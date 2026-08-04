//! Transfer functions (OETFs and EOTFs) between display-linear light and the
//! encoded signal a display expects.
//!
//! The display-encoding pass runs [`srgb_oetf_extended`], [`scrgb_encode`] and
//! the PQ inverse EOTF on the GPU. Those three mirror
//! `transfer_functions.wgsl` operation for operation and are its `f32` parity
//! reference, so keep both in sync. The EOTFs and the plain [`srgb_oetf`] are
//! CPU-only, used by the screenshot readback and save paths.
//!
//! All math uses [`bevy_math::ops`], so results are deterministic across
//! platforms and match the GT7 CPU reference (`gt7.rs` in
//! `bevy_core_pipeline::tonemapping`).

use bevy_math::ops;

/// Luminance of scRGB signal value 1.0, in nits (cd/m^2): the IEC 61966-2-2
/// D65 reference white.
pub const SCRGB_REFERENCE_WHITE_NITS: f32 = 80.0;

/// Maximum luminance the PQ (SMPTE ST 2084) signal can carry, in nits.
/// PQ luminance is always normalized against this value, not the display peak.
pub const PQ_MAX_LUMINANCE_NITS: f32 = 10000.0;

/// ST-2084 constant `m1` = (2610 / 4096) / 4.
const PQ_M1: f32 = 0.159_301_76;
/// ST-2084 constant `m2` = (2523 / 4096) * 128.
const PQ_M2: f32 = 78.84375;
/// ST-2084 constant `c1` = 3424 / 4096.
const PQ_C1: f32 = 0.8359375;
/// ST-2084 constant `c2` = (2413 / 4096) * 32 = 18.8515625 (exact in `f32`).
const PQ_C2: f32 = 18.851_563;
/// ST-2084 constant `c3` = (2392 / 4096) * 32.
const PQ_C3: f32 = 18.6875;

/// The sRGB (IEC 61966-2-1) OETF, the inverse EOTF: display-linear `[0, 1]` to
/// encoded signal.
///
/// `V = 12.92*L` for `L <= 0.0031308`, `V = 1.055*L^(1/2.4) - 0.055`
/// otherwise. Negative inputs take the linear segment, which keeps `powf` off
/// a negative base.
///
/// No counterpart in `transfer_functions.wgsl`: sRGB swapchains apply this
/// curve in hardware on the `*UnormSrgb` writeback. The screenshot path uses
/// it to quantize a display-linear capture into an 8-bit image.
pub fn srgb_oetf(linear: f32) -> f32 {
    if linear <= 0.003_130_8 {
        12.92 * linear
    } else {
        1.055 * ops::powf(linear, 1.0 / 2.4) - 0.055
    }
}

/// `f32::signum` but matching WGSL `sign`: returns `0.0` at zero rather than
/// `1.0` or `-1.0`, so the extended-sRGB encode matches the shader bit for bit
/// at `0.0` and `-0.0`.
fn wgsl_sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// The odd-symmetric ("encoded extended range") sRGB OETF: the sRGB transfer
/// continued past `[0, 1]` by mirroring the full curve through the origin, so
/// `srgb_oetf_extended(-c) == -srgb_oetf_extended(c)`.
///
/// `V = sign(c) * (|c|*12.92` for `|c| <= 0.0031308`, `1.055*|c|^(1/2.4) -
/// 0.055` otherwise `)`. This is what the
/// [`ExtendedSrgb`](bevy_window::DisplayTransfer::ExtendedSrgb) display target
/// encodes for.
///
/// [`srgb_oetf`] extends only the linear segment below zero. This one applies
/// the whole curve to the magnitude of a negative (wide-gamut) component and
/// restores its sign. `abs` keeps `powf` off a negative base, so the result is
/// finite for every finite input.
pub fn srgb_oetf_extended(c: f32) -> f32 {
    let a = c.abs();
    let lo = a * 12.92;
    let hi = 1.055 * ops::powf(a, 1.0 / 2.4) - 0.055;
    wgsl_sign(c) * if a <= 0.003_130_8 { lo } else { hi }
}

/// The odd-symmetric extended sRGB EOTF: encoded signal to display-linear.
/// Exact inverse of [`srgb_oetf_extended`], sign preserved.
pub fn srgb_eotf_extended(s: f32) -> f32 {
    let a = s.abs();
    let lo = a / 12.92;
    let hi = ops::powf((a + 0.055) / 1.055, 2.4);
    wgsl_sign(s) * if a <= 0.04045 { lo } else { hi }
}

/// Encodes paper-white-relative display-linear color (1.0 = paper white at the
/// tone-map operator output) as an scRGB-linear signal (1.0 = 80 nits):
/// `V = L * paper_white_nits / 80`.
///
/// scRGB is unbounded and allows negative components, so nothing is clamped.
pub fn scrgb_encode(color: f32, paper_white_nits: f32) -> f32 {
    color * (paper_white_nits / SCRGB_REFERENCE_WHITE_NITS)
}

/// The PQ (SMPTE ST 2084) inverse EOTF: normalized display-linear luminance
/// (`Y = nits / 10000`) to a PQ signal in `[0, 1]`.
///
/// Negative inputs are clamped to zero before the `pow`, since `powf` with a
/// negative base and the non-integer exponent `m1` is NaN. The GT7 copy in
/// `gt7.rs` skips the clamp because its callers guarantee non-negative input.
/// Inputs above 1.0 are not clamped, so the signal can exceed 1.0 and the
/// target format clamps it on store.
pub fn pq_inverse_eotf(y: f32) -> f32 {
    let y = y.max(0.0);
    let ym = ops::powf(y, PQ_M1);
    // Numerically stabler form of ((c1 + c2*ym) / (1 + c3*ym))^m2, identical
    // to the GT7 operator's own copy in gt7.wgsl and gt7.rs.
    ops::exp2(PQ_M2 * (ops::log2(PQ_C1 + PQ_C2 * ym) - ops::log2(1.0 + PQ_C3 * ym)))
}

/// [`pq_inverse_eotf`] taking absolute luminance in nits.
/// `pq_inverse_eotf_from_nits(1000.0)` is about 0.7518.
pub fn pq_inverse_eotf_from_nits(nits: f32) -> f32 {
    pq_inverse_eotf(nits / PQ_MAX_LUMINANCE_NITS)
}

/// The PQ EOTF: PQ signal (clamped to `[0, 1]`) to normalized display-linear
/// luminance (1.0 = 10000 nits). Inverse of [`pq_inverse_eotf`].
pub fn pq_eotf(signal: f32) -> f32 {
    let n = signal.clamp(0.0, 1.0);
    let np = ops::powf(n, 1.0 / PQ_M2);
    let l = (np - PQ_C1).max(0.0) / (PQ_C2 - PQ_C3 * np);
    ops::powf(l, 1.0 / PQ_M1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `f64` reference for the PQ inverse EOTF, so the expected values do not
    /// come from the `f32` implementation under test.
    fn pq_inverse_eotf_f64(y: f64) -> f64 {
        let m1 = 2610.0 / 4096.0 / 4.0;
        let m2 = 2523.0 / 4096.0 * 128.0;
        let c1 = 3424.0 / 4096.0;
        let c2 = 2413.0 / 4096.0 * 32.0;
        let c3 = 2392.0 / 4096.0 * 32.0;
        let ym = y.max(0.0).powf(m1);
        ((c1 + c2 * ym) / (1.0 + c3 * ym)).powf(m2)
    }

    #[test]
    fn pq_constants_are_exact() {
        assert_eq!(PQ_M1, (2610.0f64 / 4096.0 / 4.0) as f32);
        assert_eq!(PQ_M2, (2523.0f64 / 4096.0 * 128.0) as f32);
        assert_eq!(PQ_C1, (3424.0f64 / 4096.0) as f32);
        assert_eq!(PQ_C2, (2413.0f64 / 4096.0 * 32.0) as f32);
        assert_eq!(PQ_C3, (2392.0f64 / 4096.0 * 32.0) as f32);
    }

    #[test]
    fn pq_inverse_eotf_matches_reference_values() {
        // 1000 nits, the canonical check value from the encoder spec.
        let expected_1000 = pq_inverse_eotf_f64(0.1); // 0.75182700871...
        assert!((expected_1000 - 0.751_827).abs() < 1e-6);
        assert!((pq_inverse_eotf_from_nits(1000.0) as f64 - expected_1000).abs() < 1e-5);

        // Endpoint and mid-range sweep. The f32 form picks up about 1 ULP per
        // pow/exp2/log2 step: a wider but sub-quantization-step tolerance.
        for nits in [0.0, 0.1, 1.0, 80.0, 100.0, 203.0, 2000.0, 10000.0] {
            let expected = pq_inverse_eotf_f64(nits as f64 / 10000.0);
            let actual = pq_inverse_eotf_from_nits(nits) as f64;
            assert!(
                (actual - expected).abs() < 5e-5,
                "PQ({nits} nits): {actual} vs {expected}"
            );
        }
        // 10000 nits encodes to exactly 1.0 (within f32 rounding).
        assert!((pq_inverse_eotf(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pq_negative_input_is_clamped_not_nan() {
        // The clamp must make negative inputs behave exactly like zero. The
        // unclamped GT7 form returns NaN here.
        let at_zero = pq_inverse_eotf(0.0);
        assert!(at_zero.is_finite());
        for y in [-1e-6, -0.5, -10.0] {
            let v = pq_inverse_eotf(y);
            assert!(v.is_finite(), "PQ({y}) must be finite");
            assert_eq!(v, at_zero, "PQ({y}) must equal PQ(0)");
        }
        // PQ(0) is c1^m2, a tiny positive value, not 0.
        assert!(at_zero > 0.0 && at_zero < 1e-5);
    }

    #[test]
    fn pq_round_trips() {
        for y in [0.0, 1e-4, 0.01, 0.1, 0.5, 1.0] {
            let signal = pq_inverse_eotf(y);
            let back = pq_eotf(signal);
            assert!(
                (back - y).abs() < 2e-4,
                "PQ round trip at {y}: got {back} (signal {signal})"
            );
        }
    }

    #[test]
    fn srgb_round_trips_and_is_continuous() {
        // On `[0, 1]` `abs` and `sign` are no-ops, so the extended EOTF is the
        // plain sRGB EOTF and inverts `srgb_oetf` exactly.
        for l in [0.0, 0.001, 0.0031308, 0.004, 0.1, 0.18, 0.5, 0.9, 1.0] {
            let signal = srgb_oetf(l);
            let back = srgb_eotf_extended(signal);
            assert!(
                (back - l).abs() < 1e-6,
                "sRGB round trip at {l}: got {back}"
            );
        }
        // Continuity at the piecewise breakpoint.
        let below = srgb_oetf(0.0031308);
        let above = srgb_oetf(0.0031309);
        assert!((below - above).abs() < 1e-5);
        assert_eq!(srgb_oetf(0.0), 0.0);
        assert!((srgb_oetf(1.0) - 1.0).abs() < 1e-6);
        // 18% gray encodes to about 0.4613, a well-known sRGB anchor.
        assert!((srgb_oetf(0.18) - 0.461_356).abs() < 1e-4);
        // Negatives take the linear segment, so no NaN.
        assert_eq!(srgb_oetf(-0.5), 12.92 * -0.5);
    }

    #[test]
    fn srgb_oetf_extended_is_odd_symmetric_and_anchored() {
        // SDR reference white (scRGB 1.0 = 80 nits) encodes to 0.99999994
        // (`1.055 - 0.055` in f32, the same as `srgb_oetf(1.0)`), so an 80-nit
        // paper white round-trips SDR through the extended path.
        assert!((srgb_oetf_extended(1.0) - 1.0).abs() < 1e-6);
        assert_eq!(srgb_oetf_extended(0.0), 0.0);
        for c in [0.001, 0.0031308, 0.05, 0.18, 0.625, 1.0, 1.25, 2.0] {
            assert!(
                (srgb_oetf_extended(-c) + srgb_oetf_extended(c)).abs() < 1e-6,
                "not odd-symmetric at {c}"
            );
        }
        // Continuity at the piecewise breakpoint.
        let below = srgb_oetf_extended(0.0031308);
        let above = srgb_oetf_extended(0.0031309);
        assert!((below - above).abs() < 1e-5);
        // On [0, 1] the extended OETF is the plain sRGB OETF.
        for l in [0.0031308, 0.05, 0.18, 0.5, 1.0] {
            assert!(
                (srgb_oetf_extended(l) - srgb_oetf(l)).abs() < 1e-6,
                "diverges at {l}"
            );
        }
        // f64-evaluated fixtures. The encoder feeds color * paper_white / 80
        // into this curve, so at 100-nit paper white scRGB 0.625 is 0.5 and
        // 1.25 is 1.0, brighter than SDR.
        assert!((srgb_oetf_extended(0.625) - 0.812_366).abs() < 1e-4);
        assert!((srgb_oetf_extended(1.25) - 1.102_795).abs() < 1e-4);
        assert!((srgb_oetf_extended(-0.125) + 0.388_573).abs() < 1e-4);
        assert!((srgb_oetf_extended(0.18) - 0.461_356).abs() < 1e-4);
        // Large negatives stay finite: no NaN from a negative pow base.
        assert!(srgb_oetf_extended(-1e30).is_finite());
    }

    #[test]
    fn srgb_oetf_extended_round_trips() {
        for c in [-2.0, -1.0, -0.125, -0.001, 0.0, 0.001, 0.18, 1.0, 1.25, 3.0] {
            let back = srgb_eotf_extended(srgb_oetf_extended(c));
            assert!((back - c).abs() < 1e-4, "round trip at {c}: got {back}");
        }
        assert_eq!(srgb_oetf_extended(0.0), 0.0);
        assert_eq!(srgb_eotf_extended(0.0), 0.0);
    }

    #[test]
    fn scrgb_scale_is_paper_white_over_80() {
        // At an 80-nit paper white the encoding is the identity.
        assert_eq!(scrgb_encode(1.0, 80.0), 1.0);
        // Default SDR paper white, 100 nits.
        assert_eq!(scrgb_encode(1.0, 100.0), 1.25);
        // ITU-R BT.2408 reference paper white, 203 nits.
        assert!((scrgb_encode(1.0, 203.0) - 2.5375).abs() < 1e-6);
        // Linear in the color value, and negatives pass through.
        assert_eq!(scrgb_encode(0.5, 100.0), 0.625);
        assert_eq!(scrgb_encode(-0.5, 100.0), -0.625);
    }
}
