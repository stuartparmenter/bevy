//! CPU mirror of the display-encoding pass's out-of-gamut chroma compression.
//! Keep it in sync with `gamut_compress` in `display_encoding.wesl`. That
//! shader documents the ACES 1.3 Reference Gamut Compression this implements
//! (Academy S-2020-001, "RGC"; reference implementation `lib/RGC_common.ctl`
//! in `aces-dev`), and where the thresholds, limits and power come from.
//!
//! The rationale for choosing the ACES RGC over an exact hue-preserving
//! `ICtCp` compression is on [`DisplayGamutCompression`].
//!
//! [`DisplayGamutCompression`]: super::DisplayGamutCompression

use bevy_math::{ops, Vec3};

/// Per-channel compression thresholds (cyan, magenta, yellow), the published
/// ACES 1.3 Reference Gamut Compression values.
pub const GAMUT_COMPRESSION_THRESHOLD: Vec3 = Vec3::new(0.815, 0.803, 0.880);

/// Per-channel limits, re-derived for the Rec.2020 to Rec.709 contraction.
/// Each must be at least the maximum achromatic distance the source gamut
/// reaches in target-gamut coordinates, or the most saturated sources stay out
/// of gamut after compression.
pub const GAMUT_COMPRESSION_LIMIT: Vec3 = Vec3::new(1.62, 1.10, 1.13);

/// Exponent of the ACES RGC parametric compression curve (published value).
pub const GAMUT_COMPRESSION_POWER: f32 = 1.2;

/// Precomputed curve scales: `compression_scale(threshold, limit, power)` per
/// channel, evaluated in `f64` and baked because the WGSL side cannot evaluate
/// `pow` in a `const` expression. A test locks them to the closed form.
pub const GAMUT_COMPRESSION_SCALE: Vec3 = Vec3::new(0.21634937, 0.43270176, 0.18745117);

/// The closed-form scale of the ACES RGC compression curve:
/// `(lim - thr) / (((1 - thr) / (lim - thr))^(-power) - 1)^(1/power)`.
pub fn compression_scale(threshold: f32, limit: f32, power: f32) -> f32 {
    (limit - threshold)
        / ops::powf(
            ops::powf((1.0 - threshold) / (limit - threshold), -power) - 1.0,
            1.0 / power,
        )
}

/// The ACES RGC parametric compression curve for one channel's achromatic
/// distance. Below `threshold` it returns `threshold`. [`gamut_compress`]
/// selects the original distance there.
pub fn compress_distance(dist: f32, threshold: f32, scale: f32) -> f32 {
    let nd = (dist - threshold).max(0.0) / scale;
    let p = ops::powf(nd, GAMUT_COMPRESSION_POWER);
    threshold + scale * nd / ops::powf(1.0 + p, 1.0 / GAMUT_COMPRESSION_POWER)
}

/// Compresses an out-of-gamut color (negative components) toward the achromatic
/// axis at constant `max(r, g, b)`.
///
/// Channels whose achromatic distance is below [`GAMUT_COMPRESSION_THRESHOLD`]
/// are returned bit-identically, and colors within [`GAMUT_COMPRESSION_LIMIT`]
/// land inside the target gamut. Colors with no positive channel are returned
/// unchanged. The shader's final `max(0)` safety clip handles them, except
/// under extended sRGB.
pub fn gamut_compress(rgb: Vec3) -> Vec3 {
    let achromatic = rgb.max_element();
    if achromatic <= 0.0 {
        return rgb;
    }
    let dist = (Vec3::splat(achromatic) - rgb) / achromatic;
    let compressed_dist = Vec3::new(
        compress_distance(
            dist.x,
            GAMUT_COMPRESSION_THRESHOLD.x,
            GAMUT_COMPRESSION_SCALE.x,
        ),
        compress_distance(
            dist.y,
            GAMUT_COMPRESSION_THRESHOLD.y,
            GAMUT_COMPRESSION_SCALE.y,
        ),
        compress_distance(
            dist.z,
            GAMUT_COMPRESSION_THRESHOLD.z,
            GAMUT_COMPRESSION_SCALE.z,
        ),
    );
    let compressed = Vec3::splat(achromatic) - compressed_dist * achromatic;
    Vec3::select(dist.cmplt(GAMUT_COMPRESSION_THRESHOLD), rgb, compressed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_render::{
        transfer_functions::pq_inverse_eotf_from_nits,
        working_color_space::{REC2020_TO_REC709, REC709_TO_REC2020},
    };

    /// `ICtCp` (ITU-R BT.2100) hue angle, in degrees, of a linear Rec.709
    /// color. Out-of-gamut components are allowed: the color is converted to
    /// Rec.2020, where the test colors are in gamut, before the LMS and PQ
    /// steps. `1.0` is treated as 100 nits, so hue angles are only comparable
    /// against each other at the same scale.
    fn ictcp_hue_degrees(rgb709: Vec3) -> f32 {
        let rgb = REC709_TO_REC2020 * rgb709;
        let l = (rgb.x * 1688.0 + rgb.y * 2146.0 + rgb.z * 262.0) / 4096.0;
        let m = (rgb.x * 683.0 + rgb.y * 2951.0 + rgb.z * 462.0) / 4096.0;
        let s = (rgb.x * 99.0 + rgb.y * 309.0 + rgb.z * 3688.0) / 4096.0;
        let l_pq = pq_inverse_eotf_from_nits(l.max(0.0) * 100.0);
        let m_pq = pq_inverse_eotf_from_nits(m.max(0.0) * 100.0);
        let s_pq = pq_inverse_eotf_from_nits(s.max(0.0) * 100.0);
        let ct = (6610.0 * l_pq - 13613.0 * m_pq + 7003.0 * s_pq) / 4096.0;
        let cp = (17933.0 * l_pq - 17390.0 * m_pq - 543.0 * s_pq) / 4096.0;
        ops::atan2(cp, ct).to_degrees()
    }

    fn hue_drift_degrees(input: Vec3, output: Vec3) -> f32 {
        let mut drift = ictcp_hue_degrees(output) - ictcp_hue_degrees(input);
        if drift > 180.0 {
            drift -= 360.0;
        }
        if drift < -180.0 {
            drift += 360.0;
        }
        drift.abs()
    }

    #[test]
    fn baked_scales_match_the_closed_form() {
        for i in 0..3 {
            let closed = compression_scale(
                GAMUT_COMPRESSION_THRESHOLD[i],
                GAMUT_COMPRESSION_LIMIT[i],
                GAMUT_COMPRESSION_POWER,
            );
            let baked = GAMUT_COMPRESSION_SCALE[i];
            assert!(
                ((closed - baked) / baked).abs() < 1e-6,
                "channel {i}: closed form {closed} vs baked {baked}"
            );
        }
    }

    #[test]
    fn under_threshold_values_pass_through_bit_exactly() {
        // All distances under the smallest threshold (0.803).
        for rgb in [
            Vec3::new(0.5, 0.4, 0.3),
            Vec3::new(0.18, 0.18, 0.18),
            Vec3::new(1.0, 0.25, 0.35),
            Vec3::new(7.3, 1.6, 2.2),
            Vec3::ZERO,
            Vec3::new(-0.25, -1.0, -0.5), // no positive channel: untouched
        ] {
            let out = gamut_compress(rgb);
            assert_eq!(rgb.x.to_bits(), out.x.to_bits(), "{rgb} -> {out}");
            assert_eq!(rgb.y.to_bits(), out.y.to_bits(), "{rgb} -> {out}");
            assert_eq!(rgb.z.to_bits(), out.z.to_bits(), "{rgb} -> {out}");
        }
    }

    /// Sweeps the Rec.2020 gamut hull (cube faces) through the Rec.2020 to
    /// Rec.709 contraction and asserts every compressed color is inside the
    /// Rec.709 gamut, up to f32 residue caught by the shader's final `max(0)`.
    /// This is the property the re-derived limits exist for. The ACES
    /// camera-gamut limits fail it in the cyan direction.
    #[test]
    fn rec2020_hull_compresses_into_gamut() {
        const N: usize = 100;
        let mut checked = 0;
        for fixed_axis in 0..3 {
            for fixed_value in [0.0f32, 1.0] {
                for a in 0..=N {
                    for b in 0..=N {
                        let mut v = [0.0f32; 3];
                        v[fixed_axis] = fixed_value;
                        v[(fixed_axis + 1) % 3] = a as f32 / N as f32;
                        v[(fixed_axis + 2) % 3] = b as f32 / N as f32;
                        let rgb709 = REC2020_TO_REC709 * Vec3::from_array(v);
                        let out = gamut_compress(rgb709);
                        assert!(
                            out.min_element() >= -1e-4,
                            "2020 {v:?} -> 709 {rgb709} compressed to {out}, still out of gamut"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 6 * (N + 1) * (N + 1));
    }

    /// The Rec.2020 primaries and secondaries, the most saturated possible
    /// inputs, land in gamut with bounded `ICtCp` hue drift. The bounds below
    /// are the measured behavior of the per-channel ACES RGC formulation, not
    /// exact hue preservation. See `DisplayGamutCompression`.
    #[test]
    fn rec2020_primaries_compress_in_gamut_with_bounded_hue_drift() {
        let cases = [
            (Vec3::new(1.0, 0.0, 0.0), 6.5, "2020 red"),
            (Vec3::new(0.0, 1.0, 0.0), 6.0, "2020 green"),
            (Vec3::new(0.0, 0.0, 1.0), 18.0, "2020 blue"),
            (Vec3::new(0.0, 1.0, 1.0), 2.5, "2020 cyan"),
            (Vec3::new(1.0, 0.0, 1.0), 2.0, "2020 magenta"),
            (Vec3::new(1.0, 1.0, 0.0), 2.0, "2020 yellow"),
        ];
        for (rgb2020, max_drift, label) in cases {
            let rgb709 = REC2020_TO_REC709 * rgb2020;
            assert!(rgb709.min_element() < 0.0, "{label} should be OOG in 709");
            let out = gamut_compress(rgb709);
            assert!(out.min_element() >= -1e-5, "{label}: {out} out of gamut");
            // max(r, g, b) (the achromatic anchor) is preserved exactly.
            assert_eq!(
                out.max_element().to_bits(),
                rgb709.max_element().to_bits(),
                "{label}: achromatic anchor changed"
            );
            let drift = hue_drift_degrees(rgb709, out);
            assert!(
                drift < max_drift,
                "{label}: hue drift {drift} deg exceeds {max_drift} deg"
            );
        }
    }

    /// Moderately out-of-gamut colors, the common case of slightly wide chroma
    /// after a Rec.2020 to Rec.709 contraction, keep hue drift in the low
    /// single digits of degrees. The per-direction bounds below are the
    /// measured behavior of the per-channel formulation.
    #[test]
    fn moderate_out_of_gamut_hue_drift_is_small() {
        // The Rec.2020 color with the worst cyan-direction distance (~1.594),
        // and the same color blended 55% toward its clip (distance ~1.27).
        let worst_cyan_709 = REC2020_TO_REC709 * Vec3::new(0.0, 0.17666, 0.19333);
        let cases = [
            (worst_cyan_709, 2.0),
            (
                worst_cyan_709.lerp(worst_cyan_709.max(Vec3::ZERO), 0.55),
                1.0,
            ),
            (Vec3::new(-0.02, 0.6, 0.1), 3.5), // slightly OOG green-ish
            (Vec3::new(0.8, -0.025, 0.7), 1.0), // slightly OOG magenta-ish
            (Vec3::new(0.9, 0.4, -0.02), 4.5), // slightly OOG orange-ish
        ];
        for (rgb, max_drift) in cases {
            let out = gamut_compress(rgb);
            assert!(out.min_element() >= -1e-5, "{rgb}: {out} out of gamut");
            let drift = hue_drift_degrees(rgb, out);
            assert!(
                drift < max_drift,
                "{rgb}: hue drift {drift} deg exceeds {max_drift} deg"
            );
        }
    }

    /// The compression curve is monotonically increasing through the knee and
    /// continuous at the threshold.
    #[test]
    fn compression_is_monotonic_and_continuous_at_the_knee() {
        for i in 0..3 {
            let threshold = GAMUT_COMPRESSION_THRESHOLD[i];
            let scale = GAMUT_COMPRESSION_SCALE[i];
            // Continuity: just above the threshold the curve still tracks the
            // identity to first order.
            let eps = 1e-4;
            let at_knee = compress_distance(threshold + eps, threshold, scale);
            assert!(
                (at_knee - (threshold + eps)).abs() < 1e-5,
                "channel {i}: discontinuous at the knee ({at_knee})"
            );
            // Monotonicity of the effective per-channel mapping across the knee
            // and the whole compression range: identity below the threshold via
            // the pass-through select, the curve above it.
            let effective = |d: f32| {
                if d < threshold {
                    d
                } else {
                    compress_distance(d, threshold, scale)
                }
            };
            let mut previous = effective(0.0);
            let mut d = 0.01f32;
            while d < 2.5 {
                let current = effective(d);
                assert!(
                    current > previous,
                    "channel {i}: not monotonic at distance {d}"
                );
                previous = current;
                d += 0.01;
            }
        }
    }

    #[test]
    fn the_limit_maps_onto_the_gamut_boundary() {
        for i in 0..3 {
            let compressed = compress_distance(
                GAMUT_COMPRESSION_LIMIT[i],
                GAMUT_COMPRESSION_THRESHOLD[i],
                GAMUT_COMPRESSION_SCALE[i],
            );
            assert!(
                (compressed - 1.0).abs() < 1e-6,
                "channel {i}: compress(limit) = {compressed}, expected 1.0"
            );
        }
    }

    /// The compression is scale-invariant (distances are ratios), so it
    /// behaves identically across exposure: compress(k * c) == k * compress(c).
    #[test]
    fn compression_is_scale_invariant() {
        let rgb = REC2020_TO_REC709 * Vec3::new(0.1, 0.9, 0.2);
        let reference = gamut_compress(rgb);
        for k in [0.25f32, 4.0, 64.0] {
            let scaled = gamut_compress(rgb * k) / k;
            assert!(
                (scaled - reference).abs().max_element() < 1e-4,
                "scale {k}: {scaled} vs {reference}"
            );
        }
    }
}
