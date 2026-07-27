// Display transfer functions (OETFs) for signal encoding.
//
// These functions convert *display-linear* light into the encoded signal a
// display expects. They are the shader-side building blocks of the
// display-encoding pass (gamut transform → transfer encoding), and are kept
// separate from `bevy_render::color_operations`, whose sRGB helpers exist for
// color-*authoring* conversions.
//
// Only the encode direction the display-encoding pass runs lives here. The
// EOTFs and the plain sRGB OETF are CPU-only, in
// `bevy_render::transfer_functions` (transfer_functions.rs), which also holds
// the `f32` parity references for the functions here; keep both in sync.
//
// NOTE on duplication: `bevy_core_pipeline::tonemapping_gt7` (gt7.wgsl)
// carries its own self-contained PQ helpers (same ST-2084 constants, same
// numerically-stable exp2/log2 form) because the GT7 operator is a verbatim,
// fixture-locked port that imports nothing. gt7.wgsl is not deduplicated onto
// this module; if you change the PQ math here, check `gt7.wgsl` stays
// consistent.

#define_import_path bevy_render::transfer_functions

// ---------------------------------------------------------------------------
// Extended-range sRGB (IEC 61966-2-2 encoded form / "scRGB nonlinear")
// ---------------------------------------------------------------------------

// Odd-symmetric ("encoded extended range") sRGB OETF for one channel: the sRGB
// transfer continued past `[0, 1]` by mirroring the full curve through the
// origin, so `f(-c) == -f(c)`:
//
//   V = sign(c) * ( |c|*12.92                  if |c| <= 0.0031308
//                   1.055*|c|^(1/2.4) - 0.055  otherwise )
//
// This is the transfer the `ExtendedSrgb` / `ExtendedDisplayP3` surface color
// spaces expect (Vulkan `EXTENDED_SRGB_NONLINEAR_EXT`, Metal
// `kCGColorSpaceExtendedSRGB` / `kCGColorSpaceExtendedDisplayP3`, the browser
// WebGPU `srgb` / `display-p3` canvas with `toneMapping: "extended"`).
//
// Distinct from the plain sRGB OETF (`srgb_oetf` in transfer_functions.rs),
// which extends only the LINEAR segment below zero (`12.92 * c`): that is the
// right pow-safety behavior for an SDR `[0, 1]`-domain encode, but the
// extended-range HDR signal must apply the full gamma curve to the magnitude
// of negative (wide-gamut / out-of-gamut) components and preserve their sign.
// `abs` keeps `pow` away from a negative base, so the result is NaN-free for
// every finite input.
fn srgb_oetf_extended_channel(c: f32) -> f32 {
    let a = abs(c);
    let lo = a * 12.92;
    let hi = 1.055 * pow(a, 1.0 / 2.4) - 0.055;
    return sign(c) * select(hi, lo, a <= 0.0031308);
}

// Per-channel odd-symmetric extended sRGB OETF; see
// `srgb_oetf_extended_channel`.
fn srgb_oetf_extended(linear: vec3<f32>) -> vec3<f32> {
    return vec3(
        srgb_oetf_extended_channel(linear.x),
        srgb_oetf_extended_channel(linear.y),
        srgb_oetf_extended_channel(linear.z),
    );
}

// ---------------------------------------------------------------------------
// scRGB (IEC 61966-2-2, linear form)
// ---------------------------------------------------------------------------

// Luminance, in nits (cd/m²), of scRGB signal value 1.0 (D65 reference white).
const SCRGB_REFERENCE_WHITE_NITS: f32 = 80.0;

// Encodes paper-white-relative display-linear color (1.0 = paper white at the
// tone-map operator output) as scRGB-linear signal, where 1.0 = 80 nits:
//
//   V = L * paper_white_nits / 80
//
// scRGB is unbounded and permits negative components; no clamping is applied
// here (out-of-gamut policy belongs to the gamut stage, not the transfer).
fn scrgb_encode(color: vec3<f32>, paper_white_nits: f32) -> vec3<f32> {
    return color * (paper_white_nits / SCRGB_REFERENCE_WHITE_NITS);
}

// ---------------------------------------------------------------------------
// PQ (SMPTE ST 2084:2014 / ITU-R BT.2100)
// ---------------------------------------------------------------------------

// ST-2084 constants. Identical values to the GT7 operator's self-contained
// copies (`GT7_PQ_*` in gt7.wgsl).
const PQ_M1: f32 = 0.1593017578125; // (2610 / 4096) / 4
const PQ_M2: f32 = 78.84375;        // (2523 / 4096) * 128
const PQ_C1: f32 = 0.8359375;       // 3424 / 4096
const PQ_C2: f32 = 18.8515625;      // (2413 / 4096) * 32
const PQ_C3: f32 = 18.6875;         // (2392 / 4096) * 32

// Maximum luminance the PQ signal can carry, in nits (cd/m²). PQ luminance is
// always normalized against this value, NOT against the display's peak.
const PQ_MAX_LUMINANCE_NITS: f32 = 10000.0;

// PQ inverse EOTF for one channel: normalized display-linear luminance
// (Y = nits / 10000, so 1.0 = 10000 nits) → PQ signal in [0, 1].
//
// The input is clamped to >= 0 BEFORE the `pow`: `pow` with a negative base
// is indeterminate in WGSL (NaN on most backends), and an encoder entry point
// must tolerate slightly-negative inputs (antialiasing fringes, out-of-gamut
// residue after the gamut stage). The GT7 reference implementation
// (`gt7_inverse_eotf_st2084` in gt7.wgsl, from gt7_tone_mapping.cpp) is
// deliberately left unclamped because its callers guarantee non-negative
// inputs — do not copy that form here.
//
// Inputs above 1.0 (more than 10000 nits) are NOT clamped, mirroring the GT7
// helper; the resulting signal exceeds 1.0 and is clamped by the target
// (10-bit unorm formats clamp on store).
fn pq_inverse_eotf_channel(y_in: f32) -> f32 {
    let y = max(y_in, 0.0);
    let ym = pow(y, PQ_M1);
    // Numerically-stabler form of ((c1 + c2*ym) / (1 + c3*ym))^m2, identical
    // to the GT7 operator's copy.
    return exp2(PQ_M2 * (log2(PQ_C1 + PQ_C2 * ym) - log2(1.0 + PQ_C3 * ym)));
}

// Per-channel PQ inverse EOTF; see `pq_inverse_eotf_channel`.
fn pq_inverse_eotf(y: vec3<f32>) -> vec3<f32> {
    return vec3(
        pq_inverse_eotf_channel(y.x),
        pq_inverse_eotf_channel(y.y),
        pq_inverse_eotf_channel(y.z),
    );
}

// PQ inverse EOTF taking absolute luminance in nits (cd/m²) per channel.
// `pq_inverse_eotf_from_nits(vec3(1000.0))` ≈ vec3(0.7518).
fn pq_inverse_eotf_from_nits(nits: vec3<f32>) -> vec3<f32> {
    return pq_inverse_eotf(nits / PQ_MAX_LUMINANCE_NITS);
}
