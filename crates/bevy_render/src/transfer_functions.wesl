// Display transfer functions (OETFs) for signal encoding.
//
// These convert display-linear light into the encoded signal a display
// expects. They are the shader-side building blocks of the display-encoding
// pass, which does a gamut transform and then a transfer encoding. They are
// kept apart from `bevy_render::color_operations`, whose sRGB helpers are for
// color-authoring conversions.
//
// The EOTFs and the plain sRGB OETF are CPU-only, in
// `bevy_render::transfer_functions` (transfer_functions.rs), which also holds
// the `f32` parity references for the functions here. Keep both in sync.
//
// `bevy_core_pipeline::tonemapping_gt7` (gt7.wgsl) carries its own PQ helpers
// with the same ST-2084 constants and the same exp2/log2 form, because the GT7
// operator is a verbatim, fixture-locked port that imports nothing. If you
// change the PQ math here, check that gt7.wgsl stays consistent.

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
// This is what the `ExtendedSrgb` and `ExtendedDisplayP3` surface color spaces
// expect. See `bevy_window::DisplayTransfer::ExtendedSrgb` for the per-backend
// names.
//
// The plain sRGB OETF (`srgb_oetf` in transfer_functions.rs) extends only the
// linear segment below zero (`12.92 * c`), which is pow-safe for an SDR
// `[0, 1]`-domain encode. An extended-range HDR signal instead applies the
// whole curve to the magnitude and restores the sign. `abs` keeps `pow` off a
// negative base, so no finite input gives NaN.
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

// Luminance of scRGB signal value 1.0, in nits (cd/m^2). D65 reference white.
const SCRGB_REFERENCE_WHITE_NITS: f32 = 80.0;

// Encodes paper-white-relative display-linear color (1.0 = paper white at the
// tone-map operator output) as an scRGB-linear signal, where 1.0 = 80 nits:
//
//   V = L * paper_white_nits / 80
//
// scRGB is unbounded and allows negative components. Nothing is clamped here.
// Out-of-gamut policy belongs to the gamut stage, not the transfer.
fn scrgb_encode(color: vec3<f32>, paper_white_nits: f32) -> vec3<f32> {
    return color * (paper_white_nits / SCRGB_REFERENCE_WHITE_NITS);
}

// ---------------------------------------------------------------------------
// PQ (SMPTE ST 2084:2014 / ITU-R BT.2100)
// ---------------------------------------------------------------------------

// ST-2084 constants. Same values as the GT7 operator's own copies (`GT7_PQ_*`
// in gt7.wgsl).
const PQ_M1: f32 = 0.1593017578125; // (2610 / 4096) / 4
const PQ_M2: f32 = 78.84375;        // (2523 / 4096) * 128
const PQ_C1: f32 = 0.8359375;       // 3424 / 4096
const PQ_C2: f32 = 18.8515625;      // (2413 / 4096) * 32
const PQ_C3: f32 = 18.6875;         // (2392 / 4096) * 32

// Maximum luminance the PQ signal can carry, in nits (cd/m^2). PQ luminance is
// always normalized against this value, not against the display's peak.
const PQ_MAX_LUMINANCE_NITS: f32 = 10000.0;

// PQ inverse EOTF for one channel: normalized display-linear luminance
// (Y = nits / 10000, so 1.0 = 10000 nits) to a PQ signal in [0, 1].
//
// The input is clamped to >= 0 before the `pow`, since `pow` with a negative
// base is indeterminate in WGSL and gives NaN on most backends. An encoder
// entry point has to take slightly negative inputs from antialiasing fringes
// and out-of-gamut residue. The GT7 reference (`gt7_inverse_eotf_st2084` in
// gt7.wgsl, from gt7_tone_mapping.cpp) is deliberately unclamped because its
// callers guarantee non-negative inputs. Do not copy that form here.
//
// Inputs above 1.0 (more than 10000 nits) are not clamped, matching the GT7
// helper. A 10-bit unorm target clamps the signal on store.
fn pq_inverse_eotf_channel(y_in: f32) -> f32 {
    let y = max(y_in, 0.0);
    let ym = pow(y, PQ_M1);
    // Numerically stabler form of ((c1 + c2*ym) / (1 + c3*ym))^m2, identical
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

// PQ inverse EOTF taking absolute luminance in nits (cd/m^2) per channel.
// `pq_inverse_eotf_from_nits(vec3(1000.0))` is about vec3(0.7518).
fn pq_inverse_eotf_from_nits(nits: vec3<f32>) -> vec3<f32> {
    return pq_inverse_eotf(nits / PQ_MAX_LUMINANCE_NITS);
}
