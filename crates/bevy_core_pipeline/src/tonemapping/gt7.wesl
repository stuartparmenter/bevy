// Gran Turismo 7 tone-mapping operator.
//
// Native WGSL port of Polyphony Digital's reference implementation
// (gt7_tone_mapping.cpp, MIT License, Copyright (c) 2025 Polyphony Digital Inc.),
// published with the SIGGRAPH 2025 course "Physically Based Tone Mapping in
// Gran Turismo 7". The CPU parity reference and the fixture tests that gate
// changes to this file live in `gt7.rs`. Keep both in sync.
//
// Unit convention: the operator works on linear Rec.2020 RGB "frame buffer
// values" where 1.0 is 100 nits.

// Precomputed parameters for the GT7 tone-mapping pipeline.
//
// `k_a`, `k_b`, `k_c` and `peak_ucs` are closed-form products of the
// user-facing parameters, computed once on the CPU instead of per pixel (see
// `Gt7ParamsUniform::from_params` in gt7.rs).
//
// Under the GT7_PARAMS_UNIFORM shader def this struct is filled per view from
// the `Gt7ParamsUniform` buffer (gt7.rs; keep the field order in sync).
// Without the def it holds the baked defaults from `gt7_default_sdr_params()`.
struct Gt7Params {
    // Display peak in frame-buffer units (peak_nits / 100).
    peak: f32,
    // Shoulder constants: shoulder(x) = k_a + k_b * exp(x * k_c).
    k_a: f32,
    k_b: f32,
    k_c: f32,
    // Curve shape.
    mid_point: f32,
    linear_section: f32,
    toe_strength: f32,
    // ICtCp `I` of peak white; normalizes the luminance driving the chroma fade.
    peak_ucs: f32,
    // UCS share of the final per-channel/UCS blend.
    blend_ratio: f32,
    // Chroma fade band, as fractions of `peak_ucs` (fade_end may exceed 1.0).
    fade_start: f32,
    fade_end: f32,
    // Post-clamp output scale. In SDR mode it is 1 / 2.5, which rescales the
    // 250-nit-referred result into [0, 1]. In HDR mode it is
    // 100 / paper_white_nits, so 1.0 at the operator output is paper white.
    sdr_correction_factor: f32,
}

// Per-view GT7 parameters prepared on the CPU (`Gt7ParamsUniform` in gt7.rs).
// Bound only when the pipeline is specialized with the GT7_PARAMS_UNIFORM
// shader def (see `gt7_params_uniform_active` in tonemapping/mod.rs). The
// binding index is pushed as a shader def so other bind groups can rebind it
// elsewhere; it is 5 in the tonemapping pass.
@if(GT7_PARAMS_UNIFORM)
@group(0) @binding(constants::GT7_PARAMS_BINDING_INDEX) var<uniform> gt7_params_uniform: Gt7Params;

// Default SDR-mode parameters (curve defaults alpha = 0.25, mid_point = 0.538,
// linear_section = 0.444, toe_strength = 1.28 at peak 2.5 fb = 250 nits).
// Derived constants are f64-precision evaluations of the closed forms in
// `Gt7ParamsUniform::from_params` (gt7.rs).
fn gt7_default_sdr_params() -> Gt7Params {
    var params: Gt7Params;
    params.peak = 2.5;
    // k = (linear_section - 1) / (alpha - 1) = 0.7413333333333334
    params.k_a = 2.963333333333334;       // peak * (linear_section + k)
    params.k_b = -3.3733512380644313;     // -peak * k * exp(linear_section / k)
    params.k_c = -0.539568345323741;      // -1 / (k * peak)
    params.mid_point = 0.538;
    params.linear_section = 0.444;
    params.toe_strength = 1.28;
    params.peak_ucs = 0.6025591549907509; // ICtCp I of (2.5, 2.5, 2.5)
    params.blend_ratio = 0.6;
    params.fade_start = 0.98;
    params.fade_end = 1.16;
    params.sdr_correction_factor = 0.4;   // 1 / 2.5
    return params;
}

// ST-2084 (PQ) constants, SMPTE ST 2084:2014 / ITU-R BT.2100.
const GT7_PQ_M1: f32 = 0.1593017578125;  // (2610 / 4096) / 4
const GT7_PQ_M2: f32 = 78.84375;         // (2523 / 4096) * 128
const GT7_PQ_C1: f32 = 0.8359375;        // 3424 / 4096
const GT7_PQ_C2: f32 = 18.8515625;       // (2413 / 4096) * 32
const GT7_PQ_C3: f32 = 18.6875;          // (2392 / 4096) * 32
// Maximum luminance supported by PQ (cd/m²).
const GT7_PQ_C: f32 = 10000.0;
// Physical luminance (cd/m²) of frame-buffer value 1.0 in GT7's convention.
const GT7_REFERENCE_LUMINANCE: f32 = 100.0;
// Gran Turismo's SDR paper white (cd/m²); 2.5 in frame-buffer units.
const GT7_SDR_PAPER_WHITE: f32 = 250.0;

// Full-precision linear Rec.709 -> Rec.2020 matrix (D65, per ITU-R BT.2087).
// Bevy's working space is currently scene-linear Rec.709 and the operator is
// native Rec.2020, so the input is expanded unless WORKING_COLOR_SPACE_REC2020
// is set. The literals are duplicated from
// `bevy_render::working_color_space::REC709_TO_REC2020` because this module
// imports nothing, which keeps it self-contained and fixture-locked. Keep them
// identical: `matrices_match_wgsl_f64_literals` there locks them bitwise.
const GT7_REC_709_TO_REC_2020 = mat3x3<f32>(
    0.627403895934699, 0.06909728935823199, 0.016391438875150228,   // column 0
    0.32928303837788375, 0.919540395075459, 0.08801330787722578,    // column 1
    0.043313065687417246, 0.011362315566309154, 0.895595253247624,  // column 2
);

// Inverse of the above (linear Rec.2020 -> Rec.709). Same sync contract, with
// `bevy_render::working_color_space::REC2020_TO_REC709`.
const GT7_REC_2020_TO_REC_709 = mat3x3<f32>(
    1.6604910021084347, -0.12455047452159052, -0.01815076335490522, // column 0
    -0.5876411387885496, 1.1328998971259598, -0.10057889800800739,  // column 1
    -0.07284986331988484, -0.008349422604369487, 1.1187296613629125, // column 2
);

// ST-2084 (PQ) EOTF: normalized PQ signal (clamped to [0, 1]) -> linear
// frame-buffer value (1.0 = 100 nits).
fn gt7_eotf_st2084(n_in: f32) -> f32 {
    let n = clamp(n_in, 0.0, 1.0);
    let np = pow(n, 1.0 / GT7_PQ_M2);
    var l = max(np - GT7_PQ_C1, 0.0);
    l = l / (GT7_PQ_C2 - GT7_PQ_C3 * np);
    l = pow(l, 1.0 / GT7_PQ_M1);
    // Convert absolute luminance (cd/m²) into the frame-buffer linear scale.
    return l * GT7_PQ_C / GT7_REFERENCE_LUMINANCE;
}

// ST-2084 (PQ) inverse EOTF: linear frame-buffer value (1.0 = 100 nits) ->
// normalized PQ signal. Does not clamp its input above 1.0, mirroring the
// reference. Callers must keep the input non-negative: `pow` with a negative
// base is indeterminate in WGSL.
fn gt7_inverse_eotf_st2084(v: f32) -> f32 {
    let y = v * GT7_REFERENCE_LUMINANCE / GT7_PQ_C;
    let ym = pow(y, GT7_PQ_M1);
    // Numerically-stabler form of ((c1 + c2*ym) / (1 + c3*ym))^m2.
    return exp2(GT7_PQ_M2 * (log2(GT7_PQ_C1 + GT7_PQ_C2 * ym) - log2(1.0 + GT7_PQ_C3 * ym)));
}

// Linear Rec.2020 RGB -> ICtCp (ITU-R BT.2100 / ITU-T T.302).
// Deviation from the C++ reference: LMS is clamped at zero before the PQ
// encode to avoid NaN for inputs saturated enough to drive LMS negative.
// Matches the CPU port in gt7.rs.
fn gt7_rgb_to_ictcp(rgb: vec3<f32>) -> vec3<f32> {
    let l = (rgb.r * 1688.0 + rgb.g * 2146.0 + rgb.b * 262.0) / 4096.0;
    let m = (rgb.r * 683.0 + rgb.g * 2951.0 + rgb.b * 462.0) / 4096.0;
    let s = (rgb.r * 99.0 + rgb.g * 309.0 + rgb.b * 3688.0) / 4096.0;

    let l_pq = gt7_inverse_eotf_st2084(max(l, 0.0));
    let m_pq = gt7_inverse_eotf_st2084(max(m, 0.0));
    let s_pq = gt7_inverse_eotf_st2084(max(s, 0.0));

    return vec3(
        (2048.0 * l_pq + 2048.0 * m_pq) / 4096.0,
        (6610.0 * l_pq - 13613.0 * m_pq + 7003.0 * s_pq) / 4096.0,
        (17933.0 * l_pq - 17390.0 * m_pq - 543.0 * s_pq) / 4096.0,
    );
}

// ICtCp -> linear Rec.2020 RGB (ITU-R BT.2100 / ITU-T T.302). Output channels
// are clamped at zero, mirroring the reference.
fn gt7_ictcp_to_rgb(ictcp: vec3<f32>) -> vec3<f32> {
    let l = ictcp.x + 0.00860904 * ictcp.y + 0.11103 * ictcp.z;
    let m = ictcp.x - 0.00860904 * ictcp.y - 0.11103 * ictcp.z;
    let s = ictcp.x + 0.560031 * ictcp.y - 0.320627 * ictcp.z;

    let l_lin = gt7_eotf_st2084(l);
    let m_lin = gt7_eotf_st2084(m);
    let s_lin = gt7_eotf_st2084(s);

    return vec3(
        max(3.43661 * l_lin - 2.50645 * m_lin + 0.0698454 * s_lin, 0.0),
        max(-0.79133 * l_lin + 1.9836 * m_lin - 0.192271 * s_lin, 0.0),
        max(-0.0259499 * l_lin - 0.0989137 * m_lin + 1.12486 * s_lin, 0.0),
    );
}

// The "GT Tone Mapping" curve (V2) with a convergent shoulder, evaluated per
// channel: a power-curve toe blended into an exactly-linear middle section,
// then an exponential shoulder. The shoulder asymptote (k_a is about
// 1.185 * peak at default parameters) never reaches the operator output
// because `gt7_tone_map` clamps at the peak.
fn gt7_evaluate_curve(x: f32, params: Gt7Params) -> f32 {
    if x < 0.0 {
        return 0.0;
    }

    // WGSL's `smoothstep` takes (edge0, edge1, x), the reference's
    // (x, edge0, edge1). Semantics match for finite inputs and edge0 < edge1.
    let weight_linear = smoothstep(0.0, params.mid_point, x);
    let weight_toe = 1.0 - weight_linear;

    if x < params.linear_section * params.peak {
        let toe_mapped = params.mid_point * pow(x / params.mid_point, params.toe_strength);
        return weight_toe * toe_mapped + weight_linear * x;
    } else {
        // Shoulder mapping for highlights. For extreme inputs exp(x * k_c)
        // underflows to zero (k_c < 0), converging on k_a.
        return params.k_a + params.k_b * exp(x * params.k_c);
    }
}

// The full GT7 tone-mapping pipeline in native units: linear Rec.2020
// frame-buffer values (1.0 = 100 nits) in, tone-mapped linear Rec.2020 out
// ([0, 1] in SDR mode, [0, peak_nits / paper_white_nits] in HDR mode).
fn gt7_tone_map(rgb: vec3<f32>, params: Gt7Params) -> vec3<f32> {
    // Convert to UCS to separate luminance and chroma.
    let ucs = gt7_rgb_to_ictcp(rgb);

    // Per-channel tone mapping ("skewed" color).
    let skewed_rgb = vec3(
        gt7_evaluate_curve(rgb.r, params),
        gt7_evaluate_curve(rgb.g, params),
        gt7_evaluate_curve(rgb.b, params),
    );

    let skewed_ucs = gt7_rgb_to_ictcp(skewed_rgb);

    // Fades chroma as the original luminance approaches the peak. The band
    // extends past it (fade_end > 1) so over-peak colors keep some chroma.
    let chroma_scale = 1.0 - smoothstep(params.fade_start, params.fade_end, ucs.x / params.peak_ucs);

    // Luminance from the skewed color; chroma from the original color, faded.
    let scaled_ucs = vec3(skewed_ucs.x, ucs.yz * chroma_scale);

    let scaled_rgb = gt7_ictcp_to_rgb(scaled_ucs);

    // Final blend between per-channel and UCS-scaled results, clamped at the
    // peak.
    let blended = (1.0 - params.blend_ratio) * skewed_rgb + params.blend_ratio * scaled_rgb;
    return params.sdr_correction_factor * min(blended, vec3(params.peak));
}

// Integration seam for Bevy's tone-mapping pass, dispatched as
// `TONEMAP_METHOD_GRAN_TURISMO_7` from `tone_mapping()` in
// tonemapping.wesl. The CPU reference in gt7.rs covers the operator
// itself (`cpu_reference::apply`), not the four working-space/output-gamut
// combinations below. Those are unverified until a GPU readback test exists.
//
// Contract:
// 1. Input is Bevy's scene-linear working space (1.0 is about SDR paper
//    white). The default working space is Rec.709 and is converted to the
//    operator's native linear Rec.2020. Under WORKING_COLOR_SPACE_REC2020 the
//    input is already native Rec.2020 and the expansion is skipped. This is
//    the one Bevy tonemapper that takes the wide working space without a
//    gamut-clipping entry conversion (see tonemapping.wesl).
// 2. Multiply by 2.5, so Bevy's 1.0 maps to GT7's 250-nit paper white
//    (2.5 frame-buffer units). The input scaling is the same in SDR and HDR
//    mode (paper-white strategy (a)), so the curve always sees Gran Turismo's
//    250-nit-calibrated scene.
// 3. Run the operator. The prepared params uniform selects SDR or HDR mode
//    (see `Gt7ParamsUniform::new` in gt7.rs). The HDR output is relative to
//    paper white.
// 4. Pick the output gamut with the TONEMAP_OUTPUT_REC2020 shader def, pushed
//    exactly when the view's resolved display transfer is HDR (see
//    `ResolvedTonemapping::output_gamut` in tonemapping/mod.rs, the single
//    source of truth shared with the display encoder's input-gamut contract):
//    - Without the def (every SDR view): convert Rec.2020 -> Rec.709 and clamp
//      to [0, 1], because out-of-gamut Rec.2020 results have no SDR Rec.709
//      representation. The conversion is kept under
//      WORKING_COLOR_SPACE_REC2020 too: the pass's SDR output contract is
//      Rec.709 display-linear, which the SDR chain hardware-encodes to sRGB.
//    - With the def (HDR-transfer views): emit the operator's native linear
//      Rec.2020 display-referred output directly, with no back-conversion and
//      no clamp. The display-encoding pass then does the Rec.2020 -> display
//      transform: identity for PQ/Rec.2020, and the Rec.709 contraction plus
//      out-of-gamut compression for scRGB.
fn tone_mapping_gran_turismo_7(color: vec3<f32>) -> vec3<f32> {
@if(GT7_PARAMS_UNIFORM)
    let params = gt7_params_uniform;
@else
    let params = gt7_default_sdr_params();
@if(WORKING_COLOR_SPACE_REC2020)
    let rec2020 = color;
@else
    let rec2020 = GT7_REC_709_TO_REC_2020 * color;
    let fb = rec2020 * (GT7_SDR_PAPER_WHITE / GT7_REFERENCE_LUMINANCE);
    let mapped = gt7_tone_map(fb, params);
@if(TONEMAP_OUTPUT_REC2020)
    return mapped;
@else
    return saturate(GT7_REC_2020_TO_REC_709 * mapped);
}
