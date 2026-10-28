//! The Gran Turismo 7 tonemapping method.
//!
//! Ported from Polyphony Digital's reference implementation, gt7_tone_mapping.cpp
//! (MIT License, Copyright (c) 2025 Polyphony Digital Inc.).
//!
//! The method works on linear Rec.2020 "frame buffer values", where 1.0 is 100 nits.

import bevy_render::{
    transfer_functions::{pq_eotf_to_nits, pq_inverse_eotf_from_nits},
    working_color_space::REC2020_TO_REC709,
};

/// Curve constants. Matches `Gt7ParamsUniform` in gt7.rs.
struct Gt7Params {
    /// Display peak in frame buffer units.
    peak: f32,
    /// The shoulder is k_a + k_b * exp(x * k_c).
    k_a: f32,
    k_b: f32,
    k_c: f32,
    mid_point: f32,
    linear_section: f32,
    toe_strength: f32,
    /// ICtCp intensity of peak white.
    peak_ucs: f32,
    blend_ratio: f32,
    fade_start: f32,
    fade_end: f32,
    /// Scales the output so that 1.0 is paper white.
    output_scale: f32,
    /// Nonzero clamps the Rec.709 output to [0, 1].
    clamp_output: u32,
}

@if(GT7_PARAMS_UNIFORM)
@group(1) @binding(0) var<uniform> gt7_params_uniform: Gt7Params;

/// The parameters for an SDR target at the default `GranTurismo7Params`. Must match
/// `Gt7ParamsUniform::new` in gt7.rs. Materials that tonemap in their shaders use these,
/// since they can't bind the uniform.
const GT7_DEFAULT_SDR_PARAMS = Gt7Params(
    2.5,                 // peak
    2.963333333333334,   // k_a
    -3.3733512380644313, // k_b
    -0.539568345323741,  // k_c
    0.538,               // mid_point
    0.444,               // linear_section
    1.28,                // toe_strength
    0.6025591549907509,  // peak_ucs
    0.6,                 // blend_ratio
    0.98,                // fade_start
    1.16,                // fade_end
    0.4,                 // output_scale
    1u,                  // clamp_output
);

/// Nits per frame buffer unit.
const REFERENCE_LUMINANCE: f32 = 100.0;

/// Gran Turismo's SDR paper white of 250 nits, in frame buffer units.
const SDR_PAPER_WHITE: f32 = 2.5;

/// Unlike the reference, negative LMS values encode as zero instead of NaN. Very
/// saturated colors have them.
fn rgb_to_ictcp(rgb: vec3<f32>) -> vec3<f32> {
    let lms = vec3(
        rgb.r * 1688.0 + rgb.g * 2146.0 + rgb.b * 262.0,
        rgb.r * 683.0 + rgb.g * 2951.0 + rgb.b * 462.0,
        rgb.r * 99.0 + rgb.g * 309.0 + rgb.b * 3688.0,
    ) / 4096.0;
    let pq = pq_inverse_eotf_from_nits(lms * REFERENCE_LUMINANCE);

    return vec3(
        (2048.0 * pq.x + 2048.0 * pq.y) / 4096.0,
        (6610.0 * pq.x - 13613.0 * pq.y + 7003.0 * pq.z) / 4096.0,
        (17933.0 * pq.x - 17390.0 * pq.y - 543.0 * pq.z) / 4096.0,
    );
}

fn ictcp_to_rgb(ictcp: vec3<f32>) -> vec3<f32> {
    let lms_pq = vec3(
        ictcp.x + 0.00860904 * ictcp.y + 0.11103 * ictcp.z,
        ictcp.x - 0.00860904 * ictcp.y - 0.11103 * ictcp.z,
        ictcp.x + 0.560031 * ictcp.y - 0.320627 * ictcp.z,
    );
    let lms = pq_eotf_to_nits(lms_pq) / REFERENCE_LUMINANCE;

    return max(vec3(
        3.43661 * lms.x - 2.50645 * lms.y + 0.0698454 * lms.z,
        -0.79133 * lms.x + 1.9836 * lms.y - 0.192271 * lms.z,
        -0.0259499 * lms.x - 0.0989137 * lms.y + 1.12486 * lms.z,
    ), vec3(0.0));
}

/// A power curve toe, blended into a linear section, then an exponential shoulder.
fn evaluate_curve(x: f32, params: Gt7Params) -> f32 {
    if x < 0.0 {
        return 0.0;
    }

    let weight_linear = smoothstep(0.0, params.mid_point, x);
    let weight_toe = 1.0 - weight_linear;

    if x < params.linear_section * params.peak {
        let toe_mapped = params.mid_point * pow(x / params.mid_point, params.toe_strength);
        return weight_toe * toe_mapped + weight_linear * x;
    }
    return params.k_a + params.k_b * exp(x * params.k_c);
}

fn gt7_tone_map(rgb: vec3<f32>, params: Gt7Params) -> vec3<f32> {
    let ucs = rgb_to_ictcp(rgb);

    let skewed_rgb = vec3(
        evaluate_curve(rgb.r, params),
        evaluate_curve(rgb.g, params),
        evaluate_curve(rgb.b, params),
    );
    let skewed_ucs = rgb_to_ictcp(skewed_rgb);

    // Intensity comes from the curve. Chroma comes from the input, faded out near peak.
    let chroma_scale = 1.0 - smoothstep(params.fade_start, params.fade_end, ucs.x / params.peak_ucs);
    let scaled_rgb = ictcp_to_rgb(vec3(skewed_ucs.x, ucs.yz * chroma_scale));

    let blended = mix(skewed_rgb, scaled_rgb, params.blend_ratio);
    return params.output_scale * min(blended, vec3(params.peak));
}

/// Tonemaps a linear Rec.2020 color to linear Rec.709.
fn tone_mapping_gran_turismo_7(rec2020: vec3<f32>) -> vec3<f32> {
@if(GT7_PARAMS_UNIFORM)
    let params = gt7_params_uniform;
@else
    let params = GT7_DEFAULT_SDR_PARAMS;

    // Bevy's 1.0 maps to Gran Turismo's paper white, on SDR and HDR displays alike.
    let mapped = REC2020_TO_REC709 * gt7_tone_map(rec2020 * SDR_PAPER_WHITE, params);
    if params.clamp_output != 0u {
        return saturate(mapped);
    }
    return mapped;
}
