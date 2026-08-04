//! CPU-side tests for the auto exposure and auto white balance adaptation math.
//!
//! The functions marked as mirrors below copy `auto_exposure.wgsl` operation for
//! operation. If the shader math changes, they must be updated to match.

use super::{
    buffers::{build_uniform, initial_state},
    pipeline::{AutoExposureState, AutoExposureUniform},
    AutoExposure, AutoWhiteBalance, PhysiologicalAdaptation,
};

/// Mirror of the metering bias in `compute_average`.
fn apply_metering_bias(mut avg_lum: f32, settings: &AutoExposureUniform) -> f32 {
    if settings.metering_bias != 0.0 {
        avg_lum += settings.metering_bias;
    }
    avg_lum
}

/// Mirror of the temporal smoothing, long-term envelope update, and two-stage clamp in
/// `compute_average`. Returns the exposure that would be applied to the view.
fn adaptation_step(
    state: &mut AutoExposureState,
    target_exposure: f32,
    delta_time: f32,
    settings: &AutoExposureUniform,
) -> f32 {
    let mut exposure = state.exposure;
    let delta = target_exposure - exposure;
    if target_exposure > exposure {
        let speed_down = settings.speed_down * delta_time;
        let exp_down = speed_down / settings.exponential_transition_distance;
        exposure += f32::min(speed_down, delta * exp_down);
    } else {
        let speed_up = settings.speed_up * delta_time;
        let exp_up = speed_up / settings.exponential_transition_distance;
        exposure += f32::max(-speed_up, delta * exp_up);
    }

    let mut long_term = state.long_term;
    let long_term_delta = exposure - long_term;
    if exposure > long_term {
        let long_term_speed_down = settings.long_term_speed_down * delta_time;
        let long_term_exp_down = long_term_speed_down / settings.exponential_transition_distance;
        long_term += f32::min(long_term_speed_down, long_term_delta * long_term_exp_down);
    } else {
        let long_term_speed_up = settings.long_term_speed_up * delta_time;
        let long_term_exp_up = long_term_speed_up / settings.exponential_transition_distance;
        long_term += f32::max(-long_term_speed_up, long_term_delta * long_term_exp_up);
    }

    if settings.physiological != 0 {
        exposure = exposure.clamp(
            long_term - settings.long_term_bound_down,
            long_term + settings.long_term_bound_up,
        );
    }

    state.exposure = exposure;
    state.long_term = long_term;

    exposure
}

/// The single-stage smoothing, without the two-stage model. Reference for the tests
/// that pin the disabled path to numerically identical output.
fn legacy_single_stage_step(
    exposure: f32,
    target_exposure: f32,
    delta_time: f32,
    settings: &AutoExposureUniform,
) -> f32 {
    let delta = target_exposure - exposure;
    if target_exposure > exposure {
        let speed_down = settings.speed_down * delta_time;
        let exp_down = speed_down / settings.exponential_transition_distance;
        exposure + f32::min(speed_down, delta * exp_down)
    } else {
        let speed_up = settings.speed_up * delta_time;
        let exp_up = speed_up / settings.exponential_transition_distance;
        exposure + f32::max(-speed_up, delta * exp_up)
    }
}

const DT: f32 = 1.0 / 60.0;

fn enabled_settings(adaptation: PhysiologicalAdaptation) -> AutoExposureUniform {
    build_uniform(
        &AutoExposure {
            physiological: Some(adaptation),
            ..Default::default()
        },
        None,
    )
}

/// An [`AutoExposureState`] with the given exposure values and the neutral (D65) white
/// balance chromaticity.
fn ae_state(exposure: f32, long_term: f32) -> AutoExposureState {
    AutoExposureState {
        exposure,
        long_term,
        chroma_x: AWB_D65_XY[0],
        chroma_y: AWB_D65_XY[1],
        chroma_sums: [0; 3],
    }
}

/// A deterministic, wandering target sequence exercising both adaptation directions and
/// both the linear and exponential smoothing regions.
fn target_sequence(step: usize) -> f32 {
    match (step / 120) % 4 {
        0 => 6.0,
        1 => -4.0,
        2 => 0.25,
        _ => -7.5,
    }
}

#[test]
fn gpu_struct_layouts_match_the_wgsl_structs() {
    use bevy_render::render_resource::ShaderType;

    // naga computes span=80 for the WGSL `AutoExposure` uniform (17 4-byte scalars
    // plus three words of tail padding) and span=28 for `AutoExposureState` (four
    // scalars plus a stride-4 array of three atomic words; storage structs are not
    // rounded to 16). The encase layouts must agree.
    assert_eq!(AutoExposureUniform::min_size().get(), 80);
    assert_eq!(AutoExposureState::min_size().get(), 28);
}

#[test]
fn default_component_is_configuration_identical_to_legacy() {
    let settings = AutoExposure::default();

    // The fields that predate the two-stage model keep their documented defaults.
    assert_eq!(settings.range, -8.0..=8.0);
    assert_eq!(settings.filter, 0.10..=0.90);
    assert_eq!(settings.speed_brighten, 3.0);
    assert_eq!(settings.speed_darken, 1.0);
    assert_eq!(settings.exponential_transition_distance, 1.5);

    // The two-stage and bias fields default to neutral.
    assert_eq!(settings.metering_bias, 0.0);
    assert!(settings.physiological.is_none());
}

#[test]
fn default_uniform_is_neutral() {
    let uniform = build_uniform(&AutoExposure::default(), None);

    // The metering values the single-stage implementation uploaded, unchanged.
    assert_eq!(uniform.min_log_lum, -8.0);
    assert_eq!(uniform.inv_log_lum_range, 1.0 / 16.0);
    assert_eq!(uniform.log_lum_range, 16.0);
    assert_eq!(uniform.low_percent, 0.10);
    assert_eq!(uniform.high_percent, 0.90);
    assert_eq!(uniform.speed_up, 3.0);
    assert_eq!(uniform.speed_down, 1.0);
    assert_eq!(uniform.exponential_transition_distance, 1.5);

    // Neutral, so the shader skips both the metering bias and the two-stage clamp.
    assert_eq!(uniform.metering_bias, 0.0);
    assert_eq!(uniform.physiological, 0);

    // The initial GPU state: exposure 0 clamped into range, envelope at the same value.
    let state = initial_state(&AutoExposure::default());
    assert_eq!(state.exposure, 0.0);
    assert_eq!(state.long_term, 0.0);
}

#[test]
fn disabled_two_stage_is_bit_identical_to_legacy_single_stage() {
    let settings = build_uniform(&AutoExposure::default(), None);
    assert_eq!(settings.physiological, 0);

    let mut state = initial_state(&AutoExposure::default());
    let mut legacy_exposure = state.exposure;

    for step in 0..1200 {
        let target = target_sequence(step);
        let applied = adaptation_step(&mut state, target, DT, &settings);
        legacy_exposure = legacy_single_stage_step(legacy_exposure, target, DT, &settings);

        assert_eq!(
            applied.to_bits(),
            legacy_exposure.to_bits(),
            "exposure diverged from the legacy math at step {step}"
        );
    }
}

#[test]
fn long_term_envelope_bounds_short_term_exposure() {
    let adaptation = PhysiologicalAdaptation::default();
    let settings = enabled_settings(adaptation);
    let mut state = ae_state(0.0, 0.0);

    // The scene becomes 10 EV darker, so the short-term stage wants +10 EV at 1 EV/s.
    let target = 10.0;
    for _ in 0..(10.0 / DT) as usize {
        let applied = adaptation_step(&mut state, target, DT, &settings);
        assert!(
            applied <= state.long_term + adaptation.bound_darken + 1e-4,
            "exposure {applied} escaped the long-term bound {}",
            state.long_term + adaptation.bound_darken
        );
    }

    // After 10 simulated seconds the unbounded exposure would be ~9.4 EV. The bound
    // is the envelope (10 s x 0.01 EV/s = 0.1 EV) plus bound_darken (2 EV).
    assert!(
        state.exposure < 3.0,
        "exposure {} was not bounded by the long-term envelope",
        state.exposure
    );
    assert!(
        (state.exposure - (state.long_term + adaptation.bound_darken)).abs() < 1e-4,
        "exposure {} should be saturated at the upper bound {}",
        state.exposure,
        state.long_term + adaptation.bound_darken
    );
}

#[test]
fn bounded_exposure_converges_once_the_envelope_catches_up() {
    let settings = enabled_settings(PhysiologicalAdaptation::default());
    let mut state = ae_state(0.0, 0.0);

    // Simulate 30 minutes of dark adaptation at 10 Hz.
    let target = 10.0;
    let dt = 0.1;
    for _ in 0..18000 {
        adaptation_step(&mut state, target, dt, &settings);
    }

    assert!(
        (state.exposure - target).abs() < 0.05,
        "exposure {} did not converge to the target {target}",
        state.exposure
    );
    assert!(
        (state.long_term - target).abs() < 0.5,
        "long-term envelope {} did not follow the exposure to {target}",
        state.long_term
    );
}

#[test]
fn convergence_within_bounds_is_unaffected_by_the_envelope() {
    // A target inside the envelope's bounds adapts as fast as the single-stage path.
    let settings = enabled_settings(PhysiologicalAdaptation::default());
    let mut state = ae_state(0.0, 0.0);
    let mut legacy_exposure = 0.0;

    let target = 1.5;
    for _ in 0..600 {
        adaptation_step(&mut state, target, DT, &settings);
        legacy_exposure = legacy_single_stage_step(legacy_exposure, target, DT, &settings);
    }

    assert!(
        (state.exposure - legacy_exposure).abs() < 1e-4,
        "in-bounds adaptation ({}) should match the single-stage path ({legacy_exposure})",
        state.exposure
    );
    assert!((state.exposure - target).abs() < 0.05);
}

#[test]
fn light_adaptation_is_faster_than_dark_adaptation() {
    let adaptation = PhysiologicalAdaptation::default();
    let settings = enabled_settings(adaptation);

    // Fully adapted to a dark environment, the scene becomes 8 EV brighter.
    let mut state = ae_state(8.0, 8.0);
    let mut steps_to_brighten = None;
    for step in 0..1_000_000 {
        adaptation_step(&mut state, 0.0, DT, &settings);
        if (state.exposure - 0.0).abs() < 0.25 {
            steps_to_brighten = Some(step);
            break;
        }
    }

    // Fully adapted to a bright environment, the scene becomes 8 EV darker.
    let mut state = ae_state(0.0, 0.0);
    let mut steps_to_darken = None;
    for step in 0..1_000_000 {
        adaptation_step(&mut state, 8.0, DT, &settings);
        if (state.exposure - 8.0).abs() < 0.25 {
            steps_to_darken = Some(step);
            break;
        }
    }

    let (steps_to_brighten, steps_to_darken) = (
        steps_to_brighten.expect("light adaptation never converged"),
        steps_to_darken.expect("dark adaptation never converged"),
    );
    assert!(
        steps_to_brighten < steps_to_darken,
        "light adaptation ({steps_to_brighten} steps) must be faster than dark adaptation \
        ({steps_to_darken} steps)"
    );
}

#[test]
fn long_term_envelope_is_rate_limited() {
    let adaptation = PhysiologicalAdaptation::default();
    let settings = enabled_settings(adaptation);
    let mut state = ae_state(0.0, 0.0);

    let max_speed = adaptation.speed_brighten.max(adaptation.speed_darken);
    for step in 0..2400 {
        let previous = state.long_term;
        adaptation_step(&mut state, target_sequence(step), DT, &settings);
        assert!(
            (state.long_term - previous).abs() <= max_speed * DT + 1e-6,
            "the envelope moved faster than its speed limit at step {step}"
        );
    }
}

#[test]
fn metering_bias_math() {
    // At the neutral bias, the metered value passes through bit-identically.
    let neutral = build_uniform(&AutoExposure::default(), None);
    let avg = -3.7f32;
    assert_eq!(apply_metering_bias(avg, &neutral).to_bits(), avg.to_bits());

    // A positive bias meters the scene as brighter than it measured.
    let settings = build_uniform(
        &AutoExposure {
            metering_bias: 1.0,
            ..Default::default()
        },
        None,
    );
    assert_eq!(apply_metering_bias(3.0, &settings), 4.0);
}

#[test]
fn invalid_inputs_are_sanitized() {
    // A non-finite metering bias is ignored.
    let settings = build_uniform(
        &AutoExposure {
            metering_bias: f32::NAN,
            ..Default::default()
        },
        None,
    );
    assert_eq!(settings.metering_bias, 0.0);

    // Invalid physiological fields are reset to their defaults; valid ones are kept.
    let defaults = PhysiologicalAdaptation::default();
    let settings = enabled_settings(PhysiologicalAdaptation {
        speed_brighten: f32::NAN,
        speed_darken: -1.0,
        bound_brighten: 4.0,
        bound_darken: f32::INFINITY,
    });
    assert_eq!(settings.physiological, 1);
    assert_eq!(settings.long_term_speed_up, defaults.speed_brighten);
    assert_eq!(settings.long_term_speed_down, defaults.speed_darken);
    assert_eq!(settings.long_term_bound_down, 4.0);
    assert_eq!(settings.long_term_bound_up, defaults.bound_darken);
}

#[test]
fn initial_state_starts_neutral_inside_the_metering_range() {
    // The envelope starts at the same neutral exposure as the short-term stage.
    let state = initial_state(&AutoExposure::default());
    assert_eq!(state.exposure, 0.0);
    assert_eq!(state.long_term, state.exposure);

    // A metering range that excludes 0 EV clamps both stages into it.
    let state = initial_state(&AutoExposure {
        range: 2.0..=8.0,
        ..Default::default()
    });
    assert_eq!(state.exposure, 2.0);
    assert_eq!(state.long_term, 2.0);
}

// Auto white balance mirrors. Matrices are stored column-major, like WGSL `mat3x3<f32>`.

/// Mirror of `AWB_D65_XY`.
const AWB_D65_XY: [f32; 2] = [0.31272, 0.32903];

/// Mirror of the `AWB_REC709_TO_{X,Y,Z}` matrix rows.
#[expect(
    clippy::excessive_precision,
    reason = "the literals mirror the WGSL constants digit for digit"
)]
const AWB_REC709_TO_XYZ_ROWS: [[f32; 3]; 3] = [
    [0.4123907992659595, 0.35758433938387796, 0.1804807884018343],
    [0.21263900587151036, 0.7151686787677559, 0.07219231536073371],
    [0.01933081871559185, 0.11919477979462599, 0.9505321522496607],
];

/// Mirror of the `AWB_XYZ_TO_REC709_{R,G,B}` matrix rows.
#[expect(
    clippy::excessive_precision,
    reason = "the literals mirror the WGSL constants digit for digit"
)]
const AWB_XYZ_TO_REC709_ROWS: [[f32; 3]; 3] = [
    [3.2409699419045208, -1.537383177570093, -0.49861076029300311],
    [
        -0.96924363628087962,
        1.8759675015077208,
        0.041555057407175612,
    ],
    [
        0.055630079696993608,
        -0.20397695888897655,
        1.0569715142428784,
    ],
];

/// Mirror of `AWB_RGB_TO_LMS` (columns).
const AWB_RGB_TO_LMS: [[f32; 3]; 3] = [
    [0.311692, 0.0905138, 0.00764433],
    [0.652085, 0.901341, 0.0486554],
    [0.0362225, 0.00814478, 0.943700],
];

/// Mirror of `AWB_LMS_TO_RGB` (columns).
const AWB_LMS_TO_RGB: [[f32; 3]; 3] = [
    [4.06305, -0.40791, -0.0118812],
    [-2.93241, 1.40437, -0.0486532],
    [-0.130646, 0.00353630, 1.0605344],
];

/// `m * v` for a column-major 3x3 matrix.
fn mat3_mul_vec3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// `a * b` for column-major 3x3 matrices.
fn mat3_mul_mat3(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        mat3_mul_vec3(a, b[0]),
        mat3_mul_vec3(a, b[1]),
        mat3_mul_vec3(a, b[2]),
    ]
}

/// Mirror of `awb_xy_to_rec709` (white point `xy` at unit luminance).
fn awb_xy_to_rec709(xy: [f32; 2]) -> [f32; 3] {
    let xyz = [xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]];
    [
        xyz[0] * AWB_XYZ_TO_REC709_ROWS[0][0]
            + xyz[1] * AWB_XYZ_TO_REC709_ROWS[0][1]
            + xyz[2] * AWB_XYZ_TO_REC709_ROWS[0][2],
        xyz[0] * AWB_XYZ_TO_REC709_ROWS[1][0]
            + xyz[1] * AWB_XYZ_TO_REC709_ROWS[1][1]
            + xyz[2] * AWB_XYZ_TO_REC709_ROWS[1][2],
        xyz[0] * AWB_XYZ_TO_REC709_ROWS[2][0]
            + xyz[1] * AWB_XYZ_TO_REC709_ROWS[2][1]
            + xyz[2] * AWB_XYZ_TO_REC709_ROWS[2][2],
    ]
}

/// Mirror of `awb_cct` (the `McCamy` 1992 approximation).
fn awb_cct(xy: [f32; 2]) -> f32 {
    let d = (0.1858 - xy[1]).min(-1e-6);
    let n = ((xy[0] - 0.3320) / d).clamp(-1.2, 1.3);
    ((449.0 * n + 3525.0) * n + 6823.3) * n + 5520.33
}

/// Mirror of `awb_planckian_xy` (Kang et al. 2002).
#[expect(
    clippy::excessive_precision,
    reason = "the literals mirror the WGSL constants digit for digit"
)]
fn awb_planckian_xy(cct: f32) -> [f32; 2] {
    let t = cct;
    let t2 = t * t;
    let t3 = t2 * t;
    let x = if t <= 4000.0 {
        -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.179910
    } else {
        -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390
    };
    let x2 = x * x;
    let x3 = x2 * x;
    let y = if t <= 2222.0 {
        -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
    } else if t <= 4000.0 {
        -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
    } else {
        3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483
    };
    [x, y]
}

/// The CCT-clamped, tint-preserving white point inside `awb_balance_matrix`.
fn awb_white_point(adapted_xy: [f32; 2]) -> [f32; 2] {
    let cct = awb_cct(adapted_xy);
    let locus = awb_planckian_xy(cct.clamp(1667.0, 25000.0));
    let tint = [
        (adapted_xy[0] - locus[0]).clamp(-0.05, 0.05),
        (adapted_xy[1] - locus[1]).clamp(-0.05, 0.05),
    ];
    let out = awb_planckian_xy(cct.clamp(2500.0, 7000.0));
    [out[0] + tint[0], out[1] + tint[1]]
}

/// Mirror of `awb_balance_matrix`.
fn awb_balance_matrix(adapted_xy: [f32; 2]) -> [[f32; 3]; 3] {
    let white_xy = awb_white_point(adapted_xy);
    let d65 = mat3_mul_vec3(&AWB_RGB_TO_LMS, awb_xy_to_rec709(AWB_D65_XY));
    let white = mat3_mul_vec3(&AWB_RGB_TO_LMS, awb_xy_to_rec709(white_xy));
    let gain = [d65[0] / white[0], d65[1] / white[1], d65[2] / white[2]];
    let scaled = [
        [
            gain[0] * AWB_RGB_TO_LMS[0][0],
            gain[1] * AWB_RGB_TO_LMS[0][1],
            gain[2] * AWB_RGB_TO_LMS[0][2],
        ],
        [
            gain[0] * AWB_RGB_TO_LMS[1][0],
            gain[1] * AWB_RGB_TO_LMS[1][1],
            gain[2] * AWB_RGB_TO_LMS[1][2],
        ],
        [
            gain[0] * AWB_RGB_TO_LMS[2][0],
            gain[1] * AWB_RGB_TO_LMS[2][1],
            gain[2] * AWB_RGB_TO_LMS[2][2],
        ],
    ];
    mat3_mul_mat3(&AWB_LMS_TO_RGB, &scaled)
}

/// The CIE 1931 xy chromaticity and luminance of a linear Rec.709 color, mirroring
/// the per-pixel measurement in `compute_histogram`.
fn rec709_to_xy_lum(rgb: [f32; 3]) -> ([f32; 2], f32) {
    let x = mat3_mul_vec3(
        &[
            [
                AWB_REC709_TO_XYZ_ROWS[0][0],
                AWB_REC709_TO_XYZ_ROWS[1][0],
                AWB_REC709_TO_XYZ_ROWS[2][0],
            ],
            [
                AWB_REC709_TO_XYZ_ROWS[0][1],
                AWB_REC709_TO_XYZ_ROWS[1][1],
                AWB_REC709_TO_XYZ_ROWS[2][1],
            ],
            [
                AWB_REC709_TO_XYZ_ROWS[0][2],
                AWB_REC709_TO_XYZ_ROWS[1][2],
                AWB_REC709_TO_XYZ_ROWS[2][2],
            ],
        ],
        rgb,
    );
    let sum = x[0] + x[1] + x[2];
    ([x[0] / sum, x[1] / sum], x[1])
}

/// Mirror of the anchor blend and temporal adaptation in `compute_average`. `sums` are
/// the drained accumulators `(x*Y, y*Y, Y)` and `lum_scale` converts them to scene-linear.
fn awb_adaptation_step(
    state: &mut AutoExposureState,
    sums: [f32; 3],
    lum_scale: f32,
    delta_time: f32,
    settings: &AutoExposureUniform,
) {
    let scene_luminance = sums[2] * lum_scale;
    let denom = scene_luminance + settings.awb_anchor;
    if denom > 0.0 {
        let target_x = (sums[0] * lum_scale + settings.awb_anchor * AWB_D65_XY[0]) / denom;
        let target_y = (sums[1] * lum_scale + settings.awb_anchor * AWB_D65_XY[1]) / denom;
        let alpha = (settings.awb_speed * delta_time).clamp(0.0, 1.0);
        state.chroma_x += (target_x - state.chroma_x) * alpha;
        state.chroma_y += (target_y - state.chroma_y) * alpha;
    }
}

fn awb_settings(white_balance: AutoWhiteBalance) -> AutoExposureUniform {
    build_uniform(&AutoExposure::default(), Some(&white_balance))
}

#[test]
fn mccamy_matches_known_illuminants() {
    // D65 has a CCT of ~6504 K (McCamy's published accuracy near the locus is a few K).
    let d65 = awb_cct(AWB_D65_XY);
    assert!(
        (d65 - 6504.0).abs() < 15.0,
        "CCT of D65 was {d65}, expected ~6504 K"
    );

    // CIE standard illuminant A (tungsten): xy = (0.44757, 0.40745), CCT 2856 K.
    let a = awb_cct([0.44757, 0.40745]);
    assert!(
        (a - 2856.0).abs() < 15.0,
        "CCT of illuminant A was {a}, expected ~2856 K"
    );
}

#[test]
fn deep_blue_scenes_read_as_cool_not_warm() {
    // A saturated blue scene adapts below McCamy's y = 0.1858 epicenter (this fixture
    // is Rec.709 blue blended with the default D65 anchor). The CCT must stay at the
    // cool end; a denominator sign flip would read it as ~1632 K and boost blue.
    let adapted_xy = [0.1853, 0.1184];
    let cct = awb_cct(adapted_xy);
    assert!(
        cct > 7000.0,
        "deep-blue chromaticity produced CCT {cct} K, expected the cool extreme"
    );

    // The correction must warm the image: red and green roughly preserved, blue
    // strongly attenuated. A sign flip instead produces a blue gain of ~2.
    let matrix = awb_balance_matrix(adapted_xy);
    let corrected_white = mat3_mul_vec3(&matrix, [1.0, 1.0, 1.0]);
    assert!(
        (0.9..1.3).contains(&corrected_white[0]) && (0.9..1.2).contains(&corrected_white[1]),
        "red/green of corrected white drifted to {:?}",
        corrected_white
    );
    assert!(
        corrected_white[2] < 0.8,
        "white balance must attenuate blue on a deep-blue scene, got gain {}",
        corrected_white[2]
    );
}

#[test]
fn planckian_locus_round_trips_through_mccamy() {
    for cct in [2500.0f32, 3000.0, 4000.0, 5000.0, 6500.0, 7000.0] {
        let xy = awb_planckian_xy(cct);
        let round_trip = awb_cct(xy);
        assert!(
            (round_trip - cct).abs() < 0.015 * cct,
            "Planckian {cct} K round-tripped to {round_trip} K"
        );
    }

    // D65 sits just off the Planckian locus at ~6504 K.
    let locus = awb_planckian_xy(6504.0);
    let dx = locus[0] - AWB_D65_XY[0];
    let dy = locus[1] - AWB_D65_XY[1];
    let distance = (dx * dx + dy * dy).sqrt();
    assert!(
        distance < 0.006,
        "Planckian 6504 K is {distance} away from D65 in xy"
    );
}

#[test]
fn cct_output_is_clamped_to_the_camera_range() {
    // A 2000 K scene is clamped to the 2500 K lower bound.
    let white = awb_white_point(awb_planckian_xy(2000.0));
    let clamped = awb_cct(white);
    assert!(
        (clamped - 2500.0).abs() < 0.02 * 2500.0,
        "2000 K clamped to {clamped}, expected ~2500 K"
    );

    // A 10000 K scene is clamped to the 7000 K upper bound.
    let white = awb_white_point(awb_planckian_xy(10000.0));
    let clamped = awb_cct(white);
    assert!(
        (clamped - 7000.0).abs() < 0.02 * 7000.0,
        "10000 K clamped to {clamped}, expected ~7000 K"
    );

    // Chromaticities inside the range pass through exactly: the white point is
    // reconstructed as locus + (xy - locus) with the same locus on both sides.
    for cct in [2700.0f32, 4500.0, 6500.0] {
        let xy = awb_planckian_xy(cct);
        let white = awb_white_point(xy);
        assert_eq!(white, xy, "in-range {cct} K must pass through unchanged");
    }
}

#[test]
fn tint_is_preserved_and_clamped() {
    // A small off-locus tint within the in-range CCT band passes through exactly.
    let locus = awb_planckian_xy(5000.0);
    let tinted = [locus[0], locus[1] + 0.02];
    assert_eq!(awb_white_point(tinted), tinted);

    // A huge off-locus tint is clamped to +/-0.05 around the locus of the measured CCT.
    let extreme = [locus[0], locus[1] + 0.2];
    let white = awb_white_point(extreme);
    let cct = awb_cct(extreme);
    let reference = awb_planckian_xy(cct.clamp(2500.0, 7000.0));
    assert!((white[1] - reference[1] - 0.05).abs() < 1e-6);
}

#[test]
fn virtual_light_anchor_stabilizes_dark_scenes() {
    let settings = awb_settings(AutoWhiteBalance {
        speed: 4.0,
        virtual_light_anchor: 0.01,
    });

    // A pitch-black scene measures nothing; the anchor pulls the white point to D65.
    let mut state = ae_state(0.0, 0.0);
    state.chroma_x = 0.45;
    state.chroma_y = 0.40;
    for _ in 0..600 {
        awb_adaptation_step(&mut state, [0.0, 0.0, 0.0], 1.0, DT, &settings);
    }
    assert!((state.chroma_x - AWB_D65_XY[0]).abs() < 1e-3);
    assert!((state.chroma_y - AWB_D65_XY[1]).abs() < 1e-3);

    // A bright scene overwhelms the anchor and converges to the measurement.
    let measured = [0.40f32, 0.38];
    let luminance = 10.0f32; // 1000x the anchor.
    let sums = [measured[0] * luminance, measured[1] * luminance, luminance];
    let mut state = ae_state(0.0, 0.0);
    for _ in 0..600 {
        awb_adaptation_step(&mut state, sums, 1.0, DT, &settings);
    }
    assert!((state.chroma_x - measured[0]).abs() < 1e-4);
    assert!((state.chroma_y - measured[1]).abs() < 1e-4);

    // When the scene luminance equals the anchor luminance, the target is the
    // luminance-weighted midpoint between the measurement and D65.
    let mut state = ae_state(0.0, 0.0);
    let sums = [measured[0] * 0.01, measured[1] * 0.01, 0.01];
    for _ in 0..2400 {
        awb_adaptation_step(&mut state, sums, 1.0, DT, &settings);
    }
    assert!((state.chroma_x - (measured[0] + AWB_D65_XY[0]) / 2.0).abs() < 1e-4);
    assert!((state.chroma_y - (measured[1] + AWB_D65_XY[1]) / 2.0).abs() < 1e-4);

    // With the anchor disabled and nothing measured, the state holds still.
    let settings = awb_settings(AutoWhiteBalance {
        speed: 4.0,
        virtual_light_anchor: 0.0,
    });
    let mut state = ae_state(0.0, 0.0);
    state.chroma_x = 0.45;
    state.chroma_y = 0.40;
    awb_adaptation_step(&mut state, [0.0, 0.0, 0.0], 1.0, DT, &settings);
    assert_eq!((state.chroma_x, state.chroma_y), (0.45, 0.40));
}

#[test]
fn chromaticity_adaptation_is_rate_limited_and_convergent() {
    let settings = awb_settings(AutoWhiteBalance {
        speed: 0.5,
        virtual_light_anchor: 0.0,
    });
    let measured = [0.42f32, 0.39];
    let sums = [measured[0], measured[1], 1.0];

    // A single small step moves by exactly alpha * remaining.
    let mut state = ae_state(0.0, 0.0);
    let expected = AWB_D65_XY[0] + (measured[0] - AWB_D65_XY[0]) * (0.5 * DT).clamp(0.0, 1.0);
    awb_adaptation_step(&mut state, sums, 1.0, DT, &settings);
    assert!((state.chroma_x - expected).abs() < 1e-7);

    // It converges over time...
    for _ in 0..3600 {
        awb_adaptation_step(&mut state, sums, 1.0, DT, &settings);
    }
    assert!((state.chroma_x - measured[0]).abs() < 1e-4);
    assert!((state.chroma_y - measured[1]).abs() < 1e-4);

    // ...and a huge speed * delta_time never overshoots: it lands exactly on target.
    let settings = awb_settings(AutoWhiteBalance {
        speed: 1000.0,
        virtual_light_anchor: 0.0,
    });
    let mut state = ae_state(0.0, 0.0);
    awb_adaptation_step(&mut state, sums, 1.0, 1.0, &settings);
    assert_eq!((state.chroma_x, state.chroma_y), (measured[0], measured[1]));
}

#[test]
fn balance_matrix_is_identity_at_d65() {
    let m = awb_balance_matrix(AWB_D65_XY);
    for (column, column_values) in m.iter().enumerate() {
        for (row, value) in column_values.iter().enumerate() {
            let expected = if row == column { 1.0 } else { 0.0 };
            assert!(
                (value - expected).abs() < 5e-5,
                "balance[{column}][{row}] = {value}, expected {expected}"
            );
        }
    }
}

#[test]
fn balance_matrix_neutralizes_a_warm_illuminant() {
    // CIE standard illuminant A (tungsten, CCT 2856 K, inside the 2500 K - 7000 K
    // output clamp so the correction is unclamped) lighting a white scene, in linear
    // Rec.709 at unit luminance. Its raw R/B channel ratio is ~7.9.
    let illuminant = awb_xy_to_rec709([0.44757, 0.40745]);
    let ([x, y], luminance) = rec709_to_xy_lum(illuminant);

    // Let the adaptation converge onto the measured chromaticity.
    let settings = awb_settings(AutoWhiteBalance {
        speed: 4.0,
        virtual_light_anchor: 0.001,
    });
    let mut state = ae_state(0.0, 0.0);
    let sums = [x * luminance, y * luminance, luminance];
    for _ in 0..3600 {
        awb_adaptation_step(&mut state, sums, 1.0, DT, &settings);
    }

    // The correction must push the illuminant towards neutral and reduce the spread.
    let m = awb_balance_matrix([state.chroma_x, state.chroma_y]);
    let corrected = mat3_mul_vec3(&m, illuminant);
    let spread_before = illuminant[0] / illuminant[2];
    let spread_after = corrected[0] / corrected[2];
    assert!(
        spread_after < spread_before,
        "correction did not reduce the warm cast: {spread_before} -> {spread_after}"
    );
    // Deriving the gains in the same LMS basis they are applied in leaves a residual
    // R/B ratio of ~1.0003 (D65_XY's 5-digit rounding) plus ~0.1% of anchor pull.
    // Deriving them in raw CAM16 LMS would leave ~1.54 instead.
    assert!(
        (spread_after - 1.0).abs() < 0.01,
        "corrected illuminant {corrected:?} is not neutral (R/B {spread_after})"
    );
    let max = corrected[0].max(corrected[1]).max(corrected[2]);
    let min = corrected[0].min(corrected[1]).min(corrected[2]);
    assert!(
        max / min < 1.01,
        "corrected illuminant {corrected:?} has a residual channel spread"
    );
    // The correction preserves the white point's luminance.
    let (_, corrected_luminance) = rec709_to_xy_lum(corrected);
    assert!(
        (corrected_luminance - luminance).abs() < 0.01,
        "correction changed the illuminant's luminance: {luminance} -> {corrected_luminance}"
    );
}

#[test]
fn awb_uniform_defaults_and_sanitization() {
    // Without the component, every auto white balance field is neutral.
    let uniform = build_uniform(&AutoExposure::default(), None);
    assert_eq!(uniform.awb_enabled, 0);
    assert_eq!(uniform.awb_speed, 0.0);
    assert_eq!(uniform.awb_anchor, 0.0);

    // With the component, the sanitized values are uploaded.
    let uniform = build_uniform(&AutoExposure::default(), Some(&AutoWhiteBalance::default()));
    assert_eq!(uniform.awb_enabled, 1);
    assert_eq!(uniform.awb_speed, AutoWhiteBalance::default().speed);
    assert_eq!(
        uniform.awb_anchor,
        AutoWhiteBalance::default().virtual_light_anchor
    );

    // Invalid fields are reset to their defaults.
    let uniform = build_uniform(
        &AutoExposure::default(),
        Some(&AutoWhiteBalance {
            speed: f32::NAN,
            virtual_light_anchor: -1.0,
        }),
    );
    assert_eq!(uniform.awb_enabled, 1);
    assert_eq!(uniform.awb_speed, AutoWhiteBalance::default().speed);
    assert_eq!(
        uniform.awb_anchor,
        AutoWhiteBalance::default().virtual_light_anchor
    );
}

#[test]
fn initial_chromaticity_is_d65() {
    let state = initial_state(&AutoExposure::default());
    assert_eq!(state.chroma_x, AWB_D65_XY[0]);
    assert_eq!(state.chroma_y, AWB_D65_XY[1]);
}
