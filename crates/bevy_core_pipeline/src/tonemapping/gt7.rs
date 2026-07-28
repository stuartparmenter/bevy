//! CPU-side reference implementation of the Gran Turismo 7 tone-mapping operator.
//!
//! This is a native port of Polyphony Digital's reference implementation
//! (`gt7_tone_mapping.cpp`, MIT License, Copyright (c) 2025 Polyphony Digital Inc.),
//! published as part of the SIGGRAPH 2025 course "Physically Based Tone Mapping in
//! Gran Turismo 7".
//!
//! # Unit convention (native)
//!
//! The operator works on linear Rec.2020 RGB "frame buffer values" where `1.0`
//! corresponds to [`REFERENCE_LUMINANCE`] (100 cd/m²) of physical luminance:
//!
//! - In SDR mode the operator tone-maps against Gran Turismo's 250-nit SDR paper
//!   white ([`GRAN_TURISMO_SDR_PAPER_WHITE`]) and rescales the result by
//!   `1 / 2.5` so the output fits `[0, 1]`, ready for the sRGB OETF.
//! - In HDR mode the output range is `[0, peak_nits / 100]`, ready for the PQ
//!   inverse EOTF. HDR peak luminance is valid in the range 250–10000 nits.
//!
//! The GPU implementation lives in `gt7.wgsl` and must be kept in sync with this
//! module; the math here is the parity reference for `cargo test` fixtures
//! (tolerance `1e-4` per channel against the C++ reference output) and for future
//! GPU readback tests (tolerance `1/1024` for fp16 targets).
//!
//! All math is deliberately `f32` to mirror both the C++ reference and the WGSL
//! port operation-for-operation.

use bevy_camera::Camera;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{Has, With},
    reflect::ReflectComponent,
    system::{Commands, Query},
};
use bevy_log::warn_once;
use bevy_math::ops;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    extract_component::ExtractComponent,
    render_resource::ShaderType,
    view::{ViewDisplayTarget, ViewTarget},
    RenderApp,
};
use bevy_window::DisplayTarget;

use super::{effective_tonemapping, gt7_params_uniform_active, Tonemapping};
use crate::camera_stack::{StackRole, ViewStackContract};

/// Physical luminance in cd/m² that a linear frame-buffer value of `1.0`
/// corresponds to in Gran Turismo's native unit convention.
pub const REFERENCE_LUMINANCE: f32 = 100.0;

/// The SDR reference (paper) white level used by Gran Turismo's tone mapping,
/// in cd/m². This is Polyphony's artistic calibration, not sRGB's 80/100 nits.
pub const GRAN_TURISMO_SDR_PAPER_WHITE: f32 = 250.0;

/// The lowest HDR peak luminance, in nits, the GT7 operator supports.
///
/// The reference implementation documents 250 nits as the valid lower bound
/// (the curve parameters assume a 250-nit SDR paper white) but does not
/// enforce it; Bevy clamps to it at prepare time and warns.
pub const GT7_MIN_HDR_PEAK_NITS: f32 = 250.0;

/// The highest HDR peak luminance, in nits, the GT7 operator supports
/// (the PQ ceiling). Clamped to at prepare time, with a warning.
pub const GT7_MAX_HDR_PEAK_NITS: f32 = 10000.0;

/// Full-precision (f64-derived) linear Rec.709 → Rec.2020 conversion matrix
/// (row-major, D65 white point, derived per ITU-R BT.2087).
///
/// Used by the in-tree SDR integration seam: Bevy's working space is currently
/// scene-linear Rec.709, while the GT7 operator natively works in linear
/// Rec.2020.
///
/// Each literal is the shortest round-trip representation of the correctly
/// rounded `f32` of the corresponding f64 literal in `gt7.wgsl`
/// (`GT7_REC_709_TO_REC_2020`, note: column-major there); the two must stay
/// bit-identical so this module remains an exact parity reference for the
/// shader.
// The shared engine-wide constants live in
// `bevy_render::working_color_space` (`REC709_TO_REC2020`, column-major
// `Mat3`); the row-major copies here are the parity-fixture format and are
// bit-identity-locked against the shared constants by
// `matrices_bitwise_match_bevy_render_working_color_space`.
pub const REC_709_TO_REC_2020: [[f32; 3]; 3] = [
    [0.627_403_9, 0.329_283_03, 0.043_313_067],
    [0.069_097_29, 0.919_540_4, 0.011_362_315],
    [0.016_391_44, 0.088_013_306, 0.895_595_25],
];

/// Full-precision (f64-derived) linear Rec.2020 → Rec.709 conversion matrix
/// (row-major, D65 white point). Inverse of [`REC_709_TO_REC_2020`].
///
/// Bit-identical to `GT7_REC_2020_TO_REC_709` in `gt7.wgsl` (see
/// [`REC_709_TO_REC_2020`] for the sync contract).
pub const REC_2020_TO_REC_709: [[f32; 3]; 3] = [
    [1.660_491, -0.587_641_1, -0.072_849_86],
    [-0.124_550_48, 1.132_899_9, -0.008_349_422],
    [-0.018_150_763, -0.100_578_9, 1.118_729_7],
];

/// Per-camera parameters for the [`Tonemapping::GranTurismo7`] operator.
///
/// Defaults match Polyphony Digital's reference implementation. All parameters
/// are dimensionless except where noted; the curve parameters are expressed in
/// GT7's native frame-buffer units where `1.0` = 100 nits (see the module docs
/// for the unit contract).
///
/// Add this component to a camera that uses [`Tonemapping::GranTurismo7`] to
/// customize the operator. Whenever the view's tonemapping pipeline binds the
/// prepared parameters (`gt7_params_uniform_active` in the parent module:
/// when this component is present, and always on HDR-transfer targets),
/// [`queue_gt7_params_uniforms`] validates the values with
/// [`Self::sanitized`] each frame and produces a [`Gt7ParamsUniform`] that
/// replaces the shader's baked defaults — falling back to [`Self::default`]
/// for cameras without the component. Cameras **without** this component on
/// SDR targets keep using the shader's baked SDR defaults.
///
/// The operator's mode follows the view's resolved [`DisplayTarget`], with
/// or without this component: on a target that requests an HDR transfer, the
/// uniform is computed in HDR mode (peak taken from
/// [`DisplayTarget::peak_luminance_nits`]); otherwise in
/// SDR mode. See [`Gt7ParamsUniform::new`] for the exact rules.
///
/// [`Tonemapping::GranTurismo7`]: crate::tonemapping::Tonemapping::GranTurismo7
#[derive(Component, Debug, Clone, Copy, PartialEq, Reflect, ExtractComponent)]
#[extract_component_filter(With<Camera>)]
#[reflect(Component, Debug, Default, PartialEq, Clone)]
#[extract_app(RenderApp)]
pub struct GranTurismo7Params {
    /// Mix between the per-channel tone-mapped color and the hue-preserving
    /// UCS (`ICtCp`) processed color. `0.0` = fully per-channel ("camera-like"
    /// skew), `1.0` = fully UCS (hue-stable). Polyphony markets this as the
    /// main artistic dial. Clamped to `[0, 1]`. Default: `0.6`.
    pub blend_ratio: f32,
    /// Start of the highlight chroma fade band, as a fraction of the peak
    /// luminance in UCS (`ICtCp` `I`) units. Original-luminance values above this
    /// begin losing chroma. Clamped to `[0, 1]`. Default: `0.98`.
    pub fade_start: f32,
    /// End of the highlight chroma fade band, as a fraction of the peak
    /// luminance in UCS units. Values above this are fully desaturated.
    /// Intentionally allowed to exceed `1.0` so over-peak colors keep some
    /// chroma; must be greater than [`Self::fade_start`]. Default: `1.16`.
    pub fade_end: f32,
    /// Curvature control for the shoulder region. Must be less than `1.0`.
    /// Default: `0.25`.
    pub alpha: f32,
    /// Gray point in frame-buffer units: the end of the toe→linear blend
    /// region. The curve is exactly linear from here to the shoulder. Must be
    /// greater than `0.0`. Default: `0.538`.
    pub mid_point: f32,
    /// Fraction of the peak intensity at which the linear section ends and the
    /// convergent shoulder begins. Must be less than `1.0`. Default: `0.444`.
    pub linear_section: f32,
    /// Exponent of the toe's power curve. Must be non-negative.
    /// Default: `1.28`.
    pub toe_strength: f32,
}

impl Default for GranTurismo7Params {
    fn default() -> Self {
        Self {
            blend_ratio: 0.6,
            fade_start: 0.98,
            fade_end: 1.16,
            alpha: 0.25,
            mid_point: 0.538,
            linear_section: 0.444,
            toe_strength: 1.28,
        }
    }
}

impl GranTurismo7Params {
    /// The smallest allowed width of the chroma fade band
    /// (`fade_end - fade_start`); prevents a division by zero in the
    /// `smoothstep` underlying the chroma fade.
    const MIN_FADE_BAND: f32 = 1e-4;
    /// Margin keeping `alpha` and `linear_section` strictly below `1.0`,
    /// preventing divisions by zero in the shoulder constant derivation.
    const UNIT_MARGIN: f32 = 1e-3;

    /// Returns a copy with all parameters validated and clamped to safe ranges,
    /// emitting [`warn_once!`] if anything had to be adjusted.
    ///
    /// This implements the prepare-time validation table for the GT7 operator:
    ///
    /// - Any non-finite (NaN/∞) field is reset to its default.
    /// - `blend_ratio` is clamped to `[0, 1]`.
    /// - `fade_start` is clamped to `[0, 1]`.
    /// - `fade_end` is clamped to at least `fade_start + 1e-4`. The upper bound
    ///   is intentionally NOT clamped: `fade_end > 1` lets over-peak colors
    ///   keep some chroma.
    /// - `alpha` and `linear_section` are clamped to `[0, 1 - 1e-3]` (values of
    ///   exactly `1.0` produce divisions by zero in the closed-form shoulder
    ///   constants).
    /// - `mid_point` is clamped to at least `1e-3` (a zero mid point produces
    ///   divisions by zero in the toe).
    /// - `toe_strength` is clamped to be non-negative.
    ///
    /// Called by [`queue_gt7_params_uniforms`] before the parameters reach
    /// the GPU.
    pub fn sanitized(&self) -> Self {
        let defaults = Self::default();
        let mut sanitized = *self;
        let mut adjusted = false;

        // Reset non-finite fields to their defaults first so the range clamps
        // below operate on real numbers.
        let fields = [
            (&mut sanitized.blend_ratio, defaults.blend_ratio),
            (&mut sanitized.fade_start, defaults.fade_start),
            (&mut sanitized.fade_end, defaults.fade_end),
            (&mut sanitized.alpha, defaults.alpha),
            (&mut sanitized.mid_point, defaults.mid_point),
            (&mut sanitized.linear_section, defaults.linear_section),
            (&mut sanitized.toe_strength, defaults.toe_strength),
        ];
        for (field, default) in fields {
            if !field.is_finite() {
                *field = default;
                adjusted = true;
            }
        }

        let mut clamp = |value: &mut f32, min: f32, max: f32| {
            let clamped = value.clamp(min, max);
            if clamped != *value {
                *value = clamped;
                adjusted = true;
            }
        };

        clamp(&mut sanitized.blend_ratio, 0.0, 1.0);
        clamp(&mut sanitized.fade_start, 0.0, 1.0);
        clamp(&mut sanitized.alpha, 0.0, 1.0 - Self::UNIT_MARGIN);
        clamp(&mut sanitized.linear_section, 0.0, 1.0 - Self::UNIT_MARGIN);
        clamp(&mut sanitized.mid_point, Self::UNIT_MARGIN, f32::MAX);
        clamp(&mut sanitized.toe_strength, 0.0, f32::MAX);
        // No upper clamp on `fade_end`: values past 1.0 are intentional.
        clamp(
            &mut sanitized.fade_end,
            sanitized.fade_start + Self::MIN_FADE_BAND,
            f32::MAX,
        );

        if adjusted {
            warn_once!(
                "GranTurismo7Params contained out-of-range or non-finite values \
                 and was sanitized; see the GranTurismo7Params docs for valid ranges"
            );
        }

        sanitized
    }
}

/// `smoothstep` with the C++ reference's exact semantics: strict comparisons,
/// and the interpolant computed before the range checks (so `edge0 == edge1`
/// yields NaN/∞ rather than a clamp — parameter validation prevents that).
fn smooth_step(x: f32, edge0: f32, edge1: f32) -> f32 {
    let t = (x - edge0) / (edge1 - edge0);
    if x < edge0 {
        return 0.0;
    }
    if x > edge1 {
        return 1.0;
    }
    t * t * (3.0 - 2.0 * t)
}

/// Luminance-driven chroma fade: `1.0` below `a`, falling to `0.0` at `b`.
fn chroma_curve(x: f32, a: f32, b: f32) -> f32 {
    1.0 - smooth_step(x, a, b)
}

// ST-2084 (PQ) constants, SMPTE ST 2084:2014 / ITU-R BT.2100.
const PQ_M1: f32 = 0.159_301_76; // (2610 / 4096) / 4
const PQ_M2: f32 = 78.84375; // (2523 / 4096) * 128
const PQ_C1: f32 = 0.835_937_5; // 3424 / 4096
const PQ_C2: f32 = 18.851_563; // (2413 / 4096) * 32
const PQ_C3: f32 = 18.6875; // (2392 / 4096) * 32
/// Maximum luminance supported by PQ (cd/m²).
const PQ_C: f32 = 10000.0;

/// ST-2084 (PQ) EOTF: normalized PQ signal (clamped to `[0, 1]`) → linear
/// frame-buffer value (`1.0` = 100 nits).
fn eotf_st2084(n: f32) -> f32 {
    let n = n.clamp(0.0, 1.0);
    let np = ops::powf(n, 1.0 / PQ_M2);
    let mut l = np - PQ_C1;
    if l < 0.0 {
        l = 0.0;
    }
    l /= PQ_C2 - PQ_C3 * np;
    l = ops::powf(l, 1.0 / PQ_M1);
    // Convert absolute luminance (cd/m²) into the frame-buffer linear scale.
    l * PQ_C / REFERENCE_LUMINANCE
}

/// ST-2084 (PQ) inverse EOTF: linear frame-buffer value (`1.0` = 100 nits) →
/// normalized PQ signal.
///
/// Deliberately does NOT clamp its input (mirroring the reference): values
/// above 10000 nits encode above `1.0`. Negative inputs would produce NaN;
/// callers clamp at zero before calling (see [`rgb_to_ictcp`]).
fn inverse_eotf_st2084(v: f32) -> f32 {
    let physical = v * REFERENCE_LUMINANCE;
    let y = physical / PQ_C;
    let ym = ops::powf(y, PQ_M1);
    // Numerically-stabler form of ((c1 + c2*ym) / (1 + c3*ym))^m2.
    ops::exp2(PQ_M2 * (ops::log2(PQ_C1 + PQ_C2 * ym) - ops::log2(1.0 + PQ_C3 * ym)))
}

/// Linear Rec.2020 RGB → `ICtCp` (ITU-R BT.2100 / ITU-T T.302).
///
/// Deviation from the C++ reference: the LMS intermediates are clamped at zero
/// before the PQ encode. The reference would produce NaN for inputs saturated
/// enough to drive LMS negative; the clamp is the recommended port policy and
/// matches the WGSL implementation. All parity fixtures keep LMS positive, so
/// fixture outputs are unaffected.
fn rgb_to_ictcp(rgb: [f32; 3]) -> [f32; 3] {
    let l = (rgb[0] * 1688.0 + rgb[1] * 2146.0 + rgb[2] * 262.0) / 4096.0;
    let m = (rgb[0] * 683.0 + rgb[1] * 2951.0 + rgb[2] * 462.0) / 4096.0;
    let s = (rgb[0] * 99.0 + rgb[1] * 309.0 + rgb[2] * 3688.0) / 4096.0;

    let l_pq = inverse_eotf_st2084(l.max(0.0));
    let m_pq = inverse_eotf_st2084(m.max(0.0));
    let s_pq = inverse_eotf_st2084(s.max(0.0));

    [
        (2048.0 * l_pq + 2048.0 * m_pq) / 4096.0,
        (6610.0 * l_pq - 13613.0 * m_pq + 7003.0 * s_pq) / 4096.0,
        (17933.0 * l_pq - 17390.0 * m_pq - 543.0 * s_pq) / 4096.0,
    ]
}

/// `ICtCp` → linear Rec.2020 RGB (ITU-R BT.2100 / ITU-T T.302).
///
/// Output channels are clamped at zero, mirroring the reference. The PQ decode
/// in [`eotf_st2084`] clamps its input to `[0, 1]`, so per-LMS-channel values
/// saturate at 10000 nits.
fn ictcp_to_rgb(ictcp: [f32; 3]) -> [f32; 3] {
    let l = ictcp[0] + 0.00860904 * ictcp[1] + 0.11103 * ictcp[2];
    let m = ictcp[0] - 0.00860904 * ictcp[1] - 0.11103 * ictcp[2];
    let s = ictcp[0] + 0.560031 * ictcp[1] - 0.320627 * ictcp[2];

    let l_lin = eotf_st2084(l);
    let m_lin = eotf_st2084(m);
    let s_lin = eotf_st2084(s);

    [
        (3.43661 * l_lin - 2.50645 * m_lin + 0.0698454 * s_lin).max(0.0),
        (-0.79133 * l_lin + 1.9836 * m_lin - 0.192271 * s_lin).max(0.0),
        (-0.0259499 * l_lin - 0.0989137 * m_lin + 1.12486 * s_lin).max(0.0),
    ]
}

/// The "GT Tone Mapping" curve (V2) with a convergent shoulder, evaluated
/// per channel: a power-curve toe blended into an exactly-linear middle
/// section, followed by a convergent exponential shoulder.
///
/// The shoulder asymptote (`k_a ≈ 1.185 × peak` at default parameters) lies
/// above the peak; it is never visible in operator output because
/// [`Gt7ToneMapping::apply`] clamps at the peak.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gt7ToneMappingCurve {
    /// Display peak intensity in frame-buffer units (`peak_nits / 100`).
    pub peak_intensity: f32,
    /// Shoulder curvature control (must be `< 1`).
    pub alpha: f32,
    /// Gray point: end of the toe→linear blend (frame-buffer units).
    pub mid_point: f32,
    /// Fraction of `peak_intensity` where the shoulder begins.
    pub linear_section: f32,
    /// Toe power-curve exponent.
    pub toe_strength: f32,
    /// Precomputed shoulder asymptote.
    pub k_a: f32,
    /// Precomputed shoulder scale.
    pub k_b: f32,
    /// Precomputed shoulder exponent factor (negative).
    pub k_c: f32,
}

impl Gt7ToneMappingCurve {
    /// Initializes the curve, precomputing the closed-form shoulder constants.
    ///
    /// `monitor_intensity` is the display peak in frame-buffer units
    /// (`peak_nits / 100`); `gray_point` is the curve's mid point. The GT7
    /// defaults are `alpha = 0.25`, `gray_point = 0.538`,
    /// `linear_section = 0.444`, `toe_strength = 1.28`.
    pub fn new(
        monitor_intensity: f32,
        alpha: f32,
        gray_point: f32,
        linear_section: f32,
        toe_strength: f32,
    ) -> Self {
        let k = (linear_section - 1.0) / (alpha - 1.0);
        Self {
            peak_intensity: monitor_intensity,
            alpha,
            mid_point: gray_point,
            linear_section,
            toe_strength,
            k_a: monitor_intensity * linear_section + monitor_intensity * k,
            k_b: -monitor_intensity * k * ops::exp(linear_section / k),
            k_c: -1.0 / (k * monitor_intensity),
        }
    }

    /// Evaluates the curve at `x` (frame-buffer units). Negative inputs map
    /// to zero. With default parameters the regions are: toe→linear blend on
    /// `(0, mid_point)`, exactly linear on `[mid_point, linear_section × peak)`,
    /// shoulder on `[linear_section × peak, ∞)`.
    pub fn evaluate(&self, x: f32) -> f32 {
        if x < 0.0 {
            return 0.0;
        }

        let weight_linear = smooth_step(x, 0.0, self.mid_point);
        let weight_toe = 1.0 - weight_linear;

        // Shoulder mapping for highlights. For extreme inputs `exp(x * k_c)`
        // underflows cleanly to zero (`k_c < 0`), converging on `k_a`.
        let shoulder = self.k_a + self.k_b * ops::exp(x * self.k_c);

        if x < self.linear_section * self.peak_intensity {
            let toe_mapped = self.mid_point * ops::powf(x / self.mid_point, self.toe_strength);
            weight_toe * toe_mapped + weight_linear * x
        } else {
            shoulder
        }
    }
}

/// CPU implementation of the full GT7 tone-mapping pipeline (curve +
/// hue-preserving `ICtCp` branch). See the module docs for the unit contract.
///
/// This mirrors the `GT7ToneMapping` struct in the C++ reference: construct it
/// once per parameter set ([`Self::new_sdr`] / [`Self::new_hdr`]), then call
/// [`Self::apply`] per color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gt7ToneMapping {
    /// `1 / 2.5` in SDR mode (rescales the 250-nit-referred result into
    /// `[0, 1]`); `1.0` in HDR mode.
    pub sdr_correction_factor: f32,
    /// Display target peak in frame-buffer units (`peak_nits / 100`).
    pub framebuffer_luminance_target: f32,
    /// `ICtCp` `I` of peak white, precomputed at initialization; normalizes the
    /// luminance driving the chroma fade.
    pub framebuffer_luminance_target_ucs: f32,
    /// The per-channel tone-mapping curve.
    pub curve: Gt7ToneMappingCurve,
    /// UCS share of the final per-channel/UCS blend (constant, not
    /// chroma-dependent).
    pub blend_ratio: f32,
    /// Chroma fade band start, as a fraction of peak luminance in UCS units.
    pub fade_start: f32,
    /// Chroma fade band end (may exceed `1.0`).
    pub fade_end: f32,
}

impl Gt7ToneMapping {
    fn new(physical_target_luminance: f32, params: &GranTurismo7Params, sdr: bool) -> Self {
        let framebuffer_luminance_target = physical_target_luminance / REFERENCE_LUMINANCE;
        let curve = Gt7ToneMappingCurve::new(
            framebuffer_luminance_target,
            params.alpha,
            params.mid_point,
            params.linear_section,
            params.toe_strength,
        );
        let peak_ucs = rgb_to_ictcp([
            framebuffer_luminance_target,
            framebuffer_luminance_target,
            framebuffer_luminance_target,
        ])[0];
        Self {
            sdr_correction_factor: if sdr {
                1.0 / framebuffer_luminance_target
            } else {
                1.0
            },
            framebuffer_luminance_target,
            framebuffer_luminance_target_ucs: peak_ucs,
            curve,
            blend_ratio: params.blend_ratio,
            fade_start: params.fade_start,
            fade_end: params.fade_end,
        }
    }

    /// Initializes for SDR output with the default parameters: tone-maps
    /// against Gran Turismo's 250-nit paper white, then rescales the result by
    /// `1 / 2.5` so the output fits `[0, 1]`, ready for the sRGB OETF.
    pub fn new_sdr() -> Self {
        Self::new_sdr_with_params(&GranTurismo7Params::default())
    }

    /// Initializes for SDR output with custom [`GranTurismo7Params`]. The
    /// parameters are used as-is; call [`GranTurismo7Params::sanitized`] first
    /// for untrusted values.
    pub fn new_sdr_with_params(params: &GranTurismo7Params) -> Self {
        Self::new(GRAN_TURISMO_SDR_PAPER_WHITE, params, true)
    }

    /// Initializes for HDR output with the default parameters, targeting the
    /// given display peak luminance in nits (valid range 250–10000; the lower
    /// limit exists because the curve parameters assume a 250-nit SDR paper
    /// white). Output range is `[0, physical_target_luminance / 100]`.
    pub fn new_hdr(physical_target_luminance: f32) -> Self {
        Self::new_hdr_with_params(physical_target_luminance, &GranTurismo7Params::default())
    }

    /// Initializes for HDR output with custom [`GranTurismo7Params`]. The
    /// parameters are used as-is; call [`GranTurismo7Params::sanitized`] first
    /// for untrusted values.
    pub fn new_hdr_with_params(
        physical_target_luminance: f32,
        params: &GranTurismo7Params,
    ) -> Self {
        Self::new(physical_target_luminance, params, false)
    }

    /// Whether every precomputed constant the operator evaluates with is
    /// finite.
    ///
    /// [`GranTurismo7Params::sanitized`] bounds each parameter individually,
    /// but combinations can still overflow the closed-form shoulder and toe
    /// math — e.g. a `linear_section` close to 1 sends
    /// `exp(linear_section / k)` past `f32::MAX`, and a non-finite shoulder
    /// constant turns every highlight into NaN. The toe is probed at the
    /// shoulder seam, its largest input.
    pub fn has_finite_curve(&self) -> bool {
        let curve = &self.curve;
        let seam = curve.linear_section * curve.peak_intensity;
        let toe_probe = curve.mid_point * ops::powf(seam / curve.mid_point, curve.toe_strength);
        curve.k_a.is_finite()
            && curve.k_b.is_finite()
            && curve.k_c.is_finite()
            && toe_probe.is_finite()
            && self.framebuffer_luminance_target_ucs.is_finite()
    }

    /// Applies the tone-mapping pipeline to a linear Rec.2020 frame-buffer
    /// color (native GT7 units, `1.0` = 100 nits).
    ///
    /// Steps: per-channel curve ("skewed" color); chroma fade driven by the
    /// ORIGINAL color's UCS luminance; recombination of skewed luminance with
    /// faded original chroma; constant 60% UCS / 40% per-channel blend; clamp
    /// at peak; SDR correction factor.
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        // Convert to UCS to separate luminance and chroma.
        let ucs = rgb_to_ictcp(rgb);

        // Per-channel tone mapping ("skewed" color).
        let skewed_rgb = [
            self.curve.evaluate(rgb[0]),
            self.curve.evaluate(rgb[1]),
            self.curve.evaluate(rgb[2]),
        ];

        let skewed_ucs = rgb_to_ictcp(skewed_rgb);

        let chroma_scale = chroma_curve(
            ucs[0] / self.framebuffer_luminance_target_ucs,
            self.fade_start,
            self.fade_end,
        );

        let scaled_ucs = [
            skewed_ucs[0],         // Luminance from the skewed color.
            ucs[1] * chroma_scale, // Chroma from the original color, faded.
            ucs[2] * chroma_scale,
        ];

        let scaled_rgb = ictcp_to_rgb(scaled_ucs);

        // Final blend between per-channel and UCS-scaled results.
        let mut out = [0.0; 3];
        for i in 0..3 {
            let blended =
                (1.0 - self.blend_ratio) * skewed_rgb[i] + self.blend_ratio * scaled_rgb[i];
            out[i] = self.sdr_correction_factor * blended.min(self.framebuffer_luminance_target);
        }
        out
    }

    /// Applies the operator to a Bevy scene-linear Rec.709 color, mirroring the
    /// WGSL `tone_mapping_gran_turismo_7` integration seam used for SDR output
    /// (no `TONEMAP_OUTPUT_REC2020` shader def). This is the CPU parity
    /// reference for GPU readback tests of the SDR path.
    ///
    /// SDR-output steps:
    /// 1. Convert scene-linear Rec.709 → Rec.2020.
    /// 2. Multiply by `2.5`: Bevy's `1.0` (SDR paper white) maps to GT7's
    ///    250-nit paper white (`2.5` frame-buffer units).
    /// 3. Run the operator (SDR mode brings the result back to `[0, 1]` via
    ///    the `1 / 2.5` correction factor).
    /// 4. Convert the display-referred result Rec.2020 → Rec.709 for the
    ///    sRGB output chain, clamping to `[0, 1]` (out-of-gamut Rec.2020
    ///    results have no SDR Rec.709 representation).
    ///
    /// On HDR-transfer targets the pass instead emits the operator's native
    /// Rec.2020 output for the display encoder — see
    /// [`Self::apply_bevy_scene_linear_rec2020_output`].
    pub fn apply_bevy_scene_linear_sdr(&self, rgb: [f32; 3]) -> [f32; 3] {
        let rec2020 = mat3_mul_vec3(&REC_709_TO_REC_2020, rgb);
        let fb = [
            rec2020[0] * (GRAN_TURISMO_SDR_PAPER_WHITE / REFERENCE_LUMINANCE),
            rec2020[1] * (GRAN_TURISMO_SDR_PAPER_WHITE / REFERENCE_LUMINANCE),
            rec2020[2] * (GRAN_TURISMO_SDR_PAPER_WHITE / REFERENCE_LUMINANCE),
        ];
        let mapped = self.apply(fb);
        let rec709 = mat3_mul_vec3(&REC_2020_TO_REC_709, mapped);
        [
            rec709[0].clamp(0.0, 1.0),
            rec709[1].clamp(0.0, 1.0),
            rec709[2].clamp(0.0, 1.0),
        ]
    }

    /// CPU mirror of the `tone_mapping_gran_turismo_7` WGSL wrapper under the
    /// `WORKING_COLOR_SPACE_REC2020` shader def (gt7.wgsl), i.e. the SDR seam
    /// when Bevy's working color space is
    /// `bevy_render::working_color_space::WorkingColorSpace::Rec2020`.
    ///
    /// Identical to [`Self::apply_bevy_scene_linear_sdr`] except that the
    /// input is already native linear Rec.2020 (the working space), so the
    /// Rec.709 → Rec.2020 input expansion is skipped. The output
    /// back-conversion to Rec.709 (+ clamp) is unchanged: the tone mapping
    /// pass's SDR output contract is Rec.709 display-linear in every working
    /// space (the sRGB chain hardware-encodes from it). HDR-transfer targets
    /// take the native-output seam instead — see
    /// [`Self::apply_bevy_scene_linear_rec2020_native_rec2020_output`].
    pub fn apply_bevy_scene_linear_sdr_rec2020_native(&self, rgb: [f32; 3]) -> [f32; 3] {
        let fb = [
            rgb[0] * (GRAN_TURISMO_SDR_PAPER_WHITE / REFERENCE_LUMINANCE),
            rgb[1] * (GRAN_TURISMO_SDR_PAPER_WHITE / REFERENCE_LUMINANCE),
            rgb[2] * (GRAN_TURISMO_SDR_PAPER_WHITE / REFERENCE_LUMINANCE),
        ];
        let mapped = self.apply(fb);
        let rec709 = mat3_mul_vec3(&REC_2020_TO_REC_709, mapped);
        [
            rec709[0].clamp(0.0, 1.0),
            rec709[1].clamp(0.0, 1.0),
            rec709[2].clamp(0.0, 1.0),
        ]
    }

    /// CPU mirror of the `tone_mapping_gran_turismo_7` WGSL wrapper under the
    /// `TONEMAP_OUTPUT_REC2020` shader def (the HDR-native output path,
    /// Rec.709 working space): scene-linear Rec.709 in, the operator's native
    /// linear Rec.2020 display-referred output out — **no** Rec.2020 →
    /// Rec.709 back-conversion and **no** clamp. The display-encoding pass
    /// consumes this output (its per-view input-gamut contract is keyed by
    /// the same predicate that pushes the shader def; see
    /// `tonemap_output_gamut` in `tonemapping/mod.rs`).
    ///
    /// With the paper-white renormalization folded into
    /// [`Self::sdr_correction_factor`] (`100 / paper_white_nits`, as prepared
    /// by [`Gt7ParamsUniform::new`] for HDR targets), the output range is
    /// `[0, peak_nits / paper_white_nits]` — paper-white-relative
    /// display-linear (`1.0` = paper white), the convention the encoder
    /// expects.
    ///
    /// This is the CPU parity reference for GPU readback tests of the GT7
    /// HDR path under the default Rec.709 working space.
    pub fn apply_bevy_scene_linear_rec2020_output(&self, rgb: [f32; 3]) -> [f32; 3] {
        let rec2020 = mat3_mul_vec3(&REC_709_TO_REC_2020, rgb);
        self.apply_bevy_scene_linear_rec2020_native_rec2020_output(rec2020)
    }

    /// CPU mirror of the `tone_mapping_gran_turismo_7` WGSL wrapper under
    /// **both** the `WORKING_COLOR_SPACE_REC2020` and `TONEMAP_OUTPUT_REC2020`
    /// shader defs: native linear Rec.2020 in (the wide working space, no
    /// input expansion), native linear Rec.2020 display-referred out (no
    /// back-conversion, no clamp). See
    /// [`Self::apply_bevy_scene_linear_rec2020_output`] for the output
    /// contract.
    ///
    /// Apart from the Bevy-side `× 2.5` input scaling (Bevy `1.0` → GT7's
    /// 250-nit paper white), this **is** the reference operator: feeding it
    /// `fb / 2.5` reproduces the C++ harness fixtures directly (locked by
    /// the `hdr_native_rec2020_output_matches_cpp_fixtures_directly` test).
    pub fn apply_bevy_scene_linear_rec2020_native_rec2020_output(&self, rgb: [f32; 3]) -> [f32; 3] {
        let fb = [
            rgb[0] * (GRAN_TURISMO_SDR_PAPER_WHITE / REFERENCE_LUMINANCE),
            rgb[1] * (GRAN_TURISMO_SDR_PAPER_WHITE / REFERENCE_LUMINANCE),
            rgb[2] * (GRAN_TURISMO_SDR_PAPER_WHITE / REFERENCE_LUMINANCE),
        ];
        self.apply(fb)
    }
}

fn mat3_mul_vec3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// GPU uniform feeding the GT7 operator's `Gt7Params` WGSL struct (see
/// `gt7.wgsl`; field order and meaning must stay identical).
///
/// All derived curve constants (`k_a`/`k_b`/`k_c`, `peak_ucs`) are computed
/// CPU-side (closed forms in [`Gt7ToneMappingCurve::new`] /
/// [`Gt7ToneMapping::new_sdr_with_params`] and friends) so the shader stays
/// cheap.
///
/// This is also a render-world component: [`queue_gt7_params_uniforms`] puts
/// one on each view that needs it via [`Gt7ParamsUniform::new`], and
/// [`UniformComponentPlugin`](bevy_render::extract_component::UniformComponentPlugin)
/// packs those into the dynamic uniform buffer the tonemapping pass binds —
/// only when the `GT7_PARAMS_UNIFORM` shader def is pushed. Without that def
/// the shader keeps its baked SDR defaults (`gt7_default_sdr_params()` in
/// `gt7.wgsl`).
#[derive(Component, Clone, Copy, Debug, PartialEq, ShaderType)]
pub struct Gt7ParamsUniform {
    /// Display peak in frame-buffer units (`peak_nits / 100`).
    pub peak: f32,
    /// Precomputed shoulder asymptote ([`Gt7ToneMappingCurve::k_a`]).
    pub k_a: f32,
    /// Precomputed shoulder scale ([`Gt7ToneMappingCurve::k_b`]).
    pub k_b: f32,
    /// Precomputed shoulder exponent factor ([`Gt7ToneMappingCurve::k_c`],
    /// negative).
    pub k_c: f32,
    /// Gray point in frame-buffer units (end of the toe→linear blend).
    pub mid_point: f32,
    /// Fraction of `peak` where the linear section ends and the shoulder
    /// begins.
    pub linear_section: f32,
    /// Exponent of the toe's power curve.
    pub toe_strength: f32,
    /// `ICtCp` `I` of peak white, precomputed at prepare time; normalizes the
    /// luminance driving the chroma fade.
    pub peak_ucs: f32,
    /// UCS share of the final per-channel/UCS blend.
    pub blend_ratio: f32,
    /// Chroma fade band start, as a fraction of `peak_ucs`.
    pub fade_start: f32,
    /// Chroma fade band end (may exceed `1.0`).
    pub fade_end: f32,
    /// Post-clamp output scale. `1 / 2.5` in SDR mode (Polyphony's native
    /// rescale of the 250-nit-referred result into `[0, 1]`);
    /// `100 / paper_white_nits` in HDR mode (the paper-white renormalization
    /// at the operator/encoder seam, so `1.0` = paper white at the operator
    /// output — identity at the default 100-nit paper white).
    pub sdr_correction_factor: f32,
}

impl From<&Gt7ToneMapping> for Gt7ParamsUniform {
    fn from(tone_mapping: &Gt7ToneMapping) -> Self {
        Self {
            peak: tone_mapping.framebuffer_luminance_target,
            k_a: tone_mapping.curve.k_a,
            k_b: tone_mapping.curve.k_b,
            k_c: tone_mapping.curve.k_c,
            mid_point: tone_mapping.curve.mid_point,
            linear_section: tone_mapping.curve.linear_section,
            toe_strength: tone_mapping.curve.toe_strength,
            peak_ucs: tone_mapping.framebuffer_luminance_target_ucs,
            blend_ratio: tone_mapping.blend_ratio,
            fade_start: tone_mapping.fade_start,
            fade_end: tone_mapping.fade_end,
            sdr_correction_factor: tone_mapping.sdr_correction_factor,
        }
    }
}

impl Gt7ParamsUniform {
    /// Builds the uniform for a view from its (unsanitized) user parameters
    /// and resolved [`DisplayTarget`], implementing the prepare-time
    /// validation policy:
    ///
    /// - `params` is passed through [`GranTurismo7Params::sanitized`]
    ///   (non-finite fields reset, ranges clamped, one warning).
    /// - If the display target requests an HDR transfer (scRGB-linear, PQ, or
    ///   extended-range sRGB), the operator is configured in **HDR mode**:
    ///   - `paper_white_nits` is sanitized through
    ///     [`DisplayTarget::sanitized_paper_white_nits`] (non-finite or
    ///     non-positive → 100 nits, clamped to the 10000-nit PQ ceiling, each
    ///     with a warning) — the same method the display pipeline's uniform
    ///     writer uses, so the operator and the display encoder always fold
    ///     the identical paper-white value; non-finite
    ///     `peak_luminance_nits` is reset to 100 (with a warning);
    ///   - the peak is clamped to
    ///     [`[GT7_MIN_HDR_PEAK_NITS, GT7_MAX_HDR_PEAK_NITS]`](GT7_MIN_HDR_PEAK_NITS),
    ///     with a warning;
    ///   - a peak below `paper_white_nits` is raised to `paper_white_nits`,
    ///     with a warning;
    ///   - [`Gt7ParamsUniform::sdr_correction_factor`] is set to
    ///     `100 / paper_white_nits`: the seam renormalization that scales the
    ///     operator's native output (`1.0` = 100 nits) so `1.0` = paper white.
    ///     On these views the tonemapping pipeline is specialized with the
    ///     `TONEMAP_OUTPUT_REC2020` shader def (same HDR predicate), so the
    ///     operator emits this paper-white-relative output in its native
    ///     Rec.2020 primaries — unclamped, `[0, peak / paper_white]` — for
    ///     the display-encoding pass.
    /// - Otherwise the operator is configured in **SDR mode**, identical to
    ///   the baked defaults except for the user parameters: peak 2.5
    ///   frame-buffer units (Gran Turismo's 250-nit paper white), output
    ///   rescaled into `[0, 1]`.
    pub fn new(display_target: &DisplayTarget, params: &GranTurismo7Params) -> Self {
        let params = params.sanitized();
        // Single-source the HDR predicate with the rest of the display
        // pipeline (`DisplayTransfer::is_hdr`, which also backs
        // `ViewDisplayTarget::is_hdr_transfer`). Callers pass the *resolved*
        // display target, so a downgraded HDR request configures plain SDR
        // mode here too.
        if !display_target.transfer.is_hdr() {
            let mut tone_mapping = Gt7ToneMapping::new_sdr_with_params(&params);
            if !tone_mapping.has_finite_curve() {
                warn_once!(
                    "GranTurismo7Params produce tone-mapping curve constants that overflow \
                     f32 (the closed-form shoulder/toe math is not finite for this \
                     combination); falling back to the default parameters"
                );
                tone_mapping = Gt7ToneMapping::new_sdr();
            }
            return Self::from(&tone_mapping);
        }

        // Single-sourced with the display pipeline's uniform producer
        // (`prepare_view_display_targets` in bevy_render): the operator's
        // seam renormalization (× 100 / paper_white) and the display
        // encoder's transfer encoding (× paper_white / 80 for scRGB,
        // × paper_white for PQ) must fold the IDENTICAL paper-white value or
        // the paper-white factors fail to cancel for degenerate/out-of-range
        // inputs.
        let paper_white = display_target.sanitized_paper_white_nits();
        if !display_target.paper_white_nits.is_finite() || display_target.paper_white_nits <= 0.0 {
            warn_once!(
                "DisplayTarget::paper_white_nits is non-finite or non-positive; \
                 GranTurismo7 is using 100 nits instead"
            );
        } else if paper_white < display_target.paper_white_nits {
            warn_once!(
                "DisplayTarget::paper_white_nits exceeds the PQ ceiling of 10000 nits; \
                 GranTurismo7 is clamping it to 10000 nits"
            );
        }

        let mut peak = display_target.peak_luminance_nits;
        if !peak.is_finite() {
            warn_once!(
                "DisplayTarget::peak_luminance_nits is non-finite; \
                 GranTurismo7 is using 100 nits before range clamping"
            );
            peak = DisplayTarget::SDR_SRGB.peak_luminance_nits;
        }
        let clamped_peak = peak.clamp(GT7_MIN_HDR_PEAK_NITS, GT7_MAX_HDR_PEAK_NITS);
        if clamped_peak != peak {
            warn_once!(
                "DisplayTarget::peak_luminance_nits is outside GranTurismo7's supported \
                 HDR range [250, 10000] nits and was clamped"
            );
            peak = clamped_peak;
        }
        if peak < paper_white {
            warn_once!(
                "DisplayTarget::peak_luminance_nits is below paper_white_nits; \
                 GranTurismo7 is raising the peak to paper white"
            );
            peak = paper_white;
        }

        let mut tone_mapping = Gt7ToneMapping::new_hdr_with_params(peak, &params);
        if !tone_mapping.has_finite_curve() {
            warn_once!(
                "GranTurismo7Params produce tone-mapping curve constants that overflow \
                 f32 (the closed-form shoulder/toe math is not finite for this \
                 combination); falling back to the default parameters"
            );
            tone_mapping = Gt7ToneMapping::new_hdr(peak);
        }
        // Paper-white renormalization at the operator/encoder seam: rescale
        // the operator's native output (1.0 = 100 nits) so that 1.0 = paper
        // white. `apply` multiplies by this factor after the peak clamp, so
        // the CPU struct stays a valid parity reference for the shader.
        tone_mapping.sdr_correction_factor = REFERENCE_LUMINANCE / paper_white;
        Self::from(&tone_mapping)
    }
}

/// Gives a [`Gt7ParamsUniform`] to every view whose tonemapping pipeline
/// binds one, and takes it away from every view that stops qualifying.
///
/// [`UniformComponentPlugin`](bevy_render::extract_component::UniformComponentPlugin)
/// — registered for [`Gt7ParamsUniform`] in the tonemapping plugin — packs
/// the components this system inserts into
/// [`ComponentUniforms<Gt7ParamsUniform>`](bevy_render::extract_component::ComponentUniforms)
/// and gives each view a
/// [`DynamicUniformIndex<Gt7ParamsUniform>`](bevy_render::extract_component::DynamicUniformIndex)
/// addressing its entry, which the tonemapping node binds as the pass's
/// dynamic offset.
///
/// A view qualifies when [`gt7_params_uniform_active`] holds for it. The
/// uniform is then built from the camera's [`GranTurismo7Params`] if present
/// and [`GranTurismo7Params::default`] otherwise, with the SDR/HDR mode and
/// the HDR peak taken from the view's resolved [`ViewDisplayTarget`] (see
/// [`Gt7ParamsUniform::new`]).
///
/// Views authored with `GranTurismo7` but without the component on SDR
/// targets keep the shader's baked SDR defaults, and a stack member whose
/// tone-mapping pass is deferred to its finalizer ([`StackRole::Deferred`])
/// never runs the pass at all — neither gets a uniform. That absence is what
/// `prepare_view_tonemapping_pipelines` reads to leave `GT7_PARAMS_UNIFORM`
/// off the pipeline key, so the two systems cannot disagree about the pass's
/// bind group layout.
///
/// Runs in [`RenderSystems::Queue`](bevy_render::RenderSystems::Queue): after
/// `PrepareViews`, where [`ViewDisplayTarget`] and [`ViewStackContract`] are
/// resolved, and before `Prepare`, where the uniform packing and the pipeline
/// specialization both read what this system wrote.
pub fn queue_gt7_params_uniforms(
    mut commands: Commands,
    views: Query<
        (
            Entity,
            // Optional so a camera that drops its `Tonemapping` still
            // reaches the removal branch below instead of keeping a stale
            // `Gt7ParamsUniform`. `prepare_view_tonemapping_pipelines`, which
            // reads the resulting flag, likewise defaults a missing component
            // to `Tonemapping::None`.
            Option<&Tonemapping>,
            Option<&GranTurismo7Params>,
            Option<&ViewDisplayTarget>,
            Option<&ViewStackContract>,
            Has<Gt7ParamsUniform>,
        ),
        // The liveness gate for `ViewStackContract`, which is overwritten in
        // place and never removed; also the view set
        // `prepare_view_tonemapping_pipelines` specializes over.
        With<ViewTarget>,
    >,
) {
    for (entity, tonemapping, params, view_display_target, contract, has_uniform) in &views {
        // Cameras stacked on a shared main texture tone-map once, on the
        // stack's finalizer; the deferred members never run the pass.
        let deferred =
            contract.is_some_and(|contract| matches!(contract.tonemap, StackRole::Deferred(_)));
        let active = !deferred
            && gt7_params_uniform_active(
                effective_tonemapping(tonemapping, view_display_target),
                params.is_some(),
                view_display_target.is_some_and(ViewDisplayTarget::is_hdr_transfer),
            );

        if !active {
            // Render-world entities are retained across frames, so a view
            // that stops qualifying must have the component actively removed
            // — but only if it has one, so plain SDR views issue no command.
            if has_uniform {
                commands.entity(entity).remove::<Gt7ParamsUniform>();
            }
            continue;
        }

        let display_target = view_display_target
            .map(|view_display_target| view_display_target.resolved)
            .unwrap_or_default();
        let params = params.copied().unwrap_or_default();
        commands
            .entity(entity)
            .insert(Gt7ParamsUniform::new(&display_target, &params));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-channel absolute tolerance for CPU-port-vs-C++-reference parity.
    const TOLERANCE: f32 = 1e-4;

    #[track_caller]
    fn assert_rgb_eq(actual: [f32; 3], expected: [f32; 3]) {
        for i in 0..3 {
            assert!(
                (actual[i] - expected[i]).abs() <= TOLERANCE,
                "channel {i}: actual {:?} vs expected {:?} (diff {:e})",
                actual,
                expected,
                (actual[i] - expected[i]).abs()
            );
        }
    }

    /// Ground-truth fixtures generated by compiling the unmodified C++
    /// reference (`plans/gt7_tone_mapping.cpp`, g++ -O2 -std=c++17) with a
    /// `printf("%.9e")` harness over the canonical `main()` cases
    /// (SDR + HDR 1000/4000/10000 over three inputs) plus branch-coverage
    /// extras. Inputs are linear Rec.2020 frame-buffer values.
    #[test]
    fn cpp_parity_canonical_12() {
        let sdr = Gt7ToneMapping::new_sdr();
        let hdr1000 = Gt7ToneMapping::new_hdr(1000.0);
        let hdr4000 = Gt7ToneMapping::new_hdr(4000.0);
        let hdr10000 = Gt7ToneMapping::new_hdr(10000.0);

        let inputs: [[f32; 3]; 3] = [[0.5, 1.23, 0.75], [12.3, 34.3, 56.9], [1504.7, 64.51, 0.5]];

        // SDR (peak 250 nits internal, output rescaled into [0, 1]).
        assert_rgb_eq(
            sdr.apply(inputs[0]),
            [1.996_225e-1, 4.907_029_6e-1, 2.995_677_6e-1],
        );
        assert_rgb_eq(sdr.apply(inputs[1]), [1.0, 1.0, 1.0]);
        assert_rgb_eq(sdr.apply(inputs[2]), [1.0, 1.0, 7.387_512e-1]);

        // HDR, 1000-nit peak.
        assert_rgb_eq(
            hdr1000.apply(inputs[0]),
            [4.998_231_8e-1, 1.230_000_7, 7.499_952_3e-1],
        );
        assert_rgb_eq(hdr1000.apply(inputs[1]), [10.0, 10.0, 10.0]);
        assert_rgb_eq(hdr1000.apply(inputs[2]), [10.0, 10.0, 6.706_747_5]);

        // HDR, 4000-nit peak.
        assert_rgb_eq(
            hdr4000.apply(inputs[0]),
            [4.998_231_8e-1, 1.230_000_7, 7.499_952_3e-1],
        );
        assert_rgb_eq(hdr4000.apply(inputs[1]), [11.354_192, 30.071_972, 40.0]);
        assert_rgb_eq(hdr4000.apply(inputs[2]), [40.0, 40.0, 23.847_712]);

        // HDR, 10000-nit peak (peak UCS is exactly 1.0: PQ(10000 nits) = 1).
        assert_rgb_eq(
            hdr10000.apply(inputs[0]),
            [4.998_231_8e-1, 1.230_000_7, 7.499_952_3e-1],
        );
        assert_rgb_eq(
            hdr10000.apply(inputs[1]),
            [12.277_842, 34.240_51, 56.395_88],
        );
        assert_rgb_eq(
            hdr10000.apply(inputs[2]),
            [91.726_71, 68.024_31, 42.575_134],
        );
    }

    #[test]
    fn cpp_parity_branch_coverage_extras() {
        let sdr = Gt7ToneMapping::new_sdr();
        let hdr1000 = Gt7ToneMapping::new_hdr(1000.0);

        // Mid-band chroma fade (chromaScale ≈ 0.55): one clamped + two
        // unclamped channels.
        assert_rgb_eq(
            hdr1000.apply([20.0, 15.0, 5.0]),
            [10.0, 9.912_016, 5.249_325_3],
        );

        // Achromatic in the SDR fade band: UCS path is ~no-op on gray (tiny
        // channel asymmetry comes from the f32 ICtCp matrices, itself a
        // parity probe).
        assert_rgb_eq(
            sdr.apply([3.0, 3.0, 3.0]),
            [9.179_522e-1, 9.179_485_4e-1, 9.179_471e-1],
        );

        // Exact seam values: R == mid_point, G == linear_section * peak
        // (shoulder branch via strict <), B == linear_section.
        assert_rgb_eq(
            sdr.apply([0.538, 1.11, 0.444]),
            [2.151_980_8e-1, 4.439_385_5e-1, 1.772_700_5e-1],
        );

        // Negative channel: curve's x < 0 branch; LMS of this input stays
        // positive so the (deviating) LMS clamp is not engaged and the result
        // matches the unmodified reference.
        assert_rgb_eq(
            sdr.apply([-0.1, 0.2, 0.1]),
            [0.0, 7.863_592e-2, 3.668_311_2e-2],
        );

        // Bevy SDR paper-white anchor: 2.5 fb gray (250 nits).
        assert_rgb_eq(
            sdr.apply([2.5, 2.5, 2.5]),
            [8.351_579e-1, 8.351_547e-1, 8.351_534e-1],
        );

        // Tiny gray: toe pow + PQ near-zero (`l < 0` clamp in the EOTF).
        assert_rgb_eq(
            hdr1000.apply([1e-5, 1e-5, 1e-5]),
            [4.735_695_7e-7, 4.735_677_5e-7, 4.735_67e-7],
        );
    }

    #[test]
    fn cpp_parity_custom_params_identity_region() {
        // blend = 0 (pure per-channel), toe_strength = 1 makes the toe exactly
        // linear, so inputs below the shoulder seam (0.3 * 10 = 3.0 fb) pass
        // through unchanged.
        let params = GranTurismo7Params {
            blend_ratio: 0.0,
            alpha: 0.5,
            mid_point: 0.4,
            linear_section: 0.3,
            toe_strength: 1.0,
            ..Default::default()
        };
        let custom = Gt7ToneMapping::new_hdr_with_params(1000.0, &params);
        assert_rgb_eq(custom.apply([0.5, 1.23, 0.75]), [0.5, 1.23, 0.75]);
    }

    #[test]
    fn cpp_parity_init_products() {
        // Closed-form curve constants and precomputed peak UCS, from the C++
        // reference harness (%.9e).
        let sdr = Gt7ToneMapping::new_sdr();
        assert!((sdr.curve.k_a - 2.963_333_1).abs() < 1e-5);
        assert!((sdr.curve.k_b - -3.373_351).abs() < 1e-5);
        assert!((sdr.curve.k_c - -5.395_683_6e-1).abs() < 1e-6);
        assert!((sdr.framebuffer_luminance_target_ucs - 6.025_607_6e-1).abs() < 1e-5);
        assert!((sdr.sdr_correction_factor - 0.4).abs() < 1e-7);

        let hdr10000 = Gt7ToneMapping::new_hdr(10000.0);
        // PQ(10000 nits) = 1 exactly.
        assert!((hdr10000.framebuffer_luminance_target_ucs - 1.0).abs() < 1e-6);
        assert_eq!(hdr10000.sdr_correction_factor, 1.0);
    }

    #[test]
    fn curve_matches_reference_values() {
        // Curve-only values from the C++ reference harness; covers each
        // curve region (toe, exact-linear region, shoulder, asymptote).
        let curve25 = Gt7ToneMappingCurve::new(2.5, 0.25, 0.538, 0.444, 1.28);
        let curve10 = Gt7ToneMappingCurve::new(10.0, 0.25, 0.538, 0.444, 1.28);
        let curve40 = Gt7ToneMappingCurve::new(40.0, 0.25, 0.538, 0.444, 1.28);

        let cases25 = [
            (0.0, 0.0),
            (0.1, 6.583_988e-2),
            (0.25, 2.233_054_3e-1),
            (0.444, 4.421_193e-1),
            (0.538, 0.538),
            (1.0, 1.0),
            (1.11, 1.109_999_9),
            (2.5, 2.087_880_6),
            (4.0, 2.573_628_7),
            (10.0, 2.948_031_2),
            (1504.7, 2.963_333_1), // exp underflow → k_a
        ];
        for (x, expected) in cases25 {
            assert!(
                (curve25.evaluate(x) - expected).abs() <= TOLERANCE,
                "peak 2.5, x = {x}"
            );
        }
        // Exactly-linear region and shoulder for higher peaks.
        assert!((curve10.evaluate(2.5) - 2.5).abs() <= TOLERANCE);
        assert!((curve10.evaluate(4.44) - 4.439_999_6).abs() <= TOLERANCE);
        assert!((curve10.evaluate(10.0) - 8.351_522).abs() <= TOLERANCE);
        assert!((curve40.evaluate(40.0) - 33.406_09).abs() <= 1e-3);

        // x < 0 branch.
        assert_eq!(curve25.evaluate(-1.0), 0.0);
    }

    #[test]
    fn bevy_sdr_seam_gray_anchor() {
        // Bevy scene-linear white (1, 1, 1) maps through the seam to GT7's
        // 250-nit paper white (2.5 fb gray): the matrices are
        // gray-preserving, so the result matches the native 2.5-fb fixture.
        let sdr = Gt7ToneMapping::new_sdr();
        let out = sdr.apply_bevy_scene_linear_sdr([1.0, 1.0, 1.0]);
        assert_rgb_eq(out, [8.351_579e-1, 8.351_547e-1, 8.351_534e-1]);

        // Black stays black; the output stays in [0, 1].
        assert_rgb_eq(sdr.apply_bevy_scene_linear_sdr([0.0, 0.0, 0.0]), [0.0; 3]);
        let bright = sdr.apply_bevy_scene_linear_sdr([100.0, 100.0, 100.0]);
        for c in bright {
            assert!((0.0..=1.0).contains(&c));
        }
    }

    #[test]
    fn matrices_are_gray_preserving_inverses() {
        for v in [
            mat3_mul_vec3(&REC_709_TO_REC_2020, [1.0, 1.0, 1.0]),
            mat3_mul_vec3(&REC_2020_TO_REC_709, [1.0, 1.0, 1.0]),
        ] {
            for c in v {
                assert!((c - 1.0).abs() < 1e-6);
            }
        }
        let round_trip = mat3_mul_vec3(
            &REC_2020_TO_REC_709,
            mat3_mul_vec3(&REC_709_TO_REC_2020, [0.25, 0.5, 0.75]),
        );
        assert_rgb_eq(round_trip, [0.25, 0.5, 0.75]);
    }

    /// The local matrix constants and the shared
    /// `bevy_render::working_color_space` constants must stay bit-identical
    /// (both are the canonical f64-derived BT.2087 values; gt7.rs is the
    /// declared CPU parity reference for the shaders).
    #[test]
    fn matrices_bitwise_match_bevy_render_working_color_space() {
        let pairs = [
            (
                &REC_709_TO_REC_2020,
                bevy_render::working_color_space::REC709_TO_REC2020,
            ),
            (
                &REC_2020_TO_REC_709,
                bevy_render::working_color_space::REC2020_TO_REC709,
            ),
        ];
        for (rows, mat) in pairs {
            for (row_index, row) in rows.iter().enumerate() {
                for (col_index, value) in row.iter().enumerate() {
                    // `gt7.rs` matrices are row-major; `Mat3` is column-major.
                    let shared = mat.col(col_index)[row_index];
                    assert_eq!(
                        value.to_bits(),
                        shared.to_bits(),
                        "matrix entry [{row_index}][{col_index}] differs: {value:?} vs {shared:?}"
                    );
                }
            }
        }
    }

    /// The Rec.2020-native SDR seam (`WORKING_COLOR_SPACE_REC2020`) must be
    /// bit-identical to the Rec.709-working seam fed the same color: the
    /// Rec.709-working path computes the identical Rec.709 → Rec.2020
    /// expansion as its first step, so handing the expanded value to the
    /// native seam replays the exact same float operations.
    #[test]
    fn rec2020_native_seam_matches_rec709_seam_bitwise() {
        let sdr = Gt7ToneMapping::new_sdr();
        let inputs: [[f32; 3]; 6] = [
            [0.5, 1.23, 0.75],
            [12.3, 4.56, 7.89],
            [1504.7, 0.0, 0.001],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.538, 1.11, 0.444],
        ];
        for rec709 in inputs {
            let rec709_working = sdr.apply_bevy_scene_linear_sdr(rec709);
            let native = sdr.apply_bevy_scene_linear_sdr_rec2020_native(mat3_mul_vec3(
                &REC_709_TO_REC_2020,
                rec709,
            ));
            for (l, n) in rec709_working.iter().zip(native.iter()) {
                assert_eq!(
                    l.to_bits(),
                    n.to_bits(),
                    "seam mismatch for input {rec709:?}: {rec709_working:?} vs {native:?}"
                );
            }
        }
    }

    /// The `TONEMAP_OUTPUT_REC2020` seam without the Rec.709 back-conversion
    /// is the reference operator itself behind Bevy's `× 2.5` input scaling,
    /// so feeding it `fb / 2.5` must reproduce the C++ harness's canonical 12
    /// fixtures DIRECTLY — the cleanest possible parity statement for the
    /// HDR-native output path (the Rec.2020 → Rec.709 wrapper was always
    /// Bevy-side; the fixtures themselves are native Rec.2020 outputs).
    #[test]
    fn hdr_native_rec2020_output_matches_cpp_fixtures_directly() {
        let sdr = Gt7ToneMapping::new_sdr();
        let hdr1000 = Gt7ToneMapping::new_hdr(1000.0);
        let hdr4000 = Gt7ToneMapping::new_hdr(4000.0);
        let hdr10000 = Gt7ToneMapping::new_hdr(10000.0);

        // The canonical fixture inputs are GT7 frame-buffer values; the seam
        // multiplies Bevy scene-linear by 2.5, so divide first. The division
        // round-trips within 1 ULP, far inside the 1e-4 fixture tolerance.
        let scene = |fb: [f32; 3]| [fb[0] / 2.5, fb[1] / 2.5, fb[2] / 2.5];
        let inputs: [[f32; 3]; 3] = [[0.5, 1.23, 0.75], [12.3, 34.3, 56.9], [1504.7, 64.51, 0.5]];

        let native =
            |tm: &Gt7ToneMapping, fb| tm.apply_bevy_scene_linear_rec2020_native_rec2020_output(fb);

        // SDR mode (output [0, 1], still native Rec.2020 — no clamp needed).
        assert_rgb_eq(
            native(&sdr, scene(inputs[0])),
            [1.996_225e-1, 4.907_029_6e-1, 2.995_677_6e-1],
        );
        assert_rgb_eq(native(&sdr, scene(inputs[1])), [1.0, 1.0, 1.0]);
        assert_rgb_eq(native(&sdr, scene(inputs[2])), [1.0, 1.0, 7.387_512e-1]);

        // HDR, 1000-nit peak.
        assert_rgb_eq(
            native(&hdr1000, scene(inputs[0])),
            [4.998_231_8e-1, 1.230_000_7, 7.499_952_3e-1],
        );
        assert_rgb_eq(native(&hdr1000, scene(inputs[1])), [10.0, 10.0, 10.0]);
        assert_rgb_eq(
            native(&hdr1000, scene(inputs[2])),
            [10.0, 10.0, 6.706_747_5],
        );

        // HDR, 4000-nit peak.
        assert_rgb_eq(
            native(&hdr4000, scene(inputs[0])),
            [4.998_231_8e-1, 1.230_000_7, 7.499_952_3e-1],
        );
        assert_rgb_eq(
            native(&hdr4000, scene(inputs[1])),
            [11.354_192, 30.071_972, 40.0],
        );
        assert_rgb_eq(native(&hdr4000, scene(inputs[2])), [40.0, 40.0, 23.847_712]);

        // HDR, 10000-nit peak.
        assert_rgb_eq(
            native(&hdr10000, scene(inputs[0])),
            [4.998_231_8e-1, 1.230_000_7, 7.499_952_3e-1],
        );
        assert_rgb_eq(
            native(&hdr10000, scene(inputs[1])),
            [12.277_842, 34.240_51, 56.395_88],
        );
        assert_rgb_eq(
            native(&hdr10000, scene(inputs[2])),
            [91.726_71, 68.024_31, 42.575_134],
        );
    }

    /// The `TONEMAP_OUTPUT_REC2020` seams differ from the SDR seams by
    /// EXACTLY the dropped Rec.2020 → Rec.709 back-conversion + clamp:
    /// re-applying it on the CPU reproduces the SDR-seam output bit-for-bit,
    /// for both working spaces and both operator modes (all four wrapper def
    /// combinations are coherent).
    #[test]
    fn rec2020_output_back_conversion_is_the_only_difference() {
        let back_convert = |out: [f32; 3]| {
            let rec709 = mat3_mul_vec3(&REC_2020_TO_REC_709, out);
            [
                rec709[0].clamp(0.0, 1.0),
                rec709[1].clamp(0.0, 1.0),
                rec709[2].clamp(0.0, 1.0),
            ]
        };
        let assert_bits_eq = |a: [f32; 3], b: [f32; 3], label: &str, input: [f32; 3]| {
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{label} mismatch for input {input:?}: {a:?} vs {b:?}"
                );
            }
        };

        let mut hdr = Gt7ToneMapping::new_hdr(1000.0);
        // The paper-white renormalization, as the HDR prepare path folds it.
        hdr.sdr_correction_factor = REFERENCE_LUMINANCE / 200.0;

        let inputs: [[f32; 3]; 6] = [
            [0.5, 1.23, 0.75],
            [12.3, 4.56, 7.89],
            [1504.7, 0.0, 0.001],
            [1.0, 1.0, 1.0],
            [-0.1, 0.2, 0.1],
            [0.538, 1.11, 0.444],
        ];
        for tm in [Gt7ToneMapping::new_sdr(), hdr] {
            for input in inputs {
                assert_bits_eq(
                    tm.apply_bevy_scene_linear_sdr(input),
                    back_convert(tm.apply_bevy_scene_linear_rec2020_output(input)),
                    "rec709-working",
                    input,
                );
                assert_bits_eq(
                    tm.apply_bevy_scene_linear_sdr_rec2020_native(input),
                    back_convert(tm.apply_bevy_scene_linear_rec2020_native_rec2020_output(input)),
                    "rec2020-working",
                    input,
                );
            }
        }
    }

    /// With the paper-white renormalization prepared by
    /// `Gt7ParamsUniform::new` for an HDR target, the native-output seam's
    /// ceiling is exactly `peak / paper_white` — paper-white-relative
    /// display-linear, the encoder's input convention.
    #[test]
    fn hdr_native_output_is_paper_white_relative() {
        let params = GranTurismo7Params::default();
        let uniform = Gt7ParamsUniform::new(
            &hdr_target(1000.0, 200.0, DisplayTransfer::ScRgbLinear),
            &params,
        );
        let mut tm = Gt7ToneMapping::new_hdr_with_params(1000.0, &params);
        tm.sdr_correction_factor = uniform.sdr_correction_factor;

        // A far-over-peak gray saturates every channel at the peak clamp
        // (10 fb); the seam renormalization scales it to peak / paper_white
        // = 1000 / 200 = 5.0 exactly.
        let out = tm.apply_bevy_scene_linear_rec2020_native_rec2020_output([1e4; 3]);
        for c in out {
            assert_eq!(c, 1000.0 / 200.0);
        }
    }

    #[test]
    fn params_sanitized_clamps_and_resets() {
        // NaN/Inf reset to defaults.
        let nan_params = GranTurismo7Params {
            blend_ratio: f32::NAN,
            fade_end: f32::INFINITY,
            ..Default::default()
        };
        let sanitized = nan_params.sanitized();
        assert_eq!(sanitized, GranTurismo7Params::default());

        // Range clamps.
        let out_of_range = GranTurismo7Params {
            blend_ratio: 1.5,
            fade_start: -0.25,
            fade_end: -1.0,
            alpha: 1.0,
            mid_point: 0.0,
            linear_section: 1.0,
            toe_strength: -3.0,
        };
        let sanitized = out_of_range.sanitized();
        assert_eq!(sanitized.blend_ratio, 1.0);
        assert_eq!(sanitized.fade_start, 0.0);
        assert!(sanitized.fade_end >= sanitized.fade_start + 1e-4);
        assert!(sanitized.alpha < 1.0);
        assert!(sanitized.mid_point > 0.0);
        assert!(sanitized.linear_section < 1.0);
        assert!(sanitized.toe_strength >= 0.0);

        // fade_end > 1 is intentional: must NOT be clamped down.
        let wide_fade = GranTurismo7Params {
            fade_end: 4.0,
            ..Default::default()
        };
        assert_eq!(wide_fade.sanitized().fade_end, 4.0);

        // Defaults pass through untouched.
        let defaults = GranTurismo7Params::default();
        assert_eq!(defaults.sanitized(), defaults);

        // Sanitized params must produce finite output everywhere the raw
        // params would have produced NaN.
        let tm = Gt7ToneMapping::new_sdr_with_params(&out_of_range.sanitized());
        for c in tm.apply([0.5, 1.23, 0.75]) {
            assert!(c.is_finite());
        }
    }

    use bevy_window::{DisplayGamut, DisplayTransfer};

    fn hdr_target(peak: f32, paper_white: f32, transfer: DisplayTransfer) -> DisplayTarget {
        DisplayTarget {
            paper_white_nits: paper_white,
            peak_luminance_nits: peak,
            min_luminance_nits: 0.0,
            gamut: DisplayGamut::Rec2020,
            transfer,
        }
    }

    /// SDR-target uniform must reproduce the C++ reference's SDR init
    /// products (the same fixtures as `cpp_parity_init_products`), i.e. the
    /// values baked into `gt7_default_sdr_params()` in gt7.wgsl.
    #[test]
    fn uniform_sdr_mode_matches_init_fixtures() {
        let uniform =
            Gt7ParamsUniform::new(&DisplayTarget::SDR_SRGB, &GranTurismo7Params::default());
        assert_eq!(uniform.peak, 2.5);
        assert!((uniform.k_a - 2.963_333_1).abs() < 1e-5);
        assert!((uniform.k_b - -3.373_351).abs() < 1e-5);
        assert!((uniform.k_c - -5.395_683_6e-1).abs() < 1e-6);
        assert!((uniform.peak_ucs - 6.025_607_6e-1).abs() < 1e-5);
        assert!((uniform.sdr_correction_factor - 0.4).abs() < 1e-7);
        assert_eq!(uniform.mid_point, 0.538);
        assert_eq!(uniform.linear_section, 0.444);
        assert_eq!(uniform.toe_strength, 1.28);
        assert_eq!(uniform.blend_ratio, 0.6);
        assert_eq!(uniform.fade_start, 0.98);
        assert_eq!(uniform.fade_end, 1.16);

        // A non-HDR transfer stays SDR mode no matter how bright the target
        // claims to be.
        let bright_sdr = Gt7ParamsUniform::new(
            &DisplayTarget {
                peak_luminance_nits: 4000.0,
                ..DisplayTarget::SDR_SRGB
            },
            &GranTurismo7Params::default(),
        );
        assert_eq!(bright_sdr.peak, 2.5);
        assert!((bright_sdr.sdr_correction_factor - 0.4).abs() < 1e-7);
    }

    /// HDR-target uniform must match the C++ reference's HDR init products
    /// and apply the paper-white renormalization (`100 / paper_white`).
    #[test]
    fn uniform_hdr_mode_matches_init_fixtures() {
        let params = GranTurismo7Params::default();
        let uniform = Gt7ParamsUniform::new(
            &hdr_target(1000.0, 100.0, DisplayTransfer::ScRgbLinear),
            &params,
        );
        let reference = Gt7ToneMapping::new_hdr(1000.0);
        assert_eq!(uniform.peak, 10.0);
        assert_eq!(uniform.k_a, reference.curve.k_a);
        assert_eq!(uniform.k_b, reference.curve.k_b);
        assert_eq!(uniform.k_c, reference.curve.k_c);
        assert_eq!(uniform.peak_ucs, reference.framebuffer_luminance_target_ucs);
        // Seam renormalization is identity at the default 100-nit paper white
        // (native HDR-mode factor).
        assert_eq!(uniform.sdr_correction_factor, 1.0);

        // Non-default paper white scales the output so 1.0 = paper white.
        let uniform =
            Gt7ParamsUniform::new(&hdr_target(1000.0, 200.0, DisplayTransfer::Pq), &params);
        assert_eq!(uniform.peak, 10.0);
        assert!((uniform.sdr_correction_factor - 0.5).abs() < 1e-7);

        // The 10000-nit peak hits PQ's exact ceiling (peak UCS == 1).
        let uniform = Gt7ParamsUniform::new(
            &hdr_target(10000.0, 100.0, DisplayTransfer::ScRgbLinear),
            &params,
        );
        assert_eq!(uniform.peak, 100.0);
        assert!((uniform.peak_ucs - 1.0).abs() < 1e-6);
    }

    /// The clamp table for HDR-mode peak/paper-white selection.
    #[test]
    fn uniform_hdr_mode_clamp_table() {
        let params = GranTurismo7Params::default();

        // Peak below the documented 250-nit lower bound: clamped up.
        let uniform = Gt7ParamsUniform::new(
            &hdr_target(100.0, 100.0, DisplayTransfer::ScRgbLinear),
            &params,
        );
        assert_eq!(uniform.peak, 2.5);
        assert_eq!(uniform.sdr_correction_factor, 1.0);

        // Peak above the 10000-nit PQ ceiling: clamped down.
        let uniform =
            Gt7ParamsUniform::new(&hdr_target(20000.0, 100.0, DisplayTransfer::Pq), &params);
        assert_eq!(uniform.peak, 100.0);

        // Peak below paper white: raised to paper white.
        let uniform = Gt7ParamsUniform::new(
            &hdr_target(400.0, 600.0, DisplayTransfer::ScRgbLinear),
            &params,
        );
        assert_eq!(uniform.peak, 6.0);
        assert!((uniform.sdr_correction_factor - 100.0 / 600.0).abs() < 1e-7);

        // Non-finite peak: reset to 100 nits, then range-clamped to 250.
        let uniform = Gt7ParamsUniform::new(
            &hdr_target(f32::NAN, 100.0, DisplayTransfer::ScRgbLinear),
            &params,
        );
        assert_eq!(uniform.peak, 2.5);

        // Non-finite / non-positive paper white: reset to 100 nits.
        for paper_white in [f32::NAN, f32::INFINITY, 0.0, -50.0] {
            let uniform = Gt7ParamsUniform::new(
                &hdr_target(1000.0, paper_white, DisplayTransfer::ScRgbLinear),
                &params,
            );
            assert_eq!(uniform.peak, 10.0);
            assert_eq!(uniform.sdr_correction_factor, 1.0);
        }

        // Absurd paper white above the PQ ceiling: clamped to 10000, and the
        // peak follows it up.
        let uniform = Gt7ParamsUniform::new(
            &hdr_target(1000.0, 20000.0, DisplayTransfer::ScRgbLinear),
            &params,
        );
        assert_eq!(uniform.peak, 100.0);
        assert!((uniform.sdr_correction_factor - 0.01).abs() < 1e-9);
    }

    /// The seam renormalization must fold the *exact* value
    /// [`DisplayTarget::sanitized_paper_white_nits`] returns — the same
    /// method the display pipeline's uniform writer applies before the
    /// encoder multiplies by paper white — so the paper-white factors
    /// (`× 100 / paper_white` here, `× paper_white / 80` or
    /// `× paper_white` at the encoder) cancel bit-for-bit for every input.
    #[test]
    fn paper_white_fold_is_single_sourced_with_the_display_pipeline() {
        let params = GranTurismo7Params::default();
        for paper_white in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -50.0,
            0.0,
            50.0,
            80.0,
            100.0,
            203.0,
            1000.0,
            10000.0,
            20000.0,
        ] {
            let target = hdr_target(1000.0, paper_white, DisplayTransfer::ScRgbLinear);
            let uniform = Gt7ParamsUniform::new(&target, &params);
            assert_eq!(
                uniform.sdr_correction_factor.to_bits(),
                (REFERENCE_LUMINANCE / target.sanitized_paper_white_nits()).to_bits(),
                "diverged for paper_white_nits = {paper_white}"
            );
        }
    }

    /// User params flow through `sanitized()` before reaching the uniform.
    #[test]
    fn uniform_sanitizes_user_params() {
        let out_of_range = GranTurismo7Params {
            blend_ratio: 7.5,
            ..Default::default()
        };
        let uniform = Gt7ParamsUniform::new(&DisplayTarget::SDR_SRGB, &out_of_range);
        assert_eq!(uniform.blend_ratio, 1.0);

        let nan_params = GranTurismo7Params {
            mid_point: f32::NAN,
            ..Default::default()
        };
        let uniform = Gt7ParamsUniform::new(&DisplayTarget::SDR_SRGB, &nan_params);
        assert_eq!(uniform.mid_point, GranTurismo7Params::default().mid_point);
    }

    /// Parameter combinations whose individual values pass sanitization can
    /// still overflow the closed-form curve constants (`linear_section`
    /// close to 1 sends `exp(linear_section / k)` past `f32::MAX`); the
    /// uniform must fall back to the defaults instead of uploading
    /// non-finite constants that turn highlights into NaN.
    #[test]
    fn uniform_falls_back_when_curve_constants_overflow() {
        let overflowing = GranTurismo7Params {
            linear_section: 0.999,
            ..Default::default()
        };

        let sdr = Gt7ParamsUniform::new(&DisplayTarget::SDR_SRGB, &overflowing);
        let hdr = Gt7ParamsUniform::new(
            &DisplayTarget {
                transfer: DisplayTransfer::ScRgbLinear,
                peak_luminance_nits: 1000.0,
                ..DisplayTarget::SDR_SRGB
            },
            &overflowing,
        );
        for uniform in [&sdr, &hdr] {
            assert!(uniform.k_a.is_finite(), "k_a must stay finite");
            assert!(uniform.k_b.is_finite(), "k_b must stay finite");
            assert!(uniform.k_c.is_finite(), "k_c must stay finite");
            assert!(uniform.peak_ucs.is_finite(), "peak_ucs must stay finite");
        }
        // The fallback recomputes from the default parameters.
        assert_eq!(
            sdr.linear_section,
            GranTurismo7Params::default().linear_section
        );
    }

    /// The uniform round-trips `Gt7ToneMapping` exactly, so the CPU struct
    /// (with the same correction-factor override) remains the parity
    /// reference for GPU readback tests of the uniform path.
    #[test]
    fn uniform_mirrors_cpu_tone_mapping() {
        let params = GranTurismo7Params::default();
        let mut reference = Gt7ToneMapping::new_hdr_with_params(1000.0, &params);
        reference.sdr_correction_factor = REFERENCE_LUMINANCE / 200.0;
        let uniform = Gt7ParamsUniform::new(
            &hdr_target(1000.0, 200.0, DisplayTransfer::ScRgbLinear),
            &params,
        );
        assert_eq!(uniform, Gt7ParamsUniform::from(&reference));
    }
}
