//! Per-camera parameters for the Gran Turismo 7 tonemapping method.
//!
//! Ported from Polyphony Digital's reference implementation, `gt7_tone_mapping.cpp` (MIT
//! License, Copyright (c) 2025 Polyphony Digital Inc.).

use bevy_camera::{Camera, TonemappingPass};
use bevy_ecs::prelude::*;
use bevy_log::warn_once;
use bevy_math::ops;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    extract_component::ExtractComponent, render_resource::ShaderType,
    transfer_functions::pq_inverse_eotf_from_nits, view::ViewDisplayTarget, RenderApp,
};

/// The luminance of a frame buffer value of 1.0 in the method, in nits.
const REFERENCE_LUMINANCE: f32 = 100.0;

/// Gran Turismo's SDR paper white, in nits.
const SDR_PAPER_WHITE_NITS: f32 = 250.0;

/// Tunes [`Tonemapping::GranTurismo7`](super::Tonemapping::GranTurismo7) for a camera.
///
/// The defaults match Polyphony Digital's reference implementation.
#[derive(Component, Debug, Clone, Copy, PartialEq, Reflect, ExtractComponent)]
#[extract_component_filter(With<Camera>)]
#[reflect(Component, Debug, Default, PartialEq, Clone)]
#[extract_app(RenderApp)]
// Materials that tonemap in their shaders can't read these parameters.
#[require(TonemappingPass)]
pub struct GranTurismo7Params {
    /// How much of the hue-preserving curve to blend into the per-channel curve, from 0.0
    /// to 1.0. The per-channel curve shifts bright colors toward white, like a camera.
    pub blend_ratio: f32,
    /// Where colors start to lose saturation, as a fraction of peak brightness, from 0.0
    /// to 1.0.
    pub fade_start: f32,
    /// Where colors are fully desaturated, as a fraction of peak brightness. It must be
    /// greater than [`fade_start`](Self::fade_start), and can be greater than 1.0.
    pub fade_end: f32,
    /// The curvature of the shoulder that compresses highlights. Must be at least 0.0 and
    /// less than 1.0.
    pub alpha: f32,
    /// Where the toe ends and the linear section begins, in units where 1.0 is 100 nits.
    /// Must be greater than 0.0.
    pub mid_point: f32,
    /// Where the linear section ends and the shoulder begins, as a fraction of peak
    /// brightness. Must be at least 0.0 and less than 1.0.
    pub linear_section: f32,
    /// The exponent of the toe. Higher values darken shadows. Must be greater than 0.0.
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
    /// Whether every field is in the range its docs give.
    fn is_valid(&self) -> bool {
        let unit = 0.0..=1.0;
        let below_one = 0.0..1.0;
        unit.contains(&self.blend_ratio)
            && unit.contains(&self.fade_start)
            && self.fade_end.is_finite()
            && self.fade_end > self.fade_start
            && below_one.contains(&self.alpha)
            && below_one.contains(&self.linear_section)
            && self.mid_point.is_finite()
            && self.mid_point > 0.0
            && self.toe_strength.is_finite()
            && self.toe_strength > 0.0
    }
}

/// The curve constants for `Tonemapping::GranTurismo7`. Matches `Gt7Params` in
/// `gt7.wesl`.
#[derive(Component, Clone, Copy, Debug, PartialEq, ShaderType)]
pub struct Gt7ParamsUniform {
    peak: f32,
    k_a: f32,
    k_b: f32,
    k_c: f32,
    mid_point: f32,
    linear_section: f32,
    toe_strength: f32,
    peak_ucs: f32,
    blend_ratio: f32,
    fade_start: f32,
    fade_end: f32,
    output_scale: f32,
    clamp_output: u32,
}

impl Gt7ParamsUniform {
    /// On an HDR display, the curve peaks at the display's peak luminance, and the output
    /// is scaled so that 1.0 is paper white. Otherwise it uses Gran Turismo's SDR
    /// calibration, and the output is clamped to `[0, 1]`.
    pub(super) fn new(display_target: &ViewDisplayTarget, params: &GranTurismo7Params) -> Self {
        let is_hdr = display_target.is_hdr();
        let (peak_nits, paper_white_nits) = if is_hdr {
            // The curve needs a peak of at least Gran Turismo's SDR paper white.
            (
                display_target.peak_luminance_nits.max(SDR_PAPER_WHITE_NITS),
                display_target.paper_white_nits,
            )
        } else {
            (SDR_PAPER_WHITE_NITS, SDR_PAPER_WHITE_NITS)
        };
        let uniform = Self::from_params(peak_nits, paper_white_nits, is_hdr, params);
        if params.is_valid() && uniform.is_finite() {
            return uniform;
        }
        warn_once!(
            "GranTurismo7Params has values outside their valid ranges, or values that make \
            the tonemapping curve infinite. Using the default parameters instead."
        );
        let defaults = GranTurismo7Params::default();
        Self::from_params(peak_nits, paper_white_nits, is_hdr, &defaults)
    }

    /// The closed forms are from the reference's `GT7ToneMappingCurve`.
    fn from_params(
        peak_nits: f32,
        paper_white_nits: f32,
        is_hdr: bool,
        params: &GranTurismo7Params,
    ) -> Self {
        let peak = peak_nits / REFERENCE_LUMINANCE;
        let k = (params.linear_section - 1.0) / (params.alpha - 1.0);
        Self {
            peak,
            k_a: peak * params.linear_section + peak * k,
            k_b: -peak * k * ops::exp(params.linear_section / k),
            k_c: -1.0 / (k * peak),
            mid_point: params.mid_point,
            linear_section: params.linear_section,
            toe_strength: params.toe_strength,
            // The LMS rows sum to 1, so the `ICtCp` intensity of a gray is its PQ signal.
            peak_ucs: pq_inverse_eotf_from_nits(peak_nits),
            blend_ratio: params.blend_ratio,
            fade_start: params.fade_start,
            fade_end: params.fade_end,
            output_scale: REFERENCE_LUMINANCE / paper_white_nits,
            clamp_output: u32::from(!is_hdr),
        }
    }

    /// Some combinations of fields in their valid ranges still overflow.
    fn is_finite(&self) -> bool {
        // The toe's largest value is where it meets the shoulder.
        let toe_end = self.linear_section * self.peak;
        let toe = self.mid_point * ops::powf(toe_end / self.mid_point, self.toe_strength);
        [self.k_a, self.k_b, self.k_c, toe]
            .iter()
            .all(|value| value.is_finite())
    }
}
