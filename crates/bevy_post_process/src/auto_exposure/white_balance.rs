use super::AutoExposure;
use bevy_camera::Camera;
use bevy_core_pipeline::tonemapping::ExternalWhiteBalance;
use bevy_ecs::{prelude::*, query::QueryItem, reflect::ReflectComponent};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    extract_component::ExtractComponent, sync_component::SyncComponent,
    view::NeedsSceneLinearTarget, RenderApp,
};
use bevy_utils::once;
use tracing::warn;

/// Component that enables automatic white balance for an HDR-enabled camera,
/// following the model Polyphony Digital presented for Gran Turismo 7 at
/// SIGGRAPH 2025 ("Physically Based Tone Mapping in Gran Turismo 7").
///
/// Auto white balance estimates the scene's dominant illuminant chromaticity
/// and slowly adapts a correction towards a neutral (D65) white point,
/// mimicking the chromatic adaptation of human vision.
///
/// The measurement rides along in [`AutoExposure`]'s metering pass, so this
/// component requires [`AutoExposure`] and pulls it in when added on its own.
///
/// * Metering uses the same per-pixel metering-mask weights as the luminance
///   histogram to build a luminance-weighted average of the scene's CIE 1931
///   xy chromaticity, blended in [Yxy] space as Gran Turismo 7 does.
/// * A faint virtual light, an ideal D65 source of luminance
///   [`virtual_light_anchor`](Self::virtual_light_anchor), is blended into the
///   measurement so near-dark scenes stay anchored at neutral instead of
///   chasing measurement noise.
/// * Adaptation smooths only the xy chromaticity over time, at
///   [`speed`](Self::speed). Luminance adaptation is [`AutoExposure`]'s job.
/// * The adapted chromaticity is converted to a correlated color temperature
///   (the `McCamy` 1992 approximation) plus a tint offset from the Planckian
///   locus. The temperature is clamped to the 2500 K - 7000 K range typical of
///   real camera AWB specifications. The result is applied as a von Kries
///   adaptation in the same LMS basis Bevy's static white balance uses, by
///   multiplying the correction matrix into the view's
///   [`ColorGrading`](bevy_render::view::ColorGrading) balance matrix on the GPU.
///
/// The automatic correction composes with the artist-authored
/// [`ColorGrading`](bevy_render::view::ColorGrading) `temperature`/`tint`
/// instead of overwriting them.
///
/// # Usage Notes
///
/// Like [`AutoExposure`], the correction is consumed by the tonemapping pass,
/// so cameras with `Tonemapping::None` are unaffected, and the metering runs
/// in a compute shader (**not compatible with WebGL2**).
///
/// Hue-preserving operators (`TonyMcMapface`, `AgX`, `KhronosPbrNeutral`,
/// `GranTurismo7`, `ReinhardLuminance`) preserve the corrected white point best.
/// Per-channel operators (`Reinhard`, `SomewhatBoringDisplayTransform`) can
/// shift it slightly away from neutral again.
///
/// Add this component to a 3d camera together with
/// [`AutoExposurePlugin`](super::AutoExposurePlugin). The metering pass runs
/// only in the 3d core pipeline, so this has no effect on 2d cameras.
///
/// [Yxy]: https://en.wikipedia.org/wiki/CIE_1931_color_space#CIE_xy_chromaticity_diagram_and_the_CIE_xyY_color_space
#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect)]
#[reflect(Component, Default, Clone, PartialEq)]
#[require(AutoExposure, NeedsSceneLinearTarget)]
pub struct AutoWhiteBalance {
    /// The adaptation speed of the white-point chromaticity, per second.
    ///
    /// The rate constant of an exponential approach: each second the adapted
    /// chromaticity moves this fraction of the remaining distance towards the
    /// measured scene chromaticity. A single frame never overshoots.
    ///
    /// Must be finite and non-negative. The default value is 0.5, which settles
    /// in a few seconds, like a real camera's AWB. Chromatic adaptation is
    /// deliberately slower than exposure adaptation.
    pub speed: f32,

    /// The luminance of the virtual light, an ideal D65 illuminant blended into
    /// the scene measurement as a luminance-weighted reference, in the
    /// scene-linear units auto exposure meters.
    ///
    /// The anchor's weight relative to the measurement is
    /// `virtual_light_anchor / (virtual_light_anchor + scene_luminance)`, so it
    /// is negligible in normal lighting and dominant in near-dark scenes. Set it
    /// to `0.0` to disable the anchor and always trust the measurement.
    ///
    /// Must be finite and non-negative. The default value is 0.01.
    pub virtual_light_anchor: f32,
}

impl Default for AutoWhiteBalance {
    fn default() -> Self {
        Self {
            speed: 0.5,
            virtual_light_anchor: 0.01,
        }
    }
}

impl AutoWhiteBalance {
    /// Invalid fields are reset to their defaults, warning once.
    pub(super) fn sanitized(&self) -> Self {
        let defaults = Self::default();
        let mut invalid = false;
        let mut sanitize = |value: f32, default: f32| -> f32 {
            if value.is_finite() && value >= 0.0 {
                value
            } else {
                invalid = true;
                default
            }
        };

        let sanitized = Self {
            speed: sanitize(self.speed, defaults.speed),
            virtual_light_anchor: sanitize(
                self.virtual_light_anchor,
                defaults.virtual_light_anchor,
            ),
        };

        if invalid {
            once!(warn!(
                "AutoWhiteBalance::speed and ::virtual_light_anchor must be finite and \
                non-negative; invalid fields were reset to their defaults"
            ));
        }

        sanitized
    }
}

impl SyncComponent<RenderApp> for AutoWhiteBalance {
    type Target = (AutoWhiteBalance, ExternalWhiteBalance);
}

impl ExtractComponent<RenderApp> for AutoWhiteBalance {
    type QueryData = &'static Self;
    type QueryFilter = With<Camera>;
    // The `ExternalWhiteBalance` marker keeps the tonemapping pass's
    // `WHITE_BALANCE` shader path compiled in for this view even when the static
    // `ColorGrading` temperature/tint deltas are zero, because the metering pass
    // composes the correction matrix into `view.color_grading.balance` on the GPU.
    type Out = (Self, ExternalWhiteBalance);

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some((*item, ExternalWhiteBalance))
    }
}
