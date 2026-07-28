use bevy_camera::{Camera, Hdr};
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
/// mimicking the chromatic adaptation of human vision (and of camera AWB).
///
/// # How it works
///
/// * **Metering** shares the [`AutoExposure`](super::AutoExposure) compute
///   pass: the same per-pixel metering-mask weights drive both the luminance
///   histogram and a luminance-weighted average of the scene's CIE 1931 *xy*
///   chromaticity (a [Yxy] measurement, the blend space Gran Turismo 7
///   settled on). When both components are present on a camera, one metering
///   dispatch serves both; `AutoWhiteBalance` also works on its own, in which
///   case the shared pass runs with neutral auto-exposure settings.
/// * **Stability**: a faint *virtual light* — an ideal D65 light source of
///   luminance [`virtual_light_anchor`](Self::virtual_light_anchor) — is
///   blended into the measurement as one more luminance-weighted reference.
///   In bright scenes it is negligible; in dark scenes it dominates, anchoring
///   the white point at neutral instead of chasing measurement noise. This is
///   Gran Turismo 7's dark-scene stability mechanism.
/// * **Adaptation** smooths only the *xy* chromaticity over time at
///   [`speed`](Self::speed) (luminance adaptation is
///   [`AutoExposure`](super::AutoExposure)'s job).
/// * **Output**: the adapted chromaticity is converted to a correlated color
///   temperature (the `McCamy` 1992 approximation) plus a tint offset from the
///   Planckian locus, the temperature is clamped to the **2500 K – 7000 K**
///   range typical of real camera AWB specifications, and the result is
///   applied as a von Kries adaptation in the same LMS basis Bevy's existing
///   white-balance machinery uses: the correction matrix is multiplied into
///   the view's [`ColorGrading`](bevy_render::view::ColorGrading) balance
///   matrix on the GPU.
///
/// # Composition with manual color grading
///
/// The automatic correction composes with, and does not overwrite, the
/// artist-authored [`ColorGrading`](bevy_render::view::ColorGrading)
/// `temperature`/`tint` values: the automatic correction (towards neutral) is
/// applied to the image first, and the manual white-balance adjustment is
/// applied on top of the corrected image. A deliberate "warm tungsten" grade
/// therefore stays warm regardless of the scene's measured illuminant.
///
/// # Usage Notes
///
/// Like [`AutoExposure`](super::AutoExposure), the correction is consumed by
/// the tonemapping pass, so cameras with `Tonemapping::None` are unaffected,
/// and the metering runs in a compute shader (**not compatible with WebGL2**).
/// Hue-preserving tone-mapping operators (`TonyMcMapface`, `AgX`,
/// `KhronosPbrNeutral`, `GranTurismo7`) preserve the corrected white point
/// best; purely per-channel operators (`Reinhard`, `ReinhardLuminance`,
/// `SomewhatBoringDisplayTransform`) may shift the corrected color slightly
/// away from neutral again.
///
/// Add this component to a 3d camera together with the
/// [`AutoExposurePlugin`](super::AutoExposurePlugin) (which owns the shared
/// metering infrastructure). The metering pass runs only in the 3d core
/// pipeline, so the correction has no effect on 2d cameras.
///
/// [Yxy]: https://en.wikipedia.org/wiki/CIE_1931_color_space#CIE_xy_chromaticity_diagram_and_the_CIE_xyY_color_space
#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect)]
#[reflect(Component, Default, Clone, PartialEq)]
#[require(Hdr, NeedsSceneLinearTarget)]
pub struct AutoWhiteBalance {
    /// The adaptation speed of the white-point chromaticity, per second.
    ///
    /// This is the rate constant of an exponential approach: each second, the
    /// adapted chromaticity moves this fraction of the remaining distance
    /// towards the measured scene chromaticity (values are clamped so a
    /// single frame never overshoots). Around `0.5`, adaptation settles in a
    /// few seconds, comparable to a real camera's AWB; chromatic adaptation
    /// is deliberately slower than exposure adaptation.
    ///
    /// Must be finite and non-negative. The default value is 0.5.
    pub speed: f32,

    /// The luminance of the faint *virtual light* — an ideal D65 illuminant
    /// blended into the scene measurement as a luminance-weighted reference —
    /// in scene-linear luminance units (the same units as the rendered pixel
    /// values that auto exposure meters).
    ///
    /// The blend weight of the anchor relative to the measurement is
    /// `virtual_light_anchor / (virtual_light_anchor + scene_luminance)`, so
    /// its influence scales with the *inverse* of the mean scene luminance:
    /// negligible in normal lighting, dominant in near-dark scenes where a
    /// measured chromaticity would be noise. Set it to `0.0` to disable the
    /// anchor and always trust the measurement.
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
    /// Returns a copy of these settings with invalid fields reset to their
    /// default values.
    ///
    /// Both fields must be finite and non-negative. Warns (once) if any field
    /// had to be reset.
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
    // `WHITE_BALANCE` shader path compiled in for this view even when the
    // static `ColorGrading` temperature/tint deltas are zero, since the
    // metering compute pass composes the automatic correction matrix into
    // `view.color_grading.balance` on the GPU.
    type Out = (Self, ExternalWhiteBalance);

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some((*item, ExternalWhiteBalance))
    }
}
