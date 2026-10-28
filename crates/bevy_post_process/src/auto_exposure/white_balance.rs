use super::{settings::or_default, AutoExposure};
use bevy_core_pipeline::tonemapping::ExternalWhiteBalance;
use bevy_ecs::{prelude::*, query::QueryItem, reflect::ReflectComponent};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{extract_component::ExtractComponent, sync_component::SyncComponent, RenderApp};
use bevy_utils::once;
use tracing::warn;

/// Component that corrects the color of the scene's light toward neutral white, like a camera's
/// auto white balance or the eye's adaptation to colored light.
///
/// It meters the average chromaticity of the scene, weighted by luminance and by the
/// [`AutoExposure::metering_mask`], and gradually adapts a correction that maps it to the D65
/// white point. The correction covers color temperatures from 2500 K to 7000 K, and tints up to
/// 0.05 away from the Planckian locus in CIE 1931 xy. It is applied before the
/// [`ColorGrading`](bevy_render::view::ColorGrading) white balance, which stays available for
/// grading.
///
/// This is the auto white balance of "Physically Based Tone Mapping in Gran Turismo 7"
/// (SIGGRAPH 2025).
///
/// # Usage Notes
///
/// The tonemapping pass applies the correction, so it has no effect on a camera with
/// [`Tonemapping::None`](bevy_core_pipeline::tonemapping::Tonemapping::None).
///
/// **Auto white balance requires compute shaders and is not compatible with WebGL2.**
#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect)]
#[reflect(Component, Default, Clone, Debug, PartialEq)]
#[require(AutoExposure)]
pub struct AutoWhiteBalance {
    /// How fast the correction adapts to a change in the color of the light, per second of real
    /// time. Each frame covers this times the frame time of the remaining difference, up to all of
    /// it.
    ///
    /// Must be finite and not negative. The default value is 0.5.
    pub speed: f32,

    /// The luminance of a neutral light that is always part of the measurement, in cd/m². It keeps
    /// dim scenes close to neutral.
    ///
    /// Its share of the measurement is `virtual_light_luminance / (virtual_light_luminance +
    /// scene_luminance)`, so it matters only in scenes about as dark as it or darker.
    ///
    /// Must be finite and not negative. The default value is 1.0.
    pub virtual_light_luminance: f32,
}

impl Default for AutoWhiteBalance {
    fn default() -> Self {
        Self {
            speed: 0.5,
            virtual_light_luminance: 1.0,
        }
    }
}

impl AutoWhiteBalance {
    /// Returns these settings with each invalid value replaced by its default.
    pub(super) fn sanitized(&self) -> Self {
        let default = Self::default();
        let mut invalid = false;
        let sanitized = Self {
            speed: or_default(self.speed, default.speed, &mut invalid),
            virtual_light_luminance: or_default(
                self.virtual_light_luminance,
                default.virtual_light_luminance,
                &mut invalid,
            ),
        };
        if invalid {
            once!(warn!(
                "AutoWhiteBalance::speed and AutoWhiteBalance::virtual_light_luminance must be finite and not negative; using the defaults for invalid values"
            ));
        }
        sanitized
    }
}

impl SyncComponent<RenderApp> for AutoWhiteBalance {
    type Target = ExternalWhiteBalance;
}

impl ExtractComponent<RenderApp> for AutoWhiteBalance {
    type QueryData = Has<AutoExposure>;
    type QueryFilter = With<Self>;
    type Out = ExternalWhiteBalance;

    fn extract_component(
        has_auto_exposure: QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        has_auto_exposure.then_some(ExternalWhiteBalance)
    }
}
