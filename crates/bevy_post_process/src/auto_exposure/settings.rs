use core::ops::RangeInclusive;

use super::{
    buffers::build_uniform, compensation_curve::AutoExposureCompensationCurve,
    pipeline::AutoExposureUniform, white_balance::AutoWhiteBalance,
};
use bevy_asset::Handle;
use bevy_camera::Hdr;
use bevy_ecs::{prelude::Component, query::QueryItem, reflect::ReflectComponent};
use bevy_image::Image;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    extract_component::ExtractComponent, sync_component::SyncComponent,
    view::NeedsSceneLinearTarget, RenderApp,
};
use bevy_utils::{default, once};
use tracing::warn;

/// Component that enables auto exposure for an HDR-enabled 2d or 3d camera.
///
/// Auto exposure adjusts the exposure of the camera automatically to
/// simulate the human eye's ability to adapt to different lighting conditions.
///
/// Bevy's implementation builds a 64 bin histogram of the scene's luminance,
/// and then adjusts the exposure so that the average brightness of the final
/// render will be middle gray. Because it's using a histogram, some details can
/// be selectively ignored or emphasized. Outliers like shadows and specular
/// highlights can be ignored, and certain areas can be given more (or less)
/// weight based on a mask.
///
/// # Usage Notes
///
/// **Auto Exposure requires compute shaders and is not compatible with WebGL2.**
#[derive(Component, Clone, Reflect)]
#[reflect(Component, Default, Clone)]
#[require(Hdr, NeedsSceneLinearTarget)]
pub struct AutoExposure {
    /// The range of exposure values for the histogram.
    ///
    /// Pixel values below this range will be ignored, and pixel values above this range will be
    /// clamped in the sense that they will count towards the highest bin in the histogram.
    /// The default value is `-8.0..=8.0`.
    pub range: RangeInclusive<f32>,

    /// The portion of the histogram to consider when metering.
    ///
    /// By default, the darkest 10% and the brightest 10% of samples are ignored,
    /// so the default value is `0.10..=0.90`.
    pub filter: RangeInclusive<f32>,

    /// The speed at which the exposure adapts from dark to bright scenes, in F-stops per second.
    pub speed_brighten: f32,

    /// The speed at which the exposure adapts from bright to dark scenes, in F-stops per second.
    pub speed_darken: f32,

    /// The distance in F-stops from the target exposure from where to transition from animating
    /// in linear fashion to animating exponentially. This helps against jittering when the
    /// target exposure keeps on changing slightly from frame to frame, while still maintaining
    /// a relatively slow animation for big changes in scene brightness.
    ///
    /// ```text
    /// ev
    ///                       ➔●┐
    /// |              ⬈         ├ exponential section
    /// │        ⬈               ┘
    /// │    ⬈                   ┐
    /// │  ⬈                     ├ linear section
    /// │⬈                       ┘
    /// ●───────────────────────── time
    /// ```
    ///
    /// The default value is 1.5.
    pub exponential_transition_distance: f32,

    /// The mask to apply when metering. The mask will cover the entire screen, where:
    /// * `(0.0, 0.0)` is the top-left corner,
    /// * `(1.0, 1.0)` is the bottom-right corner.
    ///
    /// Only the red channel of the texture is used.
    /// The sample at the current screen position will be used to weight the contribution
    /// of each pixel to the histogram:
    /// * 0.0 means the pixel will not contribute to the histogram,
    /// * 1.0 means the pixel will contribute fully to the histogram.
    ///
    /// The default value is a white image, so all pixels contribute equally.
    ///
    /// # Usage Notes
    ///
    /// The mask is quantized to 16 discrete levels because of limitations in the compute shader
    /// implementation.
    pub metering_mask: Handle<Image>,

    /// Exposure compensation curve to apply after metering.
    /// The default value is a flat line at 0.0.
    /// For more information, see [`AutoExposureCompensationCurve`].
    pub compensation_curve: Handle<AutoExposureCompensationCurve>,

    /// A constant bias, in exposure values (EV), added to the metered scene luminance.
    ///
    /// A positive bias meters the scene as brighter than it is, so the image gets darker.
    /// The offset applies before the [`compensation_curve`](Self::compensation_curve).
    ///
    /// The default value is 0.0.
    pub metering_bias: f32,

    /// Two-stage adaptation, layering a slow long-term envelope on the short-term
    /// smoothing. See [`PhysiologicalAdaptation`].
    ///
    /// The default value is `None`, which uses only the short-term smoothing.
    pub physiological: Option<PhysiologicalAdaptation>,
}

impl Default for AutoExposure {
    fn default() -> Self {
        Self {
            range: -8.0..=8.0,
            filter: 0.10..=0.90,
            speed_brighten: 3.0,
            speed_darken: 1.0,
            exponential_transition_distance: 1.5,
            metering_mask: default(),
            compensation_curve: default(),
            metering_bias: 0.0,
            physiological: None,
        }
    }
}

impl SyncComponent<RenderApp> for AutoExposure {
    type Target = (AutoExposure, AutoExposureUniform);
}

impl ExtractComponent<RenderApp> for AutoExposure {
    type QueryData = (&'static AutoExposure, Option<&'static AutoWhiteBalance>);
    type QueryFilter = ();
    // The uniform folds in the optional `AutoWhiteBalance`, so extraction builds it here.
    // It runs every frame without change detection, which also covers `AutoWhiteBalance`
    // being removed on its own. The sanitization warnings in `build_uniform` stay once-only.
    type Out = (AutoExposure, AutoExposureUniform);

    fn extract_component(
        (settings, white_balance): QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        Some((settings.clone(), build_uniform(settings, white_balance)))
    }
}

/// Settings for two-stage physiological exposure adaptation, enabled through
/// [`AutoExposure::physiological`]. The model is the one Gran Turismo 7 presented at
/// SIGGRAPH 2025 ("Physically Based Tone Mapping in Gran Turismo 7").
///
/// Human vision adapts to brightness on two time scales:
///
/// * A short-term stage (pupil constriction and neural gain) covers a few EV and reacts
///   within seconds. This is the regular [`AutoExposure`] smoothing, driven by
///   [`AutoExposure::speed_brighten`] and [`AutoExposure::speed_darken`].
/// * A long-term stage (receptor sensitivity and photopigment bleaching) covers the
///   remaining range, about 12 EV, over minutes to tens of minutes. Adapting to light is
///   much faster than adapting to darkness.
///
/// The long-term envelope clamps the short-term exposure to
/// `[envelope - bound_brighten, envelope + bound_darken]`. The envelope keeps tracking the
/// short-term exposure while this setting is `None`, so enabling it at runtime is smooth.
///
/// All speeds and bounds must be finite and non-negative. Invalid values are reset to their
/// defaults when the settings are uploaded to the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
#[reflect(Default, Clone, PartialEq)]
pub struct PhysiologicalAdaptation {
    /// Speed of the long-term envelope when the scene gets brighter, in F-stops per second.
    ///
    /// The default value is 0.05, which covers 12 EV in about 4 minutes.
    pub speed_brighten: f32,

    /// Speed of the long-term envelope when the scene gets darker, in F-stops per second.
    ///
    /// The default value is 0.01, which covers 12 EV in about 20 minutes.
    pub speed_darken: f32,

    /// How far below the long-term envelope the short-term exposure may drop, in EV,
    /// when the scene gets brighter.
    ///
    /// The default value is 3.0.
    pub bound_brighten: f32,

    /// How far above the long-term envelope the short-term exposure may rise, in EV,
    /// when the scene gets darker.
    ///
    /// The default value is 2.0.
    pub bound_darken: f32,
}

impl Default for PhysiologicalAdaptation {
    fn default() -> Self {
        Self {
            speed_brighten: 0.05,
            speed_darken: 0.01,
            bound_brighten: 3.0,
            bound_darken: 2.0,
        }
    }
}

impl PhysiologicalAdaptation {
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
            speed_brighten: sanitize(self.speed_brighten, defaults.speed_brighten),
            speed_darken: sanitize(self.speed_darken, defaults.speed_darken),
            bound_brighten: sanitize(self.bound_brighten, defaults.bound_brighten),
            bound_darken: sanitize(self.bound_darken, defaults.bound_darken),
        };

        if invalid {
            once!(warn!(
                "PhysiologicalAdaptation speeds and bounds must be finite and non-negative; \
                invalid fields were reset to their defaults"
            ));
        }

        sanitized
    }
}
