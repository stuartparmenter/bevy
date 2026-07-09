use core::ops::RangeInclusive;

use super::compensation_curve::AutoExposureCompensationCurve;
use bevy_asset::Handle;
use bevy_camera::Hdr;
use bevy_ecs::{prelude::Component, reflect::ReflectComponent};
use bevy_image::Image;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{extract_component::ExtractComponent, RenderApp};
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
#[derive(Component, Clone, Reflect, ExtractComponent)]
#[reflect(Component, Default, Clone)]
#[require(Hdr)]
#[extract_app(RenderApp)]
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

    /// A constant bias, in exposure values (EV), added to the metered scene luminance after
    /// all metering references have been fused.
    ///
    /// A positive bias makes the meter believe the scene is brighter than it measured, which
    /// results in a *darker* final image; a negative bias results in a brighter image. This is
    /// equivalent to a constant offset in the [`compensation_curve`](Self::compensation_curve),
    /// but is applied at the metering seam where external references
    /// (see [`AutoExposureExternalReference`](super::AutoExposureExternalReference))
    /// are blended in.
    ///
    /// The default value is 0.0, which leaves the metered value untouched.
    pub metering_bias: f32,

    /// Optional two-stage "physiological" adaptation, layering a slow long-term adaptation
    /// envelope on top of the regular short-term smoothing. See [`PhysiologicalAdaptation`]
    /// for details.
    ///
    /// The default value is `None`, which preserves the classic single-stage behavior.
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

/// Settings for two-stage physiological exposure adaptation, enabled through
/// [`AutoExposure::physiological`].
///
/// Human vision adapts to brightness changes on two distinct time scales, a model
/// popularized for real-time rendering by Gran Turismo 7 (SIGGRAPH 2025,
/// "Physically Based Tone Mapping in Gran Turismo 7"):
///
/// * A **short-term** stage — pupil constriction and neural gain — that covers a few EV and
///   reacts within seconds. This corresponds to the regular [`AutoExposure`] smoothing,
///   driven by [`AutoExposure::speed_brighten`] and [`AutoExposure::speed_darken`].
/// * A **long-term** stage — receptor sensitivity and photopigment bleaching — that covers
///   the remaining adaptation range (on the order of 12 EV) and takes minutes to tens of
///   minutes to adapt, asymmetrically: adapting to light is much faster than adapting to
///   darkness.
///
/// When enabled, a slowly moving long-term adaptation envelope is tracked alongside the
/// short-term exposure, and the long-term envelope *bounds* the short-term result: the
/// short-term adapted exposure is clamped to
/// `[envelope - bound_brighten, envelope + bound_darken]`.
/// Walking from daylight into a cave, the image quickly brightens by at most
/// [`bound_darken`](Self::bound_darken) EV, and then continues to brighten slowly as the
/// long-term envelope catches up — dark scenes stay dark until the eye has had time to truly
/// adapt. Because the long-term speeds are asymmetric, the same scene luminance can produce
/// a different perceived exposure at dawn than at dusk.
///
/// The long-term envelope state lives on the GPU next to the short-term exposure state, and
/// keeps tracking (without bounding) even while this setting is `None`, so enabling it at
/// runtime transitions smoothly.
///
/// All speeds and bounds must be non-negative and finite; invalid values are reset to their
/// defaults when the settings are uploaded to the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
#[reflect(Default, Clone, PartialEq)]
pub struct PhysiologicalAdaptation {
    /// The speed at which the long-term envelope adapts from dark to bright scenes,
    /// in F-stops per second.
    ///
    /// The default value is 0.05, fully adapting to light across ~12 EV in about 4 minutes.
    pub speed_brighten: f32,

    /// The speed at which the long-term envelope adapts from bright to dark scenes,
    /// in F-stops per second. Dark adaptation is slower than light adaptation.
    ///
    /// The default value is 0.01, fully adapting to darkness across ~12 EV in about
    /// 20 minutes.
    pub speed_darken: f32,

    /// How far below the long-term envelope the short-term exposure may drop, in EV,
    /// when adapting to a scene that became brighter.
    ///
    /// The default value is 3.0.
    pub bound_brighten: f32,

    /// How far above the long-term envelope the short-term exposure may rise, in EV,
    /// when adapting to a scene that became darker. This is the bound that keeps dark
    /// scenes dark until the long-term envelope catches up.
    ///
    /// The default value is 2.0.
    pub bound_darken: f32,

    /// The initial value of the long-term envelope, in EV, used when the per-camera
    /// adaptation state is first created (i.e. when auto exposure is first enabled for a
    /// camera). Use this to start a camera as if it had already adapted to a known
    /// environment, e.g. bright daylight.
    ///
    /// If `None` (the default), the envelope starts at the same neutral initial value as the
    /// short-term exposure. Changing this value at runtime has no effect on cameras whose
    /// adaptation state already exists; state is intentionally preserved across settings
    /// changes to keep the adaptation continuous.
    pub initial_long_term_ev: Option<f32>,
}

impl Default for PhysiologicalAdaptation {
    fn default() -> Self {
        Self {
            speed_brighten: 0.05,
            speed_darken: 0.01,
            bound_brighten: 3.0,
            bound_darken: 2.0,
            initial_long_term_ev: None,
        }
    }
}

impl PhysiologicalAdaptation {
    /// Returns a copy of these settings with invalid fields reset to their default values.
    ///
    /// Speeds and bounds must be finite and non-negative;
    /// [`initial_long_term_ev`](Self::initial_long_term_ev) must be finite if present.
    /// Warns (once) if any field had to be reset.
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
            initial_long_term_ev: match self.initial_long_term_ev {
                Some(ev) if !ev.is_finite() => {
                    invalid = true;
                    None
                }
                ev => ev,
            },
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

/// An externally computed metering reference for [`AutoExposure`].
///
/// Bevy's auto exposure meters the rendered frame through a luminance histogram. That is
/// accurate and flexible, but inherently limited to the camera frustum: it cannot anticipate
/// brightness outside the current view, and it provides no long-term stability across camera
/// cuts. Gran Turismo 7 addresses this by fusing **five** metering references with fixed
/// weights: the frame buffer histogram, light probes, the sky dome, direct sun illuminance,
/// and SH-based sky visibility.
///
/// This component is the extensible seam for that multi-reference pattern. A system that can
/// estimate the scene's brightness from non-frame-buffer data — for example from
/// `DirectionalLight::illuminance`, an environment map's integrated luminance, or a
/// procedural sky model — can write its estimate here (typically every frame). The
/// `(ev, weight)` pair is uploaded with the auto exposure settings, and the metering compute
/// shader fuses it with the histogram measurement as a weighted average:
///
/// ```text
/// metered_ev = (histogram_ev + ev * weight) / (1.0 + weight)
/// ```
///
/// where the frame buffer histogram always carries a weight of `1.0`.
///
/// The component is user-driven, and only a single fused external reference is supported.
/// If multiple sources need to be combined (e.g. a sky dome, sun illuminance, and light
/// probe luminance), fuse them into one `ev`/`weight` pair before writing the component.
///
/// This component only has an effect on cameras that also have [`AutoExposure`].
#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect)]
#[reflect(Component, Default, Clone, PartialEq)]
pub struct AutoExposureExternalReference {
    /// The reference's estimate of the scene's average log2 luminance, in the same units as
    /// the histogram average that [`AutoExposure`] meters from the frame buffer (compare
    /// [`AutoExposure::range`]). Unlike the histogram average, this value is not clamped to
    /// [`AutoExposure::range`].
    ///
    /// Must be finite; non-finite values cause the reference to be ignored.
    pub ev: f32,

    /// The weight of this reference relative to the frame buffer histogram, which always has
    /// a weight of `1.0`. A weight of `0.0` disables the reference, `1.0` averages it equally
    /// with the histogram, and larger values let it dominate.
    ///
    /// Must be finite and non-negative; invalid values cause the reference to be ignored.
    ///
    /// The default value is 1.0.
    pub weight: f32,
}

impl Default for AutoExposureExternalReference {
    fn default() -> Self {
        Self {
            ev: 0.0,
            weight: 1.0,
        }
    }
}
