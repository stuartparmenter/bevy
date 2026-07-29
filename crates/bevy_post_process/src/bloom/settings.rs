use super::downsampling_pipeline::BloomUniforms;
use bevy_camera::{Camera, Hdr};
use bevy_ecs::{
    prelude::Component,
    query::{QueryItem, With},
    reflect::ReflectComponent,
};
use bevy_math::Vec2;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    extract_component::ExtractComponent, sync_component::SyncComponent,
    view::NeedsSceneLinearTarget, RenderApp,
};

/// Applies a bloom effect to an HDR-enabled 2d or 3d camera.
///
/// Bloom emulates an effect found in real cameras and the human eye,
/// causing halos to appear around very bright parts of the scene.
///
/// See also <https://en.wikipedia.org/wiki/Bloom_(shader_effect)>.
///
/// # Usage Notes
///
/// Often used in conjunction with `bevy_pbr::StandardMaterial::emissive` for 3d meshes.
///
/// Bloom is best used alongside a tonemapping function that desaturates bright colors,
/// such as [`bevy_core_pipeline::tonemapping::Tonemapping::TonyMcMapface`].
///
/// Bevy's implementation uses a parametric curve to blend between a set of
/// blurred (lower frequency) images generated from the camera's view.
/// See <https://starlederer.github.io/bloom/> for a visualization of the parametric curve
/// used in Bevy as well as a visualization of the curve's respective scattering profile.
#[derive(Component, Reflect, Clone)]
#[reflect(Component, Default, Clone)]
#[require(Hdr, NeedsSceneLinearTarget)]
pub struct Bloom {
    /// Controls the baseline of how much the image is scattered (default: 0.15).
    ///
    /// This parameter should be used only to control the strength of the bloom
    /// for the scene as a whole. Increasing it too much will make the scene appear
    /// blurry and over-exposed.
    ///
    /// To make a mesh glow brighter, rather than increase the bloom intensity,
    /// you should increase the mesh's `emissive` value.
    ///
    /// # In energy-conserving mode
    /// The value represents how likely the light is to scatter.
    ///
    /// The value should be between 0.0 and 1.0 where:
    /// * 0.0 means no bloom
    /// * 1.0 means the light is scattered as much as possible
    ///
    /// # In additive mode
    /// The value represents how much scattered light is added to
    /// the image to create the glow effect.
    ///
    /// In this configuration:
    /// * 0.0 means no bloom
    /// * Greater than 0.0 means a proportionate amount of scattered light is added
    pub intensity: f32,

    /// Low frequency contribution boost.
    /// Controls how much more likely the light
    /// is to scatter completely sideways (low frequency image).
    ///
    /// Comparable to a low shelf boost on an equalizer.
    ///
    /// # In energy-conserving mode
    /// The value should be between 0.0 and 1.0 where:
    /// * 0.0 means low frequency light uses base intensity for blend factor calculation
    /// * 1.0 means low frequency light contributes at full power
    ///
    /// # In additive mode
    /// The value represents how much scattered light is added to
    /// the image to create the glow effect.
    ///
    /// In this configuration:
    /// * 0.0 means no bloom
    /// * Greater than 0.0 means a proportionate amount of scattered light is added
    pub low_frequency_boost: f32,

    /// Low frequency contribution boost curve.
    /// Controls the curvature of the blend factor function
    /// making frequencies next to the lowest ones contribute more.
    ///
    /// Somewhat comparable to the Q factor of an equalizer node.
    ///
    /// Valid range:
    /// * 0.0 - base intensity and boosted intensity are linearly interpolated
    /// * 1.0 - all frequencies below maximum are at boosted intensity level
    pub low_frequency_boost_curvature: f32,

    /// Tightens how much the light scatters (default: 1.0).
    ///
    /// Valid range:
    /// * 0.0 - maximum scattering angle is 0 degrees (no scattering)
    /// * 1.0 - maximum scattering angle is 90 degrees
    pub high_pass_frequency: f32,

    /// Controls the threshold filter used for extracting the brightest regions from the input image
    /// before blurring them and compositing back onto the original image.
    ///
    /// Changing these settings creates a physically inaccurate image and makes it easy to make
    /// the final result look worse. However, they can be useful when emulating the 1990s-2000s game look.
    /// See [`BloomPrefilter`] for more information.
    pub prefilter: BloomPrefilter,

    /// Controls whether bloom textures
    /// are blended between or added to each other. Useful
    /// if image brightening is desired and a must-change
    /// if `prefilter` is used.
    ///
    /// # Recommendation
    /// Set to [`BloomCompositeMode::Additive`] if `prefilter` is
    /// configured in a non-energy-conserving way,
    /// otherwise set to [`BloomCompositeMode::EnergyConserving`].
    pub composite_mode: BloomCompositeMode,

    /// Maximum size of each dimension for the largest mipchain texture used in downscaling/upscaling.
    /// Only tweak if you are seeing visual artifacts.
    ///
    /// The [`BloomScatterModel::Gt7Glare`] glare pattern's on-screen angular
    /// size is calibrated for the default value of `512`; other values rescale
    /// the glare pattern accordingly.
    pub max_mip_dimension: u32,

    /// Amount to stretch the bloom on each axis. Artistic control, can be used to emulate
    /// anamorphic blur by using a large x-value. For large values, you may need to increase
    /// [`Bloom::max_mip_dimension`] to reduce sampling artifacts.
    pub scale: Vec2,

    /// Selects how the blurred pyramid levels are weighted when composited
    /// back onto the image (default: [`BloomScatterModel::Aesthetic`], the
    /// parametric curve).
    ///
    /// [`BloomScatterModel::Gt7Glare`] replaces the parametric curve with
    /// per-level weights derived from the diffraction pattern of a camera
    /// aperture, turning bloom into a physically based veiling glare in the
    /// style of Gran Turismo 7. See [`BloomScatterModel`] for the parameters
    /// it overrides.
    pub scatter: BloomScatterModel,
}

impl Bloom {
    const DEFAULT_MAX_MIP_DIMENSION: u32 = 512;

    /// The default bloom preset.
    ///
    /// This uses the [`EnergyConserving`](BloomCompositeMode::EnergyConserving) composite mode.
    pub const NATURAL: Self = Self {
        intensity: 0.15,
        low_frequency_boost: 0.7,
        low_frequency_boost_curvature: 0.95,
        high_pass_frequency: 1.0,
        prefilter: BloomPrefilter {
            threshold: 0.0,
            threshold_nits: None,
            threshold_softness: 0.0,
        },
        composite_mode: BloomCompositeMode::EnergyConserving,
        max_mip_dimension: Self::DEFAULT_MAX_MIP_DIMENSION,
        scale: Vec2::ONE,
        scatter: BloomScatterModel::Aesthetic,
    };

    /// Physically based veiling glare in the style of Gran Turismo 7: the
    /// pyramid levels are weighted by the diffraction pattern of an f/5.6
    /// camera aperture instead of the parametric curve. See
    /// [`BloomScatterModel::Gt7Glare`].
    pub const GT7_GLARE: Self = Self {
        scatter: BloomScatterModel::Gt7Glare {
            f_number: BloomScatterModel::DEFAULT_F_NUMBER,
        },
        ..Self::NATURAL
    };

    /// Emulates the look of stylized anamorphic bloom, stretched horizontally.
    pub const ANAMORPHIC: Self = Self {
        // The larger scale necessitates a larger resolution to reduce artifacts:
        max_mip_dimension: Self::DEFAULT_MAX_MIP_DIMENSION * 2,
        scale: Vec2::new(4.0, 1.0),
        ..Self::NATURAL
    };

    /// A preset that's similar to how older games did bloom.
    pub const OLD_SCHOOL: Self = Self {
        intensity: 0.05,
        low_frequency_boost: 0.7,
        low_frequency_boost_curvature: 0.95,
        high_pass_frequency: 1.0,
        prefilter: BloomPrefilter {
            threshold: 0.6,
            threshold_nits: None,
            threshold_softness: 0.2,
        },
        composite_mode: BloomCompositeMode::Additive,
        max_mip_dimension: Self::DEFAULT_MAX_MIP_DIMENSION,
        scale: Vec2::ONE,
        scatter: BloomScatterModel::Aesthetic,
    };

    /// A preset that applies a very strong bloom, and blurs the whole screen.
    pub const SCREEN_BLUR: Self = Self {
        intensity: 1.0,
        low_frequency_boost: 0.0,
        low_frequency_boost_curvature: 0.0,
        high_pass_frequency: 1.0 / 3.0,
        prefilter: BloomPrefilter {
            threshold: 0.0,
            threshold_nits: None,
            threshold_softness: 0.0,
        },
        composite_mode: BloomCompositeMode::EnergyConserving,
        max_mip_dimension: Self::DEFAULT_MAX_MIP_DIMENSION,
        scale: Vec2::ONE,
        scatter: BloomScatterModel::Aesthetic,
    };

    /// Returns `true` when the downsampling passes must apply the
    /// soft-threshold prefilter curve.
    ///
    /// This is [`BloomPrefilter::is_active`] gated on the scatter model:
    /// [`BloomScatterModel::Gt7Glare`] always integrates the total scene
    /// energy (a physical point-spread function has no brightness cutoff;
    /// GT7's glare is threshold-free), so any configured prefilter is
    /// ignored under it.
    pub(crate) fn thresholding_active(&self) -> bool {
        matches!(self.scatter, BloomScatterModel::Aesthetic) && self.prefilter.is_active()
    }

    /// The composite mode the upsampling pipeline actually uses.
    ///
    /// [`BloomScatterModel::Gt7Glare`] derives its blend constants as
    /// chained energy-conserving lerp factors (see `bloom::glare`), so it
    /// forces [`BloomCompositeMode::EnergyConserving`] regardless of
    /// [`composite_mode`](Self::composite_mode);
    /// [`BloomScatterModel::Aesthetic`] uses the configured mode unchanged.
    pub(crate) fn effective_composite_mode(&self) -> BloomCompositeMode {
        match self.scatter {
            BloomScatterModel::Aesthetic => self.composite_mode,
            BloomScatterModel::Gt7Glare { .. } => BloomCompositeMode::EnergyConserving,
        }
    }
}

impl Default for Bloom {
    fn default() -> Self {
        Self::NATURAL
    }
}

/// How [`Bloom`] distributes scattered light across its blur pyramid when
/// compositing it back onto the image.
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect)]
#[reflect(Default, Clone, PartialEq)]
pub enum BloomScatterModel {
    /// The hand-tuned parametric curve, shaped
    /// by [`Bloom::intensity`], [`Bloom::low_frequency_boost`],
    /// [`Bloom::low_frequency_boost_curvature`] and
    /// [`Bloom::high_pass_frequency`]. This is the default.
    #[default]
    Aesthetic,

    /// Physically based veiling glare inspired by Gran Turismo 7 (SIGGRAPH
    /// 2025): the per-level weights are derived from the Fraunhofer
    /// diffraction point-spread function (the polychromatic Airy pattern) of
    /// an ideal circular aperture at the given F-number, instead of the
    /// parametric curve.
    ///
    /// GT7's own 240 hand-calibrated weights are unpublished; Bevy derives
    /// the weights from the documented physical model (see the
    /// `bloom::glare` module docs for the derivation and references), so
    /// this is *inspired by* GT7 rather than a clone of its constants.
    ///
    /// Under this model:
    /// - [`Bloom::intensity`] keeps its meaning as the total fraction of
    ///   energy scattered out of the sharp image (energy-conserving), while
    ///   the F-number shapes how that energy spreads across blur radii.
    /// - The [`Bloom::prefilter`] threshold is ignored (a physical PSF
    ///   applies to all light; GT7's glare is threshold-free) and
    ///   [`Bloom::composite_mode`] is forced to
    ///   [`BloomCompositeMode::EnergyConserving`].
    /// - [`Bloom::low_frequency_boost`],
    ///   [`Bloom::low_frequency_boost_curvature`] and
    ///   [`Bloom::high_pass_frequency`] are unused.
    ///
    /// The glare's on-screen angular size is calibrated for the default
    /// [`Bloom::max_mip_dimension`] of `512`; other values rescale the glare
    /// pattern accordingly.
    Gt7Glare {
        /// The aperture F-number (focal length over aperture diameter) of
        /// the virtual camera, clamped to the standard full-stop ladder
        /// range `[1.0, 22.0]`.
        ///
        /// Diffraction scales with the F-number: small values (f/1.0, wide
        /// aperture) produce a tight glare that falls off steeply with
        /// radius, large values (f/22, stopped down) spread the energy into
        /// a wide, soft veil. Defaults to f/5.6
        /// ([`BloomScatterModel::DEFAULT_F_NUMBER`]). Non-finite or
        /// non-positive values fall back to the default with a warning.
        f_number: f32,
    },
}

impl BloomScatterModel {
    /// The default aperture F-number for [`Self::Gt7Glare`] (mid-ladder,
    /// a common photographic walk-around aperture).
    pub const DEFAULT_F_NUMBER: f32 = super::glare::DEFAULT_F_NUMBER;
}

/// Applies a threshold filter to the input image to extract the brightest
/// regions before blurring them and compositing back onto the original image.
/// These settings are useful when emulating the 1990s-2000s game look.
///
/// # Considerations
/// * Changing these settings creates a physically inaccurate image
/// * Changing these settings makes it easy to make the final result look worse
/// * Non-default prefilter settings should be used in conjunction with [`BloomCompositeMode::Additive`]
#[derive(Default, Clone, Reflect)]
#[reflect(Clone, Default)]
pub struct BloomPrefilter {
    /// Baseline of the quadratic threshold curve (default: 0.0).
    ///
    /// RGB values under the threshold curve will not contribute to the effect.
    ///
    /// This is expressed in raw scene-linear framebuffer values, where `1.0`
    /// corresponds to SDR paper white at the tone-map operator output. On HDR
    /// display targets, where paper white is configurable
    /// (`DisplayTarget::paper_white_nits` on the window), consider
    /// [`threshold_nits`](Self::threshold_nits) instead so the cutoff keeps a
    /// fixed physical meaning.
    pub threshold: f32,

    /// Optional luminance threshold expressed in nits (default: `None`).
    ///
    /// When set, this takes precedence over [`threshold`](Self::threshold):
    /// at prepare time it is divided by the paper white (in nits) of the
    /// view's *resolved* display target (`DisplayTarget::paper_white_nits`,
    /// 100 nits for plain SDR targets) to produce the framebuffer-value
    /// threshold the shader uses. This keeps the cutoff anchored to a
    /// physical brightness on displays with a non-default paper white,
    /// instead of silently re-scaling with it.
    ///
    /// `Some(0.0)` (or a negative value) disables thresholding, like a
    /// `threshold` of `0.0` does.
    pub threshold_nits: Option<f32>,

    /// Controls how much to blend between the thresholded and non-thresholded colors (default: 0.0).
    ///
    /// 0.0 = Abrupt threshold, no blending
    /// 1.0 = Fully soft threshold
    ///
    /// Values outside of the range [0.0, 1.0] will be clamped.
    pub threshold_softness: f32,
}

impl BloomPrefilter {
    /// Returns `true` when this prefilter requests any thresholding, i.e.
    /// when the bloom downsampling pipeline must apply the soft-threshold
    /// curve ([`threshold_nits`](Self::threshold_nits) takes precedence over
    /// [`threshold`](Self::threshold)).
    pub fn is_active(&self) -> bool {
        match self.threshold_nits {
            Some(nits) => nits > 0.0,
            None => self.threshold > 0.0,
        }
    }

    /// Resolves the threshold to scene-linear framebuffer units (`1.0` =
    /// paper white at the tone-map operator output) against the given paper
    /// white in nits.
    ///
    /// Returns [`threshold`](Self::threshold) unchanged when
    /// [`threshold_nits`](Self::threshold_nits) is `None`; `paper_white_nits`
    /// must be positive and finite (use
    /// `DisplayTarget::sanitized_paper_white_nits`).
    pub fn resolve_threshold(&self, paper_white_nits: f32) -> f32 {
        match self.threshold_nits {
            Some(nits) => (nits / paper_white_nits).max(0.0),
            None => self.threshold,
        }
    }
}

#[derive(Debug, Clone, Reflect, PartialEq, Eq, Hash, Copy)]
#[reflect(Clone, Hash, PartialEq)]
pub enum BloomCompositeMode {
    EnergyConserving,
    Additive,
}

impl SyncComponent<RenderApp> for Bloom {
    type Target = (Self, BloomUniforms);
}

impl ExtractComponent<RenderApp> for Bloom {
    type QueryData = (&'static Self, &'static Camera);
    type QueryFilter = With<Hdr>;
    type Out = Self;

    fn extract_component((bloom, camera): QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        // `prepare_bloom_uniforms` builds the `BloomUniforms` in the render
        // world, where the view's resolved display target is available. This
        // guard admits only active cameras with a drawable viewport, so the
        // uniform math (aspect ratio, viewport normalization) never sees a
        // zero dimension.
        match (
            camera.physical_viewport_rect(),
            camera.physical_viewport_size(),
            camera.physical_target_size(),
            camera.is_active,
        ) {
            (Some(_), Some(size), Some(_), true) if size.x != 0 && size.y != 0 => {
                Some(bloom.clone())
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec4;

    /// The precomputation packing must stay bit-identical to the reference
    /// inline math for `threshold`-based prefilters.
    #[test]
    fn threshold_precomputations_match_reference_inline_math() {
        for (threshold, threshold_softness) in [
            (0.0_f32, 0.0_f32),
            (0.6, 0.2),
            (1.0, 1.0),
            (2.5, 0.5),
            (0.6, 2.0),
        ] {
            let knee = threshold * threshold_softness.clamp(0.0, 1.0);
            let reference = Vec4::new(
                threshold,
                threshold - knee,
                2.0 * knee,
                0.25 / (knee + 0.00001),
            );
            let shared = BloomUniforms::threshold_precomputations(threshold, threshold_softness);
            assert_eq!(
                reference.to_array().map(f32::to_bits),
                shared.to_array().map(f32::to_bits)
            );
        }
    }

    #[test]
    fn prefilter_activity() {
        // A non-zero framebuffer-value `threshold` activates the prefilter.
        let fb = BloomPrefilter {
            threshold: 0.6,
            ..Default::default()
        };
        assert!(fb.is_active());
        assert!(!BloomPrefilter::default().is_active());

        // `threshold_nits` takes precedence, including `Some(0.0)` disabling
        // a non-zero `threshold`.
        let nits = BloomPrefilter {
            threshold: 0.0,
            threshold_nits: Some(120.0),
            ..Default::default()
        };
        assert!(nits.is_active());
        let disabled = BloomPrefilter {
            threshold: 0.6,
            threshold_nits: Some(0.0),
            ..Default::default()
        };
        assert!(!disabled.is_active());
        assert!(!BloomPrefilter {
            threshold_nits: Some(-5.0),
            ..Default::default()
        }
        .is_active());
    }

    #[test]
    fn resolve_threshold_converts_nits_by_paper_white() {
        // Without nits the framebuffer value passes through untouched, for
        // any paper white.
        let fb = BloomPrefilter {
            threshold: 0.6,
            ..Default::default()
        };
        assert_eq!(fb.resolve_threshold(100.0), 0.6);
        assert_eq!(fb.resolve_threshold(203.0), 0.6);

        // 200 nits is 2x paper white on a 100-nit target and exactly paper
        // white on a 200-nit target.
        let nits = BloomPrefilter {
            threshold: 123.0, // ignored
            threshold_nits: Some(200.0),
            ..Default::default()
        };
        assert_eq!(nits.resolve_threshold(100.0), 2.0);
        assert_eq!(nits.resolve_threshold(200.0), 1.0);

        // Degenerate values clamp to "no threshold".
        assert_eq!(
            BloomPrefilter {
                threshold_nits: Some(-50.0),
                ..Default::default()
            }
            .resolve_threshold(100.0),
            0.0
        );
    }
}
