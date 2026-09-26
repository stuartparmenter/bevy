use bevy_color::RgbPrimaries;
use bevy_ecs::prelude::Component;
use bevy_platform::sync::atomic::{AtomicBool, Ordering};

use crate::{DisplayProvenance, FieldProvenance, MonitorDisplayCapability, WindowDisplayState};

#[cfg(feature = "bevy_reflect")]
use {
    bevy_ecs::prelude::ReflectComponent,
    bevy_reflect::{std_traits::ReflectDefault, Reflect},
};

#[cfg(all(feature = "serialize", feature = "bevy_reflect"))]
use bevy_reflect::{ReflectDeserialize, ReflectSerialize};

/// Requests the color space and luminance for a [`Window`](crate::Window)'s
/// output.
///
/// The default is SDR sRGB. Set [`hdr`](Self::hdr) to get HDR output where
/// the display supports it, or
/// [`color_space_override`](Self::color_space_override) to choose the color
/// space yourself. The surface may not support the request, so the output
/// can differ from it. The wgpu [color space and HDR primer] explains what
/// each backend can present.
///
/// Leave a luminance field `None` unless your app calibrates it. The renderer
/// then uses what the display reports, or a default for the color space.
///
/// A required component of [`Window`](crate::Window). Bevy never writes this
/// component.
///
/// Adding the `Hdr` component to a camera does not give HDR display output.
/// It only changes the format of the texture the camera renders to.
///
/// # Example
///
/// Request HDR output. The color space it gets depends on the platform and
/// the display:
///
/// ```
/// # use bevy_ecs::world::World;
/// # use bevy_window::{DisplayTarget, Window};
/// # let mut world = World::new();
/// world.spawn((
///     Window::default(),
///     DisplayTarget {
///         hdr: true,
///         ..Default::default()
///     },
/// ));
/// ```
///
/// [color space and HDR primer]: https://docs.rs/wgpu/latest/wgpu/index.html#surface-color-spaces-and-hdr-output
#[derive(Component, Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Component, Default, Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct DisplayTarget {
    /// Requests HDR output. Bevy picks the best HDR color space the surface
    /// supports, or SDR when it supports none. Most apps should use this.
    pub hdr: bool,
    /// A color space to use instead of the one Bevy picks, for example to
    /// prefer scRGB over PQ. When `Some`, [`hdr`](Self::hdr) is ignored. When
    /// the surface does not support it, the output is SDR.
    pub color_space_override: Option<SurfaceColorSpace>,
    /// The luminance of a plain white UI element, called paper white, in nits.
    /// A tonemapped value of `1.0` maps to it. On an HDR display, raise it to
    /// make the image brighter.
    ///
    /// SDR uses 100 nits. [ITU-R BT.2408] recommends 203 nits for HDR
    /// television.
    ///
    /// [ITU-R BT.2408]: https://www.itu.int/pub/R-REP-BT.2408
    pub paper_white_nits: Option<f32>,
    /// The highest luminance a highlight can reach, in nits.
    pub peak_luminance_nits: Option<f32>,
    /// The lowest luminance the display can show, in nits.
    pub min_luminance_nits: Option<f32>,
}

/// A color space a surface can present in: a set of primaries, a transfer
/// function, and a range.
///
/// Each variant corresponds to a wgpu [`SurfaceColorSpace`][wgpu], whose docs
/// describe the encoding and the backends that support it. HLG is left out
/// because it needs a scene-referred signal, which Bevy does not produce.
///
/// [wgpu]: https://docs.rs/wgpu/latest/wgpu/enum.SurfaceColorSpace.html
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Default, Debug, PartialEq, Hash, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub enum SurfaceColorSpace {
    /// [sRGB](https://registry.color.org/rgb-registry/srgb) of IEC 61966-2-1:
    /// the BT.709 primaries with the sRGB transfer function, in standard
    /// dynamic range. wgpu's [`SurfaceColorSpace::Srgb`].
    ///
    /// [`SurfaceColorSpace::Srgb`]: https://docs.rs/wgpu/latest/wgpu/enum.SurfaceColorSpace.html#variant.Srgb
    #[default]
    Srgb,
    /// Linear [scRGB] of IEC 61966-2-2: the BT.709 primaries with a linear
    /// transfer function. A value of `1.0` is 80 nits, and values above `1.0`
    /// and below `0.0` are valid. Colors outside BT.709 are encoded as
    /// out-of-range values. wgpu's [`SurfaceColorSpace::ExtendedSrgbLinear`].
    ///
    /// [scRGB]: https://en.wikipedia.org/wiki/ScRGB
    /// [`SurfaceColorSpace::ExtendedSrgbLinear`]: https://docs.rs/wgpu/latest/wgpu/enum.SurfaceColorSpace.html#variant.ExtendedSrgbLinear
    ScRgbLinear,
    /// HDR10 of [ITU-R BT.2100]: the BT.2020 primaries with the [perceptual
    /// quantizer] of SMPTE ST 2084. PQ encodes absolute luminance, with `1.0`
    /// at 10000 nits. wgpu's [`SurfaceColorSpace::Bt2100Pq`].
    ///
    /// [perceptual quantizer]: https://en.wikipedia.org/wiki/Perceptual_quantizer
    /// [ITU-R BT.2100]: https://www.itu.int/rec/R-REC-BT.2100
    /// [`SurfaceColorSpace::Bt2100Pq`]: https://docs.rs/wgpu/latest/wgpu/enum.SurfaceColorSpace.html#variant.Bt2100Pq
    Pq,
    /// Extended-range sRGB of IEC 61966-2-2: the BT.709 primaries with the
    /// sRGB transfer function continued above `1.0` for colors brighter than
    /// SDR white and mirrored below `0.0` for colors outside the gamut. This
    /// is the web HDR path. wgpu's [`SurfaceColorSpace::ExtendedSrgb`].
    ///
    /// [`SurfaceColorSpace::ExtendedSrgb`]: https://docs.rs/wgpu/latest/wgpu/enum.SurfaceColorSpace.html#variant.ExtendedSrgb
    ExtendedSrgb,
    /// [`ExtendedSrgb`](Self::ExtendedSrgb) with the Display P3 primaries.
    /// wgpu's [`SurfaceColorSpace::ExtendedDisplayP3`].
    ///
    /// [`SurfaceColorSpace::ExtendedDisplayP3`]: https://docs.rs/wgpu/latest/wgpu/enum.SurfaceColorSpace.html#variant.ExtendedDisplayP3
    ExtendedDisplayP3,
}

impl SurfaceColorSpace {
    /// Returns the primaries of this color space.
    pub const fn primaries(&self) -> RgbPrimaries {
        match self {
            Self::Srgb | Self::ScRgbLinear | Self::ExtendedSrgb => RgbPrimaries::BT709,
            Self::Pq => RgbPrimaries::BT2020,
            Self::ExtendedDisplayP3 => RgbPrimaries::DISPLAY_P3,
        }
    }

    /// Returns the transfer function of this color space.
    pub const fn transfer_function(&self) -> TransferFunction {
        match self {
            Self::Srgb => TransferFunction::Srgb,
            Self::ScRgbLinear => TransferFunction::Linear,
            Self::Pq => TransferFunction::Pq,
            Self::ExtendedSrgb | Self::ExtendedDisplayP3 => TransferFunction::ExtendedSrgb,
        }
    }

    /// Returns `true` if this color space has high dynamic range. Every color
    /// space except [`Srgb`](Self::Srgb) does.
    pub const fn is_hdr(&self) -> bool {
        !matches!(self, Self::Srgb)
    }
}

/// The transfer function of a [`SurfaceColorSpace`]: how linear color maps
/// to the signal the display decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, PartialEq, Hash, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub enum TransferFunction {
    /// The sRGB transfer function of IEC 61966-2-1, over `0.0` to `1.0`.
    Srgb,
    /// No transfer function. The signal is linear light, and values above
    /// `1.0` and below `0.0` are valid.
    Linear,
    /// The perceptual quantizer of SMPTE ST 2084, over absolute luminance up
    /// to 10000 nits.
    Pq,
    /// The sRGB transfer function continued above `1.0` and mirrored below
    /// `0.0`.
    ExtendedSrgb,
}

/// Logs a warning once per call site.
///
/// `bevy_window` has no dependency on `bevy_log`, so this is a local guard
/// over [`log::warn!`].
macro_rules! warn_once {
    ($($arg:tt)+) => {{
        static FIRED: AtomicBool = AtomicBool::new(false);
        if !FIRED.swap(true, Ordering::Relaxed) {
            log::warn!($($arg)+);
        }
    }};
}

/// The paper white of SDR, in nits. sRGB reference viewing conditions specify
/// 80 nits and [ITU-R BT.2035] specifies 100 nits, and 100 is the common
/// choice for SDR content on desktop displays.
///
/// [ITU-R BT.2035]: https://www.itu.int/rec/R-REC-BT.2035
pub const SDR_PAPER_WHITE_NITS: f32 = 100.0;

/// The paper white [ITU-R BT.2408] recommends for PQ, in nits.
///
/// [ITU-R BT.2408]: https://www.itu.int/pub/R-REP-BT.2408
pub const PQ_PAPER_WHITE_NITS: f32 = 203.0;

/// The luminance of signal `1.0` in scRGB and extended sRGB, in nits. The OS
/// maps this signal to its own SDR white, so a paper white of 80 nits means
/// "match the OS SDR white".
pub const SCRGB_REFERENCE_WHITE_NITS: f32 = 80.0;

/// The peak luminance an HDR color space gets when the app has not
/// calibrated it, in nits. Most HDR displays reach at least this.
pub const HDR_PEAK_LUMINANCE_NITS: f32 = 1000.0;

/// The highest luminance a [`DisplayTarget`] field can resolve to, in nits.
/// It is the top of the PQ curve, and no display exceeds it.
pub const MAX_LUMINANCE_NITS: f32 = 10000.0;

impl SurfaceColorSpace {
    /// Returns the paper white this color space uses when the app has not
    /// calibrated it, in nits.
    ///
    /// SDR uses [`SDR_PAPER_WHITE_NITS`]. PQ uses [`PQ_PAPER_WHITE_NITS`]. The
    /// other HDR color spaces use [`SCRGB_REFERENCE_WHITE_NITS`], so
    /// tonemapped white lands on the OS SDR white.
    pub const fn default_paper_white_nits(&self) -> f32 {
        match self {
            Self::Srgb => SDR_PAPER_WHITE_NITS,
            Self::Pq => PQ_PAPER_WHITE_NITS,
            Self::ScRgbLinear | Self::ExtendedSrgb | Self::ExtendedDisplayP3 => {
                SCRGB_REFERENCE_WHITE_NITS
            }
        }
    }

    /// Returns the peak luminance this color space uses when the app has not
    /// calibrated it, in nits. SDR peaks at paper white. HDR uses
    /// [`HDR_PEAK_LUMINANCE_NITS`].
    pub const fn default_peak_luminance_nits(&self) -> f32 {
        match self {
            Self::Srgb => SDR_PAPER_WHITE_NITS,
            _ => HDR_PEAK_LUMINANCE_NITS,
        }
    }

    /// Returns the minimum luminance this color space uses when the app has
    /// not calibrated it, in nits. It is `0.0` for every color space.
    pub const fn default_min_luminance_nits(&self) -> f32 {
        0.0
    }
}

/// A [`DisplayTarget`] after surface negotiation: the color space the output
/// uses, the luminance values the renderer encodes for, and where each
/// luminance value comes from.
///
/// [`DisplayTarget::resolve_with_display`] builds it from the request and
/// what the display reports, and [`DisplayTarget::resolve`] builds it from
/// the request alone. Each luminance is the calibrated value when the app set
/// one, else what the display reports, else the default for the color space.
/// The default is SDR sRGB at 100 nits with every field from the default.
///
/// The renderer inserts and updates this component on each
/// [`Window`](crate::Window) after the window's first update. Writing to it
/// has no effect. Its color space is the one in
/// [`WindowSurfaceColorSpaces`], so it is one frame behind the surface.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Component, Default, Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct ResolvedDisplayTarget {
    /// The color space the output uses.
    pub color_space: SurfaceColorSpace,
    /// The luminance of paper white, in nits. See
    /// [`DisplayTarget::paper_white_nits`].
    pub paper_white_nits: f32,
    /// The highest luminance the display can show, in nits. It is at least
    /// [`paper_white_nits`](Self::paper_white_nits).
    pub peak_luminance_nits: f32,
    /// The lowest luminance the display can show, in nits. It is at most
    /// [`paper_white_nits`](Self::paper_white_nits).
    pub min_luminance_nits: f32,
    /// Where each luminance field comes from.
    pub provenance: DisplayProvenance,
}

impl Default for ResolvedDisplayTarget {
    fn default() -> Self {
        Self {
            color_space: SurfaceColorSpace::Srgb,
            paper_white_nits: SDR_PAPER_WHITE_NITS,
            peak_luminance_nits: SDR_PAPER_WHITE_NITS,
            min_luminance_nits: 0.0,
            provenance: DisplayProvenance::DEFAULT,
        }
    }
}

/// Returns `Some(value)` if it is finite and passes `is_valid`, so a bad
/// value the platform reports counts as not reported.
fn reported(value: Option<f32>, is_valid: fn(f32) -> bool) -> Option<f32> {
    value.filter(|value| value.is_finite() && is_valid(*value))
}

/// Resolves one luminance field of a [`DisplayTarget`] to a value and its
/// [`FieldProvenance`].
///
/// `Some` passes through as [`FieldProvenance::User`] unless it is not
/// finite, fails `is_valid`, or is above [`MAX_LUMINANCE_NITS`]. `None` takes
/// `sensed` as [`FieldProvenance::Os`] when there is one, else `default` as
/// [`FieldProvenance::Default`]. A `Some` or a sensed value above
/// [`MAX_LUMINANCE_NITS`] is clamped to it. Each problem with a `Some` warns
/// once, naming the field.
macro_rules! resolve_luminance {
    ($field:ident, $value:expr, $sensed:expr, $default:expr, $is_valid:expr) => {{
        let default: f32 = $default;
        let sensed: Option<f32> = $sensed;
        let is_valid: fn(f32) -> bool = $is_valid;
        match $value {
            None => match sensed {
                Some(value) => (value.min(MAX_LUMINANCE_NITS), FieldProvenance::Os),
                None => (default, FieldProvenance::Default),
            },
            Some(value) if !value.is_finite() || !is_valid(value) => {
                warn_once!(
                    "DisplayTarget::{} is {value}, which is not a valid luminance. Using \
                    the default of {default} nits.",
                    stringify!($field)
                );
                (default, FieldProvenance::Default)
            }
            Some(value) if value > MAX_LUMINANCE_NITS => {
                warn_once!(
                    "DisplayTarget::{} is {value}, above the {MAX_LUMINANCE_NITS} nits a \
                    display can show. Clamping it.",
                    stringify!($field)
                );
                (MAX_LUMINANCE_NITS, FieldProvenance::User)
            }
            Some(value) => (value, FieldProvenance::User),
        }
    }};
}

impl DisplayTarget {
    /// Resolves this request for the color space the surface negotiated,
    /// with nothing reported by the display.
    ///
    /// This is [`resolve_with_display`](Self::resolve_with_display) with no
    /// [`MonitorDisplayCapability`] and no [`WindowDisplayState`]: every
    /// uncalibrated luminance takes the default of `color_space`.
    pub fn resolve(&self, color_space: SurfaceColorSpace) -> ResolvedDisplayTarget {
        self.resolve_with_display(color_space, None, None)
    }

    /// Resolves this request for the color space the surface negotiated and
    /// what the display behind it reports.
    ///
    /// A calibrated luminance (`Some`) is used as is, with
    /// [`FieldProvenance::User`], unless it is not finite or not positive (the
    /// default is used, with a warning) or above [`MAX_LUMINANCE_NITS`] (it
    /// is clamped, with a warning). A minimum luminance of `0.0` is valid.
    ///
    /// An uncalibrated luminance (`None`) takes what the display reports,
    /// with [`FieldProvenance::Os`]. A reported value above
    /// [`MAX_LUMINANCE_NITS`] is clamped to it, and a reported value that is
    /// not finite or not positive counts as not reported:
    ///
    /// - Paper white takes [`WindowDisplayState::sdr_white_nits`]. On
    ///   [`ScRgbLinear`](SurfaceColorSpace::ScRgbLinear),
    ///   [`ExtendedSrgb`](SurfaceColorSpace::ExtendedSrgb) and
    ///   [`ExtendedDisplayP3`](SurfaceColorSpace::ExtendedDisplayP3), when the
    ///   display reports [`WindowDisplayState::tone_map_headroom`] but no SDR
    ///   white, paper white is [`SCRGB_REFERENCE_WHITE_NITS`]. On those
    ///   surfaces a signal of `1.0` is the OS SDR white, and 80 nits is the
    ///   scRGB reference white, so 80 keeps SDR content at the system's SDR
    ///   white and makes the peak below consistent with the encoder's scale
    ///   of paper white over 80.
    /// - Peak luminance, on an HDR color space, takes
    ///   [`MonitorDisplayCapability::max_nits`], else the resolved paper white
    ///   times [`WindowDisplayState::tone_map_headroom`]. On SDR the display
    ///   is not consulted; see the default below.
    /// - Minimum luminance takes [`MonitorDisplayCapability::min_nits`].
    ///
    /// When the display reports nothing for a field, it takes the default of
    /// `color_space`, with [`FieldProvenance::Default`]: see
    /// [`SurfaceColorSpace::default_paper_white_nits`],
    /// [`SurfaceColorSpace::default_peak_luminance_nits`] and
    /// [`SurfaceColorSpace::default_min_luminance_nits`]. On SDR nothing is
    /// sensed for the peak: it is the resolved paper white, with
    /// [`FieldProvenance::Default`].
    ///
    /// Paper white resolves first, then the peak, which can derive from it,
    /// then the minimum. The peak luminance is raised to at least the paper
    /// white, and the minimum luminance is lowered to at most the paper white.
    /// Raising or lowering does not change the provenance.
    pub fn resolve_with_display(
        &self,
        color_space: SurfaceColorSpace,
        capability: Option<&MonitorDisplayCapability>,
        state: Option<&WindowDisplayState>,
    ) -> ResolvedDisplayTarget {
        let positive = |value: f32| value > 0.0;
        let headroom = reported(state.and_then(|state| state.tone_map_headroom), positive);
        let sdr_white_nits = reported(state.and_then(|state| state.sdr_white_nits), positive);

        let signal_one_is_sdr_white = matches!(
            color_space,
            SurfaceColorSpace::ScRgbLinear
                | SurfaceColorSpace::ExtendedSrgb
                | SurfaceColorSpace::ExtendedDisplayP3
        );
        let sensed_paper_white = sdr_white_nits.or_else(|| {
            (signal_one_is_sdr_white && headroom.is_some()).then_some(SCRGB_REFERENCE_WHITE_NITS)
        });
        let (paper_white_nits, paper_white) = resolve_luminance!(
            paper_white_nits,
            self.paper_white_nits,
            sensed_paper_white,
            color_space.default_paper_white_nits(),
            positive
        );

        let (sensed_peak, default_peak) = if color_space.is_hdr() {
            let max_nits = reported(capability.and_then(|c| c.max_nits), positive);
            // The product can overflow to infinity.
            let from_headroom = reported(headroom.map(|h| paper_white_nits * h), positive);
            (
                max_nits.or(from_headroom),
                color_space.default_peak_luminance_nits(),
            )
        } else {
            (None, paper_white_nits)
        };
        let (peak_luminance_nits, peak_luminance) = resolve_luminance!(
            peak_luminance_nits,
            self.peak_luminance_nits,
            sensed_peak,
            default_peak,
            positive
        );

        let non_negative = |value: f32| value >= 0.0;
        let (min_luminance_nits, min_luminance) = resolve_luminance!(
            min_luminance_nits,
            self.min_luminance_nits,
            reported(capability.and_then(|c| c.min_nits), non_negative),
            color_space.default_min_luminance_nits(),
            non_negative
        );

        ResolvedDisplayTarget {
            color_space,
            paper_white_nits,
            peak_luminance_nits: peak_luminance_nits.max(paper_white_nits),
            min_luminance_nits: min_luminance_nits.min(paper_white_nits),
            provenance: DisplayProvenance {
                paper_white,
                peak_luminance,
                min_luminance,
            },
        }
    }
}

/// A set of [`SurfaceColorSpace`]s, stored as a bitset.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Default, Debug, PartialEq, Hash, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct SurfaceColorSpaces(u8);

impl SurfaceColorSpaces {
    /// The empty set.
    pub const EMPTY: Self = Self(0);

    /// Every color space, in declaration order.
    const ALL: [SurfaceColorSpace; 5] = [
        SurfaceColorSpace::Srgb,
        SurfaceColorSpace::ScRgbLinear,
        SurfaceColorSpace::Pq,
        SurfaceColorSpace::ExtendedSrgb,
        SurfaceColorSpace::ExtendedDisplayP3,
    ];

    /// The bit for `color_space`. A match rather than a cast, so adding a
    /// variant cannot renumber existing bits.
    const fn bit(color_space: SurfaceColorSpace) -> u8 {
        match color_space {
            SurfaceColorSpace::Srgb => 0b00001,
            SurfaceColorSpace::ScRgbLinear => 0b00010,
            SurfaceColorSpace::Pq => 0b00100,
            SurfaceColorSpace::ExtendedSrgb => 0b01000,
            SurfaceColorSpace::ExtendedDisplayP3 => 0b10000,
        }
    }

    /// Returns this set with `color_space` added.
    pub const fn with(self, color_space: SurfaceColorSpace) -> Self {
        Self(self.0 | Self::bit(color_space))
    }

    /// Returns `true` if `color_space` is a member.
    pub const fn contains(self, color_space: SurfaceColorSpace) -> bool {
        self.0 & Self::bit(color_space) != 0
    }

    /// Iterates the members in [`SurfaceColorSpace`] declaration order.
    pub fn iter(self) -> impl Iterator<Item = SurfaceColorSpace> {
        Self::ALL
            .into_iter()
            .filter(move |&color_space| self.contains(color_space))
    }
}

impl core::fmt::Debug for SurfaceColorSpaces {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// The [`SurfaceColorSpace`] a window's surface uses, and the color spaces
/// it could use.
///
/// [`DisplayTarget`] is a request. The renderer resolves it against what the
/// surface supports and reports the result here.
///
/// The renderer inserts and updates this component. Writing to it has no
/// effect. It is one frame behind the surface and is absent until the
/// surface is configured.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Component, Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct WindowSurfaceColorSpaces {
    /// The color space the surface uses.
    pub resolved: SurfaceColorSpace,
    /// The color spaces the surface can provide.
    /// [`SurfaceColorSpace::Srgb`] is included when the surface has a format
    /// for the default (automatic) wgpu color space, which is every surface
    /// outside an OS HDR mode that lists formats only in explicit color
    /// spaces.
    pub supported: SurfaceColorSpaces,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_requests_sdr() {
        let target = DisplayTarget::default();
        assert!(!target.hdr);
        assert_eq!(target.color_space_override, None);
    }

    #[test]
    fn only_srgb_is_not_hdr() {
        assert!(!SurfaceColorSpace::Srgb.is_hdr());
        assert!(SurfaceColorSpace::ScRgbLinear.is_hdr());
        assert!(SurfaceColorSpace::Pq.is_hdr());
        assert!(SurfaceColorSpace::ExtendedSrgb.is_hdr());
        assert!(SurfaceColorSpace::ExtendedDisplayP3.is_hdr());
    }

    #[test]
    fn each_color_space_has_primaries_and_a_transfer_function() {
        let expected = [
            (
                SurfaceColorSpace::Srgb,
                RgbPrimaries::BT709,
                TransferFunction::Srgb,
            ),
            (
                SurfaceColorSpace::ScRgbLinear,
                RgbPrimaries::BT709,
                TransferFunction::Linear,
            ),
            (
                SurfaceColorSpace::Pq,
                RgbPrimaries::BT2020,
                TransferFunction::Pq,
            ),
            (
                SurfaceColorSpace::ExtendedSrgb,
                RgbPrimaries::BT709,
                TransferFunction::ExtendedSrgb,
            ),
            (
                SurfaceColorSpace::ExtendedDisplayP3,
                RgbPrimaries::DISPLAY_P3,
                TransferFunction::ExtendedSrgb,
            ),
        ];
        for (color_space, primaries, transfer_function) in expected {
            assert_eq!(color_space.primaries(), primaries);
            assert_eq!(color_space.transfer_function(), transfer_function);
        }
    }

    #[test]
    fn color_space_set_membership() {
        let set = SurfaceColorSpaces::EMPTY
            .with(SurfaceColorSpace::Srgb)
            .with(SurfaceColorSpace::Pq);
        assert!(set.contains(SurfaceColorSpace::Srgb));
        assert!(set.contains(SurfaceColorSpace::Pq));
        assert!(!set.contains(SurfaceColorSpace::ScRgbLinear));
        assert!(!set.contains(SurfaceColorSpace::ExtendedSrgb));
        assert!(!set.contains(SurfaceColorSpace::ExtendedDisplayP3));
        assert!(!SurfaceColorSpaces::EMPTY.contains(SurfaceColorSpace::Srgb));
        assert_eq!(set.with(SurfaceColorSpace::Pq), set);
    }

    #[test]
    fn color_space_set_iterates_in_declaration_order() {
        let set = SurfaceColorSpaces::EMPTY
            .with(SurfaceColorSpace::ExtendedDisplayP3)
            .with(SurfaceColorSpace::Srgb)
            .with(SurfaceColorSpace::Pq);
        assert!(set.iter().eq([
            SurfaceColorSpace::Srgb,
            SurfaceColorSpace::Pq,
            SurfaceColorSpace::ExtendedDisplayP3,
        ]));
    }

    #[test]
    fn resolve_none_fills_defaults_per_color_space() {
        let expected = [
            (SurfaceColorSpace::Srgb, 100.0, 100.0),
            (SurfaceColorSpace::ScRgbLinear, 80.0, 1000.0),
            (SurfaceColorSpace::Pq, 203.0, 1000.0),
            (SurfaceColorSpace::ExtendedSrgb, 80.0, 1000.0),
            (SurfaceColorSpace::ExtendedDisplayP3, 80.0, 1000.0),
        ];
        for (color_space, paper_white_nits, peak_luminance_nits) in expected {
            assert_eq!(
                DisplayTarget::default().resolve(color_space),
                ResolvedDisplayTarget {
                    color_space,
                    paper_white_nits,
                    peak_luminance_nits,
                    min_luminance_nits: 0.0,
                    provenance: DisplayProvenance::DEFAULT,
                },
                "{color_space:?}"
            );
        }
        assert_eq!(
            DisplayTarget::default().resolve(SurfaceColorSpace::Srgb),
            ResolvedDisplayTarget::default()
        );
    }

    #[test]
    fn resolve_some_passes_through_bit_for_bit() {
        let target = DisplayTarget {
            paper_white_nits: Some(150.25),
            peak_luminance_nits: Some(1234.5678),
            min_luminance_nits: Some(0.0051),
            ..Default::default()
        };
        let resolved = target.resolve(SurfaceColorSpace::Pq);
        assert_eq!(resolved.paper_white_nits.to_bits(), 150.25f32.to_bits());
        assert_eq!(
            resolved.peak_luminance_nits.to_bits(),
            1234.5678f32.to_bits()
        );
        assert_eq!(resolved.min_luminance_nits.to_bits(), 0.0051f32.to_bits());
        assert_eq!(resolved.provenance, DisplayProvenance::USER);

        // Zero is a valid minimum luminance.
        let zero_min = DisplayTarget {
            min_luminance_nits: Some(0.0),
            ..Default::default()
        };
        let resolved = zero_min.resolve(SurfaceColorSpace::Srgb);
        assert_eq!(resolved.min_luminance_nits, 0.0);
        assert_eq!(resolved.provenance.min_luminance, FieldProvenance::User);
    }

    #[test]
    fn resolve_degenerate_some_falls_back_to_the_default() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -1.0] {
            let target = DisplayTarget {
                paper_white_nits: Some(bad),
                peak_luminance_nits: Some(bad),
                ..Default::default()
            };
            assert_eq!(
                target.resolve(SurfaceColorSpace::Pq),
                DisplayTarget::default().resolve(SurfaceColorSpace::Pq),
                "{bad}"
            );
        }
        for bad in [f32::NAN, f32::INFINITY, -1.0] {
            let target = DisplayTarget {
                min_luminance_nits: Some(bad),
                ..Default::default()
            };
            let resolved = target.resolve(SurfaceColorSpace::Srgb);
            assert_eq!(resolved.min_luminance_nits, 0.0, "{bad}");
            assert_eq!(resolved.provenance.min_luminance, FieldProvenance::Default);
        }
    }

    #[test]
    fn resolve_clamps_to_the_maximum_luminance() {
        let target = DisplayTarget {
            paper_white_nits: Some(20000.0),
            peak_luminance_nits: Some(f32::MAX),
            min_luminance_nits: Some(10001.0),
            ..Default::default()
        };
        assert_eq!(
            target.resolve(SurfaceColorSpace::ScRgbLinear),
            ResolvedDisplayTarget {
                color_space: SurfaceColorSpace::ScRgbLinear,
                paper_white_nits: MAX_LUMINANCE_NITS,
                peak_luminance_nits: MAX_LUMINANCE_NITS,
                min_luminance_nits: MAX_LUMINANCE_NITS,
                provenance: DisplayProvenance::USER,
            }
        );
    }

    #[test]
    fn resolve_raises_peak_to_paper_white() {
        let target = DisplayTarget {
            paper_white_nits: Some(500.0),
            peak_luminance_nits: Some(200.0),
            ..Default::default()
        };
        let resolved = target.resolve(SurfaceColorSpace::Pq);
        assert_eq!(resolved.peak_luminance_nits, 500.0);
        // Raising keeps the provenance of the value.
        assert_eq!(resolved.provenance.peak_luminance, FieldProvenance::User);

        // An uncalibrated peak is raised past its default too.
        let target = DisplayTarget {
            paper_white_nits: Some(300.0),
            ..Default::default()
        };
        assert_eq!(
            target.resolve(SurfaceColorSpace::Srgb).peak_luminance_nits,
            300.0
        );

        // A calibrated peak below the default paper white is raised to it.
        let target = DisplayTarget {
            peak_luminance_nits: Some(150.0),
            ..Default::default()
        };
        assert_eq!(
            target.resolve(SurfaceColorSpace::Pq).peak_luminance_nits,
            203.0
        );
    }

    #[test]
    fn resolve_keeps_a_calibrated_peak_above_the_sdr_paper_white() {
        let target = DisplayTarget {
            peak_luminance_nits: Some(400.0),
            ..Default::default()
        };
        let resolved = target.resolve(SurfaceColorSpace::Srgb);
        assert_eq!(resolved.paper_white_nits, 100.0);
        assert_eq!(resolved.peak_luminance_nits, 400.0);
    }

    #[test]
    fn resolve_lowers_min_to_paper_white() {
        let target = DisplayTarget {
            min_luminance_nits: Some(500.0),
            ..Default::default()
        };
        assert_eq!(
            target.resolve(SurfaceColorSpace::Srgb),
            ResolvedDisplayTarget {
                color_space: SurfaceColorSpace::Srgb,
                paper_white_nits: 100.0,
                peak_luminance_nits: 100.0,
                min_luminance_nits: 100.0,
                provenance: DisplayProvenance {
                    min_luminance: FieldProvenance::User,
                    ..DisplayProvenance::DEFAULT
                },
            }
        );
    }

    fn display_state(
        tone_map_headroom: Option<f32>,
        sdr_white_nits: Option<f32>,
    ) -> WindowDisplayState {
        WindowDisplayState {
            tone_map_headroom,
            sdr_white_nits,
        }
    }

    fn monitor_capability(
        max_nits: Option<f32>,
        min_nits: Option<f32>,
    ) -> MonitorDisplayCapability {
        MonitorDisplayCapability {
            max_nits,
            min_nits,
            ..Default::default()
        }
    }

    #[test]
    fn sensed_values_fill_none_fields_only() {
        // A Windows-like report: SDR white, peak and minimum in nits.
        let capability = monitor_capability(Some(1000.0), Some(0.05));
        let state = display_state(Some(5.0), Some(200.0));

        // Every field `None`: every field from the display.
        assert_eq!(
            DisplayTarget::default().resolve_with_display(
                SurfaceColorSpace::Pq,
                Some(&capability),
                Some(&state)
            ),
            ResolvedDisplayTarget {
                color_space: SurfaceColorSpace::Pq,
                paper_white_nits: 200.0,
                peak_luminance_nits: 1000.0,
                min_luminance_nits: 0.05,
                provenance: DisplayProvenance {
                    paper_white: FieldProvenance::Os,
                    peak_luminance: FieldProvenance::Os,
                    min_luminance: FieldProvenance::Os,
                },
            }
        );

        // Every field `Some`: the display is ignored.
        let calibrated = DisplayTarget {
            paper_white_nits: Some(203.0),
            peak_luminance_nits: Some(4000.0),
            min_luminance_nits: Some(0.0),
            ..Default::default()
        };
        assert_eq!(
            calibrated.resolve_with_display(SurfaceColorSpace::Pq, Some(&capability), Some(&state)),
            ResolvedDisplayTarget {
                color_space: SurfaceColorSpace::Pq,
                paper_white_nits: 203.0,
                peak_luminance_nits: 4000.0,
                min_luminance_nits: 0.0,
                provenance: DisplayProvenance::USER,
            }
        );

        // Mixed: each field resolves on its own.
        let mixed = DisplayTarget {
            peak_luminance_nits: Some(600.0),
            ..Default::default()
        };
        let resolved =
            mixed.resolve_with_display(SurfaceColorSpace::Pq, Some(&capability), Some(&state));
        assert_eq!(resolved.paper_white_nits, 200.0);
        assert_eq!(resolved.peak_luminance_nits, 600.0);
        assert_eq!(resolved.min_luminance_nits, 0.05);
        assert_eq!(
            resolved.provenance,
            DisplayProvenance {
                paper_white: FieldProvenance::Os,
                peak_luminance: FieldProvenance::User,
                min_luminance: FieldProvenance::Os,
            }
        );
    }

    #[test]
    fn nothing_sensed_falls_back_to_the_default_per_field() {
        // The components are present but report nothing.
        let capability = MonitorDisplayCapability::default();
        let state = WindowDisplayState::default();
        assert_eq!(
            DisplayTarget::default().resolve_with_display(
                SurfaceColorSpace::Pq,
                Some(&capability),
                Some(&state)
            ),
            DisplayTarget::default().resolve(SurfaceColorSpace::Pq)
        );

        // Only a minimum is reported: the other fields take the default.
        let capability = monitor_capability(None, Some(0.01));
        let resolved = DisplayTarget::default().resolve_with_display(
            SurfaceColorSpace::Pq,
            Some(&capability),
            None,
        );
        assert_eq!(resolved.paper_white_nits, 203.0);
        assert_eq!(resolved.peak_luminance_nits, 1000.0);
        assert_eq!(resolved.min_luminance_nits, 0.01);
        assert_eq!(
            resolved.provenance,
            DisplayProvenance {
                paper_white: FieldProvenance::Default,
                peak_luminance: FieldProvenance::Default,
                min_luminance: FieldProvenance::Os,
            }
        );
    }

    #[test]
    fn a_bad_sensed_value_counts_as_not_reported() {
        let capability = monitor_capability(Some(f32::NAN), Some(-1.0));
        let state = display_state(Some(f32::INFINITY), Some(0.0));
        assert_eq!(
            DisplayTarget::default().resolve_with_display(
                SurfaceColorSpace::Pq,
                Some(&capability),
                Some(&state)
            ),
            DisplayTarget::default().resolve(SurfaceColorSpace::Pq)
        );
    }

    #[test]
    fn apple_headroom_without_sdr_white_puts_paper_white_at_the_scrgb_reference() {
        let state = display_state(Some(4.0), None);
        for color_space in [
            SurfaceColorSpace::ScRgbLinear,
            SurfaceColorSpace::ExtendedSrgb,
            SurfaceColorSpace::ExtendedDisplayP3,
        ] {
            let resolved =
                DisplayTarget::default().resolve_with_display(color_space, None, Some(&state));
            assert_eq!(
                resolved.paper_white_nits, SCRGB_REFERENCE_WHITE_NITS,
                "{color_space:?}"
            );
            assert_eq!(resolved.provenance.paper_white, FieldProvenance::Os);
            // The peak is paper white times the headroom.
            assert_eq!(resolved.peak_luminance_nits, 320.0);
            assert_eq!(resolved.provenance.peak_luminance, FieldProvenance::Os);
        }

        // PQ encodes absolute luminance, so the rule does not apply there.
        let resolved = DisplayTarget::default().resolve_with_display(
            SurfaceColorSpace::Pq,
            None,
            Some(&state),
        );
        assert_eq!(resolved.paper_white_nits, PQ_PAPER_WHITE_NITS);
        assert_eq!(resolved.provenance.paper_white, FieldProvenance::Default);
        assert_eq!(resolved.peak_luminance_nits, 203.0 * 4.0);
        assert_eq!(resolved.provenance.peak_luminance, FieldProvenance::Os);

        // Nor on SDR.
        let resolved = DisplayTarget::default().resolve_with_display(
            SurfaceColorSpace::Srgb,
            None,
            Some(&state),
        );
        assert_eq!(resolved.paper_white_nits, SDR_PAPER_WHITE_NITS);
        assert_eq!(resolved.provenance.paper_white, FieldProvenance::Default);

        // A reported SDR white wins over the rule.
        let resolved = DisplayTarget::default().resolve_with_display(
            SurfaceColorSpace::ScRgbLinear,
            None,
            Some(&display_state(Some(4.0), Some(120.0))),
        );
        assert_eq!(resolved.paper_white_nits, 120.0);
        assert_eq!(resolved.provenance.paper_white, FieldProvenance::Os);
    }

    #[test]
    fn peak_prefers_max_nits_over_headroom() {
        let capability = monitor_capability(Some(1000.0), None);
        let state = display_state(Some(5.0), Some(200.0));
        let resolved = DisplayTarget::default().resolve_with_display(
            SurfaceColorSpace::Pq,
            Some(&capability),
            Some(&state),
        );
        assert_eq!(resolved.peak_luminance_nits, 1000.0);
        assert_eq!(resolved.provenance.peak_luminance, FieldProvenance::Os);

        // Without a peak in nits, the peak is paper white times headroom.
        let resolved = DisplayTarget::default().resolve_with_display(
            SurfaceColorSpace::Pq,
            None,
            Some(&state),
        );
        assert_eq!(resolved.peak_luminance_nits, 1000.0);
        assert_eq!(resolved.provenance.peak_luminance, FieldProvenance::Os);
    }

    #[test]
    fn peak_from_headroom_uses_the_resolved_paper_white() {
        // The peak is 120 * 3, from the paper white the display reports.
        let state = display_state(Some(3.0), Some(120.0));
        let resolved = DisplayTarget::default().resolve_with_display(
            SurfaceColorSpace::ScRgbLinear,
            None,
            Some(&state),
        );
        assert_eq!(resolved.paper_white_nits, 120.0);
        assert_eq!(resolved.peak_luminance_nits, 360.0);

        // A calibrated paper white feeds the peak too: 250 * 3.
        let target = DisplayTarget {
            paper_white_nits: Some(250.0),
            ..Default::default()
        };
        let resolved =
            target.resolve_with_display(SurfaceColorSpace::ScRgbLinear, None, Some(&state));
        assert_eq!(resolved.peak_luminance_nits, 750.0);
        assert_eq!(
            resolved.provenance,
            DisplayProvenance {
                paper_white: FieldProvenance::User,
                peak_luminance: FieldProvenance::Os,
                min_luminance: FieldProvenance::Default,
            }
        );
    }

    #[test]
    fn sdr_peak_is_the_paper_white() {
        // The display reports a peak, but SDR has no headroom.
        let capability = monitor_capability(Some(1000.0), None);
        let state = display_state(Some(5.0), Some(200.0));
        let resolved = DisplayTarget::default().resolve_with_display(
            SurfaceColorSpace::Srgb,
            Some(&capability),
            Some(&state),
        );
        assert_eq!(resolved.paper_white_nits, 200.0);
        assert_eq!(resolved.peak_luminance_nits, 200.0);
        assert_eq!(resolved.provenance.paper_white, FieldProvenance::Os);
        assert_eq!(resolved.provenance.peak_luminance, FieldProvenance::Default);

        // Below the SDR default too.
        let resolved = DisplayTarget::default().resolve_with_display(
            SurfaceColorSpace::Srgb,
            None,
            Some(&display_state(None, Some(80.0))),
        );
        assert_eq!(resolved.paper_white_nits, 80.0);
        assert_eq!(resolved.peak_luminance_nits, 80.0);
    }

    #[test]
    fn a_sensed_value_is_clamped_to_the_maximum_luminance() {
        // A driver that reports a peak above the top of the PQ curve.
        let capability = monitor_capability(Some(50000.0), None);
        let resolved = DisplayTarget::default().resolve_with_display(
            SurfaceColorSpace::Pq,
            Some(&capability),
            None,
        );
        assert_eq!(resolved.peak_luminance_nits, MAX_LUMINANCE_NITS);
        assert_eq!(resolved.provenance.peak_luminance, FieldProvenance::Os);

        // Paper white times headroom above the maximum: 500 * 100.
        let resolved = DisplayTarget::default().resolve_with_display(
            SurfaceColorSpace::Pq,
            None,
            Some(&display_state(Some(100.0), Some(500.0))),
        );
        assert_eq!(resolved.paper_white_nits, 500.0);
        assert_eq!(resolved.peak_luminance_nits, MAX_LUMINANCE_NITS);
        assert_eq!(resolved.provenance.peak_luminance, FieldProvenance::Os);

        // A reported SDR white and minimum are clamped too.
        let capability = monitor_capability(None, Some(20000.0));
        let resolved = DisplayTarget::default().resolve_with_display(
            SurfaceColorSpace::Pq,
            Some(&capability),
            Some(&display_state(None, Some(20000.0))),
        );
        assert_eq!(resolved.paper_white_nits, MAX_LUMINANCE_NITS);
        assert_eq!(resolved.min_luminance_nits, MAX_LUMINANCE_NITS);
        assert_eq!(
            resolved.provenance,
            DisplayProvenance {
                paper_white: FieldProvenance::Os,
                peak_luminance: FieldProvenance::Default,
                min_luminance: FieldProvenance::Os,
            }
        );
    }

    #[test]
    fn a_sensed_peak_is_raised_to_paper_white() {
        // A display that reports a peak below its SDR white.
        let capability = monitor_capability(Some(150.0), None);
        let state = display_state(None, Some(200.0));
        let resolved = DisplayTarget::default().resolve_with_display(
            SurfaceColorSpace::Pq,
            Some(&capability),
            Some(&state),
        );
        assert_eq!(resolved.peak_luminance_nits, 200.0);
        assert_eq!(resolved.provenance.peak_luminance, FieldProvenance::Os);
    }
}
