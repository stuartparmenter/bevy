use bevy_color::{Chromaticity, RgbPrimaries};
#[cfg(not(feature = "bevy_reflect"))]
use bevy_reflect::TypePath;
#[cfg(feature = "bevy_reflect")]
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use serde::{Deserialize, Serialize};

/// The color primaries that an [`Image`](crate::Image)'s RGB data is expressed in.
///
/// Metadata only: it records the gamut the pixel values were authored in. Nothing
/// converts the pixel data.
///
/// Loaders resolve the stamped value in this order:
/// 1. An explicit `source_primaries` loader setting, for example on
///    [`ImageLoaderSettings`](crate::ImageLoaderSettings).
/// 2. Color-primary metadata in the file: the KTX2 data format descriptor, a Radiance
///    HDR `PRIMARIES=` header line, or `OpenEXR` `chromaticities`.
/// 3. The [`SourceColorPrimaries::Bt709`] default, because most assets use the sRGB /
///    Rec. 709 primaries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Default, Debug, Clone, PartialEq, Hash)
)]
#[cfg_attr(not(feature = "bevy_reflect"), derive(TypePath))]
pub enum SourceColorPrimaries {
    /// The ITU-R BT.709 primaries (shared by sRGB), D65 white point.
    #[default]
    Bt709,
    /// The ITU-R BT.2020 (Rec. 2020) wide-gamut primaries, D65 white point.
    Bt2020,
    /// The Display P3 primaries (DCI-P3 primaries with a D65 white point).
    DisplayP3,
}

impl SourceColorPrimaries {
    /// The per-coordinate tolerance used by [`SourceColorPrimaries::from_chromaticities`].
    ///
    /// Files write primaries with three or four decimal places, so `2e-3` absorbs the
    /// rounding. BT.709 and Display P3 are the closest supported pair, and differ by
    /// `0.04` in the red x coordinate.
    pub const CHROMATICITY_MATCH_TOLERANCE: f32 = 2e-3;

    /// Returns the [`RgbPrimaries`] chromaticities of this primary set, for use with
    /// [`rgb_to_rgb_matrix`](bevy_color::rgb_to_rgb_matrix).
    pub const fn to_rgb_primaries(self) -> RgbPrimaries {
        match self {
            SourceColorPrimaries::Bt709 => RgbPrimaries::BT709,
            SourceColorPrimaries::Bt2020 => RgbPrimaries::BT2020,
            SourceColorPrimaries::DisplayP3 => RgbPrimaries::DISPLAY_P3,
        }
    }

    /// Matches a set of CIE 1931 xy chromaticities against the supported primary sets.
    ///
    /// Returns the matching variant when every coordinate of `red`, `green`, `blue`, and
    /// `white` is within [`CHROMATICITY_MATCH_TOLERANCE`] of one of them, `None` otherwise.
    ///
    /// [`CHROMATICITY_MATCH_TOLERANCE`]: SourceColorPrimaries::CHROMATICITY_MATCH_TOLERANCE
    pub fn from_chromaticities(
        red: Chromaticity,
        green: Chromaticity,
        blue: Chromaticity,
        white: Chromaticity,
    ) -> Option<Self> {
        let candidates = [
            SourceColorPrimaries::Bt709,
            SourceColorPrimaries::Bt2020,
            SourceColorPrimaries::DisplayP3,
        ];
        candidates.into_iter().find(|candidate| {
            let reference = candidate.to_rgb_primaries();
            [
                (red, reference.red),
                (green, reference.green),
                (blue, reference.blue),
                (white, reference.white),
            ]
            .into_iter()
            .all(|(actual, expected)| {
                (actual.x - expected.x).abs() <= Self::CHROMATICITY_MATCH_TOLERANCE
                    && (actual.y - expected.y).abs() <= Self::CHROMATICITY_MATCH_TOLERANCE
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_chromaticities_matches_within_tolerance() {
        for source in [
            SourceColorPrimaries::Bt709,
            SourceColorPrimaries::Bt2020,
            SourceColorPrimaries::DisplayP3,
        ] {
            let reference = source.to_rgb_primaries();
            assert_eq!(
                SourceColorPrimaries::from_chromaticities(
                    reference.red,
                    reference.green,
                    reference.blue,
                    reference.white,
                ),
                Some(source)
            );
            let nudge = SourceColorPrimaries::CHROMATICITY_MATCH_TOLERANCE * 0.5;
            assert_eq!(
                SourceColorPrimaries::from_chromaticities(
                    Chromaticity::new(reference.red.x + nudge, reference.red.y - nudge),
                    reference.green,
                    reference.blue,
                    reference.white,
                ),
                Some(source)
            );
        }
    }

    #[test]
    fn from_chromaticities_rejects_unknown_primaries() {
        // ACEScg (AP1) primaries: a valid file value, but not a supported variant.
        assert_eq!(
            SourceColorPrimaries::from_chromaticities(
                Chromaticity::new(0.713, 0.293),
                Chromaticity::new(0.165, 0.830),
                Chromaticity::new(0.128, 0.044),
                Chromaticity::D60,
            ),
            None
        );
    }
}
