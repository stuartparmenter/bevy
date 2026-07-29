//! CIE 1931 chromaticity coordinates, RGB primary sets, and runtime derivation of
//! RGB ↔ RGB conversion matrices.
//!
//! These are the engine-wide primitives for wide-gamut color support: asset loaders,
//! display encoders, and the configurable working color space all describe their
//! primaries with [`RgbPrimaries`] and derive conversion matrices with
//! [`rgb_to_rgb_matrix`].
//!
//! Matrices are derived from chromaticity coordinates using the standard
//! [Lindbloom method](http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html):
//! each primary's chromaticity is lifted to XYZ, and the columns are scaled so the
//! white chromaticity maps to luminance 1.0. No chromatic adaptation transform (CAT)
//! is applied; converting between primary sets with different white points (such as
//! [`RgbPrimaries::ACES_CG`], which is approximately D60, and a D65 set) will carry
//! a small white-point shift.

use bevy_math::{DMat3, DVec3, Mat3, Vec3};
#[cfg(feature = "bevy_reflect")]
use bevy_reflect::prelude::*;

/// A position in the [CIE 1931 xy chromaticity diagram](https://en.wikipedia.org/wiki/CIE_1931_color_space),
/// describing a color's hue and saturation independently of its luminance.
///
/// Chromaticity coordinates are how color standards (ITU-R BT.709, BT.2020, SMPTE
/// Display P3, ACES) define their primaries and white points. Together with a
/// luminance value `Y`, a chromaticity fully determines a CIE XYZ color (the *xyY*
/// representation); see [`Chromaticity::to_xyz`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Clone, PartialEq, Default)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct Chromaticity {
    /// The x chromaticity coordinate. Typically in `[0.0, 0.8]` for physical colors.
    pub x: f32,
    /// The y chromaticity coordinate. Typically in `(0.0, 0.9]` for physical colors.
    pub y: f32,
}

impl Chromaticity {
    /// Construct a new [`Chromaticity`] from CIE 1931 xy coordinates.
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// The [CIE Standard Illuminant D65](https://en.wikipedia.org/wiki/Illuminant_D65)
    /// white point, as specified by ITU-R BT.709 / BT.2020 and the sRGB standard.
    pub const D65: Self = Self::new(0.3127, 0.3290);

    /// The ACES white point (approximately CIE Standard Illuminant D60), used by
    /// [`RgbPrimaries::ACES_CG`].
    pub const D60: Self = Self::new(0.32168, 0.33767);

    /// Convert this chromaticity and a luminance `Y` (the CIE *xyY* representation)
    /// to a CIE 1931 XYZ tristimulus value.
    ///
    /// The returned vector is `(X, Y, Z)`. A luminance of `1.0` corresponds to the
    /// reference white luminance, matching the conventions of
    /// [`Xyza`](crate::Xyza).
    ///
    /// The conversion divides by `y`; a chromaticity with `y == 0.0` (which does not
    /// describe a physical color) produces non-finite components.
    pub const fn to_xyz(self, luminance: f32) -> Vec3 {
        Vec3::new(
            self.x / self.y * luminance,
            luminance,
            (1.0 - self.x - self.y) / self.y * luminance,
        )
    }
}

impl Default for Chromaticity {
    /// Defaults to the [D65](Chromaticity::D65) white point.
    fn default() -> Self {
        Self::D65
    }
}

/// A set of RGB primaries and a white point, defined by their
/// [`Chromaticity`] coordinates.
///
/// This fully determines the meaning of an RGB triple (up to a transfer function and
/// a luminance scale): [`rgb_to_rgb_matrix`] derives the matrix converting linear
/// RGB values between two primary sets.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Clone, PartialEq, Default)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct RgbPrimaries {
    /// The chromaticity of the red primary.
    pub red: Chromaticity,
    /// The chromaticity of the green primary.
    pub green: Chromaticity,
    /// The chromaticity of the blue primary.
    pub blue: Chromaticity,
    /// The chromaticity of the white point (the color produced by `(1, 1, 1)`).
    pub white: Chromaticity,
}

impl RgbPrimaries {
    /// The [ITU-R BT.709](https://www.itu.int/rec/R-REC-BT.709) primaries with a D65
    /// white point.
    ///
    /// These are the primaries of sRGB and of Bevy's default (SDR) working color
    /// space; [`LinearRgba`](crate::LinearRgba) values are defined relative to them.
    pub const BT709: Self = Self {
        red: Chromaticity::new(0.640, 0.330),
        green: Chromaticity::new(0.300, 0.600),
        blue: Chromaticity::new(0.150, 0.060),
        white: Chromaticity::D65,
    };

    /// The [ITU-R BT.2020](https://www.itu.int/rec/R-REC-BT.2020) (Rec. 2020)
    /// wide-gamut primaries with a D65 white point.
    ///
    /// These are the primaries of [`LinearRec2020`](crate::LinearRec2020) and the
    /// standard container gamut for HDR video (HDR10, Dolby Vision).
    pub const BT2020: Self = Self {
        red: Chromaticity::new(0.708, 0.292),
        green: Chromaticity::new(0.170, 0.797),
        blue: Chromaticity::new(0.131, 0.046),
        white: Chromaticity::D65,
    };

    /// The [Display P3](https://en.wikipedia.org/wiki/DCI-P3#Display_P3) primaries
    /// (DCI-P3 primaries with a D65 white point), used by most wide-gamut consumer
    /// displays.
    pub const DISPLAY_P3: Self = Self {
        red: Chromaticity::new(0.680, 0.320),
        green: Chromaticity::new(0.265, 0.690),
        blue: Chromaticity::new(0.150, 0.060),
        white: Chromaticity::D65,
    };

    /// The [ACEScg](https://docs.acescentral.com/specifications/acescg/) (AP1)
    /// primaries with the ACES (approximately D60) white point, used for film-grade
    /// scene-linear rendering.
    ///
    /// Note that the ACES white point differs from D65: converting between
    /// `ACES_CG` and a D65 primary set with [`rgb_to_rgb_matrix`] does not apply
    /// chromatic adaptation and will carry a small white-point shift.
    pub const ACES_CG: Self = Self {
        red: Chromaticity::new(0.713, 0.293),
        green: Chromaticity::new(0.165, 0.830),
        blue: Chromaticity::new(0.128, 0.044),
        white: Chromaticity::D60,
    };

    /// Derive the RGB → XYZ matrix in `f64` precision. The matrix maps `(1, 1, 1)`
    /// to the XYZ value of [`Self::white`] at luminance `1.0`.
    fn rgb_to_xyz_dmat3(&self) -> DMat3 {
        fn xyz_of(c: Chromaticity) -> DVec3 {
            let (x, y) = (c.x as f64, c.y as f64);
            DVec3::new(x / y, 1.0, (1.0 - x - y) / y)
        }

        // Columns are the (unscaled) XYZ coordinates of the primaries.
        let primaries = DMat3::from_cols(xyz_of(self.red), xyz_of(self.green), xyz_of(self.blue));
        // Scale each column so that (1, 1, 1) maps to the white point at Y = 1.
        let scale = primaries.inverse() * xyz_of(self.white);
        primaries * DMat3::from_diagonal(scale)
    }
}

impl Default for RgbPrimaries {
    /// Defaults to [`RgbPrimaries::BT709`], the primaries of Bevy's default (SDR)
    /// working color space.
    fn default() -> Self {
        Self::BT709
    }
}

/// Returns the 3×3 matrix converting linear RGB values in the `src` primary set to
/// linear RGB values in the `dst` primary set, by composing
/// `src` RGB → XYZ → `dst` RGB.
///
/// No chromatic adaptation transform is applied (a "CAT-free" derivation): when both
/// primary sets share the same white point (as all the D65 sets here do), none is
/// needed, and the white point maps exactly to white. For primary sets with
/// *different* white points (e.g. [`RgbPrimaries::ACES_CG`] vs. a D65 set), the
/// result carries a small white-point shift instead of adapting it.
///
/// The derivation is performed in `f64` and rounded to `f32` at the end. Use with
/// [`Mat3::mul_vec3`] or the `*` operator: `m * rgb`.
///
/// # Example
///
/// ```
/// use bevy_color::{rgb_to_rgb_matrix, RgbPrimaries};
/// use bevy_math::Vec3;
///
/// // The ITU-R BT.2087 Rec.709 → Rec.2020 conversion matrix.
/// let m = rgb_to_rgb_matrix(RgbPrimaries::BT709, RgbPrimaries::BT2020);
/// // Pure Rec.709 red expressed in Rec.2020 primaries:
/// let red_2020 = m * Vec3::X;
/// assert!((red_2020.x - 0.6274).abs() < 1e-3);
/// ```
pub fn rgb_to_rgb_matrix(src: RgbPrimaries, dst: RgbPrimaries) -> Mat3 {
    (dst.rgb_to_xyz_dmat3().inverse() * src.rgb_to_xyz_dmat3()).as_mat3()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::assert_approx_eq;
    use alloc::format;

    fn assert_mat3_approx_eq(a: Mat3, b: Mat3, tolerance: f32) {
        for col in 0..3 {
            for row in 0..3 {
                assert_approx_eq!(
                    a.col(col)[row],
                    b.col(col)[row],
                    tolerance,
                    format!("matrices differ at column {col}, row {row}: {a} != {b}")
                );
            }
        }
    }

    #[test]
    fn bt709_to_bt2020_matches_bt2087() {
        // Published ITU-R BT.2087-0 (table in section 2.2.1) Rec.709 → Rec.2020 matrix.
        let expected = Mat3::from_cols_array_2d(&[
            // columns (glam is column-major)
            [0.6274, 0.0691, 0.0164],
            [0.3293, 0.9195, 0.0880],
            [0.0433, 0.0114, 0.8956],
        ]);
        let m = rgb_to_rgb_matrix(RgbPrimaries::BT709, RgbPrimaries::BT2020);
        assert_mat3_approx_eq(m, expected, 1e-4);
    }

    #[test]
    fn rgb_to_rgb_round_trip_is_identity() {
        let sets = [
            RgbPrimaries::BT709,
            RgbPrimaries::BT2020,
            RgbPrimaries::DISPLAY_P3,
            RgbPrimaries::ACES_CG,
        ];
        for src in sets {
            for dst in sets {
                let round_trip = rgb_to_rgb_matrix(dst, src) * rgb_to_rgb_matrix(src, dst);
                assert_mat3_approx_eq(round_trip, Mat3::IDENTITY, 1e-6);
            }
        }
    }

    #[test]
    fn same_primaries_is_identity() {
        let m = rgb_to_rgb_matrix(RgbPrimaries::BT709, RgbPrimaries::BT709);
        assert_mat3_approx_eq(m, Mat3::IDENTITY, 1e-7);
    }

    #[test]
    fn white_maps_to_white_point() {
        for primaries in [
            RgbPrimaries::BT709,
            RgbPrimaries::BT2020,
            RgbPrimaries::DISPLAY_P3,
            RgbPrimaries::ACES_CG,
        ] {
            let white_xyz = primaries.rgb_to_xyz_dmat3().as_mat3() * Vec3::ONE;
            let expected = primaries.white.to_xyz(1.0);
            assert_approx_eq!(white_xyz.x, expected.x, 1e-6);
            assert_approx_eq!(white_xyz.y, expected.y, 1e-6);
            assert_approx_eq!(white_xyz.z, expected.z, 1e-6);
        }
    }

    #[test]
    fn bt709_matrix_matches_crate_srgb_matrix() {
        // The crate's `Xyza` ↔ `LinearRgba` constants are the Lindbloom sRGB matrices,
        // which use the tabulated ASTM E308 D65 white (0.95047, 1.0, 1.08883) rather
        // than the white derived from the (0.3127, 0.3290) chromaticity. The
        // difference is ~2e-4 in the worst coefficient.
        let expected = Mat3::from_cols_array_2d(&[
            [0.4124564, 0.2126729, 0.0193339],
            [0.3575761, 0.7151522, 0.119192],
            [0.1804375, 0.072175, 0.9503041],
        ]);
        let m = RgbPrimaries::BT709.rgb_to_xyz_dmat3().as_mat3();
        assert_mat3_approx_eq(m, expected, 5e-4);
    }

    #[test]
    fn xyz_round_trip() {
        let m = RgbPrimaries::BT2020.rgb_to_xyz_dmat3();
        assert_mat3_approx_eq(m.inverse().as_mat3() * m.as_mat3(), Mat3::IDENTITY, 1e-6);
    }

    #[test]
    fn chromaticity_to_xyz() {
        // D65 at unit luminance.
        let xyz = Chromaticity::D65.to_xyz(1.0);
        assert_approx_eq!(xyz.x, 0.9504559, 1e-6);
        assert_approx_eq!(xyz.y, 1.0, 1e-6);
        assert_approx_eq!(xyz.z, 1.0890578, 1e-6);

        // Luminance scales linearly.
        let xyz = Chromaticity::D65.to_xyz(2.0);
        assert_approx_eq!(xyz.y, 2.0, 1e-6);
        assert_approx_eq!(xyz.x, 2.0 * 0.9504559, 1e-5);
    }
}
