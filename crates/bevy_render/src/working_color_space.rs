//! The project-global working color space of the renderer.
//!
//! Set the working color space on [`RenderPlugin`](crate::RenderPlugin). The
//! shader def is registered when the renderer initializes, so a later change to
//! the extracted resource does not reach the shaders.

use bevy_color::LinearRgba;
use bevy_ecs::{reflect::ReflectResource, resource::Resource};
use bevy_math::{Mat3, Vec3, Vec4};
use bevy_reflect::{prelude::ReflectDefault, Reflect};

/// The shader def registered globally on the
/// [`PipelineCache`](crate::render_resource::PipelineCache) when the
/// [`WorkingColorSpace`] is [`WorkingColorSpace::Rec2020`].
pub const WORKING_COLOR_SPACE_REC2020_SHADER_DEF: &str = "WORKING_COLOR_SPACE_REC2020";

/// The color primaries of the renderer's scene-referred working space.
///
/// All scene-referred buffers, material/light/clear colors, and lighting math
/// share one set of primaries, because shared assets and buffers make
/// per-camera working spaces impractical.
///
/// Under `Rec2020`:
///
/// * Scene-linear intermediate textures hold linear Rec.2020 values.
/// * Colors that enter the render world without shader-side texture
///   composition (light colors, ambient light, fog, clear colors) convert on
///   the CPU at their extract/prepare seams, through
///   [`linear_rgba_rec709_to_working`].
/// * Colors composed in shaders from Rec.709 factors (material, texture, and
///   vertex color; environment map and skybox samples) convert once at the end
///   of composition, under the `WORKING_COLOR_SPACE_REC2020` shader def. The
///   renderer assumes every sampled color texture is Rec.709. A texture with
///   wide primaries has no escape hatch and is over-converted. See
///   `GpuImage::source_primaries`.
/// * The Gran Turismo 7 tone mapping operator takes the working space
///   natively, so its Rec.709 to Rec.2020 input expansion is skipped. Every
///   other operator and the color-grading stack are fit to Rec.709 and get a
///   Rec.2020 to Rec.709 conversion at the tone mapping pass entry, which
///   clips colors outside the Rec.709 gamut.
///
/// `LinearRgba` and the rest of `bevy_color` stay defined as linear Rec.709.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
#[reflect(Resource, Debug, Default, Clone, PartialEq, Hash)]
pub enum WorkingColorSpace {
    /// Linear Rec.709 / sRGB primaries, D65 white point. Every working-space
    /// conversion is an identity.
    #[default]
    Rec709,
    /// Linear ITU-R BT.2020 (Rec.2020) primaries, D65 white point. Opt-in wide
    /// working space for HDR display output.
    Rec2020,
}

impl WorkingColorSpace {
    /// Returns `true` for [`WorkingColorSpace::Rec2020`].
    #[inline]
    pub const fn is_rec2020(self) -> bool {
        matches!(self, WorkingColorSpace::Rec2020)
    }
}

/// Linear Rec.709 to Rec.2020 conversion matrix, D65 white point, derived per
/// ITU-R BT.2087.
///
/// Each literal is the shortest decimal that round-trips the correctly rounded
/// `f32` of the matching f64 literal in `working_color_space.wesl` and
/// `gt7.wesl`. The Rust and WGSL constants must stay bit-identical so the CPU
/// code is an exact parity reference for the shaders
/// (`matrices_match_wgsl_f64_literals`). Equal to
/// `RgbPrimaries::BT709.matrix_to(RgbPrimaries::BT2020)` within a few ULP.
pub const REC709_TO_REC2020: Mat3 = Mat3::from_cols(
    Vec3::new(0.627_403_9, 0.069_097_29, 0.016_391_44),
    Vec3::new(0.329_283_03, 0.919_540_4, 0.088_013_306),
    Vec3::new(0.043_313_067, 0.011_362_315, 0.895_595_25),
);

/// Linear Rec.2020 to Rec.709 conversion matrix, D65 white point. Inverse of
/// [`REC709_TO_REC2020`].
///
/// See [`REC709_TO_REC2020`] for the WGSL bit-identity contract.
pub const REC2020_TO_REC709: Mat3 = Mat3::from_cols(
    Vec3::new(1.660_491, -0.124_550_48, -0.018_150_763),
    Vec3::new(-0.587_641_1, 1.132_899_9, -0.100_578_9),
    Vec3::new(-0.072_849_86, -0.008_349_422, 1.118_729_7),
);

/// Linear Rec.709 (sRGB) to Display-P3 conversion matrix, both D65.
///
/// Display-P3 uses the DCI-P3 primaries, matching
/// [`bevy_color::RgbPrimaries::DISPLAY_P3`]. Its blue primary is the same as
/// Rec.709's, so the third column's off-diagonal entries are exactly zero.
///
/// The display-encoding pass uses this to carry Rec.709 tone-map output into
/// the P3-gamut
/// [`ExtendedDisplayP3`](bevy_window::DisplayTransfer::ExtendedSrgb) signal.
/// Bit-identical to `REC709_TO_DISPLAYP3` in `working_color_space.wesl`.
pub const REC709_TO_DISPLAYP3: Mat3 = Mat3::from_cols(
    Vec3::new(0.822_461_96, 0.033_194_2, 0.017_082_632),
    Vec3::new(0.177_538_04, 0.966_805_8, 0.072_397_44),
    Vec3::new(0.0, 0.0, 0.910_519_96),
);

/// Linear Display-P3 to Rec.709 (sRGB) conversion matrix, both D65. The
/// f64-derived inverse of [`REC709_TO_DISPLAYP3`].
///
/// Decodes a Display-P3 screenshot readback back into Rec.709 display-linear.
pub const DISPLAYP3_TO_REC709: Mat3 = Mat3::from_cols(
    Vec3::new(1.224_940_2, -0.042_056_955, -0.019_637_555),
    Vec3::new(-0.224_940_18, 1.042_056_9, -0.078_636_04),
    Vec3::new(0.0, 0.0, 1.098_273_6),
);

/// Linear Rec.2020 to Display-P3 conversion matrix, both D65. Bit-identical to
/// `REC2020_TO_DISPLAYP3` in `working_color_space.wesl`.
///
/// The display-encoding pass uses this on the GT7-on-HDR path, whose tone-map
/// output is native Rec.2020. Display-P3 sits inside Rec.2020, so this
/// contracts the gamut.
pub const REC2020_TO_DISPLAYP3: Mat3 = Mat3::from_cols(
    Vec3::new(1.343_578_2, -0.065_297_455, 0.002_821_787_3),
    Vec3::new(-0.282_179_68, 1.075_787_9, -0.019_598_495),
    Vec3::new(-0.061_398_58, -0.010_490_463, 1.016_776_7),
);

/// Linear Display-P3 to Rec.2020 conversion matrix, both D65. The f64-derived
/// inverse of [`REC2020_TO_DISPLAYP3`], kept for symmetry and the
/// mutual-inverse test.
pub const DISPLAYP3_TO_REC2020: Mat3 = Mat3::from_cols(
    Vec3::new(0.753_833_06, 0.045_743_85, -0.001_210_340_3),
    Vec3::new(0.198_597_37, 0.941_777_2, 0.017_601_717),
    Vec3::new(0.047_569_595, 0.012_478_931, 0.983_608_6),
);

/// Converts a linear Rec.709 color into the given working color space.
///
/// * [`WorkingColorSpace::Rec709`]: returns `color` unchanged, bit for bit.
/// * [`WorkingColorSpace::Rec2020`]: applies [`REC709_TO_REC2020`] to the RGB
///   channels and leaves alpha alone. Nothing is clamped, so out-of-gamut
///   inputs convert like any other value.
#[inline]
pub fn linear_rgba_rec709_to_working(color: LinearRgba, working: WorkingColorSpace) -> LinearRgba {
    match working {
        WorkingColorSpace::Rec709 => color,
        WorkingColorSpace::Rec2020 => {
            let rgb = REC709_TO_REC2020 * Vec3::new(color.red, color.green, color.blue);
            LinearRgba {
                red: rgb.x,
                green: rgb.y,
                blue: rgb.z,
                alpha: color.alpha,
            }
        }
    }
}

/// [`Vec4`] variant of [`linear_rgba_rec709_to_working`]. Converts `xyz` and
/// passes `w` (alpha) through.
#[inline]
pub fn vec4_rec709_to_working(color: Vec4, working: WorkingColorSpace) -> Vec4 {
    match working {
        WorkingColorSpace::Rec709 => color,
        WorkingColorSpace::Rec2020 => (REC709_TO_REC2020 * color.truncate()).extend(color.w),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_color::RgbPrimaries;

    fn assert_mat3_rel_eq(a: Mat3, b: Mat3, max_rel: f32, context: &str) {
        let a = a.to_cols_array();
        let b = b.to_cols_array();
        for (index, (lhs, rhs)) in a.iter().zip(b.iter()).enumerate() {
            // Entries that are both near zero compare by absolute error. A
            // shared primary makes some off-diagonals exactly zero in closed
            // form, so those literals are locked to `0.0` while the runtime
            // derivation leaves about 1e-17 of noise.
            let scale = lhs.abs().max(rhs.abs());
            let rel = if scale < 1e-6 {
                0.0
            } else {
                (lhs - rhs).abs() / scale
            };
            assert!(
                rel <= max_rel,
                "{context}: entry {index} differs by relative {rel:e}: {lhs:?} ({:#010x}) vs {rhs:?} ({:#010x})",
                lhs.to_bits(),
                rhs.to_bits(),
            );
        }
    }

    /// The Rust literals must round to the same `f32` as the f64 literals in
    /// `working_color_space.wesl` and `gt7.wesl`.
    #[test]
    fn matrices_match_wgsl_f64_literals() {
        // Transcribed from the WGSL sources, one column per line.
        #[rustfmt::skip]
        let cases: [(&str, Mat3, [f64; 9]); 2] = [
            ("REC709_TO_REC2020", REC709_TO_REC2020, [
                0.627403895934699,    0.06909728935823199,   0.016391438875150228,
                0.32928303837788375,  0.919540395075459,     0.08801330787722578,
                0.043313065687417246, 0.011362315566309154,  0.895595253247624,
            ]),
            ("REC2020_TO_REC709", REC2020_TO_REC709, [
                1.6604910021084347,   -0.12455047452159052,  -0.01815076335490522,
                -0.5876411387885496,  1.1328998971259598,    -0.10057889800800739,
                -0.07284986331988484, -0.008349422604369487, 1.1187296613629125,
            ]),
        ];
        for (name, mat, wgsl) in cases {
            for (index, (value, wgsl_value)) in mat.to_cols_array().iter().zip(wgsl).enumerate() {
                assert_eq!(
                    value.to_bits(),
                    (wgsl_value as f32).to_bits(),
                    "{name} entry {index} differs: {value:?} vs {wgsl_value:?}"
                );
            }
        }
    }

    /// The literals are not bit-identical to the `bevy_color` runtime
    /// derivation: `RgbPrimaries::matrix_to` starts from the `f32` chromaticity
    /// fields, while these constants come from an f64 derivation. They must
    /// still agree to within a few ULP.
    #[test]
    fn matrices_match_bevy_color_primaries_within_tolerance() {
        assert_mat3_rel_eq(
            REC709_TO_REC2020,
            RgbPrimaries::BT709.matrix_to(RgbPrimaries::BT2020),
            1e-5,
            "REC709_TO_REC2020 vs BT709.matrix_to(BT2020)",
        );
        assert_mat3_rel_eq(
            REC2020_TO_REC709,
            RgbPrimaries::BT2020.matrix_to(RgbPrimaries::BT709),
            1e-5,
            "REC2020_TO_REC709 vs BT2020.matrix_to(BT709)",
        );
        assert_mat3_rel_eq(
            REC709_TO_DISPLAYP3,
            RgbPrimaries::BT709.matrix_to(RgbPrimaries::DISPLAY_P3),
            1e-5,
            "REC709_TO_DISPLAYP3 vs BT709.matrix_to(DISPLAY_P3)",
        );
        assert_mat3_rel_eq(
            DISPLAYP3_TO_REC709,
            RgbPrimaries::DISPLAY_P3.matrix_to(RgbPrimaries::BT709),
            1e-5,
            "DISPLAYP3_TO_REC709 vs DISPLAY_P3.matrix_to(BT709)",
        );
        assert_mat3_rel_eq(
            REC2020_TO_DISPLAYP3,
            RgbPrimaries::BT2020.matrix_to(RgbPrimaries::DISPLAY_P3),
            1e-5,
            "REC2020_TO_DISPLAYP3 vs BT2020.matrix_to(DISPLAY_P3)",
        );
        assert_mat3_rel_eq(
            DISPLAYP3_TO_REC2020,
            RgbPrimaries::DISPLAY_P3.matrix_to(RgbPrimaries::BT2020),
            1e-5,
            "DISPLAYP3_TO_REC2020 vs DISPLAY_P3.matrix_to(BT2020)",
        );
    }

    #[test]
    fn display_p3_matrices_are_gray_preserving_inverses() {
        let white = Vec3::ONE;
        for m in [
            REC709_TO_DISPLAYP3,
            DISPLAYP3_TO_REC709,
            REC2020_TO_DISPLAYP3,
            DISPLAYP3_TO_REC2020,
        ] {
            let mapped = m * white;
            assert!(
                (mapped - white).abs().max_element() < 1e-6,
                "white drifted: {mapped}"
            );
        }
        let sample = Vec3::new(0.25, 0.5, 0.75);
        for (fwd, inv) in [
            (REC709_TO_DISPLAYP3, DISPLAYP3_TO_REC709),
            (REC2020_TO_DISPLAYP3, DISPLAYP3_TO_REC2020),
        ] {
            let round_trip = inv * (fwd * sample);
            assert!(
                (round_trip - sample).abs().max_element() < 1e-5,
                "round trip drifted: {round_trip}"
            );
        }
    }

    #[test]
    fn matrices_are_gray_preserving_inverses() {
        let white = Vec3::ONE;
        let to_2020 = REC709_TO_REC2020 * white;
        let to_709 = REC2020_TO_REC709 * white;
        for v in [to_2020, to_709] {
            assert!((v - white).abs().max_element() < 1e-6, "white drifted: {v}");
        }
        let round_trip = REC2020_TO_REC709 * (REC709_TO_REC2020 * Vec3::new(0.25, 0.5, 0.75));
        assert!(
            (round_trip - Vec3::new(0.25, 0.5, 0.75))
                .abs()
                .max_element()
                < 1e-6,
            "round trip drifted: {round_trip}"
        );
    }

    #[test]
    fn rec709_is_bitwise_identity() {
        let color = LinearRgba::new(1.5, -0.25, 0.000123, 0.5);
        let converted = linear_rgba_rec709_to_working(color, WorkingColorSpace::Rec709);
        assert_eq!(color.red.to_bits(), converted.red.to_bits());
        assert_eq!(color.green.to_bits(), converted.green.to_bits());
        assert_eq!(color.blue.to_bits(), converted.blue.to_bits());
        assert_eq!(color.alpha.to_bits(), converted.alpha.to_bits());

        let v = Vec4::new(2.0, -1.0, 0.5, 0.25);
        assert_eq!(
            v.to_array().map(f32::to_bits),
            vec4_rec709_to_working(v, WorkingColorSpace::Rec709)
                .to_array()
                .map(f32::to_bits)
        );
    }

    /// Rec.709 red maps to the first column of the matrix; alpha passes
    /// through.
    #[test]
    fn rec2020_conversion_known_values() {
        let red = linear_rgba_rec709_to_working(LinearRgba::RED, WorkingColorSpace::Rec2020);
        assert_eq!(red.red.to_bits(), 0.627_403_9_f32.to_bits());
        assert_eq!(red.green.to_bits(), 0.069_097_29_f32.to_bits());
        assert_eq!(red.blue.to_bits(), 0.016_391_44_f32.to_bits());
        assert_eq!(red.alpha, 1.0);

        let v = vec4_rec709_to_working(Vec4::new(1.0, 0.0, 0.0, 0.25), WorkingColorSpace::Rec2020);
        assert_eq!(v.x.to_bits(), 0.627_403_9_f32.to_bits());
        assert_eq!(v.w, 0.25);
    }
}
