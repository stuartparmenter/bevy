//! The project-global working color space of the renderer.
//!
//! Bevy's scene-referred rendering uses linear Rec.709 (the sRGB primaries)
//! as its default working color space. [`WorkingColorSpace`] makes that axis
//! explicit and configurable: the default, [`WorkingColorSpace::Rec709`], is
//! a pass-through that leaves scene-referred buffers and lighting math in
//! linear Rec.709, while the opt-in [`WorkingColorSpace::Rec2020`] switches
//! them to the wide-gamut ITU-R BT.2020 primaries (D65 white point
//! throughout).
//!
//! The working color space is configured on
//! [`RenderPlugin`](crate::RenderPlugin) and is **immutable after the app is
//! built**: render pipelines are specialized against it exactly once, so
//! mutating the extracted resource at runtime has no effect.
//!
//! See the `WORKING_COLOR_SPACE_REC2020` shader def
//! ([`WORKING_COLOR_SPACE_REC2020_SHADER_DEF`]) and the WGSL helper library
//! importable as `bevy_render::working_color_space` for the shader-side
//! counterparts.

use bevy_color::LinearRgba;
use bevy_ecs::{reflect::ReflectResource, resource::Resource};
use bevy_math::{Mat3, Vec3, Vec4};
use bevy_reflect::{prelude::ReflectDefault, Reflect};

/// The name of the shader def pushed into every working-space-aware render
/// pipeline when the [`WorkingColorSpace`] is [`WorkingColorSpace::Rec2020`].
///
/// When the working color space is [`WorkingColorSpace::Rec709`] (the
/// default), the def is *not* pushed and every shader composes with no
/// working-space def.
pub const WORKING_COLOR_SPACE_REC2020_SHADER_DEF: &str = "WORKING_COLOR_SPACE_REC2020";

/// The color primaries of the renderer's scene-referred working space.
///
/// This is a project-global, immutable axis configured on
/// [`RenderPlugin`](crate::RenderPlugin) (Unreal-style project setting): all
/// scene-referred buffers, material/light/clear colors, and lighting math
/// share one set of primaries, because shared assets and buffers make
/// per-camera working spaces impractical.
///
/// # Behavior under `Rec2020`
///
/// * Scene-linear intermediate textures hold linear Rec.2020 values.
/// * Colors entering the render world without shader-side texture composition
///   (light colors, ambient light, fog, clear colors) are converted
///   Rec.709 → Rec.2020 on the CPU at their extract/prepare seams via
///   [`linear_rgba_rec709_to_working`].
/// * Color quantities composed in shaders from Rec.709 factors (material
///   color × texture × vertex color, environment map and skybox samples) are
///   converted once in the shader, at the end of composition, under the
///   `WORKING_COLOR_SPACE_REC2020` shader def. All sampled color textures are
///   assumed to be authored against Rec.709 primaries (the overwhelmingly
///   common case); textures stamped with wide primaries
///   (`Image::source_primaries`) currently have no per-texture escape hatch
///   and will be over-converted (see `GpuImage::source_primaries`).
/// * The Gran Turismo 7 tone mapping operator consumes the working space
///   natively (its Rec.709 → Rec.2020 input expansion is skipped); all other
///   operators and the color-grading stack are Rec.709-fit and receive a
///   Rec.2020 → Rec.709 conversion at the tone mapping pass entry, which
///   clips colors outside the Rec.709 gamut.
///
/// `LinearRgba` (and the rest of `bevy_color`) remains *defined* as linear
/// Rec.709: the conversion to the working space happens exactly once, at the
/// seams above — user-facing color APIs do not reinterpret.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
#[reflect(Resource, Debug, Default, Clone, PartialEq, Hash)]
pub enum WorkingColorSpace {
    /// Linear Rec.709 / sRGB primaries, D65 white point (the default).
    ///
    /// The working-space conversions are identities, so rendering matches a
    /// build with no working-space support compiled in.
    #[default]
    Rec709,
    /// Linear ITU-R BT.2020 (Rec.2020) primaries, D65 white point.
    ///
    /// Opt-in wide working space. Recommended for HDR display output
    /// pipelines; see the type-level docs for the semantics.
    Rec2020,
}

impl WorkingColorSpace {
    /// Returns `true` if this is the wide [`WorkingColorSpace::Rec2020`]
    /// working space.
    #[inline]
    pub const fn is_rec2020(self) -> bool {
        matches!(self, WorkingColorSpace::Rec2020)
    }
}

/// Full-precision (f64-derived) linear Rec.709 → Rec.2020 conversion matrix
/// (D65 white point, derived per ITU-R BT.2087).
///
/// Each literal is the shortest round-trip representation of the correctly
/// rounded `f32` of the corresponding f64 literal in
/// `working_color_space.wgsl` / `gt7.wgsl`; the Rust and WGSL constants must
/// stay bit-identical so CPU code remains an exact parity reference for the
/// shaders. Bit-identical to `REC_709_TO_REC_2020` in
/// `bevy_core_pipeline::tonemapping::gt7` (verified by a test there) and
/// equal to `bevy_color::rgb_to_rgb_matrix(RgbPrimaries::BT709,
/// RgbPrimaries::BT2020)` within a few ULP (the runtime derivation uses the
/// chromaticity-derived D65 white, while these constants use the BT.2087
/// convention; verified by test).
pub const REC709_TO_REC2020: Mat3 = Mat3::from_cols(
    Vec3::new(0.627_403_9, 0.069_097_29, 0.016_391_44),
    Vec3::new(0.329_283_03, 0.919_540_4, 0.088_013_306),
    Vec3::new(0.043_313_067, 0.011_362_315, 0.895_595_25),
);

/// Full-precision (f64-derived) linear Rec.2020 → Rec.709 conversion matrix
/// (D65 white point). Inverse of [`REC709_TO_REC2020`].
///
/// See [`REC709_TO_REC2020`] for the bit-identity contract with the WGSL and
/// `bevy_color` counterparts.
pub const REC2020_TO_REC709: Mat3 = Mat3::from_cols(
    Vec3::new(1.660_491, -0.124_550_48, -0.018_150_763),
    Vec3::new(-0.587_641_1, 1.132_899_9, -0.100_578_9),
    Vec3::new(-0.072_849_86, -0.008_349_422, 1.118_729_7),
);

/// Full-precision (f64-derived) linear Rec.709 (sRGB) → Display-P3 conversion
/// matrix (both D65). Display-P3 uses the DCI-P3 primaries with a D65 white,
/// matching [`bevy_color::RgbPrimaries::DISPLAY_P3`]; its blue primary
/// coincides with Rec.709's, so the third column's off-diagonal entries are
/// exactly zero.
///
/// Used by the display-encoding pass to carry Rec.709 tone-map output into the
/// P3-gamut [`ExtendedDisplayP3`](bevy_window::DisplayTransfer::ExtendedSrgb)
/// signal. The Rust literals are bit-identical to `REC709_TO_DISPLAYP3` in
/// `working_color_space.wgsl`; both round to the same `f32` as the f64
/// derivation and agree with `bevy_color::rgb_to_rgb_matrix(BT709,
/// DISPLAY_P3)` within a few ULP (verified by test).
pub const REC709_TO_DISPLAYP3: Mat3 = Mat3::from_cols(
    Vec3::new(0.822_461_96, 0.033_194_2, 0.017_082_632),
    Vec3::new(0.177_538_04, 0.966_805_8, 0.072_397_44),
    Vec3::new(0.0, 0.0, 0.910_519_96),
);

/// Full-precision (f64-derived) linear Display-P3 → Rec.709 (sRGB) conversion
/// matrix (both D65). Inverse of [`REC709_TO_DISPLAYP3`]; used to decode a
/// Display-P3 screenshot readback back into Rec.709 display-linear.
pub const DISPLAYP3_TO_REC709: Mat3 = Mat3::from_cols(
    Vec3::new(1.224_940_2, -0.042_056_955, -0.019_637_555),
    Vec3::new(-0.224_940_18, 1.042_056_9, -0.078_636_04),
    Vec3::new(0.0, 0.0, 1.098_273_6),
);

/// Full-precision (f64-derived) linear Rec.2020 → Display-P3 conversion matrix
/// (both D65). Used by the display-encoding pass for the GT7-on-HDR path,
/// whose tone-map output is native Rec.2020, onto a P3-gamut signal (a gamut
/// *contraction*: Display-P3 ⊂ Rec.2020). Bit-identical to
/// `REC2020_TO_DISPLAYP3` in `working_color_space.wgsl`.
pub const REC2020_TO_DISPLAYP3: Mat3 = Mat3::from_cols(
    Vec3::new(1.343_578_2, -0.065_297_455, 0.002_821_787_3),
    Vec3::new(-0.282_179_68, 1.075_787_9, -0.019_598_495),
    Vec3::new(-0.061_398_58, -0.010_490_463, 1.016_776_7),
);

/// Full-precision (f64-derived) linear Display-P3 → Rec.2020 conversion matrix
/// (both D65). Inverse of [`REC2020_TO_DISPLAYP3`], kept for symmetry and the
/// mutual-inverse test.
pub const DISPLAYP3_TO_REC2020: Mat3 = Mat3::from_cols(
    Vec3::new(0.753_833_06, 0.045_743_85, -0.001_210_340_3),
    Vec3::new(0.198_597_37, 0.941_777_2, 0.017_601_717),
    Vec3::new(0.047_569_595, 0.012_478_931, 0.983_608_6),
);

/// Converts a linear Rec.709 color into the given working color space.
///
/// This is THE shared CPU seam helper for the working-space axis: every
/// extract/prepare-time color conversion routes through it so the matrix and
/// the identity guarantee live in exactly one place.
///
/// * [`WorkingColorSpace::Rec709`]: returns `color` **unchanged, bit-for-bit**
///   (an exact identity for the SDR default).
/// * [`WorkingColorSpace::Rec2020`]: applies [`REC709_TO_REC2020`] to the RGB
///   channels (alpha is untouched). Out-of-gamut inputs (negative or > 1
///   components) convert linearly like any other value.
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

/// [`Vec4`] variant of [`linear_rgba_rec709_to_working`]: converts `rgb`
/// (`xyz`) and passes `w` (alpha) through. Identity (bit-for-bit) for
/// [`WorkingColorSpace::Rec709`].
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
    use bevy_color::{rgb_to_rgb_matrix, RgbPrimaries};

    fn assert_mat3_rel_eq(a: Mat3, b: Mat3, max_rel: f32, context: &str) {
        let a = a.to_cols_array();
        let b = b.to_cols_array();
        for (index, (lhs, rhs)) in a.iter().zip(b.iter()).enumerate() {
            // Entries that are both effectively zero (a shared primary makes
            // some off-diagonals exact zero in closed form — we lock those
            // literals to `0.0`, while the runtime derivation leaves ~1e-17
            // floating-point noise) compare by absolute, not relative, error:
            // 0 vs 1e-17 is a 100% *relative* difference but a 0% real one.
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

    /// The hardcoded BT.2087-derived shortest-f32 literals must agree with
    /// the `bevy_color` runtime derivation to a tight relative tolerance.
    /// They are NOT bit-identical: `rgb_to_rgb_matrix` derives the D65 white
    /// from the (0.3127, 0.3290) chromaticity, while the BT.2087 constants
    /// (shared bit-for-bit with `gt7.rs`/`gt7.wgsl`/`working_color_space.wgsl`)
    /// follow the tabulated-white convention; observed disagreement is a few
    /// ULP (relative ~1e-6). The bitwise gt7-parity test lives in
    /// `bevy_core_pipeline::tonemapping::gt7`.
    #[test]
    fn matrices_match_bevy_color_primaries_within_tolerance() {
        assert_mat3_rel_eq(
            REC709_TO_REC2020,
            rgb_to_rgb_matrix(RgbPrimaries::BT709, RgbPrimaries::BT2020),
            1e-5,
            "REC709_TO_REC2020 vs rgb_to_rgb_matrix(BT709, BT2020)",
        );
        assert_mat3_rel_eq(
            REC2020_TO_REC709,
            rgb_to_rgb_matrix(RgbPrimaries::BT2020, RgbPrimaries::BT709),
            1e-5,
            "REC2020_TO_REC709 vs rgb_to_rgb_matrix(BT2020, BT709)",
        );
        // Display-P3 shares the D65 white point with both BT.709 and BT.2020,
        // so (unlike the BT.2087 Rec.2020 constants) there is no
        // tabulated-white divergence — the literals agree with the runtime
        // derivation to the same tight tolerance.
        assert_mat3_rel_eq(
            REC709_TO_DISPLAYP3,
            rgb_to_rgb_matrix(RgbPrimaries::BT709, RgbPrimaries::DISPLAY_P3),
            1e-5,
            "REC709_TO_DISPLAYP3 vs rgb_to_rgb_matrix(BT709, DISPLAY_P3)",
        );
        assert_mat3_rel_eq(
            DISPLAYP3_TO_REC709,
            rgb_to_rgb_matrix(RgbPrimaries::DISPLAY_P3, RgbPrimaries::BT709),
            1e-5,
            "DISPLAYP3_TO_REC709 vs rgb_to_rgb_matrix(DISPLAY_P3, BT709)",
        );
        assert_mat3_rel_eq(
            REC2020_TO_DISPLAYP3,
            rgb_to_rgb_matrix(RgbPrimaries::BT2020, RgbPrimaries::DISPLAY_P3),
            1e-5,
            "REC2020_TO_DISPLAYP3 vs rgb_to_rgb_matrix(BT2020, DISPLAY_P3)",
        );
        assert_mat3_rel_eq(
            DISPLAYP3_TO_REC2020,
            rgb_to_rgb_matrix(RgbPrimaries::DISPLAY_P3, RgbPrimaries::BT2020),
            1e-5,
            "DISPLAYP3_TO_REC2020 vs rgb_to_rgb_matrix(DISPLAY_P3, BT2020)",
        );
    }

    /// The Display-P3 matrices are gray-preserving (rows sum to 1, shared D65
    /// white) and mutual inverses with their Rec.709 / Rec.2020 counterparts.
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

    /// Both matrices are gray-preserving (rows sum to 1) and mutual inverses.
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

    /// `Rec709` must be a bit-for-bit identity through the shared helper
    /// (the exact-identity guarantee for the SDR default).
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

    /// Known-value conversion: Rec.709 red maps to the first column of the
    /// matrix; alpha passes through.
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
