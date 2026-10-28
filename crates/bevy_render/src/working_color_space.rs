use bevy_color::{Alpha, ColorToComponents, LinearRgba};
use bevy_ecs::resource::Resource;
use bevy_math::{Mat3, Vec3};

/// The primaries of the linear color space that lighting and blending happen in, before
/// tonemapping.
///
/// Use [`WorkingColorSpace::Rec2020`] to keep lit colors outside the Rec.709 gamut, which a
/// wide-gamut HDR display can show. Colors you author, such as [`LinearRgba`] values and
/// textures, stay Rec.709, and the renderer converts them. Custom shaders that write color
/// before tonemapping must convert it themselves.
///
/// Set it on [`RenderPlugin`](crate::RenderPlugin). It can't change after startup.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkingColorSpace {
    /// Linear Rec.709 primaries, the same as sRGB.
    #[default]
    Rec709,
    /// Linear Rec.2020 primaries.
    Rec2020,
}

const REC709_TO_REC2020: Mat3 = Mat3::from_cols(
    Vec3::new(0.627_403_9, 0.069_097_29, 0.016_391_44),
    Vec3::new(0.329_283_03, 0.919_540_4, 0.088_013_306),
    Vec3::new(0.043_313_067, 0.011_362_315, 0.895_595_25),
);

impl WorkingColorSpace {
    /// Converts a linear Rec.709 color to this color space. Alpha is unchanged.
    pub fn convert_rec709(self, color: LinearRgba) -> LinearRgba {
        match self {
            Self::Rec709 => color,
            Self::Rec2020 => {
                LinearRgba::from_vec3(REC709_TO_REC2020 * color.to_vec3()).with_alpha(color.alpha)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::REC709_TO_REC2020;
    use bevy_color::RgbPrimaries;

    /// The column-major entries of the `mat3x3` constant `name` in `working_color_space.wesl`.
    fn wgsl_matrix(name: &str) -> [f32; 9] {
        let source = include_str!("working_color_space.wesl");
        let start = source
            .find(&format!("const {name} = mat3x3<f32>("))
            .unwrap();
        let body = &source[start..];
        let body = &body[body.find('(').unwrap() + 1..body.find(')').unwrap()];
        let values: Vec<f32> = body
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| value.parse().unwrap())
            .collect();
        values.try_into().unwrap()
    }

    #[test]
    fn matrices_match_primaries() {
        let (bt709, bt2020) = (RgbPrimaries::BT709, RgbPrimaries::BT2020);
        for (name, matrix, from, to) in [
            (
                "WGSL REC709_TO_REC2020",
                wgsl_matrix("REC709_TO_REC2020"),
                bt709,
                bt2020,
            ),
            (
                "WGSL REC2020_TO_REC709",
                wgsl_matrix("REC2020_TO_REC709"),
                bt2020,
                bt709,
            ),
            (
                "REC709_TO_REC2020",
                REC709_TO_REC2020.to_cols_array(),
                bt709,
                bt2020,
            ),
        ] {
            for (value, expected) in matrix.into_iter().zip(from.matrix_to(to).to_cols_array()) {
                assert!(
                    (value - expected).abs() < 1e-5,
                    "{name}: {value} vs {expected}"
                );
            }
        }
    }
}
