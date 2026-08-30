//! CPU conversion of an equirectangular (lat-long) panorama into a cubemap [`Image`].

use crate::{Image, TextureFormatPixelInfo};
use alloc::borrow::Cow;
use bevy_color::Srgba;
use bevy_math::{ops, Vec3, Vec4};
use core::f32::consts::{PI, TAU};
use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
use thiserror::Error;
use wgpu_types::{
    Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
};

/// Largest finite `f16` value.
const F16_MAX: f32 = half::f16::MAX.to_f32_const();

/// An error from [`Image::equirectangular_to_cubemap`].
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum EquirectangularToCubemapError {
    /// The source is not a single-layer 2D image.
    #[error("the equirectangular source must be a single-layer 2D image")]
    WrongDimension,
    /// The source has no CPU-side pixel data.
    #[error("the equirectangular source has no CPU-side pixel data")]
    Uninitialized,
    /// The source format cannot be decoded by the converter.
    #[error("unsupported source texture format {0:?}; use Rgba32Float, Rgba16Float, Rgba8Unorm or Rgba8UnormSrgb")]
    UnsupportedSourceFormat(TextureFormat),
    /// `face_size` was zero.
    #[error("cubemap face size must be at least 1")]
    InvalidFaceSize,
}

impl Image {
    /// Converts this equirectangular (lat-long) panorama into a six-face cubemap with
    /// `face_size` x `face_size` faces of [`TextureFormat::Rgba16Float`].
    ///
    /// The result is a 2D array image with six layers whose
    /// [`texture_view_descriptor`](Image::texture_view_descriptor) is already set to
    /// [`TextureViewDimension::Cube`], so it can be assigned directly to a `Skybox`,
    /// `EnvironmentMapLight` or `GeneratedEnvironmentMapLight`. The source's
    /// [`sampler`](Image::sampler) and [`asset_usage`](Image::asset_usage) carry over.
    ///
    /// # Source layout
    ///
    /// The source is a single-layer 2D image in the standard lat-long parameterisation used
    /// by Poly Haven, RTXMG and most `.hdr` panoramas: for a world direction `dir`,
    ///
    /// ```text
    /// u = 0.5 + atan2(dir.z, dir.x) / (2 pi)     // u = 0.5 at +X, 0.75 at +Z, 0 (and 1) at -X, 0.25 at -Z
    /// v = acos(dir.y) / pi                        // v = 0 at +Y (top row), 1 at -Y (bottom row)
    /// ```
    ///
    /// So the centre column of the panorama faces world `+X` and its left and right edges meet
    /// at `-X`. [`equirectangular_uv`] is this mapping.
    ///
    /// Supported source formats are `Rgba32Float` (what the `.hdr` loader produces),
    /// `Rgba16Float`, `Rgba8Unorm` and `Rgba8UnormSrgb`; sRGB sources are decoded to linear.
    /// Each output texel samples the source bilinearly at the direction through its centre,
    /// wrapping in `u` and clamping in `v`.
    ///
    /// Every channel is clamped to the `f16` maximum (65504) and non-finite source values are
    /// written as 0, so an unclipped `.hdr` sun never becomes an infinity that would poison
    /// downstream sums such as `bevy_solari`'s importance pyramid.
    ///
    /// # Face layout
    ///
    /// Layers are ordered `+X, -X, +Y, -Y, +Z, -Z` in cube space with the texel-to-direction
    /// mapping wgpu (Vulkan/D3D/GL) uses for cube sampling. Bevy's skybox and environment-map
    /// shaders look cubemaps up with `dir * vec3(1, 1, -1)`, so the faces are built with world Z
    /// negated: layer 4 (`+Z`) holds what is seen looking down world `-Z` and layer 5 (`-Z`)
    /// holds world `+Z`. The panorama's `+Z` therefore appears at world `+Z` in a `Skybox`.
    pub fn equirectangular_to_cubemap(
        &self,
        face_size: u32,
    ) -> Result<Image, EquirectangularToCubemapError> {
        if face_size == 0 {
            return Err(EquirectangularToCubemapError::InvalidFaceSize);
        }
        if self.texture_descriptor.dimension != TextureDimension::D2
            || self.texture_descriptor.size.depth_or_array_layers != 1
            || self.texture_descriptor.mip_level_count != 1
        {
            return Err(EquirectangularToCubemapError::WrongDimension);
        }
        let Some(data) = self.data.as_deref() else {
            return Err(EquirectangularToCubemapError::Uninitialized);
        };
        let source_format = self.texture_descriptor.format;
        let width = self.width() as usize;
        let height = self.height() as usize;
        if width == 0 || height == 0 {
            return Err(EquirectangularToCubemapError::WrongDimension);
        }
        let pixel_size = source_format
            .pixel_size()
            .map_err(|_| EquirectangularToCubemapError::UnsupportedSourceFormat(source_format))?;
        let bytes = data
            .get(..width * height * pixel_size)
            .ok_or(EquirectangularToCubemapError::Uninitialized)?;

        let texels: Cow<[[f32; 4]]> = match source_format {
            TextureFormat::Rgba32Float => bytemuck::try_cast_slice(bytes)
                .map(Cow::Borrowed)
                .unwrap_or_else(|_| Cow::Owned(bytemuck::pod_collect_to_vec(bytes))),
            TextureFormat::Rgba16Float => Cow::Owned(decode(bytes, pixel_size, |texel| {
                core::array::from_fn(|i| {
                    half::f16::from_le_bytes([texel[2 * i], texel[2 * i + 1]]).to_f32()
                })
            })),
            TextureFormat::Rgba8Unorm => Cow::Owned(decode(bytes, pixel_size, |texel| {
                core::array::from_fn(|i| texel[i] as f32 / u8::MAX as f32)
            })),
            TextureFormat::Rgba8UnormSrgb => {
                let lut: [f32; 256] =
                    core::array::from_fn(|i| Srgba::gamma_function(i as f32 / u8::MAX as f32));
                Cow::Owned(decode(bytes, pixel_size, |texel| {
                    [
                        lut[texel[0] as usize],
                        lut[texel[1] as usize],
                        lut[texel[2] as usize],
                        texel[3] as f32 / u8::MAX as f32,
                    ]
                }))
            }
            other => {
                return Err(EquirectangularToCubemapError::UnsupportedSourceFormat(
                    other,
                ))
            }
        };

        let source = Equirect {
            width,
            height,
            texels: &texels,
        };
        let mut faces = source.render(face_size);
        for channel in &mut faces {
            // f16::from_f32 saturates to infinity above f16::MAX; keep the value finite.
            *channel = if channel.is_finite() {
                channel.clamp(-F16_MAX, F16_MAX)
            } else {
                0.0
            };
        }
        let mut halves = vec![0u16; faces.len()];
        halves
            .reinterpret_cast_mut::<half::f16>()
            .convert_from_f32_slice(&faces);
        drop(faces);

        let mut cubemap = Image::new(
            Extent3d {
                width: face_size,
                height: face_size,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            bytemuck::cast_slice(&halves).to_vec(),
            TextureFormat::Rgba16Float,
            self.asset_usage,
        );
        cubemap.sampler = self.sampler.clone();
        cubemap.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..Default::default()
        });
        Ok(cubemap)
    }
}

/// Decodes every `pixel_size`-byte texel of `bytes` to linear RGBA.
fn decode(bytes: &[u8], pixel_size: usize, texel: impl Fn(&[u8]) -> [f32; 4]) -> Vec<[f32; 4]> {
    bytes.chunks_exact(pixel_size).map(texel).collect()
}

/// World direction through the centre of texel `(x, y)` of cube `face`, with `u`/`v` already
/// mapped to `-1..1` (`v = -1` on the top row).
///
/// The first three components are wgpu's cube-space direction for the face
/// (`+X: (1, -v, -u)`, `-X: (-1, -v, u)`, `+Y: (u, 1, v)`, `-Y: (u, -1, -v)`,
/// `+Z: (u, -v, 1)`, `-Z: (-u, -v, -1)`); Bevy samples cubemaps with the world Z negated, so
/// the world direction is that vector with Z flipped.
pub(crate) fn cubemap_texel_world_direction(face: u32, u: f32, v: f32) -> Vec3 {
    let cube = match face {
        0 => Vec3::new(1.0, -v, -u),
        1 => Vec3::new(-1.0, -v, u),
        2 => Vec3::new(u, 1.0, v),
        3 => Vec3::new(u, -1.0, -v),
        4 => Vec3::new(u, -v, 1.0),
        _ => Vec3::new(-u, -v, -1.0),
    };
    Vec3::new(cube.x, cube.y, -cube.z).normalize()
}

/// Lat-long texture coordinates in `0..1` of a unit world direction.
pub fn equirectangular_uv(dir: Vec3) -> (f32, f32) {
    let u = 0.5 + ops::atan2(dir.z, dir.x) / TAU;
    let v = ops::acos(dir.y.clamp(-1.0, 1.0)) / PI;
    (u, v)
}

/// A decoded lat-long panorama, row-major linear RGBA.
struct Equirect<'a> {
    width: usize,
    height: usize,
    texels: &'a [[f32; 4]],
}

impl Equirect<'_> {
    /// Renders all six faces, `face_size` square, into one tightly packed RGBA `f32` buffer in
    /// layer, row, column order.
    fn render(&self, face_size: u32) -> Vec<f32> {
        let fetch = |x: usize, y: usize| Vec4::from(self.texels[y * self.width + x]);
        let width_f = self.width as f32;
        let height_f = self.height as f32;
        let max_y = self.height - 1;

        let face_size = face_size as usize;
        let mut out = vec![0.0f32; 6 * face_size * face_size * 4];
        let inv_face_size = 1.0 / face_size as f32;
        for (i, texel) in out.chunks_exact_mut(4).enumerate() {
            let x = i % face_size;
            let y = i / face_size % face_size;
            let face = (i / (face_size * face_size)) as u32;
            let u = (x as f32 + 0.5) * inv_face_size * 2.0 - 1.0;
            let v = (y as f32 + 0.5) * inv_face_size * 2.0 - 1.0;
            let dir = cubemap_texel_world_direction(face, u, v);
            let (su, sv) = equirectangular_uv(dir);

            // Bilinear tap around the continuous pixel position; wrap in u, clamp in v.
            // `py` is clamped before the split into row and fraction so the top and
            // bottom rows are not blended past the edge.
            let px = su * width_f - 0.5;
            let py = (sv * height_f - 0.5).clamp(0.0, max_y as f32);
            let x0f = px.floor();
            let y0f = py.floor();
            let fx = px - x0f;
            let fy = py - y0f;
            let x0 = (x0f as isize).rem_euclid(self.width as isize) as usize;
            let x1 = if x0 + 1 == self.width { 0 } else { x0 + 1 };
            let y0 = (y0f as isize).clamp(0, max_y as isize) as usize;
            let y1 = (y0 + 1).min(max_y);

            let a = fetch(x0, y0);
            let b = fetch(x1, y0);
            let c = fetch(x0, y1);
            let d = fetch(x1, y1);
            // Written out rather than `Vec4::lerp` so the rounding matches the scalar
            // `a + (b - a) * t` form exactly.
            let top = a + (b - a) * fx;
            let bottom = c + (d - c) * fx;
            texel.copy_from_slice(&(top + (bottom - top) * fy).to_array());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::RenderAssetUsages;
    use bevy_math::UVec3;

    fn equirect_rgba32(width: u32, height: u32, texel: impl Fn(u32, u32) -> [f32; 4]) -> Image {
        let mut data = Vec::with_capacity((width * height * 16) as usize);
        for y in 0..height {
            for x in 0..width {
                for channel in texel(x, y) {
                    data.extend_from_slice(&channel.to_le_bytes());
                }
            }
        }
        Image::new(
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            data,
            TextureFormat::Rgba32Float,
            RenderAssetUsages::all(),
        )
    }

    fn texel(cubemap: &Image, face: u32, x: u32, y: u32) -> [f32; 4] {
        assert_eq!(
            cubemap.texture_descriptor.format,
            TextureFormat::Rgba16Float
        );
        let bytes = cubemap.pixel_bytes(UVec3::new(x, y, face)).unwrap();
        core::array::from_fn(|i| {
            half::f16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]).to_f32()
        })
    }

    fn assert_close(actual: [f32; 4], expected: [f32; 4], tolerance: f32) {
        for i in 0..4 {
            assert!(
                (actual[i] - expected[i]).abs() <= tolerance,
                "channel {i}: {actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn output_is_a_cube_view_over_six_layers() {
        let cubemap = equirect_rgba32(8, 4, |_, _| [0.0; 4])
            .equirectangular_to_cubemap(4)
            .unwrap();
        assert_eq!(cubemap.texture_descriptor.size.depth_or_array_layers, 6);
        assert_eq!(cubemap.texture_descriptor.dimension, TextureDimension::D2);
        assert_eq!(
            cubemap.texture_descriptor.format,
            TextureFormat::Rgba16Float
        );
        assert_eq!(
            cubemap
                .texture_view_descriptor
                .as_ref()
                .and_then(|view| view.dimension),
            Some(TextureViewDimension::Cube)
        );
        assert_eq!(cubemap.data.as_ref().unwrap().len(), 6 * 4 * 4 * 8);
    }

    /// The u landmarks the docs promise, checked against the sentence rather than the formula.
    #[test]
    fn lat_long_u_matches_the_documented_landmarks() {
        let u = |dir| equirectangular_uv(dir).0;
        assert!((u(Vec3::X) - 0.5).abs() < 1e-6, "+X: {}", u(Vec3::X));
        assert!((u(Vec3::Z) - 0.75).abs() < 1e-6, "+Z: {}", u(Vec3::Z));
        assert!(
            (u(Vec3::NEG_Z) - 0.25).abs() < 1e-6,
            "-Z: {}",
            u(Vec3::NEG_Z)
        );
        let minus_x = u(Vec3::NEG_X);
        assert!(
            minus_x.abs() < 1e-6 || (minus_x - 1.0).abs() < 1e-6,
            "-X: {minus_x}"
        );
        assert!(equirectangular_uv(Vec3::Y).1.abs() < 1e-6);
        assert!((equirectangular_uv(Vec3::NEG_Y).1 - 1.0).abs() < 1e-6);
    }

    /// Values beyond the f16 range saturate instead of becoming infinities, and non-finite
    /// source texels are written as 0.
    #[test]
    fn rgba16_output_stays_finite() {
        let hot = [1.0e6, f32::INFINITY, f32::NAN, 1.0];
        let cubemap = equirect_rgba32(16, 8, |_, _| hot)
            .equirectangular_to_cubemap(4)
            .unwrap();
        for face in 0..6 {
            for y in 0..4 {
                for x in 0..4 {
                    let t = texel(&cubemap, face, x, y);
                    assert!(t.iter().all(|c| c.is_finite()), "{t:?}");
                    assert_eq!(t[0], F16_MAX);
                    assert_eq!(t[1], 0.0);
                    assert_eq!(t[2], 0.0);
                    assert_eq!(t[3], 1.0);
                }
            }
        }
    }

    #[test]
    fn constant_panorama_gives_constant_faces() {
        let color = [0.25, 0.5, 2.0, 1.0];
        let cubemap = equirect_rgba32(16, 8, |_, _| color)
            .equirectangular_to_cubemap(8)
            .unwrap();
        for face in 0..6 {
            for y in 0..8 {
                for x in 0..8 {
                    assert_close(texel(&cubemap, face, x, y), color, 1e-3);
                }
            }
        }
    }

    #[test]
    fn hemispheres_land_on_the_y_faces() {
        let white = [1.0; 4];
        let black = [0.0, 0.0, 0.0, 1.0];
        let size = 16;
        let cubemap = equirect_rgba32(64, 32, |_, y| if y < 16 { white } else { black })
            .equirectangular_to_cubemap(size)
            .unwrap();
        for y in 0..size {
            for x in 0..size {
                assert_close(texel(&cubemap, 2, x, y), white, 1e-3);
                assert_close(texel(&cubemap, 3, x, y), black, 1e-3);
            }
        }
        // Side faces: top row looks up, bottom row looks down.
        for face in [0, 1, 4, 5] {
            for x in 0..size {
                assert_close(texel(&cubemap, face, x, 0), white, 1e-3);
                assert_close(texel(&cubemap, face, x, size - 1), black, 1e-3);
            }
        }
    }

    /// One unique colour per axis direction, painted as a small disc around each of the six
    /// lat-long positions; each must land at the centre of its face, and nowhere else.
    #[test]
    fn axis_directions_land_on_their_faces() {
        let width = 256;
        let height = 128;
        // (world direction, colour), in layer order: cube-space +Z (layer 4) is world -Z
        // and cube-space -Z (layer 5) is world +Z because Bevy samples cubemaps with Z negated.
        let axes = [
            (Vec3::X, [1.0, 0.0, 0.0, 1.0]),
            (Vec3::NEG_X, [0.0, 1.0, 0.0, 1.0]),
            (Vec3::Y, [0.0, 0.0, 1.0, 1.0]),
            (Vec3::NEG_Y, [1.0, 1.0, 0.0, 1.0]),
            (Vec3::NEG_Z, [1.0, 0.0, 1.0, 1.0]),
            (Vec3::Z, [0.0, 1.0, 1.0, 1.0]),
        ];
        let panorama = equirect_rgba32(width, height, |x, y| {
            // Direction of this lat-long texel, inverting the documented mapping.
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;
            let phi = (u - 0.5) * TAU;
            let theta = v * PI;
            let (sin_theta, cos_theta) = ops::sin_cos(theta);
            let (sin_phi, cos_phi) = ops::sin_cos(phi);
            let dir = Vec3::new(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
            for (axis, color) in axes {
                if dir.dot(axis) > 0.97 {
                    return color;
                }
            }
            [0.0, 0.0, 0.0, 1.0]
        });
        let size = 32;
        let cubemap = panorama.equirectangular_to_cubemap(size).unwrap();
        let centre = size / 2;
        for (face, (_, color)) in axes.iter().enumerate() {
            let face = face as u32;
            assert_close(texel(&cubemap, face, centre, centre), *color, 1e-3);
            assert_close(texel(&cubemap, face, centre - 1, centre - 1), *color, 1e-3);
            // Corners look 54.7 degrees off-axis and must be black.
            assert_close(texel(&cubemap, face, 0, 0), [0.0, 0.0, 0.0, 1.0], 1e-3);
            assert_close(
                texel(&cubemap, face, size - 1, size - 1),
                [0.0, 0.0, 0.0, 1.0],
                1e-3,
            );
        }
    }

    /// The texel-to-direction mapping must be self-consistent: adjacent faces share edges.
    #[test]
    fn face_edges_are_continuous() {
        let size = 8;
        let edge = |face, x, y| cubemap_texel_world_direction(face, x, y);
        let step = 2.0 / size as f32;
        for i in 0..size {
            let t = -1.0 + (i as f32 + 0.5) * step;
            // +X right edge meets -Z left edge, +X left edge meets +Z right edge, +Y bottom
            // row meets +Z top row, -Y top row meets +Z bottom row (all in cube space).
            let a = edge(0, 1.0, t);
            let b = edge(5, -1.0, t);
            assert!(a.abs_diff_eq(b, 1e-5), "{a} vs {b}");
            let a = edge(0, -1.0, t);
            let b = edge(4, 1.0, t);
            assert!(a.abs_diff_eq(b, 1e-5), "{a} vs {b}");
            let a = edge(2, t, 1.0);
            let b = edge(4, t, -1.0);
            assert!(a.abs_diff_eq(b, 1e-5), "{a} vs {b}");
            let a = edge(3, t, -1.0);
            let b = edge(4, t, 1.0);
            assert!(a.abs_diff_eq(b, 1e-5), "{a} vs {b}");
        }
    }

    #[test]
    fn srgb_source_is_linearised() {
        let mut data = Vec::new();
        for _ in 0..(8 * 4) {
            data.extend_from_slice(&[128, 255, 0, 255]);
        }
        let image = Image::new(
            Extent3d {
                width: 8,
                height: 4,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            data,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::all(),
        );
        let cubemap = image.equirectangular_to_cubemap(2).unwrap();
        let expected = [Srgba::gamma_function(128.0 / 255.0), 1.0, 0.0, 1.0];
        assert_close(texel(&cubemap, 0, 0, 0), expected, 2e-3);
    }

    #[test]
    fn rejects_bad_inputs() {
        let mut image = equirect_rgba32(8, 4, |_, _| [0.0; 4]);
        assert_eq!(
            image.equirectangular_to_cubemap(0),
            Err(EquirectangularToCubemapError::InvalidFaceSize)
        );
        image.texture_descriptor.format = TextureFormat::Rg32Float;
        assert_eq!(
            image.equirectangular_to_cubemap(2),
            Err(EquirectangularToCubemapError::UnsupportedSourceFormat(
                TextureFormat::Rg32Float
            ))
        );
        image.texture_descriptor.format = TextureFormat::Rgba32Float;
        image.data = None;
        assert_eq!(
            image.equirectangular_to_cubemap(2),
            Err(EquirectangularToCubemapError::Uninitialized)
        );
    }
}
