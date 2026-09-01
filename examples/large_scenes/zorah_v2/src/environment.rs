//! The sidecar's equirectangular HDR as the camera's environment light: loaded
//! as an `Image`, measured, converted to a cubemap off the main thread, then
//! bound as `EnvironmentMapLight` (Solari importance-samples its specular map)
//! and `Skybox` (the backdrop behind the geometry Solari's primary rays miss),
//! with the sun's excess split into a matching directional light.

use std::f32::consts::{PI, TAU};

use bevy::{
    asset::{LoadState, RenderAssetUsages},
    image::{EquirectangularToCubemapError, HdrTextureLoaderSettings},
    light::Skybox,
    math::ops,
    prelude::*,
    render::render_resource::TextureFormat,
    tasks::{block_on, poll_once, AsyncComputeTaskPool, Task},
};

use crate::{scene::SceneEnvironmentMap, setup::ZorahCamera};

/// Cube face edge, from the 4096x2048 source: each face spans 90 degrees, so
/// 1024 texels keep the panorama's angular resolution along the equator.
const CUBEMAP_FACE_SIZE: u32 = 1024;
/// The map's values are in arbitrary units. Its intensity is chosen so that
/// its sky, sun excluded, averages the radiance of a uniform sky of this
/// illuminance (`E / pi` nits), a clear-day sky; the sun's excess above the
/// sky threshold is split into a directional light of the same energy.
const ENVIRONMENT_MAP_SKY_ILLUMINANCE: f32 = 15_000.0;
/// Texels brighter than this multiple of the whole hemisphere's mean count as
/// the sun (and its aureole) rather than sky, both when the sky mean is
/// measured and when `split_sun` clamps the disc.
/// An unclipped sun disc is thousands of times the sky; a clipped one still
/// tens; haze near it a few.
const SUN_LUMINANCE_RATIO: f32 = 32.0;

/// The `EnvironmentMapLight::intensity` that gives a map whose sky averages
/// `sky_mean_luminance` (per unit intensity) the target mean radiance, times
/// the sidecar's own multiplier.
pub fn intensity(sky_mean_luminance: f32, multiplier: f32) -> f32 {
    let target = ENVIRONMENT_MAP_SKY_ILLUMINANCE / PI;
    let scale = if sky_mean_luminance > 0.0 {
        target / sky_mean_luminance
    } else {
        1.0
    };
    multiplier * scale
}

/// The measuring geometry of an `Rgba32Float` panorama's upper hemisphere,
/// shared by the sky measurement and the sun split so the two cannot disagree
/// about a texel's solid angle.
struct PanoramaGeometry {
    width: usize,
    upper_rows: usize,
    d_theta: f32,
    d_phi: f32,
}

impl PanoramaGeometry {
    fn of(image: &Image) -> Option<Self> {
        if image.texture_descriptor.format != TextureFormat::Rgba32Float {
            return None;
        }
        let width = image.width() as usize;
        let height = image.height() as usize;
        if width == 0 || height == 0 {
            return None;
        }
        Some(Self {
            width,
            upper_rows: height.div_ceil(2),
            d_theta: PI / height as f32,
            d_phi: TAU / width as f32,
        })
    }

    /// A row's texel solid angle: `sin(theta) dtheta dphi`.
    fn row_solid_angle(&self, y: usize) -> f32 {
        ops::sin((y as f32 + 0.5) * self.d_theta) * self.d_theta * self.d_phi
    }

    /// Direction of the texel's centre; the inverse of `equirectangular_uv`.
    fn direction(&self, x: usize, y: usize) -> Vec3 {
        let (sin_theta, cos_theta) = ops::sin_cos((y as f32 + 0.5) * self.d_theta);
        let (sin_phi, cos_phi) = ops::sin_cos((x as f32 + 0.5) * self.d_phi - PI);
        Vec3::new(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi)
    }
}

/// The texel's linear color, NaNs and negatives clamped away.
fn texel_rgb(bytes: &[u8]) -> LinearRgba {
    let channel = |at: usize| {
        let value = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        }
    };
    LinearRgba::rgb(channel(0), channel(4), channel(8))
}

/// Solid-angle-weighted mean luminance of an equirectangular panorama's upper
/// hemisphere, sun texels excluded; `None` unless it is `Rgba32Float` (what
/// the `.hdr` loader produces).
pub fn measure_sky_mean_luminance(image: &Image) -> Option<f32> {
    let geometry = PanoramaGeometry::of(image)?;
    let data = image
        .data
        .as_deref()?
        .get(..geometry.width * geometry.upper_rows * 16)?;
    let luminances = data
        .chunks_exact(16)
        .map(|texel| texel_rgb(texel).luminance())
        .collect::<Vec<f32>>();
    let mean = |sky: &dyn Fn(f32) -> bool| {
        let mut solid_angle = 0.0f64;
        let mut weighted_luminance = 0.0f64;
        for (y, row) in luminances.chunks_exact(geometry.width).enumerate() {
            let weight = geometry.row_solid_angle(y);
            for &luminance in row.iter().filter(|&&luminance| sky(luminance)) {
                solid_angle += weight as f64;
                weighted_luminance += (weight * luminance) as f64;
            }
        }
        (solid_angle > 0.0).then(|| (weighted_luminance / solid_angle) as f32)
    };
    let hemisphere_mean = mean(&|_| true)?;
    let sun_threshold = SUN_LUMINANCE_RATIO * hemisphere_mean;
    Some(mean(&|luminance| luminance <= sun_threshold).unwrap_or(hemisphere_mean))
}

/// The sun measured out of the panorama. The sidecar's RTXMG configuration
/// lights the scene with an analytic sun next to the map, and a delta light is
/// also what Solari's reservoir-less world cache updates can sample reliably.
pub struct MeasuredSun {
    /// Excess-weighted mean direction toward the sun, in the map's frame.
    pub direction_in_map: Vec3,
    /// Integral of luminance above the sky threshold over the sun texels
    /// (map units times steradians); the map's `intensity` factor makes it
    /// lux.
    pub excess_illuminance: f32,
    /// Excess-weighted sun chromaticity, normalized to unit luminance.
    pub color: Color,
}

/// Splits the panorama's sun: texels above the sky threshold are clamped to
/// it in place, and the removed excess is returned for an equivalent
/// directional light, conserving the total energy.
pub fn split_sun(image: &mut Image) -> Option<MeasuredSun> {
    let geometry = PanoramaGeometry::of(image)?;
    let data = image
        .data
        .as_deref_mut()?
        .get_mut(..geometry.width * geometry.upper_rows * 16)?;

    let mut solid_angle = 0.0f64;
    let mut weighted_luminance = 0.0f64;
    for (y, row) in data.chunks_exact(geometry.width * 16).enumerate() {
        let weight = geometry.row_solid_angle(y);
        for texel in row.chunks_exact(16) {
            solid_angle += weight as f64;
            weighted_luminance += (weight * texel_rgb(texel).luminance()) as f64;
        }
    }
    if solid_angle <= 0.0 {
        return None;
    }
    let sun_threshold = SUN_LUMINANCE_RATIO * (weighted_luminance / solid_angle) as f32;

    let mut excess = 0.0f64;
    let mut direction = Vec3::ZERO;
    let mut rgb_excess = Vec3::ZERO;
    for (y, row) in data.chunks_exact_mut(geometry.width * 16).enumerate() {
        let weight = geometry.row_solid_angle(y);
        for (x, texel) in row.chunks_exact_mut(16).enumerate() {
            let rgb = texel_rgb(texel);
            let luminance = rgb.luminance();
            if luminance <= sun_threshold {
                continue;
            }
            let texel_excess = (luminance - sun_threshold) * weight;
            excess += texel_excess as f64;
            direction += geometry.direction(x, y) * texel_excess;
            let kept = sun_threshold / luminance;
            rgb_excess += Vec3::new(rgb.red, rgb.green, rgb.blue) * ((1.0 - kept) * weight);
            for (channel, value) in [rgb.red, rgb.green, rgb.blue].into_iter().enumerate() {
                texel[channel * 4..channel * 4 + 4].copy_from_slice(&(value * kept).to_le_bytes());
            }
        }
    }
    if excess <= 0.0 {
        return None;
    }
    Some(MeasuredSun {
        direction_in_map: direction.normalize(),
        excess_illuminance: excess as f32,
        color: Color::LinearRgba(
            LinearRgba::rgb(rgb_excess.x, rgb_excess.y, rgb_excess.z).with_luminance(1.0),
        ),
    })
}

/// The `EnvironmentMapLight`/`Skybox` rotation for the sidecar's
/// `envmap rotation`. RTXMG applies minus the sidecar angle (about +Y) to its
/// lookup direction, and the converter's seam is half a turn from RTXMG's:
/// hence `degrees - 180`.
pub fn environment_map_rotation(degrees: f32) -> Quat {
    Quat::from_rotation_y((degrees - 180.0).to_radians())
}

/// The sidecar HDR's load and conversion in flight; removed once the map is
/// on the camera or the map has proved unusable.
#[derive(Resource)]
pub struct EnvironmentMapLoad {
    map: SceneEnvironmentMap,
    source: Handle<Image>,
    /// The cubemap, the source's measured sky mean luminance, and its sun.
    task: Option<
        Task<Result<(Image, Option<f32>, Option<MeasuredSun>), EquirectangularToCubemapError>>,
    >,
}

impl EnvironmentMapLoad {
    /// Starts loading the sidecar's `.hdr`. It stays on the CPU only: the
    /// cubemap is what the GPU gets.
    pub fn start(asset_server: &AssetServer, map: SceneEnvironmentMap) -> Self {
        let source = asset_server
            .load_builder()
            .with_settings(|settings: &mut HdrTextureLoaderSettings| {
                settings.asset_usage = RenderAssetUsages::MAIN_WORLD;
            })
            .load::<Image>(map.path.clone());
        Self {
            map,
            source,
            task: None,
        }
    }
}

/// Waits for the panorama, converts it on `AsyncComputeTaskPool` (the 4k
/// source takes a third of a second in release, too long for a frame), then
/// puts it on the camera as `EnvironmentMapLight` and `Skybox`, with the
/// sun's excess as a directional light. A load or
/// conversion failure warns and leaves only the emissive and fire lights. Runs while
/// `EnvironmentMapLoad` exists.
pub fn install_environment_map(
    mut commands: Commands,
    mut load: ResMut<EnvironmentMapLoad>,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    camera: Single<Entity, With<ZorahCamera>>,
) {
    let settled: Result<_, String> = match load.task.as_mut() {
        Some(task) => match block_on(poll_once(task)) {
            Some(converted) => converted.map_err(|error| error.to_string()),
            None => return,
        },
        None => match asset_server.load_state(&load.source) {
            LoadState::NotLoaded | LoadState::Loading => return,
            LoadState::Failed(error) => Err(error.to_string()),
            LoadState::Loaded => {
                // Ownership rather than a clone: nothing else wants the
                // 128 MiB source.
                let Some(mut source) = images.remove(&load.source) else {
                    return;
                };
                info!(
                    path = load.map.path,
                    source = format!("{}x{}", source.width(), source.height()),
                    face_size = CUBEMAP_FACE_SIZE,
                    "converting the environment map"
                );
                load.task = Some(AsyncComputeTaskPool::get().spawn(async move {
                    // Clamp the sun out before the sky is measured or converted, so
                    // neither the mean nor the cubemap keeps the disc's energy.
                    let sun = split_sun(&mut source);
                    let sky_mean_luminance = measure_sky_mean_luminance(&source);
                    let mut cubemap = source.equirectangular_to_cubemap(CUBEMAP_FACE_SIZE)?;
                    // The converter copies the source's CPU-only usage; the
                    // cube is for the GPU, and its 48 MiB need not stay behind
                    // after the upload.
                    cubemap.asset_usage = RenderAssetUsages::RENDER_WORLD;
                    Ok((cubemap, sky_mean_luminance, sun))
                }));
                return;
            }
        },
    };
    match settled {
        Ok((cubemap, sky_mean_luminance, sun)) => {
            let intensity = sky_mean_luminance.map_or(load.map.intensity, |mean| {
                intensity(mean, load.map.intensity)
            });
            let rotation = environment_map_rotation(load.map.rotation_degrees);
            info!(
                path = load.map.path,
                rotation_degrees = load.map.rotation_degrees,
                multiplier = load.map.intensity,
                intensity,
                sky_mean_luminance = ?sky_mean_luminance,
                "environment map converted and bound"
            );
            let cubemap = images.add(cubemap);
            commands.entity(*camera).insert((
                // The same unfiltered cube serves as the diffuse map: the
                // raster path only lights the preview while the BLASes build,
                // and Solari reads the specular map's mip 0 alone.
                EnvironmentMapLight {
                    diffuse_map: cubemap.clone(),
                    specular_map: cubemap.clone(),
                    intensity,
                    rotation,
                    ..default()
                },
                Skybox {
                    image: Some(cubemap),
                    brightness: intensity,
                    rotation,
                },
            ));
            if let Some(sun) = sun {
                // The excess scales like the map, so this is the disc's real
                // illuminance; the direction turns with the map. Solari
                // replaces shadow maps with its own visibility rays, and the
                // default DirectionalLight casts none.
                let illuminance = sun.excess_illuminance * intensity;
                let direction = rotation * sun.direction_in_map;
                info!(
                    ?direction,
                    illuminance, "sun split from the environment map into a directional light"
                );
                commands.spawn((
                    Name::new("Environment map sun"),
                    DirectionalLight {
                        color: sun.color,
                        illuminance,
                        ..default()
                    },
                    Transform::IDENTITY.looking_to(-direction, Vec3::Y),
                ));
            }
        }
        Err(reason) => {
            warn!(
                path = load.map.path,
                "environment map unusable ({reason}); only the emissive and fire lights remain"
            );
        }
    }
    commands.remove_resource::<EnvironmentMapLoad>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{
        image::equirectangular_uv,
        render::render_resource::{Extent3d, TextureDimension},
    };

    /// A panorama of uniform luminance 1 with one texel of luminance `sun`
    /// at direction `dir`.
    fn panorama(width: u32, height: u32, sun_dir: Vec3, sun: f32) -> Image {
        let mut data = vec![0u8; (width * height * 16) as usize];
        let (u, v) = equirectangular_uv(sun_dir);
        let sun_x = ((u * width as f32) as u32).min(width - 1);
        let sun_y = ((v * height as f32) as u32).min(height - 1);
        for y in 0..height {
            for x in 0..width {
                let value = if (x, y) == (sun_x, sun_y) { sun } else { 1.0 };
                let at = ((y * width + x) * 16) as usize;
                for channel in 0..3 {
                    data[at + channel * 4..at + channel * 4 + 4]
                        .copy_from_slice(&value.to_le_bytes());
                }
                data[at + 12..at + 16].copy_from_slice(&1.0f32.to_le_bytes());
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
            RenderAssetUsages::MAIN_WORLD,
        )
    }

    #[test]
    fn uniform_sky_measures_as_itself() {
        let mean = measure_sky_mean_luminance(&panorama(256, 128, Vec3::Y, 1.0)).unwrap();
        assert!((mean - 1.0).abs() < 1e-3, "{mean}");
        // The intensity makes that sky the target one.
        let unit = intensity(mean, 1.0);
        assert!((unit - ENVIRONMENT_MAP_SKY_ILLUMINANCE / PI).abs() < 5.0);
        assert!((intensity(mean, 2.0) - 2.0 * unit).abs() < 1e-3);
    }

    #[test]
    fn sun_texel_is_excluded_from_the_sky() {
        let sun_dir = Vec3::new(0.6, 0.7, 0.36).normalize();
        let mean = measure_sky_mean_luminance(&panorama(512, 256, sun_dir, 1.0e5)).unwrap();
        assert!((mean - 1.0).abs() < 1e-3, "{mean}");
    }

    #[test]
    fn sun_split_takes_the_excess_and_clamps_the_disc() {
        let sun_dir = Vec3::new(0.6, 0.7, 0.36).normalize();
        let mut image = panorama(512, 256, sun_dir, 1.0e5);
        let sun = split_sun(&mut image).unwrap();
        assert!(
            sun.direction_in_map.dot(sun_dir) > 0.999,
            "{}",
            sun.direction_in_map
        );
        // The clamp is the energy check: with the disc's excess taken, the
        // map reads as pure sky again.
        let mean = measure_sky_mean_luminance(&image).unwrap();
        assert!((mean - 1.0).abs() < 1e-2, "{mean}");
    }

    #[test]
    fn a_sunless_sky_has_nothing_to_split() {
        assert!(split_sun(&mut panorama(64, 32, Vec3::Y, 1.0)).is_none());
    }

    #[test]
    fn rotation_turns_the_map_by_the_documented_angle() {
        // With the seam offset alone (rotation 0), the converter's +X faces -X.
        let flipped = environment_map_rotation(0.0) * Vec3::X;
        assert!(flipped.abs_diff_eq(Vec3::NEG_X, 1e-5), "{flipped}");
        // A positive sidecar rotation turns the map the way RTXMG's
        // transposed lookup does: +90 degrees is `from_rotation_y(-90)` on
        // top, which takes +X to +Z (azimuth 0 to 90).
        let quarter = environment_map_rotation(90.0) * Vec3::X;
        assert!(quarter.abs_diff_eq(Vec3::Z, 1e-5), "{quarter}");
    }
}
