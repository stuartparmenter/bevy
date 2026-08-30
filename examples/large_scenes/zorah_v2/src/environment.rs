//! The sidecar's equirectangular HDR as the camera's environment light: loaded
//! as an `Image`, measured, converted to a cubemap off the main thread, then
//! bound as `EnvironmentMapLight` (Solari importance-samples its specular map)
//! and `Skybox` (the backdrop behind the geometry Solari's primary rays miss).

use std::f32::consts::PI;

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
/// illuminance (`E / pi` nits), a clear-day sky; the map's own sun then
/// lands wherever the HDR put it.
const ENVIRONMENT_MAP_SKY_ILLUMINANCE: f32 = 15_000.0;
/// Texels brighter than this multiple of the whole hemisphere's mean count as
/// the sun (and its aureole) rather than sky when the sky mean is measured.
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

/// Solid-angle-weighted mean luminance of an equirectangular panorama's upper
/// hemisphere, sun texels excluded; `None` unless it is `Rgba32Float` (what
/// the `.hdr` loader produces).
pub fn measure_sky_mean_luminance(image: &Image) -> Option<f32> {
    if image.texture_descriptor.format != TextureFormat::Rgba32Float {
        return None;
    }
    let width = image.width() as usize;
    let height = image.height() as usize;
    let upper_rows = height.div_ceil(2);
    let data = image.data.as_deref()?.get(..width * upper_rows * 16)?;
    if width == 0 || height == 0 {
        return None;
    }
    let channel = |bytes: &[u8]| {
        let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        }
    };
    let luminances = data
        .chunks_exact(16)
        .map(|texel| {
            LinearRgba::rgb(
                channel(&texel[0..4]),
                channel(&texel[4..8]),
                channel(&texel[8..12]),
            )
            .luminance()
        })
        .collect::<Vec<f32>>();
    // A row's texel solid angle: sin(theta) dtheta dphi.
    let d_theta = PI / height as f32;
    let d_phi = 2.0 * PI / width as f32;
    let row_weight = |y: usize| ops::sin((y as f32 + 0.5) * d_theta) * d_theta * d_phi;
    let mean = |sky: &dyn Fn(f32) -> bool| {
        let mut solid_angle = 0.0f64;
        let mut weighted_luminance = 0.0f64;
        for (y, row) in luminances.chunks_exact(width).enumerate() {
            let weight = row_weight(y);
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
    /// The cubemap and the source's measured sky mean luminance.
    task: Option<Task<Result<(Image, Option<f32>), EquirectangularToCubemapError>>>,
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
/// puts it on the camera as `EnvironmentMapLight` and `Skybox`. A load or
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
                let Some(source) = images.remove(&load.source) else {
                    return;
                };
                info!(
                    path = load.map.path,
                    source = format!("{}x{}", source.width(), source.height()),
                    face_size = CUBEMAP_FACE_SIZE,
                    "converting the environment map"
                );
                load.task = Some(AsyncComputeTaskPool::get().spawn(async move {
                    let sky_mean_luminance = measure_sky_mean_luminance(&source);
                    let mut cubemap = source.equirectangular_to_cubemap(CUBEMAP_FACE_SIZE)?;
                    // The converter copies the source's CPU-only usage; the
                    // cube is for the GPU, and its 48 MiB need not stay behind
                    // after the upload.
                    cubemap.asset_usage = RenderAssetUsages::RENDER_WORLD;
                    Ok((cubemap, sky_mean_luminance))
                }));
                return;
            }
        },
    };
    match settled {
        Ok((cubemap, sky_mean_luminance)) => {
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
