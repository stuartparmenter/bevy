//! Procedural Atmospheric Scattering.
//!
//! This plugin implements [Hillaire's 2020 paper](https://sebh.github.io/publications/egsr2020.pdf)
//! on real-time atmospheric scattering. While it *will* work simply as a
//! procedural skybox, it also does much more. It supports dynamic time-of-
//! -day, multiple directional lights, and since it's applied as a post-processing
//! effect *on top* of the existing skybox, a starry skybox would automatically
//! show based on the time of day. Scattering in front of terrain (similar
//! to distance fog, but more complex) is handled as well, and takes into
//! account the directional light color and direction.
//!
//! To add the atmosphere to your scene, spawn an entity with the [`Atmosphere`] component and
//! [`bevy_transform::components::GlobalTransform`], optionally add a [`VolumetricCloudLayer`]
//! (e.g. [`VolumetricCloudLayer::cumulus`]) and/or a [`ThinCloudLayer`] (e.g.
//! [`ThinCloudLayer::cirrus`]), both moved by the atmosphere's [`AtmosphereWind`] profile, and add
//! [`AtmosphereSettings`] to each 3D camera that should render it. Detailed
//! documentation is on the [`Atmosphere`] component.
//!
//! Placement and scene scale come from the entity's transform. With several atmospheres in one
//! scene, each camera picks the atmosphere whose origin is closest in world space.
//!
//! Performance-wise, the effect should be fairly cheap since the LUTs (Look
//! Up Tables) that encode most of the data are small, and take advantage of the
//! fact that the atmosphere is symmetric. Performance is also proportional to
//! the number of directional lights in the scene. In order to tune
//! performance more finely, the [`AtmosphereSettings`] camera component
//! manages the size of each LUT and the sample count for each ray.
//!
//! Given how similar it is to [`crate::volumetric_fog`], it might be expected
//! that these two modules would work together well. However for now using both
//! at once is untested, and might not be physically accurate. These may be
//! integrated into a single module in the future.
//!
//! On web platforms, atmosphere rendering will look slightly different. Specifically, when calculating how light travels
//! through the atmosphere, we use a simpler averaging technique instead of the more
//! complex blending operations. This difference will be resolved for WebGPU in a future release.
//!
//! [Shadertoy]: https://www.shadertoy.com/view/slSXRW
//!
//! [Unreal Engine Implementation]: https://github.com/sebh/UnrealEngineSkyAtmosphere

mod cloud_profile;
mod environment;
pub mod fbm_noise;
mod debug_dump;
pub mod debug_view;
mod gpu_timing;
mod node;
pub mod perlin_worley_noise;
pub mod detail_noise;
pub mod resources;
pub mod weather;

use bevy_app::{App, Plugin, Update};
use bevy_asset::{embedded_asset, AssetId, Handle};
use bevy_image::Image;
use bevy_camera::{Camera3d, Hdr};
use bevy_core_pipeline::{
    core_3d::{main_opaque_pass_3d, main_transparent_pass_3d},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{Changed, With},
    schedule::IntoScheduleConfigs,
    system::{Commands, Query, Res, ResMut},
};
use bevy_light::{atmosphere::ScatteringMedium, Atmosphere};
use bevy_math::{Mat4, Quat, UVec2, UVec3, Vec2, Vec3};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    extract_component::{ExtractComponentPlugin, UniformComponentPlugin},
    render_resource::{DownlevelFlags, ShaderType, SpecializedRenderPipelines},
    renderer::RenderDevice,
    sync_component::{SyncComponent, SyncComponentPlugin},
    sync_world::RenderEntity,
    Extract, ExtractSchedule, RenderStartup,
};
use bevy_render::{
    render_resource::{TextureFormat, TextureUsages},
    renderer::RenderAdapter,
    Render, RenderApp, RenderSystems,
};
use bevy_time::Time;
use bevy_transform::components::GlobalTransform;

use bevy_shader::load_shader_library;
use environment::{
    atmosphere_environment, init_atmosphere_probe_layout, init_atmosphere_probe_pipeline,
    on_insert_atmosphere_environment_map_light, on_remove_atmosphere_environment_map_light,
    prepare_atmosphere_probe_bind_groups, prepare_atmosphere_probe_sources, prepare_probe_textures,
    AtmosphereEnvironmentMap,
};
use fbm_noise::{
    generate_fbm_noise_once, init_fbm_noise_params_buffer, init_fbm_noise_texture,
    prepare_fbm_noise_bind_group, FbmNoiseBindGroupLayout, FbmNoisePipeline, NoiseGenerated,
};
pub use cloud_profile::{
    physical_profile_bounds, CloudProfiles, CLOUD_PROFILE_HEIGHT_RES, CLOUD_PROFILE_RANGE_M,
    CLOUD_PROFILE_TYPE_RES,
};
pub use weather::{CloudBlend, CloudInfluence, CloudInfluenceShape, CloudInfluenceValues};
pub use debug_view::{
    AtmosphereDebugPanel, AtmosphereDebugSource, AtmosphereDebugView, AtmosphereDebugViewPlugin,
    DebugChannels, DebugDecode, DebugMapping,
};
use node::{atmosphere_luts, render_sky};
use perlin_worley_noise::{
    generate_perlin_worley_noise_once, init_perlin_worley_noise_params_buffer,
    init_perlin_worley_noise_texture, prepare_perlin_worley_noise_bind_group,
    PerlinWorleyNoiseBindGroupLayout, PerlinWorleyNoiseGenerated, PerlinWorleyNoisePipeline,
};
use detail_noise::{
    generate_detail_noise_once, init_detail_noise_params_buffer, init_detail_noise_texture,
    prepare_detail_noise_bind_group, DetailNoiseBindGroupLayout, DetailNoiseGenerated,
    DetailNoisePipeline,
};
use perlin_worley_noise::{NoiseMipDownsampleBindGroupLayout, NoiseMipDownsamplePipeline};
use resources::{
    prepare_atmosphere_transforms, prepare_atmosphere_uniforms, queue_render_sky_pipelines,
    AtmosphereTransforms, GpuAtmosphere, RenderSkyBindGroupLayouts,
};
use tracing::warn;

use crate::resources::prepare_atmosphere_buffers;

use self::resources::{
    prepare_atmosphere_bind_groups, prepare_atmosphere_textures,
    prepare_cloud_shadow_temporal_params, prepare_clouds_temporal_params, AtmosphereBindGroupLayouts, AtmosphereBindGroups,
    AtmosphereLutPipelines, AtmosphereTextures, AtmosphereTransformsOffset,
    CloudNoiseSampler,
    RenderSkyPipelineId,
};

#[doc(hidden)]
pub struct AtmospherePlugin;

impl Plugin for AtmospherePlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "types.wesl");
        load_shader_library!(app, "functions.wesl");
        load_shader_library!(app, "bruneton_functions.wesl");
        load_shader_library!(app, "bindings.wesl");
        load_shader_library!(app, "shadows.wesl");
        load_shader_library!(app, "clouds.wesl");

        embedded_asset!(app, "transmittance_lut.wesl");
        embedded_asset!(app, "multiscattering_lut.wesl");
        embedded_asset!(app, "sky_view_lut.wesl");
        embedded_asset!(app, "aerial_view_lut.wesl");
        embedded_asset!(app, "render_sky.wesl");
        embedded_asset!(app, "environment.wesl");
        embedded_asset!(app, "fbm_noise_3d.wesl");
        embedded_asset!(app, "cloud_noise_quantiles.wesl");
        embedded_asset!(app, "weather_window.wesl");
        embedded_asset!(app, "cloud_shadow_mips.wesl");
        embedded_asset!(app, "perlin_worley_noise_3d.wesl");
        embedded_asset!(app, "detail_noise_3d.wesl");
        embedded_asset!(app, "noise_mip_downsample_3d.wesl");
        embedded_asset!(app, "cloud_shadow_map.wesl");
        embedded_asset!(app, "cloud_field.wesl");
        embedded_asset!(app, "cloud_shadow_filter.wesl");
        embedded_asset!(app, "cloud_shadow_temporal.wesl");
        embedded_asset!(app, "clouds_raymarch.wesl");
        embedded_asset!(app, "clouds_reconstruct.wesl");

        app.add_plugins((
            ExtractComponentPlugin::<AtmosphereEnvironmentMap>::default(),
            SyncComponentPlugin::<AtmosphereSettings>::default(),
            UniformComponentPlugin::<GpuAtmosphere>::default(),
            UniformComponentPlugin::<GpuAtmosphereSettings>::default(),
            UniformComponentPlugin::<GpuCloudLayers>::default(),
        ))
        .add_observer(on_insert_atmosphere_environment_map_light)
        .add_observer(on_remove_atmosphere_environment_map_light)
        .init_resource::<CloudTimeScale>()
        .add_systems(Update, (advance_cloud_wind, apply_volumetric_cloud_settings));

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<weather::ExtractedCloudInfluences>()
                .add_systems(ExtractSchedule, (extract_atmosphere, weather::extract_cloud_influences));
        }
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        let render_adapter = render_app.world().resource::<RenderAdapter>();

        if !render_adapter
            .get_downlevel_capabilities()
            .flags
            .contains(DownlevelFlags::COMPUTE_SHADERS)
        {
            warn!("AtmospherePlugin not loaded. GPU lacks support for compute shaders.");
            return;
        }

        if !render_adapter
            .get_texture_format_features(TextureFormat::Rgba16Float)
            .allowed_usages
            .contains(TextureUsages::STORAGE_BINDING)
        {
            warn!("AtmospherePlugin not loaded. GPU lacks support: TextureFormat::Rgba16Float does not support TextureUsages::STORAGE_BINDING.");
            return;
        }

        // Check the `RenderDevice` in addition to the `RenderAdapter`. The
        // former takes the `WGPU_SETTINGS_PRIO` environment variable into
        // account, and the latter doesn't.
        let render_device = render_app.world().resource::<RenderDevice>();
        if render_device.limits().max_storage_textures_per_shader_stage == 0 {
            warn!("AtmospherePlugin not loaded. GPU lacks support: `max_storage_textures_per_shader_stage` is 0");
            return;
        }

        render_app
            .insert_resource(AtmosphereBindGroupLayouts::new())
            .init_resource::<RenderSkyBindGroupLayouts>()
            .init_resource::<CloudNoiseSampler>()
            .init_resource::<environment::AtmosphereProbeSource>()
            .init_resource::<fbm_noise::CloudNoiseFormats>()
            .init_resource::<AtmosphereLutPipelines>()
            .init_resource::<node::PerViewLutState>()
            .init_resource::<resources::CloudHistoryWrites>()
            .init_resource::<AtmosphereTransforms>()
            .init_resource::<SpecializedRenderPipelines<RenderSkyBindGroupLayouts>>()
            .init_resource::<FbmNoiseBindGroupLayout>()
            .init_resource::<FbmNoisePipeline>()
            .init_resource::<NoiseGenerated>()
            .init_resource::<fbm_noise::PendingCloudProfiles>()
            .init_resource::<fbm_noise::CloudNoiseQuantilePipeline>()
            .init_resource::<PerlinWorleyNoiseBindGroupLayout>()
            .init_resource::<PerlinWorleyNoisePipeline>()
            .init_resource::<PerlinWorleyNoiseGenerated>()
            .init_resource::<NoiseMipDownsampleBindGroupLayout>()
            .init_resource::<NoiseMipDownsamplePipeline>()
            .init_resource::<DetailNoiseBindGroupLayout>()
            .init_resource::<DetailNoisePipeline>()
            .init_resource::<DetailNoiseGenerated>()
            // Registered here, after the capability checks: it needs `NoiseGenerated`, which only
            // exists when the plugin loads (registered in `build`, a GPU without compute support
            // panicked on the missing resource instead of falling back).
            .add_systems(
                ExtractSchedule,
                (extract_cloud_weather_regen, fbm_noise::extract_cloud_profiles),
            )
            .add_systems(
                RenderStartup,
                (
                    init_atmosphere_probe_layout,
                    init_atmosphere_probe_pipeline,
                    init_fbm_noise_texture,
                    init_fbm_noise_params_buffer,
                    init_perlin_worley_noise_texture,
                    init_perlin_worley_noise_params_buffer,
                    init_detail_noise_texture,
                    init_detail_noise_params_buffer,
                )
                    .chain(),
            )
            .add_systems(
                Render,
                (
                    configure_camera_depth_usages.in_set(RenderSystems::PrepareViews),
                    queue_render_sky_pipelines.in_set(RenderSystems::Queue),
                    prepare_atmosphere_textures.in_set(RenderSystems::PrepareResources),
                    prepare_probe_textures
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_atmosphere_textures),
                    prepare_atmosphere_uniforms
                        .in_set(RenderSystems::Prepare)
                        .before(RenderSystems::PrepareResources),
                    prepare_atmosphere_probe_sources
                        .in_set(RenderSystems::Prepare)
                        .before(RenderSystems::PrepareResources),
                    prepare_atmosphere_probe_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                    prepare_atmosphere_transforms.in_set(RenderSystems::PrepareResources),
                    prepare_cloud_shadow_temporal_params
                        .in_set(RenderSystems::PrepareBindGroups)
                        .before(prepare_atmosphere_bind_groups),
                    (prepare_clouds_temporal_params, resources::prepare_cloud_reference_accumulators)
                        .in_set(RenderSystems::PrepareBindGroups)
                        .before(prepare_atmosphere_bind_groups),
                    prepare_atmosphere_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                    prepare_fbm_noise_bind_group.in_set(RenderSystems::PrepareBindGroups),
                    prepare_perlin_worley_noise_bind_group.in_set(RenderSystems::PrepareBindGroups),
                    prepare_detail_noise_bind_group.in_set(RenderSystems::PrepareBindGroups),
                    generate_fbm_noise_once
                        .in_set(RenderSystems::Render)
                        .after(RenderSystems::PrepareBindGroups),
                    generate_perlin_worley_noise_once
                        .in_set(RenderSystems::Render)
                        .after(RenderSystems::PrepareBindGroups),
                    fbm_noise::compute_cloud_noise_quantiles
                        .in_set(RenderSystems::Render)
                        .after(generate_perlin_worley_noise_once),
                    generate_detail_noise_once
                        .in_set(RenderSystems::Render)
                        .after(RenderSystems::PrepareBindGroups),
                    prepare_atmosphere_buffers.in_set(RenderSystems::PrepareResources),
                    fbm_noise::upload_cloud_profiles.in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_systems(
                Core3d,
                (
                    (atmosphere_luts, atmosphere_environment)
                        .chain()
                        .after(Core3dSystems::Prepass)
                        .before(Core3dSystems::MainPass),
                    render_sky
                        .after(main_opaque_pass_3d)
                        .before(main_transparent_pass_3d),
                ),
            );
    }
}

/// For each camera with [`AtmosphereSettings`], picks the nearest [`Atmosphere`] by world-space
/// distance to its origin, copies it as [`ExtractedAtmosphere`], combines its cloud layers (layer
/// components on the atmosphere entity and on its [`CloudLayers`]) into [`GpuCloudLayers`], and
/// builds [`GpuAtmosphereSettings`].
pub fn extract_atmosphere(
    mut commands: Commands,
    mut shadow_anchors: bevy_ecs::system::Local<bevy_platform::collections::HashMap<Entity, Vec3>>,
    atmosphere_entities: Extract<
        Query<(
            Entity,
            &Atmosphere,
            &GlobalTransform,
            Option<&AtmosphereWind>,
            Option<&CloudLayers>,
        )>,
    >,
    layer_entities: Extract<
        Query<(
            Option<&VolumetricCloudLayer>,
            Option<&ThinCloudLayer>,
            Option<&CloudProfiles>,
        )>,
    >,
    cameras: Extract<
        Query<(RenderEntity, Option<&AtmosphereSettings>, &GlobalTransform), With<Camera3d>>,
    >,
) {
    let candidates: Vec<_> = atmosphere_entities.iter().collect();
    // Shadow-map anchors of cameras that no longer exist.
    shadow_anchors.retain(|entity, _| cameras.iter().any(|(e, ..)| e == *entity));

    for (render_entity, settings, cam_global) in &cameras {
        // Remove any stale render-world state when AtmosphereSettings is removed or candidates is empty
        let (Some(settings), false) = (settings, candidates.is_empty()) else {
            commands.entity(render_entity).try_remove::<(
                ExtractedAtmosphere,
                GpuAtmosphereSettings,
                GpuAtmosphere,
                AtmosphereTextures,
                AtmosphereTransformsOffset,
                RenderSkyPipelineId,
                AtmosphereBindGroups,
                GpuCloudLayers,
            )>();
            continue;
        };

        let cam_world = cam_global.translation();
        let selected = candidates
            .iter()
            .min_by(|(ea, _, gt_a, _, _), (eb, _, gt_b, _, _)| {
                let da = cam_world.distance(gt_a.translation());
                let db = cam_world.distance(gt_b.translation());
                da.total_cmp(&db).then_with(|| ea.cmp(eb))
            })
            .expect("checked non-empty above");
        let atmo = selected.1;
        let gt = selected.2;
        let wind = selected.3;
        // The atmosphere entity's own layer components first, then its related layers.
        let mut volumetric: Option<(&VolumetricCloudLayer, Option<&CloudProfiles>)> = None;
        let mut thin_layers: Vec<&ThinCloudLayer> = Vec::new();
        let related = selected.4.into_iter().flat_map(|l| l.0.iter().copied());
        for entity in core::iter::once(selected.0).chain(related) {
            let Ok((vol, thin, profiles)) = layer_entities.get(entity) else {
                continue;
            };
            if let Some(vol) = vol {
                if volumetric.is_some() {
                    bevy_log::warn_once!(
                        "an atmosphere has several VolumetricCloudLayers; only the first is rendered"
                    );
                } else {
                    volumetric = Some((vol, profiles));
                }
            }
            thin_layers.extend(thin);
        }
        let cloud_layer = volumetric.map(|(v, _)| v);
        let profiles = volumetric.and_then(|(_, p)| p);
        // Atmospheres are spherically symmetric, so ignore their orientation.
        let (scale, _, translation) = gt.to_scale_rotation_translation();

        let atmosphere_to_world =
            Mat4::from_scale_rotation_translation(scale, Quat::IDENTITY, translation);
        let extracted = ExtractedAtmosphere {
            inner_radius: atmo.inner_radius,
            outer_radius: atmo.outer_radius,
            ground_albedo: atmo.ground_albedo,
            medium: atmo.medium.id(),
            world_to_atmosphere: atmosphere_to_world.inverse(),
            atmosphere_to_world,
        };
        let extracted_world_to_atmosphere = extracted.world_to_atmosphere;
        commands.entity(render_entity).insert(extracted);
        let mut gpu_settings = GpuAtmosphereSettings::from(settings.clone());
        let has_layers = cloud_layer.is_some() || !thin_layers.is_empty();
        if has_layers {
            let mut layers =
                GpuCloudLayers::new(cloud_layer, &thin_layers, wind, profiles, atmo.inner_radius);
            layers.weather_window_origin = weather_window_origin(
                extracted_world_to_atmosphere.transform_point3(cam_world),
                &layers,
            );
            gpu_settings.cloud_shadow_anchor_max_radius = layers.cloud_layer_end;
            // The thin layers' shadow (the map's alpha) applies below the lowest mid level.
            gpu_settings.cloud_thin_shadow_max_radius =
                layers.lowest_thin().map_or(0.0, GpuThinCloudLayer::mid_radius);
            commands.entity(render_entity).insert(layers);
        }
        // The map's centre: re-snapped after the camera moves 1/16 of the map's half extent
        // (4 km at 64 km), which keeps it within the warped map's fine-texel centre.
        let camera_as = extracted_world_to_atmosphere.transform_point3(cam_world);
        let anchor = cloud_shadow_anchor(
            camera_as,
            atmo.inner_radius,
            gpu_settings.cloud_shadow_anchor_max_radius,
            0.0625 * gpu_settings.cloud_shadow_map_extent,
            shadow_anchors.get(&render_entity).copied(),
        );
        shadow_anchors.insert(render_entity, anchor);
        gpu_settings.cloud_shadow_anchor = anchor;
        commands.entity(render_entity).insert(gpu_settings);
        if !has_layers {
            commands.entity(render_entity).remove::<GpuCloudLayers>();
        }
    }
}

/// Weather-map texel at the weather window's texel 0 for a camera at `camera` (atmosphere space):
/// the camera in the map's frame (advected by the layer's offset, rotated 30 degrees as
/// `clouds.wesl` sample_weather), in texels of the 512^2 map, minus half the window, floored.
fn weather_window_origin(camera: Vec3, layers: &GpuCloudLayers) -> Vec2 {
    let q = Vec2::new(camera.x - layers.noise_offset.x, camera.z - layers.noise_offset.z);
    let rotated = Vec2::new(0.866_025_4 * q.x + 0.5 * q.y, -0.5 * q.x + 0.866_025_4 * q.y);
    let texels = fbm_noise::NoiseTextureSize::default().size.as_vec2();
    let uv = rotated / layers.noise_scale.max(1.0);
    (uv * texels).floor() - Vec2::splat((resources::WEATHER_WINDOW_SIZE / 2) as f32)
}

/// The render-world representation of an `Atmosphere`, but which
/// hasn't been converted into shader uniforms yet.
#[derive(Clone, Component)]
pub struct ExtractedAtmosphere {
    pub inner_radius: f32,
    pub outer_radius: f32,
    pub ground_albedo: Vec3,
    pub medium: AssetId<ScatteringMedium>,
    pub world_to_atmosphere: Mat4,
    pub atmosphere_to_world: Mat4,
}

/// This component controls the resolution of the atmosphere LUTs, and
/// how many samples are used when computing them.
///
/// The transmittance LUT stores the optical depth from a point in the
/// atmosphere to the outer edge of the atmosphere in any direction,
/// parametrized by the point's radius and the cosine of the zenith angle
/// of the ray.
///
/// The multiscattering LUT stores the log of the factor representing luminance
/// scattered towards the camera with scattering order >2, parametrized by the
/// point's radius and the cosine of the zenith angle of the sun.
///
/// The sky-view lut is essentially the actual skybox, storing the light scattered
/// towards the camera in every direction with a cubemap.
///
/// The aerial-view lut is a 3d LUT fit to the view frustum, which stores the log of
/// the luminance scattered towards the camera at each point.
///
/// The sky-view and aerial-view LUTs hold luminance pre-exposed by the view's
/// exposure, so twilight and night skies stay in f16's normal range.
#[derive(Clone, Component, Reflect)]
#[reflect(Clone, Default)]
#[require(Hdr)]
pub struct AtmosphereSettings {
    /// The size of the transmittance LUT
    pub transmittance_lut_size: UVec2,

    /// The size of the multiscattering LUT
    pub multiscattering_lut_size: UVec2,

    /// The size of the sky-view LUT.
    pub sky_view_lut_size: UVec2,

    /// The size of the aerial-view LUT.
    pub aerial_view_lut_size: UVec3,

    /// The number of points to sample along each ray when
    /// computing the transmittance LUT
    pub transmittance_lut_samples: u32,

    /// The number of rays to sample when computing each
    /// pixel of the multiscattering LUT
    pub multiscattering_lut_dirs: u32,

    /// The number of points to sample when integrating along each
    /// multiscattering ray
    pub multiscattering_lut_samples: u32,

    /// The number of points to sample along each ray when
    /// computing the sky-view LUT.
    pub sky_view_lut_samples: u32,

    /// The number of points to sample for each slice along the z-axis
    /// of the aerial-view LUT.
    pub aerial_view_lut_samples: u32,

    /// The maximum distance from the camera to evaluate the
    /// aerial view LUT. The slices along the z-axis of the
    /// texture will be distributed linearly from the camera
    /// to this value.
    ///
    /// units: m
    pub aerial_view_lut_max_distance: f32,

    /// The number of points to sample for each fragment when the using
    /// ray marching to render the sky
    pub sky_max_samples: u32,

    /// The rendering method to use for the atmosphere.
    pub rendering_method: AtmosphereMode,

    /// Whether to sample shadow maps for volumetric shadow shafts.
    pub shadows_enabled: bool,
    /// Resolution of the cloud shadow map used to shadow directional lights when
    /// rendering in [`AtmosphereMode::Raymarched`].
    ///
    /// Higher resolution improves stability and reduces aliasing at grazing angles,
    /// but increases compute cost.
    pub cloud_shadow_map_size: UVec2,

    /// Half-extent of the orthographic cloud shadow map volume in meters.
    ///
    /// The shadow map covers a square region of size \(2 \cdot extent\) around the view
    /// position, oriented in light space.
    pub cloud_shadow_map_extent: f32,

    /// Number of samples used when building the cloud shadow map.
    pub cloud_shadow_map_samples: u32,

    /// Strength multiplier applied to the cloud shadow map's stored extinction / optical depth.
    pub cloud_shadow_map_strength: f32,

    /// Number of spatial filter iterations applied to the cloud shadow map after tracing.
    ///
    /// This is the main knob to trade performance for stability once the shadow map uses jittered sampling.
    /// Values above 4 are usually not worth it.
    pub cloud_shadow_map_spatial_filter_iterations: u32,

    /// Time-slices the cloud shadow map tracing: each frame only one texel out of every
    /// `N x N` tile is re-traced (in a Bayer order), and the remaining texels are reprojected
    /// from the previous frame's map (see Bauer 2019, "Creating the Atmospheric World of RDR2":
    /// the cloud shadow map updates 1 ray per 4x4 tile per frame).
    ///
    /// `1` traces every texel every frame. Values above `1` cut the tracing cost by roughly
    /// `N * N` and implicitly enable the temporal reprojection pass (blending of the traced texels
    /// with history is still controlled by [`Self::cloud_shadow_temporal_enabled`]). The spatial
    /// filter is skipped while time-slicing, since it would repeatedly blur the reprojected texels.
    /// Supported values: `1`, `2` and `4`.
    pub cloud_shadow_map_update_interleave: u32,

    /// Enable temporal filtering for the cloud shadow map to reduce edge flickering.
    ///
    /// When enabled, the shadow map is blended with the previous frame's result,
    /// stabilizing edges over time. Based on Unreal's approach.
    pub cloud_shadow_temporal_enabled: bool,

    /// Weight of the current frame in the temporal blend (0.0–1.0).
    ///
    /// Lower values (e.g. 0.1–0.2) produce more stable shadows but slower convergence.
    /// Higher values (e.g. 0.5) converge faster but may retain more flicker.
    pub cloud_shadow_temporal_alpha: f32,

    /// Light rotation threshold (degrees) beyond which history is invalidated.
    ///
    /// When the sun moves more than this angle, temporal accumulation is reset
    /// to avoid ghosting. Unreal uses 10°.
    pub cloud_shadow_temporal_light_rotation_cut_deg: f32,

    /// Minimum step budget of the cloud march per ray (set from
    /// [`VolumetricCloudSettings::max_steps`]). Rays inside the cloud layer get
    /// `1024 / 96` times as many.
    pub cloud_max_steps: u32,

    /// Light taps toward the sun per cloud sample (geometric spacing from 20 m, growth 1.4;
    /// set from [`VolumetricCloudSettings::light_taps`]).
    pub cloud_light_taps: u32,

    /// Farthest distance the cloud march covers, in meters (set from
    /// [`VolumetricCloudSettings::max_distance`]).
    pub cloud_max_distance: f32,

    /// Density samples per long in-cloud view step (set from
    /// [`VolumetricCloudSettings::step_samples`]).
    pub cloud_step_samples: u32,

    /// Largest internal size of the cloud and haze buffers, in pixels (set from
    /// [`VolumetricCloudSettings::max_render_size`]).
    pub cloud_max_render_size: UVec2,
}

impl Default for AtmosphereSettings {
    fn default() -> Self {
        Self {
            transmittance_lut_size: UVec2::new(256, 128),
            transmittance_lut_samples: 64,
            // Wider along the sun angle, where the factor falls fastest near the horizon.
            multiscattering_lut_size: UVec2::new(64, 32),
            multiscattering_lut_dirs: 64,
            multiscattering_lut_samples: 20,
            sky_view_lut_size: UVec2::new(400, 200),
            sky_view_lut_samples: 16,
            aerial_view_lut_size: UVec3::new(32, 32, 32),
            aerial_view_lut_samples: 10,
            aerial_view_lut_max_distance: 3.2e4,
            sky_max_samples: 16,
            rendering_method: AtmosphereMode::LookupTexture,
            shadows_enabled: true,

            // Cloud shadow map defaults (only used by Raymarched mode).
            cloud_shadow_map_size: UVec2::new(512, 512),
            cloud_shadow_map_extent: 64_000.0,
            cloud_shadow_map_samples: 24,
            cloud_shadow_map_strength: 1.0,
            cloud_shadow_map_spatial_filter_iterations: 0,
            cloud_shadow_map_update_interleave: 4,
            cloud_shadow_temporal_enabled: false,
            cloud_shadow_temporal_alpha: 0.15,
            cloud_shadow_temporal_light_rotation_cut_deg: 10.0,
            cloud_max_steps: VolumetricCloudSettings::MEDIUM.max_steps,
            cloud_light_taps: VolumetricCloudSettings::MEDIUM.light_taps,
            cloud_max_distance: VolumetricCloudSettings::MEDIUM.max_distance,
            cloud_step_samples: VolumetricCloudSettings::MEDIUM.step_samples,
            cloud_max_render_size: VolumetricCloudSettings::MEDIUM.max_render_size,
        }
    }
}

#[derive(Clone, Component, Reflect, ShaderType)]
#[reflect(Default)]
pub struct GpuAtmosphereSettings {
    pub transmittance_lut_size: UVec2,
    pub multiscattering_lut_size: UVec2,
    pub sky_view_lut_size: UVec2,
    pub aerial_view_lut_size: UVec3,
    pub transmittance_lut_samples: u32,
    pub multiscattering_lut_dirs: u32,
    pub multiscattering_lut_samples: u32,
    pub sky_view_lut_samples: u32,
    pub aerial_view_lut_samples: u32,
    pub aerial_view_lut_max_distance: f32,
    pub sky_max_samples: u32,
    pub rendering_method: u32,
    pub shadows_enabled: u32,
    pub cloud_shadow_map_size: UVec2,
    pub cloud_shadow_map_extent: f32,
    pub cloud_shadow_map_samples: u32,
    pub cloud_shadow_map_strength: f32,
    pub cloud_shadow_map_spatial_filter_iterations: u32,
    pub cloud_shadow_map_update_interleave: u32,
    pub cloud_shadow_temporal_enabled: u32,
    pub cloud_shadow_temporal_alpha: f32,
    pub cloud_shadow_temporal_light_rotation_cut_deg: f32,
    /// Radius (from the planet centre) the cloud shadow map's anchor is lowered to when the
    /// camera is higher: the cumulus tops. Set at extraction from [`GpuCloudLayers`]; shared by
    /// the map's producer, the CPU reprojection and every consumer (atmosphere and PBR).
    pub cloud_shadow_anchor_max_radius: f32,
    /// Radius below which the thin layer's shadow (the map's alpha) applies: the cirrus mid
    /// level.
    pub cloud_thin_shadow_max_radius: f32,
    pub cloud_max_steps: u32,
    pub cloud_light_taps: u32,
    pub cloud_max_distance: f32,
    pub cloud_step_samples: u32,
    pub cloud_max_render_size: UVec2,
    /// Centre of the cloud shadow map (atmosphere space): [`cloud_shadow_anchor`], set at
    /// extraction and read by the map's producer, the CPU reprojection and every consumer.
    pub cloud_shadow_anchor: Vec3,
}

/// Centre of the cloud shadow map with hysteresis: the camera (atmosphere space, kept above the
/// surface), lowered to `max_radius` (the cumulus tops) when higher, held where it was until the
/// camera has moved `snap` (m) from it. Independent of the light: turning the sun rotates the
/// map about a fixed centre. Rounding the camera's coordinates in the light's basis (UE5's
/// stabilised anchor) re-snapped on every frame of a turning light: the planet-centred position
/// sweeps through the light-plane grid (1 degree moves it ~110 km at Earth's radius), so the
/// map's centre jumped by up to a snap step each frame and its history was re-mapped with it.
/// The map stays fixed in the world while the camera flies within `snap` of it, as before.
pub fn cloud_shadow_anchor(
    camera: Vec3,
    inner_radius: f32,
    max_radius: f32,
    snap: f32,
    previous: Option<Vec3>,
) -> Vec3 {
    let r = camera.length();
    let surface = inner_radius + 1.0;
    let origin = camera.normalize_or_zero() * r.max(surface).min(max_radius.max(surface));
    match previous {
        Some(anchor) if anchor.distance(origin) < snap => anchor,
        _ => origin,
    }
}

impl GpuAtmosphereSettings {
    /// Whether the cloud shadow map goes through the temporal reprojection pass, either because
    /// temporal filtering is enabled or because the tracing is time-sliced.
    pub fn cloud_shadow_uses_temporal_pass(&self) -> bool {
        self.cloud_shadow_temporal_enabled != 0 || self.cloud_shadow_map_update_interleave > 1
    }
}

impl Default for GpuAtmosphereSettings {
    fn default() -> Self {
        AtmosphereSettings::default().into()
    }
}

impl From<AtmosphereSettings> for GpuAtmosphereSettings {
    fn from(s: AtmosphereSettings) -> Self {
        Self {
            transmittance_lut_size: s.transmittance_lut_size,
            multiscattering_lut_size: s.multiscattering_lut_size,
            sky_view_lut_size: s.sky_view_lut_size,
            aerial_view_lut_size: s.aerial_view_lut_size,
            transmittance_lut_samples: s.transmittance_lut_samples,
            multiscattering_lut_dirs: s.multiscattering_lut_dirs,
            multiscattering_lut_samples: s.multiscattering_lut_samples,
            sky_view_lut_samples: s.sky_view_lut_samples,
            aerial_view_lut_samples: s.aerial_view_lut_samples,
            aerial_view_lut_max_distance: s.aerial_view_lut_max_distance,
            sky_max_samples: s.sky_max_samples,
            rendering_method: s.rendering_method as u32,
            shadows_enabled: s.shadows_enabled as u32,
            cloud_shadow_map_size: s.cloud_shadow_map_size,
            cloud_shadow_map_extent: s.cloud_shadow_map_extent,
            cloud_shadow_map_samples: s.cloud_shadow_map_samples,
            cloud_shadow_map_strength: s.cloud_shadow_map_strength,
            cloud_shadow_map_spatial_filter_iterations: s
                .cloud_shadow_map_spatial_filter_iterations,
            cloud_shadow_map_update_interleave: match s.cloud_shadow_map_update_interleave {
                0 | 1 => 1,
                2 | 3 => 2,
                _ => 4,
            },
            cloud_shadow_temporal_enabled: s.cloud_shadow_temporal_enabled as u32,
            cloud_shadow_temporal_alpha: s.cloud_shadow_temporal_alpha,
            cloud_shadow_temporal_light_rotation_cut_deg: s
                .cloud_shadow_temporal_light_rotation_cut_deg,
            // No cloud layers: no lowering, no thin shadow (set at extraction otherwise).
            cloud_shadow_anchor_max_radius: f32::MAX,
            cloud_thin_shadow_max_radius: 0.0,
            cloud_max_steps: s.cloud_max_steps,
            cloud_light_taps: s.cloud_light_taps,
            cloud_max_distance: s.cloud_max_distance,
            cloud_step_samples: s.cloud_step_samples.max(1),
            cloud_max_render_size: s.cloud_max_render_size.max(UVec2::ONE),
            // Set at extraction (`cloud_shadow_anchor`).
            cloud_shadow_anchor: Vec3::ZERO,
        }
    }
}

impl SyncComponent<RenderApp> for AtmosphereSettings {
    type Target = GpuAtmosphereSettings;
}

fn configure_camera_depth_usages(
    mut cameras: Query<&mut Camera3d, (Changed<Camera3d>, With<ExtractedAtmosphere>)>,
) {
    for mut camera in &mut cameras {
        camera.depth_texture_usages.0 |= TextureUsages::TEXTURE_BINDING.bits();
    }
}

/// Selects how the atmosphere is rendered. Choose based on scene scale and
/// volumetric shadow quality, and based on performance needs.
#[repr(u32)]
#[derive(Clone, Default, Reflect, Copy)]
pub enum AtmosphereMode {
    /// High-performance solution tailored to scenes that are mostly inside of the atmosphere.
    /// Uses a set of lookup textures to approximate scattering integration.
    /// Slightly less accurate for very long-distance/space views (lighting precision
    /// tapers as the camera moves far from the scene origin) and for sharp volumetric
    /// (cloud/fog) shadows.
    #[default]
    LookupTexture = 0,
    /// Slower, more accurate rendering method for any type of scene.
    /// Integrates the scattering numerically with raymarching and produces sharp volumetric
    /// (cloud/fog) shadows.
    /// Best for cinematic shots, planets seen from orbit, and scenes requiring
    /// accurate long-distance lighting.
    Raymarched = 1,
}

/// Procedural / painted weather input for volumetric clouds (design.md §3.2).
///
/// Both modes resolve to the same GPU weather map binding (512² RGBA:
/// coverage, type, density scale, spare). Procedural is the default and is
/// generated once on the GPU at startup; a `Texture` override can be wired
/// later without changing the density sampler.
#[derive(Clone, Component, Reflect)]
#[reflect(Clone, Default)]
pub struct CloudWeather {
    /// World-space extent covered by one weather-map tile, in meters.
    pub scale: f32,
    /// Seed for procedural generation (reserved; current generator is fixed).
    pub seed: u32,
    /// Base cloud type in [0, 1] (0 stratus → 1 cumulus), blended with the map's G.
    pub cloud_type: f32,
    /// How much the weather-map type channel can deviate from `cloud_type`.
    pub type_variation: f32,
    /// UV offset applied when sampling the weather map.
    pub offset: Vec2,
    /// Meters/second advection of the weather UV offset (applied on the CPU later).
    pub offset_velocity: Vec2,
    /// Optional painted weather map (RGBA: coverage, type, density scale, mesoscale cloud
    /// fraction). The cloud life cycle applies to the procedural map only.
    /// When set, the procedural GPU weather map is skipped and this image is bound
    /// instead (design.md §3.2). Changing the handle or clearing it triggers a rebind.
    pub texture: Option<Handle<Image>>,
}

impl Default for CloudWeather {
    fn default() -> Self {
        Self {
            scale: 32_000.0,
            seed: 0,
            cloud_type: 0.5,
            type_variation: 0.5,
            offset: Vec2::ZERO,
            offset_velocity: Vec2::ZERO,
            texture: None,
        }
    }
}

/// Per-camera volumetric cloud quality (design.md §3.3).
///
/// Insert on a camera to override defaults. Absent cameras get [`VolumetricCloudSettings::MEDIUM`].
#[derive(Clone, Component, Reflect)]
#[reflect(Clone, Default)]
pub struct VolumetricCloudSettings {
    /// Minimum step budget of the cloud march per ray (rays inside the cloud layer get
    /// `1024 / 96` times as many).
    pub max_steps: u32,
    /// Light taps toward the sun per cloud sample (geometric spacing from 20 m, growth 1.4:
    /// 8 taps reach ~0.7 km, 12 taps ~2.8 km, 16 ~11 km).
    pub light_taps: u32,
    /// Farthest distance the cloud march covers, in meters.
    pub max_distance: f32,
    /// Density samples, evenly spaced, per in-cloud view step longer than 100 m; 1 = one.
    ///
    /// One density sample per step makes clouds too transparent where the density varies
    /// within the step (the mean of `exp(-sigma dt)` over the jitter exceeds `exp` of the mean
    /// optical depth). Against a 16x finer march (hero view, sun 35 deg, Apple M5), the
    /// cloudy-pixel transmittance error is +0.078 with 1 sample, +0.037 with 2 (+20 % march
    /// time) and +0.018 with 4 (+57 %); half and quarter steps give +0.032 (+37 %) and +0.012
    /// (+105 %). Lighting stays one evaluation per step.
    pub step_samples: u32,
    /// Cloud shadow map update interleave (1 = full, 4 = 1/16 texels/frame).
    pub shadow_update_interleave: u32,
    /// Enable temporal accumulation on the cloud shadow map.
    pub shadow_temporal: bool,
    /// Largest internal size of the cloud and haze buffers, in pixels. They are half the
    /// viewport up to this size (same aspect ratio) and upscaled beyond it, so their cost stops
    /// growing with the window: at half resolution a 2560x1440 view cost 3x a 1280x720 one
    /// (clouds 25 vs 8 ms, haze 17 vs 5 ms, Apple M5), and the lower frame rate meant more
    /// camera motion per frame and more visible temporal flicker.
    pub max_render_size: UVec2,
    /// Temporal blend alpha for the cloud shadow map when temporal is enabled.
    pub shadow_temporal_alpha: f32,
}

impl VolumetricCloudSettings {
    pub const LOW: Self = Self {
        max_steps: 48,
        light_taps: 8,
        max_distance: 200_000.0,
        step_samples: 1,
        shadow_update_interleave: 4,
        shadow_temporal: true,
        shadow_temporal_alpha: 0.15,
        max_render_size: UVec2::new(640, 360),
    };
    pub const MEDIUM: Self = Self {
        max_steps: 96,
        light_taps: 12,
        max_distance: 350_000.0,
        step_samples: 1,
        shadow_update_interleave: 4,
        shadow_temporal: true,
        shadow_temporal_alpha: 0.1,
        max_render_size: UVec2::new(960, 540),
    };
    pub const HIGH: Self = Self {
        max_steps: 128,
        light_taps: 16,
        max_distance: 400_000.0,
        step_samples: 2,
        shadow_update_interleave: 2,
        shadow_temporal: true,
        shadow_temporal_alpha: 0.08,
        max_render_size: UVec2::new(1280, 720),
    };
    pub const CINEMATIC: Self = Self {
        max_steps: 160,
        light_taps: 20,
        max_distance: 500_000.0,
        step_samples: 4,
        shadow_update_interleave: 1,
        shadow_temporal: true,
        shadow_temporal_alpha: 0.05,
        max_render_size: UVec2::new(1280, 720),
    };
}

impl Default for VolumetricCloudSettings {
    fn default() -> Self {
        Self::MEDIUM
    }
}

/// Component that adds a volumetric cloud layer to the atmosphere.
///
/// Add this component to an [`Atmosphere`] entity to enable volumetric clouds for cameras
/// that select that atmosphere.
#[derive(Clone, Component, Reflect)]
#[reflect(Clone, Default)]
#[require(CloudWeather, CloudProfiles)]
pub struct VolumetricCloudLayer {
    /// Altitude at which the cloud layer starts (from planet center)
    /// units: m
    ///
    /// With [`CloudProfiles::Physical`] (the default) the layer is the profiles' own range
    /// instead ([`physical_profile_bounds`]): each air mass has its cloud base, so the bounds
    /// follow the profiles. The other profiles use these bounds.
    pub cloud_layer_start: f32,

    /// Altitude at which the cloud layer ends (from planet center)
    /// units: m (see [`VolumetricCloudLayer::cloud_layer_start`])
    pub cloud_layer_end: f32,

    /// Density multiplier for the clouds
    pub cloud_density: f32,

    /// Absorption coefficient for clouds at density 1. Use [`VolumetricCloudLayer::with_liquid_water`]
    /// to derive it from the droplet microphysics.
    /// units: m^-1
    pub cloud_absorption: f32,

    /// Scattering coefficient for clouds at density 1. Use [`VolumetricCloudLayer::with_liquid_water`]
    /// to derive it from the droplet microphysics.
    /// units: m^-1
    pub cloud_scattering: f32,

    /// Liquid water content at density 1, the reference of [`VolumetricCloudLayer::cloud_scattering`]
    /// and [`VolumetricCloudLayer::cloud_absorption`]: the in-cloud liquid water of the
    /// [`CloudProfiles`] (kg/m^3) divided by it is the density, so extinction is
    /// `mass coefficient * liquid water`. Set by [`VolumetricCloudLayer::with_liquid_water`].
    /// units: kg/m^3
    pub liquid_water: f32,

    /// World-space size (m) of one weather-map tile.
    pub noise_scale: f32,

    /// World-space period (m, isotropic) of the 128³ shape noise, which sets the size of the
    /// individual clouds. The default 6.4 km puts the cover-weighted cloud size at ~600-760 m at
    /// 5-10 % cloud fraction with the physical profiles, the cumulus scale where cloud cover
    /// peaks in LES (BOMEX 700 m; 400-1250 m across cases; Neggers et al. 2003, J. Atmos. Sci.
    /// 60, Table 2), measured on the noise in `.clouds-handoff/noise_study`; size scales
    /// linearly with the period.
    pub shape_noise_scale: f32,

    /// Accumulated displacement of the cloud field (m): xz by [`VolumetricCloudLayer::wind`] (advection;
    /// the density at `p` is the pattern at `p - noise_offset`), y by
    /// [`VolumetricCloudLayer::updraft_speed`] (the noise rises through the fixed vertical envelope:
    /// thermals rise and turrets form at the base and dissolve at the top). Advanced each frame by
    /// [`advance_cloud_wind`].
    pub noise_offset: Vec3,

    /// Tile size of the Nubis Cubed detail noise. 600 m puts the low-frequency billow cells (tile / 3)
    /// at ~200 m, the scale of cumulus turrets (thermal bubbles ~100-500 m); Nubis's own 100 m tile
    /// is for close-up voxel clouds and left large sky clouds without mid-scale breakup.
    /// units: m
    pub detail_noise_scale: f32,

    /// Strength of the detail noise modulation in [0,1].
    /// 0 = disabled, 1 = full modulation.
    pub detail_strength: f32,

    /// Scale of the weather map's coverage (its R channel, after the mesoscale clearing), clamped
    /// to 1: with the physical profiles, coverage times the vertical profile is the cloud
    /// fraction, so this scales the cloud fraction everywhere. 1 = the map as authored.
    pub coverage: f32,

    /// Override of the wind at the cloud base (m/s, horizontal), which advects the cloud field.
    /// `None`: [`AtmosphereWind`] at the base altitude.
    pub wind: Option<Vec3>,

    /// With [`CloudProfiles::Envelope`]: cloud-top height (fraction of the layer) of the thinnest
    /// cells; the top rises linearly with weather coverage to the layer top. The physical profiles
    /// take the column top from the distribution of parcel tops instead.
    pub profile_top_min: f32,

    /// Override of the vertical wind shear dU/dz above the cloud base (1/s, horizontal; 1e-3 =
    /// 1 m/s per km). `None`: the mean shear of [`AtmosphereWind`] over the layer. Clouds lean
    /// downwind like the trajectory of their updraft through the sheared flow,
    /// x(z) = shear int_0^z z' / w(z') dz' at height z above the base: with
    /// [`CloudProfiles::Physical`] w(z) is the cloudy updraft of each air mass (baked with the
    /// profiles: marine cumulus lean more than continental ones), otherwise the constant
    /// [`VolumetricCloudLayer::updraft_speed`], x = shear z^2 / (2 w). The lean is static (a cloud
    /// is continuously renewed by new thermals), not accumulated over time. Across air masses the
    /// lean differs, and the leaned field folds where it changes faster than 1 m per m
    /// horizontally: the procedural weather stays below that up to ~2x the default shear
    /// (~3.6e-3 /s); stronger shear tears real cumulus apart, which a displaced field cannot show.
    pub wind_shear: Option<Vec3>,
    /// Cumulus updraft speed (m/s): the rise of the internal structure (the noise moves up
    /// through the profile), and the shear lean of the Nubis, envelope and image profiles.
    /// Fair-weather cumulus: ~1-5 m/s (textbook range, e.g. Stull, Practical Meteorology).
    pub updraft_speed: f32,
    /// Life time of a cloud (s), 0 = static weather: the procedural weather map cross-fades
    /// between independent coverage patterns over this period, so each cloud grows (wider and
    /// taller, as the top follows coverage) and decays. Small cumulus live ~10-30 min (textbook
    /// range, e.g. Houze, Cloud Dynamics).
    ///
    /// The weather map is shared by all views: with several cameras or atmospheres, the life
    /// cycle of the lowest-order camera whose cumulus layer has one drives everyone's weather.
    pub lifetime: f32,
    /// Accumulated cloud time (s), advanced by [`advance_cloud_wind`]; drives the life cycle.
    pub evolution_time: f32,

}

impl VolumetricCloudLayer {
    /// Peak cumulus liquid water content, i.e. the LWC at density 1 (the peak of the adiabatic x
    /// entrainment vertical profile, ~mid-height). The saturated-adiabatic liquid water gradient
    /// is ~2 g/m^3 per km for a ~10 C cloud base, so ~1 km above the base the adiabatic LWC is ~2
    /// g/m^3; cumulus cores keep ~half of it after entrainment (Warner 1955), so ~1 g/m^3.
    /// units: kg/m^3
    pub const CUMULUS_LIQUID_WATER: f32 = 1.0e-3;
    /// Typical cumulus droplet effective radius, matching the droplet Mie phase the clouds
    /// use (Jendersie & d'Eon 2023 fit at diameter 20 um).
    /// units: m
    pub const CUMULUS_EFFECTIVE_RADIUS: f32 = 10.0e-6;

    /// Sets the scattering and absorption coefficients from droplet microphysics: the liquid
    /// water content at density 1 (kg/m^3) and the droplet effective radius (m).
    ///
    /// For a droplet size distribution, `sigma = (3/4) <Q> LWC / (rho_water r_e)`.
    /// The size-averaged Mie efficiencies come from exact Mie at 550 nm for a gamma distribution
    /// (Hansen 1971, effective variance 0.1) with the water index of Hale & Querry 1973
    /// (`validation/mie/extinction.py`). `<Q_ext>` = 2.09 (2.08-2.11 for r_e 8-12 um,
    /// 450-650 nm), and `<Q_abs>` = 8.7e-7 at r_e 10 um, which scales ~linearly with r_e
    /// (weak absorption). The resulting single-scattering albedo is 0.9999996.
    pub fn with_liquid_water(mut self, liquid_water_content: f32, effective_radius: f32) -> Self {
        const RHO_WATER: f32 = 1000.0;
        const Q_EXT: f32 = 2.09;
        const Q_ABS_PER_M: f32 = 8.7e-7 / 10.0e-6;
        let geometric = 0.75 * liquid_water_content / (RHO_WATER * effective_radius);
        let q_abs = Q_ABS_PER_M * effective_radius;
        self.cloud_scattering = (Q_EXT - q_abs) * geometric;
        self.cloud_absorption = q_abs * geometric;
        self.liquid_water = liquid_water_content;
        self
    }
}

impl Default for VolumetricCloudLayer {
    fn default() -> Self {
        Self {
            cloud_layer_start: 6_361_500.0, // cloud base 1.5 km above the surface
            cloud_layer_end: 6_364_000.0,   // tops 4 km (2.5 km thick layer)
            cloud_density: 1.0,
            cloud_absorption: 0.0,
            cloud_scattering: 0.0,
            liquid_water: Self::CUMULUS_LIQUID_WATER,
            noise_scale: 32_000.0, // weather tile
            shape_noise_scale: 6_400.0,
            noise_offset: Vec3::ZERO,
            detail_noise_scale: 600.0,
            detail_strength: 1.0,
            coverage: 1.0,
            wind: None,
            profile_top_min: 0.25,
            wind_shear: None,
            updraft_speed: 3.0,
            lifetime: 1200.0,
            evolution_time: 0.0,
        }
        .with_liquid_water(Self::CUMULUS_LIQUID_WATER, Self::CUMULUS_EFFECTIVE_RADIUS)
    }
}

impl SyncComponent<RenderApp> for VolumetricCloudLayer {
    type Target = Self;
}

/// A high-altitude cirrus layer (thin ice cloud). Add it to an [`Atmosphere`] entity, with or
/// without a [`VolumetricCloudLayer`] (cumulus).
///
/// Structure after Hogan & Kew (2005, Q. J. R. Meteorol. Soc. 131, "A 3D stochastic cloud model
/// for investigating the radiative properties of inhomogeneous cirrus clouds"): at the generating
/// level (cloud top) the log ice water content is a turbulent tracer with a -5/3 power spectrum up
/// to an outer scale, and below it the ice falls at [`ThinCloudLayer::fall_speed`] through the wind
/// shear, displacing that pattern into fall streaks (Marshall 1953: x = S dz^2 / (2 w) for
/// constant shear S and fall speed w). Light: single scattering by an ice slab with the
/// asymmetry of roughened ice crystals.
#[derive(Clone, Component, Reflect)]
#[reflect(Clone, Default)]
pub struct ThinCloudLayer {
    /// Radius (from the planet center, like [`VolumetricCloudLayer::cloud_layer_start`]) of the generating
    /// level, the cirrus top. High clouds: 5-13 km in mid-latitudes (WMO International Cloud
    /// Atlas); Hogan & Kew's case: 9.1 km.
    /// units: m
    pub top: f32,
    /// Depth of the ice below the generating level (fall-streak length). Cirrus are typically
    /// 1-2 km thick (e.g. Sassen & Campbell 2001, J. Atmos. Sci. 58).
    /// units: m
    pub thickness: f32,
    /// Mean optical depth of the cloudy columns at normal incidence; 0 disables the layer. Sassen
    /// & Cho 1992 (J. Appl. Meteor. 31): subvisual < 0.03 < thin < 0.3 < opaque < 3.
    pub optical_depth: f32,
    /// Cloud fraction at each level, in [0, 1].
    pub coverage: f32,
    /// Standard deviation of ln(ice water content) across the layer: Hogan & Kew Fig. 2(d),
    /// 0.5-2.5 for the underlying distribution (<= 1.5 near cloud base).
    pub ln_iwc_std: f32,
    /// Outer scale: the -5/3 spectrum flattens beyond it (Hogan & Kew: ~50 km). Sets the
    /// generating texture's tile (1024 texels).
    /// units: m
    pub outer_scale: f32,
    /// Ice fall speed (Hogan & Kew: a constant 0.8 m/s reproduced the observed fall streaks).
    /// units: m/s
    pub fall_speed: f32,
    /// Relative spread of fall speeds (std / mean): it spreads the fall-streak displacement and
    /// smooths the pattern below the top, steepening the spectrum with depth as observed.
    pub fall_speed_spread: f32,
    /// Override of the wind at the generating level (horizontal), which advects the layer.
    /// `None`: [`AtmosphereWind`] at [`ThinCloudLayer::top`].
    /// units: m/s
    pub wind: Option<Vec3>,
    /// Override of the vertical wind shear dU/dz across the layer (horizontal). Fall streaks
    /// trail by -shear dz^2 / (2 fall_speed) at depth dz below the top. `None`: the mean shear of
    /// [`AtmosphereWind`] over the layer's thickness.
    /// units: 1/s
    pub wind_shear: Option<Vec3>,
    /// Asymmetry parameter g of the ice phase function: severely roughened aggregates, ~0.75 in
    /// the visible (Baum et al. 2014, JQSRT 146).
    pub asymmetry: f32,
    /// Accumulated displacement (m), advanced by [`advance_cloud_wind`].
    pub offset: Vec3,
}

impl Default for ThinCloudLayer {
    fn default() -> Self {
        Self {
            top: 6_369_500.0,
            thickness: 1500.0,
            optical_depth: 0.3,
            coverage: 0.35,
            ln_iwc_std: 1.0,
            outer_scale: 50_000.0,
            fall_speed: 0.8,
            fall_speed_spread: 0.25,
            wind: None,
            wind_shear: None,
            asymmetry: 0.75,
            offset: Vec3::ZERO,
        }
    }
}

impl VolumetricCloudLayer {
    /// Fair-weather cumulus (the default): base 1.5 km, tops up to 4 km, adiabatic profile,
    /// 1 g/m^3 peak liquid water.
    pub fn cumulus() -> Self {
        Self::default()
    }

    /// Wind at the base and shear over the layer: the overrides, or the atmosphere's profile.
    pub fn resolved_wind(&self, profile: &AtmosphereWind, inner_radius: f32) -> (Vec3, Vec3) {
        let base = self.cloud_layer_start - inner_radius;
        let top = self.cloud_layer_end - inner_radius;
        (
            self.wind.unwrap_or_else(|| profile.at(base)),
            self.wind_shear.unwrap_or_else(|| profile.mean_shear(base, top)),
        )
    }
}

impl ThinCloudLayer {
    /// Cirrus (the default): thin (tau 0.3), patchy (35 %), fall streaks.
    pub fn cirrus() -> Self {
        Self::default()
    }

    /// Cirrostratus: a nearly continuous, more uniform ice veil. Optical depth 0.6 (Sassen & Cho
    /// 1992's "opaque" class 0.3-3, the range of halo-producing sheets), cloud fraction 0.9, a
    /// smaller ln IWC spread (0.5) and a deeper layer (2 km).
    pub fn cirrostratus() -> Self {
        Self {
            optical_depth: 0.6,
            coverage: 0.9,
            ln_iwc_std: 0.5,
            thickness: 2000.0,
            ..Self::default()
        }
    }

    /// Wind at the generating level and shear over the thickness: the overrides, or the
    /// atmosphere's profile.
    pub fn resolved_wind(&self, profile: &AtmosphereWind, inner_radius: f32) -> (Vec3, Vec3) {
        let top = self.top - inner_radius;
        (
            self.wind.unwrap_or_else(|| profile.at(top)),
            self.wind_shear
                .unwrap_or_else(|| profile.mean_shear(top - self.thickness, top)),
        )
    }
}

/// Horizontal wind profile of the atmosphere, u(z): one profile for all cloud layers, so the
/// cumulus advection and lean, the cirrus drift and fall streaks, and their relative motion are
/// consistent. Layers take their wind and shear at their own altitude unless they override them
/// ([`VolumetricCloudLayer::wind`], [`ThinCloudLayer::wind`]). Required by the cloud layers
/// (default: [`AtmosphereWind::fair_weather`]).
#[derive(Clone, Component, Reflect, Debug)]
#[reflect(Clone, Default, Debug)]
pub struct AtmosphereWind {
    /// (altitude above the ground in m, horizontal wind in m/s), sorted by altitude; linear in
    /// between, constant beyond the first and last level.
    pub levels: Vec<(f32, Vec3)>,
}

impl Default for AtmosphereWind {
    fn default() -> Self {
        Self::fair_weather()
    }
}

impl AtmosphereWind {
    /// No wind.
    pub fn calm() -> Self {
        Self::constant(Vec3::ZERO)
    }

    /// The same wind at every altitude (no shear).
    pub fn constant(velocity: Vec3) -> Self {
        Self {
            levels: vec![(0.0, velocity)],
        }
    }

    /// A fair-weather cumulus profile (an input, like a sounding), with the structure of the
    /// shallow-cumulus LES cases:
    /// - a well-mixed subcloud layer, uniform wind up to 700 m (BOMEX: u = -8.75 m/s from the
    ///   surface to 700 m; Siebesma et al. 2003, Table B1);
    /// - 1.9 m/s per km of shear through the cumulus layer, to 4.5 km (BOMEX 1.8, -8.75 to
    ///   -4.61 m/s from 700 to 3000 m; RICO 2.0, -9.9 to -1.9 m/s from 0 to 4 km; van Zanten et
    ///   al. 2011, Table 2);
    /// - stronger shear (~3.4 m/s per km) toward the upper-tropospheric jet where cirrus form.
    ///
    /// The cloud layer's shear sets the clouds' lean ([`VolumetricCloudLayer::wind_shear`]).
    pub fn fair_weather() -> Self {
        let dir = Vec3::new(3.0, 0.0, 1.0).normalize();
        Self {
            levels: vec![
                (0.0, dir * 8.0),
                (700.0, dir * 8.0),
                (4500.0, dir * 15.2),
                (8000.0, dir * 27.0),
                (9500.0, dir * 33.3),
            ],
        }
    }

    /// Wind at `altitude` (m above the ground).
    pub fn at(&self, altitude: f32) -> Vec3 {
        let Some(&(z0, v0)) = self.levels.first() else {
            return Vec3::ZERO;
        };
        if altitude <= z0 {
            return v0;
        }
        for pair in self.levels.windows(2) {
            let ((za, va), (zb, vb)) = (pair[0], pair[1]);
            if altitude <= zb {
                return va.lerp(vb, (altitude - za) / (zb - za).max(1e-3));
            }
        }
        self.levels.last().map_or(Vec3::ZERO, |&(_, v)| v)
    }

    /// Mean shear dU/dz between two altitudes (1/s).
    pub fn mean_shear(&self, low: f32, high: f32) -> Vec3 {
        (self.at(high) - self.at(low)) / (high - low).max(1.0)
    }
}

/// The atmosphere a cloud layer belongs to: put [`VolumetricCloudLayer`] or [`ThinCloudLayer`] on
/// an entity with `CloudLayerOf(atmosphere)` to add a layer to that atmosphere (any number of thin
/// layers; one volumetric layer is rendered, see [`GpuCloudLayers`]). Layer components on the
/// atmosphere entity itself count as its layers too, so a single-layer scene needs no extra
/// entities. Wind comes from the atmosphere's [`AtmosphereWind`] (per-layer `wind` /
/// `wind_shear` override it).
///
/// ```ignore
/// commands
///     .spawn((Atmosphere::earthlike(medium), AtmosphereWind::fair_weather()))
///     .with_related_entities::<CloudLayerOf>(|layers| {
///         layers.spawn(VolumetricCloudLayer::cumulus());
///         layers.spawn(ThinCloudLayer::cirrus());
///         layers.spawn(ThinCloudLayer::cirrostratus());
///     });
/// ```
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
#[relationship(relationship_target = CloudLayers)]
pub struct CloudLayerOf(#[entities] pub Entity);

/// The cloud layers related to an atmosphere ([`CloudLayerOf`]); despawned with it.
#[derive(Component, Default, Debug, PartialEq, Eq)]
#[relationship_target(relationship = CloudLayerOf, linked_spawn)]
pub struct CloudLayers(Vec<Entity>);

/// Thin layers rendered per atmosphere (lowest first); further layers are ignored with a warning.
pub const MAX_THIN_CLOUD_LAYERS: usize = 4;

/// One thin (ice) layer of [`GpuCloudLayers`]: a [`ThinCloudLayer`] resolved against the wind
/// profile. Field names match `clouds.wesl` GpuThinCloudLayer.
#[derive(Clone, Copy, Default, ShaderType)]
pub struct GpuThinCloudLayer {
    pub offset: Vec3,
    pub top: f32,
    pub wind_shear: Vec3,
    pub thickness: f32,
    pub optical_depth: f32,
    /// Generating-field (unit Gaussian) threshold below which a level is clear: the
    /// (1 - coverage) quantile of N(0, 1).
    pub threshold: f32,
    pub ln_iwc_std: f32,
    /// Normalisation of exp(sigma g) to an in-cloud mean of 1 above the threshold.
    pub norm: f32,
    pub fall_speed: f32,
    pub fall_speed_spread: f32,
    pub outer_scale: f32,
    pub asymmetry: f32,
}

impl GpuThinCloudLayer {
    pub fn new(ci: &ThinCloudLayer, wind: &AtmosphereWind, inner_radius: f32) -> Self {
        let (_, shear) = ci.resolved_wind(wind, inner_radius);
        // Log-normal extinction exp(sigma g), g ~ N(0, 1), cloudy where g > g_c with
        // P(g > g_c) = coverage. In-cloud mean of exp(sigma g):
        // E[exp(sigma g) | g > g_c] = exp(sigma^2 / 2) Q(g_c - sigma) / Q(g_c), Q = 1 - Phi.
        let coverage = ci.coverage.clamp(1e-4, 1.0);
        let threshold = if coverage >= 1.0 { -40.0 } else { normal_quantile(1.0 - coverage) };
        let sigma = ci.ln_iwc_std.max(0.0);
        let in_cloud_mean = (0.5 * sigma * sigma).exp() * normal_sf(threshold - sigma)
            / normal_sf(threshold).max(1e-12);
        Self {
            offset: ci.offset,
            top: ci.top,
            wind_shear: shear,
            thickness: ci.thickness.max(0.0),
            optical_depth: ci.optical_depth.max(0.0),
            threshold,
            ln_iwc_std: sigma,
            norm: 1.0 / in_cloud_mean.max(1e-12),
            fall_speed: ci.fall_speed.max(1e-3),
            fall_speed_spread: ci.fall_speed_spread.max(0.0),
            outer_scale: ci.outer_scale.max(1.0),
            asymmetry: ci.asymmetry.clamp(-0.99, 0.99),
        }
    }

    /// Radius of the layer's mid level (its compositing shell).
    pub fn mid_radius(&self) -> f32 {
        self.top - 0.5 * self.thickness
    }
}

/// Render-world uniform of the cloud layers on a view's atmosphere: one [`VolumetricCloudLayer`]
/// (cumulus, optional) and up to [`MAX_THIN_CLOUD_LAYERS`] [`ThinCloudLayer`]s sorted by altitude.
/// Field names match `clouds.wesl`.
#[derive(Clone, Component, ShaderType)]
pub struct GpuCloudLayers {
    pub cloud_layer_start: f32,
    pub cloud_layer_end: f32,
    pub cloud_density: f32,
    pub cloud_absorption: f32,
    pub cloud_scattering: f32,
    pub liquid_water: f32,
    /// 0: profile lookups (physical or image), 1: analytic envelope, 2: Nubis's vertical profile.
    pub profile_mode: u32,
    pub profile_bottom_exponent: f32,
    pub profile_top_exponent: f32,
    pub profile_top_min: f32,
    pub noise_scale: f32,
    pub noise_offset: Vec3,
    pub detail_noise_scale: f32,
    pub detail_strength: f32,
    pub wind: Vec3,
    pub wind_shear: Vec3,
    pub updraft_speed: f32,
    pub lifetime: f32,
    pub evolution_time: f32,
    pub coverage: f32,
    pub shape_noise_scale: f32,
    /// Weather-map texel at the weather window's texel 0 (integer valued; `weather_window.wesl`),
    /// set per view from the camera at extraction.
    pub weather_window_origin: Vec2,
    /// Default cloud droplet number concentration of the physical channels (cm^-3): the layer's
    /// reference microphysics, LWC 1 g/m^3 at r_e 10 um, k N_d = 239 with k = 0.8 (Martin et al.
    /// 1994). Not read by the clouds yet.
    pub droplet_number: f32,
    /// Number of valid entries of [`GpuCloudLayers::thin`].
    pub thin_count: u32,
    /// Thin layers, lowest mid level first.
    pub thin: [GpuThinCloudLayer; MAX_THIN_CLOUD_LAYERS],
}

impl GpuCloudLayers {
    /// The lowest thin layer (the temporal histories follow its motion).
    pub fn lowest_thin(&self) -> Option<&GpuThinCloudLayer> {
        (self.thin_count > 0).then(|| &self.thin[0])
    }

    pub fn new(
        cumulus: Option<&VolumetricCloudLayer>,
        thin_layers: &[&ThinCloudLayer],
        wind: Option<&AtmosphereWind>,
        profiles: Option<&CloudProfiles>,
        inner_radius: f32,
    ) -> Self {
        let default_wind = AtmosphereWind::default();
        let wind = wind.unwrap_or(&default_wind);
        let disabled_cumulus = VolumetricCloudLayer {
            cloud_density: 0.0,
            ..VolumetricCloudLayer::default()
        };
        let c = cumulus.unwrap_or(&disabled_cumulus);
        let mut thin: Vec<GpuThinCloudLayer> = thin_layers
            .iter()
            .filter(|l| l.optical_depth > 0.0)
            .map(|l| GpuThinCloudLayer::new(l, wind, inner_radius))
            .collect();
        thin.sort_by(|a, b| a.mid_radius().total_cmp(&b.mid_radius()));
        if thin.len() > MAX_THIN_CLOUD_LAYERS {
            bevy_log::warn_once!(
                "an atmosphere has {} thin cloud layers; the lowest {MAX_THIN_CLOUD_LAYERS} are rendered",
                thin.len()
            );
        }
        let thin_count = thin.len().min(MAX_THIN_CLOUD_LAYERS);
        let mut thin_array = [GpuThinCloudLayer::default(); MAX_THIN_CLOUD_LAYERS];
        thin_array[..thin_count].copy_from_slice(&thin[..thin_count]);
        // Physical profiles carry their air masses' cloud bases and tops: the layer spans them.
        let physical = matches!(profiles, None | Some(CloudProfiles::Physical));
        let physical_layer;
        let c = if physical {
            let (base, top) = physical_profile_bounds();
            physical_layer = VolumetricCloudLayer {
                cloud_layer_start: inner_radius + base,
                cloud_layer_end: inner_radius + top,
                ..c.clone()
            };
            &physical_layer
        } else {
            c
        };
        let (c_wind, c_shear) = c.resolved_wind(wind, inner_radius);
        // 0: physical lookups, 1: analytic envelope (in the shader), 2: Nubis's vertical profile.
        let (profile_mode, profile_bottom_exponent, profile_top_exponent) = match profiles {
            Some(CloudProfiles::Envelope { bottom, top }) => (1, *bottom, *top),
            Some(CloudProfiles::Nubis) => (2, 1.0, 1.0),
            _ => (0, 1.0, 1.0),
        };
        Self {
            cloud_layer_start: c.cloud_layer_start,
            cloud_layer_end: c.cloud_layer_end,
            cloud_density: c.cloud_density,
            cloud_absorption: c.cloud_absorption,
            cloud_scattering: c.cloud_scattering,
            liquid_water: c.liquid_water,
            profile_mode,
            profile_bottom_exponent,
            profile_top_exponent,
            profile_top_min: c.profile_top_min,
            noise_scale: c.noise_scale,
            noise_offset: c.noise_offset,
            detail_noise_scale: c.detail_noise_scale,
            detail_strength: c.detail_strength,
            wind: c_wind,
            wind_shear: c_shear,
            updraft_speed: c.updraft_speed,
            lifetime: c.lifetime,
            evolution_time: c.evolution_time,
            coverage: c.coverage.max(0.0),
            shape_noise_scale: c.shape_noise_scale.max(1.0),
            weather_window_origin: Vec2::ZERO,
            droplet_number: 299.0,
            thin_count: thin_count as u32,
            thin: thin_array,
        }
    }
}

/// Upper tail of the standard normal distribution, Q(x) = erfc(x / sqrt 2) / 2.
fn normal_sf(x: f32) -> f32 {
    0.5 * erfc(x as f64 / core::f64::consts::SQRT_2) as f32
}

/// Complementary error function (Numerical Recipes `erfcc`, Chebyshev fit, |rel. error| < 1.2e-7).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
        .exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

/// Standard normal quantile Phi^-1(p) (Acklam's rational approximation, |rel. error| < 1.2e-9).
fn normal_quantile(p: f32) -> f32 {
    let p = (p as f64).clamp(1e-12, 1.0 - 1e-12);
    const A: [f64; 6] = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
    const B: [f64; 5] = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const C: [f64; 6] = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const D: [f64; 4] = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    let x = if p < 0.02425 {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p > 1.0 - 0.02425 {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    };
    x as f32
}

/// Multiplier on the cloud clock (advection, rising thermals, life cycle), e.g. for time-lapse.
#[derive(bevy_ecs::resource::Resource, Clone, Copy, Debug, Reflect)]
#[reflect(Clone, Debug)]
pub struct CloudTimeScale(pub f32);

impl Default for CloudTimeScale {
    fn default() -> Self {
        Self(1.0)
    }
}

/// Advances the cloud field each frame: advection by [`VolumetricCloudLayer::wind`], rise at
/// [`VolumetricCloudLayer::updraft_speed`] and the life-cycle clock [`VolumetricCloudLayer::evolution_time`].
pub fn advance_cloud_wind(
    time: Res<Time>,
    scale: Option<Res<CloudTimeScale>>,
    mut layers: Query<(Entity, &mut VolumetricCloudLayer, Option<&CloudLayerOf>)>,
    mut thin_layers: Query<(Entity, &mut ThinCloudLayer, Option<&CloudLayerOf>)>,
    atmospheres: Query<(&Atmosphere, Option<&AtmosphereWind>)>,
) {
    let default_wind = AtmosphereWind::default();
    let dt = time.delta_secs() * scale.map_or(1.0, |s| s.0);
    if dt <= 0.0 {
        return;
    }
    // A layer's atmosphere: the entity itself, or the one it is related to.
    let atmosphere_of = |entity: Entity, of: Option<&CloudLayerOf>| {
        atmospheres.get(of.map_or(entity, |o| o.0)).ok()
    };
    for (entity, mut layer, of) in &mut layers {
        let Some((atmosphere, wind)) = atmosphere_of(entity, of) else {
            continue;
        };
        let (wind, _) = layer.resolved_wind(wind.unwrap_or(&default_wind), atmosphere.inner_radius);
        let step = Vec3::new(wind.x, layer.updraft_speed, wind.z) * dt;
        layer.noise_offset += step;
        layer.evolution_time += dt;
    }
    for (entity, mut cirrus, of) in &mut thin_layers {
        let Some((atmosphere, wind)) = atmosphere_of(entity, of) else {
            continue;
        };
        let (wind, _) = cirrus.resolved_wind(wind.unwrap_or(&default_wind), atmosphere.inner_radius);
        let step = Vec3::new(wind.x, 0.0, wind.z) * dt;
        cirrus.offset += step;
    }
}

/// When [`CloudWeather`] changes (and no texture override), regenerate the procedural weather map.
pub fn extract_cloud_weather_regen(
    weathers: Extract<Query<&CloudWeather, Changed<CloudWeather>>>,
    mut generated: ResMut<NoiseGenerated>,
) {
    for weather in &weathers {
        if weather.texture.is_none() {
            generated.0 = false;
        }
    }
}

/// Apply [`VolumetricCloudSettings`] from the camera onto [`AtmosphereSettings`] (design.md §3.3).
pub fn apply_volumetric_cloud_settings(
    mut cameras: Query<(&VolumetricCloudSettings, &mut AtmosphereSettings), With<Camera3d>>,
) {
    for (cloud_settings, mut atmosphere_settings) in &mut cameras {
        atmosphere_settings.cloud_shadow_map_update_interleave =
            cloud_settings.shadow_update_interleave;
        atmosphere_settings.cloud_shadow_temporal_enabled = cloud_settings.shadow_temporal;
        atmosphere_settings.cloud_shadow_temporal_alpha = cloud_settings.shadow_temporal_alpha;
        atmosphere_settings.cloud_max_steps = cloud_settings.max_steps;
        atmosphere_settings.cloud_light_taps = cloud_settings.light_taps;
        atmosphere_settings.cloud_max_distance = cloud_settings.max_distance;
        atmosphere_settings.cloud_step_samples = cloud_settings.step_samples;
        atmosphere_settings.cloud_max_render_size = cloud_settings.max_render_size;
    }
}


#[cfg(test)]
mod cloud_layer_tests {
    use super::*;

    #[test]
    fn wind_profile_interpolates_and_clamps() {
        let wind = AtmosphereWind {
            levels: vec![(1000.0, Vec3::X * 10.0), (3000.0, Vec3::X * 30.0)],
        };
        assert_eq!(wind.at(0.0), Vec3::X * 10.0);
        assert!((wind.at(2000.0) - Vec3::X * 20.0).length() < 1e-4);
        assert_eq!(wind.at(5000.0), Vec3::X * 30.0);
        assert!((wind.mean_shear(1000.0, 3000.0) - Vec3::X * 0.01).length() < 1e-6);
        assert_eq!(AtmosphereWind::calm().at(1234.0), Vec3::ZERO);
    }

    #[test]
    fn layer_wind_overrides_profile() {
        let profile = AtmosphereWind::constant(Vec3::X * 5.0);
        let cumulus = VolumetricCloudLayer {
            wind: Some(Vec3::Z * 2.0),
            ..VolumetricCloudLayer::cumulus()
        };
        let (w, shear) = cumulus.resolved_wind(&profile, 6_360_000.0);
        assert_eq!(w, Vec3::Z * 2.0);
        assert_eq!(shear, Vec3::ZERO);
        let (w, _) = ThinCloudLayer::cirrus().resolved_wind(&profile, 6_360_000.0);
        assert_eq!(w, Vec3::X * 5.0);
    }

    #[test]
    fn normal_quantile_and_tail_are_consistent() {
        for p in [0.01_f32, 0.2, 0.5, 0.65, 0.99] {
            let x = normal_quantile(p);
            assert!((1.0 - normal_sf(x) - p).abs() < 1e-5, "p = {p}, x = {x}");
        }
        assert!((normal_quantile(0.975) - 1.959_964).abs() < 1e-4);
    }
}
