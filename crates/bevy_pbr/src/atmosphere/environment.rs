use crate::{
    environment_map::EnvironmentMapExposures,
    resources::{
        AtmosphereTextures, AtmosphereTransform, AtmosphereTransforms, AtmosphereTransformsOffset,
        GpuAtmosphere,
    },
    GpuCloudLayers, ExtractedAtmosphere, GpuAtmosphereSettings, GpuLights, LightMeta,
    LinearSampler, ScatteringMediumSampler, ViewLightsUniformOffset,
};
use bevy_asset::{load_embedded_asset, AssetServer, Assets, Handle, RenderAssetUsages};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    lifecycle::{Insert, Remove},
    observer::On,
    query::{Has, With},
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
    template::FromTemplate,
};
use bevy_image::Image;
use bevy_light::{
    AtmosphereEnvironmentMapLight, EnvironmentMapLight, GeneratedEnvironmentMapLight,
};
use bevy_math::UVec2;
use bevy_render::{
    camera::{ExtractedCamera, SortedCameras},
    diagnostic::RecordDiagnostics,
    extract_component::{ComponentUniforms, DynamicUniformIndex, ExtractComponent},
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{RenderContext, RenderDevice, ViewQuery},
    texture::{CachedTexture, FallbackImage, GpuImage},
    view::{ViewUniform, ViewUniformOffset, ViewUniforms},
    RenderApp,
};
use bevy_utils::default;
use tracing::warn;

use crate::detail_noise::DetailNoiseTexture;
use crate::fbm_noise::FbmNoiseTexture;
use crate::perlin_worley_noise::PerlinWorleyNoiseTexture;
use crate::GpuScatteringMedium;

// Render world representation of an environment map light for the atmosphere
#[derive(Component, ExtractComponent, Clone, FromTemplate)]
#[extract_app(RenderApp)]
#[extract_component_sync_target((Self, AtmosphereProbeTextures, AtmosphereProbeBindGroups))]
pub struct AtmosphereEnvironmentMap {
    pub environment_map: Handle<Image>,
    pub size: UVec2,
}

#[derive(Component)]
pub struct AtmosphereProbeTextures {
    pub environment: TextureView,
    pub transmittance_lut: CachedTexture,
    pub multiscattering_lut: CachedTexture,
    pub sky_view_lut: CachedTexture,
    pub aerial_view_lut: CachedTexture,
    pub cloud_shadow_map: CachedTexture,
    pub weather_window_art: CachedTexture,
    pub weather_window_physical: CachedTexture,
}

#[derive(Component)]
pub struct AtmosphereProbeBindGroups {
    pub environment: BindGroup,
    pub environment_clouds: Option<BindGroup>,
}

#[derive(Resource)]
pub struct AtmosphereProbeLayouts {
    pub environment: BindGroupLayoutDescriptor,
    pub environment_clouds: BindGroupLayoutDescriptor,
}

#[derive(Resource)]
pub struct AtmosphereProbePipeline {
    pub environment: CachedComputePipelineId,
    pub environment_clouds: CachedComputePipelineId,
}

pub fn init_atmosphere_probe_layout(mut commands: Commands) {
    let environment = BindGroupLayoutDescriptor::new(
        "environment_bind_group_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                // uniforms
                (0, uniform_buffer::<GpuAtmosphere>(true)),
                (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                (2, uniform_buffer::<AtmosphereTransform>(true)),
                (3, uniform_buffer::<ViewUniform>(true)),
                (4, uniform_buffer::<GpuLights>(true)),
                // scattering medium luts and sampler (required by `raymarch_atmosphere`)
                (5, texture_2d(TextureSampleType::default())),
                (6, texture_2d(TextureSampleType::default())),
                (7, sampler(SamplerBindingType::Filtering)),
                // atmosphere luts and sampler
                (8, texture_2d(TextureSampleType::default())), // transmittance
                (9, texture_2d(TextureSampleType::default())), // multiscattering
                (10, texture_2d(TextureSampleType::default())), // sky view
                (11, texture_3d(TextureSampleType::default())), // aerial view
                (12, sampler(SamplerBindingType::Filtering)),
                // output 2D array texture
                (
                    13,
                    texture_storage_2d_array(
                        TextureFormat::Rgba16Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                ),
            ),
        ),
    );

    // Clouds-enabled variant: adds scattering medium LUTs + cloud bindings used by `raymarch_atmosphere()`.
    let environment_clouds = BindGroupLayoutDescriptor::new(
        "environment_bind_group_layout_clouds",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                // uniforms
                (0, uniform_buffer::<GpuAtmosphere>(true)),
                (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                (2, uniform_buffer::<AtmosphereTransform>(true)),
                (3, uniform_buffer::<ViewUniform>(true)),
                (4, uniform_buffer::<GpuLights>(true)),
                // scattering medium luts and sampler (required by raymarch_atmosphere)
                (5, texture_2d(TextureSampleType::default())),
                (6, texture_2d(TextureSampleType::default())),
                (7, sampler(SamplerBindingType::Filtering)),
                // atmosphere luts and sampler
                (8, texture_2d(TextureSampleType::default())), // transmittance
                (9, texture_2d(TextureSampleType::default())), // multiscattering
                (10, texture_2d(TextureSampleType::default())), // sky view
                (11, texture_3d(TextureSampleType::default())), // aerial view
                (12, sampler(SamplerBindingType::Filtering)),
                // cloud bindings (match `clouds.wesl` / `bindings.wesl`)
                (20, uniform_buffer::<GpuCloudLayers>(true)),
                (
                    21,
                    texture_2d(TextureSampleType::Float { filterable: true }),
                ),
                (22, sampler(SamplerBindingType::Filtering)),
                (
                    23,
                    texture_2d(TextureSampleType::Float { filterable: true }),
                ), // cloud shadow map
                (
                    24,
                    texture_3d(TextureSampleType::Float { filterable: true }),
                ), // perlin-worley shape
                (
                    25,
                    texture_2d_array(TextureSampleType::Float { filterable: true }),
                ), // STBN
                (
                    33,
                    texture_3d(TextureSampleType::Float { filterable: true }),
                ), // detail
                (34, texture_2d(TextureSampleType::Float { filterable: true })), // vertical profiles
                // Physical cloud scheme fluctuation (g, g^2, b, b^2), shape tile + mips (`fbm_noise.rs`).
                (35, texture_3d(TextureSampleType::Float { filterable: true })),
                // Weather window around the camera: art (36) and physical (37) channels (`weather_window.wesl`).
                (36, texture_2d(TextureSampleType::Float { filterable: true })),
                (37, texture_2d(TextureSampleType::Float { filterable: true })),
                // output 2D array texture
                (
                    13,
                    texture_storage_2d_array(
                        TextureFormat::Rgba16Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                ),
            ),
        ),
    );

    commands.insert_resource(AtmosphereProbeLayouts {
        environment,
        environment_clouds,
    });
}

/// Picks the view that renders the probes not attached to a camera ([`probe_view`]) and records
/// the exposure each probe's environment map is stored with: the sky-view LUT it samples, and the
/// raymarched sky and clouds, are pre-exposed by the view that renders it.
///
/// The source is the view that renders first, so that later views sample this frame's
/// environment map.
pub(super) fn prepare_atmosphere_probe_sources(
    views: Query<(Entity, &ExtractedCamera), With<ExtractedAtmosphere>>,
    sorted_cameras: Res<SortedCameras>,
    probes: Query<(Entity, &AtmosphereEnvironmentMap)>,
    mut source: ResMut<AtmosphereProbeSource>,
    mut exposures: ResMut<EnvironmentMapExposures>,
) {
    source.0 = sorted_cameras
        .0
        .iter()
        .find_map(|camera| views.get(camera.entity).ok())
        .map(|(view, _)| view);

    for (probe, render_env_map) in &probes {
        let Some((_, camera)) =
            probe_view(probe, views.contains(probe), &source).and_then(|view| views.get(view).ok())
        else {
            continue;
        };
        exposures.insert(render_env_map.environment_map.id(), camera.exposure);
    }
}

pub(super) fn prepare_atmosphere_probe_bind_groups(
    probes: Query<(Entity, &AtmosphereProbeTextures), With<AtmosphereEnvironmentMap>>,
    probe_source: Res<AtmosphereProbeSource>,
    render_device: Res<RenderDevice>,
    layouts: Res<AtmosphereProbeLayouts>,
    atmosphere_sampler: Res<LinearSampler>,
    (
        medium_sampler,
        view_uniforms,
        lights_uniforms,
        atmosphere_transforms,
        atmosphere_uniforms,
        settings_uniforms,
        cloud_layer_uniforms,
        fbm_noise_texture,
        perlin_worley_noise_texture,
        detail_noise_texture,
        cloud_noise_sampler,
        (bluenoise, render_images, fallback_image),
        gpu_media,
        views,
        pipeline_cache,
    ): (
        Res<ScatteringMediumSampler>,
        Res<ViewUniforms>,
        Res<LightMeta>,
        Res<AtmosphereTransforms>,
        Res<ComponentUniforms<GpuAtmosphere>>,
        Res<ComponentUniforms<GpuAtmosphereSettings>>,
        Res<ComponentUniforms<GpuCloudLayers>>,
        Res<FbmNoiseTexture>,
        Res<PerlinWorleyNoiseTexture>,
        Res<DetailNoiseTexture>,
        Res<crate::resources::CloudNoiseSampler>,
        (
            Res<crate::Bluenoise>,
            Res<RenderAssets<GpuImage>>,
            Res<FallbackImage>,
        ),
        Res<RenderAssets<GpuScatteringMedium>>,
        Query<&ExtractedAtmosphere, With<ExtractedAtmosphere>>,
        Res<PipelineCache>,
    ),
    mut commands: Commands,
) {
    // STBN texture for environment_clouds (raymarch_atmosphere uses it when CLOUDS_ENABLED)
    let stbn_view: std::borrow::Cow<'_, TextureView> = match render_images.get(&bluenoise.texture) {
        Some(gpu) => std::borrow::Cow::Owned(gpu.texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        })),
        None => std::borrow::Cow::Borrowed(&fallback_image.d2_array.texture_view),
    };

    for (entity, textures) in &probes {
        let Some(atmosphere) = probe_view(entity, views.contains(entity), &probe_source)
            .and_then(|view| views.get(view).ok())
        else {
            continue;
        };
        let Some(gpu_medium) = gpu_media.get(atmosphere.medium) else {
            continue;
        };

        let environment = render_device.create_bind_group(
            "environment_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.environment),
            &BindGroupEntries::with_indices((
                // uniforms
                (0, atmosphere_uniforms.binding().unwrap()),
                (1, settings_uniforms.binding().unwrap()),
                (2, atmosphere_transforms.uniforms().binding().unwrap()),
                (3, view_uniforms.uniforms.binding().unwrap()),
                (4, lights_uniforms.view_gpu_lights.binding().unwrap()),
                // scattering medium luts and sampler
                (5, &gpu_medium.density_lut_view),
                (6, &gpu_medium.scattering_lut_view),
                (7, medium_sampler.sampler()),
                // atmosphere luts and sampler
                (8, &textures.transmittance_lut.default_view),
                (9, &textures.multiscattering_lut.default_view),
                (10, &textures.sky_view_lut.default_view),
                (11, &textures.aerial_view_lut.default_view),
                (12, &atmosphere_sampler.0),
                // output 2D array texture
                (13, &textures.environment),
            )),
        );

        // Optional clouds-enabled bind group (requires VolumetricCloudLayer + scattering medium + noise textures).
        let environment_clouds = (|| -> Option<BindGroup> {
            let cloud_layer_binding = cloud_layer_uniforms.binding()?;
            // `gpu_medium` comes from the selected view's atmosphere above.

            Some(render_device.create_bind_group(
                "environment_bind_group_clouds",
                &pipeline_cache.get_bind_group_layout(&layouts.environment_clouds),
                &BindGroupEntries::with_indices((
                    // uniforms
                    (0, atmosphere_uniforms.binding().unwrap()),
                    (1, settings_uniforms.binding().unwrap()),
                    (2, atmosphere_transforms.uniforms().binding().unwrap()),
                    (3, view_uniforms.uniforms.binding().unwrap()),
                    (4, lights_uniforms.view_gpu_lights.binding().unwrap()),
                    // scattering medium luts and sampler
                    (5, &gpu_medium.density_lut_view),
                    (6, &gpu_medium.scattering_lut_view),
                    (7, medium_sampler.sampler()),
                    // atmosphere luts and sampler
                    (8, &textures.transmittance_lut.default_view),
                    (9, &textures.multiscattering_lut.default_view),
                    (10, &textures.sky_view_lut.default_view),
                    (11, &textures.aerial_view_lut.default_view),
                    (12, &atmosphere_sampler.0),
                    // cloud bindings
                    (20, cloud_layer_binding.clone()),
                    (21, &fbm_noise_texture.texture.default_view),
                    (22, &**cloud_noise_sampler),
                    (23, &textures.cloud_shadow_map.default_view),
                    (24, &perlin_worley_noise_texture.texture.default_view),
                    (25, stbn_view.as_ref()),
                    (33, &detail_noise_texture.texture.default_view),
                    (34, &fbm_noise_texture.profile.default_view),
                    (35, &fbm_noise_texture.scheme_noise.default_view),
                    (36, &textures.weather_window_art.default_view),
                    (37, &textures.weather_window_physical.default_view),
                    // output
                    (13, &textures.environment),
                )),
            ))
        })();

        commands.entity(entity).insert(AtmosphereProbeBindGroups {
            environment,
            environment_clouds,
        });
    }
}

/// The view that renders an atmosphere environment probe, with its own uniforms, medium, LUTs
/// and cloud shadow map, so all of them describe one atmosphere (audit 39: probes borrowed the
/// first view's LUTs while every view dispatched them with its own uniforms):
/// - a probe on a camera ([`AtmosphereEnvironmentMapLight`] on a `Camera3d`) is that view;
/// - a [`LightProbe`](bevy_light::LightProbe) region is rendered by one deterministic source view,
///   the first atmosphere view in [`SortedCameras`] ([`AtmosphereProbeSource`]).
pub(super) fn probe_view(probe: Entity, probe_is_view: bool, source: &AtmosphereProbeSource) -> Option<Entity> {
    if probe_is_view {
        Some(probe)
    } else {
        source.0
    }
}

/// The view that renders the atmosphere probes not attached to a camera ([`probe_view`]).
#[derive(Resource, Default)]
pub struct AtmosphereProbeSource(pub Option<Entity>);

pub(super) fn prepare_probe_textures(
    view_textures: Query<&AtmosphereTextures, With<ExtractedAtmosphere>>,
    probes: Query<(Entity, &AtmosphereEnvironmentMap)>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    source: Res<AtmosphereProbeSource>,
    mut commands: Commands,
) {
    // The rendering view's textures, every frame: they are its own and can be reallocated.
    for (probe, render_env_map) in &probes {
        let Some(view) = probe_view(probe, view_textures.contains(probe), &source) else {
            continue;
        };
        let Ok(view_textures) = view_textures.get(view) else {
            continue;
        };
        let Some(environment) = gpu_images.get(&render_env_map.environment_map) else {
            continue;
        };

        let environment_view = environment.texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        });

        commands.entity(probe).insert(AtmosphereProbeTextures {
            environment: environment_view,
            transmittance_lut: view_textures.transmittance_lut.clone(),
            multiscattering_lut: view_textures.multiscattering_lut.clone(),
            sky_view_lut: view_textures.sky_view_lut.clone(),
            aerial_view_lut: view_textures.aerial_view_lut.clone(),
            cloud_shadow_map: view_textures.cloud_shadow_map.clone(),
            weather_window_art: view_textures.weather_window_art.clone(),
            weather_window_physical: view_textures.weather_window_physical.clone(),
        });
    }
}

pub fn init_atmosphere_probe_pipeline(
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<AtmosphereProbeLayouts>,
    asset_server: Res<AssetServer>,
    mut commands: Commands,
) {
    let environment = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("environment_pipeline".into()),
        layout: vec![layouts.environment.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "environment.wesl"),
        shader_defs: vec!["ATMOSPHERE_NO_SHADOW_MAPS".into()],
        ..default()
    });

    let environment_clouds = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("environment_pipeline_clouds".into()),
        layout: vec![layouts.environment_clouds.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "environment.wesl"),
        shader_defs: vec!["CLOUDS_ENABLED".into(), "ATMOSPHERE_NO_SHADOW_MAPS".into(), "ATMOSPHERE_ENV_PROBE".into()],
        ..default()
    });

    commands.insert_resource(AtmosphereProbePipeline {
        environment,
        environment_clouds,
    });
}

// Ensure power-of-two dimensions to avoid edge update issues on cubemap faces
pub fn validate_environment_map_size(size: UVec2) -> UVec2 {
    let new_size = UVec2::new(
        size.x.max(1).next_power_of_two(),
        size.y.max(1).next_power_of_two(),
    );
    if new_size != size {
        warn!(
            "Non-power-of-two AtmosphereEnvironmentMapLight size {}, correcting to {new_size}",
            size
        );
    }
    new_size
}

/// Allocates a cubemap and inserts filtered or unfiltered environment lighting.
pub fn on_insert_atmosphere_environment_map_light(
    insert: On<Insert<AtmosphereEnvironmentMapLight>>,
    lights: Query<(
        &AtmosphereEnvironmentMapLight,
        Has<AtmosphereEnvironmentMap>,
    )>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    let Ok((env_map_light, already_added_env_map)) = lights.get(insert.entity) else {
        return;
    };
    let Ok(mut entity) = commands.get_entity(insert.entity) else {
        return;
    };

    // Handle case where someone inserts a new AtmosphereEnvironmentMapLight component
    // on an entity that already has an AtmosphereEnvironmentMapLight component
    if already_added_env_map {
        entity.try_remove::<(
            AtmosphereEnvironmentMap,
            GeneratedEnvironmentMapLight,
            EnvironmentMapLight,
        )>();
    }

    let new_size = validate_environment_map_size(env_map_light.size);
    let mut environment_image = Image::new_fill(
        Extent3d {
            width: new_size.x,
            height: new_size.y,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        &[0; 8],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::all(),
    );
    environment_image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..Default::default()
    });
    environment_image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC;

    let environment_handle = images.add(environment_image);

    entity.insert(AtmosphereEnvironmentMap {
        environment_map: environment_handle.clone(),
        size: new_size,
    });

    if env_map_light.filtered {
        entity.insert(GeneratedEnvironmentMapLight {
            environment_map: environment_handle,
            intensity: env_map_light.intensity,
            affects_lightmapped_mesh_diffuse: env_map_light.affects_lightmapped_mesh_diffuse,
            ..default()
        });
    } else {
        entity.insert(EnvironmentMapLight {
            diffuse_map: environment_handle.clone(),
            specular_map: environment_handle,
            intensity: env_map_light.intensity,
            affects_lightmapped_mesh_diffuse: env_map_light.affects_lightmapped_mesh_diffuse,
            ..default()
        });
    }
}

/// Removes the cubemap and environment lighting inserted with it.
pub fn on_remove_atmosphere_environment_map_light(
    remove: On<Remove<AtmosphereEnvironmentMapLight>>,
    mut commands: Commands,
) {
    if let Ok(mut entity) = commands.get_entity(remove.entity) {
        entity.try_remove::<(
            AtmosphereEnvironmentMap,
            GeneratedEnvironmentMapLight,
            EnvironmentMapLight,
        )>();
    }
}

pub fn atmosphere_environment(
    view: ViewQuery<(
        &DynamicUniformIndex<GpuAtmosphere>,
        &DynamicUniformIndex<GpuAtmosphereSettings>,
        &AtmosphereTransformsOffset,
        &ViewUniformOffset,
        &ViewLightsUniformOffset,
        Option<&DynamicUniformIndex<GpuCloudLayers>>,
    )>,
    probe_query: Query<(
        Entity,
        &AtmosphereProbeBindGroups,
        &AtmosphereEnvironmentMap,
        Has<ExtractedAtmosphere>,
    )>,
    probe_source: Res<AtmosphereProbeSource>,
    pipeline_cache: Res<PipelineCache>,
    pipelines: Res<AtmosphereProbePipeline>,
    mut ctx: RenderContext,
) {
    let this_view = view.entity();
    let Some(environment_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.environment)
    else {
        return;
    };

    let (
        atmosphere_uniforms_offset,
        settings_uniforms_offset,
        atmosphere_transforms_offset,
        view_uniforms_offset,
        lights_uniforms_offset,
        cloud_layer_offset,
    ) = view.into_inner();

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(ctx.command_encoder(), "atmosphere_environment");

    for (probe, bind_groups, env_map_light, probe_is_view) in probe_query.iter() {
        // Each probe is rendered by its own view only (`probe_view`): its bind groups hold that
        // view's medium and LUTs, matching the uniform offsets below.
        if probe_view(probe, probe_is_view, &probe_source) != Some(this_view) {
            continue;
        }
        let use_clouds = cloud_layer_offset.is_some() && bind_groups.environment_clouds.is_some();

        let pipeline = if use_clouds {
            pipeline_cache.get_compute_pipeline(pipelines.environment_clouds)
        } else {
            Some(environment_pipeline)
        };

        let Some(pipeline) = pipeline else {
            continue;
        };

        let command_encoder = ctx.command_encoder();
        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("environment_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(pipeline);

        if use_clouds {
            pass.set_bind_group(
                0,
                bind_groups.environment_clouds.as_ref().unwrap(),
                &[
                    atmosphere_uniforms_offset.index(),
                    settings_uniforms_offset.index(),
                    atmosphere_transforms_offset.index(),
                    view_uniforms_offset.offset,
                    lights_uniforms_offset.offset,
                    cloud_layer_offset.as_ref().unwrap().index(),
                ],
            );
        } else {
            pass.set_bind_group(
                0,
                &bind_groups.environment,
                &[
                    atmosphere_uniforms_offset.index(),
                    settings_uniforms_offset.index(),
                    atmosphere_transforms_offset.index(),
                    view_uniforms_offset.offset,
                    lights_uniforms_offset.offset,
                ],
            );
        }

        pass.dispatch_workgroups(
            env_map_light.size.x / 8,
            env_map_light.size.y / 8,
            6, // 6 cubemap faces
        );
    }

    time_span.end(ctx.command_encoder());
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::Update;
    use bevy_asset::{uuid_handle, AssetId};
    use bevy_camera::{CameraOutputMode, ClearColorConfig, MsaaWriteback};
    use bevy_ecs::{schedule::ScheduleLabel, world::World};
    use bevy_math::{Mat4, Vec3};
    use bevy_render::camera::SortedCamera;

    fn atmosphere_view(order: isize, exposure: f32) -> (ExtractedCamera, ExtractedAtmosphere) {
        (
            ExtractedCamera {
                target: None,
                physical_viewport_size: None,
                physical_target_size: None,
                viewport: None,
                schedule: Update.intern(),
                order,
                output_mode: CameraOutputMode::default(),
                msaa_writeback: MsaaWriteback::default(),
                clear_color: ClearColorConfig::default(),
                sorted_camera_index_for_target: 0,
                exposure,
                hdr: true,
                tonemap_in_shader: false,
                runs_tonemapping_pass: true,
            },
            ExtractedAtmosphere {
                inner_radius: 6_360_000.0,
                outer_radius: 6_460_000.0,
                ground_albedo: Vec3::splat(0.3),
                medium: AssetId::default(),
                world_to_atmosphere: Mat4::IDENTITY,
                atmosphere_to_world: Mat4::IDENTITY,
            },
        )
    }

    fn probe(environment_map: Handle<Image>) -> AtmosphereEnvironmentMap {
        AtmosphereEnvironmentMap {
            environment_map,
            size: UVec2::splat(128),
        }
    }

    #[test]
    fn probes_record_the_exposure_of_the_view_that_renders_them() {
        let own_map = uuid_handle!("6d2f86a4-3c1b-4b9e-9a51-0c6f1e0b7a11");
        let standalone_map = uuid_handle!("0b9c7f3e-58d2-4e8a-b3a4-71d5c2e9f622");

        let mut world = World::new();
        world.init_resource::<EnvironmentMapExposures>();
        world.init_resource::<AtmosphereProbeSource>();
        let late_view = world
            .spawn((atmosphere_view(1, 0.25), probe(own_map.clone())))
            .id();
        let first_view = world.spawn(atmosphere_view(0, 4.0)).id();
        let standalone = world.spawn(probe(standalone_map.clone())).id();
        let sorted = |entity, order| SortedCamera {
            entity,
            order,
            target: None,
            output_mode: CameraOutputMode::default(),
        };
        world.insert_resource(SortedCameras(vec![
            sorted(first_view, 0),
            sorted(late_view, 1),
        ]));

        world
            .run_system_cached(prepare_atmosphere_probe_sources)
            .unwrap();

        let source = world.resource::<AtmosphereProbeSource>();
        assert_eq!(probe_view(late_view, true, source), Some(late_view));
        assert_eq!(probe_view(standalone, false, source), Some(first_view));
        let exposures = world.resource::<EnvironmentMapExposures>();
        assert_eq!(exposures.get(own_map.id()), 0.25);
        assert_eq!(exposures.get(standalone_map.id()), 4.0);
    }
}
