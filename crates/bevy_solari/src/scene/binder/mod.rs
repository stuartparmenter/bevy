mod allocator;
mod assets;
mod bind_group;
mod instances;
mod lights;
mod tlas;
mod tlas_build;

use self::assets::{AssetState, MAX_TEXTURE_COUNT};
pub use self::bind_group::prepare_raytracing_scene_bind_group;
use self::bind_group::BindGroupCacheState;
use self::instances::{
    ChangedInstanceFilter, InstanceInputs, InstanceQueryData, InstanceState, MAX_MESH_SLAB_COUNT,
};
use self::lights::LightState;
use self::tlas::TlasState;
pub use self::tlas::{build_raytracing_tlas, TlasInstanceSetupPipeline};
use super::{blas::BlasManager, extract::StandardMaterialAssets, RaytracingMesh3d};
use bevy_ecs::{
    entity::Entity,
    lifecycle::RemovedComponents,
    resource::Resource,
    system::{Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_pbr::ExtractedDirectionalLight;
use bevy_render::{
    mesh::allocator::MeshAllocator,
    render_asset::{ExtractedAssets, RenderAssets},
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    texture::GpuImage,
};
use tracing::{info, info_span};

/// Small scene constants the shaders read alongside Solari's other scene data.
///
/// wgpu forbids uniform-buffer bindings in a bind group that also contains binding arrays, so
/// these live in a read-only storage buffer instead.
#[derive(ShaderType, Clone, Copy, Default, PartialEq)]
struct GpuSceneParameters {
    max_world_geometry_error: f32,
}

/// Logs the scene's composition once it has held still for a couple of frames, so a settling
/// scene reports its final shape rather than every loading step.
#[derive(Default)]
struct SceneSummaryLog {
    last: Option<(u32, usize, usize, usize)>,
    stable_frames: u8,
    reported: Option<(u32, usize, usize, usize)>,
}

impl SceneSummaryLog {
    fn observe(&mut self, summary: (u32, usize, usize, usize)) {
        if self.last == Some(summary) {
            self.stable_frames = self.stable_frames.saturating_add(1);
        } else {
            self.last = Some(summary);
            self.stable_frames = 0;
        }
        if self.stable_frames >= 2 && self.reported != Some(summary) {
            info!(
                raytracing_instances = summary.0,
                emissive_mesh_lights = summary.1,
                directional_lights = summary.2,
                total_light_sources = summary.3,
                "prepared Solari raytracing scene"
            );
            self.reported = Some(summary);
        }
    }
}

/// Insert this resource into the render world to make the raytracing scene retain the previous
/// frame's TLAS and the light id translation table that maps into it.
///
/// This is useful for temporal techniques that need last frame's data. Retaining it costs a second
/// TLAS allocation and rebuild, so the scene only does so while something asks for it.
#[derive(Resource, Default)]
pub struct RaytracingSceneNeedsPreviousFrameData;

#[derive(Resource)]
pub struct RaytracingSceneBindings {
    pub bind_group: Option<BindGroup>,
    pub bind_group_layout: BindGroupLayoutDescriptor,
    assets: AssetState,
    instances: InstanceState,
    lights: LightState,
    tlas: TlasState,
    bind_groups: BindGroupCacheState,
    scene_parameters: StorageBuffer<GpuSceneParameters>,
    last_scene_parameters: Option<GpuSceneParameters>,
    summary: SceneSummaryLog,
}

impl RaytracingSceneBindings {
    /// Records that a lighting pass read `previous_frame_light_id_translations`, so the next
    /// frame's table translates from this frame's light ids rather than older ones.
    pub fn note_light_translations_consumed(&self) {
        self.lights.note_translations_consumed();
    }
}

fn bind_group_layout_descriptor() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "raytracing_scene_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None).count(MAX_MESH_SLAB_COUNT),
                storage_buffer_read_only_sized(false, None).count(MAX_MESH_SLAB_COUNT),
                texture_2d(TextureSampleType::Float { filterable: true }).count(MAX_TEXTURE_COUNT),
                sampler(SamplerBindingType::Filtering).count(MAX_TEXTURE_COUNT),
                storage_buffer_read_only_sized(false, None),
                acceleration_structure(),
                acceleration_structure(),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                storage_buffer_read_only::<GpuSceneParameters>(false),
            ),
        ),
    )
}

impl FromWorld for RaytracingSceneBindings {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        Self {
            bind_group: None,
            bind_group_layout: bind_group_layout_descriptor(),
            assets: AssetState::new(),
            instances: InstanceState::new(),
            lights: LightState::new(),
            tlas: TlasState::new(render_device),
            bind_groups: BindGroupCacheState::new(render_device),
            scene_parameters: StorageBuffer::from(GpuSceneParameters::default()),
            last_scene_parameters: None,
            summary: SceneSummaryLog::default(),
        }
    }
}

/// Applies this frame's scene changes to the retained buffers, binding arrays and TLAS.
pub fn prepare_raytracing_scene_resources(
    instances: Query<InstanceQueryData>,
    changed_instances: Query<Entity, ChangedInstanceFilter>,
    mut removed_instances: RemovedComponents<RaytracingMesh3d>,
    directional_lights: Query<(Entity, &ExtractedDirectionalLight)>,
    needs_previous_frame_data: Option<Res<RaytracingSceneNeedsPreviousFrameData>>,
    mesh_allocator: Res<MeshAllocator>,
    blas_manager: Res<BlasManager>,
    material_assets: Res<StandardMaterialAssets>,
    texture_assets: Res<RenderAssets<GpuImage>>,
    extracted_images: Res<ExtractedAssets<GpuImage>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    instance_setup_pipeline: Res<TlasInstanceSetupPipeline>,
    mut bindings: ResMut<RaytracingSceneBindings>,
) {
    let bindings = &mut *bindings;
    let needs_previous_frame_data = needs_previous_frame_data.is_some();

    // Roll light ids over before any removal or compaction writes this frame's translations
    bindings.lights.begin_frame(needs_previous_frame_data);

    // Update material and texture assets
    bindings
        .assets
        .update_materials(&mut bindings.instances, &material_assets, &texture_assets);
    bindings.assets.update_textures(
        &mut bindings.instances,
        &extracted_images,
        &texture_assets,
        &material_assets,
    );

    // Apply structural instance changes, now that asset slots are current
    bindings
        .instances
        .remove_instances(&mut bindings.lights, removed_instances.read());
    let inputs = InstanceInputs {
        assets: &bindings.assets,
        blas_manager: &blas_manager,
        mesh_allocator: &mesh_allocator,
    };
    bindings.instances.refresh_instances(
        &inputs,
        &mut bindings.lights,
        &instances,
        &changed_instances,
    );

    // Update the light set, now that emissive instances are resolved
    bindings.lights.update(&directional_lights);

    write_scene_parameters(bindings, &render_device, &render_queue);

    bindings.summary.observe((
        bindings.instances.live_count,
        bindings.lights.emissive_light_count(),
        bindings.lights.directional_light_count(),
        bindings.lights.index.len(),
    ));

    // Upload the above writes
    write_sparse_buffers(bindings, &render_device, &render_queue);

    // Prepare the next TLAS
    let build_ready = !bindings.tlas.uses_raw_build()
        || instance_setup_pipeline
            .id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id))
            .is_some();
    bindings.tlas.advance(
        &bindings.instances,
        &mut bindings.bind_groups,
        &render_device,
        build_ready,
        needs_previous_frame_data,
    );
}

/// Refreshes the scene-constant storage buffer whenever one of its inputs changed.
fn write_scene_parameters(
    bindings: &mut RaytracingSceneBindings,
    device: &RenderDevice,
    queue: &RenderQueue,
) {
    let parameters = GpuSceneParameters {
        max_world_geometry_error: bindings.instances.max_world_geometry_error(),
    };

    if bindings.last_scene_parameters != Some(parameters) {
        bindings.scene_parameters.set(parameters);
        bindings.scene_parameters.write_buffer(device, queue);
        bindings.last_scene_parameters = Some(parameters);
    }
}

/// Grows every sparse buffer to hold at least one element, then snapshots its dirty set into
/// either a staged sparse update or a full reupload.
fn write_sparse_buffers(
    bindings: &mut RaytracingSceneBindings,
    device: &RenderDevice,
    queue: &RenderQueue,
) {
    let _span = info_span!("write_buffers").entered();

    let assets = &mut bindings.assets;
    assets.materials.grow(1);
    assets.materials.write_buffers(device, queue);

    let instances = &mut bindings.instances;
    instances.transforms.grow(1);
    instances.transforms.write_buffers(device, queue);
    instances.previous_frame_transforms.grow(1);
    instances
        .previous_frame_transforms
        .write_buffers(device, queue);
    instances.geometry_ids.grow(1);
    instances.geometry_ids.write_buffers(device, queue);
    instances.material_ids.grow(1);
    instances.material_ids.write_buffers(device, queue);
    if bindings.tlas.uses_raw_build() {
        instances.blas_refs.grow(1);
        instances.blas_refs.write_buffers(device, queue);
    }

    let lights = &mut bindings.lights;
    lights.sources.grow(1);
    lights.sources.write_buffers(device, queue);
    lights.directional_lights.grow(1);
    lights.directional_lights.write_buffers(device, queue);
    lights.previous_frame_id_translations.grow(1);
    lights
        .previous_frame_id_translations
        .write_buffers(device, queue);
}

#[cfg(test)]
mod tests {
    use super::bind_group_layout_descriptor;
    use bevy_render::render_resource::{BindingType, BufferBindingType};

    #[test]
    fn raytracing_scene_binding_arrays_do_not_share_a_group_with_uniform_buffers() {
        let layout = bind_group_layout_descriptor();
        assert!(layout.entries.iter().any(|entry| entry.count.is_some()));
        assert!(layout.entries.iter().all(|entry| {
            !matches!(
                &entry.ty,
                BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    ..
                }
            )
        }));
    }

    #[test]
    fn only_a_rasterized_surface_pays_the_shading_normal_safety_factor() {
        // Both bias sites resolve an unknown error through the same bound, so the scene-wide maximum
        // cannot end up less conservative than the per-instance path of the instance that set it.
        // Only the rasterized one is offset along a normal-mapped normal, so only it needs the
        // factor; a ray hit carries a true geometric normal.
        let shader = include_str!("../bindings.wesl");
        let body = |name: &str| {
            shader
                .split_once(&format!("fn {name}"))
                .unwrap_or_else(|| panic!("{name} must exist"))
                .1
                .split_once("\n}")
                .unwrap_or_else(|| panic!("{name} must have a body"))
                .0
                .to_string()
        };

        let rasterized = body("rasterized_surface_ray_origin_bias");
        let per_instance = body("ray_origin_bias_for_instance");
        for bias in [&rasterized, &per_instance] {
            assert!(bias.contains("bounded_world_geometry_error("), "{bias}");
        }
        assert!(rasterized.contains("RAY_ORIGIN_BIAS_SHADING_NORMAL_SAFETY *"));
        assert!(
            !per_instance.contains("RAY_ORIGIN_BIAS_SHADING_NORMAL_SAFETY"),
            "{per_instance}"
        );
        assert!(body("bounded_world_geometry_error")
            .contains("max(scene_parameters.max_world_geometry_error, 0.0)"));
    }
}
