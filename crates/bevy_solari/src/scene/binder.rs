use super::{
    blas::BlasManager, extract::StandardMaterialAssets, RaytracingMesh3d, SolariEnvironmentLight,
};
use bevy_asset::{AssetId, Handle};
use bevy_color::{ColorToComponents, LinearRgba};
use bevy_ecs::{
    entity::Entity,
    resource::Resource,
    system::{Query, Res, ResMut},
};
use bevy_math::{ops::cos, Affine3, Affine3Ext, Mat3, Mat4, Vec3, Vec4};
use bevy_pbr::{
    world_geometry_error, DfgLut, ExtractedDirectionalLight, MeshGeometryError, MeshMaterial3d,
    PreviousGlobalTransform, StandardMaterial,
};
use bevy_platform::{collections::HashMap, hash::FixedHasher};
use bevy_render::{
    diagnostic::{DiagnosticsRecorder, RecordDiagnostics},
    mesh::allocator::MeshAllocator,
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    texture::{FallbackImage, GpuImage},
};
use bevy_transform::components::GlobalTransform;
use core::{f32::consts::TAU, hash::Hash, num::NonZeroU32, ops::Deref};
use tracing::{error, info};

const MAX_MESH_SLAB_COUNT: NonZeroU32 = NonZeroU32::new(500).unwrap();
const MAX_TEXTURE_COUNT: NonZeroU32 = NonZeroU32::new(5_000).unwrap();

const TEXTURE_MAP_NONE: u32 = u32::MAX;
const LIGHT_NOT_PRESENT_THIS_FRAME: u32 = u32::MAX;
const MAX_EMISSIVE_TRIANGLES_PER_LIGHT: u32 = u16::MAX as u32;
const MAX_LIGHT_SOURCES: usize = u16::MAX as usize;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum LightKey {
    EmissiveMesh { entity: Entity, first_triangle: u32 },
    Directional(Entity),
    Environment(Entity),
}

#[derive(Resource)]
pub struct RaytracingSceneBindings {
    pub bind_group: Option<BindGroup>,
    pub bind_group_layout: BindGroupLayoutDescriptor,
    previous_frame_tlas: Option<Tlas>,
    previous_frame_light_keys: Vec<LightKey>,
    last_scene_summary: Option<(usize, usize, usize, usize, usize)>,
    scene_summary_stable_frames: u8,
    reported_scene_summary: Option<(usize, usize, usize, usize, usize)>,
    reported_light_source_overflow: bool,
}

pub fn prepare_raytracing_scene_bindings(
    instances_query: Query<(
        Entity,
        &RaytracingMesh3d,
        &MeshGeometryError,
        &MeshMaterial3d<StandardMaterial>,
        &GlobalTransform,
        Option<&PreviousGlobalTransform>,
    )>,
    directional_lights_query: Query<(Entity, &ExtractedDirectionalLight)>,
    environment_lights_query: Query<(Entity, &SolariEnvironmentLight)>,
    mesh_allocator: Res<MeshAllocator>,
    blas_manager: Res<BlasManager>,
    material_assets: Res<StandardMaterialAssets>,
    texture_assets: Res<RenderAssets<GpuImage>>,
    fallback_texture: Res<FallbackImage>,
    dfg_lut: Res<DfgLut>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    render_queue: Res<RenderQueue>,
    mut diagnostics: Option<ResMut<DiagnosticsRecorder>>,
    mut raytracing_scene_bindings: ResMut<RaytracingSceneBindings>,
) {
    raytracing_scene_bindings.bind_group = None;

    let previous_frame_tlas = raytracing_scene_bindings.previous_frame_tlas.take();

    let mut this_frame_light_key_to_id = HashMap::<LightKey, u32>::default();
    let previous_frame_light_keys: Vec<_> = raytracing_scene_bindings
        .previous_frame_light_keys
        .drain(..)
        .collect();

    if instances_query.iter().len() == 0 {
        return;
    }

    let mut vertex_buffers = CachedBindingArray::new();
    let mut index_buffers = CachedBindingArray::new();
    let mut textures = CachedBindingArray::new();
    let mut samplers = Vec::new();
    let mut materials = StorageBufferList::<GpuMaterial>::default();
    let mut tlas = render_device
        .wgpu_device()
        .create_tlas(&CreateTlasDescriptor {
            label: Some("tlas"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
            max_instances: instances_query.iter().len() as u32,
        });
    let mut transforms = StorageBufferList::<[Vec4; 3]>::default();
    let mut previous_frame_transforms = StorageBufferList::<[Vec4; 3]>::default();
    let mut geometry_ids = StorageBufferList::<GpuInstanceGeometryIds>::default();
    let mut material_ids = StorageBufferList::<u32>::default();
    let mut light_sources = StorageBufferList::<GpuLightSource>::default();
    let mut directional_lights = StorageBufferList::<GpuDirectionalLight>::default();
    let mut previous_frame_light_id_translations = StorageBufferList::<u32>::default();
    // wgpu forbids uniform-buffer bindings in a bind group that also contains
    // binding arrays. Keep these small scene constants in a read-only storage
    // buffer alongside Solari's other scene data.
    let mut scene_parameters = StorageBuffer::from(GpuSceneParameters::default());

    let mut material_id_map: HashMap<AssetId<StandardMaterial>, u32, FixedHasher> =
        HashMap::default();
    let mut material_id = 0;
    let mut process_texture = |texture_handle: &Option<Handle<_>>| -> Option<u32> {
        match texture_handle {
            Some(texture_handle) => match texture_assets.get(texture_handle.id()) {
                Some(texture) => {
                    let (texture_id, is_new) =
                        textures.push_if_absent(texture.texture_view.deref(), texture_handle.id());
                    if is_new {
                        samplers.push(texture.sampler.deref());
                    }
                    Some(texture_id)
                }
                None => None,
            },
            None => Some(TEXTURE_MAP_NONE),
        }
    };
    for (asset_id, material) in material_assets.iter() {
        let Some(base_color_texture_id) = process_texture(&material.base_color_texture) else {
            continue;
        };
        let Some(normal_map_texture_id) = process_texture(&material.normal_map_texture) else {
            continue;
        };
        let Some(emissive_texture_id) = process_texture(&material.emissive_texture) else {
            continue;
        };
        let Some(metallic_roughness_texture_id) =
            process_texture(&material.metallic_roughness_texture)
        else {
            continue;
        };

        materials.get_mut().push(GpuMaterial {
            normal_map_texture_id,
            base_color_texture_id,
            emissive_texture_id,
            metallic_roughness_texture_id,

            base_color: LinearRgba::from(material.base_color).to_vec3(),
            perceptual_roughness: material.perceptual_roughness,
            emissive: material.emissive.to_vec3(),
            metallic: material.metallic,
            flags: if material.flip_normal_map_y {
                MATERIAL_FLAGS_FLIP_NORMAL_MAP_Y
            } else {
                0
            },
            reflectance: material.reflectance,
            _padding_a: 0,
            _padding_b: 0,
            uv_transform: material.uv_transform.into(),
        });

        material_id_map.insert(*asset_id, material_id);
        material_id += 1;
    }

    if material_id == 0 {
        return;
    }

    if textures.is_empty() {
        textures.vec.push(fallback_texture.d2.texture_view.deref());
        samplers.push(fallback_texture.d2.sampler.deref());
    }

    let mut instance_id = 0;
    let mut emissive_mesh_light_count = 0usize;
    // Bounds every instance, so it is what a rasterized surface falls back to when its G-buffer
    // texel carries no geometry error of its own.
    let mut max_world_geometry_error = 0.0f32;
    for (entity, mesh, geometry_error, material, transform, previous_frame_transform) in
        &instances_query
    {
        let Some(blas) = blas_manager.get(&mesh.id()) else {
            continue;
        };
        let Some(vertex_slice) = mesh_allocator.mesh_vertex_slice(&mesh.id()) else {
            continue;
        };
        let Some(index_slice) = mesh_allocator.mesh_index_slice(&mesh.id()) else {
            continue;
        };
        let Some(material_id) = material_id_map.get(&material.id()).copied() else {
            continue;
        };
        let Some(material) = materials.get().get(material_id as usize) else {
            continue;
        };

        *tlas.get_mut_single(instance_id).unwrap() = Some(TlasInstance::new(
            blas,
            tlas_transform(&transform.to_matrix()),
            Default::default(),
            0xFF,
        ));

        // Per instance, because a ray leaving this surface can only self-intersect this instance's
        // own simplified BLAS. The scene-wide maximum below is only the fallback for a rasterized
        // surface whose G-buffer texel states no error at all.
        let world_geometry_error = world_geometry_error(geometry_error.0, &transform.affine());
        let transform = Affine3::from(transform.affine()).to_transpose();
        transforms.get_mut().push(transform);
        previous_frame_transforms.get_mut().push(
            previous_frame_transform
                .map(|t| Affine3::from(t.0).to_transpose())
                .unwrap_or(transform),
        );

        let (vertex_buffer_id, _) = vertex_buffers.push_if_absent(
            vertex_slice.buffer.as_entire_buffer_binding(),
            vertex_slice.buffer.id(),
        );
        let (index_buffer_id, _) = index_buffers.push_if_absent(
            index_slice.buffer.as_entire_buffer_binding(),
            index_slice.buffer.id(),
        );

        max_world_geometry_error = max_world_geometry_error.max(world_geometry_error);
        geometry_ids.get_mut().push(GpuInstanceGeometryIds {
            vertex_buffer_id,
            vertex_buffer_offset: vertex_slice.range.start,
            index_buffer_id,
            index_buffer_offset: index_slice.range.start,
            triangle_count: (index_slice.range.len() / 3) as u32,
            vertex_stride_words: blas_manager
                .vertex_stride(&mesh.id())
                .expect("a bound BLAS must retain its input vertex stride")
                / 4,
            world_geometry_error,
            _padding: 0.0,
        });

        material_ids.get_mut().push(material_id);

        if material.emissive != Vec3::ZERO {
            let triangle_count = (index_slice.range.len() / 3) as u32;
            for (first_triangle, _) in emissive_triangle_chunks(triangle_count) {
                light_sources
                    .get_mut()
                    .push(GpuLightSource::new_emissive_mesh_light(
                        instance_id as u32,
                        first_triangle,
                    ));

                let light_key = LightKey::EmissiveMesh {
                    entity,
                    first_triangle,
                };
                this_frame_light_key_to_id.insert(light_key, light_sources.get().len() as u32 - 1);
                raytracing_scene_bindings
                    .previous_frame_light_keys
                    .push(light_key);
                emissive_mesh_light_count += 1;
            }
        }

        instance_id += 1;
    }

    if instance_id == 0 {
        return;
    }

    for (entity, directional_light) in &directional_lights_query {
        let directional_lights = directional_lights.get_mut();
        let directional_light_id = directional_lights.len() as u32;

        directional_lights.push(GpuDirectionalLight::new(directional_light));

        light_sources
            .get_mut()
            .push(GpuLightSource::new_directional_light(directional_light_id));

        let light_key = LightKey::Directional(entity);
        this_frame_light_key_to_id.insert(light_key, light_sources.get().len() as u32 - 1);
        raytracing_scene_bindings
            .previous_frame_light_keys
            .push(light_key);
    }

    let mut environment_light_count = 0usize;
    let mut environment_radiance = Vec3::ZERO;
    for (entity, environment_light) in &environment_lights_query {
        let directional_lights = directional_lights.get_mut();
        let directional_light_id = directional_lights.len() as u32;

        directional_lights.push(GpuDirectionalLight::new_environment(environment_light));
        light_sources
            .get_mut()
            .push(GpuLightSource::new_environment_light(directional_light_id));

        let light_key = LightKey::Environment(entity);
        this_frame_light_key_to_id.insert(light_key, light_sources.get().len() as u32 - 1);
        raytracing_scene_bindings
            .previous_frame_light_keys
            .push(light_key);
        environment_radiance += GpuDirectionalLight::environment_radiance(environment_light);
        environment_light_count += 1;
    }

    // Light ids are packed into 16 bits alongside a 16-bit triangle id, and index 65535 with
    // triangle 65535 would alias NULL_LIGHT_ID. Drop the overflow instead of failing the frame:
    // the instances still render, the extra emitters just stop being explicitly sampled.
    let mut sampled_environment_light_count = environment_light_count;
    if light_sources.get().len() > MAX_LIGHT_SOURCES {
        let dropped = light_sources.get().len() - MAX_LIGHT_SOURCES;
        // Environment entries are pushed last, so they are the first to be dropped. Whatever is
        // left is what NEE can still reach, and the shaders MIS against exactly that count.
        sampled_environment_light_count = environment_light_count.saturating_sub(dropped);
        light_sources.get_mut().truncate(MAX_LIGHT_SOURCES);
        raytracing_scene_bindings
            .previous_frame_light_keys
            .truncate(MAX_LIGHT_SOURCES);
        this_frame_light_key_to_id.retain(|_, id| (*id as usize) < MAX_LIGHT_SOURCES);
        if !raytracing_scene_bindings.reported_light_source_overflow {
            error!(
                dropped,
                maximum = MAX_LIGHT_SOURCES,
                "too many light sources in the scene; the excess will not be sampled"
            );
            raytracing_scene_bindings.reported_light_source_overflow = true;
        }
    }

    let scene_summary = (
        instance_id,
        emissive_mesh_light_count,
        directional_lights_query.iter().len(),
        environment_light_count,
        light_sources.get().len(),
    );
    if raytracing_scene_bindings.last_scene_summary == Some(scene_summary) {
        raytracing_scene_bindings.scene_summary_stable_frames = raytracing_scene_bindings
            .scene_summary_stable_frames
            .saturating_add(1);
    } else {
        raytracing_scene_bindings.last_scene_summary = Some(scene_summary);
        raytracing_scene_bindings.scene_summary_stable_frames = 0;
    }
    if raytracing_scene_bindings.scene_summary_stable_frames >= 2
        && raytracing_scene_bindings.reported_scene_summary != Some(scene_summary)
    {
        info!(
            raytracing_instances = scene_summary.0,
            emissive_mesh_lights = scene_summary.1,
            directional_lights = scene_summary.2,
            environment_lights = scene_summary.3,
            total_light_sources = scene_summary.4,
            "prepared Solari raytracing scene"
        );
        raytracing_scene_bindings.reported_scene_summary = Some(scene_summary);
    }

    for previous_frame_light_key in previous_frame_light_keys {
        let current_frame_index = this_frame_light_key_to_id
            .get(&previous_frame_light_key)
            .copied()
            .unwrap_or(LIGHT_NOT_PRESENT_THIS_FRAME);
        previous_frame_light_id_translations
            .get_mut()
            .push(current_frame_index);
    }

    materials.write_buffer(&render_device, &render_queue);
    transforms.write_buffer(&render_device, &render_queue);
    previous_frame_transforms.write_buffer(&render_device, &render_queue);
    geometry_ids.write_buffer(&render_device, &render_queue);
    material_ids.write_buffer(&render_device, &render_queue);
    light_sources.write_buffer(&render_device, &render_queue);
    directional_lights.write_buffer(&render_device, &render_queue);
    previous_frame_light_id_translations.write_buffer(&render_device, &render_queue);
    let scene_parameter_values = scene_parameters.get_mut();
    scene_parameter_values.max_world_geometry_error = max_world_geometry_error;
    scene_parameter_values.environment_radiance = environment_radiance;
    scene_parameter_values.inverse_environment_light_pdf = if sampled_environment_light_count == 0 {
        0.0
    } else {
        TAU * light_sources.get().len() as f32 / sampled_environment_light_count as f32
    };
    scene_parameters.write_buffer(&render_device, &render_queue);

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("tlas_build_command_encoder"),
    });
    let time_span = diagnostics
        .as_mut()
        .map(|diagnostics| diagnostics.time_span(&mut command_encoder, "tlas_build"));
    command_encoder.build_acceleration_structures(&[], [&tlas]);
    if let Some(time_span) = time_span {
        time_span.end(&mut command_encoder);
    }
    render_queue.submit([command_encoder.finish()]);

    let (dfg_view, dfg_sampler) = texture_assets
        .get(&dfg_lut.texture)
        .map(|img| (&img.texture_view, &img.sampler))
        .unwrap_or((
            &fallback_texture.d2.texture_view,
            &fallback_texture.d2.sampler,
        ));

    raytracing_scene_bindings.bind_group = Some(render_device.create_bind_group(
        "raytracing_scene_bind_group",
        &pipeline_cache.get_bind_group_layout(&raytracing_scene_bindings.bind_group_layout),
        &BindGroupEntries::sequential((
            vertex_buffers.as_slice(),
            index_buffers.as_slice(),
            textures.as_slice(),
            samplers.as_slice(),
            materials.binding().unwrap(),
            tlas.as_binding(),
            previous_frame_tlas.as_ref().unwrap_or(&tlas).as_binding(),
            transforms.binding().unwrap(),
            previous_frame_transforms.binding().unwrap(),
            geometry_ids.binding().unwrap(),
            material_ids.binding().unwrap(),
            light_sources.binding().unwrap(),
            directional_lights.binding().unwrap(),
            previous_frame_light_id_translations.binding().unwrap(),
            dfg_view,
            dfg_sampler,
            scene_parameters.binding().unwrap(),
        )),
    ));

    raytracing_scene_bindings.previous_frame_tlas = Some(tlas);
}

impl RaytracingSceneBindings {
    pub fn new() -> Self {
        Self {
            bind_group: None,
            bind_group_layout: BindGroupLayoutDescriptor::new(
                "raytracing_scene_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        storage_buffer_read_only_sized(false, None).count(MAX_MESH_SLAB_COUNT),
                        storage_buffer_read_only_sized(false, None).count(MAX_MESH_SLAB_COUNT),
                        texture_2d(TextureSampleType::Float { filterable: true })
                            .count(MAX_TEXTURE_COUNT),
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
            ),
            previous_frame_tlas: None,
            previous_frame_light_keys: Vec::new(),
            last_scene_summary: None,
            scene_summary_stable_frames: 0,
            reported_scene_summary: None,
            reported_light_source_overflow: false,
        }
    }
}

impl Default for RaytracingSceneBindings {
    fn default() -> Self {
        Self::new()
    }
}

struct CachedBindingArray<T, I: Eq + Hash> {
    map: HashMap<I, u32>,
    vec: Vec<T>,
}

impl<T, I: Eq + Hash> CachedBindingArray<T, I> {
    fn new() -> Self {
        Self {
            map: HashMap::default(),
            vec: Vec::default(),
        }
    }

    fn push_if_absent(&mut self, item: T, item_id: I) -> (u32, bool) {
        let mut is_new = false;
        let i = *self.map.entry(item_id).or_insert_with(|| {
            is_new = true;
            let i = self.vec.len() as u32;
            self.vec.push(item);
            i
        });
        (i, is_new)
    }

    fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    fn as_slice(&self) -> &[T] {
        self.vec.as_slice()
    }
}

type StorageBufferList<T> = StorageBuffer<Vec<T>>;

#[derive(ShaderType)]
struct GpuInstanceGeometryIds {
    vertex_buffer_id: u32,
    vertex_buffer_offset: u32,
    index_buffer_id: u32,
    index_buffer_offset: u32,
    triangle_count: u32,
    vertex_stride_words: u32,
    world_geometry_error: f32,
    _padding: f32,
}

#[derive(ShaderType, Default)]
struct GpuSceneParameters {
    environment_radiance: Vec3,
    max_world_geometry_error: f32,
    inverse_environment_light_pdf: f32,
    _padding: Vec3,
}

#[cfg(test)]
mod tests {
    use super::{
        emissive_triangle_chunks, GpuDirectionalLight, GpuLightSource, RaytracingSceneBindings,
        MATERIAL_FLAGS_FLIP_NORMAL_MAP_Y,
    };
    use crate::scene::SolariEnvironmentLight;
    use bevy_color::LinearRgba;
    use bevy_math::Vec3;
    use bevy_render::render_resource::{BindingType, BufferBindingType};

    #[test]
    fn raytracing_scene_binding_arrays_do_not_share_a_group_with_uniform_buffers() {
        let layout = RaytracingSceneBindings::new().bind_group_layout;
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
    fn environment_illuminance_converts_to_uniform_hemisphere_radiance() {
        let light = SolariEnvironmentLight {
            color: LinearRgba::new(1.0, 0.5, 0.25, 1.0),
            illuminance: core::f32::consts::PI,
        };
        let gpu = GpuDirectionalLight::new_environment(&light);

        assert_eq!(gpu.direction_to_light, Vec3::Y);
        assert_eq!(gpu.cos_theta_max, 0.0);
        assert!(gpu.luminance.abs_diff_eq(Vec3::new(1.0, 0.5, 0.25), 1e-6));
        assert_eq!(gpu.inverse_pdf, core::f32::consts::TAU);
    }

    #[test]
    fn emissive_triangle_chunks_preserve_every_triangle() {
        let cases = [
            (0, vec![]),
            (1, vec![(0, 1)]),
            (65_535, vec![(0, 65_535)]),
            (65_536, vec![(0, 65_535), (65_535, 1)]),
            (109_512, vec![(0, 65_535), (65_535, 43_977)]),
            (131_070, vec![(0, 65_535), (65_535, 65_535)]),
        ];

        for (triangle_count, expected) in cases {
            assert_eq!(
                emissive_triangle_chunks(triangle_count).collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn emissive_light_encodes_chunk_offset_without_changing_instance() {
        let light = GpuLightSource::new_emissive_mesh_light(42, 65_535);
        assert_eq!(light.kind, 65_535 << 1);
        assert_eq!(light.id, 42);
    }

    #[test]
    fn light_source_kinds_and_material_flags_match_the_shader_constants() {
        let shader = include_str!("raytracing_scene_bindings.wgsl");

        let environment = GpuLightSource::new_environment_light(7);
        assert_eq!(environment.id, 7);
        assert!(shader.contains(&format!(
            "const LIGHT_SOURCE_KIND_ENVIRONMENT = {}u;",
            environment.kind
        )));
        // The environment kind must still read as non-emissive-mesh in the shader's low-bit test.
        assert_eq!(environment.kind & 1, 1);
        assert_eq!(GpuLightSource::new_directional_light(0).kind & 1, 1);
        assert_eq!(GpuLightSource::new_emissive_mesh_light(0, 0).kind & 1, 0);

        assert!(shader.contains(&format!(
            "const MATERIAL_FLAGS_FLIP_NORMAL_MAP_Y = {MATERIAL_FLAGS_FLIP_NORMAL_MAP_Y}u;"
        )));
    }

    #[test]
    fn only_a_rasterized_surface_pays_the_shading_normal_safety_factor() {
        // Both bias sites resolve an unknown error through the same bound, so the scene-wide maximum
        // cannot end up less conservative than the per-instance path of the instance that set it.
        // Only the rasterized one is offset along a normal-mapped normal, so only it needs the
        // factor; a ray hit carries a true geometric normal.
        let shader = include_str!("raytracing_scene_bindings.wgsl");
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

#[derive(ShaderType)]
struct GpuMaterial {
    normal_map_texture_id: u32,
    base_color_texture_id: u32,
    emissive_texture_id: u32,
    metallic_roughness_texture_id: u32,

    base_color: Vec3,
    perceptual_roughness: f32,
    emissive: Vec3,
    metallic: f32,
    flags: u32,
    _padding_a: u32,
    _padding_b: u32,
    reflectance: f32,
    uv_transform: Mat3,
}

/// Matches `MATERIAL_FLAGS_FLIP_NORMAL_MAP_Y` in `raytracing_scene_bindings.wgsl`.
const MATERIAL_FLAGS_FLIP_NORMAL_MAP_Y: u32 = 1;

#[derive(ShaderType)]
struct GpuLightSource {
    kind: u32,
    id: u32,
}

impl GpuLightSource {
    fn new_emissive_mesh_light(instance_id: u32, first_triangle: u32) -> GpuLightSource {
        assert!(
            first_triangle <= u32::MAX >> 1,
            "emissive light triangle offset exceeds its 31-bit encoding"
        );
        Self {
            kind: first_triangle << 1,
            id: instance_id,
        }
    }

    fn new_directional_light(directional_light_id: u32) -> GpuLightSource {
        Self {
            kind: 1,
            id: directional_light_id,
        }
    }

    /// Resolves like a directional light, but flagged so the shaders can MIS-weight it against the
    /// environment radiance a missed BRDF ray picks up.
    fn new_environment_light(directional_light_id: u32) -> GpuLightSource {
        Self {
            kind: 3,
            id: directional_light_id,
        }
    }
}

fn emissive_triangle_chunks(triangle_count: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..triangle_count)
        .step_by(MAX_EMISSIVE_TRIANGLES_PER_LIGHT as usize)
        .map(move |first_triangle| {
            (
                first_triangle,
                (triangle_count - first_triangle).min(MAX_EMISSIVE_TRIANGLES_PER_LIGHT),
            )
        })
}

#[derive(ShaderType, Default)]
struct GpuDirectionalLight {
    direction_to_light: Vec3,
    cos_theta_max: f32,
    luminance: Vec3,
    inverse_pdf: f32,
}

impl GpuDirectionalLight {
    fn new(directional_light: &ExtractedDirectionalLight) -> Self {
        let cos_theta_max = cos(directional_light.sun_disk_angular_size / 2.0);
        let solid_angle = TAU * (1.0 - cos_theta_max);
        let luminance =
            (directional_light.color.to_vec3() * directional_light.illuminance) / solid_angle;

        Self {
            direction_to_light: directional_light.transform.back().into(),
            cos_theta_max,
            luminance,
            inverse_pdf: solid_angle,
        }
    }

    fn new_environment(environment_light: &SolariEnvironmentLight) -> Self {
        Self {
            direction_to_light: Vec3::Y,
            // A cone with a 90-degree half-angle is the world +Y hemisphere.
            cos_theta_max: 0.0,
            // Uniform hemispheric radiance produces E = PI * L on a
            // horizontal surface.
            luminance: Self::environment_radiance(environment_light),
            inverse_pdf: TAU,
        }
    }

    fn environment_radiance(environment_light: &SolariEnvironmentLight) -> Vec3 {
        environment_light.color.to_vec3()
            * (environment_light.illuminance.max(0.0) / core::f32::consts::PI)
    }
}

fn tlas_transform(transform: &Mat4) -> [f32; 12] {
    transform.transpose().to_cols_array()[..12]
        .try_into()
        .unwrap()
}
