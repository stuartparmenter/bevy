//! Render high-poly 3d meshes using an efficient GPU-driven method. See [`MeshletPlugin`] and [`MeshletMesh`] for details.

mod asset;
#[cfg(feature = "meshlet_processor")]
mod from_mesh;
mod instance_manager;
mod material_pipeline_prepare;
mod material_shade_nodes;
mod meshlet_mesh_manager;
mod pipelines;
mod resource_manager;
mod visibility_buffer_raster_node;

pub(crate) use self::{
    instance_manager::{queue_material_meshlet_meshes, InstanceManager},
    material_pipeline_prepare::{
        prepare_material_meshlet_meshes_main_opaque_pass, prepare_material_meshlet_meshes_prepass,
    },
};

#[cfg(feature = "meshlet_processor")]
pub use self::asset::MeshletRaytracingGeometry;
pub use self::asset::{
    MeshletMesh, MeshletMeshLoader, MeshletMeshSaveOrLoadError, MeshletMeshSaver,
    MESHLET_MESH_ASSET_VERSION,
};
#[cfg(feature = "meshlet_processor")]
pub use self::from_mesh::{
    quantize_vertex_position, vertex_position_quantization_scale, MeshToMeshletMeshConversionError,
    MESHLET_DEFAULT_VERTEX_POSITION_QUANTIZATION_FACTOR,
};
use self::{
    instance_manager::extract_meshlet_mesh_entities,
    material_pipeline_prepare::{
        MeshletViewMaterialsDeferredGBufferPrepass, MeshletViewMaterialsMainOpaquePass,
        MeshletViewMaterialsPrepass,
    },
    material_shade_nodes::{
        meshlet_deferred_gbuffer_prepass, meshlet_main_opaque_pass, meshlet_prepass,
    },
    meshlet_mesh_manager::perform_pending_meshlet_mesh_writes,
    pipelines::*,
    resource_manager::{
        prepare_meshlet_per_frame_resources, prepare_meshlet_view_bind_groups, ResourceManager,
    },
    visibility_buffer_raster_node::meshlet_visibility_buffer_raster,
};
use crate::render::{per_view_shadow_pass, EARLY_SHADOW_PASS};
use crate::{meshlet::meshlet_mesh_manager::init_meshlet_mesh_manager, PreviousGlobalTransform};
use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, AssetApp, AssetId, Handle};
use bevy_camera::visibility::{self, Visibility, VisibilityClass};
use bevy_core_pipeline::{
    core_3d::main_opaque_pass_3d,
    deferred::node::late_deferred_prepass,
    prepass::node::early_prepass,
    prepass::{DeferredPrepass, MotionVectorPrepass, NormalPrepass},
    schedule::{Core3d, Core3dSystems},
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::Has,
    reflect::ReflectComponent,
    schedule::IntoScheduleConfigs,
    system::{Commands, Query, Res},
    template::FromTemplate,
};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    renderer::RenderDevice,
    settings::WgpuFeatures,
    view::{prepare_view_targets, Msaa},
    ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::load_shader_library;
use bevy_transform::components::Transform;
use derive_more::From;
use tracing::error;

/// Provides a plugin for rendering large amounts of high-poly 3d meshes using an efficient GPU-driven method. See also [`MeshletMesh`].
///
/// Rendering dense scenes made of high-poly meshes with thousands or millions of triangles is extremely expensive in Bevy's standard renderer.
/// Once meshes are pre-processed into a [`MeshletMesh`], this plugin can render these kinds of scenes very efficiently.
///
/// In comparison to Bevy's standard renderer:
/// * Much more efficient culling. Meshlets can be culled individually, instead of all or nothing culling for entire meshes at a time.
///   Additionally, occlusion culling can eliminate meshlets that would cause overdraw.
/// * Much more efficient batching. All geometry can be rasterized in a single draw.
/// * Scales better with large amounts of dense geometry and overdraw. Bevy's standard renderer will bottleneck sooner.
/// * Near-seamless level of detail (LOD).
/// * Much greater base overhead. Rendering will be slower and use more memory than Bevy's standard renderer
///   with small amounts of geometry and overdraw.
/// * Requires preprocessing meshes. See [`MeshletMesh`] for details.
/// * Limitations on the kinds of materials you can use. See [`MeshletMesh`] for details.
///
/// This plugin requires a fairly recent GPU that supports [`WgpuFeatures::TEXTURE_INT64_ATOMIC`]
/// and non-uniform storage-buffer binding arrays. Mesh data is stored in fixed 64 MiB pages so
/// large scenes never require a monolithic storage buffer or a whole-heap reallocation.
///
/// This plugin currently works only on the Vulkan and Metal backends.
///
/// This plugin is not compatible with [`Msaa`]. Any camera rendering a [`MeshletMesh`] must have
/// [`Msaa`] set to [`Msaa::Off`].
///
/// Mixing forward+prepass and deferred rendering for opaque materials is not currently supported when using this plugin.
/// You must use one or the other by setting [`crate::DefaultOpaqueRendererMethod`].
/// Do not override [`crate::Material::opaque_render_method`] for any material when using this plugin.
///
/// ![A render of the Stanford dragon as a `MeshletMesh`](https://raw.githubusercontent.com/bevyengine/bevy/main/crates/bevy_pbr/src/meshlet/meshlet_preview.png)
pub struct MeshletPlugin {
    /// The maximum amount of clusters that can be processed at once,
    /// used to control the size of a pre-allocated GPU buffer.
    ///
    /// If this number is too low, you'll see rendering artifacts like missing or blinking meshes.
    ///
    /// Each cluster slot costs 4 bytes of VRAM.
    ///
    /// Must not be greater than 2^25.
    pub cluster_buffer_slots: u32,
}

impl MeshletPlugin {
    /// [`WgpuFeatures`] required for this plugin to function.
    pub fn required_wgpu_features() -> WgpuFeatures {
        WgpuFeatures::TEXTURE_INT64_ATOMIC
            | WgpuFeatures::TEXTURE_ATOMIC
            | WgpuFeatures::SHADER_INT64
            | WgpuFeatures::SUBGROUP
            | WgpuFeatures::DEPTH_CLIP_CONTROL
            | WgpuFeatures::IMMEDIATES
            | WgpuFeatures::BUFFER_BINDING_ARRAY
            | WgpuFeatures::STORAGE_RESOURCE_BINDING_ARRAY
            | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
    }
}

impl Plugin for MeshletPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(target_endian = "big")]
        compile_error!("MeshletPlugin is only supported on little-endian processors.");

        if self.cluster_buffer_slots > 2_u32.pow(25) {
            error!("MeshletPlugin::cluster_buffer_slots must not be greater than 2^25.");
            std::process::exit(1);
        }

        load_shader_library!(app, "meshlet_bindings.wgsl");
        load_shader_library!(app, "visibility_buffer_resolve.wgsl");
        load_shader_library!(app, "meshlet_cull_shared.wgsl");
        embedded_asset!(app, "clear_visibility_buffer.wgsl");
        embedded_asset!(app, "cull_instances.wgsl");
        embedded_asset!(app, "cull_bvh.wgsl");
        embedded_asset!(app, "cull_clusters.wgsl");
        embedded_asset!(app, "visibility_buffer_software_raster.wgsl");
        embedded_asset!(app, "visibility_buffer_hardware_raster.wgsl");
        embedded_asset!(app, "meshlet_mesh_material.wgsl");
        embedded_asset!(app, "resolve_render_targets.wgsl");
        embedded_asset!(app, "remap_1d_to_2d_dispatch.wgsl");
        embedded_asset!(app, "fill_counts.wgsl");

        app.init_asset::<MeshletMesh>()
            .register_asset_loader(MeshletMeshLoader);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        // Create a variable here so we can move-capture it.
        let cluster_buffer_slots = self.cluster_buffer_slots;
        let init_resource_manager_system =
            move |mut commands: Commands, render_device: Res<RenderDevice>| {
                commands
                    .insert_resource(ResourceManager::new(cluster_buffer_slots, &render_device));
            };

        render_app
            .insert_resource(InstanceManager::new())
            .add_systems(
                RenderStartup,
                (
                    check_meshlet_features,
                    (
                        (init_resource_manager_system, init_meshlet_pipelines).chain(),
                        init_meshlet_mesh_manager,
                    ),
                )
                    .chain(),
            )
            .add_systems(ExtractSchedule, extract_meshlet_mesh_entities)
            .add_systems(
                Render,
                (
                    perform_pending_meshlet_mesh_writes.in_set(RenderSystems::PrepareAssets),
                    configure_meshlet_views
                        .after(prepare_view_targets)
                        .in_set(RenderSystems::PrepareViews),
                    prepare_meshlet_per_frame_resources.in_set(RenderSystems::PrepareResources),
                    prepare_meshlet_view_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                    queue_material_meshlet_meshes.in_set(RenderSystems::QueueMeshes),
                    prepare_material_meshlet_meshes_main_opaque_pass
                        .in_set(RenderSystems::QueueMeshes)
                        .before(queue_material_meshlet_meshes),
                ),
            )
            .add_systems(
                Core3d,
                (
                    meshlet_visibility_buffer_raster
                        .before(early_prepass)
                        .before(per_view_shadow_pass::<EARLY_SHADOW_PASS>),
                    meshlet_prepass
                        .after(per_view_shadow_pass::<EARLY_SHADOW_PASS>)
                        .after(late_deferred_prepass)
                        .in_set(Core3dSystems::Prepass),
                    meshlet_deferred_gbuffer_prepass
                        .after(meshlet_prepass)
                        .in_set(Core3dSystems::Prepass),
                    meshlet_main_opaque_pass
                        .before(main_opaque_pass_3d)
                        .in_set(Core3dSystems::MainPass),
                ),
            );
    }
}

fn check_meshlet_features(render_device: Res<RenderDevice>) {
    let features = render_device.features();
    if !features.contains(MeshletPlugin::required_wgpu_features()) {
        error!(
            "MeshletPlugin can't be used. GPU lacks support for required features: {:?}.",
            MeshletPlugin::required_wgpu_features().difference(features)
        );
        std::process::exit(1);
    }
}

/// The meshlet mesh equivalent of [`bevy_mesh::Mesh3d`].
#[derive(
    Component, FromTemplate, Clone, Debug, Default, Deref, DerefMut, Reflect, PartialEq, Eq, From,
)]
#[reflect(Component, Default, Clone, PartialEq)]
#[require(Transform, PreviousGlobalTransform, Visibility, VisibilityClass)]
#[component(on_add = visibility::add_visibility_class::<MeshletMesh3d>)]
pub struct MeshletMesh3d(pub Handle<MeshletMesh>);

impl From<MeshletMesh3d> for AssetId<MeshletMesh> {
    fn from(mesh: MeshletMesh3d) -> Self {
        mesh.id()
    }
}

impl From<&MeshletMesh3d> for AssetId<MeshletMesh> {
    fn from(mesh: &MeshletMesh3d) -> Self {
        mesh.id()
    }
}

fn configure_meshlet_views(
    mut views_3d: Query<(
        Entity,
        &Msaa,
        Has<NormalPrepass>,
        Has<MotionVectorPrepass>,
        Has<DeferredPrepass>,
    )>,
    mut commands: Commands,
) {
    for (entity, msaa, normal_prepass, motion_vector_prepass, deferred_prepass) in &mut views_3d {
        if *msaa != Msaa::Off {
            error!("MeshletPlugin can't be used with MSAA. Add Msaa::Off to your camera to use this plugin.");
            std::process::exit(1);
        }

        if !(normal_prepass || motion_vector_prepass || deferred_prepass) {
            commands
                .entity(entity)
                .insert(MeshletViewMaterialsMainOpaquePass::default());
        } else {
            // TODO: Should we add both Prepass and DeferredGBufferPrepass materials here, and in other systems/nodes?
            commands.entity(entity).insert((
                MeshletViewMaterialsMainOpaquePass::default(),
                MeshletViewMaterialsPrepass::default(),
                MeshletViewMaterialsDeferredGBufferPrepass::default(),
            ));
        }
    }
}

#[cfg(test)]
mod shader_validation_tests {
    use bevy_shader::Shader;
    use naga_oil::compose::{Composer, NagaModuleDescriptor, ShaderDefValue};
    use std::collections::HashMap;

    fn add_stub_module(composer: &mut Composer, source: &'static str, name: &'static str) {
        let shader = Shader::from_wgsl(source, name);
        composer.add_composable_module((&shader).into()).unwrap();
    }

    #[test]
    fn paged_meshlet_addressing_passes_naga_validation() {
        let mut composer = Composer::default().with_capabilities(naga::valid::Capabilities::all());
        add_stub_module(
            &mut composer,
            concat!(
                "#define_import_path bevy_pbr::mesh_types\n",
                "struct Mesh {\n",
                "    world_from_local: mat3x4<f32>,\n",
                "    previous_world_from_local: mat3x4<f32>,\n",
                "    local_from_world_transpose_a: mat2x4<f32>,\n",
                "    local_from_world_transpose_b: f32,\n",
                "    flags: u32,\n",
                "    material_and_lightmap_bind_group_slot: u32,\n",
                "}\n",
                "const MESH_FLAGS_SIGN_DETERMINANT_MODEL_3X3_BIT: u32 = 1u << 31u;\n",
            ),
            "mesh_types_stub.wgsl",
        );
        add_stub_module(
            &mut composer,
            r#"
#define_import_path bevy_render::view
struct View {
    clip_from_world: mat4x4<f32>,
    main_pass_viewport: vec4<f32>,
    world_position: vec3<f32>,
}

fn frag_coord_to_ndc(frag_coord: vec4<f32>, viewport: vec4<f32>) -> vec3<f32> {
    return vec3(frag_coord.xy / viewport.zw * 2.0 - 1.0, frag_coord.z);
}
"#,
            "view_stub.wgsl",
        );
        add_stub_module(
            &mut composer,
            concat!(
                "#define_import_path bevy_pbr::prepass_bindings\n",
                "struct PreviousViewUniforms { value: u32 }\n",
                // Group 3 is unused by every meshlet pass, so this cannot collide.
                "@group(3) @binding(0) var<uniform> previous_view_uniforms: PreviousViewUniforms;\n",
            ),
            "previous_view_stub.wgsl",
        );
        add_stub_module(
            &mut composer,
            concat!(
                "#define_import_path bevy_pbr::pbr_prepass_functions\n",
                "fn calculate_motion_vector(world_position: vec4<f32>, previous_world_position: vec4<f32>) -> vec2<f32> {\n",
                "    return world_position.xy - previous_world_position.xy;\n",
                "}\n",
            ),
            "pbr_prepass_functions_stub.wgsl",
        );
        add_stub_module(
            &mut composer,
            r#"
#define_import_path bevy_render::utils
fn octahedral_decode_signed(value: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(value, 0.0);
}
"#,
            "render_utils_stub.wgsl",
        );
        add_stub_module(
            &mut composer,
            r#"
#define_import_path bevy_render::maths
fn affine3_to_square(_affine: mat3x4<f32>) -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
}

fn mat2x4_f32_to_mat3x3_unpack(a: mat2x4<f32>, b: f32) -> mat3x3<f32> {
    return mat3x3<f32>(a[0].xyz, vec3<f32>(a[0].w, a[1].xy), vec3<f32>(a[1].zw, b));
}
"#,
            "render_maths_stub.wgsl",
        );
        add_stub_module(
            &mut composer,
            r#"
#define_import_path bevy_pbr::mesh_functions
fn mesh_position_local_to_world(world_from_local: mat4x4<f32>, position: vec4<f32>) -> vec4<f32> {
    return world_from_local * position;
}
"#,
            "mesh_functions_stub.wgsl",
        );
        add_stub_module(
            &mut composer,
            r#"
#define_import_path bevy_pbr::view_transformations
fn ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return ndc * vec2<f32>(0.5, -0.5) + 0.5;
}

fn position_world_to_clip(world_pos: vec3<f32>) -> vec4<f32> {
    return vec4(world_pos, 1.0);
}
"#,
            "view_transformations_stub.wgsl",
        );
        let bindings = Shader::from_wgsl(
            include_str!("meshlet_bindings.wgsl"),
            "meshlet_bindings.wgsl",
        );
        composer.add_composable_module((&bindings).into()).unwrap();
        add_stub_module(
            &mut composer,
            r#"
#define_import_path bevy_pbr::meshlet_cull_shared
#import bevy_pbr::meshlet_bindings::MeshletAabb

fn lod_error_is_imperceptible(_sphere: vec4<f32>, _error: f32, _instance_id: u32) -> bool {
    return false;
}

fn aabb_in_frustum(_aabb: MeshletAabb, _instance_id: u32) -> bool {
    return true;
}

fn should_occlusion_cull_aabb(_aabb: MeshletAabb, _instance_id: u32) -> bool {
    return false;
}
"#,
            "meshlet_cull_shared_stub.wgsl",
        );

        for pass_def in [
            "MESHLET_BVH_CULLING_PASS",
            "MESHLET_CLUSTER_CULLING_PASS",
            "MESHLET_VISIBILITY_BUFFER_RASTER_PASS",
            "MESHLET_MESH_MATERIAL_PASS",
        ] {
            let test_shader = Shader::from_wgsl(
                r#"
enable wgpu_binding_array;

#import bevy_pbr::meshlet_bindings::{
    MeshletGpuDescriptor,
    get_meshlet_vertex_id,
    get_meshlet_vertex_position,
    get_meshlet_vertex_normal,
    get_meshlet_vertex_uv,
    load_bvh_subnode,
    load_meshlet,
    load_meshlet_geometry,
    load_meshlet_cull_data,
}

@compute @workgroup_size(1)
fn validate_paged_loads() {
    let descriptor = MeshletGpuDescriptor(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    var meshlet = load_meshlet(descriptor, 0u);
    var geometry_meshlet = load_meshlet_geometry(descriptor, 0u);
    let vertex_id = get_meshlet_vertex_id(descriptor, meshlet.start_index_id);
    let position = get_meshlet_vertex_position(descriptor, &geometry_meshlet, vertex_id);
    let normal = get_meshlet_vertex_normal(descriptor, &meshlet, vertex_id);
    let uv = get_meshlet_vertex_uv(descriptor, &meshlet, vertex_id);
    let bvh = load_bvh_subnode(descriptor, u32(position.x), 0u);
    let cull = load_meshlet_cull_data(descriptor, u32(normal.x + uv.x));
}
"#,
                "paged_meshlet_validation.wgsl",
            );
            let shader_defs = HashMap::from([
                (
                    "MESHLET_PAGE_ACCESS".to_string(),
                    ShaderDefValue::Bool(true),
                ),
                (pass_def.to_string(), ShaderDefValue::Bool(true)),
            ]);
            composer
                .make_naga_module(NagaModuleDescriptor {
                    shader_defs,
                    ..(&test_shader).into()
                })
                .unwrap_or_else(|error| panic!("{pass_def} failed Naga validation: {error:?}"));
        }

        // Compile an actual page-consuming entry point in both of its pipeline variants. The
        // culling policy is stubbed because it is orthogonal; all real bindings, descriptor
        // address translation, queue accesses, and BVH page loads remain intact. Naga validates
        // composition and types here; bind-group creation and Vulkan non-uniform indexing still
        // require the Windows/device integration run.
        let cull_bvh = Shader::from_wgsl(include_str!("cull_bvh.wgsl"), "cull_bvh.wgsl");
        for first_pass in [false, true] {
            let mut shader_defs = HashMap::from([
                (
                    "MESHLET_PAGE_ACCESS".to_string(),
                    ShaderDefValue::Bool(true),
                ),
                (
                    "MESHLET_BVH_CULLING_PASS".to_string(),
                    ShaderDefValue::Bool(true),
                ),
            ]);
            if first_pass {
                shader_defs.insert(
                    "MESHLET_FIRST_CULLING_PASS".to_string(),
                    ShaderDefValue::Bool(true),
                );
            }
            composer
                .make_naga_module(NagaModuleDescriptor {
                    shader_defs,
                    ..(&cull_bvh).into()
                })
                .unwrap_or_else(|error| {
                    panic!("cull_bvh first_pass={first_pass} failed Naga validation: {error:?}")
                });
        }

        let cull_instances =
            Shader::from_wgsl(include_str!("cull_instances.wgsl"), "cull_instances.wgsl");
        for first_pass in [false, true] {
            let mut shader_defs = HashMap::from([(
                "MESHLET_INSTANCE_CULLING_PASS".to_string(),
                ShaderDefValue::Bool(true),
            )]);
            if first_pass {
                shader_defs.insert(
                    "MESHLET_FIRST_CULLING_PASS".to_string(),
                    ShaderDefValue::Bool(true),
                );
            }
            composer
                .make_naga_module(NagaModuleDescriptor {
                    shader_defs,
                    ..(&cull_instances).into()
                })
                .unwrap_or_else(|error| {
                    panic!(
                        "cull_instances first_pass={first_pass} failed Naga validation: {error:?}"
                    )
                });
        }

        for (name, source) in [
            (
                "visibility_buffer_software_raster.wgsl",
                include_str!("visibility_buffer_software_raster.wgsl"),
            ),
            (
                "visibility_buffer_hardware_raster.wgsl",
                include_str!("visibility_buffer_hardware_raster.wgsl"),
            ),
        ] {
            let raster = Shader::from_wgsl(source, name);
            for full_output in [false, true] {
                let mut shader_defs = HashMap::from([
                    (
                        "MESHLET_PAGE_ACCESS".to_string(),
                        ShaderDefValue::Bool(true),
                    ),
                    (
                        "MESHLET_VISIBILITY_BUFFER_RASTER_PASS".to_string(),
                        ShaderDefValue::Bool(true),
                    ),
                ]);
                if full_output {
                    shader_defs.insert(
                        "MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT".to_string(),
                        ShaderDefValue::Bool(true),
                    );
                }
                composer
                    .make_naga_module(NagaModuleDescriptor {
                        shader_defs,
                        ..(&raster).into()
                    })
                    .unwrap_or_else(|error| {
                        panic!("{name} full_output={full_output} failed Naga validation: {error:?}")
                    });
            }
        }

        // The material-pass resolve reconstructs the triangle, its facing bit and its texture
        // gradients from the visibility buffer. It has no entry point of its own, so drive it
        // from a throwaway fragment shader that reads every field of the resolved VertexOutput.
        add_stub_module(
            &mut composer,
            concat!(
                "#define_import_path bevy_pbr::mesh_view_bindings\n",
                "#import bevy_render::view::View\n",
                "@group(0) @binding(0) var<uniform> view: View;\n",
            ),
            "mesh_view_bindings_stub.wgsl",
        );
        let resolve = Shader::from_wgsl(
            include_str!("visibility_buffer_resolve.wgsl"),
            "visibility_buffer_resolve.wgsl",
        );
        composer.add_composable_module((&resolve).into()).unwrap();
        for motion_vectors in [false, true] {
            let motion_vector_read = if motion_vectors {
                "+ out.motion_vector.x"
            } else {
                ""
            };
            let resolve_consumer = Shader::from_wgsl(
                format!(
                    r#"
enable wgpu_binding_array;

#import bevy_pbr::meshlet_visibility_buffer_resolve::resolve_vertex_output

@fragment
fn fragment(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {{
    let out = resolve_vertex_output(frag_coord);
    return vec4(
        out.position.xyz + out.world_position.xyz + out.world_normal + out.world_tangent.xyz,
        f32(out.is_front)
            + f32(out.mesh_flags + out.cluster_id + out.material_bind_group_slot)
            + out.uv.x + out.ddx_uv.y + out.ddy_uv.x {motion_vector_read},
    );
}}
"#
                ),
                "visibility_buffer_resolve_validation.wgsl",
            );
            let mut shader_defs = HashMap::from([
                (
                    "MESHLET_PAGE_ACCESS".to_string(),
                    ShaderDefValue::Bool(true),
                ),
                (
                    "MESHLET_MESH_MATERIAL_PASS".to_string(),
                    ShaderDefValue::Bool(true),
                ),
            ]);
            if motion_vectors {
                shader_defs.insert("PREPASS_FRAGMENT".to_string(), ShaderDefValue::Bool(true));
                shader_defs.insert(
                    "MOTION_VECTOR_PREPASS".to_string(),
                    ShaderDefValue::Bool(true),
                );
            }
            composer
                .make_naga_module(NagaModuleDescriptor {
                    shader_defs,
                    ..(&resolve_consumer).into()
                })
                .unwrap_or_else(|error| {
                    panic!("visibility_buffer_resolve.wgsl motion_vectors={motion_vectors} failed Naga validation: {error:?}")
                });
        }
    }
}
