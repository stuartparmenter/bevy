mod binder;
mod blas;
mod extract;
mod producer;
mod types;

use bevy_asset::embedded_asset;
use bevy_shader::load_shader_library;
pub use binder::prepare_raytracing_scene_resources;
pub use binder::{RaytracingSceneBindings, RaytracingSceneNeedsPreviousFrameData};
pub use producer::RaytracingProducerEncoder;
pub use types::{
    RaytracingGeometry, RaytracingGeometryBuffers, RaytracingGeometryPreviousVertices,
    RaytracingGeometryUpdateMode, RaytracingMesh3d,
};

use crate::SolariPlugins;
use bevy_app::{App, Plugin, PreUpdate};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    mesh::{
        allocator::{allocate_and_free_meshes, MeshAllocatorSettings},
        RenderMesh,
    },
    render_asset::prepare_assets,
    render_resource::{update_sparse_buffers, BufferUsages},
    renderer::{RenderDevice, RenderGraph, RenderGraphSystems},
    ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderSystems,
};
use binder::{
    build_raytracing_tlas, prepare_raytracing_scene_bind_group, TlasInstanceSetupPipeline,
};
use blas::{
    build_raytracing_geometry_blas, compact_raytracing_blas, delete_raytracing_blas,
    prepare_raytracing_blas, prepare_raytracing_geometry_blas, BlasManager, GeometryBlasManager,
};
use extract::{
    extract_raytracing_environment_map_light, extract_raytracing_material_assets,
    extract_raytracing_scene_meshes_and_materials, extract_raytracing_scene_structural,
    extract_raytracing_scene_transforms, update_raytracing_previous_global_transforms,
    ExtractedEnvironmentMapLight, StandardMaterialAssets,
};
use producer::submit_raytracing_producers;
use tracing::warn;

/// Creates acceleration structures and binding arrays of resources for raytracing.
pub struct RaytracingScenePlugin;

impl Plugin for RaytracingScenePlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "brdf.wesl");
        load_shader_library!(app, "bindings.wesl");
        load_shader_library!(app, "sampling.wesl");
        embedded_asset!(app, "binder/setup_tlas_instances.wesl");

        // Meshlet raytracing entities can match both transform updaters.
        // Both write the same value, so their relative order does not matter.
        app.add_systems(
            PreUpdate,
            update_raytracing_previous_global_transforms
                .ambiguous_with(bevy_pbr::update_mesh_previous_global_transforms),
        );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        let render_device = render_app.world().resource::<RenderDevice>();
        let features = render_device.features();
        if !features.contains(SolariPlugins::required_wgpu_features()) {
            warn!(
                "RaytracingScenePlugin not loaded. GPU lacks support for required features: {:?}.",
                SolariPlugins::required_wgpu_features().difference(features)
            );
            return;
        }

        render_app
            .world_mut()
            .resource_mut::<MeshAllocatorSettings>()
            .extra_buffer_usages |= BufferUsages::BLAS_INPUT | BufferUsages::STORAGE;

        render_app
            .init_resource::<ExtractedEnvironmentMapLight>()
            .init_gpu_resource::<BlasManager>()
            .init_gpu_resource::<GeometryBlasManager>()
            .init_gpu_resource::<StandardMaterialAssets>()
            .init_gpu_resource::<RaytracingProducerEncoder>()
            .init_gpu_resource::<RaytracingSceneBindings>()
            .init_gpu_resource::<TlasInstanceSetupPipeline>()
            .add_systems(
                ExtractSchedule,
                (
                    extract_raytracing_scene_structural,
                    extract_raytracing_scene_transforms,
                    extract_raytracing_scene_meshes_and_materials,
                    extract_raytracing_material_assets,
                    extract_raytracing_environment_map_light,
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_raytracing_blas
                        .in_set(RenderSystems::PrepareAssets)
                        .before(prepare_assets::<RenderMesh>)
                        .after(allocate_and_free_meshes),
                    compact_raytracing_blas
                        .in_set(RenderSystems::PrepareAssets)
                        .after(prepare_raytracing_blas),
                    // Allocate before binding instances; build after producers fill their buffers.
                    prepare_raytracing_geometry_blas
                        .in_set(RenderSystems::PrepareResources)
                        .before(prepare_raytracing_scene_resources),
                    prepare_raytracing_scene_resources.in_set(RenderSystems::PrepareResources),
                    // Record BLAS builds after the producer passes, then submit both before the TLAS build.
                    build_raytracing_geometry_blas.in_set(RenderSystems::PrepareBindGroups),
                    submit_raytracing_producers
                        .in_set(RenderSystems::PrepareBindGroups)
                        .after(build_raytracing_geometry_blas),
                    prepare_raytracing_scene_bind_group.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                RenderGraph,
                (
                    build_raytracing_tlas
                        .after(update_sparse_buffers)
                        .in_set(RenderGraphSystems::Begin),
                    delete_raytracing_blas.in_set(RenderGraphSystems::Finish),
                ),
            );
    }
}
