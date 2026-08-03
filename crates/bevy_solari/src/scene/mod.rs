mod binder;
mod blas;
mod extract;
mod types;

use bevy_shader::load_shader_library;
pub use binder::RaytracingSceneBindings;
pub use blas::{RaytracingSceneStatus, RaytracingSceneStatusSnapshot};
pub use types::{RaytracingMesh3d, RaytracingMesh3dGeometryError, SolariEnvironmentLight};

use crate::SolariPlugins;
use bevy_app::{App, Plugin};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    extract_resource::ExtractResourcePlugin,
    mesh::{
        allocator::{allocate_and_free_meshes, MeshAllocatorSettings},
        RenderMesh,
    },
    render_asset::prepare_assets,
    render_resource::BufferUsages,
    renderer::RenderDevice,
    ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderSystems,
};
use binder::prepare_raytracing_scene_bindings;
use blas::{
    compact_raytracing_blas, prepare_raytracing_blas, update_raytracing_scene_status, BlasManager,
};
use extract::{extract_raytracing_scene, StandardMaterialAssets};
use tracing::warn;

/// Creates acceleration structures and binding arrays of resources for raytracing.
pub struct RaytracingScenePlugin;

impl Plugin for RaytracingScenePlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "brdf.wgsl");
        load_shader_library!(app, "raytracing_scene_bindings.wgsl");
        load_shader_library!(app, "sampling.wgsl");
        app.init_resource::<RaytracingSceneStatus>();
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

        app.add_plugins((
            ExtractResourcePlugin::<StandardMaterialAssets>::default(),
            ExtractResourcePlugin::<RaytracingSceneStatus>::default(),
        ));

        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .world_mut()
            .resource_mut::<MeshAllocatorSettings>()
            .extra_buffer_usages |= BufferUsages::BLAS_INPUT | BufferUsages::STORAGE;

        render_app
            .init_gpu_resource::<BlasManager>()
            .init_gpu_resource::<StandardMaterialAssets>()
            .insert_resource(RaytracingSceneBindings::new())
            .add_systems(ExtractSchedule, extract_raytracing_scene)
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
                    update_raytracing_scene_status
                        .in_set(RenderSystems::PrepareAssets)
                        .after(compact_raytracing_blas),
                    prepare_raytracing_scene_bindings.in_set(RenderSystems::PrepareBindGroups),
                ),
            );
    }
}

#[cfg(test)]
mod shader_validation_tests {
    use bevy_shader::Shader;
    use naga_oil::compose::Composer;

    #[test]
    fn scene_libraries_pass_naga_validation() {
        let mut composer = Composer::default();
        let pbr_functions = Shader::from_wgsl(
            r#"
#define_import_path bevy_pbr::pbr_functions
fn calculate_tbn_mikktspace(normal: vec3<f32>, tangent: vec4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(tangent.xyz, cross(normal, tangent.xyz), normal);
}
"#,
            "pbr_functions_stub.wgsl",
        );
        let maths = Shader::from_wgsl(
            r#"
#define_import_path bevy_render::maths
fn affine3_to_square(affine: mat3x4<f32>) -> mat4x4<f32> {
    return transpose(mat4x4<f32>(
        affine[0], affine[1], affine[2], vec4<f32>(0.0, 0.0, 0.0, 1.0)
    ));
}
const PI_2: f32 = 6.283185307179586;
fn orthonormalize(direction: vec3<f32>) -> mat3x3<f32> {
    let z = normalize(direction);
    let x = normalize(cross(select(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), abs(z.x) > 0.9), z));
    return mat3x3<f32>(x, cross(z, x), z);
}
"#,
            "maths_stub.wgsl",
        );
        let lighting = Shader::from_wgsl(
            r#"
#define_import_path bevy_pbr::lighting
fn D_GGX(roughness: f32, n_dot_h: f32) -> f32 { return roughness + n_dot_h; }
"#,
            "lighting_stub.wgsl",
        );
        let pbr_utils = Shader::from_wgsl(
            r#"
#define_import_path bevy_pbr::utils
fn rand_vec2f(rng: ptr<function, u32>) -> vec2<f32> { return vec2<f32>(0.5); }
fn rand_u(rng: ptr<function, u32>) -> u32 { return 0u; }
fn rand_range_u(range: u32, rng: ptr<function, u32>) -> u32 { return 0u; }
"#,
            "pbr_utils_stub.wgsl",
        );
        composer
            .add_composable_module((&pbr_functions).into())
            .unwrap();
        composer.add_composable_module((&maths).into()).unwrap();
        composer.add_composable_module((&lighting).into()).unwrap();
        composer.add_composable_module((&pbr_utils).into()).unwrap();

        let scene_bindings = Shader::from_wgsl(
            include_str!("raytracing_scene_bindings.wgsl"),
            "raytracing_scene_bindings.wgsl",
        );
        composer
            .add_composable_module((&scene_bindings).into())
            .expect("Solari scene bindings must pass Naga validation");

        let sampling = Shader::from_wgsl(include_str!("sampling.wgsl"), "sampling.wgsl");
        composer
            .add_composable_module((&sampling).into())
            .expect("Solari sampling library must pass Naga validation");
    }
}
