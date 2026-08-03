#![expect(missing_docs, reason = "Not all docs are written yet, see #3492.")]

//! Provides raytraced lighting.
//!
//! See [`SolariPlugins`] for more info.
//!
//! ![`bevy_solari` logo](https://raw.githubusercontent.com/bevyengine/bevy/refs/heads/main/assets/branding/bevy_solari.svg)

extern crate alloc;

pub mod pathtracer;
pub mod realtime;
pub mod scene;

/// The solari prelude.
///
/// This includes the most common types in this crate, re-exported for your convenience.
pub mod prelude {
    pub use super::SolariPlugins;
    pub use crate::realtime::SolariLighting;
    pub use crate::scene::{
        RaytracingMesh3d, RaytracingSceneStatus, RaytracingSceneStatusSnapshot,
        SolariEnvironmentLight,
    };
    pub use bevy_pbr::MeshGeometryError;
}

use crate::realtime::SolariLightingPlugin;
use crate::scene::RaytracingScenePlugin;
use bevy_app::{PluginGroup, PluginGroupBuilder};
use bevy_render::settings::WgpuFeatures;

/// An experimental set of plugins for raytraced lighting.
///
/// This plugin group provides:
/// * [`SolariLightingPlugin`] - Raytraced direct and indirect lighting.
/// * [`RaytracingScenePlugin`] - BLAS building, resource and lighting binding.
///
/// There's also:
/// * [`pathtracer::PathtracingPlugin`] - A non-realtime pathtracer for validation purposes (not added by default).
///
/// To get started, add this plugin to your app, and then add `RaytracingMesh3d` and `MeshMaterial3d::<StandardMaterial>` to your entities.
pub struct SolariPlugins;

impl PluginGroup for SolariPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(RaytracingScenePlugin)
            .add(SolariLightingPlugin)
    }
}

impl SolariPlugins {
    /// [`WgpuFeatures`] required for these plugins to function.
    pub fn required_wgpu_features() -> WgpuFeatures {
        WgpuFeatures::EXPERIMENTAL_RAY_QUERY
            | WgpuFeatures::BUFFER_BINDING_ARRAY
            | WgpuFeatures::TEXTURE_BINDING_ARRAY
            | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | WgpuFeatures::PARTIALLY_BOUND_BINDING_ARRAY
    }
}

/// Composes every Solari compute entry point out of the real shader sources, against stubs for the
/// `bevy_pbr` and `bevy_render` helpers they import. Solari's WGSL is otherwise only compiled on a
/// device with raytracing support, so this is the only build-time check that the shaders agree with
/// each other and with the structs in `binder.rs`.
#[cfg(test)]
mod entry_point_shader_validation_tests {
    use bevy_shader::{Shader, ShaderDefVal};
    use naga_oil::compose::{Composer, NagaModuleDescriptor, ShaderDefValue, ShaderType};
    use std::collections::HashMap;

    const WORLD_CACHE_SIZE: u32 = 1 << 20;

    const STUBS: &[(&str, &str)] = &[
        (
            "tonemapping_stub.wgsl",
            r#"
#define_import_path bevy_core_pipeline::tonemapping
fn tonemapping_luminance(v: vec3<f32>) -> f32 { return v.g; }
"#,
        ),
        (
            "pbr_utils_stub.wgsl",
            r#"
#define_import_path bevy_pbr::utils
fn rand_f(rng: ptr<function, u32>) -> f32 { return 0.5; }
fn rand_u(rng: ptr<function, u32>) -> u32 { return 0u; }
fn rand_vec2f(rng: ptr<function, u32>) -> vec2<f32> { return vec2<f32>(0.5); }
fn rand_range_u(range: u32, rng: ptr<function, u32>) -> u32 { return 0u; }
fn sample_disk(radius: f32, rng: ptr<function, u32>) -> vec2<f32> { return vec2<f32>(radius); }
fn sample_cosine_hemisphere(normal: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> { return normal; }
"#,
        ),
        (
            "pbr_functions_stub.wgsl",
            r#"
#define_import_path bevy_pbr::pbr_functions
fn calculate_tbn_mikktspace(normal: vec3<f32>, tangent: vec4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(tangent.xyz, cross(normal, tangent.xyz) * tangent.w, normal);
}
fn calculate_F0_dielectric(reflectance: vec3<f32>) -> vec3<f32> { return reflectance; }
fn calculate_diffuse_color(base_color: vec3<f32>, metallic: f32, specular_transmission: f32, diffuse_transmission: f32) -> vec3<f32> {
    return base_color * (1.0 - metallic - specular_transmission - diffuse_transmission);
}
fn calculate_F0(base_color: vec3<f32>, metallic: f32, reflectance: vec3<f32>) -> vec3<f32> {
    return mix(reflectance, base_color, metallic);
}
"#,
        ),
        (
            // Only reachable under DLSS_RR_GUIDE_BUFFERS, but naga_oil resolves imports before
            // preprocessing, so the module still has to exist.
            "resolve_dlss_rr_textures_stub.wgsl",
            r#"
#define_import_path bevy_solari::resolve_dlss_rr_textures
fn env_brdf_approx2(specular_color: vec3<f32>, alpha: f32, N: vec3<f32>, V: vec3<f32>) -> vec3<f32> {
    return specular_color * (alpha + abs(dot(N, V)));
}
"#,
        ),
        (
            "lighting_stub.wgsl",
            r#"
#define_import_path bevy_pbr::lighting
fn D_GGX(roughness: f32, n_dot_h: f32) -> f32 { return roughness + n_dot_h; }
fn V_SmithGGXCorrelated(roughness: f32, n_dot_v: f32, n_dot_l: f32) -> f32 { return roughness; }
fn specular_multiscatter(D: f32, V: f32, F: vec3<f32>, F0: vec3<f32>, F_ab: vec2<f32>, specular_intensity: f32) -> vec3<f32> {
    return F * F0 * (D * V * specular_intensity * F_ab.x);
}
"#,
        ),
        (
            "rgb9e5_stub.wgsl",
            r#"
#define_import_path bevy_pbr::rgb9e5
fn vec3_to_rgb9e5_(rgb: vec3<f32>) -> u32 { return u32(rgb.r); }
fn rgb9e5_to_vec3_(packed: u32) -> vec3<f32> { return vec3<f32>(f32(packed)); }
"#,
        ),
        (
            "pbr_deferred_types_stub.wgsl",
            r#"
#define_import_path bevy_pbr::pbr_deferred_types
fn unpack_24bit_normal(packed: u32) -> vec2<f32> { return vec2<f32>(f32(packed)); }
fn deferred_geometry_error(gbuffer_a: u32) -> f32 { return -1.0; }
"#,
        ),
        (
            "prepass_bindings_stub.wgsl",
            r#"
#define_import_path bevy_pbr::prepass_bindings
struct PreviousViewUniforms {
    clip_from_world: mat4x4<f32>,
    unjittered_clip_from_world: mat4x4<f32>,
    world_from_clip: mat4x4<f32>,
    clip_from_view: mat4x4<f32>,
    view_from_clip: mat4x4<f32>,
}
"#,
        ),
        (
            "maths_stub.wgsl",
            r#"
#define_import_path bevy_render::maths
const PI: f32 = 3.141592653589793;
const PI_2: f32 = 6.283185307179586;
fn affine3_to_square(affine: mat3x4<f32>) -> mat4x4<f32> {
    return transpose(mat4x4<f32>(
        affine[0], affine[1], affine[2], vec4<f32>(0.0, 0.0, 0.0, 1.0)
    ));
}
fn orthonormalize(direction: vec3<f32>) -> mat3x3<f32> {
    let z = normalize(direction);
    let x = normalize(cross(select(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), abs(z.x) > 0.9), z));
    return mat3x3<f32>(x, cross(z, x), z);
}
"#,
        ),
        (
            "render_utils_stub.wgsl",
            r#"
#define_import_path bevy_render::utils
fn octahedral_encode(v: vec3<f32>) -> vec2<f32> { return v.xy; }
fn octahedral_decode(v: vec2<f32>) -> vec3<f32> { return normalize(vec3<f32>(v, 1.0)); }
"#,
        ),
        (
            "view_stub.wgsl",
            r#"
#define_import_path bevy_render::view
struct View {
    clip_from_world: mat4x4<f32>,
    unjittered_clip_from_world: mat4x4<f32>,
    world_from_clip: mat4x4<f32>,
    clip_from_view: mat4x4<f32>,
    view_from_clip: mat4x4<f32>,
    world_position: vec3<f32>,
    exposure: f32,
    viewport: vec4<f32>,
    main_pass_viewport: vec4<f32>,
}
fn depth_ndc_to_view_z(ndc_depth: f32, clip_from_view: mat4x4<f32>, view_from_clip: mat4x4<f32>) -> f32 {
    return ndc_depth + clip_from_view[0][0] + view_from_clip[0][0];
}
"#,
        ),
    ];

    /// Solari libraries in dependency order.
    const LIBRARIES: &[(&str, &str)] = &[
        (
            "raytracing_scene_bindings.wgsl",
            include_str!("scene/raytracing_scene_bindings.wgsl"),
        ),
        ("sampling.wgsl", include_str!("scene/sampling.wgsl")),
        ("brdf.wgsl", include_str!("scene/brdf.wgsl")),
        (
            "realtime_bindings.wgsl",
            include_str!("realtime/realtime_bindings.wgsl"),
        ),
        (
            "world_cache_query.wgsl",
            include_str!("realtime/world_cache_query.wgsl"),
        ),
        (
            "presample_light_tiles.wgsl",
            include_str!("realtime/presample_light_tiles.wgsl"),
        ),
        (
            "gbuffer_utils.wgsl",
            include_str!("realtime/gbuffer_utils.wgsl"),
        ),
        (
            "initial_path.wgsl",
            include_str!("realtime/initial_path.wgsl"),
        ),
    ];

    const ENTRY_POINTS: &[(&str, &str)] = &[
        ("restir.wgsl", include_str!("realtime/restir.wgsl")),
        (
            "world_cache_update.wgsl",
            include_str!("realtime/world_cache_update.wgsl"),
        ),
        (
            "world_cache_compact.wgsl",
            include_str!("realtime/world_cache_compact.wgsl"),
        ),
        (
            "presample_light_tiles.wgsl",
            include_str!("realtime/presample_light_tiles.wgsl"),
        ),
        ("pathtracer.wgsl", include_str!("pathtracer/pathtracer.wgsl")),
    ];

    fn world_cache_size_defs() -> Vec<ShaderDefVal> {
        vec![ShaderDefVal::UInt(
            "WORLD_CACHE_SIZE".into(),
            WORLD_CACHE_SIZE,
        )]
    }

    #[test]
    fn every_solari_entry_point_composes() {
        // Parsing is what this test wants; the final naga validation pass would additionally
        // demand the ray-query and binding-array capabilities, which are a device property.
        let mut composer = Composer::non_validating();
        for (path, source) in STUBS.iter().chain(LIBRARIES) {
            let shader = Shader::from_wgsl_with_defs(*source, *path, world_cache_size_defs());
            if let Err(error) = composer.add_composable_module((&shader).into()) {
                panic!("{path} failed to compose: {}", error.emit_to_string(&composer));
            }
        }

        let shader_defs = HashMap::from([(
            "WORLD_CACHE_SIZE".to_string(),
            ShaderDefValue::UInt(WORLD_CACHE_SIZE),
        )]);
        for (path, source) in ENTRY_POINTS {
            let result = composer.make_naga_module(NagaModuleDescriptor {
                source,
                file_path: path,
                shader_type: ShaderType::Wgsl,
                shader_defs: shader_defs.clone(),
                additional_imports: &[],
            });
            if let Err(error) = result {
                panic!("{path} failed to compose: {}", error.emit_to_string(&composer));
            }
        }
    }
}
