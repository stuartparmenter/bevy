mod extract;
mod node;
mod prepare;

use crate::SolariPlugins;
use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_camera::Hdr;
use bevy_core_pipeline::{
    core_3d::main_opaque_pass_3d,
    prepass::{
        DeferredPrepass, DeferredPrepassDoubleBuffer, DepthPrepass, DepthPrepassDoubleBuffer,
        MotionVectorPrepass,
    },
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::{component::Component, reflect::ReflectComponent, schedule::IntoScheduleConfigs};
use bevy_pbr::DefaultOpaqueRendererMethod;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    renderer::RenderDevice, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::load_shader_library;
use extract::extract_solari_lighting;
use node::{init_solari_lighting_pipelines, solari_lighting};
use prepare::prepare_solari_lighting_resources;
use tracing::warn;

/// Raytraced direct and indirect lighting.
///
/// When using this plugin, it's highly recommended to set `shadow_maps_enabled: false` on all lights, as Solari replaces
/// traditional shadow mapping.
pub struct SolariLightingPlugin;

impl Plugin for SolariLightingPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "gbuffer_utils.wgsl");
        load_shader_library!(app, "realtime_bindings.wgsl");
        load_shader_library!(app, "presample_light_tiles.wgsl");
        load_shader_library!(app, "initial_path.wgsl");
        embedded_asset!(app, "restir.wgsl");
        load_shader_library!(app, "world_cache_query.wgsl");
        embedded_asset!(app, "world_cache_compact.wgsl");
        embedded_asset!(app, "world_cache_update.wgsl");

        load_shader_library!(app, "resolve_dlss_rr_textures.wgsl");

        app.insert_resource(DefaultOpaqueRendererMethod::deferred());
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        let render_device = render_app.world().resource::<RenderDevice>();
        let features = render_device.features();
        if !features.contains(SolariPlugins::required_wgpu_features()) {
            warn!(
                "SolariLightingPlugin not loaded. GPU lacks support for required features: {:?}.",
                SolariPlugins::required_wgpu_features().difference(features)
            );
            return;
        }

        render_app
            .add_systems(RenderStartup, init_solari_lighting_pipelines)
            .add_systems(ExtractSchedule, extract_solari_lighting)
            .add_systems(
                Render,
                prepare_solari_lighting_resources.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Core3d,
                solari_lighting
                    .before(main_opaque_pass_3d)
                    .in_set(Core3dSystems::MainPass),
            );
    }
}

/// A component for a 3d camera entity to enable the Solari raytraced lighting system.
///
/// Must be used with `CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING)`, and
/// `Msaa::Off`.
#[derive(Component, Reflect, Clone)]
#[reflect(Component, Default, Clone)]
#[require(
    Hdr,
    DeferredPrepass,
    DepthPrepass,
    MotionVectorPrepass,
    DeferredPrepassDoubleBuffer,
    DepthPrepassDoubleBuffer
)]
pub struct SolariLighting {
    /// Maximum confidence weight (effective temporal history length) a pixel
    /// can accumulate during temporal resampling.
    ///
    /// Higher values are more stable but slower to react to lighting changes
    /// and will lead to increased artifacts.
    pub confidence_weight_cap: f32,

    /// Number of direct light samples taken for the camera's primary hit during
    /// initial sampling.
    ///
    /// Higher values reduce noise in directly-lit areas at the cost of more work
    /// per frame. Lower values are faster but noisier.
    pub primary_di_samples: u32,

    /// Number of direct light samples taken at each indirect bounce during
    /// initial sampling.
    ///
    /// Higher values reduce noise in indirect lighting at the cost of more work
    /// per frame. Lower values are faster but noisier.
    pub secondary_di_samples: u32,

    /// Maximum number of bounces traced when generating an initial path.
    ///
    /// Higher values capture more indirect light for greater accuracy at the cost
    /// of more rays traced per frame. Lower values are faster but lose
    /// multi-bounce lighting for specular paths.
    pub max_bounces: u32,

    /// How responsive the world cache is to changes in lighting.
    ///
    /// Higher values accumulate more temporal history, giving more stable but
    /// less responsive (slower to update) lighting. Lower values react faster
    /// but are noisier and less stable.
    pub world_cache_max_temporal_samples: f32,

    /// How many direct light samples each world cache cell takes when updating
    /// each frame.
    ///
    /// Higher values reduce noise in cached lighting at the cost of more work
    /// per frame. Lower values are faster but noisier.
    pub world_cache_direct_light_sample_count: u32,

    /// Maximum distance to trace GI rays between two world cache cells.
    ///
    /// Higher values capture indirect light from farther away for more accurate
    /// GI at the cost of longer (more expensive) ray traversal and increased noise.
    /// Lower values are faster and less noisy but may miss distant lighting.
    pub world_cache_max_gi_ray_distance: f32,

    /// Soft upper limit on the number of world cache cells to update each frame.
    ///
    /// Higher values let the cache converge faster after lighting changes at the
    /// cost of more work per frame. Lower values are cheaper but make the cache
    /// slower to update.
    ///
    /// This is a stochastic target that only takes effect when the number of
    /// active cells exceeds it: each active cell is then updated with
    /// probability `target / active_cells`, so on average this many cells
    /// update, though individual frames may update more or fewer. When there
    /// are fewer active cells than the target, all of them update every frame.
    pub world_cache_cell_updates_soft_target: u32,

    /// Size of a world cache cell at the lowest LOD, in meters.
    ///
    /// Smaller values give finer spatial resolution and more detailed indirect
    /// lighting at the cost of more cells to fill and update. Larger values are
    /// cheaper but coarser, which can cause light leaking.
    pub world_cache_position_base_cell_size: f32,

    /// How fast the world cache transitions between LODs as a function of
    /// distance to the camera.
    ///
    /// Higher values keep cells small (high detail) out to greater distances for
    /// better quality at the cost of more cells to fill. Lower values transition
    /// to larger cells sooner, which is cheaper but coarser farther from the
    /// camera.
    pub world_cache_position_lod_scale: f32,

    /// Set to true to delete the saved temporal history (past frames).
    ///
    /// Useful for preventing ghosting when the history is no longer
    /// representative of the current frame, such as in sudden camera cuts.
    ///
    /// After setting this to true, it will automatically be toggled
    /// back to false at the end of the frame.
    pub reset: bool,
}

impl Default for SolariLighting {
    fn default() -> Self {
        Self {
            confidence_weight_cap: 8.0,
            primary_di_samples: 8,
            secondary_di_samples: 4,
            max_bounces: 3,
            world_cache_max_temporal_samples: 32.0,
            world_cache_direct_light_sample_count: 32,
            world_cache_max_gi_ray_distance: 50.0,
            world_cache_cell_updates_soft_target: 40000,
            world_cache_position_base_cell_size: 0.15,
            world_cache_position_lod_scale: 15.0,
            reset: true, // No temporal history on the first frame
        }
    }
}

#[cfg(test)]
mod shader_source_tests {
    #[test]
    fn ris_visibility_is_part_of_candidate_evaluation() {
        let initial_path = include_str!("initial_path.wgsl");
        let candidate_visibility = initial_path
            .find("let visibility = trace_visibility(ray_origin, geometric_normal")
            .expect("initial-path RIS candidates must test visibility");
        let candidate_target = initial_path[candidate_visibility..]
            .find("let target_function = luminance(brdf_radiance);")
            .expect("initial-path RIS must evaluate a target after visibility");
        assert!(candidate_target > 0);
        assert!(
            !initial_path.contains("unbiased_contribution_weight *= trace_visibility"),
            "the selected RIS winner must not issue a redundant visibility ray"
        );

        let world_cache_update = include_str!("world_cache_update.wgsl");
        assert!(world_cache_update
            .contains("let visibility = trace_visibility(world_position, world_normal"));
        assert!(!world_cache_update.contains("unbiased_contribution_weight *= trace_visibility"));
    }

    #[test]
    fn world_cache_linear_probe_wraps() {
        let world_cache_query = include_str!("world_cache_query.wgsl");
        assert!(world_cache_query.contains("key = wrap_key(key + 1u);"));
        assert!(!world_cache_query.contains("\n            key += 1u;"));
    }

    #[test]
    fn ray_misses_consume_the_shared_environment() {
        let initial_path = include_str!("initial_path.wgsl");
        assert!(initial_path.contains("sample_environment_radiance(next_bounce.wi)"));
        assert!(initial_path.contains("environment_light_pdf(next_bounce.wi)"));
        assert!(
            initial_path.contains("environment_light_pdf(di.wi)"),
            "environment NEE must MIS against the BRDF-miss strategy that owns the same radiance"
        );
    }

    #[test]
    fn world_cache_gi_misses_do_not_re_add_the_environment() {
        // sample_di samples the environment for every cell with a full-range visibility ray, and
        // the GI ray is truncated at world_cache_max_gi_ray_distance, so a miss must add nothing.
        let world_cache_update = include_str!("world_cache_update.wgsl");
        assert!(!world_cache_update.contains("sample_environment_radiance"));
    }

    #[test]
    fn rasterized_surfaces_widen_the_ray_origin_bias_with_distance() {
        // The G-buffer surface is whichever meshlet LOD stays under ~1px of screen error, so it
        // drifts from the traced geometry by an amount that grows with camera distance. A constant
        // geometry-error bias leaves distant shading points inside the traced proxy, where every
        // visibility and GI ray is self-occluded and the pixel resolves to black.
        let initial_path = include_str!("initial_path.wgsl");
        assert!(initial_path.contains(
            "ray_origin_bias_with_raster_lod(primary_ray_origin_bias(), pixel_world_size(world_position, view.world_position))"
        ));

        // One import plus the canonical and other domains of a reservoir merge.
        let restir = include_str!("restir.wgsl");
        assert_eq!(restir.matches("ray_origin_bias_with_raster_lod").count(), 3);

        // Cache rays must escape their surface by the same distance the DI rays do.
        let world_cache_update = include_str!("world_cache_update.wgsl");
        assert!(world_cache_update.contains("fn cell_ray_origin_bias"));
        assert_eq!(
            world_cache_update
                .matches("cell_ray_origin_bias(geometry_data)")
                .count(),
            2
        );

        // The added term must stay bounded, or a distant surface biases its rays straight through
        // the wall behind it.
        let scene_bindings = include_str!("../scene/raytracing_scene_bindings.wgsl");
        assert!(scene_bindings.contains("min(lod_error, RAY_ORIGIN_BIAS_RASTER_LOD_MAX)"));
    }
}
