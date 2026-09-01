mod extract;
mod node;
mod prepare;
pub use prepare::WORLD_CACHE_SIZE;

use crate::{
    scene::{prepare_raytracing_scene_resources, RaytracingSceneBindings},
    SolariPlugins,
};
use bevy_app::{App, Plugin, PostUpdate};
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
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::Has,
    reflect::ReflectComponent,
    schedule::IntoScheduleConfigs,
    system::{Commands, Query},
};
use bevy_pbr::DefaultOpaqueRendererMethod;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    init_gpu_resource, renderer::RenderDevice, ExtractSchedule, Render, RenderApp, RenderStartup,
    RenderSystems,
};
use bevy_shader::load_shader_library;
use extract::extract_solari_lighting;
use node::{init_solari_lighting_pipelines, solari_lighting};
use prepare::{
    prepare_solari_lighting_resources, setup_raytracing_scene_needs_previous_frame_data,
};
use tracing::warn;

/// Raytraced direct and indirect lighting.
///
/// When using this plugin, it's highly recommended to set `shadow_maps_enabled: false` on all lights, as Solari replaces
/// traditional shadow mapping.
pub struct SolariLightingPlugin;

impl Plugin for SolariLightingPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "gbuffer_utils.wesl");
        load_shader_library!(app, "bindings.wesl");
        load_shader_library!(app, "presample_light_tiles.wesl");
        load_shader_library!(app, "initial_path.wesl");
        embedded_asset!(app, "restir.wesl");
        embedded_asset!(app, "no_restir.wesl");
        load_shader_library!(app, "world_cache_query.wesl");
        embedded_asset!(app, "world_cache_compact.wesl");
        embedded_asset!(app, "world_cache_update.wesl");

        load_shader_library!(app, "resolve_dlss_rr_textures.wesl");

        app.insert_resource(DefaultOpaqueRendererMethod::deferred());
    }

    fn finish(&self, app: &mut App) {
        let render_device = app.sub_app(RenderApp).world().resource::<RenderDevice>();
        let features = render_device.features();
        if !features.contains(SolariPlugins::required_wgpu_features()) {
            warn!(
                "SolariLightingPlugin not loaded. GPU lacks support for required features: {:?}.",
                SolariPlugins::required_wgpu_features().difference(features)
            );
            return;
        }
        let limits = render_device.limits();
        if (limits.max_storage_buffer_binding_size as u64) < prepare::WORLD_CACHE_BUFFER_SIZE
            || limits.max_buffer_size < prepare::WORLD_CACHE_BUFFER_SIZE
        {
            warn!(
                "SolariLightingPlugin not loaded. GPU buffer limits cannot hold the {} byte world cache.",
                prepare::WORLD_CACHE_BUFFER_SIZE
            );
            return;
        }

        app.add_systems(PostUpdate, manage_prepass_double_buffers);

        app.sub_app_mut(RenderApp)
            .add_systems(
                RenderStartup,
                init_solari_lighting_pipelines.after(init_gpu_resource::<RaytracingSceneBindings>),
            )
            .add_systems(ExtractSchedule, extract_solari_lighting)
            .add_systems(
                Render,
                (
                    prepare_solari_lighting_resources,
                    setup_raytracing_scene_needs_previous_frame_data
                        .before(prepare_raytracing_scene_resources),
                )
                    .in_set(RenderSystems::PrepareResources),
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
#[require(Hdr, DeferredPrepass, DepthPrepass, MotionVectorPrepass)]
pub struct SolariLighting {
    /// [ReSTIR](https://en.wikipedia.org/wiki/Spatiotemporal_reservoir_resampling) is a technique to reuse path samples
    /// between pixels and frames. This dramatically reduces noise, at the cost of a few extra rays per pixel.
    ///
    /// However, modern denoisers cope well with very noisy input. In many cases, turning this on
    /// won't dramatically improve quality after denoising.
    ///
    /// If you want more fine shadow detail, or have scenes with more difficult lighting conditions,
    /// turning this on may improve quality and stability, at the cost of a decent chunk of performance.
    ///
    /// Whether to enable this setting or not will be very scene dependent.
    ///
    /// Defaults to `false`.
    pub restir: bool,

    /// Maximum confidence weight (effective temporal history length) a pixel
    /// can accumulate during temporal resampling.
    ///
    /// Has no effect when [`SolariLighting::restir`] is `false`.
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
    /// Higher values capture more detail in nested reflections and more indirect lighting,
    /// at the cost of more rays traced per frame. Lower values are faster but lose
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
            restir: false,
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

/// Adds or removes the prepass double-buffer components according to [`SolariLighting::restir`].
fn manage_prepass_double_buffers(
    views: Query<(
        Entity,
        &SolariLighting,
        Has<DeferredPrepassDoubleBuffer>,
        Has<DepthPrepassDoubleBuffer>,
    )>,
    mut commands: Commands,
) {
    for (entity, solari_lighting, deferred_double_buffered, depth_double_buffered) in &views {
        let mut entity = commands.entity(entity);
        if solari_lighting.restir {
            if !deferred_double_buffered {
                entity.insert(DeferredPrepassDoubleBuffer);
            }
            if !depth_double_buffered {
                entity.insert(DepthPrepassDoubleBuffer);
            }
        } else {
            if deferred_double_buffered {
                entity.remove::<DeferredPrepassDoubleBuffer>();
            }
            if depth_double_buffered {
                entity.remove::<DepthPrepassDoubleBuffer>();
            }
        }
    }
}

#[cfg(test)]
mod shader_source_tests {
    use bevy_math::{ops, Vec3};

    #[test]
    fn ris_visibility_is_part_of_candidate_evaluation() {
        let initial_path = include_str!("initial_path.wesl");
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

        let world_cache_update = include_str!("world_cache_update.wesl");
        assert!(world_cache_update
            .contains("let visibility = trace_visibility(world_position, world_normal"));
        assert!(!world_cache_update.contains("unbiased_contribution_weight *= trace_visibility"));
    }

    #[test]
    fn shadow_rays_do_not_spend_the_origin_bias_at_the_light_end() {
        // The light point lies exactly on the traced triangle, so truncating the ray by the shading
        // surface's BLAS error charged that error twice and hid every occluder standing closer to
        // its light than the bias.
        let sampling = include_str!("../scene/sampling.wesl");
        assert!(!sampling.contains("dist - ray_origin_bias"));
        assert!(
            sampling.contains("dist - max(RAY_T_MIN, LIGHT_SAMPLE_END_EPSILON_RELATIVE * dist)")
        );

        // A light inside the offset shell has nothing between it and the surface, so it is visible.
        // Reporting it occluded is what stopped emitters lighting the geometry they sit on.
        assert!(
            sampling.contains("if t_max < RAY_T_MIN { return visibility_without_tracing(1.0); }")
        );
    }

    #[test]
    fn world_cache_linear_probe_wraps() {
        let world_cache_query = include_str!("world_cache_query.wesl");
        assert!(world_cache_query.contains("key = wrap_key(key + 1u);"));
        assert!(!world_cache_query.contains("\n            key += 1u;"));
    }

    #[test]
    fn world_cache_query_overflows_are_counted() {
        let world_cache_query = include_str!("world_cache_query.wesl");
        assert!(world_cache_query.contains("atomicAdd(&world_cache.query_overflows, 1u);"));
    }

    #[test]
    fn world_cache_new_cells_always_take_their_first_update() {
        // A never-blended cell (sample count of zero) must not wait out the stochastic budget
        // while returning black. Only sample_di and blend bootstrap - the first sample is
        // DI-only so bootstraps cannot chain new cells through their GI-hit queries - and the
        // two must agree on selection.
        let world_cache_update = include_str!("world_cache_update.wesl");
        assert!(world_cache_update
            .contains("if world_cache.radiance[cell_index].a == 0.0 { return true; }"));
        assert_eq!(
            world_cache_update
                .matches("if !should_update_cell(cell_index, &rng) { return; }")
                .count(),
            2
        );
        assert_eq!(
            world_cache_update
                .matches("if rand_f(&rng) >= f32(constants.world_cache_cell_updates_soft_target)")
                .count(),
            1
        );
    }

    #[test]
    fn world_cache_block_totals_include_the_last_cell() {
        // `a` is a block-local exclusive scan, so a block's total is its last cell's scan value
        // plus that cell's own flag; dropping the flag collides compacted indices across block
        // boundaries and the colliding cells silently miss their update passes.
        let world_cache_compact = include_str!("world_cache_compact.wesl");
        assert!(world_cache_compact
            .contains("return world_cache.a[last_cell] + u32(world_cache.life[last_cell] != 0u);"));
        assert!(!world_cache_compact.contains("w1[t] = world_cache.a[t * 1024u - 1u];"));
    }

    #[test]
    fn world_cache_normal_buckets_split_perpendicular_surfaces() {
        // A bucket one unit wide over a component's [-1, 1] range reduces the key to the sign of
        // each component, so a floor and the wall it meets share a cell whenever both normals tip
        // into the same octant, and the seeding surface owns the hemisphere sample_gi samples and
        // the albedo folded into that cell's radiance.
        let world_cache_query = include_str!("world_cache_query.wesl");
        assert!(world_cache_query.contains(
            "return quantize_position(world_normal, WORLD_CACHE_NORMAL_QUANTIZATION_FACTOR);"
        ));

        // Read the width and the epsilon out of the shader rather than repeating them, so widening
        // the bucket fails the separation check below instead of only the text pin above.
        let shader_f32 = |prefix: &str, terminator: char| -> f32 {
            world_cache_query
                .split_once(prefix)
                .and_then(|(_, rest)| rest.split_once(terminator))
                .and_then(|(value, _)| value.trim().parse().ok())
                .unwrap_or_else(|| panic!("the shader must declare a parsable {prefix}"))
        };
        let factor = shader_f32("const WORLD_CACHE_NORMAL_QUANTIZATION_FACTOR: f32 = ", ';');
        let epsilon = shader_f32("return floor(world_position / quantization_factor + ", ')');

        // The key is bitcast, so two normals share a cell only when their bucket bits match.
        let quantize = |normal: Vec3| {
            (normal / factor + Vec3::splat(epsilon))
                .floor()
                .to_array()
                .map(f32::to_bits)
        };

        const GRID: usize = 16;
        let mut buckets = Vec::new();
        for i in 0..GRID {
            for j in 0..GRID {
                let (sin_theta, cos_theta) =
                    ops::sin_cos(core::f32::consts::PI * (i as f32 + 0.5) / GRID as f32);
                let (sin_phi, cos_phi) =
                    ops::sin_cos(core::f32::consts::TAU * (j as f32 + 0.5) / GRID as f32);
                let normal = Vec3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
                buckets.push((quantize(normal), normal));
            }
        }

        for (index, (key, normal)) in buckets.iter().enumerate() {
            for (other_key, other_normal) in &buckets[index + 1..] {
                assert!(
                    key != other_key || normal.dot(*other_normal) > 0.5,
                    "{normal} and {other_normal} are over 60 degrees apart in one cell"
                );
            }
        }
    }

    #[test]
    fn ray_misses_consume_the_shared_environment() {
        let initial_path = include_str!("initial_path.wesl");
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
        let world_cache_update = include_str!("world_cache_update.wesl");
        assert!(!world_cache_update.contains("sample_environment_radiance"));
    }

    #[test]
    fn light_tile_selection_is_reshuffled_every_frame() {
        // The tiles are refilled every frame, so a frozen choice is not a frozen set of lights. It
        // does keep each workgroup sharing its tile with exactly the same others, whose variance
        // then correlates into a pattern the temporal filter reproduces instead of averaging out.
        let initial_path = include_str!("initial_path.wesl");
        assert!(initial_path.contains(
            "var workgroup_rng = (workgroup_id.x * 0x9E3779B9u) + workgroup_id.y + bounce + constants.frame_rng;"
        ));
        let world_cache_update = include_str!("world_cache_update.wesl");
        assert!(world_cache_update.contains(
            "var workgroup_rng = (workgroup_id.x * 0x9E3779B9u) + workgroup_id.y + constants.frame_rng;"
        ));
    }

    #[test]
    fn rasterized_surfaces_widen_the_ray_origin_bias_with_distance() {
        // The G-buffer surface is whichever meshlet LOD stays under ~1px of screen error, so it
        // drifts from the traced geometry by an amount that grows with camera distance. A constant
        // geometry-error bias leaves distant shading points inside the traced proxy, where every
        // visibility and GI ray is self-occluded and the pixel resolves to black.
        let initial_path = include_str!("initial_path.wesl");
        assert!(initial_path.contains(
            "ray_origin_bias_with_raster_lod(rasterized_surface_ray_origin_bias(surface_world_geometry_error), pixel_world_size(world_position, view.world_position))"
        ));
        // The scene-wide maximum is the unknown fallback now, not the primary vertex's answer.
        assert!(!initial_path.contains("primary_ray_origin_bias"));

        // One import plus the canonical and other domains of a reservoir merge.
        let restir = include_str!("restir.wesl");
        assert_eq!(restir.matches("ray_origin_bias_with_raster_lod").count(), 3);

        // The two merge domains are different surfaces and can be different instances. Reusing the
        // canonical bias for both compiles, shows no acne, and only leaks as energy loss and blotchy
        // convergence, so pin the two expressions apart.
        assert!(
            restir.contains("rasterized_surface_ray_origin_bias(canonical_world_geometry_error)")
        );
        assert!(restir.contains("rasterized_surface_ray_origin_bias(other_world_geometry_error)"));

        // Cache rays must escape their surface by the same distance the DI rays do.
        let world_cache_update = include_str!("world_cache_update.wesl");
        assert!(world_cache_update.contains("fn cell_ray_origin_bias"));
        assert_eq!(
            world_cache_update
                .matches("cell_ray_origin_bias(geometry_data)")
                .count(),
            2
        );

        // The added term must stay bounded, or a distant surface biases its rays straight through
        // the wall behind it.
        let scene_bindings = include_str!("../scene/bindings.wesl");
        assert!(scene_bindings.contains("min(lod_error, RAY_ORIGIN_BIAS_RASTER_LOD_MAX)"));
    }
}
