import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance;
import bevy_pbr::render::utils::{rand_f, rand_range_u, sample_cosine_hemisphere};
import package::realtime::presample_light_tiles::unpack_resolved_light_sample;
import package::scene::sampling::{calculate_resolved_light_contribution, isfinite, isfinite3, trace_visibility};
import package::scene::bindings::{trace_ray, resolve_ray_hit_full, offset_ray_origin, ray_origin_bias_with_raster_lod, RAY_T_MIN};
import package::realtime::world_cache_query::query_world_cache;
import package::realtime::bindings::{
    light_tile_resolved_samples,
    pixel_world_size,
    view,
    constants,
    world_cache,
    WorldCacheGeometryData,
};

enable wgpu_ray_query;

// The stored bias is the geometry error of whatever surface seeded the cell. The raster LOD term is
// recomputed here instead of being stored with it, because a cell outlives camera movement and the
// term tracks the camera. Using the same formula the primary surface uses keeps the cache's DI and
// GI rays escaping their surface by the same distance the ReSTIR DI rays do.
fn cell_ray_origin_bias(geometry_data: WorldCacheGeometryData) -> f32 {
    return ray_origin_bias_with_raster_lod(
        geometry_data.ray_origin_bias,
        pixel_world_size(geometry_data.world_position, view.world_position),
    );
}

@compute @workgroup_size(64, 1, 1)
fn sample_di(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(global_invocation_id) active_cell_id: vec3<u32>) {
    if active_cell_id.x >= world_cache.active_cells_count { return; }

    let cell_index = world_cache.active_cell_indices[active_cell_id.x];
    let geometry_data = world_cache.geometry_data[cell_index];
    var rng = cell_index + constants.frame_rng;

    if rand_f(&rng) >= f32(constants.world_cache_cell_updates_soft_target) / f32(world_cache.active_cells_count) { return; }

    let new_radiance = sample_random_light_ris(geometry_data.world_position, geometry_data.world_normal, cell_ray_origin_bias(geometry_data), workgroup_id.xy, &rng);

    world_cache.active_cells_new_radiance[active_cell_id.x] = new_radiance;
}

@compute @workgroup_size(64, 1, 1)
fn sample_gi(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(global_invocation_id) active_cell_id: vec3<u32>) {
    if active_cell_id.x >= world_cache.active_cells_count { return; }

    let cell_index = world_cache.active_cell_indices[active_cell_id.x];
    let geometry_data = world_cache.geometry_data[cell_index];
    var rng = cell_index + constants.frame_rng;

    if rand_f(&rng) >= f32(constants.world_cache_cell_updates_soft_target) / f32(world_cache.active_cells_count) { return; }

    let ray_direction = sample_cosine_hemisphere(geometry_data.world_normal, &rng);
    let ray_origin = offset_ray_origin(geometry_data.world_position, geometry_data.world_normal, ray_direction, cell_ray_origin_bias(geometry_data));
    let ray = trace_ray(ray_origin, ray_direction, RAY_T_MIN, constants.world_cache_max_gi_ray_distance, RAY_FLAG_NONE);
    if ray.kind != RAY_QUERY_INTERSECTION_NONE {
        let ray_hit = resolve_ray_hit_full(ray);
        let cell_life = atomicLoad(&world_cache.life[cell_index]);
        let radiance = query_world_cache(ray_hit.world_position, ray_hit.geometric_world_normal, ray_hit.ray_origin_bias, view.world_position, ray.t, cell_life, &rng);
        world_cache.active_cells_new_radiance[active_cell_id.x] += ray_hit.material.base_color * radiance;
    }
}

@compute @workgroup_size(64, 1, 1)
fn blend_new_samples(@builtin(global_invocation_id) active_cell_id: vec3<u32>) {
    if active_cell_id.x >= world_cache.active_cells_count { return; }

    let cell_index = world_cache.active_cell_indices[active_cell_id.x];
    var rng = cell_index + constants.frame_rng;

    if rand_f(&rng) >= f32(constants.world_cache_cell_updates_soft_target) / f32(world_cache.active_cells_count) { return; }

    let old_radiance_raw = world_cache.radiance[cell_index];
    let old_radiance = select(vec4(0.0), old_radiance_raw, isfinite3(old_radiance_raw.rgb) && isfinite(old_radiance_raw.a));
    let new_radiance_raw = world_cache.active_cells_new_radiance[active_cell_id.x];
    let new_radiance = select(vec3(0.0), new_radiance_raw, isfinite3(new_radiance_raw));
    let luminance_delta_raw = world_cache.luminance_deltas[cell_index];
    let luminance_delta = select(0.0, luminance_delta_raw, isfinite(luminance_delta_raw));

    // https://bsky.app/profile/gboisse.bsky.social/post/3m5blga3ftk2a
    let sample_count = select(
        min(max(old_radiance.a, 0.0) + 1.0, constants.world_cache_max_temporal_samples),
        1.0,
        bool(constants.reset),
    );
    let alpha = abs(luminance_delta) / max(luminance(old_radiance.rgb), 0.001);
    let max_sample_count = mix(constants.world_cache_max_temporal_samples, 1.0, pow(saturate(alpha), 1.0 / 8.0));
    var blend_amount = 1.0 / min(sample_count, max_sample_count);
    if bool(constants.reset) {
        blend_amount = 1.0;
    }

    let blended_radiance = mix(old_radiance.rgb, new_radiance, blend_amount);
    let blended_luminance_delta = select(mix(luminance_delta, luminance(blended_radiance) - luminance(old_radiance.rgb), 1.0 / 8.0), 0.0, bool(constants.reset));

    world_cache.radiance[cell_index] = vec4(blended_radiance, sample_count);
    world_cache.luminance_deltas[cell_index] = blended_luminance_delta;
}

fn sample_random_light_ris(world_position: vec3<f32>, world_normal: vec3<f32>, ray_origin_bias: f32, workgroup_id: vec2<u32>, rng: ptr<function, u32>) -> vec3<f32> {
    if constants.world_cache_direct_light_sample_count == 0u {
        return vec3(0.0);
    }
    // Indirect 1D dispatch, so `workgroup_id.y` is always 0 and without a frame term each block's
    // cells share one fixed light-tile subset for the life of the app.
    var workgroup_rng = (workgroup_id.x * 0x9E3779B9u) + workgroup_id.y + constants.frame_rng;
    let light_tile_start = rand_range_u(128u, &workgroup_rng) * 1024u;

    var weight_sum = 0.0;
    var selected_sample_radiance = vec3(0.0);
    var selected_sample_target_function = 0.0;
    let mis_weight = 1.0 / f32(constants.world_cache_direct_light_sample_count);
    for (var i = 0u; i < constants.world_cache_direct_light_sample_count; i++) {
        let tile_sample = light_tile_start + rand_range_u(1024u, rng);
        let resolved_light_sample = unpack_resolved_light_sample(light_tile_resolved_samples[tile_sample], view.exposure);
        let light_contribution = calculate_resolved_light_contribution(resolved_light_sample, world_position, world_normal);

        let visibility = trace_visibility(world_position, world_normal, resolved_light_sample.world_position, ray_origin_bias);
        let contribution = light_contribution.radiance * saturate(dot(light_contribution.wi, world_normal)) * visibility;
        let target_function = luminance(contribution);
        let resampling_weight = mis_weight * (target_function * light_contribution.inverse_pdf);
        if !(resampling_weight > 0.0) || !isfinite(resampling_weight) || !isfinite3(contribution) {
            continue;
        }

        weight_sum += resampling_weight;

        if rand_f(rng) * weight_sum < resampling_weight {
            selected_sample_radiance = contribution;
            selected_sample_target_function = target_function;
        }
    }

    var unbiased_contribution_weight = 0.0;
    if any(selected_sample_radiance != vec3(0.0)) && isfinite(weight_sum) {
        let inverse_target_function = select(0.0, 1.0 / selected_sample_target_function, selected_sample_target_function > 0.0);
        unbiased_contribution_weight = weight_sum * inverse_target_function;
    }

    return selected_sample_radiance * unbiased_contribution_weight;
}
