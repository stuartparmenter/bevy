//! Directional light visibility in distance tiers (`ShadowTiers`): the scene inside the near
//! sphere, partial occluders inside the mid sphere, and the application's far field beyond
//! (`FarFieldVisibility`).
import bevy_pbr::render::utils::rand_u;
import package::scene::bindings::{occluded, shadow_tiers, RAY_T_MIN, RAY_T_MAX, RAY_MASK_ALL_RAYS};
@if(FAR_FIELD_VISIBILITY) import package::scene::far_field::far_field_visibility;

enable wgpu_ray_query;

/// Metres of origin that share a partial occluder bit, so a surface keeps its bit under jitter
/// and reuse while its light sample does.
const PARTIAL_MASK_CELL = 0.25;

/// Visibility of a directional light from `origin` along the unit `direction`, in this frame's
/// scene or with `previous` the previous frame's.
fn directional_visibility(origin: vec3<f32>, direction: vec3<f32>, previous: bool) -> f32 {
    if shadow_tiers.tiered == 0u {
        return f32(!occluded(previous, origin, direction, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE, RAY_MASK_ALL_RAYS));
    }
    let sphere = shadow_tiers.spheres[u32(previous)];
    let near_t = sphere_exit(origin, direction, sphere.center, sphere.near_radius);
    if near_t > RAY_T_MIN && occluded(previous, origin, direction, RAY_T_MIN, near_t, RAY_FLAG_NONE, RAY_MASK_ALL_RAYS) {
        return 0.0;
    }
    // Without a far field the mid tier reaches as far as the scene does.
    var mid_t = RAY_T_MAX;
    @if(FAR_FIELD_VISIBILITY) { mid_t = sphere_exit(origin, direction, sphere.center, sphere.mid_radius); }
    if mid_t > near_t + RAY_T_MIN && occluded(previous, origin, direction, max(near_t, RAY_T_MIN), mid_t, RAY_FLAG_CULL_BACK_FACING, partial_mask(origin, direction)) {
        return 0.0;
    }
    var visibility = 1.0;
    @if(FAR_FIELD_VISIBILITY) { visibility = far_field_visibility(origin, direction, near_t, mid_t, previous); }
    return visibility;
}

/// Distance along the ray at which it leaves the sphere, 0 for an origin outside it.
fn sphere_exit(origin: vec3<f32>, direction: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    let oc = origin - center;
    let c = dot(oc, oc) - radius * radius;
    if c >= 0.0 {
        return 0.0;
    }
    let b = dot(oc, direction);
    return sqrt(b * b - c) - b;
}

/// One of the partial occluder mask bits, chosen by the ray's origin cell and direction.
fn partial_mask(origin: vec3<f32>, direction: vec3<f32>) -> u32 {
    let cell = bitcast<vec3<u32>>(vec3<i32>(floor(origin / PARTIAL_MASK_CELL)));
    var state = (cell.x * 0x9E3779B1u) ^ (cell.y * 0x85EBCA77u) ^ (cell.z * 0xC2B2AE3Du)
        ^ (bitcast<u32>(direction.x) * 0x27D4EB2Fu) ^ (bitcast<u32>(direction.y) * 0x165667B1u) ^ bitcast<u32>(direction.z);
    return 2u << (rand_u(&state) % 7u);
}
