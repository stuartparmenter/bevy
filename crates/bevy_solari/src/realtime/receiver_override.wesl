//! Receiver overrides: one `rgba32uint` texel per pixel that replaces the ray origin policy of
//! that pixel's primary surface. Producers write texels with the `encode_*` functions during
//! `SolariLightingSystems::WriteReceiverOverrides`; lighting reads them back with
//! `apply_receiver_override_texel`. `texel.x >> 24` is the texel kind; kind 0 is no override,
//! so a cleared texture leaves every pixel on the ordinary policy.
import bevy_render::utils::octahedral_decode;
import package::realtime::gbuffer_utils::ResolvedGPixel;
import package::scene::bindings::EXACT_NORMAL_TANGENT_GUARD;
import package::scene::triangle_anchor::resolve_triangle_anchor;

enable wgpu_ray_query;

/// Rays start at the reconstructed surface position, without the shading normal push.
/// `x & 1` = a face normal follows; `y` = bits(tangent guard); `z, w` = bits(octahedral
/// coordinates of the normal in [0, 1], as `bevy_render::utils::octahedral_decode` takes them).
const RECEIVER_OVERRIDE_SURFACE = 1u;
/// Rays start on a triangle of the raytracing scene, offset along its face normal.
/// `x & 0xFFFFFF` = instance slot; `y` = primitive index; `z, w` = bits(barycentrics).
const RECEIVER_OVERRIDE_TRIANGLE = 2u;

/// A thin two-sided surface whose face normal is unknown. Its ray origins are not offset.
fn encode_receiver_override_surface() -> vec4<u32> {
    return vec4<u32>(RECEIVER_OVERRIDE_SURFACE << 24u, 0u, 0u, 0u);
}

/// A thin two-sided surface with a face normal of either sign. `tangent_guard` is the sine of
/// the normal's angular uncertainty; see `offset_surface_ray_origin`.
fn encode_receiver_override_surface_normal(octahedral_normal: vec2<f32>, tangent_guard: f32) -> vec4<u32> {
    return vec4<u32>((RECEIVER_OVERRIDE_SURFACE << 24u) | 1u, bitcast<u32>(tangent_guard), bitcast<vec2<u32>>(octahedral_normal));
}

/// A point on a raytracing scene triangle that coincides with the rasterized surface.
/// `instance_id` is this frame's instance slot, as `RayIntersection::instance_custom_data` reports it.
fn encode_receiver_override_triangle(instance_id: u32, primitive_id: u32, barycentrics: vec2<f32>) -> vec4<u32> {
    return vec4<u32>((RECEIVER_OVERRIDE_TRIANGLE << 24u) | (instance_id & 0xFFFFFFu), primitive_id, bitcast<vec2<u32>>(barycentrics));
}

/// `previous`: `texel` and `surface` are from the previous frame. A previous-frame triangle that
/// does not resolve clears `ray_origin_valid`: no other origin describes that surface's history.
fn apply_receiver_override_texel(surface: ResolvedGPixel, texel: vec4<u32>, previous: bool) -> ResolvedGPixel {
    var overridden = surface;
    let kind = texel.x >> 24u;
    if kind == RECEIVER_OVERRIDE_SURFACE {
        overridden.ray_origin = surface.world_position;
        overridden.geometric_normal = vec4(0.0);
        if (texel.x & 1u) != 0u {
            overridden.geometric_normal = vec4(octahedral_decode(bitcast<vec2<f32>>(texel.zw)), bitcast<f32>(texel.y));
        }
    } else if kind == RECEIVER_OVERRIDE_TRIANGLE {
        let anchor = resolve_triangle_anchor(texel.x & 0xFFFFFFu, texel.y, bitcast<vec2<f32>>(texel.zw), previous);
        if anchor.valid {
            overridden.ray_origin = anchor.position;
            overridden.geometric_normal = vec4(anchor.normal, EXACT_NORMAL_TANGENT_GUARD);
        } else if previous {
            overridden.ray_origin_valid = false;
        } else {
            overridden.ray_origin = surface.world_position;
            overridden.geometric_normal = vec4(0.0);
        }
    }
    return overridden;
}
