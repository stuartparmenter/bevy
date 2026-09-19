//! Resolve visible exact grass geometry once per frame. Lighting reuses the
//! anchor, including the prior frame's geometry for temporal visibility.
import package::realtime::bindings::{constants, view, gbuffer, depth_buffer, grass_receiver_anchors, previous_grass_receiver_anchors};
import package::realtime::gbuffer_utils::{ResolvedGPixel, gpixel_resolve};
import package::realtime::grass_receiver_geometry::{match_receiver, encode_receiver_anchor, resolve_receiver_anchor};

@compute @workgroup_size(8, 8, 1)
fn resolve_grass_receivers(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    if any(pixel >= vec2<u32>(view.main_pass_viewport.zw)) { return; }
    // Always overwrite misses/background/disabled pixels, never retain an old owner.
    var anchor = vec4<u32>(0xFFFFFFFFu, 0u, 0u, 0u);
    let depth = textureLoad(depth_buffer, pixel, 0);
    if depth > 0.0 && bool(constants.grass_rt_receiver_origin) && bool(constants.grass_geometric_ray_origin) {
        let surface = gpixel_resolve(textureLoad(gbuffer, pixel, 0), depth, pixel, view.main_pass_viewport.zw, view.world_from_clip);
        anchor = encode_receiver_anchor(match_receiver(surface, pixel));
    }
    textureStore(grass_receiver_anchors, pixel, anchor);
}

fn refine_grass_receiver(surface: ResolvedGPixel, pixel: vec2<u32>, previous: bool) -> ResolvedGPixel {
    var refined = surface;
    if !surface.is_grass || !bool(constants.grass_rt_receiver_origin) || !bool(constants.grass_geometric_ray_origin) {
        return refined;
    }
    var anchor: vec4<u32>;
    if previous {
        // No previous texture reads on allocation, skipped frames, or explicit reset.
        if bool(constants.reset) { return refined; }
        anchor = textureLoad(previous_grass_receiver_anchors, pixel, 0);
    } else {
        anchor = textureLoad(grass_receiver_anchors, pixel);
    }
    if anchor.x == 0xFFFFFFFFu { return refined; }
    let receiver = resolve_receiver_anchor(anchor, previous);
    if receiver.matched {
        refined.ray_origin = receiver.position;
        refined.geometric_normal = vec4(receiver.normal, 0.000001);
    } else {
        // A formerly matched receiver was destroyed/reassigned. Reject its
        // temporal reservoir rather than substitute a different ray origin.
        refined.ray_origin_valid = false;
    }
    return refined;
}
