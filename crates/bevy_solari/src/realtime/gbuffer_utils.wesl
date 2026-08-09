import bevy_pbr::deferred::types::unpack_24bit_normal;
import bevy_pbr::render::rgb9e5::rgb9e5_to_vec3_;
import bevy_render::utils::octahedral_decode;
import bevy_render::view::{View, depth_ndc_to_view_z};
import bevy_solari::scene::bindings::ResolvedMaterial;

struct ResolvedGPixel {
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    material: ResolvedMaterial,
}

fn gpixel_resolve(gpixel: vec4<u32>, depth: f32, pixel_id: vec2<u32>, view_size: vec2<f32>, world_from_clip: mat4x4<f32>) -> ResolvedGPixel {
    let world_position = reconstruct_world_position(pixel_id, depth, view_size, world_from_clip);
    let world_normal = octahedral_decode(unpack_24bit_normal(gpixel.a));

    let base_rough = unpack4x8unorm(gpixel.r);
    let base_color = pow(base_rough.rgb, vec3(2.2));
    let perceptual_roughness = base_rough.a;
    let roughness = perceptual_roughness * perceptual_roughness;
    let props = unpack4x8unorm(gpixel.b);
    let reflectance = props.r;
    let metallic = props.g;
    let emissive = rgb9e5_to_vec3_(gpixel.g);
    let material = ResolvedMaterial(base_color, emissive, reflectance, perceptual_roughness, roughness, metallic);

    return ResolvedGPixel(world_position, world_normal, material);
}

fn reconstruct_world_position(pixel_id: vec2<u32>, depth: f32, view_size: vec2<f32>, world_from_clip: mat4x4<f32>) -> vec3<f32> {
    let uv = (vec2<f32>(pixel_id) + 0.5) / view_size;
    let xy_ndc = (uv - vec2(0.5)) * vec2(2.0, -2.0);
    let world_pos = world_from_clip * vec4(xy_ndc, depth, 1.0);
    return world_pos.xyz / world_pos.w;
}

// Reject if tangent plane differs more than 0.3% or angle between normals more than 90 degrees
fn pixel_dissimilar(depth: f32, world_position: vec3<f32>, other_world_position: vec3<f32>, normal: vec3<f32>, other_normal: vec3<f32>, view: View) -> bool {
    return pixel_dissimilar_motion_compensated(depth, world_position, other_world_position, normal, other_normal, 0.0, view);
}

// Same test, with the tangent-plane threshold widened by `plane_slack` world units. Temporal call
// sites pass the slack from `temporal_object_motion_slack` so a moving surface's own frame-to-frame
// displacement doesn't reject its history; zero slack is exactly the strict test.
fn pixel_dissimilar_motion_compensated(depth: f32, world_position: vec3<f32>, other_world_position: vec3<f32>, normal: vec3<f32>, other_normal: vec3<f32>, plane_slack: f32, view: View) -> bool {
    // https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22699-fast-denoising-with-self-stabilizing-recurrent-blurs.pdf#page=45
    let tangent_plane_distance = abs(dot(normal, other_world_position - world_position));
    let view_z = -depth_ndc_to_view_z(depth, view.clip_from_view, view.view_from_clip);

    return (tangent_plane_distance - plane_slack) / view_z > 0.003 || dot(normal, other_normal) < 0.0;
}

// Sub-pixel motion-vector residual is camera jitter and reconstruction noise, not object motion
const TEMPORAL_MOTION_DEAD_BAND_PIXELS = 1.5;
// Cap on motion slack as a fraction of view-space depth, so history from a genuinely different
// surface (e.g. background revealed by a fast-moving object) is still rejected
const TEMPORAL_MOTION_SLACK_CAP = 0.1;

// Estimates how far the surface visible at `pixel_id` moved in world space since the previous
// frame, for widening the temporal tangent-plane test. The camera-induced part of the motion
// vector cancels by reprojecting the current world position with the previous frame's unjittered
// view-projection; the residual is pure object motion as seen on screen, converted to world units
// at the pixel's depth. Only motion perpendicular to the view ray is observable this way — motion
// along it gets no slack (conservative). Sub-pixel residuals return exactly 0.0, keeping static
// scenes and camera-only motion bit-identical to the uncompensated test.
fn temporal_object_motion_slack(
    pixel_id: vec2<u32>,
    depth: f32,
    world_position: vec3<f32>,
    motion_vector: vec2<f32>,
    view: View,
    previous_unjittered_clip_from_world: mat4x4<f32>,
) -> f32 {
    let previous_clip = previous_unjittered_clip_from_world * vec4(world_position, 1.0);
    if previous_clip.w <= 0.0 {
        return 0.0;
    }
    let static_previous_uv = (previous_clip.xy / previous_clip.w) * vec2(0.5, -0.5) + 0.5;

    let uv = (vec2<f32>(pixel_id) + 0.5) / view.main_pass_viewport.zw;
    let residual_uv = (uv - motion_vector) - static_previous_uv;
    let residual_pixels = residual_uv * view.main_pass_viewport.zw;
    if dot(residual_pixels, residual_pixels) < TEMPORAL_MOTION_DEAD_BAND_PIXELS * TEMPORAL_MOTION_DEAD_BAND_PIXELS {
        return 0.0;
    }

    // World-space span of the residual at this pixel's depth, valid for any projection
    let xy_ndc = (uv - vec2(0.5)) * vec2(2.0, -2.0);
    let residual_ndc = residual_uv * vec2(2.0, -2.0);
    let shifted_world = view.world_from_clip * vec4(xy_ndc + residual_ndc, depth, 1.0);
    let world_motion = length(shifted_world.xyz / shifted_world.w - world_position);

    let view_z = -depth_ndc_to_view_z(depth, view.clip_from_view, view.view_from_clip);
    return min(world_motion, TEMPORAL_MOTION_SLACK_CAP * view_z);
}

fn permute_pixel(pixel_id: vec2<u32>, frame_index: u32, view_size: vec2<f32>) -> vec2<u32> {
    let r = frame_index;
    let offset = vec2<i32>(i32(r & 3u), i32((r >> 2u) & 3u));
    var shifted_pixel_id = vec2<i32>(pixel_id) + offset;
    shifted_pixel_id ^= vec2<i32>(3);
    shifted_pixel_id -= offset;
    if any(shifted_pixel_id < vec2<i32>(0)) || any(shifted_pixel_id >= vec2<i32>(view_size)) {
        return pixel_id;
    }
    return vec2<u32>(shifted_pixel_id);
}
