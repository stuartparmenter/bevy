import bevy_pbr::deferred::types::{NO_DIRECTIONAL_OCCLUSION, decode_fresnel_ratio, has_directional_occlusion, unpack_24bit_normal, unpack_directional_occlusion, unpack_diffuse_transmission, unpack_emissive, unpack_specular_tint};
import bevy_pbr::render::pbr_functions::dielectric_reflectance;
import bevy_render::utils::octahedral_decode;
import bevy_render::view::{View, depth_ndc_to_view_z};
import package::realtime::bindings::{gbuffer, gbuffer_specular_tint, previous_gbuffer, previous_gbuffer_specular_tint, previous_view, view};
import package::scene::bindings::{NORMAL_OFFSET_ORIGIN, ResolvedMaterial};

struct ResolvedGPixel {
    /// The point rays leave this surface from, before `offset_surface_ray_origin`.
    ray_origin: vec3<f32>,
    /// The surface's `geometric_normal` for `offset_surface_ray_origin`.
    geometric_normal: vec4<f32>,
    /// False when `ray_origin` could not be reconstructed, which makes the surface's history unusable.
    ray_origin_valid: bool,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    material: ResolvedMaterial,
}

/// Loads and resolves the current frame's G-buffer at `pixel_id`, which has depth `depth`.
fn load_gpixel(pixel_id: vec2<u32>, depth: f32) -> ResolvedGPixel {
    return gpixel_resolve(textureLoad(gbuffer, pixel_id, 0), textureLoad(gbuffer_specular_tint, pixel_id, 0).r, depth, pixel_id, view.main_pass_viewport.zw, view.world_from_clip);
}

/// Loads and resolves the previous frame's G-buffer at `pixel_id`, which had depth `depth`.
fn load_previous_gpixel(pixel_id: vec2<u32>, depth: f32) -> ResolvedGPixel {
    return gpixel_resolve(textureLoad(previous_gbuffer, pixel_id, 0), textureLoad(previous_gbuffer_specular_tint, pixel_id, 0).r, depth, pixel_id, view.main_pass_viewport.zw, previous_view.world_from_clip);
}

fn gpixel_resolve(gpixel: vec4<u32>, gpixel_tint_transmission: u32, depth: f32, pixel_id: vec2<u32>, view_size: vec2<f32>, world_from_clip: mat4x4<f32>) -> ResolvedGPixel {
    let world_position = reconstruct_world_position(pixel_id, depth, view_size, world_from_clip);
    let wo = normalize(reconstruct_world_position(pixel_id, 1.0, view_size, world_from_clip) - world_position);
    let world_normal = face_view(octahedral_decode(unpack_24bit_normal(gpixel.a)), wo);

    let base_rough = unpack4x8unorm(gpixel.r);
    let base_color = pow(base_rough.rgb, vec3(2.2));
    let perceptual_roughness = base_rough.a;
    let roughness = perceptual_roughness * perceptual_roughness;
    let props = unpack4x8unorm(gpixel.b);
    let dielectric = dielectric_reflectance(decode_fresnel_ratio(props.r));
    let metallic = props.g;
    let emissive = unpack_emissive(gpixel);
    let directional_occlusion = select(NO_DIRECTIONAL_OCCLUSION, unpack_directional_occlusion(gpixel), has_directional_occlusion(gpixel));
    let specular_tint = unpack_specular_tint(gpixel_tint_transmission);
    let diffuse_transmission = unpack_diffuse_transmission(gpixel_tint_transmission);
    let material = ResolvedMaterial(base_color, emissive, dielectric.F0, dielectric.F90, specular_tint, perceptual_roughness, roughness, metallic, diffuse_transmission, false, directional_occlusion);

    return ResolvedGPixel(world_position, vec4(world_normal, NORMAL_OFFSET_ORIGIN), true, world_position, world_normal, material);
}

fn reconstruct_world_position(pixel_id: vec2<u32>, depth: f32, view_size: vec2<f32>, world_from_clip: mat4x4<f32>) -> vec3<f32> {
    let uv = (vec2<f32>(pixel_id) + 0.5) / view_size;
    let xy_ndc = (uv - vec2(0.5)) * vec2(2.0, -2.0);
    let world_pos = world_from_clip * vec4(xy_ndc, depth, 1.0);
    return world_pos.xyz / world_pos.w;
}

/// Reject if tangent plane differs more than 0.3% or angle between normals more than 90 degrees
fn pixel_dissimilar(depth: f32, world_position: vec3<f32>, other_world_position: vec3<f32>, normal: vec3<f32>, other_normal: vec3<f32>, view: View) -> bool {
    return pixel_dissimilar_motion_compensated(depth, world_position, other_world_position, normal, other_normal, 0.0, view);
}

/// Applies `pixel_dissimilar` with an additional tangent-plane tolerance in world units.
/// Temporal reuse supplies object motion slack; zero slack keeps the original test.
fn pixel_dissimilar_motion_compensated(depth: f32, world_position: vec3<f32>, other_world_position: vec3<f32>, normal: vec3<f32>, other_normal: vec3<f32>, plane_slack: f32, view: View) -> bool {
    // https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22699-fast-denoising-with-self-stabilizing-recurrent-blurs.pdf#page=45
    let tangent_plane_distance = abs(dot(normal, other_world_position - world_position));
    let view_z = -depth_ndc_to_view_z(depth, view.clip_from_view, view.view_from_clip);

    return (tangent_plane_distance - plane_slack) / view_z > 0.003 || dot(normal, other_normal) < 0.0;
}

/// Ignore small motion residuals caused by camera jitter and reconstruction noise
const TEMPORAL_MOTION_DEAD_BAND_PIXELS = 1.5;
/// Limit slack relative to view depth to reject history from disoccluded surfaces
const TEMPORAL_MOTION_SLACK_CAP = 0.1;

/// Estimates object motion in world units for temporal validation.
/// Subtracts camera motion using the previous unjittered view-projection, then converts
/// the screen-space residual at the current depth. Motion along the view ray is not
/// observable. Residuals below the dead band return zero slack.
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

/// Least cosine between a primary surface's shading normal and the direction to the camera.
const MIN_PRIMARY_N_DOT_V = 0.01;

/// `normal` leaned toward `wo` until it faces the camera. A normal map tilts some texels of a
/// surface seen edge-on past its silhouette, where the BRDF has no value and the pixel goes black.
fn face_view(normal: vec3<f32>, wo: vec3<f32>) -> vec3<f32> {
    let n_dot_v = dot(normal, wo);
    if n_dot_v >= MIN_PRIMARY_N_DOT_V {
        return normal;
    }
    return normalize(normal + (MIN_PRIMARY_N_DOT_V - n_dot_v) * wo);
}
