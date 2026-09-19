//! Shared grass receiver geometry for production cached ray origins and the
//! diagnostic reference. Camera-ray matching is conservative: coincident blades
//! remain ambiguous without raster primitive IDs. Cached anchors store RT
//! instance/triangle/barycentric identity, not displaced shading positions.
import bevy_render::maths::affine3_to_square;
import package::realtime::gbuffer_utils::{ResolvedGPixel, reconstruct_world_position};
import package::realtime::bindings::view;
import package::scene::bindings::{geometry_ids, transforms, previous_frame_transforms,
    previous_frame_instance_id_translations, INSTANCE_NOT_PRESENT_THIS_FRAME,
    load_vertices, transform_positions, trace_ray, RAY_T_MIN, RAY_T_MAX};

enable wgpu_ray_query;

struct Receiver {
    matched: bool,
    // Current instance slot even when resolving a previous-frame anchor.
    instance_id: u32,
    primitive_id: u32,
    barycentrics: vec2<f32>,
    // Previous-frame positions/normals when `previous` is true.
    position: vec3<f32>,
    normal: vec3<f32>,
}

const NO_RECEIVER = 0xFFFFFFFFu;

fn encode_receiver_anchor(receiver: Receiver) -> vec4<u32> {
    if !receiver.matched { return vec4<u32>(NO_RECEIVER, 0u, 0u, 0u); }
    return vec4<u32>(receiver.instance_id, receiver.primitive_id,
        bitcast<u32>(receiver.barycentrics.x), bitcast<u32>(receiver.barycentrics.y));
}

fn resolve_receiver_anchor(anchor: vec4<u32>, previous: bool) -> Receiver {
    var receiver: Receiver;
    if anchor.x == NO_RECEIVER { return receiver; }
    var instance = anchor.x;
    if previous {
        if instance >= arrayLength(&previous_frame_instance_id_translations) { return receiver; }
        instance = previous_frame_instance_id_translations[instance];
        // This mapping is invalidated on removal, slot reuse, or any topology
        // generation change. Reject BEFORE touching current indices/vertices.
        if instance == INSTANCE_NOT_PRESENT_THIS_FRAME { return receiver; }
    }
    if instance >= arrayLength(&geometry_ids) || instance >= arrayLength(&transforms) { return receiver; }
    if previous && instance >= arrayLength(&previous_frame_transforms) { return receiver; }
    var geometry = geometry_ids[instance];
    if geometry.ray_mask != 0x02u || geometry.diagnostic_primitives_per_group != 9u
        || anchor.y >= geometry.triangle_count { return receiver; }
    let bary = bitcast<vec2<f32>>(anchor.zw);
    // Positive comparisons also reject NaNs; tolerate tiny ray-query edge
    // roundoff without clamping away the actual intersection barycentrics.
    if !all(bary >= vec2<f32>(-1e-5)) || !all(bary <= vec2<f32>(1.00001))
        || !(bary.x + bary.y <= 1.00001) { return receiver; }
    var transform = transforms[instance];
    if previous {
        geometry.vertex_buffer_id = geometry.previous_vertex_buffer_id;
        transform = previous_frame_transforms[instance];
    }
    let vertices = load_vertices(geometry, anchor.y);
    let points = transform_positions(affine3_to_square(transform), vertices);
    // Edge form avoids summing three large world positions at grass scale.
    let position = points[0] + (points[1] - points[0]) * bary.x + (points[2] - points[0]) * bary.y;
    let cross_normal = cross(points[1] - points[0], points[2] - points[0]);
    let normal_length_squared = dot(cross_normal, cross_normal);
    if !(normal_length_squared > 1e-20) || !(normal_length_squared < 1e30)
        || !all(abs(position) < vec3<f32>(1e30)) { return receiver; }
    let normal = cross_normal * inverseSqrt(normal_length_squared);
    return Receiver(true, instance, anchor.y, bary, position, normal);
}

fn match_receiver(surface: ResolvedGPixel, pixel: vec2<u32>) -> Receiver {
    var receiver: Receiver;
    if !surface.is_grass || !(dot(surface.receiver_geometric_normal, surface.receiver_geometric_normal) > 0.5) { return receiver; }
    // Start at this pixel's near plane, supporting perspective and orthographic
    // cameras without accidentally selecting geometry clipped from the raster view.
    // Like Solari's G-buffer resolve, receiver matching assumes viewport origin zero.
    if any(view.main_pass_viewport.xy != vec2(0.0)) { return receiver; }
    let camera_origin = reconstruct_world_position(pixel, 1.0, view.main_pass_viewport.zw, view.world_from_clip);
    let delta = surface.world_position - camera_origin;
    let distance = length(delta);
    if !(distance > RAY_T_MIN) || !(distance < RAY_T_MAX) { return receiver; }
    // Conservative matching, not a mathematical reconstruction-error bound.
    // Cap the tolerance so large world coordinates cannot silently match another
    // blade centimetres away. Prefer an unmatched fallback to a false anchor.
    let magnitude = max(max(abs(surface.world_position.x), abs(surface.world_position.y)), abs(surface.world_position.z));
    let tolerance = clamp(8.0 * 1.1920928955078125e-7 * magnitude, 0.00025, 0.002);
    let hit = trace_ray(camera_origin, delta / distance, RAY_T_MIN, distance + tolerance, RAY_FLAG_NONE);
    if hit.kind == RAY_QUERY_INTERSECTION_NONE { return receiver; }
    receiver = resolve_receiver_anchor(vec4<u32>(hit.instance_custom_data, hit.primitive_index,
        bitcast<u32>(hit.barycentrics.x), bitcast<u32>(hit.barycentrics.y)), false);
    if !receiver.matched { return receiver; }
    if !(length(receiver.position - surface.world_position) <= tolerance)
        || !(abs(dot(receiver.normal, surface.receiver_geometric_normal)) >= 0.995) {
        receiver.matched = false;
    }
    return receiver;
}
