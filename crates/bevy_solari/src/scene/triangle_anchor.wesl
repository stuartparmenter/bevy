//! Resolves a triangle anchor (instance slot, primitive index, barycentrics) to a
//! world-space point and face normal, on current or previous-frame geometry.
import bevy_render::maths::affine3_to_square;
import package::scene::bindings::{geometry_ids, transforms, previous_frame_transforms,
    previous_frame_instance_id_translations, INSTANCE_NOT_PRESENT_THIS_FRAME,
    load_vertices, transform_positions};

enable wgpu_ray_query;

struct TriangleAnchor {
    valid: bool,
    // On previous-frame vertices and transform when `previous` is true.
    position: vec3<f32>,
    normal: vec3<f32>,
}

/// `previous`: `instance_id` is a previous-frame slot; resolve on previous vertices and transform.
fn resolve_triangle_anchor(instance_id: u32, primitive_id: u32, barycentrics: vec2<f32>, previous: bool) -> TriangleAnchor {
    var anchor: TriangleAnchor;
    var instance = instance_id;
    if previous {
        if instance >= arrayLength(&previous_frame_instance_id_translations) { return anchor; }
        instance = previous_frame_instance_id_translations[instance];
        // Translate and reject before any geometry read: the previous slot can hold another instance.
        if instance == INSTANCE_NOT_PRESENT_THIS_FRAME { return anchor; }
    }
    if instance >= arrayLength(&geometry_ids) || instance >= arrayLength(&transforms) { return anchor; }
    if previous && instance >= arrayLength(&previous_frame_transforms) { return anchor; }
    var geometry = geometry_ids[instance];
    if primitive_id >= geometry.triangle_count { return anchor; }
    // Positive comparisons also reject NaNs; tolerate tiny ray-query edge
    // roundoff without clamping away the actual intersection barycentrics.
    if !all(barycentrics >= vec2<f32>(-1e-5)) || !all(barycentrics <= vec2<f32>(1.00001))
        || !(barycentrics.x + barycentrics.y <= 1.00001) { return anchor; }
    var transform = transforms[instance];
    if previous {
        geometry.vertex_buffer_id = geometry.previous_vertex_buffer_id;
        transform = previous_frame_transforms[instance];
    }
    let vertices = load_vertices(geometry, primitive_id);
    let points = transform_positions(affine3_to_square(transform), vertices);
    // Edge form avoids summing three large world positions.
    let position = points[0] + (points[1] - points[0]) * barycentrics.x + (points[2] - points[0]) * barycentrics.y;
    let cross_normal = cross(points[1] - points[0], points[2] - points[0]);
    let normal_length_squared = dot(cross_normal, cross_normal);
    if !(normal_length_squared > 1e-20) || !(normal_length_squared < 1e30)
        || !all(abs(position) < vec3<f32>(1e30)) { return anchor; }
    let normal = cross_normal * inverseSqrt(normal_length_squared);
    return TriangleAnchor(true, position, normal);
}
