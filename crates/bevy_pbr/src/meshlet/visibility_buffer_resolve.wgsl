#define_import_path bevy_pbr::meshlet_visibility_buffer_resolve

#import bevy_pbr::{
    meshlet_bindings::{
        Meshlet,
        meshlet_visibility_buffer,
        meshlet_raster_clusters,
        MeshletGpuDescriptor,
        meshlet_instance_descriptors,
        load_meshlet,
        meshlet_instance_uniforms,
        get_meshlet_vertex_id,
        get_meshlet_vertex_position,
        get_meshlet_vertex_normal,
        get_meshlet_vertex_uv,
    },
    mesh_view_bindings::view,
    mesh_functions::mesh_position_local_to_world,
    mesh_types::{Mesh, MESH_FLAGS_SIGN_DETERMINANT_MODEL_3X3_BIT},
    view_transformations::position_world_to_clip,
}
#import bevy_render::{
    maths::{affine3_to_square, mat2x4_f32_to_mat3x3_unpack},
    view::frag_coord_to_ndc,
}

#ifdef PREPASS_FRAGMENT
#ifdef MOTION_VECTOR_PREPASS
#import bevy_pbr::{
    prepass_bindings::previous_view_uniforms,
    pbr_prepass_functions::calculate_motion_vector,
}
#endif
#endif

/// Functions to be used by materials for reading from a meshlet visibility buffer texture.

#ifdef MESHLET_MESH_MATERIAL_PASS
struct PartialDerivatives {
    barycentrics: vec3<f32>,
    ddx: vec3<f32>,
    ddy: vec3<f32>,
    // Twice the signed NDC area of the triangle. Positive means the triangle winds
    // counter-clockwise on screen, i.e. what `@builtin(front_facing)` reports as front.
    ndc_double_area: f32,
}

// A triangle that rasterized has a non-zero homogeneous area, and a pixel inside it has
// non-zero homogeneous weights. Both still reach zero for a triangle rasterized
// degenerately or one whose plane passes through the eye, and the resulting inf/NaN
// propagates into the deferred G-buffer as pure black. Floor the divisor well below
// anything a well-formed triangle produces so healthy fragments are unchanged.
const RECIPROCAL_EPSILON: f32 = 1e-20;

fn safe_inverse(x: f32) -> f32 {
    // Sign-preserving, and bit-identical to 1.0 / x whenever abs(x) >= RECIPROCAL_EPSILON.
    return select(-1.0, 1.0, x >= 0.0) / max(abs(x), RECIPROCAL_EPSILON);
}

// Perspective-correct barycentrics of the pixel at `ndc` on a triangle given by its
// clip-space (x, y, w) vertices, in 2D homogeneous coordinates (Olano & Greer, "Triangle
// Scan Conversion using 2D Homogeneous Coordinates"). The clip position of any point on
// the triangle is the same barycentric mix of the vertex clip positions, so its screen
// position matches the pixel exactly when `dot(b, x - ndc.x * w) == 0` and
// `dot(b, y - ndc.y * w) == 0`; the cross product of those two rows solves for `b` up to
// scale, and the scale is fixed by `dot(b, 1) == 1`. Nothing here divides by a vertex
// `w`, so a triangle with vertices behind the eye resolves as exactly as one in front:
// the resolve reprojects unclipped world positions, and any ground plane the camera
// stands on has vertices behind it.
fn homogeneous_barycentrics(clip_x: vec3<f32>, clip_y: vec3<f32>, clip_w: vec3<f32>, ndc: vec2<f32>) -> vec3<f32> {
    let weights = cross(clip_x - ndc.x * clip_w, clip_y - ndc.y * clip_w);
    return weights * safe_inverse(dot(weights, vec3(1.0)));
}

// two_over_screen_size converts the per-NDC derivatives to per-pixel: one pixel spans
// 2/viewport NDC units. NDC y points up the screen, so the y derivative steps down.
fn compute_partial_derivatives(vertex_world_positions: array<vec4<f32>, 3>, ndc_uv: vec2<f32>, two_over_screen_size: vec2<f32>) -> PartialDerivatives {
    var result: PartialDerivatives;

    let vertex_clip_position_0 = position_world_to_clip(vertex_world_positions[0].xyz);
    let vertex_clip_position_1 = position_world_to_clip(vertex_world_positions[1].xyz);
    let vertex_clip_position_2 = position_world_to_clip(vertex_world_positions[2].xyz);

    let clip_x = vec3(vertex_clip_position_0.x, vertex_clip_position_1.x, vertex_clip_position_2.x);
    let clip_y = vec3(vertex_clip_position_0.y, vertex_clip_position_1.y, vertex_clip_position_2.y);
    let clip_w = vec3(vertex_clip_position_0.w, vertex_clip_position_1.w, vertex_clip_position_2.w);

    // The homogeneous determinant carries the screen winding of the rasterized part of the
    // triangle whatever the sign of each vertex's w; with every vertex in front of the eye
    // it is the NDC double area scaled by the product of the three positive w.
    result.ndc_double_area = determinant(mat3x3(
        vertex_clip_position_0.xyw,
        vertex_clip_position_1.xyw,
        vertex_clip_position_2.xyw,
    ));

    result.barycentrics = homogeneous_barycentrics(clip_x, clip_y, clip_w, ndc_uv);
    let barycentrics_ddx = homogeneous_barycentrics(
        clip_x, clip_y, clip_w, ndc_uv + vec2(two_over_screen_size.x, 0.0),
    );
    let barycentrics_ddy = homogeneous_barycentrics(
        clip_x, clip_y, clip_w, ndc_uv - vec2(0.0, two_over_screen_size.y),
    );
    result.ddx = barycentrics_ddx - result.barycentrics;
    result.ddy = barycentrics_ddy - result.barycentrics;
    return result;
}

struct VertexOutput {
    position: vec4<f32>,
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
    uv: vec2<f32>,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
    world_tangent: vec4<f32>,
    // Screen-space winding facing, matching `@builtin(front_facing)`. The instance
    // determinant is *not* folded in here: consumers pass this through
    // `pbr_functions::winding_corrected_front_facing` to get logical facing.
    is_front: bool,
    mesh_flags: u32,
    cluster_id: u32,
    material_bind_group_slot: u32,
#ifdef PREPASS_FRAGMENT
#ifdef MOTION_VECTOR_PREPASS
    motion_vector: vec2<f32>,
#endif
#endif
}

/// Load the visibility buffer texture and resolve it into a VertexOutput.
fn resolve_vertex_output(frag_coord: vec4<f32>) -> VertexOutput {
    let packed_ids = u32(textureLoad(meshlet_visibility_buffer, vec2<u32>(frag_coord.xy)).r);
    let cluster_id = packed_ids >> 7u;
    let instanced_offset = meshlet_raster_clusters[cluster_id];
    let meshlet_id = instanced_offset.offset;
    let descriptor = meshlet_instance_descriptors[instanced_offset.instance_id];
    var meshlet = load_meshlet(descriptor, meshlet_id);

    let triangle_id = extractBits(packed_ids, 0u, 7u);
    let index_ids = meshlet.start_index_id + (triangle_id * 3u) + vec3(0u, 1u, 2u);
    let vertex_ids = vec3(
        get_meshlet_vertex_id(descriptor, index_ids[0]),
        get_meshlet_vertex_id(descriptor, index_ids[1]),
        get_meshlet_vertex_id(descriptor, index_ids[2]),
    );
    let vertex_0 = load_vertex(descriptor, &meshlet, vertex_ids[0]);
    let vertex_1 = load_vertex(descriptor, &meshlet, vertex_ids[1]);
    let vertex_2 = load_vertex(descriptor, &meshlet, vertex_ids[2]);

    let instance_id = instanced_offset.instance_id;
    var instance_uniform = meshlet_instance_uniforms[instance_id];

    let world_from_local = affine3_to_square(instance_uniform.world_from_local);
    let world_position_0 = mesh_position_local_to_world(world_from_local, vec4(vertex_0.position, 1.0));
    let world_position_1 = mesh_position_local_to_world(world_from_local, vec4(vertex_1.position, 1.0));
    let world_position_2 = mesh_position_local_to_world(world_from_local, vec4(vertex_2.position, 1.0));

    let frag_coord_ndc = frag_coord_to_ndc(frag_coord, view.main_pass_viewport).xy;
    let partial_derivatives = compute_partial_derivatives(
        array(world_position_0, world_position_1, world_position_2),
        frag_coord_ndc,
        2.0 / view.main_pass_viewport.zw,
    );

    let world_position = mat3x4(world_position_0, world_position_1, world_position_2) * partial_derivatives.barycentrics;
    let world_positions_camera_relative = mat3x3(
        world_position_0.xyz - view.world_position,
        world_position_1.xyz - view.world_position,
        world_position_2.xyz - view.world_position,
    );
    let ddx_world_position = world_positions_camera_relative * partial_derivatives.ddx;
    let ddy_world_position = world_positions_camera_relative * partial_derivatives.ddy;

    let interpolated_normal = mat3x3(
        normal_local_to_world(vertex_0.normal, &instance_uniform),
        normal_local_to_world(vertex_1.normal, &instance_uniform),
        normal_local_to_world(vertex_2.normal, &instance_uniform),
    ) * partial_derivatives.barycentrics;

    // Coarse LODs can collapse a plate into a single triangle carrying near-opposite vertex
    // normals, whose interpolation is zero-length. That divides by zero in
    // calculate_tbn_mikktspace and in the tangent-plane projection below, so fall back to the
    // triangle plane. The cross product is in index order, which the instance determinant flips.
    let geometric_normal = cross(
        world_position_1.xyz - world_position_0.xyz,
        world_position_2.xyz - world_position_0.xyz,
    ) * select(-1.0, 1.0, (instance_uniform.flags & MESH_FLAGS_SIGN_DETERMINANT_MODEL_3X3_BIT) != 0u);
    let world_normal = normalize(select(
        geometric_normal,
        interpolated_normal,
        dot(interpolated_normal, interpolated_normal) > 1e-12,
    ));

    let uv = mat3x2(vertex_0.uv, vertex_1.uv, vertex_2.uv) * partial_derivatives.barycentrics;
    let ddx_uv = mat3x2(vertex_0.uv, vertex_1.uv, vertex_2.uv) * partial_derivatives.ddx;
    let ddy_uv = mat3x2(vertex_0.uv, vertex_1.uv, vertex_2.uv) * partial_derivatives.ddy;

    let world_tangent = calculate_world_tangent(world_normal, ddx_world_position, ddy_world_position, ddx_uv, ddy_uv);

#ifdef PREPASS_FRAGMENT
#ifdef MOTION_VECTOR_PREPASS
    let previous_world_from_local = affine3_to_square(instance_uniform.previous_world_from_local);
    let previous_world_position_0 = mesh_position_local_to_world(previous_world_from_local, vec4(vertex_0.position, 1.0));
    let previous_world_position_1 = mesh_position_local_to_world(previous_world_from_local, vec4(vertex_1.position, 1.0));
    let previous_world_position_2 = mesh_position_local_to_world(previous_world_from_local, vec4(vertex_2.position, 1.0));
    let previous_world_position = mat3x4(previous_world_position_0, previous_world_position_1, previous_world_position_2) * partial_derivatives.barycentrics;
    let motion_vector = calculate_motion_vector(world_position, previous_world_position);
#endif
#endif

    return VertexOutput(
        frag_coord,
        world_position,
        world_normal,
        uv,
        ddx_uv,
        ddy_uv,
        world_tangent,
        partial_derivatives.ndc_double_area > 0.0,
        instance_uniform.flags,
        instance_id ^ meshlet_id,
        instance_uniform.material_and_lightmap_bind_group_slot & 0xffffu,
#ifdef PREPASS_FRAGMENT
#ifdef MOTION_VECTOR_PREPASS
        motion_vector,
#endif
#endif
    );
}

struct MeshletVertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
}

fn load_vertex(descriptor: MeshletGpuDescriptor, meshlet: ptr<function, Meshlet>, vertex_id: u32) -> MeshletVertex {
    return MeshletVertex(
        get_meshlet_vertex_position(descriptor, meshlet, vertex_id),
        get_meshlet_vertex_normal(descriptor, meshlet, vertex_id),
        get_meshlet_vertex_uv(descriptor, meshlet, vertex_id),
    );
}

fn normal_local_to_world(vertex_normal: vec3<f32>, instance_uniform: ptr<function, Mesh>) -> vec3<f32> {
    if any(vertex_normal != vec3<f32>(0.0)) {
        return normalize(
            mat2x4_f32_to_mat3x3_unpack(
                (*instance_uniform).local_from_world_transpose_a,
                (*instance_uniform).local_from_world_transpose_b,
            ) * vertex_normal
        );
    } else {
        return vertex_normal;
    }
}

// https://www.jeremyong.com/graphics/2023/12/16/surface-gradient-bump-mapping/#surface-gradient-from-a-tangent-space-normal-vector-without-an-explicit-tangent-basis
fn calculate_world_tangent(
    world_normal: vec3<f32>,
    ddx_world_position: vec3<f32>,
    ddy_world_position: vec3<f32>,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
) -> vec4<f32> {
    // Project the position gradients onto the tangent plane
    let ddx_world_position_s = ddx_world_position - dot(ddx_world_position, world_normal) * world_normal;
    let ddy_world_position_s = ddy_world_position - dot(ddy_world_position, world_normal) * world_normal;

    // Compute the jacobian matrix to leverage the chain rule
    let jacobian_sign = sign(ddx_uv.x * ddy_uv.y - ddx_uv.y * ddy_uv.x);

    var world_tangent = jacobian_sign * (ddy_uv.y * ddx_world_position_s - ddx_uv.y * ddy_world_position_s);

    // The sign intrinsic returns 0 if the argument is 0, and collinear projected position
    // gradients (edge-on slivers) zero the tangent itself
    if jacobian_sign != 0.0 && dot(world_tangent, world_tangent) > 1e-20 {
        world_tangent = normalize(world_tangent);
    }

    // The second factor here ensures a consistent handedness between
    // the tangent frame and surface basis w.r.t. screenspace.
    let w = jacobian_sign * sign(dot(ddy_world_position, cross(world_normal, ddx_world_position)));

    return vec4(world_tangent, -w); // TODO: Unclear why we need to negate this to match mikktspace generated tangents
}
#endif
