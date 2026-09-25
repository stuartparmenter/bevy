//! Vertex/fragment shader for rasterizing large clusters into a visibility buffer.

import package::{
    meshlet::bindings::{
        meshlet_instance_descriptors,
        load_meshlet_geometry,
        meshlet_instance_uniforms,
        meshlet_raster_clusters,
        meshlet_previous_raster_counts,
        meshlet_visibility_buffer,
        view,
        get_meshlet_triangle_count,
        get_meshlet_vertex_id,
        get_meshlet_vertex_position,
    },
    render::{
        mesh_functions::mesh_position_local_to_world,
        mesh_types::{
            MESH_FLAGS_CULL_BACK_FACES_BIT,
            MESH_FLAGS_CULL_FRONT_FACES_BIT,
            MESH_FLAGS_SIGN_DETERMINANT_MODEL_3X3_BIT,
        },
    },
};
import bevy_render::maths::affine3_to_square;

enable wgpu_binding_array;

var<immediate> meshlet_raster_cluster_rightmost_slot: u32;

// Bits of `VertexOutput::culled_faces`: the screen-space faces the fragment stage discards.
const CULL_SCREEN_FRONT_FACES: u32 = 1u;
const CULL_SCREEN_BACK_FACES: u32 = 2u;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @if(MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT)
    @location(0) @interpolate(flat) packed_ids: u32,
    @location(1) @interpolate(flat) culled_faces: u32,
}

@vertex
fn vertex(@builtin(instance_index) instance_index: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let cluster_in_draw = meshlet_previous_raster_counts[1] + instance_index;
    let cluster_id = meshlet_raster_cluster_rightmost_slot - cluster_in_draw;
    let instanced_offset = meshlet_raster_clusters[cluster_id];
    let descriptor = meshlet_instance_descriptors[instanced_offset.instance_id];
    var meshlet = load_meshlet_geometry(descriptor, instanced_offset.offset);

    let triangle_id = vertex_index / 3u;
    if triangle_id >= get_meshlet_triangle_count(&meshlet) { return dummy_vertex(); }
    let index_id = vertex_index;
    let vertex_id = get_meshlet_vertex_id(descriptor, meshlet.start_index_id + index_id);

    let instance_uniform = meshlet_instance_uniforms[instanced_offset.instance_id];

    let vertex_position = get_meshlet_vertex_position(descriptor, &meshlet, vertex_id);
    let world_from_local = affine3_to_square(instance_uniform.world_from_local);
    let world_position = mesh_position_local_to_world(world_from_local, vec4(vertex_position, 1.0));
    let clip_position = view.clip_from_world * vec4(world_position.xyz, 1.0);

    // The pipeline culls nothing, as it rasterizes every material; cull the instance's faces in
    // the fragment stage instead, matching the software rasterizer. `front_facing` there reports
    // screen-space winding, which a negative-determinant transform reverses, so swap the faces
    // for a mirrored instance.
    let mirrored = (instance_uniform.flags & MESH_FLAGS_SIGN_DETERMINANT_MODEL_3X3_BIT) == 0u;
    let cull_front = (instance_uniform.flags & MESH_FLAGS_CULL_FRONT_FACES_BIT) != 0u;
    let cull_back = (instance_uniform.flags & MESH_FLAGS_CULL_BACK_FACES_BIT) != 0u;
    let culled_faces = select(0u, CULL_SCREEN_FRONT_FACES, select(cull_front, cull_back, mirrored))
        | select(0u, CULL_SCREEN_BACK_FACES, select(cull_back, cull_front, mirrored));

    @if(MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT)
    return VertexOutput(
        clip_position,
        (cluster_id << 7u) | triangle_id,
        culled_faces,
    );
    @else
    return VertexOutput(
        clip_position,
        culled_faces,
    );
}

@fragment
fn fragment(vertex_output: VertexOutput, @builtin(front_facing) front_facing: bool) {
    let face = select(CULL_SCREEN_BACK_FACES, CULL_SCREEN_FRONT_FACES, front_facing);
    if (vertex_output.culled_faces & face) != 0u { return; }

    let depth = bitcast<u32>(vertex_output.position.z);
    @if(MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT)
    let visibility = (u64(depth) << 32u) | u64(vertex_output.packed_ids);
    @else
    let visibility = depth;
    textureAtomicMax(meshlet_visibility_buffer, vec2<u32>(vertex_output.position.xy), visibility);
}

fn dummy_vertex() -> VertexOutput {
    @if(MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT)
    return VertexOutput(
        vec4(divide(0.0, 0.0)), // NaN vertex position
        0u,
        0u,
    );
    @else
    return VertexOutput(
        vec4(divide(0.0, 0.0)),
        0u,
    );
}

/// Naga doesn't allow divide by zero literals, but this lets us work around it
fn divide(a: f32, b: f32) -> f32 {
    return a / b;
}
