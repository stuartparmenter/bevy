import package::scene::bindings::PackedVertex;

struct DeformationParams {
    entity_from_world: mat4x4<f32>,
    world_from_entity: mat4x4<f32>,
    vertex: vec4<u32>,
    attributes: vec4<u32>,
    skin: vec4<u32>,
    morph: vec4<u32>,
}

struct MorphAttributes {
    position: vec3<f32>,
    pad_a: f32,
    normal: vec3<f32>,
    pad_b: f32,
    tangent: vec3<f32>,
    pad_c: f32,
}

@group(0) @binding(0) var<uniform> params: DeformationParams;
@group(0) @binding(1) var<storage> source: array<u32>;
@group(0) @binding(2) var<storage> joints: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage> morph_weights: array<f32>;
@group(0) @binding(4) var<storage> morph_targets: array<MorphAttributes>;
@group(0) @binding(5) var<storage, read_write> vertices: array<PackedVertex>;
@group(0) @binding(6) var<storage, read_write> previous_vertices: array<PackedVertex>;

fn read_vec3(offset: u32) -> vec3<f32> {
    return bitcast<vec3<f32>>(vec3(source[offset], source[offset + 1u], source[offset + 2u]));
}

fn read_vec4(offset: u32) -> vec4<f32> {
    return vec4(read_vec3(offset), bitcast<f32>(source[offset + 3u]));
}

fn normalize_or_zero(value: vec3<f32>) -> vec3<f32> {
    if dot(value, value) > 0.0 {
        return normalize(value);
    }
    return vec3(0.0);
}

fn transform_normal(transform: mat3x3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let cofactor = mat3x3(cross(transform[1], transform[2]), cross(transform[2], transform[0]), cross(transform[0], transform[1]));
    return normalize_or_zero(sign(dot(transform[0], cofactor[0])) * cofactor * normal);
}

@compute @workgroup_size(64)
fn deform(@builtin(global_invocation_id) id: vec3<u32>) {
    let vertex = id.x;
    if vertex >= params.vertex.x {
        return;
    }
    let base = params.vertex.y + vertex * params.vertex.z;
    var position = read_vec3(base + params.attributes.x);
    var normal = read_vec3(base + params.attributes.y);
    var uv = vec2(0.0);
    if params.attributes.z != 0xFFFFFFFFu {
        let offset = base + params.attributes.z;
        uv = bitcast<vec2<f32>>(vec2(source[offset], source[offset + 1u]));
    }
    var tangent = vec4(0.0);
    if params.attributes.w != 0xFFFFFFFFu {
        tangent = read_vec4(base + params.attributes.w);
    }
    for (var target_index = 0u; target_index < params.morph.z; target_index++) {
        let weight = morph_weights[params.morph.y + target_index];
        if weight != 0.0 {
            let delta = morph_targets[params.morph.x + target_index * params.vertex.x + vertex];
            position += weight * delta.position;
            normal += weight * delta.normal;
            tangent = vec4(tangent.xyz + weight * delta.tangent, tangent.w);
        }
    }
    var world_from_local = params.world_from_entity;
    if params.vertex.w != 0xFFFFFFFFu {
        let weights = read_vec4(base + params.skin.x);
        let offset = base + params.skin.y;
        var indices: vec4<u32>;
        if params.skin.z == 0u {
            indices = vec4(source[offset] & 0xFFFFu, source[offset] >> 16u,
                source[offset + 1u] & 0xFFFFu, source[offset + 1u] >> 16u);
        } else {
            indices = vec4(source[offset], source[offset + 1u], source[offset + 2u], source[offset + 3u]);
        }
        indices += vec4(params.vertex.w);
        world_from_local = weights.x * joints[indices.x] + weights.y * joints[indices.y]
            + weights.z * joints[indices.z] + weights.w * joints[indices.w];
    }
    let world_matrix = mat3x3(world_from_local[0].xyz, world_from_local[1].xyz, world_from_local[2].xyz);
    let entity_matrix = mat3x3(params.world_from_entity[0].xyz, params.world_from_entity[1].xyz, params.world_from_entity[2].xyz);
    let inverse_entity_matrix = mat3x3(params.entity_from_world[0].xyz, params.entity_from_world[1].xyz, params.entity_from_world[2].xyz);
    let world_position = (world_from_local * vec4(position, 1.0)).xyz;
    let local_position = (params.entity_from_world * vec4(world_position, 1.0)).xyz;
    // Normalize in world space as the raster vertex shader does, then undo the instance transform.
    // Do not normalize again locally: that would change interpolation under nonuniform scale.
    let local_normal = transpose(entity_matrix) * transform_normal(world_matrix, normal);
    let local_tangent = inverse_entity_matrix * normalize_or_zero(world_matrix * tangent.xyz);
    var result: PackedVertex;
    result.a = vec4(local_position, local_normal.x);
    result.b = vec4(local_normal.yz, uv);
    // Hit shading applies the entity's handedness, matching the raster tangent convention.
    result.tangent = vec4(local_tangent, tangent.w);
    if params.skin.w != 0u {
        previous_vertices[vertex] = vertices[vertex];
    } else {
        previous_vertices[vertex] = result;
    }
    vertices[vertex] = result;
}
