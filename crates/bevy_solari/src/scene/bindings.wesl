import bevy_pbr::render::pbr_functions::calculate_tbn_mikktspace;
import bevy_render::maths::affine3_to_square;

enable wgpu_ray_query;
enable wgpu_binding_array;

struct InstanceGeometryIds {
    vertex_buffer_id: u32,
    vertex_buffer_offset: u32,
    previous_vertex_buffer_id: u32,
    index_buffer_id: u32,
    index_buffer_offset: u32,
    triangle_count: u32,
    ray_mask: u32,
    diagnostic_primitives_per_group: u32,
}

struct VertexBuffer { vertices: array<PackedVertex> }

struct IndexBuffer { indices: array<u32> }

struct PackedVertex {
    a: vec4<f32>,
    b: vec4<f32>,
    tangent: vec4<f32>,
}

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    tangent: vec4<f32>,
}

fn unpack_vertex(packed: PackedVertex) -> Vertex {
    var vertex: Vertex;
    vertex.position = packed.a.xyz;
    vertex.normal = vec3(packed.a.w, packed.b.xy);
    vertex.uv = packed.b.zw;
    vertex.tangent = packed.tangent;
    return vertex;
}

struct Material {
    normal_map_texture_id: u32,
    base_color_texture_id: u32,
    emissive_texture_id: u32,
    metallic_roughness_texture_id: u32,

    base_color: vec3<f32>,
    perceptual_roughness: f32,
    emissive: vec3<f32>,
    metallic: f32,
    _padding: vec3<f32>,
    reflectance: f32,
}

const TEXTURE_MAP_NONE = 0xFFFFFFFFu;

const MIRROR_ROUGHNESS_THRESHOLD = 0.001f;

struct LightSource {
    kind: u32, // 1 bit for kind, 31 bits for extra data
    id: u32,
}

const LIGHT_SOURCE_KIND_EMISSIVE_MESH = 0u;
const LIGHT_SOURCE_KIND_DIRECTIONAL = 1u;

struct DirectionalLight {
    direction_to_light: vec3<f32>,
    cos_theta_max: f32,
    luminance: vec3<f32>,
    inverse_pdf: f32,
}

struct EnvironmentMapLight {
    light_from_world: mat3x3<f32>,
    intensity: f32,
}

const LIGHT_NOT_PRESENT_THIS_FRAME = 0xFFFFFFFFu;
const INSTANCE_NOT_PRESENT_THIS_FRAME = 0xFFFFFFFFu;

@group(0) @binding(0) var<storage> vertex_buffers: binding_array<VertexBuffer>;
@group(0) @binding(1) var<storage> index_buffers: binding_array<IndexBuffer>;
@group(0) @binding(2) var textures: binding_array<texture_2d<f32>>;
@group(0) @binding(3) var samplers: binding_array<sampler>;
@group(0) @binding(4) var<storage> materials: array<Material>;
@group(0) @binding(5) var tlas: acceleration_structure;
@group(0) @binding(6) var previous_frame_tlas: acceleration_structure;
@group(0) @binding(7) var<storage> transforms: array<mat3x4<f32>>;
@group(0) @binding(8) var<storage> previous_frame_transforms: array<mat3x4<f32>>;
@group(0) @binding(9) var<storage> geometry_ids: array<InstanceGeometryIds>;
@group(0) @binding(10) var<storage> material_ids: array<u32>;
@group(0) @binding(11) var<storage> light_sources: array<LightSource>;
@group(0) @binding(12) var<storage> directional_lights: array<DirectionalLight>;
@group(0) @binding(13) var<storage> previous_frame_light_id_translations: array<u32>;
@group(0) @binding(14) var brdf_dfg_lut: texture_2d<f32>;
@group(0) @binding(15) var brdf_dfg_lut_sampler: sampler;
@group(0) @binding(16) var environment_map: texture_cube<f32>;
@group(0) @binding(17) var environment_map_sampler: sampler;
@group(0) @binding(18) var<storage> environment_map_light: EnvironmentMapLight;
@group(0) @binding(19) var<storage> previous_frame_instance_id_translations: array<u32>;

const RAY_T_MIN = 0.001f;
const RAY_T_MAX = 100000.0f;

const RAY_NO_CULL = 0xFFu;

/// Grass primary surfaces carry (triangle normal, angular uncertainty).
/// Packed oct8x2 normals require a 0.02 guard; exact RT triangles use 1e-6.
/// Zero context preserves ordinary/secondary surfaces' existing origin policy.
/// `direction` must be normalized.
fn offset_surface_ray_origin(position: vec3<f32>, geometric_normal: vec4<f32>, direction: vec3<f32>) -> vec3<f32> {
    let n_dot_d = dot(geometric_normal.xyz, direction);
    if !(abs(n_dot_d) > geometric_normal.w) { return position; }
    return offset_triangle_ray_origin(position, geometric_normal.xyz, direction);
}

/// An unquantized triangle normal from the RT vertices does not need the oct8x2
/// uncertainty band. Retain only a small arithmetic guard near exact tangency:
/// cross/normalize/dot are still f32, not exact geometry predicates.
/// Used for verified RT receiver anchors as well as the surface reference.
fn offset_triangle_ray_origin(position: vec3<f32>, geometric_normal: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    let n_dot_d = dot(geometric_normal, direction);
    if !(abs(n_dot_d) > 0.000001) { return position; }
    let normal = select(-geometric_normal, geometric_normal, n_dot_d > 0.0);
    // A small, position-scaled allowance for f32 reconstruction/rounding error.
    // This is a heuristic, not an error bound for arbitrary depth transforms.
    // Four epsilons avoids the centimetre-scale offsets of 256-ULP schemes at
    // common world coordinates; the 10um floor handles positions near zero.
    let distance = max(0.00001, 4.0 * 1.1920928955078125e-7 * dot(abs(normal), abs(position)));
    return position + normal * distance;
}

fn trace_ray(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32, ray_flag: u32) -> RayIntersection {
    return trace_ray_masked(ray_origin, ray_direction, ray_t_min, ray_t_max, ray_flag, RAY_NO_CULL);
}

fn trace_ray_masked(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32, ray_flag: u32, mask: u32) -> RayIntersection {
    let ray = RayDesc(ray_flag, mask, ray_t_min, ray_t_max, ray_origin, ray_direction);
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, ray);
    rayQueryProceed(&rq);
    return rayQueryGetCommittedIntersection(&rq);
}

fn trace_ray_previous_frame(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32, ray_flag: u32) -> RayIntersection {
    let ray = RayDesc(ray_flag, RAY_NO_CULL, ray_t_min, ray_t_max, ray_origin, ray_direction);
    var rq: ray_query;
    rayQueryInitialize(&rq, previous_frame_tlas, ray);
    rayQueryProceed(&rq);
    return rayQueryGetCommittedIntersection(&rq);
}

fn sample_texture(id: u32, uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(textures[id], samplers[id], uv, 0.0).rgb; // TODO: Mipmap
}

struct ResolvedMaterial {
    base_color: vec3<f32>,
    emissive: vec3<f32>,
    reflectance: f32,
    perceptual_roughness: f32,
    roughness: f32,
    metallic: f32,
}

struct ResolvedRayHitFull {
    world_position: vec3<f32>,
    /// Includes deformation motion when the producer supplies previous vertices.
    previous_frame_world_position: vec3<f32>,
    world_normal: vec3<f32>,
    geometric_world_normal: vec3<f32>,
    world_tangent: vec4<f32>,
    uv: vec2<f32>,
    triangle_area: f32,
    triangle_count: u32,
    material: ResolvedMaterial,
}

fn resolve_material(material: Material, uv: vec2<f32>) -> ResolvedMaterial {
    var m: ResolvedMaterial;

    m.base_color = material.base_color.rgb;
    if material.base_color_texture_id != TEXTURE_MAP_NONE {
        m.base_color *= sample_texture(material.base_color_texture_id, uv);
    }

    m.emissive = material.emissive.rgb;
    if material.emissive_texture_id != TEXTURE_MAP_NONE {
        m.emissive *= sample_texture(material.emissive_texture_id, uv);
    }

    m.reflectance = material.reflectance;

    m.perceptual_roughness = material.perceptual_roughness;
    m.metallic = material.metallic;
    if material.metallic_roughness_texture_id != TEXTURE_MAP_NONE {
        let metallic_roughness = sample_texture(material.metallic_roughness_texture_id, uv);
        m.perceptual_roughness *= metallic_roughness.g;
        m.metallic *= metallic_roughness.b;
    }

    m.roughness = m.perceptual_roughness * m.perceptual_roughness;

    return m;
}

fn resolve_ray_hit_full(ray_hit: RayIntersection) -> ResolvedRayHitFull {
    let barycentrics = vec3(1.0 - ray_hit.barycentrics.x - ray_hit.barycentrics.y, ray_hit.barycentrics);
    return resolve_triangle_data_full(ray_hit.instance_custom_data, ray_hit.primitive_index, barycentrics);
}

fn load_vertices(instance_geometry_ids: InstanceGeometryIds, triangle_id: u32) -> array<Vertex, 3> {
    let index_buffer = &index_buffers[instance_geometry_ids.index_buffer_id].indices;
    let vertex_buffer = &vertex_buffers[instance_geometry_ids.vertex_buffer_id].vertices;

    let indices_i = (triangle_id * 3u) + vec3(0u, 1u, 2u) + instance_geometry_ids.index_buffer_offset;
    let indices = vec3((*index_buffer)[indices_i.x], (*index_buffer)[indices_i.y], (*index_buffer)[indices_i.z]) + instance_geometry_ids.vertex_buffer_offset;

    return array<Vertex, 3>(
        unpack_vertex((*vertex_buffer)[indices.x]),
        unpack_vertex((*vertex_buffer)[indices.y]),
        unpack_vertex((*vertex_buffer)[indices.z])
    );
}

fn transform_positions(transform: mat4x4<f32>, vertices: array<Vertex, 3>) -> array<vec3<f32>, 3> {
    return array<vec3<f32>, 3>(
        (transform * vec4(vertices[0].position, 1.0)).xyz,
        (transform * vec4(vertices[1].position, 1.0)).xyz,
        (transform * vec4(vertices[2].position, 1.0)).xyz
    );
}

struct ResolvedSamplePoint {
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
}

/// Resolves a geometry anchor using current transforms and vertices.
/// Returns the interpolated vertex normal without material or normal-map sampling.
fn resolve_sample_point(instance_id: u32, triangle_id: u32, barycentrics: vec3<f32>) -> ResolvedSamplePoint {
    let transform = affine3_to_square(transforms[instance_id]);
    let vertices = load_vertices(geometry_ids[instance_id], triangle_id);
    let world_vertices = transform_positions(transform, vertices);
    let world_position = mat3x3(world_vertices[0], world_vertices[1], world_vertices[2]) * barycentrics;
    let local_normal = mat3x3(vertices[0].normal, vertices[1].normal, vertices[2].normal) * barycentrics;
    let transform_3x3 = mat3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz);
    let cofactor = mat3x3(cross(transform_3x3[1], transform_3x3[2]), cross(transform_3x3[2], transform_3x3[0]), cross(transform_3x3[0], transform_3x3[1]));
    let determinant = dot(transform_3x3[0], cofactor[0]);
    let determinant_sign = select(sign(determinant), 1.0, determinant == 0.0);
    let world_normal = normalize(determinant_sign * (cofactor * local_normal));
    return ResolvedSamplePoint(world_position, world_normal);
}

fn resolve_triangle_data_full(instance_id: u32, triangle_id: u32, barycentrics: vec3<f32>) -> ResolvedRayHitFull {
    let material_id = material_ids[instance_id];
    let material = materials[material_id];

    let transform = affine3_to_square(transforms[instance_id]);
    let previous_frame_transform = affine3_to_square(previous_frame_transforms[instance_id]);

    let instance_geometry_ids = geometry_ids[instance_id];
    let vertices = load_vertices(instance_geometry_ids, triangle_id);

    let world_vertices = transform_positions(transform, vertices);
    let world_position = mat3x3(world_vertices[0], world_vertices[1], world_vertices[2]) * barycentrics;

    var previous_vertices = vertices;
    if instance_geometry_ids.previous_vertex_buffer_id != instance_geometry_ids.vertex_buffer_id {
        var previous_geometry_ids = instance_geometry_ids;
        previous_geometry_ids.vertex_buffer_id = instance_geometry_ids.previous_vertex_buffer_id;
        previous_vertices = load_vertices(previous_geometry_ids, triangle_id);
    }
    let previous_frame_world_vertices = transform_positions(previous_frame_transform, previous_vertices);
    let previous_frame_world_position = mat3x3(previous_frame_world_vertices[0], previous_frame_world_vertices[1], previous_frame_world_vertices[2]) * barycentrics;

    let uv = mat3x2(vertices[0].uv, vertices[1].uv, vertices[2].uv) * barycentrics;

    let transform_3x3 = mat3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz);

    let cofactor = mat3x3(cross(transform_3x3[1], transform_3x3[2]), cross(transform_3x3[2], transform_3x3[0]), cross(transform_3x3[0], transform_3x3[1]));
    let determinant = dot(transform_3x3[0], cofactor[0]);
    let determinant_sign = select(sign(determinant), 1.0, determinant == 0.0);

    let local_tangent = mat3x3(vertices[0].tangent.xyz, vertices[1].tangent.xyz, vertices[2].tangent.xyz) * barycentrics;
    let world_tangent = vec4(normalize(transform_3x3 * local_tangent), vertices[0].tangent.w * determinant_sign);

    let local_normal = mat3x3(vertices[0].normal, vertices[1].normal, vertices[2].normal) * barycentrics; // TODO: Use barycentric lerp, ray_hit.object_to_world, cross product geo normal
    var world_normal = normalize(determinant_sign * (cofactor * local_normal));

    let geometric_world_normal = world_normal;
    if material.normal_map_texture_id != TEXTURE_MAP_NONE {
        let TBN = calculate_tbn_mikktspace(world_normal, world_tangent);
        let T = TBN[0];
        let B = TBN[1];
        let N = TBN[2];
        var Nt = sample_texture(material.normal_map_texture_id, uv) * 2.0 - 1.0;
        Nt.z = sqrt(max(1.0 - dot(Nt.xy, Nt.xy), 0.0)); // Reconstruct Z to support two-channel normal maps
        world_normal = normalize(Nt.x * T + Nt.y * B + Nt.z * N);
    }

    let triangle_edge0 = world_vertices[0] - world_vertices[1];
    let triangle_edge1 = world_vertices[0] - world_vertices[2];
    let triangle_area = length(cross(triangle_edge0, triangle_edge1)) / 2.0;

    let resolved_material = resolve_material(material, uv);

    return ResolvedRayHitFull(
        world_position,
        previous_frame_world_position,
        world_normal,
        geometric_world_normal,
        world_tangent,
        uv,
        triangle_area,
        instance_geometry_ids.triangle_count,
        resolved_material,
    );
}
