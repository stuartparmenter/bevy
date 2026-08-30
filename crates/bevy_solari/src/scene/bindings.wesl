import bevy_pbr::render::pbr_functions::calculate_tbn_mikktspace;
import bevy_pbr::render::utils::{dir_to_cube_uv, rand_vec2f, sample_cube_dir, CubeUV};
import bevy_render::maths::affine3_to_square;
import package::scene::environment_map::{atlas_position_to_face, cube_face_jacobian, face_to_atlas_texel, ENVIRONMENT_ATLAS_EMPTY_FACE};

enable wgpu_ray_query;
enable wgpu_binding_array;

struct InstanceGeometryIds {
    vertex_buffer_id: u32,
    vertex_buffer_offset: u32,
    index_buffer_id: u32,
    index_buffer_offset: u32,
    triangle_count: u32,
}

struct SceneParameters {
    // Face size and mip count of the bound importance pyramid.
    environment_face_size: u32,
    environment_mip_count: u32,
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
// Resolves like a directional light, but its radiance is also what a missed BRDF ray picks up from
// sample_environment_map_light, so the two strategies have to be MIS-weighted against each other.
const LIGHT_SOURCE_KIND_ENVIRONMENT = 3u;

fn light_source_is_emissive_mesh(light_source: LightSource) -> bool {
    return (light_source.kind & 1u) == 0u;
}

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
@group(0) @binding(19) var<storage, read> scene_parameters: SceneParameters;
@group(0) @binding(20) var environment_importance_map: texture_2d<f32>;

const RAY_T_MIN = 0.001f;
const RAY_T_MAX = 100000.0f;

const RAY_NO_CULL = 0xFFu;

// The cube direction sample_environment_map_light looks `world_direction` up at: rotated into the
// light's frame, then Z flipped for the left-handed cubemap.
fn world_to_cube_direction(world_direction: vec3<f32>) -> vec3<f32> {
    let c = environment_map_light.light_from_world * world_direction;
    return vec3(c.x, c.y, -c.z);
}

// The inverse of world_to_cube_direction; light_from_world is a rotation, so its inverse is its
// transpose.
fn cube_to_world_direction(cube_direction: vec3<f32>) -> vec3<f32> {
    return transpose(environment_map_light.light_from_world) * vec3(cube_direction.x, cube_direction.y, -cube_direction.z);
}

fn environment_importance_root() -> f32 {
    return textureLoad(environment_importance_map, vec2(0u, 0u), scene_parameters.environment_mip_count - 1u).r;
}

// Density per steradian of the jittered sample inside the leaf texel at `coords`, given the leaf's
// weight and the root total. The Jacobian is evaluated at the sample's own uv, which is exactly the
// density the jitter has there.
fn environment_leaf_pdf(leaf: f32, root: f32, coords: CubeUV) -> f32 {
    return (leaf / root) * cube_face_jacobian(coords.uv, scene_parameters.environment_face_size);
}

// Density per steradian with which sample_environment_direction produces `direction`, for one draw
// of the environment light entry.
fn environment_direction_pdf(direction: vec3<f32>) -> f32 {
    let root = environment_importance_root();
    if !(root > 0.0) { return 0.0; }
    let coords = dir_to_cube_uv(world_to_cube_direction(direction));
    let texel = face_to_atlas_texel(coords, scene_parameters.environment_face_size);
    let leaf = textureLoad(environment_importance_map, texel, 0u).r;
    return environment_leaf_pdf(leaf, root, coords);
}

// 1 / light_count while the environment entry, which the binder always pushes last, is in
// light_sources; else 0.
fn environment_light_selection_probability() -> f32 {
    let light_count = arrayLength(&light_sources);
    if light_count == 0u || light_sources[light_count - 1u].kind != LIGHT_SOURCE_KIND_ENVIRONMENT {
        return 0.0;
    }
    return 1.0 / f32(light_count);
}

// Density of the whole environment NEE strategy: pick the environment entry, then a direction.
// Every MIS site (BRDF miss and NEE) uses this one function, so the two strategies always agree.
fn environment_light_pdf(direction: vec3<f32>) -> f32 {
    let selection = environment_light_selection_probability();
    if selection == 0.0 { return 0.0; }
    return environment_direction_pdf(direction) * selection;
}

struct EnvironmentDirectionSample {
    world_direction: vec3<f32>,
    // 0 when nothing could be sampled (all-zero pyramid).
    pdf: f32,
}

const ONE_MINUS_EPSILON = 0.99999994f;

// Hierarchical warp (RTXDI-style): two uniforms from `seed` pick a quadrant per level from the root
// down, rescaled after every choice, and what is left of them jitters inside the leaf texel.
// Deterministic per seed: ReSTIR re-resolves (light_id, seed) every frame.
fn sample_environment_direction(seed: u32) -> EnvironmentDirectionSample {
    var rng = seed;
    var u = min(rand_vec2f(&rng), vec2(ONE_MINUS_EPSILON));
    let mip_count = scene_parameters.environment_mip_count;
    var texel = vec2(0u, 0u);
    var leaf = 0.0;
    var root = 0.0;
    var level = mip_count - 1u; // root
    while level > 0u {
        level -= 1u;
        let base = texel * 2u;
        let w00 = textureLoad(environment_importance_map, base, level).r;
        let w10 = textureLoad(environment_importance_map, base + vec2(1u, 0u), level).r;
        var w01 = 0.0;
        var w11 = 0.0;
        // Every level is even-sized in both directions except the 2x1 one under the root, whose
        // second row does not exist.
        let under_root = level + 2u == mip_count;
        if !under_root {
            w01 = textureLoad(environment_importance_map, base + vec2(0u, 1u), level).r;
            w11 = textureLoad(environment_importance_map, base + vec2(1u, 1u), level).r;
        }
        let left = w00 + w01;
        let total = left + w10 + w11;
        if !(total > 0.0) { return EnvironmentDirectionSample(vec3(0.0, 1.0, 0.0), 0.0); }
        // Parents are sums of their children, so the first level's total is the root.
        if under_root { root = total; }
        let p_left = left / total;
        var column = vec2(w00, w01); // (top, bottom) of the chosen column
        var child = vec2(0u, 0u);
        if u.x < p_left {
            u.x = u.x / p_left;
        } else {
            u.x = (u.x - p_left) / (1.0 - p_left);
            child.x = 1u;
            column = vec2(w10, w11);
        }
        let p_top = column.x / (column.x + column.y);
        if u.y < p_top {
            u.y = u.y / p_top;
        } else {
            u.y = (u.y - p_top) / (1.0 - p_top);
            child.y = 1u;
        }
        u = clamp(u, vec2(0.0), vec2(ONE_MINUS_EPSILON));
        texel = base + child;
        leaf = select(column.x, column.y, child.y == 1u);
    }
    let coords = atlas_position_to_face(texel, u, scene_parameters.environment_face_size);
    if coords.face == ENVIRONMENT_ATLAS_EMPTY_FACE {
        return EnvironmentDirectionSample(vec3(0.0, 1.0, 0.0), 0.0);
    }
    let pdf = environment_leaf_pdf(leaf, root, coords);
    return EnvironmentDirectionSample(cube_to_world_direction(sample_cube_dir(coords.uv, coords.face)), pdf);
}

fn trace_ray(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32, ray_flag: u32) -> RayIntersection {
    let ray = RayDesc(ray_flag, RAY_NO_CULL, ray_t_min, ray_t_max, ray_origin, ray_direction);
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

fn resolve_triangle_data_full(instance_id: u32, triangle_id: u32, barycentrics: vec3<f32>) -> ResolvedRayHitFull {
    let material_id = material_ids[instance_id];
    let material = materials[material_id];

    let transform = affine3_to_square(transforms[instance_id]);
    let previous_frame_transform = affine3_to_square(previous_frame_transforms[instance_id]);

    let instance_geometry_ids = geometry_ids[instance_id];
    let vertices = load_vertices(instance_geometry_ids, triangle_id);

    let world_vertices = transform_positions(transform, vertices);
    let world_position = mat3x3(world_vertices[0], world_vertices[1], world_vertices[2]) * barycentrics;

    let previous_frame_world_vertices = transform_positions(previous_frame_transform, vertices);
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
