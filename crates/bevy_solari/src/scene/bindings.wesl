import bevy_pbr::render::pbr_functions::calculate_tbn_mikktspace;
import bevy_pbr::render::utils::{dir_to_cube_uv, rand_vec2f, sample_cube_dir, CubeUV};
import bevy_render::maths::{affine3_to_square, orthonormalize};
import bevy_render::utils::{decompress_vertex_position, decompress_vertex_normal, decompress_vertex_tangent, decompress_vertex_uv};
import bevy_render::mesh::metadata_types::{MeshMetadata, COMPRESS_POSITION, COMPRESS_NORMAL, COMPRESS_TANGENT, COMPRESS_UV0};
import package::scene::environment_map::{atlas_position_to_face, cube_face_jacobian, face_to_atlas_texel, ENVIRONMENT_ATLAS_EMPTY_FACE};

enable wgpu_ray_query;
enable wgpu_binding_array;

struct InstanceGeometryIds {
    vertex_buffer_id: u32,
    vertex_buffer_offset: u32,
    index_buffer_id: u32,
    index_buffer_offset: u32,
    triangle_count: u32,
    metadata_buffer_id: u32,
    metadata_offset: u32,
    vertex_stride_words: u32,
    world_geometry_error: f32,
    vertex_flags: u32,
}

/// The `metadata_buffer_id` of meshes without allocator metadata, whose attributes are all
/// uncompressed.
const NO_MESH_METADATA = 0xFFFFFFFFu;

const INDEX_BUFFER_FLAG_U16 = 1u << 31u;

/// Set in `InstanceGeometryIds::vertex_flags` when the vertices end in a tangent. Without one,
/// the tangent frame is reconstructed from the UVs.
const VERTEX_FLAG_HAS_TANGENT = 1u << 0u;

struct MeshMetadataBuffer { metadata: array<MeshMetadata> }

/// Interleaved position, normal, UV0 and optionally tangent, each either uncompressed or
/// compressed as `MeshMetadata::attribute_compression` describes.
struct VertexBuffer { words: array<u32> }

struct SceneParameters {
    max_world_geometry_error: f32,
    // Face size and mip count of the bound importance pyramid.
    environment_face_size: u32,
    environment_mip_count: u32,
}

struct IndexBuffer { indices: array<u32> }

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    tangent: vec4<f32>,
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
    reflectance: f32,
    flags: u32,
    uv_translation: vec2<f32>,
    uv_transform: mat2x2<f32>,
    alpha: f32,
    alpha_cutoff: f32,
    _padding: vec2<f32>,
}

const TEXTURE_MAP_NONE = 0xFFFFFFFFu;

const MATERIAL_FLAG_DOUBLE_SIDED = 1u << 0u;
const MATERIAL_FLAG_FLIP_NORMAL_MAP_Y = 1u << 1u;
// Mirrors StandardMaterialFlags::METALLIC_ROUGHNESS_RG: the metallic-roughness texture is a
// two-channel image packed R = roughness, G = metallic.
const MATERIAL_FLAG_METALLIC_ROUGHNESS_RG = 1u << 2u;

const MIRROR_ROUGHNESS_THRESHOLD = 0.001f;

struct LightSource {
    // LIGHT_SOURCE_KIND_BITS bits for kind, rest of the bits for extra data. For emissive meshes,
    // that is the first triangle in this logical light's at-most-65535-triangle chunk.
    kind: u32,
    id: u32,
}

const LIGHT_SOURCE_KIND_BITS = 3u;

const LIGHT_SOURCE_KIND_EMISSIVE_MESH = 0u;
const LIGHT_SOURCE_KIND_DIRECTIONAL = 1u;
const LIGHT_SOURCE_KIND_POINT = 2u;
const LIGHT_SOURCE_KIND_RECT = 3u;
// Resolves like a directional light, but its radiance is also what a missed BRDF ray picks up from
// sample_environment_map_light, so the two strategies have to be MIS-weighted against each other.
const LIGHT_SOURCE_KIND_ENVIRONMENT = 4u;

fn light_source_kind(light_source: LightSource) -> u32 {
    return light_source.kind & ((1u << LIGHT_SOURCE_KIND_BITS) - 1u);
}

struct DirectionalLight {
    direction_to_light: vec3<f32>,
    cos_theta_max: f32,
    luminance: vec3<f32>,
    inverse_pdf: f32,
}

/// A point or spot light. Point lights have no cone, e.g. `cos_outer <= -1`.
struct PointLight {
    position: vec3<f32>,
    radius: f32,
    intensity: vec3<f32>,
    cos_inner: f32,
    axis: vec3<f32>,
    cos_outer: f32,
}

/// A one-sided rectangular light facing `cross(edge_y, edge_x)`.
struct RectLight {
    center: vec3<f32>,
    area: f32,
    edge_x: vec3<f32>,
    _padding0: f32,
    edge_y: vec3<f32>,
    _padding1: f32,
    radiance: vec3<f32>,
    _padding2: f32,
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
@group(0) @binding(11) var brdf_dfg_lut: texture_2d<f32>;
@group(0) @binding(12) var brdf_dfg_lut_sampler: sampler;
@group(0) @binding(13) var<storage> light_sources: array<LightSource>;
@group(0) @binding(14) var<storage> directional_lights: array<DirectionalLight>;
@group(0) @binding(15) var<storage> point_lights: array<PointLight>;
@group(0) @binding(16) var<storage> rect_lights: array<RectLight>;
@group(0) @binding(17) var environment_map: texture_cube<f32>;
@group(0) @binding(18) var environment_map_sampler: sampler;
@group(0) @binding(19) var<storage> environment_map_light: EnvironmentMapLight;
@group(0) @binding(20) var<storage> previous_frame_light_id_translations: array<u32>;
@group(0) @binding(21) var<storage> mesh_metadata_buffers: binding_array<MeshMetadataBuffer>;
@group(0) @binding(22) var<storage, read> scene_parameters: SceneParameters;
@group(0) @binding(23) var environment_importance_map: texture_2d<f32>;

const RAY_T_MIN = 0.001f;
const RAY_T_MAX = 100000.0f;

const RAY_NO_CULL = 0xFFu;

// A rasterized G-buffer surface is offset along the NORMAL-MAPPED shading normal, because that is
// the only normal the G-buffer stores. An offset of `b` only buys `b * cos(theta)` of clearance from
// the geometric surface, so scale the geometry-error term up to cover a normal map that tilts the
// shading normal by up to ~48 degrees. Ray-hit surfaces carry a true geometric normal and need no
// such margin, which also keeps one instance's primary bias at or above its secondary bias.
const RAY_ORIGIN_BIAS_SHADING_NORMAL_SAFETY = 1.5f;

// The bias is per instance because a ray leaving a surface can only self-intersect the simplified
// BLAS of the instance it left: charging every ray the scene's largest geometry error starts small
// props' shadow rays above the prop itself. `world_geometry_error` comes from the G-buffer, and is
// negative when the texel was written by something that states none - a custom deferred material,
// the prepass fallback - in which case fall back to the scene-wide maximum, which bounds every
// instance that states one. That fallback is a rasterized surface like any other and takes the same
// safety factor: the instance setting the maximum is biased by the factor on its own texels, and it
// is a likely occupant of the fallback, since an error past the quantizer's top step also codes as
// unknown. A scene that states no error anywhere has a zero maximum, where the factor is a no-op.
fn rasterized_surface_ray_origin_bias(world_geometry_error: f32) -> f32 {
    return RAY_T_MIN
        + RAY_ORIGIN_BIAS_SHADING_NORMAL_SAFETY * bounded_world_geometry_error(world_geometry_error);
}

// A negative error states none at all, in which case the only bound available is the scene-wide
// maximum over the instances that do state one.
fn bounded_world_geometry_error(world_geometry_error: f32) -> f32 {
    return select(
        world_geometry_error,
        max(scene_parameters.max_world_geometry_error, 0.0),
        world_geometry_error < 0.0,
    );
}

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
    if light_count == 0u || light_source_kind(light_sources[light_count - 1u]) != LIGHT_SOURCE_KIND_ENVIRONMENT {
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

// A ray hit lies on the traced surface and carries a true geometric normal, so unlike a rasterized
// surface it needs no shading-normal margin.
fn ray_origin_bias_for_instance(instance_id: u32) -> f32 {
    return RAY_T_MIN
        + bounded_world_geometry_error(geometry_ids[instance_id].world_geometry_error);
}

// The visibility buffer rasterizes whichever meshlet LOD keeps its simplification error under about
// one pixel, so a shading point read back from the G-buffer sits up to a pixel's worth of world
// space away from the fixed-LOD surface the BLAS holds - a gap that grows linearly with camera
// distance. Past a few tens of meters it exceeds the BLAS build error that
// `ray_origin_bias_for_instance` covers, the shading point ends up inside the traced proxy, and
// every visibility and GI ray from it is instantly self-occluded. Widen the bias by the same bound
// the LOD selector enforces (bevy_pbr::meshlet_cull_shared::lod_error_is_imperceptible).
const RAY_ORIGIN_BIAS_RASTER_LOD_SCALE = 3.0f;
// Capped rather than left to grow: a bias thicker than the thinnest occluder Solari must respect
// leaks light through it, and this scene's walls and floors are on the order of a meter. Reached at
// roughly 150 m at 1080p, by which point a pixel already covers 17 cm.
const RAY_ORIGIN_BIAS_RASTER_LOD_MAX = 0.5f;

// `pixel_world_size` is the world-space size of one pixel at the shading point, passed in because
// this module is bound with the scene and cannot see any pass's view uniform. The comparison rejects
// NaN, which reaches here whenever a degenerate G-buffer texel produces a garbage world position.
fn ray_origin_bias_with_raster_lod(base_bias: f32, pixel_world_size: f32) -> f32 {
    let lod_error = RAY_ORIGIN_BIAS_RASTER_LOD_SCALE * pixel_world_size;
    return base_bias + select(0.0, min(lod_error, RAY_ORIGIN_BIAS_RASTER_LOD_MAX), lod_error > 0.0);
}

fn offset_ray_origin(
    world_position: vec3<f32>,
    geometric_world_normal: vec3<f32>,
    ray_direction: vec3<f32>,
    bias: f32,
) -> vec3<f32> {
    let side = select(-1.0, 1.0, dot(geometric_world_normal, ray_direction) >= 0.0);
    return world_position + geometric_world_normal * side * max(bias, RAY_T_MIN);
}

fn trace_ray(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32, ray_flag: u32) -> RayIntersection {
    let ray = RayDesc(ray_flag, RAY_NO_CULL, ray_t_min, ray_t_max, ray_origin, ray_direction);
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, ray);
    while rayQueryProceed(&rq) {
        let candidate = rayQueryGetCandidateIntersection(&rq);
        if passes_alpha_test(candidate) {
            rayQueryConfirmIntersection(&rq);
        }
    }
    return rayQueryGetCommittedIntersection(&rq);
}

fn trace_ray_previous_frame(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32, ray_flag: u32) -> RayIntersection {
    let ray = RayDesc(ray_flag, RAY_NO_CULL, ray_t_min, ray_t_max, ray_origin, ray_direction);
    var rq: ray_query;
    rayQueryInitialize(&rq, previous_frame_tlas, ray);
    while rayQueryProceed(&rq) {
        let candidate = rayQueryGetCandidateIntersection(&rq);
        if passes_alpha_test(candidate) {
            rayQueryConfirmIntersection(&rq);
        }
    }
    return rayQueryGetCommittedIntersection(&rq);
}

fn passes_alpha_test(candidate: RayIntersection) -> bool {
    let barycentrics = vec3(1.0 - candidate.barycentrics.x - candidate.barycentrics.y, candidate.barycentrics);
    return !resolve_triangle_data_full(candidate.instance_custom_data, candidate.primitive_index, barycentrics).material.alpha_masked_out;
}

fn sample_texture(id: u32, uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(textures[id], samplers[id], uv, 0.0); // TODO: Mipmap
}

struct ResolvedMaterial {
    base_color: vec3<f32>,
    emissive: vec3<f32>,
    reflectance: f32,
    perceptual_roughness: f32,
    roughness: f32,
    metallic: f32,
    alpha_masked_out: bool,
}

struct ResolvedRayHitFull {
    world_position: vec3<f32>,
    previous_frame_world_position: vec3<f32>,
    world_normal: vec3<f32>,
    geometric_world_normal: vec3<f32>,
    world_tangent: vec4<f32>,
    uv: vec2<f32>,
    triangle_area: f32,
    triangle_id: u32,
    triangle_count: u32,
    ray_origin_bias: f32,
    material: ResolvedMaterial,
    double_sided: bool,
}

fn resolve_material(material: Material, uv: vec2<f32>) -> ResolvedMaterial {
    var m: ResolvedMaterial;

    m.base_color = material.base_color.rgb;
    var alpha = material.alpha;
    if material.base_color_texture_id != TEXTURE_MAP_NONE {
        let base_color = sample_texture(material.base_color_texture_id, uv);
        m.base_color *= base_color.rgb;
        alpha *= base_color.a;
    }

    m.emissive = material.emissive.rgb;
    if material.emissive_texture_id != TEXTURE_MAP_NONE {
        m.emissive *= sample_texture(material.emissive_texture_id, uv).rgb;
    }
    m.alpha_masked_out = alpha < material.alpha_cutoff;
    if m.alpha_masked_out {
        m.emissive = vec3(0.0);
    }

    m.reflectance = material.reflectance;

    m.perceptual_roughness = material.perceptual_roughness;
    m.metallic = material.metallic;
    if material.metallic_roughness_texture_id != TEXTURE_MAP_NONE {
        let metallic_roughness = sample_texture(material.metallic_roughness_texture_id, uv);
        if (material.flags & MATERIAL_FLAG_METALLIC_ROUGHNESS_RG) != 0u {
            m.perceptual_roughness *= metallic_roughness.r;
            m.metallic *= metallic_roughness.g;
        } else {
            m.perceptual_roughness *= metallic_roughness.g;
            m.metallic *= metallic_roughness.b;
        }
    }

    m.roughness = m.perceptual_roughness * m.perceptual_roughness;

    return m;
}

fn resolve_ray_hit_full(ray_hit: RayIntersection) -> ResolvedRayHitFull {
    let barycentrics = vec3(1.0 - ray_hit.barycentrics.x - ray_hit.barycentrics.y, ray_hit.barycentrics);
    var resolved = resolve_triangle_data_full(ray_hit.instance_custom_data, ray_hit.primitive_index, barycentrics);

    if !ray_hit.front_face {
        // Flip normals for backfacing hits of double sided materials
        if resolved.double_sided {
            resolved.world_normal = -resolved.world_normal;
            resolved.geometric_world_normal = -resolved.geometric_world_normal;
        }

        // Emissive surfaces only emit from their front side, matching NEE, where cos_theta_light
        // culls the back side
        resolved.material.emissive = vec3(0.0);
    }

    return resolved;
}

fn load_word(vertex_buffer_id: u32, i: u32) -> u32 {
    return vertex_buffers[vertex_buffer_id].words[i];
}

fn load_f32x2(vertex_buffer_id: u32, i: u32) -> vec2<f32> {
    return bitcast<vec2<f32>>(vec2(load_word(vertex_buffer_id, i), load_word(vertex_buffer_id, i + 1u)));
}

fn load_f32x3(vertex_buffer_id: u32, i: u32) -> vec3<f32> {
    return vec3(load_f32x2(vertex_buffer_id, i), bitcast<f32>(load_word(vertex_buffer_id, i + 2u)));
}

fn load_f32x4(vertex_buffer_id: u32, i: u32) -> vec4<f32> {
    return vec4(load_f32x2(vertex_buffer_id, i), load_f32x2(vertex_buffer_id, i + 2u));
}

// `stride` is the vertex size in words. Without a tangent, the returned one is zero.
fn load_vertex(vertex_buffer_id: u32, metadata: MeshMetadata, vertex_index: u32, stride: u32, has_tangent: bool) -> Vertex {
    let c = metadata.attribute_compression;
    let position_words = select(3u, 2u, (c & COMPRESS_POSITION) != 0u);
    let normal_words = select(3u, 1u, (c & COMPRESS_NORMAL) != 0u);
    let uv_words = select(2u, 1u, (c & COMPRESS_UV0) != 0u);

    var i = vertex_index * stride;
    var vertex: Vertex;

    if (c & COMPRESS_POSITION) != 0u {
        let compressed = vec4(unpack2x16snorm(load_word(vertex_buffer_id, i)), unpack2x16snorm(load_word(vertex_buffer_id, i + 1u)));
        vertex.position = decompress_vertex_position(compressed, metadata.aabb_center, metadata.aabb_half_extents);
    } else {
        vertex.position = load_f32x3(vertex_buffer_id, i);
    }
    i += position_words;

    if (c & COMPRESS_NORMAL) != 0u {
        vertex.normal = decompress_vertex_normal(unpack2x16snorm(load_word(vertex_buffer_id, i)));
    } else {
        vertex.normal = load_f32x3(vertex_buffer_id, i);
    }
    i += normal_words;

    if (c & COMPRESS_UV0) != 0u {
        vertex.uv = decompress_vertex_uv(unpack2x16unorm(load_word(vertex_buffer_id, i)), metadata.uv_channels_min_and_extents[0]);
    } else {
        vertex.uv = load_f32x2(vertex_buffer_id, i);
    }
    i += uv_words;

    vertex.tangent = vec4(0.0);
    if has_tangent {
        if (c & COMPRESS_TANGENT) != 0u {
            vertex.tangent = decompress_vertex_tangent(unpack2x16snorm(load_word(vertex_buffer_id, i)));
        } else {
            vertex.tangent = load_f32x4(vertex_buffer_id, i);
        }
    }

    return vertex;
}

fn load_vertices(instance_geometry_ids: InstanceGeometryIds, triangle_id: u32) -> array<Vertex, 3> {
    let index_buffer = &index_buffers[instance_geometry_ids.index_buffer_id].indices;

    let index_offset = instance_geometry_ids.index_buffer_offset;
    let indices_i = (triangle_id * 3u) + vec3(0u, 1u, 2u) + (index_offset & ~INDEX_BUFFER_FLAG_U16);
    var indices: vec3<u32>;
    if (index_offset & INDEX_BUFFER_FLAG_U16) != 0u {
        let words = vec3((*index_buffer)[indices_i.x >> 1u], (*index_buffer)[indices_i.y >> 1u], (*index_buffer)[indices_i.z >> 1u]);
        indices = (words >> ((indices_i & vec3(1u)) * 16u)) & vec3(0xFFFFu);
    } else {
        indices = vec3((*index_buffer)[indices_i.x], (*index_buffer)[indices_i.y], (*index_buffer)[indices_i.z]);
    }
    indices += instance_geometry_ids.vertex_buffer_offset;

    var metadata: MeshMetadata;
    if instance_geometry_ids.metadata_buffer_id != NO_MESH_METADATA {
        metadata = mesh_metadata_buffers[instance_geometry_ids.metadata_buffer_id].metadata[instance_geometry_ids.metadata_offset];
    }

    let vertex_buffer_id = instance_geometry_ids.vertex_buffer_id;
    let stride = instance_geometry_ids.vertex_stride_words;
    let has_tangent = (instance_geometry_ids.vertex_flags & VERTEX_FLAG_HAS_TANGENT) != 0u;
    return array<Vertex, 3>(
        load_vertex(vertex_buffer_id, metadata, indices.x, stride, has_tangent),
        load_vertex(vertex_buffer_id, metadata, indices.y, stride, has_tangent),
        load_vertex(vertex_buffer_id, metadata, indices.z, stride, has_tangent)
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

    let mesh_uv = mat3x2(vertices[0].uv, vertices[1].uv, vertices[2].uv) * barycentrics;
    let uv = material.uv_transform * mesh_uv + material.uv_translation;

    let local_normal = mat3x3(vertices[0].normal, vertices[1].normal, vertices[2].normal) * barycentrics;
    var local_tangent = mat3x3(vertices[0].tangent.xyz, vertices[1].tangent.xyz, vertices[2].tangent.xyz) * barycentrics;
    var tangent_sign = vertices[0].tangent.w;
    if (instance_geometry_ids.vertex_flags & VERTEX_FLAG_HAS_TANGENT) == 0u {
        let edge1 = vertices[1].position - vertices[0].position;
        let edge2 = vertices[2].position - vertices[0].position;
        let uv_edge1 = vertices[1].uv - vertices[0].uv;
        let uv_edge2 = vertices[2].uv - vertices[0].uv;
        let determinant = uv_edge1.x * uv_edge2.y - uv_edge1.y * uv_edge2.x;
        // The determinant is in squared UV units, so the degeneracy test has to be relative to the
        // chart's own scale. An absolute epsilon rejects valid dense-mesh triangles whose UV edges
        // are only a fraction of a texel apart.
        let uv_edge_scale_squared = max(dot(uv_edge1, uv_edge1), dot(uv_edge2, uv_edge2));
        if abs(determinant) > 1e-6 * uv_edge_scale_squared {
            local_tangent = normalize((edge1 * uv_edge2.y - edge2 * uv_edge1.y) / determinant);
            let local_bitangent = normalize((edge2 * uv_edge1.x - edge1 * uv_edge2.x) / determinant);
            // Only the sign of the triple product matters and it survives a positive scale, so use
            // the raw interpolation, which cancelling vertex normals can zero out but not misdirect.
            tangent_sign = select(-1.0, 1.0, dot(cross(local_normal, local_tangent), local_bitangent) >= 0.0);
        } else {
            // The tangent guard below builds a fallback frame from the normalized world normal.
            // Picking an axis against a raw vertex normal instead needs that normal to be unit,
            // which an arbitrary mesh's NORMAL attribute does not promise.
            local_tangent = vec3(0.0);
            tangent_sign = 1.0;
        }
    }
    let linear_transform = mat3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz);
    let linear_determinant = determinant(linear_transform);
    var normal_transform = linear_transform;
    if abs(linear_determinant) > 1e-12 {
        // The cofactor matrix is inverse-transpose(M) * determinant(M).
        // Scale its vector columns individually because WGSL/Naga does not
        // define matrix-by-scalar division.
        let inverse_determinant = 1.0 / linear_determinant;
        normal_transform = mat3x3(
            cross(linear_transform[1], linear_transform[2]) * inverse_determinant,
            cross(linear_transform[2], linear_transform[0]) * inverse_determinant,
            cross(linear_transform[0], linear_transform[1]) * inverse_determinant,
        );
    }
    let transform_handedness = select(-1.0, 1.0, linear_determinant >= 0.0);

    let interpolated_world_normal = normal_transform * local_normal;
    // A coarse LOD can collapse a plate into a single triangle carrying near-opposite vertex
    // normals, and a mesh can supply zero normals outright; either way the interpolation is
    // zero-length and normalizing it is NaN. Test the local normal, a convex combination of unit
    // normals, so the threshold carries no instance scale - the same test and threshold
    // bevy_pbr::meshlet_visibility_buffer_resolve uses.
    let interpolated_normal_is_usable = dot(local_normal, local_normal) > 1e-12;

    let triangle_cross = cross(
        world_vertices[1] - world_vertices[0],
        world_vertices[2] - world_vertices[0],
    );
    let triangle_cross_length_squared = dot(triangle_cross, triangle_cross);

    // Only a triangle with no direction at all and no usable vertex normals reaches the axis; it is
    // here so that no NaN can leave this function.
    var world_normal = vec3<f32>(0.0, 1.0, 0.0);
    if interpolated_normal_is_usable {
        world_normal = normalize(interpolated_world_normal);
    }
    var geometric_world_normal = world_normal;
    if triangle_cross_length_squared > 1e-20 {
        // cross(M*e1, M*e2) carries a factor of det(M), which points the winding normal inward on
        // mirrored (negative-determinant) instances while the cofactor-transformed shading normal
        // still points outward. Undo that sign so both agree, and so the geometric normal keeps the
        // same sidedness as RayIntersection::front_face, which the driver derives in object space.
        geometric_world_normal = triangle_cross * (inverseSqrt(triangle_cross_length_squared) * transform_handedness);
        if !interpolated_normal_is_usable {
            world_normal = geometric_world_normal;
        }
    }

    var transformed_tangent = linear_transform * local_tangent;
    transformed_tangent -= world_normal * dot(world_normal, transformed_tangent);
    if dot(transformed_tangent, transformed_tangent) <= 1e-12 {
        transformed_tangent = orthonormalize(world_normal)[0];
    }
    let world_tangent = vec4(normalize(transformed_tangent), tangent_sign * transform_handedness);

    if material.normal_map_texture_id != TEXTURE_MAP_NONE {
        let TBN = calculate_tbn_mikktspace(world_normal, world_tangent);
        let T = TBN[0];
        let B = TBN[1];
        let N = TBN[2];
        var Nt = sample_texture(material.normal_map_texture_id, uv).rgb * 2.0 - 1.0;
        if (material.flags & MATERIAL_FLAG_FLIP_NORMAL_MAP_Y) != 0u {
            Nt.y = -Nt.y; // Normal maps authored for DirectX, matching apply_normal_mapping
        }
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
        triangle_id,
        instance_geometry_ids.triangle_count,
        ray_origin_bias_for_instance(instance_id),
        resolved_material,
        (material.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u,
    );
}
