import bevy_pbr::render::pbr_functions::calculate_tbn_mikktspace;
import bevy_render::maths::{affine3_to_square, orthonormalize};

enable wgpu_ray_query;
enable wgpu_binding_array;

struct InstanceGeometryIds {
    vertex_buffer_id: u32,
    vertex_buffer_offset: u32,
    index_buffer_id: u32,
    index_buffer_offset: u32,
    triangle_count: u32,
    vertex_stride_words: u32,
    world_geometry_error: f32,
    _padding: f32,
}

struct SceneParameters {
    max_world_geometry_error: f32,
}

struct VertexBuffer { words: array<u32> }

struct IndexBuffer { indices: array<u32> }

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    tangent: vec4<f32>,
}

fn load_vertex(vertex_buffer_id: u32, index: u32, stride: u32) -> Vertex {
    let base = index * stride;
    var vertex: Vertex;
    vertex.position = vec3<f32>(
        bitcast<f32>(vertex_buffers[vertex_buffer_id].words[base]),
        bitcast<f32>(vertex_buffers[vertex_buffer_id].words[base + 1u]),
        bitcast<f32>(vertex_buffers[vertex_buffer_id].words[base + 2u]),
    );
    vertex.normal = vec3<f32>(
        bitcast<f32>(vertex_buffers[vertex_buffer_id].words[base + 3u]),
        bitcast<f32>(vertex_buffers[vertex_buffer_id].words[base + 4u]),
        bitcast<f32>(vertex_buffers[vertex_buffer_id].words[base + 5u]),
    );
    vertex.uv = vec2<f32>(
        bitcast<f32>(vertex_buffers[vertex_buffer_id].words[base + 6u]),
        bitcast<f32>(vertex_buffers[vertex_buffer_id].words[base + 7u]),
    );
    vertex.tangent = vec4<f32>(0.0);
    if stride >= 12u {
        vertex.tangent = vec4<f32>(
            bitcast<f32>(vertex_buffers[vertex_buffer_id].words[base + 8u]),
            bitcast<f32>(vertex_buffers[vertex_buffer_id].words[base + 9u]),
            bitcast<f32>(vertex_buffers[vertex_buffer_id].words[base + 10u]),
            bitcast<f32>(vertex_buffers[vertex_buffer_id].words[base + 11u]),
        );
    }
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
    flags: u32,
    _padding_a: u32,
    _padding_b: u32,
    reflectance: f32,
    uv_transform: mat3x3<f32>,
}

const TEXTURE_MAP_NONE = 0xFFFFFFFFu;

// Mirrors the behavior of StandardMaterialFlags::FLIP_NORMAL_MAP_Y in the raster path. The bit
// value is Solari's own; binder.rs sets it from StandardMaterial::flip_normal_map_y.
const MATERIAL_FLAGS_FLIP_NORMAL_MAP_Y = 1u;
// Mirrors StandardMaterialFlags::METALLIC_ROUGHNESS_RG: the metallic-roughness texture is a
// two-channel image packed R = roughness, G = metallic. binder.rs sets it from the format.
const MATERIAL_FLAGS_METALLIC_ROUGHNESS_RG = 2u;

const MIRROR_ROUGHNESS_THRESHOLD = 0.001f;

struct LightSource {
    // The low bit is the kind. For emissive meshes, the upper 31 bits are the
    // first triangle in this logical light's at-most-65535-triangle chunk.
    kind: u32,
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
@group(0) @binding(16) var<storage, read> scene_parameters: SceneParameters;

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
    triangle_id: u32,
    triangle_count: u32,
    ray_origin_bias: f32,
    // False when the ray reached the triangle from behind its outward-facing side. Emission is
    // one-sided, so BRDF paths must gate emissive collection on this to match NEE, which culls the
    // back side through cos_theta_light.
    front_face: bool,
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
        if (material.flags & MATERIAL_FLAGS_METALLIC_ROUGHNESS_RG) != 0u {
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
    var hit = resolve_triangle_data_full(ray_hit.instance_custom_data, ray_hit.primitive_index, barycentrics);
    hit.front_face = ray_hit.front_face;
    if !ray_hit.front_face {
        hit.world_normal = -hit.world_normal;
        hit.geometric_world_normal = -hit.geometric_world_normal;
    }
    return hit;
}

fn load_vertices(instance_geometry_ids: InstanceGeometryIds, triangle_id: u32) -> array<Vertex, 3> {
    let indices_i = (triangle_id * 3u) + vec3(0u, 1u, 2u) + instance_geometry_ids.index_buffer_offset;
    let index_buffer_id = instance_geometry_ids.index_buffer_id;
    let indices = vec3(
        index_buffers[index_buffer_id].indices[indices_i.x],
        index_buffers[index_buffer_id].indices[indices_i.y],
        index_buffers[index_buffer_id].indices[indices_i.z],
    ) + instance_geometry_ids.vertex_buffer_offset;

    return array<Vertex, 3>(
        load_vertex(instance_geometry_ids.vertex_buffer_id, indices.x, instance_geometry_ids.vertex_stride_words),
        load_vertex(instance_geometry_ids.vertex_buffer_id, indices.y, instance_geometry_ids.vertex_stride_words),
        load_vertex(instance_geometry_ids.vertex_buffer_id, indices.z, instance_geometry_ids.vertex_stride_words)
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

    let raw_uv = mat3x2(vertices[0].uv, vertices[1].uv, vertices[2].uv) * barycentrics;
    let uv = (material.uv_transform * vec3(raw_uv, 1.0)).xy;

    let local_normal = mat3x3(vertices[0].normal, vertices[1].normal, vertices[2].normal) * barycentrics;
    var local_tangent = mat3x3(vertices[0].tangent.xyz, vertices[1].tangent.xyz, vertices[2].tangent.xyz) * barycentrics;
    var tangent_sign = vertices[0].tangent.w;
    if instance_geometry_ids.vertex_stride_words < 12u {
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
        var Nt = sample_texture(material.normal_map_texture_id, uv) * 2.0 - 1.0;
        if (material.flags & MATERIAL_FLAGS_FLIP_NORMAL_MAP_Y) != 0u {
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
        // Triangle data resolved without a ray (light samples) is described from its emitting side.
        true,
        resolved_material,
    );
}
