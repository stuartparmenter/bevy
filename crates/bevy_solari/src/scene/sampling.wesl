import bevy_pbr::render::pbr_lighting::D_GGX;
import bevy_pbr::render::utils::{rand_vec2f, rand_u, rand_range_u};
import bevy_render::maths::{PI_2, orthonormalize};
import package::scene::bindings::{trace_ray, trace_ray_previous_frame, RAY_T_MIN, RAY_T_MAX, geometry_ids, light_sources, directional_lights, LightSource, LIGHT_SOURCE_KIND_ENVIRONMENT, light_source_is_emissive_mesh, resolve_triangle_data_full, ResolvedRayHitFull, MIRROR_ROUGHNESS_THRESHOLD, sample_environment_direction, sample_environment_radiance};

enable wgpu_ray_query;

fn power_heuristic(f: f32, g: f32) -> f32 {
    return balance_heuristic(f * f, g * g);
}

fn balance_heuristic(f: f32, g: f32) -> f32 {
    // Need to guard against NaNs since ReSTIR reservoirs can have UCW=0
    if f == 0.0 {
        return 0.0;
    }
    return max(0.0, 1.0 / (1.0 + (g / f)));
}

/// https://gpuopen.com/download/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf (Listing 1)
/// Result is invalid when output.z <= 0.0, and must be discarded
fn sample_ggx_vndf(wi_tangent: vec3<f32>, roughness: f32, rng: ptr<function, u32>) -> vec3<f32> {
    // Mirror BRDF case
    if roughness <= MIRROR_ROUGHNESS_THRESHOLD {
        return vec3(-wi_tangent.xy, wi_tangent.z);
    }

    let i = wi_tangent;
    let rand = rand_vec2f(rng);
    let i_std = normalize(vec3(i.xy * roughness, i.z));
    let phi = PI_2 * rand.x;
    let a = roughness;
    let s = 1.0 + length(vec2(i.xy));
    let a2 = a * a;
    let s2 = s * s;
    let k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
    let b = select(i_std.z, k * i_std.z, i.z > 0.0);
    let z = fma(1.0 - rand.y, 1.0 + b, -b);
    let sin_theta = sqrt(saturate(1.0 - z * z));
    let o_std = vec3(sin_theta * cos(phi), sin_theta * sin(phi), z);
    let m_std = i_std + o_std;
    let m = normalize(vec3(m_std.xy * roughness, m_std.z));
    return 2.0 * dot(i, m) * m - i;
}

fn ggx_vndf_sample_invalid(ray_tangent: vec3<f32>) -> bool {
    return !(ray_tangent.z > 0.0);
}

/// https://gpuopen.com/download/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf (Listing 2)
fn ggx_vndf_pdf(wi_tangent: vec3<f32>, wo_tangent: vec3<f32>, roughness: f32) -> f32 {
    // Mirror BRDF case
    if roughness <= MIRROR_ROUGHNESS_THRESHOLD {
        let mirror_wo = vec3(-wi_tangent.xy, wi_tangent.z);
        if all(abs(mirror_wo - wo_tangent) < vec3(0.0001)) {
            return bitcast<f32>(0x7F800000u); // INF
        } else {
            return 0.0;
        }
    }

    let i = wi_tangent;
    let o = wo_tangent;
    let m = normalize(i + o);
    let ndf = D_GGX(roughness, saturate(m.z));
    let ai = roughness * i.xy;
    let len2 = dot(ai, ai);
    let t = sqrt(len2 + i.z * i.z);
    var pdf: f32;
    if i.z >= 0.0 {
        let a = roughness;
        let s = 1.0 + length(i.xy);
        let a2 = a * a;
        let s2 = s * s;
        let k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
        pdf = ndf / (2.0 * (k * i.z + t));
    } else {
        pdf = ndf * (t - i.z) / (2.0 * len2);
    }

    return select(pdf, 0.0, isnan(pdf));
}

fn isinf(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u;
}

fn isnan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

fn isfinite(x: f32) -> bool {
    return !isinf(x) && !isnan(x);
}

fn isfinite3(x: vec3<f32>) -> bool {
    return isfinite(x.x) && isfinite(x.y) && isfinite(x.z);
}

const NULL_LIGHT_ID = 0xFFFFFFFFu;
const MAX_EMISSIVE_TRIANGLES_PER_LIGHT = 0xFFFFu;

struct LightSample {
    light_id: u32,
    seed: u32,
}

struct ResolvedLightSample {
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
    radiance: vec3<f32>,
    inverse_pdf: f32,
}

struct LightContribution {
    radiance: vec3<f32>,
    inverse_pdf: f32,
    /// inverse_pdf may not be in solid angle measure (e.g. area measure)
    /// but inverse_solid_angle_pdf is always in solid angle measure
    /// for cases where it's required, e.g. MIS between light sampling techniques
    inverse_solid_angle_pdf: f32,
    wi: vec3<f32>,
    brdf_rays_can_hit: bool,
    is_environment: bool,
}

struct GenerateRandomLightSampleResult {
    light_sample: LightSample,
    resolved_light_sample: ResolvedLightSample,
}

// True when a missed BRDF ray would collect this light's radiance from sample_environment_radiance,
// so the two strategies have to be MIS-weighted. Sun disks are directional but are not part of the
// environment radiance, so they stay NEE-only.
fn light_sample_is_environment(light_sample: LightSample) -> bool {
    if light_sample.light_id == NULL_LIGHT_ID {
        return false;
    }
    return light_sources[light_sample.light_id >> 16u].kind == LIGHT_SOURCE_KIND_ENVIRONMENT;
}

fn emissive_light_first_triangle(light_source: LightSource) -> u32 {
    return light_source.kind >> 1u;
}

fn emissive_light_triangle_count(light_source: LightSource) -> u32 {
    let first_triangle = emissive_light_first_triangle(light_source);
    let total_triangle_count = geometry_ids[light_source.id].triangle_count;
    let remaining = total_triangle_count - min(first_triangle, total_triangle_count);
    return min(remaining, MAX_EMISSIVE_TRIANGLES_PER_LIGHT);
}

fn emissive_hit_triangle_count(hit: ResolvedRayHitFull) -> u32 {
    let first_triangle =
        (hit.triangle_id / MAX_EMISSIVE_TRIANGLES_PER_LIGHT) * MAX_EMISSIVE_TRIANGLES_PER_LIGHT;
    let remaining = hit.triangle_count - min(first_triangle, hit.triangle_count);
    return min(remaining, MAX_EMISSIVE_TRIANGLES_PER_LIGHT);
}

fn sample_random_light(ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, ray_origin_bias: f32, rng: ptr<function, u32>) -> LightContribution {
    let sample = generate_random_light_sample(rng);
    var light_contribution = calculate_resolved_light_contribution(sample.resolved_light_sample, ray_origin, origin_world_normal);
    if light_sample_is_environment(sample.light_sample) {
        light_contribution.is_environment = true;
        light_contribution.brdf_rays_can_hit = true;
    }
    light_contribution.radiance *= trace_visibility(ray_origin, origin_world_normal, sample.resolved_light_sample.world_position, ray_origin_bias);
    return light_contribution;
}

fn random_emissive_light_pdf(hit: ResolvedRayHitFull, ray_distance: f32, NdotV: f32) -> f32 {
    let light_count = arrayLength(&light_sources);
    let triangle_count = emissive_hit_triangle_count(hit);
    if light_count == 0u || triangle_count == 0u || hit.triangle_area <= 0.0 || NdotV <= 0.0 {
        return 0.0;
    }
    let area_pdf = 1.0 / (f32(light_count) * f32(triangle_count) * hit.triangle_area);
    let pdf = area_pdf * (ray_distance * ray_distance) / NdotV;
    return select(0.0, pdf, isfinite(pdf) && pdf >= 0.0);
}

fn generate_random_light_sample(rng: ptr<function, u32>) -> GenerateRandomLightSampleResult {
    let light_count = arrayLength(&light_sources);
    if light_count == 0u {
        return GenerateRandomLightSampleResult(
            LightSample(NULL_LIGHT_ID, 0u),
            ResolvedLightSample(vec4(0.0), vec3(0.0), vec3(0.0), 0.0),
        );
    }
    let light_id = rand_range_u(light_count, rng);

    let light_source = light_sources[light_id];

    var triangle_id = 0u;
    if light_source_is_emissive_mesh(light_source) {
        let triangle_count = emissive_light_triangle_count(light_source);
        triangle_id = rand_range_u(triangle_count, rng);
    }

    let seed = rand_u(rng);
    let light_sample = LightSample((light_id << 16u) | triangle_id, seed);

    var resolved_light_sample = resolve_light_sample(light_sample, light_source);
    resolved_light_sample.inverse_pdf *= f32(light_count);

    return GenerateRandomLightSampleResult(light_sample, resolved_light_sample);
}

fn resolve_light_sample(light_sample: LightSample, light_source: LightSource) -> ResolvedLightSample {
    if light_source.kind == LIGHT_SOURCE_KIND_ENVIRONMENT {
        // Deterministic per seed: ReSTIR re-resolves (light_id, seed) every frame. After
        // generate_random_light_sample multiplies inverse_pdf by the light count, its reciprocal is
        // exactly environment_light_pdf(direction), which every MIS site relies on.
        let sample = sample_environment_direction(light_sample.seed);
        if !(sample.pdf > 0.0) {
            return ResolvedLightSample(vec4(0.0, 1.0, 0.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0), 0.0);
        }
        return ResolvedLightSample(
            vec4(sample.world_direction, 0.0),
            -sample.world_direction,
            sample_environment_radiance(sample.world_direction),
            1.0 / sample.pdf,
        );
    } else if !light_source_is_emissive_mesh(light_source) {
        let directional_light = directional_lights[light_source.id];

        var direction_to_light: vec3<f32>;
@if(!NO_DIRECTIONAL_LIGHT_SOFT_SHADOWS) {
        // Sample a random direction within a cone whose base is the sun approximated as a disk
        // https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec30%3A305
        var rng = light_sample.seed;
        let random = rand_vec2f(&rng);
        let cos_theta = (1.0 - random.x) + random.x * directional_light.cos_theta_max;
        let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        let phi = random.y * PI_2;
        let x = cos(phi) * sin_theta;
        let y = sin(phi) * sin_theta;
        direction_to_light = vec3(x, y, cos_theta);

        // Rotate the ray so that the cone it was sampled from is aligned with the light direction
        direction_to_light = orthonormalize(directional_light.direction_to_light) * direction_to_light;
} @else {
        direction_to_light = directional_light.direction_to_light;
}

        return ResolvedLightSample(
        vec4(direction_to_light, 0.0),
        -direction_to_light,
        directional_light.luminance,
        directional_light.inverse_pdf,
    );
} else {
        let triangle_count = emissive_light_triangle_count(light_source);
        let triangle_id = emissive_light_first_triangle(light_source)
            + (light_sample.light_id & 0xFFFFu);
        let barycentrics = triangle_barycentrics(light_sample.seed);
        let triangle_data = resolve_triangle_data_full(light_source.id, triangle_id, barycentrics);

        return ResolvedLightSample(
            vec4(triangle_data.world_position, 1.0),
            triangle_data.world_normal,
            triangle_data.material.emissive.rgb,
            f32(triangle_count) * triangle_data.triangle_area,
        );
    }
}

fn calculate_resolved_light_contribution(resolved_light_sample: ResolvedLightSample, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>) -> LightContribution {
    let ray = resolved_light_sample.world_position.xyz - (resolved_light_sample.world_position.w * ray_origin);
    let light_distance_squared = dot(ray, ray);
    if !(light_distance_squared > 1e-12) || !isfinite(light_distance_squared) {
        return LightContribution(vec3(0.0), 0.0, 0.0, vec3(0.0), false, false);
    }
    let wi = ray * inverseSqrt(light_distance_squared);

    let cos_theta_light = saturate(dot(-wi, resolved_light_sample.world_normal));
    let denominator = cos_theta_light / light_distance_squared;

    let radiance = resolved_light_sample.radiance * denominator;
    let inverse_solid_angle_pdf = resolved_light_sample.inverse_pdf * denominator;
    if !isfinite3(radiance) || !isfinite(resolved_light_sample.inverse_pdf) || !isfinite(inverse_solid_angle_pdf) {
        return LightContribution(vec3(0.0), 0.0, 0.0, wi, false, false);
    }

    // Environment samples also satisfy brdf_rays_can_hit, but the packed tile sample cannot say so:
    // callers that know the light source set it themselves.
    return LightContribution(radiance, resolved_light_sample.inverse_pdf, inverse_solid_angle_pdf, wi, resolved_light_sample.world_position.w == 1.0, false);
}

// The light sample point is resolved from the same vertex buffers and the same instance transform
// the BLAS was built from, so it lies exactly on the traced triangle. All the far end of a shadow
// ray has to survive is float rounding - not a geometry error. Spending the origin's error there as
// well truncated every shadow ray by that error a second time, which erased the shadow of any
// occluder standing closer to its light than the bias.
const LIGHT_SAMPLE_END_EPSILON_RELATIVE = 0.001f;

struct VisibilityRay {
    origin: vec3<f32>,
    direction: vec3<f32>,
    t_max: f32,
    // False when the two endpoints settle the query on their own and no ray is needed.
    trace: bool,
    // Visibility to report when `trace` is false.
    visibility: f32,
}

fn visibility_without_tracing(visibility: f32) -> VisibilityRay {
    return VisibilityRay(vec3(0.0), vec3(0.0), 0.0, false, visibility);
}

fn setup_visibility_ray(ray_origin_in: vec3<f32>, origin_world_normal: vec3<f32>, point: vec4<f32>, ray_origin_bias: f32) -> VisibilityRay {
    let toward_light = select(point.xyz, point.xyz - ray_origin_in, point.w == 1.0);
    if !isfinite3(toward_light) { return visibility_without_tracing(0.0); }

    // Lift the origin clear of its own instance's simplified BLAS first, then measure the ray from
    // where it actually starts, so the bias is spent once instead of at both ends. A light below
    // `origin_world_normal` lifts the origin into the surface, which is correct for a geometric
    // normal and would not be for a shading normal - but every caller weights the result by a cosine
    // against that same normal, so no shading normal reaches the flip with a nonzero weight.
    let side = select(-1.0, 1.0, dot(origin_world_normal, toward_light) >= 0.0);
    let ray_origin = ray_origin_in + origin_world_normal * side * max(ray_origin_bias, RAY_T_MIN);

    if point.w != 1.0 {
        return VisibilityRay(ray_origin, point.xyz, RAY_T_MAX, true, 0.0);
    }

    let ray = point.xyz - ray_origin;
    let dist = length(ray);
    if !isfinite(dist) { return visibility_without_tracing(0.0); }

    let t_max = dist - max(RAY_T_MIN, LIGHT_SAMPLE_END_EPSILON_RELATIVE * dist);
    // The light sits inside the shell the origin was lifted into, so nothing can lie between the
    // two points. Reporting that as occluded is what stopped a candle body being lit by its flame.
    if t_max < RAY_T_MIN { return visibility_without_tracing(1.0); }

    return VisibilityRay(ray_origin, ray / dist, t_max, true, 0.0);
}

fn trace_visibility(ray_origin_in: vec3<f32>, origin_world_normal: vec3<f32>, point: vec4<f32>, ray_origin_bias: f32) -> f32 {
    let ray = setup_visibility_ray(ray_origin_in, origin_world_normal, point, ray_origin_bias);
    if !ray.trace { return ray.visibility; }

    let ray_hit = trace_ray(ray.origin, ray.direction, RAY_T_MIN, ray.t_max, RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    return f32(ray_hit.kind == RAY_QUERY_INTERSECTION_NONE);
}

fn trace_visibility_previous_frame(ray_origin_in: vec3<f32>, origin_world_normal: vec3<f32>, point: vec4<f32>, ray_origin_bias: f32) -> f32 {
    let ray = setup_visibility_ray(ray_origin_in, origin_world_normal, point, ray_origin_bias);
    if !ray.trace { return ray.visibility; }

    let ray_hit = trace_ray_previous_frame(ray.origin, ray.direction, RAY_T_MIN, ray.t_max, RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    return f32(ray_hit.kind == RAY_QUERY_INTERSECTION_NONE);
}

/// https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec22%3A297
fn triangle_barycentrics(seed: u32) -> vec3<f32> {
    var rng = seed;
    var barycentrics = rand_vec2f(&rng);
    if barycentrics.x + barycentrics.y > 1.0 { barycentrics = 1.0 - barycentrics; }
    return vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics);
}
