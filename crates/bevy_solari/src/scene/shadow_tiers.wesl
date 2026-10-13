//! Directional light visibility in distance tiers (`ShadowTiers`): the scene inside the near
//! sphere, partial occluders inside the mid sphere, and a ground and canopy heightfield beyond.
import bevy_pbr::render::utils::rand_u;
import package::scene::bindings::{occluded, shadow_tiers, shadow_ground, shadow_canopy, RAY_T_MIN, RAY_T_MAX, RAY_MASK_ALL_RAYS, SHADOW_TIERS_TIERED, SHADOW_TIERS_HEIGHTFIELD};

enable wgpu_ray_query;

/// Steps after which the heightfield march stops and returns the transmittance so far.
const HEIGHTFIELD_MAX_STEPS = 96u;
/// Texels a ray may cross at the level it marches while rising half the greatest canopy height.
const HEIGHTFIELD_RISE_TEXELS = 64.0;
/// Transmittance below which the march stops.
const TRANSMITTANCE_FLOOR = 0.01;
/// Metres of origin that share a partial occluder bit, so a surface keeps its bit under jitter
/// and reuse while its light sample does.
const PARTIAL_MASK_CELL = 0.25;

/// Visibility of a directional light from `origin` along the unit `direction`, in this frame's
/// scene or with `previous` the previous frame's.
fn directional_visibility(origin: vec3<f32>, direction: vec3<f32>, previous: bool) -> f32 {
    if (shadow_tiers.flags & SHADOW_TIERS_TIERED) == 0u {
        return f32(!occluded(previous, origin, direction, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE, RAY_MASK_ALL_RAYS));
    }
    let sphere = shadow_tiers.spheres[u32(previous)];
    let near_t = sphere_exit(origin, direction, sphere.center, sphere.near_radius);
    if near_t > RAY_T_MIN && occluded(previous, origin, direction, RAY_T_MIN, near_t, RAY_FLAG_NONE, RAY_MASK_ALL_RAYS) {
        return 0.0;
    }
    // Without a heightfield the mid tier reaches as far as the scene does.
    let heightfield = (shadow_tiers.flags & SHADOW_TIERS_HEIGHTFIELD) != 0u;
    let mid_t = select(RAY_T_MAX, sphere_exit(origin, direction, sphere.center, sphere.mid_radius), heightfield);
    if mid_t > near_t + RAY_T_MIN && occluded(previous, origin, direction, max(near_t, RAY_T_MIN), mid_t, RAY_FLAG_CULL_BACK_FACING, partial_mask(origin, direction)) {
        return 0.0;
    }
    if !heightfield {
        return 1.0;
    }
    return heightfield_visibility(origin, direction, near_t, mid_t);
}

/// Distance along the ray at which it leaves the sphere, 0 for an origin outside it.
fn sphere_exit(origin: vec3<f32>, direction: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    let oc = origin - center;
    let c = dot(oc, oc) - radius * radius;
    if c >= 0.0 {
        return 0.0;
    }
    let b = dot(oc, direction);
    return sqrt(b * b - c) - b;
}

/// One of the partial occluder mask bits, chosen by the ray's origin cell and direction.
fn partial_mask(origin: vec3<f32>, direction: vec3<f32>) -> u32 {
    let cell = bitcast<vec3<u32>>(vec3<i32>(floor(origin / PARTIAL_MASK_CELL)));
    var state = (cell.x * 0x9E3779B1u) ^ (cell.y * 0x85EBCA77u) ^ (cell.z * 0xC2B2AE3Du)
        ^ (bitcast<u32>(direction.x) * 0x27D4EB2Fu) ^ (bitcast<u32>(direction.y) * 0x165667B1u) ^ bitcast<u32>(direction.z);
    return 2u << (rand_u(&state) % 7u);
}

/// The heightfield textures' layout, read once per march.
struct Heightfield {
    /// Metres per ground texel at level 0.
    texel: f32,
    ground_levels: i32,
    canopy_levels: i32,
    /// Canopy texels are ground texels this many levels up; both sizes are powers of two.
    canopy_shift: i32,
}

fn heightfield() -> Heightfield {
    let ground_size = textureDimensions(shadow_ground).x;
    return Heightfield(
        shadow_tiers.extent / f32(ground_size),
        i32(textureNumLevels(shadow_ground)),
        i32(textureNumLevels(shadow_canopy)),
        i32(firstLeadingBit(ground_size)) - i32(firstLeadingBit(textureDimensions(shadow_canopy).x)),
    );
}

/// Transmittance of the heightfield along the ray: the ground from `ground_from` on and the
/// canopy from `canopy_from` on. Either 0 means the march starts at the surface the ray leaves:
/// the ground march then starts above the ground that surface stands on, and the canopy march
/// `canopy_self_radius` out when the surface is in the canopy.
fn heightfield_visibility(origin: vec3<f32>, direction: vec3<f32>, ground_from: f32, canopy_from: f32) -> f32 {
    let hf = heightfield();

    var start = origin;
    var canopy_start = canopy_from;
    if ground_from == 0.0 {
        let ground = ground_height(hf, origin.xz);
        start.y = max(origin.y, ground);
        if canopy_from == 0.0 && origin.y > ground + shadow_tiers.canopy_surface_height {
            canopy_start = shadow_tiers.canopy_self_radius / max(length(direction.xz), 0.0001);
        }
    }

    // A ray at a low elevation crosses many texels while inside the canopy band, so it marches
    // coarser texels, whose coverage averages and heights bound the texels they hold.
    let rise = max(direction.y, 0.001);
    let rise_texels = 0.5 * shadow_tiers.canopy_height_max / (rise * hf.texel);
    let min_level = clamp(i32(ceil(log2(rise_texels / HEIGHTFIELD_RISE_TEXELS))), 0, hf.ground_levels - 1);

    var t = ground_from;
    var level = min_level;
    var transmittance = 1.0;
    // The ground under the last step's exit, which the next step enters.
    var ground_cache_t = -1.0;
    var ground_cache = 0.0;
    for (var step = 0u; step < HEIGHTFIELD_MAX_STEPS; step++) {
        let p = start + direction * t;
        let local = p.xz - shadow_tiers.origin;
        if any(local < vec2(0.0)) || any(local >= vec2(shadow_tiers.extent)) || (direction.y > 0.0 && p.y > shadow_tiers.height_max) {
            break;
        }

        let texel = hf.texel * f32(1u << u32(level));
        let cell = floor(local / texel);
        let t_exit = t + texel_exit(local, direction.xz, cell, texel);
        let in_canopy = t_exit > canopy_start;

        // Nothing in the texel reaches the ray: skip it.
        var top = textureLoad(shadow_ground, vec2<i32>(cell), level).r;
        if in_canopy {
            top += canopy_max(hf, cell, level) * shadow_tiers.canopy_height_max;
        }
        let y_low = min(p.y, start.y + direction.y * t_exit);
        if y_low >= top {
            t = t_exit;
            level = min(level + 1, hf.ground_levels - 1);
            continue;
        }
        if level > min_level {
            level -= 1;
            continue;
        }

        // The ground is tested at full resolution: a coarse texel's maximum stands above a
        // receiver on its slope. The midpoint catches a ridge inside a coarse texel.
        let p_exit = start + direction * t_exit;
        let p_mid = 0.5 * (p + p_exit);
        let ground_in = select(ground_height(hf, p.xz), ground_cache, t == ground_cache_t);
        let ground_out = ground_height(hf, p_exit.xz);
        ground_cache_t = t_exit;
        ground_cache = ground_out;
        let slack = shadow_tiers.ground_slack;
        if p.y < ground_in - slack || p_exit.y < ground_out - slack
            || (level > 0 && p_mid.y < ground_height(hf, p_mid.xz) - slack) {
            return 0.0;
        }
        if in_canopy {
            let canopy = canopy_sample(hf, p_mid.xz, level);
            let height = canopy.g * shadow_tiers.canopy_height_max;
            let ground = 0.5 * (ground_in + ground_out);
            let band = vec2(ground + shadow_tiers.canopy_base * height, ground + height);
            let t_in = max(t, canopy_start);
            let y_in = start.y + direction.y * t_in;
            let y_range = vec2(min(y_in, p_exit.y), max(y_in, p_exit.y));
            let overlap = min(y_range.y, band.y) - max(y_range.x, band.x);
            let share = select(f32(y_range.x >= band.x && y_range.x <= band.y), saturate(overlap / (y_range.y - y_range.x)), y_range.y - y_range.x > 0.0001);
            let crossed = share * (t_exit - t_in);
            transmittance *= 1.0 - canopy.r * (1.0 - exp(-shadow_tiers.canopy_extinction * crossed));
            if transmittance < TRANSMITTANCE_FLOOR {
                return 0.0;
            }
        }
        t = t_exit;
        level = min(level + 1, hf.ground_levels - 1);
    }
    return transmittance;
}

/// Distance along the ray from `local` (XZ relative to the heightfield origin) to where it leaves
/// the texel `cell` of `texel` metres, with the ray's unit direction's XZ.
fn texel_exit(local: vec2<f32>, direction: vec2<f32>, cell: vec2<f32>, texel: f32) -> f32 {
    let edge = (cell + step(vec2(0.0), direction)) * texel;
    let distance = select(vec2(shadow_tiers.extent), (edge - local) / direction, abs(direction) > vec2(0.000001));
    // Past the edge by a hair, so the next step starts in the next texel.
    return max(min(distance.x, distance.y), 0.0) + 0.001 * texel;
}

/// Bilinear load of `level` of a heightfield texture at `xz`, whose texels are `texel` metres.
fn load_bilinear(texture: texture_2d<f32>, level: i32, texel: f32, xz: vec2<f32>) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(texture, level));
    let u = (xz - shadow_tiers.origin) / texel - 0.5;
    let i = vec2<i32>(floor(u));
    let f = u - floor(u);
    let i0 = clamp(i, vec2(0), size - 1);
    let i1 = clamp(i + 1, vec2(0), size - 1);
    let c00 = textureLoad(texture, i0, level);
    let c10 = textureLoad(texture, vec2(i1.x, i0.y), level);
    let c01 = textureLoad(texture, vec2(i0.x, i1.y), level);
    let c11 = textureLoad(texture, i1, level);
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

/// Bilinear ground height at `xz`.
fn ground_height(hf: Heightfield, xz: vec2<f32>) -> f32 {
    return load_bilinear(shadow_ground, 0, hf.texel, xz).r;
}

/// The canopy level whose texels are at least as coarse as ground `level`'s.
fn canopy_level(hf: Heightfield, level: i32) -> i32 {
    return clamp(level - hf.canopy_shift, 0, hf.canopy_levels - 1);
}

/// The greatest canopy height (over `canopy_height_max`) in the ground texel `cell` of `level`.
fn canopy_max(hf: Heightfield, cell: vec2<f32>, level: i32) -> f32 {
    // Below the canopy's finest level a ground texel is part of a canopy texel.
    let shift = u32(max(hf.canopy_shift - level, 0));
    return textureLoad(shadow_canopy, vec2<i32>(cell) >> vec2(shift), canopy_level(hf, level)).b;
}

/// Bilinear canopy coverage (r) and height over `canopy_height_max` (g) at `xz`, from the canopy
/// level matching ground `level`.
fn canopy_sample(hf: Heightfield, xz: vec2<f32>, level: i32) -> vec2<f32> {
    let canopy_level = canopy_level(hf, level);
    let texel = hf.texel * f32(1u << u32(hf.canopy_shift + canopy_level));
    return load_bilinear(shadow_canopy, canopy_level, texel, xz).rg;
}
