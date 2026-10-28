
// Temporal filter for cloud shadow map. Reduces edge flickering by blending
// with history. Based on Unreal's MainShadowTemporalProcessCS and
// three-geospatial's ShadowResolveMaterial.
//
// Layout matches cloud_shadow_map:
// - R: front depth (km) from light-volume near plane
// - G: mean extinction (1/m)
// - B: max optical depth (unitless)
// - A: thin cloud layers' (cirrus) similarity-reduced slant optical depth along the light ray
//
// Note: We do NOT import bindings::settings here because it would conflict
// with our binding 1 (curr_cloud_shadow). Size is passed via temporal_params.

@group(0) @binding(3) var prev_cloud_sampler: sampler;

@group(0) @binding(0) var<uniform> temporal_params: CloudShadowTemporalParams;
// Traced output (storage, use textureLoad)
@group(0) @binding(1) var curr_cloud_shadow: texture_storage_2d<rgba16float, read>;
// History (sampled for bilinear)
@group(0) @binding(2) var prev_cloud_shadow: texture_2d<f32>;
@group(0) @binding(4) var out_cloud_shadow: texture_storage_2d<rgba16float, write>;

struct CloudShadowTemporalParams {
    curr_anchor: vec3<f32>,
    curr_light_dir: vec3<f32>,
    curr_basis_x: vec3<f32>,
    curr_basis_y: vec3<f32>,
    prev_anchor: vec3<f32>,
    prev_light_dir: vec3<f32>,
    prev_basis_x: vec3<f32>,
    prev_basis_y: vec3<f32>,
    extent: f32,
    temporal_alpha: f32,
    history_valid: u32,
    anchor_moved: u32,
    size: vec2<u32>,
    interleave: u32,
    frame_index: u32,
    // Cloud field displacement since the previous frame (atmosphere space, wind advection).
    cloud_motion: vec3<f32>,
    // Thin (cirrus) layer displacement since the previous frame: transports alpha.
    cirrus_motion: vec3<f32>,
    // |cos(light zenith)| / cloud layer thickness (1/m).
    history_angle_scale: f32,
    // Angle the light turned (rad) over the last k + 1 frames, k = 0..16 (component k % 4 of
    // element k / 4).
    rotation_since: array<vec4<f32>, 4>,
}

/// Angle the light turned over the last `frames` frames (0 for none).
fn light_turned_over(frames: u32) -> f32 {
    if (frames == 0u) {
        return 0.0;
    }
    let k = min(frames, 16u) - 1u;
    return temporal_params.rotation_since[k / 4u][k % 4u];
}

// Time-slicing: the texel (within each N x N tile) traced on a given frame, in Bayer order so
// that consecutive frames are spread out spatially. Must match `cloud_shadow_temporal.wesl`.
fn cloud_shadow_interleave_slot(frame: u32, n: u32) -> vec2<u32> {
    if (n >= 4u) {
        // Inverse 4x4 Bayer matrix: rank -> packed (x + 4 * y).
        const ORDER4 = array<u32, 16>(0u, 10u, 2u, 8u, 5u, 15u, 7u, 13u, 1u, 11u, 3u, 9u, 4u, 14u, 6u, 12u);
        let packed = ORDER4[frame % 16u];
        return vec2(packed % 4u, packed / 4u);
    }
    if (n == 2u) {
        // Inverse 2x2 Bayer matrix: rank -> packed (x + 2 * y).
        const ORDER2 = array<u32, 4>(0u, 3u, 1u, 2u);
        let packed = ORDER2[frame % 4u];
        return vec2(packed % 2u, packed / 2u);
    }
    return vec2(0u);
}

// Reprojects the previous frame's cloud shadow map into the current texel. Returns the history
// value in `rgb` (front depth already converted to the current near plane) and `w = 1` when the
// reprojection is in bounds.
//
// `nearest` point-samples the history. Untraced texels must use it: they are re-reprojected every
// frame, so bilinear resampling would keep blurring the map (and blend front depths of cloudy and
// clear texels into meaningless values).
//
// The reprojection happens at the occluder depth (the history's front depth at this texel), not
// at the near plane: the near plane sits `2 * extent` away from the anchor, so even the slow sun
// rotation of a day/night cycle moves it by more than a texel per frame.
// Must match clouds.wesl.
/// Warped cloud-shadow-map coordinates: light-space |xy|/extent = a rho + (1 - a) rho^3, rho = |2 uv - 1|.
/// Texels are a x the linear size at the anchor and (3 - 2a) x at the border, a single-map "near
/// cascade": looking along the light, near haze and ground shadows read one texel per screen
/// region, and uniform 125 m texels showed as straight-edged patches. The cubic keeps the fine
/// texels further out than a quadratic.
const CLOUD_SHADOW_WARP_A: f32 = 0.125;
fn cloud_shadow_uv_to_xy(uv: vec2<f32>, extent: f32) -> vec2<f32> {
    // Radial: the warp acts on the distance from the anchor, so texels stay near square (aspect
    // <= ~3). The separable form (per axis) made texels slabs up to (3 - 2a) / a = 22:1 along
    // the map axes, and nearly binary cumulus shadows on the ground came out as blocky
    // staircases (streaks at low sun).
    let s = 2.0 * uv - 1.0;
    let rho = length(s);
    let r = CLOUD_SHADOW_WARP_A * rho + (1.0 - CLOUD_SHADOW_WARP_A) * rho * rho * rho;
    return extent * s * (r / max(rho, 1e-6));
}
fn cloud_shadow_xy_to_uv(xy: vec2<f32>, extent: f32) -> vec2<f32> {
    // rho^3 + p rho + q = 0, p > 0: the hyperbolic form of the single real root is stable near
    // rho = 0 (Cardano's difference of cube roots cancelled there, so uv -> xy -> uv drifted and
    // the temporal reprojection crawled).
    let r = length(xy) / extent;
    let p = CLOUD_SHADOW_WARP_A / (1.0 - CLOUD_SHADOW_WARP_A);
    let q = -r / (1.0 - CLOUD_SHADOW_WARP_A);
    let k = 2.0 * sqrt(p / 3.0);
    let rho = -k * sinh(asinh(1.5 * q / p * sqrt(3.0 / p)) / 3.0);
    return 0.5 * (xy / extent) * (rho / max(r, 1e-6)) + 0.5;
}

/// Bilinear lookup of the history with the cumulus optical depth (b) interpolated as
/// transmittance. Cumulus optical depths are ~50: a linear blend of the stored depths put a clear
/// texel next to a cloud at depth 1-2 (transmittance 0.1-0.4) from a few percent of weight, and
/// untraced texels are reprojected every frame until their next trace, so the shadows grew and
/// darkened for up to 15 frames and snapped back at the trace (audit 1 validation: at the
/// default interleave, transmittance 0.087 too low on average against a dense trace, 0.41 at
/// shadow edges, and a 0.03 sawtooth flicker in a frozen scene). Interpolating transmittance
/// keeps its mean. The other channels are smooth or small and stay linear.
fn sample_history(uv: vec2<f32>) -> vec4<f32> {
    let size = vec2<i32>(temporal_params.size);
    let st = uv * vec2<f32>(size) - 0.5;
    let i0 = vec2<i32>(floor(st));
    let f = st - floor(st);
    var linear = vec4(0.0);
    var vis = 0.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2(k & 1, k >> 1);
        let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
        let tap = textureLoad(prev_cloud_shadow, clamp(i0 + o, vec2(0), size - 1), 0);
        linear += w * tap;
        vis += w * exp(-max(tap.b, 0.0));
    }
    // Below ~1e-30 (depth ~69) the blend is opaque either way: keep the linear depth.
    linear.b = select(linear.b, -log(vis), vis > 1e-30);
    return linear;
}

fn reproject_cloud_shadow_history(coord: vec2<i32>, uv: vec2<f32>, extent: f32, half_depth: f32, nearest: bool, motion: vec3<f32>) -> vec4<f32> {
    let max_t = 2.0 * half_depth;
    let curr_xy = cloud_shadow_uv_to_xy(uv, extent);
    // Positions relative to the anchors, not absolute: at planetary radius (~6.4e6 m) a float32
    // position resolves ~0.5 m, and the absolute round trip moved a static history by ~0.05 of
    // a 20 m central texel every frame, a blur that the untraced texels accumulated until their
    // next trace. The anchor difference is exactly 0 for a static anchor.
    let anchor_delta = temporal_params.curr_anchor - temporal_params.prev_anchor;
    // Depth estimate: history front depth at the same texel ("no cloud" texels use the
    // anchor depth, where the visible shadows are).
    let depth_estimate_km = textureLoad(prev_cloud_shadow, coord, 0).r;
    var depth_estimate = depth_estimate_km * 1000.0;
    if (depth_estimate >= 0.999 * max_t) {
        depth_estimate = half_depth;
    }
    // The occluder moved with the wind: look it up where it was last frame. Relative to the
    // current anchor:
    let curr_rel = temporal_params.curr_basis_x * curr_xy.x
        + temporal_params.curr_basis_y * curr_xy.y
        + temporal_params.curr_light_dir * (depth_estimate - half_depth)
        - motion;
    let prev_rel = curr_rel + anchor_delta;
    let prev_xy = vec2(
        dot(prev_rel, temporal_params.prev_basis_x),
        dot(prev_rel, temporal_params.prev_basis_y),
    );
    if (any(abs(prev_xy) > vec2(extent))) {
        return vec4(-1.0);
    }
    // Differential: this texel's uv plus the change of the inverse warp. The inverse goes
    // through sinh / asinh, whose GPU error moved even a static history systematically (up to a
    // tenth of a texel per frame at the rim, so clear texels gathered optical depth until their
    // next trace: map transmittance 0.09 below the trace at the rim, against 0.001 at the
    // centre). Differencing two evaluations cancels that error; a static map maps to itself.
    let prev_uv = uv + (cloud_shadow_xy_to_uv(prev_xy, extent) - cloud_shadow_xy_to_uv(curr_xy, extent));
    if (any(prev_uv < vec2(0.0)) || any(prev_uv > vec2(1.0))) {
        return vec4(-1.0);
    }

    var prev = sample_history(prev_uv);
    if (nearest) {
        // Untraced texels are reprojected every frame until their next trace. Front depth is
        // discontinuous at cloud edges, so it is taken from the nearest texel (bilinear would
        // leak depths across edges, compounding every frame). The extinction, optical depths
        // and thin-layer depth keep the bilinear lookup: rounding them to the nearest texel
        // dropped the sub-texel part of each frame's wind motion, so shadows stood still and
        // jumped at the next trace. Repeated bilinear lookups blur by at most ~0.4 texel before
        // the refresh (16 frames at a 0.01-texel/frame wind), under the consumers' 2-texel
        // B-spline.
        let size = vec2<i32>(temporal_params.size);
        let prev_coord = clamp(vec2<i32>(floor(prev_uv * vec2<f32>(size))), vec2(0), size - vec2(1));
        prev.r = textureLoad(prev_cloud_shadow, prev_coord, 0).r;
    }
    // Front depth: previous near plane -> current near plane (keeps "no cloud" at max depth).
    // Only when the anchor moved: the map is Rgba16Float, so converting every frame (e.g. for the
    // slow sun rotation) would re-quantize the depth, which is stored in km at up to 4 * extent.
    if (temporal_params.anchor_moved != 0u && prev.r * 1000.0 < 0.999 * max_t) {
        // The previous front point relative to the current ray origin (anchor-relative, as above).
        let prev_front = temporal_params.prev_basis_x * prev_xy.x
            + temporal_params.prev_basis_y * prev_xy.y
            + temporal_params.prev_light_dir * (prev.r * 1000.0 - half_depth)
            - anchor_delta;
        let curr_origin = temporal_params.curr_basis_x * curr_xy.x
            + temporal_params.curr_basis_y * curr_xy.y
            - temporal_params.curr_light_dir * half_depth;
        prev.r = clamp(dot(prev_front - curr_origin, temporal_params.curr_light_dir), 0.0, max_t) * 0.001;
    }
    return prev;
}

/// History of both layers: the cumulus channels (rgb) moved with the cumulus wind, the thin-layer
/// optical depth (alpha) with the cirrus wind. One cumulus motion for all four channels left the
/// cirrus shadow trailing (or, without cumulus, standing still) between traces.
fn reproject_layers(coord: vec2<i32>, uv: vec2<f32>, extent: f32, half_depth: f32, nearest: bool) -> vec4<f32> {
    var prev = reproject_cloud_shadow_history(coord, uv, extent, half_depth, nearest, temporal_params.cloud_motion);
    if (prev.r >= 0.0) {
        let thin = reproject_cloud_shadow_history(coord, uv, extent, half_depth, nearest, temporal_params.cirrus_motion);
        if (thin.r >= 0.0) {
            prev.a = thin.a;
        }
    }
    return prev;
}

/// Frames since the texel in tile slot `slot` was last traced (0: this frame), from the Bayer
/// order of `cloud_shadow_interleave_slot`.
fn cloud_shadow_texel_age(slot: vec2<u32>, n: u32) -> u32 {
    let frames = n * n;
    for (var k = 0u; k < frames; k++) {
        let f = temporal_params.frame_index + frames - k;
        if (all(cloud_shadow_interleave_slot(f, n) == slot)) {
            return k;
        }
    }
    return frames;
}

/// Trust in a history that saw the light `turned` (rad) away from the current one: the optical
/// depth of a turned ray is a different path through the cloud, not a displaced one, so
/// reprojection cannot correct it. The ray's far end in the cloud layer moves sideways by
/// turned x thickness / |mu|; the history is trusted while that stays within the texel,
/// exp(-turned / (texel x |mu| / thickness)). A static light keeps it at 1.
fn cloud_shadow_history_trust(uv: vec2<f32>, turned: f32) -> f32 {
    let rho = length(2.0 * uv - 1.0);
    let a = CLOUD_SHADOW_WARP_A;
    let texel = temporal_params.extent * (a + 3.0 * (1.0 - a) * rho * rho)
        * 2.0 / f32(temporal_params.size.x);
    return exp(-turned / max(texel * temporal_params.history_angle_scale, 1e-9));
}

/// This frame's traces, interpolated to `coord`: bilinear over the sparse lattice of traced
/// texels (one per n x n tile, at `slot`), optical depth as transmittance, front depth from the
/// nearest trace. Correct for the current light at 1/n resolution.
fn cloud_shadow_fresh(coord: vec2<i32>, slot: vec2<u32>, n: u32) -> vec4<f32> {
    let size = vec2<i32>(temporal_params.size);
    let ni = i32(n);
    let q = (vec2<f32>(coord) - vec2<f32>(slot)) / f32(n);
    let i0 = vec2<i32>(floor(q));
    let f = q - floor(q);
    let lattice = vec2<i32>(slot);
    var linear = vec4(0.0);
    var vis = 0.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2(k & 1, k >> 1);
        let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
        // The nearest traced lattice point on the map (the lattice ends inside the last tile).
        var c = (i0 + o) * ni + lattice;
        c = select(c, c - ni, c >= size);
        c = select(c, c + ni, c < vec2(0));
        let tap = textureLoad(curr_cloud_shadow, c);
        linear += w * tap;
        vis += w * exp(-max(tap.b, 0.0));
    }
    linear.b = select(linear.b, -log(vis), vis > 1e-30);
    var near = vec2<i32>(round(q)) * ni + lattice;
    near = select(near, near - ni, near >= size);
    near = select(near, near + ni, near < vec2(0));
    linear.r = textureLoad(curr_cloud_shadow, near).r;
    return linear;
}

/// mix of two shadow-map values, the cumulus optical depth (b) as transmittance.
fn cloud_shadow_mix(a: vec4<f32>, b: vec4<f32>, t: f32) -> vec4<f32> {
    var m = mix(a, b, t);
    let vis = mix(exp(-max(a.b, 0.0)), exp(-max(b.b, 0.0)), t);
    m.b = select(m.b, -log(vis), vis > 1e-30);
    return m;
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = temporal_params.size;
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let size_f = vec2<f32>(size);
    let uv = (vec2<f32>(gid.xy) + vec2(0.5)) / size_f;
    let extent = temporal_params.extent;
    let half_depth = extent * 2.0;

    // Current frame data (traced output). With time-slicing, only one texel per N x N tile was
    // traced this frame; the others still hold the value traced on an earlier frame.
    let curr_data = textureLoad(curr_cloud_shadow, coord);
    let interleave = max(1u, temporal_params.interleave);
    let slot = cloud_shadow_interleave_slot(temporal_params.frame_index, interleave);
    let traced_this_frame = all((gid.xy % vec2(interleave)) == slot);

    var filtered_data = curr_data;
    // Untraced texels of a time-sliced map hold whatever was traced N² frames ago, which is the
    // cleared (zero optical depth = fully lit) texture right after startup / a history cut, or
    // was traced for another light direction. Without a usable history they take this frame's
    // traces, interpolated.
    if (!traced_this_frame) {
        filtered_data = cloud_shadow_fresh(coord, slot, interleave);
    }

    if (temporal_params.history_valid != 0u && !traced_this_frame) {
        // Not traced this frame: reproject the previous filtered result, trusted as far as the
        // light has turned since this texel was traced; the rest from this frame's traces. A
        // static light keeps the reprojected history (full resolution); a turning light gets
        // the current light's shadows at 1/N resolution instead of a smeared history.
        let reprojected = reproject_layers(coord, uv, extent, half_depth, true);
        if (reprojected.r >= 0.0) {
            let age = cloud_shadow_texel_age(gid.xy % vec2(interleave), interleave);
            let trust = cloud_shadow_history_trust(uv, light_turned_over(age));
            filtered_data = cloud_shadow_mix(filtered_data, reprojected, trust);
            filtered_data.r = select(filtered_data.r, reprojected.r, trust >= 0.5);
        }
    } else if (temporal_params.history_valid != 0u) {
        // Clamp-to-neighborhood (AABB): compute min/max of G and B in 3x3 neighborhood.
        // Prevents history from contributing outlier values at edges, reducing jitter.
        // Based on Unreal VolumetricRenderTarget and TSR/TAA.
        var g_min = curr_data.g;
        var g_max = curr_data.g;
        var b_min = curr_data.b;
        var b_max = curr_data.b;
        var a_min = curr_data.a;
        var a_max = curr_data.a;
        let size_i = vec2<i32>(size);
        for (var dy = -1; dy <= 1; dy += 1) {
            for (var dx = -1; dx <= 1; dx += 1) {
                let nc = clamp(coord + vec2<i32>(dx, dy), vec2<i32>(0, 0), size_i - vec2<i32>(1, 1));
                let n = textureLoad(curr_cloud_shadow, nc);
                a_min = min(a_min, n.a);
                a_max = max(a_max, n.a);
                g_min = min(g_min, n.g);
                g_max = max(g_max, n.g);
                b_min = min(b_min, n.b);
                b_max = max(b_max, n.b);
            }
        }

        let reprojected = reproject_layers(coord, uv, extent, half_depth, false);
        if (reprojected.r >= 0.0) {
            var prev = reprojected;

            // Clamp history to neighborhood AABB before blending (reduces edge jitter)
            prev.g = clamp(prev.g, g_min, g_max);
            prev.b = clamp(prev.b, b_min, b_max);
            prev.a = clamp(prev.a, a_min, a_max);

            // Temporal blend: prev + alpha * (curr - prev)
            // Do NOT filter depth (Unreal: precision/convergence issues)
            // The history was traced N² frames ago: weighted by the trust left after the light
            // turned since (all of the trace for a fast-turning light).
            let trust = cloud_shadow_history_trust(uv, light_turned_over(interleave * interleave));
            let alpha = 1.0 - (1.0 - temporal_params.temporal_alpha) * trust;
            filtered_data.g = prev.g + alpha * (curr_data.g - prev.g);
            // The cumulus optical depth blends as transmittance (see `sample_history`): a linear
            // blend of depths is darker than the blend of the transmittances (exp is convex), so
            // a history that differs from the trace (after a reset, or blurred by reprojection
            // under motion) darkened and dilated the shadows. With the fixed quadrature the trace
            // is deterministic and either space converges to it; transmittance recovers faster
            // from a reset (audit 1: 0.05 vs 0.11 mean error 300 frames after startup).
            let vis = mix(exp(-max(prev.b, 0.0)), exp(-max(curr_data.b, 0.0)), alpha);
            filtered_data.b = select(prev.b + alpha * (curr_data.b - prev.b), -log(vis), vis > 1e-30);
            // ATM_DBG_SHADOW_JITTER: the traces are noisy, unbiased estimates of the optical depth,
            // and only their depth average is unbiased (an average of transmittances is too bright).
            @if(ATM_DBG_SHADOW_JITTER) { filtered_data.b = prev.b + alpha * (curr_data.b - prev.b); }
            filtered_data.a = prev.a + alpha * (curr_data.a - prev.a);
            filtered_data.r = curr_data.r;
        }
    }

    textureStore(out_cloud_shadow, coord, max(filtered_data, vec4(0.0)));
}
