// The physical cloud scheme's moisture fluctuation and its tables (`clouds.wesl`
// cloud_scheme_density):
// - `bake`: the fluctuation texture (`fbm_noise.rs` FbmNoiseTexture::scheme_noise), 128^3 on the
//   shape-noise tile: R the shape noise Gaussianized, g = Phi^-1(F(s)), G its variance (0 here;
//   the mips hold the variance over each block, `noise_mip_downsample_3d.wesl` moments), B the
//   thermal bubbles b = sum_k 2^(-H k) max(Phi^-1(F(m_k)), 0) over three octaves of bubble cells
//   (8, 16, 32 cells per tile), A its variance. m_k is the largest weighted spherical bump of the jittered
//   cell points (the alligator primitive without the second-largest subtraction): a bubble per
//   cell, high inside it. The amplitude 2^(-H k), H = 2/3, continues the spectrum of the shape
//   noise (base frequency 4 per tile): level sets of a field of Hurst exponent H have dimension
//   3 - H = 7/3, the measured dimension of cumulus surfaces (Siebesma & Jonker 2000, LES 2.32 +-
//   0.01; observed via D_p ~ 4/3). The shader samples the texture three times, at the tile and
//   at 1/8 and 1/64 of it (bubbles of tile/8 .. tile/2048), so the octaves continue down to a
//   few metres.
// - rows of the cloud profile texture after the cloud-type rows: one per shape-noise mip, the
//   distribution F(v) of the shape noise (R), texel i at v = i / (width - 1), and in every texel
//   the mean (G) and variance (B) of the bubbles b; then the scheme row, texel i at cloud fraction
//   p = (i / (width - 1))^2: R the threshold t that Y = g + r (b1 + s b2 + s^2 b3) exceeds with
//   probability p (s = 8^-H the second and third taps; r the bubble weight), G the depth c of the
//   mixed shell in Y (CLOUD_EDGE_SHELL times the mean of Y - t over the cloud), B the liquid-water
//   magnitude 1 / E[min(Y - t, c) / c] over the cloud, A 1.
//   Liquid water is min(Y - t, c) / c times the magnitude: linear in the excess within the shell,
//   flat in the core (`clouds.wesl` cloud_scheme_density).
//   P(Y > t) and E[(Y - t)+] integrate the measured distribution of b1 against g + r (s b2 +
//   s^2 b3), the latter Gaussian with the measured mean and variance (s r b2 + s^2 r b3 is a
//   small, nearly Gaussian addition: s r = 0.16).

@group(0) @binding(0) var noise: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>, 4096>;
@group(0) @binding(2) var profile_lut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var detail: texture_3d<f32>;

struct QuantileParams {
    _pad0: f32,
    // Weight r of the thermal bubbles (`clouds.wesl` cloud_bubble_weight).
    bubble_weight: f32,
    _pad1: f32,
    _pad2: f32,
}
@group(0) @binding(4) var<uniform> params: QuantileParams;
@group(0) @binding(5) var scheme_out: texture_storage_3d<rgba16float, write>;

// After the cloud-type rows (`cloud_profile.rs` CLOUD_PROFILE_TYPE_RES).
const FIRST_DISTRIBUTION_ROW: u32 = 64u;
const MIPS: u32 = 8u;
const SCHEME_ROW: u32 = FIRST_DISTRIBUTION_ROW + MIPS;
const BINS: u32 = 256u;
// Bin blocks: shape noise per mip (on [0, 1]), the bubble bump m (on [0, 1]), the bubbles b (on
// [0, B_RANGE], B_BINS bins), and the cumulative counts of the shape mip 0 and bump blocks.
const SHAPE_BLOCK: u32 = 0u;
const BUMP_BLOCK: u32 = MIPS * BINS;
const B_BLOCK: u32 = BUMP_BLOCK + BINS;
const B_BINS: u32 = 1024u;
const B_RANGE: f32 = 10.0;
const CUM_SHAPE: u32 = B_BLOCK + B_BINS;
const CUM_BUMP: u32 = CUM_SHAPE + BINS;

// Must match clouds.wesl: CLOUD_SCHEME_H, CLOUD_SCHEME_TAP.
const H: f32 = 0.6666667;
const BUBBLE_OCTAVES: u32 = 3u;
const BUBBLE_BASE_CELLS: i32 = 8;
const TAP: f32 = 8.0;

// Must match clouds.wesl: remap and the shape-noise composite.
fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    let t = clamp((x - a) / max(1e-6, b - a), 0.0, 1.0);
    return mix(c, d, t);
}

fn shape_noise(pw: vec4<f32>) -> f32 {
    let wfbm = pw.y * 0.625 + pw.z * 0.25 + pw.w * 0.125;
    return saturate(remap(pw.x, wfbm - 1.0, 1.0, 0.0, 1.0));
}

fn levels() -> u32 {
    return min(textureNumLevels(noise), MIPS);
}

fn bin_of(v: f32) -> u32 {
    return min(u32(saturate(v) * f32(BINS)), BINS - 1u);
}

fn pcg3d(v_in: vec3<u32>) -> vec3<u32> {
    var v = v_in * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v = v ^ (v >> vec3(16u));
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

fn hash4(c: vec3<i32>, seed: u32) -> vec4<f32> {
    let h = pcg3d(vec3<u32>(c) + vec3(seed * 1013u, seed * 7919u, seed * 104729u));
    let h2 = pcg3d(h ^ vec3(0x9e3779b9u));
    return vec4(vec3<f32>(h), f32(h2.x)) * (1.0 / 4294967295.0);
}

/// Bubble field: the largest weighted smoothstep bump of the jittered cell points (one per cell,
/// radius one cell), periodic with `period` cells over the unit cube.
fn bubble_bump(p: vec3<f32>, period: i32, seed: u32) -> f32 {
    let q = p * f32(period);
    let ip = vec3<i32>(floor(q));
    let fp = q - floor(q);
    var m = 0.0;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let o = vec3(x, y, z);
                let c = ((ip + o) % period + period) % period;
                let h = hash4(c, seed);
                let d = length(fp - (h.xyz + vec3<f32>(o)));
                let t = saturate(1.0 - d);
                m = max(m, h.w * t * t * (3.0 - 2.0 * t));
            }
        }
    }
    return m;
}

fn octave_seed(k: u32) -> u32 {
    return 811u + 97u * k;
}

@compute @workgroup_size(4, 4, 4)
fn histogram_marginals(@builtin(global_invocation_id) gid: vec3<u32>) {
    for (var m = 0u; m < levels(); m++) {
        if (all(gid < textureDimensions(noise, m))) {
            let s = shape_noise(textureLoad(noise, vec3<i32>(gid), i32(m)));
            atomicAdd(&bins[SHAPE_BLOCK + m * BINS + bin_of(s)], 1u);
        }
    }
    // Every octave has the same distribution (the same construction at another scale).
    let dims = textureDimensions(scheme_out);
    if (all(gid < dims)) {
        let p = (vec3<f32>(gid) + 0.5) / vec3<f32>(dims);
        atomicAdd(&bins[BUMP_BLOCK + bin_of(bubble_bump(p, BUBBLE_BASE_CELLS, octave_seed(0u)))], 1u);
    }
}

// Cumulative counts (exclusive) of the shape mip-0 and bump blocks, for constant-time CDFs.
@compute @workgroup_size(2)
fn prefix(@builtin(global_invocation_id) gid: vec3<u32>) {
    let src = select(SHAPE_BLOCK, BUMP_BLOCK, gid.x == 1u);
    let dst = select(CUM_SHAPE, CUM_BUMP, gid.x == 1u);
    if (gid.x > 1u) {
        return;
    }
    var sum = 0u;
    for (var b = 0u; b < BINS; b++) {
        atomicStore(&bins[dst + b], sum);
        sum += atomicLoad(&bins[src + b]);
    }
}

// Distribution at v from a block and its cumulative counts (piecewise linear within a bin).
fn cdf_fast(block: u32, cum: u32, v: f32) -> f32 {
    let total = atomicLoad(&bins[cum + BINS - 1u]) + atomicLoad(&bins[block + BINS - 1u]);
    let x = saturate(v) * f32(BINS);
    let xb = min(u32(x), BINS - 1u);
    let below = f32(atomicLoad(&bins[cum + xb])) + (x - f32(xb)) * f32(atomicLoad(&bins[block + xb]));
    return saturate(below / f32(max(total, 1u)));
}

// Standard normal quantile, Abramowitz & Stegun 26.2.23 (as clouds.wesl normal_quantile).
fn normal_quantile(p: f32) -> f32 {
    let q = clamp(p, 1e-6, 1.0 - 1e-6);
    let lower = q < 0.5;
    let t = sqrt(-2.0 * log(select(1.0 - q, q, lower)));
    let x = t - (2.515517 + t * (0.802853 + t * 0.010328))
        / (1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308)));
    return select(x, -x, lower);
}

@compute @workgroup_size(4, 4, 4)
fn bake(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(scheme_out);
    if (any(gid >= dims)) {
        return;
    }
    // Same tile and size as the shape noise: texel for texel.
    let s = shape_noise(textureLoad(noise, vec3<i32>(gid), 0));
    let g = normal_quantile(cdf_fast(SHAPE_BLOCK, CUM_SHAPE, s));
    let p = (vec3<f32>(gid) + 0.5) / vec3<f32>(dims);
    var b = 0.0;
    var cells = BUBBLE_BASE_CELLS;
    for (var k = 0u; k < BUBBLE_OCTAVES; k++) {
        let m = bubble_bump(p, cells, octave_seed(k));
        b += pow(2.0, -H * f32(k)) * max(normal_quantile(cdf_fast(BUMP_BLOCK, CUM_BUMP, m)), 0.0);
        cells *= 2;
    }
    textureStore(scheme_out, vec3<i32>(gid), vec4(g, 0.0, b, 0.0));
    atomicAdd(&bins[B_BLOCK + min(u32(b / B_RANGE * f32(B_BINS)), B_BINS - 1u)], 1u);
}

// Distribution of a [0, 1] histogram block at v (piecewise linear within a bin).
fn cdf(block: u32, v: f32) -> f32 {
    var total = 0u;
    for (var b = 0u; b < BINS; b++) {
        total += atomicLoad(&bins[block + b]);
    }
    let x = saturate(v) * f32(BINS);
    let xb = min(u32(x), BINS - 1u);
    var below = 0u;
    for (var k = 0u; k < xb; k++) {
        below += atomicLoad(&bins[block + k]);
    }
    return saturate((f32(below) + (x - f32(xb)) * f32(atomicLoad(&bins[block + xb]))) / f32(max(total, 1u)));
}

// One invocation per texel of every mip row.
@compute @workgroup_size(64)
fn distributions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = textureDimensions(profile_lut).x;
    let i = gid.x % width;
    let m = gid.x / width;
    if (m >= levels()) {
        return;
    }
    let v = f32(i) / f32(width - 1u);
    let mb = bubble_moments();
    textureStore(profile_lut, vec2<i32>(i32(i), i32(FIRST_DISTRIBUTION_ROW + m)),
        vec4(cdf(SHAPE_BLOCK + m * BINS, v), mb.mean, mb.var_, 0.0));
}

fn phi(x: f32) -> f32 {
    return 0.39894228 * exp(-0.5 * x * x);
}

// Mills ratio R(x) = Q(x) / phi(x), Abramowitz & Stegun 7.1.26.
fn mills_ratio(x: f32) -> f32 {
    let z = abs(x) * 0.70710678;
    let k = 1.0 / (1.0 + 0.3275911 * z);
    let poly = k * (0.254829592 + k * (-0.284496736 + k * (1.421413741 + k * (-1.453152027 + k * 1.061405429))));
    let r_abs = 0.5 * poly * 2.50662827;
    if (x >= 0.0) {
        return r_abs;
    }
    let ph = phi(x);
    return (1.0 - r_abs * ph) / ph;
}

// Upper tail Q(x) = P(N > x).
fn upper(x: f32) -> f32 {
    if (x >= 0.0) {
        return mills_ratio(x) * phi(x);
    }
    return 1.0 - mills_ratio(-x) * phi(x);
}

struct BubbleMoments {
    total: f32,
    mean: f32,
    var_: f32,
}

fn bubble_moments() -> BubbleMoments {
    var n = 0.0;
    var s1 = 0.0;
    var s2 = 0.0;
    for (var i = 0u; i < B_BINS; i++) {
        let c = f32(atomicLoad(&bins[B_BLOCK + i]));
        let b = (f32(i) + 0.5) * B_RANGE / f32(B_BINS);
        n += c;
        s1 += c * b;
        s2 += c * b * b;
    }
    let mean = s1 / max(n, 1.0);
    return BubbleMoments(max(n, 1.0), mean, max(s2 / max(n, 1.0) - mean * mean, 0.0));
}

// (P(Y > t), E[(Y - t)+]) for Y = g + r b1 + r (s b2 + s^2 b3): the measured b1 distribution
// against N(mean_d, sigma_d^2), the Gaussian g plus the two small taps.
fn scheme_moments(t: f32, mean_d: f32, sigma_d: f32, total: f32) -> vec2<f32> {
    var out = vec2(0.0);
    for (var i = 0u; i < B_BINS; i++) {
        let c = f32(atomicLoad(&bins[B_BLOCK + i]));
        if (c == 0.0) {
            continue;
        }
        let b = (f32(i) + 0.5) * B_RANGE / f32(B_BINS);
        let d = (params.bubble_weight * b + mean_d - t) / sigma_d;
        let above = upper(-d);
        out += c * vec2(above, sigma_d * (phi(d) + d * above));
    }
    return out / total;
}

// Depth of the mixed shell, in units of the mean excess (must match clouds.wesl): liquid water
// grows linearly with the excess over the outer part of a cloud and is flat in its core. Fitted to
// the aircraft composite of 1624 cumulus penetrations (Wang, Geerts & French 2009, Fig. 5a:
// liquid water flat in the interior, dropping within ~10 % of the radius of the edge), rms 0.05
// at cloud fractions 0.05-0.2 (.clouds-handoff/noise_study/physical_detail/edge_fit.py).
const CLOUD_EDGE_SHELL: f32 = 0.5;

// One invocation per scheme texel.
@compute @workgroup_size(64)
fn scheme(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = textureDimensions(profile_lut).x;
    let i = gid.x;
    if (i >= width) {
        return;
    }
    let u = f32(i) / f32(width - 1u);
    let p = clamp(u * u, 1e-5, 1.0 - 1e-5);
    let mb = bubble_moments();
    let r = params.bubble_weight;
    let s = pow(TAP, -H);
    let mean_d = r * (s + s * s) * mb.mean;
    let sigma_d = sqrt(1.0 + r * r * (s * s + s * s * s * s) * mb.var_);
    // Threshold: bisection on P(Y > t) = p over the support of Y.
    var lo = -8.0;
    var hi = 8.0 + r * B_RANGE * (1.0 + s + s * s);
    for (var it = 0u; it < 40u; it++) {
        let mid = 0.5 * (lo + hi);
        if (scheme_moments(mid, mean_d, sigma_d, mb.total).x > p) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = 0.5 * (lo + hi);
    let mean_excess = max(scheme_moments(t, mean_d, sigma_d, mb.total).y / p, 1e-4);
    let c = CLOUD_EDGE_SHELL * mean_excess;
    // E[min(Y - t, c)+] = E[(Y - t)+] - E[(Y - t - c)+].
    let shell = (scheme_moments(t, mean_d, sigma_d, mb.total).y
        - scheme_moments(t + c, mean_d, sigma_d, mb.total).y) / (c * p);
    textureStore(profile_lut, vec2<i32>(i32(i), i32(SCHEME_ROW)), vec4(t, c, 1.0 / max(shell, 1e-3), 1.0));
}
