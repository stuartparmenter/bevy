// Cloud rendering functions using 3D FBM noise

import bevy_render::maths::{PI, ray_sphere_intersect};
import package::render::utils::interleaved_gradient_noise;
import package::atmosphere::{
    types::{Atmosphere, AtmosphereSettings},
    bindings::{settings, atmosphere},
};

struct GpuCloudLayers {
    cloud_layer_start: f32,
    cloud_layer_end: f32,
    cloud_density: f32,
    cloud_absorption: f32,
    cloud_scattering: f32,
    // Liquid water (kg/m^3) at density 1.
    liquid_water: f32,
    // 0: physical lookups (`cloud_profile.rs`), 1: analytic envelope x^a (1 - x)^b,
    // 2: Nubis's vertical profile (height in the layer, 1 - p threshold).
    profile_mode: u32,
    profile_bottom_exponent: f32,
    profile_top_exponent: f32,
    profile_top_min: f32,
    noise_scale: f32,
    noise_offset: vec3<f32>,
    detail_noise_scale: f32,
    detail_strength: f32,
    wind: vec3<f32>,
    wind_shear: vec3<f32>,
    updraft_speed: f32,
    lifetime: f32,
    evolution_time: f32,
    coverage: f32,
    shape_noise_scale: f32,
    weather_window_origin: vec2<f32>,
    droplet_number: f32,
    // Thin layers (ThinCloudLayer, mod.rs GpuCloudLayers), lowest mid level first.
    thin_count: u32,
    thin: array<GpuThinCloudLayer, 4>,
}

/// One thin (ice) layer (mod.rs GpuThinCloudLayer).
struct GpuThinCloudLayer {
    offset: vec3<f32>,
    top: f32,
    wind_shear: vec3<f32>,
    thickness: f32,
    optical_depth: f32,
    threshold: f32,
    ln_iwc_std: f32,
    norm: f32,
    fall_speed: f32,
    fall_speed_spread: f32,
    outer_scale: f32,
    asymmetry: f32,
}

@group(0) @binding(20) var<uniform> cloud_layer: GpuCloudLayers;
@group(0) @binding(21) var noise_texture_2d: texture_2d<f32>;
@group(0) @binding(22) var noise_sampler_3d: sampler;
@group(0) @binding(24) var perlin_worley_noise_3d: texture_3d<f32>;
@group(0) @binding(33) var detail_worley_noise_3d: texture_3d<f32>;
// Vertical profile lookups (`cloud_profile.rs`): R presence bottom, G presence top, B in-cloud
// liquid water; x = height in the column (texel i at i / 63), y = cloud type (texel j at j / 31).
@group(0) @binding(34) var cloud_profile_lut: texture_2d<f32>;

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if (l2 <= 1e-12) {
        return vec3(0.0, 1.0, 0.0);
    }
    return v * inverseSqrt(l2);
}

// Sample weather map (RGBA) at a given world-space scale.
// - R: coverage
// - G: cloud type (0 stratus → 0.5 stratocumulus → 1 cumulus)
// - B: density scale
// - A: spare
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

/// Size (m) of the warped cloud shadow map's texel at `uv` along its smaller, tangential axis:
/// extent (a + (1 - a) rho^2) x 2 / size. The radial axis is (a + 3 (1 - a) rho^2) / (a + (1 - a)
/// rho^2) times larger (2.75 x at the rim). Level-of-detail selection takes the smaller axis, as
/// hardware mip selection takes the larger footprint-to-texel ratio.
fn cloud_shadow_texel_m(uv: vec2<f32>, extent: f32, size: f32) -> f32 {
    let rho = length(2.0 * uv - 1.0);
    return extent * (CLOUD_SHADOW_WARP_A + (1.0 - CLOUD_SHADOW_WARP_A) * rho * rho) * 2.0 / max(size, 1.0);
}

/// Mip level of the cloud shadow map (`cloud_shadow_mips.wesl`) for a lookup whose pixel covers
/// `footprint` m of the light plane: log2 of the texels it spans, 0 to the last level.
fn cloud_shadow_lod(footprint: f32, uv: vec2<f32>, extent: f32, size: f32, levels: u32) -> f32 {
    let texels = max(footprint, 1e-3) / cloud_shadow_texel_m(uv, extent, size);
    return clamp(log2(texels), 0.0, f32(levels) - 1.0);
}

/// Angle one pixel subtends (rad) for a perspective projection (Bevy's: clip_from_view[1][1] =
/// 1 / tan(fov_y / 2), [3][3] = 0), 0 for an orthographic one (no footprint growth).
fn pixel_angle(clip_from_view: mat4x4<f32>, viewport_height: f32) -> f32 {
    if (clip_from_view[3][3] != 0.0) {
        return 0.0;
    }
    return 2.0 / (clip_from_view[1][1] * max(viewport_height, 1.0));
}

/// Footprint (m) in the light plane (basis `bx`, `by`) of a pixel of angle `angle` at distance
/// `d`, seen along `v` (unit, camera to point) on a surface with normal `n`: the pixel covers
/// d angle across the view and d angle / |v.n| along it on the surface (grazing views stretch
/// it), each projected along the light onto the map. The larger of the two.
fn cloud_shadow_surface_footprint(d: f32, angle: f32, v: vec3<f32>, n: vec3<f32>, bx: vec3<f32>, by: vec3<f32>) -> f32 {
    let w = d * angle;
    let across = safe_normalize(cross(n, v)) * w;
    let along = safe_normalize(v - n * dot(v, n)) * (w / max(abs(dot(v, n)), 1e-3));
    let a = length(vec2(dot(across, bx), dot(across, by)));
    let b = length(vec2(dot(along, bx), dot(along, by)));
    return max(a, b);
}

/// Weather-map uv to weather-window texel coordinates (`weather_window.wesl`).
fn weather_window_local(uv: vec2<f32>) -> vec2<f32> {
    return uv * vec2<f32>(textureDimensions(noise_texture_2d)) - cloud_layer.weather_window_origin;
}

/// Physical weather channels at a sample position (the same frame as `sample_weather`): the
/// window's B inside it, the defaults outside (cloud-fraction scale 1, air mass = the map's type,
/// the default droplet number, no base offset).
fn sample_weather_physical(world_pos: vec3<f32>, weather_scale: f32, map_type: f32) -> vec4<f32> {
    let rot = mat2x2<f32>(0.8660254, -0.5, 0.5, 0.8660254);
    let uv = (rot * world_pos.xz) / max(1.0, weather_scale);
    let local = weather_window_local(uv);
    let window = f32(textureDimensions(weather_window_physical).x);
    if (all(local > vec2(1.0)) && all(local < vec2(window - 1.0))) {
        return textureSampleLevel(weather_window_physical, noise_sampler_3d, local / window, 0.0);
    }
    return vec4(1.0, map_type, cloud_layer.droplet_number, 0.0);
}

/// Extinction of the droplets relative to the layer's reference microphysics at the same liquid
/// water: sigma = 3 Q LWC / (4 rho_w r_e), r_e^3 = 3 LWC / (4 pi rho_w k N_d), so at fixed LWC
/// sigma ∝ (k N_d)^(1/3) (Twomey: more droplets, more opaque clouds; k ~ 0.8 maritime, 0.67
/// continental, Martin, Johnson & Spice 1994). Relative to the layer's reference droplet number
/// (its LWC and r_e), with k cancelling within one air mass.
fn cloud_droplet_extinction_factor(droplet_number: f32) -> f32 {
    return pow(max(droplet_number, 1.0) / max(cloud_layer.droplet_number, 1.0), 1.0 / 3.0);
}

fn sample_weather(world_pos: vec3<f32>, weather_scale: f32) -> vec4<f32> {
    let rot = mat2x2<f32>(
        0.8660254, -0.5,
        0.5, 0.8660254
    ); // 30 degrees
    let xz = rot * world_pos.xz;
    let uv = xz / max(1.0, weather_scale);
    // R is the current coverage: the weather generator bakes the cloud life cycle into the map
    // (fbm_noise_3d.wesl), so one sample serves every density lookup. Inside the weather window
    // around the camera (`weather_window.wesl`, texel j = map texel origin + j) the window is
    // read, the tiled map outside it; the same bilinear weights, so they agree where no source
    // has changed the window.
    let local = weather_window_local(uv);
    let window = f32(textureDimensions(weather_window_art).x);
    var w: vec4<f32>;
    if (all(local > vec2(1.0)) && all(local < vec2(window - 1.0))) {
        w = textureSampleLevel(weather_window_art, noise_sampler_3d, local / window, 0.0);
    } else {
        w = textureSampleLevel(noise_texture_2d, noise_sampler_3d, uv, 0.0);
    }
    // Mesoscale cloud fraction from the A channel at 4x the tile (64/32/16 km): subtracts from
    // the coverage, keeping it in cloudy regions and clearing gaps between them. Subtractive,
    // not renormalised (Nubis ValueErosion, (v - e) / (1 - e)): renormalising stretched the tip
    // of every small coverage bump in a suppressed region back to full coverage, i.e. full layer
    // depth (the cloud top follows coverage), and those narrow blobs became tall extruded
    // columns. Lowering the coverage makes them smaller and shallower together, as cumulus in
    // suppressed regions are (depth scales with width).
    let meso = textureSampleLevel(noise_texture_2d, noise_sampler_3d, uv * 0.25 + vec2(0.37, 0.61), 0.0).a;
    let erosion = 1.0 - smoothstep(0.30, 0.65, meso);
    w.r = saturate((w.r - erosion) * cloud_layer.coverage);
    return w;
}

/// Shape-noise mip at the distance LOD `mip` (`cloud_mip_from_distance`, set for a 4 km shape
/// tile): offset by the tile size so a texel covers the same metres at a given distance with any
/// `shape_noise_scale` (a 6.4 km tile at the plain distance mip was 1.6x coarser: ~250 m at
/// 16 km). The detail noise and the camera distance keep the distance mip.
fn cloud_shape_mip(mip: f32) -> f32 {
    return max(mip - log2(max(cloud_layer.shape_noise_scale, 1.0) / 4000.0), 0.0);
}

fn sample_shape_noise(world_pos: vec3<f32>, h: f32, mip: f32) -> vec4<f32> {
    let rot = mat2x2<f32>(
        0.8660254, -0.5,
        0.5, 0.8660254
    );
    let xz = rot * world_pos.xz;
    // Shape tiles at shape_noise_scale (isotropic, m; it sets the cloud size). Using normalized
    // layer height for w made the vertical period ≈ layer thickness while xz spanned 16 km.
    let period = max(1.0, cloud_layer.shape_noise_scale);
    let uv = xz / period;
    // Rising thermals: the noise moves up through the fixed vertical envelope (noise_offset.y).
    let w = (h * cloud_layer_thickness_m() - cloud_layer.noise_offset.y) / period;
    return textureSampleLevel(perlin_worley_noise_3d, noise_sampler_3d, vec3(uv, w), mip);
}

fn sample_detail_noise(world_pos: vec3<f32>, h: f32, mip: f32) -> vec4<f32> {
    let rot = mat2x2<f32>(
        0.8660254, -0.5,
        0.5, 0.8660254
    );
    let xz = rot * world_pos.xz;
    let period = max(1.0, cloud_layer.detail_noise_scale);
    let uv = xz / period;
    let w = (h * cloud_layer_thickness_m() - cloud_layer.noise_offset.y) / period;
    return textureSampleLevel(detail_worley_noise_3d, noise_sampler_3d, vec3(uv, w), mip);
}

fn cloud_layer_thickness_m() -> f32 {
    return max(1.0, cloud_layer.cloud_layer_end - cloud_layer.cloud_layer_start);
}


/// Get cloud scattering coefficient per unit density
fn get_cloud_scattering_coeff() -> f32 {
    return cloud_layer.cloud_scattering;
}

/// Get cloud absorption coefficient per unit density
fn get_cloud_absorption_coeff() -> f32 {
    return cloud_layer.cloud_absorption;
}

// PERF/DEBUG: Hardcoded repeating sphere SDF volume (no texture fetches).
const CELL_SIZE_M: f32 = 4000.0;
const RADIUS_M: f32 = 750.0;
const SOFTNESS_M: f32 = 10.0;
const CLOUD_DEBUG_SINGLE_SPHERE: bool = false;
// Primary cloud shape toggle (debug spheres vs Nubis density). Keep a debug path (roadmap 2e).
const CLOUD_USE_NOISE_SHAPE: bool = true;
const CLOUD_DEBUG_GRADIENT_SUBSTEPS: bool = false;

fn debug_sphere_center() -> vec3<f32> {
    let center_r = 0.5 * (cloud_layer.cloud_layer_start + cloud_layer.cloud_layer_end);
    return vec3(0.0, center_r, 0.0);
}

fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    let t = clamp((x - a) / max(1e-6, b - a), 0.0, 1.0);
    return mix(c, d, t);
}

/// Height range of the profile lookups (m above the cloud base; `cloud_profile.rs`
/// CLOUD_PROFILE_RANGE_M).
const CLOUD_PROFILE_RANGE_M: f32 = 4000.0;
/// Rows of the profile texture: one per cloud type, then the shape- and billow-noise
/// distributions of each noise mip, then the cloud-scheme row (`cloud_noise_quantiles.wesl`).
const CLOUD_PROFILE_TYPE_ROWS: f32 = 64.0;
const CLOUD_NOISE_QUANTILE_ROWS: f32 = 8.0;
const CLOUD_SCHEME_ROW: f32 = 72.0;
/// First of the lean rows, one per cloud type (`fbm_noise.rs` CLOUD_LEAN_FIRST_ROW), and the shear
/// they are tabulated for (`cloud_profile.rs` CLOUD_LEAN_SHEAR).
const CLOUD_LEAN_FIRST_ROW: f32 = 73.0;
const CLOUD_LEAN_SHEAR: f32 = 1.0e-3;

/// Cloud base above the surface (m) of a cloud type with the profile lookups: the A channel, km
/// (`cloud_profile.rs`).
fn cloud_profile_base_m(cloud_type: f32) -> f32 {
    let n = vec2<f32>(textureDimensions(cloud_profile_lut));
    let v = (saturate(cloud_type) * (CLOUD_PROFILE_TYPE_ROWS - 1.0) + 0.5) / n.y;
    return textureSampleLevel(cloud_profile_lut, noise_sampler_3d, vec2(0.5 / n.x, v), 0.0).a * 1000.0;
}

/// Cloud type (weather G) at a position, as `sample_weather` reads it.
fn cloud_weather_type(world_pos: vec3<f32>) -> f32 {
    return cloud_column_weather(world_pos).y;
}

/// The weather of a column at a sample position, as every density function reads it:
/// (coverage, cloud type, density scale, droplet extinction factor). With the physical profiles
/// the physical channels (weather window B) scale the cloud fraction, set the air mass (the
/// profile row) and the droplet number; the others read the art channels only.
fn cloud_column_weather(sample_pos: vec3<f32>) -> vec4<f32> {
    let weather = sample_weather(sample_pos, cloud_layer.noise_scale);
    var coverage = clamp(weather.r, 0.0, 1.0);
    var cloud_type = clamp(weather.g, 0.0, 1.0);
    var droplets = 1.0;
    if (cloud_layer.profile_mode == 0u) {
        let physical = sample_weather_physical(sample_pos, cloud_layer.noise_scale, cloud_type);
        coverage = saturate(coverage * physical.r);
        cloud_type = saturate(physical.g);
        droplets = cloud_droplet_extinction_factor(physical.b);
    }
    return vec4(coverage, cloud_type, clamp(weather.b, 0.0, 1.0), droplets);
}

/// Vertical profile of a column: (dimensional profile, in-cloud liquid water / the layer's).
/// - Lookups (`CloudProfiles` physical or image, `cloud_profile.rs`), the Nubis Evolved (2022)
///   vertical-profile structure with physical contents: one row per cloud type (air mass),
///   against metres above that row's cloud base (A, km above the surface). R x G is the
///   cloud fraction (peak 1), B the in-cloud liquid water (kg/m^3). Dimensional profile =
///   cloud fraction x coverage; the erosion turns it into clouds that share a base and stop at
///   different heights, as in a cumulus field (statistical cloud scheme, Sommeria & Deardorff
///   1977; Siebesma et al. 2003). Liquid water / the layer's reference scales the density.
/// - Analytic envelope (profile_mode 1): h'^a (1 - h')^b, h' = height / column top, the top
///   rising with coverage from profile_top_min to the layer top; uniform liquid water.
/// Top and bottom type are one weather channel until the weather map carries both.
fn cloud_vertical_profile(h: f32, coverage: f32, cloud_type: f32) -> vec2<f32> {
    if (coverage <= 0.0) {
        return vec2(0.0);
    }
    if (cloud_layer.profile_mode == 1u) {
        let a = cloud_layer.profile_bottom_exponent;
        let b = cloud_layer.profile_top_exponent;
        let top = mix(cloud_layer.profile_top_min, 1.0, coverage);
        let x = h / max(top, 1e-3);
        if (x >= 1.0) {
            return vec2(0.0);
        }
        let peak = a / max(a + b, 1e-6);
        let max_v = max(pow(peak, a) * pow(1.0 - peak, b), 1e-6);
        return vec2(saturate(pow(x, a) * pow(1.0 - x, b) / max_v * coverage), 1.0);
    }
    // Texel i holds u = i / (N - 1): sample the ends at texel centres.
    let n = vec2<f32>(textureDimensions(cloud_profile_lut));
    let v = (saturate(cloud_type) * (CLOUD_PROFILE_TYPE_ROWS - 1.0) + 0.5) / n.y;
    if (cloud_layer.profile_mode == 2u) {
        // Nubis vertical profile: height as a fraction of the layer, bottom x top gradient.
        let lut = textureSampleLevel(cloud_profile_lut, noise_sampler_3d,
            vec2((saturate(h) * (n.x - 1.0) + 0.5) / n.x, v), 0.0);
        return vec2(saturate(lut.r * lut.g * coverage), 1.0);
    }
    // Height above the row's cloud base (each air mass has its own lifting condensation level).
    let base = cloud_profile_base_m(cloud_type);
    let z = h * cloud_layer_thickness_m() + (cloud_layer.cloud_layer_start - atmosphere.inner_radius) - base;
    if (z < 0.0 || z >= CLOUD_PROFILE_RANGE_M) {
        return vec2(0.0);
    }
    let u = (z / CLOUD_PROFILE_RANGE_M * (n.x - 1.0) + 0.5) / n.x;
    let lut = textureSampleLevel(cloud_profile_lut, noise_sampler_3d, vec2(u, v), 0.0);
    return vec2(saturate(lut.r * lut.g * coverage), lut.b / max(cloud_layer.liquid_water, 1e-9));
}

/// Mip offset of the detail noise relative to the shape noise (footprint LOD, see
/// sample_cumulus_shape_lod). Must match `fbm_noise.rs`.
fn cloud_detail_mip_offset() -> f32 {
    let detail_texel = max(cloud_layer.detail_noise_scale, 1.0) / 128.0;
    return max(log2(4000.0 * 2.2e-3 / detail_texel), 0.0);
}

/// Standard normal quantile, Abramowitz & Stegun 26.2.23 (|error| < 4.5e-4).
fn normal_quantile(p: f32) -> f32 {
    let q = clamp(p, 1e-6, 1.0 - 1e-6);
    let lower = q < 0.5;
    let t = sqrt(-2.0 * log(select(1.0 - q, q, lower)));
    let x = t - (2.515517 + t * (0.802853 + t * 0.010328))
        / (1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308)));
    return select(x, -x, lower);
}

/// Physical cloud scheme fluctuation (`cloud_noise_quantiles.wesl` bake): R g, G its variance, B b,
/// A its variance, on the shape-noise tile; each mip holds the mean and the variance over the block
/// it averages (0 at mip 0).
@group(0) @binding(35) var cloud_scheme_noise: texture_3d<f32>;
/// Weather window around the camera (`weather_window.wesl`): art channels (A) and physical
/// channels (B: cloud-fraction scale, air mass, droplet number, base offset).
@group(0) @binding(36) var weather_window_art: texture_2d<f32>;
@group(0) @binding(37) var weather_window_physical: texture_2d<f32>;

/// Hurst exponent of the moisture fluctuation: level sets of dimension 3 - H = 7/3, the measured
/// dimension of cumulus surfaces (Siebesma & Jonker 2000). Must match `cloud_noise_quantiles.wesl`.
const CLOUD_SCHEME_H: f32 = 0.6666667;
/// Scale ratio between the fluctuation texture's three taps (8 = its three bubble octaves).
const CLOUD_SCHEME_TAP: f32 = 8.0;
/// Angular footprint of a half-resolution cloud pixel (rad), as `cloud_detail_mip_offset`.
const CLOUD_PIXEL_FOOTPRINT: f32 = 2.2e-3;

/// Weight of the thermal bubbles against the shape noise (`fbm_noise.rs` cloud_bubble_weight):
/// the first bubble octave (cells of 1/8 of the shape tile) continues the shape noise's spectrum
/// (base frequency 4) as (1/2)^H, times the detail strength (0 = no bubbles).
fn cloud_bubble_weight() -> f32 {
    return pow(0.5, CLOUD_SCHEME_H) * max(cloud_layer.detail_strength, 0.0);
}

/// One tap of the fluctuation texture at `period` (m), rotated and offset per tap so the taps of
/// one texture are uncorrelated, at the level of detail of a footprint of `footprint` m.
fn cloud_scheme_tap(world_pos: vec3<f32>, h: f32, period: f32, tap: u32, footprint: f32) -> vec4<f32> {
    let angle = 0.5235988 + 1.1 * f32(tap);
    let c = cos(angle);
    let s = sin(angle);
    let xz = mat2x2<f32>(c, -s, s, c) * world_pos.xz;
    let w = h * cloud_layer_thickness_m() - cloud_layer.noise_offset.y;
    let uvw = vec3(xz.x, w, xz.y) / period + vec3(0.37, 0.61, 0.13) * f32(tap);
    let lod = log2(max(footprint / (period / 128.0), 1.0));
    return textureSampleLevel(cloud_scheme_noise, noise_sampler_3d, vec3(uvw.x, uvw.z, uvw.y), lod);
}

/// Standard normal density.
fn normal_pdf(x: f32) -> f32 {
    return 0.39894228 * exp(-0.5 * x * x);
}

/// Standard normal distribution, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
fn normal_cdf(x: f32) -> f32 {
    let z = abs(x) * 0.70710678;
    let k = 1.0 / (1.0 + 0.3275911 * z);
    let erfc = k * (0.254829592 + k * (-0.284496736 + k * (1.421413741 + k * (-1.453152027 + k * 1.061405429)))) * exp(-z * z);
    return select(0.5 * erfc, 1.0 - 0.5 * erfc, x >= 0.0);
}

/// Statistical cloud scheme (Sommeria & Deardorff 1977) at every scale, as a union of thermals:
/// the moisture fluctuation Y = g + r (b1 + s b2 + s^2 b3), g the Gaussianized shape noise (the
/// coherent plumes), b the thermal bubbles of three taps of the fluctuation texture (nine octaves
/// of bubble cells from 1/8 to 1/2048 of the shape tile; s = 8^-H), cloud where Y > t, liquid
/// water proportional to Y - t (`cloud_noise_quantiles.wesl`). A cumulus is a cluster of rising
/// bubbles (Scorer & Ludlam 1953; Blyth et al. 2005): bubbles add turrets and never carve, so the
/// cores stay solid while the surface reaches the measured dimension 7/3.
/// Each sample stands for its footprint (half-resolution pixel at the sample's distance): the
/// texture mips give the footprint mean mu of Y and the variance sigma^2 the footprint leaves
/// unresolved, and the expected excess over the footprint is the
/// Gaussian closure of the scheme, E[(mu + sigma Z - t)+] = sigma phi(d) + (mu - t) Phi(d),
/// d = (mu - t) / sigma, applied to the liquid water min(Y - t, c)+ / c: linear in the excess over
/// a mixed shell of depth c, flat in the core, as aircraft composites of cumulus show (Wang,
/// Geerts & French 2009: liquid water flat in the interior, dropping within ~10 % of the radius
/// of the edge; `cloud_noise_quantiles.wesl` CLOUD_EDGE_SHELL). Near the camera sigma -> 0 and the excess is the resolved Y - t; far away
/// the mean liquid water of the unresolved bubbles is kept (the footprint mean of the true
/// excess within a few percent at 400 m footprints, against -15 % for thresholding the filtered
/// field: .clouds-handoff noise study). Returns the liquid water over its in-cloud mean at cloud
/// fraction p (in-cloud mean 1).
fn cloud_scheme_density(world_pos: vec3<f32>, h: f32, p: f32, mip: f32) -> f32 {
    let cam_dist = (exp2(mip) - 1.0) * 4000.0;
    let footprint = cam_dist * CLOUD_PIXEL_FOOTPRINT;
    let period = max(1.0, cloud_layer.shape_noise_scale);
    let a = cloud_scheme_tap(world_pos, h, period, 0u, footprint);
    let r = cloud_bubble_weight();
    let s = pow(CLOUD_SCHEME_TAP, -CLOUD_SCHEME_H);
    let scheme = cloud_scheme(p);
    let bubbles = cloud_bubble_moments();
    // The finer taps (bubbles below 1/64 of the tile) only matter near the cloud surface: where
    // even their largest contribution cannot reach the threshold the sample is clear, and deep
    // in a cloud's core (beyond the mixed shell) the liquid water is flat, so both use their
    // exact mean and variance instead of the texture.
    let fine_mean = r * (s + s * s) * bubbles.x;
    let fine_var = r * r * (s * s + s * s * s * s) * bubbles.y;
    let fine_max = r * (s + s * s) * (bubbles.x + 7.0 * sqrt(bubbles.y));
    let mu0 = a.x + r * a.z;
    let sigma0 = sqrt(a.y + r * r * a.w);
    if (mu0 + fine_max < scheme.x - 3.0 * sigma0) {
        return 0.0;
    }
    var mu = mu0 + fine_mean;
    var var_ = sigma0 * sigma0 + fine_var;
    if (mu0 - scheme.x < scheme.y + fine_max + 3.0 * sigma0) {
        let b = cloud_scheme_tap(world_pos, h, period / CLOUD_SCHEME_TAP, 1u, footprint);
        let c = cloud_scheme_tap(world_pos, h, period / (CLOUD_SCHEME_TAP * CLOUD_SCHEME_TAP), 2u, footprint);
        mu = mu0 + r * (s * b.z + s * s * c.z);
        var_ = sigma0 * sigma0 + r * r * s * s * (b.w + s * s * c.w);
    }
    let sigma = sqrt(var_);
    let dm = mu - scheme.x;
    // Liquid water min(Y - t, c)+ / c (mixed shell of depth c, flat core), its footprint mean
    // E[(Y - t)+] - E[(Y - t - c)+].
    let shell = cloud_expected_excess(dm, sigma) - cloud_expected_excess(dm - scheme.y, sigma);
    return shell / max(scheme.y, 1e-4) * scheme.z;
}

/// E[(dm + sigma Z)+], Z standard normal: sigma phi(d) + dm Phi(d), d = dm / sigma (the resolved
/// max(dm, 0) when sigma -> 0). Beyond 3 sigma below zero it is < 4e-4 sigma: zero, so the
/// march's empty test (and its return to coarse steps) still sees clear air.
fn cloud_expected_excess(dm: f32, sigma: f32) -> f32 {
    if (sigma <= 1e-3) {
        return max(dm, 0.0);
    }
    let d = dm / sigma;
    return select(0.0, sigma * normal_pdf(d) + dm * normal_cdf(d), d > -3.0);
}

/// Mean and variance of the bubbles b (`cloud_noise_quantiles.wesl`, distribution rows G, B).
fn cloud_bubble_moments() -> vec2<f32> {
    let size = vec2<f32>(textureDimensions(cloud_profile_lut));
    return textureSampleLevel(cloud_profile_lut, noise_sampler_3d,
        vec2(0.5 / size.x, (CLOUD_PROFILE_TYPE_ROWS + 0.5) / size.y), 0.0).gb;
}


/// The cloud-scheme row at cloud fraction p (texel i at p = (i / (N - 1))^2): (threshold t, depth c
/// of the mixed shell, liquid-water magnitude, 1), `cloud_noise_quantiles.wesl` scheme.
fn cloud_scheme(p: f32) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(cloud_profile_lut));
    let u = (sqrt(saturate(p)) * (size.x - 1.0) + 0.5) / size.x;
    return textureSampleLevel(cloud_profile_lut, noise_sampler_3d,
        vec2(u, (CLOUD_SCHEME_ROW + 0.5) / size.y), 0.0);
}

/// Coarse density context: weather + dimensional profile (Nubis 2022 vertical_profile *
/// coverage) at the same sample position as the fine density (sample_cumulus_shape_lod).
/// Returns (bound of the eroded profile, density_scale^4, h, liquid water): the dimensional
/// profile with the analytic envelope (noise - (1 - p) <= p), 1 with the lookups (renormalized
/// excess).
fn cloud_density_context(r: f32, world_pos: vec3<f32>) -> vec4<f32> {
    let layer_thickness = cloud_layer.cloud_layer_end - cloud_layer.cloud_layer_start;
    let height_in_layer = r - cloud_layer.cloud_layer_start;
    let h = clamp(height_in_layer / max(1.0, layer_thickness), 0.0, 1.0);

    // Must match the fine density's sample position: the coarse density is the empty-space
    // predicate (ray march, SDF seeds).
    let sample_pos = cloud_sample_pos(world_pos, r);
    let column = cloud_column_weather(sample_pos);
    let density_scale = pow(column.z, 4.0);
    let profile = cloud_vertical_profile(h, column.x, column.y);
    // Lookups: cloud is possible wherever the cloud fraction is positive (the scheme's excess has
    // no small upper bound; the coarse density only tests for possible cloud).
    let bound = select(0.0, select(profile.x, 1.0, cloud_layer.profile_mode == 0u), profile.x > 0.0);
    // Return bound, density_scale, h, liquid water (sample_pos reconstructed by callers).
    return vec4(bound, density_scale, h, profile.y);
}

/// Sample position in the moving cloud field (all density inputs: weather, shape, detail):
/// - advection: the field moves with the cloud-base wind, density at `p` is the pattern at
///   `p - noise_offset` (horizontal);
/// - shear lean: a parcel rising at updraft speed w(z) through wind shear S above the base drifts
///   x(z) = S int_0^z z' / w(z') dz' downwind at height z, so the column is displaced by that much
///   (static: clouds are renewed by new thermals, the lean does not accumulate). With the lookup
///   profiles w(z) is the cloudy updraft of the column's air mass (the lean rows, `cloud_profile.rs`);
///   otherwise the layer's constant updraft speed, x = S z^2 / (2 w).
/// Nubis/HZD scroll the noise with the wind (pseudo-motion); HZD15 also skews with height.
fn cloud_sample_pos(world_pos: vec3<f32>, r: f32) -> vec3<f32> {
    // The lean starts at the cloud's own base: with the physical profiles, its air mass's (the
    // type at the advected, unleaned position). From the bottom of the layer, which spans every
    // air mass's base, a cloud with a higher base was tilted from its base up.
    var lean: vec3<f32>;
    if (cloud_layer.profile_mode == 0u) {
        let advected = world_pos - vec3(cloud_layer.noise_offset.x, 0.0, cloud_layer.noise_offset.z);
        let cloud_type = cloud_weather_type(advected);
        let z = max(r - atmosphere.inner_radius - cloud_profile_base_m(cloud_type), 0.0);
        let n = vec2<f32>(textureDimensions(cloud_profile_lut));
        let u = (min(z / CLOUD_PROFILE_RANGE_M, 1.0) * (n.x - 1.0) + 0.5) / n.x;
        let v = (CLOUD_LEAN_FIRST_ROW + saturate(cloud_type) * (CLOUD_PROFILE_TYPE_ROWS - 1.0) + 0.5) / n.y;
        let drift = textureSampleLevel(cloud_profile_lut, noise_sampler_3d, vec2(u, v), 0.0).r;
        lean = cloud_layer.wind_shear * (drift / CLOUD_LEAN_SHEAR);
    } else {
        let z = max(r - cloud_layer.cloud_layer_start, 0.0);
        lean = cloud_layer.wind_shear * (z * z / (2.0 * max(cloud_layer.updraft_speed, 0.1)));
    }
    return world_pos - vec3(cloud_layer.noise_offset.x + lean.x, 0.0, cloud_layer.noise_offset.z + lean.z);
}

/// Coarse density: weather + vertical profile only (no 3D noise), an upper bound of the fine
/// density (`sample_cumulus_shape_lod`) used as the empty-space predicate (ray march, SDF seeds).
/// The fine density is at most (profile * scale^4)^lerp(0.3, 0.6, scale^4) (noise <= 1, erosion
/// only removes); the bound must carry the same power, or thin low-density-scale clouds, which
/// the power lifts by orders of magnitude (2e-4 -> 0.08), fall under the empty threshold.
fn sample_cumulus_shape_coarse(r: f32, world_pos: vec3<f32>) -> f32 {
    let ctx = cloud_density_context(r, world_pos);
    let x = saturate(ctx.x * ctx.y);
    return select(0.0, pow(x, mix(0.3, 0.6, max(1e-4, ctx.y))), x > 0.0) * ctx.w;
}

/// Fine density with mip LOD (design.md §1.3 / §1.4).
fn sample_cumulus_shape_lod(r: f32, world_pos: vec3<f32>, mip: f32) -> f32 {
    let layer_thickness = cloud_layer.cloud_layer_end - cloud_layer.cloud_layer_start;
    let height_in_layer = r - cloud_layer.cloud_layer_start;
    let h = clamp(height_in_layer / max(1.0, layer_thickness), 0.0, 1.0);
    let sample_pos = cloud_sample_pos(world_pos, r);

    // Nubis detail type (0 wispy, 1 billowy) and density scale: authored weather fields; with
    // the physical profiles the air mass and droplet number of the physical channels.
    let column = cloud_column_weather(sample_pos);
    let coverage = column.x;
    let detail_type = column.y;
    let density_scale = column.z;
    let droplets = column.w;
    // Nubis Evolved (2022) sky model: dimensional_profile = vertical_profile * coverage.
    let vertical = cloud_vertical_profile(h, coverage, detail_type);
    let base_profile = vertical.x;
    if (base_profile <= 0.0) {
        return 0.0;
    }
    // Physical lookups: the statistical cloud scheme at the sample's footprint, its detail the
    // scheme's own thermal bubbles (`cloud_scheme_density`); liquid water from the lookups, the
    // weather's density scale as for the coarse bound, the droplets' extinction per liquid water.
    if (cloud_layer.profile_mode == 0u) {
        return cloud_scheme_density(sample_pos, h, base_profile, mip) * pow(density_scale, 4.0)
            * vertical.y * droplets;
    }

    // Large-scale shape from the 2022 noise (low-frequency Perlin-Worley eroded by Worley fBm,
    // HZD15: fbm = 0.625 lo + 0.25 med + 0.125 hi, remap(pw, fbm - 1, 1, 0, 1)), applied as the
    // 2022 sky density: saturate(noise - threshold), threshold = 1 - dimensional_profile.
    let shape_mip = cloud_shape_mip(mip);
    let pw = sample_shape_noise(sample_pos, h, shape_mip);
    let wfbm = pw.y * 0.625 + pw.z * 0.25 + pw.w * 0.125;
    let shape_noise = saturate(remap(pw.x, wfbm - 1.0, 1.0, 0.0, 1.0));
    // Detail noise. Detail LOD from the sample
    // footprint (anti-aliasing and cache): `mip` is log2(1 + d / 4 km) (shape noise); a half-res
    // pixel spans ~2.2 mrad, and a detail texel is detail_noise_scale / 128, so the footprint LOD
    // is mip + log2(4000 m * 2.2e-3 / texel).
    let n = sample_detail_noise(sample_pos, h, mip + cloud_detail_mip_offset());
    let profile = saturate(shape_noise - (1.0 - base_profile));

    // Nubis Cubed (2023) up-res of that profile (Guerrilla's VoxelCloudSampler,
    // GetUprezzedVoxelCloudDensity): wispy = lerp(R, G, profile), billowy = lerp(B, A,
    // profile^0.25) * 0.3, composite = lerp(wispy, billowy, detail_type), value erosion of the
    // profile by the composite, density scale^4, sharpened by pow(d, lerp(0.3, 0.6, scale^4)).
    // Nubis's extra near-camera density fade (x0.25 at 10 m, for player visibility) is not
    // applied: density does not depend on the camera.
    let wispy = mix(n.r, n.g, profile);
    let billowy = mix(n.b * 0.3, n.a * 0.3, pow(profile, 0.25));
    var composite = mix(wispy, billowy, detail_type) * cloud_layer.detail_strength;
    // Near-camera high-frequency detail (Nubis Cubed sampler, inHFDetails): the highest-frequency
    // wispy (G) and billowy (A) noise folded twice over 0, blended in within 150 m of the camera.
    // The camera distance follows from the shape LOD, mip = log2(1 + d / 4 km).
    let cam_dist = (exp2(mip) - 1.0) * 4000.0;
    // Scaled by detail_strength like the other detail (0 = no detail erosion, audit 26).
    let hhf = saturate(mix(1.0 - pow(abs(abs(n.g * 2.0 - 1.0) * 2.0 - 1.0), 4.0),
        pow(abs(abs(n.a * 2.0 - 1.0) * 2.0 - 1.0), 2.0), detail_type)) * cloud_layer.detail_strength;
    let hf_blend = saturate((cam_dist - 50.0) / 100.0);
    composite = mix(hhf, composite, mix(0.9, 1.0, hf_blend));
    var dens = saturate((profile - composite) / max(1.0 - composite, 1e-4));
    let powered_scale = pow(density_scale, 4.0);
    dens *= powered_scale;
    dens = pow(dens, mix(0.3, 0.6, max(1e-4, powered_scale)));
    // In-cloud liquid water with height (peak 1): the coarse bound carries the same factor.
    return near_camera_reshape(dens, mip) * vertical.y;
}

/// Nubis Cubed near-camera density reshape (within 150 m of the camera, from the shape LOD
/// mip = log2(1 + d / 4 km)): pow(d, 0.5) * 0.666 at 50 m, identity beyond 150 m. Increasing in
/// `d`, so applying it to an upper bound keeps it an upper bound (the march's coarse test).
fn near_camera_reshape(dens: f32, mip: f32) -> f32 {
    let cam_dist = (exp2(mip) - 1.0) * 4000.0;
    let hf_blend = saturate((cam_dist - 50.0) / 100.0);
    return pow(dens, mix(0.5, 1.0, hf_blend)) * mix(0.666, 1.0, hf_blend);
}

fn sample_cumulus_shape(r: f32, world_pos: vec3<f32>) -> f32 {
    return sample_cumulus_shape_lod(r, world_pos, 0.0);
}

/// Cloud shape / coverage term in [0, 1].
/// This is *not* a physical density by itself: it is the normalized field used to
/// shape the cloud volume (noise + height falloff).
fn get_cloud_coverage(r: f32, world_pos: vec3<f32>) -> f32 {
    // Check if we're within the cloud layer
    if (r < cloud_layer.cloud_layer_start || r > cloud_layer.cloud_layer_end) {
        return 0.0;
    }

    if (CLOUD_USE_NOISE_SHAPE) {
        return sample_cumulus_shape(r, world_pos);
    }

    // Debug sphere volume path
    if (CLOUD_DEBUG_SINGLE_SPHERE) {
        let c = debug_sphere_center();
        let sdf = length(world_pos - c) - RADIUS_M;
        let density = 1.0 - smoothstep(-SOFTNESS_M, SOFTNESS_M, sdf);
        return clamp(density, 0.0, 1.0);
    }

    // Repeating spheres in XZ only.
    let layer_thickness = cloud_layer.cloud_layer_end - cloud_layer.cloud_layer_start;
    let height_in_layer = r - cloud_layer.cloud_layer_start;
    let center_y = 0.5 * layer_thickness;
    let y_rel = height_in_layer - center_y;
    let cell_xz = fract(world_pos.xz / CELL_SIZE_M) - vec2(0.5);
    let q = vec3(cell_xz * CELL_SIZE_M, y_rel);
    let sdf = length(q) - RADIUS_M;
    let sphere_density = 1.0 - smoothstep(-SOFTNESS_M, SOFTNESS_M, sdf);
    return clamp(sphere_density, 0.0, 1.0);
}

/// Returns (density, grad_mag) for the current debug cloud field.
/// - density is the normalized coverage term in [0,1]
/// - grad_mag is an estimate of |∇density| in 1/m, useful for adaptive stepping.
fn sample_cloud_field_density_and_grad(r: f32, world_pos: vec3<f32>) -> vec2<f32> {
    // Outside cloud layer => empty and flat field.
    if (r < cloud_layer.cloud_layer_start || r > cloud_layer.cloud_layer_end) {
        return vec2(0.0, 0.0);
    }

    // Noise shape: skip expensive finite-diff gradient unless debug flag is on (roadmap 2e).
    if (CLOUD_USE_NOISE_SHAPE) {
        let d0 = sample_cumulus_shape(r, world_pos);
        if (!CLOUD_DEBUG_GRADIENT_SUBSTEPS) {
            return vec2(d0, 0.0);
        }
        const EPS_M: f32 = 500.0;
        let up = safe_normalize(world_pos);
        let east = safe_normalize(vec3(-up.z, 0.0, up.x));
        let north = safe_normalize(cross(up, east));
        let dx = sample_cumulus_shape(r, world_pos + east * EPS_M);
        let dz = sample_cumulus_shape(r, world_pos + north * EPS_M);
        let ddx = abs(dx - d0) / EPS_M;
        let ddz = abs(dz - d0) / EPS_M;
        let grad_mag = sqrt(ddx * ddx + ddz * ddz);
        return vec2(d0, grad_mag);
    }

    // Debug sphere field: analytic gradient proxy from SDF smoothstep.
    var sdf: f32;
    if (CLOUD_DEBUG_SINGLE_SPHERE) {
        let c = debug_sphere_center();
        sdf = length(world_pos - c) - RADIUS_M;
    } else {
        let layer_thickness = cloud_layer.cloud_layer_end - cloud_layer.cloud_layer_start;
        let height_in_layer = r - cloud_layer.cloud_layer_start;
        let center_y = 0.5 * layer_thickness;
        let y_rel = height_in_layer - center_y;
        let cell_xz = fract(world_pos.xz / CELL_SIZE_M) - vec2(0.5);
        let q = vec3(cell_xz * CELL_SIZE_M, y_rel);
        sdf = length(q) - RADIUS_M;
    }

    let edge0 = -SOFTNESS_M;
    let edge1 = SOFTNESS_M;
    let denom = (edge1 - edge0);
    let tt = clamp((sdf - edge0) / denom, 0.0, 1.0);
    let smoothen = tt * tt * (3.0 - 2.0 * tt);
    let sphere_density = 1.0 - smoothen;
    let d_sphere_density_dsdf = 6.0 * tt * (1.0 - tt) / denom;
    let density = clamp(sphere_density, 0.0, 1.0);
    let grad_mag = d_sphere_density_dsdf;
    return vec2(density, grad_mag);
}

/// Returns (start_t, end_t, valid) for the segment of the ray inside the cloud layer shell.
/// - start_t/end_t are distances along the ray direction (meters), relative to ray_origin.
/// - valid is 1.0 if the segment exists, 0.0 otherwise.
fn cloud_layer_segment(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec3<f32> {
    // IMPORTANT: do the ray/sphere intersection math in kilometers to improve numerical stability.
    // In meters, `r` is ~6.3e6 and for grazing angles the discriminant can become ill-conditioned,
    // producing "banded" invalid intersections as the sun elevation changes.
    // Doing the math in km keeps magnitudes ~6.3e3 and greatly reduces cancellation.
    const M_TO_KM: f32 = 0.001;
    const KM_TO_M: f32 = 1000.0;

    let r_km = length(ray_origin) * M_TO_KM;
    let up = normalize(ray_origin);
    let mu = dot(ray_dir, up);

    let bottom_radius_km = cloud_layer.cloud_layer_start * M_TO_KM;
    let top_radius_km = cloud_layer.cloud_layer_end * M_TO_KM;

    // Unreal-style intersection selection:
    // Compute intersections with the top and bottom spheres, then derive a single [TMin, TMax]
    // interval from carefully selected roots. This avoids sign/branch issues at grazing angles.
    let t_top = ray_sphere_intersect(r_km, mu, top_radius_km);
    if (t_top.x < 0.0 && t_top.y < 0.0) {
        return vec3(0.0, 0.0, 0.0);
    }
    let t_bottom = ray_sphere_intersect(r_km, mu, bottom_radius_km);

    var t_min = t_top.x;
    var t_max = t_top.y;

    // If we also intersect the bottom sphere, combine both.
    if (!(t_bottom.x < 0.0 && t_bottom.y < 0.0)) {
        // If we see both intersections in front of us, keep the min/closest, otherwise the max/furthest.
        var temp_top = select(max(t_top.x, t_top.y), min(t_top.x, t_top.y), (t_top.x > 0.0) && (t_top.y > 0.0));
        var temp_bottom = select(max(t_bottom.x, t_bottom.y), min(t_bottom.x, t_bottom.y), (t_bottom.x > 0.0) && (t_bottom.y > 0.0));

        if ((t_bottom.x > 0.0) && (t_bottom.y > 0.0)) {
            // If we can see the bottom of the layer, make sure we use the camera (0) or the closest top intersection.
            temp_top = max(0.0, min(t_top.x, t_top.y));
        }

        t_min = min(temp_bottom, temp_top);
        t_max = max(temp_bottom, temp_top);
    }

    t_min = max(0.0, t_min);
    t_max = max(0.0, t_max);
    let valid = (t_max > t_min);

    // Convert km back to meters for callers.
    return vec3(t_min * KM_TO_M, t_max * KM_TO_M, select(0.0, 1.0, valid));
}

/// Cloud *medium density* used for extinction / scattering integration.
/// This is the normalized coverage term scaled by `cloud_layer.cloud_density`.
fn get_cloud_medium_density(r: f32, world_pos: vec3<f32>) -> f32 {
    return get_cloud_coverage(r, world_pos) * cloud_layer.cloud_density;
}

fn get_cloud_medium_density_coarse(r: f32, world_pos: vec3<f32>) -> f32 {
    if (r < cloud_layer.cloud_layer_start || r > cloud_layer.cloud_layer_end) {
        return 0.0;
    }
    if (!CLOUD_USE_NOISE_SHAPE) {
        return get_cloud_medium_density(r, world_pos);
    }
    return sample_cumulus_shape_coarse(r, world_pos) * cloud_layer.cloud_density;
}

fn get_cloud_medium_density_lod(r: f32, world_pos: vec3<f32>, mip: f32) -> f32 {
    if (r < cloud_layer.cloud_layer_start || r > cloud_layer.cloud_layer_end) {
        return 0.0;
    }
    if (!CLOUD_USE_NOISE_SHAPE) {
        return get_cloud_medium_density(r, world_pos);
    }
    return sample_cumulus_shape_lod(r, world_pos, mip) * cloud_layer.cloud_density;
}

fn cloud_mip_from_distance(dist_m: f32) -> f32 {
    // Nubis3: mip = log2(1 + dist · k). k chosen so ~mip 0 near, ~mip 4 at ~16 km.
    @if(ATM_DBG_MIP0) { return 0.0; }
    return log2(1.0 + dist_m * (1.0 / 4000.0));
}

/// Raymarch through clouds towards the sun to compute volumetric shadow
/// Returns the light transmittance factor [0,1] where 0 = fully shadowed, 1 = no shadow
/// Properly handles viewer inside clouds and grazing angles
fn compute_cloud_shadow(
    world_pos: vec3<f32>,
    sun_dir: vec3<f32>,
    steps: u32,
    pixel_coords: vec2<f32>,
) -> f32 {
    // Early exit if clouds are disabled
    if (cloud_layer.cloud_density <= 0.0) {
        return 1.0;
    }

    // March bounds: only integrate within the cloud layer shell along the sun ray.
    let seg = cloud_layer_segment(world_pos, sun_dir);
    if (seg.z < 0.5) {
        return 1.0;
    }

    var march_start = max(seg.x, 0.0);
    var march_end = seg.y;
    if (march_start >= march_end || march_end <= 0.0) {
        return 1.0;
    }

    // Earth (planet) shadow term:
    // If the sun ray hits the planet surface before it exits the cloud layer,
    // the sun is fully occluded and there is no direct lighting.
    let r0 = length(world_pos);
    let up0 = normalize(world_pos);
    let mu0 = dot(sun_dir, up0);
    let ground_i = ray_sphere_intersect(r0, mu0, atmosphere.inner_radius);
    // `ground_i.x` is the nearest positive intersection along the ray, if any.
    if (ground_i.x > 0.0 && ground_i.x < march_end) {
        return 0.0;
    }

    let march_distance = march_end - march_start;

    // Adaptive step count: long segments need more samples, otherwise we can miss small dense regions
    // (especially with the repeating-sphere debug density field).
    const TARGET_STEP_M: f32 = 1500.0;
    const MAX_SHADOW_STEPS: f32 = 16.0;
    let desired = clamp(ceil(march_distance / TARGET_STEP_M), f32(steps), MAX_SHADOW_STEPS);
    let shadow_steps = max(1u, u32(desired));

    let step_size = march_distance / f32(shadow_steps);
    var optical_depth = 0.0;

    // Raymarch through clouds towards sun with per-step jitter.
    for (var i = 0u; i < shadow_steps; i++) {
        let j = interleaved_gradient_noise(pixel_coords, 100u + i); // [0,1]
        let t = march_start + (f32(i) + j) * step_size;
        let sample_pos = world_pos + sun_dir * t;
        let sample_r = length(sample_pos);

        let density = get_cloud_medium_density(sample_r, sample_pos);
        if (density > 0.0) {
            let extinction = density * (cloud_layer.cloud_scattering + cloud_layer.cloud_absorption);
            optical_depth += extinction * step_size;
            // Early out when essentially fully shadowed
            if (optical_depth > 8.0) {
                return 0.0;
            }
        }
    }

    return exp(-optical_depth);
}

/// Simplified cloud contribution for a single sample point
/// Returns (luminance_added, transmittance_multiplier)
fn sample_cloud_contribution(
    world_pos: vec3<f32>,
    step_size: f32,
) -> vec2<f32> {
    let r = length(world_pos);
    let density = get_cloud_medium_density(r, world_pos);
    
    if (density < 0.01) {
        return vec2(0.0, 1.0);
    }
    
    // Physically correct coefficients (units: m^-1 per unit density)
    let extinction = density * (cloud_layer.cloud_scattering + cloud_layer.cloud_absorption);
    let scattering = density * cloud_layer.cloud_scattering;
    
    // Beer's law
    let transmittance = exp(-extinction * step_size);
    
    // Simple uniform scattering (could be enhanced with actual sun direction)
    let in_scatter = scattering * (1.0 - transmittance);
    
    return vec2(in_scatter, transmittance);
}

// ---------------------------------------------------------------------------------------------
// Coarse cloud signed-distance field (Nubis Cubed, Advances 2023: "SDF", 512x512x64 voxels,
// -256 m .. 4096 m; GetVoxelCloudDistance feeds the secondary-scattering remap and SDF ray
// marching). Here it is generated each frame from the coarse (weather + height profile + shape)
// density over a camera-centred grid (`cloud_field.wesl`): x/z horizontal in atmosphere space,
// z slices span the cloud layer radially.
const CLOUD_FIELD_RES_XY: u32 = 128u;
const CLOUD_FIELD_RES_Z: u32 = 32u;
const CLOUD_FIELD_EXTENT_M: f32 = 24000.0;

fn cloud_field_voxel_m() -> f32 {
    return CLOUD_FIELD_EXTENT_M / f32(CLOUD_FIELD_RES_XY);
}

/// Horizontal position in the cloud frame: atmosphere-space x/z minus the wind advection. Every
/// density input is sampled at `p - noise_offset` (`cloud_sample_pos`; the shear lean too), so
/// the clouds are fixed in this frame up to the rising noise and the life cycle. The field is
/// indexed in it: a field built some frames ago (the rebuild is spread over 8 frames) still lies
/// on the clouds that moved since. Indexed in atmosphere space, the field fell behind the clouds
/// by the advection of up to 8 frames and jumped back at each rebuild: its SDF, escape and
/// diffusion terms flashed every 8 frames in time-lapse (`CloudTimeScale` 60: ~1 s of wind per
/// frame, a voxel per rebuild at 180x).
fn cloud_field_frame_xz(p: vec3<f32>) -> vec2<f32> {
    return p.xz - cloud_layer.noise_offset.xz;
}

/// A cloud-frame horizontal position back in atmosphere space (this frame's advection).
fn cloud_field_frame_to_atmosphere_xz(xz: vec2<f32>) -> vec2<f32> {
    return xz + cloud_layer.noise_offset.xz;
}

/// Grid origin (min corner, cloud-frame x/z), snapped to whole voxels so it does not swim.
/// Field cell of the camera: integer voxel index (stored in the field so consumers use the
/// origin the field was actually built with; the build is spread over several frames).
fn cloud_field_cell(cam: vec3<f32>) -> vec2<f32> {
    // Whole escape-volume blocks (cloud_field.wesl ESC_XY_DOWNSAMPLE = 2 voxels), so both the
    // field voxels and the half-resolution escape voxels stay fixed in the world as the grid
    // follows the camera. An odd cell shifted the escape lattice by half a block, and every
    // interpolated escape probability (multiple scattering) jumped when the camera crossed a
    // voxel boundary.
    // Rounded to half-float precision: the cell travels to the field's consumers in RGBA16F
    // metadata, so the field is built at exactly the origin they reconstruct (audit 21: past
    // cell 2048 the stored cell rounded and the consumers read the field up to 750 m (at
    // 1,500 km) off the place it was built, beyond the SDF skip margin). Rounded cells stay even
    // (half-float spacing is >= 2 there). The grid then follows the camera in coarser steps far
    // from the origin (6 km at Earth-radius coordinates, within the 12 km half extent).
    let cell = 2.0 * floor(cloud_field_frame_xz(cam) / (2.0 * cloud_field_voxel_m()));
    return unpack2x16float(pack2x16float(cell));
}

fn cloud_field_origin_from_cell(cell: vec2<f32>) -> vec2<f32> {
    return cell * cloud_field_voxel_m() - 0.5 * CLOUD_FIELD_EXTENT_M;
}

fn cloud_field_origin(cam: vec3<f32>) -> vec2<f32> {
    return cloud_field_origin_from_cell(cloud_field_cell(cam));
}

/// Normalised field coordinate of an atmosphere-space point (through the cloud frame); outside
/// [0,1]^3 = not covered. `cell` is the stored build cell (`cloud_field_sdf.ba`).
fn cloud_field_uvw(p: vec3<f32>, cell: vec2<f32>) -> vec3<f32> {
    let uv = (cloud_field_frame_xz(p) - cloud_field_origin_from_cell(cell)) / CLOUD_FIELD_EXTENT_M;
    let w = (length(p) - cloud_layer.cloud_layer_start)
        / max(cloud_layer.cloud_layer_end - cloud_layer.cloud_layer_start, 1.0);
    return vec3(uv, w);
}

// Cirrus generating field (fbm_noise_3d.wesl `cirrus`): unit Gaussian, -5/3 spectrum, tile =
// outer scale, mips = band-limited fields.
@group(0) @binding(49) var cirrus_field: texture_2d<f32>;
/// Depth slices for rendering (the shadow map uses a quarter: its texel footprint smooths the streaks).
const CIRRUS_SLICES: u32 = 8u;

/// Generating field at `q` (m, in the layer's moving frame) smoothed to `width` (m), on a tile of
/// `outer_scale` (m): trilinear over the band-limited mips (the noise sampler selects mips nearest).
/// Layers share the field, decorrelated by a per-layer offset of a fraction of the tile.
fn cirrus_field_at(q: vec2<f32>, width: f32, outer_scale: f32, layer: u32) -> f32 {
    let texel = outer_scale / f32(textureDimensions(cirrus_field).x);
    let lod = clamp(log2(max(width, texel) / texel), 0.0, f32(textureNumLevels(cirrus_field) - 1u));
    let uv = q / outer_scale + vec2(0.371, 0.613) * f32(layer);
    let a = textureSampleLevel(cirrus_field, noise_sampler_3d, uv, floor(lod)).r;
    let b = textureSampleLevel(cirrus_field, noise_sampler_3d, uv, ceil(lod)).r;
    return mix(a, b, fract(lod));
}

/// Cirrus optical depth along the view ray through the layer (Hogan & Kew 2005, QJRMS 131):
/// ice at depth dz below the generating level (cloud top) came from the generating cell
/// displaced by the fall streak x(dz) = -S dz^2 / (2 w) (their eq. 2 with constant shear S and fall
/// speed w; Marshall 1953), so each depth slice samples the generating field g at p - x. The
/// fall-speed spread spreads that displacement, smoothing deeper slices (the observed spectral
/// steepening below the top). ln IWC = sigma g (log-normal), clear where g < the coverage
/// threshold; normalised to an in-cloud mean of 1 per level, so cloudy columns average the layer's
/// optical depth at normal incidence. Slices are taken where the view ray crosses each depth
/// (parallax across the layer's thickness); the result is the slant optical depth over mu.
fn cirrus_slant_optical_depth(layer: u32, origin: vec3<f32>, dir: vec3<f32>, t_mid: f32,
    up_comp: f32, footprint_m: f32, slices: u32, jitter: f32) -> f32 {
    let thin = cloud_layer.thin[layer];
    let h = thin.thickness;
    let w = thin.fall_speed;
    let shear = thin.wind_shear.xz;
    var sum = 0.0;
    for (var k = 0u; k < slices; k++) {
        // Stratified, jittered depths: fixed slice depths drew each feature as discrete shifted
        // copies (fall-streak displacement + parallax) at slant views; the jitter (per pixel and
        // frame) lets the temporal reconstruction integrate the depth continuously.
        let dz = h * (f32(k) + jitter) / f32(slices);
        let t_k = t_mid + (0.5 * h - dz) / up_comp;
        let p_k = origin + dir * t_k;
        let x = -shear * (dz * dz / (2.0 * w));
        let q = p_k.xz - thin.offset.xz - x;
        let spread = length(x) * thin.fall_speed_spread;
        let g = cirrus_field_at(q, max(footprint_m, spread), thin.outer_scale, layer);
        // Cloud boundary (Hogan & Kew's IWC threshold), antialiased: a hard step drew the
        // overlapping slices' edges as angular facets.
        let cloudy = smoothstep(thin.threshold - 0.15, thin.threshold + 0.15, g);
        sum += cloudy * thin.norm * exp(thin.ln_iwc_std * g);
    }
    return thin.optical_depth * sum / f32(slices) / abs(up_comp);
}
