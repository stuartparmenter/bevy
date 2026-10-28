// Dedicated low-resolution volumetric cloud raymarch (design.md §1.2, roadmap step 1b).
//
// Traces 1 of every 2x2 half-res pixels per frame (checkerboard order from `view.frame_count`,
// RDR2 slides 51-55); `clouds_reconstruct.wesl` fills the rest from reprojected history.
// Runs at a fraction of the view resolution (half res by default) as its own compute pass, so the
// cloud cost is decoupled from `render_sky` and from the atmosphere `AtmosphereMode`.
// Outputs:
// - `out_clouds` RGBA16F: rgb = cloud in-scattered luminance (pre-exposed, *without* aerial
//   perspective), a = cloud transmittance (wavelength independent).
// - `out_clouds_depth` R32F: transmittance-weighted cloud distance in meters (FB16 §5.9.1, eq. 21),
//   0 when the ray sees no cloud. `render_sky` applies aerial perspective once at this distance.

import bevy_render::maths::{PI, ray_sphere_intersect};
import package::render::utils::interleaved_gradient_noise;
import package::atmosphere::{
    bindings::{atmosphere, settings, view, lights, stbn_texture, atmosphere_lut_sampler, cloud_field_sdf, cloud_shadow_map, medium_density_lut},
    bruneton_functions::{ray_intersects_ground, sample_transmittance_lut_segment, sample_transmittance_lut_segment_tau_red, distance_to_top_atmosphere_boundary},
    functions::{
        uv_to_ray_direction, uv_to_ndc, ndc_to_camera_dist, get_view_position,
        sample_transmittance_lut, sample_multiscattering_lut, sample_sky_view_lut, direction_world_to_atmosphere, sample_cloud_shadow_map, cloud_shadow_frame, cloud_shadow_ground_uv, cloud_shadow_optical_depth, cloud_far_transmittance,
        hg_phase, cloud_mie_phase, CLOUD_MIE_MEAN_COS, CLOUD_MIE_G_HG, CLOUD_MIE_G_D, CLOUD_MIE_ALPHA, CLOUD_MIE_W_D, sample_aerial_view_lut, raymarch_atmosphere, sample_density_lut, sample_scattering_lut,
        ABSORPTION_DENSITY, SCATTERING_DENSITY,
    },
    clouds::{
        cloud_layer_segment, get_cloud_medium_density_coarse, near_camera_reshape,
        get_cloud_medium_density_lod, cloud_mip_from_distance, get_cloud_scattering_coeff,
        get_cloud_absorption_coeff, cloud_density_context, cloud_layer, cirrus_slant_optical_depth, CIRRUS_SLICES,
        cloud_field_uvw, cloud_field_voxel_m, CLOUD_FIELD_EXTENT_M,
        cloud_profile_lut, CLOUD_PROFILE_TYPE_ROWS, CLOUD_SCHEME_ROW,
    },
};

@if(MULTISAMPLED_DEPTH)
@group(0) @binding(13) var depth_texture: texture_depth_multisampled_2d;
@else
@group(0) @binding(13) var depth_texture: texture_depth_2d;
@group(0) @binding(26) var out_clouds: texture_storage_2d<rgba16float, write>;
@group(0) @binding(27) var out_clouds_depth: texture_storage_2d<r32float, write>;

const CLOUD_DIST_UNKNOWN: f32 = 1e9;
// SDF empty-space skip safety margin in voxels: the windowed EDT is exact only to ~1 voxel and
// trilinear filtering of a distance field under-/over-shoots by up to ~1 voxel (Hart 1996 sphere
// tracing needs a conservative bound), plus 0.5 voxel for the coarse-density seed threshold.
const CLOUD_SDF_SKIP_MARGIN_VOXELS: f32 = 2.5;
// Escape probability of multiply scattered light (r) and the vertical optical depths to the
// layer top (g) and bottom (b) (cloud_field.wesl `escape`; cell from cloud_field_sdf).
@group(0) @binding(48) var cloud_field_esc: texture_3d<f32>;
// Diffuse sunlight of the cloud field per unit sun irradiance, U = 4 pi J_diff / E (r), and its
// build cell (gb) (cloud_field.wesl diffusion solve).
@group(0) @binding(59) var cloud_field_diff: texture_3d<f32>;

/// The cloud field's diffusion solution at p (0 outside the grid).
fn cloud_field_diffuse(p: vec3<f32>) -> f32 {
    let uvw = cloud_field_uvw(p, textureLoad(cloud_field_diff, vec3(0), 0).gb);
    if (any(uvw.xy < vec2(0.0)) || any(uvw.xy > vec2(1.0))) {
        return 0.0;
    }
    return textureSampleLevel(cloud_field_diff, atmosphere_lut_sampler, saturate(uvw), 0.0).r;
}
/// Second exponential integral E2(x) = int_1^inf exp(-x t) / t^2 dt (the hemispheric escape of a
/// slab of optical depth x). Rational fit exp(-x) (x^2 + a x + b) / (x^3 + c x^2 + d x + b),
/// E2(0) = 1, max relative error 0.64 % on [0, 40] (least squares against scipy.special.expn).
fn exp_integral_e2(x: f32) -> f32 {
    return exp(-x) * (x * x + 1.6706291 * x + 0.13822509)
        / (x * x * x + 3.6251676 * x * x + 2.2005648 * x + 0.13822509);
}

/// (P_esc, tau_up, tau_down). Outside the grid: the plane-parallel P_esc from the coarse march
/// toward the layer top and bottom (cloud_far_transmittance).
fn cloud_escape_and_column(p: vec3<f32>) -> vec3<f32> {
    let uvw = cloud_field_uvw(p, textureLoad(cloud_field_sdf, vec3(0), 0).ba);
    var outside = any(uvw.xy < vec2(0.0)) || any(uvw.xy > vec2(1.0));
    @if(ATM_DBG_ESC_FALLBACK) { outside = true; }
    if (outside) {
        // Plane-parallel limit of the same escape integral (1/4pi) int exp(-tau'(w)) dw over the
        // two hemispheres, from the vertical columns: P_esc = (E2(tau'_up) + E2(tau'_down)) / 2,
        // tau' = (1 - <cos>) tau (similarity). The former constant (a slab surface's 0.5)
        // under-lit every cloud beyond the field grid (seen from far away or from space).
        let n = normalize(p);
        let tau_up = -log(max(cloud_far_transmittance(p, n, true), 1e-6));
        let tau_down = -log(max(cloud_far_transmittance(p, -n, true), 1e-6));
        let k = 1.0 - CLOUD_MIE_MEAN_COS;
        let p_esc = 0.5 * (exp_integral_e2(k * tau_up) + exp_integral_e2(k * tau_down));
        return vec3(p_esc, tau_up, tau_down);
    }
    return textureSampleLevel(cloud_field_esc, atmosphere_lut_sampler, saturate(uvw), 0.0).rgb;
}
/// Nubis Cubed GetVoxelCloudDistance: signed distance (m, negative inside) from the coarse SDF
/// (cloud_field.wesl). Outside the grid: CLOUD_DIST_UNKNOWN. Positive distances are capped by
/// the distance to the grid's side so clouds beyond the grid are never skipped over.
fn cloud_distance(p: vec3<f32>, cam: vec3<f32>) -> f32 {
    let uvw = cloud_field_uvw(p, textureLoad(cloud_field_sdf, vec3(0), 0).ba);
    if (any(uvw.xy < vec2(0.0)) || any(uvw.xy > vec2(1.0))) {
        return CLOUD_DIST_UNKNOWN;
    }
    let d = textureSampleLevel(cloud_field_sdf, atmosphere_lut_sampler, saturate(uvw), 0.0).r;
    let border = min(min(uvw.x, 1.0 - uvw.x), min(uvw.y, 1.0 - uvw.y)) * CLOUD_FIELD_EXTENT_M;
    return select(d, min(d, border), d > 0.0);
}

// Hard caps for the first version of the pass; these become `VolumetricCloudSettings` (design §3.3).
// Step budgets: `settings.cloud_max_steps` (VolumetricCloudSettings::max_steps, 96 at MEDIUM)
// per ray, and this many times more for rays inside the layer (1024 at MEDIUM).
const CLOUD_IN_LAYER_STEP_RATIO: f32 = 1024.0 / 96.0;
// Distance-adaptive step (Nubis3): near + far_offset · dist / 16384.
const CLOUD_STEP_NEAR_M: f32 = 40.0;
const CLOUD_STEP_FAR_OFFSET_M: f32 = 220.0;
const CLOUD_STEP_FAR_REF_M: f32 = 16384.0;
// Near the camera the step is a fixed fraction of the distance (like exponential froxel slices):
// a constant 40 m step at 100 m puts a cloud into a few translucent slices whose spacing is a
// large fraction of the view distance, and they slide visibly as the camera moves (crawl).
// Floor: the detail-noise texel (the density has no finer structure to sample).
const CLOUD_STEP_REL: f32 = 0.1;

/// View-ray step length (m) at distance `t`.
fn cloud_view_step(t: f32) -> f32 {
    let far = CLOUD_STEP_NEAR_M + CLOUD_STEP_FAR_OFFSET_M * clamp(t / CLOUD_STEP_FAR_REF_M, 0.0, 8.0);
    let floor_m = max(cloud_layer.detail_noise_scale, 1.0) / 128.0;
    var step = min(far, max(CLOUD_STEP_REL * t, floor_m));
    // Step-size convergence (audit 24 validation).
    @if(ATM_DBG_STEP_DIV2) { step *= 0.5; }
    @if(ATM_DBG_STEP_DIV4) { step *= 0.25; }
    @if(ATM_DBG_STEP_DIV16) { step *= 0.0625; }
    return step;
}
// In-cloud view steps longer than this take `settings.cloud_step_samples` density samples
// (VolumetricCloudSettings::step_samples). Shorter steps are near the camera, where the
// density changes little within one.
const CLOUD_STEP_SUB_MIN_M: f32 = 100.0;
const CLOUD_FINE_EMPTY_TO_COARSE: u32 = 10u;
const CLOUD_COARSE_DENSITY_EPS: f32 = 0.0005;
// Grazing rays enter a 1.5-4 km shell ~140-230 km away, so keep the horizon covered.
const CLOUD_MIN_TRANSMITTANCE: f32 = 0.01;

// Short light taps toward the sun for near-surface self-shadowing detail (Nubis3 2:09:52: "the
// first two samples are taken as before"; the cloud shadow map covers the long range).
// 12 taps growing x1.4 from 20 m (~2.8 km) with a gentler LOD ramp: the old 6 doubling taps
// (1.26 km) under-estimated sun optical depth (single scatter +25-40% vs CLOUD_REFERENCE).
const LIGHT_TAP_STEP_M: f32 = 20.0;
const LIGHT_TAP_GROWTH: f32 = 1.4;
const LIGHT_TAP_LOD: f32 = 0.25;
const LIGHT_LONG_TAP_M: f32 = 1500.0;

// Order in which the 4 pixels of each 2x2 half-res block are traced (shared with reconstruct).
const CHECKER_OFFSETS: array<vec2<u32>, 4> = array(
    vec2(0u, 0u), vec2(1u, 1u), vec2(1u, 0u), vec2(0u, 1u)
);

fn cloud_phase(cos_theta: f32) -> f32 {
    // Droplet Mie phase (Jendersie & d'Eon 2023 HG + Draine fit, d = 20 um; functions.wesl).
    return cloud_mie_phase(cos_theta);
}

/// Flux-equivalent sky radiance E_sky / pi at the camera radius: the boundary value of the sky
/// side of the ambient two-stream (Marshak boundary condition: incoming hemispheric flux) and the
/// sky part of the Lambertian ground. Cosine-weighted quadrature over the sky-view LUT (atmosphere
/// space): mu = sqrt(1 - u) for stratified u, golden-angle azimuths, so the plain mean of the
/// samples is E / pi.
const CLOUD_SKY_IRR_DIRS: u32 = 16u;
fn sample_sky_irradiance(r: f32) -> vec3<f32> {
    var sum = vec3(0.0);
    for (var i = 0u; i < CLOUD_SKY_IRR_DIRS; i++) {
        let u = (f32(i) + 0.5) / f32(CLOUD_SKY_IRR_DIRS);
        let mu = sqrt(1.0 - u);
        let s = sqrt(u);
        let a = f32(i) * 2.399963;
        sum += sample_sky_view_lut(r, vec3(s * sin(a), mu, -s * cos(a)));
    }
    return sum / f32(CLOUD_SKY_IRR_DIRS);
}

/// Isotropic ambient in-scatter radiance (per unit sigma_s): the mean intensity of the diffuse
/// field driven by the sky above (`sky_rad`, mean sky radiance) and the ground below
/// (`ground_rad`, Lambertian bounce of sun + sky); Nubis Cubed's sky-ward ambient attenuation
/// (talk 135:16) in physical form. A conservative medium leaves a uniform isotropic field
/// unchanged, so the in-cloud gradient comes from the sky/ground contrast: the diffusion
/// (two-stream) solution of a slab in similarity-reduced depth tau' = (1 - g) tau, with the
/// extrapolation length z0 = 2/3,
///   J = L_sky + (L_ground - L_sky) (tau'_up + z0) / (tau'_up + tau'_down + 2 z0).
/// No cloud: (L_sky + L_ground) / 2. vs MC (validation/sky/amb_slab_mc.py): within 3 %.
const CLOUD_DIFFUSION_Z0: f32 = 2.0 / 3.0;
fn cloud_ambient_term(sky_rad: vec3<f32>, ground_rad: vec3<f32>, tau_up: f32, tau_down: f32) -> vec3<f32> {
    let k = 1.0 - CLOUD_MIE_MEAN_COS;
    let w = (k * tau_up + CLOUD_DIFFUSION_Z0) / (k * (tau_up + tau_down) + 2.0 * CLOUD_DIFFUSION_Z0);
    return mix(sky_rad, ground_rad, w);
}

/// Diffuse (multiply scattered) sunlight in a conservative cloud slab lit from above at cosine
/// mu0, per unit sun irradiance: M = 4 pi J_diff / E at optical depth tau_up below the top of a
/// slab of optical depth tau_up + tau_down (the shader's multiple-scattering source is
/// E M / (4 pi) per unit sigma_s). Delta-Eddington (Joseph, Wiscombe & Weinman 1976): forward
/// peak g^2 left in the direct beam, scaled depths tau' = (1 - g^2) tau, scaled asymmetry
/// g' = g / (1 + g); with U = 4 pi J_diff, U'' = -3 E (1 + g') exp(-tau'/mu0) and diffuse flux
/// F = -U'/3 + g' mu0 E exp(-tau'/mu0), so U = A + B tau' + c exp(-tau'/mu0),
/// c = -3 (1 + g') mu0^2, with A, B from Marshak boundaries (no diffuse light entering: U/4 + F/2
/// = 0 at the top, U/4 - F/2 = 0 at the bottom). Against a slab Monte Carlo with the droplet phase
/// function (.clouds-handoff/validation/sky/ms_vertical_slab.py), max(this, the escape series):
/// 1.06-1.23 x the MC (5-95 %: 0.84-1.39) at optical depths 20-60 and sun cosines 0.3-1, where the
/// series alone gives 0.00-1.0 (~0 on the shaded side); thin slabs (2-5) stay under (0.4-0.8).
fn cloud_diffuse_sun(tau_up: f32, tau_down: f32, mu0: f32) -> f32 {
    let g = CLOUD_MIE_MEAN_COS;
    let s = 1.0 - g * g;
    let gs = g / (1.0 + g);
    let tp = max(tau_up, 0.0) * s;
    let t_total = (max(tau_up, 0.0) + max(tau_down, 0.0)) * s;
    let c = -3.0 * (1.0 + gs) * mu0 * mu0;
    let k = c / (3.0 * mu0) + gs * mu0;
    let e_t = exp(-t_total / mu0);
    let r1 = -(0.25 * c + 0.5 * k);
    let r2 = -(0.25 * c - 0.5 * k) * e_t;
    let det = 0.0625 * t_total + 1.0 / 12.0;
    let a = (r1 * (0.25 * t_total + 1.0 / 6.0) + r2 / 6.0) / det;
    let b = 0.25 * (r2 - r1) / det;
    return max(a + b * tp + c * exp(-tp / mu0), 0.0);
}

/// Mean sun visibility of the ground a point at `p` sees below it: the ground radiance enters the
/// ambient as a mean intensity (uniform in solid angle, i.e. uniform in mu = cos(nadir angle)),
/// so it averages the ground over kilometres. Sampling only the point straight below projected
/// the crisp cloud-shadow pattern on the ground back onto the cloud bases, offset by
/// H tan(sun zenith) (it lined up only with the sun overhead). Quadrature: midpoint nodes
/// mu = 1/4, 3/4 x 8 azimuths (staggered); ground point at horizontal distance
/// H sqrt(1 - mu^2) / mu. The ground's sun visibility is a sharp pattern, and the nodes sweep
/// across it as the height changes: with 8 nodes each crossing stepped the average by 1/8 (16 nodes: 1/16),
/// which showed as fixed horizontal bands on the shadowed sides of clouds (ambient-lit).
fn ground_sun_visibility(p: vec3<f32>, l: vec3<f32>) -> f32 {
    let r_g = atmosphere.inner_radius;
    let up = normalize(p);
    let height = max(length(p) - r_g, 0.0);
    let t1 = normalize(cross(up, select(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), abs(up.x) > 0.9)));
    let t2 = cross(up, t1);
    var vis = 0.0;
    var n_off_map = 0.0;
    let frame = cloud_shadow_frame(l);
    for (var i = 0u; i < 2u; i++) {
        let mu = (f32(i) + 0.5) / 2.0;
        let rho = height * sqrt(1.0 - mu * mu) / mu;
        for (var k = 0u; k < 8u; k++) {
            let a = (f32(k) + 0.5 * f32(i)) * 0.78539816;
            let p_g = normalize(up * r_g + (t1 * cos(a) + t2 * sin(a)) * rho) * r_g;
            // Shadow map, optical depth only (the ground is below every cloud), one bilinear tap.
            let uv = cloud_shadow_ground_uv(frame, p_g);
            if (uv.x < 0.0) {
                n_off_map += 1.0;
            } else {
                let d = textureSampleLevel(cloud_shadow_map, atmosphere_lut_sampler, uv, 0.0);
                vis += cloud_shadow_optical_depth(d.rgb) * exp(-d.a);
            }
        }
    }
    // Ground points off the map (clouds near its 64 km edge) share one far-field march from the
    // ground below p. Marching per point inlined a full density loop into these 16 taps, inside
    // the light loop inside the view march: the cloud pass spilled 160 bytes of registers
    // (Metal compiler, Apple M5), 0 without the ambient path.
    if (n_off_map > 0.0) {
        vis += n_off_map * cloud_far_transmittance(up * r_g, l, true);
    }
    return vis / 16.0;
}


// Aerial perspective between the camera and the cloud (at the transmittance-weighted depth d).
// The cloud shadow is baked into the aerial-perspective froxel LUT and the sky-view LUT (Hillaire
// 2020 AP volume; RDR2 / Bauer 2019 froxel fog with cloud shadows; UE5 volumetric cloud shadow on
// the SkyAtmosphere): low-frequency and temporally stable. The former per-pixel shadow-map march
// along the ray was noisy at the horizon and blue-shifted it (purple line).

/// Single scattering by a plane-parallel slab of optical depth `tau` (Liou, An Introduction to
/// Atmospheric Radiation, ch. 3/6): radiance per unit (albedo x phase x normal sun irradiance).
/// Sun and viewer on the same side (reflection): mu0 / (mu0 + mu) (1 - exp(-tau (1/mu0 + 1/mu)));
/// opposite sides (transmission): mu0 / (mu0 - mu) (exp(-tau/mu0) - exp(-tau/mu)).
/// mu0, mu: |cosines| of the sun and view directions to the slab normal. Multiple scattering is
/// neglected: the similarity-reduced depth (1 - g) tau <= 0.08 for thin cirrus.
fn slab_single_scatter(tau: f32, mu0: f32, mu: f32, same_side: bool) -> f32 {
    if (same_side) {
        return mu0 / (mu0 + mu) * (1.0 - exp(-tau * (1.0 / mu0 + 1.0 / mu)));
    }
    if (abs(mu0 - mu) < 1e-3) {
        return tau / mu * exp(-tau / mu);
    }
    return mu0 / (mu0 - mu) * (exp(-tau / mu0) - exp(-tau / mu));
}

/// Composites the cirrus shell into the cloud result (rgb pre-exposed, a = transmittance) and its
/// transmittance-weighted depth. Behind the cumulus when the camera is below the shell, in front
/// when above. Haze in front of the cirrus as for the cumulus: T_ap(0..t) L_ci + (1 - T_ci) L_ap,
/// with L_ap(0..t) ~ the sky-view radiance of the full path (the in-scatter above 9 km is
/// neglected: ~25 % of the Rayleigh depth, weighted by the cirrus opacity 1 - T_ci).
fn composite_cirrus(layer: u32, c: vec4<f32>, depth_io: ptr<function, f32>, origin: vec3<f32>,
    dir: vec3<f32>, r: f32, mu: f32, t_scene: f32, slice_jitter: f32) -> vec4<f32> {
    // The layer's mid-level shell sets the compositing distance; the optical depth integrates
    // the whole thickness along the ray.
    let thin = cloud_layer.thin[layer];
    let r_ci = thin.top - 0.5 * thin.thickness;
    let b = r * mu;
    let disc = b * b - (r * r - r_ci * r_ci);
    if (disc <= 0.0) {
        return c;
    }
    let below = r < r_ci;
    let t_ci = select(-b - sqrt(disc), -b + sqrt(disc), below);
    if (t_ci <= 0.0 || t_ci > t_scene || (below && ray_intersects_ground(r, mu))) {
        return c;
    }
    let p = origin + dir * t_ci;
    let up = p / r_ci;
    let up_comp = dot(dir, up);
    let mu_v = max(abs(up_comp), 1e-3);
    // Pixel footprint on the shell (half-res pixel ~2.2 mrad, stretched by the slant).
    let tau_slant = cirrus_slant_optical_depth(layer, origin, dir, t_ci, select(-mu_v, mu_v, up_comp > 0.0), t_ci * 2.2e-3 / mu_v, CIRRUS_SLICES, slice_jitter);
    if (tau_slant <= 0.0) {
        return c;
    }
    // Normal-incidence optical depth of the column the ray crosses (slab formulas below).
    let tau = tau_slant * mu_v;
    let t_view = exp(-tau_slant);
    var l_ci = vec3(0.0);
    let sky_down = sample_sky_irradiance(r_ci);
    var ground_up = sample_sky_irradiance(atmosphere.inner_radius);
    for (var light_i: u32 = 0u; light_i < lights.n_directional_lights; light_i++) {
        let light = &lights.directional_lights[light_i];
        let l = (*light).direction_to_light;
        let mu0 = dot(l, up);
        let t_atm = sample_transmittance_lut(r_ci, mu0) * f32(!ray_intersects_ground(r_ci, mu0));
        let k = slab_single_scatter(tau, max(abs(mu0), 1e-3), mu_v, (mu0 > 0.0) != below);
        l_ci += (*light).color.rgb * t_atm * hg_phase(dot(l, dir), thin.asymmetry) * k;
        // Sunlit ground below (Lambertian), seen as flux-equivalent radiance from beneath.
        let mu_g = dot(l, normalize(p));
        ground_up += (*light).color.rgb * sample_transmittance_lut(atmosphere.inner_radius, mu_g)
            * max(mu_g, 0.0) / PI;
    }
    // Diffuse sky (above) and ground (below, albedo) light: isotropic mean intensity over both
    // hemispheres; single scattering of diffuse light by the slab ~ (1 - T) J.
    let j = 0.5 * (sky_down + atmosphere.ground_albedo * ground_up);
    l_ci += j * (1.0 - t_view);
    var t_ap = sample_transmittance_lut_segment(r, mu, t_ci);
    t_ap = t_ap * exp(sample_transmittance_lut_segment_tau_red(r, mu, t_ci));
    let lum_ci = (t_ap * l_ci + (1.0 - t_view) * sample_sky_view_lut(r, dir)) * view.exposure;
    // Representative depth weighted by each layer's visible opacity: the layer behind is seen
    // through the one in front (audit 27: above the sheet the cumulus kept its full weight 1 - T_c,
    // so an opaque cirrus foreground still pulled the depth to the cumulus behind it).
    var out: vec4<f32>;
    var w_ci: f32;
    var w_c: f32;
    if (below) {
        out = vec4(c.rgb + c.a * lum_ci, c.a * t_view);
        w_c = 1.0 - c.a;
        w_ci = c.a * (1.0 - t_view);
    } else {
        out = vec4(lum_ci + t_view * c.rgb, c.a * t_view);
        w_ci = 1.0 - t_view;
        w_c = t_view * (1.0 - c.a);
    }
    if (w_c + w_ci > 0.0) {
        *depth_io = (w_c * *depth_io + w_ci * t_ci) / (w_c + w_ci);
    }
    return out;
}

fn cloud_shadowed_haze(uv: vec2<f32>, origin: vec3<f32>, ray_dir: vec3<f32>, d: f32, r: f32, mu: f32, j: f32) -> vec3<f32> {
    // Composited by the caller as L = T_ap * L_cloud + (1 - T_cloud) * L_ap (Hillaire 2020 §5.2):
    // the cloud is never replaced by sky colour. L_ap is the same per-pixel, cloud-shadowed
    // atmosphere raymarch the sky pixels use (RDR2 integrates haze along the cloud ray too), so
    // light shafts are continuous across cloud edges. The 32x32-froxel AP LUT (Hillaire 2020,
    // which assumes no clouds) made shafts in front of clouds blocky and speckled
    // (ATM_DBG_HAZE_AP_LUT restores it). Cost: ~+0.1 ms clouds_raymarch (hero:35).
    @if(ATM_DBG_HAZE_AP_LUT) {
        if (d <= settings.aerial_view_lut_max_distance) {
            return sample_aerial_view_lut(uv, d);
        }
    }
    return raymarch_atmosphere(origin, ray_dir, d, settings.sky_max_samples, uv, false).inscattering;
}

// ---------------------------------------------------------------------------------------------
// Reference path tracer (CLOUD_REFERENCE): ground truth for clouds AND atmosphere.
// One combined participating medium = Bevy's atmosphere (same medium density / scattering LUTs
// that the realtime LUTs are built from: Rayleigh + Mie + ozone absorption, per-altitude, with
// their phase functions) + the cloud layer (same density WESL at mip 0, same sigma_s/sigma_a,
// same dual-lobe HG). Delta tracking for free flights (piecewise majorants: air everywhere,
// + cloud inside the cloud shell), ratio tracking for camera-ray transmittance and sun shadow
// rays through the clouds; air transmittance toward the sun from the transmittance LUT (exact
// integral of the same medium). Sun is the only light: no sky LUT, space is black; ground is a
// Lambertian sphere (albedo atmosphere.ground_albedo) with sun NEE and cosine-sampled bounces.
// Unlimited bounces with Russian roulette. RGB via hero-channel sampling: each traced sample
// handles one channel (cycled per pixel sample), weighted by 3.
// Output: rgb = in-scattered radiance along the camera ray up to the scene depth (sky pixels: the
// whole sky minus the sun disk), a = luminance-weighted camera-ray transmittance.
struct RefOut { c: vec4<f32>, d: f32 }

var<private> ref_rng: u32;
fn ref_hash(v: u32) -> u32 {
    let st = v * 747796405u + 2891336453u;
    let w = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u;
    return (w >> 22u) ^ w;
}
fn ref_rand() -> f32 {
    ref_rng = ref_rng * 747796405u + 2891336453u;
    var w = ((ref_rng >> ((ref_rng >> 28u) + 4u)) ^ ref_rng) * 277803737u;
    w = (w >> 22u) ^ w;
    return f32(w) * (1.0 / 4294967296.0);
}

// Majorants of the null-collision estimators (audit 33): the air's per-channel extinction and an
// upper bound of the cloud density, from the data the tracer samples. Linear interpolation never
// exceeds the largest texel, so the maxima over the texels bound every lookup: the air over the
// medium's density LUT (absorption + scattering rows), the clouds `cloud_density` times the
// factors of the up-res density (`sample_cumulus_shape_lod`): the profile's liquid water
// relative to `liquid_water` and the thermal magnitude (each can exceed 1); the eroded density
// and its near-camera reshape stay <= 1 at any LOD (`ref_cloud_density`). Each workgroup
// reduces them cooperatively before tracing (`ref_compute_bounds`). The previous majorant
// (cloud_density) did not bound the physical profiles, whose density reaches ~3.
@if(CLOUD_REFERENCE)
var<workgroup> ref_wg_bounds: array<vec4<f32>, 64>;
@if(CLOUD_REFERENCE)
var<workgroup> ref_wg_magnitude: array<f32, 64>;
var<private> ref_air_majorant: vec3<f32>;
var<private> ref_cloud_bound: f32;

@if(CLOUD_REFERENCE)
fn ref_compute_bounds(li: u32) {
    var b = vec4(0.0);
    var magnitude = 0.0;
    let n_air = textureDimensions(medium_density_lut);
    for (var i = li; i < n_air.x; i += 64u) {
        var e = vec3(0.0);
        for (var row = 0u; row < n_air.y; row++) {
            e += textureLoad(medium_density_lut, vec2(i, row), 0).xyz;
        }
        b = vec4(max(b.xyz, e), b.w);
    }
    if (cloud_layer.profile_mode == 0u) {
        let n = textureDimensions(cloud_profile_lut);
        for (var i = li; i < n.x * u32(CLOUD_PROFILE_TYPE_ROWS); i += 64u) {
            b.w = max(b.w, textureLoad(cloud_profile_lut, vec2(i % n.x, i / n.x), 0).b);
        }
        for (var i = li; i < n.x; i += 64u) {
            magnitude = max(magnitude, textureLoad(cloud_profile_lut, vec2(i, u32(CLOUD_SCHEME_ROW)), 0).b);
        }
    }
    ref_wg_bounds[li] = b;
    ref_wg_magnitude[li] = magnitude;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (li < stride) {
            ref_wg_bounds[li] = max(ref_wg_bounds[li], ref_wg_bounds[li + stride]);
            ref_wg_magnitude[li] = max(ref_wg_magnitude[li], ref_wg_magnitude[li + stride]);
        }
        workgroupBarrier();
    }
    let bounds = ref_wg_bounds[0];
    ref_air_majorant = max(bounds.xyz, vec3(1e-9));
    var shape = 1.0;
    if (cloud_layer.profile_mode == 0u) {
        shape *= bounds.w / max(cloud_layer.liquid_water, 1e-9) * ref_wg_magnitude[0];
    }
    ref_cloud_bound = max(cloud_layer.cloud_density, 0.0) * shape;
}

// The camera of the traced pixel (atmosphere space), for the cloud LOD.
var<private> ref_camera: vec3<f32>;

// The cloud medium at p as the realtime march defines it for this camera: the density at the
// LOD of p's distance to the camera (Q22). Mip 0 everywhere also applied the near-camera reshape
// and high-frequency detail (keyed to the LOD's inferred camera distance, < 150 m) to every
// cloud, a different medium from the realtime one beyond 150 m. Bounces and sun paths use the
// same field (one medium per camera), so the comparison isolates transport. Like the realtime
// march, the field ends at `cloud_max_distance` from the camera: the reference traced every
// cloud to the horizon, 0.09 more opaque on average in a hero view.
fn ref_cloud_density(p: vec3<f32>) -> f32 {
    @if(CLOUD_REF_NO_CLOUDS) { return 0.0; }
    let d = distance(p, ref_camera);
    if (d > settings.cloud_max_distance) { return 0.0; }
    return get_cloud_medium_density_lod(length(p), p, cloud_mip_from_distance(d));
}

// Cap diagnostics (ATM_DBG_REF_CAPS, Q21): x the bounce cap, y a flight's segment budget, z the
// sun ray's null-collision budget was reached by this sample (each truncates the path), w a
// flight continued past a segment's null-collision budget (exact, no truncation).
var<private> ref_caps: vec4<f32>;
const REF_NULL_CANDIDATES: u32 = 8192u;
const REF_SEGMENTS: u32 = 64u;
const REF_SUN_CANDIDATES: u32 = 65536u;
const REF_MAX_BOUNCES: u32 = 16384u;
// Beyond this many events, each further event also survives with REF_DEEP_SURVIVAL (unbiased
// Russian roulette, weight 1 / q): analog cloud events keep throughput 1, so the throughput
// roulette never ended diffusion paths in thick clouds (~tau^2 events, thousands at tau ~ 50),
// which a hard cap of 1024 truncated (Q21). Termination and variance only, not physics.
const REF_DEEP_EVENTS: u32 = 256u;
const REF_DEEP_SURVIVAL: f32 = 0.99;

// Distance from p0 + d t to the next intersection of the ray p0 + d s with a sphere of radius R
// beyond s = t (inf if none). The roots are always solved from the ray's origin p0, so a
// boundary just crossed is never found again: stepping a little past it from the crossing point
// (5 cm) could land back on the sphere, as a float32 position resolves only ~0.5 m at planetary
// radius, and a ray re-hit the same boundary until its segment budget ran out (and counted as
// escaped: the clear colour showed through the ground from orbit, audit 35/36).
fn ref_sphere_after(p0: vec3<f32>, d: vec3<f32>, radius: f32, t: f32) -> f32 {
    let r = length(p0);
    let b = dot(p0, d);
    let c = (r - radius) * (r + radius);
    let disc = b * b - c;
    if (disc < 0.0) { return 1e30; }
    let s = sqrt(disc);
    if (-b - s > t) { return -b - s - t; }
    if (-b + s > t) { return -b + s - t; }
    return 1e30;
}

fn ref_air(r: f32) -> mat2x3<f32> {
    // [0] = absorption, [1] = scattering (rgb, m^-1)
    return mat2x3<f32>(sample_density_lut(r, ABSORPTION_DENSITY), sample_density_lut(r, SCATTERING_DENSITY));
}

struct RefFlight {
    t: f32,
    // 0 = reached t_max, 1 = air scatter, 2 = cloud scatter, 3 = absorbed, 4 = ground, 5 = space
    event: u32,
    tr: f32, // ratio-tracking transmittance (ratio mode only)
}

// Free flight (delta tracking) or transmittance (ratio tracking, `ratio` = true) along `d` from
// `p` up to `t_max` in channel `ch`, through air + clouds.
fn ref_flight(p0: vec3<f32>, d: vec3<f32>, t_max: f32, ch: u32, air_max: f32, ratio: bool) -> RefFlight {
    var o: RefFlight;
    o.tr = 1.0;
    let sigma_s_c = get_cloud_scattering_coeff();
    let sigma_t_c = sigma_s_c + get_cloud_absorption_coeff();
    let r_cs = cloud_layer.cloud_layer_start;
    let r_ce = cloud_layer.cloud_layer_end;
    var t = 0.0;
    for (var seg = 0u; seg < REF_SEGMENTS; seg++) {
        let p = p0 + d * t;
        // Next boundary: cloud shell spheres, ground, top of atmosphere, t_max.
        var tb = t_max - t;
        var ev = 0u;
        let t_g = ref_sphere_after(p0, d, atmosphere.inner_radius, t);
        if (t_g < tb) { tb = t_g; ev = 4u; }
        let t_top = ref_sphere_after(p0, d, atmosphere.outer_radius, t);
        if (t_top < tb) { tb = t_top; ev = 5u; }
        var t_shell = min(ref_sphere_after(p0, d, r_cs, t), ref_sphere_after(p0, d, r_ce, t));
        if (t_shell < tb) { tb = t_shell; ev = 6u; }
        let pm = p + d * (0.5 * tb);
        let rm = length(pm);
        // The top of the atmosphere at the end of a segment in vacuum is entry from space, crossed
        // like the shell boundaries; from inside it is escape (audit 36).
        if (ev == 5u && rm > atmosphere.outer_radius) { ev = 6u; }
        let in_shell = rm > r_cs && rm < r_ce;
        // Vacuum outside the atmosphere: no collisions (the density LUT clamps to the top value).
        var maj = select(air_max, 0.0, rm > atmosphere.outer_radius);
        @if(!CLOUD_REF_NO_CLOUDS) { if (in_shell) { maj += sigma_t_c * ref_cloud_bound; } }
        var tt = 0.0;
        var reached = maj <= 0.0;
        for (var i = 0u; i < REF_NULL_CANDIDATES && !reached; i++) {
            tt += -log(1.0 - ref_rand()) / maj;
            if (tt >= tb) { reached = true; break; }
            let q = p + d * tt;
            let rq = length(q);
            var air = ref_air(rq);
            let a_air = air[0][ch];
            let s_air = air[1][ch];
            var dens = 0.0;
            if (in_shell) { dens = ref_cloud_density(q); }
            let s_cl = dens * sigma_s_c;
            let t_cl = dens * sigma_t_c;
            if (ratio) {
                o.tr *= max(0.0, 1.0 - (a_air + s_air + t_cl) / maj);
                if (o.tr < 1e-5) { o.t = t + tt; o.event = 3u; return o; }
                continue;
            }
            let u = ref_rand() * maj;
            if (u < s_cl) { o.t = t + tt; o.event = 2u; return o; }
            if (u < t_cl) { o.t = t + tt; o.event = 3u; return o; }
            if (u < t_cl + s_air) { o.t = t + tt; o.event = 1u; return o; }
            if (u < t_cl + s_air + a_air) { o.t = t + tt; o.event = 3u; return o; }
        }
        if (!reached) {
            // Candidate budget used up inside the segment: continue from the last candidate
            // (exact: the candidates are a memoryless Poisson process).
            ref_caps.w = 1.0;
            t += tt;
            continue;
        }
        t += tb;
        if (ev != 6u) { o.t = t; o.event = ev; return o; }
    }
    ref_caps.y = 1.0;
    o.t = t;
    o.event = 0u;
    return o;
}

fn ref_sample_hg(g: f32) -> f32 {
    let u = ref_rand();
    if (abs(g) < 1e-3) { return 1.0 - 2.0 * u; }
    let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
    return (1.0 + g * g - s * s) / (2.0 * g);
}

/// Samples cos(theta) from cloud_mie_phase exactly: the HG lobe with probability 1 - w_D, else the
/// Draine lobe by rejection (Draine = HG(g_D) * (1 + alpha mu^2) / norm: propose HG(g_D), accept
/// with (1 + alpha mu^2) / (1 + alpha)).
fn ref_sample_cloud_phase() -> f32 {
    if (ref_rand() >= CLOUD_MIE_W_D) {
        return ref_sample_hg(CLOUD_MIE_G_HG);
    }
    var mu = ref_sample_hg(CLOUD_MIE_G_D);
    for (var i = 0u; i < 64u; i++) {
        if (ref_rand() * (1.0 + CLOUD_MIE_ALPHA) <= 1.0 + CLOUD_MIE_ALPHA * mu * mu) { break; }
        mu = ref_sample_hg(CLOUD_MIE_G_D);
    }
    return mu;
}

fn ref_dir_around(d: vec3<f32>, cos_t: f32) -> vec3<f32> {
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    let phi = 2.0 * PI * ref_rand();
    let a = select(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), abs(d.x) > 0.9);
    let x = normalize(cross(a, d));
    let y = cross(d, x);
    return normalize(x * (cos(phi) * sin_t) + y * (sin(phi) * sin_t) + d * cos_t);
}

const REF_AIR_HG_G: f32 = 0.8;

// Sun visibility from p (channel ch): air via transmittance LUT, clouds via ratio tracking.
fn ref_sun_vis(p: vec3<f32>, l: vec3<f32>, ch: u32) -> f32 {
    let r = length(p);
    let mu = dot(l, p / r);
    if (ray_intersects_ground(r, mu)) { return 0.0; }
    var v = sample_transmittance_lut(r, mu)[ch];
    @if(CLOUD_REF_QUAD_T) {
    // Direct quadrature of the density LUT (bypasses the transmittance LUT).
    let t_top = distance_to_top_atmosphere_boundary(atmosphere, r, mu);
    var od = 0.0;
    let n = 256u;
    for (var i = 0u; i < n; i++) {
        let u0 = f32(i) / f32(n); let u1 = f32(i + 1u) / f32(n);
        let ta = t_top * u0 * u0; let tb = t_top * u1 * u1;
        let q = p + l * (0.5 * (ta + tb));
        let m = ref_air(length(q));
        od += (m[0][ch] + m[1][ch]) * (tb - ta);
    }
    v = exp(-od);
    }
    @if(!CLOUD_REF_NO_CLOUDS) {
    let seg = cloud_layer_segment(p, l);
    if (seg.z > 0.5 && v > 0.0 && ref_cloud_bound > 0.0) {
        // Ratio tracking with the cloud majorant sigma_t * ref_cloud_bound.
        let sigma_t_c = get_cloud_scattering_coeff() + get_cloud_absorption_coeff();
        var t = max(seg.x, 0.0);
        let t_end = min(seg.y, settings.cloud_max_distance);
        var done = false;
        for (var i = 0u; i < REF_SUN_CANDIDATES; i++) {
            t += -log(1.0 - ref_rand()) / (sigma_t_c * ref_cloud_bound);
            if (t >= t_end || v < 1e-6) { done = true; break; }
            v *= 1.0 - ref_cloud_density(p + l * t) / ref_cloud_bound;
        }
        if (!done) { ref_caps.z = 1.0; }
    }
    }
    return v;
}

fn cloud_reference_trace(origin: vec3<f32>, view_dir: vec3<f32>, t_scene: f32, px: vec2<u32>) -> RefOut {
    let sample_index = view.frame_count / 4u;
    // Decorrelated per pixel and per frame (nested PCG hashes; a linear seed correlates).
    var salt = 0u;
    @if(CLOUD_REF_SEED2) { salt = 0x9e3779b9u; }
    ref_rng = ref_hash(px.x ^ ref_hash(px.y ^ ref_hash(view.frame_count ^ salt)));
    ref_camera = origin;
    ref_caps = vec4(0.0);
    let ch = (sample_index + px.x * 7u + px.y * 13u) % 3u;
    var lum_w = array<f32, 3>(0.2126, 0.7152, 0.0722);

    let air_max = ref_air_majorant[ch];

    let has_light = lights.n_directional_lights > 0u;
    var l = vec3(0.0, 1.0, 0.0);
    var sun = 0.0;
    if (has_light) {
        l = lights.directional_lights[0].direction_to_light;
        sun = lights.directional_lights[0].color.rgb[ch];
    }

    var out: RefOut;
    // Camera-ray transmittance (ratio tracking, same channel).
    let tf = ref_flight(origin, view_dir, t_scene, ch, air_max, true);
    // The ground is opaque (audit 35): only reaching the scene depth or escaping transmits.
    let t_cam = select(tf.tr, 0.0, tf.event == 3u || tf.event == 4u);

    var L = 0.0;
    var thr = 1.0;
    var p = origin;
    var d = view_dir;
    var t_max = t_scene;
    var first_t = 0.0;
    var bounce = 0u;
    for (; bounce < REF_MAX_BOUNCES; bounce++) {
        let f = ref_flight(p, d, t_max, ch, air_max, false);
        if (bounce == 0u) { first_t = f.t; }
        if (f.event == 0u || f.event == 5u || f.event == 3u) { break; }
        p = p + d * f.t;
        t_max = 1e30;
        if (f.event == 4u) {
            if (bounce == 0u && t_scene < 1e29) { break; } // scene geometry handled by the raster
            // Lambertian ground.
            let n = normalize(p);
            p = n * (atmosphere.inner_radius + 0.5);
            if (has_light) {
                L += thr * atmosphere.ground_albedo[ch] / PI * sun * max(dot(l, n), 0.0) * ref_sun_vis(p, l, ch);
            }
            thr *= atmosphere.ground_albedo[ch];
            // cosine-weighted hemisphere
            let u1 = ref_rand();
            d = ref_dir_around(n, sqrt(u1));
        } else if (f.event == 2u) {
            // Cloud scatter (albedo folded into the analog absorption event).
            if (has_light) {
                L += thr * cloud_phase(dot(l, d)) * sun * ref_sun_vis(p, l, ch);
            }
            d = ref_dir_around(d, clamp(ref_sample_cloud_phase(), -1.0, 1.0));
        } else {
            // Air scatter: phase = scattering LUT (sum_i sigma_s,i * p_i) / sigma_s.
            let r = length(p);
            let s_air = max(sample_density_lut(r, SCATTERING_DENSITY)[ch], 1e-20);
            if (has_light) {
                let ph = sample_scattering_lut(r, dot(l, d))[ch] / s_air;
                L += thr * ph * sun * ref_sun_vis(p, l, ch);
            }
            // Sample 50% uniform sphere / 50% HG(0.8), weight by phase / mixture pdf.
            var nd: vec3<f32>;
            if (ref_rand() < 0.5) {
                nd = ref_dir_around(d, 1.0 - 2.0 * ref_rand());
            } else {
                nd = ref_dir_around(d, clamp(ref_sample_hg(REF_AIR_HG_G), -1.0, 1.0));
            }
            let cs = dot(nd, d);
            let pdf = 0.5 / (4.0 * PI) + 0.5 * hg_phase(cs, REF_AIR_HG_G);
            thr *= (sample_scattering_lut(r, cs)[ch] / s_air) / pdf;
            d = nd;
        }
        // CLOUD_TERMS=s1: single scattering only (sun NEE at the first event), to validate the
        // realtime single-scatter octave in isolation.
        @if(CLOUD_TERM_S1) { break; }
        if (bounce > 3u) {
            var q = clamp(thr, 0.05, 1.0);
            if (bounce >= REF_DEEP_EVENTS) { q *= REF_DEEP_SURVIVAL; }
            if (ref_rand() > q) { break; }
            thr /= q;
        }
    }
    if (bounce == REF_MAX_BOUNCES) { ref_caps.x = 1.0; }
    var rgb = vec3(0.0);
    rgb[ch] = 3.0 * L;
    out.c = vec4(rgb, 3.0 * lum_w[ch] * t_cam);
    @if(ATM_DBG_REF_CAPS) { out.c = ref_caps; }
    out.d = first_t;
    return out;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    // Before any invocation returns: the whole workgroup takes part (barriers).
    @if(CLOUD_REFERENCE) { ref_compute_bounds(li); }
    let trace_size = textureDimensions(out_clouds);
    if (any(id.xy >= trace_size)) {
        return;
    }
    // The cloud targets cover the camera's viewport; the depth texture covers the whole render
    // target, so its texels are offset by the viewport origin.
    let full_size_u = max(vec2<u32>(view.viewport.zw), vec2(1u));
    let viewport_origin = vec2<i32>(view.viewport.xy);
    // Cloud-buffer pixel traced by this invocation this frame. The buffer is half the viewport,
    // or smaller (`VolumetricCloudSettings::max_render_size`; then even-sized, so the trace is
    // exactly half of it): its size is the smaller of the two.
    let half_size = max(min(full_size_u / 2u, trace_size * 2u), vec2(1u));
    let half_px = min(id.xy * 2u + CHECKER_OFFSETS[view.frame_count % 4u], half_size - 1u);
    let uv = (vec2<f32>(half_px) + 0.5) / vec2<f32>(half_size);

    // Conservative scene depth over the low-res footprint: the farthest of the covered texels
    // (reverse-Z: smallest depth), so thin occluders don't punch holes into the reconstruction.
    // The footprint is the full-res pixels the texel overlaps (2 x 2 at half resolution, up to
    // ~4 x 4 under the size cap).
    let scale = vec2<f32>(full_size_u) / vec2<f32>(half_size);
    let base_px = vec2<i32>(floor(vec2<f32>(half_px) * scale));
    let footprint = vec2<i32>(ceil(vec2<f32>(half_px + 1u) * scale)) - base_px;
    var depth = 1.0;
    for (var y = 0; y < footprint.y; y++) {
        for (var x = 0; x < footprint.x; x++) {
            let p = viewport_origin + min(base_px + vec2(x, y), vec2<i32>(full_size_u) - 1);
            @if(MULTISAMPLED_DEPTH) {
                // MSAA: the farthest sample of the pixel too (edges partly cover the sky).
                for (var s = 0u; s < textureNumSamples(depth_texture); s++) {
                    depth = min(depth, textureLoad(depth_texture, p, i32(s)));
                }
            } @else {
                depth = min(depth, textureLoad(depth_texture, p, 0));
            }
        }
    }

    var result = vec4(0.0, 0.0, 0.0, 1.0);
    var weighted_depth = 0.0;

    let ray_dir = uv_to_ray_direction(uv);
    let ray_origin = get_view_position();
    let r = length(ray_origin);
    let mu = dot(ray_dir, normalize(ray_origin));

    let seg = cloud_layer_segment(ray_origin, ray_dir);
    var t_start = seg.x;
    var t_end = min(seg.y, settings.cloud_max_distance);
    // Early exits: shell miss, planet occlusion, scene depth bound.
    if (ray_intersects_ground(r, mu)) {
        t_end = min(t_end, ray_sphere_intersect(r, mu, atmosphere.inner_radius).x);
    }
    if (depth > 0.0) {
        t_end = min(t_end, ndc_to_camera_dist(vec3(uv_to_ndc(uv), depth)));
    }

    @if(CLOUD_REFERENCE) {
    var t_scene = 1e30;
    if (depth > 0.0) {
        t_scene = ndc_to_camera_dist(vec3(uv_to_ndc(uv), depth));
    }
    let rp = cloud_reference_trace(ray_origin, ray_dir, t_scene, half_px);
    weighted_depth = rp.d;
    result = vec4(rp.c.rgb * view.exposure, rp.c.a);
    @if(ATM_DBG_REF_CAPS) { result = rp.c; }
    }
    @if(!CLOUD_REFERENCE) {
    if (seg.z > 0.5 && t_end > t_start && cloud_layer.cloud_density > 0.0) {
        // Per-pixel jitter of the ray start (STBN when bound, IGN otherwise).
        let stbn_dims = textureDimensions(stbn_texture);
        var jitter: f32;
        if (all(stbn_dims > vec2(1u))) {
            let layer = i32(view.frame_count % u32(textureNumLayers(stbn_texture)));
            // Indexed by the trace texel (dense), not the half-res pixel (both noises): the
            // checkerboard's half_px samples at a stride of 2 with a per-frame offset, which
            // loses the blue-noise spectrum (IGN: its diagonal structure aliases) and left
            // diagonal hatching in near-camera fog when flying through clouds, which the
            // reconstruction then took frames to average out.
            jitter = textureLoad(stbn_texture, vec2<i32>(id.xy) % vec2<i32>(stbn_dims), layer, 0).r;
        } else {
            jitter = interleaved_gradient_noise(vec2<f32>(id.xy), view.frame_count % 64u);
        }

        let sigma_s_unit = get_cloud_scattering_coeff();
        let sigma_t_unit = sigma_s_unit + get_cloud_absorption_coeff();
        var transmittance = 1.0;
        // Optical depth along the view ray since it entered the current cloud (reset on empty
        // samples): the view-side half of the local escape estimate below.
        var view_od_in_cloud = 0.0;
        // ATM_DBG_OUT_SUN_OD: (sun optical depth local, the field's diffusion solution, the
        // field's escape probability, density) at the first in-cloud sample.
        var dbg_sun = vec4(-1.0);
        var depth_weight_sum = 0.0;
        var depth_sum = 0.0;
        var luminance = vec3(0.0);
        // Sky irradiance once per ray (reduction stand-in); height/profile modulate per sample.
        let sky_irr = sample_sky_irradiance(r);


        // Coarse/fine march (HZD15 / Nubis17 / design.md §1.4): large steps with weather+profile
        // only; on hit, step back and switch to fine with full noise. After N empty fine samples
        // return to coarse. Light taps only in fine mode with density > 0.
        var t = t_start + jitter * cloud_view_step(t_start);
        var fine_mode = false;
        // Ground-truth march: no coarse gate at all.
        @if(ATM_DBG_NO_COARSE) { fine_mode = true; }
        @if(ATM_DBG_NO_COARSE) { fine_mode = true; }
        var fine_empty: u32 = 0u;
        // Length of the last coarse leap: a coarse hit steps back over it, and only it.
        var last_leap = 0.0;
        var step_i: u32 = 0u;
        // Step budget from the traced in-layer distance (UE5 volumetric clouds scale the view
        // sample count with it up to a maximum), not from where the camera is: a near-horizontal
        // ray from just below the base (or above the top) is as long an in-slab path as one from
        // inside, and a fixed 96 steps ran out in the first clouds, leaving the rest unmarched.
        let in_layer = r > cloud_layer.cloud_layer_start && r < cloud_layer.cloud_layer_end;
        let base_steps = f32(max(settings.cloud_max_steps, 1u));
        let in_layer_steps = round(base_steps * CLOUD_IN_LAYER_STEP_RATIO);
        let budget = clamp((t_end - t_start) / CLOUD_STEP_NEAR_M, base_steps, in_layer_steps);
        var max_steps = select(u32(budget), u32(in_layer_steps), in_layer);
        // Ground-truth march for skip validation: fine samples everywhere, no budget limit.
        @if(ATM_DBG_NO_COARSE) { max_steps = 16384u; }

        loop {
            if (t >= t_end || step_i >= max_steps || transmittance < CLOUD_MIN_TRANSMITTANCE) {
                break;
            }
            step_i += 1u;

            // Never let the step budget end before the segment does (Hillaire 2016 / UE5: the
            // sample count is spread over the in-layer segment): inside the slab a grazing ray
            // spends its budget on near wisps and distant clouds were clipped away.
            // Only empty-space (coarse) steps stretch; in-cloud steps keep their spacing.
            let remaining = max(t_end - t, 0.0) / f32(max_steps - step_i + 1u);
            let dist_step = cloud_view_step(t);
            // Coarse (empty-space) samples use the fine spacing: the coarse predicate is only
            // sampled, so a longer step (HZD15's 3x) hops over the thin cloud caps a grazing ray
            // crosses, and those clouds pop in and out as the camera moves. Coarse mode stays
            // cheaper per sample (weather only), and the SDF and the budget stretch make the leaps.
            let dt = select(dist_step, max(dist_step, remaining), !fine_mode);
            let p = ray_origin + ray_dir * t;
            let r_p = length(p);
            let mip = cloud_mip_from_distance(t);

            var density = 0.0;
            if (fine_mode) {
                density = get_cloud_medium_density_lod(r_p, p, mip);
                // Long steps: the step's density is the mean of evenly spaced samples across
                // it, also when the first is empty, which keeps the step's optical depth
                // unbiased (audit 24). exp of one jittered sample averages too transparent
                // where the density varies within the step. Lighting stays one evaluation per
                // step (sigma_s / sigma_t does not depend on density).
                let sub_n = max(settings.cloud_step_samples, 1u);
                if (sub_n > 1u && dt > CLOUD_STEP_SUB_MIN_M) {
                    var density_sum = density;
                    for (var k = 1u; k < sub_n; k++) {
                        let pk = p + ray_dir * (dt * f32(k) / f32(sub_n));
                        density_sum += get_cloud_medium_density_lod(length(pk), pk, mip);
                    }
                    density = density_sum / f32(sub_n);
                }
                if (density <= 0.0) {
                    view_od_in_cloud = 0.0;
                    fine_empty += 1u;
                    var to_coarse = fine_empty >= CLOUD_FINE_EMPTY_TO_COARSE;
                    @if(ATM_DBG_NO_COARSE) { to_coarse = false; }
                    if (to_coarse) {
                        fine_mode = false;
                        last_leap = 0.0;
                        fine_empty = 0u;
                    }
                    t += dt;
                    continue;
                }
                fine_empty = 0u;
            } else {
                // The fine density's near-camera reshape raises small densities (pow 0.5), so the
                // bound gets it too, or thin near-camera cloud failed the coarse test (audit 25).
                let coarse = near_camera_reshape(
                    get_cloud_medium_density_coarse(r_p, p) / max(cloud_layer.cloud_density, 1e-6),
                    mip,
                ) * cloud_layer.cloud_density;
                if (coarse > CLOUD_COARSE_DENSITY_EPS) {
                    // Step back and enter fine mode (Nubis17).
                    // Undo the leap that landed here (not the current step, which with the
                    // budget stretch can exceed the fine run since the last hit: the march
                    // then moved backwards until the budget ran out).
                    // The fine run then starts a jittered fraction of a fine step in: coarse
                    // leaps (budget stretch, SDF) are much longer than the ray-start jitter, so
                    // hits land at nearly the same distances on every pixel and frame, and the
                    // fine slices line up into bands and rings that crawl with the camera.
                    // Only after a real leap: with no leap yet (the first sample, already at the
                    // jittered ray start) adding the jitter again moved the first fine sample by
                    // up to two steps and lost near-entry optical depth (audit 23).
                    if (last_leap > 0.0) {
                        t = max(t_start, t - last_leap + jitter * dist_step);
                    }
                    fine_mode = true;
                    fine_empty = 0u;
                    continue;
                }
                // Nubis Cubed SDF ray march: step = max(sdf_cloud_distance, adaptive_step).
                // Margin: trilinear SDF error (~half a voxel diagonal) plus up to one voxel of
                // camera-snap offset, since the field is rebuilt over 4 of every 8 frames.
                // Inside the field grid the SDF bounds the skip; the budget stretch (`remaining`)
                // applies only beyond it, where it would otherwise step over whole clouds that the
                // SDF knows about (holes that depend on how much budget nearer clouds used).
                var skip = dt;
                @if(!ATM_DBG_NO_SDF_SKIP) {
                    let d_sdf = cloud_distance(p, ray_origin);
                    if (d_sdf < CLOUD_DIST_UNKNOWN) {
                        skip = max(dist_step,
                            d_sdf - CLOUD_SDF_SKIP_MARGIN_VOXELS * cloud_field_voxel_m());
                    }
                }
                t += skip;
                last_leap = skip;
                continue;
            }

            let sigma_t = density * sigma_t_unit;
            let sigma_s = density * sigma_s_unit;
            let up = p / r_p;
            let dens_ctx = cloud_density_context(r_p, p);
            let esc_col = cloud_escape_and_column(p);

            var direct_sum = vec3(0.0);
            for (var light_i: u32 = 0u; light_i < lights.n_directional_lights; light_i++) {
                let light = &lights.directional_lights[light_i];
                let l = (*light).direction_to_light;
                let mu_light = dot(l, up);
                let t_atm = sample_transmittance_lut(r_p, mu_light)
                    * f32(!ray_intersects_ground(r_p, mu_light));
                // Local light march through the real (fine) density (Nubis/HZD cone march): 6 taps with
                // doubling step (~1.3 km) + one long tap, so billows self-shadow. The cloud shadow map
                // (coarse density, 4x4 time-sliced, temporally filtered) is only used for the
                // occlusion *beyond* the local march; using it at `p` replaced all per-billow detail
                // with smooth weather-map blobs once it converged.
                var light_od = 0.0;
                var tap_t = 0.0;
                var tap_dt = LIGHT_TAP_STEP_M;
                var n_taps = settings.cloud_light_taps;
                var tap_growth = LIGHT_TAP_GROWTH;
                var tap_lod = LIGHT_TAP_LOD;
                // Validation: a dense uniform march toward the light (256 x 30 m at the sample's
                // mip) to test the light taps' quadrature against the reference.
                @if(ATM_DBG_LIGHT_BRUTE) {
                    n_taps = 256u;
                    tap_dt = 30.0;
                    tap_growth = 1.0;
                    tap_lod = 0.0;
                }
                for (var k = 0u; k < n_taps; k++) {
                    let pk = p + l * (tap_t + 0.5 * tap_dt);
                    let dk = get_cloud_medium_density_lod(length(pk), pk, mip + tap_lod * f32(k));
                    light_od += dk * tap_dt;
                    tap_t += tap_dt;
                    tap_dt *= tap_growth;
                }
                let p_far = p + l * (tap_t + LIGHT_LONG_TAP_M * 0.5);
                let d_far = get_cloud_medium_density_lod(length(p_far), p_far, mip + 3.0);
                light_od += d_far * LIGHT_LONG_TAP_M;
                let light_optical_depth = light_od * sigma_t_unit;
                let p_beyond = p + l * (tap_t + LIGHT_LONG_TAP_M);
                var t_far = sample_cloud_shadow_map(p_beyond, l, false);
                @if(ATM_DBG_NO_FAR_SHADOW) { t_far = 1.0; }
                let t_local = exp(-light_optical_depth);
                @if(ATM_DBG_OUT_SUN_OD) {
                    if (dbg_sun.x < 0.0) { dbg_sun = vec4(light_optical_depth, cloud_field_diffuse(p), esc_col.x, density); }
                }
                // ATM_DBG_OUT_SUN_TAU: (sun optical depth, local march + map beyond; the map alone
                // at p; the field's diffusion solution; density) at the first in-cloud sample.
                @if(ATM_DBG_OUT_SUN_TAU) {
                    if (dbg_sun.x < 0.0) {
                        dbg_sun = vec4(light_optical_depth - log(max(t_far, 1e-6)), -log(max(sample_cloud_shadow_map(p, l, false), 1e-6)), cloud_field_diffuse(p), density);
                    }
                }
                let t_cloud = t_far * t_local;
                let cos_theta = dot(l, ray_dir);
                // Nubis Cubed structure (Schneider, Advances 2023, "Direct = Transmittance * Phase +
                // Multiple Scattering * Secondary Phase") with physically derived terms. Multiple
                // scattering (orders >= 2), isotropic: E_sun * exp(-(1 - g) tau_sun) * F_ms / 4pi.
                //  - exp(-(1 - g) tau_sun): diffuse sunlight under similarity scaling (Joseph,
                //    Wiscombe & Weinman 1976), tau_sun = local march + shadow-map optical depth.
                //  - F_ms = f_ms / (1 - f_ms), f_ms = albedo (1 - P_esc): Hillaire 2020's transfer
                //    series with the escape probability on the reduced optical depth (cloud field).
                // vs MC (validation/sky/ms_models_vs_mc.py, tau <= 5): median error -29..+25 %.
                var direct = t_local * t_far * cloud_phase(cos_theta);
                @if(!CLOUD_TERM_S1) {
                    let tau_sun = light_optical_depth - log(max(t_far, 1e-6));
                    let albedo = sigma_s_unit / max(sigma_t_unit, 1e-9);
                    // P_esc of a finite cloud is positive; the escape volume stores it in
                    // RGBA16F, whose smallest normal (6.1e-5) is the floor kept here, so a thick
                    // conservative cloud (zero absorption) cannot make 1 - f_ms zero (audit 22).
                    // Escape probability: the cloud field's (375 m cells; validated for clouds
                    // larger than that) or, where smaller, a local estimate from the fine density
                    // along the two directions the march already knows, toward the sun and back
                    // along the view ray: P_esc ~ mean of exp(-(1 - g) tau) (similarity-reduced,
                    // as the field's). Small cumuli are much smaller than a field cell, which
                    // diluted their extinction there: P_esc ~ 1, no multiple scattering, and
                    // their sunlit sides rendered gray although tau ~ 10-20 across them.
                    let tau_view = view_od_in_cloud + 0.5 * sigma_t * dt;
                    let p_esc_local = 0.5 * (exp(-(1.0 - CLOUD_MIE_MEAN_COS) * tau_sun)
                        + exp(-(1.0 - CLOUD_MIE_MEAN_COS) * tau_view));
                    var p_esc = min(esc_col.x, p_esc_local);
                    @if(ATM_DBG_ESC_ZERO) { p_esc = 0.0; }
                    @if(ATM_DBG_ESC_FIELD_ONLY) { p_esc = esc_col.x; }
                    let f_ms = albedo * (1.0 - max(p_esc, 6.1035e-5));
                    var ms_vol = exp(-(1.0 - CLOUD_MIE_MEAN_COS) * tau_sun) * f_ms / (1.0 - f_ms);
                    // Thick cloud: the escape series decays as exp(-(1 - g) tau_sun) and leaves
                    // the shaded side dark (MC: ~0 below a reduced depth of a few), while the
                    // diffuse light of a conservative medium falls off linearly through it. The
                    // delta-Eddington diffusion solution of the vertical slab (the field's
                    // columns to the layer top and bottom), lit by the sunlight that reaches the
                    // column (the far shadow map), takes over where it is larger
                    // (ATM_DBG_MS_DIFFUSION).
                    // The cloud field's 3D diffusion solve (cloud_field.wesl): the diffuse light of
                    // the whole cloud around p, lit by the first directional light. The escape
                    // series stays where it is larger (thin cloud, beyond the grid).
                    var field_on = light_i == 0u;
                    @if(ATM_DBG_MS_SERIES_ONLY) { field_on = false; }
                    if (field_on) {
                        ms_vol = max(ms_vol, cloud_field_diffuse(p));
                    }
                    // Experiment: the vertical-slab solution (lights only the layer top).
                    @if(ATM_DBG_MS_DIFFUSION) {
                        if (mu_light > 0.0) {
                            ms_vol = max(ms_vol, t_far * cloud_diffuse_sun(esc_col.y, esc_col.z, max(mu_light, 0.05)));
                        }
                    }
                    direct += ms_vol * (0.25 / PI);
                }
                direct_sum += (*light).color.rgb * t_atm * direct;
            }
            // Sunlit ground below this sample (Lambert, albedo): the first directional light's
            // irradiance through the atmosphere times the mean cloud-shadow visibility of the
            // ground it sees. After the light loop, not in it: its 16 shadow taps kept their state
            // live through the multiple-scattering work (register spills, Apple M5).
            var ground_rad = sky_irr * atmosphere.ground_albedo;
            if (lights.n_directional_lights > 0u) {
                let light0 = &lights.directional_lights[0];
                let l0 = (*light0).direction_to_light;
                let mu_g = dot(l0, up);
                let e_sun = (*light0).color.rgb * sample_transmittance_lut(atmosphere.inner_radius, mu_g)
                    * max(mu_g, 0.0) * ground_sun_visibility(p, l0);
                ground_rad = atmosphere.ground_albedo * (e_sun / PI + sky_irr);
            }
            var ambient = cloud_ambient_term(sky_irr, ground_rad, esc_col.y, esc_col.z);
            // Validation term toggles (CLOUD_TERMS=s1|sun|amb, see resources.rs).
            @if(CLOUD_TERM_NO_AMB) { ambient = vec3(0.0); }
            @if(CLOUD_TERM_NO_SUN) { direct_sum = vec3(0.0); }
            let scattered = (direct_sum + ambient) * sigma_s;

            let step_t = exp(-sigma_t * dt);
            luminance += transmittance * (scattered - scattered * step_t) / max(sigma_t, 1e-9);
            let w = transmittance * (1.0 - step_t);
            depth_sum += w * t;
            depth_weight_sum += w;
            transmittance *= step_t;
            view_od_in_cloud += sigma_t * dt;
            t += dt;
        }

        if (transmittance < CLOUD_MIN_TRANSMITTANCE) {
            luminance /= max(1.0 - transmittance, 1e-4);
            transmittance = 0.0;
        }
        weighted_depth = select(0.0, depth_sum / depth_weight_sum, depth_weight_sum > 0.0);
        // Pre-composite the aerial perspective in front of the cloud here (half-res, jittered,
        // temporally reconstructed) instead of in the full-res render_sky pass:
        //   rgb = T_ap(0..d) * L_c + (1 - T_c) * L_ap_shadowed(0..d)
        var lum = luminance;
        var haze_on = true;
        @if(CLOUD_NO_HAZE) { haze_on = false; }
        if (haze_on && transmittance < 0.999 && weighted_depth > 0.0) {
            let j = fract(jitter + f32(view.frame_count % 64u) * 0.61803398875);
            var t_ap = sample_transmittance_lut_segment(r, mu, weighted_depth);
            // Similarity-scaled AP transmittance for extended sources (clouds behind haze/Mie):
            // T_sim = exp(-(τ_a + τ_s_ray + (1-g)τ_s_mie)) = T_bal · exp(τ_red),
            // τ_red = ∫ g·σ_s_mie from the transmittance LUT alpha (Rayleigh stays ballistic).
            // Homogeneous-haze MC: τ=1.2 T_bal 0.30 vs MC vis 0.73; T_sim 0.64 (progress run 10).
            // Optional ATM_DBG_AP_BAL keeps ballistic T.
            let tau_red = sample_transmittance_lut_segment_tau_red(r, mu, weighted_depth);
            @if(!ATM_DBG_AP_BAL) {
                // Similarity (delta-scaling, Joseph, Wiscombe & Weinman 1976) applied uniformly:
                // assumes the cloud is extended relative to the haze's forward lobe, so light
                // scattered out of the line of sight is replaced from neighbouring cloud.
                t_ap = t_ap * exp(tau_red);
            }
            var haze = cloud_shadowed_haze(uv, ray_origin, ray_dir, weighted_depth, r, mu, j);
            @if(ATM_DBG_TAP_ONE) { t_ap = vec3(1.0); }
            lum = t_ap * luminance + (1.0 - transmittance) * haze;
        }
        result = vec4(lum * view.exposure, transmittance);
        @if(ATM_DBG_OUT_SUN_OD) { result = dbg_sun; }
        @if(ATM_DBG_OUT_SUN_TAU) { result = dbg_sun; }
        @if(ATM_DBG_OUT_H) {
            let ph = ray_origin + ray_dir * weighted_depth;
            let ch = cloud_density_context(length(ph), ph);
            result = vec4(ch.z, ch.x, weighted_depth * 1e-4, transmittance);
        }
    }

    // Thin (ice) layers, lowest first: a shell above the camera goes behind what is composited
    // (nearest first, the lowest), one below in front of it (farthest first, again the lowest).
    if (cloud_layer.thin_count > 0u) {
        var t_scene = 1e30;
        if (depth > 0.0) {
            t_scene = ndc_to_camera_dist(vec3(uv_to_ndc(uv), depth));
        }
        // Slice-depth jitter: blue noise per pixel and frame (decorrelated from the ray start).
        let j = fract(interleaved_gradient_noise(vec2<f32>(half_px), view.frame_count % 64u) + 0.5);
        for (var layer = 0u; layer < min(cloud_layer.thin_count, 4u); layer++) {
            result = composite_cirrus(layer, result, &weighted_depth, ray_origin, ray_dir, r, mu, t_scene, j);
        }
    }
    }
    textureStore(out_clouds, id.xy, result);
    textureStore(out_clouds_depth, id.xy, vec4(weighted_depth, 0.0, 0.0, 0.0));
}
