// Coarse cloud signed-distance field (Nubis Cubed, Advances 2023, "SDF" / "Voxel Cloud
// Ray-Marcher"). Three steps:
//   init:  inside/outside of the coarse density (same predicate the raymarch uses to enter fine
//          mode) -> squared-distance seeds (rg) + vertical voxel size (b).
//   edt_x/edt_y/edt_z: separable exact Euclidean distance transform (Felzenszwalb & Huttenlocher
//          2012 lower-envelope form, brute-force within a window) for both the outside and the
//          inside distance. edt_z writes the final signed distance in metres (negative inside).
//   escape: per-voxel escape probability of multiply scattered light (Hillaire 2020 f_ms with the
//          similarity-reduced optical depth, see `escape`), from the extinction written by init.
//   diffusion_source / sor_red / sor_black / diffusion_publish: the diffuse (multiply scattered)
//          sunlight of the cloud field from a delta-Eddington diffusion solve (see `sor`).

import package::atmosphere::bindings::{atmosphere, view, lights, settings, cloud_shadow_map, cloud_field_sdf};
import package::atmosphere::functions::CLOUD_MIE_MEAN_COS;
// The diffusion source's sun optical depth (its pipelines alone set CLOUDS_ENABLED).
@if(CLOUDS_ENABLED)
import package::atmosphere::functions::{cloud_shadow_frame, cloud_shadow_transmittance, cloud_thin_shadow_weight};
import package::atmosphere::clouds::{
    cloud_layer,
    get_cloud_medium_density_coarse,
    get_cloud_medium_density_lod,
    get_cloud_scattering_coeff,
    get_cloud_absorption_coeff,
    cloud_field_origin_from_cell,
    cloud_field_frame_to_atmosphere_xz,
    cloud_field_cell,
    cloud_field_voxel_m,
    CLOUD_FIELD_RES_XY,
    CLOUD_FIELD_RES_Z,
    cloud_shadow_xy_to_uv,
};

@group(0) @binding(40) var out_field_init: texture_storage_3d<rgba16float, write>;
@group(0) @binding(41) var in_field: texture_3d<f32>;
@group(0) @binding(42) var out_field: texture_storage_3d<rgba16float, write>;
@group(0) @binding(43) var out_field_sdf: texture_storage_3d<rgba16float, write>;
// Extinction per voxel (r, m^-1), vertical voxel size (g, m), build cell (ba). Written by init.
@group(0) @binding(45) var out_field_ext: texture_storage_3d<rgba16float, write>;
@group(0) @binding(47) var in_field_ext: texture_3d<f32>;
// Escape probability (r), optical depth to the layer top (g) and bottom (b). Written by escape
// in the same frame as edt_z, so it shares cloud_field_sdf's build cell.
@group(0) @binding(46) var out_field_esc: texture_storage_3d<rgba16float, write>;

// Squared distances are stored in xy-voxel units (fp16-safe: < 2048 within the windows).
const BIG: f32 = 60000.0;
// Same threshold as the raymarch's coarse -> fine switch (clouds_raymarch.wesl).
const INSIDE_EPS: f32 = 1e-4;
// Search windows (voxels): ~1.5 km horizontally, the whole layer vertically.
const WIN_XY: i32 = 8;
const WIN_Z: i32 = 31;

fn view_position() -> vec3<f32> {
    let p = (atmosphere.world_to_atmosphere * vec4(view.world_position, 1.0)).xyz;
    let r = length(p);
    let rmin = atmosphere.inner_radius + 1.0;
    return select(p, p / max(r, 1e-6) * rmin, r < rmin);
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= CLOUD_FIELD_RES_XY || id.y >= CLOUD_FIELD_RES_XY || id.z >= CLOUD_FIELD_RES_Z) {
        return;
    }
    let vox = cloud_field_voxel_m();
    let thick = cloud_layer.cloud_layer_end - cloud_layer.cloud_layer_start;
    let dz = thick / f32(CLOUD_FIELD_RES_Z);
    let cell = cloud_field_cell(view_position());
    let xz = cloud_field_frame_to_atmosphere_xz(cloud_field_origin_from_cell(cell) + (vec2<f32>(id.xy) + 0.5) * vox);
    let r = cloud_layer.cloud_layer_start + (f32(id.z) + 0.5) * dz;
    let p = vec3(xz.x, sqrt(max(r * r - dot(xz, xz), 0.0)), xz.y);
    let coarse = get_cloud_medium_density_coarse(length(p), p);
    let inside = coarse > INSIDE_EPS;
    // a: optical depth of the voxel's vertical extent (same mip-3 density as the shadow map),
    // blurred by edt_x/edt_y and summed by edt_z into the column optical depth.
    var od = 0.0;
    if (coarse > 0.0) {
        od = get_cloud_medium_density_lod(length(p), p, AO_DENSITY_MIP)
            * (get_cloud_scattering_coeff() + get_cloud_absorption_coeff()) * dz;
    }
    // r: squared distance to the nearest inside voxel, g: to the nearest outside voxel.
    // b: the build cell (x at z = 0, z at z = 1) and dz / vox (z >= 2) travel to edt_z.
    let cell_c = select(select(dz / vox, cell.y, id.z == 1u), cell.x, id.z == 0u);
    textureStore(out_field_init, id, vec4(select(BIG, 0.0, inside), select(0.0, BIG, inside), cell_c, od));
    textureStore(out_field_ext, id, vec4(od / dz, dz, cell));
}

// Escape probability of multiply scattered light, P_esc(x) = (1/4pi) ∫ exp(-tau'(x, w)) dw, with
// the similarity-reduced optical depth tau' = (1 - <cos>) tau (Joseph, Wiscombe & Weinman 1976:
// for orders >= 2 a forward-scattering medium acts like an isotropic one with sigma (1 - g)).
// The cloud pass turns it into Hillaire 2020's transfer fraction f_ms = albedo (1 - P_esc) and
// the series 1 / (1 - f_ms). Validated against MC (validation/sky/ms_depth_mc.py,
// ms_models_vs_mc.py). Quadrature: the 14 cube directions (6 faces, 8 corners), equal weights.
// March: steps growing x1.35 from 40 m (12 steps, ~4 km) through the voxel extinction; the
// layer top/bottom and the grid side end the path (no cloud beyond). Stored at half the field's
// horizontal resolution (375 m): P_esc varies over the reduced mean free path 1/((1-g) sigma),
// ~400 m at full density.
const ESC_STEPS: u32 = 12u;
const ESC_STEP0_M: f32 = 40.0;
const ESC_STEP_GROWTH: f32 = 1.35;
const ESC_XY_DOWNSAMPLE: u32 = 2u;
const ESC_DIRS: array<vec3<f32>, 14> = array(
    vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0),
    vec3(0.0, -1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, -1.0),
    vec3(0.57735, 0.57735, 0.57735), vec3(-0.57735, 0.57735, 0.57735),
    vec3(0.57735, -0.57735, 0.57735), vec3(-0.57735, -0.57735, 0.57735),
    vec3(0.57735, 0.57735, -0.57735), vec3(-0.57735, 0.57735, -0.57735),
    vec3(0.57735, -0.57735, -0.57735), vec3(-0.57735, -0.57735, -0.57735),
);

@compute @workgroup_size(4, 4, 4)
fn escape(@builtin(global_invocation_id) id: vec3<u32>) {
    let res_xy = CLOUD_FIELD_RES_XY / ESC_XY_DOWNSAMPLE;
    if (id.x >= res_xy || id.y >= res_xy || id.z >= CLOUD_FIELD_RES_Z) {
        return;
    }
    // Centre of the 2x2 block in field-voxel coordinates.
    let centre = vec3(vec2<f32>(id.xy * ESC_XY_DOWNSAMPLE) + 0.5 * f32(ESC_XY_DOWNSAMPLE), f32(id.z) + 0.5);
    let e0 = textureLoad(in_field_ext, vec3<u32>(id.xy * ESC_XY_DOWNSAMPLE, id.z), 0);
    let dz = e0.g;
    let vox = cloud_field_voxel_m();
    // Grid space (voxels): x, y horizontal (field xy), z vertical.
    let scale = vec3(1.0 / vox, 1.0 / vox, 1.0 / max(dz, 1e-3));
    let res = vec3(f32(CLOUD_FIELD_RES_XY), f32(CLOUD_FIELD_RES_XY), f32(CLOUD_FIELD_RES_Z));
    let k = 1.0 - CLOUD_MIE_MEAN_COS;
    var p_esc = 0.0;
    var tau_vert = vec2(0.0);
    for (var d = 0u; d < 14u; d++) {
        let dir = ESC_DIRS[d] * scale;
        var t = 0.0;
        var dt = ESC_STEP0_M;
        var tau = 0.0;
        for (var s = 0u; s < ESC_STEPS; s++) {
            let q = centre + dir * (t + 0.5 * dt);
            if (any(q < vec3(0.0)) || any(q >= res)) { break; }
            tau += textureLoad(in_field_ext, vec3<i32>(q), 0).r * dt;
            t += dt;
            dt *= ESC_STEP_GROWTH;
        }
        p_esc += exp(-k * tau);
        // Directions 4 and 5 are straight up and down (grid z): the column above and below,
        // for the ambient (clouds_raymarch `cloud_ambient_term`).
        if (d == 4u) { tau_vert.x = tau; }
        if (d == 5u) { tau_vert.y = tau; }
    }
    textureStore(out_field_esc, id, vec4(p_esc / 14.0, tau_vert, 0.0));
}

// One workgroup per grid line: the line is loaded into workgroup memory once, so each voxel's
// windowed min reads shared memory instead of 2*WIN+1 global texels (was ~5.7 ms, bandwidth bound).
var<workgroup> line: array<vec2<f32>, 256>;
var<workgroup> line_od: array<f32, 256>;
var<workgroup> line_cell: array<f32, 32>;

// Cloud AO (UE5 volumetric clouds: a top-down cloud optical-depth map, blurred, occludes the
// sky light on the ground). Gaussian blur of the voxel optical depth along x and y, sigma in
// voxels (~ 750 m, of the order of the cloud-base height times the skylight cone).
const AO_BLUR_SIGMA: f32 = 4.0;
const AO_DENSITY_MIP: f32 = 3.0;

fn blur_line(lane: u32, n: u32) -> f32 {
    var acc = 0.0;
    var wsum = 0.0;
    let c = i32(lane);
    for (var k = -WIN_XY; k <= WIN_XY; k++) {
        let j = clamp(c + k, 0, i32(n) - 1);
        let w = exp(-0.5 * f32(k * k) / (AO_BLUR_SIGMA * AO_BLUR_SIGMA));
        acc += w * line_od[j];
        wsum += w;
    }
    return acc / wsum;
}

fn edt_line(lane: u32, n: u32, win: i32, s: f32) -> vec2<f32> {
    var best = vec2(BIG);
    let c = i32(lane);
    for (var k = -win; k <= win; k++) {
        let j = c + k;
        if (j < 0 || j >= i32(n)) { continue; }
        let d = f32(k) * s;
        best = min(best, line[j] + vec2(d * d));
    }
    return best;
}

// Dispatch (1, RES_XY, RES_Z): lane = x.
@compute @workgroup_size(128, 1, 1)
fn edt_x(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let q = vec3<i32>(i32(lane), i32(wg.y), i32(wg.z));
    let v = textureLoad(in_field, q, 0);
    line[lane] = v.xy;
    line_od[lane] = v.w;
    workgroupBarrier();
    textureStore(out_field, q, vec4(edt_line(lane, CLOUD_FIELD_RES_XY, WIN_XY, 1.0), v.z, blur_line(lane, CLOUD_FIELD_RES_XY)));
}

// Dispatch (RES_XY, 1, RES_Z): lane = y.
@compute @workgroup_size(128, 1, 1)
fn edt_y(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let q = vec3<i32>(i32(wg.x), i32(lane), i32(wg.z));
    let v = textureLoad(in_field, q, 0);
    line[lane] = v.xy;
    line_od[lane] = v.w;
    workgroupBarrier();
    textureStore(out_field, q, vec4(edt_line(lane, CLOUD_FIELD_RES_XY, WIN_XY, 1.0), v.z, blur_line(lane, CLOUD_FIELD_RES_XY)));
}

// Dispatch (RES_XY, RES_XY, 1): lane = z.
@compute @workgroup_size(32, 1, 1)
fn edt_z(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let q = vec3<i32>(i32(wg.x), i32(wg.y), i32(lane));
    let v4 = textureLoad(in_field, q, 0);
    line_cell[lane] = v4.z;
    workgroupBarrier();
    let dz = line_cell[2];
    line[lane] = v4.xy;
    line_od[lane] = v4.w;
    workgroupBarrier();
    var column_od = 0.0;
    for (var k = 0u; k < CLOUD_FIELD_RES_Z; k++) {
        column_od += line_od[k];
    }
    let v = edt_line(lane, CLOUD_FIELD_RES_Z, WIN_Z, dz) ;
    let vox = cloud_field_voxel_m();
    // Voxel centres -> surface: the boundary lies half a voxel between an inside and an outside
    // centre (smallest voxel axis).
    let half = 0.5 * min(1.0, dz);
    // Nothing found inside the window: the window radius is a safe lower bound.
    let d_out = min(sqrt(v.x), f32(WIN_XY));
    let d_in = min(sqrt(v.y), f32(WIN_XY));
    let sdf = select(d_out - half, half - d_in, v.x == 0.0) * vox;
    textureStore(out_field_sdf, q, vec4(sdf, column_od, line_cell[0], line_cell[1]));
}

// ---------------------------------------------------------------------------------------------
// Diffuse sunlight: the delta-Eddington diffusion equation on the field's cells (Joseph, Wiscombe &
// Weinman 1976: forward peak g^2 left in the direct beam, sigma' = (1 - g^2) sigma, g' = g / (1 + g)),
// for U = 4 pi J_diff per unit sun irradiance (the cloud pass's multiple-scattering source is
// E U / (4 pi) per unit sigma_s):
//     -div(D grad U) = (1 + g') sigma' T'_sun,   D = 1 / (3 sigma'),
// with T'_sun the delta-scaled transmittance toward the sun through the field, solved on the cloud
// cells (sigma > DIFF_SIGMA_MIN) with the Marshak condition at every face to clear air or the grid
// edge (no diffuse light entering: net outward flux U / 2). The diffuse light at a point depends
// on the whole cloud around it (it peaks inside the sunlit side and falls toward the far side),
// which no local estimate reproduces: against a 3D Monte Carlo of the IPRT C2 cube (Emde et al.
// 2019) the solve gives 0.82 x the MC field at 100 m cells, 0.65-0.69 x on this grid, where the
// escape series gives 0.44 x (.clouds-handoff/validation/iprt_c2/c2_field.py, c2_sor.py).
// Red-black successive over-relaxation in place, warm-started from the previous frame: 32
// iterations converge from zero (0.3 %), 1-2 track a sun moved by 2 degrees.
@group(0) @binding(55) var diff_src: texture_storage_3d<r32float, read_write>;
@group(0) @binding(56) var diff_u: texture_storage_3d<r32float, read_write>;
@group(0) @binding(57) var out_field_diff: texture_storage_3d<rgba16float, write>;
// The solution and source moved with the grid (z < RES_Z: U, above: the source).
@group(0) @binding(58) var diff_tmp: texture_storage_3d<r32float, read_write>;
// (0, 1): the build cell the solution and source are indexed by; (2, 3): the pending shift.
@group(0) @binding(60) var diff_state: texture_storage_3d<r32float, read_write>;

const DIFF_SIGMA_MIN: f32 = 1e-4;
const DIFF_OMEGA: f32 = 1.85;
const DIFF_SUN_STEPS: u32 = 16u;

fn diff_g() -> vec2<f32> {
    let g = CLOUD_MIE_MEAN_COS;
    return vec2(1.0 - g * g, g / (1.0 + g));
}

fn field_in_range(id: vec3<i32>) -> bool {
    return all(id >= vec3(0)) && all(id < vec3(i32(CLOUD_FIELD_RES_XY), i32(CLOUD_FIELD_RES_XY), i32(CLOUD_FIELD_RES_Z)));
}

/// Optical depth toward the first directional light from p, past the field's own march, from the
/// cloud shadow map (one texel: the field's cells are coarser than its texels), including the thin
/// layers above p; 0 off the map. Only beyond the march: inside a cloud the map's front depth and
/// mean extinction give 5-10x too little optical depth (hero/backlit views against the light
/// taps, ATM_DBG_OUT_SUN_TAU), as for the cloud pass's own lookup past its light taps.
@if(CLOUDS_ENABLED)
fn diffusion_sun_tau_beyond(p: vec3<f32>, l: vec3<f32>) -> f32 {
    let frame = cloud_shadow_frame(l);
    let extent = settings.cloud_shadow_map_extent;
    let rel = p - frame.anchor;
    let xy = vec2(dot(rel, frame.bx), dot(rel, frame.by));
    if (!frame.mapped || extent <= 0.0 || any(abs(xy) > vec2(extent))) {
        return 0.0;
    }
    let d_sample = dot(rel, normalize(-l)) + 2.0 * extent;
    let size = vec2<i32>(textureDimensions(cloud_shadow_map));
    let texel = clamp(vec2<i32>(cloud_shadow_xy_to_uv(xy, extent) * vec2<f32>(size)), vec2(0), size - 1);
    let q = textureLoad(cloud_shadow_map, texel, 0);
    let t = cloud_shadow_transmittance(q.rgb, d_sample) * exp(-cloud_thin_shadow_weight(p) * q.a);
    return -log(max(t, 1e-6));
}

/// Source of one cell: (1 + g') sigma' T'_sun, the sun's optical depth marched toward the first
/// directional light through the field's extinction (16 steps growing from 40 m, ~14 km) and the
/// shadow map beyond the march's end (clouds past the grid, the thin layers).
@if(CLOUDS_ENABLED)
fn diffusion_source_cell(id: vec3<u32>) {
    if (id.x >= CLOUD_FIELD_RES_XY || id.y >= CLOUD_FIELD_RES_XY || id.z >= CLOUD_FIELD_RES_Z) {
        return;
    }
    let e = textureLoad(in_field_ext, id, 0);
    let gg = diff_g();
    var src = 0.0;
    if (e.r > DIFF_SIGMA_MIN && lights.n_directional_lights > 0u) {
        let l = lights.directional_lights[0].direction_to_light;
        let vox = cloud_field_voxel_m();
        // Grid space (voxels): x, y horizontal (atmosphere xz), z vertical.
        let dir = vec3(l.x / vox, l.z / vox, l.y / max(e.g, 1e-3));
        let res = vec3(f32(CLOUD_FIELD_RES_XY), f32(CLOUD_FIELD_RES_XY), f32(CLOUD_FIELD_RES_Z));
        let centre = vec3<f32>(id) + 0.5;
        var tau = 0.0;
        var t = 0.0;
        var dt = ESC_STEP0_M;
        for (var s = 0u; s < DIFF_SUN_STEPS; s++) {
            let q = centre + dir * (t + 0.5 * dt);
            if (any(q < vec3(0.0)) || any(q >= res)) { break; }
            tau += textureLoad(in_field_ext, vec3<i32>(q), 0).r * dt;
            t += dt;
            dt *= ESC_STEP_GROWTH;
        }
        // The cell centre in atmosphere space (as `init`, with the extinction's own build cell,
        // where the cloud it holds is this frame).
        let xz = cloud_field_frame_to_atmosphere_xz(cloud_field_origin_from_cell(e.ba) + (vec2<f32>(id.xy) + 0.5) * vox);
        let r = cloud_layer.cloud_layer_start + (f32(id.z) + 0.5) * e.g;
        let p = vec3(xz.x, sqrt(max(r * r - dot(xz, xz), 0.0)), xz.y);
        tau += diffusion_sun_tau_beyond(p + l * t, l);
        src = (1.0 + gg.y) * gg.x * e.r * exp(-gg.x * tau);
    }
    textureStore(diff_src, id, vec4(src));
}

/// The source of every cell (a full build).
@if(CLOUDS_ENABLED)
@compute @workgroup_size(4, 4, 4)
fn diffusion_source(@builtin(global_invocation_id) id: vec3<u32>) {
    diffusion_source_cell(id);
}

/// The source of one cell in every 2 x 2 x 2 block (the frame's corner, frame mod 8), dispatched
/// over an eighth of the cells: each frame refreshes 1/8 of the cells spread over the whole field
/// (layers instead put the cloudy middle of the layer in two frames: 0.13-0.57 ms), all of them
/// every 8 frames (the field's own rebuild cadence).
@if(CLOUDS_ENABLED)
@compute @workgroup_size(4, 4, 4)
fn diffusion_source_slice(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = view.frame_count % 8u;
    diffusion_source_cell(2u * id + vec3(k & 1u, (k >> 1u) & 1u, k >> 2u));
}

/// One colour of a red-black sweep, dispatched over half the cells in x: cell x = 2 i + the
/// parity of (y + z + colour), so every invocation updates a cell of that colour.
fn sor(half_id: vec3<u32>, parity: u32) {
    let id = vec3(2u * half_id.x + ((half_id.y + half_id.z + parity) & 1u), half_id.y, half_id.z);
    if (id.x >= CLOUD_FIELD_RES_XY || id.y >= CLOUD_FIELD_RES_XY || id.z >= CLOUD_FIELD_RES_Z) {
        return;
    }
    let e = textureLoad(in_field_ext, id, 0);
    if (e.r <= DIFF_SIGMA_MIN) {
        textureStore(diff_u, id, vec4(0.0));
        return;
    }
    let gg = diff_g();
    let d = 1.0 / (3.0 * gg.x * e.r);
    let vox = cloud_field_voxel_m();
    let hz = max(e.g, 1e-3);
    let c = vec3<i32>(id);
    var sum = 0.0;
    var diag = 0.0;
    for (var k = 0; k < 6; k++) {
        let axis = k / 2;
        var o = vec3(0);
        o[axis] = select(-1, 1, (k & 1) == 1);
        let h = select(vox, hz, axis == 2);
        let n = c + o;
        var dn = 0.0;
        if (field_in_range(n)) {
            let sn = textureLoad(in_field_ext, n, 0).r;
            if (sn > DIFF_SIGMA_MIN) {
                dn = 1.0 / (3.0 * gg.x * sn);
            }
        }
        if (dn > 0.0) {
            let cond = 2.0 * d * dn / (d + dn) / (h * h);
            sum += cond * textureLoad(diff_u, n).r;
            diag += cond;
        } else {
            diag += 1.0 / (h * (h / (2.0 * d) + 2.0));
        }
    }
    let u = textureLoad(diff_u, c).r;
    let new_u = (textureLoad(diff_src, c).r + sum) / max(diag, 1e-30);
    textureStore(diff_u, c, vec4(max(mix(u, new_u, DIFF_OMEGA), 0.0)));
}

@compute @workgroup_size(4, 4, 4)
fn sor_red(@builtin(global_invocation_id) id: vec3<u32>) {
    sor(id, 0u);
}

@compute @workgroup_size(4, 4, 4)
fn sor_black(@builtin(global_invocation_id) id: vec3<u32>) {
    sor(id, 1u);
}

/// Copies the solution into the filterable volume the cloud pass samples (r: U per unit sun
/// irradiance, gb: build cell); `diffusion_reset` restarts the solve from zero.
@compute @workgroup_size(4, 4, 4)
fn diffusion_publish(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= CLOUD_FIELD_RES_XY || id.y >= CLOUD_FIELD_RES_XY || id.z >= CLOUD_FIELD_RES_Z) {
        return;
    }
    // g, b: the build cell of the extinction the solve used (cloud_field_uvw), so the cloud pass
    // maps positions with the solution's own cell while the SDF of a new build is pending.
    let e = textureLoad(in_field_ext, id, 0);
    var u = textureLoad(diff_u, id).r;
    // Clear cells next to cloud take the mean of their cloud neighbours: the camera sees the
    // outer tens of metres of a cloud, within a cell of clear air, and trilinear filtering
    // against a stored 0 halved the diffuse light at every surface. Cells far from cloud (the
    // field's distance, more than two cells) skip the neighbourhood.
    // The SDF may still be indexed by the previous build cell (it switches at edt_z, three
    // frames after the extinction): read it at this cell's position in its own grid.
    let sdf_cell = textureLoad(cloud_field_sdf, vec3(0), 0).ba;
    let qs = vec3<i32>(id) + vec3(vec2<i32>(e.ba - sdf_cell), 0);
    let far = field_in_range(qs) && textureLoad(cloud_field_sdf, qs, 0).r > 2.0 * cloud_field_voxel_m();
    if (e.r <= DIFF_SIGMA_MIN && far) {
        u = 0.0;
    } else if (e.r <= DIFF_SIGMA_MIN) {
        var sum = 0.0;
        var n = 0.0;
        for (var k = 0; k < 6; k++) {
            var o = vec3(0);
            o[k / 2] = select(-1, 1, (k & 1) == 1);
            let q = vec3<i32>(id) + o;
            if (field_in_range(q) && textureLoad(in_field_ext, q, 0).r > DIFF_SIGMA_MIN) {
                sum += textureLoad(diff_u, q).r;
                n += 1.0;
            }
        }
        u = select(0.0, sum / max(n, 1.0), n > 0.0);
    }
    textureStore(out_field_diff, id, vec4(u, e.ba, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn diffusion_reset(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= CLOUD_FIELD_RES_XY || id.y >= CLOUD_FIELD_RES_XY || id.z >= CLOUD_FIELD_RES_Z) {
        return;
    }
    textureStore(diff_u, id, vec4(0.0));
    if (all(id == vec3(0u))) {
        let cell = textureLoad(in_field_ext, id, 0).ba;
        textureStore(diff_state, vec3(0, 0, 0), vec4(cell.x));
        textureStore(diff_state, vec3(1, 0, 0), vec4(cell.y));
        textureStore(diff_state, vec3(2, 0, 0), vec4(0.0));
        textureStore(diff_state, vec3(3, 0, 0), vec4(0.0));
    }
}

// The grid follows the camera in whole blocks (cloud_field_cell): when the extinction is rebuilt
// at a new cell, the warm-started solution and the source move with it (they are indexed by
// cell), else the light sat two cells off the clouds and blinked until the SOR re-converged
// (every ~375 m of flight). Two passes, so no invocation reads the state another writes:
// shift copies into diff_tmp and records the offset, commit copies back and records the cell.
fn diffusion_shift_offset() -> vec2<i32> {
    return vec2<i32>(vec2(textureLoad(diff_state, vec3(2, 0, 0)).r, textureLoad(diff_state, vec3(3, 0, 0)).r));
}

@compute @workgroup_size(4, 4, 4)
fn diffusion_shift(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= CLOUD_FIELD_RES_XY || id.y >= CLOUD_FIELD_RES_XY || id.z >= CLOUD_FIELD_RES_Z) {
        return;
    }
    let cell = textureLoad(in_field_ext, vec3(0), 0).ba;
    let old = vec2(textureLoad(diff_state, vec3(0, 0, 0)).r, textureLoad(diff_state, vec3(1, 0, 0)).r);
    let o = vec2<i32>(round(cell - old));
    if (all(id == vec3(0u))) {
        textureStore(diff_state, vec3(2, 0, 0), vec4(f32(o.x)));
        textureStore(diff_state, vec3(3, 0, 0), vec4(f32(o.y)));
    }
    if (all(o == vec2(0))) {
        return;
    }
    // Cell id of the new grid is cell id + o of the old one (cloud_field_origin_from_cell).
    let q = vec3<i32>(id) + vec3(o, 0);
    let inside = field_in_range(q);
    let u = select(0.0, textureLoad(diff_u, q).r, inside);
    let src = select(0.0, textureLoad(diff_src, q).r, inside);
    textureStore(diff_tmp, vec3<i32>(id), vec4(u));
    textureStore(diff_tmp, vec3<i32>(id) + vec3(0, 0, i32(CLOUD_FIELD_RES_Z)), vec4(src));
}

@compute @workgroup_size(4, 4, 4)
fn diffusion_shift_commit(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= CLOUD_FIELD_RES_XY || id.y >= CLOUD_FIELD_RES_XY || id.z >= CLOUD_FIELD_RES_Z) {
        return;
    }
    let o = diffusion_shift_offset();
    if (all(o == vec2(0))) {
        return;
    }
    textureStore(diff_u, id, textureLoad(diff_tmp, vec3<i32>(id)));
    textureStore(diff_src, id, textureLoad(diff_tmp, vec3<i32>(id) + vec3(0, 0, i32(CLOUD_FIELD_RES_Z))));
    if (all(id == vec3(0u))) {
        let cell = textureLoad(in_field_ext, vec3(0), 0).ba;
        textureStore(diff_state, vec3(0, 0, 0), vec4(cell.x));
        textureStore(diff_state, vec3(1, 0, 0), vec4(cell.y));
    }
}
