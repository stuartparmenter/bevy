// Auto exposure
//
// This shader computes an auto exposure value for the current frame,
// which is then used as an exposure correction in the tone mapping shader.
//
// The auto exposure value is computed in two passes:
// * The compute_histogram pass calculates a histogram of the luminance values in the scene,
// taking into account the metering mask texture. The metering mask is a grayscale texture
// that defines the areas of the screen that should be given more weight when calculating
// the average luminance value. For example, the middle area of the screen might be more important
// than the edges.
// * The compute_average pass calculates the average luminance value of the scene, taking
// into account the low_percent and high_percent settings. These settings define the
// percentage of the histogram that should be excluded when calculating the average. This
// is useful to avoid overexposure when you have a lot of shadows, or underexposure when you
// have a lot of bright specular reflections.
//
// The final target_exposure is finally used to smoothly adjust the exposure value over time.
// Optionally, a two-stage physiological adaptation model bounds this short-term exposure by a
// slowly moving long-term adaptation envelope (see `PhysiologicalAdaptation` on the Rust side).
//
// The same metering pass optionally measures the scene's luminance-weighted average
// chromaticity for auto white balance (see `AutoWhiteBalance` on the Rust side): the
// compute_histogram pass accumulates fixed-point Yxy sums with the same metering-mask weights,
// and the compute_average pass blends in a faint D65 "virtual light" anchor, temporally adapts
// the white-point chromaticity, and composes a von Kries correction matrix into the view's
// color-grading balance matrix. Every auto-white-balance statement is gated on the
// `awb_enabled` uniform flag, so the auto-exposure-only configuration runs no white-balance
// arithmetic at all.

#import bevy_render::view::View
#import bevy_render::globals::Globals

// Constant to convert RGB to luminance, taken from Real Time Rendering, Vol 4 pg. 278, 4th edition
const RGB_TO_LUM = vec3<f32>(0.2125, 0.7154, 0.0721);

struct AutoExposure {
    min_log_lum: f32,
    inv_log_lum_range: f32,
    log_lum_range: f32,
    low_percent: f32,
    high_percent: f32,
    speed_up: f32,
    speed_down: f32,
    exponential_transition_distance: f32,
    metering_bias: f32,
    long_term_speed_up: f32,
    long_term_speed_down: f32,
    long_term_bound_up: f32,
    long_term_bound_down: f32,
    physiological: u32,
    // Auto white balance chromaticity adaptation speed, per second.
    awb_speed: f32,
    // Auto white balance virtual-light anchor luminance, scene-linear units.
    awb_anchor: f32,
    // Non-zero when auto white balance is enabled for this view.
    awb_enabled: u32,
    // Tail padding to 80 bytes. WGSL rounds a uniform struct's size up to a
    // multiple of 16 anyway; spelling it out keeps this struct field-for-field
    // identical to its CPU mirror, which has to pad explicitly.
    pad_0: u32,
    pad_1: u32,
    pad_2: u32,
}

// The per-view adaptation state, persisted on the GPU from frame to frame.
struct AutoExposureState {
    // The smoothed short-term exposure correction, in EV.
    exposure: f32,
    // The long-term physiological adaptation envelope, in EV.
    long_term: f32,
    // The adapted white-point chromaticity (CIE 1931 xy) for auto white
    // balance.
    chroma_x: f32,
    chroma_y: f32,
    // Fixed-point accumulators for the auto white balance measurement: the
    // luminance-weighted sums of x, of y, and the luminance weight itself
    // (Yxy space). Flushed by compute_histogram, drained and cleared by
    // compute_average.
    chroma_sums: array<atomic<u32>, 3>,
}

struct CompensationCurve {
    min_log_lum: f32,
    inv_log_lum_range: f32,
    min_compensation: f32,
    compensation_range: f32,
}

@group(0) @binding(0) var<uniform> globals: Globals;

@group(0) @binding(1) var<uniform> settings: AutoExposure;

@group(0) @binding(2) var tex_color: texture_2d<f32>;

@group(0) @binding(3) var tex_mask: texture_2d<f32>;

@group(0) @binding(4) var tex_compensation: texture_1d<f32>;

@group(0) @binding(5) var<uniform> compensation_curve: CompensationCurve;

@group(0) @binding(6) var<storage, read_write> histogram: array<atomic<u32>, 64>;

@group(0) @binding(7) var<storage, read_write> state: AutoExposureState;

@group(0) @binding(8) var<storage, read_write> view: View;

var<workgroup> histogram_shared: array<atomic<u32>, 64>;

var<workgroup> chroma_shared: array<atomic<u32>, 3>;

// === Auto white balance constants and helpers ===============================

// Fixed-point scale used by the per-pixel accumulation into chroma_shared.
// Per-pixel contributions are bounded by the maximum metering weight (16), so
// a 256-pixel workgroup stays below 16 * 256 * 2^16 = 2^28 < 2^32.
const CHROMA_WORKGROUP_SCALE: f32 = 65536.0;

// Fixed-point scale used by the per-workgroup flush into the state's chroma
// accumulators. The flushed values are normalized by the total pixel count,
// so the accumulated sums are bounded by 16 * 2^24 = 2^28 < 2^32.
const CHROMA_FLUSH_SCALE: f32 = 16777216.0;

// The CIE 1931 xy chromaticity of the D65 white point, matching `D65_XY` in
// `bevy_render/src/view/mod.rs` (and `D65_XY` in `buffers.rs`); keep in sync.
const AWB_D65_XY = vec2<f32>(0.31272, 0.32903);

// Rows of the linear-RGB -> CIE 1931 XYZ matrices (D65 white point) for the
// two supported working color spaces, derived from the Rec.709 (IEC
// 61966-2-1 / ITU-R BT.709) and ITU-R BT.2020 primaries.
const AWB_REC709_TO_X = vec3<f32>(0.4123907992659595, 0.35758433938387796, 0.1804807884018343);
const AWB_REC709_TO_Y = vec3<f32>(0.21263900587151036, 0.7151686787677559, 0.07219231536073371);
const AWB_REC709_TO_Z = vec3<f32>(0.01933081871559185, 0.11919477979462599, 0.9505321522496607);
const AWB_REC2020_TO_X = vec3<f32>(0.6369580483012913, 0.14461690358620838, 0.16888097516417205);
const AWB_REC2020_TO_Y = vec3<f32>(0.26270021201126703, 0.677998071518871, 0.059301716469861945);
const AWB_REC2020_TO_Z = vec3<f32>(0.0, 0.028072693049087508, 1.0609850577107909);

// Rows of the inverse of the `AWB_REC709_TO_*` matrix (CIE 1931 XYZ ->
// linear Rec.709 RGB, D65 white point), used to express white points in the
// working RGB space when deriving the von Kries gains below.
const AWB_XYZ_TO_REC709_R = vec3<f32>(3.2409699419045208, -1.537383177570093, -0.49861076029300311);
const AWB_XYZ_TO_REC709_G = vec3<f32>(-0.96924363628087962, 1.8759675015077208, 0.041555057407175612);
const AWB_XYZ_TO_REC709_B = vec3<f32>(0.055630079696993608, -0.20397695888897655, 1.0569715142428784);

// The CAM16-derived RGB -> LMS / LMS -> RGB matrices, copied verbatim from
// `RGB_TO_LMS` / `LMS_TO_RGB` in `bevy_render/src/view/mod.rs` — the same
// matrices the CPU uses to build the user's static white-balance matrix from
// `ColorGrading`'s temperature/tint; keep them in sync. Note that this is
// bevy's own white-normalized variant of the CAM16 LMS basis (its rows sum
// to 1.0, mapping RGB white to LMS (1, 1, 1)), NOT the raw CAM16 transform
// of CIE XYZ — the von Kries gains below must therefore be derived in THIS
// basis, not in raw CAM16 LMS.
const AWB_RGB_TO_LMS = mat3x3<f32>(
    vec3(0.311692, 0.0905138, 0.00764433),
    vec3(0.652085, 0.901341, 0.0486554),
    vec3(0.0362225, 0.00814478, 0.943700),
);
const AWB_LMS_TO_RGB = mat3x3<f32>(
    vec3(4.06305, -0.40791, -0.0118812),
    vec3(-2.93241, 1.40437, -0.0486532),
    vec3(-0.130646, 0.00353630, 1.0605344),
);

// The linear Rec.709 RGB coordinates of the white point with CIE 1931
// chromaticity `xy` and unit luminance (Y = 1).
fn awb_xy_to_rec709(xy: vec2<f32>) -> vec3<f32> {
    let xyz = vec3(xy.x / xy.y, 1.0, (1.0 - xy.x - xy.y) / xy.y);
    return vec3(
        dot(xyz, AWB_XYZ_TO_REC709_R),
        dot(xyz, AWB_XYZ_TO_REC709_G),
        dot(xyz, AWB_XYZ_TO_REC709_B),
    );
}

// The correlated color temperature of a CIE 1931 chromaticity, using
// McCamy's approximation (C. S. McCamy 1992, "Correlated color temperature
// as an explicit function of chromaticity coordinates", Color Research &
// Application 17(2)). The denominator is clamped to stay negative: every
// real illuminant sits above the y = 0.1858 epicenter, so chromaticities
// below it (deep blue/violet scenes) must keep reading as cool — letting
// the sign flip would classify them as extremely warm and drive the
// correction in the wrong direction. The intermediate `n` is clamped so
// that extreme chromaticities produce a finite CCT inside the
// Planckian-locus approximation's validity range.
fn awb_cct(xy: vec2<f32>) -> f32 {
    let d = min(0.1858 - xy.y, -1e-6);
    let n = clamp((xy.x - 0.3320) / d, -1.2, 1.3);
    return ((449.0 * n + 3525.0) * n + 6823.3) * n + 5520.33;
}

// The CIE 1931 xy chromaticity of a Planckian (blackbody) radiator at the
// given correlated color temperature, using the cubic approximation of
// Kang et al. 2002, "Design of advanced color temperature control system
// for HDTV applications", J. Korean Phys. Soc. 41(6) (valid 1667 K-25000 K).
fn awb_planckian_xy(cct: f32) -> vec2<f32> {
    let t = cct;
    let t2 = t * t;
    let t3 = t2 * t;
    var x = 0.0;
    if t <= 4000.0 {
        x = -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.179910;
    } else {
        x = -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390;
    }
    let x2 = x * x;
    let x3 = x2 * x;
    var y = 0.0;
    if t <= 2222.0 {
        y = -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683;
    } else if t <= 4000.0 {
        y = -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867;
    } else {
        y = 3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483;
    }
    return vec2(x, y);
}

// The white-balance correction matrix for a scene whose adapted illuminant
// chromaticity is `adapted_xy`. The CCT is clamped to the 2500 K - 7000 K
// output range of typical real-camera AWB specifications, the off-locus tint
// component is preserved (clamped to +/-0.05 in xy for stability), and the
// resulting white point is neutralized towards D65 by a von Kries adaptation
// in bevy's CAM16-derived RGB <-> LMS basis (the same matrix pair the CPU
// uses for `ColorGrading`'s temperature/tint).
fn awb_balance_matrix(adapted_xy: vec2<f32>) -> mat3x3<f32> {
    let cct = awb_cct(adapted_xy);
    let locus = awb_planckian_xy(clamp(cct, 1667.0, 25000.0));
    let tint = clamp(adapted_xy - locus, vec2(-0.05), vec2(0.05));
    let white_xy = awb_planckian_xy(clamp(cct, 2500.0, 7000.0)) + tint;

    // Von Kries adaptation: independent gain on each cone response, scaling
    // the estimated illuminant to D65. The gains MUST be derived in the same
    // LMS basis the `AWB_LMS_TO_RGB * diag(gain) * AWB_RGB_TO_LMS` sandwich
    // uses, so both white points are mapped through `AWB_RGB_TO_LMS` from
    // their unit-luminance Rec.709 coordinates. (Deriving the gains in the
    // raw CAM16 LMS basis instead — whose cone axes genuinely differ from
    // bevy's white-normalized variant — only partially neutralizes the
    // illuminant: a converged correction would leave tungsten light at an
    // R/B channel ratio of ~1.54 instead of ~1.0.) Both vectors use Y = 1,
    // so the correction preserves the white point's luminance. A D65 input
    // yields a gain of exactly 1.0 (identical expressions on both sides).
    let gain = (AWB_RGB_TO_LMS * awb_xy_to_rec709(AWB_D65_XY))
        / (AWB_RGB_TO_LMS * awb_xy_to_rec709(white_xy));
    let scaled = mat3x3<f32>(
        gain * AWB_RGB_TO_LMS[0],
        gain * AWB_RGB_TO_LMS[1],
        gain * AWB_RGB_TO_LMS[2],
    );
    return AWB_LMS_TO_RGB * scaled;
}

// =============================================================================

// For a given color, return the histogram bin index
fn color_to_bin(hdr: vec3<f32>) -> u32 {
    // Convert color to luminance, using the Y row that matches the working
    // space the main texture holds (the same choice the white-balance
    // measurement below makes).
#ifdef WORKING_COLOR_SPACE_REC2020
    let lum = dot(hdr, AWB_REC2020_TO_Y);
#else
    let lum = dot(hdr, RGB_TO_LUM);
#endif

    if lum < exp2(settings.min_log_lum) {
        return 0u;
    }

    // Calculate the log_2 luminance and express it as a value in [0.0, 1.0]
    // where 0.0 represents the minimum luminance, and 1.0 represents the max.
    let log_lum = saturate((log2(lum) - settings.min_log_lum) * settings.inv_log_lum_range);

    // Map [0, 1] to [1, 63]. The zeroth bin is handled by the epsilon check above.
    return u32(log_lum * 62.0 + 1.0);
}

// Read the metering mask at the given UV coordinates, returning a weight for the histogram.
//
// Since the histogram is summed in the compute_average step, there is a limit to the amount of
// distinct values that can be represented. When using the chosen value of 16, the maximum
// amount of pixels that can be weighted and summed is 2^32 / 16 = 16384^2.
fn metering_weight(coords: vec2<f32>) -> u32 {
    let pos = vec2<i32>(coords * vec2<f32>(textureDimensions(tex_mask)));
    let mask = textureLoad(tex_mask, pos, 0).r;
    return u32(mask * 16.0);
}

@compute @workgroup_size(16, 16, 1)
fn compute_histogram(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
) {
    // Clear the workgroup shared histogram
    if local_invocation_index < 64 {
        atomicStore(&histogram_shared[local_invocation_index], 0u);
    }

    // Clear the workgroup shared chroma accumulators (auto white balance).
    // `settings.awb_enabled` comes from a uniform buffer, so all the auto
    // white balance branches below are uniform control flow and the shared
    // barriers stay valid.
    if settings.awb_enabled != 0u && local_invocation_index < 3u {
        atomicStore(&chroma_shared[local_invocation_index], 0u);
    }

    // Wait for all workgroup threads to clear the shared histogram
    workgroupBarrier();

    let dim = vec2<u32>(textureDimensions(tex_color));
    let uv = vec2<f32>(global_invocation_id.xy) / vec2<f32>(dim);

    if global_invocation_id.x < dim.x && global_invocation_id.y < dim.y {
        let col = textureLoad(tex_color, vec2<i32>(global_invocation_id.xy), 0).rgb;
        let index = color_to_bin(col);
        let weight = metering_weight(uv);

        // Increment the shared histogram bin by the weight obtained from the metering mask
        atomicAdd(&histogram_shared[index], weight);

        // Accumulate the auto white balance chromaticity measurement, reusing
        // the texel and metering-mask weight already loaded for the histogram.
        // No barriers run inside this block, so nesting the non-uniform flag
        // check here keeps the workgroup barriers below in uniform control
        // flow.
        if settings.awb_enabled != 0u {
            let awb_col = max(col, vec3(0.0));

            // The CIE 1931 XYZ of the working-space color (linear Rec.709 by
            // default, linear Rec.2020 when the project opted into the wide
            // working space).
#ifdef WORKING_COLOR_SPACE_REC2020
            let big_x = dot(awb_col, AWB_REC2020_TO_X);
            let big_y = dot(awb_col, AWB_REC2020_TO_Y);
            let big_z = dot(awb_col, AWB_REC2020_TO_Z);
#else
            let big_x = dot(awb_col, AWB_REC709_TO_X);
            let big_y = dot(awb_col, AWB_REC709_TO_Y);
            let big_z = dot(awb_col, AWB_REC709_TO_Z);
#endif

            let xyz_sum = big_x + big_y + big_z;
            if xyz_sum > 0.0 {
                // Luminance-weighted Yxy measurement: each pixel contributes
                // its chromaticity (x, y) weighted by the metering mask and by
                // its luminance, normalized (and saturated) against the top of
                // the metering range so the fixed-point sums cannot overflow.
                let y_cap = exp2(settings.min_log_lum + settings.log_lum_range);
                let y_norm = clamp(big_y / y_cap, 0.0, 1.0);
                let lum_weight = y_norm * f32(weight);
                let contribution = vec3(big_x / xyz_sum, big_y / xyz_sum, 1.0) * lum_weight;

                let fixed_point = contribution * CHROMA_WORKGROUP_SCALE + vec3(0.5);
                atomicAdd(&chroma_shared[0], u32(fixed_point.x));
                atomicAdd(&chroma_shared[1], u32(fixed_point.y));
                atomicAdd(&chroma_shared[2], u32(fixed_point.z));
            }
        }
    }

    // Wait for all workgroup threads to finish updating the workgroup histogram
    workgroupBarrier();

    // Accumulate the workgroup histogram into the global histogram.
    // Note that the global histogram was not cleared at the beginning,
    // as it will be cleared in compute_average. The guard matches the clear
    // above: the workgroup has 256 invocations but only 64 bins, and an
    // unguarded flush would clamp lanes 64..255 onto bin 63 (naga's default
    // bounds-check policy), inflating the brightest bin ~193x.
    if local_invocation_index < 64u {
        atomicAdd(
            &histogram[local_invocation_index],
            atomicLoad(&histogram_shared[local_invocation_index]),
        );
    }

    // Accumulate the workgroup chroma sums into the per-view state
    // accumulators, converting from the workgroup fixed-point scale to the
    // flush scale and normalizing by the total pixel count for headroom (the
    // normalization cancels out exactly in compute_average).
    if settings.awb_enabled != 0u && local_invocation_index < 3u {
        let workgroup_sum = f32(atomicLoad(&chroma_shared[local_invocation_index])) / CHROMA_WORKGROUP_SCALE;
        let normalized = workgroup_sum / (f32(dim.x) * f32(dim.y));
        atomicAdd(&state.chroma_sums[local_invocation_index], u32(normalized * CHROMA_FLUSH_SCALE + 0.5));
    }
}

@compute @workgroup_size(1, 1, 1)
fn compute_average(@builtin(local_invocation_index) local_index: u32) {
    var histogram_sum = 0u;

    // Calculate the cumulative histogram and clear the histogram bins.
    // Each bin in the cumulative histogram contains the sum of all bins up to that point.
    // This way we can quickly exclude the portion of lowest and highest samples as required by
    // the low_percent and high_percent settings.
    for (var i=0u; i<64u; i+=1u) {
        histogram_sum += atomicLoad(&histogram[i]);
        atomicStore(&histogram_shared[i], histogram_sum);

        // Clear the histogram bin for the next frame
        atomicStore(&histogram[i], 0u);
    }

    let first_index = u32(f32(histogram_sum) * settings.low_percent);
    let last_index = u32(f32(histogram_sum) * settings.high_percent);

    var count = 0u;
    var sum = 0.0;
    for (var i=1u; i<64u; i+=1u) {
        // The number of pixels in the bin. The histogram values are clamped to
        // first_index and last_index to exclude the lowest and highest samples.
        let bin_count =
            clamp(atomicLoad(&histogram_shared[i]), first_index, last_index) -
            clamp(atomicLoad(&histogram_shared[i - 1u]), first_index, last_index);

        sum += f32(bin_count) * f32(i);
        count += bin_count;
    }

    var avg_lum = settings.min_log_lum;

    if count > 0u {
        // The average luminance of the included histogram samples.
        avg_lum = sum / (f32(count) * 63.0)
            * settings.log_lum_range
            + settings.min_log_lum;
    }

    // A constant EV offset on the metered luminance, applied before the compensation
    // curve is sampled. The branch keeps the default configuration on the exact
    // histogram average.
    if settings.metering_bias != 0.0 {
        avg_lum += settings.metering_bias;
    }

    // The position in the compensation curve texture to sample for avg_lum.
    let u = (avg_lum - compensation_curve.min_log_lum) * compensation_curve.inv_log_lum_range;

    // The target exposure is the negative of the average log luminance.
    // The compensation value is added to the target exposure to adjust the exposure for
    // artistic purposes.
    let target_exposure = textureLoad(tex_compensation, i32(saturate(u) * 255.0), 0).r
        * compensation_curve.compensation_range
        + compensation_curve.min_compensation
        - avg_lum;

    // Smoothly adjust the short-term `exposure` towards the `target_exposure`
    var exposure = state.exposure;
    let delta = target_exposure - exposure;
    if target_exposure > exposure {
        let speed_down = settings.speed_down * globals.delta_time;
        let exp_down = speed_down / settings.exponential_transition_distance;
        exposure = exposure + min(speed_down, delta * exp_down);
    } else {
        let speed_up = settings.speed_up * globals.delta_time;
        let exp_up = speed_up / settings.exponential_transition_distance;
        exposure = exposure + max(-speed_up, delta * exp_up);
    }

    // Track the long-term physiological adaptation envelope: a slow, asymmetric follower
    // of the short-term exposure, modeling receptor sensitivity (photopigment bleaching
    // and recovery). The envelope is updated even when physiological adaptation is
    // disabled — it never feeds back into the exposure in that case — so that enabling
    // it at runtime starts from an envelope that already follows the current exposure.
    var long_term = state.long_term;
    let long_term_delta = exposure - long_term;
    if exposure > long_term {
        let long_term_speed_down = settings.long_term_speed_down * globals.delta_time;
        let long_term_exp_down = long_term_speed_down / settings.exponential_transition_distance;
        long_term = long_term + min(long_term_speed_down, long_term_delta * long_term_exp_down);
    } else {
        let long_term_speed_up = settings.long_term_speed_up * globals.delta_time;
        let long_term_exp_up = long_term_speed_up / settings.exponential_transition_distance;
        long_term = long_term + max(-long_term_speed_up, long_term_delta * long_term_exp_up);
    }

    // Two-stage adaptation: the long-term envelope bounds the short-term result, so e.g.
    // dark scenes stay dark — even if the short-term stage wants to lift them — until the
    // long-term envelope has had time to adapt.
    if settings.physiological != 0u {
        exposure = clamp(
            exposure,
            long_term - settings.long_term_bound_down,
            long_term + settings.long_term_bound_up,
        );
    }

    state.exposure = exposure;
    state.long_term = long_term;

    // Apply the exposure to the color grading settings, from where it will be used for the color
    // grading pass.
    view.color_grading.exposure += exposure;

    // Auto white balance. Gated on a uniform flag, so the auto-exposure-only
    // configuration executes exactly the statements above and nothing else.
    if settings.awb_enabled != 0u {
        // Drain and clear the fixed-point chroma accumulators (see
        // compute_histogram for the encoding).
        let sum_yx = f32(atomicLoad(&state.chroma_sums[0])) / CHROMA_FLUSH_SCALE;
        let sum_yy = f32(atomicLoad(&state.chroma_sums[1])) / CHROMA_FLUSH_SCALE;
        let sum_y = f32(atomicLoad(&state.chroma_sums[2])) / CHROMA_FLUSH_SCALE;
        atomicStore(&state.chroma_sums[0], 0u);
        atomicStore(&state.chroma_sums[1], 0u);
        atomicStore(&state.chroma_sums[2], 0u);

        // The mean metered scene luminance in scene-linear units. The sums
        // were normalized by the total pixel count (for fixed-point headroom)
        // and by the top of the metering range; histogram_sum is the total
        // metering weight over the same pixels, so the pixel count cancels
        // out exactly.
        let dim = vec2<f32>(textureDimensions(tex_color));
        var lum_scale = 0.0;
        if histogram_sum > 0u {
            let y_cap = exp2(settings.min_log_lum + settings.log_lum_range);
            lum_scale = dim.x * dim.y * y_cap / f32(histogram_sum);
        }
        let scene_luminance = sum_y * lum_scale;

        // Blend the faint D65 "virtual light" anchor into the measurement as
        // one more luminance-weighted Yxy reference — GT7's dark-scene
        // stability mechanism. The anchor's relative weight scales with the
        // inverse of the mean scene luminance: negligible in bright scenes,
        // dominant in near-dark ones (where the measurement would be noise).
        let denom = scene_luminance + settings.awb_anchor;
        if denom > 0.0 {
            let target_x = (sum_yx * lum_scale + settings.awb_anchor * AWB_D65_XY.x) / denom;
            let target_y = (sum_yy * lum_scale + settings.awb_anchor * AWB_D65_XY.y) / denom;

            // Temporal adaptation of the chromaticity only — luminance
            // adaptation is auto exposure's job. Exponential approach,
            // clamped so a single frame never overshoots.
            let alpha = saturate(settings.awb_speed * globals.delta_time);
            state.chroma_x += (target_x - state.chroma_x) * alpha;
            state.chroma_y += (target_y - state.chroma_y) * alpha;
        }

        // Compose the automatic correction with the artist-authored white
        // balance. The tonemapping pass applies `balance * color`, so
        // right-multiplying puts the automatic correction (towards neutral)
        // innermost: it corrects the image first, and the user's manual
        // temperature/tint grade applies on top of the corrected image.
        view.color_grading.balance = view.color_grading.balance
            * awb_balance_matrix(vec2(state.chroma_x, state.chroma_y));
    }
}
