//! Auto exposure
//!
//! This shader meters the scene's absolute luminance and moves an adapted exposure gradually
//! toward it. Tonemapping applies the difference between the adapted exposure and the exposure the
//! view was rendered with.
//!
//! The adapted exposure is computed in two passes:
//! * The compute_histogram pass calculates a histogram of the scene luminance in EV100,
//! taking into account the metering mask texture. The metering mask is a grayscale texture
//! that defines the areas of the screen that should be given more weight when calculating
//! the average luminance value. For example, the middle area of the screen might be more important
//! than the edges.
//! * The compute_average pass calculates the average luminance value of the scene, taking
//! into account the low_percent and high_percent settings. These settings define the
//! percentage of the histogram that should be excluded when calculating the average. This
//! is useful to avoid overexposure when you have a lot of shadows, or underexposure when you
//! have a lot of bright specular reflections.
//!
//! The target exposure displays that average at the compensated level, and the adapted exposure
//! moves toward it over real time.
//!
//! With auto white balance, the same passes meter the scene's average chromaticity and adapt a
//! white balance correction toward it.

import bevy_render::view::View;
@if(WORKING_COLOR_SPACE_REC2020)
import bevy_render::working_color_space::REC2020_TO_REC709;

/// Constant to convert RGB to luminance, taken from Real Time Rendering, Vol 4 pg. 278, 4th edition
const RGB_TO_LUM = vec3<f32>(0.2125, 0.7154, 0.0721);

/// The number of histogram bins; matches `HISTOGRAM_BIN_COUNT`. The first and last bins hold the
/// luminance below and above the metering range, the others split the range evenly.
const BIN_COUNT = 128u;

/// `log2(1.2)`. A camera at `ev100` stores luminance `L` as `L * 2^-ev100 / 1.2`
/// (`Exposure::exposure`).
const LOG2_EXPOSURE_SCALE = 0.26303441;

/// The EV100 of luminance `L` (cd/m²) is `log2(L) + 3`: a reflected-light meter at ISO 100 with
/// calibration constant K = 12.5, `log2(L * 100 / 12.5)`.
const EV100_OF_LOG2_LUMINANCE = 3.0;

/// The CIE 1931 xy chromaticity of the D65 white point.
const D65_XY = vec2<f32>(0.31272, 0.32903);

const REC709_TO_XYZ = mat3x3<f32>(
    0.4123908, 0.21263901, 0.019330819,
    0.35758434, 0.71516868, 0.11919478,
    0.18048079, 0.072192315, 0.95053215,
);

const XYZ_TO_REC709 = mat3x3<f32>(
    3.24096994, -0.96924364, 0.05563008,
    -1.53738318, 1.8759675, -0.20397696,
    -0.49861076, 0.04155506, 1.05697151,
);

/// The RGB to LMS matrices of `ColorGrading` white balance. Their rows sum to 1, so RGB white maps
/// to LMS (1, 1, 1).
const RGB_TO_LMS = mat3x3<f32>(
    0.311692, 0.0905138, 0.00764433,
    0.652085, 0.901341, 0.0486554,
    0.0362225, 0.00814478, 0.943700,
);
const LMS_TO_RGB = mat3x3<f32>(
    4.06305, -0.40791, -0.0118812,
    -2.93241, 1.40437, -0.0486532,
    -0.130646, 0.00353630, 1.0605344,
);

/// White balance weighs each pixel by its luminance at the adapted exposure, which displays the
/// metered average at 1, up to this value.
const CHROMA_LUMINANCE_CAP = 16.0;

/// Fixed-point scale of the workgroup chromaticity sums. A pixel adds at most 16 (the maximum
/// metering weight), so a workgroup of 256 pixels stays below 2^12 * 2^16 = 2^28.
const CHROMA_WORKGROUP_SCALE = 65536.0;

/// Fixed-point scale of the per-view chromaticity sums. The workgroup sums are divided by the pixel
/// count before they are added, so the total stays below 16 * 2^24 = 2^28.
const CHROMA_FLUSH_SCALE = 16777216.0;

struct AutoExposure {
    min_ev100: f32,
    inv_ev100_range: f32,
    ev100_range: f32,
    low_percent: f32,
    high_percent: f32,
    speed_brighten: f32,
    speed_darken: f32,
    exponential_transition_distance: f32,
    correction_min: f32,
    correction_max: f32,
    /// The camera's `Exposure`, in EV100, which the correction range is relative to.
    camera_ev100: f32,
    /// Real time since the previous frame, in seconds.
    delta_time: f32,
    long_term_speed_brighten: f32,
    long_term_speed_darken: f32,
    long_term_bound_brighten: f32,
    long_term_bound_darken: f32,
    /// Nonzero when the long-term adaptation bounds the adapted exposure.
    physiological: u32,
    white_balance_speed: f32,
    /// In cd/m².
    virtual_light_luminance: f32,
    /// Nonzero when auto white balance is on.
    white_balance: u32,
    /// Nonzero when no pipeline is waiting to compile, or the startup timeout has passed.
    pipelines_ready: u32,
}

struct CompensationCurve {
    min_ev100: f32,
    inv_ev100_range: f32,
    min_compensation: f32,
    compensation_range: f32,
}

/// The adaptation state, persistent across frames.
struct AutoExposureState {
    /// The adapted exposure, in EV100.
    ev100: f32,
    /// Zero until the first fully drawn frame with luminance above the bottom of the metering
    /// range sets `ev100` and `white_point` to their targets.
    initialized: u32,
    /// The adapted white point, in CIE 1931 xy.
    white_point: vec2<f32>,
    /// The long-term adapted exposure, in EV100.
    long_term_ev100: f32,
    /// The fixed-point sums of luminance-weighted x and y, and of the weights. `compute_histogram`
    /// adds to them and `compute_average` clears them.
    chroma_sums: array<atomic<u32>, 3>,
}

@group(0) @binding(0) var<uniform> settings: AutoExposure;

@group(0) @binding(1) var tex_color: texture_2d<f32>;

@group(0) @binding(2) var tex_mask: texture_2d<f32>;

@group(0) @binding(3) var tex_compensation: texture_1d<f32>;

@group(0) @binding(4) var<uniform> compensation_curve: CompensationCurve;

@group(0) @binding(5) var<storage, read_write> histogram: array<atomic<u32>, BIN_COUNT>;

@group(0) @binding(6) var<storage, read_write> state: AutoExposureState;

@group(0) @binding(7) var<storage, read_write> view: View;

/// The view's `ViewExposureTexture`: the factor from scene luminance to the tonemapping input.
@group(0) @binding(8) var exposure_texture: texture_storage_2d<r32float, write>;

var<workgroup> histogram_shared: array<atomic<u32>, BIN_COUNT>;

var<workgroup> chroma_shared: array<atomic<u32>, 3>;

/// Converts a working color space color to linear Rec.709.
fn working_to_rec709(color: vec3<f32>) -> vec3<f32> {
@if(WORKING_COLOR_SPACE_REC2020) {
    return REC2020_TO_REC709 * color;
} @else {
    return color;
}
}

/// For a given stored color, return the histogram bin index of its scene luminance.
fn color_to_bin(hdr: vec3<f32>, log2_exposure: f32) -> u32 {
    // Convert color to luminance
    let lum = dot(hdr, RGB_TO_LUM);

    // Zero, negative and NaN luminance have no logarithm; count them in the lowest bin.
    if !(lum > 0.0) {
        return 0u;
    }

    // Undo the view's exposure to get the scene luminance in EV100, and express it as a
    // position where [0.0, 1.0] spans the metering range.
    let ev100 = log2(lum) - log2_exposure + EV100_OF_LOG2_LUMINANCE;
    let position = (ev100 - settings.min_ev100) * settings.inv_ev100_range;

    // Map [0, 1) to [1, BIN_COUNT - 2], below the range to the first bin, and the top of the range
    // and above to the last bin.
    return u32(clamp(position * f32(BIN_COUNT - 2u) + 1.0, 0.0, f32(BIN_COUNT - 1u)));
}

/// The position of a histogram bin within the metering range, from 0.0 at the minimum to 1.0 at
/// the maximum. The first and last bins sit at the ends, the others at their centers.
fn bin_position(index: u32) -> f32 {
    return saturate((f32(index) - 0.5) / f32(BIN_COUNT - 2u));
}

/// Read the metering mask at the given UV coordinates, returning a weight for the histogram.
///
/// Since the histogram is summed in the compute_average step, there is a limit to the amount of
/// distinct values that can be represented. When using the chosen value of 16, the maximum
/// amount of pixels that can be weighted and summed is 2^32 / 16 = 16384^2.
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
    if local_invocation_index < BIN_COUNT {
        atomicStore(&histogram_shared[local_invocation_index], 0u);
    }
    if settings.white_balance != 0u && local_invocation_index < 3u {
        atomicStore(&chroma_shared[local_invocation_index], 0u);
    }

    // Wait for all workgroup threads to clear the shared histogram
    workgroupBarrier();

    let dim = vec2<u32>(textureDimensions(tex_color));
    let uv = vec2<f32>(global_invocation_id.xy) / vec2<f32>(dim);

    if global_invocation_id.x < dim.x && global_invocation_id.y < dim.y {
        let col = textureLoad(tex_color, vec2<i32>(global_invocation_id.xy), 0).rgb;
        let log2_exposure = log2(view.exposure);
        let rec709 = working_to_rec709(col);
        let index = color_to_bin(rec709, log2_exposure);
        let weight = metering_weight(uv);

        // Increment the shared histogram bin by the weight obtained from the metering mask
        atomicAdd(&histogram_shared[index], weight);

        if settings.white_balance != 0u {
            let xyz = REC709_TO_XYZ * max(rec709, vec3(0.0));
            let xyz_sum = xyz.x + xyz.y + xyz.z;
            // Skips black and infinite pixels, which have no chromaticity.
            if xyz_sum > 0.0 && xyz_sum < 3.4e38 {
                let relative_luminance =
                    xyz.y * exp2(-log2_exposure - state.ev100 - LOG2_EXPOSURE_SCALE);
                let luminance_weight =
                    saturate(relative_luminance / CHROMA_LUMINANCE_CAP) * f32(weight);
                let chroma = vec3(xyz.xy / xyz_sum, 1.0) * luminance_weight;
                let fixed_point = vec3<u32>(chroma * CHROMA_WORKGROUP_SCALE + 0.5);
                atomicAdd(&chroma_shared[0], fixed_point.x);
                atomicAdd(&chroma_shared[1], fixed_point.y);
                atomicAdd(&chroma_shared[2], fixed_point.z);
            }
        }
    }

    // Wait for all workgroup threads to finish updating the workgroup histogram
    workgroupBarrier();

    // Accumulate the workgroup histogram into the global histogram, one bin per invocation.
    // Note that the global histogram was not cleared at the beginning,
    // as it will be cleared in compute_average.
    if local_invocation_index < BIN_COUNT {
        let histogram_value = atomicLoad(&histogram_shared[local_invocation_index]);
        if histogram_value != 0u {
            atomicAdd(&histogram[local_invocation_index], histogram_value);
        }
    }

    if settings.white_balance != 0u && local_invocation_index < 3u {
        let sum = f32(atomicLoad(&chroma_shared[local_invocation_index])) / CHROMA_WORKGROUP_SCALE;
        let normalized = sum / (f32(dim.x) * f32(dim.y));
        atomicAdd(
            &state.chroma_sums[local_invocation_index],
            u32(normalized * CHROMA_FLUSH_SCALE + 0.5),
        );
    }
}

@compute @workgroup_size(1, 1, 1)
fn compute_average() {
    // The cumulative histogram is accumulated on the fly in the averaging loop rather than stored
    // in a function-local array: NVIDIA's shader compiler (driver 621.01) asserts on that array.
    var histogram_sum = 0u;
    for (var i=0u; i<BIN_COUNT; i+=1u) {
        histogram_sum += atomicLoad(&histogram[i]);
    }

    // A frame with every pixel in the lowest bin, such as a black frame before the scene has
    // loaded, doesn't start adaptation.
    let metered = atomicLoad(&histogram[0]) < histogram_sum;

    let first_index = u32(f32(histogram_sum) * settings.low_percent);
    let last_index = u32(f32(histogram_sum) * settings.high_percent);

    var count = 0u;
    var sum = 0.0;
    // Include the lowest bin at the minimum luminance when averaging.
    // The cumulative histogram before this bin is zero, clamped to first_index.
    var previous = first_index;
    // The sum of all bins up to and including bin i.
    var running_sum = 0u;
    for (var i=0u; i<BIN_COUNT; i+=1u) {
        running_sum += atomicLoad(&histogram[i]);

        // Clear the histogram bin for the next frame
        atomicStore(&histogram[i], 0u);

        // The number of pixels in the bin. The histogram values are clamped to
        // first_index and last_index to exclude the lowest and highest samples.
        let cumulative = clamp(running_sum, first_index, last_index);
        let bin_count = cumulative - previous;
        previous = cumulative;

        sum += f32(bin_count) * bin_position(i);
        count += bin_count;
    }

    // The average scene luminance of the included histogram samples, in EV100.
    var avg_ev100 = settings.min_ev100;

    if count > 0u {
        avg_ev100 = sum / f32(count) * settings.ev100_range + settings.min_ev100;
    }

    // The position in the compensation curve texture to sample for avg_ev100.
    let u = (avg_ev100 - compensation_curve.min_ev100) * compensation_curve.inv_ev100_range;
    let compensation = textureLoad(tex_compensation, i32(saturate(u) * 255.0), 0).r
        * compensation_curve.compensation_range
        + compensation_curve.min_compensation;

    // The target exposure stores the average luminance at 2^compensation:
    // log2(L) - ev100 - log2(1.2) = compensation.
    let target_ev100 = avg_ev100 - EV100_OF_LOG2_LUMINANCE - LOG2_EXPOSURE_SCALE - compensation;

    if settings.white_balance != 0u {
        auto_white_balance(avg_ev100);
    } else {
        // Turning auto white balance on starts from no correction.
        state.white_point = D65_XY;
    }

    var ev100 = target_ev100;
    var long_term_ev100 = target_ev100;
    if state.initialized != 0u {
        // Smoothly adjust the adapted exposure towards the target. A brighter scene raises the
        // exposure value.
        ev100 = adapt(state.ev100, target_ev100, settings.speed_brighten, settings.speed_darken);
        long_term_ev100 = ev100;
        if settings.physiological != 0u {
            long_term_ev100 = adapt(
                state.long_term_ev100,
                target_ev100,
                settings.long_term_speed_brighten,
                settings.long_term_speed_darken,
            );
            ev100 = clamp(
                ev100,
                long_term_ev100 - settings.long_term_bound_darken,
                long_term_ev100 + settings.long_term_bound_brighten,
            );
        }
    }

    // Keep the correction relative to the camera's `Exposure` within the configured range, and
    // store the clamped exposure so adaptation continues from what is displayed.
    let min_ev100 = settings.camera_ev100 - settings.correction_max;
    let max_ev100 = settings.camera_ev100 - settings.correction_min;
    ev100 = clamp(ev100, min_ev100, max_ev100);
    state.ev100 = ev100;
    state.long_term_ev100 = clamp(long_term_ev100, min_ev100, max_ev100);
    if metered && settings.pipelines_ready != 0u {
        state.initialized = 1u;
    }

    // The view was rendered with the camera's `Exposure` or, with pre-exposure, an earlier adapted
    // exposure. Apply the difference through the color grading exposure, which tonemapping reads.
    let view_ev100 = -log2(view.exposure) - LOG2_EXPOSURE_SCALE;
    view.color_grading.exposure += view_ev100 - ev100;
    textureStore(
        exposure_texture,
        vec2(0),
        vec4(view.exposure * exp2(view.color_grading.exposure), 0.0, 0.0, 0.0),
    );
}

/// Moves `ev100` toward `target_ev100`, at the speed for its direction in F-stops per second.
fn adapt(ev100: f32, target_ev100: f32, speed_brighten: f32, speed_darken: f32) -> f32 {
    let delta = target_ev100 - ev100;
    let speed = select(speed_darken, speed_brighten, delta > 0.0) * settings.delta_time;
    let exponential = min(speed / settings.exponential_transition_distance, 1.0);
    return ev100 + clamp(delta * exponential, -speed, speed);
}

/// Adapts the white point toward the metered chromaticity, and applies its correction to the view.
/// `avg_ev100` is the metered scene luminance.
fn auto_white_balance(avg_ev100: f32) {
    let sums = vec3(
        f32(atomicLoad(&state.chroma_sums[0])),
        f32(atomicLoad(&state.chroma_sums[1])),
        f32(atomicLoad(&state.chroma_sums[2])),
    );
    atomicStore(&state.chroma_sums[0], 0u);
    atomicStore(&state.chroma_sums[1], 0u);
    atomicStore(&state.chroma_sums[2], 0u);

    var measured = D65_XY;
    if sums.z > 0.0 {
        measured = sums.xy / sums.z;
    }
    let luminance = exp2(avg_ev100 - EV100_OF_LOG2_LUMINANCE);
    let virtual_light = settings.virtual_light_luminance;
    let target_xy = mix(measured, D65_XY, virtual_light / (virtual_light + luminance));

    var white_point = target_xy;
    if state.initialized != 0u {
        white_point = mix(
            state.white_point,
            target_xy,
            saturate(settings.white_balance_speed * settings.delta_time),
        );
    }
    state.white_point = white_point;

    // Right-multiplying applies the automatic correction before the color grading white balance.
    view.color_grading.balance = view.color_grading.balance * white_balance_matrix(white_point);
}

/// The linear Rec.709 color of the white point with CIE 1931 chromaticity `xy`, at luminance 1.
fn xy_to_rec709(xy: vec2<f32>) -> vec3<f32> {
    return XYZ_TO_REC709 * vec3(xy.x / xy.y, 1.0, (1.0 - xy.x - xy.y) / xy.y);
}

/// The correlated color temperature of a CIE 1931 chromaticity, from McCamy's approximation
/// ("Correlated color temperature as an explicit function of chromaticity coordinates", 1992).
///
/// The denominator stays negative. Chromaticities below y = 0.1858, such as deep blue light, would
/// otherwise read as very warm. The clamp on `n` keeps the result between 1632 K and 21334 K.
fn correlated_color_temperature(xy: vec2<f32>) -> f32 {
    let d = min(0.1858 - xy.y, -1e-6);
    let n = clamp((xy.x - 0.3320) / d, -1.2, 1.3);
    return ((449.0 * n + 3525.0) * n + 6823.3) * n + 5520.33;
}

/// The CIE 1931 chromaticity of a blackbody at temperature `t` in kelvin, from the cubic fit of
/// Kang et al. ("Design of advanced color temperature control system for HDTV applications",
/// 2002). Valid from 1667 K to 25000 K.
fn planckian_xy(t: f32) -> vec2<f32> {
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

/// The von Kries correction that maps light with chromaticity `adapted_xy` to D65.
///
/// The color temperature is limited to 2500 K to 7000 K, and the distance from the Planckian locus
/// to 0.05 in xy.
fn white_balance_matrix(adapted_xy: vec2<f32>) -> mat3x3<f32> {
    let temperature = correlated_color_temperature(adapted_xy);
    let locus = planckian_xy(clamp(temperature, 1667.0, 25000.0));
    let tint = clamp(adapted_xy - locus, vec2(-0.05), vec2(0.05));
    let white_xy = planckian_xy(clamp(temperature, 2500.0, 7000.0)) + tint;

    // The gains come from the same RGB to LMS matrix that converts the colors, so the white point
    // maps to D65 white. Both white points have luminance 1, so the correction keeps luminance.
    // A warm, green white can fall outside Rec.709. Clamping it keeps every gain positive.
    let white = max(xy_to_rec709(white_xy), vec3(0.0));
    let gain = (RGB_TO_LMS * xy_to_rec709(D65_XY)) / (RGB_TO_LMS * white);
    let scaled = mat3x3<f32>(gain * RGB_TO_LMS[0], gain * RGB_TO_LMS[1], gain * RGB_TO_LMS[2]);
    return LMS_TO_RGB * scaled;
}
