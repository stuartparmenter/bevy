// Display-encoding pass: paper-white-relative display-linear color, in the
// tone-map operator's output space with UI already composited, to an encoded
// display signal. The `display_encoding` module docs describe where the pass
// sits in the display pipeline.
//
// There is no sRGB transfer def: the hardware applies the sRGB OETF on the
// upscaling blit's `*UnormSrgb` writeback, so plain SDR views never run this
// pass.

import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
import bevy_render::view::display_target::DisplayTargetUniform;
import bevy_render::transfer_functions::{scrgb_encode, pq_inverse_eotf_from_nits, srgb_oetf_extended};
import bevy_render::working_color_space::{REC709_TO_REC2020, REC2020_TO_REC709, REC709_TO_DISPLAYP3, REC2020_TO_DISPLAYP3};
@if(SRGB_TO_LINEAR)
import bevy_render::color_operations::srgb_to_linear;
@if(OKLAB_TO_LINEAR)
import bevy_render::color_operations::oklab_to_linear_rgb;

@group(0) @binding(0) var in_texture: texture_2d<f32>;
@group(0) @binding(1) var in_sampler: sampler;
// Per-view display-target calibration. The gamut and transfer are compile-time
// shader defs, so paper white is the only value read at runtime. The uniform
// producer `prepare_view_display_targets` sanitizes it (finite, positive,
// <= 10000) with the same rules the tone-map operators fold at prepare time, so
// the seam scale factors cancel exactly.
@group(0) @binding(2) var<uniform> display_target: DisplayTargetUniform;

// Out-of-gamut chroma compression in the style of the ACES 1.3 Reference Gamut
// Compression (Academy S-2020-001 / ACES "RGC", aces-dev lib/RGC_common.ctl).
// Each channel's distance from the achromatic axis is `dist = (ach - c) / ach`
// with `ach = max(r, g, b)`. It is 0 on the axis, 1 exactly on the gamut
// boundary (channel value 0), and above 1 outside the gamut. The parametric
// power curve compresses distances above the threshold so that `dist == limit`
// lands exactly on the boundary.
//
// Thresholds and power are the published ACES RGC values. The ACES limits
// (1.147 / 1.264 / 1.312) were derived from digital-cinema camera gamuts and
// under-cover the Rec.2020 to Rec.709 contraction this pass performs: the
// Rec.2020 hull reaches a distance of ~1.594 in the cyan direction when
// expressed in Rec.709 coordinates. The limits below are re-derived from the
// Rec.2020 hull maxima (~1.594 / ~1.087 / ~1.117) plus headroom.
//
// CPU mirror and tests: `gamut_compression.rs` next to this shader. Keep both
// in sync.
@if(DISPLAY_GAMUT_COMPRESSION)
const GAMUT_COMPRESSION_THRESHOLD = vec3<f32>(0.815, 0.803, 0.880);
@if(DISPLAY_GAMUT_COMPRESSION)
const GAMUT_COMPRESSION_POWER: f32 = 1.2;
// Kept for the derivation only: the shader consumes the precomputed scales.
@if(DISPLAY_GAMUT_COMPRESSION)
const GAMUT_COMPRESSION_LIMIT = vec3<f32>(1.62, 1.10, 1.13);
// scale = (limit - thr) / (((1 - thr) / (limit - thr))^(-power) - 1)^(1/power),
// evaluated in f64. See `compression_scale` in gamut_compression.rs; a test
// locks these literals to the closed form.
@if(DISPLAY_GAMUT_COMPRESSION)
const GAMUT_COMPRESSION_SCALE = vec3<f32>(0.21634937, 0.43270176, 0.18745117);

// The ACES RGC parametric compression curve, vectorized over the three chroma
// directions. Monotonically increasing above the threshold, approaching
// `threshold + scale` asymptotically. The `max` keeps `pow` off negative bases,
// so below the threshold this returns `threshold`. `gamut_compress` selects the
// original value there.
@if(DISPLAY_GAMUT_COMPRESSION)
fn gamut_compress_distance(dist: vec3<f32>) -> vec3<f32> {
    let nd = max(dist - GAMUT_COMPRESSION_THRESHOLD, vec3(0.0)) / GAMUT_COMPRESSION_SCALE;
    let p = pow(nd, vec3(GAMUT_COMPRESSION_POWER));
    return GAMUT_COMPRESSION_THRESHOLD
        + GAMUT_COMPRESSION_SCALE * nd / pow(1.0 + p, vec3(1.0 / GAMUT_COMPRESSION_POWER));
}

// Compresses out-of-gamut colors (negative components) toward the achromatic
// axis at constant `max(r, g, b)`. Channels whose distance is below the
// threshold are returned bit-identically. Colors whose distance does not exceed
// the limit land inside the target gamut. Hue is only approximately preserved;
// see `DisplayGamutCompression` in mod.rs.
@if(DISPLAY_GAMUT_COMPRESSION)
fn gamut_compress(rgb: vec3<f32>) -> vec3<f32> {
    let achromatic = max(rgb.r, max(rgb.g, rgb.b));
    if achromatic <= 0.0 {
        // No positive channel to compress toward. The final safety clip below
        // handles all-non-positive colors, except under extended sRGB.
        return rgb;
    }
    let dist = (vec3(achromatic) - rgb) / achromatic;
    let compressed = vec3(achromatic) - gamut_compress_distance(dist) * achromatic;
    return select(compressed, rgb, dist < GAMUT_COMPRESSION_THRESHOLD);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(in_texture, in_sampler, in.uv);

    // 1. Decode the compositing space, if the main texture is not already
    // display-linear. Same defs and math as the upscaling blit.
@if(SRGB_TO_LINEAR) {
    color = vec4(srgb_to_linear(color.rgb), color.a);
}
@if(OKLAB_TO_LINEAR) {
    color = vec4(oklab_to_linear_rgb(color.rgb), color.a);
}

    var rgb = color.rgb;

    // 2. Gamut transform: tone-map-output primaries to display primaries.
    //
    // Input contract: the tone-mapping pass emits Rec.709 display-linear,
    // except `Tonemapping::GranTurismo7` on an HDR-transfer target, which emits
    // its native linear Rec.2020 through the `TONEMAP_OUTPUT_REC2020` path in
    // gt7.wesl. Prepare keys exactly one of the defs below, or none for an
    // identity transform.
@if(DISPLAY_GAMUT_REC2020) {
    rgb = REC709_TO_REC2020 * rgb;
}
@if(GAMUT_REC2020_TO_REC709) {
    // A contraction: GT7's Rec.2020 output onto the Rec.709-coordinate scRGB
    // or extended-sRGB signal. It can produce negative components, so prepare
    // normally keys in `DISPLAY_GAMUT_COMPRESSION` below.
    rgb = REC2020_TO_REC709 * rgb;
}
@if(DISPLAY_GAMUT_DISPLAYP3) {
    // Rec.709 source to a Display-P3 signal (the `ExtendedDisplayP3` color
    // space). An expansion, so the result is in gamut by construction.
    rgb = REC709_TO_DISPLAYP3 * rgb;
}
@if(GAMUT_REC2020_TO_DISPLAYP3) {
    // GT7's Rec.2020 source to a Display-P3 signal. A contraction, handled like
    // the Rec.2020 to Rec.709 case above.
    rgb = REC2020_TO_DISPLAYP3 * rgb;
}

    // 3. Out-of-gamut handling. The compression def is pushed only when the
    // gamut stage can produce out-of-gamut colors, or under
    // `DisplayGamutCompression::Always`.
@if(DISPLAY_GAMUT_COMPRESSION) {
    rgb = gamut_compress(rgb);
}
    // Final safety clip of negative components. PQ needs non-negative input
    // before its `pow`. After compression this only catches floating-point
    // residue and scene-referred negatives that did not come from the gamut
    // stage. With no compression active it is the whole out-of-gamut handling.
    //
    // Skipped for the encoded extended-range sRGB transfer. Its OETF is
    // odd-symmetric, and the extended signal exists to carry out-of-gamut and
    // scene-referred negative components past the SDR floor.
@if(!DISPLAY_TRANSFER_EXTENDED_SRGB) {
    rgb = max(rgb, vec3(0.0));
}

    // 4. Transfer encoding. Input is paper-white-relative display-linear
    // (1.0 = paper white at the operator output).
@if(DISPLAY_TRANSFER_SCRGB) {
    // scRGB-linear: 1.0 = 80 nits, so scale by paper_white / 80.
    rgb = scrgb_encode(rgb, display_target.paper_white_nits);
} @elif(DISPLAY_TRANSFER_PQ) {
    // PQ encodes absolute luminance normalized to 10000 nits, so convert
    // paper-white-relative values to nits first.
    rgb = pq_inverse_eotf_from_nits(rgb * display_target.paper_white_nits);
} @elif(DISPLAY_TRANSFER_EXTENDED_SRGB) {
    // Encoded extended-range sRGB (the `ExtendedSrgb` and `ExtendedDisplayP3`
    // color spaces): the same `paper_white / 80` normalization as scRGB, then
    // the odd-symmetric extended sRGB OETF.
    rgb = srgb_oetf_extended(scrgb_encode(rgb, display_target.paper_white_nits));
}

    // Alpha passes through for the multi-camera alpha-blended upscale path.
    return vec4(rgb, color.a);
}
