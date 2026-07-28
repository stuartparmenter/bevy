// Display-encoding pass: paper-white-relative display-linear color (the
// tone-map operator's output space, with UI already composited) → encoded
// display signal.
//
// Stages, per the separated display pipeline:
//   1. (optional) decode the view's compositing space back to linear
//      (`SRGB_TO_LINEAR` / `OKLAB_TO_LINEAR`, normally done by the upscaling
//      blit — which passes encoded output through untouched instead when this
//      pass ran),
//   2. gamut transform from the per-view source primaries (the tonemapping
//      pass's output: Rec.709, or GT7's native Rec.2020 on HDR targets) to
//      the display *signal* primaries:
//        - `DISPLAY_GAMUT_REC2020`: Rec.709 source → PQ/Rec.2020 signal
//          (an expansion, in-gamut by construction),
//        - `GAMUT_REC2020_TO_REC709`: GT7's Rec.2020 source → scRGB signal
//          (a contraction; the out-of-gamut compression below keys in for it
//          under `DisplayGamutCompression::Auto`) — scRGB signals are
//          definitionally expressed in (extended) Rec.709 coordinates
//          whatever the panel's physical gamut, so prepare coerces the scRGB
//          encoding gamut to Rec.709 (the compositor maps to the panel
//          itself),
//        - `DISPLAY_GAMUT_DISPLAYP3`: Rec.709 source → Display-P3 signal
//          (the `ExtendedDisplayP3` color space; an expansion, in-gamut by
//          construction),
//        - `GAMUT_REC2020_TO_DISPLAYP3`: GT7's Rec.2020 source → Display-P3
//          signal (a contraction; keys the compression like the Rec.709 case),
//        - no def: identity (Rec.709 → scRGB / extended-sRGB, GT7's Rec.2020 →
//          PQ/Rec.2020),
//   3. out-of-gamut handling: ACES-RGC-style hue-approximate chroma
//      compression toward the achromatic axis (`DISPLAY_GAMUT_COMPRESSION`),
//      with the plain hue-shifting per-channel clip as the debug fallback
//      (`DISPLAY_GAMUT_CLIP_DEBUG`), followed by a `max(0)` safety clip (PQ
//      requires non-negative input) — skipped for the sign-preserving
//      `DISPLAY_TRANSFER_EXTENDED_SRGB` transfer,
//   4. transfer encoding (`DISPLAY_TRANSFER_SCRGB` /
//      `DISPLAY_TRANSFER_PQ` / `DISPLAY_TRANSFER_EXTENDED_SRGB`).
//
// This shader is never specialized for sRGB targets: the exact sRGB OETF is
// hardware-applied on the upscaling blit's `*UnormSrgb` writeback, so plain
// SDR views never run this pass.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::display_target::DisplayTargetUniform
#import bevy_render::transfer_functions::{scrgb_encode, pq_inverse_eotf_from_nits, srgb_oetf_extended}
// Gamut matrices for stage 2; that module documents their derivation and the
// bit-identity contract with the Rust constants. Each is used under exactly one
// of the gamut defs below.
#import bevy_render::working_color_space::{REC709_TO_REC2020, REC2020_TO_REC709, REC709_TO_DISPLAYP3, REC2020_TO_DISPLAYP3}
#ifdef SRGB_TO_LINEAR
#import bevy_render::color_operations::srgb_to_linear
#endif
#ifdef OKLAB_TO_LINEAR
#import bevy_render::color_operations::oklab_to_linear_rgb
#endif

@group(0) @binding(0) var in_texture: texture_2d<f32>;
@group(0) @binding(1) var in_sampler: sampler;
// Per-view display-target calibration. The gamut and transfer are
// compile-time shader defs here; paper white is the only value read at
// runtime. It is sanitized by the uniform producer
// (`prepare_view_display_targets`: finite, positive, <= 10000) with the same
// rules the tone-map operators fold at prepare time, so the seam scale
// factors cancel exactly.
@group(0) @binding(2) var<uniform> display_target: DisplayTargetUniform;

#ifdef DISPLAY_GAMUT_COMPRESSION
// Out-of-gamut chroma compression in the style of the ACES 1.3 Reference
// Gamut Compression (Academy S-2020-001 / ACES "RGC", aces-dev
// lib/RGC_common.ctl): per-channel distance from the achromatic axis,
// `dist = (ach - c) / ach` with `ach = max(r, g, b)`, smoothly compressed
// with the parametric power curve so that `dist == limit` lands exactly on
// the gamut boundary (`dist == 1`, i.e. channel value 0) while distances
// below the threshold pass through bit-identically.
//
// Thresholds and power are the published ACES RGC values (cyan 0.815,
// magenta 0.803, yellow 0.880, power 1.2). The ACES *limits*
// (1.147 / 1.264 / 1.312) were derived from digital-cinema camera gamuts and
// under-cover the Rec.2020 → Rec.709 contraction this pass performs (the
// Rec.2020 hull reaches a distance of ~1.594 in the cyan direction when
// expressed in Rec.709 coordinates), so the limits below are re-derived from
// the Rec.2020 hull maxima (~1.594 / ~1.087 / ~1.117) plus headroom.
// CPU mirror + tests: `gamut_compression.rs` next to this shader — keep both
// in sync.
const GAMUT_COMPRESSION_THRESHOLD = vec3<f32>(0.815, 0.803, 0.880);
const GAMUT_COMPRESSION_POWER: f32 = 1.2;
// Limits (the distance that maps exactly onto the gamut boundary); kept for
// documentation/derivation only — the shader consumes the precomputed scales.
const GAMUT_COMPRESSION_LIMIT = vec3<f32>(1.62, 1.10, 1.13);
// scale = (limit - thr) / (((1 - thr) / (limit - thr))^(-power) - 1)^(1/power),
// evaluated in f64 (see `compression_scale` in gamut_compression.rs; a test
// locks these literals to the closed form).
const GAMUT_COMPRESSION_SCALE = vec3<f32>(0.21634937, 0.43270176, 0.18745117);

// The ACES RGC parametric compression curve, vectorized over the three
// chroma directions. Identity below the threshold (the `max` keeps `pow`
// away from negative bases; callers select the original value there anyway),
// monotonically increasing above it, mapping `limit` to 1 and approaching
// `threshold + scale` asymptotically.
fn gamut_compress_distance(dist: vec3<f32>) -> vec3<f32> {
    let nd = max(dist - GAMUT_COMPRESSION_THRESHOLD, vec3(0.0)) / GAMUT_COMPRESSION_SCALE;
    let p = pow(nd, vec3(GAMUT_COMPRESSION_POWER));
    return GAMUT_COMPRESSION_THRESHOLD
        + GAMUT_COMPRESSION_SCALE * nd / pow(1.0 + p, vec3(1.0 / GAMUT_COMPRESSION_POWER));
}

// Compresses out-of-gamut colors (negative components) toward the achromatic
// axis at constant `max(r, g, b)`. Channels whose distance is below the
// threshold are returned bit-identically; colors whose distance does not
// exceed the limit land inside the target gamut. Hue is approximately (not
// exactly) preserved — the per-channel formulation is the standard
// cost/robustness trade-off of the ACES RGC (no boundary search, no
// iteration, monotonic, NaN-free for finite inputs).
fn gamut_compress(rgb: vec3<f32>) -> vec3<f32> {
    let achromatic = max(rgb.r, max(rgb.g, rgb.b));
    if achromatic <= 0.0 {
        // No positive channel to compress toward; the final safety clip
        // below handles all-non-positive colors (exactly like the previous
        // clip-only behavior).
        return rgb;
    }
    let dist = (vec3(achromatic) - rgb) / achromatic;
    let compressed = vec3(achromatic) - gamut_compress_distance(dist) * achromatic;
    // Bit-identical pass-through for in-gamut channels under the threshold.
    return select(compressed, rgb, dist < GAMUT_COMPRESSION_THRESHOLD);
}
#endif

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(in_texture, in_sampler, in.uv);

    // 1. Decode the compositing space, if the main texture is not already
    // display-linear. (Same defs and math as the upscaling blit.)
#ifdef SRGB_TO_LINEAR
    color = vec4(srgb_to_linear(color.rgb), color.a);
#endif
#ifdef OKLAB_TO_LINEAR
    color = vec4(oklab_to_linear_rgb(color.rgb), color.a);
#endif

    var rgb = color.rgb;

    // 2. Gamut transform: tone-map-output primaries → display primaries.
    //
    // Input contract: the tone-mapping pass emits Rec.709 display-linear for
    // every operator under every `WorkingColorSpace` (Rec.709-fit operators
    // receive a Rec.2020 → Rec.709 conversion at the pass entry), EXCEPT
    // `Tonemapping::GranTurismo7` on an HDR-transfer target, which emits its
    // native linear Rec.2020 (the `TONEMAP_OUTPUT_REC2020` path in
    // gt7.wgsl). The stack contract resolver derives the per-view source
    // gamut from the same predicate (`tonemap_output_gamut`) and prepare
    // keys exactly one of the defs below — or none for the identity
    // stages (Rec.709 → scRGB, Rec.2020 → PQ/Rec.2020).
#ifdef DISPLAY_GAMUT_REC2020
    rgb = REC709_TO_REC2020 * rgb;
#endif
#ifdef GAMUT_REC2020_TO_REC709
    // A gamut *contraction* (GT7's Rec.2020 output onto the
    // Rec.709-coordinate scRGB signal): can produce out-of-gamut (negative)
    // components, for which prepare keys in the out-of-gamut compression
    // below (`DISPLAY_GAMUT_COMPRESSION`; see `DisplayGamutCompression` and
    // `is_gamut_contraction` in mod.rs).
    rgb = REC2020_TO_REC709 * rgb;
#endif
#ifdef DISPLAY_GAMUT_DISPLAYP3
    // Rec.709 source → Display-P3 signal (the `ExtendedDisplayP3` color space):
    // an expansion (Display-P3 ⊃ Rec.709), in-gamut by construction.
    rgb = REC709_TO_DISPLAYP3 * rgb;
#endif
#ifdef GAMUT_REC2020_TO_DISPLAYP3
    // GT7's Rec.2020 source → Display-P3 signal: a contraction
    // (Display-P3 ⊂ Rec.2020), for which prepare keys in the out-of-gamut
    // compression below, exactly like the Rec.2020 → Rec.709 case.
    rgb = REC2020_TO_DISPLAYP3 * rgb;
#endif

    // 3. Out-of-gamut handling (perceptual compression, with the per-channel
    // clip as the debug fallback). The compression def
    // is pushed only when the gamut stage can actually produce out-of-gamut
    // colors (a gamut *contraction*, or `DisplayGamutCompression::Always`);
    // expansions and identity transforms keep the plain clip below, which is
    // a no-op for their in-gamut-by-construction inputs.
#ifdef DISPLAY_GAMUT_COMPRESSION
    rgb = gamut_compress(rgb);
#endif
    // Final safety clip of negative components (PQ additionally requires
    // non-negative input before its `pow`). After compression this only
    // catches floating-point residue and scene-referred negatives that did
    // not come from the gamut stage (compressed colors land in-gamut by
    // construction); under DISPLAY_GAMUT_CLIP_DEBUG — or when no compression
    // is active — it IS the entire out-of-gamut handling: the hue-shifting
    // per-channel clip, kept for A/B comparison against the compression.
    //
    // Skipped for the encoded extended-range sRGB transfer: its OETF is
    // odd-symmetric (sign-preserving by design), and the whole point of the
    // extended signal is to carry wide-gamut / out-of-gamut and
    // scene-referred negative components past the SDR floor — clipping them
    // would discard exactly the range the transfer exists to encode. (Its
    // contraction paths already land in-gamut via the compression above; any
    // residue the OETF carries safely.)
#ifndef DISPLAY_TRANSFER_EXTENDED_SRGB
    rgb = max(rgb, vec3(0.0));
#endif

    // 4. Transfer encoding. Input is paper-white-relative display-linear
    // (1.0 = paper white at the operator output).
#ifdef DISPLAY_TRANSFER_SCRGB
    // scRGB-linear: 1.0 = 80 nits, so scale by paper_white / 80.
    rgb = scrgb_encode(rgb, display_target.paper_white_nits);
#else ifdef DISPLAY_TRANSFER_PQ
    // PQ encodes absolute luminance normalized to 10000 nits: convert
    // paper-white-relative values to nits first.
    rgb = pq_inverse_eotf_from_nits(rgb * display_target.paper_white_nits);
#else ifdef DISPLAY_TRANSFER_EXTENDED_SRGB
    // Encoded extended-range sRGB (the `ExtendedSrgb` / `ExtendedDisplayP3`
    // color spaces): the same scRGB `paper_white / 80` normalization as the
    // linear path, then the odd-symmetric extended sRGB OETF.
    rgb = srgb_oetf_extended(scrgb_encode(rgb, display_target.paper_white_nits));
#endif

    // Alpha passes through for the multi-camera alpha-blended upscale path.
    return vec4(rgb, color.a);
}
