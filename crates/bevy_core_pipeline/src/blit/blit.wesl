import package::fullscreen_vertex_shader::FullscreenVertexOutput;
@if(SRGB_TO_LINEAR) import bevy_render::color_operations::srgb_to_linear;
@if(OKLAB_TO_LINEAR) import bevy_render::color_operations::oklab_to_linear_rgb;
@if(DISPLAY_ENCODE) import bevy_render::transfer_functions::{scrgb_encode, pq_inverse_eotf_from_nits, srgb_oetf_extended};

@if(DISPLAY_TARGET_UNIFORM)
struct DisplayTargetUniform {
    gamut: mat3x3<f32>,
    paper_white_nits: f32,
}

@group(0) @binding(0) var in_texture: texture_2d<f32>;
@group(0) @binding(1) var in_sampler: sampler;
@if(DISPLAY_TARGET_UNIFORM)
@group(1) @binding(0) var<uniform> display_target: DisplayTargetUniform;

// The ACES Reference Gamut Compression (Academy S-2020-001). It has a closed form, and it
// doesn't change colors well inside the gamut. The limits are larger than the ACES ones so
// that they cover Rec.2020 in Rec.709.
@if(GAMUT_COMPRESSION)
const GAMUT_COMPRESSION_THRESHOLD = vec3<f32>(0.815, 0.803, 0.880);
@if(GAMUT_COMPRESSION)
const GAMUT_COMPRESSION_LIMIT = vec3<f32>(1.62, 1.10, 1.13);
@if(GAMUT_COMPRESSION)
const GAMUT_COMPRESSION_POWER: f32 = 1.2;
@if(GAMUT_COMPRESSION)
const GAMUT_COMPRESSION_SCALE = (GAMUT_COMPRESSION_LIMIT - GAMUT_COMPRESSION_THRESHOLD) / pow(
    pow(
        (1.0 - GAMUT_COMPRESSION_THRESHOLD) / (GAMUT_COMPRESSION_LIMIT - GAMUT_COMPRESSION_THRESHOLD),
        vec3(-GAMUT_COMPRESSION_POWER),
    ) - 1.0,
    vec3(1.0 / GAMUT_COMPRESSION_POWER),
);

/// Moves each channel that is far from `max(r, g, b)` toward it. A channel whose distance
/// is the limit ends at zero, on the gamut boundary.
@if(GAMUT_COMPRESSION)
fn gamut_compress(rgb: vec3<f32>) -> vec3<f32> {
    let achromatic = max(rgb.r, max(rgb.g, rgb.b));
    if achromatic <= 0.0 {
        return rgb;
    }
    let distance = (achromatic - rgb) / achromatic;
    let excess = max(distance - GAMUT_COMPRESSION_THRESHOLD, vec3(0.0)) / GAMUT_COMPRESSION_SCALE;
    let compressed = GAMUT_COMPRESSION_THRESHOLD + GAMUT_COMPRESSION_SCALE * excess
        / pow(1.0 + pow(excess, vec3(GAMUT_COMPRESSION_POWER)), vec3(1.0 / GAMUT_COMPRESSION_POWER));
    return select(achromatic - compressed * achromatic, rgb, distance < GAMUT_COMPRESSION_THRESHOLD);
}

@fragment
fn fs_main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(in_texture, in_sampler, in.uv);
    @if(SRGB_TO_LINEAR)
    color = vec4(srgb_to_linear(color.rgb), color.a);
    @if(OKLAB_TO_LINEAR)
    color = vec4(oklab_to_linear_rgb(color.rgb), color.a);
@if(DISPLAY_TARGET_UNIFORM) {
    var rgb = display_target.gamut * color.rgb;
    @if(GAMUT_COMPRESSION)
    rgb = gamut_compress(rgb);

@if(DISPLAY_ENCODE) {
    let paper_white = display_target.paper_white_nits;
    @if(DISPLAY_TRANSFER_PQ)
    rgb = pq_inverse_eotf_from_nits(rgb * paper_white);
    @else
    rgb = scrgb_encode(rgb, paper_white);
    @if(DISPLAY_TRANSFER_EXTENDED_SRGB)
    rgb = srgb_oetf_extended(rgb);
} @else {
    rgb = max(rgb, vec3(0.0));
}
    color = vec4(rgb, color.a);
}
    return color;
}
