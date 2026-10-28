// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;

struct CASUniforms {
    sharpness: f32,
};

@group(0) @binding(0) var screenTexture: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> uniforms: CASUniforms;

// This is set at the limit of providing unnatural results for sharpening.
const FSR_RCAS_LIMIT = 0.1875;
// -4.0 instead of -1.0 to avoid issues with MSAA.
const peakC = vec2<f32>(10.0, -40.0);

// On HDR display targets, RCAS runs on colors compressed into -1 to 1, following
// AMD's guidance for RCAS on HDR input. The curve keeps the sign of negative
// channels. A ring with mixed signs in a channel sets the lobe to 0, so those
// pixels are not sharpened.
fn range_compress(c: vec4<f32>) -> vec4<f32> {
@if(HDR_DISPLAY_TARGET) {
    return vec4<f32>(c.rgb / (1.0 + abs(c.rgb)), c.a);
} @else {
    return c;
}
}

@if(HDR_DISPLAY_TARGET)
fn range_decompress(c: vec3<f32>) -> vec3<f32> {
    // The floor at 1 / 65504, the f16 maximum, keeps the result finite.
    return c / max(1.0 - abs(c), vec3<f32>(1.0 / 65504.0));
}

/// Robust Contrast Adaptive Sharpening (RCAS)
/// Based on the following implementation:
/// https://github.com/GPUOpen-Effects/FidelityFX-FSR2/blob/ea97a113b0f9cadf519fbcff315cc539915a3acd/src/ffx-fsr2-api/shaders/ffx_fsr1.h#L672
/// RCAS is based on the following logic.
/// RCAS uses a 5 tap filter in a cross pattern (same as CAS),
///    W                b
///  W 1 W  for taps  d e f
///    W                h
/// Where 'W' is the negative lobe weight.
///  output = (W*(b+d+f+h)+e)/(4*W+1)
/// RCAS solves for 'W' by seeing where the signal might clip out of the {0 to 1} input range,
///  0 == (W*(b+d+f+h)+e)/(4*W+1) -> W = -e/(b+d+f+h)
///  1 == (W*(b+d+f+h)+e)/(4*W+1) -> W = (1-e)/(b+d+f+h-4)
/// Then chooses the 'W' which results in no clipping, limits 'W', and multiplies by the 'sharp' amount.
/// This solution above has issues with MSAA input as the steps along the gradient cause edge detection issues.
/// So RCAS uses 4x the maximum and 4x the minimum (depending on equation)in place of the individual taps.
/// As well as switching from 'e' to either the minimum or maximum (depending on side), to help in energy conservation.
/// This stabilizes RCAS.
/// RCAS does a simple highpass which is normalized against the local contrast then shaped,
///       0.25
///  0.25  -1  0.25
///       0.25
/// This is used as a noise detection filter, to reduce the effect of RCAS on grain, and focus on real edges.
/// The CAS node runs after tonemapping, so on SDR display targets the input is in the range of 0 to 1.
@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Algorithm uses minimal 3x3 pixel neighborhood.
    //    b
    //  d e f
    //    h
    let b = range_compress(textureSample(screenTexture, samp, in.uv, vec2<i32>(0, -1))).rgb;
    let d = range_compress(textureSample(screenTexture, samp, in.uv, vec2<i32>(-1, 0))).rgb;
    // We need the alpha value of the pixel we're working on for the output
    let e = range_compress(textureSample(screenTexture, samp, in.uv));
    let f = range_compress(textureSample(screenTexture, samp, in.uv, vec2<i32>(1, 0))).rgb;
    let h = range_compress(textureSample(screenTexture, samp, in.uv, vec2<i32>(0, 1))).rgb;
    // Min and max of ring.
    let mn4 = min(min(b, d), min(f, h));
    let mx4 = max(max(b, d), max(f, h));
    // Limiters
    // 4.0 to avoid issues with MSAA.
    let hitMin = mn4 / (4.0 * mx4);
    let hitMax = (peakC.x - mx4) / (peakC.y + 4.0 * mn4);
    let lobeRGB = max(-hitMin, hitMax);
    var lobe = max(-FSR_RCAS_LIMIT, min(0.0, max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)))) * uniforms.sharpness;
@if(RCAS_DENOISE) {
    // Luma times 2.
    let bL = b.b * 0.5 + (b.r * 0.5 + b.g);
    let dL = d.b * 0.5 + (d.r * 0.5 + d.g);
    let eL = e.b * 0.5 + (e.r * 0.5 + e.g);
    let fL = f.b * 0.5 + (f.r * 0.5 + f.g);
    let hL = h.b * 0.5 + (h.r * 0.5 + h.g);
    // Noise detection.
    var noise = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;;
    noise = saturate(abs(noise) / (max(max(bL, dL), max(fL, hL)) - min(min(bL, dL), min(fL, hL))));
    noise = 1.0 - 0.5 * noise;
    // Apply noise removal.
    lobe *= noise;
}
    let sharpened = (lobe * b + lobe * d + lobe * f + lobe * h + e.rgb) / (4.0 * lobe + 1.0);
@if(HDR_DISPLAY_TARGET) {
    // Decompression magnifies overshoot near -1 and 1. Limit the result to the
    // range of the taps, widened to 0. A positive channel may also reach paper
    // white (0.5 compressed), as on SDR. The bounds keep the sign of the taps
    // and stay inside -1 to 1.
    let lower = min(min(mn4, e.rgb), vec3<f32>(0.0));
    let tap_max = max(mx4, e.rgb);
    let upper = select(vec3<f32>(0.0), max(tap_max, vec3<f32>(0.5)), tap_max > vec3<f32>(0.0));
    return vec4<f32>(range_decompress(clamp(sharpened, lower, upper)), e.w);
} @else {
    return vec4<f32>(sharpened, e.w);
}
}
