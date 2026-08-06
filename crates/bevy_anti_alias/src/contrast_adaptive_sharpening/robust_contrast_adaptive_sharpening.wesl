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

// range compression following AMD's HDR guidance for RCAS
fn rcas_range_compress(c: vec3<f32>) -> vec3<f32> {
    // Negative inputs are not expected post-tonemap. Clamp them so the
    // compression stays monotonic and invertible.
    let v = max(c, vec3<f32>(0.0));
    return v / (1.0 + v);
}

fn rcas_range_decompress(c: vec3<f32>) -> vec3<f32> {
    // The denominator floor of `1 / 65504` (the f16 maximum an Rgba16Float
    // target can store) keeps the inverse finite for inputs at or above 1.0.
    let v = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return v / max(1.0 - v, vec3<f32>(1.0 / 65504.0));
}

// Robust Contrast Adaptive Sharpening (RCAS)
// Based on the following implementation:
// https://github.com/GPUOpen-Effects/FidelityFX-FSR2/blob/ea97a113b0f9cadf519fbcff315cc539915a3acd/src/ffx-fsr2-api/shaders/ffx_fsr1.h#L672
// RCAS is based on the following logic.
// RCAS uses a 5 tap filter in a cross pattern (same as CAS),
//    W                b
//  W 1 W  for taps  d e f 
//    W                h
// Where 'W' is the negative lobe weight.
//  output = (W*(b+d+f+h)+e)/(4*W+1)
// RCAS solves for 'W' by seeing where the signal might clip out of the {0 to 1} input range,
//  0 == (W*(b+d+f+h)+e)/(4*W+1) -> W = -e/(b+d+f+h)
//  1 == (W*(b+d+f+h)+e)/(4*W+1) -> W = (1-e)/(b+d+f+h-4)
// Then chooses the 'W' which results in no clipping, limits 'W', and multiplies by the 'sharp' amount.
// This solution above has issues with MSAA input as the steps along the gradient cause edge detection issues.
// So RCAS uses 4x the maximum and 4x the minimum (depending on equation)in place of the individual taps.
// As well as switching from 'e' to either the minimum or maximum (depending on side), to help in energy conservation.
// This stabilizes RCAS.
// RCAS does a simple highpass which is normalized against the local contrast then shaped,
//       0.25
//  0.25  -1  0.25
//       0.25
// This is used as a noise detection filter, to reduce the effect of RCAS on grain, and focus on real edges.
// The CAS node runs after tonemapping. On SDR display targets the input is in
// the range of 0 to 1.
@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Algorithm uses minimal 3x3 pixel neighborhood.
    //    b
    //  d e f
    //    h
    var b: vec3<f32>;
    var d: vec3<f32>;
    var e: vec4<f32>;
    var f: vec3<f32>;
    var h: vec3<f32>;
    @if(HDR_DISPLAY_TARGET)
    var local_max: vec3<f32>;
@if(HDR_DISPLAY_TARGET) {
    let b_raw = textureSample(screenTexture, samp, in.uv, vec2<i32>(0, -1)).rgb;
    let d_raw = textureSample(screenTexture, samp, in.uv, vec2<i32>(-1, 0)).rgb;
    // We need the alpha value of the pixel we're working on for the output
    let e_raw = textureSample(screenTexture, samp, in.uv).rgba;
    let f_raw = textureSample(screenTexture, samp, in.uv, vec2<i32>(1, 0)).rgb;
    let h_raw = textureSample(screenTexture, samp, in.uv, vec2<i32>(0, 1)).rgb;
    b = rcas_range_compress(b_raw);
    d = rcas_range_compress(d_raw);
    e = vec4<f32>(rcas_range_compress(e_raw.rgb), e_raw.a);
    f = rcas_range_compress(f_raw);
    h = rcas_range_compress(h_raw);
    local_max = max(max(b_raw, d_raw), max(max(f_raw, h_raw), e_raw.rgb));
} @else {
    b = textureSample(screenTexture, samp, in.uv, vec2<i32>(0, -1)).rgb;
    d = textureSample(screenTexture, samp, in.uv, vec2<i32>(-1, 0)).rgb;
    // We need the alpha value of the pixel we're working on for the output
    e = textureSample(screenTexture, samp, in.uv).rgba;
    f = textureSample(screenTexture, samp, in.uv, vec2<i32>(1, 0)).rgb;
    h = textureSample(screenTexture, samp, in.uv, vec2<i32>(0, 1)).rgb;
}
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
@if(HDR_DISPLAY_TARGET) {
    let sharpened = rcas_range_decompress((lobe * b + lobe * d + lobe * f + lobe * h + e.rgb) / (4.0 * lobe + 1.0));
    // ensure clamp RCAS limiter output to paper white for HDR values
    return vec4<f32>(min(sharpened, max(local_max, vec3<f32>(1.0))), e.w);
} @else {
    return vec4<f32>((lobe * b + lobe * d + lobe * f + lobe * h + e.rgb) / (4.0 * lobe + 1.0), e.w);
}
}
