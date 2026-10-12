const RGB9E5_EXPONENT_BITS        = 5u;
const RGB9E5_MANTISSA_BITSU       = 9u;
const RGB9E5_EXP_BIAS             = 15;
const RGB9E5_MAX_VALID_BIASED_EXP = 31u;

//#define MAX_RGB9E5_EXP               (RGB9E5_MAX_VALID_BIASED_EXP - RGB9E5_EXP_BIAS)
//#define RGB9E5_MANTISSA_VALUES       (1<<RGB9E5_MANTISSA_BITS)
//#define MAX_RGB9E5_MANTISSA          (RGB9E5_MANTISSA_VALUES-1)
//#define MAX_RGB9E5                   ((f32(MAX_RGB9E5_MANTISSA))/RGB9E5_MANTISSA_VALUES * (1<<MAX_RGB9E5_EXP))
//#define EPSILON_RGB9E5_              ((1.0/RGB9E5_MANTISSA_VALUES) / (1<<RGB9E5_EXP_BIAS))

const MAX_RGB9E5_EXP              = 16u;
const MAX_RGB9E5_MANTISSA         = 511;
const MAX_RGB9E5_MANTISSAU        = 511u;
const EPSILON_RGB9E5_             = 0.000000059604645;

fn floor_log2_(x: f32) -> i32 {
    let f = bitcast<u32>(x);
    let biasedexponent = (f & 0x7F800000u) >> 23u;
    return i32(biasedexponent) - 127;
}

/// https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_shared_exponent.txt
fn vec3_to_rgb9e5_(rgb_in: vec3<f32>) -> u32 {
    return vec3_to_shared_exponent_(rgb_in, RGB9E5_MANTISSA_BITSU);
}

/// The shared-exponent encoding of `vec3_to_rgb9e5_` with `mantissa_bits`-bit mantissas: R, G
/// and B from bit 0 up, then the 5-bit exponent biased by 15.
fn vec3_to_shared_exponent_(rgb_in: vec3<f32>, mantissa_bits: u32) -> u32 {
    let mantissa_values = f32(1u << mantissa_bits);
    let max_value = (mantissa_values - 1.0) / mantissa_values * exp2(f32(MAX_RGB9E5_EXP));
    let rgb = clamp(rgb_in, vec3(0.0), vec3(max_value));

    let maxrgb = max(rgb.r, max(rgb.g, rgb.b));
    var exp_shared = max(-RGB9E5_EXP_BIAS - 1, floor_log2_(maxrgb)) + 1 + RGB9E5_EXP_BIAS;
    var denom = exp2(f32(exp_shared - RGB9E5_EXP_BIAS - i32(mantissa_bits)));

    let maxm = floor(maxrgb / denom + 0.5);
    if (maxm == mantissa_values) {
        denom *= 2.0;
        exp_shared += 1;
    }

    let n = vec3<u32>(floor(rgb / denom + 0.5));

    return (u32(exp_shared) << (3u * mantissa_bits)) | (n.b << (2u * mantissa_bits)) | (n.g << mantissa_bits) | n.r;
}

/// Builtin extractBits() is not working on WEBGL or DX12
/// DX12: HLSL: Unimplemented("write_expr_math ExtractBits")
fn extract_bits(value: u32, offset: u32, bits: u32) -> u32 {
    let mask = (1u << bits) - 1u;
    return (value >> offset) & mask;
}

fn rgb9e5_to_vec3_(v: u32) -> vec3<f32> {
    return shared_exponent_to_vec3_(v, RGB9E5_MANTISSA_BITSU);
}

/// Decodes `vec3_to_shared_exponent_`'s encoding with `mantissa_bits`-bit mantissas.
fn shared_exponent_to_vec3_(v: u32, mantissa_bits: u32) -> vec3<f32> {
    let exponent = i32(extract_bits(v, 3u * mantissa_bits, RGB9E5_EXPONENT_BITS)) - RGB9E5_EXP_BIAS - i32(mantissa_bits);
    let scale = exp2(f32(exponent));

    return vec3(
        f32(extract_bits(v, 0u, mantissa_bits)),
        f32(extract_bits(v, mantissa_bits, mantissa_bits)),
        f32(extract_bits(v, 2u * mantissa_bits, mantissa_bits))
    ) * scale;
}
