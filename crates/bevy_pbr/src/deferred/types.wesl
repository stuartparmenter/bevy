import package::render::{
    mesh_types::{MESH_FLAGS_SHADOW_RECEIVER_BIT, MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT},
    pbr_types::{STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT, STANDARD_MATERIAL_FLAGS_UNLIT_BIT},
    rgb9e5::{shared_exponent_to_vec3_, vec3_to_shared_exponent_},
};

// Maximum of 8 bits available
const DEFERRED_FLAGS_UNLIT_BIT: u32                 = 1u << 0u;
const DEFERRED_FLAGS_FOG_ENABLED_BIT: u32           = 1u << 1u;
const DEFERRED_MESH_FLAGS_SHADOW_RECEIVER_BIT: u32  = 1u << 2u;
const DEFERRED_MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT: u32 = 1u << 3u;

fn deferred_flags_from_mesh_material_flags(mesh_flags: u32, mat_flags: u32) -> u32 {
    var flags = 0u;
    flags |= u32((mesh_flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u) * DEFERRED_MESH_FLAGS_SHADOW_RECEIVER_BIT;
    flags |= u32((mesh_flags & MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT) != 0u) * DEFERRED_MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT;
    flags |= u32((mat_flags & STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT) != 0u) * DEFERRED_FLAGS_FOG_ENABLED_BIT;
    flags |= u32((mat_flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) != 0u) * DEFERRED_FLAGS_UNLIT_BIT;
    return flags;
}

fn mesh_material_flags_from_deferred_flags(deferred_flags: u32) -> vec2<u32> {
    var mat_flags = 0u;
    var mesh_flags = 0u;
    mesh_flags |= u32((deferred_flags & DEFERRED_MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u) * MESH_FLAGS_SHADOW_RECEIVER_BIT;
    mesh_flags |= u32((deferred_flags & DEFERRED_MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT) != 0u) * MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT;
    mat_flags |= u32((deferred_flags & DEFERRED_FLAGS_FOG_ENABLED_BIT) != 0u) * STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;
    mat_flags |= u32((deferred_flags & DEFERRED_FLAGS_UNLIT_BIT) != 0u) * STANDARD_MATERIAL_FLAGS_UNLIT_BIT;
    return vec2(mesh_flags, mat_flags);
}

const U12MAXF = 4095.0;
const U16MAXF = 65535.0;
const U20MAXF = 1048575.0;

// Storing normals as oct24.
// Flags are stored in the remaining 8 bits.
// https://jcgt.org/published/0003/02/01/paper.pdf
// Could possibly go down to oct20 if the space is needed.

fn pack_24bit_normal_and_flags(octahedral_normal: vec2<f32>, flags: u32) -> u32 {
    let unorm1 = u32(saturate(octahedral_normal.x) * U12MAXF + 0.5);
    let unorm2 = u32(saturate(octahedral_normal.y) * U12MAXF + 0.5);
    return (unorm1 & 0xFFFu) | ((unorm2 & 0xFFFu) << 12u) | ((flags & 0xFFu) << 24u);
}

fn unpack_24bit_normal(packed: u32) -> vec2<f32> {
    let unorm1 = packed & 0xFFFu;
    let unorm2 = (packed >> 12u) & 0xFFFu;
    return vec2(f32(unorm1) / U12MAXF, f32(unorm2) / U12MAXF);
}

fn unpack_flags(packed: u32) -> u32 {
    return (packed >> 24u) & 0xFFu;
}

// The G-buffer stores the magnitude of the dielectric Fresnel ratio after the clearcoat and
// specular weight, from `pbr_functions::dielectric_fresnel_ratio`, as a unorm on a power curve
// that gives more codes to low values.

/// The exponent of the Fresnel ratio curve, `log(0.2) / log(7 / 15)`. It puts 0.2, the ratio of
/// the default IOR of 1.5, exactly on a 4-bit and an 8-bit code, since 7 / 15 = 119 / 255.
const FRESNEL_RATIO_EXPONENT: f32 = 2.1117351;

/// Encodes the magnitude of a Fresnel ratio as a unorm value.
fn encode_fresnel_ratio(ratio: f32) -> f32 {
    return pow(saturate(abs(ratio)), 1.0 / FRESNEL_RATIO_EXPONENT);
}

/// Decodes a Fresnel ratio magnitude encoded by `encode_fresnel_ratio`.
fn decode_fresnel_ratio(encoded: f32) -> f32 {
    return pow(encoded, FRESNEL_RATIO_EXPONENT);
}

/// The builtin one didn't work in webgl.
/// "'unpackUnorm4x8' : no matching overloaded function found"
/// https://github.com/gfx-rs/naga/issues/2006
fn unpack_unorm4x8_(v: u32) -> vec4<f32> {
    return vec4(
        f32(v & 0xFFu),
        f32((v >> 8u) & 0xFFu),
        f32((v >> 16u) & 0xFFu),
        f32((v >> 24u) & 0xFFu)
    ) / 255.0;
}

/// 'packUnorm4x8' : no matching overloaded function found
/// https://github.com/gfx-rs/naga/issues/2006
fn pack_unorm4x8_(values: vec4<f32>) -> u32 {
    let v = vec4<u32>(saturate(values) * 255.0 + 0.5);
    return (v.w << 24u) | (v.z << 16u) | (v.y << 8u) | v.x;
}

/// Pack 3x 4bit unorm + 1x 20bit
fn pack_unorm3x4_plus_unorm_20_(v: vec4<f32>) -> u32 {
    let sm = vec3<u32>(saturate(v.xyz) * 15.0 + 0.5);
    let bg = u32(saturate(v.w) * U20MAXF + 0.5);
    return (bg << 12u) | (sm.z << 8u) | (sm.y << 4u) | sm.x;
}

/// Unpack 3x 4bit unorm + 1x 20bit
fn unpack_unorm3x4_plus_unorm_20_(v: u32) -> vec4<f32> {
    return vec4(
        f32(v & 0xfu) / 15.0,
        f32((v >> 4u) & 0xFu) / 15.0,
        f32((v >> 8u) & 0xFu) / 15.0,
        f32((v >> 12u) & 0xFFFFFFu) / U20MAXF,
    );
}

// The deferred specular tint texture's word: the specular tint in `vec3_to_shared_exponent_`'s
// encoding with 7-bit mantissas (bits 0-25) XORed with the encoding of white, and the diffuse
// transmission as a 6-bit unorm in bits 26-31. White with no transmission packs to 0, so the
// cleared texture and unwritten `FragmentOutput` fields decode to it.
const SPECULAR_TINT_MANTISSA_BITS: u32 = 7u;
/// `vec3_to_shared_exponent_(vec3(1.0), SPECULAR_TINT_MANTISSA_BITS)`.
const SPECULAR_TINT_WHITE: u32 = 0x02102040u;
const DIFFUSE_TRANSMISSION_SHIFT: u32 = 26u;
const DIFFUSE_TRANSMISSION_MAX: f32 = 63.0;

/// Packs a specular tint and a diffuse transmission for the deferred specular tint texture.
fn pack_specular_tint_transmission(specular_tint: vec3<f32>, diffuse_transmission: f32) -> u32 {
    let tint = vec3_to_shared_exponent_(specular_tint, SPECULAR_TINT_MANTISSA_BITS) ^ SPECULAR_TINT_WHITE;
    let transmission = u32(round(saturate(diffuse_transmission) * DIFFUSE_TRANSMISSION_MAX));
    return tint | (transmission << DIFFUSE_TRANSMISSION_SHIFT);
}

/// Unpacks the specular tint packed by `pack_specular_tint_transmission`.
fn unpack_specular_tint(packed: u32) -> vec3<f32> {
    return shared_exponent_to_vec3_(packed ^ SPECULAR_TINT_WHITE, SPECULAR_TINT_MANTISSA_BITS);
}

/// Unpacks the diffuse transmission packed by `pack_specular_tint_transmission`.
fn unpack_diffuse_transmission(packed: u32) -> f32 {
    return f32(packed >> DIFFUSE_TRANSMISSION_SHIFT) / DIFFUSE_TRANSMISSION_MAX;
}
