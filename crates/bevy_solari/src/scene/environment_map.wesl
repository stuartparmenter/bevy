// Atlas layout and cube-face math shared by the importance-map build pass and the scene bindings;
// binding-free. environment.rs's module doc describes the pyramid.

import bevy_pbr::render::utils::CubeUV;

const ENVIRONMENT_ATLAS_FACES_X = 4u;
const ENVIRONMENT_ATLAS_FACES_Y = 2u;
const ENVIRONMENT_ATLAS_EMPTY_FACE = 0xFFFFFFFFu;
// Keeps every real texel's probability strictly positive.
const ENVIRONMENT_IMPORTANCE_FLOOR = 1e-6f;

fn atlas_face_origin(face: u32, face_size: u32) -> vec2<u32> {
    return vec2(face % ENVIRONMENT_ATLAS_FACES_X, face / ENVIRONMENT_ATLAS_FACES_X) * face_size;
}

// Mip-0 atlas texel holding `coords` (a dir_to_cube_uv result).
fn face_to_atlas_texel(coords: CubeUV, face_size: u32) -> vec2<u32> {
    let uv = clamp(coords.uv, vec2(0.0), vec2(1.0));
    let local = min(vec2<u32>(uv * f32(face_size)), vec2(face_size - 1u));
    return atlas_face_origin(coords.face, face_size) + local;
}

// Face and uv (for sample_cube_dir) of the point `jitter` in [0, 1) inside mip-0 atlas `texel`;
// face is ENVIRONMENT_ATLAS_EMPTY_FACE in the two unused slots.
fn atlas_position_to_face(texel: vec2<u32>, jitter: vec2<f32>, face_size: u32) -> CubeUV {
    let face_xy = texel / face_size;
    var face = face_xy.y * ENVIRONMENT_ATLAS_FACES_X + face_xy.x;
    if face >= 6u { face = ENVIRONMENT_ATLAS_EMPTY_FACE; }
    let local = vec2<f32>(texel - face_xy * face_size) + jitter;
    return CubeUV(local / f32(face_size), face);
}

// 1 / (solid angle of a face texel) at uv: d(s,t)/dω for a face of `face_size` texels.
fn cube_face_jacobian(uv: vec2<f32>, face_size: u32) -> f32 {
    let st = uv * 2.0 - 1.0;
    let n = f32(face_size);
    let x = dot(st, st) + 1.0;
    return 0.25 * n * n * x * sqrt(x);
}

fn cube_face_texel_solid_angle(uv: vec2<f32>, face_size: u32) -> f32 {
    return 1.0 / cube_face_jacobian(uv, face_size);
}
