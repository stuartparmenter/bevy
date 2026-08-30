// Writes mip 0 of the environment importance atlas, the mean luminance of the cubemap over each
// face texel times the texel's solid angle; environment.rs's module doc describes the pyramid.

import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance;
import bevy_pbr::render::utils::sample_cube_dir;
import package::scene::environment_map::{atlas_position_to_face, cube_face_texel_solid_angle, ENVIRONMENT_ATLAS_EMPTY_FACE, ENVIRONMENT_ATLAS_FACES_X, ENVIRONMENT_ATLAS_FACES_Y, ENVIRONMENT_IMPORTANCE_FLOOR};
import package::scene::sampling::isfinite;

struct BuildConstants {
    face_size: u32,
    source_mip: f32,
    oversample: u32,
}

@group(0) @binding(0) var environment_map: texture_cube<f32>;
@group(0) @binding(1) var environment_map_sampler: sampler;
@group(0) @binding(2) var importance_map_mip0: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> constants: BuildConstants;

// Non-finite luminance counts as 0 so one bad texel cannot turn the pyramid's root into Inf/NaN
// and zero every other pdf.
fn finite_luminance(radiance: vec3<f32>) -> f32 {
    let l = luminance(radiance);
    return select(0.0, l, isfinite(l));
}

@compute @workgroup_size(8, 8, 1)
fn build_environment_importance_map(@builtin(global_invocation_id) id: vec3<u32>) {
    let atlas_size = vec2(ENVIRONMENT_ATLAS_FACES_X, ENVIRONMENT_ATLAS_FACES_Y) * constants.face_size;
    if any(id.xy >= atlas_size) { return; }
    let texel = atlas_position_to_face(id.xy, vec2(0.5), constants.face_size);
    var weight = 0.0;
    if texel.face != ENVIRONMENT_ATLAS_EMPTY_FACE {
        // `oversample^2` taps spread over the texel when the source has no mip matching the atlas,
        // so a small bright feature between the texel centres still contributes.
        let taps = max(constants.oversample, 1u);
        let inv_taps = 1.0 / f32(taps);
        var sum = 0.0;
        for (var j = 0u; j < taps; j++) {
            for (var i = 0u; i < taps; i++) {
                let jitter = (vec2(f32(i), f32(j)) + 0.5) * inv_taps;
                let coords = atlas_position_to_face(id.xy, jitter, constants.face_size);
                let direction = sample_cube_dir(coords.uv, coords.face);
                let radiance = textureSampleLevel(environment_map, environment_map_sampler, direction, constants.source_mip).rgb;
                sum += finite_luminance(radiance);
            }
        }
        let mean = sum * inv_taps * inv_taps;
        weight = max(mean, ENVIRONMENT_IMPORTANCE_FLOOR) * cube_face_texel_solid_angle(texel.uv, constants.face_size);
    }
    textureStore(importance_map_mip0, id.xy, vec4(weight, 0.0, 0.0, 0.0));
}
