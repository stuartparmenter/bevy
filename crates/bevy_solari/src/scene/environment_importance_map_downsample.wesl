// One mip step of the importance pyramid: each parent is the sum of its up to four children;
// environment.rs's module doc describes the pyramid.

@group(0) @binding(0) var source_level: texture_2d<f32>;
@group(0) @binding(1) var destination_level: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn downsample_environment_importance_map(@builtin(global_invocation_id) id: vec3<u32>) {
    let destination_size = textureDimensions(destination_level);
    if any(id.xy >= destination_size) { return; }
    let source_size = textureDimensions(source_level);
    let base = id.xy * 2u;
    var sum = 0.0;
    for (var dy = 0u; dy < 2u; dy++) {
        for (var dx = 0u; dx < 2u; dx++) {
            let coords = base + vec2(dx, dy);
            if all(coords < source_size) { sum += textureLoad(source_level, coords, 0).r; }
        }
    }
    textureStore(destination_level, id.xy, vec4(sum, 0.0, 0.0, 0.0));
}
