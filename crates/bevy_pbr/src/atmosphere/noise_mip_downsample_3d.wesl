// Box-filter downsample for one 3D noise mip level (src mip N -> dst mip N+1).
@group(0) @binding(13) var src_texture: texture_3d<f32>;
@group(0) @binding(14) var dst_texture: texture_storage_3d<rgba16float, write>;

@compute
@workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_texture);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let base = vec3<i32>(gid) * 2;
    var sum = vec4<f32>(0.0);
    for (var zo = 0; zo < 2; zo++) {
        for (var yo = 0; yo < 2; yo++) {
            for (var xo = 0; xo < 2; xo++) {
                sum += textureLoad(src_texture, base + vec3<i32>(xo, yo, zo), 0);
            }
        }
    }
    textureStore(dst_texture, vec3<i32>(gid), sum * 0.125);
}

// Mean and variance pairs (R, G) and (B, A) of the physical cloud scheme's fluctuation
// (`cloud_noise_quantiles.wesl` bake): the block's mean of the means and its total variance, the
// mean of the variances plus the variance of the means.
@compute
@workgroup_size(4, 4, 4)
fn moments(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst_texture);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }
    let base = vec3<i32>(gid) * 2;
    var mean = vec2<f32>(0.0);
    var second = vec2<f32>(0.0);
    for (var zo = 0; zo < 2; zo++) {
        for (var yo = 0; yo < 2; yo++) {
            for (var xo = 0; xo < 2; xo++) {
                let t = textureLoad(src_texture, base + vec3<i32>(xo, yo, zo), 0);
                mean += t.xz;
                second += vec2(t.y, t.w) + t.xz * t.xz;
            }
        }
    }
    mean *= 0.125;
    let var_ = max(second * 0.125 - mean * mean, vec2(0.0));
    textureStore(dst_texture, vec3<i32>(gid), vec4(mean.x, var_.x, mean.y, var_.y));
}
