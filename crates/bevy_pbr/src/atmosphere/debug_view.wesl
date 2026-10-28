// Atmosphere debug view (`debug_view.rs`): maps one texture, or a z slice of a 3D one, into a
// display image for the UI. Nearest texels (textureLoad: R32Float and Rgba32Float are not
// filterable), fitted into the image with its aspect ratio kept.

struct DebugParams {
    // 0 rgb, 1 r, 2 g, 3 b, 4 a, 5 r * g.
    channels: u32,
    // Stored encoding: 0 the value, 1 log (shown as exp(x)), 2 optical depth (shown as exp(-x)).
    decode: u32,
    // 0 linear (value * scale + bias), 1 log2 of value * scale over LOG_STOPS stops up to 2^2,
    // 2 tonemapped x / (1 + x), x = value * scale (radiance).
    mapping: u32,
    mip: u32,
    scale: f32,
    bias: f32,
    // 3D sources: the z slice, as a fraction of the depth.
    slice: f32,
    _pad: f32,
}

@if(SOURCE_3D)
@group(0) @binding(0) var source: texture_3d<f32>;
@else
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var out_image: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: DebugParams;

const LOG_STOPS: f32 = 12.0;
const BACKGROUND: vec4<f32> = vec4(0.05, 0.05, 0.05, 1.0);
const NAN_COLOUR: vec4<f32> = vec4(1.0, 0.0, 1.0, 1.0);

fn mip() -> u32 {
    return min(params.mip, textureNumLevels(source) - 1u);
}

fn load_texel(texel: vec2<u32>) -> vec4<f32> {
    @if(SOURCE_3D) {
        let depth = textureDimensions(source, mip()).z;
        let z = min(u32(params.slice * f32(depth)), depth - 1u);
        return textureLoad(source, vec3(texel, z), mip());
    }
    @if(!SOURCE_3D) {
        return textureLoad(source, texel, mip());
    }
}

fn source_size() -> vec2<u32> {
    return textureDimensions(source, mip()).xy;
}

@compute @workgroup_size(8, 8, 1)
fn blit(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_size = textureDimensions(out_image);
    if (any(id.xy >= out_size)) {
        return;
    }
    // Fit the source into the image (letterbox), texel (0, 0) at the top left.
    let src = vec2<f32>(source_size());
    let dst = vec2<f32>(out_size);
    let s = min(dst.x / src.x, dst.y / src.y);
    let origin = 0.5 * (dst - src * s);
    let q = (vec2<f32>(id.xy) + 0.5 - origin) / s;
    if (any(q < vec2(0.0)) || any(q >= src)) {
        textureStore(out_image, id.xy, BACKGROUND);
        return;
    }
    let v = load_texel(vec2<u32>(q));
    var c: vec3<f32>;
    switch params.channels {
        case 1u: { c = vec3(v.r); }
        case 2u: { c = vec3(v.g); }
        case 3u: { c = vec3(v.b); }
        case 4u: { c = vec3(v.a); }
        case 5u: { c = vec3(v.r * v.g); }
        default: { c = v.rgb; }
    }
    if (params.decode == 1u) {
        c = exp(c);
    } else if (params.decode == 2u) {
        c = exp(-c);
    }
    if (any(c != c)) {
        textureStore(out_image, id.xy, NAN_COLOUR);
        return;
    }
    if (params.mapping == 1u) {
        c = saturate((log2(max(c * params.scale, vec3(1e-30))) + LOG_STOPS - 2.0) / LOG_STOPS);
    } else if (params.mapping == 2u) {
        let x = max(c * params.scale, vec3(0.0));
        c = x / (1.0 + x);
    } else {
        c = saturate(c * params.scale + params.bias);
    }
    textureStore(out_image, id.xy, vec4(c, 1.0));
}
