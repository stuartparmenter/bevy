// Weather window (`.clouds-handoff/notes/weather_layers_design.md`, A2): the weather around one
// view's camera, composed each frame from the tiled weather map, the physical defaults and the
// world-anchored influences (`weather.rs` CloudInfluence). Window texel j holds tile texel
// origin + j of the weather map, in the map's own (rotated, advected) frame, so inside the window
// and away from influences the clouds sample exactly what the tiled map gives (`clouds.wesl`
// sample_weather).
// - out_art (A): coverage, type, density scale, mesoscale fraction.
// - out_physical (B): cloud-fraction scale, air mass (the map's type unless set), droplet number
//   N_d (cm^-3), cloud-base offset (m).

@group(0) @binding(0) var weather_tile: texture_2d<f32>;
@group(0) @binding(1) var out_art: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var out_physical: texture_storage_2d<rgba16float, write>;

const MAX_INFLUENCES: u32 = 16u;

struct Influence {
    // Centre (atmosphere-space xz) and the local x axis (unit, xz).
    center_axis: vec4<f32>,
    // Half extents (m), feather (m), shape (bits: 0 rectangle, 1 ellipse).
    size: vec4<f32>,
    // Blend mode (bits), opacity, channel mask (bits: 0-3 art, 4-7 physical).
    mode: vec4<f32>,
    art: vec4<f32>,
    physical: vec4<f32>,
}

struct WeatherWindowParams {
    // Window texel 0 in weather-map texels (integer valued), `clouds.wesl` weather_window_origin.
    origin: vec2<f32>,
    droplet_number: f32,
    // Influence count (bits).
    count: f32,
    // The map's frame: the advection offset (xz) the weather drifts by, and the tile size (m).
    map_offset: vec2<f32>,
    map_scale: f32,
    _pad: f32,
    influences: array<Influence, MAX_INFLUENCES>,
}
@group(0) @binding(3) var<uniform> params: WeatherWindowParams;

fn blend(base: f32, value: f32, mode: u32) -> f32 {
    switch mode {
        case 1u: { return base + value; }
        case 2u: { return base - value; }
        case 3u: { return base * value; }
        case 4u: { return max(base, value); }
        case 5u: { return min(base, value); }
        default: { return value; }
    }
}

/// Mask of an influence at atmosphere-space xz: 1 inside, falling to 0 over the feather width
/// at the edge.
fn influence_mask(inf: Influence, p: vec2<f32>) -> f32 {
    let d = p - inf.center_axis.xy;
    let ax = inf.center_axis.zw;
    let local = vec2(dot(d, ax), dot(d, vec2(-ax.y, ax.x)));
    let half = inf.size.xy;
    let feather = max(inf.size.z, 1e-3);
    var inside: f32;
    if (bitcast<u32>(inf.size.w) == 1u) {
        // Distance inside the ellipse along its normal, approximately (length in units of the
        // semi-axes, scaled back by the smaller axis).
        inside = (1.0 - length(local / half)) * min(half.x, half.y);
    } else {
        let q = half - abs(local);
        inside = min(q.x, q.y);
    }
    return smoothstep(0.0, feather, inside);
}

@compute @workgroup_size(8, 8, 1)
fn compose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_art);
    if (any(gid.xy >= size)) {
        return;
    }
    let dims = vec2<i32>(textureDimensions(weather_tile));
    let texel = vec2<i32>(params.origin) + vec2<i32>(gid.xy);
    let t = (texel % dims + dims) % dims;
    var art = textureLoad(weather_tile, t, 0);
    var phys = vec4(1.0, art.g, params.droplet_number, 0.0);
    let count = min(bitcast<u32>(params.count), MAX_INFLUENCES);
    if (count > 0u) {
        // The texel's position in the map frame (m), unrotated (sample_weather rotates world xz
        // by 30 degrees) and moved by the advection offset: the world point whose clouds it feeds.
        let m = (vec2<f32>(texel) + 0.5) / vec2<f32>(dims) * params.map_scale;
        let unrot = mat2x2<f32>(0.8660254, 0.5, -0.5, 0.8660254) * m;
        let p = unrot + params.map_offset;
        for (var i = 0u; i < count; i++) {
            let inf = params.influences[i];
            let w = inf.mode.y * influence_mask(inf, p);
            if (w <= 0.0) {
                continue;
            }
            let mode = bitcast<u32>(inf.mode.x);
            let channels = bitcast<u32>(inf.mode.z);
            for (var c = 0u; c < 4u; c++) {
                if ((channels & (1u << c)) != 0u) {
                    art[c] = mix(art[c], blend(art[c], inf.art[c], mode), w);
                }
                if ((channels & (1u << (c + 4u))) != 0u) {
                    phys[c] = mix(phys[c], blend(phys[c], inf.physical[c], mode), w);
                }
            }
        }
        art = vec4(saturate(art.rgb), art.a);
        phys = vec4(max(phys.r, 0.0), saturate(phys.g), max(phys.b, 1.0), phys.a);
    }
    textureStore(out_art, vec2<i32>(gid.xy), art);
    textureStore(out_physical, vec2<i32>(gid.xy), phys);
}
