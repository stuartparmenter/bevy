// Mip chain of the cloud shadow map (`node.rs`, after the temporal pass): each level averages
// 2x2 texels of the one above, so lookups can read the map at their pixel footprint (Williams
// 1983, pyramidal parametrics; Reeves, Salesin & Cook 1987, the filter region of a shadow lookup
// is the pixel's footprint projected into the map). Far from the camera, and at grazing views of
// the ground, a pixel covers several texels; reading one texel there aliased the cloud shadows
// (a checker in the far haze, flicker at the map's rim while the light turns).
//
// What is averaged is what the lookups need averaged: the visibility (Lokovic & Veach 2000, deep
// shadow maps, prefilter and mip-map the fractional visibility over each pixel's area).
// - B, A (cumulus and thin-layer optical depth): averaged as transmittance. Averaging optical
//   depth turns a texel 10 % covered by a cumulus (depth ~50) into transmittance exp(-5) = 0.007
//   instead of 0.9.
// - G (mean extinction): linear mean, as UE5's cloud shadow downsample (VolumetricCloud.usf
//   MainShadowFilterCS).
// - R (front depth): the nearest of the four. Clear texels hold the far plane; UE5's downsample
//   (whose "mean - mean abs deviation" reduces to the mean, the sum not being divided by 4) put a
//   texel with one cloud and three clear texels behind the ground, and its shadow was lost.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(dst);
    if (any(id.xy >= size)) {
        return;
    }
    let src_max = vec2<i32>(textureDimensions(src)) - 1;
    var front = 1e30;
    var ext = 0.0;
    var vis = 0.0;
    var vis_thin = 0.0;
    for (var k = 0; k < 4; k++) {
        let c = min(vec2<i32>(id.xy) * 2 + vec2(k & 1, k >> 1), src_max);
        let q = textureLoad(src, c, 0);
        front = min(front, q.r);
        ext += q.g;
        vis += exp(-max(q.b, 0.0));
        vis_thin += exp(-max(q.a, 0.0));
    }
    // Below ~1e-30 (depth ~69) every texel is opaque: keep the deepest representable depth.
    let b = select(69.0, -log(vis * 0.25), vis > 4e-30);
    let a = select(69.0, -log(vis_thin * 0.25), vis_thin > 4e-30);
    textureStore(dst, id.xy, vec4(front, ext * 0.25, b, a));
}
