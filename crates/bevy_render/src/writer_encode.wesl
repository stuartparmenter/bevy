// Shared writer-side encode for shaders that compose a Rec.709-authored color
// and write it straight into a compositing buffer (sprites, 2D meshes and
// materials, gizmos, UI).
//
// Two def-gated steps, in buffer order:
//   1. `OUTPUT_GAMUT_REC2020`: the destination buffer uses Rec.2020 primaries,
//      so the composed Rec.709 color converts once, after composition.
//      Pre-tonemap writers key this off the project-global
//      `WorkingColorSpace`. Post-tonemap writers (UI) key it per view off the
//      buffer's resolved source gamut.
//   2. `COMPOSITING_SPACE_SRGB` or `COMPOSITING_SPACE_OKLAB`: the view
//      composites in an encoded space, so the output is encoded to match and
//      blending happens in that space. The terminal decode reverses it.

@if(OUTPUT_GAMUT_REC2020)
import package::working_color_space::rec709_to_rec2020;
@if(COMPOSITING_SPACE_SRGB)
import package::color_operations::linear_to_srgb;
@if(COMPOSITING_SPACE_OKLAB)
import package::color_operations::linear_rgb_to_oklab;

fn writer_encode(color_in: vec4<f32>) -> vec4<f32> {
    var color = color_in;
@if(OUTPUT_GAMUT_REC2020) {
    color = vec4(rec709_to_rec2020(color.rgb), color.a);
}
@if(COMPOSITING_SPACE_SRGB) {
    color = vec4(linear_to_srgb(color.rgb), color.a);
}
@if(COMPOSITING_SPACE_OKLAB) {
    color = vec4(linear_rgb_to_oklab(color.rgb), color.a);
}
    return color;
}
