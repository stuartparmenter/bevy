//! Encodes a fragment's output color for the view's compositing space, so
//! blending happens in that space. The upscaling blit decodes back to linear.
//! Pipelines whose main texture holds working color space values push
//! `OUTPUT_WORKING_COLOR_SPACE` to convert their Rec.709 colors first.

@if(WORKING_COLOR_SPACE_REC2020 && OUTPUT_WORKING_COLOR_SPACE)
import package::working_color_space::rec709_to_rec2020;
@if(COMPOSITING_SPACE_SRGB)
import package::color_operations::linear_to_srgb;
@if(COMPOSITING_SPACE_OKLAB)
import package::color_operations::linear_rgb_to_oklab;

fn writer_encode(color_in: vec4<f32>) -> vec4<f32> {
    var color = color_in;
@if(WORKING_COLOR_SPACE_REC2020 && OUTPUT_WORKING_COLOR_SPACE)
    color = rec709_to_rec2020(color);
@if(COMPOSITING_SPACE_SRGB) {
    color = vec4(linear_to_srgb(color.rgb), color.a);
}
@if(COMPOSITING_SPACE_OKLAB) {
    color = vec4(linear_rgb_to_oklab(color.rgb), color.a);
}
    return color;
}
