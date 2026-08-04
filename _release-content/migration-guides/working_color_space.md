---
title: "`RenderPlugin::working_color_space` and `GpuImage::source_primaries`"
pull_requests: []
---

`RenderPlugin` has a new field `working_color_space: bevy_render::working_color_space::WorkingColorSpace`. Add it if you construct `RenderPlugin` field-by-field. The default `WorkingColorSpace::Rec709` renders bit-for-bit as before.

A `WorkingColorSpace` resource is now in both the main and render worlds. `RenderPlugin` registers the `WORKING_COLOR_SPACE_REC2020` shader def globally on the `PipelineCache`. Changing the resource at runtime has no effect.

New `working_color_space` fields appear on `Mesh2dPipeline` and `SpritePipeline`. If you construct these directly, fill them from the `WorkingColorSpace` resource. `GpuImage` gains `source_primaries: bevy_image::SourceColorPrimaries`, propagated from `Image::source_primaries`. When constructing a `GpuImage` manually, stamp the source image's value or `Default::default()` (`Bt709`).

Under `WorkingColorSpace::Rec2020`:

- Scene-linear buffers, light/fog/clear colors, and composed material colors hold linear Rec.2020 values. Custom materials and custom render passes that inject Rec.709 colors must convert them, with `linear_rgba_rec709_to_working` on the CPU or `rec709_to_rec2020` in WGSL under the `WORKING_COLOR_SPACE_REC2020` shader def, both in `bevy_render::working_color_space`. 2D writer pipelines (sprites, 2D meshes and materials, gizmos, UI) receive the composed-color conversion as the `OUTPUT_GAMUT_REC2020` writer-encode def instead. A custom `Material2d` fragment shader can call `bevy_render::writer_encode::writer_encode` on its composed color to handle it.
- Every camera needs an active `Tonemapping` operator, since the Rec.2020 to display conversion runs in the tonemapping pass. `Tonemapping::None` cameras (the `Camera2d` default) render reinterpreted, desaturated colors, and a `warn_once` diagnoses this. Use `Tonemapping::Linear` for the conversion with no tone curve.
- Operators other than `Tonemapping::GranTurismo7` are Rec.709-fit and clip working-space colors outside the Rec.709 gamut at the tonemapping pass entry.
- Parts of the renderer are not yet converted and stay Rec.709-fit. The "Wide-gamut color" release note lists them.
- `LinearRgba` (and the rest of `bevy_color`) stays defined as linear Rec.709. The conversion happens at the render-world seams above, so do not pre-convert colors you hand to standard Bevy APIs.
