---
title: "`RenderPlugin::working_color_space` and `GpuImage::source_primaries`"
pull_requests: []
---

`RenderPlugin` has a new field `working_color_space: bevy_render::working_color_space::WorkingColorSpace`. `..default()` is unaffected; if you construct it field-by-field, add the new field. The default `WorkingColorSpace::Rec709` preserves the previous rendering bit-for-bit.

A `WorkingColorSpace` resource is now present in both the main and render worlds, but it is read exactly once when `RenderPlugin` builds — pipelines are specialized against it at startup, so changing it at runtime has no effect.

New `working_color_space` fields appear on `MeshPipeline`, `PrepassPipeline`, `Mesh2dPipeline`, and `SpritePipeline`; add them if you construct these directly (rare). `GpuImage` gains `source_primaries: bevy_image::SourceColorPrimaries`, propagated from `Image::source_primaries` at `prepare_asset` time; when constructing a `GpuImage` manually, stamp the source image's value or `Default::default()` (`Bt709`).

`WorkingColorSpace::Rec2020` is opt-in (new; nothing changes otherwise). When enabled:

- Scene-linear buffers, light/fog/clear colors, and composed material colors hold linear Rec.2020 values. Custom materials and custom render passes that inject Rec.709 colors must convert them with `bevy_render::working_color_space::linear_rgba_rec709_to_working` (CPU) or the `bevy_render::working_color_space` WGSL module's `rec709_to_rec2020` under the `WORKING_COLOR_SPACE_REC2020` shader def. 2D writer pipelines (sprites, 2D meshes and materials, gizmos, UI) receive the composed-color conversion as the `OUTPUT_GAMUT_REC2020` writer-encode def instead; a custom `Material2d` fragment shader can call `bevy_render::writer_encode::writer_encode` on its composed color to handle it.
- Every camera needs an active `Tonemapping` operator, since the Rec.2020 → display conversion runs in the tonemapping pass. `Tonemapping::None` cameras (the `Camera2d` default) render reinterpreted, desaturated colors (a `warn_once` diagnoses this); use `Tonemapping::Linear` for the conversion with no tone curve.
- Operators other than `Tonemapping::GranTurismo7` are Rec.709-fit and clip working-space colors outside the Rec.709 gamut at the tone mapping pass entry.
- Some parts of the renderer are not yet converted and stay Rec.709-fit: `CompositingSpace::Oklab` and the bloom luminance weights; clustered decals and irradiance volumes; the `specular_tint` and clearcoat tint material inputs; `bevy_solari`; and atmosphere-generated sky values.
- `LinearRgba` (and the rest of `bevy_color`) stays defined as linear Rec.709; the conversion to the working space happens once, at the render-world seams above. Do not pre-convert colors you hand to standard Bevy APIs.
