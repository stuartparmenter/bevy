---
title: "`RenderPlugin::working_color_space` and `GpuImage::source_primaries`"
pull_requests: []
---

`RenderPlugin` has a new field, `working_color_space: WorkingColorSpace`. Add it if
you construct `RenderPlugin` field-by-field. The default, `WorkingColorSpace::Rec709`,
renders bit-for-bit as before; the opt-in `WorkingColorSpace::Rec2020` is covered in
the release note "Wide-gamut color: Rec.2020 in `bevy_color` and an opt-in wide
working color space".

`Mesh2dPipeline` has a new `working_color_space` field. If you construct it directly,
fill it from the `WorkingColorSpace` resource, now present in both the main and render
worlds.

`GpuImage` has a new field, `source_primaries: bevy_image::SourceColorPrimaries`,
propagated from `Image::source_primaries`. When constructing a `GpuImage` manually,
use the source image's value or `Default::default()` (`Bt709`).
