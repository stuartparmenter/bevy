---
title: Rec.2020 working color space
authors: ["@stuartparmenter"]
pull_requests: []
---

Bevy lights and blends colors in linear Rec.709, the primaries of sRGB, so lit colors can't be more saturated than Rec.709 allows. Many HDR displays can show more saturated colors than that. To render them, set the new `RenderPlugin::working_color_space` field to `WorkingColorSpace::Rec2020`. Lighting and blending then use the wider Rec.2020 primaries:

```rust
App::new().add_plugins(DefaultPlugins.set(RenderPlugin {
    working_color_space: WorkingColorSpace::Rec2020,
    ..default()
}));
```

The working color space can't change after startup. Colors you author, such as `Color` values and textures, stay Rec.709, and Bevy converts them. Custom shaders that write color before tonemapping must convert their Rec.709 inputs themselves. Use `WorkingColorSpace::convert_rec709` on the CPU, or `rec709_to_rec2020` from the `bevy_render::working_color_space` shader module under the `WORKING_COLOR_SPACE_REC2020` shader def.

`Tonemapping::GranTurismo7` keeps the wider colors on an HDR display. `Tonemapping::None` passes them through, and so does `Tonemapping::Linear` when color grading is at its default. The other methods are designed for Rec.709, so they convert to Rec.709 before they tonemap.

## Gamut compression

A display whose gamut is narrower than Rec.2020 can't show every color that a camera with one of these methods outputs. Bevy compresses those colors toward neutral, in the style of the ACES Reference Gamut Compression, instead of clipping each channel. This also shifts the most saturated colors inside the display's gamut a little. Compression applies on every display except PQ, which uses the Rec.2020 primaries.
