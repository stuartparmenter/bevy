---
title: "Wide working color space: opt-in Rec.2020 rendering"
authors: ["@stuartparmenter"]
pull_requests: []
---

Bevy's scene-referred rendering has always used linear Rec.709 (the sRGB
primaries). The new `working_color_space` field on `RenderPlugin` makes it
configurable:

```rust
use bevy::render::{RenderPlugin, WorkingColorSpace};

App::new().add_plugins(DefaultPlugins.set(RenderPlugin {
    working_color_space: WorkingColorSpace::Rec2020,
    ..default()
}));
```

The default, `WorkingColorSpace::Rec709`, is bit-for-bit identical to previous
releases. `WorkingColorSpace::Rec2020` switches the scene-referred buffers and
lighting math to the Rec.2020 primaries. Rec.2020 reaches saturated real-world
colors, such as car paints, neon, and lasers, that Rec.709 cannot represent with
non-negative components.

The setting is project-global. `RenderPlugin` reads it once at build time, and
mutating the resource afterwards has no effect. Bevy converts light, fog, and
clear colors on the CPU, and converts shader-composed colors once after
composition: PBR base color and emissive, lightmaps, environment maps, skyboxes,
sprites, `ColorMaterial`, and tilemaps. Every sampled color texture is assumed to
be authored in Rec.709.

`LinearRgba` and the rest of `bevy_color` stay defined as linear Rec.709. The
conversion happens at these render-world seams, so do not pre-convert colors you
hand to standard Bevy APIs. Custom materials and custom render passes that inject
Rec.709 colors must convert them: `linear_rgba_rec709_to_working` on the CPU, or
`rec709_to_rec2020` in WGSL under the `WORKING_COLOR_SPACE_REC2020` shader def,
both in `bevy_render::working_color_space`. 2D writer pipelines (sprites, 2D
meshes and materials, gizmos, UI) get the `OUTPUT_GAMUT_REC2020` writer-encode
def instead; a custom `Material2d` fragment shader can call
`bevy_render::writer_encode::writer_encode` on its composed color.

`Tonemapping::GranTurismo7` runs natively on Rec.2020 values. Every other
operator, and the color-grading stack, is Rec.709-fit, so the tone-mapping pass
converts Rec.2020 to Rec.709 at its entry and clips out-of-gamut colors.
`Tonemapping::None` cameras (the `Camera2d` default) skip that conversion and
render desaturated, with a `warn_once`; the new `Tonemapping::Linear` runs the
conversion, grading, and dither with no tone curve.

These parts stay Rec.709-fit under `Rec2020`: `CompositingSpace::Oklab`, the
bloom luminance weights, clustered decals, irradiance volumes, the
`specular_tint` and clearcoat tint material inputs, `bevy_solari`, and
atmosphere-generated sky values.
