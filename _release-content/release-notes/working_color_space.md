---
title: "Wide working color space (Rec.2020, opt-in)"
authors: ["@stuartparmenter"]
pull_requests: []
---

Bevy's scene-referred rendering has always used linear Rec.709 (the sRGB
primaries) as its working color space, implicitly. That axis is now explicit
and configurable: `RenderPlugin` has a new `working_color_space` field of type
`bevy_render::working_color_space::WorkingColorSpace`.

```rust
use bevy::render::{working_color_space::WorkingColorSpace, RenderPlugin};

App::new().add_plugins(DefaultPlugins.set(RenderPlugin {
    working_color_space: WorkingColorSpace::Rec2020,
    ..default()
}));
```

The default, `WorkingColorSpace::Rec709`, is bit-for-bit identical to previous
releases. `WorkingColorSpace::Rec2020` switches the scene-referred buffers and
lighting math to the wide-gamut ITU-R BT.2020 primaries (D65 white point
throughout), which cover roughly twice the visible gamut of Rec.709 — including
saturated real-world colors (car paints, neon, lasers) that Rec.709 cannot
represent with non-negative components. Rec.2020 is also the native space of
the new `Tonemapping::GranTurismo7` operator, which consumes it without
conversion.

This is a project-global, startup-time setting (Unreal-style "working color
space"): shared assets and buffers make per-camera working spaces impractical,
so all cameras share one set of primaries. It is read once when `RenderPlugin`
builds; mutating the resource afterwards has no effect.

Under `Rec2020`:

- Light colors (point/spot/directional/rect), ambient light, distance fog and
  clear colors convert Rec.709 → Rec.2020 on the CPU at their extract/prepare
  seams, through `bevy_render::working_color_space::linear_rgba_rec709_to_working`.
- Color quantities composed in shaders from Rec.709 factors — PBR base color
  and emissive (material factor × texture × vertex color), lightmap samples,
  environment-map radiance, skybox samples, sprite / `ColorMaterial` /
  tilemap colors — convert once after composition. 3D pipelines key the
  conversion on the global `WORKING_COLOR_SPACE_REC2020` shader def; 2D
  writer pipelines on the `OUTPUT_GAMUT_REC2020` writer-encode def, pushed
  from the same setting. All sampled color textures are
  assumed authored against Rec.709 (this also covers compressed textures, which
  cannot be converted on the CPU). Textures stamped with wide primaries via
  `Image::source_primaries` have no per-texture escape hatch yet, but the stamp
  is now propagated to `GpuImage::source_primaries` so one can be added later.
- `Tonemapping::GranTurismo7` runs natively on the Rec.2020 values. Every other
  operator (and the color-grading stack) is Rec.709-fit — the AgX / Tony
  McMapface / Blender Filmic LUTs have no algorithmic source to rebake from — so
  the tone mapping pass converts Rec.2020 → Rec.709 at its entry for them,
  clipping colors outside the Rec.709 gamut.
- The tone mapping pass outputs Rec.709 display-linear for every operator,
  except `Tonemapping::GranTurismo7` on an HDR-transfer display target: there
  the operator emits its native linear Rec.2020 straight into the
  display-encoding pass, and Rec.2020 values flow all the way to the HDR display
  signal (see the Gran Turismo 7 tonemapping and display-encoding pass release
  notes). Every other configuration keeps Rec.709
  output; FXAA/SMAA and the display-encoding pass are unaffected.

Some parts of the renderer are not yet converted under `Rec2020` and stay
Rec.709-fit:

- `Tonemapping::None` cameras (the `Camera2d` default) skip the conversion and
  render desaturated; a `warn_once` fires. Give them the new
  `Tonemapping::Linear`, which runs the conversion, grading, and dither with no
  tone curve.
- `CompositingSpace::Oklab` and the bloom luminance weights.
- Clustered decals and irradiance volumes.
- The `specular_tint` and clearcoat tint material inputs.
- `bevy_solari` (the experimental real-time path tracer).
- Atmosphere-generated sky values.
