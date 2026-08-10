---
title: "Wide-gamut color: Rec.2020 in `bevy_color` and an opt-in wide working color space"
authors: ["@stuartparmenter"]
pull_requests: []
---

`bevy_color` gained a `LinearRec2020` color space: linear RGB with the ITU-R
BT.2020 (Rec.2020) primaries, the standard container gamut for HDR displays and
video, about twice the area of sRGB. It converts to and from every other space in
the crate, has a `Color::LinearRec2020` variant, and supports `Mix`, `Luminance`
with the Rec.2020 weights, `VectorSpace` splines, reflection, and serialization.

Wide-gamut and HDR colors no longer need hand-computed negative sRGB components:

```rust
// A vivid Rec.2020 red, far outside the sRGB gamut:
let red = Color::rec2020(1.0, 0.0, 0.0);
// A Display P3 color, as shown in a macOS/CSS color picker:
let p3 = Color::display_p3(1.0, 0.2, 0.1);
// Any visible chromaticity via CIE xyY, here D65 white at 5x paper white:
let bright = Color::cie_xy_y(0.3127, 0.3290, 5.0);
```

The new `primaries` module holds `Chromaticity` (CIE 1931 xy coordinates),
`RgbPrimaries` (with constants for `BT709`, `BT2020`, `DISPLAY_P3`, and
`ACES_CG`), and `rgb_to_rgb_matrix`, which derives a conversion matrix between
any two primary sets.

`lighter`, `with_luminance`, and the `Laba`/`Lcha` conversions now preserve
brighter-than-white and out-of-sRGB values instead of clamping them. SDR colors
behave as before.

## Wide working color space (Rec.2020, opt-in)

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

## Color-primaries metadata for image assets

Image assets now carry their source gamut in the new `Image::source_primaries`
field (`SourceColorPrimaries`: `Bt709`, `Bt2020`, or `DisplayP3`). The KTX2,
Radiance HDR, and OpenEXR loaders read it from file metadata (the KTX2 data
format descriptor's `colorPrimaries`, Radiance `PRIMARIES=` header lines, the
OpenEXR `chromaticities` attribute), and a new optional
`source_primaries` loader setting pins it per asset, in code or in a `.meta`
file. Resolution order is setting, then file metadata, then `Bt709`. For now this
is metadata only: decoding and rendering are unchanged.
