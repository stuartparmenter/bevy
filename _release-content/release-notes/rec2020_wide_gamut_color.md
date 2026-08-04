---
title: "Wide-gamut color: Rec.2020 in `bevy_color` and an opt-in wide working color space"
authors: ["@stuartparmenter"]
pull_requests: []
---

`bevy_color` gained a `LinearRec2020` color space: linear RGB with the ITU-R
BT.2020 (Rec.2020) primaries, the standard container gamut for HDR displays and
video, about twice the area of sRGB. It converts to and from every other color
space in the crate, has a `Color::LinearRec2020` variant, and supports `Mix`,
`Luminance` with the BT.2020 luminance weights, splines via `VectorSpace`,
reflection, and serialization.

Authoring wide-gamut and HDR colors no longer needs hand-computed negative sRGB
components:

```rust
// A vivid Rec.2020 red, far outside the sRGB gamut:
let red = Color::rec2020(1.0, 0.0, 0.0);
// A Display P3 color, exactly as shown in a macOS/CSS color picker:
let p3 = Color::display_p3(1.0, 0.2, 0.1);
// Any visible chromaticity via CIE xyY coordinates, e.g. D65 white at 5x paper white:
let bright = Color::cie_xy_y(0.3127, 0.3290, 5.0);
```

The new `primaries` module holds `Chromaticity` (CIE 1931 xy coordinates),
`RgbPrimaries` (primary sets, with constants for `BT709`, `BT2020`,
`DISPLAY_P3`, and `ACES_CG`), and `rgb_to_rgb_matrix`, which derives a
conversion matrix between any two primary sets at runtime.

`lighter`, `with_luminance`, and the `Laba`/`Lcha` conversions now preserve
brighter-than-white and out-of-sRGB values instead of clamping them. SDR colors
behave as before. See the `bevy_color` HDR-safe clamping migration guide.

## Wide working color space (Rec.2020, opt-in)

Bevy's scene-referred rendering has always used linear Rec.709 (the sRGB
primaries) as its working color space. The new `working_color_space` field on
`RenderPlugin` makes it configurable:

```rust
use bevy::render::{RenderPlugin, WorkingColorSpace};

App::new().add_plugins(DefaultPlugins.set(RenderPlugin {
    working_color_space: WorkingColorSpace::Rec2020,
    ..default()
}));
```

The default, `WorkingColorSpace::Rec709`, is bit-for-bit identical to previous
releases. `WorkingColorSpace::Rec2020` switches the scene-referred buffers and
lighting math to the BT.2020 primaries, D65 white point throughout. Rec.2020
reaches saturated real-world colors, such as car paints, neon, and lasers,
that Rec.709 cannot represent with non-negative components.

The setting is project-global. `RenderPlugin` reads it once at build time, and
mutating the resource afterwards has no effect. Custom materials and custom
render passes need changes; see the "`RenderPlugin::working_color_space` and
`GpuImage::source_primaries`" migration guide.

Under `Rec2020`:

- Light colors (point/spot/directional/rect), ambient light, distance fog and
  clear colors convert Rec.709 to Rec.2020 on the CPU.
- Colors composed in shaders from Rec.709 factors convert once after
  composition: PBR base color and emissive (material factor * texture * vertex
  color), lightmap samples, environment-map radiance, skybox samples, and
  sprite / `ColorMaterial` / tilemap colors. Every sampled color texture is
  assumed to be authored in Rec.709, compressed ones included, since those
  cannot be converted on the CPU.
- `Tonemapping::GranTurismo7` runs natively on Rec.2020 values. Every other
  operator, and the color-grading stack, is Rec.709-fit. The tonemapping pass
  converts Rec.2020 to Rec.709 at its entry, clipping out-of-gamut colors.
- FXAA/SMAA and the display-encoding pass are unaffected by the working color
  space setting.

Some parts are not yet converted under `Rec2020` and stay Rec.709-fit:

- `Tonemapping::None` cameras (the `Camera2d` default) skip the conversion and
  render desaturated, with a `warn_once`. The new `Tonemapping::Linear` runs the
  conversion, grading, and dither with no tone curve.
- `CompositingSpace::Oklab` and the bloom luminance weights.
- Clustered decals and irradiance volumes.
- The `specular_tint` and clearcoat tint material inputs.
- `bevy_solari` (the experimental real-time path tracer).
- Atmosphere-generated sky values.

## Color-primaries metadata for image assets

Image assets now carry their source gamut in the new `Image::source_primaries`
field (`SourceColorPrimaries`: `Bt709`, `Bt2020`, or `DisplayP3`). The KTX2,
Radiance HDR, and OpenEXR loaders fill it in from file metadata. A new optional
`source_primaries` loader setting pins it per asset, in code or in a `.meta`
file. The `Image::source_primaries` migration guide lists the per-format
metadata sources and the resolution order.

For now this is metadata only: decoding and rendering are unchanged.
