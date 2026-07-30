---
title: "Wide-gamut color: Rec. 2020 in `bevy_color` and an opt-in wide working color space"
authors: ["@stuartparmenter"]
pull_requests: []
---

`bevy_color` can now represent wide-gamut colors as first-class citizens. The new
`LinearRec2020` color space is a linear RGB space using the ITU-R BT.2020
(Rec. 2020) primaries — the standard container gamut for HDR displays and video,
covering roughly twice the area of the sRGB gamut. It converts to and from every
other color space in the crate, has a `Color::LinearRec2020` variant, and supports
all the standard color operations (`Mix`, `Luminance` with the correct BT.2020
luminance weights, splines via `VectorSpace`, reflection, and serialization).

Authoring wide-gamut and HDR colors no longer requires hand-computed negative sRGB
components:

```rust
// A vivid Rec. 2020 red, far outside the sRGB gamut:
let red = Color::rec2020(1.0, 0.0, 0.0);
// A Display P3 color, exactly as shown in a macOS/CSS color picker:
let p3 = Color::display_p3(1.0, 0.2, 0.1);
// Any visible chromaticity via CIE xyY coordinates, e.g. D65 white at 5× paper white:
let bright = Color::xy_y(0.3127, 0.3290, 5.0);
```

Underneath, the new `primaries` module provides the building blocks the renderer's
wide-gamut working-space support (below) is built on: `Chromaticity` (CIE 1931 xy
coordinates), `RgbPrimaries` (primary sets with constants for `BT709`, `BT2020`,
`DISPLAY_P3`, and `ACES_CG`), and `rgb_to_rgb_matrix` for deriving conversion
matrices between any two primary sets at runtime.

Color operations are also HDR-safe: `lighter`, `with_luminance`, and the
`Laba`/`Lcha` conversions preserve brighter-than-white and out-of-sRGB values
instead of silently clamping them, while SDR colors behave exactly as before
(details in the `bevy_color` HDR-safe clamping migration guide).

## Wide working color space (Rec.2020, opt-in)

Bevy's scene-referred rendering has always used linear Rec.709 (the sRGB
primaries) as its working color space, implicitly. That axis is now explicit
and configurable: `RenderPlugin` has a new `working_color_space` field of type
`bevy_render::WorkingColorSpace`.

```rust
use bevy::render::{RenderPlugin, WorkingColorSpace};

App::new().add_plugins(DefaultPlugins.set(RenderPlugin {
    working_color_space: WorkingColorSpace::Rec2020,
    ..default()
}));
```

The default, `WorkingColorSpace::Rec709`, is bit-for-bit identical to previous
releases. `WorkingColorSpace::Rec2020` switches the scene-referred buffers and
lighting math to the wide-gamut ITU-R BT.2020 primaries (D65 white point
throughout) — including saturated real-world colors (car paints, neon, lasers)
that Rec.709 cannot represent with non-negative components. Rec.2020 is also
the native space of the new `Tonemapping::GranTurismo7` operator, which
consumes it without conversion.

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
  cannot be converted on the CPU); see the source-primaries metadata below for
  the path toward per-texture conversion.
- `Tonemapping::GranTurismo7` runs natively on the Rec.2020 values. Every other
  operator (and the color-grading stack) is Rec.709-fit — the AgX / Tony
  McMapface / Blender Filmic LUTs have no algorithmic source to rebake from — so
  the tone mapping pass converts Rec.2020 → Rec.709 at its entry for them,
  clipping colors outside the Rec.709 gamut.
- The tone mapping pass outputs Rec.709 display-linear for every operator,
  except `Tonemapping::GranTurismo7` on an HDR-transfer display target: there
  the operator emits its native linear Rec.2020 straight into the
  display-encoding pass, and Rec.2020 values flow all the way to the HDR display
  signal (see the Gran Turismo 7 and HDR display output release
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

## Color-primaries metadata for image assets

Image assets now carry their source gamut explicitly: `Image` has a new
`source_primaries: SourceColorPrimaries` field (`Bt709`, `Bt2020`, or `DisplayP3`),
and the image loaders fill it in from real file metadata:

- **KTX2**: the `colorPrimaries` field of the data format descriptor. The loader
  also warns (once) when the file's declared transfer function contradicts the
  `is_srgb` loader setting, or when the file declares an HDR transfer function such
  as PQ or HLG.
- **Radiance HDR (`.hdr`)**: `PRIMARIES=` header lines.
- **OpenEXR (`.exr`)**: the standardized `chromaticities` header attribute.
- **glTF**: textures are stamped BT.709, as the glTF 2.0 specification mandates.

You can also pin the primaries per asset via the new optional `source_primaries`
field on `ImageLoaderSettings`, `HdrTextureLoaderSettings`, and
`ExrTextureLoaderSettings` (in code or in `.meta` files). The resolution order is:
explicit setting, then file metadata, then the BT.709 default.

For now this is pure metadata — decoding and rendering are unchanged, untagged
assets behave as before, and the `Rec2020` working space still assumes every
sampled texture is authored against Rec.709. But the stamp is propagated to
`GpuImage::source_primaries`, so a per-texture conversion — letting a Rec. 709
HDRI, a Display P3 texture, and a Rec. 2020 video frame all land in the working
space with their colors intact instead of silently shifting saturation — can be
added later.
