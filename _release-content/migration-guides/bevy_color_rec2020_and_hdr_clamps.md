---
title: "`bevy_color`: new `Color::LinearRec2020` variant and HDR-safe clamping"
pull_requests: []
---

## New `Color::LinearRec2020` variant

`Color` has a new variant, `Color::LinearRec2020(LinearRec2020)`, a linear RGB color space
with wide-gamut Rec.2020 (ITU-R BT.2020) primaries. Exhaustive matches on `Color` must
handle it (or add a wildcard arm). `From`/`Into` cover conversions to and from every other
space, so `color.into()`, `Color::to_linear`, and `Color::to_srgba` still work.

## Clamping in color operations is now HDR-aware

Several `bevy_color` operations clamped results into the SDR `[0.0, 1.0]` range, losing
HDR (brighter-than-white) and wide-gamut (out-of-sRGB) values. They now clamp only SDR
inputs:

- `LinearRgba::with_luminance` and `LinearRec2020::with_luminance` no longer clamp each channel
  to `[0, 1]` when the input color or the target luminance is outside SDR.
- `Luminance::lighter` for `LinearRgba` and `LinearRec2020` still clamps to white for SDR colors.
  HDR colors (any channel or the luminance above `1.0`) now keep brightening at constant
  chromaticity: `LinearRgba::new(4.0, 0.0, 0.0, 1.0)` is no longer desaturated.
- `Luminance::lighter` for `Xyza`, `Laba`, `Lcha`, `Oklaba`, `Oklcha`, `Hsla`, and `Okhsla` still
  clamps to white when luminance/lightness is at most `1.0`. Colors already above `1.0` keep
  brightening instead of being pulled down to `1.0`.
- `Luminance::darker` for `Xyza`, `Hsla`, and `Okhsla` no longer clamps to an upper bound of `1.0`
  (it still clamps to black). This affects only inputs already brighter than white, or negative
  `amount` values outside the documented `[0.0, 1.0]` range.
- The `Laba` -> `Lcha` conversion no longer clamps chroma to `[0.0, 1.5]`. Every sRGB-gamut color
  has chroma below `1.5`, so only wide-gamut inputs change, and they now round-trip losslessly.

Quantization methods (`ColorToPacked::to_u8_array` and friends) still clamp to `[0, 1]`.

To force colors into SDR range, clamp explicitly, e.g.
`LinearRgba { red: c.red.clamp(0., 1.), .. }`, or convert through `ColorToPacked`.
