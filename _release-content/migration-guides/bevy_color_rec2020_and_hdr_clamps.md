---
title: "`bevy_color`: new `Color::LinearRec2020` variant and HDR-safe clamping"
pull_requests: []
---

## New `Color::LinearRec2020` variant

`Color` has a new variant, `Color::LinearRec2020(LinearRec2020)`: linear RGB with
wide-gamut Rec.2020 primaries. If you match exhaustively on `Color`, add an arm for it
(or a wildcard). Conversions (`From`/`Into`, `Color::to_linear`, `Color::to_srgba`)
cover it like every other space.

## Color operations no longer clamp HDR values

Operations like `with_luminance`, `lighter`, and `darker` used to clamp their results
into `[0.0, 1.0]`, which silently threw away HDR (brighter-than-white) and wide-gamut
colors. They now clamp only SDR inputs: colors inside `[0.0, 1.0]` behave exactly as
before, while HDR and wide-gamut colors keep their extra brightness and saturation.
The `Laba` -> `Lcha` conversion also no longer clamps chroma, so wide-gamut colors
round-trip losslessly.

If you relied on these methods to pull values back into SDR range, clamp explicitly
(e.g. `c.red.clamp(0., 1.)`) or convert through `ColorToPacked`, which still quantizes
to `[0, 1]`.
