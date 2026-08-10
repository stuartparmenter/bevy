---
title: "`bevy_color`: new `Color::LinearRec2020` variant and HDR-safe clamping"
pull_requests: []
---

`Color` has a new variant, `Color::LinearRec2020(LinearRec2020)` (linear RGB with
wide-gamut Rec.2020 primaries), which must be handled during exhaustive `match`
statements. Conversions (`From`/`Into`, `Color::to_linear`, `Color::to_srgba`) cover
it like every other space.

`with_luminance`, `lighter`, and `darker` no longer clamp their results into
`[0.0, 1.0]` for HDR (brighter-than-white) and wide-gamut inputs. Colors inside
`[0.0, 1.0]` behave exactly as before. The `Laba` to `Lcha` conversion no longer
clamps chroma. If you relied on these methods to pull values back into SDR range,
clamp explicitly (`c.red.clamp(0., 1.)`) or convert through `ColorToPacked`, which
still quantizes to `[0, 1]`.
