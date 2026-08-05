---
title: "`Window` now requires the `DisplayTarget` and `EffectiveDisplayTarget` components"
pull_requests: []
---

`Window` now requires two new components: `DisplayTarget`, the calibration of the
display the window is presented on (paper white and peak luminance, black level,
color gamut, transfer function), and `EffectiveDisplayTarget`, the resolved
calibration the renderer consumes (written by the engine; treat it as read-only).
Every window gets a `DisplayTarget` defaulting to `DisplayTarget::SDR_SRGB`, which
matches the previous behavior, so most code needs no change. See the "HDR display
output (scRGB-linear and HDR10/PQ)" release note for what `DisplayTarget` enables.

- Archetype-based assumptions about window entities (e.g. exact component sets in
  tests or editors) must account for the two extra components.
- Window entities serialized with reflection-based scene formats now include
  `DisplayTarget` and `EffectiveDisplayTarget` alongside `Window`.
