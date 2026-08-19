---
title: "`Window` now requires the `DisplayTarget` and `EffectiveDisplayTarget` components"
pull_requests: []
---

`Window` now requires two new components: `DisplayTarget`, the calibration of the
display the window is presented on, and `EffectiveDisplayTarget`, the resolved
calibration the renderer consumes (written by the engine; treat it as read-only).
`DisplayTarget` defaults to `DisplayTarget::SDR_SRGB`, which matches 0.19
behavior, so most code needs no change. See the release note
"`DisplayTarget`: per-display calibration for windows" for what these components
hold.

- Code that assumes an exact component set on window entities, in tests or
  editors, must account for the two extra components.
- Window entities serialized with reflection-based scene formats now include
  `DisplayTarget` and `EffectiveDisplayTarget`.
