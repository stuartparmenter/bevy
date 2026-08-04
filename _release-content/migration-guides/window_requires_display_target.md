---
title: "`Window` now requires the `DisplayTarget` and `EffectiveDisplayTarget` components"
pull_requests: []
---

`Window` now requires the new `DisplayTarget` component: the paper white
luminance, peak luminance, black level, color gamut, and transfer function of
the display the window is presented on. It also requires `EffectiveDisplayTarget`,
the resolved calibration the renderer consumes. The engine writes that one, so
treat it as read-only. See the "HDR display output (scRGB-linear and HDR10/PQ)"
release note.

Every `Window` therefore gets a `DisplayTarget` defaulting to
`DisplayTarget::SDR_SRGB` (100 nits paper white and peak, Rec.709 gamut, sRGB
transfer) and a matching `EffectiveDisplayTarget`.
That matches the previous behavior, so no action is required.

- `Query<&DisplayTarget, With<Window>>` now matches every window. Archetype-based
  assumptions about window entities (e.g. exact component sets in tests or
  editors) must account for the two extra components.
- Window entities serialized with reflection-based scene formats now include
  `DisplayTarget` and `EffectiveDisplayTarget` alongside `Window`.
- To override the default, insert your own value when spawning, e.g.
  `commands.spawn((Window::default(), DisplayTarget { peak_luminance_nits: 1000.0, ..Default::default() }))`.

Bevy never mutates `DisplayTarget`, so a window moving to a different monitor
does not change it. To react to monitor moves, watch the existing `OnMonitor`
relationship with `Changed<OnMonitor>` / `RemovedComponents<OnMonitor>`.
