---
title: "`Window` now requires the `DisplayTarget` and `EffectiveDisplayTarget` components"
pull_requests: []
---

`Window` now requires the new `DisplayTarget` component, which describes the
display the window is presented on (paper white luminance, peak luminance, black
level, color gamut, transfer function) and is the foundation for HDR display
output (see the "HDR display output (scRGB-linear and HDR10/PQ)" release note).
It also requires the derived `EffectiveDisplayTarget`: the resolved calibration
the renderer actually consumes, which the engine rewrites in place from
`DisplayTarget`, `DisplayCalibrationPolicy`, and sensed display information.
Treat `EffectiveDisplayTarget` as read-only.

Through the required-component machinery every `Window` receives a
`DisplayTarget` defaulting to `DisplayTarget::SDR_SRGB` (100 nits paper white and
peak, Rec.709 gamut, sRGB transfer) and a matching `EffectiveDisplayTarget`.
This matches Bevy's previous behavior, so output is unchanged and no action is
required.

Note that:

- `Query<&DisplayTarget, With<Window>>` now matches all window entities, and
  archetype-based assumptions about window entities (e.g. exact component sets in
  tests or editors) must account for the extra components, `DisplayTarget` and
  `EffectiveDisplayTarget`.
- Window entities serialized with reflection-based scene formats now include
  `DisplayTarget` and `EffectiveDisplayTarget` alongside `Window`.
- To override the default, insert your own value when spawning, e.g.
  `commands.spawn((Window::default(), DisplayTarget { peak_luminance_nits: 1000.0, ..Default::default() }))`.

A new `WindowMonitorChanged` message is registered by `WindowPlugin` and emitted
by `bevy_winit` when the monitor a window is on changes. Bevy never mutates
`DisplayTarget` automatically — the component is user-authoritative — so existing
code is unaffected unless you listen for the new message.
