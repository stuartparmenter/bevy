---
title: "`DisplayTarget`: per-display calibration for windows"
authors: ["@stuartparmenter"]
pull_requests: []
---

The new `DisplayTarget` component in `bevy_window` describes the display a
window is presented on: paper white, peak luminance, black level, color gamut
(`DisplayGamut`: Rec.709, Display P3, or Rec.2020), and the transfer function
the final signal is encoded with (`DisplayTransfer`: sRGB, scRGB-linear, PQ, or
the encoded extended-range `ExtendedSrgb`). It is a required component of
`Window`. The default, `DisplayTarget::SDR_SRGB` (paper white and peak of 100
nits, Rec.709, sRGB), reproduces Bevy's existing SDR output. This release's HDR
output is parameterized by it; see the release note "HDR display output
(scRGB-linear and HDR10/PQ)".

`EffectiveDisplayTarget`, also required on `Window`, is the resolved
calibration the renderer consumes: your `DisplayTarget` plus anything
`DisplayCalibrationPolicy` lets the engine sense. The engine rewrites it in
place, so read it, don't write it. See the migration guide "`Window` now
requires the `DisplayTarget` and `EffectiveDisplayTarget` components".

To see what a window actually got, read its `WindowSurfaceTransfers` component:
`resolved` is the transfer the surface ended up with, `supported` lists the
transfers it could present. It is read-only and one frame behind negotiation.

In the render world, each camera view carries its target's resolved calibration
as a `ViewDisplayTarget`. A per-view `DisplayTargetUniform` carries the
paper-white luminance to the GPU, importable in WGSL as
`bevy_render::view::display_target`. It exists only on HDR-target views and
is not in the view bind group, so a custom pass must bind it itself and handle
the SDR case.

Render targets that aren't windows, such as `RenderTarget::Image` and
`RenderTarget::TextureView`, have no window entity to hold the component. The
new `ManualDisplayTargets` resource in `bevy_render` maps a
`NormalizedRenderTarget` to its `DisplayTarget` for those, and targets without
an entry fall back to `DisplayTarget::SDR_SRGB`.

Bevy never overwrites values you set on `DisplayTarget`. To react to a window
moving to a different display, watch the `OnMonitor` relationship on the window
entity. `bevy_winit` retargets it whenever the window's monitor changes.

`DisplayTarget` also gained builder-style helpers:

```rust
let hdr = DisplayTarget::SDR_SRGB
    .with_paper_white(200.0)
    .with_peak(1000.0)
    .with_transfer(DisplayTransfer::ScRgbLinear);
```

## The `hdr_calibration` example

`WindowDisplayState` and `MonitorDisplayCapability` now report values sensed by
wgpu: peak, black, full-frame and SDR-white nits, headroom, and a gamut bucket.
But paper white is a viewing preference no display can report, and sensed
values are missing or coarse on many platforms.

`examples/3d/hdr_calibration.rs` fills the gaps with a three-step HGIG-style
wizard, the way most games calibrate: a probe square on a reference card. Raise
peak until the max-signal probe disappears into the card, set paper white to a
comfortable white level, and lower black level until the probe disappears into
the black background. It persists the result via `bevy_settings` and reloads it
on the next run. `M` picks the mode up front: manual HGIG, where you tune every
value, or trust-OS, where peak, black, and gamut resolve from sensed data
(`DisplayCalibrationPolicy`) and you only set paper white.

All adjustments mutate the primary window's `DisplayTarget` live. Calibrating
needs an HDR display; without one the example shows a "no HDR output" notice.
