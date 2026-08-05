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
output is parameterized by it: peak-aware tone mapping
(`Tonemapping::GranTurismo7`), wide-gamut Rec.2020 output through the
display-encoding pass, and the `Rgba16Float` scRGB swapchain format (see the
"HDR display output (scRGB-linear and HDR10/PQ)" release note).

`EffectiveDisplayTarget`, also required on `Window`, is the resolved
calibration the renderer consumes: your `DisplayTarget` plus anything
`DisplayCalibrationPolicy` lets the engine sense. The engine rewrites it in
place, so read it, don't write it. Two extra required components mean new
window archetypes and reflected scene data. See the migration guide "`Window`
now requires the `DisplayTarget` and `EffectiveDisplayTarget` components".

To see what a window actually got, read its `WindowSurfaceTransfers` component:
`resolved` is the transfer the surface ended up with, `supported` lists the
transfers it could present. It is read-only and one frame behind negotiation.

In the render world, each camera view carries its target's resolved calibration
as a `ViewDisplayTarget`. A per-view `DisplayTargetUniform` carries the
paper-white luminance to the GPU, importable in WGSL as
`bevy_render::display_target`. It is only present on views whose resolved
transfer is HDR, and it is not in the view bind group, so a custom pass has to
bind it itself and handle the SDR case.

Render targets that aren't windows, such as `RenderTarget::Image` and
`RenderTarget::TextureView` (used by OpenXR), have no window entity to hold the
component. The new `ManualDisplayTargets` resource in `bevy_render` maps a
`NormalizedRenderTarget` to its `DisplayTarget` for those, and targets without
an entry fall back to `DisplayTarget::SDR_SRGB`.

Bevy never overwrites values you set on `DisplayTarget`. To react to a window
moving to a different display, watch the `OnMonitor` relationship on the window
entity. `bevy_winit` retargets it whenever the window's monitor changes,
including when it first becomes known.

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
values are missing or coarse on many platforms. An in-app HGIG-style
calibration flow fills the gaps.

`examples/3d/hdr_calibration.rs` runs that flow as a three-step wizard, the way
most games calibrate: a probe square on a reference card, with per-step
luminances. Raise peak until the max-signal probe disappears into the card, set
paper white to a comfortable white level, and lower black level until the probe
disappears into the black background. It persists the result via
`bevy_settings`, reloads it on the next run, and prompts to recalibrate when
the window moves to another monitor. `M` picks the mode up front: manual HGIG,
where you tune every value, or trust-OS, where peak, black, and gamut resolve
from sensed data (`DisplayCalibrationPolicy`) and you only set paper white.

All adjustments mutate the primary window's `DisplayTarget` live. The pattern
renders with `Tonemapping::Linear` and unlit materials, so it reaches the
display encoder at exact paper-white-relative values (above `1.0` reaches
peak). Calibrating needs an HDR display: macOS/iOS Metal, Windows Vulkan or
DX12, or Wayland Vulkan with Mesa 25.1+. Without one the example shows a "no
HDR output" notice.
