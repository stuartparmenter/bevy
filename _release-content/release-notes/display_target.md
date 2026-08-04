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

`examples/helpers/hdr_calibration.rs` packages that flow as
`HdrCalibrationPlugin<S>`. Drop it into one of your own `States` and it runs a
three-step wizard, persists the result next to the executable, emits a
`CalibrationComplete` event on confirm, and prompts to recalibrate when the
window moves to another monitor. Pick a `CalibrationStrategy` up front: manual
HGIG, where you tune every value, or trust-OS, where peak, black, and gamut
resolve from sensed data and you only set paper white. The three steps are:

- `peak_luminance_nits`: a solid clipped near-peak surround, a true-black
  separating frame, and a center patch at the candidate peak
  (`PeakFraction(1.0)`). Raise the value until the patch merges into the
  surround, then back off one tap.
- `paper_white_nits`: a reference white card at exactly `1.0` next to a
  203-nit ITU-R BT.2408 strip.
- `min_luminance_nits`: near-black steps at fixed absolute luminances.

All adjustments mutate the primary window's `DisplayTarget` live. The patterns
render with `Tonemapping::None` and unlit materials, so they reach the display
encoder at exact paper-white-relative values (above `1.0` reaches peak).

`examples/3d/hdr_calibration.rs` is a harness around the plugin plus the `hdr`
helper's `HdrPlugin`. `T` cycles the requested transfer through sRGB,
scRGB-linear, extended-range sRGB, and PQ, renegotiating the swapchain on the
fly. `G` toggles a Gran Turismo 7 tone-mapping preview over the same patterns.
Backtick shows an engine-telemetry overlay. Calibrating needs an HDR display:
macOS/iOS Metal, Windows Vulkan, or Wayland Vulkan with Mesa 25.1+. On SDR
systems the example still runs, on the warn-and-degrade path.
