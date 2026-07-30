---
title: "`DisplayTarget`: per-display calibration for windows"
authors: ["@stuartparmenter"]
pull_requests: []
---

Bevy now describes the display a window is presented on with the new
`DisplayTarget` component in `bevy_window`. It captures how bright "paper
white" is, the display's peak luminance, its black level, its color gamut
(`DisplayGamut`: Rec.709, Display P3, or Rec.2020), and the transfer function
the final signal is encoded with (`DisplayTransfer`: sRGB, scRGB-linear, PQ,
or the encoded extended-range `ExtendedSrgb` — the web HDR path, which pairs
with `DisplayGamut::DisplayP3` for wide-gamut HDR). It is a required component
of `Window`, so every window gets one. The default, `DisplayTarget::SDR_SRGB`
(paper white and peak of 100 nits, Rec.709, sRGB), reproduces Bevy's existing
SDR output. It is the foundation this release's HDR output is parameterized
by: peak-aware tone mapping (`Tonemapping::GranTurismo7`), wide-gamut Rec.2020
output through the display-encoding pass, and the `Rgba16Float` scRGB
swapchain format (see the "HDR display output (scRGB-linear and HDR10/PQ)"
release note).

In the render world, every camera view resolves its target's calibration into
a `ViewDisplayTarget` component carrying the post-negotiation calibration.
The outcome is also mirrored back
to the main world as a read-only `WindowSurfaceTransfers` component on the
window: `resolved` is the transfer the surface actually presents, so apps can
detect a downgraded HDR request, and `supported` is the set of transfers this
surface could present, so an app can offer only the modes that will work
instead of requesting one that silently downgrades. Both lag negotiation by one
frame. A per-view `DisplayTargetUniform` (the paper-white luminance,
importable in WGSL as
`bevy_render::display_target`) is prepared each frame for views whose resolved
transfer is HDR — SDR views carry no display-target uniform — and bound solely
by the display-encoding (gamut-mapping and transfer-encoding) pass. The GT7
operator's HDR mode is driven separately, by a `Gt7ParamsUniform` baked from
the target's peak luminance at prepare time and bound in the tonemapping pass.

Render targets that aren't windows, such as `RenderTarget::Image` and
`RenderTarget::TextureView` (used by OpenXR), have no window entity to host
the component. For those, the new `ManualDisplayTargets` resource in
`bevy_render` maps a `NormalizedRenderTarget` to its `DisplayTarget`; targets
without an entry fall back to `DisplayTarget::SDR_SRGB`.

`DisplayTarget` is user-authoritative: Bevy never overwrites values you set.
To react to a window being dragged to a different display, watch the
`OnMonitor` relationship on the window entity — `bevy_winit` retargets it
whenever the monitor a window is on changes (including when it first becomes
known):

```rust
fn react_to_monitor_change(
    windows: Query<(Entity, &OnMonitor), Changed<OnMonitor>>,
    monitors: Query<&Monitor>,
) {
    for (window, on_monitor) in &windows {
        if let Ok(monitor) = monitors.get(on_monitor.0) {
            // Inspect the new monitor and decide whether to update the
            // window's `DisplayTarget`.
        }
    }
}
```

`DisplayTarget` also gained builder-style helpers for deriving calibrated
targets from a base value:

```rust
let hdr = DisplayTarget::SDR_SRGB
    .with_paper_white(200.0)
    .with_peak(1000.0)
    .with_transfer(DisplayTransfer::ScRgbLinear);
```

## Calibrating the values: the `hdr_calibration` example

HDR output is only as good as the calibration values in the window's
`DisplayTarget`. wgpu now senses some of them — `DisplayHdrInfo` feeds peak,
black, full-frame and SDR-white nits, headroom, and a gamut bucket into
`WindowDisplayState` / `MonitorDisplayCapability` — but paper white is a viewing
preference no display can report, and sensed values are missing or coarse on many
platforms. So an in-app, HGIG-style calibration flow complements that sensing
rather than replacing it, and doubles as a reference for shipping an "HDR
settings" screen in your own game.

`examples/helpers/hdr_calibration.rs` packages the flow as a reusable
`HdrCalibrationPlugin<S>`: drop it into one of your own `States` and it runs a
guided three-step wizard, persists the result next to the executable, emits a
`CalibrationComplete` event on confirm, and prompts to recalibrate when the
window moves to another monitor (watching `OnMonitor`). A `CalibrationStrategy`
is chosen up front: manual HGIG (you tune every value) or trust-OS (peak, black,
and gamut auto-resolve from sensed data; you only set paper white).
`examples/3d/hdr_calibration.rs` is a thin harness around it — it adds the
`hdr` helper's `HdrPlugin` plus the calibration plugin and wires up
`T`/`G`/backtick controls.

The three steps:

- **Peak luminance** (`peak_luminance_nits`): a solid clipped near-peak surround,
  a true-black separating frame, and a center patch at the candidate peak
  (`PeakFraction(1.0)`); raise the value until the patch merges into the
  surround, then back off one tap.
- **Paper white** (`paper_white_nits`): a reference white card at exactly `1.0`
  next to a 203-nit ITU-R BT.2408 strip.
- **Black level** (`min_luminance_nits`): near-black steps at fixed absolute
  luminances.

All adjustments mutate the primary window's `DisplayTarget` live. The patterns
render with `Tonemapping::None` and unlit materials so they reach the display
encoder at exact paper-white-relative values (above `1.0` reaches peak). In the
harness, `T` cycles the requested transfer through sRGB → scRGB-linear →
extended-sRGB → PQ, renegotiating the swapchain on the fly; `G` toggles a Gran
Turismo 7 tone-mapping preview over the same patterns; and backtick shows an
engine-telemetry overlay.

An HDR display is required to calibrate anything real (macOS/iOS Metal, Windows
Vulkan, or Wayland Vulkan with Mesa 25.1+); on SDR systems the example still runs
on the documented warn-and-degrade path.
