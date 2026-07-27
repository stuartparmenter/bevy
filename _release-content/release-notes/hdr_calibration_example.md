---
title: "HDR calibration example"
authors: ["@stuartparmenter"]
pull_requests: []
---

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
window moves to another monitor (`WindowMonitorChanged`). A `CalibrationStrategy`
is chosen up front: manual HGIG (you tune every value) or trust-OS (peak, black,
and gamut auto-resolve from sensed data; you only set paper white).
`examples/3d/hdr_calibration.rs` is a thin harness around it — it adds the
`hdr_helper` `HdrPlugin` plus the plugin and wires up `T`/`G`/backtick controls.

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

`DisplayTarget` gained matching builder-style helpers for deriving calibrated
targets from a base value:

```rust
let hdr = DisplayTarget::SDR_SRGB
    .with_paper_white(200.0)
    .with_peak(1000.0)
    .with_transfer(DisplayTransfer::ScRgbLinear);
```

An HDR display is required to calibrate anything real (macOS/iOS Metal, Windows
Vulkan, or Wayland Vulkan with Mesa 25.1+); on SDR systems the example still runs
on the documented warn-and-degrade path.
