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
swapchain format (see the "HDR display output (scRGB-linear)" release note).

In the render world, every camera view resolves its target's calibration into
a `ViewDisplayTarget` component, carrying both the `requested` calibration and
the `resolved` one after surface negotiation. The outcome is also mirrored back
to the main world as a read-only `WindowSurfaceTransfers` component on the
window: `resolved` is the transfer the surface actually presents, so apps can
detect a downgraded HDR request, and `supported` is the set of transfers this
surface could present, so an app can offer only the modes that will work
instead of requesting one that silently downgrades. Both lag negotiation by one
frame. A per-view `DisplayTargetUniform` (luminance values
plus gamut/transfer indices, importable in WGSL as
`bevy_render::display_target`) is prepared each frame and bound solely by the
display-encoding (gamut-mapping and transfer-encoding) pass. The GT7 operator's
HDR mode is driven separately, by a `Gt7ParamsUniform` baked from the target's
peak luminance at prepare time and bound in the tonemapping pass.

Render targets that aren't windows, such as `RenderTarget::Image` and
`RenderTarget::TextureView` (used by OpenXR), have no window entity to host
the component. For those, the new `ManualDisplayTargets` resource in
`bevy_render` maps a `NormalizedRenderTarget` to its `DisplayTarget`; targets
without an entry fall back to `DisplayTarget::SDR_SRGB`.

`DisplayTarget` is user-authoritative: Bevy never overwrites values you set.
To react to a window being dragged to a different display, the new
`WindowMonitorChanged` message is sent whenever the monitor a window is on
changes (including when it first becomes known):

```rust
fn react_to_monitor_change(
    mut events: MessageReader<WindowMonitorChanged>,
    monitors: Query<&Monitor>,
) {
    for event in events.read() {
        if let Some(monitor) = event.monitor.and_then(|m| monitors.get(m).ok()) {
            // Inspect the new monitor and decide whether to update the
            // window's `DisplayTarget`.
        }
    }
}
```
