---
title: "HDR display output (scRGB-linear and HDR10/PQ)"
authors: ["@stuartparmenter"]
pull_requests: []
---

Bevy can now present real high-dynamic-range output. Set the window's
`DisplayTarget` to request an HDR transfer, and give the camera an HDR-aware
tone-mapping operator:

```rust
fn enable_hdr_output(
    mut commands: Commands,
    mut window: Single<&mut DisplayTarget, With<PrimaryWindow>>,
    camera: Single<Entity, With<Camera>>,
) {
    **window = DisplayTarget {
        paper_white_nits: 200.0,
        peak_luminance_nits: 1000.0,
        // Or `DisplayTransfer::Pq` for HDR10 output.
        transfer: DisplayTransfer::ScRgbLinear,
        ..DisplayTarget::SDR_SRGB
    };
    // On HDR targets GT7 runs in its HDR mode automatically, driven by the
    // display target's peak luminance. Add a `GranTurismo7Params` component
    // to customize the operator's artistic dials.
    commands.entity(*camera).insert(Tonemapping::GranTurismo7);
}
```

HDR output is presented as paper-white-relative display-linear light: `1.0`
maps to `paper_white_nits` and values above it reach into the display's HDR
headroom up to `peak_luminance_nits`. Surface negotiation selects a real
(format, color space) pair through wgpu's surface color-space API:

- **`DisplayTransfer::ScRgbLinear`** configures an `Rgba16Float` swapchain in
  the extended-sRGB-**linear** color space (1.0 = 80 nits). Available on
  macOS/iOS (Metal EDR), Windows (Vulkan/DX12), and Wayland (Vulkan, Mesa
  25.1+ color management) — **native-only** (browser WebGPU cannot present a
  linear-transfer canvas; use `ExtendedSrgb` there).
- **`DisplayTransfer::ExtendedSrgb`** is the *encoded* (gamma) sibling of
  scRGB-linear: same 1.0 = 80 nits normalization, but the signal is run
  through the odd-symmetric extended sRGB OETF instead of staying linear. It is
  the **web HDR path** and is also available on Metal and Vulkan. Its surface
  color space follows `DisplayTarget::gamut`: `DisplayGamut::Rec709` selects
  the `ExtendedSrgb` color space, and `DisplayGamut::DisplayP3` selects
  `ExtendedDisplayP3`, where the encoder converts tone-mapped output into P3
  primaries before encoding (Metal and browser WebGPU). This is the first
  transfer for which `DisplayGamut::DisplayP3` is a real, non-coerced encoder
  target.
- **`DisplayTransfer::Pq`** configures an HDR10 swapchain — `Rgb10a2Unorm`
  preferred (PQ's native 10-bit container), `Rgba16Float` where that is what
  the backend advertises — carrying the PQ (SMPTE ST 2084) signal in Rec.2020
  primaries. Available on Vulkan, DX12, and Metal when the OS has HDR output
  enabled. With GT7 this is a fully native path: GT7's HDR mode emits linear
  Rec.2020 directly, the display-encoding pass applies the PQ OETF with no
  intermediate gamut round-trip, and the encoded signal is presented as-is.

Press `O` in the `tonemapping` example (cycles sRGB → scRGB →
extended-sRGB 709/P3 → PQ) or `T` in the `hdr_calibration` example to try it
on an HDR-capable display.

## The display-encoding pass

The tone-mapped image has to be converted into the exact signal the display
expects — the right primaries and the right transfer function — or highlights
and saturated colors come out wrong. A dedicated display-encoding pass does
this per view, with no per-camera setup. It runs after the UI pass (UI
composites in display-linear, paper-white-relative space, so a white UI panel
lands at `DisplayTarget::paper_white_nits`) and before the final upscaling
blit. Reading the view's resolved `DisplayTarget`, it performs, in order:

- a full-precision gamut transform from the tone-map operator's output
  primaries to the display signal's primaries (for example Rec.2020 → Rec.709
  when `Tonemapping::GranTurismo7` drives an scRGB signal, or Rec.709 →
  Display-P3 for an `ExtendedDisplayP3` signal);
- out-of-gamut compression (below), so colors a contraction pushes past the
  display gamut compress gracefully instead of clipping;
- the display transfer function selected by the target's `DisplayTransfer` —
  scRGB scaling (`paper_white_nits / 80`), PQ (SMPTE ST 2084, from absolute
  nits), or the encoded extended-range sRGB OETF.

The final upscaling blit hands the encoded signal to the surface unchanged —
these formats have no hardware sRGB encode. With GT7 running in its HDR mode,
highlights above paper white finally make it to the panel.

A gamut contraction can land the most saturated colors outside what the
display can show, and a per-channel clip collapses their saturation unevenly
and shifts hue (the classic `(1500, 1200, 500) → (1000, 1000, 500)` problem —
a vivid orange reads as a duller one). Instead the pass compresses
out-of-gamut colors smoothly toward the achromatic axis, in the style of the
ACES 1.3 Reference Gamut Compression (Academy S-2020-001): in-gamut colors
pass through unchanged, the most saturated are eased back to the boundary
along a smooth curve, and brightness and hue are preserved as closely as the
closed-form mapping allows. The `DisplayGamutCompression` resource controls
it: `Auto` (default) compresses only when the gamut stage can actually go out
of gamut (a contraction); `Always` forces it on for every HDR view (this also
desaturates highly saturated in-gamut colors, so use it only to exercise the
path); `Clip` keeps the hue-shifting per-channel clip as a debug fallback for
A/B comparison.

The transfer functions the pass encodes with live in an importable WGSL
library, `bevy_render::transfer_functions` — the odd-symmetric extended-range
sRGB OETF, scRGB scaling, and the PQ inverse EOTF — with CPU mirrors and
parity tests in the matching Rust module, so you can reuse them in your own
shaders. The decode direction (the extended sRGB and PQ EOTFs) is CPU-only in
the Rust module, where the screenshot readback path uses it to decode HDR
captures.

## SDR-only operators degrade gracefully

SDR-only tone-mapping operators — everything except `GranTurismo7`, `Linear`, and `None`,
including the `Camera3d` default `TonyMcMapface` — cap their output at paper
white, leaving an HDR display's headroom unused. A camera using one on an
HDR-transfer target therefore degrades gracefully instead of silently
rendering an SDR-capped image: the view runs `Tonemapping::GranTurismo7`
instead (with the camera's `GranTurismo7Params` if present, otherwise the
defaults) and Bevy warns once. The `Tonemapping` component itself is never
modified; set `Tonemapping::GranTurismo7` explicitly to adopt the substitute
and silence the warning, or switch back to an SDR display target to keep the
authored operator. `Tonemapping::None` is not substituted — it is a deliberate
pass-through — but also warns on HDR targets.

## Unfulfillable requests downgrade with a warning

When a request cannot be fulfilled it is **downgraded with a warning at each
step**: PQ falls back to scRGB-linear where available, and any HDR request
falls back to plain SDR sRGB — byte-identical to a default window — when the
surface offers nothing better (SDR displays, OS HDR disabled, X11, GLES). A
cross-HDR downgrade (PQ → scRGB) keeps your calibration values and swaps only
the transfer. The outcome is visible in the render world: `ViewDisplayTarget`
carries the post-negotiation display target, and every consumer (the encoding
pass, the upscaling blit, GT7's HDR mode, the display-target uniform) keys on
it, so an unfulfilled HDR request can never mis-encode the image. Bevy also renegotiates defensively if the
capabilities change at runtime (e.g. the OS HDR toggle is flipped) rather than
failing surface validation.

Changing `DisplayTarget::transfer` at runtime reconfigures the surface with
fresh (format, color space) negotiation (and invalidates the window's view
targets), so HDR output can be toggled from a settings menu. A
`DisplayTarget::gamut` change is treated the same way when the transfer is
`ExtendedSrgb` (it selects `ExtendedSrgb` vs `ExtendedDisplayP3`); paper white,
peak, and every other gamut change take effect through per-view uniforms and
pipeline respecialization without any surface work.

Views on the default `DisplayTarget::SDR_SRGB` (or any sRGB-transfer target)
never run the encoding pass: it records no GPU work, and the exact sRGB encode
remains the free hardware conversion on swapchain writeback, byte-identical to
0.19. A plain single-camera SDR sRGB window with an active operator likewise
keeps folding tone mapping into its material shaders (the in-shader path) on
an 8-bit main texture. A camera gets the scene-linear `Rgba16Float`
intermediate and the node-side tonemapping pass only when it is `Hdr`, renders
to an HDR-transfer target, or runs an operator the in-shader fold cannot
reproduce; see the migration guides for the visual implications on those
previously-SDR cameras.

Screenshots understand the new surfaces too: scRGB (`Rgba16Float`) captures
read back as display-linear floats, HDR10 captures are decoded from the PQ
signal through the PQ EOTF, and encoded extended-range sRGB captures are
decoded through the extended sRGB EOTF (converting `ExtendedDisplayP3` back to
Rec.709) — all to the same display-linear Rec.709 scale (1.0 = 80 nits).
`save_to_disk` writes float images losslessly to float-capable containers
(OpenEXR `.exr` with Bevy's `exr` feature, Radiance `.hdr`); saving to an
8-bit format clamps, sRGB-encodes, and warns.
