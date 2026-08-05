---
title: "HDR display output (scRGB-linear and HDR10/PQ)"
authors: ["@stuartparmenter"]
pull_requests: []
---

Bevy can now present high-dynamic-range output. Set the window's
`DisplayTarget` to an HDR transfer and give the camera an HDR-aware
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
    // On HDR targets GT7 runs in its HDR mode automatically. Add a
    // `GranTurismo7Params` component to change the operator's defaults.
    commands.entity(*camera).insert(Tonemapping::GranTurismo7);
}
```

Output is display-linear: `1.0` is `paper_white_nits`, and values above it
reach the display's headroom up to `peak_luminance_nits`. Surface negotiation
picks a (format, color space) pair:

- `DisplayTransfer::ScRgbLinear` configures an `Rgba16Float` swapchain in the
  `ExtendedSrgbLinear` color space (1.0 = 80 nits). Native only: macOS/iOS
  (Metal EDR), Windows (Vulkan/DX12), Wayland (Vulkan, Mesa 25.1+ color
  management). Browser WebGPU has no linear-transfer canvas, so use
  `ExtendedSrgb` there.
- `DisplayTransfer::ExtendedSrgb` is the encoded (gamma) form of scRGB-linear:
  the same 1.0 = 80 nits normalization, run through the odd-symmetric extended
  sRGB OETF. This is the web HDR path, and works on Metal and Vulkan too.
  `DisplayTarget::gamut` picks the color space. `Rec709` gives `ExtendedSrgb`,
  and `DisplayP3` gives `ExtendedDisplayP3` (Metal and browser WebGPU). This is
  the only transfer where `DisplayGamut::DisplayP3` selects a P3 encoder.
- `DisplayTransfer::Pq` configures an HDR10 swapchain: the PQ (SMPTE ST 2084)
  signal in Rec.2020 primaries, preferring `Rgb10a2Unorm`, PQ's native 10-bit
  container, and taking `Rgba16Float` where the backend advertises that
  instead. Vulkan, DX12, and Metal, when the OS has HDR output enabled.

Press `O` in the `tonemapping` example to cycle sRGB, scRGB, extended-range
sRGB 709/P3, and PQ.

## The display-encoding pass

A per-view pass turns the tone-mapped image into the signal the display
expects, with no per-camera setup. It converts the operator's output primaries
to the display signal's primaries, such as Rec.2020 to Rec.709 for an scRGB
signal. Then it applies the transfer function `DisplayTransfer` selects: scRGB
scaling (`paper_white_nits / 80`), PQ from absolute nits, or the extended sRGB
OETF.

A gamut contraction can push saturated colors past what the display can show.
The pass compresses those colors toward the achromatic axis, in the style of
the ACES 1.3 Reference Gamut Compression (Academy S-2020-001), and leaves
in-gamut colors untouched. The `DisplayGamutCompression` resource controls it:
`Auto` (the default) compresses only when the gamut stage can go out of gamut
(a contraction), `Always` compresses on every HDR view and also desaturates
highly saturated in-gamut colors, and `Clip` keeps the per-channel clip as a
debug fallback.

The encode functions live in an importable WGSL library,
`bevy_render::transfer_functions`, mirrored in a matching Rust module. The
decode direction, the extended sRGB and PQ EOTFs, is CPU-only.

## Cameras without an HDR-capable operator

Every operator except `GranTurismo7`, `Linear`, and `None` caps its output at
paper white, including `TonyMcMapface`, the `Camera3d` default. On an
HDR-transfer target such a view runs `Tonemapping::GranTurismo7` instead, with
the camera's `GranTurismo7Params` if present, and Bevy warns once. The
`Tonemapping` component is never modified. Set `Tonemapping::GranTurismo7`
yourself to silence the warning, or go back to an SDR display target to keep
the authored operator. `Tonemapping::None` is a deliberate pass-through and is
never substituted, but it warns on HDR targets too.

## Negotiation and downgrade

A request that cannot be fulfilled is downgraded, with a warning at each step.
PQ falls back to scRGB-linear where available. Any HDR request falls back to
plain SDR sRGB, byte-identical to a default window, when the surface offers
nothing better: SDR displays, OS HDR disabled, X11, GLES. A cross-HDR
downgrade keeps your calibration values and swaps only the transfer. The
outcome shows up on the window's `WindowSurfaceTransfers` and on
`ViewDisplayTarget` in the render world. Read those rather than the request.
Bevy also renegotiates when capabilities change at runtime, such as the OS
HDR toggle being flipped.

Changing `DisplayTarget::transfer` at runtime reconfigures the surface with
fresh (format, color space) negotiation and invalidates the window's view
targets. A `DisplayTarget::gamut` change does the same under `ExtendedSrgb`.
Paper white, peak, and every other gamut change take effect without
reconfiguring the surface.

Views on any sRGB-transfer target, including the default
`DisplayTarget::SDR_SRGB`, never run the encoding pass. The sRGB encode is
unchanged from 0.19. Requesting an HDR transfer does move a camera off the
in-shader tone-mapping fast path and its 8-bit main texture onto a
scene-linear `Rgba16Float` intermediate. The "Tone mapping render path"
migration guide has the conditions and the visual implications.

## Screenshots

Every capture lands on the same display-linear Rec.709 scale (1.0 = 80 nits).
scRGB (`Rgba16Float`) reads back as display-linear floats. HDR10 is decoded
through the PQ EOTF, and encoded extended-range sRGB through the extended sRGB
EOTF, converting `ExtendedDisplayP3` back to Rec.709. `save_to_disk` writes
float images losslessly to float-capable containers (OpenEXR `.exr` with
Bevy's `exr` feature, Radiance `.hdr`). Saving to an 8-bit format clamps,
sRGB-encodes, and warns.
