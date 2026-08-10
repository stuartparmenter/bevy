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
    // On HDR display targets GT7 runs in its HDR mode automatically. Add a
    // `GranTurismo7Params` component to change the operator's defaults.
    commands.entity(*camera).insert(Tonemapping::GranTurismo7);
}
```

Output is display-linear: `1.0` is `paper_white_nits`, and highlights run up to
`peak_luminance_nits`. `DisplayTarget::transfer` picks the swapchain:

- `DisplayTransfer::ScRgbLinear` gives an `Rgba16Float` swapchain in the
  `ExtendedSrgbLinear` color space (1.0 = 80 nits). Native only: macOS/iOS
  (Metal), Windows (Vulkan/DX12), Wayland (Vulkan, Mesa 25.1+).
- `DisplayTransfer::ExtendedSrgb` is the encoded (gamma) form of scRGB-linear,
  and the only HDR path browser WebGPU can present. Metal and Vulkan advertise
  it too. It is the one transfer that reads `DisplayTarget::gamut`:
  `DisplayGamut::Rec709` gives `ExtendedSrgb`, `DisplayGamut::DisplayP3` gives
  `ExtendedDisplayP3` (Metal and browser WebGPU).
- `DisplayTransfer::Pq` gives an HDR10 swapchain: the PQ (SMPTE ST 2084) signal
  in Rec.2020 primaries, preferring `Rgb10a2Unorm` and taking `Rgba16Float`
  where the backend offers that instead. Vulkan, DX12, and Metal, when the OS
  has HDR output enabled.

Press `O` in the `tonemapping` example to cycle sRGB, scRGB, extended-range
sRGB 709/P3, and PQ.

A request the surface cannot fulfill is downgraded, with a warning. PQ falls
back to scRGB-linear, and any HDR request falls back to plain SDR sRGB,
identical to a default window, when the surface offers nothing better: SDR
displays, OS HDR disabled, X11, GLES. A cross-HDR downgrade keeps your
calibration and swaps only the transfer. Read the outcome from the window's
`WindowSurfaceTransfers` component, not from the request; see the release note
"`DisplayTarget`: per-display calibration for windows".

Changing `DisplayTarget::transfer` at runtime reconfigures the surface and
renegotiates, as does a `DisplayTarget::gamut` change under `ExtendedSrgb`.
Paper white, peak, and other gamut changes take effect without reconfiguring.

## The display-encoding pass

A per-view pass turns the tone-mapped image into the signal the display
expects, with no per-camera setup: it converts the operator's output primaries
to the display's, then applies the transfer function. The encode functions are
importable in WGSL as `bevy_render::transfer_functions`, mirrored in a Rust
module; the decodes (the extended sRGB and PQ EOTFs) are CPU-only.

Narrowing the gamut, Rec.2020 to Rec.709 for an scRGB signal, can push
saturated colors past what the display can show. The pass compresses those
toward the achromatic axis, in the style of the ACES 1.3 Reference Gamut
Compression (Academy S-2020-001), and leaves in-gamut colors untouched. The
`DisplayGamutCompression` resource controls it: `Auto` (the default) compresses
only when the gamut narrows, `Always` compresses on every HDR-target view and
also desaturates saturated in-gamut colors, and `Clip` restores the per-channel
clip.

## Cameras without an HDR-capable operator

Every operator except `GranTurismo7`, `Linear`, and `None` caps its output at
paper white, including `TonyMcMapface`, the `Camera3d` default. On an
HDR display target such a view runs `Tonemapping::GranTurismo7` instead, with
the camera's `GranTurismo7Params` if present, and Bevy warns once. Your
`Tonemapping` component is never modified: set `Tonemapping::GranTurismo7`
yourself to silence the warning, or use an SDR display target to keep the
authored operator. `Tonemapping::None` is never substituted, but also warns on
HDR display targets.

Views on an SDR display target, including the default
`DisplayTarget::SDR_SRGB`, never run the encoding pass, and their sRGB encode
is unchanged from 0.19. An HDR transfer does move a camera off the in-shader
tone-mapping fast path and its 8-bit main texture onto a scene-linear
`Rgba16Float` intermediate. See the migration guide "Tone mapping render path:
node-based tone mapping and the `Rgba16Float` intermediate".

## Screenshots

Every capture lands on the same display-linear Rec.709 scale (1.0 = 80 nits),
whatever transfer the surface used. `save_to_disk` writes float containers:
OpenEXR `.exr` (Bevy's `exr` feature) losslessly, Radiance `.hdr` with
negatives clipped. Saving to an 8-bit format clamps, sRGB-encodes, and warns.
