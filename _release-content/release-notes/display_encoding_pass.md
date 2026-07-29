---
title: "Display-encoding pass: gamut transform, out-of-gamut compression, and HDR transfer functions"
authors: ["@stuartparmenter"]
pull_requests: []
---

When a camera presents to an HDR display, the tone-mapped image has to be
converted into the exact signal the display expects — the right primaries and
the right transfer function — or highlights and saturated colors come out wrong.
Bevy now does this in a dedicated display-encoding pass, with no per-camera
setup: request an HDR transfer on the window's `DisplayTarget` and give the
camera an HDR-aware tone-mapping operator (see the "HDR display output
(scRGB-linear and HDR10/PQ)" release note), and everything below happens
automatically for that view.

The pass runs after the UI pass (UI composites in display-linear,
paper-white-relative space, so a white UI panel lands at
`DisplayTarget::paper_white_nits`) and before the final upscaling blit. Reading
the view's resolved `DisplayTarget`, it performs, in order:

- a full-precision gamut transform from the tone-map operator's output
  primaries to the display signal's primaries (for example Rec.2020 → Rec.709
  when `Tonemapping::GranTurismo7` drives an scRGB signal, or Rec.709 →
  Display-P3 for an `ExtendedDisplayP3` signal);
- out-of-gamut compression (below), so colors a contraction pushes past the
  display gamut compress gracefully instead of clipping;
- the display transfer function — scRGB-linear, PQ (SMPTE ST 2084), or the
  encoded extended-range sRGB OETF — selected by the target's `DisplayTransfer`.

## Out-of-gamut compression

A gamut contraction can land the most saturated colors outside what the display
can show. A per-channel clip collapses their saturation unevenly and shifts hue
(the classic `(1500, 1200, 500) → (1000, 1000, 500)` problem — a vivid orange
reads as a duller one). Instead the pass compresses out-of-gamut colors smoothly
toward the achromatic axis, in the style of the ACES 1.3 Reference Gamut
Compression (Academy S-2020-001): in-gamut colors pass through unchanged, the
most saturated are eased back to the boundary along a smooth curve (no banding,
distinct bright colors stay distinct), and brightness and hue are preserved as
closely as the closed-form mapping allows.

The `DisplayGamutCompression` resource controls it:

- `Auto` (default) — compress only when the gamut stage can actually go out of
  gamut (a contraction), e.g. a `Tonemapping::GranTurismo7` camera fitting its
  wide Rec.2020 output into a Rec.709-coordinate scRGB / extended-sRGB or
  Display-P3 signal. Identity and expansion paths keep a no-op plain clip.
- `Always` — force it on for every HDR view; this also desaturates highly
  saturated in-gamut colors on paths that can't go out of gamut, so use it only
  to exercise the path.
- `Clip` — the hue-shifting per-channel clip, kept as a debug fallback for A/B
  comparison.

The transfer functions the pass encodes with live in a new importable WGSL
library, `bevy_render::transfer_functions` — the odd-symmetric extended-range
sRGB OETF, scRGB scaling, and the PQ inverse EOTF — with CPU mirrors and
parity tests in the matching Rust module, so you can reuse them in your own
shaders. The decode direction (the extended sRGB and PQ EOTFs) is CPU-only in
the Rust module, where the screenshot readback path uses it to decode HDR
captures.

The pass only activates for display targets whose resolved transfer is HDR
(`ScRgbLinear`, `Pq`, or `ExtendedSrgb`) — exactly when surface selection has
configured a matching HDR swapchain (see the scRGB HDR output release note,
including its note on the currently required wgpu patch). Views on the default
`DisplayTarget::SDR_SRGB` (or any sRGB-transfer target) never run it: it records
no GPU work, and the exact sRGB encode remains the free hardware conversion on
swapchain writeback, byte-identical to 0.19. A plain single-camera SDR sRGB
window with an active operator likewise keeps folding tone mapping into its
material shaders (the in-shader path) on an 8-bit main texture. A camera gets
the scene-linear `Rgba16Float` intermediate and the node-side tonemapping pass
only when it is `Hdr`, renders to an HDR-transfer target, or runs an operator
the in-shader fold cannot reproduce; see the migration guides for the visual
implications on those previously-SDR cameras.
