---
title: "The composition contract: one resolution for camera compositing"
authors: ["@stuartparmenter"]
pull_requests: []
---

A camera's `CompositingSpace` is a *request* — the space you want the camera's
buffer composited in — much like the transfer function on its `DisplayTarget`.
Previously several end-of-frame stages (tone mapping, the display encoder, and
the upscaling blit) each independently re-derived a stack's shape and the
buffer's color space, and the drift between them caused bugs on stacked and HDR
cameras.

Bevy now resolves all of it once per frame, in two phases, and every consumer
reads the result instead of re-deriving it:

- `ResolvedCompositingSpace`, a per-view component, holds the first phase's
  result: cameras that share a main-texture ping-pong resolve as one group to a
  single compositing space. A solo camera, or a
  non-stack shape (cameras that clear or render to a viewport), keeps its own
  request verbatim. A compositing stack collapses to one space: the space its
  members request, or linear (with a warning) if they conflict or if any member
  is not a 2D camera.
- `ViewStackContract` is the per-view result: each view's role for the tone-map
  pass and, separately, the display-encode pass (solo, deferred to a finalizer,
  or the finalizer); its upscaling-blit disposition; its resolved compositing
  space; the color primaries the encoder should treat the buffer as holding; and
  the resolved encode parameters.

The two pass roles stay distinct — a `Tonemapping::None` overlay can make one
camera finalize tone mapping and a different camera finalize encoding — but are
tied by one coherence rule: encoding can never be deferred past a camera that
tone mapping defers over, because that would encode a buffer before it is
tone-mapped. When the stack shape would force that, tone-map deferral is
cancelled for the whole stack and Bevy warns.

## What this fixes

- **Encoder input keying.** The display encoder keyed its source gamut and source
  space off the finalizer camera's *own* operator and request; it now reads them
  from the stack contract — the gamut of whatever camera actually tone-mapped the
  buffer, and the resolved compositing space. Example: a
  `Tonemapping::GranTurismo7` 3D camera composited under a `Tonemapping::None` 2D
  overlay on a PQ display target was double-expanded Rec.709 → Rec.2020 over a
  buffer already in Rec.2020, oversaturating the image; it now encodes correctly.

- **Deferral coherence.** Encode-before-tone-map was reachable when a
  `Tonemapping::None` member broke the encode shape but not the tone-map shape;
  the conflict is now detected and tone-map deferral is cancelled with a warning.

- **Deferred-member blits and out-texture ownership.** Every member below the
  finalizer now skips its upscaling blit, so the finalizer is the only one
  presenting and does so with replace, owning the output texture. Previously each
  deferred member blit its un-finalized buffer and the finalizer alpha-blended its
  encoded result over those pixels. The finalizer's clear color is CPU-encoded
  through the resolved transfer, gamut, and paper white, so on HDR targets the
  regions no camera covers present a correct clear instead of a raw linear value
  that read as full-peak nits.

- **Deterministic mixed-HDR stacks.** Sorted-camera indices now count per render
  target rather than per `(target, hdr)`, so a default-tone-mapped 3D camera and
  an `Hdr` 2D overlay on one target order deterministically and the overlay
  composites over the base, instead of both landing at index 0 with
  replace-over-replace blits.

- **FXAA luma and UI encoding.** FXAA's edge luma used a Rec.601 dot that can go
  NaN on an Oklab buffer's signed chroma; it now reads the Oklab L channel on
  resolved-Oklab views. UI ran after tone mapping and wrote linear values into the
  camera's encoded buffer; each UI fragment shader now writer-encodes its final
  color into the resolved compositing space, the same way the sprite shader does.

- **UI and gizmo wide-gamut colors.** UI and gizmos authored their colors in
  Rec.709 and wrote them unconverted into a post-tone-map buffer that, on a GT7 HDR
  view, holds Rec.2020 primaries — so saturated colors oversaturated. They now
  convert to the buffer's working gamut (`rec709_to_rec2020`) the same way sprites
  do, and 2D gizmos additionally writer-encode into the camera's compositing space.

Default SDR projects render byte-for-byte identically: a solo camera resolves to
its own request, so every pipeline key and shader-def vector hashes exactly as
before. The changes above are confined to camera stacks, non-default compositing
spaces, and HDR output. See the migration guide for adapting custom render code.
