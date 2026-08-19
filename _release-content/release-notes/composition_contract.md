---
title: "The composition contract: one resolution for camera compositing"
authors: ["@stuartparmenter"]
pull_requests: []
---

Cameras that share a main texture and form a stack now resolve their
`CompositingSpace` as one group. A group is a stack when every camera after the
first has `ClearColorConfig::None` and no viewport. Give every stack member the
same `CompositingSpace`, or none: if they conflict, or if any member is not a 2D
camera, the group resolves to linear with a warning. Cameras that do not form a
stack keep their own requests.

Tone mapping and display encoding resolve per stack too. A buffer is never
encoded before it is tone-mapped, so when the stack shape would force that,
tone-map deferral is cancelled for the whole stack, with a warning.

## What this fixes

- The display encoder now takes its source gamut from the camera that
  tone-mapped the stack. A `Tonemapping::None` overlay over a
  `Tonemapping::GranTurismo7` camera used to expand Rec.709 to Rec.2020 twice on
  a PQ target, oversaturating the image.

- Regions of an HDR display target that no camera covers used to show a raw
  linear value that read as full-peak nits. They now show the clear color,
  encoded for the resolved transfer, gamut, and paper white.

- A default-tone-mapped 3D camera and an `Hdr` 2D overlay on one target used to
  blit over each other with a replacing blend. The overlay now composites over
  the base.

- FXAA's edge luma called `sqrt` on a Rec.601 dot that goes negative on a
  `CompositingSpace::Oklab` buffer's signed chroma, producing NaN. UI wrote
  linear Rec.709 values into the camera's buffer, so its colors were wrong on
  `Srgb` and `Oklab` views, and oversaturated on an HDR-target GT7 view whose
  buffer holds Rec.2020 primaries.

Default SDR projects render byte-for-byte identically. For adapting custom
render code, see the migration guide "`ViewTarget::compositing_space` removed:
camera compositing resolves once per frame".
