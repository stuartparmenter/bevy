---
title: "The composition contract: one resolution for camera compositing"
authors: ["@stuartparmenter"]
pull_requests: []
---

Cameras that share a main texture and form a stack now resolve their
`CompositingSpace` as one group. Give every stack member the same
`CompositingSpace`, or none: if they conflict, or if any member is not a 2D
camera, the group resolves to linear with a warning. A solo camera, or one that
clears or renders to a viewport, keeps its own request.

A `Tonemapping::None` overlay can make one camera finalize tone mapping and a
different camera finalize encoding. Bevy will not encode a buffer before it is
tone-mapped, so when the stack shape would force that, tone-map deferral is
cancelled for the whole stack and Bevy warns.

## What this fixes

- Encoder source gamut. A `Tonemapping::GranTurismo7` 3D camera under a
  `Tonemapping::None` 2D overlay on a PQ target used to double-expand Rec.709
  to Rec.2020 over a buffer already in Rec.2020, oversaturating the image.

- Uncovered regions on HDR targets. Regions no camera covers used to show a raw
  linear value that read as full-peak nits. They now show the clear color,
  encoded for the resolved transfer, gamut, and paper white.

- Deterministic mixed-HDR stacks. A default-tone-mapped 3D camera and an `Hdr`
  2D overlay on one target used to overwrite each other in a nondeterministic
  order. The overlay now composites over the base.

- FXAA luma and UI encoding. FXAA's edge luma took the square root of a Rec.601
  dot that goes negative on an Oklab buffer's signed chroma, producing NaN. UI
  wrote linear values into the camera's encoded buffer, so its colors were wrong
  on `Srgb` and `Oklab` views.

- UI and gizmo wide-gamut colors. UI and gizmos authored their colors in Rec.709
  and wrote them unconverted into the post-tone-map buffer. On a GT7 HDR view
  that buffer holds Rec.2020 primaries, so saturated colors oversaturated.

Default SDR projects render byte-for-byte identically. The changes above only
affect camera stacks, non-default compositing spaces, and HDR output. See the
"Camera compositing resolves once per frame" migration guide for adapting custom
render code.
