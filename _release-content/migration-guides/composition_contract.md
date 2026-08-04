---
title: "Camera compositing resolves once per frame: `ViewTarget::compositing_space` removed, stacks share one space"
pull_requests: []
---

Bevy resolves a camera's `CompositingSpace` request once per frame. Two new
render-world components hold the result:

- `ResolvedCompositingSpace`: the view's `Option<CompositingSpace>`. Written by
  `resolve_composition_spaces` in `RenderSystems::CreateViews`.
- `ViewStackContract`: the view's compositing space, encoder source gamut,
  tone-map and display-encode roles, and blit disposition. Written by
  `resolve_camera_stack_contracts` in `RenderSystems::PrepareViews`.

A solo camera resolves to its own request, so its pipeline keys and pixels are
unchanged.

## Removed API

`ViewTarget::compositing_space` is removed. Read `ResolvedCompositingSpace` or
`ViewStackContract::compositing_space` instead:

```rust
// 0.19
let space = view_target.compositing_space;

// 0.20, in bevy_render (`ResolvedCompositingSpace` component on the view entity)
let space = resolved_space.0;

// 0.20, in bevy_core_pipeline (per-view contract)
let space = contract.compositing_space;
```

`ViewStackContract` is overwritten in place and never removed, so keep a
`ViewTarget` term in your query to skip views whose target was dropped.

`ExtractedCamera::compositing_space` still holds the raw per-camera request.
Don't use it for pass or pipeline-key decisions.

`encoder_input_gamut` is removed from `bevy_core_pipeline::display_encoding`. The
encoder's source gamut now comes from `ViewStackContract::source_gamut`.

`bevy_core_pipeline::tonemapping::resolve_tonemapping` replaces
`effective_tonemapping` and `tonemap_output_gamut`. It returns the substituted
operator and its output gamut together (`ResolvedTonemapping::output_gamut`).

`SortedCamera::hdr` is removed: cameras sort by `(order, target)`, and their
per-target stacking index keys on the target alone. Read `ExtractedCamera::hdr`
on the view entity if you need a camera's raw HDR request.

## Renamed writer-encode shader defs

The writer-side encode defs that 2D pipelines (sprites, 2D meshes and materials,
gizmos, UI) push into their shaders are renamed:

| 0.19 | 0.20 |
| ---- | ---- |
| `SRGB_OUTPUT` | `COMPOSITING_SPACE_SRGB` |
| `OKLAB_OUTPUT` | `COMPOSITING_SPACE_OKLAB` |

The gamut-convert def in writer shaders is now `OUTPUT_GAMUT_REC2020`: the
buffer being written has Rec.2020 primaries. UI used to push
`WORKING_COLOR_SPACE_REC2020` for that. That name now keeps only its
project-global meaning.

A custom `Material2d` or UI shader that read the old defs should switch to the
new names, or call `bevy_render::writer_encode::writer_encode` on its composed
color, which handles all three defs.

## Behavior changes

These apply to camera stacks, non-default compositing spaces, and HDR output,
never to default SDR cameras. The release note "The composition contract: one
resolution for camera compositing" covers the bugs they fix.

- Cameras that share a main texture and form a stack (each later camera
  composites full-screen over the previous output) resolve to one compositing
  space. Conflicting requests, or any non-2D-camera member, resolve to linear
  with a warning naming the misconfiguration. Give every stack member the same
  `CompositingSpace`, or none.
- A 3D camera and an `Hdr` 2D overlay on the same target now order
  deterministically, and the overlay composites over the base instead of
  overwriting it.
- Every stack member ordered below a finalizing camera skips its upscaling blit,
  and the finalizer presents once. A finalizer with `CameraOutputMode::Skip`
  cancels the skipping for its group.
- On HDR display targets, regions no camera covers no longer show a raw linear
  value that reads as full-peak nits.
- The display encoder takes its source gamut from the camera that tone-mapped
  the buffer, not from a `Tonemapping::None` overlay composited on top.
- UI, 2D gizmos, and the tilemap chunk material writer-encode each fragment's
  final color into the view's resolved compositing space, instead of writing
  linear values into a `CompositingSpace::Srgb`/`Oklab` buffer. UI over 3D
  cameras is unaffected (non-2D cameras resolve to linear), as are 3D gizmos,
  which render pre-tone-map.
- On a Rec.2020 (GT7) HDR view, UI fragment shaders and the 2D and 3D gizmo
  line/joint shaders convert their Rec.709-authored colors with
  `rec709_to_rec2020` (the shared `OUTPUT_GAMUT_REC2020` writer-encode) before
  the compositing-space encode.
- FXAA on a `CompositingSpace::Oklab` view reads the Oklab L channel for edge
  luma. Other compositing spaces are unchanged.
