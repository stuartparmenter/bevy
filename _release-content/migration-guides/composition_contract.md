---
title: "Camera compositing resolves once per frame: `ViewTarget::compositing_space` removed, stacks share one space"
pull_requests: []
---

Bevy now resolves a camera's `CompositingSpace` request once per frame — after a
render target's cameras are grouped — into shared state that every render stage
reads instead of re-deriving:

- `ResolvedCompositionSpaces` (render-world resource, built in
  `RenderSystems::CreateViews` after `sort_cameras`): one
  `Option<CompositingSpace>` per camera view, after grouping cameras that share a
  main-texture ping-pong.
- `ViewStackContract` (render-world component, built in
  `RenderSystems::PrepareViews` after `prepare_view_targets` and
  `prepare_view_display_targets`): each view's tone-map and display-encode roles,
  blit disposition, resolved compositing space, the encoder's source gamut, and
  resolved encode parameters.

A solo camera resolves to its own request, so its pipeline keys and pixels are
unchanged. The changes below affect custom render code that read the old per-view
state, and camera stacks using a non-default compositing space or HDR output.

## Removed API

`ViewTarget::compositing_space` is removed. Read the resolved space from
`ResolvedCompositionSpaces` (keyed by the render-world view entity) in
`bevy_render`, or from `ViewStackContract::compositing_space` in
`bevy_core_pipeline`. The raw per-camera request remains as
`ExtractedCamera::compositing_space`, but it feeds only the extract-time
main-texture format choice — don't consult it for pass or pipeline-key decisions.

```rust
// 0.19
let space = view_target.compositing_space;

// 0.20, in bevy_render (needs `Res<ResolvedCompositionSpaces>` and the view entity)
let space = resolved_spaces.get(entity, camera.compositing_space);

// 0.20, in bevy_core_pipeline (per-view contract)
let space = contract.compositing_space;
```

`encoder_input_gamut` is removed from `bevy_core_pipeline::display_encoding`. The
encoder's source gamut now comes from `ViewStackContract::source_gamut`.
`tonemap_output_gamut` remains the single source of truth for an operator's output
gamut.

`SortedCamera::hdr` is removed: cameras sort by `(order, target)`, and their
per-target stacking index keys on the target alone. Read `ExtractedCamera::hdr`
on the view entity if you need a camera's raw HDR request.

## Behavior changes

These affect non-default compositing-space and HDR configurations, never default
SDR cameras.

- **Camera stacks resolve one compositing space per shared buffer.** Cameras that
  share a main texture and form a stack (each later camera composites full-screen
  over the previous output) resolve to a single compositing space. Conflicting
  requests, or any non-2D-camera member, resolve to linear with a warning naming
  the misconfiguration. Give every member of a stack the same `CompositingSpace`,
  or none.

- **Sorted-camera indices count per render target.** A 3D camera and an `Hdr` 2D
  overlay on the same target now order deterministically, so the overlay
  composites over the base instead of overwriting it.

- **Stack members below a finalizer skip their upscaling blit.** When a stack
  defers its tone-map or display-encode pass to a finalizing camera, every member
  ordered below that finalizer — deferred and pass-disabled members alike — skips
  its upscaling blit, and the finalizer presents once, with replace, owning the
  output texture. On HDR display targets the out-texture clear color is CPU-encoded
  through the resolved transfer, gamut, and paper white, so regions no camera
  covers present a correct clear instead of a raw linear value (which reads as
  full-peak nits). A finalizer with `CameraOutputMode::Skip` cancels the skipping
  for its group.

- **Stacked GT7 under an overlay encodes correctly on HDR output.** The encoder
  keys its source gamut off the camera that tone-mapped the buffer, not a
  `Tonemapping::None` overlay composited on top. A `Tonemapping::GranTurismo7` 3D
  camera under a 2D overlay on a PQ display target no longer gets a spurious
  Rec.709 → Rec.2020 double expansion that oversaturated the image. See the
  composition-contract release note for the before/after capture.

- **UI on a `CompositingSpace::Srgb` or `CompositingSpace::Oklab` camera encodes
  into that space.** UI runs after tone mapping, which leaves the buffer encoded in
  the camera's compositing space; each UI fragment shader now writer-encodes its
  final color into the resolved compositing space, the same way the sprite shader
  does. UI over 3D cameras is unaffected (non-2D cameras resolve to linear).

- **UI and gizmos convert Rec.709-authored colors to the buffer's working gamut on
  a Rec.2020 (GT7) HDR view.** UI fragment shaders, and the 2D and 3D gizmo
  line/joint shaders, now apply `rec709_to_rec2020` (the shared
  `WORKING_COLOR_SPACE_REC2020` writer-encode) before the compositing-space encode.
  UI keys this per view off the buffer's `source_gamut`; gizmos render pre-tone-map
  and key off the global `WorkingColorSpace` like sprites and meshes. SDR and
  Rec.709 views are byte-identical (no shader def, no conversion).

- **2D gizmos writer-encode into the camera's compositing space.** Like UI and
  sprites, 2D gizmos blend into a `CompositingSpace::Srgb`/`Oklab` buffer, so they
  now encode their output to match it. 3D gizmos render pre-tone-map and are
  unaffected. SDR/linear views are byte-identical.

- **FXAA on a `CompositingSpace::Oklab` view uses the Oklab L channel for edge
  luma.** FXAA's Rec.601-weighted luma dot can go negative (and NaN under the
  square root) on an Oklab buffer's signed chroma channels, so on an Oklab-resolved
  view FXAA reads the Oklab L channel directly. Other compositing spaces are
  unchanged.
