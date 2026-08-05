---
title: "`ViewTarget::compositing_space` removed: camera compositing resolves once per frame"
pull_requests: []
---

A camera's `CompositingSpace` request is now resolved once per frame into two
render-world components on the view entity: `ResolvedCompositingSpace` (the view's
`Option<CompositingSpace>`, written in `RenderSystems::CreateViews`) and
`ViewStackContract` (the compositing space plus encoder source gamut, tone-map and
display-encode roles, and blit disposition, written in `RenderSystems::PrepareViews`).
A solo camera resolves to its own request, so its pipeline keys and pixels are
unchanged.

- `ViewTarget::compositing_space` has been removed. Read `ResolvedCompositingSpace`
  (in `bevy_render`) or `ViewStackContract::compositing_space` (in
  `bevy_core_pipeline`) instead. `ViewStackContract` is overwritten in place and
  never removed, so keep a `ViewTarget` term in your query to skip views whose
  target was dropped. Don't use `ExtractedCamera::compositing_space` (the raw
  per-camera request) for pass or pipeline-key decisions.
- `encoder_input_gamut` has been removed from `bevy_core_pipeline::display_encoding`.
  Read `ViewStackContract::source_gamut` instead.
- `bevy_core_pipeline::tonemapping::effective_tonemapping` and
  `tonemap_output_gamut` have been replaced by `resolve_tonemapping`, which returns
  the substituted operator and its output gamut together
  (`ResolvedTonemapping::output_gamut`).
- `SortedCamera::hdr` has been removed: cameras sort by `(order, target)` alone.
  Read `ExtractedCamera::hdr` on the view entity if you need a camera's raw HDR
  request.

The writer-side encode shader defs that 2D pipelines (sprites, 2D meshes and
materials, gizmos, UI) push into their shaders are renamed: `SRGB_OUTPUT` is now
`COMPOSITING_SPACE_SRGB`, and `OKLAB_OUTPUT` is now `COMPOSITING_SPACE_OKLAB`. The
gamut-convert def in writer shaders is now `OUTPUT_GAMUT_REC2020` (UI previously
pushed `WORKING_COLOR_SPACE_REC2020` for this; that name now keeps only its
project-global meaning). A custom `Material2d` or UI shader that read the old defs
should switch to the new names, or call `bevy_render::writer_encode::writer_encode`
on its composed color, which handles all three defs.

Cameras that share a main texture and form a stack now resolve to a single
compositing space. Conflicting requests, or any non-2D-camera member, resolve to
linear with a warning naming the misconfiguration: give every stack member the same
`CompositingSpace`, or none. The other behavior changes in this area — deterministic
stack ordering, one final blit per stack, writer-encode into the resolved space,
FXAA luma on `CompositingSpace::Oklab` views — are bug fixes to camera stacks,
non-default compositing spaces, and HDR output that need no migration and never
affect default SDR cameras. The "The composition contract: one resolution for
camera compositing" release note covers them.
