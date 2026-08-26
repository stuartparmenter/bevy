---
title: "`ViewTarget::compositing_space` removed: camera compositing resolves once per frame"
pull_requests: []
---

A camera's `CompositingSpace` request now resolves once per frame into two
render-world components on the view entity: `ResolvedCompositingSpace` (the view's
`Option<CompositingSpace>`), written in `RenderSystems::CreateViews`, and
`ViewStackContract`, written in `RenderSystems::PrepareViews`.

- `ViewTarget::compositing_space` and `ExtractedCamera::compositing_space` have
  been removed (see the `resolved_compositing_space` guide). In
  `bevy_core_pipeline`, `ViewStackContract::compositing_space` carries the
  resolved value. `ViewStackContract` is overwritten in place and never removed,
  so keep a `ViewTarget` term in your query to skip views whose target was
  dropped.
- `encoder_input_gamut` has been removed from `bevy_core_pipeline::display_encoding`.
  Read `ViewStackContract::source_gamut` instead.
- `bevy_core_pipeline::tonemapping::effective_tonemapping` and
  `tonemap_output_gamut` have been replaced by `resolve_tonemapping`, which returns
  the substituted operator and its output gamut together
  (`ResolvedTonemapping::output_gamut`).
The writer-side encode shader defs that 2D pipelines (sprites, 2D meshes and
materials, gizmos, UI) push into their shaders are renamed: `SRGB_OUTPUT` is now
`COMPOSITING_SPACE_SRGB`, and `OKLAB_OUTPUT` is now `COMPOSITING_SPACE_OKLAB`. A
custom `Material2d` or UI shader that read the old defs should switch to the new
names, or call `bevy_render::writer_encode::writer_encode` on its composed color,
which also applies the `OUTPUT_GAMUT_REC2020` gamut conversion.

The rendering behavior changes that come with this, in how camera stacks compose,
tonemap, and display-encode, need no migration and never affect default SDR
cameras. See the release note "The composition contract: one resolution for
camera compositing".
