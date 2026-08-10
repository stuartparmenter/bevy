---
title: "Tone mapping render path: node-based tone mapping and the `Rgba16Float` intermediate"
pull_requests: []
---

In 0.19, `Hdr` cameras tone-mapped in the post-process pass on an `Rgba16Float` main
texture. Every other camera applied its `Tonemapping` operator in the sprite, mesh2d, and
PBR fragment shaders (the `TONEMAP_IN_SHADER` shader def) on an 8-bit main texture.

In 0.20, a tone-mapped camera keeps the in-shader path only when all of these hold:

- no `Hdr`
- no `bevy_render::view::NeedsSceneLinearTarget`, a required component of bloom, auto
  exposure, auto white balance, depth of field, motion blur, TAA, and DLSS
- no `bevy_camera::TonemappingPass`, a required component of `GranTurismo7Params`
- a `CompositingSpace` that is absent or `Linear`
- an SDR sRGB `Window` render target
- no other active camera on that target

The default `Camera3d` meets all of these and renders unchanged from 0.19. The render
world marks these cameras with `bevy_render::camera::TonemapInShader`. Read it to detect
the path; do not insert or remove it yourself.

Every other tone-mapped camera now tone-maps in the post-process pass on a scene-linear
`Rgba16Float` intermediate. Expect small visual differences:

- Transparents blend in scene-linear before tone mapping, instead of tone-mapped values
  being alpha-blended.
- Everything the camera renders is tone-mapped, including gizmos and custom materials
  that did not call `tone_mapping()` themselves. Custom material shaders should output
  scene-linear color.
- `DebandDither` runs once on the blended image instead of per fragment, and the fp16
  intermediate removes the banding the old 8-bit intermediate introduced.
- Stacked cameras rendering to the same target compose in scene-linear and tone-map once,
  on the last camera. Earlier cameras' operators do not run (Bevy warns if they differ).

`Tonemapping` and `DebandDither` moved from `bevy_core_pipeline::tonemapping` to
`bevy_render::view`. Rust imports still work through re-exports, but the reflected type
paths changed: update scene files and Bevy Remote Protocol component keys that name them.

`Tonemapping::None` is now a pure passthrough: no tone-mapping pass runs, and no
`ColorGrading` exposure or post-saturation is applied. If you used `ColorGrading` or
`AutoExposure` with `Tonemapping::None`, switch to `Tonemapping::Linear`, which applies
grading and dither with no tone curve.

`NeedsSceneLinearTarget` and `TonemappingPass` are ordinary required components, so
`remove_with_requires` on one effect also strips a marker other effects on the camera may
still need; use plain `remove::<T>()` instead. On WebGL2 the fp16 path requires
`EXT_color_buffer_float`, already required for `Hdr` cameras.
