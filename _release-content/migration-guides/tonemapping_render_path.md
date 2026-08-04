---
title: "Tone mapping render path: node-based tone mapping and the `Rgba16Float` intermediate"
pull_requests: []
---

In 0.19, `Hdr` cameras tone-mapped in the post-process pass on an `Rgba16Float`
main texture. Every other camera applied its `Tonemapping` operator in the
sprite, mesh2d, and PBR fragment shaders (the `TONEMAP_IN_SHADER` shader def) on
an 8-bit (`Rgba8UnormSrgb` / `Rgba8Unorm`) main texture.

A tone-mapped camera (`Tonemapping` other than `Tonemapping::None`) keeps that
in-shader path, with no tonemapping node, as long as all of these hold:

- tone mapping enabled and no `Hdr`,
- no `bevy_render::view::NeedsSceneLinearTarget`, which bloom, auto exposure,
  auto white balance, depth of field, motion blur, TAA and DLSS all pull in as a
  required component,
- no `bevy_camera::TonemappingPass`, which `GranTurismo7Params` pulls in as
  a required component,
- `CompositingSpace` absent or `Linear`,
- resolved `DisplayTarget` is `DisplayTarget::SDR_SRGB`,
- the render target is a `Window`,
- it is the sole active camera on that target.

The default `Camera3d` (`TonyMcMapface`, no `Hdr`, sole camera on a plain SDR
sRGB window) meets them all and is unchanged from 0.19.

Bevy records the decision as the `bevy_render::camera::TonemapInShader` marker
on the render-world camera entity. Do not insert or remove it yourself. Read it
instead of inferring a view's path from its main-texture format.

A tone-mapped camera that fails any condition tone-maps in the post-process pass
and renders to a scene-linear `Rgba16Float` intermediate. Expect small visual
differences:

- Transparents blend in scene-linear before tone mapping, instead of
  tone-mapped values being alpha-blended.
- Everything the camera renders is tone-mapped, including gizmos and custom
  materials that did not call `tone_mapping()` themselves. Custom material
  shaders should output scene-linear color and let the pass tone-map.
- `DebandDither` is applied once in the pass, on the blended image, rather than
  per fragment.
- Stacked cameras rendering to the same target (`ClearColorConfig::None`, no
  viewport) compose in scene-linear and tone-map once on the last camera.
  Earlier cameras' operators do not run (Bevy warns if they differ).
- The fp16 intermediate removes the banding the old 8-bit intermediate
  introduced.

`Tonemapping` and `DebandDither` moved from `bevy_core_pipeline::tonemapping` to
`bevy_render::view`. Both are re-exported from the old path, so Rust imports
still work. Their reflected type paths changed to
`bevy_render::view::Tonemapping` and `bevy_render::view::DebandDither`: update
scene files and Bevy Remote Protocol component keys that name them.

`NeedsSceneLinearTarget` and `TonemappingPass` are ordinary required
components, so `remove_with_requires` on an effect also strips the shared
marker, re-enabling the in-shader fold. Any other scene-linear effect still on
the camera then samples a tone-mapped 8-bit buffer. Use plain `remove::<T>()`
instead, which leaves the marker in place at the cost of an fp16 intermediate
and renders identical pixels. If you match custom render graphs against the
exact set of components on a camera entity, account for the two markers. On
WebGL2 the fp16 path requires `EXT_color_buffer_float`, already required for
`Hdr` cameras.

`Tonemapping::None` is a passthrough: no tonemapping pass runs and no
`ColorGrading` exposure or post-saturation is applied. If you used `ColorGrading`
or `AutoExposure` with `Tonemapping::None`, switch to `Tonemapping::Linear`,
which applies grading and dither with no tone curve.
