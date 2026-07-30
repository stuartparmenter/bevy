---
title: "Tone mapping render path: node-based tone mapping and the `Rgba16Float` intermediate"
pull_requests: []
---

In 0.19, cameras with the `Hdr` marker tone-mapped in the post-process
tonemapping pass and rendered to an `Rgba16Float` main texture, while every
other ("SDR") camera applied its `Tonemapping` operator inside the sprite,
mesh2d, and PBR fragment shaders (the `TONEMAP_IN_SHADER` shader def) on an
8-bit (`Rgba8UnormSrgb` / `Rgba8Unorm`) main texture.

That in-shader fast path still exists. A tone-mapped camera (`Tonemapping` other
than `Tonemapping::None`) keeps it — per-fragment in-shader tone mapping, its
8-bit main texture, and no tonemapping node — as long as all of these hold
(`eligible_in_shader_tonemap` in `bevy_render/src/camera.rs`):

- tone mapping enabled and no `Hdr`,
- no `bevy_render::view::NeedsSceneLinearTarget`, which bloom, auto exposure,
  auto white balance, depth of field, motion blur, TAA and DLSS all pull in as a
  required component,
- no `bevy_camera::NeedsNodeTonemapping`, which `GranTurismo7Params` pulls in as
  a required component,
- `CompositingSpace` absent or `Linear`,
- resolved `DisplayTarget` is `DisplayTarget::SDR_SRGB`,
- the render target is a `Window`,
- it is the sole active camera on that target.

The default `Camera3d` (`TonyMcMapface`, no `Hdr`, sole camera on a plain SDR
sRGB window) meets these conditions: it keeps its 8-bit path and memory
footprint and is unchanged from 0.19.

Camera extraction publishes the decision on the camera's render-world entity as
the `bevy_render::camera::TonemapInShader` marker (auto-managed; do not insert
or remove it yourself). Render-world code that needs to know which path a view
took should read that marker instead of inferring it from the main-texture
format.

A tone-mapped camera that fails any condition instead tone-maps in the
post-process pass and renders to a scene-linear `Rgba16Float` intermediate.
Triggers: the `Hdr` marker, scene-linear post-processing or anti-aliasing,
`NeedsNodeTonemapping`, a non-`Linear` `CompositingSpace`, an HDR display
target, a non-window render target, or sharing the target with another active
camera. For those cameras, expect small visual differences:

- Transparents blend in scene-linear before tone mapping, instead of
  tone-mapped values being alpha-blended.
- Everything the camera renders is tone-mapped, including gizmos and custom
  materials that did not call `tone_mapping()` themselves. Custom material
  shaders should output scene-linear color and let the pass tone-map.
- `DebandDither` is applied once in the pass, on the blended image, rather than
  per fragment.
- Stacked cameras rendering to the same target (`ClearColorConfig::None`, no
  viewport) compose in scene-linear and tone-map once on the last camera;
  earlier cameras' operators do not run (Bevy warns if they differ).
- The fp16 intermediate also removes the banding the old 8-bit intermediate
  introduced.

`Tonemapping` and `DebandDither` moved from `bevy_core_pipeline::tonemapping` to
`bevy_render::view`, so camera extraction can pick the main-texture format from
the operator directly. Both are re-exported from their old path and Rust imports
are unaffected, but their reflected type paths changed to
`bevy_render::view::Tonemapping` and `bevy_render::view::DebandDither`: update
scene files and Bevy Remote Protocol component keys that name them.

The two veto markers above, `NeedsSceneLinearTarget` and
`NeedsNodeTonemapping`, are ordinary required components. Removing an effect
with `remove_with_requires` also strips the shared marker — which re-enables the
in-shader fold for any other scene-linear effect still on the camera, and that
effect then samples a tone-mapped 8-bit buffer. Use plain `remove::<T>()`
instead; it leaves the marker in place, costing an fp16 intermediate and
rendering identical pixels. If you match custom render graphs against the exact
set of components on a camera entity, account for the two markers. On WebGL2 the
fp16 path requires `EXT_color_buffer_float` (already required for `Hdr` cameras
and widely supported).

`Tonemapping::None` is a true passthrough: no tonemapping pass runs and no
`ColorGrading` exposure or post-saturation is applied. If you used `ColorGrading`
or `AutoExposure` with `Tonemapping::None`, switch to `Tonemapping::Linear`,
which applies grading and dither with no tone curve.
