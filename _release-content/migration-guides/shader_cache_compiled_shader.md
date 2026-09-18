---
title: "`ShaderCache::get` returns `CompiledShader`"
pull_requests: []
---

`bevy_shader::ShaderCache::get` now returns `Arc<CompiledShader<M>>` instead of `Arc<M>`. Use `.module` for the shader module and `.overrides.resolve(&constants)` to map the keys in a pipeline descriptor's `constants` to the identifiers in the compiled module.

WESL mangles the name of an `override` declared in an imported module, so a `constants` key for one is now the override's module path: `pkg::module::NAME` for an embedded module, or `package::path::to::module::NAME` for a module loaded by asset path (`package::shaders::util::TAPS` for `shaders/util.wesl`). An override declared in the stage's own shader keeps its bare name. Keys whose override does not exist in the compiled module are skipped, so a value can be set for a module that is only conditionally imported. WGSL and SPIR-V shaders receive bare keys unchanged.
