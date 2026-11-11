---
title: "GPU mesh preprocessing systems are merged"
pull_requests: []
---

The render graph systems `allocate_uniforms`, `unpack_bins` and `early_prepass_build_indirect_parameters` are merged into `early_gpu_preprocess`, and `late_prepass_build_indirect_parameters` is merged into `late_gpu_preprocess`.
Systems ordered against a removed system should be ordered against the system it was merged into:

```rust
// 0.20
my_render_system.after(early_prepass_build_indirect_parameters)

// 0.21
my_render_system.after(early_gpu_preprocess)
```
