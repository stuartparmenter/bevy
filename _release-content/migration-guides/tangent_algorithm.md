---
title: "`Mesh::generate_tangents` is now `Mesh::compute_tangents`"
pull_requests: []
---

`Mesh::generate_tangents` and `Mesh::with_generated_tangents` are now `Mesh::compute_tangents` and `Mesh::with_computed_tangents`, and take a `TangentAlgorithm`. Pass `TangentAlgorithm::Mikktspace` to keep the previous behavior.

```rust
// 0.20
mesh.generate_tangents()?;

// 0.21
mesh.compute_tangents(TangentAlgorithm::Mikktspace)?;
```

`GltfPlugin` and `GltfLoaderSettings` have a new `tangent_algorithm` field, and `GltfLoader` has a new `default_tangent_algorithm` field. Add them to struct literals that don't use `..Default::default()`.
