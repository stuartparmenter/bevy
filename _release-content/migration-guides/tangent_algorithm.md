---
title: Configurable tangent generation
pull_requests: []
---

`Mesh::generate_tangents()` is now `Mesh::compute_tangents(TangentAlgorithm)`, and `Mesh::with_generated_tangents()` is now `Mesh::with_computed_tangents(TangentAlgorithm)`. Pass `TangentAlgorithm::Mikktspace` or `Default::default()` to keep using MikkTSpace.

`GltfPlugin` has a new `tangent_algorithm` field, defaulting to MikkTSpace. `GltfLoader` has a corresponding `default_tangent_algorithm` field. Add these fields to exhaustive struct literals, or use `..Default::default()` for `GltfPlugin`.

`GltfLoaderSettings::tangent_algorithm` optionally overrides the plugin choice for an individual asset. It defaults to `None`; add this field to exhaustive settings literals. Existing tangents in glTF assets are preserved regardless of the selected algorithm.
