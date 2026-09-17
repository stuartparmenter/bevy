# Bevy Solari

[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/bevyengine/bevy#license)
[![Crates.io](https://img.shields.io/crates/v/bevy_solari.svg)](https://crates.io/crates/bevy_solari)
[![Downloads](https://img.shields.io/crates/d/bevy_solari.svg)](https://crates.io/crates/bevy_solari)
[![Docs](https://docs.rs/bevy_solari/badge.svg)](https://docs.rs/bevy_solari/latest/bevy_solari/)
[![Discord](https://img.shields.io/discord/691052431525675048.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/bevy)

![Logo](../../assets/branding/bevy_solari.svg)

## Animated meshes

Add `RaytracingMesh3d(mesh_handle.clone())` to an entity using Bevy's ordinary
`SkinnedMesh` and/or `MeshMorphWeights` components. Solari applies morph targets,
then skinning, and updates raytracing geometry and deformation motion history
automatically. The entity must use `MeshMaterial3d<StandardMaterial>`, and its mesh
must have `enable_raytracing` enabled (the default).

The same components work without `Mesh3d` for geometry that should only be visible
to rays. Solari requests animation inputs even when the mesh is outside the camera
frustum; it does not need `NoFrustumCulling`. Raster rendering still needs suitable
bounds for the animation, just as it does without Solari.

Animated meshes support triangle lists with U16 or U32 indices, or no index buffer.
Positions and normals must be Float32x3. Optional UVs and tangents use Float32x2
and Float32x4, respectively. Skinning requires Float32x4 joint weights and Uint16x4
or Uint32x4 joint indices. Additional vertex attributes are accepted; compressed
animation attributes are not supported. Static meshes retain the format
requirements documented on `RaytracingMesh3d`.

Run the procedural demonstration with:

```sh
cargo run --example solari --features bevy_solari,https,free_camera -- --animated-meshes
```

From left to right, the meshes demonstrate skinning, morph targets, and both
together. Add `--pathtracer` to render the scene entirely with rays; press Space
to pause the animation and accumulate samples. The meshes,
animation, materials, and lighting are generated locally.

## Migrating an application-side skinning producer

For ordinary Bevy animation, remove the custom raytracing skinning plugin, compute
shader, source/output buffer caches, proxy components, and producer registration.
Keep `RaytracingMesh3d` on the animated mesh entity instead of removing it when
`SkinnedMesh` is added. Existing skeleton animation and morph weight updates then
feed both rendering paths. Remove any `NoFrustumCulling` workaround that existed
solely to make joint data available to Solari.

Keep application policy for deciding which entities participate in raytracing.
In particular, rebuilding animated geometry has a per-frame cost: an animated
mesh cutoff might enter at 30 m and exit at 38 m, while static geometry enters at
150 m and exits at 175 m. Those distances are application choices, not Solari
defaults. Continue to apply LOD and animation-distance policies independently.
Each animated instance currently uses two of the scene's 500 vertex-buffer
bindings, even when instances share a source mesh.

An explicit `RaytracingGeometry` component overrides the automatic mesh path.
Keep that API for custom GPU deformation or geometry generation that does not use
Bevy's skinning and morph components.
