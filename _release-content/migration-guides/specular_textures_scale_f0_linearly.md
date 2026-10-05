---
title: "`StandardMaterial` specular textures scale F0 linearly"
pull_requests: []
---

`StandardMaterial::specular_texture` and `specular_tint_texture` now scale F0 linearly, matching the `KHR_materials_specular` spec.
Previously, the specular texture's alpha was halved and both textures were effectively squared, so specular maps came out much dimmer than intended.

If you set `reflectance` to 2.0 to make up for this, as the old docs suggested, you can probably go back to the default of 0.5:

```rust
// 0.20
StandardMaterial {
    reflectance: 2.0,
    specular_texture: Some(specular_map),
    ..default()
}

// 0.21
StandardMaterial {
    specular_texture: Some(specular_map),
    ..default()
}
```

If you'd rather keep your materials looking exactly the same, leave `reflectance` alone and convert your textures instead: a specular alpha of `v` becomes `v * v / 4`, and a linear tint of `t` becomes `t * t`.

If you build your own materials from `GltfMaterial`, note that `reflectance` now includes the IOR and `specular_tint` is now the square root of `specularColorFactor`, so don't apply the IOR a second time.
