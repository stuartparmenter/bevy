---
title: "Deferred rendering stores diffuse transmission"
pull_requests: []
---

`StandardMaterial`s with `diffuse_transmission` above `0.0` no longer switch to forward rendering
when the `OpaqueRendererMethod` is `Auto`. The deferred specular tint target stores the
transmission beside the tint, and the deferred lighting pass and Solari light the transmitted lobe.
On devices without that target (see `DeferredSpecularTintSupport`), deferred materials lose their
transmission; set `opaque_render_method` to `OpaqueRendererMethod::Forward` to keep it there.

`pack_specular_tint(tint)` is now `pack_specular_tint_transmission(tint, diffuse_transmission)`,
and `unpack_diffuse_transmission` reads the transmission back. The tint keeps 7 mantissa bits
instead of 9.

`prepare_world_normal` now inverts the normal of a double-sided back face unless the shader also
applies normal mapping (`VERTEX_UVS`, `VERTEX_TANGENTS` and `STANDARD_MATERIAL_NORMAL_MAP`), which
inverts it instead. A custom shader that calls `apply_normal_mapping` without all three and builds
its TBN from the prepared `world_normal` must flip it back.
