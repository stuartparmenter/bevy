---
title: Transmission blur tap count is a pipeline override
pull_requests: []
---

The `SCREEN_SPACE_SPECULAR_TRANSMISSION_BLUR_TAPS` shader def is no longer set. `bevy_pbr::transmission` declares `override SCREEN_SPACE_SPECULAR_TRANSMISSION_BLUR_TAPS: i32 = 8;` instead, and `MeshPipeline` sets it on the fragment stage from `ScreenSpaceTransmissionQuality`.

Custom shaders that read the shader def with `#{...}` (or guard on it with `#ifdef`) should import the override:

```wgsl
// Bevy 0.19
#ifdef SCREEN_SPACE_SPECULAR_TRANSMISSION_BLUR_TAPS
let num_taps = #{SCREEN_SPACE_SPECULAR_TRANSMISSION_BLUR_TAPS};
#else
let num_taps = 8;
#endif

// Bevy 0.20
import bevy_pbr::transmission::SCREEN_SPACE_SPECULAR_TRANSMISSION_BLUR_TAPS;
let num_taps = SCREEN_SPACE_SPECULAR_TRANSMISSION_BLUR_TAPS;
```

Custom pipelines that import the module without going through `MeshPipeline` can set it with the key in `bevy_pbr::SCREEN_SPACE_SPECULAR_TRANSMISSION_BLUR_TAPS`:

```rust
FragmentState {
    constants: vec![(bevy_pbr::SCREEN_SPACE_SPECULAR_TRANSMISSION_BLUR_TAPS.into(), 16.0)],
    ..default()
}
```
