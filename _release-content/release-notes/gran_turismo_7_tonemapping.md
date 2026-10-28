---
title: Gran Turismo 7 tonemapping and glare
authors: ["@stuartparmenter"]
pull_requests: []
---

Bevy has a new tonemapping method, `Tonemapping::GranTurismo7`, from Polyphony Digital's SIGGRAPH 2025 course "Physically Based Tone Mapping in Gran Turismo 7". Use it when your game runs on HDR displays. On an HDR display, it fits its curve to the display's peak luminance, so highlights use the display's full range. On an SDR display, it uses Gran Turismo's SDR curve. It doesn't need the `tonemapping_luts` feature.

The method blends a per-channel filmic curve, which shifts bright colors toward white like a camera, with a curve that keeps hue. Colors near peak brightness lose saturation. `GranTurismo7Params` tunes both curves per camera:

```rust
commands.spawn((
    Camera3d::default(),
    Tonemapping::GranTurismo7,
    // Optional. The defaults match Polyphony Digital's reference implementation.
    GranTurismo7Params {
        blend_ratio: 0.8,
        ..default()
    },
));
```

On an HDR display, a camera with a method that limits its output to SDR, such as `TonyMcMapface`, the `Camera3d` default, uses `GranTurismo7` instead and warns once. Set `Tonemapping::GranTurismo7` on the camera to silence the warning. Press `0` in the `tonemapping` example to compare it with the other methods.

## Glare

`Bloom` has a new `scatter` field. `BloomScatterModel::Gt7Glare` replaces bloom's hand-tuned curve with lens glare, from the same course. It computes the glare from the diffraction of light through a camera aperture, so you set it with an f-number instead of curve settings. A low f-number keeps the glare tight around bright lights. A high f-number spreads it into a wide, soft glow. `Bloom::GT7_GLARE` is a preset at f/5.6:

```rust
commands.spawn((
    Camera3d::default(),
    Bloom {
        scatter: BloomScatterModel::Gt7Glare { f_number: 8.0 },
        ..Bloom::GT7_GLARE
    },
));
```

Glare applies to all light, so it ignores the prefilter threshold and always uses `BloomCompositeMode::EnergyConserving`. `Bloom::intensity` sets the fraction of light scattered out of the sharp image. In the `bloom_3d` example, press `B` to switch to glare, and `O` and `L` to change the f-number.
