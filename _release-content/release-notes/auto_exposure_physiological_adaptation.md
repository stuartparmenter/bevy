---
title: Physiological auto exposure and automatic white balance
authors: ["@stuartparmenter"]
pull_requests: []
---

Bevy's `AutoExposure` can now model how human vision adapts to brightness on two time scales,
following the model Polyphony Digital presented for Gran Turismo 7 at SIGGRAPH 2025
("Physically Based Tone Mapping in Gran Turismo 7"). A short-term stage (pupil and neural gain)
covers a few EV within seconds. A long-term stage (receptor sensitivity and photopigment
bleaching) covers the rest of the ~12 EV range over minutes, asymmetrically: adapting to
light is much faster than adapting to darkness.

Bevy's existing smoothing is the short-term stage. The new opt-in `physiological` setting adds
the long-term stage: a slow adaptation envelope that clamps the short-term exposure to a
bounded range around itself.

```rust
commands.spawn((
    Camera3d::default(),
    AutoExposure {
        physiological: Some(PhysiologicalAdaptation::default()),
        ..default()
    },
));
```

Defaults are tuned to real physiological time scales. Games will often want faster values.
`PhysiologicalAdaptation` exposes the long-term speeds (`speed_brighten`/`speed_darken`, EV per
second per direction) and the bounding range (`bound_brighten`/`bound_darken`, EV below/above
the envelope).

`AutoExposure` also gains a `metering_bias`, a constant EV offset applied to the metered scene
luminance. A positive bias meters the scene as brighter than it is, so the final image is
darker.

Try it in the `auto_exposure` example with the `P` key.

## Automatic white balance

Add the new `AutoWhiteBalance` component to a camera (with the `AutoExposurePlugin`) and the
renderer estimates the scene's dominant illuminant and slowly adapts the white point toward
neutral at a configurable `speed`. It is modeled on the system from the same SIGGRAPH 2025
course.

```rust
commands.spawn((
    Camera3d::default(),
    AutoExposure::default(),
    AutoWhiteBalance::default(),
));
```

- Shared metering. The same `metering_mask` weights both the luminance histogram and the scene
  chromaticity measurement, so `AutoWhiteBalance` requires `AutoExposure` and pulls it in when
  spawned on its own.
- Dark-scene stability. Near-black scenes anchor at neutral instead of chasing measurement
  noise. `virtual_light_anchor` (default: 0.01) sets the luminance of the faint D65 reference
  that holds them there, and `0.0` turns it off.
- Bounded output. The adapted chromaticity is converted to a correlated color temperature
  (McCamy's approximation) plus an off-locus tint, with the temperature clamped to 2500-7000 K,
  so deliberate extreme lighting is never fully "corrected" away.

The correction composes with manual `ColorGrading` temperature/tint: it neutralizes the scene
first, then the artist's grade applies on top.

Try it in the `auto_exposure` example: press `L` to switch the room light to tungsten,
then `B` to watch the camera adapt the orange cast away.
