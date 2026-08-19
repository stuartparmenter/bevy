---
title: Physiological auto exposure and automatic white balance
authors: ["@stuartparmenter"]
pull_requests: []
---

`AutoExposure` can now model human brightness adaptation on two time scales,
following Polyphony Digital's Gran Turismo 7 presentation at SIGGRAPH 2025
("Physically Based Tone Mapping in Gran Turismo 7"). A short-term stage (pupil
and neural gain) covers a few EV within seconds. A long-term stage (receptor
sensitivity and photopigment bleaching) covers the rest of the ~12 EV range over
minutes, and adapts to light much faster than to darkness.

Bevy's existing smoothing is the short-term stage; the new opt-in
`physiological` field adds the long-term one, a slow envelope that clamps the
short-term exposure to a bounded range around itself.

```rust
commands.spawn((
    Camera3d::default(),
    AutoExposure {
        physiological: Some(PhysiologicalAdaptation::default()),
        ..default()
    },
));
```

The defaults use real physiological time scales; games will often want faster
values. `PhysiologicalAdaptation` sets the long-term speeds
(`speed_brighten`/`speed_darken`, EV per second) and the bounding range
(`bound_brighten`/`bound_darken`, EV below/above the envelope).

`AutoExposure` also gains `metering_bias`, a constant EV offset added to the
metered scene luminance. A positive bias meters the scene as brighter than it
is, so the image gets darker.

Try it in the `auto_exposure` example with the `P` key.

## Automatic white balance

Add the new `AutoWhiteBalance` component to a camera, along with
`AutoExposurePlugin`. Bevy estimates the scene's dominant illuminant and adapts
the white point toward neutral at a configurable `speed`. It is modeled on the
system from the same SIGGRAPH 2025 course.

`AutoWhiteBalance` measures in `AutoExposure`'s metering pass, so it requires
`AutoExposure` and pulls it in when spawned on its own. The same `metering_mask`
weights both the luminance histogram and the scene chromaticity.
`virtual_light_anchor` (default: 0.01) blends in a faint D65 reference at that
luminance, so near-black scenes anchor at neutral instead of chasing measurement
noise; `0.0` turns it off. The adapted color temperature is clamped to
2500-7000 K, so deliberate extreme lighting is never fully "corrected" away.

The correction composes with manual `ColorGrading` temperature/tint: it
neutralizes the scene, then your grade applies on top.

In the `auto_exposure` example, press `L` for a tungsten room light, then `B` to
watch the camera adapt the orange cast away.
