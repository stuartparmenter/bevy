---
title: Physiological auto exposure and automatic white balance
authors: ["@stuartparmenter"]
pull_requests: []
---

Bevy's `AutoExposure` can now model how human vision adapts to brightness on two time scales,
following the two-stage model Polyphony Digital presented for Gran Turismo 7 at SIGGRAPH 2025
("Physically Based Tone Mapping in Gran Turismo 7"). A short-term stage (pupil and neural gain)
covers a few EV within seconds; a long-term stage (receptor sensitivity and photopigment
bleaching) covers the rest of the ~12 EV range over minutes, asymmetrically — adapting to light
is much faster than to darkness. The long-term stage *bounds* the short-term one: walk from
daylight into a cave and the scene brightens a little immediately but stays dark until you've
truly adapted.

Bevy's existing smoothing is the short-term stage. The new opt-in `physiological` setting adds
the long-term stage: a slow, asymmetric adaptation envelope tracked per camera on the GPU that
clamps the short-term exposure to a bounded range around itself.

```rust
commands.spawn((
    Camera3d::default(),
    AutoExposure {
        physiological: Some(PhysiologicalAdaptation::default()),
        ..default()
    },
));
```

Defaults are tuned to real physiological time scales; games will often want faster values.
`PhysiologicalAdaptation` exposes the long-term speeds (`speed_brighten`/`speed_darken`, EV per
second per direction) and the bounding range (`bound_brighten`/`bound_darken`, EV below/above
the envelope).

`AutoExposure` also gains a `metering_bias`: a constant EV offset applied to the metered scene
luminance, so the meter can sit above or below what the histogram measured. A positive bias
meters the scene as brighter than it is, and darkens the final image.

Try it in the `auto_exposure` example with the `P` key.

## Automatic white balance

Add the new `AutoWhiteBalance` component to a camera (with the `AutoExposurePlugin`) and the
renderer estimates the scene's dominant illuminant and slowly adapts the white point toward
neutral, so a scene lit by warm tungsten or cool daylight no longer carries a permanent color
cast. It is modeled on the system from the same SIGGRAPH 2025 course.

```rust
commands.spawn((
    Camera3d::default(),
    AutoExposure::default(),
    AutoWhiteBalance::default(),
));
```

* **Shared metering** — the measurement rides along in the auto-exposure compute pass: the same
  metering-mask weights that build the luminance histogram also accumulate a luminance-weighted
  average of the scene chromaticity in Yxy space. One dispatch serves both adaptations, so
  `AutoWhiteBalance` requires `AutoExposure` and pulls it in when spawned on its own.
* **Dark-scene stability** — a faint *virtual light* (an ideal D65 source with a configurable
  luminance) is blended in as one more luminance-weighted reference. Its influence scales with the
  inverse of the scene luminance, so near-black scenes anchor at neutral instead of chasing
  measurement noise.
* **Bounded output** — the adapted chromaticity is converted to a correlated color temperature
  (McCamy's approximation) plus an off-locus tint, with the temperature clamped to 2500 K–7000 K,
  so deliberate extreme lighting is never fully "corrected" away.

The correction goes through Bevy's existing CAM16 white-balance machinery and *composes* with
manual `ColorGrading` temperature/tint: the automatic correction neutralizes the scene first, and
the artist's grade applies on top — a deliberate warm look stays warm.

Try it in the updated `auto_exposure` example: press `L` to switch the room light to tungsten,
then `B` to watch the camera adapt the orange cast away.
