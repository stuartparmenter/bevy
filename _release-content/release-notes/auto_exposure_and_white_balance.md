---
title: Auto exposure and auto white balance
authors: ["@stuartparmenter"]
pull_requests: [26210, 26211]
---

`AutoExposure` now meters the scene in absolute luminance, in EV100. Its metering `range` and the x-axis of its compensation curve describe real light levels. A mid-gray surface is about 15 EV100 in direct sunlight, 8 in an office and -3 under a full moon. The camera's `Exposure` no longer changes what these settings mean.

The view now renders with the adapted exposure instead of the camera's `Exposure`. This keeps the HDR main texture precise in very bright and very dark scenes. Read the adapted exposure from the camera's `AutoExposureState`. At a camera cut, set `AutoExposure::reset` to jump to the new exposure instead of adapting to it. Adaptation now runs on real time, so it continues while virtual time is paused.

## Adaptation like the human eye

Eyes adapt to a few stops of change within seconds, and to the rest over minutes. The new `AutoExposure::physiological` field adds this slow adaptation, from the SIGGRAPH 2025 course "Physically Based Tone Mapping in Gran Turismo 7". After a long time in the dark, a bright scene stays overexposed until the slow adaptation catches up. The defaults take minutes, like the human eye, so games often use faster speeds:

```rust
commands.spawn((
    Camera3d::default(),
    AutoExposure {
        physiological: Some(PhysiologicalAdaptation {
            speed_brighten: 0.4,
            speed_darken: 0.1,
            ..default()
        }),
        ..default()
    },
    AutoWhiteBalance::default(),
));
```

## Auto white balance

The new `AutoWhiteBalance` component corrects the color of the scene's light toward neutral white, like the auto white balance of a camera. Under warm indoor light, the orange cast fades as the camera adapts. It corrects color temperatures from 2500 K to 7000 K. `ColorGrading` white balance still applies after it. It requires `AutoExposure`, because it measures the scene in the same pass.

Both features need `AutoExposurePlugin` and compute shaders. In the `auto_exposure` example, press `P` to toggle physiological adaptation, `L` to toggle a warm light, and `B` to toggle auto white balance.
