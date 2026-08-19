---
title: Gran Turismo 7 tone mapping and physically based glare
authors: ["@stuartparmenter"]
pull_requests: []
---

Bevy has a new tone-mapping operator, `Tonemapping::GranTurismo7`. It ports the
operator Polyphony Digital published with their SIGGRAPH 2025 course
"Physically Based Tone Mapping in Gran Turismo 7". Their reference
implementation is MIT licensed, Copyright (c) 2025 Polyphony Digital Inc.

It blends a per-channel filmic curve with a hue-preserving ICtCp branch (60%
ICtCp, 40% per-channel by default) and adds a luminance-driven chroma fade, so
colors desaturate near peak white instead of clipping. Unlike the LUT-based
operators (AgX, TonyMcMapface, BlenderFilmic) it is algorithmic and does not
need the `tonemapping_luts` cargo feature.

```rust
commands.spawn((
    Camera3d::default(),
    Tonemapping::GranTurismo7,
));
```

On an SDR display target it tone-maps against Gran Turismo's 250-nit paper
white and rescales into the sRGB range.

Add `GranTurismo7Params` to the camera to replace the operator's baked
defaults: `blend_ratio`, the chroma fade band, and the curve shape parameters.
Out-of-range values are clamped, with a warning.

On an HDR display target, the tone curve is rebuilt around the display's
`peak_luminance_nits`, clamped to 250-10000 nits and to at least
`paper_white_nits`. Output is linear Rec.2020, scaled so `1.0` is paper white
and unclamped up to `peak / paper_white`. The release note "HDR display output
(scRGB-linear and HDR10/PQ)" covers the display-encoding pass that takes it
from there, including gamut compression.

## Physically based veiling glare as a bloom scatter model

`Bloom` has a new `scatter` field selecting how the blurred pyramid levels are
weighted when composited back onto the image:

- `BloomScatterModel::Aesthetic` (the default) is the hand-tuned parametric
  curve Bevy's bloom has always used.
- `BloomScatterModel::Gt7Glare { f_number }` replaces that curve with per-level
  weights derived from the far-field (Fraunhofer) diffraction point-spread
  function of a camera aperture, the veiling glare Polyphony Digital presented
  at SIGGRAPH 2025.

```rust
commands.spawn((
    Camera3d::default(),
    // or: Bloom { scatter: BloomScatterModel::Gt7Glare { f_number: 8.0 }, ..default() }
    Bloom::GT7_GLARE,
));
```

The weights are precomputed for the f/1 to f/22 full-stop ladder and
interpolated in between. A wide aperture gives a tight glare that falls off
steeply around bright sources. At f/22 the energy spreads into a wide, soft
veil.

A physical point-spread function applies to all light, so the glare model is
threshold-free: any configured `BloomPrefilter` is ignored with a warning, and
compositing is forced to `BloomCompositeMode::EnergyConserving`.
`Bloom::intensity` is the total fraction of energy scattered out of the sharp
image.

Try it in the `bloom_3d` example: `B` toggles the scatter model and `O`/`L`
change the aperture.
