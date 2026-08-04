---
title: Gran Turismo 7 tone mapping and physically based glare
authors: ["@stuartparmenter"]
pull_requests: []
---

Bevy has a new tone-mapping operator: `Tonemapping::GranTurismo7`.
It is a native port of the operator Polyphony Digital uses in Gran Turismo 7,
published with their SIGGRAPH 2025 course "Physically Based Tone Mapping in
Gran Turismo 7" (reference implementation MIT licensed, Copyright (c) 2025
Polyphony Digital Inc.).

It blends a per-channel filmic curve with a hue-preserving ICtCp branch
(60% ICtCp, 40% per-channel by default) and adds a luminance-driven
chroma fade, so colors desaturate near peak white instead of clipping.
Unlike the LUT-based operators (AgX, TonyMcMapface, BlenderFilmic) it is
algorithmic and does not need the `tonemapping_luts` cargo feature.

```rust
commands.spawn((
    Camera3d::default(),
    Tonemapping::GranTurismo7,
));
```

On an SDR target it tone-maps against Gran Turismo's 250-nit paper-white
calibration and rescales into the sRGB range.

Add the `GranTurismo7Params` component to a camera using
`Tonemapping::GranTurismo7` to replace the operator's baked defaults:
`blend_ratio`, the chroma fade band, and the curve shape parameters.
Out-of-range values are clamped, with a warning.

```rust
commands.spawn((
    Camera3d::default(),
    Tonemapping::GranTurismo7,
    GranTurismo7Params {
        blend_ratio: 1.0, // fully hue-preserving
        ..Default::default()
    },
));
```

HDR mode turns on whenever the camera's resolved `DisplayTarget` requests an
HDR transfer (scRGB-linear, PQ, or extended-range sRGB), with or without
`GranTurismo7Params`. The tone curve is rebuilt around the display's
`peak_luminance_nits`, clamped to the supported 250-10000 nit range and to at
least `paper_white_nits`. The output is rescaled so `1.0` equals the display's
paper white.

On those views the operator emits its native linear Rec.2020 output,
unclamped over `[0, peak / paper_white]`. Converting Rec.2020 to a narrower
display gamut can push components out of gamut, so out-of-gamut compression is
now on by default (`DisplayGamutCompression::Auto`). The "HDR display output"
release note describes the display-encoding pass.

On SDR targets, cameras without the component produce byte-identical output.

## Physically based veiling glare as a bloom scatter model

`Bloom` has a new `scatter` field selecting how the blurred pyramid levels are
weighted when composited back onto the image:

- `BloomScatterModel::Aesthetic` (the default) is the hand-tuned parametric
  curve Bevy's bloom has always used.
- `BloomScatterModel::Gt7Glare { f_number }` replaces that curve with per-level
  weights derived from the far-field (Fraunhofer) diffraction point-spread
  function of a camera aperture. This is the physically based veiling glare
  Polyphony Digital presented for Gran Turismo 7 at SIGGRAPH 2025.

```rust
commands.spawn((
    Camera3d::default(),
    // or: Bloom { scatter: BloomScatterModel::Gt7Glare { f_number: 8.0 }, ..default() }
    Bloom::GT7_GLARE,
));
```

The weights are precomputed for the standard f/1-f/22 full-stop ladder and
interpolated in between. A wide aperture gives a tight glare that falls off
steeply around bright sources. At f/22 the energy spreads into a wide, soft
veil. The `bloom::glare` module docs carry the derivation and references.

A physical point-spread function applies to all light, so the glare model is
threshold-free: any configured `BloomPrefilter` is ignored with a warning, and
compositing is forced to energy-conserving blending. `Bloom::intensity` still
means the total fraction of energy scattered out of the sharp image.

Try it in the `bloom_3d` example: `B` toggles the scatter model and `O`/`L`
step the aperture through the F-stop ladder.
