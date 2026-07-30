---
title: Gran Turismo 7 tone mapping and physically based glare
authors: ["@stuartparmenter"]
pull_requests: []
---

Bevy has a new tone-mapping operator: `Tonemapping::GranTurismo7`, a native port of the
operator Polyphony Digital uses in Gran Turismo 7, published with their SIGGRAPH 2025 course
"Physically Based Tone Mapping in Gran Turismo 7" (reference implementation MIT licensed,
Copyright (c) 2025 Polyphony Digital Inc.).

It blends a per-channel filmic curve with a hue-preserving ICtCp branch (60% hue-preserving /
40% per-channel by default), plus a luminance-driven chroma fade so colors desaturate as they
approach peak white instead of clipping to hard white — a "camera-like" highlight response
that keeps hues stable. Unlike the LUT-based operators (AgX, TonyMcMapface, BlenderFilmic),
GranTurismo7 is fully algorithmic and does not require the `tonemapping_luts` cargo feature.

```rust
commands.spawn((
    Camera3d::default(),
    Tonemapping::GranTurismo7,
));
```

The operator is peak-luminance aware and drives both SDR and HDR displays. On SDR targets it
tone-maps against Gran Turismo's 250-nit paper-white calibration and rescales into the sRGB
range; on HDR targets it runs in HDR mode and feeds the display-encoding pass directly.

A new `GranTurismo7Params` component exposes the operator's artistic dials (`blend_ratio`,
the chroma fade band, and the curve shape parameters). Adding it to a camera that uses
`Tonemapping::GranTurismo7` uploads the (validated) parameters to the GPU each frame,
replacing the operator's baked defaults:

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

When the camera renders to a target whose resolved `DisplayTarget` requests an HDR transfer
(scRGB-linear, PQ, or extended-range sRGB), the operator is configured in HDR mode — with or
without `GranTurismo7Params` (defaults are used if absent): the tone curve is rebuilt around
the display's `peak_luminance_nits` (clamped to the supported 250–10000 nit range, and to at
least `paper_white_nits`), and the output is rescaled so `1.0` equals the display's paper white.

On those HDR views the operator emits its native linear Rec.2020 display-referred output —
unclamped, `[0, peak / paper_white]` — straight into the display-encoding pass, so no highlight
or wide-gamut information is lost between the operator and the display signal. The encoder's
gamut stage then becomes a true source → display transform, keyed on the display's primaries
rather than its transfer (extended-range sRGB pairs with either Rec.709 or Display-P3):

- **PQ targets** (Rec.2020 primaries): identity — GT7's output is already in the signal's primaries.
- **Rec.709-primaried targets** (scRGB, and Rec.709 extended-range sRGB / "web HDR"): a
  full-precision Rec.2020 → Rec.709 contraction.
- **Display-P3 targets** (Display-P3 extended-range sRGB): a Rec.2020 → Display-P3 contraction.

The contraction paths can produce out-of-gamut components, for which the ACES-RGC-style
out-of-gamut compression is now active by default (`DisplayGamutCompression::Auto`)
so wide-gamut colors compress gracefully instead of clipping per channel.

On plain SDR targets, cameras without the component are completely unaffected, and SDR
output remains byte-identical.

## Physically based veiling glare as a bloom scatter model

`Bloom` has a new `scatter` field selecting how the blurred pyramid levels are
weighted when composited back onto the image:

- `BloomScatterModel::Aesthetic` (the default) is the hand-tuned parametric
  curve Bevy's bloom has always used — existing scenes render exactly as before.
- `BloomScatterModel::Gt7Glare { f_number }` replaces the curve with per-level
  weights derived from the far-field (Fraunhofer) diffraction point-spread
  function of a camera aperture — the physically based veiling glare Polyphony
  Digital presented for Gran Turismo 7 at SIGGRAPH 2025, where glare *is* the
  bloom and the aperture F-number drives how the light spreads.

```rust
commands.spawn((
    Camera3d::default(),
    // or: Bloom { scatter: BloomScatterModel::Gt7Glare { f_number: 8.0 }, ..default() }
    Bloom::GT7_GLARE,
));
```

The per-level weights are the photopically weighted energy of the polychromatic
Airy pattern integrated over each pyramid level's annulus of blur radii,
precomputed for the standard f/1–f/22 full-stop ladder and interpolated per
F-stop in between. Wide apertures give a tight glare that falls off steeply
around bright sources; stopping down to f/22 spreads the energy into a wide,
soft veil, as diffraction does on a real camera. GT7's own hand-calibrated
weights are unpublished, so Bevy derives its table from the documented physical
model rather than cloning constants — see the `bloom::glare` module docs for the
derivation and references.

Because a physical point-spread function applies to *all* light, the glare model
is threshold-free: any configured `BloomPrefilter` is ignored (with a warning),
and compositing is forced to energy-conserving blending. `Bloom::intensity`
keeps its meaning as the total fraction of energy scattered out of the sharp
image, so the artistic exposure dial and the physical distribution dial stay
independent. The effect reuses the existing bloom downsample/upsample pyramid —
switching models costs nothing beyond different blend constants.

Try it in the `bloom_3d` example: `B` toggles the scatter model and `O`/`L` walk
the aperture through the F-stop ladder.
