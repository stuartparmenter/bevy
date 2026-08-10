---
title: "Anti-aliasing and bloom are now correct on HDR display targets"
authors: ["@stuartparmenter"]
pull_requests: []
---

On an HDR display target the tone-mapped image is no longer confined to
`[0, 1]`: highlights reach `peak_luminance_nits / paper_white_nits`, 10.0 on a
1000-nit display at 100-nit paper white (see the release note "HDR display
output (scRGB-linear and HDR10/PQ)"). Several post-process effects assumed that
range and are now HDR-aware.
Cameras presenting to SDR display targets render exactly as before.

Contrast adaptive sharpening (CAS) assumed the image never went brighter than
paper white, producing fireflies and inverted sharpening around highlights. On
HDR-target views those artifacts are gone.

FXAA and SMAA detect edges with luma thresholds calibrated for `[0, 1]`. On
HDR-target views the edge-detection luma is saturated to `[0, 1]`, so bright
edges are detected at their paper-white-clamped contrast.

Bloom has two HDR changes:

- On HDR-target views the bloom pyramid is `Rgba16Float` instead of
  `Rg11b10Ufloat`. This removes visible banding above 1.0, at twice the memory
  cost.
- `BloomPrefilter` has a new `threshold_nits: Option<f32>` field. The existing
  `threshold` is a raw framebuffer value whose physical meaning rescales with
  paper white. `threshold_nits` is an absolute luminance. See the migration
  guide "`Bloom` and `BloomPrefilter` have new fields".

```rust
Bloom {
    prefilter: BloomPrefilter {
        // Bloom only above 250 nits, on any display calibration.
        threshold_nits: Some(250.0),
        threshold_softness: 0.2,
        ..default()
    },
    composite_mode: BloomCompositeMode::Additive,
    ..Bloom::OLD_SCHOOL
}
```
