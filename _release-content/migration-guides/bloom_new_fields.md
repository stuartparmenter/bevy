---
title: "`Bloom` and `BloomPrefilter` have new fields"
pull_requests: []
---

`Bloom` gained a `scatter: BloomScatterModel` field, and its nested
`BloomPrefilter` gained a `threshold_nits: Option<f32>` field. `..default()`,
functional update (`..Bloom::NATURAL`), and presets are unaffected. Exhaustive
struct literals need both:

```rust
// 0.19
Bloom {
    intensity: 0.2,
    low_frequency_boost: 0.7,
    low_frequency_boost_curvature: 0.95,
    high_pass_frequency: 1.0,
    prefilter: BloomPrefilter {
        threshold: 0.6,
        threshold_softness: 0.2,
    },
    composite_mode: BloomCompositeMode::EnergyConserving,
    max_mip_dimension: 512,
    scale: Vec2::ONE,
}

// 0.20
Bloom {
    intensity: 0.2,
    low_frequency_boost: 0.7,
    low_frequency_boost_curvature: 0.95,
    high_pass_frequency: 1.0,
    prefilter: BloomPrefilter {
        threshold: 0.6,
        threshold_nits: None,
        threshold_softness: 0.2,
    },
    composite_mode: BloomCompositeMode::EnergyConserving,
    max_mip_dimension: 512,
    scale: Vec2::ONE,
    scatter: BloomScatterModel::Aesthetic,
}
```

- `scatter` selects how bloom spreads light. `BloomScatterModel::Aesthetic` is
  the existing parametric curve and the default, and every preset (`NATURAL`,
  `ANAMORPHIC`, `OLD_SCHOOL`, `SCREEN_BLUR`) uses it. For the new
  `BloomScatterModel::Gt7Glare { f_number }` variant, see the "Gran Turismo 7
  tone mapping and physically based glare" release note.
- `threshold_nits` expresses the bloom cutoff as a physical luminance in nits
  (default `None`). When set, it overrides `threshold` and is divided by
  `DisplayTarget::paper_white_nits` (100 for SDR targets).
