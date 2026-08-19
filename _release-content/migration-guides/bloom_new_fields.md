---
title: "`Bloom` and `BloomPrefilter` have new fields"
pull_requests: []
---

`Bloom` has a new `scatter: BloomScatterModel` field, and `BloomPrefilter` has a
new `threshold_nits: Option<f32>` field. Their defaults,
`BloomScatterModel::Aesthetic` and `None`, keep the previous behavior, so
`..default()` and presets like `Bloom::NATURAL` need no change. If you write out
every field, add both:

```rust
// 0.20
Bloom {
    prefilter: BloomPrefilter {
        threshold: 0.6,
        threshold_nits: None,
        threshold_softness: 0.2,
    },
    scatter: BloomScatterModel::Aesthetic,
    // ...
}
```
