---
title: "`Bloom` and `BloomPrefilter` have new fields"
pull_requests: []
---

`Bloom` gained a `scatter` field and `BloomPrefilter` gained a `threshold_nits` field.
If you build these structs with `..default()` or a preset like `Bloom::NATURAL`,
nothing changes. If you write out every field, add the new ones:

```rust
// 0.19
Bloom {
    // ...
    prefilter: BloomPrefilter {
        threshold: 0.6,
        threshold_softness: 0.2,
    },
    // ...
}

// 0.20
Bloom {
    // ...
    prefilter: BloomPrefilter {
        threshold: 0.6,
        threshold_nits: None,
        threshold_softness: 0.2,
    },
    // ...
    scatter: BloomScatterModel::Aesthetic,
}
```

- `scatter` picks how bloom spreads light. The default,
  `BloomScatterModel::Aesthetic`, is the curve bloom has always used, so existing
  scenes look the same. The new `BloomScatterModel::Gt7Glare` variant is a physically
  based alternative; see the "Gran Turismo 7 tone mapping and physically based glare"
  release note.
- `threshold_nits` sets the bloom cutoff as a brightness in nits instead of the
  unitless `threshold`. Leave it `None` to keep the old behavior; when set, it
  overrides `threshold`.
