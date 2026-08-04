---
title: New `Tonemapping::GranTurismo7` and `Tonemapping::Linear` variants
pull_requests: []
---

The `Tonemapping` enum has two new variants: `GranTurismo7`, a port of Polyphony
Digital's Gran Turismo 7 tone-mapping operator, and `Linear`, which runs the
tonemapping pass with no tone curve. `Tonemapping` is not `#[non_exhaustive]`,
so exhaustive `match`es on it need an arm for each, or a wildcard:

```rust
match tonemapping {
    // existing arms
    Tonemapping::GranTurismo7 => { /* ... */ }
    Tonemapping::Linear => { /* ... */ }
}
```

Existing tonemappers are unchanged, and the default is still `Tonemapping::TonyMcMapface`.
