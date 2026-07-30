---
title: "`AutoExposure` has new `metering_bias` and `physiological` fields"
pull_requests: []
---

The `AutoExposure` component gained two new public fields, `metering_bias: f32` and `physiological: Option<PhysiologicalAdaptation>`, defaulting to a no-op (`0.0` and `None`) — see the "Physiological auto exposure and automatic white balance" release note. `..default()` and `AutoExposure::default()` users need no change; full struct literals must add both fields:

```rust
// 0.19
AutoExposure {
    range: -8.0..=8.0,
    filter: 0.10..=0.90,
    speed_brighten: 3.0,
    speed_darken: 1.0,
    exponential_transition_distance: 1.5,
    metering_mask: default(),
    compensation_curve: default(),
};

// 0.20
AutoExposure {
    range: -8.0..=8.0,
    filter: 0.10..=0.90,
    speed_brighten: 3.0,
    speed_darken: 1.0,
    exponential_transition_distance: 1.5,
    metering_mask: default(),
    compensation_curve: default(),
    metering_bias: 0.0,
    physiological: None,
};
```
