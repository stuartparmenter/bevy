---
title: "`AutoExposure` has new `metering_bias` and `physiological` fields"
pull_requests: []
---

The `AutoExposure` component has two new public fields, `metering_bias: f32` and `physiological: Option<PhysiologicalAdaptation>`. Both default to a no-op, so `..default()` and `AutoExposure::default()` need no change. If you write out every field, add `metering_bias: 0.0` and `physiological: None` to keep the previous behavior.
