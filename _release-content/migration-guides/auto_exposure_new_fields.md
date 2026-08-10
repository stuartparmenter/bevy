---
title: "`AutoExposure` has new `metering_bias` and `physiological` fields"
pull_requests: []
---

`AutoExposure` has two new public fields, `metering_bias: f32` and `physiological: Option<PhysiologicalAdaptation>`. Their defaults, `0.0` and `None`, keep the previous behavior, so `..default()` and `AutoExposure::default()` need no change. If you write out every field, add `metering_bias: 0.0` and `physiological: None`.
