---
title: "`RenderPlugin` has a new `working_color_space` field"
pull_requests: []
---

`RenderPlugin` has a new `working_color_space: WorkingColorSpace` field. Add `working_color_space: WorkingColorSpace::Rec709` where you construct it, or use `..default()`.
