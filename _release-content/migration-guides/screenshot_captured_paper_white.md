---
title: "`ScreenshotCaptured` has a new `paper_white_nits` field"
pull_requests: []
---

`ScreenshotCaptured` has a new `paper_white_nits: Option<f32>` field. Add `paper_white_nits: None` where you construct it.

`CapturedScreenshots` now holds a `Receiver<ScreenshotCaptured>` instead of a `Receiver<(Entity, Image)>`. Read `entity` and `image` from the received `ScreenshotCaptured`.
