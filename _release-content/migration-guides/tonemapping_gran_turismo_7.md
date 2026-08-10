---
title: New `Tonemapping::GranTurismo7` and `Tonemapping::Linear` variants
pull_requests: []
---

The `Tonemapping` enum has two new variants, `Tonemapping::GranTurismo7` and
`Tonemapping::Linear`, which must be handled during `match` statements. The
default is still `Tonemapping::TonyMcMapface`.
