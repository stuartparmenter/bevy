---
title: New `Tonemapping::GranTurismo7` variant
pull_requests: []
---

The `Tonemapping` enum has a new variant, `Tonemapping::GranTurismo7`, which must
be handled during `match` statements. The default is still
`Tonemapping::TonyMcMapface`.
