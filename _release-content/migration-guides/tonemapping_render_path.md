---
title: "Camera stacks tonemap once"
pull_requests: []
---

Stacked cameras rendering to the same target now compose in scene-linear space and
tonemap once, in the last camera's tonemapping pass. Earlier cameras' `Tonemapping`,
`ColorGrading`, `GranTurismo7Params`, and `DebandDither` settings do not run, and Bevy
warns when their tonemapping methods differ. Set those components on the last camera of
the stack.

Adding `GranTurismo7Params` to a camera moves its tonemapping from the material shaders
to the tonemapping pass. This costs an `Rgba16Float` intermediate and one fullscreen
pass. Transparents blend in scene-linear before tonemapping instead of after it, so
scenes with transparency can look slightly different. `TonemappingPass` is a required
component of `GranTurismo7Params` and outlives it, so removing the params alone does
not restore the in-shader path; remove `TonemappingPass` too.

An HDR display target, or an SDR target with a recalibrated `DisplayTarget`, moves a
camera onto the same pass and intermediate. Budget for the fp16 main texture on such
targets.
