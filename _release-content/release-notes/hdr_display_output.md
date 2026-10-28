---
title: HDR display output
authors: ["@stuartparmenter"]
pull_requests: [25658, 25961]
---

HDR displays can show highlights much brighter than the white of a web page, such as sunlight, fire and reflections on metal. Bevy can now render HDR output to them. Request it on the window's `DisplayTarget`:

```rust
App::new().add_plugins(DefaultPlugins.set(WindowPlugin {
    primary_display_target: Some(DisplayTarget {
        hdr: true,
        ..default()
    }),
    ..default()
}));
```

Bevy picks the best HDR color space that the window surface supports: PQ (HDR10), then linear scRGB, then extended sRGB, then extended Display P3. If the surface supports none of them, or the Windows HDR setting is off, the window stays SDR. When the OS HDR setting changes while the app runs, Bevy negotiates the surface again. To choose the color space yourself, set `DisplayTarget::color_space_override`. The window's `WindowSurfaceColorSpaces` component shows the color space the surface uses and the ones it supports. Press `O` in the `tonemapping` example to cycle through them.

The `Hdr` component on a camera does not give HDR output. It only changes the format of the texture the camera renders to.

## Brightness

Paper white is the brightness of a plain white UI element. A tonemapped value of `1.0` maps to it, and highlights go above it, up to the display's peak luminance. Leave `DisplayTarget::paper_white_nits` and `DisplayTarget::peak_luminance_nits` at `None` to use what the display reports. If the display reports nothing, Bevy uses a default for the color space: a paper white of 203 nits for PQ, the OS SDR white for the other HDR color spaces, and a peak of 1000 nits.

To let players set these values, see the new `hdr_calibration` example. It shows a test pattern for the peak luminance, the paper white and the black level, and writes each value to the window's `DisplayTarget`. Each window's `ResolvedDisplayTarget` component holds the values Bevy renders with, and `WindowDisplayHdrInfo` holds what the display reports.

## Cameras and effects

A camera that renders to an HDR window gets an HDR main texture, with or without the `Hdr` component. Most tonemapping methods limit their output to SDR. On an HDR window, Bevy runs `Tonemapping::GranTurismo7` in their place and warns once. See the release note on Gran Turismo 7 tonemapping. If no camera on an HDR window tonemaps, Bevy warns, because colors above `1.0` reach the display unchanged.

FXAA, SMAA and contrast adaptive sharpening now work on values above `1.0`. On HDR windows, bloom uses a texture with more precision, so bright areas don't band. The new `BloomPrefilter::threshold_nits` field sets the bloom threshold in nits, so the cutoff stays at the same brightness when paper white changes:

```rust
Bloom {
    prefilter: BloomPrefilter {
        threshold_nits: Some(400.0),
        threshold_softness: 0.2,
        ..default()
    },
    composite_mode: BloomCompositeMode::Additive,
    ..default()
}
```

To render HDR output into an image, add its `DisplayTarget` with a `color_space_override` to the `ManualDisplayTargets` resource. The image then holds the encoded signal for that color space.

## Screenshots

A screenshot of an HDR window is decoded to linear Rec.709 in `Rgba32Float`, with `1.0` at paper white. `ScreenshotCaptured::paper_white_nits` holds that paper white. `save_to_disk` keeps the HDR range in `.exr` files, and in `.png` files, which it writes as 16-bit PNGs with the PQ transfer function. Image viewers that support HDR show them as HDR.

```rust
commands
    .spawn(Screenshot::primary_window())
    .observe(save_to_disk("screenshot.png"));
```
