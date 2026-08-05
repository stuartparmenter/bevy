---
title: "`Image` has a new `source_primaries` field"
pull_requests: []
---

`Image` (in `bevy_image`, available as `bevy::image::Image`) has a new public field,
`source_primaries: SourceColorPrimaries`, recording which color primaries (gamut) the
image data is expressed in: `Bt709` (the sRGB primaries, and the default), `Bt2020`,
or `DisplayP3`. It is metadata only: decoding, storage, and rendering are unchanged.
See the "Wide-gamut color" release note.

Every `Image` constructor (`Image::new`, `Image::new_fill`, `Image::default`,
`Image::from_buffer`, `Image::from_dynamic`, ...) initializes the field to
`SourceColorPrimaries::Bt709`. Struct-literal construction must add it:

```rust
// 0.19
let image = Image {
    data,
    texture_descriptor,
    // ...
};

// 0.20
let image = Image {
    data,
    texture_descriptor,
    // ...
    source_primaries: Default::default(),
};
```

`ImageLoaderSettings`, `HdrTextureLoaderSettings`, and `ExrTextureLoaderSettings`
also have a new optional `source_primaries: Option<SourceColorPrimaries>` setting.
The default `None` reads the file's metadata where present, so existing `.meta`
files keep working.
