---
title: "`Image` has a new `source_primaries` field"
pull_requests: []
---

`Image` (in `bevy_image`, available as `bevy::image::Image`) has a new public field:

```rust
pub source_primaries: SourceColorPrimaries,
```

It records which color primaries (gamut) the image data is expressed in: `Bt709`
(the sRGB primaries, and the default), `Bt2020`, or `DisplayP3`. It is metadata
only: decoding, storage, and rendering are unchanged.

Struct-literal construction must now provide the field:

```rust
// 0.19
let image = Image {
    data,
    texture_descriptor,
    // ...
    copy_on_resize: false,
};

// 0.20
let image = Image {
    data,
    texture_descriptor,
    // ...
    copy_on_resize: false,
    source_primaries: Default::default(),
};
```

Every `Image` constructor (`Image::new`, `Image::new_fill`, `Image::default`,
`Image::from_buffer`, `Image::from_dynamic`, ...) initializes the field to
`SourceColorPrimaries::Bt709`.

`ImageLoaderSettings`, `HdrTextureLoaderSettings`, and `ExrTextureLoaderSettings`
gained an optional `source_primaries: Option<SourceColorPrimaries>` setting
(default `None`, so existing `.meta` files keep working). Resolution order is
setting > file metadata > BT.709. The file metadata read is the KTX2 data format
descriptor's `colorPrimaries`, Radiance HDR `PRIMARIES=` header lines, and the
OpenEXR `chromaticities` attribute. The KTX2 loader also logs a one-time warning
when the file's declared transfer function contradicts the loader's `is_srgb`
setting (the setting still wins) or declares an HDR transfer function (PQ/HLG,
still loaded as-is). The glTF loader stamps `Bt709`, as the glTF 2.0
specification requires.

The opt-in wide working color space (`RenderPlugin::working_color_space`) still
assumes every sampled texture is Rec.709.
See the "Wide-gamut color" release note.
