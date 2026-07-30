---
title: "`Image` has a new `source_primaries` field"
pull_requests: []
---

`Image` (in `bevy_image`, available as `bevy::image::Image`) has a new public field:

```rust
pub source_primaries: SourceColorPrimaries,
```

It records which color primaries (gamut) the image data is expressed in — `Bt709`
(the sRGB primaries, and the default), `Bt2020`, or `DisplayP3`. This is metadata
only: it does not change how any image is decoded, stored, or rendered.

Struct-literal construction must now provide the field; `SourceColorPrimaries::default()`
(BT.709) preserves the previous behavior:

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

All of `Image`'s constructors (`Image::new`, `Image::new_fill`, `Image::default`,
`Image::from_buffer`, `Image::from_dynamic`, ...) initialize the field to
`SourceColorPrimaries::Bt709`, so code using them is unaffected.

`ImageLoaderSettings`, `HdrTextureLoaderSettings`, and `ExrTextureLoaderSettings`
gained an optional `source_primaries: Option<SourceColorPrimaries>` setting
(default `None`; existing `.meta` files keep working). Resolution order is
setting > file metadata > BT.709. With `None`, the loaders honor per-format file
metadata: the KTX2 data format descriptor's `colorPrimaries`, Radiance HDR
`PRIMARIES=` header lines, and the OpenEXR `chromaticities` attribute. The KTX2
loader also logs a one-time warning when the file's declared transfer function
contradicts the loader's `is_srgb` setting (the setting still wins, byte-for-byte
identical) or declares an HDR transfer function (PQ/HLG, still loaded as-is). The
glTF loader stamps `Bt709` explicitly, as mandated by the glTF 2.0 specification.

The stamp is propagated to `GpuImage::source_primaries` in the render world for use
by the configurable wide working color space that also ships this release
(`RenderPlugin::working_color_space` — see the "Wide-gamut color" release
note).
