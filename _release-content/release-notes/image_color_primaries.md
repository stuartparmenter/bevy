---
title: Color primaries for image assets
authors: ["@stuartparmenter"]
pull_requests: [25472]
---

Wide-gamut textures, such as Display P3 photos and Rec.2020 HDR images, show the wrong colors when they are read with the sRGB primaries. Images now record the color primaries of their pixel data in the new `Image::source_color_primaries` field. The supported primaries are `Bt709`, which sRGB uses, `Bt2020` and `DisplayP3`. `Bt709` is the default. The KTX2, PNG, Radiance HDR and OpenEXR loaders read the primaries from the file's color metadata. To override them, set `source_color_primaries` in the loader settings or in a `.meta` file:

```rust
let texture: Handle<Image> = asset_server
    .load_builder()
    .with_settings(|settings: &mut ImageLoaderSettings| {
        settings.source_color_primaries = Some(SourceColorPrimaries::DisplayP3);
    })
    .load("textures/photo.png");
```

The field is metadata only. Bevy's renderer still treats every texture as Rec.709, so code that needs the primaries, such as a custom material, must convert the colors itself.
