# Zero-Day Example

Beeple's "Zero-Day" sci-fi corridor (NVIDIA ORCA), **path-traced with Bevy Solari**
and wired to the HDR output pipeline (GT7 tonemapping + the shared `HdrPlugin`, like
the `solari` and `neon_sign` examples).

Zero-Day is authored to be lit entirely by ~10,000 emissive triangles with no punctual
lights — the way NVIDIA's original real-time ["Measure 1"](https://www.youtube.com/watch?v=0WE7CgJMuVc)
demo renders it. That needs a path tracer, so this example requires Solari: the
emissive meshes become real area lights with global illumination. It plays the film's
~13.7 s take (~550 animated objects plus the original `DynamicCamera2` flythrough) and
drives the render camera from that camera.

Requires a **ray-tracing capable GPU** (Solari currently needs the Vulkan backend in
wgpu).

## Getting the scene

Download "Zero-Day" [from developer.nvidia.com](https://developer.nvidia.com/orca/beeple-zero-day).

The download ships as `MEASURE_ONE/MEASURE_ONE.fbx` plus a `tex/` folder of `.dds`
textures. Bevy can't load FBX, and Blender's FBX importer mis-reads this Octane-exported
asset's material conventions, so [`convert.py`](convert.py) rebuilds each material from
the naming/channel convention documented in the download's README, bakes the animation
into one scene-length clip, and exports a single self-contained `.glb`:

| Texture          | Channels                                        |
|:-----------------|:------------------------------------------------|
| `_BaseColor.dds` | RGB = base color (alpha = opacity, kept opaque) |
| `_Specular.dds`  | R = occlusion, **G = roughness, B = metallic**  |
| `_Normal.dds`    | DirectX normal (green flipped in the example)   |
| `_Emissive.dds`  | RGB = emissive color                            |

Convert it with the headless Blender helper (Blender 4.x/5.x), dropping the result in
this example's `assets/` folder (which is `.gitignore`d and never committed):

```console
blender --background --python-exit-code 1 --python convert.py -- \
  "MEASURE_ONE/MEASURE_ONE.fbx" "examples/large_scenes/zero_day/assets/zero_day.glb"
```

## Running

```console
cargo run -p zero_day --release
# Solari without DLSS (a ray-tracing GPU with no DLSS SDK):
cargo run -p zero_day --release --no-default-features
```

Controls:

- **C** — toggle the film flythrough vs. free-fly (WASD + mouse).
- **N** — toggle DLSS Ray Reconstruction (with the `dlss` feature).
- **B** — run a short benchmark over the flythrough (printed to the console).

```console
Options:
  --emissive   emissive multiplier for the accent panels (they are the scene's only
               lights, so they must be bright). ~150000 reads about right; lower it
               if the scene is blown out.
  --no-pulse   disable the synthetic emissive pulse (the panels breathe by default to
               evoke the film's animated lights, which weren't in the exported asset).
  --help       display usage information
```
