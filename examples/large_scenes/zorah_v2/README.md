# Zorah v2: NVIDIA's glTF export in meshlets + Solari

This example renders NVIDIA's `zorah_textured_public.v1` glTF export of the
Zorah sample directly: no Unreal project, no converter, no bundles. The root
glTF (13,079 nodes, 2,812 meshes, 1,514 materials, 4,418 KTX2 textures,
3.31 G unique triangles) is parsed with the `gltf` crate at startup; each
mesh's geometry is baked into full-detail meshlet meshes on first run and
cached beside the export, then pruned to an error bound as it loads.

```
cargo run --release -p zorah_v2 -- --scene-root C:\path\to\zorah_textured_public.v1
```

`--scene-root` takes the export's directory (or its root `.gltf`); the
`ZORAH_ROOT` environment variable is the fallback. Without either the run
fails with a message pointing here.

## What it loads

- Scene 0 of the root glTF, walked with `EXT_mesh_gpu_instancing` expanded
  (7,654 instances from 1,092 instanced nodes; 659 mirrored nodes keep their
  negative scale). Units are metres, Y-up, right-handed: nothing is converted.
- Geometry through the per-mesh `meshes/<stem>.mesh.gltf` wrappers, decoded
  from `EXT_meshopt_compression` with bevy_gltf's own decode pass. The
  wrappers are never handed to the asset server, whose glTF loader would pull
  every 4k texture in as a dependency.
- Materials as `StandardMaterial`: base colour, metallic-roughness (the
  export's BC5 textures carry roughness in R and metallic in G, which
  bevy detects from the format), normal, and emissive maps, plus
  `KHR_texture_transform`. `KHR_materials_transmission` and `_ior` are read
  for the glass decision below; `KHR_materials_specular` is ignored unless
  asked for (see flags).
- Textures as KTX2 with the largest mips dropped at load (`--max-texture-size`).
- The sidecar `scene.json` view for the starting camera.

## The bake cache

Every referenced mesh is baked into `<cache>/<mesh stem>/`:

- `p<primitive>_<partition>.meshlet_mesh`, bevy's `MeshletMesh` asset at
  full detail, every LOD;
- `manifest.json`, written last through a temp file and rename. It lists
  every part (primitive, material, triangle count, AABB) and the settings
  stamp the directory was baked under.

The cache defaults to `<scene root>/.bevy_zorah_cache`; `--cache-dir` moves
it. Meshes sharing a `.mesh.bin` (82 do) are baked once.

The manifest is the checkpoint: an interrupted run resumes at mesh
granularity, redoing only the mesh that was in flight. A manifest whose
stamp disagrees with the current `--quantization`, `--partition-triangles`,
meshlet asset version or bake pipeline version is rebaked. To reset, delete
the cache directory (or one mesh's subdirectory). Caches from before the
load-time LOD cut also hold a `.zblas` BLAS cut per part; those manifests
still load, and the `.zblas` files can be deleted.

Primitives above `--partition-triangles` are cut into spatial partitions by
median bisection on the longest axis of their triangle centroids. The
partitions of one primitive, and the primitives of a multi-primitive mesh
(the export's UDIM tiles meet edge to edge), are built with locked borders
so coarser meshlet LODs cannot open cracks along the seams. The bake runs on
`--bake-workers` threads while the window stays responsive. Each worker
holds one whole decoded mesh (every attribute of every primitive, tangents
included) plus the partition being built, so the default of
`clamp((total RAM - 8 GiB) / 2 GiB, 1, cores)` assumes about 2 GiB per
worker; that is a typical figure, not a bound, and the largest meshes
(32 M triangles across 12 primitives) will hold several GiB at their peak.
Lower `--bake-workers` if the machine runs short of memory during the full
bake. Progress is logged every 5 s; a mesh that fails (or panics inside the
meshlet builder) is reported and skipped rather than aborting the bake.

A part that is present but unreadable (a truncated file) fails to load at
run time; its mesh's manifest is then deleted so the next run rebakes it.

## Residency: the load-time LOD cut

Full detail does not fit. Measured over 1,015 baked meshes, 676 M raster
triangles took 16 GiB of meshlet data (25.4 bytes per triangle across all
LODs), which projects to 78 GiB for the 3.31 G-triangle scene. The meshlet
manager holds 128 pages of 64 MiB, 8 GiB, and an upload past the last free
run fails ("the 128 meshlet data pages have no free run of N bytes") and
that part never renders; this fork streams neither meshlets nor BLASes.

So every part is cut as it loads, in memory, and the cache stays at full
detail. `MeshletMesh::pruned(--raster-error)` drops every LOD finer than
the bound; the meshlets that become the finest level are marked error-free
so the runtime rasterizes them up close instead of culling them and
leaving holes, and every coarser transition happens at the same distance
as before. The BLAS cut is selected from the same data at
`max(--raytracing-error, --raster-error)` and handed to Solari with the
error it achieved. Pruning at load rather than in the bake means one cache
serves every bound: a new `--raster-error` costs a restart, not a rebake.

Once the scene is in, the run logs the pruned meshlet bytes against the
8 GiB budget, as a warning when over. Choosing values:

- `--raster-error` (default 4 mm) is the finest detail the raster image
  can show, however close the camera gets. Instancing costs nothing extra,
  since the bytes are per mesh. Take the smallest value whose report line
  (below) fits the budget. Measured on half the cache (1,258 of 2,034
  geometry files, 824 M source triangles): 16.8 GiB at full detail,
  8.4 GiB at 1 mm, 6.2 at 2 mm, 5.1 at 4 mm, 4.5 at 8 mm, 3.9 at 32 mm.
  The ladder flattens because the coarsest LODs, which locked partition
  borders keep from simplifying further, are most of the bytes at 4 mm and
  beyond; the bound mainly trades the fine end.
- `--raytracing-error` (default 5 cm) is how far the surface Solari traces
  against may sit from the rasterized one. Solari biases its rays by the
  achieved error, so a larger value shows as light leaking at contact
  points; a smaller one costs BLAS memory and build time. Values below
  `--raster-error` are raised to it.

`--report-lod-budget` measures a cache without opening a window or touching
the GPU: it decodes every part of every complete manifest on
`--bake-workers` threads, prunes it at each of 0, 1, 2, 4, 8, 16 and 32 mm
and cuts it at each of 2, 5, 10 and 20 cm, and logs one line per rung with
the resident bytes and triangles across LODs, the BLAS triangles and an
input-geometry estimate (32 B per vertex + 4 B per index, the buffers Solari
reads; the acceleration structure the driver builds is extra), the page
budget line and the cache coverage (geometry files and root meshes with a
complete manifest, and the source triangles they hold). It reads a cache
that is still being written: directories without a manifest and parts that
fail to open are counted and skipped.

## How a run proceeds

The root glTF is parsed and its scene walked before the window opens; the
bake starts from the app's first frame and the states go `Baking` ->
`LoadingScene` -> `WarmingRaytracing` -> `Running`:

- While baking and loading, every mesh whose manifest is complete has its
  parts loaded from the `cache://` asset source (at most 32 new parts per
  frame; each is pruned and its BLAS cut decoded on the IO pool as it
  loads) and, once they are on the GPU, one entity per (instance, part) is
  spawned (at most 512 per frame): `MeshletMesh3d` normally, `Mesh3d` of the
  BLAS cut for alpha-tested materials under `--preserve-alpha`. Materials
  and their textures are created the first time a part uses them, so a
  `--limit-meshes` run only reads the textures it shows.
- When everything is spawned, `RaytracingMesh3d` is attached in batches and
  the app waits for Solari's measured BLAS readiness (`available_blas >=
  expected`, no queued builds or compactions), then inserts `SolariLighting`
  (and DLSS Ray Reconstruction when the `dlss` feature is on and the GPU
  supports it). Until then the meshlet raster preview is lit by a flat
  ambient term of the same sky colour and radiance (illuminance / pi),
  which is zeroed again when Solari takes over.
- Placements whose world matrix carries shear (a rotated child under a
  non-uniformly scaled parent; the palm trees here) are spawned as a chain of
  parent entities mirroring their glTF nodes, since one `Transform` cannot
  hold shear; everything else is a single flattened `Transform`.
- `--bake-only` skips the app entirely: the bake runs on the calling thread's
  watch with no window and the process exits non-zero if a mesh failed.

## Lighting caveat

The export has no lights. RTXMG, the reference renderer, lights it with the
sidecar's equirectangular HDR alone, which bevy_solari cannot bind yet (its
environment light is a uniform hemisphere), and lets sky light into the
throne room through its stained glass as thin-wall transmission, which
bevy_solari has no notion of either. The `--sky-*`, `--sun-illuminance`,
`--emissive-boost` and `--fire-lumens` flags are stand-ins until a textured
environment light and thin-wall transmission exist in bevy_solari. By
default transmissive materials are left out of the BLAS so the sky reaches
the throne room through its windows.

## Flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--scene-root <dir or .gltf>` | `ZORAH_ROOT` | The export. |
| `--cache-dir <dir>` | `<scene root>/.bevy_zorah_cache` | Bake output. |
| `--bake-workers N` | `clamp((RAM - 8 GiB) / 2 GiB, 1, cores)` | Concurrent mesh bakes, or report threads. |
| `--partition-triangles N` | `500000` | Cut larger primitives into spatial partitions. |
| `--raster-error m` | `0.004` | Geometric error the finest resident meshlet LOD may carry; parts are pruned to it at load. |
| `--raytracing-error m` | `0.05` | Geometric error the BLAS LOD cut may carry (never below `--raster-error`). |
| `--report-lod-budget` | off | Measure the cache at a ladder of both errors without a window, then exit. |
| `--quantization n` | `4` | Meshlet positions snap to 1/2^n cm. |
| `--max-texture-size px` | `1024` | Drop KTX2 mips above this edge; `0` keeps 4k. |
| `--limit-meshes N` | all | Dev: bake and spawn only the first N meshes by glTF index (fire proxies appear only for selected props). |
| `--bake-only` | off | Bake without opening a window, then exit (non-zero if any mesh failed). |
| `--screenshot-after s` | off | Dev: F12 this many seconds after Solari comes on, then exit. |
| `--diagnostics` | off | Print frame-time diagnostics. |
| `--camera-position x,y,z` | scene.json view | Starting camera position. |
| `--camera-target x,y,z` | scene.json view | Starting look target. |
| `--exposure-ev100 ev` | auto | Fixed exposure. |
| `--no-auto-exposure` | off | Fixed exposure at the base EV100 (Blender's default minus the bias) instead of histogram auto exposure. |
| `--exposure-bias ev` | `0` | Exposure compensation, positive brighter. |
| `--hide-nodes substr` | none | Skip nodes whose name contains the substring; repeatable. |
| `--preserve-alpha` | off | Keep MASK/BLEND materials as alpha-tested `Mesh3d` (meshlets cannot alpha-test). |
| `--double-sided-all` | off | Render everything double-sided like the reference renderer. |
| `--gltf-specular` | off | Honour `KHR_materials_specular`. Off because the export's 0.498 is UE's default Specular 0.5 written literally, and taking it as glTF would halve F0. |
| `--glass-in-blas` | off | Include transmissive materials in the BLAS. |
| `--sky-illuminance lux` | `15000` | Uniform sky light. |
| `--sky-color r,g,b` | `0.78,0.86,1.0` | Sky colour, linear. |
| `--sun-illuminance lux` | `0` | Directional sun from the .cfg direction (0.6, 0.7, 0.36) and colour (1.0, 0.8, 0.5); `0` = none. |
| `--emissive-boost k` | `4` | Multiplies every emissive material (lamp glass, coals). |
| `--fire-lumens lm` | `800` | Emissive proxy sphere at every `FirePot`/`FireGrate`/`Firewood_Coal` node; `0` = none. |
| `--clay` | off | Flat grey clay everywhere, lighting only. |
| `--solari-albedo` | off | Surfaces emit their base colour and reflect nothing. |

## Keys

- `P` logs the camera as a `--camera-position ... --camera-target ...` fragment.
- `F12` writes `zorah_v2-<unix time>.png`.

The camera is a `FreeCamera` (walk 3 m/s, run 20 m/s).
