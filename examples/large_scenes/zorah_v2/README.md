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
  every part (primitive and partition index, material, triangle and vertex
  counts, locked vertices, AABB, whether its winding was repaired) and the
  settings stamp the directory was baked under.

The cache defaults to `<scene root>/.bevy_zorah_cache`; `--cache-dir` moves
it. Meshes sharing a `.mesh.bin` (82 do) are baked once.

The manifest is the checkpoint: an interrupted run resumes at mesh
granularity, redoing only the meshes that were in flight. A manifest whose
stamp disagrees with the current `--quantization`, `--partition-triangles`,
meshlet asset version or bake pipeline version is rebaked. To reset, delete
the cache directory (or one mesh's subdirectory). Older caches may also hold
`.zblas` files, which nothing reads.

Primitives above `--partition-triangles` are cut into spatial partitions by
median bisection on the longest axis of their triangle centroids. Every part
- each partition, and each primitive of a multi-primitive mesh (the export's
UDIM tiles) - builds its own LOD chain, so a vertex two parts share would
drift apart as they simplify and open the seam. The bake therefore locks
exactly the positions that occur in more than one part of the mesh: the
partition cuts and the edges tiles share, matched on the meshlet build's own
quantization grid since the export's tiles agree only to within a few ulps.
Nothing else is locked; an edge that is open in the source mesh meets nothing
and simplifies freely. The manifest records each part's `locked_vertices`.

Work is scheduled per part: `--bake-workers` threads take parts from a
queue, and a worker that finds it empty opens the next mesh - decodes it,
partitions it, finds its seams - and queues its parts, largest meshes first,
so a 32 M-triangle mesh spreads over every worker instead of holding one for
an hour. At most `workers` decoded meshes are held at once; the largest
here (32 M triangles across 12 primitives) is about 1 GiB decoded, and a part
under construction adds its meshlet build on top. The default worker count
is `clamp((total RAM - 8 GiB) / 2 GiB, 1, cores)`. Each part's
simplification fans out over bevy's `AsyncComputeTaskPool`, which the app sizes to a
thread per core for exactly this (the default pool would cap it at four
threads and every worker would queue behind them), but a part's clustering,
partitioning and BVH build run on its worker, so fewer workers than cores
leaves cores idle: sixteen on thirty-two measured seventeen busy. Lower
`--bake-workers` if the machine runs short of memory. The `mimalloc`
feature (on by default) replaces the system allocator, which on Windows
serialises the bake's threads. Progress is logged every 5 s; a mesh whose
part fails (or panics inside the meshlet builder) is reported and skipped
rather than aborting the bake, and the rest of its parts still finish.

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
8 GiB budget, as a warning when over. The budget is an upper bound on what
fits, not a guarantee: the manager packs each part best-fit into one 64 MiB
page and a part cannot straddle two, so the slack at the end of every page
is lost and a total near the budget can still leave its last parts
unrendered, and a part that prunes to more than a page is rejected however
empty the pages are (each such part is warned about as it loads, and the
residency line counts them). Choosing values:

- `--raster-error` (default 4 mm) is the finest detail the raster image
  can show, however close the camera gets. Instancing costs nothing extra,
  since the bytes are per mesh. Take the smallest value whose report line
  (below) fits the budget; the ladder flattens towards the coarsest LODs,
  which the bound cannot prune, so measure the cache rather than assume.
- `--raytracing-error` (default 5 cm) is how far the surface Solari traces
  against may sit from the rasterized one. Solari biases its rays by the
  achieved error, so a larger value shows as light leaking at contact
  points; a smaller one costs BLAS memory and build time. Values below
  `--raster-error` are raised to it.

`--report-lod-budget` measures a cache without opening a window or touching
the GPU: it decodes every part of every complete manifest on
`--bake-workers` threads, prunes it at each of 0, 1, 2, 4, 8, 16 and 32 mm
and cuts it at each of 2, 5, 10 and 20 cm, and logs one line per rung with
the resident bytes and triangles across LODs, the largest single part and
the number of parts over a page, the BLAS triangles and an
input-geometry estimate (32 B per vertex + 4 B per index, the buffers Solari
reads; the acceleration structure the driver builds is extra), the page
budget line and the cache coverage (geometry files and root meshes with a
complete manifest, and the source triangles they hold). It reads a cache
that is still being written: directories without a manifest, manifests
baked under other `--quantization` or `--partition-triangles` settings (a
run would rebake them), and parts that fail to open are counted and
skipped. A scene with no cache directory yet is reported as nothing to
measure, and none is created.

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
  supports it). Until then the meshlet raster preview is lit by the
  camera's `EnvironmentMapLight`.
- Placements whose world matrix carries shear (a rotated child under a
  non-uniformly scaled parent; the palm trees here) are spawned as a chain of
  parent entities mirroring their glTF nodes, since one `Transform` cannot
  hold shear; everything else is a single flattened `Transform`.
- `--bake-only` skips the app entirely: the bake runs on the calling thread's
  watch with no window and the process exits non-zero if a mesh failed.

## Lighting caveat

The export has no lights. RTXMG, the reference renderer, lights it with the
sidecar's equirectangular HDR alone (`settings.envmap` in
`<name>.scene.json`, with its `envmap rotation` and `envmap intensity`)
and lets sky light into the throne room through its stained glass as
thin-wall transmission, which bevy_solari has no notion of. By default
transmissive materials are left out of the BLAS so the sky reaches the
throne room through its windows; `--emissive-boost` and `--fire-lumens`
remain stand-ins for the interiors.

The HDR is converted to a cubemap at start-up and put on the camera as an
`EnvironmentMapLight`, which bevy_solari importance-samples, and as a
`Skybox` behind the geometry Solari's primary rays miss. Its values are in
arbitrary units, so its intensity is normalised so that the map's sky (sun
texels excluded) averages the radiance of a 15000 lux uniform sky;
`--envmap-intensity` multiplies that. The map is rotated by the sidecar's
`envmap rotation` minus 180 degrees, since RTXMG turns its lookup direction
by minus that angle and the converter's seam is half a turn from RTXMG's.
The analytic `--sun-illuminance` sun stays on by default: the map's sun
disc is a few texels and Solari resolves it poorly; `--sun-illuminance 0`
lights from the map alone. A map that fails to load or convert leaves the
sun as the only light.

The export has no exposure data either; auto exposure uses the reference
renderer's metering (the mean of the 80th..95th luminance percentiles
scaled to -0.5 EV before ACES), and `--exposure-bias` adds to it.

## Flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--scene-root <dir or .gltf>` | `ZORAH_ROOT` | The export. |
| `--cache-dir <dir>` | `<scene root>/.bevy_zorah_cache` | Bake output. |
| `--bake-workers N` | `clamp((RAM - 8 GiB) / 2 GiB, 1, cores)` | Threads building bake parts, or report threads. |
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
| `--exposure-bias ev` | `0` | Exposure compensation on top of the reference's, positive brighter. |
| `--hide-nodes substr` | none | Skip nodes whose name contains the substring; repeatable. |
| `--preserve-alpha` | off | Keep MASK/BLEND materials as alpha-tested `Mesh3d` (meshlets cannot alpha-test). |
| `--double-sided-all` | off | Render everything double-sided like the reference renderer. |
| `--gltf-specular` | off | Honour `KHR_materials_specular`. Off because the export's 0.498 is UE's default Specular 0.5 written literally, and taking it as glTF would halve F0. |
| `--glass-in-blas` | off | Include transmissive materials in the BLAS. |
| `--envmap-intensity k` | `1` | Multiplies the environment map over its normalisation (sky mean = a 15000 lux uniform sky). |
| `--sun-illuminance lux` | `100000` | Directional sun from the .cfg direction (0.6, 0.7, 0.36) and colour (1.0, 0.8, 0.5); `0` = none. |
| `--emissive-boost k` | `4` | Multiplies every emissive material (lamp glass, coals). |
| `--fire-lumens lm` | `800` | Emissive proxy sphere at every `FirePot`/`FireGrate`/`Firewood_Coal` node; `0` = none. |
| `--clay` | off | Flat grey clay everywhere, lighting only. |
| `--solari-albedo` | off | Surfaces emit their base colour and reflect nothing. |

## Keys

- `P` logs the camera as a `--camera-position ... --camera-target ...` fragment.
- `F12` writes `zorah_v2-<unix time>.png`.

The camera is a `FreeCamera` (walk 3 m/s, run 20 m/s).
