# Zorah: UE 5.4 source data in Bevy meshlets + Solari

This example converts the Zorah 1.1.0 UE 5.4 source project without Unreal
Engine. The converter is intentionally Zorah-specific: it knows the sample's
three World Partition levels, Blueprint/static-mesh-group conventions,
uncooked `FMeshDescription` layout, material-instance families, and editor
texture payloads.

Unreal is useful for visual comparison, but it is not required by conversion
or runtime.

## Current data set

The source sample resolves to:

- 3 levels and 10,494 flattened actor records
- 648 referenced meshes
- 436,559,393 full-detail triangles in 5,422 bounded partitions
- 321 recursive source-material records and 854 source textures
- 300 runtime materials and 823 runtime textures after exact material baking
- 572 `DecalActor`s projecting 50 decal materials, which reach no mesh slot and
  so are additional to both material counts above

The manifests also retain 27 GreenHouse static-mesh components whose referenced
packages are not present in the downloadable source project. They are marked
`source-package-not-in-sample` rather than guessed or silently reassigned; the
other two levels have no unresolved static-mesh components.

Static-mesh material assignment follows each mesh's exact UE `StaticMaterials`
and `SectionInfoMap` records. A partition records both its LOD0 section index
and the material slot that section resolves to, because a component's
`OverrideMaterials` array is indexed by material slot while sections are not.
Inherited Blueprint meshes are resolved from the exact generated component
template. The converter contains no filename,
directory-distance, or approximate-name material/mesh fallback. Four material
packages referenced by the authored project are absent from the download; their
original object identities are retained and rendered with an explicit magenta
diagnostic material rather than being substituted with a similarly named asset.

Each runtime partition has two representations derived from one tangent-free
loose GLB:

- full-detail geometry converted offline to Bevy `MeshletMesh` data;
- a fixed-error LOD selected from that meshlet hierarchy and decoded into the
  compact indexed triangle buffers required for wgpu BLAS construction.

Meshlets rasterize the visible surface while Solari traces the meshlet-derived
LOD. wgpu exposes triangle BLAS/TLAS and ray queries, but no Vulkan
cluster/partitioned acceleration-structure API, so hardware ray tracing still
requires decoded position/index buffers. The compact companion omits tangents;
Solari reconstructs its tangent frame from triangle positions and UVs at a hit.

The branch's meshlet format also packs UVs per meshlet. Runtime meshlet data is
packed whole-asset into lazily allocated 64 MiB GPU pages exposed through a
fixed 128-entry storage-buffer binding array. This avoids monolithic allocation
and binding-size limits while preserving asset-local offsets.

## Requirements

- stable Rust 1.96.1 (no nightly)
- .NET SDK 10 for the CUE4Parse manifest reader
- `sfw uv` for isolated Python dependencies
- Visual Studio 2022 Build Tools with Desktop C++ for the texture compressor
- the extracted `ZorahSample` project

`pyooz` is used only by the isolated conversion environment. It is not
vendored or linked into Bevy.

## Convert once

From the repository root:

```sh
sfw uv run --with numpy --with pyooz==0.0.8 --with pillow python \
  examples/large_scenes/zorah/convert/convert.py ../ZorahSample
```

The final output defaults to `examples/large_scenes/zorah/assets`. Pass a
second positional path to choose another destination. Conversion is always
incremental: the adjacent `.assets.convert` work directory is a persistent
cache, and an existing final tree is replaced atomically only after its changed
domains verify. The work cache is retained after success for later
converter/runtime iteration; pass `--discard-work` only when that cache is no
longer useful.

The normal command is also the update command. It fingerprints loose geometry,
runtime materials/textures, scenes, and relevant pack settings independently.
An unchanged run performs only fast manifest/bundle validation. A
material/texture change hard-links the already-verified geometry bundles into
the staging tree, reuses unchanged packed texture entries, and compresses only
changed textures. Meshlet generation and the Solari capacity audit rerun only
when geometry changes. Under WSL, source inspection and texture/material
conversion use WSL-native `.NET` with WSL paths. A Windows output path
automatically selects native Windows Cargo for the Windows packer output.

After the first complete conversion, light/actor export changes do not rebuild
geometry, meshlets, textures, or bundles:

```sh
sfw uv run --with numpy --with pyooz==0.0.8 --with pillow python \
  examples/large_scenes/zorah/convert/convert.py ../ZorahSample --scenes-only
```

Use `--refresh-source` when the fixed UE source files changed, and
`--rebuild-geometry` only when geometry extraction or partitioning must be
redone.

Useful tuning arguments are:

- `--triangles 250000`
- `--raytracing-error 0.02` (metres)
- `--max-texture-size 8192`
- `--material-bake-size 8192`
- `--texture-jobs 16` (defaults to min(16, host cores); the block compressors
  are single-threaded per texture)
- `--geometry-jobs 24` (0, the default, sizes from the packer host: cores
  minus four, between 4 and 24; meshlet building is serial per partition, so
  this is the packer's main throughput knob)
- `--geometry-shard-gib 0.5`
- `--texture-shard-gib 2`

Packer and capacity-audit builds use their own `target/packer` directory with
`-C target-cpu=native` Rust codegen, so they never invalidate interactive
`cargo run` artifacts (and vice versa). The bundled C compressors keep their
own flags: they build scalar/SSE/AVX variants with runtime dispatch. The first
conversion after this change compiles that directory once.

The entry point performs these phases:

1. build the CUE4Parse reader and flatten all three levels;
2. stream uncooked mesh descriptions, retaining position/normal/UV seams, and
   produce bounded tangent-free geometry;
3. resolve recursive materials and export editor texture sources with valid
   4×4 block-aligned extents;
4. verify changed loose domains (without reopening unchanged GLB payloads);
5. run the stable-Rust `zorah_pack` tool, reusing verified geometry and packed
   textures where possible; changed geometry creates meshlets and the
   fixed-error Solari LOD, while changed textures become mipmapped KTX2 data;
6. verify every labeled bundle reference, enforce at most 50 physical files in
   the final tree, and run the full Solari capacity audit only for new geometry.

Geometry uses smaller shards than textures because every geometry bundle is
expanded into CPU assets and scheduled for GPU upload as one unit. Existing
runtime assets can be quickly re-sharded without re-extracting or rebuilding
meshlets:

```sh
sfw uv run python examples/large_scenes/zorah/convert/convert.py \
  ../ZorahSample --reshard-existing
```

The loose thousands-file tree exists only inside
`.assets.convert/loose`, outside Bevy's asset root. The final `assets/`
directory contains the manifests plus approximately 512 MiB geometry and 2 GiB
texture `.zorah_bundle` shards. It does not contain thousands of GLBs, PNGs,
metadata files, or a pre-populated `imported_assets` cache.

## Why the Solari companion exists

`MeshletMesh` uses variable-bit positions and meshlet-local `u8` indices, while
the current hardware BLAS API accepts fixed-stride positions and conventional
indices. The packer therefore selects a complete LOD from the meshlet BVH and
decodes only that LOD to `POSITION/NORMAL/UV_0` plus `u32` indices. This is a
32-byte vertex rather than the previous 48-byte tangent-bearing vertex. The
raster meshlet asset remains the authoritative full-detail representation.

## UDIM texture atlases

Zorah authors many materials as UDIM tile sets. The converter assembles each set
into a single texture and records the grid it used as `source_grid_columns` and
`source_grid_rows`. Block `(bx, by)` is pasted at pixel
`(bx * tile_width, (rows - 1 - by) * tile_height)`, so UDIM block row 0 is the
bottom row of the image. That matches UE's V flip, where UDIM row `k` covers
mesh `v` in `[-k, -k + 1]`.

The runtime derives a material's `uv_transform` from that grid alone: scale
`(1/columns, 1/rows)` and translation `(0, (rows - 1)/rows)`, so
`u_atlas = u/columns` and `v_atlas = (v + rows - 1)/rows`. A 1x1 grid stays
`Affine2::IDENTITY`. Nothing consults per-material UV bounds, so a material
occupying only columns 2-4 of a five-column atlas keeps those columns instead of
collapsing onto column 0.

Two consequences are logged rather than silently absorbed. A `StandardMaterial`
carries one `uv_transform` for every map, so when a material's base color,
normal, and ORM textures disagree on grid size the base color grid addresses all
three and the mismatch is named. UVs reaching outside `[0, columns]` by
`[-(rows - 1), 1]` are listed as well; they still render, wrapped by the
`Repeat` sampler onto another tile of the same atlas.

Not every grid cell has an authored tile; 23 of the exported atlases have gaps.
Those cells are filled with neutral values: `(128, 128, 128, 255)` for color,
`(255, 128, 0, 255)` for ORM (occlusion 1, roughness 0.5, metallic 0),
`(128, 128, 255, 255)` for normal maps, and 255 for single-channel masks. The
fill is load-bearing twice over. A zero-filled cell is a black albedo with zero
occlusion, roughness, and metallic, which shades to a black mirror; and mips are
generated across the whole atlas with no tile awareness, so every coarse level
of a holed atlas is pulled toward whatever the gaps hold.

## Runtime asset flow

The example loads unprocessed assets. The converter already emits
runtime-ready bundles, so the asset processor had nothing to do except read and
blake3-hash every multi-GiB shard on every launch before deciding nothing
changed, plus write a full duplicate tree into `imported_assets` on the first
run. The example reads the JSON manifests directly and asks the asset server
only for the bundle shards the selected level needs; the example-local
`.zorah_bundle` loader publishes their meshes, meshlets, and images as labeled
subassets. An `imported_assets` directory left by an earlier run is unused and
can be deleted.

Startup is state-driven and bounded:

1. Only texture bundles referenced by the selected level's effective
   materials are loaded, sequentially, before materials are built. Once the
   materials hold direct image handles, the bundle-root handles are released
   so images no material references are freed before geometry loads. Bundle
   images are render-world only; their CPU copies are freed after upload. A
   bundle that fails to load is recorded and its materials are built
   untextured, so the affected geometry still renders instead of waiting
   forever on images that will never exist.
2. Geometry bundles are loaded one at a time. Asset requests, unique geometry
   residency, and raster instance spawning have per-frame budgets.
3. Solari queues a BLAS build for every compatible `Mesh` asset as it loads,
   so BLAS work overlaps step 2 rather than starting after it. Attaching the
   ray-tracing components in bounded batches stages TLAS instance registration;
   build pacing comes from Solari's own per-frame budgets.
4. The example reports the actual queued, allocator-waiting, available,
   compacted, and failed counters while it waits for every expected unique
   BLAS to settle. The older vertex/frame estimate is retained only as a
   diagnostic timeout and never enables Solari early. `SolariLighting` is
   added only after measured readiness.

This keeps the window responsive and prevents the old all-at-once tangent,
meshlet upload, BLAS, TLAS, and Solari workload burst.

Because BLAS building is driven by mesh assets rather than by the staged
component attach, `--raster-only` omits Solari's plugins entirely; that, not
withholding `RaytracingMesh3d`, is what avoids the level's BLAS set. An adapter
without ray-query support is detected the same way and reported once, after
which the example keeps running as meshlet raster instead of waiting on
readiness counters nothing will ever advance.

## Run on Windows Vulkan

```powershell
$env:WGPU_BACKEND = "vulkan"
cargo +1.96.1 run --release -p zorah
```

Choose another level with:

```powershell
cargo +1.96.1 run --release -p zorah -- --scene scenes/Restir_Level.json
```

Each included level has a deterministic starting view. Override it for focused
debugging with Bevy-space coordinates:

```powershell
cargo +1.96.1 run --release -p zorah -- --camera-position 0,4.5,48 --camera-target 0,4,40
```

The target machine is an RTX 5090. DLSS frame generation can be integrated
later without changing the conversion format.

## Advanced phase debugging

The normal entry point is preferred. Its component scripts remain usable for
focused diagnosis:

- `convert_geometry.py` extracts and partitions Zorah meshes.
- `mesh_description.py` inspects or extracts the known uncooked mesh schema.
- `texture_source.py` reconstructs and resizes editor texture sources.
- `verify_conversion.py` validates the loose intermediate.
- `verify_bundles.py` validates the final runtime tree.
- `zorah_pack` is the Rust offline meshlet/texture/bundle builder.

The converter auto-detects native or WSL-hosted Windows `dotnet`. Use
`--dotnet` to select another executable or `--skip-dotnet-build` when the
reader is already built.

## UE-specific gaps

Implemented source conventions include World Partition actors, Zorah grass,
grass-edge, candle, bench, static-mesh-group and throne-platform Blueprints,
local substitutes for the referenced `/Engine/BasicShapes`, recursive
material parameters, and common Base Color/Normal/ORM mappings.

This is not a general UE material or Blueprint interpreter. Complex material
graphs, layer blending, WPO/wind, subsurface profiles, and water/translucency
remain approximations or gaps.

Foliage cutouts are converted but not switched on. `M_LS_Foliage` takes its
opacity mask either from the base color texture's alpha or from a separate
mask texture, chosen by one static switch; eleven instances pick the separate
texture, and the bake composites it into the alpha channel of the baked base
color, where a `StandardMaterial` alpha-tests it against
`OpacityMaskClipValue`. Instances on the base-color-alpha side keep whatever
alpha their albedo carries, which in this sample is opaque everywhere -
including `MI_Tree_A1_Atlas` and the two `MI_Flowers_Atlas_A0*` materials,
whose "atlas" is UV packing over fully modelled geometry rather than
billboards.

`--preserve-alpha` turns the masks on. Non-opaque partitions then spawn as
`Mesh3d` instead of `MeshletMesh3d`, because meshlets rasterize into the
visibility buffer before any material runs and
`meshlet/material_pipeline_prepare.rs` collects only opaque materials, so a
masked meshlet would write depth and never be shaded. Both paths draw the
plain `Mesh` the BLAS already loads, so the switch costs no extra conversion
and no extra GPU memory. It moves 1,365 of GreenHouse's 8,832 partitions, 392
of Restir's 27,394, and 31 of Throne Room's 23,190.

It stays off by default because Solari would still trace those instances
opaque and disagree with the raster image. Making it the default needs, in
`bevy_solari`: an alpha (and `OpacityMaskClipValue`) field on `GpuMaterial` in
`scene/binder.rs`, a `rayQueryProceed` candidate loop in
`raytracing_scene_bindings.wgsl` that samples base color alpha and rejects
clipped hits, and dropping the hardcoded
`AccelerationStructureGeometryFlags::OPAQUE` in `scene/blas.rs`.

The sample's one water surface, the GreenHouse reflecting pond, is
`MSM_SingleLayerWater`. Its own parameter family is mapped to the four terms
Bevy has: Water Base Color, Water Roughness, Water Specular, and the first
authored wave normal with its tiling and pan speed, which the runtime advances
through `uv_transform` every frame so both meshlet raster and Solari see the
motion. The volume half of that shading model has nowhere to land - neither
`StandardMaterial` nor Solari's `GpuMaterial` carries IOR, transmission or
thickness, and `brdf.wgsl` has no transmission lobe - so the pond renders as a
dark reflective surface rather than one the bed shows through, and its
absorption tint and caustics are dropped. UE also layers three wave normals
over world-space UVs; one normal map and one `uv_transform` leave room for the
first, applied to UV0.

Every `DecalActor` becomes a `ClusteredDecal`: 232 in GreenHouse, 323 in
Restir, 17 in Throne Room. Each one serializes a material and a transform and
nothing else, so all 572 take `UDecalComponent`'s (128, 256, 256) cm class
default box, and none authors a `SortOrder` or a fade. UE projects along the
component's X through half-extents of `DecalSize * RelativeScale3D` and reads
the image off Y and Z; `ue_world_to_bevy` already lands those on the depth and
image axes of Bevy's unit cube. The three `M_LS_Decal_*_VT` parents pack the
surface into one `ROT` texture, green the coverage and red the roughness, so
the bake moves green into the base color's alpha - which is where
`apply_decals` reads a decal's coverage - and restates red as glTF ORM.
`MI_Wetness_Puddles_Decal_A1` binds no base color texture at all, so its four
Restir instances are skipped rather than projected as a no-op.

Decals reach the raster image only. `apply_decals` runs in `pbr.wgsl`, which
the meshlet material pass shares, so meshlet geometry receives them, but Solari
traces the surface underneath unchanged. Three authored terms have nowhere to
land: the `Opacity Angle Fade` switch that 42 of the 53 decal materials tick,
because `clustered_decal_iterator_next` tests only box containment; the decal
normal maps, because `apply_decals` blends a tangent-space vector straight into
the world-space `N` and offers no flip for UE's DirectX-format maps; and
`SortOrder`, because a cluster's decals are visited in an unspecified order.

The scene manifest preserves UE directional, point, spot, and sky-light
components, including physical units, colors, temperature, source dimensions,
cone/source angles, range, transforms, and IES/light-function references.
Directional lights feed Solari directly. Point and spot lights use Bevy's
analytical lights for the responsive raster warm-up and two shared emissive
proxy meshes for Solari. Intensities are restated in each consumer's
convention: UE derives spot candela from the cone solid angle while Bevy
divides `SpotLight::intensity` by 4pi, so a spot authored in
`ELightUnits::Lumens` is rescaled for the raster light and left as authored
total flux for its emissive proxy. The 6,421 Throne Room candle actors share one tiny
1,800 K flame mesh/material, positioned with the same top-of-static-mesh rule
used by Zorah's Niagara Blueprint, so they add one BLAS rather than thousands.
The two `BP_HoodLight` spotlight templates are recovered from their Blueprint
class package because their World Partition instance contains no component
exports.
Source emissive intensity/color/mask parameters are also mapped, and the
runtime marks emission explicitly from Zorah's authored switches and custom
emission graphs. Solari exposes large emissive instances as stable logical
chunks of at most 65,535 triangles while all chunks continue to reference the
same mesh and BLAS.

Remaining lighting approximations are explicit: Solari does not yet bind
analytical point/spot lights, so their emissive proxies have Lambertian rather
than exact UE angular profiles. UE IES profiles and procedural light-function
materials retain uniform-flux proxies but the profiles themselves are not
evaluated. SkyLight real-time/environment capture is a raster ambient
approximation and is not yet a traced environment light; Niagara flame
animation/flicker is not reproduced. UE light-blocker meshes are retained as
ray-tracing occluders but intentionally omitted from camera-visible meshlet
rasterization. The 225 components UE excludes from the shadow pass (1
GreenHouse, 216 Restir, 8 Throne Room) become `NotShadowCaster`, which meshlet
raster honors; Solari's TLAS carries every instance, so they still occlude
there.
