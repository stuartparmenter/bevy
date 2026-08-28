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
- `--raytracing-error 0.02` (metres). An upper bound, not a claim: each partition
  records `blas_achieved_error`, the largest error its selected cut actually
  carries, which is zero for geometry that simplifies to itself. Solari biases
  rays off a rasterized surface by that measurement, so a flat quad no longer
  raises every other instance's bias with it.
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

Exposure follows the level's post-process volume: every level authors
histogram auto-exposure, so the camera meters the frame the way UE does
(10%-90% band, 3 EV/s up, 1 EV/s down, the volume's compensation) and clamps
the metered EV100 to the volume's range. Restir leaves UE's open range with a
-2.5 EV compensation, so its shade brightness tracks what is on screen;
GreenHouse allows 4-5 EV100 and ThroneRoom pins 8-8, which is a fixed exposure
in effect. `--no-auto-exposure` meters every level fixed at the middle of its
range, and `--exposure-ev100` fixes any level by hand.

`--ue-editor-camera` starts instead from the perspective viewport each map was
last saved with in the UE editor (the download places no PlayerStart or camera
actor, so this is the nearest thing to an authored view; `ZorahConvert inspect
Levels/<level>.umap` prints it as `ZORAH_EDITOR_VIEW`).

Press `P` while running to log the current view as exactly that fragment
(`--scene`, `--camera-position`, `--camera-target` and the active
`--data-layers`), ready to paste back for a reproducible start; `F12` saves the
frame as `zorah-<level>-<unix time>.png` in the working directory. To find out
what an on-screen object is, `--hide-actors Eaves,LightCap` skips every actor
whose label, class or mesh path contains one of the substrings.

For a lit-versus-unlit pair, take the Solari frame first, press `P`, then rerun
with the printed fragment plus `--raster-only --unlit-textures`: every surface
shows its base-colour texture with no lighting, normal maps, emission, bloom or
tone curve, so the two frames differ only by what Solari adds. Add
`--missing-materials grey` if the magenta stand-ins would distract. The
complementary view is `--clay`, which keeps Solari and every lighting input but
paints all surfaces a rough 50% grey (normal maps and emissive lights kept), so
the frame is the illumination alone; it stands on its own or pairs with the
textured Solari frame. `--solari-albedo` is the albedo view rendered by Solari's
own path rather than the raster one: every surface emits its base-colour
texture at the camera's exposure and reflects nothing, so the traced frame is
the texture values themselves. It costs an emissive light source per surface,
so it is a screenshot mode, not a way to play.

Slots UE itself cannot texture - the engine's unassigned-slot fallback and the
four vegetation materials the download omits - render magenta so they are
impossible to miss. `--missing-materials grey` renders them as UE's own
`WorldGridMaterial` fallback would instead, a rough mid grey, for screenshots.

The target machine is an RTX 5090. DLSS frame generation can be integrated
later without changing the conversion format.

### Lighting data layers

Zorah authors its lighting scenarios as World Partition data layers
(`DL_Lighting_Day`, `DL_Lighting_Night`, `DL_Lighting_Orb`,
`DL_Lighting_Candles`, ...). The scene manifest records each actor's layers and
the level's `WorldDataLayers` initial states, and the runtime starts with the
layers UE would activate. While running, digit keys `1`-`9` toggle the level's
runtime layers in manifest order and `L` prints their state to the log; a toggle
covers lights, Solari emitters, geometry and decals, while the sky atmosphere,
fog and exposure stay as chosen at launch. The authored states are: Restir
`Day` + `Day_Support` on; GreenHouse `Sunset` + `Sunset_Support` on,
`Sunset_Clouds` off; ThroneRoom `Night` + `Orb` on and `Candles` off (the
6,443 candle actors carry no `InitialRuntimeState`, which is UE's `Unloaded`),
so ThroneRoom's candles need a keypress. Start from a different set with:

```powershell
cargo +1.96.1 run --release -p zorah -- --scene scenes/ThroneRoom_Level.json --data-layers DL_Lighting_Night,DL_Lighting_Candles
```

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
cone/source angles, range, transforms, and IES/light-function references with
the light function's scale, fade distance and disabled brightness.
Directional lights feed Solari directly. Point and spot lights use Bevy's
analytical lights for the responsive raster warm-up and two shared emissive
proxy meshes for Solari. Intensities are restated in each consumer's
convention: UE derives spot candela from the cone solid angle while Bevy
divides `SpotLight::intensity` by 4pi, so a spot authored in
`ELightUnits::Lumens` is rescaled for the raster light and left as authored
total flux for its emissive proxy. A legacy `Unitless` intensity becomes lumens
over the whole sphere, so a `Unitless` spot's proxy takes cone flux off that
same candela; skipping the restatement made the emissive disk radiate exactly
4x the punctual light's on-axis intensity for any cone.
Neither renderer evaluates an IES profile or a light function, but a uniform
proxy can still carry a light function's mean, the factor UE multiplies the
light by on average. The converter measures that mean per material and exports
it; an unmeasured light function keeps its whole flux and is counted in
`unsupported_profiles`. Throne Room's "Caustics" light is the one measured case:
`LF_LIghtCaustics_01_Inst` multiplies two panning samples of the sparse
`T_Caustics_01_MSK` ripple, whose emissive averages 0.0081, so 250,000 cd that
read as 67% of the room's emitted flux now contribute 25,447 lm. The pattern
itself is still missing, so that region shows flat light where UE shows moving
filaments reaching 6x the local mean.
The two `BP_HoodLight` spotlight templates are recovered from their Blueprint
class package because their World Partition instance contains no component
exports.
Throne Room's 6,421 candles are lit by a Niagara Light Renderer rather than a
light component, so their actors export no light at all. The runtime spawns one
flame per candle when the level runs `NS_CandleFlame_04`, placed by the same
"top of static mesh" rule the Niagara Blueprint uses: the horizontal centre of
the candle's converted bounds, one UE unit below the top. The sprite renderer's
`PivotInUVSpace` puts the particle 90% down its quad, so that anchor is the
flame's base and the geometry grows upward from it. All 6,421 share one mesh,
one meshlet mesh and one material, so they add one BLAS and no material clone,
but they do add one TLAS instance and one uniformly sampled Solari light source
each: Throne Room's 23,189 partitions plus 27 light proxies grow from 23,216
instances to 29,637, and its 29 light sources (27 proxies, one directional, one
environment) become 6,450. That is 10x under the 65,535 light-source ceiling,
but `generate_random_light_sample` picks uniformly, so every key light becomes
222x rarer to sample. `--no-candle-lights` turns the flames off - geometry and
light alike - to isolate that cost.

Unlike the exported fixtures' proxies the flames rasterize, as meshlets, so they
are visible in the primary image as well as in reflections and global
illumination. Solari adds directly visible emission only out of the deferred
G-buffer and traces no camera ray, so a TLAS-only emitter lights the room
without ever covering a pixel. Meshlets rather than `Mesh3d`, because the
meshlet deferred G-buffer pass depth-tests against the meshlet material depth
alone and runs after every plain-`Mesh3d` deferred write: a `Mesh3d` flame in
front of a candle body would keep its depth and lose its material.

The geometry is a solid of revolution of the measured flipbook silhouette:
`r(t) = scale * t^0.5 * (1 - t)^1.0625` revolved over 12 segments and 9 rings,
4.25 cm tall and 1.36 cm across at its widest, 168 triangles, starting 2.6 mm
below the anchor. The dimensions come from `NS_CandleFlame_04`'s `User.Flame
Size` (6,5) and `User.Flame Size Max` (7,7) UE units, averaged, times the
silhouette `candle_flame_02` occupies inside a flipbook tile, measured across
all 256 frames. That is 1.08M triangles across the level, 25% fewer than the
proxy sphere it replaces and 0.25% of the scene.

Sizing the flame from the sprite rather than from the light renderer's
`DefaultSourceRadius` leaves the flux derivation untouched and changes only the
area term in `luminance = flux / (pi * area)`: 3,895 cd/m2 rather than 15,279,
which is 12.7x Throne Room's 307.2 cd/m2 white point at EV100 8 and within 2.6x
of a real candle. It is also 58 px tall at a metre where the 5 mm sphere was
14 px and sub-pixel past 14 m.

Their colour is exact and their brightness is not. The particle colour
`(5, 2.246849, 1.1349998)` and the `2400` alpha that scales it are read verbatim
from the asset, and the runtime derives the tint from them rather than typing a
literal, so the authored ratio survives; it is CCT 3015 K, matching the 3000 K
the level's own `SPT_CandleFill` spots author, not a physical flame's 1700 K.
The magnitude rests on one engine constant, `1 cd = 10,000 internal light
units`, which comes from `UPointLightComponent::ComputeLightBrightness` scaling
candelas by `100 * 100` for UE's centimetre world - read from public 5.3/5.5
mirrors, because Zorah ships on an NVIDIA NvRTX branch that is not public and is
not in the sample download. That puts each flame at 1.2 cd / 15.08 lm, against a
real wax candle's roughly 1 cd. Two further judgement calls are labelled in
`main.rs`: the RGB triple is collapsed to a scalar by its peak channel, which is
1.8x brighter than a luminance-weighted reading; and the total flux is spread
uniformly over the lathe, which makes the flame brighter side-on than from above
in roughly the proportion a real flame is, but not by a measured profile.
`UE_LIGHT_UNITS_PER_CANDELA` is the single knob. A frame of Throne Room rendered
in Zorah's own build would settle all three.

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
approximation and is not yet a traced environment light.

The candle flames are opaque where UE's are additive. Nothing else is
available: `bevy_solari` builds every BLAS geometry `OPAQUE`, its `trace_ray`
commits the first hit with no any-hit test, and its `GpuMaterial` carries no
alpha at all, so an additive or alpha-masked flame cannot appear in the traced
image. The visible consequences are that a flame occludes what is behind it
instead of adding to it, that it casts a shadow-ray occlusion it should not, and
that its lowest centimetre may visibly clip through the candle body where UE's
additive sprite hid the intersection. A faithful port needs engine work in
dependency order: alpha in `GpuMaterial`, a `rayQueryConfirmIntersection`
candidate loop in place of the single `rayQueryProceed`, the per-instance cull
mask the TLAS already reserves 8 bits for, and ultimately pass-through radiance
accumulation in ReSTIR.

Three smaller flame approximations follow from the same choice. The lathe shows
the measured silhouette from every azimuth, which is what a camera-facing sprite
gives a floor-level viewer, but a viewer directly above sees a small disc rather
than a flame. Emission is uniform over the surface, where the flipbook's bottom
15% is a dark wick zone; no emissive texture, because the measured contrast is
only 1.82x peak-to-mean and Solari samples textures at LOD 0. And every flame is
the mean of UE's per-particle 6-7 x 5-7 UE unit randomisation, because
per-instance scale would mean per-instance area and therefore 6,421 materials.
The flames also do not animate. The light never did - `NS_CandleFlame_04` has no
curve, noise, random or time input anywhere in its chain - so only the sprite's
+/-14% per-frame variation is lost, and its 256-frame flipbook is a seamless 4 s
loop of a stationary particle, which makes a static mesh a fair port of its time
average. The Niagara systems that do animate - the nebula orb's translucent
shells and the GreenHouse butterflies - are exported but not spawned. UE
light-blocker meshes are retained as
ray-tracing occluders but intentionally omitted from camera-visible meshlet
rasterization, and they trace black: their `Placeholder` and
`M_Placeholder_Dark3` materials have no shading graph in UE, so the `Tint =
FFFFFFFF` the converter fabricates for them would otherwise make Throne Room's
6,087 m2 of invisible blockers the room's brightest diffuse reflector. The 225 components UE excludes from the shadow pass (1
GreenHouse, 216 Restir, 8 Throne Room) become `NotShadowCaster`, which meshlet
raster honors; Solari's TLAS carries every instance, so they still occlude
there.
