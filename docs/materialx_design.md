# Design sketch: MaterialX support in Bevy

Status: **draft / research sketch** — no implementation yet.
Target: this fork, currently `bevy 0.20.0-dev`.

## 1. Goal and scope

[MaterialX](https://materialx.org/) (Academy Software Foundation, spec v1.39.5 as of
May 2026) is the industry-standard interchange format for materials and
look-development: an XML document (`.mtlx`) describing typed node graphs that
terminate in physically based shading models (`standard_surface`,
`open_pbr_surface`, `gltf_pbr`, `usd_preview_surface`), plus material/geometry
binding ("looks"), variants, color management, and real-world units. It is the
material representation used by USD, and is exported by Blender, Houdini, Maya,
Substance, and the big free material libraries (AMD GPUOpen, ambientCG).

"Full support" for Bevy decomposes into these capability layers, roughly in
order of value:

1. **Document layer** — parse/validate `.mtlx`, resolve node definitions,
   nodegraph instances, defaults, inheritance, and file references.
2. **Surface parameter layer** — map graphs whose inputs are constants or plain
   image files onto Bevy's `StandardMaterial`. This alone covers the majority
   of real-world `.mtlx` files in the wild (tileable texture-set materials).
3. **Pattern-graph shading layer** — compile arbitrary *pattern* subgraphs
   (procedurals, noises, mixes, transforms, ramps) to WGSL that computes the
   inputs of the shading model per-fragment, while reusing Bevy's own lighting.
4. **Shading-model layer** — faithful `standard_surface` / OpenPBR / glTF PBR /
   UsdPreviewSurface semantics, documenting where Bevy's BSDF cannot represent
   a feature (e.g. thin-film, fuzz/sheen) and what the approximation is.
5. **Look/interop layer** — `<look>`, `<materialassign>`, `<collection>`,
   variants, UDIM, units, color spaces; assignment onto spawned scenes;
   glTF/USD entry points.

Explicit non-goals for the foreseeable future: OSL/MDL closures with no
rasterizer meaning, volume/VDF shading, and MaterialX's own lighting/environment
code (Bevy's lights, shadows, and probes must be the single source of truth).

## 2. Constraints discovered in research

**Upstream MaterialX shader generation is not directly usable.** The C++
library gained a WGSL backend in v1.39.4 and a Slang backend in v1.39.5, but it
generates *monolithic* shaders containing MaterialX's own light loop and
environment sampling. Those shaders cannot participate in Bevy's clustered
forward/deferred lighting, shadows, or probes, and consuming them would require
C++ FFI in the hot path. three.js faced the same problem and chose to
*interpret* MaterialX graphs into its own node-material system instead; that is
the right model for Bevy too. (The C++ library remains useful out-of-band — see
§8 on baking/validation.)

**There are no maintained Rust bindings.** The closest prior art is
[killercup/materialx](https://github.com/killercup/materialx), a proof-of-concept
`materialx-parser` + `bevy-materialx-importer` that flattens simple graphs onto
`StandardMaterial`. Good validation of Tier A below; not a foundation for
shader generation.

**Bevy-side extension points are in good shape** (all paths relative to
`crates/`):

- `Material` trait (`bevy_pbr/src/material.rs:147`): a custom material only
  needs `fragment_shader()`; `specialize()` + `#[bind_group_data]` drive
  per-material pipeline permutations. Materials bind at group 3.
- Runtime shader generation is first-class: `Shader::from_wgsl(String, path)`
  (`bevy_shader/src/shader.rs:86`) → `Assets<Shader>::add` →
  `ShaderRef::Handle`. naga_oil `#import`s inside generated source resolve
  against registered shader libraries, so generated code can import
  `bevy_pbr::pbr_functions` etc.
- The lighting seam: `pbr_input_from_vertex_output` / `PbrInput`
  (`bevy_pbr/src/render/pbr_types.wgsl:97`) →
  `apply_pbr_lighting(pbr_input)` (`pbr_functions.wgsl:346`) →
  `main_pass_post_lighting_processing`. A generated fragment shader that fills
  `PbrInput` (base color, metallic, perceptual roughness, N, emissive,
  clearcoat, anisotropy, transmission…) gets Bevy's exact lighting for free.
- Asset loaders support labeled sub-assets with per-label dependency tracking
  (`LoadContext::labeled_asset_scope`, `bevy_asset/src/loader.rs:467`) — the
  `Gltf` asset (`bevy_gltf/src/assets.rs:18`) is the pattern to copy.
- glTF has an extension-handler hook (`GltfExtensionHandler`,
  `bevy_gltf/src/loader/extensions/mod.rs:62`) that bevy_pbr already uses to
  turn `GltfMaterial` into `StandardMaterial`; a MaterialX handler can plug in
  the same way.
- Texture-count ceiling: non-bindless materials get one binding per
  texture/sampler; bindless slabs allow up to 2048 resources (64 on
  macOS/iOS) via fixed binding arrays
  (`bevy_render/src/render_resource/bindless.rs`). Arbitrary-size MaterialX
  graphs need a per-graph generated binding layout (non-bindless) and can
  adopt bindless packing later.

## 3. Proposed architecture

Two new crates, mirroring how `bevy_gltf` is wired:

```
crates/materialx_core/     # engine-agnostic: no bevy deps (could be published standalone)
crates/bevy_materialx/     # loader + materials + WGSL codegen
```

`bevy_materialx` depends on `bevy_asset`, `bevy_image`, `bevy_material`,
`bevy_shader`, `bevy_render`, `bevy_pbr` (for `StandardMaterial`,
`ExtendedMaterial`, and the PBR shader library), optionally `bevy_gltf`. It is
exposed from `bevy_internal` as an optional `dep:`-gated `bevy_materialx`
feature, exactly like the `bevy_gltf` entry in `bevy_internal/Cargo.toml`.

### 3.1 `materialx_core`: document model and graph IR

- XML parse (e.g. `quick-xml`) → typed `Document`: `NodeDef`, `Node`,
  `NodeGraph`, `Input`/`Output` (typed: `float, color3, color4, vector2/3/4,
  boolean, integer, string, filename, matrix33/44` + arrays), `Material`,
  `Look`, `Collection`, `Variant`, `PropertySet`, includes (`XInclude`-style
  file references).
- Ship the **standard data library** (the `libraries/` nodedefs from the
  MaterialX distribution, Apache-2.0) as embedded data so documents referencing
  stdlib nodes resolve without the C++ runtime. Many stdlib "nodes" are
  themselves nodegraphs over a small primitive set — resolving them shrinks the
  set of primitives the code generator must implement (§3.4).
- Graph operations: nodedef resolution, default filling, nodegraph inlining,
  dead-branch elimination (conditionals with constant inputs), constant
  folding, topological sort, and a stable content hash per flattened graph
  (this hash becomes the shader/pipeline cache key).
- Resolution passes for color spaces (`srgb_texture`, `lin_rec709`, `acescg`,
  …) and real-world units (`unittype`/`unit` attributes → scale factors),
  annotating edges so downstream tiers can act on them.
- Validation with spans/diagnostics; golden tests against the official
  MaterialX example suite and the killercup downloader corpus (ambientCG,
  GPUOpen).

### 3.2 `bevy_materialx`: asset pipeline

`MaterialXLoader: AssetLoader` with `extensions() = ["mtlx"]`, producing:

```rust
#[derive(Asset)]
pub struct MaterialXDocument {
    pub materials: Vec<Handle<MaterialXMaterialAsset>>,   // labeled: "Material/<name>"
    pub named_materials: HashMap<Box<str>, Handle<…>>,
    pub looks: Vec<MaterialXLook>,                        // name-based assignment data
    pub default_material: Option<Handle<…>>,
    // + diagnostics, source graph (for tooling/hot-reload)
}
```

Referenced images load through `load_context.load` with
`ImageLoaderSettings { is_srgb: <from colorspace>, sampler: <from
uaddressmode/filtertype attrs> }`, following the glTF loader's
`load_image`/`texture_sampler` pattern. A `MaterialXAssetLabel` enum mirrors
`GltfAssetLabel` so `asset_server.load("foo.mtlx#Material/wood")` works.

### 3.3 Two-tier material realization

Each `<surfacematerial>` is classified after flattening:

**Tier A — flatten to `StandardMaterial`.** If every shading-model input is a
constant or a plain `image`/`tiledimage` node (optionally through trivial
adapters: `normalmap`, channel swizzles, `multiply` by constant), emit a
`StandardMaterial` directly. Mappings:

- `gltf_pbr` → near 1:1 (Bevy's material *is* glTF PBR: base color,
  metallic/roughness, normal, occlusion, emissive, clearcoat, transmission,
  specular, anisotropy, ior — all present on `StandardMaterial`).
- `usd_preview_surface` → straightforward; `useSpecularWorkflow=1` needs a
  spec-to-metallic conversion.
- `standard_surface` / `open_pbr_surface` → map base/specular/coat/emission/
  transmission/anisotropy; document lossy cases (§5).

This tier needs no shader generation, works with bindless out of the box, and
covers the bulk of downloadable `.mtlx` content.

**Tier B — WGSL codegen.** For graphs with real pattern networks, generate a
fragment shader per unique flattened-graph hash:

```wgsl
#import bevy_pbr::{
    pbr_types::{PbrInput, pbr_input_new},
    pbr_fragment,          // for pbr_input_from_vertex_output
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    forward_io::{VertexOutput, FragmentOutput},
}
#import bevy_materialx::stdlib   // hand-written mx_* node library (§3.4)

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr: PbrInput = pbr_input_from_vertex_output(in, is_front, false);
    // --- generated from the flattened graph (SSA over topo order) ---
    let n3 = mx_image_color3(mx_tex_0, mx_samp_0, in.uv);
    let n4 = mx_fractal3d_color3(..., in.world_position.xyz);
    let n5 = mx_mix_color3(n3, n4, n2);
    // --- BSDF input hookup ---
    pbr.material.base_color = vec4(n5, 1.0);
    pbr.material.perceptual_roughness = n7;
    pbr.N = mx_apply_normal_map(...);
    // ----------------------------------------------------------------
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr);
    out.color = main_pass_post_lighting_processing(pbr, out.color);
    return out;
}
```

The generated `String` becomes a `Shader` via `Shader::from_wgsl` at
load/prepare time, cached by graph hash in a `MaterialXShaderCache` resource.
Prepass/deferred variants are generated the same way (or Tier B materials
initially opt out of deferred via `OpaqueRendererMethod::Forward`).

The Rust-side asset is **one** material type, not one type per graph:

```rust
#[derive(Asset, Clone)]
pub struct MaterialXMaterial {
    graph: Arc<CompiledGraph>,        // hash, binding plan, shader handle
    scalars: Vec<f32>,                // packed uniform values (graph "public inputs")
    textures: Vec<Handle<Image>>,     // in binding-plan order
    alpha_mode: AlphaMode,
    double_sided: bool,
}
```

with a **manual `AsBindGroup` impl** (the derive assumes a static field list):
one uniform buffer of packed scalars at binding 0, then texture/sampler pairs
in plan order. `bind_group_data()` returns the graph hash so
`specialize()` swaps in the per-graph generated shader and the per-graph
`BindGroupLayout`. This keeps `MaterialPlugin<MaterialXMaterial>` and the whole
queue/specialize machinery untouched, and makes editing a *parameter* (uniform
rewrite) cheap while editing the *graph topology* recompiles.

Non-bindless first (per-graph explicit bindings — realistic graphs use well
under the guaranteed 16 sampled-texture limit after constant-folding; graphs
exceeding device limits fall back to Tier A approximation with a warning).
Bindless packing into Bevy's binding-array scheme is a later optimization.

### 3.4 The WGSL node library

A hand-written `bevy_materialx/src/render/stdlib.wgsl` (registered with
`load_shader_library!`) implementing the *primitive* node set — after stdlib
nodegraph-inlining this is on the order of 60–100 functions, not 700:

- math/logic: add, sub, mul, div, mod, power, clamp, min/max, smoothstep,
  remap, ifgreater/ifequal, and/or/not, dot/cross/normalize/magnitude
- color: mix, luminance, contrast, saturate, hsv↔rgb, gamma/color-space
  transforms (matrices generated from `materialx_core`'s CMS pass)
- procedural: noise2d/3d (Perlin), fractal2d/3d, cellnoise, worleynoise,
  unifiednoise, checkerboard, ramp4/ramplr/tb, splitlr/tb, circle/hexagon
- texture: image, tiledimage, triplanarprojection, normalmap, hextiledimage
  (later), UDIM (later, needs texture-array plumbing)
- geometric: texcoord, position, normal, tangent, bitangent, geomcolor,
  geompropvalue (→ Bevy custom vertex attributes), viewdirection, frame/time
  (→ Bevy globals)
- transforms: transformpoint/vector/normal, rotate2d/3d, place2d

Porting reference: MaterialX's own `libraries/stdlib/genglsl/*.glsl` is
Apache-2.0 and translates mechanically to WGSL. This library is also
unit-testable headlessly (compute-shader harness comparing against CPU
reference evaluation).

### 3.5 Shading-model mapping (Tier A and B share this)

The terminal shader node maps onto `PbrInput`/`StandardMaterial` fields:

| MaterialX | Bevy | Notes |
|---|---|---|
| `gltf_pbr` | direct | reference target, lossless |
| `usd_preview_surface` | direct | spec-workflow conversion when enabled |
| `standard_surface` base/metalness/roughness/specular/coat/emission | direct | coat → clearcoat fields |
| `open_pbr_surface` v1.1 | mostly direct | Bevy reflectance/specular_tint covers F0 tinting |
| transmission/subsurface | `specular_transmission`/`diffuse_transmission` + thickness/attenuation | needs `pbr_transmission_textures` feature for maps |
| anisotropy | `anisotropy_strength/rotation` | tangent-frame conventions need care |
| sheen/fuzz, thin-film iridescence | **gap** | approximate (fold into roughness/F0) + warn; candidate upstream `bevy_pbr` additions |
| displacement | **gap initially** | later: vertex-stage codegen or bake-to-mesh |

Gaps are reported as structured load diagnostics, not silent drops.

### 3.6 Looks, variants, scene integration

- `MaterialXDocument` retains `<look>`/`<materialassign>`/`<collection>` data.
  An `ApplyMaterialXLook` API (system + command) walks a spawned scene's
  `Name`/hierarchy-path components, matches geometry patterns (`/a/b*`
  globbing per spec), and swaps `MeshMaterial3d` handles. This is the piece
  USD-style pipelines actually need.
- Variants/variantsets surface as labeled material permutations
  (`"Material/wood?variant=aged"`).
- glTF interop: a `GltfExtensionHandler` in `bevy_materialx` so a future
  MaterialX-in-glTF vendor extension (or sidecar `.mtlx` next to the `.gltf`)
  can override materials without touching the core loader.

## 4. What "full support" will still exclude at first

- Volume shading (`volume_material`, VDFs) — no Bevy counterpart.
- OSL-specific nodes and closures with no raster semantics.
- MaterialX environment/light nodes — intentionally ignored; Bevy owns lighting.
- Displacement (until a vertex-stage codegen pass exists).
- UDIM and hex-tiled images in v1; both are tractable follow-ups.

## 5. Phasing

| Phase | Deliverable | Rough size |
|---|---|---|
| 0 | `materialx_core`: parser, document model, stdlib data library, flatten/fold/hash, diagnostics; golden tests vs. official examples | medium — the spec surface is large but mechanical |
| 1 | `bevy_materialx` loader + Tier A → `StandardMaterial`; `MaterialXAssetLabel`; examples with ambientCG/GPUOpen assets | small–medium; ships user value immediately |
| 2 | Tier B codegen MVP: WGSL emitter, `stdlib.wgsl` core subset (math/mix/image/noise/texcoord), `MaterialXMaterial` + manual `AsBindGroup`, shader cache, forward-path only | large — the heart of the project |
| 3 | Shading-model fidelity: full `standard_surface`/OpenPBR input coverage, prepass/deferred generation, alpha modes, double-sided, tangent frames; render-comparison tests vs. MaterialXView | medium |
| 4 | Full-spec layer: looks/collections/variants, color management + units end-to-end, UDIM, `geompropvalue`, glTF handler, hot-reload of `.mtlx` edits | medium, parallelizable |
| 5 (opt) | Asset-processor integration: optional C++ MaterialX via FFI in the *offline* processor for validation and bake-to-texture of unsupported subgraphs | optional, keeps runtime pure-Rust |

## 6. Risks / open questions

- **Pipeline-permutation explosion**: every unique graph is a pipeline. The
  graph hash dedupes identical graphs (texture-set materials collapse to a
  handful), but a scene of hand-authored graphs pays real compile cost —
  mitigations: async pipeline compilation (already Bevy's model), Tier A
  preference, and offline baking (Phase 5).
- **`PbrInput` API stability**: Tier B couples to `bevy_pbr`'s WGSL internals,
  which are import-stable but not semver-guarded; being in-tree (this repo)
  largely neutralizes that.
- **Bindless interplay**: manual per-graph layouts opt Tier B out of bindless
  batching initially; acceptable, but worth a design pass with
  `material_bind_groups.rs` maintainers before Phase 2 lands.
- **Color management depth**: full OCIO-style CMS is out of scope; matrix-based
  transforms for the spec's named spaces are enough for parity with other
  real-time implementations (three.js, Babylon).
- **Spec conformance target**: propose tracking the official
  `resources/Materials/Examples` render tests and publishing a conformance
  matrix rather than claiming blanket "full support".
