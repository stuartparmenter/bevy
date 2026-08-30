//! Validate Zorah's packed geometry against the meshlet renderer's paged GPU heap.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::File,
    io::{Cursor, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use argh::FromArgs;
use lz4_flex::frame::FrameDecoder;
use serde_json::Value;

const BUNDLE_MAGIC: [u8; 8] = *b"ZORAHB01";
const BUNDLE_VERSION: u32 = 2;
const MESHLET_MAGIC: u64 = 1_717_551_717_668;
const MESHLET_VERSION: u64 = 4;
const MESHLET_LZ4_MAGIC: [u8; 4] = [0x04, 0x22, 0x4d, 0x18];
const BLAS_MAGIC: [u8; 8] = *b"ZBLAS001";
const BLAS_VERSION: u32 = 1;
const LEVELS: [&str; 3] = ["GreenHouse_Level", "Restir_Level", "ThroneRoom_Level"];

// The serialized meshlet header: magic, version, MeshletAabb, and bvh_depth,
// all written before the LZ4 frame by MeshletMeshSaver.
const MESHLET_HEADER_BYTES: usize = 8 + 8 + 24 + 4;
// Serialized v4 assets use local u32 bit offsets and are limited to 512 MiB each.
const LOCAL_POSITION_ADDRESS_BYTES: u64 = (u32::MAX as u64 + 1) / 8;
const MESHLET_PAGE_BYTES: u64 = 64 * 1024 * 1024;
const MESHLET_MAX_PAGES: usize = 128;
const SECTION_ALIGNMENT: u64 = 16;
const ARRAY_NAMES: [&str; 7] = [
    "positions",
    "normals",
    "uvs",
    "indices",
    "bvh",
    "meshlets",
    "cull",
];
// Wire element sizes for Bevy meshlet asset format v4. The asset version guard
// makes a layout change fail closed rather than silently producing bad totals.
const ARRAY_ELEMENT_BYTES: [u64; 7] = [4, 4, 4, 1, 400, 48, 48];

// Solari's binding limits. `bevy_solari` keeps these private, so bump them
// together with crates/bevy_solari/src/scene/binder.rs.
const MAX_LIGHT_SOURCES: u64 = u16::MAX as u64;
const MAX_EMISSIVE_TRIANGLES_PER_LIGHT: u64 = u16::MAX as u64;
const MAX_TEXTURE_COUNT: usize = 5_000;
const MAX_MESH_SLAB_COUNT: u64 = 500;
// SlabAllocatorSettings::default in crates/bevy_render/src/slab_allocator.rs.
const MESH_SLAB_MAX_BYTES: u64 = 512 * 1024 * 1024;
const BLAS_VERTEX_BYTES: u64 = 32;
const BLAS_INDEX_BYTES: u64 = 4;
// Mirrors EMISSIVE_INTENSITY_NAMES in the runner (examples/.../src/main.rs),
// which gates whether a converted material emits at all.
const EMISSIVE_INTENSITY_NAMES: [&str; 3] = [
    "globalemissionintensity",
    "emissiveintensity",
    "emissionintensity",
];

#[derive(FromArgs)]
/// Report per-scene Zorah meshlet and BLAS capacity, failing on unsafe limits.
struct Args {
    /// packed Zorah runtime asset root
    #[argh(positional)]
    asset_root: PathBuf,
}

#[derive(Clone)]
struct EntryLocation {
    bundle: PathBuf,
    offset: u64,
    byte_length: u64,
    kind: String,
}

#[derive(Clone, Copy, Default)]
struct MeshletStats {
    bytes: [u64; 7],
}

impl MeshletStats {
    fn packed_bytes(self) -> Result<u64, &'static str> {
        let mut length = 0u64;
        for section in self.bytes {
            length = align_up(length, SECTION_ALIGNMENT)
                .checked_add(section)
                .ok_or("meshlet packed byte count overflow")?;
        }
        Ok(align_up(length, SECTION_ALIGNMENT))
    }
}

#[derive(Clone, Debug)]
struct PageAllocator {
    free: BTreeMap<u64, u64>,
}

impl PageAllocator {
    fn new() -> Self {
        Self {
            free: BTreeMap::from([(0, MESHLET_PAGE_BYTES)]),
        }
    }

    fn best_fit(&self, size: u64) -> Option<(u64, u64)> {
        self.free
            .iter()
            .filter(|(_, length)| **length >= size)
            .min_by_key(|(start, length)| (**length, **start))
            .map(|(start, length)| (*start, *length))
    }

    fn allocate(&mut self, size: u64) -> bool {
        let Some((start, length)) = self.best_fit(size) else {
            return false;
        };
        self.free.remove(&start);
        if length != size {
            self.free.insert(start + size, length - size);
        }
        true
    }
}

#[derive(Default)]
struct PagedHeap {
    pages: Vec<PageAllocator>,
    allocated_bytes: u64,
}

impl PagedHeap {
    fn allocate(&mut self, size: u64) -> Result<(), &'static str> {
        if size == 0 {
            return Err("meshlet asset packed to zero bytes");
        }
        if size > MESHLET_PAGE_BYTES {
            return Err("meshlet asset exceeds one 64 MiB page");
        }
        let allocated_bytes = self
            .allocated_bytes
            .checked_add(size)
            .ok_or("scene meshlet byte count overflow")?;
        let page = self
            .pages
            .iter()
            .enumerate()
            .filter_map(|(id, page)| page.best_fit(size).map(|fit| (id, fit)))
            .min_by_key(|(id, (start, length))| (*length, *id, *start))
            .map(|(id, _)| id);
        let page = match page {
            Some(page) => page,
            None if self.pages.len() < MESHLET_MAX_PAGES => {
                self.pages.push(PageAllocator::new());
                self.pages.len() - 1
            }
            None => return Err("scene exceeds the 128-page meshlet heap"),
        };
        assert!(self.pages[page].allocate(size));
        self.allocated_bytes = allocated_bytes;
        Ok(())
    }
}

#[derive(Clone, Copy, Default)]
struct BlasStats {
    vertices: u64,
    indices: u64,
    bytes: u64,
}

/// What a level costs Solari's fixed-size bindings.
///
/// The runtime gives every partition of every visible component instance its
/// own TLAS instance, and each emissive one contributes a light source per
/// 65,535-triangle chunk. Point and spot lights add one emissive proxy instance
/// each; directional and sky lights bind one light source each.
#[derive(Default)]
struct SolariUsage {
    instances: u64,
    emissive_instances: u64,
    analytic_lights: u64,
    emissive_instances_by_blas: HashMap<String, u64>,
    textures: HashSet<String>,
}

/// One inherited material parameter: normalized name, association, layer index,
/// and value.
type Parameters<T> = Vec<(String, String, i64, T)>;

#[derive(Clone, Default)]
struct EffectiveMaterial {
    emissive: bool,
    scalars: Parameters<f64>,
    textures: Parameters<Option<String>>,
}

#[derive(Clone, Default)]
struct MaterialUse {
    emissive: bool,
    textures: Vec<String>,
}

fn normalized_parameter_name(name: &str) -> String {
    name.chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect()
}

fn read_parameters<T>(
    record: &Value,
    key: &str,
    value: impl Fn(&Value) -> Option<T>,
) -> Parameters<T> {
    record[key]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|parameter| {
            Some((
                normalized_parameter_name(parameter["name"].as_str()?),
                parameter["association"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                parameter["index"].as_i64().unwrap_or(0),
                value(&parameter["value"])?,
            ))
        })
        .collect()
}

fn merge_parameters<T>(target: &mut Parameters<T>, source: Parameters<T>) {
    for parameter in source {
        match target.iter_mut().find(|existing| {
            (&existing.0, &existing.1, existing.2) == (&parameter.0, &parameter.1, parameter.2)
        }) {
            Some(existing) => *existing = parameter,
            None => target.push(parameter),
        }
    }
}

/// Pick the highest-ranked parameter, matching the runner's name preference and
/// its global-before-layer scope order.
fn select_parameter<'a, T>(parameters: &'a Parameters<T>, names: &[&str]) -> Option<&'a T> {
    parameters
        .iter()
        .filter_map(|(name, _, index, value)| {
            let name_rank = names.iter().position(|desired| desired == name)?;
            let scope_rank = match *index {
                -1 => 0,
                0 => 1,
                index if index > 0 => 2 + index as usize,
                _ => usize::MAX,
            };
            Some(((name_rank, scope_rank), value))
        })
        .min_by_key(|(rank, _)| *rank)
        .map(|(_, value)| value)
}

fn resolve_material(
    object: &str,
    records: &HashMap<&str, &Value>,
    cache: &mut HashMap<String, EffectiveMaterial>,
    stack: &mut Vec<String>,
) -> EffectiveMaterial {
    if let Some(cached) = cache.get(object) {
        return cached.clone();
    }
    if stack.iter().any(|entry| entry == object) {
        return EffectiveMaterial::default();
    }
    let Some(record) = records.get(object) else {
        return EffectiveMaterial::default();
    };
    stack.push(object.to_string());
    let mut effective = record["parent"]
        .as_str()
        .map_or_else(EffectiveMaterial::default, |parent| {
            resolve_material(parent, records, cache, stack)
        });
    if let Some(emissive) = record["emissive"].as_bool() {
        effective.emissive = emissive;
    }
    merge_parameters(
        &mut effective.scalars,
        read_parameters(record, "scalars", Value::as_f64),
    );
    merge_parameters(
        &mut effective.textures,
        read_parameters(record, "textures", |value| {
            Some(value.as_str().map(str::to_string))
        }),
    );
    stack.pop();
    cache.insert(object.to_string(), effective.clone());
    effective
}

fn material_use(
    object: &str,
    records: &HashMap<&str, &Value>,
    cache: &mut HashMap<String, EffectiveMaterial>,
    uses: &mut HashMap<String, MaterialUse>,
) -> MaterialUse {
    if let Some(cached) = uses.get(object) {
        return cached.clone();
    }
    let effective = resolve_material(object, records, cache, &mut Vec::new());
    let intensity = select_parameter(&effective.scalars, &EMISSIVE_INTENSITY_NAMES)
        .copied()
        .unwrap_or(0.0);
    let value = MaterialUse {
        emissive: effective.emissive && intensity > 0.0,
        // The runner binds at most four of these per material, so counting
        // every referenced texture bounds the binding array from above.
        textures: effective
            .textures
            .iter()
            .filter_map(|(_, _, _, texture)| texture.clone())
            .collect(),
    };
    uses.insert(object.to_string(), value.clone());
    value
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();
    report(&args.asset_root)
}

fn report(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let root = root.canonicalize()?;
    let locations = index_bundles(&root)?;
    let geometry: Value = read_json(&root.join("geometry.json"))?;
    let mut meshes = HashMap::new();
    for mesh in geometry["meshes"]
        .as_array()
        .ok_or("geometry.json has no meshes array")?
    {
        meshes.insert(mesh["object"].as_str().ok_or("mesh has no object")?, mesh);
    }

    let materials: Value = read_json(&root.join("materials.json"))?;
    let material_records = materials["materials"]
        .as_array()
        .ok_or("materials.json has no materials array")?
        .iter()
        .filter_map(|record| Some((record["object"].as_str()?, record)))
        .collect::<HashMap<_, _>>();
    let mut effective_materials = HashMap::new();
    let mut material_uses = HashMap::new();

    let mut needed_meshlets = HashSet::new();
    let mut needed_blas = HashSet::new();
    let mut scene_references = Vec::new();
    for level in LEVELS {
        let scene: Value = read_json(&root.join("scenes").join(format!("{level}.json")))?;
        let mut meshlet_references = Vec::new();
        let mut seen_meshlets = HashSet::new();
        let mut blas_references = HashSet::new();
        let mut solari = SolariUsage::default();
        for actor in scene["actors"]
            .as_array()
            .ok_or_else(|| format!("{level} has no actors array"))?
        {
            if actor["hidden"].as_bool().unwrap_or(false) {
                continue;
            }
            count_actor_lights(actor, &mut solari);
            let Some(components) = actor["components"].as_array() else {
                continue;
            };
            for component in components {
                if !component["visible"].as_bool().unwrap_or(true)
                    || component["hidden_in_game"].as_bool().unwrap_or(false)
                {
                    continue;
                }
                let Some(mesh) = component["mesh"]
                    .as_str()
                    .and_then(|object| meshes.get(object))
                else {
                    continue;
                };
                // A component with an instance array spawns one instance per
                // entry, an empty one spawns none, and no array spawns one.
                let instances = match component["instances"].as_array() {
                    Some(instances) => instances.len() as u64,
                    None => 1,
                };
                for partition in mesh["partitions"]
                    .as_array()
                    .ok_or("mesh has no partitions array")?
                {
                    let meshlet_reference = partition["meshlet"]
                        .as_str()
                        .ok_or("partition has no meshlet reference")?
                        .to_string();
                    if seen_meshlets.insert(meshlet_reference.clone()) {
                        meshlet_references.push(meshlet_reference);
                    }
                    let blas_reference = partition["geometry"]
                        .as_str()
                        .ok_or("partition has no BLAS reference")?
                        .to_string();
                    blas_references.insert(blas_reference.clone());
                    solari.instances += instances;
                    let Some(object) = partition_material(component, mesh, partition) else {
                        continue;
                    };
                    let used = material_use(
                        object,
                        &material_records,
                        &mut effective_materials,
                        &mut material_uses,
                    );
                    solari.textures.extend(used.textures);
                    if used.emissive {
                        solari.emissive_instances += instances;
                        *solari
                            .emissive_instances_by_blas
                            .entry(blas_reference)
                            .or_default() += instances;
                    }
                }
            }
        }
        needed_meshlets.extend(meshlet_references.iter().cloned());
        needed_blas.extend(blas_references.iter().cloned());
        scene_references.push((level, meshlet_references, blas_references, solari));
    }

    let mut meshlet_stats = HashMap::new();
    for reference in needed_meshlets {
        let location = require_kind(&locations, &reference, "meshlet")?;
        let bytes = read_payload(location)?;
        let stats = decode_meshlet_stats(&bytes)?;
        let packed_bytes = stats.packed_bytes()?;
        if packed_bytes > MESHLET_PAGE_BYTES {
            return Err(format!(
                "meshlet asset {reference} packs to {packed_bytes} bytes, exceeding the {MESHLET_PAGE_BYTES}-byte page limit"
            )
            .into());
        }
        meshlet_stats.insert(reference, stats);
    }
    let mut blas_stats = HashMap::new();
    for reference in needed_blas {
        let location = require_kind(&locations, &reference, "meshlet_blas")?;
        let bytes = read_payload(location)?;
        blas_stats.insert(reference, decode_blas_stats(&bytes)?);
    }

    let mut failures = Vec::new();
    for (level, meshlet_references, blas_references, solari) in scene_references {
        let mut meshlet = MeshletStats::default();
        let mut heap = PagedHeap::default();
        let mut max_asset = (None, 0u64);
        let mut heap_failed = false;
        for reference in &meshlet_references {
            let stats = meshlet_stats
                .get(reference)
                .ok_or("missing decoded meshlet statistics")?;
            for (total, value) in meshlet.bytes.iter_mut().zip(stats.bytes) {
                *total = total
                    .checked_add(value)
                    .ok_or("meshlet byte total overflow")?;
            }
            let packed_bytes = stats.packed_bytes()?;
            if packed_bytes > max_asset.1 {
                max_asset = (Some(reference.as_str()), packed_bytes);
            }
            if !heap_failed && let Err(reason) = heap.allocate(packed_bytes) {
                failures.push(format!(
                    "level={level} asset={reference} bytes={packed_bytes} reason={reason}"
                ));
                heap_failed = true;
            }
        }
        let mut blas = BlasStats::default();
        for reference in &blas_references {
            let stats = blas_stats
                .get(reference)
                .ok_or("missing decoded BLAS statistics")?;
            blas.vertices = blas
                .vertices
                .checked_add(stats.vertices)
                .ok_or("BLAS vertex total overflow")?;
            blas.indices = blas
                .indices
                .checked_add(stats.indices)
                .ok_or("BLAS index total overflow")?;
            blas.bytes = blas
                .bytes
                .checked_add(stats.bytes)
                .ok_or("BLAS byte total overflow")?;
        }

        let raw_meshlet_bytes = meshlet.bytes.iter().sum::<u64>();
        let page_capacity = heap.pages.len() as u64 * MESHLET_PAGE_BYTES;
        let utilization = if page_capacity == 0 {
            0.0
        } else {
            heap.allocated_bytes as f64 * 100.0 / page_capacity as f64
        };
        println!(
            "ZORAH_CAPACITY level={level} partitions={} meshlet_packed_bytes={} meshlet_raw_bytes={} meshlet_pages={} meshlet_page_capacity={} meshlet_page_utilization_percent={utilization:.2} max_meshlet_asset_bytes={} max_meshlet_asset={} blas_meshes={} blas_vertices={} blas_triangles={} blas_buffer_bytes={}",
            meshlet_references.len(),
            heap.allocated_bytes,
            raw_meshlet_bytes,
            heap.pages.len(),
            page_capacity,
            max_asset.1,
            max_asset.0.unwrap_or("none"),
            blas_references.len(),
            blas.vertices,
            blas.indices / 3,
            blas.bytes,
        );
        println!(
            "ZORAH_CAPACITY_BUFFERS level={level} {}",
            ARRAY_NAMES
                .iter()
                .zip(meshlet.bytes)
                .map(|(name, bytes)| format!("{name}={bytes}"))
                .collect::<Vec<_>>()
                .join(" ")
        );

        let mut emissive_light_sources = 0u64;
        for (reference, instances) in &solari.emissive_instances_by_blas {
            let stats = blas_stats
                .get(reference)
                .ok_or("missing decoded BLAS statistics")?;
            let chunks = (stats.indices / 3).div_ceil(MAX_EMISSIVE_TRIANGLES_PER_LIGHT);
            emissive_light_sources = emissive_light_sources
                .checked_add(
                    chunks
                        .checked_mul(*instances)
                        .ok_or("light source overflow")?,
                )
                .ok_or("light source overflow")?;
        }
        let light_sources = emissive_light_sources + solari.analytic_lights;
        // The mesh allocator packs many BLAS meshes per slab, so this bounds the
        // binding count from below; it catches only gross overruns.
        let least_vertex_slabs = blas
            .vertices
            .saturating_mul(BLAS_VERTEX_BYTES)
            .div_ceil(MESH_SLAB_MAX_BYTES);
        let least_index_slabs = blas
            .indices
            .saturating_mul(BLAS_INDEX_BYTES)
            .div_ceil(MESH_SLAB_MAX_BYTES);
        if light_sources > MAX_LIGHT_SOURCES {
            failures.push(format!(
                "level={level} light_sources={light_sources} reason=scene exceeds the {MAX_LIGHT_SOURCES} Solari light sources"
            ));
        }
        if solari.textures.len() > MAX_TEXTURE_COUNT {
            failures.push(format!(
                "level={level} textures={} reason=scene exceeds the {MAX_TEXTURE_COUNT} Solari texture bindings",
                solari.textures.len()
            ));
        }
        if least_vertex_slabs.max(least_index_slabs) > MAX_MESH_SLAB_COUNT {
            failures.push(format!(
                "level={level} vertex_slabs={least_vertex_slabs} index_slabs={least_index_slabs} reason=scene exceeds the {MAX_MESH_SLAB_COUNT} Solari mesh slab bindings"
            ));
        }
        println!(
            "ZORAH_CAPACITY_SOLARI level={level} raytracing_instances={} emissive_instances={} emissive_light_sources={emissive_light_sources} analytic_light_sources={} light_sources={light_sources}/{MAX_LIGHT_SOURCES} textures={}/{MAX_TEXTURE_COUNT} least_vertex_slabs={least_vertex_slabs} least_index_slabs={least_index_slabs} max_mesh_slabs={MAX_MESH_SLAB_COUNT}",
            solari.instances,
            solari.emissive_instances,
            solari.analytic_lights,
            solari.textures.len(),
        );
    }

    if !failures.is_empty() {
        for failure in &failures {
            eprintln!("ZORAH_CAPACITY_ERROR {failure}");
        }
        return Err(format!(
            "Zorah capacity validation failed with {} error(s)",
            failures.len()
        )
        .into());
    }
    println!(
        "ZORAH_CAPACITY_OK local_position_address_bytes={LOCAL_POSITION_ADDRESS_BYTES} meshlet_page_bytes={MESHLET_PAGE_BYTES} meshlet_max_pages={MESHLET_MAX_PAGES} max_light_sources={MAX_LIGHT_SOURCES} max_textures={MAX_TEXTURE_COUNT} max_mesh_slabs={MAX_MESH_SLAB_COUNT}"
    );
    Ok(())
}

/// Count the light sources an actor's exported lights bind, applying the same
/// gates the runner spawns them under.
fn count_actor_lights(actor: &Value, solari: &mut SolariUsage) {
    let lights = actor["lights"].as_array().map(Vec::as_slice).unwrap_or(&[]);
    let template = |light: &Value| {
        light["name"]
            .as_str()
            .unwrap_or_default()
            .ends_with("_GEN_VARIABLE")
    };
    let has_concrete_lights = lights.iter().any(|light| !template(light));
    for light in lights {
        if (has_concrete_lights && template(light))
            || !light["visible"].as_bool().unwrap_or(true)
            || light["hidden_in_game"].as_bool().unwrap_or(false)
            || !light["affects_world"].as_bool().unwrap_or(true)
            || light["intensity"].as_f64().unwrap_or(0.0) <= 0.0
        {
            continue;
        }
        match light["type"].as_str().unwrap_or_default() {
            // Point and spot lights bind through a small emissive proxy mesh,
            // whose triangle count is far below one chunk; directional and sky
            // lights bind one light source each.
            "point" | "spot" | "directional" | "sky" => solari.analytic_lights += 1,
            _ => {}
        }
    }
}

/// Resolve a partition's material the way the runner does: a non-empty
/// component override, otherwise the mesh's own slot material. UE indexes
/// `OverrideMaterials` by static-mesh material slot while a partition's
/// `material_slot` is its LOD0 section index, and the two differ whenever the
/// mesh carries a non-identity `FMeshSectionInfoMap`; manifests written before
/// `material_index` existed are section-indexed throughout.
fn partition_material<'a>(
    component: &'a Value,
    mesh: &'a Value,
    partition: &Value,
) -> Option<&'a str> {
    let slot = usize::try_from(partition["material_slot"].as_u64()?).ok()?;
    let override_index = partition["material_index"]
        .as_u64()
        .and_then(|index| usize::try_from(index).ok())
        .unwrap_or(slot);
    component["override_materials"]
        .as_array()
        .and_then(|overrides| overrides.get(override_index))
        .and_then(Value::as_str)
        .filter(|object| !object.is_empty())
        .or_else(|| {
            mesh["material_slots"]
                .as_array()
                .and_then(|slots| slots.get(slot))
                .and_then(Value::as_str)
        })
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

fn index_bundles(
    root: &Path,
) -> Result<HashMap<String, EntryLocation>, Box<dyn std::error::Error>> {
    let mut result = HashMap::new();
    let bundle_root = root.join("bundles");
    for entry in std::fs::read_dir(&bundle_root)? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("zorah_bundle") {
            continue;
        }
        let relative = path
            .strip_prefix(root)?
            .to_string_lossy()
            .replace('\\', "/");
        let mut file = File::open(&path)?;
        let mut magic = [0; 8];
        file.read_exact(&mut magic)?;
        if magic != BUNDLE_MAGIC {
            return Err(format!("wrong bundle magic: {}", path.display()).into());
        }
        let version = read_u32(&mut file)?;
        if version != BUNDLE_VERSION {
            return Err(format!("unsupported bundle version {version}: {}", path.display()).into());
        }
        let index_length = read_u64(&mut file)?;
        let mut index_bytes = vec![0; usize::try_from(index_length)?];
        file.read_exact(&mut index_bytes)?;
        let index: Value = serde_json::from_slice(&index_bytes)?;
        if index["format_version"].as_u64() != Some(BUNDLE_VERSION as u64) {
            return Err(format!("wrong bundle index version: {}", path.display()).into());
        }
        let mut offset = file.stream_position()?;
        for record in index["entries"]
            .as_array()
            .ok_or("bundle index has no entries array")?
        {
            let label = record["label"]
                .as_str()
                .ok_or("bundle entry has no label")?;
            let byte_length = record["byte_length"]
                .as_u64()
                .ok_or("bundle entry has no byte length")?;
            let kind = record["kind"]
                .as_str()
                .ok_or("bundle entry has no kind")?
                .to_string();
            let reference = format!("{relative}#{label}");
            if result
                .insert(
                    reference.clone(),
                    EntryLocation {
                        bundle: path.clone(),
                        offset,
                        byte_length,
                        kind,
                    },
                )
                .is_some()
            {
                return Err(format!("duplicate bundle reference {reference}").into());
            }
            offset = offset
                .checked_add(byte_length)
                .ok_or("bundle payload offset overflow")?;
        }
        if file.seek(SeekFrom::End(0))? != offset {
            return Err(format!("bundle payload length mismatch: {}", path.display()).into());
        }
    }
    Ok(result)
}

fn require_kind<'a>(
    locations: &'a HashMap<String, EntryLocation>,
    reference: &str,
    kind: &str,
) -> Result<&'a EntryLocation, Box<dyn std::error::Error>> {
    let location = locations
        .get(reference)
        .ok_or_else(|| format!("missing bundled asset {reference}"))?;
    if location.kind != kind {
        return Err(format!(
            "bundled asset {reference} has kind {}, expected {kind}",
            location.kind
        )
        .into());
    }
    Ok(location)
}

fn read_payload(location: &EntryLocation) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut file = File::open(&location.bundle)?;
    file.seek(SeekFrom::Start(location.offset))?;
    let mut bytes = vec![0; usize::try_from(location.byte_length)?];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn decode_meshlet_stats(bytes: &[u8]) -> Result<MeshletStats, Box<dyn std::error::Error>> {
    if bytes.len() < 20 {
        return Err("meshlet payload is truncated".into());
    }
    let mut header = Cursor::new(bytes);
    if read_u64(&mut header)? != MESHLET_MAGIC {
        return Err("meshlet payload has wrong magic".into());
    }
    let version = read_u64(&mut header)?;
    if version != MESHLET_VERSION {
        return Err(format!(
            "meshlet payload version {version} is unsupported; capacity parser expects {MESHLET_VERSION}"
        )
        .into());
    }
    // The header is fixed width, and its AABB floats can contain the LZ4 magic,
    // so the frame is taken from its known offset rather than searched for.
    let frame = bytes
        .get(MESHLET_HEADER_BYTES..)
        .ok_or("meshlet payload is truncated")?;
    if !frame.starts_with(&MESHLET_LZ4_MAGIC) {
        return Err("meshlet payload has no LZ4 frame after its header".into());
    }
    let mut decoder = FrameDecoder::new(frame);
    let mut decoded = Vec::new();
    decoder.read_to_end(&mut decoded)?;
    decode_meshlet_arrays(&decoded)
}

fn decode_meshlet_arrays(bytes: &[u8]) -> Result<MeshletStats, Box<dyn std::error::Error>> {
    let mut cursor = Cursor::new(bytes);
    let mut stats = MeshletStats::default();
    for (index, element_bytes) in ARRAY_ELEMENT_BYTES.into_iter().enumerate() {
        let count = read_u64(&mut cursor)?;
        let byte_length = count
            .checked_mul(element_bytes)
            .ok_or("meshlet array byte count overflow")?;
        let next = cursor
            .position()
            .checked_add(byte_length)
            .ok_or("meshlet array offset overflow")?;
        if next > bytes.len() as u64 {
            return Err(format!("meshlet {} array is truncated", ARRAY_NAMES[index]).into());
        }
        cursor.set_position(next);
        stats.bytes[index] = byte_length;
    }
    if cursor.position() != bytes.len() as u64 {
        return Err("meshlet payload has trailing decoded bytes".into());
    }
    if stats.bytes[0] > LOCAL_POSITION_ADDRESS_BYTES {
        return Err(format!(
            "one meshlet asset has {} position bytes, exceeding its local u32 bit-offset limit of {LOCAL_POSITION_ADDRESS_BYTES}",
            stats.bytes[0],
        )
        .into());
    }
    if let Some(index) = stats.bytes.iter().position(|bytes| *bytes == 0) {
        return Err(format!("meshlet {} array is empty", ARRAY_NAMES[index]).into());
    }
    if stats.bytes[1] != stats.bytes[2] {
        return Err("meshlet normal and UV arrays have different lengths".into());
    }
    if stats.bytes[5] != stats.bytes[6] {
        return Err("meshlet and cull-data arrays have different lengths".into());
    }
    Ok(stats)
}

fn decode_blas_stats(bytes: &[u8]) -> Result<BlasStats, Box<dyn std::error::Error>> {
    if bytes.len() < 20 || bytes[..8] != BLAS_MAGIC {
        return Err("BLAS payload has wrong magic or is truncated".into());
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into()?);
    if version != BLAS_VERSION {
        return Err(format!("unsupported BLAS payload version {version}").into());
    }
    let vertices = u32::from_le_bytes(bytes[12..16].try_into()?) as u64;
    let indices = u32::from_le_bytes(bytes[16..20].try_into()?) as u64;
    if indices % 3 != 0 {
        return Err("BLAS index count is not divisible by three".into());
    }
    let payload_bytes = vertices
        .checked_mul(32)
        .and_then(|size| size.checked_add(indices.checked_mul(4)?))
        .ok_or("BLAS payload byte count overflow")?;
    if payload_bytes.checked_add(20) != Some(bytes.len() as u64) {
        return Err("BLAS payload length does not match its counts".into());
    }
    Ok(BlasStats {
        vertices,
        indices,
        bytes: payload_bytes,
    })
}

fn read_json(path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn read_u32(reader: &mut impl Read) -> std::io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> std::io::Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_seven_version_four_meshlet_arrays() {
        let counts = [3u64, 5, 5, 9, 11, 13, 13];
        let mut bytes = Vec::new();
        for (count, element_bytes) in counts.into_iter().zip(ARRAY_ELEMENT_BYTES) {
            bytes.extend_from_slice(&count.to_le_bytes());
            bytes.resize(bytes.len() + (count * element_bytes) as usize, 0);
        }
        let stats = decode_meshlet_arrays(&bytes).unwrap();
        assert_eq!(
            stats.bytes,
            std::array::from_fn(|index| counts[index] * ARRAY_ELEMENT_BYTES[index])
        );
        let expected = stats.bytes.into_iter().fold(0, |length, section| {
            align_up(length, SECTION_ALIGNMENT) + section
        });
        assert_eq!(stats.packed_bytes().unwrap(), align_up(expected, 16));
    }

    #[test]
    fn section_packing_matches_runtime_alignment() {
        let stats = MeshletStats {
            bytes: [1, 2, 3, 4, 5, 6, 7],
        };
        assert_eq!(stats.packed_bytes(), Ok(112));
    }

    #[test]
    fn paged_heap_uses_best_fit_and_enforces_page_limit() {
        let mut heap = PagedHeap::default();
        heap.allocate(MESHLET_PAGE_BYTES - 32).unwrap();
        heap.allocate(64).unwrap();
        heap.allocate(32).unwrap();
        assert_eq!(heap.pages.len(), 2);
        assert_eq!(
            heap.allocate(MESHLET_PAGE_BYTES + 1),
            Err("meshlet asset exceeds one 64 MiB page")
        );
    }

    #[test]
    fn meshlet_frame_is_read_at_the_fixed_header_offset() {
        use std::io::Write as _;

        let counts = [3u64, 5, 5, 9, 11, 13, 13];
        let mut arrays = Vec::new();
        for (count, element_bytes) in counts.into_iter().zip(ARRAY_ELEMENT_BYTES) {
            arrays.extend_from_slice(&count.to_le_bytes());
            arrays.resize(arrays.len() + (count * element_bytes) as usize, 0);
        }
        let mut compressed = Vec::new();
        let mut encoder = lz4_flex::frame::FrameEncoder::new(&mut compressed);
        encoder.write_all(&arrays).unwrap();
        encoder.finish().unwrap();

        let mut payload = Vec::new();
        payload.extend_from_slice(&MESHLET_MAGIC.to_le_bytes());
        payload.extend_from_slice(&MESHLET_VERSION.to_le_bytes());
        // An AABB whose float bits happen to spell the LZ4 frame magic.
        let mut aabb = [0; 24];
        aabb[8..12].copy_from_slice(&MESHLET_LZ4_MAGIC);
        payload.extend_from_slice(&aabb);
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&compressed);

        let stats = decode_meshlet_stats(&payload).unwrap();
        assert_eq!(
            stats.bytes,
            std::array::from_fn(|index| counts[index] * ARRAY_ELEMENT_BYTES[index])
        );
    }

    #[test]
    fn emissive_gate_follows_the_material_parent_chain() {
        let parent = serde_json::json!({
            "object": "parent",
            "emissive": true,
            "scalars": [
                {"name": "Emissive Intensity", "association": "Global", "index": -1, "value": 4.0}
            ],
            "textures": [
                {"name": "Emissive", "association": "Global", "index": -1, "value": "T_Emissive"}
            ],
        });
        let child = serde_json::json!({"object": "child", "parent": "parent"});
        let dark = serde_json::json!({
            "object": "dark",
            "parent": "parent",
            "scalars": [
                {"name": "EmissiveIntensity", "association": "Global", "index": -1, "value": 0.0}
            ],
        });
        let records = HashMap::from([("parent", &parent), ("child", &child), ("dark", &dark)]);
        let mut cache = HashMap::new();
        let mut uses = HashMap::new();

        let child = material_use("child", &records, &mut cache, &mut uses);
        assert!(child.emissive);
        assert_eq!(child.textures, ["T_Emissive"]);
        assert!(!material_use("dark", &records, &mut cache, &mut uses).emissive);
        assert!(!material_use("absent", &records, &mut cache, &mut uses).emissive);
    }

    #[test]
    fn light_sources_apply_the_runner_spawn_gates() {
        let actor = serde_json::json!({
            "lights": [
                {"name": "Point", "type": "point", "intensity": 100.0},
                {"name": "Spot", "type": "spot", "intensity": 100.0, "visible": false},
                {"name": "Sky", "type": "sky", "intensity": 1.0},
                {"name": "Dark", "type": "point", "intensity": 0.0},
                {"name": "Lamp_GEN_VARIABLE", "type": "point", "intensity": 100.0},
            ],
        });
        let mut solari = SolariUsage::default();
        count_actor_lights(&actor, &mut solari);
        assert_eq!(solari.analytic_lights, 2);

        // Template records are the only lights an actor has when its concrete
        // components were not exported.
        let templates = serde_json::json!({
            "lights": [{"name": "Lamp_GEN_VARIABLE", "type": "point", "intensity": 100.0}],
        });
        let mut solari = SolariUsage::default();
        count_actor_lights(&templates, &mut solari);
        assert_eq!(solari.analytic_lights, 1);
    }

    #[test]
    fn partition_material_prefers_non_empty_overrides() {
        let mesh = serde_json::json!({"material_slots": ["MI_Base", null, "MI_Third"]});
        let component = serde_json::json!({"override_materials": ["", "MI_Override"]});
        let slot = |slot: u64| serde_json::json!({"material_slot": slot});
        assert_eq!(
            partition_material(&component, &mesh, &slot(0)),
            Some("MI_Base")
        );
        assert_eq!(
            partition_material(&component, &mesh, &slot(1)),
            Some("MI_Override")
        );
        // A non-identity section map indexes the overrides by material_index
        // while the mesh's own slots stay section-indexed.
        assert_eq!(
            partition_material(
                &component,
                &mesh,
                &serde_json::json!({"material_slot": 2, "material_index": 1})
            ),
            Some("MI_Override")
        );
        assert_eq!(
            partition_material(
                &component,
                &mesh,
                &serde_json::json!({"material_slot": 2, "material_index": 0})
            ),
            Some("MI_Third")
        );
    }

    #[test]
    fn paged_heap_rejects_the_129th_full_page_atomically() {
        let mut heap = PagedHeap::default();
        for _ in 0..MESHLET_MAX_PAGES {
            heap.allocate(MESHLET_PAGE_BYTES).unwrap();
        }
        let bytes_before = heap.allocated_bytes;
        assert_eq!(
            heap.allocate(16),
            Err("scene exceeds the 128-page meshlet heap")
        );
        assert_eq!(heap.pages.len(), MESHLET_MAX_PAGES);
        assert_eq!(heap.allocated_bytes, bytes_before);
    }
}
