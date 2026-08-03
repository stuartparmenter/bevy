//! Convert the loose Zorah intermediate tree into runtime-ready bundle assets.

#[path = "../zorah_bundle.rs"]
#[allow(dead_code)]
mod zorah_bundle;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Condvar, Mutex},
    task::{Context, Poll},
};

use argh::FromArgs;
use bevy::{
    asset::{
        saver::{AssetSaver, SavedAsset},
        AssetPath, RenderAssetUsages,
    },
    image::{
        CompressedImageFormats, CompressedImageSaver, CompressedImageSaverSettings,
        ImageCompressorAlphaMode, ImageCompressorQuality, ImageSampler, ImageType,
    },
    mesh::VertexAttributeValues,
    pbr::experimental::meshlet::{MeshletMesh, MeshletMeshSaver, MESHLET_MESH_ASSET_VERSION},
    prelude::{Image, Mesh, Vec3},
    render::render_resource::TextureFormat,
    tasks::{block_on, AsyncComputeTaskPool, ComputeTaskPool, TaskPool},
};
use futures_io::AsyncWrite;
use serde_json::Value;
use zorah_bundle::{
    mesh_from_converter_glb, BundleEntry, BundleEntryKind, BundleIndex, ZORAH_BUNDLE_MAGIC,
    ZORAH_BUNDLE_VERSION,
};

const DEFAULT_GEOMETRY_SHARD_BYTES: u64 = 512 * 1024 * 1024;
const DEFAULT_TEXTURE_SHARD_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// Meshlet building is dominated by serial METIS/meshopt phases, so throughput
/// scales with concurrent partitions, not with threads inside one. Each lane
/// peaks at a few hundred MiB of scratch; leave four cores for the task pools
/// that absorb `from_mesh`'s parallel simplification bursts.
fn default_geometry_jobs() -> usize {
    std::thread::available_parallelism()
        .map_or(4, |cores| cores.get().saturating_sub(4).clamp(4, 24))
}
const DEFAULT_TEXTURE_QUALITY: &str = "very-fast";
const LEVELS: [&str; 3] = ["GreenHouse_Level", "Restir_Level", "ThroneRoom_Level"];
/// Identity of this packer's own encoding rules, stamped into every packed mesh
/// and texture so incremental reuse rejects artifacts from an older packer.
///
/// Bump together with `PACK_PIPELINE_VERSION` in `convert/convert.py`, which
/// hashes the same number into its fingerprints.
const PACK_PIPELINE_VERSION: u32 = 1;

#[derive(FromArgs)]
/// Pack loose converted Zorah assets into deterministic runtime bundles.
struct Args {
    /// loose converter output containing geometry, textures, materials, and scenes
    #[argh(positional)]
    source: PathBuf,

    /// final Bevy assets directory (must not already exist)
    #[argh(positional)]
    output: PathBuf,

    /// approximate maximum geometry payload bytes per bundle
    #[argh(option, default = "DEFAULT_GEOMETRY_SHARD_BYTES")]
    geometry_shard_bytes: u64,

    /// approximate maximum texture payload bytes per bundle
    #[argh(option, default = "DEFAULT_TEXTURE_SHARD_BYTES")]
    texture_shard_bytes: u64,

    /// absolute meshlet LOD error in metres for Solari BLAS geometry
    #[argh(option, default = "0.02")]
    raytracing_error: f32,

    /// existing runtime tree whose unchanged geometry and textures may be reused
    #[argh(option)]
    reuse_from: Option<PathBuf>,

    /// stable fingerprint of the loose geometry inputs and geometry pack settings
    #[argh(option)]
    geometry_fingerprint: Option<String>,

    /// stable fingerprint of materials, textures, scenes, and runtime pack settings
    #[argh(option)]
    runtime_fingerprint: Option<String>,

    /// maximum number of textures compressed concurrently
    #[argh(option, default = "2")]
    texture_jobs: usize,

    /// maximum number of partitions converted to meshlet and BLAS geometry concurrently
    /// (default: host cores minus four, between 4 and 24)
    #[argh(option, default = "default_geometry_jobs()")]
    geometry_jobs: usize,

    /// block-compression search effort: ultra-fast, very-fast, fast, basic, slow, very-slow
    #[argh(option, default = "DEFAULT_TEXTURE_QUALITY.to_string()")]
    texture_quality: String,
}

/// Every setting that changes packed bytes, resolved from the command line.
struct PackSettings {
    geometry_shard_bytes: u64,
    texture_shard_bytes: u64,
    raytracing_error: f32,
    geometry_jobs: usize,
    texture_jobs: usize,
    texture_quality: ImageCompressorQuality,
    texture_quality_name: String,
}

fn parse_texture_quality(name: &str) -> Option<ImageCompressorQuality> {
    Some(match name {
        "ultra-fast" => ImageCompressorQuality::UltraFast,
        "very-fast" => ImageCompressorQuality::VeryFast,
        "fast" => ImageCompressorQuality::Fast,
        "basic" => ImageCompressorQuality::Basic,
        "slow" => ImageCompressorQuality::Slow,
        "very-slow" => ImageCompressorQuality::VerySlow,
        _ => return None,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();
    AsyncComputeTaskPool::get_or_init(TaskPool::default);
    ComputeTaskPool::get_or_init(TaskPool::default);
    let source = args.source.canonicalize()?;
    if args.output.exists() {
        return Err(format!("output already exists: {}", args.output.display()).into());
    }
    if args.geometry_shard_bytes < 64 * 1024 * 1024 {
        return Err("--geometry-shard-bytes must be at least 64 MiB".into());
    }
    if args.texture_shard_bytes < 64 * 1024 * 1024 {
        return Err("--texture-shard-bytes must be at least 64 MiB".into());
    }
    if args.texture_jobs == 0 {
        return Err("--texture-jobs must be at least one".into());
    }
    if args.geometry_jobs == 0 {
        return Err("--geometry-jobs must be at least one".into());
    }
    let Some(texture_quality) = parse_texture_quality(&args.texture_quality) else {
        return Err(format!("unknown --texture-quality: {}", args.texture_quality).into());
    };
    let output_parent = args
        .output
        .parent()
        .ok_or("output must have a parent directory")?;
    fs::create_dir_all(output_parent)?;
    let temporary = tempfile_path(output_parent, args.output.file_name().unwrap_or_default());
    fs::create_dir(&temporary)?;

    if !args.raytracing_error.is_finite() || args.raytracing_error < 0.0 {
        return Err("--raytracing-error must be finite and non-negative".into());
    }
    let settings = PackSettings {
        geometry_shard_bytes: args.geometry_shard_bytes,
        texture_shard_bytes: args.texture_shard_bytes,
        raytracing_error: args.raytracing_error,
        geometry_jobs: args.geometry_jobs,
        texture_jobs: args.texture_jobs,
        texture_quality,
        texture_quality_name: args.texture_quality,
    };
    let result = pack(
        &source,
        &temporary,
        &settings,
        args.reuse_from.as_deref(),
        args.geometry_fingerprint.as_deref(),
        args.runtime_fingerprint.as_deref(),
    );
    if let Err(error) = result {
        let _ = fs::remove_dir_all(&temporary);
        return Err(error);
    }
    fs::rename(&temporary, &args.output)?;
    Ok(())
}

fn pack(
    source: &Path,
    output: &Path,
    settings: &PackSettings,
    reuse_from: Option<&Path>,
    geometry_fingerprint: Option<&str>,
    runtime_fingerprint: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let reuse_from = reuse_from.map(Path::canonicalize).transpose()?;
    let reuse_from = reuse_from.as_deref();
    let reserved_bundle_id = reuse_from.map_or(0, next_bundle_id);
    let mut bundles = BundleWriter::new(output, settings.geometry_shard_bytes, reserved_bundle_id)?;
    let mut geometry: Value = read_json(&source.join("geometry.json"))?;
    // Bundles the reused records still point at. Linking waits until the
    // texture pass has decided too, so each shard is copied forward exactly
    // once and without the entries this generation no longer references.
    let mut reused_references = HashSet::new();
    let mut reused_mesh_count = 0usize;
    if let Some(reuse) = reuse_from {
        let previous = read_json(&reuse.join("geometry.json"))?;
        let previous_meshes = previous["meshes"]
            .as_array()
            .ok_or("reused geometry.json has no meshes array")?
            .iter()
            .filter_map(|mesh| Some((mesh["object"].as_str()?.to_string(), mesh)))
            .collect::<HashMap<_, _>>();
        for mesh in geometry["meshes"]
            .as_array_mut()
            .ok_or("geometry.json has no meshes array")?
        {
            let Some(object) = mesh["object"].as_str() else {
                return Err("geometry mesh has no object".into());
            };
            let Some(previous_mesh) = previous_meshes.get(object) else {
                continue;
            };
            if reusable_geometry_mesh(source, mesh, previous_mesh, settings.raytracing_error)? {
                let material_slots = mesh["material_slots"].take();
                *mesh = (*previous_mesh).clone();
                mesh["material_slots"] = material_slots;
                reused_mesh_count += 1;
            }
        }
        collect_bundle_references(&geometry, &mut reused_references);
    }
    let mesh_usage = scene_mesh_usage(source)?;
    let meshes = geometry["meshes"]
        .as_array_mut()
        .ok_or("geometry.json has no meshes array")?;
    // Keep each usage signature in its own shards. The three Zorah levels are
    // almost entirely disjoint, so mixing meshes alphabetically makes every
    // level load every geometry shard and defeats the purpose of bundling.
    meshes.sort_by(|left, right| {
        let left_object = left["object"].as_str().unwrap_or_default();
        let right_object = right["object"].as_str().unwrap_or_default();
        mesh_usage
            .get(left_object)
            .cmp(&mesh_usage.get(right_object))
            .then_with(|| left_object.cmp(right_object))
    });
    let mesh_count = meshes.len();
    let mut totals = GeometryTotals::default();
    if reused_mesh_count != 0 {
        totals = reused_geometry_totals(meshes);
        println!(
            "ZORAH_BUNDLE_GEOMETRY_REUSED meshes={}/{} partitions={}",
            reused_mesh_count, mesh_count, totals.partitions
        );
    }
    pack_geometry(
        source,
        meshes,
        &mesh_usage,
        &mut bundles,
        settings,
        &mut totals,
    )?;

    // Texture shards are preloaded independently. Never let the first texture
    // share a bundle with the tail of a level's geometry group.
    bundles.finish_current()?;
    bundles.target_bytes = settings.texture_shard_bytes;
    let runtime_textures = source.join("textures.runtime.json");
    let texture_manifest = if runtime_textures.is_file() {
        runtime_textures
    } else {
        source.join("textures.exported.json")
    };
    let mut textures: Value = read_json(&texture_manifest)?;
    let exported = textures["exported"]
        .as_array_mut()
        .ok_or("texture manifest has no exported array")?;
    let texture_count = exported.len();
    let previous_textures = reuse_from
        .map(|root| read_json(&root.join("textures.exported.json")))
        .transpose()?
        .and_then(|document| document["exported"].as_array().cloned())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|record| Some((record["object"].as_str()?.to_string(), record)))
        .collect::<HashMap<_, _>>();
    let mut pending = Vec::new();
    let mut reused_textures = 0usize;
    for (index, texture) in exported.iter_mut().enumerate() {
        let source_path = source.join(
            texture["output"]
                .as_str()
                .ok_or("texture record has no output")?,
        );
        let object = texture["object"]
            .as_str()
            .ok_or("texture record has no object")?
            .to_string();
        let metadata = source_path.metadata()?;
        let source_size = metadata.len();
        let source_modified_ns = modified_ns(&metadata)?;
        if let Some(previous) = previous_textures.get(&object)
            && reusable_texture(
                texture,
                previous,
                source_size,
                source_modified_ns,
                &settings.texture_quality_name,
            )
        {
            let reference = previous["output"]
                .as_str()
                .ok_or("reused texture has no bundle reference")?;
            reused_references.insert(reference.to_string());
            texture["output"] = Value::String(reference.to_string());
            stamp_texture_record(texture, source_size, source_modified_ns, settings);
            reused_textures += 1;
            continue;
        }
        pending.push(PendingTexture {
            index,
            source_path,
            srgb: texture["srgb"].as_bool().unwrap_or(true),
            normal_map: texture["normal_map"].as_bool().unwrap_or(false),
            source_size,
            source_modified_ns,
        });
    }
    if let Some(reuse) = reuse_from {
        link_reused_bundles(&reused_references, reuse, output)?;
    }

    let pending_count = pending.len();
    let mut compressed_textures = 0usize;
    let compressed = OrderedWork::new(pending_count, settings.texture_jobs);
    let compress = |index: usize| {
        let pending: &PendingTexture = &pending[index];
        compress_texture(
            &pending.source_path,
            pending.srgb,
            pending.normal_map,
            settings.texture_quality,
        )
        .map_err(|error| error.to_string())
    };
    let compression = std::thread::scope(|scope| {
        for _ in 0..settings.texture_jobs {
            scope.spawn(|| compressed.run(&compress));
        }
        let mut store = || -> Result<(), Box<dyn std::error::Error>> {
            for pending in &pending {
                let encoded = compressed.next()?;
                let texture = &mut exported[pending.index];
                texture["output"] = Value::String(bundles.add(
                    format!("t/{:04}", pending.index),
                    BundleEntryKind::Image { srgb: encoded.srgb },
                    &encoded.bytes,
                )?);
                stamp_texture_record(
                    texture,
                    pending.source_size,
                    pending.source_modified_ns,
                    settings,
                );
                compressed_textures += 1;
                if compressed_textures.is_multiple_of(32) || compressed_textures == pending_count {
                    println!(
                        "ZORAH_BUNDLE_TEXTURE compressed={}/{} total={}",
                        compressed_textures, pending_count, texture_count,
                    );
                }
            }
            Ok(())
        };
        let result = store();
        // Release any worker still parked on the lookahead limit, otherwise the
        // scope cannot join them after an early error.
        compressed.finish();
        result
    });
    compression?;
    println!(
        "ZORAH_BUNDLE_TEXTURE_REUSE reused={} compressed={} total={}",
        reused_textures, pending_count, texture_count
    );

    let stats = bundles.finish()?;
    let _ = exported;
    write_json(&output.join("geometry.json"), &geometry)?;
    write_json(&output.join("textures.exported.json"), &textures)?;
    let runtime_materials = source.join("materials.runtime.json");
    let material_manifest = if runtime_materials.is_file() {
        runtime_materials
    } else {
        source.join("materials.json")
    };
    fs::copy(material_manifest, output.join("materials.json"))?;
    let scenes_output = output.join("scenes");
    fs::create_dir(&scenes_output)?;
    for level in LEVELS {
        fs::copy(
            source.join("scenes").join(format!("{level}.json")),
            scenes_output.join(format!("{level}.json")),
        )?;
    }
    write_json(
        &output.join("pack.json"),
        &serde_json::json!({
            "format": "zorah-pack-state-v1",
            "bundle_format_version": ZORAH_BUNDLE_VERSION,
            "meshlet_asset_version": MESHLET_MESH_ASSET_VERSION,
            "pack_pipeline_version": PACK_PIPELINE_VERSION,
            "geometry_fingerprint": geometry_fingerprint,
            "runtime_fingerprint": runtime_fingerprint,
            "geometry_shard_bytes": settings.geometry_shard_bytes,
            "texture_shard_bytes": settings.texture_shard_bytes,
            "raytracing_error": settings.raytracing_error,
            "texture_quality": settings.texture_quality_name,
        }),
    )?;
    println!(
        "ZORAH_BUNDLE_DONE bundles={} entries={} payload_bytes={} partitions={} winding_repairs={} textures={} raster_triangles={} raytracing_triangles={} raytracing_vertices={}",
        stats.bundles,
        stats.entries,
        stats.payload_bytes,
        totals.partitions,
        totals.winding_repairs,
        texture_count,
        totals.raster_triangles,
        totals.raytracing_triangles,
        totals.raytracing_vertices,
    );
    Ok(())
}

#[derive(Default)]
struct GeometryTotals {
    partitions: usize,
    raster_triangles: u64,
    raytracing_triangles: u64,
    raytracing_vertices: u64,
    winding_repairs: usize,
}

/// Totals already covered by the meshes spliced in from the reused tree.
fn reused_geometry_totals(meshes: &[Value]) -> GeometryTotals {
    let reused = || {
        meshes
            .iter()
            .filter(|mesh| mesh.get("parts_manifest").is_none())
            .flat_map(|mesh| mesh["partitions"].as_array().into_iter().flatten())
    };
    GeometryTotals {
        partitions: reused().count(),
        raster_triangles: reused()
            .filter_map(|partition| partition["triangles"].as_u64())
            .sum(),
        raytracing_triangles: reused()
            .filter_map(|partition| partition["blas_triangles"].as_u64())
            .sum(),
        raytracing_vertices: reused()
            .filter_map(|partition| partition["blas_vertices"].as_u64())
            .sum(),
        winding_repairs: 0,
    }
}

/// A mesh still holding a loose parts manifest, resolved ahead of packing so
/// its partitions can be converted while earlier ones are still being written.
struct PlannedMesh {
    mesh_index: usize,
    object: String,
    usage: u8,
    manifest: Value,
}

struct PackedPartition {
    meshlet: Vec<u8>,
    raytracing: Vec<u8>,
    blas_vertices: u64,
    blas_triangles: u64,
    winding_repaired: bool,
}

fn pack_geometry(
    source: &Path,
    meshes: &mut [Value],
    mesh_usage: &HashMap<String, u8>,
    bundles: &mut BundleWriter,
    settings: &PackSettings,
    totals: &mut GeometryTotals,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut plan = Vec::new();
    let mut partition_sources = Vec::new();
    for (mesh_index, mesh) in meshes.iter().enumerate() {
        let Some(parts_manifest) = mesh.get("parts_manifest") else {
            continue;
        };
        let object = mesh["object"]
            .as_str()
            .ok_or("geometry mesh has no object")?
            .to_string();
        let usage = *mesh_usage
            .get(&object)
            .ok_or_else(|| format!("geometry mesh is unused by every level: {object}"))?;
        let manifest_path = source.join(
            parts_manifest
                .as_str()
                .ok_or("geometry mesh has no parts_manifest")?,
        );
        let manifest: Value = read_json(&manifest_path)?;
        let parent = manifest_path
            .parent()
            .ok_or("parts manifest has no parent")?;
        for partition in manifest["partitions"]
            .as_array()
            .ok_or("parts manifest has no partitions")?
        {
            partition_sources.push(
                parent.join(
                    partition["meshlet"]
                        .as_str()
                        .ok_or("partition has no meshlet geometry")?,
                ),
            );
        }
        plan.push(PlannedMesh {
            mesh_index,
            object,
            usage,
            manifest,
        });
    }

    // `MeshletMesh::from_mesh` already spreads its group simplification over the
    // compute task pool, but the per-partition read, parse, meshlet build and
    // encode phases are serial. Overlap those across a bounded number of
    // partitions and consume the results in manifest order, which is what makes
    // the packed bundle bytes independent of thread timing.
    let packed = OrderedWork::new(partition_sources.len(), settings.geometry_jobs);
    let convert = |index: usize| {
        pack_partition(&partition_sources[index], settings.raytracing_error)
            .map_err(|error| error.to_string())
    };
    std::thread::scope(|scope| {
        for _ in 0..settings.geometry_jobs {
            scope.spawn(|| packed.run(&convert));
        }
        let result = write_planned_meshes(&mut plan, meshes, bundles, &packed, settings, totals);
        // Release any worker still parked on the lookahead limit, otherwise the
        // scope cannot join them after an early error.
        packed.finish();
        result
    })
}

fn pack_partition(
    meshlet_path: &Path,
    raytracing_error: f32,
) -> Result<PackedPartition, Box<dyn std::error::Error>> {
    let meshlet_source = fs::read(meshlet_path)?;
    let mut mesh = mesh_from_converter_glb(&meshlet_source, false)?;
    let winding_repaired = repair_inverted_winding(&mut mesh)?;
    let meshlet = MeshletMesh::from_mesh(&mesh, 4)?;
    let raytracing = meshlet.raytracing_geometry(raytracing_error);
    if raytracing.indices.is_empty() {
        return Err(format!(
            "meshlet LOD produced no BLAS triangles: {}",
            meshlet_path.display()
        )
        .into());
    }
    Ok(PackedPartition {
        blas_vertices: raytracing.positions.len() as u64,
        blas_triangles: (raytracing.indices.len() / 3) as u64,
        raytracing: encode_meshlet_blas(&raytracing)?,
        meshlet: encode_meshlet(&meshlet)?,
        winding_repaired,
    })
}

fn write_planned_meshes(
    plan: &mut [PlannedMesh],
    meshes: &mut [Value],
    bundles: &mut BundleWriter,
    packed: &OrderedWork<PackedPartition>,
    settings: &PackSettings,
    totals: &mut GeometryTotals,
) -> Result<(), Box<dyn std::error::Error>> {
    let mesh_count = meshes.len();
    let mut current_usage = None;
    for planned in plan.iter_mut() {
        if current_usage != Some(planned.usage) {
            bundles.finish_current()?;
            current_usage = Some(planned.usage);
            println!("ZORAH_BUNDLE_GEOMETRY_GROUP usage={:03b}", planned.usage);
        }
        let partitions = planned.manifest["partitions"]
            .as_array_mut()
            .ok_or("parts manifest has no partitions")?;
        for partition in partitions.iter_mut() {
            totals.raster_triangles = totals.raster_triangles.saturating_add(
                partition["triangles"]
                    .as_u64()
                    .ok_or("partition has no triangle count")?,
            );
            let converted = packed.next()?;
            if converted.winding_repaired {
                totals.winding_repairs += 1;
            }
            bundles.ensure_room(
                (converted.raytracing.len() as u64).saturating_add(converted.meshlet.len() as u64),
            )?;

            let id = format!("g/{:06}", totals.partitions);
            let mesh_reference = bundles.add(
                format!("{id}/meshlet_blas"),
                BundleEntryKind::MeshletBlas,
                &converted.raytracing,
            )?;
            partition["geometry"] = Value::String(mesh_reference.clone());
            partition["mesh"] = Value::String(mesh_reference);
            partition["meshlet"] = Value::String(bundles.add(
                format!("{id}/meshlet"),
                BundleEntryKind::Meshlet,
                &converted.meshlet,
            )?);
            partition["blas_vertices"] = Value::from(converted.blas_vertices);
            partition["blas_triangles"] = Value::from(converted.blas_triangles);
            // The requested LOD bound, which every selected meshlet satisfies.
            // `raytracing_geometry` does not report the selected cut's own
            // maximum error, so this stays an upper bound.
            partition["blas_achieved_error"] = Value::from(settings.raytracing_error as f64);
            totals.raytracing_triangles = totals
                .raytracing_triangles
                .saturating_add(converted.blas_triangles);
            totals.raytracing_vertices = totals
                .raytracing_vertices
                .saturating_add(converted.blas_vertices);
            totals.partitions += 1;
        }
        let mesh = meshes
            .get_mut(planned.mesh_index)
            .ok_or("planned mesh is out of range")?;
        let record = mesh
            .as_object_mut()
            .ok_or("geometry mesh is not an object")?;
        record.remove("parts_manifest");
        // Reuse in later generations hard-links these payloads instead of
        // reading them, so the formats that produced them are recorded here.
        record.insert(
            "meshlet_asset_version".to_string(),
            Value::from(MESHLET_MESH_ASSET_VERSION),
        );
        record.insert(
            "bundle_format_version".to_string(),
            Value::from(ZORAH_BUNDLE_VERSION),
        );
        record.insert(
            "pack_pipeline_version".to_string(),
            Value::from(PACK_PIPELINE_VERSION),
        );
        mesh["partitions"] = planned.manifest["partitions"].take();
        println!(
            "ZORAH_BUNDLE_GEOMETRY {}/{} mesh={} partitions={}",
            planned.mesh_index + 1,
            mesh_count,
            planned.object,
            mesh["partitions"].as_array().map_or(0, Vec::len),
        );
    }
    Ok(())
}

fn reusable_geometry_mesh(
    source: &Path,
    mesh: &Value,
    previous: &Value,
    raytracing_error: f32,
) -> Result<bool, Box<dyn std::error::Error>> {
    if mesh["asset_id"] != previous["asset_id"] {
        return Ok(false);
    }
    // Payloads are hard-linked, never re-read, so a format or algorithm change
    // is only visible in these stamps.
    if previous["meshlet_asset_version"].as_u64() != Some(MESHLET_MESH_ASSET_VERSION)
        || previous["bundle_format_version"].as_u64() != Some(ZORAH_BUNDLE_VERSION as u64)
        || previous["pack_pipeline_version"].as_u64() != Some(PACK_PIPELINE_VERSION as u64)
    {
        return Ok(false);
    }
    let Some(parts_manifest) = mesh["parts_manifest"].as_str() else {
        return Ok(false);
    };
    let manifest = read_json(&source.join(parts_manifest))?;
    let source_partitions = manifest["partitions"]
        .as_array()
        .ok_or("parts manifest has no partitions")?;
    let Some(previous_partitions) = previous["partitions"].as_array() else {
        return Ok(false);
    };
    if source_partitions.len() != previous_partitions.len() {
        return Ok(false);
    }
    // material_index rides along in the packed record but is invariant to the
    // shas, so a section-map-only edit would otherwise splice the stale one in.
    const IDENTITY_KEYS: [&str; 6] = [
        "material_slot",
        "material_index",
        "triangles",
        "vertices",
        "meshlet_sha256",
        "geometry_sha256",
    ];
    for (source_partition, previous_partition) in source_partitions.iter().zip(previous_partitions)
    {
        if IDENTITY_KEYS
            .iter()
            .any(|key| source_partition[*key] != previous_partition[*key])
        {
            return Ok(false);
        }
        let previous_error = previous_partition["blas_achieved_error"]
            .as_f64()
            .unwrap_or(f64::NAN) as f32;
        let tolerance = 1.0e-6_f32.max(raytracing_error.abs() * 1.0e-5);
        if !previous_error.is_finite()
            || (previous_error - raytracing_error).abs() > tolerance
            || !previous_partition["geometry"]
                .as_str()
                .is_some_and(|value| value.starts_with("bundles/") && value.contains('#'))
            || !previous_partition["meshlet"]
                .as_str()
                .is_some_and(|value| value.starts_with("bundles/") && value.contains('#'))
        {
            return Ok(false);
        }
    }
    Ok(true)
}

struct PendingTexture {
    index: usize,
    source_path: PathBuf,
    srgb: bool,
    normal_map: bool,
    source_size: u64,
    source_modified_ns: u64,
}

/// Repair intermediates written by early versions of the Zorah converter.
///
/// Those versions reflected UE coordinates into Bevy space and also reversed
/// the indices, leaving extracted meshes inside-out. Generated Bevy-space
/// fallback meshes were already correct, so detect the defect from face and
/// vertex normal agreement instead of unconditionally flipping every input.
fn repair_inverted_winding(mesh: &mut Mesh) -> Result<bool, Box<dyn std::error::Error>> {
    let should_invert = {
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            return Err("mesh positions are not float3".into());
        };
        let Some(VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            return Err("mesh normals are not float3".into());
        };
        let Some(indices) = mesh.indices() else {
            return Err("mesh has no indices".into());
        };

        let sampled_indices = indices
            .iter()
            .take(8192 * 3)
            .map(|index| index as usize)
            .collect::<Vec<_>>();
        let mut aligned = 0usize;
        let mut opposed = 0usize;
        for triangle in sampled_indices.chunks_exact(3) {
            let [i0, i1, i2] = triangle else {
                unreachable!();
            };
            if [*i0, *i1, *i2]
                .into_iter()
                .any(|index| index >= positions.len() || index >= normals.len())
            {
                return Err("mesh index is out of bounds".into());
            }
            let (p0, p1, p2) = (positions[*i0], positions[*i1], positions[*i2]);
            let (n0, n1, n2) = (normals[*i0], normals[*i1], normals[*i2]);
            let face = (Vec3::from(p1) - Vec3::from(p0)).cross(Vec3::from(p2) - Vec3::from(p0));
            let shading = Vec3::from(n0) + Vec3::from(n1) + Vec3::from(n2);
            let denominator = face.length() * shading.length();
            if denominator <= 1e-12 {
                continue;
            }
            let agreement = face.dot(shading) / denominator;
            if agreement > 0.1 {
                aligned += 1;
            } else if agreement < -0.1 {
                opposed += 1;
            }
        }
        opposed != 0 && opposed > aligned.saturating_mul(3)
    };

    if should_invert {
        mesh.invert_winding()?;
    }
    Ok(should_invert)
}

fn scene_mesh_usage(source: &Path) -> Result<HashMap<String, u8>, Box<dyn std::error::Error>> {
    let mut usage = HashMap::new();
    for (level_index, level) in LEVELS.into_iter().enumerate() {
        let scene: Value = read_json(&source.join("scenes").join(format!("{level}.json")))?;
        let actors = scene["actors"]
            .as_array()
            .ok_or_else(|| format!("scene {level} has no actors array"))?;
        for actor in actors {
            let Some(components) = actor["components"].as_array() else {
                continue;
            };
            for component in components {
                let Some(mesh) = component["mesh"].as_str() else {
                    continue;
                };
                *usage.entry(mesh.to_string()).or_insert(0) |= 1 << level_index;
            }
        }
    }
    Ok(usage)
}

fn modified_ns(metadata: &fs::Metadata) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(metadata
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos()
        .try_into()?)
}

const PACKED_TEXTURE_STAMPS: [&str; 5] = [
    "packed_source_size",
    "packed_source_modified_ns",
    "packed_texture_quality",
    "packed_bundle_format_version",
    "packed_pack_pipeline_version",
];

fn normalized_texture_record(record: &Value) -> Value {
    let mut normalized = record.clone();
    if let Some(object) = normalized.as_object_mut() {
        for key in ["output", "resumed"].iter().chain(&PACKED_TEXTURE_STAMPS) {
            object.remove(*key);
        }
    }
    normalized
}

/// Record what this pack produced the payload from, so a later generation can
/// tell whether hard-linking it forward is still correct.
fn stamp_texture_record(
    texture: &mut Value,
    source_size: u64,
    source_modified_ns: u64,
    settings: &PackSettings,
) {
    texture["packed_source_size"] = Value::from(source_size);
    texture["packed_source_modified_ns"] = Value::from(source_modified_ns);
    texture["packed_texture_quality"] = Value::String(settings.texture_quality_name.clone());
    texture["packed_bundle_format_version"] = Value::from(ZORAH_BUNDLE_VERSION);
    texture["packed_pack_pipeline_version"] = Value::from(PACK_PIPELINE_VERSION);
}

fn reusable_texture(
    current: &Value,
    previous: &Value,
    source_size: u64,
    source_modified_ns: u64,
    texture_quality: &str,
) -> bool {
    if normalized_texture_record(current) != normalized_texture_record(previous) {
        return false;
    }
    let Some(reference) = previous["output"].as_str() else {
        return false;
    };
    if !reference.starts_with("bundles/") || !reference.contains('#') {
        return false;
    }
    // Compressed payloads are hard-linked, never decoded, so the encoder
    // settings and container format that produced them are only visible here.
    // Trees packed before these stamps existed are recompressed once.
    previous["packed_texture_quality"].as_str() == Some(texture_quality)
        && previous["packed_bundle_format_version"].as_u64() == Some(ZORAH_BUNDLE_VERSION as u64)
        && previous["packed_pack_pipeline_version"].as_u64() == Some(PACK_PIPELINE_VERSION as u64)
        && previous["packed_source_size"].as_u64() == Some(source_size)
        && previous["packed_source_modified_ns"].as_u64() == Some(source_modified_ns)
}

fn bundle_file(reference: &str) -> Option<&str> {
    let (path, _) = reference.split_once('#')?;
    path.strip_prefix("bundles/")
        .filter(|name| !name.is_empty() && !name.contains(['/', '\\']))
}

fn bundle_label(reference: &str) -> Option<&str> {
    reference.split_once('#').map(|(_, label)| label)
}

fn link_bundle_file(file_name: &str, source: &Path, output: &Path) -> std::io::Result<()> {
    let source = source.join("bundles").join(file_name);
    let destination = output.join("bundles").join(file_name);
    if destination.exists() {
        return Ok(());
    }
    match fs::hard_link(&source, &destination) {
        Ok(()) => Ok(()),
        Err(_) => fs::copy(source, destination).map(|_| ()),
    }
}

/// Copy every shard the reused records point at into the new tree.
///
/// A shard whose entries are all still referenced is hard-linked. One that also
/// holds entries this generation replaced is rewritten without them: the
/// runtime decodes and retains every entry of every bundle it loads, so linking
/// such a shard forward would carry its dead payloads through each generation.
fn link_reused_bundles(
    references: &HashSet<String>,
    source: &Path,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut live: BTreeMap<&str, HashSet<&str>> = BTreeMap::new();
    for reference in references {
        let (Some(file_name), Some(label)) = (bundle_file(reference), bundle_label(reference))
        else {
            return Err(format!("invalid bundle reference: {reference}").into());
        };
        live.entry(file_name).or_default().insert(label);
    }
    for (file_name, labels) in live {
        let path = source.join("bundles").join(file_name);
        let (index, payload_offset) = read_bundle_index(&path)?;
        if index.entries.len() == labels.len() {
            link_bundle_file(file_name, source, output)?;
            continue;
        }
        let dropped = index.entries.len() - labels.len();
        let dropped_bytes = rewrite_bundle_without(
            &path,
            &index,
            payload_offset,
            &labels,
            &output.join("bundles").join(file_name),
        )?;
        println!(
            "ZORAH_BUNDLE_COMPACTED bundle={file_name} kept={} dropped={dropped} dropped_bytes={dropped_bytes}",
            labels.len(),
        );
    }
    Ok(())
}

fn read_bundle_index(path: &Path) -> Result<(BundleIndex, u64), Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut magic = [0; 8];
    file.read_exact(&mut magic)?;
    let mut version = [0; 4];
    file.read_exact(&mut version)?;
    let mut index_length = [0; 8];
    file.read_exact(&mut index_length)?;
    if magic != ZORAH_BUNDLE_MAGIC || u32::from_le_bytes(version) != ZORAH_BUNDLE_VERSION {
        return Err(format!(
            "reused bundle is not a current Zorah bundle: {}",
            path.display()
        )
        .into());
    }
    let mut index_bytes = vec![0; usize::try_from(u64::from_le_bytes(index_length))?];
    file.read_exact(&mut index_bytes)?;
    let index: BundleIndex = serde_json::from_slice(&index_bytes)?;
    Ok((index, file.stream_position()?))
}

fn rewrite_bundle_without(
    source: &Path,
    index: &BundleIndex,
    payload_offset: u64,
    labels: &HashSet<&str>,
    destination: &Path,
) -> Result<u64, Box<dyn std::error::Error>> {
    let mut kept = Vec::new();
    let mut ranges = Vec::new();
    let mut dropped_bytes = 0u64;
    let mut offset = payload_offset;
    for entry in &index.entries {
        if labels.contains(entry.label.as_str()) {
            ranges.push((offset, entry.byte_length));
            kept.push(entry.clone());
        } else {
            dropped_bytes = dropped_bytes.saturating_add(entry.byte_length);
        }
        offset = offset
            .checked_add(entry.byte_length)
            .ok_or("bundle payload offset overflow")?;
    }
    if kept.len() != labels.len() {
        return Err(format!(
            "reused bundle is missing referenced entries: {}",
            source.display()
        )
        .into());
    }
    let kept_index = BundleIndex {
        format_version: ZORAH_BUNDLE_VERSION,
        entries: kept,
    };
    let kept_index = serde_json::to_vec(&kept_index)?;
    let mut reader = File::open(source)?;
    let mut writer = File::create(destination)?;
    writer.write_all(&ZORAH_BUNDLE_MAGIC)?;
    writer.write_all(&ZORAH_BUNDLE_VERSION.to_le_bytes())?;
    writer.write_all(&(kept_index.len() as u64).to_le_bytes())?;
    writer.write_all(&kept_index)?;
    for (offset, byte_length) in ranges {
        reader.seek(SeekFrom::Start(offset))?;
        let copied = std::io::copy(&mut (&mut reader).take(byte_length), &mut writer)?;
        if copied != byte_length {
            return Err(format!("reused bundle is truncated: {}", source.display()).into());
        }
    }
    writer.flush()?;
    Ok(dropped_bytes)
}

fn collect_bundle_references(value: &Value, references: &mut HashSet<String>) {
    match value {
        Value::String(value) if value.starts_with("bundles/") && value.contains('#') => {
            references.insert(value.clone());
        }
        Value::Array(values) => {
            for value in values {
                collect_bundle_references(value, references);
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                collect_bundle_references(value, references);
            }
        }
        _ => {}
    }
}

fn next_bundle_id(root: &Path) -> usize {
    root.join("bundles")
        .read_dir()
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter_map(|name| {
            name.strip_prefix("zorah-")?
                .strip_suffix(".zorah_bundle")?
                .parse::<usize>()
                .ok()
        })
        .max()
        .map_or(0, |id| id + 1)
}

/// A fixed set of numbered jobs run on a worker pool and delivered in order.
///
/// Bundle bytes depend on the order entries are added, so results are handed to
/// the single writing thread by index no matter which worker finished first.
/// Workers stop claiming jobs once the pool runs `lookahead` results ahead of
/// the writer, which bounds how many encoded payloads are held in memory.
struct OrderedWork<T> {
    state: Mutex<OrderedState<T>>,
    progress: Condvar,
    count: usize,
    lookahead: usize,
}

struct OrderedState<T> {
    claimed: usize,
    delivered: usize,
    finished: bool,
    results: BTreeMap<usize, Result<T, String>>,
}

impl<T: Send> OrderedWork<T> {
    fn new(count: usize, workers: usize) -> Self {
        Self {
            state: Mutex::new(OrderedState {
                claimed: 0,
                delivered: 0,
                finished: false,
                results: BTreeMap::new(),
            }),
            progress: Condvar::new(),
            count,
            lookahead: workers.max(1) * 2,
        }
    }

    fn run(&self, work: &(impl Fn(usize) -> Result<T, String> + Sync)) {
        loop {
            let mut state = self.state.lock().unwrap();
            while !state.finished
                && state.claimed != self.count
                && state.claimed >= state.delivered + self.lookahead
            {
                state = self.progress.wait(state).unwrap();
            }
            if state.finished || state.claimed == self.count {
                return;
            }
            let index = state.claimed;
            state.claimed += 1;
            drop(state);

            let result = std::panic::catch_unwind(AssertUnwindSafe(|| work(index)))
                .unwrap_or_else(|_| Err(format!("job {index} panicked")));
            let mut state = self.state.lock().unwrap();
            state.finished |= result.is_err();
            state.results.insert(index, result);
            self.progress.notify_all();
        }
    }

    fn next(&self) -> Result<T, Box<dyn std::error::Error>> {
        let mut state = self.state.lock().unwrap();
        let index = state.delivered;
        if index == self.count {
            return Err("requested more results than jobs".into());
        }
        let result = loop {
            if let Some(result) = state.results.remove(&index) {
                break result;
            }
            state = self.progress.wait(state).unwrap();
        };
        state.delivered = index + 1;
        self.progress.notify_all();
        drop(state);
        result.map_err(Into::into)
    }

    fn finish(&self) {
        self.state.lock().unwrap().finished = true;
        self.progress.notify_all();
    }
}

fn read_json(path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn write_json(path: &Path, value: &Value) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn encode_meshlet(meshlet: &MeshletMesh) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut writer = VecWriter::default();
    block_on(MeshletMeshSaver.save(
        &mut writer,
        SavedAsset::from_asset(meshlet),
        &(),
        AssetPath::from("partition.meshlet_mesh"),
    ))?;
    Ok(writer.bytes)
}

fn encode_meshlet_blas(
    geometry: &bevy::pbr::experimental::meshlet::MeshletRaytracingGeometry,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let vertex_count = u32::try_from(geometry.positions.len())?;
    let index_count = u32::try_from(geometry.indices.len())?;
    if geometry.normals.len() != vertex_count as usize
        || geometry.uvs.len() != vertex_count as usize
        || index_count % 3 != 0
    {
        return Err("invalid meshlet BLAS geometry".into());
    }
    let mut bytes = Vec::with_capacity(
        20 + vertex_count as usize * 32 + index_count as usize * size_of::<u32>(),
    );
    bytes.extend_from_slice(b"ZBLAS001");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&vertex_count.to_le_bytes());
    bytes.extend_from_slice(&index_count.to_le_bytes());
    for ((position, normal), uv) in geometry
        .positions
        .iter()
        .zip(&geometry.normals)
        .zip(&geometry.uvs)
    {
        for value in position.iter().chain(normal).chain(uv) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    for index in &geometry.indices {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{
        mesh::{Indices, VertexAttributeValues},
        pbr::experimental::meshlet::MeshletRaytracingGeometry,
        prelude::Mesh,
        render::render_resource::PrimitiveTopology,
    };

    #[test]
    fn meshlet_blas_payload_round_trips_compact_vertices() {
        let source = MeshletRaytracingGeometry {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
        };
        let bytes = encode_meshlet_blas(&source).unwrap();
        let mesh = zorah_bundle::meshlet_blas_from_bytes(&bytes).unwrap();

        assert_eq!(mesh.count_vertices(), 3);
        assert_eq!(mesh.indices().unwrap().len(), 3);
        assert!(matches!(
            mesh.attribute(Mesh::ATTRIBUTE_POSITION),
            Some(VertexAttributeValues::Float32x3(values)) if values == &source.positions
        ));
        assert!(mesh.attribute(Mesh::ATTRIBUTE_TANGENT).is_none());
        assert_eq!(mesh.get_vertex_size(), 32);
    }

    #[test]
    fn repairs_only_winding_that_opposes_vertex_normals() {
        let make_mesh = |indices: [u32; 3], triangle_count: usize| {
            Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
                .with_inserted_attribute(
                    Mesh::ATTRIBUTE_POSITION,
                    vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                )
                .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 3])
                .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; 3])
                .with_inserted_indices(Indices::U32(indices.repeat(triangle_count)))
        };

        // Some extracted Zorah partitions contain fewer than the old 16-triangle
        // confidence threshold, including single-triangle partitions.
        let mut inverted = make_mesh([0, 2, 1], 1);
        assert!(repair_inverted_winding(&mut inverted).unwrap());
        assert_eq!(
            inverted
                .indices()
                .unwrap()
                .iter()
                .take(3)
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );

        // Generated Bevy-space fallback primitives are already aligned and must
        // not be flipped, even when they are equally small.
        let mut aligned = make_mesh([0, 1, 2], 1);
        assert!(!repair_inverted_winding(&mut aligned).unwrap());
        assert_eq!(
            aligned
                .indices()
                .unwrap()
                .iter()
                .take(3)
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
    }

    fn packed_texture(extra: Value) -> Value {
        let mut record = serde_json::json!({
            "object": "texture",
            "output": "bundles/zorah-002.zorah_bundle#t/0000",
            "srgb": true,
            "packed_source_size": 100,
            "packed_source_modified_ns": 200,
            "packed_texture_quality": DEFAULT_TEXTURE_QUALITY,
            "packed_bundle_format_version": ZORAH_BUNDLE_VERSION,
            "packed_pack_pipeline_version": PACK_PIPELINE_VERSION,
        });
        for (key, value) in extra.as_object().unwrap() {
            record[key] = value.clone();
        }
        record
    }

    #[test]
    fn texture_reuse_ignores_runtime_location_but_not_conversion_metadata() {
        let current = serde_json::json!({
            "object": "texture",
            "output": "textures/texture.png",
            "output_size": [1024, 1024],
            "srgb": true,
            "normal_map": false,
            "resumed": true,
        });
        let previous = packed_texture(serde_json::json!({
            "output_size": [1024, 1024],
            "normal_map": false,
        }));
        assert!(reusable_texture(
            &current,
            &previous,
            100,
            200,
            DEFAULT_TEXTURE_QUALITY
        ));

        let mut changed = current;
        changed["srgb"] = Value::Bool(false);
        assert!(!reusable_texture(
            &changed,
            &previous,
            100,
            200,
            DEFAULT_TEXTURE_QUALITY
        ));
    }

    #[test]
    fn stamped_texture_reuse_requires_unchanged_source_stat() {
        let current = serde_json::json!({
            "object": "texture",
            "output": "texture.png",
            "srgb": true,
        });
        let previous = packed_texture(Value::Object(Default::default()));
        assert!(reusable_texture(
            &current,
            &previous,
            100,
            200,
            DEFAULT_TEXTURE_QUALITY
        ));
        assert!(!reusable_texture(
            &current,
            &previous,
            101,
            200,
            DEFAULT_TEXTURE_QUALITY
        ));
        assert!(!reusable_texture(
            &current,
            &previous,
            100,
            201,
            DEFAULT_TEXTURE_QUALITY
        ));
    }

    #[test]
    fn texture_reuse_requires_matching_pack_stamps() {
        let current = serde_json::json!({
            "object": "texture",
            "output": "texture.png",
            "srgb": true,
        });
        // A different encoder preset or bundle format produced different bytes,
        // and the payload is hard-linked rather than re-encoded.
        assert!(!reusable_texture(
            &current,
            &packed_texture(Value::Object(Default::default())),
            100,
            200,
            "slow"
        ));
        assert!(!reusable_texture(
            &current,
            &packed_texture(serde_json::json!({
                "packed_bundle_format_version": ZORAH_BUNDLE_VERSION + 1
            })),
            100,
            200,
            DEFAULT_TEXTURE_QUALITY
        ));
        assert!(!reusable_texture(
            &current,
            &packed_texture(serde_json::json!({
                "packed_pack_pipeline_version": PACK_PIPELINE_VERSION + 1
            })),
            100,
            200,
            DEFAULT_TEXTURE_QUALITY
        ));
        // Trees packed before the stamps existed cannot be validated at all.
        let unstamped = serde_json::json!({
            "object": "texture",
            "output": "bundles/zorah-002.zorah_bundle#t/0000",
            "srgb": true,
        });
        assert!(!reusable_texture(
            &current,
            &unstamped,
            100,
            200,
            DEFAULT_TEXTURE_QUALITY
        ));
    }

    #[test]
    fn geometry_reuse_requires_matching_format_versions() {
        let source = tempdir();
        let manifest = source.join("parts.json");
        let partitions = serde_json::json!({
            "partitions": [{
                "material_slot": 0,
                "triangles": 12,
                "vertices": 24,
                "meshlet_sha256": "aa",
                "geometry_sha256": "bb",
            }],
        });
        fs::write(&manifest, serde_json::to_vec(&partitions).unwrap()).unwrap();
        let mesh = serde_json::json!({
            "object": "mesh",
            "asset_id": "id",
            "parts_manifest": "parts.json",
        });
        let mut previous = serde_json::json!({
            "object": "mesh",
            "asset_id": "id",
            "meshlet_asset_version": MESHLET_MESH_ASSET_VERSION,
            "bundle_format_version": ZORAH_BUNDLE_VERSION,
            "pack_pipeline_version": PACK_PIPELINE_VERSION,
            "partitions": [{
                "material_slot": 0,
                "triangles": 12,
                "vertices": 24,
                "meshlet_sha256": "aa",
                "geometry_sha256": "bb",
                "blas_achieved_error": 0.02,
                "geometry": "bundles/zorah-000.zorah_bundle#g/000000/meshlet_blas",
                "meshlet": "bundles/zorah-000.zorah_bundle#g/000000/meshlet",
            }],
        });
        assert!(reusable_geometry_mesh(&source, &mesh, &previous, 0.02).unwrap());

        previous["meshlet_asset_version"] = Value::from(MESHLET_MESH_ASSET_VERSION + 1);
        assert!(!reusable_geometry_mesh(&source, &mesh, &previous, 0.02).unwrap());
        previous["meshlet_asset_version"] = Value::from(MESHLET_MESH_ASSET_VERSION);
        previous["pack_pipeline_version"] = Value::from(PACK_PIPELINE_VERSION + 1);
        assert!(!reusable_geometry_mesh(&source, &mesh, &previous, 0.02).unwrap());

        // Trees packed before the stamps existed cannot be validated at all.
        let unstamped = previous
            .as_object()
            .unwrap()
            .iter()
            .filter(|(key, _)| !key.ends_with("_version"))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        assert!(!reusable_geometry_mesh(&source, &mesh, &Value::Object(unstamped), 0.02).unwrap());
        fs::remove_dir_all(&source).unwrap();
    }

    #[test]
    fn geometry_reuse_rejects_a_changed_material_index() {
        let source = tempdir();
        let manifest = source.join("parts.json");
        let write_manifest = |partition: Value| {
            fs::write(
                &manifest,
                serde_json::to_vec(&serde_json::json!({ "partitions": [partition] })).unwrap(),
            )
            .unwrap();
        };
        write_manifest(serde_json::json!({
            "material_slot": 0,
            "material_index": 2,
            "triangles": 12,
            "vertices": 24,
            "meshlet_sha256": "aa",
            "geometry_sha256": "bb",
        }));
        let mesh = serde_json::json!({
            "object": "mesh",
            "asset_id": "id",
            "parts_manifest": "parts.json",
        });
        let mut previous = serde_json::json!({
            "object": "mesh",
            "asset_id": "id",
            "meshlet_asset_version": MESHLET_MESH_ASSET_VERSION,
            "bundle_format_version": ZORAH_BUNDLE_VERSION,
            "pack_pipeline_version": PACK_PIPELINE_VERSION,
            "partitions": [{
                "material_slot": 0,
                "material_index": 2,
                "triangles": 12,
                "vertices": 24,
                "meshlet_sha256": "aa",
                "geometry_sha256": "bb",
                "blas_achieved_error": 0.02,
                "geometry": "bundles/zorah-000.zorah_bundle#g/000000/meshlet_blas",
                "meshlet": "bundles/zorah-000.zorah_bundle#g/000000/meshlet",
            }],
        });
        assert!(reusable_geometry_mesh(&source, &mesh, &previous, 0.02).unwrap());

        // A section-map edit leaves every sha untouched, so material_index is
        // the only thing separating the packed record from the source.
        previous["partitions"][0]["material_index"] = Value::from(1);
        assert!(!reusable_geometry_mesh(&source, &mesh, &previous, 0.02).unwrap());

        // Manifests and records predating material_index compare Null to Null.
        write_manifest(serde_json::json!({
            "material_slot": 0,
            "triangles": 12,
            "vertices": 24,
            "meshlet_sha256": "aa",
            "geometry_sha256": "bb",
        }));
        previous["partitions"][0]
            .as_object_mut()
            .unwrap()
            .remove("material_index");
        assert!(reusable_geometry_mesh(&source, &mesh, &previous, 0.02).unwrap());
        fs::remove_dir_all(&source).unwrap();
    }

    #[test]
    fn compaction_drops_entries_the_new_generation_replaced() {
        let root = tempdir();
        let mut writer = BundleWriter::new(&root, 1 << 30, 0).unwrap();
        let kept = writer
            .add(
                "t/0000".to_string(),
                BundleEntryKind::Image { srgb: true },
                b"kept",
            )
            .unwrap();
        writer
            .add(
                "t/0001".to_string(),
                BundleEntryKind::Image { srgb: true },
                b"stale payload",
            )
            .unwrap();
        writer.finish().unwrap();

        let output = root.join("output");
        fs::create_dir_all(output.join("bundles")).unwrap();
        link_reused_bundles(&HashSet::from([kept.clone()]), &root, &output).unwrap();
        let file_name = bundle_file(&kept).unwrap();
        let (index, payload_offset) =
            read_bundle_index(&output.join("bundles").join(file_name)).unwrap();
        assert_eq!(index.entries.len(), 1);
        assert_eq!(index.entries[0].label, "t/0000");
        assert_eq!(
            fs::metadata(output.join("bundles").join(file_name))
                .unwrap()
                .len(),
            payload_offset + 4
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn ordered_work_delivers_results_in_job_order() {
        let jobs = 64;
        let work = OrderedWork::new(jobs, 4);
        // Later jobs finish first, so any delivery by completion order fails.
        let compute = |index: usize| {
            std::thread::sleep(std::time::Duration::from_micros((jobs - index) as u64 * 20));
            Ok(index)
        };
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| work.run(&compute));
            }
            for index in 0..jobs {
                assert_eq!(work.next().unwrap(), index);
            }
            work.finish();
        });
    }

    #[test]
    fn ordered_work_reports_failed_and_panicking_jobs() {
        let failing = OrderedWork::new(4, 2);
        let compute = |index: usize| {
            if index == 2 {
                Err("job failed".to_string())
            } else {
                Ok(index)
            }
        };
        std::thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| failing.run(&compute));
            }
            assert_eq!(failing.next().unwrap(), 0);
            assert_eq!(failing.next().unwrap(), 1);
            assert_eq!(failing.next().unwrap_err().to_string(), "job failed");
            failing.finish();
        });

        // A panicking job must surface as an error instead of stalling the
        // writer that is waiting for its result.
        let panicking = OrderedWork::new(2, 1);
        let compute = |index: usize| {
            assert!(index == 0, "job panicked");
            Ok(index)
        };
        std::thread::scope(|scope| {
            scope.spawn(|| panicking.run(&compute));
            assert_eq!(panicking.next().unwrap(), 0);
            assert!(panicking.next().is_err());
            panicking.finish();
        });
    }

    fn tempdir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "zorah-pack-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }
}

struct CompressedTexture {
    bytes: Vec<u8>,
    /// Colour space of the encoded payload, which is not always the requested
    /// one: `Image::from_buffer` decodes 16-bit PNG sources to R16/Rgba16Unorm,
    /// formats with no sRGB variant, and `CompressedImageSaver` then ships them
    /// uncompressed. The runtime must not ask for an sRGB view of those.
    srgb: bool,
}

fn compress_texture(
    path: &Path,
    srgb: bool,
    normal_map: bool,
    quality: ImageCompressorQuality,
) -> Result<CompressedTexture, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let image = Image::from_buffer(
        &bytes,
        ImageType::Extension("png"),
        CompressedImageFormats::NONE,
        srgb,
        ImageSampler::Default,
        RenderAssetUsages::all(),
    )?;
    let format = image.texture_descriptor.format;
    let block_compressed = normal_map
        || matches!(
            format,
            TextureFormat::R8Unorm
                | TextureFormat::Rg8Unorm
                | TextureFormat::Rgba8Unorm
                | TextureFormat::Rgba8UnormSrgb
        );
    if !block_compressed || srgb != format.is_srgb() {
        println!(
            "ZORAH_BUNDLE_TEXTURE_UNCOMPRESSED texture={} format={format:?} srgb_requested={srgb} srgb_packed={}",
            path.display(),
            format.is_srgb(),
        );
    }
    let alpha = if normal_map {
        ImageCompressorAlphaMode::Opaque
    } else {
        ImageCompressorAlphaMode::Straight
    };
    let settings = CompressedImageSaverSettings {
        is_normal_map: normal_map,
        input_alpha_mode: alpha,
        output_alpha_mode: alpha,
        generate_mipmaps: true,
        quality,
    };
    let mut writer = VecWriter::default();
    block_on(CompressedImageSaver::default().save(
        &mut writer,
        SavedAsset::from_asset(&image),
        &settings,
        AssetPath::from("texture.ktx2"),
    ))?;
    Ok(CompressedTexture {
        bytes: writer.bytes,
        srgb: format.is_srgb(),
    })
}

#[derive(Default)]
struct VecWriter {
    bytes: Vec<u8>,
}

impl AsyncWrite for VecWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        self.bytes.extend_from_slice(bytes);
        Poll::Ready(Ok(bytes.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

struct BundleWriter {
    output: PathBuf,
    target_bytes: u64,
    next_id: usize,
    current: Option<ShardWriter>,
    stats: BundleStats,
}

#[derive(Default)]
struct BundleStats {
    bundles: usize,
    entries: usize,
    payload_bytes: u64,
}

impl BundleWriter {
    fn new(output: &Path, target_bytes: u64, next_id: usize) -> std::io::Result<Self> {
        fs::create_dir(output.join("bundles"))?;
        Ok(Self {
            output: output.to_path_buf(),
            target_bytes,
            next_id,
            current: None,
            stats: BundleStats::default(),
        })
    }

    fn add(
        &mut self,
        label: String,
        kind: BundleEntryKind,
        bytes: &[u8],
    ) -> std::io::Result<String> {
        self.ensure_room(bytes.len() as u64)?;
        if self.current.is_none() {
            let id = self.next_id;
            self.next_id += 1;
            self.current = Some(ShardWriter::new(&self.output, id)?);
        }
        let current = self.current.as_mut().unwrap();
        current.add(label.clone(), kind, bytes)?;
        Ok(format!("bundles/{}#{label}", current.file_name))
    }

    fn ensure_room(&mut self, bytes: u64) -> std::io::Result<()> {
        if self.current.as_ref().is_some_and(|current| {
            current.payload_bytes != 0
                && current.payload_bytes.saturating_add(bytes) > self.target_bytes
        }) {
            self.finish_current()?;
        }
        Ok(())
    }

    fn finish(mut self) -> std::io::Result<BundleStats> {
        self.finish_current()?;
        Ok(self.stats)
    }

    fn finish_current(&mut self) -> std::io::Result<()> {
        let Some(current) = self.current.take() else {
            return Ok(());
        };
        let entries = current.entries.len();
        let payload_bytes = current.payload_bytes;
        current.finish()?;
        self.stats.bundles += 1;
        self.stats.entries += entries;
        self.stats.payload_bytes += payload_bytes;
        Ok(())
    }
}

struct ShardWriter {
    file_name: String,
    destination: PathBuf,
    payload_path: PathBuf,
    payload: File,
    payload_bytes: u64,
    entries: Vec<BundleEntry>,
}

impl ShardWriter {
    fn new(output: &Path, id: usize) -> std::io::Result<Self> {
        let file_name = format!("zorah-{id:03}.zorah_bundle");
        let destination = output.join("bundles").join(&file_name);
        let payload_path = output.join("bundles").join(format!(".{file_name}.payload"));
        let payload = File::create(&payload_path)?;
        Ok(Self {
            file_name,
            destination,
            payload_path,
            payload,
            payload_bytes: 0,
            entries: Vec::new(),
        })
    }

    fn add(&mut self, label: String, kind: BundleEntryKind, bytes: &[u8]) -> std::io::Result<()> {
        self.payload.write_all(bytes)?;
        self.payload_bytes += bytes.len() as u64;
        self.entries.push(BundleEntry {
            label,
            byte_length: bytes.len() as u64,
            kind,
        });
        Ok(())
    }

    fn finish(mut self) -> std::io::Result<()> {
        self.payload.flush()?;
        drop(self.payload);
        let index = BundleIndex {
            format_version: ZORAH_BUNDLE_VERSION,
            entries: self.entries,
        };
        let index = serde_json::to_vec(&index).map_err(std::io::Error::other)?;
        let mut destination = File::create(&self.destination)?;
        destination.write_all(&ZORAH_BUNDLE_MAGIC)?;
        destination.write_all(&ZORAH_BUNDLE_VERSION.to_le_bytes())?;
        destination.write_all(&(index.len() as u64).to_le_bytes())?;
        destination.write_all(&index)?;
        let mut payload = File::open(&self.payload_path)?;
        std::io::copy(&mut payload, &mut destination)?;
        destination.flush()?;
        fs::remove_file(self.payload_path)?;
        Ok(())
    }
}

fn tempfile_path(parent: &Path, output_name: &std::ffi::OsStr) -> PathBuf {
    let mut name = std::ffi::OsString::from(".");
    name.push(output_name);
    name.push(format!(".pack.{}", std::process::id()));
    parent.join(name)
}
