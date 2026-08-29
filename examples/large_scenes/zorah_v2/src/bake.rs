//! Bakes the export's meshes into meshlet meshes plus their BLAS companions.
//!
//! The bake runs on plain std threads and knows nothing about a bevy `App`: a
//! caller starts it, polls [`BakeHandle::progress`] and drains
//! [`BakeHandle::try_recv`], and never blocks on it. Each mesh's finished
//! directory is checkpointed by its `manifest.json`, so an interrupted run
//! resumes at mesh granularity and a run with different settings rebakes
//! everything its stamps disagree with.
//!
//! Geometry is read through the per-mesh `meshes/<stem>.mesh.gltf` wrappers
//! with the `gltf` crate rather than the asset server, because bevy's glTF
//! loader registers every image of a file as a dependency and the wrappers
//! reference their 4k textures.

use std::{
    collections::HashMap,
    fs,
    io::Write,
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
    task::{Context, Poll},
    thread::JoinHandle,
    time::{Duration, Instant},
};

use bevy::{
    asset::{
        saver::{AssetSaver, SavedAsset},
        AssetPath, RenderAssetUsages,
    },
    math::Vec3,
    mesh::{Indices, Mesh},
    pbr::experimental::meshlet::{MeshletMesh, MeshletMeshSaver, MESHLET_MESH_ASSET_VERSION},
    render::render_resource::PrimitiveTopology,
    tasks::{block_on, AsyncComputeTaskPool, TaskPool},
};
use futures_io::AsyncWrite;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    blas,
    geometry::{partition_triangles, reindex, repair_inverted_winding, smooth_normals, Geometry},
};

/// Bumped whenever the bake's own logic changes what it writes (partition
/// rules, border locking, attribute synthesis, winding repair), so stale
/// caches rebake.
pub const BAKE_PIPELINE_VERSION: u32 = 2;
pub const MANIFEST_FILE: &str = "manifest.json";

/// How the bake runs and where it writes.
#[derive(Clone, Debug)]
pub struct BakeSettings {
    /// Directory holding the root glTF and its `meshes/` folder.
    pub scene_root: PathBuf,
    /// One subdirectory per mesh stem goes here.
    pub cache_dir: PathBuf,
    /// Threads baking meshes concurrently; each holds one decoded mesh plus
    /// its meshlet build in memory.
    pub workers: usize,
    /// Primitives above this many triangles are cut into spatial partitions
    /// before the meshlet build, which is superlinear in triangle count.
    pub partition_triangles: usize,
    /// Metres of geometric error the raytracing LOD cut may carry.
    pub raytracing_error: f32,
    /// Meshlet vertex position quantization factor (`MeshletMesh::from_mesh`).
    pub quantization: u8,
}

impl BakeSettings {
    /// The stamp a manifest must carry to be reused under these settings.
    pub fn stamp(&self) -> BakeStamp {
        BakeStamp {
            pipeline_version: BAKE_PIPELINE_VERSION,
            meshlet_asset_version: MESHLET_MESH_ASSET_VERSION,
            blas_version: blas::ZBLAS_VERSION,
            quantization: self.quantization,
            partition_triangles: self.partition_triangles,
            raytracing_error: self.raytracing_error,
        }
    }
}

/// One geometry file to bake. Root meshes that share a `.mesh.bin` are exact
/// duplicates in this export, so a job carries every mesh index using it.
#[derive(Clone, Debug)]
pub struct MeshJob {
    /// `<stem>.mesh.bin` / `<stem>.mesh.gltf` under `meshes/`; also the
    /// cache subdirectory name.
    pub stem: String,
    /// Root glTF mesh indices whose geometry this is, ascending.
    pub mesh_indices: Vec<usize>,
    /// The first mesh's name, for logs.
    pub name: String,
    /// Per primitive, as the root glTF describes it; the wrapper must agree.
    pub primitives: Vec<PrimitiveJob>,
}

/// What the root glTF says about one primitive.
#[derive(Clone, Debug)]
pub struct PrimitiveJob {
    /// Root glTF material index.
    pub material: Option<usize>,
    pub vertex_count: usize,
    /// Zero for an unindexed primitive.
    pub index_count: usize,
}

impl MeshJob {
    /// Total triangles across the job's primitives, before partitioning.
    pub fn triangles(&self) -> u64 {
        self.primitives
            .iter()
            .map(|primitive| {
                let indices = if primitive.index_count == 0 {
                    primitive.vertex_count
                } else {
                    primitive.index_count
                };
                indices as u64 / 3
            })
            .sum()
    }
}

/// Groups `mesh_indices` (ascending, deduplicated) of `document` by geometry
/// file. Fails when a mesh's primitives do not all live in one `.mesh.bin`.
pub fn plan_jobs(
    document: &gltf::Document,
    mesh_indices: impl IntoIterator<Item = usize>,
) -> Result<Vec<MeshJob>, BakeError> {
    let mut jobs: Vec<MeshJob> = Vec::new();
    let mut by_stem: HashMap<String, usize> = HashMap::new();
    let meshes = document.meshes().collect::<Vec<_>>();
    for mesh_index in mesh_indices {
        let mesh = meshes
            .get(mesh_index)
            .ok_or_else(|| BakeError::Invalid(format!("mesh {mesh_index} does not exist")))?;
        let name = mesh.name().unwrap_or_default().to_string();
        let mut stem = None;
        let mut primitives = Vec::new();
        for primitive in mesh.primitives() {
            let positions = primitive.get(&gltf::Semantic::Positions).ok_or_else(|| {
                BakeError::Invalid(format!(
                    "mesh {mesh_index} ({name}) has a primitive without positions"
                ))
            })?;
            let primitive_stem = buffer_stem(document, &positions).ok_or_else(|| {
                BakeError::Invalid(format!(
                    "mesh {mesh_index} ({name}) has positions outside a .mesh.bin"
                ))
            })?;
            match &stem {
                None => stem = Some(primitive_stem),
                Some(stem) if *stem != primitive_stem => {
                    return Err(BakeError::Invalid(format!(
                        "mesh {mesh_index} ({name}) spans {stem} and {primitive_stem}"
                    )));
                }
                Some(_) => {}
            }
            primitives.push(PrimitiveJob {
                material: primitive.material().index(),
                vertex_count: positions.count(),
                index_count: primitive.indices().map_or(0, |indices| indices.count()),
            });
        }
        let Some(stem) = stem else {
            // A mesh without primitives spawns nothing; skip it.
            continue;
        };
        match by_stem.get(&stem) {
            Some(&job) => jobs[job].mesh_indices.push(mesh_index),
            None => {
                by_stem.insert(stem.clone(), jobs.len());
                jobs.push(MeshJob {
                    stem,
                    mesh_indices: vec![mesh_index],
                    name,
                    primitives,
                });
            }
        }
    }
    Ok(jobs)
}

/// `meshes/<stem>.mesh.bin` -> `<stem>` for the buffer an accessor reads from.
///
/// A meshopt-compressed view points at its URI-less fallback buffer; the
/// file it really decodes from is the one its extension block names.
fn buffer_stem(document: &gltf::Document, accessor: &gltf::Accessor) -> Option<String> {
    let view = accessor.view()?;
    let buffer = match view.buffer().source() {
        gltf::buffer::Source::Uri(_) => view.buffer(),
        gltf::buffer::Source::Bin => {
            let index = view
                .extension_value("EXT_meshopt_compression")?
                .get("buffer")?
                .as_u64()?;
            document.buffers().nth(index as usize)?
        }
    };
    let gltf::buffer::Source::Uri(uri) = buffer.source() else {
        return None;
    };
    let file = uri.rsplit('/').next()?;
    file.strip_suffix(".mesh.bin").map(str::to_string)
}

/// The settings a manifest was written under; any difference forces a rebake.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BakeStamp {
    pub pipeline_version: u32,
    pub meshlet_asset_version: u64,
    pub blas_version: u32,
    pub quantization: u8,
    pub partition_triangles: usize,
    pub raytracing_error: f32,
}

/// The checkpoint of one baked mesh directory.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshManifest {
    pub stamp: BakeStamp,
    pub stem: String,
    pub mesh_name: String,
    pub source_triangles: u64,
    pub parts: Vec<PartManifest>,
}

/// One meshlet mesh plus BLAS companion in a mesh directory.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartManifest {
    /// Primitive index within the mesh.
    pub primitive: usize,
    /// Partition index within the primitive.
    pub partition: usize,
    /// Root glTF material index of the primitive, as the first mesh sharing
    /// the geometry declares it.
    pub material: Option<usize>,
    /// `p<primitive>_<partition>.meshlet_mesh`, relative to the directory.
    pub meshlet_file: String,
    /// `p<primitive>_<partition>.zblas`, relative to the directory.
    pub blas_file: String,
    pub triangles: u64,
    pub blas_triangles: u64,
    pub blas_achieved_error: f32,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub locked_borders: bool,
    pub winding_repaired: bool,
}

impl MeshManifest {
    /// Reads `<dir>/manifest.json` if it exists, matches `stamp`, and every
    /// part file it lists is present.
    pub fn reusable(dir: &Path, stamp: &BakeStamp) -> Option<MeshManifest> {
        let bytes = fs::read(dir.join(MANIFEST_FILE)).ok()?;
        let manifest: MeshManifest = serde_json::from_slice(&bytes).ok()?;
        if manifest.stamp != *stamp {
            return None;
        }
        let complete = manifest.parts.iter().all(|part| {
            dir.join(&part.meshlet_file).is_file() && dir.join(&part.blas_file).is_file()
        });
        complete.then_some(manifest)
    }

    /// Writes `<dir>/manifest.json` through a temp file so a crash mid-write
    /// leaves no half manifest to be mistaken for a checkpoint.
    pub fn write(&self, dir: &Path) -> Result<(), BakeError> {
        let temp = dir.join(format!("{MANIFEST_FILE}.tmp"));
        let mut file = fs::File::create(&temp)?;
        serde_json::to_writer_pretty(&mut file, self)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, dir.join(MANIFEST_FILE))?;
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum BakeError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Gltf(#[from] gltf::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("meshopt decode: {0}")]
    Meshopt(String),
    #[error("meshlet build: {0}")]
    Meshlet(String),
    #[error(transparent)]
    Blas(#[from] blas::ZblasError),
    #[error("{0}")]
    Invalid(String),
}

/// What a worker reports for one job.
#[derive(Debug)]
pub enum BakeEvent {
    /// The job's manifest is on disk, freshly baked or reused.
    Complete {
        job: usize,
        manifest: MeshManifest,
        reused: bool,
    },
    /// The job failed; the bake carries on with the others.
    Failed { job: usize, error: String },
}

/// A snapshot of the counters.
#[derive(Clone, Copy, Debug)]
pub struct BakeProgress {
    pub total: usize,
    /// Jobs finished either way: baked, reused or failed.
    pub finished: usize,
    pub baked: usize,
    pub reused: usize,
    pub failed: usize,
    /// Partitions written by fresh bakes so far.
    pub partitions: usize,
    pub elapsed: Duration,
}

impl BakeProgress {
    pub fn is_finished(&self) -> bool {
        self.finished >= self.total
    }
}

#[derive(Default)]
struct Counters {
    baked: AtomicUsize,
    reused: AtomicUsize,
    failed: AtomicUsize,
    partitions: AtomicUsize,
}

/// The running bake. Dropping it lets the workers finish their current mesh
/// and stop; they never outlive the process's interest in them for long.
pub struct BakeHandle {
    total: usize,
    started: Instant,
    counters: Arc<Counters>,
    cancel: Arc<AtomicBool>,
    events: Mutex<Receiver<BakeEvent>>,
    workers: Vec<JoinHandle<()>>,
}

impl BakeHandle {
    pub fn progress(&self) -> BakeProgress {
        let baked = self.counters.baked.load(Ordering::Acquire);
        let reused = self.counters.reused.load(Ordering::Acquire);
        let failed = self.counters.failed.load(Ordering::Acquire);
        BakeProgress {
            total: self.total,
            finished: baked + reused + failed,
            baked,
            reused,
            failed,
            partitions: self.counters.partitions.load(Ordering::Acquire),
            elapsed: self.started.elapsed(),
        }
    }

    /// The next finished job, without waiting. Every job produces exactly one
    /// event, sent before the counters count it, so a caller that reads
    /// `progress()` first and drains afterwards sees every event of the jobs
    /// that snapshot counted; draining first can leave the last one behind.
    pub fn try_recv(&self) -> Option<BakeEvent> {
        self.events.lock().ok()?.try_recv().ok()
    }

    /// Stops the workers after the meshes they are on; the counters then stay
    /// short of `total`.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    /// Cancels and joins the workers.
    pub fn stop(mut self) {
        self.cancel();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl Drop for BakeHandle {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// Starts baking `jobs` in index order on `settings.workers` threads.
///
/// `MeshletMesh::from_mesh` spreads its simplification over bevy's
/// `AsyncComputeTaskPool`, which is initialised here if no `App` has done so
/// yet (the call is a no-op otherwise).
pub fn start(settings: BakeSettings, jobs: Vec<MeshJob>) -> BakeHandle {
    AsyncComputeTaskPool::get_or_init(TaskPool::default);
    let total = jobs.len();
    let settings = Arc::new(settings);
    let jobs = Arc::new(jobs);
    let counters = Arc::new(Counters::default());
    let cancel = Arc::new(AtomicBool::new(false));
    let next = Arc::new(AtomicUsize::new(0));
    let (sender, events) = mpsc::channel();
    let workers = (0..settings.workers.max(1))
        .map(|worker| {
            let settings = Arc::clone(&settings);
            let jobs = Arc::clone(&jobs);
            let counters = Arc::clone(&counters);
            let cancel = Arc::clone(&cancel);
            let next = Arc::clone(&next);
            let sender = sender.clone();
            std::thread::Builder::new()
                .name(format!("zorah-bake-{worker}"))
                .spawn(move || run_worker(&settings, &jobs, &counters, &cancel, &next, &sender))
                .expect("spawning a bake worker thread")
        })
        .collect();
    BakeHandle {
        total,
        started: Instant::now(),
        counters,
        cancel,
        events: Mutex::new(events),
        workers,
    }
}

fn run_worker(
    settings: &BakeSettings,
    jobs: &[MeshJob],
    counters: &Counters,
    cancel: &AtomicBool,
    next: &AtomicUsize,
    sender: &Sender<BakeEvent>,
) {
    loop {
        if cancel.load(Ordering::Acquire) {
            return;
        }
        let index = next.fetch_add(1, Ordering::AcqRel);
        let Some(job) = jobs.get(index) else {
            return;
        };
        let dir = settings.cache_dir.join(&job.stem);
        let stamp = settings.stamp();
        // The wrapper is parsed without validation (its required
        // EXT_meshopt_compression would fail it), so a malformed index panics
        // inside the gltf crate's readers rather than erroring, and the
        // meshlet builder panics on degenerate input too. A panic has to count
        // as this mesh's failure, or the counters never reach `total` and the
        // caller waits on the bake forever.
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
            match MeshManifest::reusable(&dir, &stamp) {
                Some(manifest) => Ok((manifest, true)),
                None => bake_mesh(settings, job, &dir, stamp, counters)
                    .map(|manifest| (manifest, false)),
            }
        }));
        let (event, counter) = match outcome {
            Ok(Ok((manifest, reused))) => (
                BakeEvent::Complete {
                    job: index,
                    manifest,
                    reused,
                },
                if reused {
                    &counters.reused
                } else {
                    &counters.baked
                },
            ),
            Ok(Err(error)) => (
                BakeEvent::Failed {
                    job: index,
                    error: error.to_string(),
                },
                &counters.failed,
            ),
            Err(payload) => (
                BakeEvent::Failed {
                    job: index,
                    error: format!("panicked: {}", panic_message(payload.as_ref())),
                },
                &counters.failed,
            ),
        };
        // A closed receiver only means nobody is listening any more.
        let _ = sender.send(event);
        counter.fetch_add(1, Ordering::AcqRel);
    }
}

/// The `&str` or `String` a panic carried, as `std`'s hook prints it.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        String::from("unknown payload")
    }
}

fn bake_mesh(
    settings: &BakeSettings,
    job: &MeshJob,
    dir: &Path,
    stamp: BakeStamp,
    counters: &Counters,
) -> Result<MeshManifest, BakeError> {
    let meshes_dir = settings.scene_root.join("meshes");
    let wrapper_path = meshes_dir.join(format!("{}.mesh.gltf", job.stem));
    let gltf = gltf::Gltf::from_slice_without_validation(&fs::read(&wrapper_path)?)?;
    let document = gltf.document;
    let mut buffers = Vec::with_capacity(document.buffers().count());
    for buffer in document.buffers() {
        buffers.push(match buffer.source() {
            gltf::buffer::Source::Uri(uri) => fs::read(meshes_dir.join(uri))?,
            // The wrapper is JSON with an external .bin, so a `Bin` source can
            // only be a meshopt fallback buffer the file omits.
            gltf::buffer::Source::Bin => vec![0; buffer.length()],
        });
        // An omitted fallback buffer still declares its decoded length.
        if buffers.last().unwrap().len() < buffer.length() {
            buffers.last_mut().unwrap().resize(buffer.length(), 0);
        }
    }
    bevy::gltf::meshopt::decode_buffer_views(&document, &mut buffers)
        .map_err(|error| BakeError::Meshopt(error.to_string()))?;

    let mut wrapper_meshes = document.meshes();
    let (Some(mesh), None) = (wrapper_meshes.next(), wrapper_meshes.next()) else {
        return Err(BakeError::Invalid(format!(
            "{} does not hold exactly one mesh",
            wrapper_path.display()
        )));
    };
    let primitives = mesh.primitives().collect::<Vec<_>>();
    if primitives.len() != job.primitives.len() {
        return Err(BakeError::Invalid(format!(
            "wrapper has {} primitives, the root glTF {}",
            primitives.len(),
            job.primitives.len()
        )));
    }

    fs::create_dir_all(dir)?;
    let mut parts = Vec::new();
    let mut source_triangles = 0;
    for (primitive_index, (primitive, expected)) in
        primitives.iter().zip(&job.primitives).enumerate()
    {
        let mut geometry = read_primitive(primitive, &buffers, expected)
            .map_err(|error| BakeError::Invalid(format!("primitive {primitive_index}: {error}")))?;
        let winding_repaired = repair_inverted_winding(&mut geometry);
        let triangles = geometry.indices.len() / 3;
        source_triangles += triangles as u64;

        let partitions = if triangles > settings.partition_triangles {
            partition_triangles(&geometry, settings.partition_triangles)
        } else {
            vec![(0..triangles as u32).collect()]
        };
        // A mesh cut into partitions meets itself along their open edges and
        // each partition builds its LOD chain alone, so those edges are locked
        // or coarser LODs open cracks along the seams. The export's UDIM tiles
        // are separate primitives that meet the same way, so a multi-primitive
        // mesh is locked too. A single primitive in one piece keeps its open
        // edges free, since nothing of its own meets them.
        let locked_borders = partitions.len() > 1 || primitives.len() > 1;
        let mut remap = vec![u32::MAX; geometry.positions.len()];
        for (partition_index, triangle_ids) in partitions.iter().enumerate() {
            let part = if locked_borders {
                reindex(&geometry, triangle_ids, &mut remap)
            } else {
                std::mem::replace(&mut geometry, Geometry::empty())
            };
            let slot = PartSlot {
                primitive: primitive_index,
                partition: partition_index,
                material: expected.material,
                locked_borders,
                winding_repaired,
            };
            let manifest = bake_partition(settings, dir, part, slot).map_err(|error| {
                BakeError::Invalid(format!(
                    "primitive {primitive_index} partition {partition_index}: {error}"
                ))
            })?;
            parts.push(manifest);
            counters.partitions.fetch_add(1, Ordering::AcqRel);
        }
    }

    let manifest = MeshManifest {
        stamp,
        stem: job.stem.clone(),
        mesh_name: job.name.clone(),
        source_triangles,
        parts,
    };
    manifest.write(dir)?;
    Ok(manifest)
}

/// Reads one wrapper primitive into the attribute set the meshlet builder
/// wants, checking it against what the root glTF declared for it.
fn read_primitive(
    primitive: &gltf::Primitive,
    buffers: &[Vec<u8>],
    expected: &PrimitiveJob,
) -> Result<Geometry, String> {
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        return Err(format!("mode {:?} is not TRIANGLES", primitive.mode()));
    }
    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice));
    let positions = reader
        .read_positions()
        .ok_or("no positions")?
        .collect::<Vec<_>>();
    // The meshlet builder indexes its first vertex unconditionally.
    if positions.is_empty() {
        return Err("no vertices".into());
    }
    if positions.len() != expected.vertex_count {
        return Err(format!(
            "wrapper has {} vertices, the root glTF {}",
            positions.len(),
            expected.vertex_count
        ));
    }
    let indices = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect::<Vec<_>>(),
        None => (0..positions.len() as u32).collect(),
    };
    if expected.index_count != 0 && indices.len() != expected.index_count {
        return Err(format!(
            "wrapper has {} indices, the root glTF {}",
            indices.len(),
            expected.index_count
        ));
    }
    if indices.is_empty() {
        return Err("no triangles".into());
    }
    if indices.len() % 3 != 0 {
        return Err(format!(
            "{} indices is not a whole number of triangles",
            indices.len()
        ));
    }
    if indices
        .iter()
        .any(|index| *index as usize >= positions.len())
    {
        return Err("an index exceeds the vertex count".into());
    }
    let normals = match reader.read_normals() {
        Some(normals) => {
            let normals = normals.collect::<Vec<_>>();
            if normals.len() != positions.len() {
                return Err("normal count differs from the vertex count".into());
            }
            normals
        }
        None => smooth_normals(&positions, &indices),
    };
    let uvs = match reader.read_tex_coords(0) {
        Some(uvs) => {
            let uvs = uvs.into_f32().collect::<Vec<_>>();
            if uvs.len() != positions.len() {
                return Err("uv count differs from the vertex count".into());
            }
            uvs
        }
        // Flat UVs keep the attribute layout uniform; the material has no
        // texture to sample for such a primitive anyway.
        None => vec![[0.0, 0.0]; positions.len()],
    };
    Ok(Geometry {
        positions,
        normals,
        uvs,
        indices,
    })
}

/// Where a partition sits in its mesh, for the manifest entry and file names.
struct PartSlot {
    primitive: usize,
    partition: usize,
    material: Option<usize>,
    locked_borders: bool,
    winding_repaired: bool,
}

/// Builds one partition's meshlet mesh and BLAS cut and writes both files as
/// `p<primitive>_<partition>.meshlet_mesh` / `.zblas` under `dir`.
fn bake_partition(
    settings: &BakeSettings,
    dir: &Path,
    part: Geometry,
    slot: PartSlot,
) -> Result<PartManifest, BakeError> {
    let triangles = (part.indices.len() / 3) as u64;
    let (aabb_min, aabb_max) = part.positions.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(min, max), position| {
            let position = Vec3::from(*position);
            (min.min(position), max.max(position))
        },
    );
    let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, part.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, part.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, part.uvs)
        .with_inserted_indices(Indices::U32(part.indices));
    let meshlet = if slot.locked_borders {
        MeshletMesh::from_mesh_with_locked_borders(&mesh, settings.quantization)
    } else {
        MeshletMesh::from_mesh(&mesh, settings.quantization)
    }
    .map_err(|error| BakeError::Meshlet(error.to_string()))?;
    drop(mesh);
    let raytracing = meshlet.raytracing_geometry(settings.raytracing_error);
    if raytracing.indices.is_empty() {
        return Err(BakeError::Meshlet("the LOD cut has no triangles".into()));
    }
    let blas_bytes = blas::encode(&raytracing)?;
    let meshlet_bytes = encode_meshlet(&meshlet)?;
    drop(meshlet);

    let meshlet_file = format!("p{}_{}.meshlet_mesh", slot.primitive, slot.partition);
    let blas_file = format!("p{}_{}.zblas", slot.primitive, slot.partition);
    fs::write(dir.join(&meshlet_file), meshlet_bytes)?;
    fs::write(dir.join(&blas_file), blas_bytes)?;
    Ok(PartManifest {
        primitive: slot.primitive,
        partition: slot.partition,
        material: slot.material,
        meshlet_file,
        blas_file,
        triangles,
        blas_triangles: (raytracing.indices.len() / 3) as u64,
        blas_achieved_error: raytracing.achieved_error,
        aabb_min: aabb_min.to_array(),
        aabb_max: aabb_max.to_array(),
        locked_borders: slot.locked_borders,
        winding_repaired: slot.winding_repaired,
    })
}

fn encode_meshlet(meshlet: &MeshletMesh) -> Result<Vec<u8>, BakeError> {
    let mut writer = VecWriter::default();
    block_on(MeshletMeshSaver.save(
        &mut writer,
        SavedAsset::from_asset(meshlet),
        &(),
        AssetPath::from("partition.meshlet_mesh"),
    ))
    .map_err(|error| BakeError::Meshlet(error.to_string()))?;
    Ok(writer.bytes)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(dir: &Path) -> BakeSettings {
        BakeSettings {
            scene_root: dir.to_path_buf(),
            cache_dir: dir.to_path_buf(),
            workers: 1,
            partition_triangles: 500_000,
            raytracing_error: 0.02,
            quantization: 4,
        }
    }

    fn manifest(stamp: BakeStamp) -> MeshManifest {
        MeshManifest {
            stamp,
            stem: "SM_Test_0123_4567".into(),
            mesh_name: "SM_Test_0123".into(),
            source_triangles: 12,
            parts: vec![PartManifest {
                primitive: 0,
                partition: 0,
                material: Some(3),
                meshlet_file: "p0_0.meshlet_mesh".into(),
                blas_file: "p0_0.zblas".into(),
                triangles: 12,
                blas_triangles: 8,
                blas_achieved_error: 0.003,
                aabb_min: [-1.0, 0.0, -1.0],
                aabb_max: [1.0, 2.0, 1.0],
                locked_borders: false,
                winding_repaired: true,
            }],
        }
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zorah_v2_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn manifest_round_trips_and_is_reused() {
        let dir = scratch_dir("manifest");
        let settings = settings(&dir);
        let written = manifest(settings.stamp());
        written.write(&dir).unwrap();
        assert!(dir.join(MANIFEST_FILE).is_file());
        assert!(!dir.join(format!("{MANIFEST_FILE}.tmp")).exists());
        let read: MeshManifest =
            serde_json::from_slice(&fs::read(dir.join(MANIFEST_FILE)).unwrap()).unwrap();
        assert_eq!(read, written);

        // Listed part files must exist before the manifest counts as a checkpoint.
        assert!(MeshManifest::reusable(&dir, &settings.stamp()).is_none());
        fs::write(dir.join("p0_0.meshlet_mesh"), b"").unwrap();
        fs::write(dir.join("p0_0.zblas"), b"").unwrap();
        assert_eq!(
            MeshManifest::reusable(&dir, &settings.stamp()),
            Some(written)
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stamp_mismatch_forces_a_rebake() {
        let dir = scratch_dir("stamp");
        let settings = settings(&dir);
        manifest(settings.stamp()).write(&dir).unwrap();
        fs::write(dir.join("p0_0.meshlet_mesh"), b"").unwrap();
        fs::write(dir.join("p0_0.zblas"), b"").unwrap();
        assert!(MeshManifest::reusable(&dir, &settings.stamp()).is_some());

        let mut quantized = settings.clone();
        quantized.quantization = 6;
        assert!(MeshManifest::reusable(&dir, &quantized.stamp()).is_none());
        let mut coarser = settings.clone();
        coarser.raytracing_error = 0.05;
        assert!(MeshManifest::reusable(&dir, &coarser.stamp()).is_none());
        let mut split = settings.clone();
        split.partition_triangles = 100_000;
        assert!(MeshManifest::reusable(&dir, &split.stamp()).is_none());
        let mut stale = settings.stamp();
        stale.pipeline_version += 1;
        assert!(MeshManifest::reusable(&dir, &stale).is_none());
        fs::remove_dir_all(&dir).unwrap();
    }
}
