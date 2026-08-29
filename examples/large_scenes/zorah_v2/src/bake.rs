//! Bakes the export's meshes into full-detail meshlet meshes.
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
    collections::{HashMap, HashSet, VecDeque},
    fs,
    io::Write,
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc, Condvar, Mutex,
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

use crate::geometry::{
    partition_triangles, reindex, repair_inverted_winding, smooth_normals, Geometry,
};

/// Bumped whenever the bake's own logic changes what it writes (partition
/// rules, seam locking, attribute synthesis, winding repair), so stale
/// caches rebake. Version 3 locked only shared seams instead of every open
/// edge; version 4 matches seams on the quantization grid rather than by
/// exact bits, which locks the tile edges an ulp apart that 3 missed.
pub const BAKE_PIPELINE_VERSION: u32 = 4;
pub const MANIFEST_FILE: &str = "manifest.json";

/// How the bake runs and where it writes.
#[derive(Clone, Debug)]
pub struct BakeSettings {
    /// Directory holding the root glTF and its `meshes/` folder.
    pub scene_root: PathBuf,
    /// One subdirectory per mesh stem goes here.
    pub cache_dir: PathBuf,
    /// Threads building parts concurrently. Each builds one part at a time and
    /// at most `workers` decoded meshes are held between them.
    pub workers: usize,
    /// Primitives above this many triangles are cut into spatial partitions
    /// before the meshlet build, which is superlinear in triangle count.
    pub partition_triangles: usize,
    /// Meshlet vertex position quantization factor (`MeshletMesh::from_mesh`).
    pub quantization: u8,
}

impl BakeSettings {
    /// The stamp a manifest must carry to be reused under these settings.
    pub fn stamp(&self) -> BakeStamp {
        BakeStamp {
            pipeline_version: BAKE_PIPELINE_VERSION,
            meshlet_asset_version: MESHLET_MESH_ASSET_VERSION,
            quantization: self.quantization,
            partition_triangles: self.partition_triangles,
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
///
/// `--raster-error` and `--raytracing-error` are applied at load and are not
/// part of it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BakeStamp {
    pub pipeline_version: u32,
    pub meshlet_asset_version: u64,
    pub quantization: u8,
    pub partition_triangles: usize,
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

/// One meshlet mesh in a mesh directory.
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
    pub triangles: u64,
    pub vertices: u32,
    /// Vertices whose position another part of the mesh shares, held in place
    /// at every LOD so the seam stays closed.
    pub locked_vertices: u32,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub winding_repaired: bool,
}

impl MeshManifest {
    /// The stamp of `<dir>/manifest.json`, if one parses.
    pub fn stamped(dir: &Path) -> Option<BakeStamp> {
        Self::parse(dir).map(|manifest| manifest.stamp)
    }

    fn parse(dir: &Path) -> Option<MeshManifest> {
        let bytes = fs::read(dir.join(MANIFEST_FILE)).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    /// Reads `<dir>/manifest.json` if it exists, matches `stamp`, and every
    /// part file it lists is present.
    pub fn reusable(dir: &Path, stamp: &BakeStamp) -> Option<MeshManifest> {
        let manifest = Self::parse(dir)?;
        if manifest.stamp != *stamp {
            return None;
        }
        let complete = manifest
            .parts
            .iter()
            .all(|part| dir.join(&part.meshlet_file).is_file());
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

/// The running bake. Dropping it lets the workers finish their current part
/// and stop; they never outlive the process's interest in them for long.
pub struct BakeHandle {
    total: usize,
    started: Instant,
    counters: Arc<Counters>,
    scheduler: Arc<Scheduler>,
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

    /// Stops the workers after the parts they are on; the counters then stay
    /// short of `total`.
    pub fn cancel(&self) {
        self.scheduler.cancel();
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

/// Starts baking `jobs` on `settings.workers` threads.
///
/// Work is scheduled per part, not per mesh: a worker that finds nothing
/// queued opens the next mesh - decodes it, partitions it, finds its seams -
/// and queues every part, so a 32 M-triangle mesh spreads over every worker
/// instead of holding one for an hour while the rest idle. Meshes open
/// largest first, which keeps the tail short. A mesh is opened only when the
/// queue is empty, so at most `workers` decoded meshes are held at once: every
/// mesh alive at the last claim was being opened or had a part in flight, one
/// per worker.
///
/// `MeshletMesh::from_mesh` spreads its simplification over bevy's
/// `AsyncComputeTaskPool`, which is initialised here if no `App` has done so
/// yet (the call is a no-op otherwise).
pub fn start(settings: BakeSettings, jobs: Vec<MeshJob>) -> BakeHandle {
    AsyncComputeTaskPool::get_or_init(TaskPool::default);
    let total = jobs.len();
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by_key(|&job| std::cmp::Reverse(jobs[job].triangles()));
    let shared = Arc::new(Shared {
        settings,
        jobs,
        order,
        counters: Arc::new(Counters::default()),
    });
    let scheduler = Arc::new(Scheduler::default());
    let (sender, events) = mpsc::channel();
    let workers = (0..shared.settings.workers.max(1))
        .map(|worker| {
            let shared = Arc::clone(&shared);
            let scheduler = Arc::clone(&scheduler);
            let sender = sender.clone();
            std::thread::Builder::new()
                .name(format!("zorah-bake-{worker}"))
                .spawn(move || run_worker(&shared, &scheduler, &sender))
                .expect("spawning a bake worker thread")
        })
        .collect();
    BakeHandle {
        total,
        started: Instant::now(),
        counters: Arc::clone(&shared.counters),
        scheduler,
        events: Mutex::new(events),
        workers,
    }
}

/// What every worker reads.
struct Shared {
    settings: BakeSettings,
    jobs: Vec<MeshJob>,
    /// Job indices, largest first.
    order: Vec<usize>,
    counters: Arc<Counters>,
}

/// The part queue and the cursor into `Shared::order`.
#[derive(Default)]
struct Scheduler {
    state: Mutex<SchedulerState>,
    /// Signalled when parts are queued, an opening finishes, or the bake is
    /// cancelled.
    ready: Condvar,
    cancel: AtomicBool,
}

#[derive(Default)]
struct SchedulerState {
    queue: VecDeque<PartTask>,
    next: usize,
    /// Workers between claiming a job and queueing its parts; a worker with
    /// nothing to do waits for them rather than exiting.
    opening: usize,
}

impl Scheduler {
    fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
        let _guard = self.state.lock();
        self.ready.notify_all();
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }
}

/// One partition of one primitive of an open mesh.
struct PartTask {
    mesh: Arc<OpenMesh>,
    slot: usize,
}

/// A mesh whose geometry is decoded and whose parts are queued or in flight.
/// Dropped, with its geometry, when the last part reports.
struct OpenMesh {
    job: usize,
    dir: PathBuf,
    stamp: BakeStamp,
    source_triangles: u64,
    primitives: Vec<OpenPrimitive>,
    /// Where each part slot sits.
    slots: Vec<PartSlot>,
    /// Positions (as f32 bit patterns) that more than one part uses: the
    /// partition cuts and the edges primitives share.
    shared_positions: HashSet<[i32; 3]>,
    remaining: AtomicUsize,
    parts: Mutex<Vec<Option<PartManifest>>>,
    failure: Mutex<Option<String>>,
}

struct OpenPrimitive {
    geometry: Geometry,
    material: Option<usize>,
    winding_repaired: bool,
    /// Triangle ids per partition.
    partitions: Vec<Vec<u32>>,
}

/// Where a partition sits in its mesh, for the manifest entry and file names.
#[derive(Clone, Copy)]
struct PartSlot {
    primitive: usize,
    partition: usize,
}

fn run_worker(shared: &Shared, scheduler: &Scheduler, sender: &Sender<BakeEvent>) {
    let mut state = scheduler.state.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if scheduler.cancelled() {
            return;
        }
        if let Some(task) = state.queue.pop_front() {
            drop(state);
            run_part(shared, task, sender);
            state = scheduler.state.lock().unwrap_or_else(|e| e.into_inner());
            continue;
        }
        if let Some(&job) = shared.order.get(state.next) {
            state.next += 1;
            state.opening += 1;
            drop(state);
            let tasks = open_job(shared, job, sender);
            state = scheduler.state.lock().unwrap_or_else(|e| e.into_inner());
            state.opening -= 1;
            state.queue.extend(tasks);
            scheduler.ready.notify_all();
            continue;
        }
        if state.opening == 0 {
            return;
        }
        state = scheduler
            .ready
            .wait(state)
            .unwrap_or_else(|e| e.into_inner());
    }
}

/// Reuses `job`'s manifest or decodes and partitions the mesh and returns
/// its part tasks. A job that is reused, fails or panics reports here and
/// yields no tasks.
fn open_job(shared: &Shared, job: usize, sender: &Sender<BakeEvent>) -> Vec<PartTask> {
    let spec = &shared.jobs[job];
    let dir = shared.settings.cache_dir.join(&spec.stem);
    let stamp = shared.settings.stamp();
    if let Some(manifest) = MeshManifest::reusable(&dir, &stamp) {
        report(
            sender,
            BakeEvent::Complete {
                job,
                manifest,
                reused: true,
            },
            &shared.counters.reused,
        );
        return Vec::new();
    }
    // The wrapper is parsed without validation (its required
    // EXT_meshopt_compression would fail it), so a malformed index panics
    // inside the gltf crate's readers rather than erroring. A panic has to
    // count as this mesh's failure, or the counters never reach `total` and
    // the caller waits on the bake forever.
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
        open_mesh(&shared.settings, job, spec, dir, stamp)
    }));
    match outcome {
        Ok(Ok(mesh)) => {
            let mesh = Arc::new(mesh);
            (0..mesh.slots.len())
                .map(|slot| PartTask {
                    mesh: Arc::clone(&mesh),
                    slot,
                })
                .collect()
        }
        Ok(Err(error)) => {
            report(
                sender,
                BakeEvent::Failed {
                    job,
                    error: error.to_string(),
                },
                &shared.counters.failed,
            );
            Vec::new()
        }
        Err(payload) => {
            report(
                sender,
                BakeEvent::Failed {
                    job,
                    error: format!("panicked: {}", panic_message(payload.as_ref())),
                },
                &shared.counters.failed,
            );
            Vec::new()
        }
    }
}

/// Sends a job's one event, then counts it (see `BakeHandle::try_recv`).
fn report(sender: &Sender<BakeEvent>, event: BakeEvent, counter: &AtomicUsize) {
    // A closed receiver only means nobody is listening any more.
    let _ = sender.send(event);
    counter.fetch_add(1, Ordering::AcqRel);
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

/// Decodes a mesh's wrapper, partitions its primitives and finds the
/// positions its parts share.
fn open_mesh(
    settings: &BakeSettings,
    job: usize,
    spec: &MeshJob,
    dir: PathBuf,
    stamp: BakeStamp,
) -> Result<OpenMesh, BakeError> {
    let meshes_dir = settings.scene_root.join("meshes");
    let wrapper_path = meshes_dir.join(format!("{}.mesh.gltf", spec.stem));
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
    let wrapper_primitives = mesh.primitives().collect::<Vec<_>>();
    if wrapper_primitives.len() != spec.primitives.len() {
        return Err(BakeError::Invalid(format!(
            "wrapper has {} primitives, the root glTF {}",
            wrapper_primitives.len(),
            spec.primitives.len()
        )));
    }

    let mut primitives = Vec::with_capacity(wrapper_primitives.len());
    let mut slots = Vec::new();
    let mut source_triangles = 0;
    for (primitive_index, (primitive, expected)) in
        wrapper_primitives.iter().zip(&spec.primitives).enumerate()
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
        for partition in 0..partitions.len() {
            slots.push(PartSlot {
                primitive: primitive_index,
                partition,
            });
        }
        primitives.push(OpenPrimitive {
            geometry,
            material: expected.material,
            winding_repaired,
            partitions,
        });
    }
    drop(buffers);

    let shared_positions = shared_positions(&primitives, &slots, settings.quantization);
    fs::create_dir_all(&dir)?;
    Ok(OpenMesh {
        job,
        dir,
        stamp,
        source_triangles,
        remaining: AtomicUsize::new(slots.len()),
        parts: Mutex::new(vec![None; slots.len()]),
        failure: Mutex::new(None),
        slots,
        primitives,
        shared_positions,
    })
}

/// The positions more than one part of the mesh uses.
///
/// Every part is built into its own LOD chain, so a position two parts share
/// - along a partition cut, or where two primitives (the export's UDIM tiles)
/// meet edge to edge - must stay put in both or the seam opens as they
/// simplify. Nothing else is locked: an edge that is open in the source mesh
/// meets nothing and may simplify freely, which is what keeps this scene's
/// tiled floors and loose ornaments from being pinned at full detail.
///
/// Positions compare on the meshlet build's own quantization grid (1/2^q cm),
/// which is where the build snaps every vertex anyway: the parts of one
/// primitive copy the same source vertices, but this export's tiles carry
/// nominally coincident coordinates that differ in their last bits, and a
/// seam missed for an ulp would be a seam left unlocked.
fn shared_positions(
    primitives: &[OpenPrimitive],
    slots: &[PartSlot],
    quantization: u8,
) -> HashSet<[i32; 3]> {
    if slots.len() < 2 {
        return HashSet::new();
    }
    const SHARED: u32 = u32::MAX;
    let vertices = primitives
        .iter()
        .map(|primitive| primitive.geometry.positions.len())
        .sum();
    let mut owner: HashMap<[i32; 3], u32> = HashMap::with_capacity(vertices);
    for (part, slot) in slots.iter().enumerate() {
        let primitive = &primitives[slot.primitive];
        for triangle in &primitive.partitions[slot.partition] {
            let base = *triangle as usize * 3;
            for index in &primitive.geometry.indices[base..base + 3] {
                let key = position_key(primitive.geometry.positions[*index as usize], quantization);
                let entry = owner.entry(key).or_insert(part as u32);
                if *entry != part as u32 {
                    *entry = SHARED;
                }
            }
        }
    }
    owner
        .into_iter()
        .filter_map(|(key, part)| (part == SHARED).then_some(key))
        .collect()
}

/// A position snapped to the meshlet quantization grid, as `MeshletMesh::from_mesh`
/// snaps it (metres to centimetres, then 2^q steps per centimetre).
fn position_key(position: [f32; 3], quantization: u8) -> [i32; 3] {
    let scale = 100.0 * (1u32 << quantization) as f32;
    position.map(|component| (component * scale).round() as i32)
}

/// Builds one part and, when it is the mesh's last, writes the manifest and
/// reports the mesh.
fn run_part(shared: &Shared, task: PartTask, sender: &Sender<BakeEvent>) {
    let mesh = task.mesh;
    let slot = mesh.slots[task.slot];
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
        bake_part(&shared.settings, &mesh, slot)
    }));
    match outcome {
        Ok(Ok(part)) => {
            let mut parts = mesh.parts.lock().unwrap_or_else(|e| e.into_inner());
            parts[task.slot] = Some(part);
            shared.counters.partitions.fetch_add(1, Ordering::AcqRel);
        }
        Ok(Err(error)) => record_failure(&mesh, slot, error.to_string()),
        Err(payload) => record_failure(
            &mesh,
            slot,
            format!("panicked: {}", panic_message(payload.as_ref())),
        ),
    }
    if mesh.remaining.fetch_sub(1, Ordering::AcqRel) != 1 {
        return;
    }
    // Last part in: every other part has stored its result or failure.
    let failure = mesh
        .failure
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take();
    if let Some(error) = failure {
        report(
            sender,
            BakeEvent::Failed {
                job: mesh.job,
                error,
            },
            &shared.counters.failed,
        );
        return;
    }
    let parts = mesh
        .parts
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter_mut()
        .map(|part| part.take().expect("every part reported"))
        .collect();
    let spec = &shared.jobs[mesh.job];
    let manifest = MeshManifest {
        stamp: mesh.stamp.clone(),
        stem: spec.stem.clone(),
        mesh_name: spec.name.clone(),
        source_triangles: mesh.source_triangles,
        parts,
    };
    match manifest.write(&mesh.dir) {
        Ok(()) => report(
            sender,
            BakeEvent::Complete {
                job: mesh.job,
                manifest,
                reused: false,
            },
            &shared.counters.baked,
        ),
        Err(error) => report(
            sender,
            BakeEvent::Failed {
                job: mesh.job,
                error: format!("writing the manifest: {error}"),
            },
            &shared.counters.failed,
        ),
    }
}

fn record_failure(mesh: &OpenMesh, slot: PartSlot, error: String) {
    let mut failure = mesh.failure.lock().unwrap_or_else(|e| e.into_inner());
    if failure.is_none() {
        *failure = Some(format!(
            "primitive {} partition {}: {error}",
            slot.primitive, slot.partition
        ));
    }
}

/// Builds one part's meshlet mesh and writes it as
/// `p<primitive>_<partition>.meshlet_mesh` under the mesh directory.
fn bake_part(
    settings: &BakeSettings,
    mesh: &OpenMesh,
    slot: PartSlot,
) -> Result<PartManifest, BakeError> {
    let primitive = &mesh.primitives[slot.primitive];
    let triangle_ids = &primitive.partitions[slot.partition];
    let mut remap = vec![u32::MAX; primitive.geometry.positions.len()];
    let part = reindex(&primitive.geometry, triangle_ids, &mut remap);
    drop(remap);
    let triangles = (part.indices.len() / 3) as u64;
    let vertices = part.positions.len() as u32;
    let (aabb_min, aabb_max) = part.positions.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(min, max), position| {
            let position = Vec3::from(*position);
            (min.min(position), max.max(position))
        },
    );
    let locked: Vec<bool> = if mesh.shared_positions.is_empty() {
        Vec::new()
    } else {
        part.positions
            .iter()
            .map(|position| {
                mesh.shared_positions
                    .contains(&position_key(*position, settings.quantization))
            })
            .collect()
    };
    let locked_vertices = locked.iter().filter(|flag| **flag).count() as u32;
    let bevy_mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, part.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, part.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, part.uvs)
        .with_inserted_indices(Indices::U32(part.indices));
    let meshlet = if locked_vertices == 0 {
        MeshletMesh::from_mesh(&bevy_mesh, settings.quantization)
    } else {
        MeshletMesh::from_mesh_with_locks(&bevy_mesh, settings.quantization, &locked)
    }
    .map_err(|error| BakeError::Meshlet(error.to_string()))?;
    drop(bevy_mesh);
    drop(locked);
    let meshlet_bytes = encode_meshlet(&meshlet)?;
    drop(meshlet);

    let meshlet_file = format!("p{}_{}.meshlet_mesh", slot.primitive, slot.partition);
    fs::write(mesh.dir.join(&meshlet_file), meshlet_bytes)?;
    Ok(PartManifest {
        primitive: slot.primitive,
        partition: slot.partition,
        material: primitive.material,
        meshlet_file,
        triangles,
        vertices,
        locked_vertices,
        aabb_min: aabb_min.to_array(),
        aabb_max: aabb_max.to_array(),
        winding_repaired: primitive.winding_repaired,
    })
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
                triangles: 12,
                vertices: 9,
                locked_vertices: 0,
                aabb_min: [-1.0, 0.0, -1.0],
                aabb_max: [1.0, 2.0, 1.0],
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
        assert!(MeshManifest::reusable(&dir, &settings.stamp()).is_some());

        let mut quantized = settings.clone();
        quantized.quantization = 6;
        assert!(MeshManifest::reusable(&dir, &quantized.stamp()).is_none());
        let mut split = settings.clone();
        split.partition_triangles = 100_000;
        assert!(MeshManifest::reusable(&dir, &split.stamp()).is_none());
        let mut stale = settings.stamp();
        stale.pipeline_version += 1;
        assert!(MeshManifest::reusable(&dir, &stale).is_none());
        fs::remove_dir_all(&dir).unwrap();
    }
}

#[cfg(test)]
mod seam_tests {
    use super::*;

    /// A flat `n` by `n` grid of quads on the XZ plane, `x` first.
    fn grid(n: u32, x_range: std::ops::Range<u32>) -> Geometry {
        let columns = x_range.end - x_range.start + 1;
        let mut positions = Vec::new();
        for z in 0..=n {
            for x in x_range.clone().chain(std::iter::once(x_range.end)) {
                positions.push([x as f32, 0.0, z as f32]);
            }
        }
        let mut indices = Vec::new();
        for z in 0..n {
            for x in 0..columns - 1 {
                let a = z * columns + x;
                let (b, c, d) = (a + 1, a + columns, a + columns + 1);
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        let count = positions.len();
        Geometry {
            positions,
            normals: vec![[0.0, 1.0, 0.0]; count],
            uvs: vec![[0.0, 0.0]; count],
            indices,
        }
    }

    fn primitive(geometry: Geometry, partitions: Vec<Vec<u32>>) -> OpenPrimitive {
        OpenPrimitive {
            geometry,
            material: None,
            winding_repaired: false,
            partitions,
        }
    }

    fn column(shared: &HashSet<[i32; 3]>, x: f32, n: u32) -> bool {
        (0..=n).all(|z| shared.contains(&position_key([x, 0.0, z as f32], 4)))
    }

    #[test]
    fn a_partition_cut_is_the_only_seam_of_one_primitive() {
        // One 4x4 grid cut between its second and third quad columns: the
        // cut column x = 2 is shared, the open outer border is not.
        let geometry = grid(4, 0..4);
        let left: Vec<u32> = (0..16u32)
            .filter(|quad| quad % 4 < 2)
            .flat_map(|quad| [quad * 2, quad * 2 + 1])
            .collect();
        let right: Vec<u32> = (0..32u32).filter(|t| !left.contains(t)).collect();
        let primitives = vec![primitive(geometry, vec![left, right])];
        let slots = vec![
            PartSlot {
                primitive: 0,
                partition: 0,
            },
            PartSlot {
                primitive: 0,
                partition: 1,
            },
        ];
        let shared = shared_positions(&primitives, &slots, 4);
        assert_eq!(shared.len(), 5);
        assert!(column(&shared, 2.0, 4));
        assert!(!shared.contains(&position_key([0.0, 0.0, 0.0], 4)));
        assert!(!shared.contains(&position_key([4.0, 0.0, 4.0], 4)));
    }

    #[test]
    fn two_primitives_share_only_the_edge_they_meet_on() {
        // Two tiles with their own vertices meeting along x = 2, as the
        // export's UDIM primitives do.
        let a = grid(4, 0..2);
        let b = grid(4, 2..4);
        let all_a: Vec<u32> = (0..(a.indices.len() / 3) as u32).collect();
        let all_b: Vec<u32> = (0..(b.indices.len() / 3) as u32).collect();
        let primitives = vec![primitive(a, vec![all_a]), primitive(b, vec![all_b])];
        let slots = vec![
            PartSlot {
                primitive: 0,
                partition: 0,
            },
            PartSlot {
                primitive: 1,
                partition: 0,
            },
        ];
        let shared = shared_positions(&primitives, &slots, 4);
        assert_eq!(shared.len(), 5);
        assert!(column(&shared, 2.0, 4));
    }

    #[test]
    fn a_mesh_in_one_part_shares_nothing() {
        let geometry = grid(4, 0..4);
        let all: Vec<u32> = (0..32).collect();
        let primitives = vec![primitive(geometry, vec![all])];
        let slots = vec![PartSlot {
            primitive: 0,
            partition: 0,
        }];
        assert!(shared_positions(&primitives, &slots, 4).is_empty());
    }

    #[test]
    fn keys_match_within_the_quantization_step_and_differ_beyond_it() {
        // q = 4: 1/16 cm steps. An ulp apart is the same key; 1 mm is not.
        assert_eq!(
            position_key([-0.0, 1.0, -0.0], 4),
            position_key([0.0, 1.0, 0.0], 4)
        );
        assert_eq!(
            position_key([3.9762273, 5.27106, 0.0], 4),
            position_key([3.9762292, 5.2710595, 0.0], 4)
        );
        assert_ne!(
            position_key([0.0, 1.0, 0.0], 4),
            position_key([0.0, 1.0, 0.001], 4)
        );
    }
}
