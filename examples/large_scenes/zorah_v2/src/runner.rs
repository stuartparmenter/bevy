//! `Baking` -> `LoadingScene` -> `WarmingRaytracing` -> `Running`.
//!
//! One system polls the bake and, for every mesh whose manifest is complete,
//! loads its parts from the `cache://` source and spawns one entity per
//! (instance, part) under a per-frame budget, so the window stays responsive
//! while thousands of meshlet meshes stream in. Once everything is spawned
//! the warm-up attaches `RaytracingMesh3d` in batches and waits on Solari's
//! measured BLAS readiness before enabling it, exactly as v1 does.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::PathBuf,
    time::{Duration, Instant},
};

use bevy::{
    asset::LoadState,
    pbr::experimental::meshlet::{MeshletMesh, MeshletMesh3d},
    prelude::*,
    render::renderer::RenderDevice,
    solari::prelude::{
        MeshGeometryError, RaytracingMesh3d, RaytracingSceneStatus, SolariLighting, SolariPlugins,
    },
};

#[cfg(feature = "dlss")]
use bevy::anti_alias::dlss::{
    Dlss, DlssPerfQualityMode, DlssRayReconstructionFeature, DlssRayReconstructionSupported,
};

use crate::{
    bake::{BakeEvent, BakeHandle, MeshManifest, PartManifest, MANIFEST_FILE},
    lod::{fits_a_page, human_bytes, PartStats, ZorahPart, MESHLET_PAGE_BUDGET, MESHLET_PAGE_SIZE},
    materials::MaterialCache,
    scene::SceneInstance,
    setup::ZorahCamera,
};

/// Part loads started per frame: each decodes a full-detail meshlet mesh,
/// prunes it and cuts its BLAS on the IO pool.
const MAX_NEW_PART_LOADS_PER_FRAME: usize = 32;
/// Entities spawned per frame, counting every part of every instance.
const MAX_SPAWNED_INSTANCES_PER_FRAME: usize = 512;
/// `RaytracingMesh3d` attached per frame during warm-up.
const MAX_RAYTRACING_INSTANCES_PER_FRAME: usize = 512;
/// The diagnostic estimate of how long BLAS builds take: this many triangles
/// per frame plus a margin. Exceeding it only logs.
const BLAS_BUILD_TRIANGLES_PER_FRAME: u64 = 2_000_000;
const BLAS_WARMUP_MARGIN_FRAMES: u64 = 60;
const BLAS_PROGRESS_LOG_INTERVAL_FRAMES: u64 = 120;
const BAKE_PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

#[derive(States, Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZorahState {
    #[default]
    Baking,
    LoadingScene,
    WarmingRaytracing,
    Running,
}

/// The flags the spawner consults.
#[derive(Resource)]
pub struct SpawnOptions {
    pub glass_in_blas: bool,
    /// Where the manifests live, so a corrupt part can invalidate its mesh.
    pub cache_dir: PathBuf,
}

/// The scene as parsed at startup: what to spawn where.
#[derive(Resource)]
pub struct SceneData {
    pub instances: Vec<SceneInstance>,
    /// Instance indices by root glTF mesh index.
    pub instances_by_mesh: HashMap<usize, Vec<usize>>,
    /// Bake job index -> the root mesh indices sharing its geometry.
    pub meshes_of_job: Vec<Vec<usize>>,
    /// Root glTF material index per primitive, per root mesh index.
    pub primitive_materials: Vec<Vec<Option<usize>>>,
}

#[derive(Clone)]
struct PartAssets {
    manifest: PartManifest,
    meshlet: Handle<MeshletMesh>,
    blas: Handle<Mesh>,
    stats: PartStats,
}

/// A manifest whose parts are being requested, then awaited.
struct LoadingJob {
    job: usize,
    manifest: MeshManifest,
    parts: Vec<Handle<ZorahPart>>,
}

/// A job whose parts are on the GPU, being spawned across its instances.
struct SpawnJob {
    job: usize,
    parts: Vec<PartAssets>,
    instances: Vec<usize>,
    cursor: usize,
}

pub struct PendingRaytracingInstance {
    pub entity: Entity,
    pub mesh: Handle<Mesh>,
    pub geometry_error: f32,
}

#[derive(Resource)]
pub struct PendingScene {
    bake: Option<BakeHandle>,
    bake_last_report: Instant,
    bake_failed: usize,
    loading: VecDeque<LoadingJob>,
    spawning: VecDeque<SpawnJob>,
    pub raytracing_instances: Vec<PendingRaytracingInstance>,
    raytracing_cursor: usize,
    expected_blas: usize,
    /// Triangles of every distinct BLAS cut, for the warm-up estimate.
    blas_triangles: u64,
    /// Bytes the meshlet manager holds for every distinct pruned part, against
    /// `MESHLET_PAGE_BUDGET`.
    meshlet_bytes: u64,
    /// Triangles across LODs of every distinct pruned part.
    meshlet_triangles: u64,
    /// Distinct pruned parts larger than a page, which the manager rejects
    /// however empty its pages are.
    parts_oversized: usize,
    /// The largest BLAS error any part achieved.
    blas_error_max: f32,
    warmup_frames_remaining: u64,
    warmup_timeout_reported: bool,
    warmup_progress_log_frames_remaining: u64,
    warmup_started_at: Option<Instant>,
    spawned: usize,
    mesh_raster: usize,
    parts_failed: usize,
}

impl PendingScene {
    pub fn new(bake: BakeHandle, raytracing_instances: Vec<PendingRaytracingInstance>) -> Self {
        Self {
            bake: Some(bake),
            bake_last_report: Instant::now(),
            bake_failed: 0,
            loading: VecDeque::new(),
            spawning: VecDeque::new(),
            raytracing_instances,
            raytracing_cursor: 0,
            expected_blas: 0,
            blas_triangles: 0,
            meshlet_bytes: 0,
            meshlet_triangles: 0,
            parts_oversized: 0,
            blas_error_max: 0.0,
            warmup_frames_remaining: 0,
            warmup_timeout_reported: false,
            warmup_progress_log_frames_remaining: 0,
            warmup_started_at: None,
            spawned: 0,
            mesh_raster: 0,
            parts_failed: 0,
        }
    }
}

/// Polls the bake, streams finished meshes in, and moves the state on.
pub fn stream_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut pending: ResMut<PendingScene>,
    scene: Res<SceneData>,
    parts: Res<Assets<ZorahPart>>,
    mut material_cache: ResMut<MaterialCache>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    options: Res<SpawnOptions>,
    render_device: Res<RenderDevice>,
    state: Res<State<ZorahState>>,
    mut next_state: ResMut<NextState<ZorahState>>,
) {
    let pending = &mut *pending;
    poll_bake(pending, &scene);
    request_part_loads(pending, &asset_server);
    settle_loads(pending, &scene, &asset_server, &parts, &options);
    spawn_instances(
        pending,
        &scene,
        &mut commands,
        &asset_server,
        &mut material_cache,
        &mut materials,
        &options,
    );

    let bake_done = pending.bake.is_none();
    if *state == ZorahState::Baking && bake_done {
        next_state.set(ZorahState::LoadingScene);
        return;
    }
    if !(bake_done && pending.loading.is_empty() && pending.spawning.is_empty()) {
        return;
    }
    report_residency(pending);

    // Solari's plugins register nothing when the adapter lacks ray queries,
    // leaving the readiness counters at zero forever.
    let raytracing_supported = render_device
        .features()
        .contains(SolariPlugins::required_wgpu_features());
    if !raytracing_supported {
        warn!(
            missing_features = ?SolariPlugins::required_wgpu_features()
                .difference(render_device.features()),
            "this GPU or backend cannot run Solari; rendering meshlet raster only"
        );
        info!(
            "Zorah raster-only scene ready: spawned={} mesh_raster={} parts_failed={}",
            pending.spawned, pending.mesh_raster, pending.parts_failed
        );
        next_state.set(ZorahState::Running);
        return;
    }
    pending.expected_blas = pending
        .raytracing_instances
        .iter()
        .map(|instance| instance.mesh.id())
        .collect::<HashSet<_>>()
        .len();
    pending.warmup_frames_remaining = pending
        .blas_triangles
        .div_ceil(BLAS_BUILD_TRIANGLES_PER_FRAME)
        .saturating_add(BLAS_WARMUP_MARGIN_FRAMES);
    pending.warmup_timeout_reported = false;
    pending.warmup_progress_log_frames_remaining = 0;
    pending.warmup_started_at = Some(Instant::now());
    info!(
        spawned = pending.spawned,
        mesh_raster = pending.mesh_raster,
        parts_failed = pending.parts_failed,
        materials = material_cache.created,
        textures = material_cache.textures_requested,
        expected_blas = pending.expected_blas,
        diagnostic_timeout_frames = pending.warmup_frames_remaining,
        "Zorah raster scene submitted; waiting for measured BLAS readiness",
    );
    next_state.set(ZorahState::WarmingRaytracing);
}

/// Logs what the pruned scene asks of the meshlet manager, once every part is
/// in. Over the budget, the parts that did not fit have already failed to
/// upload (the manager logs each) and are missing from the raster image;
/// the fix is a coarser `--raster-error`. Under it, the sum is only an upper
/// bound on fitting: parts pack best-fit into 64 MiB pages and cannot
/// straddle one, so a total near the budget can still lose its last parts,
/// and a part over a page (counted and warned about as it loads) never
/// uploads.
fn report_residency(pending: &PendingScene) {
    let share = pending.meshlet_bytes as f64 * 100.0 / MESHLET_PAGE_BUDGET as f64;
    if pending.meshlet_bytes > MESHLET_PAGE_BUDGET {
        warn!(
            "pruned meshlet data is {} ({share:.0}% of the {} page budget): parts past the budget do not render; raise --raster-error",
            human_bytes(pending.meshlet_bytes),
            human_bytes(MESHLET_PAGE_BUDGET)
        );
    } else if pending.parts_oversized > 0 {
        warn!(
            "pruned meshlet data is {} ({share:.0}% of the {} page budget), but {} part(s) exceed the {} page and never render; raise --raster-error",
            human_bytes(pending.meshlet_bytes),
            human_bytes(MESHLET_PAGE_BUDGET),
            pending.parts_oversized,
            human_bytes(MESHLET_PAGE_SIZE)
        );
    } else {
        info!(
            "pruned meshlet data is {} ({share:.0}% of the {} page budget), {} triangles across LODs; BLAS cuts total {} triangles, achieved error at most {:.4} m",
            human_bytes(pending.meshlet_bytes),
            human_bytes(MESHLET_PAGE_BUDGET),
            pending.meshlet_triangles,
            pending.blas_triangles,
            pending.blas_error_max
        );
    }
}

/// Drains finished bake jobs into the load queue; drops the handle once the
/// bake is over so its workers are joined.
fn poll_bake(pending: &mut PendingScene, scene: &SceneData) {
    let Some(bake) = &pending.bake else {
        return;
    };
    // Workers send a job's event before counting it, so the counters are read
    // first: every event of the jobs this snapshot counts is then in the
    // channel, and none is lost when the handle is dropped below.
    let progress = bake.progress();
    while let Some(event) = bake.try_recv() {
        match event {
            BakeEvent::Complete {
                job,
                manifest,
                reused,
            } => {
                if !reused {
                    info!(
                        "baked {} ({}): {} parts, {} triangles",
                        manifest.mesh_name,
                        manifest.stem,
                        manifest.parts.len(),
                        manifest.source_triangles
                    );
                }
                if scene.meshes_of_job.get(job).is_some_and(|meshes| {
                    meshes
                        .iter()
                        .any(|mesh| scene.instances_by_mesh.contains_key(mesh))
                }) {
                    pending.loading.push_back(LoadingJob {
                        job,
                        manifest,
                        parts: Vec::new(),
                    });
                }
            }
            BakeEvent::Failed { job, error } => {
                pending.bake_failed += 1;
                let name = scene
                    .meshes_of_job
                    .get(job)
                    .and_then(|meshes| meshes.first())
                    .map_or(String::from("?"), |mesh| format!("mesh {mesh}"));
                error!("bake job {job} ({name}) failed: {error}");
            }
        }
    }
    if progress.is_finished() {
        info!(
            "bake complete: {} baked, {} reused, {} failed of {} geometry files in {:.1?}",
            progress.baked, progress.reused, progress.failed, progress.total, progress.elapsed
        );
        if let Some(bake) = pending.bake.take() {
            bake.stop();
        }
    } else if pending.bake_last_report.elapsed() >= BAKE_PROGRESS_INTERVAL {
        info!(
            "baked {}/{} meshes ({} reused, {} failed), {} partitions written, elapsed {:.0?}",
            progress.finished,
            progress.total,
            progress.reused,
            progress.failed,
            progress.partitions,
            progress.elapsed
        );
        pending.bake_last_report = Instant::now();
    }
}

/// Starts up to the per-frame budget of part loads, oldest manifests first.
fn request_part_loads(pending: &mut PendingScene, asset_server: &AssetServer) {
    let mut started = 0;
    for loading in pending.loading.iter_mut() {
        while loading.parts.len() < loading.manifest.parts.len() {
            if started >= MAX_NEW_PART_LOADS_PER_FRAME {
                return;
            }
            let part = &loading.manifest.parts[loading.parts.len()];
            let stem = &loading.manifest.stem;
            loading
                .parts
                .push(asset_server.load(format!("cache://{stem}/{}", part.meshlet_file)));
            started += 1;
        }
    }
}

/// Moves jobs whose every part has loaded (or failed) to the spawn queue.
fn settle_loads(
    pending: &mut PendingScene,
    scene: &SceneData,
    asset_server: &AssetServer,
    parts: &Assets<ZorahPart>,
    options: &SpawnOptions,
) {
    let mut index = 0;
    while index < pending.loading.len() {
        let loading = &pending.loading[index];
        let all_requested = loading.parts.len() == loading.manifest.parts.len();
        let mut all_settled = all_requested;
        let mut failed = Vec::new();
        for (part_index, handle) in loading.parts.iter().enumerate() {
            match asset_server.load_state(handle) {
                LoadState::Failed(error) => {
                    error!(
                        "{}/{}: {error}",
                        loading.manifest.stem, loading.manifest.parts[part_index].meshlet_file
                    );
                    failed.push(part_index);
                }
                LoadState::Loaded => {}
                _ => all_settled = false,
            }
        }
        if !all_settled {
            index += 1;
            continue;
        }
        let Some(loading) = pending.loading.remove(index) else {
            break;
        };
        if !failed.is_empty() {
            // The manifest only checks that its files exist, so a truncated
            // part would otherwise be reused, and fail, on every run.
            let manifest_path = options
                .cache_dir
                .join(&loading.manifest.stem)
                .join(MANIFEST_FILE);
            match std::fs::remove_file(&manifest_path) {
                Ok(()) => warn!(
                    "{}: {} part(s) failed to load; removed its manifest so the next run rebakes it",
                    loading.manifest.stem,
                    failed.len()
                ),
                Err(error) => warn!(
                    "{}: {} part(s) failed to load and {} could not be removed ({error}); delete the directory to rebake it",
                    loading.manifest.stem,
                    failed.len(),
                    manifest_path.display()
                ),
            }
        }
        pending.parts_failed += failed.len();
        let mut loaded = Vec::with_capacity(loading.parts.len());
        for (part_index, handle) in loading.parts.into_iter().enumerate() {
            if failed.contains(&part_index) {
                continue;
            }
            let Some(part) = parts.get(&handle) else {
                // Loaded, so it is in the store; treat anything else as a
                // failure rather than spawning a part with no geometry.
                pending.parts_failed += 1;
                error!(
                    "{}/{}: loaded but absent from the asset store",
                    loading.manifest.stem, loading.manifest.parts[part_index].meshlet_file
                );
                continue;
            };
            pending.blas_triangles += part.stats.blas_triangles;
            pending.meshlet_bytes += part.stats.packed_bytes;
            if !fits_a_page(part.stats.packed_bytes) {
                pending.parts_oversized += 1;
                warn!(
                    "{}/{}: pruned to {}, more than the {} page an upload must fit in; it never renders. Raise --raster-error or lower --partition-triangles and rebake",
                    loading.manifest.stem,
                    loading.manifest.parts[part_index].meshlet_file,
                    human_bytes(part.stats.packed_bytes),
                    human_bytes(MESHLET_PAGE_SIZE)
                );
            }
            pending.meshlet_triangles += part.stats.raster_triangles;
            pending.blas_error_max = pending.blas_error_max.max(part.stats.blas_achieved_error);
            loaded.push(PartAssets {
                manifest: loading.manifest.parts[part_index].clone(),
                meshlet: part.meshlet.clone(),
                blas: part.blas.clone(),
                stats: part.stats,
            });
        }
        let instances = scene
            .meshes_of_job
            .get(loading.job)
            .map(|meshes| {
                meshes
                    .iter()
                    .filter_map(|mesh| scene.instances_by_mesh.get(mesh))
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        pending.spawning.push_back(SpawnJob {
            job: loading.job,
            parts: loaded,
            instances,
            cursor: 0,
        });
    }
}

/// Spawns whole instances (every part of one placement) until the budget is
/// spent; the first instance of a frame always goes through.
fn spawn_instances(
    pending: &mut PendingScene,
    scene: &SceneData,
    commands: &mut Commands,
    asset_server: &AssetServer,
    material_cache: &mut MaterialCache,
    materials: &mut Assets<StandardMaterial>,
    options: &SpawnOptions,
) {
    let mut spawned_this_frame = 0;
    while let Some(job) = pending.spawning.front_mut() {
        while job.cursor < job.instances.len() {
            if spawned_this_frame != 0
                && spawned_this_frame + job.parts.len() > MAX_SPAWNED_INSTANCES_PER_FRAME
            {
                return;
            }
            let instance = &scene.instances[job.instances[job.cursor]];
            job.cursor += 1;
            let primitive_materials = scene.primitive_materials.get(instance.mesh);
            // A sheared placement keeps its node chain: the ancestors become
            // parent entities and the parts take the last local transform,
            // so the propagated affine `GlobalTransform` is the file's matrix.
            let (parent, transform) = match instance.chain.split_last() {
                Some((last, ancestors)) => {
                    let mut parent = None;
                    for (depth, local) in ancestors.iter().enumerate() {
                        let mut entity = commands.spawn((
                            Name::new(format!(
                                "{} (node {}) ancestor {depth}",
                                instance.name, instance.node
                            )),
                            *local,
                            Visibility::default(),
                        ));
                        if let Some(parent) = parent {
                            entity.insert(ChildOf(parent));
                        }
                        parent = Some(entity.id());
                    }
                    (parent, *last)
                }
                None => (None, instance.transform),
            };
            for part in &job.parts {
                // The manifest records the first sharing mesh's material;
                // the root glTF's own primitive is the authority for this mesh.
                let material_index = primitive_materials
                    .and_then(|materials| materials.get(part.manifest.primitive).copied())
                    .unwrap_or(part.manifest.material);
                let slot = material_cache.get(material_index, asset_server, materials);
                let name = Name::new(format!(
                    "{} (node {}) p{}_{}",
                    instance.name, instance.node, part.manifest.primitive, part.manifest.partition
                ));
                let mut entity = if slot.mesh_raster {
                    pending.mesh_raster += 1;
                    commands.spawn((
                        name,
                        Mesh3d(part.blas.clone()),
                        MeshMaterial3d(slot.handle.clone()),
                        transform,
                    ))
                } else {
                    commands.spawn((
                        name,
                        MeshletMesh3d(part.meshlet.clone()),
                        MeshMaterial3d(slot.handle.clone()),
                        transform,
                    ))
                };
                if let Some(parent) = parent {
                    entity.insert(ChildOf(parent));
                }
                let entity = entity.id();
                // Without thin-wall transmission in Solari, glass in the BLAS
                // would be an opaque wall to sky light; leaving it out lets
                // the sky into the throne room through its windows.
                if options.glass_in_blas || !slot.transmissive {
                    pending
                        .raytracing_instances
                        .push(PendingRaytracingInstance {
                            entity,
                            mesh: part.blas.clone(),
                            geometry_error: part.stats.blas_achieved_error,
                        });
                }
                pending.spawned += 1;
                spawned_this_frame += 1;
            }
        }
        let job = pending.spawning.pop_front().expect("front exists");
        debug!(
            "spawned job {} across {} instances, {} parts",
            job.job,
            job.instances.len(),
            job.parts.len()
        );
    }
}

/// Attaches `RaytracingMesh3d` in batches, then waits for Solari's counters
/// to settle before enabling it (and DLSS-RR).
pub fn warm_up_raytracing(
    mut commands: Commands,
    mut pending: ResMut<PendingScene>,
    raytracing_status: Res<RaytracingSceneStatus>,
    camera: Single<Entity, With<ZorahCamera>>,
    mut ambient_light: ResMut<GlobalAmbientLight>,
    mut next_state: ResMut<NextState<ZorahState>>,
    #[cfg(feature = "dlss")] dlss_rr_supported: Option<Res<DlssRayReconstructionSupported>>,
) {
    // Solari enqueues a BLAS build per extracted compatible `Mesh` asset, so
    // that work already started as the parts loaded. Attaching
    // `RaytracingMesh3d` only registers TLAS instances; batch it anyway to
    // keep the per-frame instance extraction bounded.
    let end = pending
        .raytracing_cursor
        .saturating_add(MAX_RAYTRACING_INSTANCES_PER_FRAME)
        .min(pending.raytracing_instances.len());
    for instance in &pending.raytracing_instances[pending.raytracing_cursor..end] {
        commands.entity(instance.entity).insert((
            RaytracingMesh3d(instance.mesh.clone()),
            MeshGeometryError(instance.geometry_error),
        ));
    }
    pending.raytracing_cursor = end;
    if end != pending.raytracing_instances.len() {
        return;
    }

    let snapshot = raytracing_status.snapshot();
    let settled = snapshot.is_settled_for(pending.expected_blas);
    let elapsed_seconds = pending
        .warmup_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f32());
    if pending.warmup_progress_log_frames_remaining == 0 {
        info!(
            expected_blas = pending.expected_blas,
            available_blas = snapshot.available_blas,
            queued_builds = snapshot.queued_builds,
            allocator_waiting = snapshot.allocator_waiting,
            pending_compactions = snapshot.pending_compactions,
            compacted_blas = snapshot.compacted_blas,
            failed_compactions = snapshot.failed_compactions,
            compaction_disabled = snapshot.compaction_disabled,
            elapsed_seconds,
            settled,
            "Zorah BLAS preparation progress",
        );
        pending.warmup_progress_log_frames_remaining = BLAS_PROGRESS_LOG_INTERVAL_FRAMES;
    } else {
        pending.warmup_progress_log_frames_remaining -= 1;
    }

    if !settled {
        if pending.warmup_frames_remaining != 0 {
            pending.warmup_frames_remaining -= 1;
        } else if !pending.warmup_timeout_reported {
            pending.warmup_timeout_reported = true;
            error!(
                expected_blas = pending.expected_blas,
                available_blas = snapshot.available_blas,
                queued_builds = snapshot.queued_builds,
                allocator_waiting = snapshot.allocator_waiting,
                elapsed_seconds,
                "Zorah BLAS preparation exceeded its conservative diagnostic estimate; continuing to wait rather than enabling Solari early",
            );
        }
        return;
    }

    // The preview ambient has done its job. Deferred and meshlet geometry
    // skip it under Solari anyway, but a forward-shaded `Mesh3d` (a BLEND
    // material under `--preserve-alpha`) would add it on top of the traced light.
    *ambient_light = GlobalAmbientLight::NONE;
    let mut camera = commands.entity(*camera);
    camera.insert(SolariLighting {
        // Upstream turned ReSTIR off by default and leans on the denoiser instead. Zorah's
        // reservoir merges carry the raster-LOD ray bias and the envmap MIS, so keep the
        // resampled path until the plain path tracer is compared against it.
        restir: true,
        // Scene tuning: Zorah's throne room and courtyard have sightlines well
        // past the 50 m default, at which a world-cache GI ray is truncated and
        // contributes nothing, leaving the large interiors short of bounce
        // energy.
        world_cache_max_gi_ray_distance: 200.0,
        ..default()
    });
    // Ray Reconstruction consumes guide buffers produced by Solari, so enable
    // it at the same measured-ready transition rather than during BLAS warmup.
    #[cfg(feature = "dlss")]
    if dlss_rr_supported.is_some() {
        camera.insert(Dlss::<DlssRayReconstructionFeature> {
            perf_quality_mode: DlssPerfQualityMode::Auto,
            reset: Default::default(),
            _phantom_data: Default::default(),
        });
        info!("DLSS Ray Reconstruction enabled");
    }
    info!(
        spawned = pending.spawned,
        mesh_raster = pending.mesh_raster,
        parts_failed = pending.parts_failed,
        bake_failed = pending.bake_failed,
        expected_blas = pending.expected_blas,
        available_blas = snapshot.available_blas,
        compacted_blas = snapshot.compacted_blas,
        failed_compactions = snapshot.failed_compactions,
        elapsed_seconds,
        "Zorah ready: Solari enabled over meshlet raster with measured-ready BLAS instances",
    );
    next_state.set(ZorahState::Running);
}
