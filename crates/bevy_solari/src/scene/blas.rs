use alloc::collections::VecDeque;
use alloc::sync::Arc;
use bevy_asset::AssetId;
use bevy_ecs::{
    resource::Resource,
    system::{Res, ResMut},
    world::World,
};
use bevy_mesh::{
    DecodedPositions, Indices, Mesh, MeshAttributeCompressionFlags, MeshRaytracingFlags,
    MeshVertexAttribute, VertexFormat,
};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::settings::WgpuFeatures;
use bevy_render::{
    diagnostic::{DiagnosticsRecorder, RecordDiagnostics},
    extract_resource::ExtractResource,
    mesh::{
        allocator::{MeshAllocator, MeshBufferSlice},
        RenderMesh,
    },
    render_asset::ExtractedAssets,
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
    RenderApp,
};
use bevy_tasks::futures::now_or_never;
use bevy_utils::once;
use core::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use tracing::warn;

/// Limits BLAS build and compaction submissions without unnecessarily throttling high-end GPUs.
///
/// Building every mesh extracted in one frame has a very large transient cost: all un-compacted
/// BLAS allocations coexist until asynchronous compaction catches up. Large scenes can exhaust
/// VRAM even when their final compacted acceleration structures would fit.
const MAX_BUILD_VERTICES_PER_FRAME: u32 = 2_000_000;
const MAX_COMPACTION_VERTICES_PER_FRAME: u32 = 2_000_000;
/// Kept tight not only for the transient VRAM spike itself: every un-compacted BLAS in flight
/// interleaves with permanent allocations in the allocator's memory blocks, and a block that
/// picked up even one long-lived allocation is never returned to the driver when the doomed ones
/// free. A large-scene load with a wide window has been measured leaving gigabytes of
/// reserved-but-dead block space behind.
const MAX_UNCOMPACTED_VERTICES: u32 = 4_000_000;

/// How many frames a queued build may be rotated for missing mesh allocator slices before it is
/// dropped. Progressive loading routinely takes a few frames; a mesh the allocator will never serve
/// would otherwise keep `queued_builds` nonzero forever and wedge every readiness check.
const MAX_ALLOCATOR_DEFERRALS: u32 = 600;

/// Under the `wgpu_hal` build path, we need to manage BLAS lifetimes ourselves.
/// Since solari keeps both current and previous frame TLAS's around, only after
/// two TLAS builds since we marked it for deletion is it safe to delete a BLAS.
const TLAS_BUILDS_BEFORE_DELETION_ALLOWED: usize = 2;

const COMPACTION_NOT_STARTED: u8 = 0;
const COMPACTION_PENDING: u8 = 1;
const COMPACTION_READY: u8 = 2;
const COMPACTION_FAILED: u8 = 3;

fn record_compaction_result(state: &AtomicU8, succeeded: bool) {
    state.store(
        if succeeded {
            COMPACTION_READY
        } else {
            COMPACTION_FAILED
        },
        Ordering::Release,
    );
}

struct PendingCompaction {
    key: BlasKey,
    vertex_count: u32,
    state: Arc<AtomicU8>,
}

/// How a raytracing compatible mesh's vertices are laid out, recorded when the mesh is extracted
/// because its BLAS may be built frames later.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlasVertexLayout {
    /// Bytes per vertex.
    pub stride: u32,
    /// Whether the vertices end in a tangent. Without one, the shaders reconstruct the tangent
    /// frame from the UVs.
    pub has_tangent: bool,
    /// The format of the positions the BLAS is built from.
    position_format: VertexFormat,
    /// The transform from compressed positions back to mesh space, if positions are compressed.
    position_transform: Option<[f32; 12]>,
}

impl BlasVertexLayout {
    fn new(mesh: &Mesh) -> Self {
        // Compressed positions are relative to the mesh's AABB, so their geometry is built with a
        // transform back to mesh space.
        let position_transform = match mesh.decoded_positions() {
            Some(DecodedPositions::Compressed {
                center, half_size, ..
            }) => Some([
                half_size.x,
                0.0,
                0.0,
                center.x, //
                0.0,
                half_size.y,
                0.0,
                center.y, //
                0.0,
                0.0,
                half_size.z,
                center.z,
            ]),
            _ => None,
        };
        Self {
            stride: mesh.get_vertex_size() as u32,
            has_tangent: mesh.contains_attribute(Mesh::ATTRIBUTE_TANGENT),
            position_format: if position_transform.is_some() {
                VertexFormat::Snorm16x4
            } else {
                Mesh::ATTRIBUTE_POSITION.format
            },
            position_transform,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct QueuedBuild {
    key: BlasKey,
    vertex_layout: BlasVertexLayout,
    index_format: IndexFormat,
    /// Frames this entry was rotated because the mesh allocator had no slices for it.
    allocator_deferrals: u32,
}

#[derive(Resource, Default)]
pub struct BlasManager {
    blas: HashMap<AssetId<Mesh>, MeshBlas>,
    /// How many BLASes `blas` holds, across both opacities.
    blas_count: usize,
    vertex_layouts: HashMap<AssetId<Mesh>, BlasVertexLayout>,
    build_queue: VecDeque<QueuedBuild>,
    compaction_queue: VecDeque<PendingCompaction>,
    compacted: HashSet<BlasKey>,
    failed_compactions: HashSet<BlasKey>,
    compaction_disabled: bool,
    allocator_waiting: usize,
    changed: Vec<AssetId<Mesh>>,
    /// BLAS that are pending deletion, one batch per TLAS build. The back batch collects
    /// retirements since the last build, and every batch ahead of it has one more build to wait
    /// out.
    pending_deletions: VecDeque<Vec<Blas>>,
}

impl BlasManager {
    pub fn get(&self, key: &BlasKey) -> Option<&Blas> {
        self.blas.get(&key.mesh)?.get(key.opacity)
    }

    pub fn device_address(&self, key: &BlasKey) -> Option<u64> {
        self.get(key)?.handle()
    }

    /// If a mesh is raytracing compatible, but its [`Mesh::raytracing`] flags don't declare
    /// the required opacity.
    pub fn is_undeclared(&self, key: &BlasKey) -> bool {
        self.blas
            .get(&key.mesh)
            .is_some_and(|mesh| !mesh.requires.contains(key.opacity.flag()))
    }

    fn require(
        &mut self,
        mesh: AssetId<Mesh>,
        flags: MeshRaytracingFlags,
        index_format: IndexFormat,
    ) {
        let blas = self.blas.entry(mesh).or_default();
        blas.requires = flags;
        blas.index_format = index_format;
    }

    pub fn index_format(&self, mesh: &AssetId<Mesh>) -> Option<IndexFormat> {
        self.blas.get(mesh).map(|blas| blas.index_format)
    }

    pub fn vertex_layout(&self, mesh: &AssetId<Mesh>) -> Option<BlasVertexLayout> {
        self.vertex_layouts.get(mesh).copied()
    }

    pub fn changed_meshes(&self) -> &[AssetId<Mesh>] {
        &self.changed
    }

    pub fn note_tlas_build(&mut self) {
        if !self.pending_deletions.is_empty() {
            self.pending_deletions.push_back(Vec::new());
        }
    }

    fn insert(&mut self, key: BlasKey, blas: Blas) {
        let slot = self.blas.entry(key.mesh).or_default().slot_mut(key.opacity);
        match slot.replace(blas) {
            Some(old) => self.retire(old),
            None => self.blas_count += 1,
        }

        self.changed.push(key.mesh);
    }

    /// Retires every BLAS of `mesh` and drops all queued work for it.
    fn remove_mesh(&mut self, mesh: AssetId<Mesh>) {
        self.changed.push(mesh);
        self.vertex_layouts.remove(&mesh);
        self.build_queue.retain(|build| build.key.mesh != mesh);
        self.compaction_queue
            .retain(|pending| pending.key.mesh != mesh);
        self.compacted.retain(|key| key.mesh != mesh);
        self.failed_compactions.retain(|key| key.mesh != mesh);

        if let Some(removed) = self.blas.remove(&mesh) {
            for blas in removed.into_iter() {
                self.blas_count -= 1;
                self.retire(blas);
            }
        }
    }

    /// Queues a replaced or removed BLAS for deletion once no in-flight TLAS can reference it.
    fn retire(&mut self, blas: Blas) {
        match self.pending_deletions.back_mut() {
            Some(batch) => batch.push(blas),
            None => self.pending_deletions.push_back(vec![blas]),
        }
    }

    fn defer_build_waiting_for_allocator(&mut self) {
        let Some(mut build) = self.build_queue.pop_front() else {
            return;
        };
        build.allocator_deferrals += 1;
        if build.allocator_deferrals >= MAX_ALLOCATOR_DEFERRALS {
            warn!(
                key = ?build.key,
                "mesh allocator never produced slices for this mesh; dropping its BLAS build so the raytracing scene can settle"
            );
            return;
        }
        self.build_queue.push_back(build);
        self.allocator_waiting += 1;
    }

    fn disable_compaction(&mut self, failed: BlasKey) {
        self.failed_compactions.insert(failed);
        self.compaction_disabled = true;
        self.compaction_queue.clear();
    }
}

/// A main-world readable snapshot of Solari's asynchronous BLAS preparation state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RaytracingSceneStatusSnapshot {
    /// Number of BLASes currently available for TLAS binding. A mesh that declares both
    /// opacities has two.
    pub available_blas: usize,
    /// Number of BLASes still waiting to be built.
    pub queued_builds: usize,
    /// Number of queued builds whose mesh allocator slices were not available on the last frame.
    pub allocator_waiting: usize,
    /// Number of valid BLASes still waiting for asynchronous compaction.
    pub pending_compactions: usize,
    /// Number of BLASes that were successfully compacted.
    pub compacted_blas: usize,
    /// Number of BLASes whose compaction failed. Their valid un-compacted BLASes are retained.
    pub failed_compactions: usize,
    /// Whether further compaction was disabled after a failure.
    pub compaction_disabled: bool,
}

impl RaytracingSceneStatusSnapshot {
    /// Returns true once at least `expected_blas` are available and no queued build or compaction
    /// work remains. Failed compactions count as settled because their valid un-compacted BLASes
    /// remain available.
    pub fn is_settled_for(&self, expected_blas: usize) -> bool {
        self.available_blas >= expected_blas
            && self.queued_builds == 0
            && self.allocator_waiting == 0
            && self.pending_compactions == 0
    }
}

#[derive(Default)]
struct RaytracingSceneStatusCounters {
    available_blas: AtomicUsize,
    queued_builds: AtomicUsize,
    allocator_waiting: AtomicUsize,
    pending_compactions: AtomicUsize,
    compacted_blas: AtomicUsize,
    failed_compactions: AtomicUsize,
    compaction_disabled: AtomicBool,
}

/// Shared BLAS preparation status.
///
/// This resource is cloned into the render world but keeps shared atomic counters, allowing
/// main-world loading states to observe render-world BLAS progress without guessing a frame count.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct RaytracingSceneStatus {
    counters: Arc<RaytracingSceneStatusCounters>,
}

impl RaytracingSceneStatus {
    /// Returns the most recently published render-world BLAS preparation counters.
    pub fn snapshot(&self) -> RaytracingSceneStatusSnapshot {
        RaytracingSceneStatusSnapshot {
            available_blas: self.counters.available_blas.load(Ordering::Acquire),
            queued_builds: self.counters.queued_builds.load(Ordering::Acquire),
            allocator_waiting: self.counters.allocator_waiting.load(Ordering::Acquire),
            pending_compactions: self.counters.pending_compactions.load(Ordering::Acquire),
            compacted_blas: self.counters.compacted_blas.load(Ordering::Acquire),
            failed_compactions: self.counters.failed_compactions.load(Ordering::Acquire),
            compaction_disabled: self.counters.compaction_disabled.load(Ordering::Acquire),
        }
    }

    fn publish(&self, manager: &BlasManager) {
        self.counters
            .available_blas
            .store(manager.blas_count, Ordering::Release);
        self.counters
            .queued_builds
            .store(manager.build_queue.len(), Ordering::Release);
        self.counters
            .allocator_waiting
            .store(manager.allocator_waiting, Ordering::Release);
        self.counters
            .pending_compactions
            .store(manager.compaction_queue.len(), Ordering::Release);
        self.counters
            .compacted_blas
            .store(manager.compacted.len(), Ordering::Release);
        self.counters
            .failed_compactions
            .store(manager.failed_compactions.len(), Ordering::Release);
        self.counters
            .compaction_disabled
            .store(manager.compaction_disabled, Ordering::Release);
    }
}

pub fn update_raytracing_scene_status(
    blas_manager: Res<BlasManager>,
    status: Res<RaytracingSceneStatus>,
) {
    status.publish(&blas_manager);
}

struct MeshBlas {
    requires: MeshRaytracingFlags,
    index_format: IndexFormat,
    opaque: Option<Blas>,
    non_opaque: Option<Blas>,
}

impl Default for MeshBlas {
    fn default() -> Self {
        Self {
            requires: MeshRaytracingFlags::empty(),
            index_format: IndexFormat::Uint32,
            opaque: None,
            non_opaque: None,
        }
    }
}

impl MeshBlas {
    fn get(&self, opacity: BlasOpacity) -> Option<&Blas> {
        match opacity {
            BlasOpacity::Opaque => self.opaque.as_ref(),
            BlasOpacity::NonOpaque => self.non_opaque.as_ref(),
        }
    }

    fn slot_mut(&mut self, opacity: BlasOpacity) -> &mut Option<Blas> {
        match opacity {
            BlasOpacity::Opaque => &mut self.opaque,
            BlasOpacity::NonOpaque => &mut self.non_opaque,
        }
    }

    fn into_iter(self) -> impl Iterator<Item = Blas> {
        self.opaque.into_iter().chain(self.non_opaque)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlasKey {
    pub mesh: AssetId<Mesh>,
    pub opacity: BlasOpacity,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BlasOpacity {
    Opaque,
    NonOpaque,
}

impl BlasOpacity {
    fn declared(flags: MeshRaytracingFlags) -> impl Iterator<Item = Self> {
        [Self::Opaque, Self::NonOpaque]
            .into_iter()
            .filter(move |opacity: &BlasOpacity| flags.contains(opacity.flag()))
    }

    pub fn flag(self) -> MeshRaytracingFlags {
        match self {
            Self::Opaque => MeshRaytracingFlags::OPAQUE,
            Self::NonOpaque => MeshRaytracingFlags::NON_OPAQUE,
        }
    }

    fn geometry_flags(self) -> AccelerationStructureGeometryFlags {
        match self {
            Self::Opaque => AccelerationStructureGeometryFlags::OPAQUE,
            Self::NonOpaque => AccelerationStructureGeometryFlags::empty(),
        }
    }
}

/// A BLAS built this frame.
struct BlasInput<'a> {
    key: BlasKey,
    vertex_stride: u64,
    vertex_slice: MeshBufferSlice<'a>,
    index_slice: MeshBufferSlice<'a>,
    size: BlasTriangleGeometrySizeDescriptor,
    /// The offset of the transform from compressed positions to mesh space, if positions are
    /// compressed.
    transform_offset: Option<BufferAddress>,
}

pub fn prepare_raytracing_blas(
    mut blas_manager: ResMut<BlasManager>,
    extracted_meshes: Res<ExtractedAssets<RenderMesh>>,
    mesh_allocator: Res<MeshAllocator>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut diagnostics: Option<ResMut<DiagnosticsRecorder>>,
) {
    blas_manager.changed.clear();

    // Delete BLAS and queued work for deleted or modified meshes.
    for asset_id in extracted_meshes
        .removed
        .iter()
        .chain(extracted_meshes.modified.iter())
    {
        blas_manager.remove_mesh(*asset_id);
    }

    // Retain newly extracted meshes in a persistent queue. The mesh allocator keeps their GPU
    // input slices alive, so later frames can build them without retaining ExtractedAssets.
    // Each queues one build per opacity it declares, and records the declaration even if empty.
    let features = render_device.features();
    for (asset_id, mesh) in &extracted_meshes.extracted {
        let Some(index_format) = raytracing_index_format(mesh, features) else {
            continue;
        };
        blas_manager.require(*asset_id, mesh.raytracing, index_format);
        if blas_manager.vertex_layouts.contains_key(asset_id) {
            continue;
        }
        let vertex_layout = BlasVertexLayout::new(mesh);
        blas_manager.vertex_layouts.insert(*asset_id, vertex_layout);
        for opacity in BlasOpacity::declared(mesh.raytracing) {
            blas_manager.build_queue.push_back(QueuedBuild {
                key: BlasKey {
                    mesh: *asset_id,
                    opacity,
                },
                vertex_layout,
                index_format,
                allocator_deferrals: 0,
            });
        }
    }

    let mut uncompacted_vertices = if blas_manager.compaction_disabled {
        0
    } else {
        blas_manager
            .compaction_queue
            .iter()
            .map(|pending| pending.vertex_count)
            .sum::<u32>()
    };
    // Keep the last measured value on throttled frames: the queue is not examined below, so zeroing
    // it here would report "nothing is allocator-blocked" without having looked.
    if !blas_manager.compaction_disabled && uncompacted_vertices >= MAX_UNCOMPACTED_VERTICES {
        return;
    }
    blas_manager.allocator_waiting = 0;

    // Allocate only as much new un-compacted geometry as the compactor can keep bounded. At
    // least one mesh is allowed when the queue is empty so a single unusually large mesh cannot
    // deadlock the queue. The position transforms of this frame's compressed builds share one
    // buffer.
    let mut blas_inputs = Vec::new();
    let mut transforms = Vec::<[f32; 12]>::new();
    let mut vertices_built = 0u32;
    let mut builds_to_check = blas_manager.build_queue.len();
    while builds_to_check != 0 {
        builds_to_check -= 1;
        let Some(QueuedBuild {
            key,
            vertex_layout,
            index_format,
            ..
        }) = blas_manager.build_queue.front().copied()
        else {
            break;
        };
        let (Some(vertex_slice), Some(index_slice)) = (
            mesh_allocator.mesh_vertex_slice(&key.mesh),
            mesh_allocator.mesh_index_slice(&key.mesh),
        ) else {
            // Mesh allocation normally precedes this system, but progressive loading and render
            // recovery can make the asset and its allocator slices arrive on different frames.
            // Keep the request alive and rotate it so one delayed mesh cannot block ready work.
            blas_manager.defer_build_waiting_for_allocator();
            continue;
        };
        let vertex_count = vertex_slice.range.len() as u32;
        if !blas_inputs.is_empty()
            && (vertices_built.saturating_add(vertex_count) > MAX_BUILD_VERTICES_PER_FRAME
                || (!blas_manager.compaction_disabled
                    && uncompacted_vertices.saturating_add(vertex_count)
                        > MAX_UNCOMPACTED_VERTICES))
        {
            break;
        }
        if !blas_manager.compaction_disabled
            && blas_inputs.is_empty()
            && uncompacted_vertices != 0
            && uncompacted_vertices.saturating_add(vertex_count) > MAX_UNCOMPACTED_VERTICES
        {
            break;
        }
        blas_manager.build_queue.pop_front();

        let (blas, size) = allocate_blas(
            &vertex_slice,
            &index_slice,
            vertex_layout.position_format,
            index_format,
            key,
            &render_device,
        );

        blas_manager.insert(key, blas);
        if !blas_manager.compaction_disabled {
            blas_manager.compaction_queue.push_back(PendingCompaction {
                key,
                vertex_count: size.vertex_count,
                state: Arc::new(AtomicU8::new(COMPACTION_NOT_STARTED)),
            });
        }
        vertices_built = vertices_built.saturating_add(size.vertex_count);
        if !blas_manager.compaction_disabled {
            uncompacted_vertices = uncompacted_vertices.saturating_add(size.vertex_count);
        }
        let transform_offset = vertex_layout.position_transform.map(|transform| {
            transforms.push(transform);
            ((transforms.len() - 1) * size_of::<[f32; 12]>()) as BufferAddress
        });
        blas_inputs.push(BlasInput {
            key,
            vertex_stride: vertex_layout.stride as u64,
            vertex_slice,
            index_slice,
            size,
            transform_offset,
        });
    }

    if blas_inputs.is_empty() {
        return;
    }

    let transform_buffer = (!transforms.is_empty()).then(|| {
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("blas_position_transforms"),
            contents: bytemuck::cast_slice(&transforms),
            usage: BufferUsages::BLAS_INPUT,
        })
    });

    // Build geometry into each BLAS
    let build_entries = blas_inputs
        .iter()
        .map(|input| {
            let geometry = BlasTriangleGeometry {
                size: &input.size,
                vertex_buffer: input.vertex_slice.buffer,
                first_vertex: input.vertex_slice.range.start,
                vertex_stride: input.vertex_stride,
                index_buffer: Some(input.index_slice.buffer),
                first_index: Some(input.index_slice.range.start),
                transform_buffer: input.transform_offset.and(transform_buffer.as_deref()),
                transform_buffer_offset: input.transform_offset,
            };
            BlasBuildEntry {
                blas: blas_manager.get(&input.key).unwrap(),
                geometry: BlasGeometries::TriangleGeometries(vec![geometry]),
            }
        })
        .collect::<Vec<_>>();

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("blas_build_command_encoder"),
    });
    let time_span = diagnostics
        .as_mut()
        .map(|diagnostics| diagnostics.time_span(&mut command_encoder, "blas_build"));
    command_encoder.build_acceleration_structures(&build_entries, &[]);
    if let Some(time_span) = time_span {
        time_span.end(&mut command_encoder);
    }
    render_queue.submit([command_encoder.finish()]);
}

/// Compacts BLASes without overlapping other render-world systems.
///
/// `wgpu-core` 30's `Queue::compact_blas` acquires its pending-writes and command-index locks in
/// the opposite order from `Queue::submit`. Running this as an exclusive system prevents a
/// concurrent queue submission from deadlocking with BLAS compaction. Keep this exclusive until
/// wgpu makes concurrent `compact_blas` and `submit` safe.
pub fn compact_raytracing_blas(world: &mut World) {
    let render_device = world.resource::<RenderDevice>().clone();
    let render_queue = world.resource::<RenderQueue>().clone();
    let mut blas_manager = world.resource_mut::<BlasManager>();
    compact_raytracing_blas_inner(&mut blas_manager, &render_device, &render_queue);
}

fn compact_raytracing_blas_inner(
    blas_manager: &mut BlasManager,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
) {
    if blas_manager.compaction_disabled {
        return;
    }

    let queue_size = blas_manager.compaction_queue.len();
    let mut meshes_processed = 0;
    let mut vertices_compacted = 0;

    while !blas_manager.compaction_queue.is_empty()
        && vertices_compacted < MAX_COMPACTION_VERTICES_PER_FRAME
        && meshes_processed < queue_size
    {
        meshes_processed += 1;

        let pending = blas_manager.compaction_queue.pop_front().unwrap();
        let key = pending.key;
        let vertex_count = pending.vertex_count;
        let Some(blas) = blas_manager.get(&key) else {
            continue;
        };

        match pending.state.load(Ordering::Acquire) {
            COMPACTION_NOT_STARTED => {
                pending.state.store(COMPACTION_PENDING, Ordering::Release);
                let callback_state = Arc::clone(&pending.state);
                blas.prepare_compaction_async(move |result| {
                    record_compaction_result(&callback_state, result.is_ok());
                });
                blas_manager.compaction_queue.push_back(pending);
            }
            COMPACTION_PENDING => blas_manager.compaction_queue.push_back(pending),
            COMPACTION_READY => match compact_blas_checked(blas, render_device, render_queue) {
                Some(compacted_blas) => {
                    blas_manager.insert(key, compacted_blas);
                    blas_manager.compacted.insert(key);
                    vertices_compacted += vertex_count;
                }
                None => {
                    warn!(
                        ?key,
                        "BLAS compaction failed; retaining valid un-compacted BLAS and disabling further compaction"
                    );
                    blas_manager.disable_compaction(key);
                    break;
                }
            },
            COMPACTION_FAILED => {
                warn!(
                    ?key,
                    "asynchronous BLAS compaction preparation failed; retaining valid un-compacted BLAS and disabling further compaction"
                );
                blas_manager.disable_compaction(key);
                break;
            }
            state => {
                warn!(?key, state, "BLAS entered an unknown compaction state");
                blas_manager.disable_compaction(key);
                break;
            }
        }
    }
}

pub fn delete_raytracing_blas(
    mut blas_manager: ResMut<BlasManager>,
    scene_bindings: Res<super::RaytracingSceneBindings>,
    render_queue: Res<RenderQueue>,
) {
    // A frame with no TLAS allocated is as strong a release point as a TLAS build: nothing
    // submitted or still to come can reference any BLAS, and the compaction copies are covered
    // by the completion callback below. Without this, a scene that builds and compacts its
    // BLASes before spawning its first raytracing instance never builds a TLAS, so not one
    // retired original is freed for the whole load; thousands of doomed allocations then
    // interleave with the load's permanent ones inside the allocator's shared memory blocks,
    // measured stranding gigabytes of reserved-but-dead VRAM that never returns to the driver.
    if scene_bindings.no_tlas_allocated() {
        blas_manager.note_tlas_build();
    }

    if blas_manager.pending_deletions.len() <= TLAS_BUILDS_BEFORE_DELETION_ALLOWED {
        return;
    }

    if let Some(deletable) = blas_manager
        .pending_deletions
        .pop_front()
        .filter(|b| !b.is_empty())
    {
        render_queue.on_submitted_work_done(move || drop(deletable));
    }
}

/// wgpu's public compaction API reports allocation and validation failures through error scopes
/// while returning an error handle. Only publish the compacted BLAS when both scopes completed
/// successfully. Keeping the source BLAS is always safe because compaction does not mutate it.
fn compact_blas_checked(
    blas: &Blas,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
) -> Option<Blas> {
    let error_scope_transaction = render_device.lock_error_scope_transaction();
    let validation_scope = render_device
        .wgpu_device()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let out_of_memory_scope = render_device
        .wgpu_device()
        .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal_scope = render_device
        .wgpu_device()
        .push_error_scope(wgpu::ErrorFilter::Internal);
    let compacted_blas = render_queue.compact_blas(blas);

    // These scopes complete immediately on wgpu's native backends. If a backend makes any
    // asynchronous, conservatively retain the source rather than publishing an unchecked handle.
    let internal_error = now_or_never(internal_scope.pop());
    let out_of_memory_error = now_or_never(out_of_memory_scope.pop());
    let validation_error = now_or_never(validation_scope.pop());
    drop(error_scope_transaction);
    if matches!(internal_error, Some(None))
        && matches!(out_of_memory_error, Some(None))
        && matches!(validation_error, Some(None))
    {
        Some(compacted_blas)
    } else {
        warn!(
            ?internal_error,
            ?out_of_memory_error,
            ?validation_error,
            "wgpu rejected BLAS compaction"
        );
        None
    }
}

fn allocate_blas(
    vertex_slice: &MeshBufferSlice,
    index_slice: &MeshBufferSlice,
    position_format: VertexFormat,
    index_format: IndexFormat,
    key: BlasKey,
    render_device: &RenderDevice,
) -> (Blas, BlasTriangleGeometrySizeDescriptor) {
    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: position_format,
        vertex_count: vertex_slice.range.len() as u32,
        index_format: Some(index_format),
        index_count: Some(index_slice.range.len() as u32),
        flags: key.opacity.geometry_flags(),
    };

    // TODO: If we ever introduce BLAS refits, we need to be aware of the TLAS double-buffer
    // to avoid invalidating the previous frame TLAS
    let blas = render_device.wgpu_device().create_blas(
        &CreateBlasDescriptor {
            label: Some(&format!("{} {:?}", key.mesh, key.opacity)),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE
                | AccelerationStructureFlags::ALLOW_COMPACTION
                | if position_format == Mesh::ATTRIBUTE_POSITION.format {
                    AccelerationStructureFlags::empty()
                } else {
                    AccelerationStructureFlags::USE_TRANSFORM
                },
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );

    (blas, blas_size)
}

/// The index format of a raytracing compatible mesh, or `None` if the mesh is not raytracing
/// compatible.
fn raytracing_index_format(mesh: &Mesh, features: WgpuFeatures) -> Option<IndexFormat> {
    const ATTRIBUTES: [MeshVertexAttribute; 4] = [
        Mesh::ATTRIBUTE_POSITION,
        Mesh::ATTRIBUTE_NORMAL,
        Mesh::ATTRIBUTE_UV_0,
        Mesh::ATTRIBUTE_TANGENT,
    ];
    let triangle_list = mesh.primitive_topology() == PrimitiveTopology::TriangleList;
    // Each attribute is either uncompressed or in the format its compression flag gives it. The
    // tangent is optional: without one, the shaders reconstruct the tangent frame from the UVs.
    let compression = mesh.attribute_compression();
    let expected = ATTRIBUTES.map(|attribute| {
        let format = match MeshAttributeCompressionFlags::for_attribute(attribute.id) {
            Some((flag, format)) if compression.contains(flag) => format,
            _ => attribute.format,
        };
        (attribute.id, format)
    });
    let vertex_attributes = mesh
        .attributes()
        .map(|(attribute, _)| (attribute.id, attribute.format))
        .collect::<Vec<_>>();
    let vertex_attributes =
        vertex_attributes.as_slice() == expected || vertex_attributes.as_slice() == &expected[..3];
    // Acceleration structures built from compressed positions need an extended vertex format.
    let position_format_supported = !compression
        .contains(MeshAttributeCompressionFlags::COMPRESS_POSITION)
        || features.contains(WgpuFeatures::EXTENDED_ACCELERATION_STRUCTURE_VERTEX_FORMATS);
    if !position_format_supported {
        once!(warn!(
            "Meshes with compressed positions are not raytraced, as the GPU lacks {:?}.",
            WgpuFeatures::EXTENDED_ACCELERATION_STRUCTURE_VERTEX_FORMATS
        ));
    }
    // An empty mesh can never be allocated a vertex buffer slice, so it would rotate in the build
    // queue forever and keep the scene from ever settling.
    if !(triangle_list
        && vertex_attributes
        && position_format_supported
        && mesh.count_vertices() != 0)
    {
        return None;
    }
    match mesh.indices()? {
        indices if indices.is_empty() => None,
        Indices::U16(_) => Some(IndexFormat::Uint16),
        Indices::U32(_) => Some(IndexFormat::Uint32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::RenderAssetUsages;
    use bevy_ecs::system::{IntoSystem, System};

    fn triangle_mesh() -> Mesh {
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_POSITION,
                vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 3])
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_UV_0,
                vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            )
            .with_inserted_indices(Indices::U32(vec![0, 1, 2]))
    }

    fn vertex_layout(stride: u32, has_tangent: bool) -> BlasVertexLayout {
        BlasVertexLayout {
            stride,
            has_tangent,
            position_format: Mesh::ATTRIBUTE_POSITION.format,
            position_transform: None,
        }
    }

    #[test]
    fn accepts_compact_and_legacy_raytracing_vertex_layouts() {
        let features = WgpuFeatures::empty();
        let compact = triangle_mesh();
        assert_eq!(
            raytracing_index_format(&compact, features),
            Some(IndexFormat::Uint32)
        );
        assert_eq!(BlasVertexLayout::new(&compact), vertex_layout(32, false));

        let legacy = triangle_mesh()
            .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[1.0, 0.0, 0.0, 1.0]; 3]);
        assert_eq!(
            raytracing_index_format(&legacy, features),
            Some(IndexFormat::Uint32)
        );
        assert_eq!(BlasVertexLayout::new(&legacy), vertex_layout(48, true));

        let indexed_16 = triangle_mesh().with_inserted_indices(Indices::U16(vec![0, 1, 2]));
        assert_eq!(
            raytracing_index_format(&indexed_16, features),
            Some(IndexFormat::Uint16)
        );
    }

    #[test]
    fn compaction_system_is_exclusive() {
        let mut world = World::new();
        let mut system = IntoSystem::into_system(compact_raytracing_blas);
        assert!(system.initialize(&mut world).is_exclusive());
    }

    fn opaque_key() -> BlasKey {
        BlasKey {
            mesh: AssetId::default(),
            opacity: BlasOpacity::Opaque,
        }
    }

    #[test]
    fn missing_allocator_slices_leave_build_queued_for_retry() {
        let key = opaque_key();
        let mut manager = BlasManager::default();
        manager
            .vertex_layouts
            .insert(key.mesh, vertex_layout(32, false));
        manager.build_queue.push_back(QueuedBuild {
            key,
            vertex_layout: vertex_layout(32, false),
            index_format: IndexFormat::Uint32,
            allocator_deferrals: 0,
        });

        manager.defer_build_waiting_for_allocator();

        assert_eq!(
            manager.build_queue.front(),
            Some(&QueuedBuild {
                key,
                vertex_layout: vertex_layout(32, false),
                index_format: IndexFormat::Uint32,
                allocator_deferrals: 1,
            })
        );
        assert_eq!(
            manager.vertex_layouts.get(&key.mesh),
            Some(&vertex_layout(32, false))
        );
        assert_eq!(manager.allocator_waiting, 1);
    }

    #[test]
    fn permanently_unallocatable_builds_are_dropped_so_the_queue_drains() {
        let mut manager = BlasManager::default();
        manager.build_queue.push_back(QueuedBuild {
            key: opaque_key(),
            vertex_layout: vertex_layout(32, false),
            index_format: IndexFormat::Uint32,
            allocator_deferrals: MAX_ALLOCATOR_DEFERRALS - 1,
        });

        manager.defer_build_waiting_for_allocator();

        assert!(manager.build_queue.is_empty());

        let status = RaytracingSceneStatus::default();
        status.publish(&manager);
        assert!(status.snapshot().is_settled_for(0));
    }

    #[test]
    fn empty_meshes_are_never_queued_for_a_blas() {
        let empty = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new())
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new())
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, Vec::<[f32; 2]>::new())
            .with_inserted_indices(Indices::U32(Vec::new()));
        assert_eq!(raytracing_index_format(&empty, WgpuFeatures::empty()), None);
    }

    #[test]
    fn failed_compaction_completion_is_observable_and_does_not_remain_pending() {
        let key = opaque_key();
        let state = Arc::new(AtomicU8::new(COMPACTION_PENDING));
        record_compaction_result(&state, false);
        assert_eq!(state.load(Ordering::Acquire), COMPACTION_FAILED);

        let mut manager = BlasManager::default();
        manager.compaction_queue.push_back(PendingCompaction {
            key,
            vertex_count: 3,
            state,
        });
        manager.disable_compaction(key);

        assert!(manager.compaction_disabled);
        assert!(manager.compaction_queue.is_empty());
        assert!(manager.failed_compactions.contains(&key));
    }

    #[test]
    fn shared_status_reports_settled_uncompacted_fallback() {
        let mut manager = BlasManager::default();
        manager.compaction_disabled = true;
        manager.failed_compactions.insert(opaque_key());

        let status = RaytracingSceneStatus::default();
        status.publish(&manager);
        let snapshot = status.snapshot();

        assert_eq!(snapshot.failed_compactions, 1);
        assert!(snapshot.compaction_disabled);
        assert!(snapshot.is_settled_for(0));
        assert!(!snapshot.is_settled_for(1));
    }

    #[test]
    fn retired_blas_wait_out_both_tlas_structures_before_deletion() {
        let mesh = AssetId::default();
        let mut manager = BlasManager::default();

        // A remove with no BLAS still marks the mesh changed but retires nothing.
        manager.remove_mesh(mesh);
        assert_eq!(manager.changed_meshes(), &[mesh]);
        assert!(manager.pending_deletions.is_empty());

        // With nothing retired, TLAS builds must not accumulate empty batches.
        manager.note_tlas_build();
        assert!(manager.pending_deletions.is_empty());
    }
}
