use alloc::collections::VecDeque;
use alloc::sync::Arc;
use bevy_asset::AssetId;
use bevy_ecs::{
    resource::Resource,
    system::{Res, ResMut},
    world::World,
};
use bevy_mesh::{Indices, Mesh};
use bevy_platform::collections::{HashMap, HashSet};
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
use core::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use tracing::warn;

/// Limits BLAS build and compaction submissions without unnecessarily throttling high-end GPUs.
///
/// Building every mesh extracted in one frame has a very large transient cost: all un-compacted
/// BLAS allocations coexist until asynchronous compaction catches up. Large scenes can exhaust
/// VRAM even when their final compacted acceleration structures would fit.
const MAX_BUILD_VERTICES_PER_FRAME: u32 = 2_000_000;
const MAX_COMPACTION_VERTICES_PER_FRAME: u32 = 2_000_000;
const MAX_UNCOMPACTED_VERTICES: u32 = 8_000_000;

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
    mesh: AssetId<Mesh>,
    vertex_count: u32,
    state: Arc<AtomicU8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QueuedBuild {
    mesh: AssetId<Mesh>,
    vertex_stride: u32,
    /// Frames this entry was rotated because the mesh allocator had no slices for it.
    allocator_deferrals: u32,
}

#[derive(Resource, Default)]
pub struct BlasManager {
    blas: HashMap<AssetId<Mesh>, Blas>,
    vertex_strides: HashMap<AssetId<Mesh>, u32>,
    build_queue: VecDeque<QueuedBuild>,
    compaction_queue: VecDeque<PendingCompaction>,
    compacted: HashSet<AssetId<Mesh>>,
    failed_compactions: HashSet<AssetId<Mesh>>,
    compaction_disabled: bool,
    allocator_waiting: usize,
    changed: Vec<AssetId<Mesh>>,
    /// BLAS that are pending deletion, one batch per TLAS build. The back batch collects
    /// retirements since the last build, and every batch ahead of it has one more build to wait
    /// out.
    pending_deletions: VecDeque<Vec<Blas>>,
}

impl BlasManager {
    pub fn get(&self, mesh: &AssetId<Mesh>) -> Option<&Blas> {
        self.blas.get(mesh)
    }

    pub fn device_address(&self, mesh: &AssetId<Mesh>) -> Option<u64> {
        self.blas.get(mesh)?.handle()
    }

    pub fn vertex_stride(&self, mesh: &AssetId<Mesh>) -> Option<u32> {
        self.vertex_strides.get(mesh).copied()
    }

    pub fn changed_meshes(&self) -> &[AssetId<Mesh>] {
        &self.changed
    }

    pub fn note_tlas_build(&mut self) {
        if !self.pending_deletions.is_empty() {
            self.pending_deletions.push_back(Vec::new());
        }
    }

    fn insert(&mut self, mesh: AssetId<Mesh>, blas: Blas) {
        if let Some(old) = self.blas.insert(mesh, blas) {
            self.retire(old);
        }

        self.changed.push(mesh);
    }

    fn remove(&mut self, mesh: AssetId<Mesh>) {
        self.changed.push(mesh);

        if let Some(removed) = self.blas.remove(&mesh) {
            self.retire(removed);
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
                mesh = %build.mesh,
                "mesh allocator never produced slices for this mesh; dropping its BLAS build so the raytracing scene can settle"
            );
            return;
        }
        self.build_queue.push_back(build);
        self.allocator_waiting += 1;
    }

    fn disable_compaction(&mut self, failed_mesh: AssetId<Mesh>) {
        self.failed_compactions.insert(failed_mesh);
        self.compaction_disabled = true;
        self.compaction_queue.clear();
    }
}

/// A main-world readable snapshot of Solari's asynchronous BLAS preparation state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RaytracingSceneStatusSnapshot {
    /// Number of compatible mesh assets that currently have a BLAS available for TLAS binding.
    pub available_blas: usize,
    /// Number of mesh assets still waiting to be built.
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
            .store(manager.blas.len(), Ordering::Release);
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
        blas_manager.remove(*asset_id);
        blas_manager.vertex_strides.remove(asset_id);
        blas_manager
            .build_queue
            .retain(|build| build.mesh != *asset_id);
        blas_manager
            .compaction_queue
            .retain(|pending| pending.mesh != *asset_id);
        blas_manager.compacted.remove(asset_id);
        blas_manager.failed_compactions.remove(asset_id);
    }

    // Retain newly extracted meshes in a persistent queue. The mesh allocator keeps their GPU
    // input slices alive, so later frames can build them without retaining ExtractedAssets.
    for (asset_id, mesh) in &extracted_meshes.extracted {
        if let Some(vertex_stride) = raytracing_vertex_stride(mesh)
            && !blas_manager.vertex_strides.contains_key(asset_id)
            && !blas_manager.blas.contains_key(asset_id)
        {
            blas_manager.vertex_strides.insert(*asset_id, vertex_stride);
            blas_manager.build_queue.push_back(QueuedBuild {
                mesh: *asset_id,
                vertex_stride,
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
    // deadlock the queue.
    let mut blas_resources = Vec::new();
    let mut vertices_built = 0u32;
    let mut builds_to_check = blas_manager.build_queue.len();
    while builds_to_check != 0 {
        builds_to_check -= 1;
        let Some(QueuedBuild {
            mesh: asset_id,
            vertex_stride,
            ..
        }) = blas_manager.build_queue.front().copied()
        else {
            break;
        };
        let (Some(vertex_slice), Some(index_slice)) = (
            mesh_allocator.mesh_vertex_slice(&asset_id),
            mesh_allocator.mesh_index_slice(&asset_id),
        ) else {
            // Mesh allocation normally precedes this system, but progressive loading and render
            // recovery can make the asset and its allocator slices arrive on different frames.
            // Keep the request alive and rotate it so one delayed mesh cannot block ready work.
            blas_manager.defer_build_waiting_for_allocator();
            continue;
        };
        let vertex_count = vertex_slice.range.len() as u32;
        if !blas_resources.is_empty()
            && (vertices_built.saturating_add(vertex_count) > MAX_BUILD_VERTICES_PER_FRAME
                || (!blas_manager.compaction_disabled
                    && uncompacted_vertices.saturating_add(vertex_count)
                        > MAX_UNCOMPACTED_VERTICES))
        {
            break;
        }
        if !blas_manager.compaction_disabled
            && blas_resources.is_empty()
            && uncompacted_vertices != 0
            && uncompacted_vertices.saturating_add(vertex_count) > MAX_UNCOMPACTED_VERTICES
        {
            break;
        }
        blas_manager.build_queue.pop_front();

        let (blas, blas_size) = allocate_blas(
            &vertex_slice,
            &index_slice,
            &asset_id,
            vertex_stride,
            &render_device,
        );

        blas_manager.insert(asset_id, blas);
        if !blas_manager.compaction_disabled {
            blas_manager.compaction_queue.push_back(PendingCompaction {
                mesh: asset_id,
                vertex_count: blas_size.vertex_count,
                state: Arc::new(AtomicU8::new(COMPACTION_NOT_STARTED)),
            });
        }
        vertices_built = vertices_built.saturating_add(blas_size.vertex_count);
        if !blas_manager.compaction_disabled {
            uncompacted_vertices = uncompacted_vertices.saturating_add(blas_size.vertex_count);
        }
        blas_resources.push((
            asset_id,
            vertex_slice,
            index_slice,
            blas_size,
            vertex_stride,
        ));
    }

    if blas_resources.is_empty() {
        return;
    }

    // Build geometry into each BLAS
    let build_entries = blas_resources
        .iter()
        .map(
            |(asset_id, vertex_slice, index_slice, blas_size, vertex_stride)| {
                let geometry = BlasTriangleGeometry {
                    size: blas_size,
                    vertex_buffer: vertex_slice.buffer,
                    first_vertex: vertex_slice.range.start,
                    vertex_stride: *vertex_stride as u64,
                    index_buffer: Some(index_slice.buffer),
                    first_index: Some(index_slice.range.start),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                };
                BlasBuildEntry {
                    blas: &blas_manager.blas[asset_id],
                    geometry: BlasGeometries::TriangleGeometries(vec![geometry]),
                }
            },
        )
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
        let mesh = pending.mesh;
        let vertex_count = pending.vertex_count;
        let Some(blas) = blas_manager.get(&mesh) else {
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
                    blas_manager.insert(mesh, compacted_blas);
                    blas_manager.compacted.insert(mesh);
                    vertices_compacted += vertex_count;
                }
                None => {
                    warn!(
                        %mesh,
                        "BLAS compaction failed; retaining valid un-compacted BLAS and disabling further compaction"
                    );
                    blas_manager.disable_compaction(mesh);
                    break;
                }
            },
            COMPACTION_FAILED => {
                warn!(
                    %mesh,
                    "asynchronous BLAS compaction preparation failed; retaining valid un-compacted BLAS and disabling further compaction"
                );
                blas_manager.disable_compaction(mesh);
                break;
            }
            state => {
                warn!(%mesh, state, "BLAS entered an unknown compaction state");
                blas_manager.disable_compaction(mesh);
                break;
            }
        }
    }
}

pub fn delete_raytracing_blas(
    mut blas_manager: ResMut<BlasManager>,
    render_queue: Res<RenderQueue>,
) {
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
    asset_id: &AssetId<Mesh>,
    _vertex_stride: u32,
    render_device: &RenderDevice,
) -> (Blas, BlasTriangleGeometrySizeDescriptor) {
    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: Mesh::ATTRIBUTE_POSITION.format,
        vertex_count: vertex_slice.range.len() as u32,
        index_format: Some(IndexFormat::Uint32),
        index_count: Some(index_slice.range.len() as u32),
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };

    // TODO: If we ever introduce BLAS refits, we need to be aware of the TLAS double-buffer
    // to avoid invalidating the previous frame TLAS
    let blas = render_device.wgpu_device().create_blas(
        &CreateBlasDescriptor {
            label: Some(&asset_id.to_string()),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE
                | AccelerationStructureFlags::ALLOW_COMPACTION,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );

    (blas, blas_size)
}

fn raytracing_vertex_stride(mesh: &Mesh) -> Option<u32> {
    let triangle_list = mesh.primitive_topology() == PrimitiveTopology::TriangleList;
    let vertex_attributes = mesh
        .attributes()
        .map(|(attribute, _)| (attribute.id, attribute.format))
        .collect::<Vec<_>>();
    let compact = [
        (Mesh::ATTRIBUTE_POSITION.id, Mesh::ATTRIBUTE_POSITION.format),
        (Mesh::ATTRIBUTE_NORMAL.id, Mesh::ATTRIBUTE_NORMAL.format),
        (Mesh::ATTRIBUTE_UV_0.id, Mesh::ATTRIBUTE_UV_0.format),
    ];
    let legacy = [
        (Mesh::ATTRIBUTE_POSITION.id, Mesh::ATTRIBUTE_POSITION.format),
        (Mesh::ATTRIBUTE_NORMAL.id, Mesh::ATTRIBUTE_NORMAL.format),
        (Mesh::ATTRIBUTE_UV_0.id, Mesh::ATTRIBUTE_UV_0.format),
        (Mesh::ATTRIBUTE_TANGENT.id, Mesh::ATTRIBUTE_TANGENT.format),
    ];
    // An empty mesh can never be allocated a vertex buffer slice, so it would rotate in the build
    // queue forever and keep the scene from ever settling.
    let indexed_32 = matches!(mesh.indices(), Some(Indices::U32(indices)) if !indices.is_empty());
    if mesh.enable_raytracing
        && triangle_list
        && indexed_32
        && mesh.count_vertices() != 0
        && (vertex_attributes.as_slice() == compact || vertex_attributes.as_slice() == legacy)
    {
        Some(
            vertex_attributes
                .iter()
                .map(|(_, format)| format.size() as u32)
                .sum(),
        )
    } else {
        None
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

    #[test]
    fn accepts_compact_and_legacy_raytracing_vertex_layouts() {
        let compact = triangle_mesh();
        assert_eq!(raytracing_vertex_stride(&compact), Some(32));

        let legacy = triangle_mesh()
            .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[1.0, 0.0, 0.0, 1.0]; 3]);
        assert_eq!(raytracing_vertex_stride(&legacy), Some(48));
    }

    #[test]
    fn compaction_system_is_exclusive() {
        let mut world = World::new();
        let mut system = IntoSystem::into_system(compact_raytracing_blas);
        assert!(system.initialize(&mut world).is_exclusive());
    }

    #[test]
    fn missing_allocator_slices_leave_build_queued_for_retry() {
        let mesh = AssetId::default();
        let mut manager = BlasManager::default();
        manager.vertex_strides.insert(mesh, 32);
        manager.build_queue.push_back(QueuedBuild {
            mesh,
            vertex_stride: 32,
            allocator_deferrals: 0,
        });

        manager.defer_build_waiting_for_allocator();

        assert_eq!(
            manager.build_queue.front(),
            Some(&QueuedBuild {
                mesh,
                vertex_stride: 32,
                allocator_deferrals: 1,
            })
        );
        assert_eq!(manager.vertex_strides.get(&mesh), Some(&32));
        assert_eq!(manager.allocator_waiting, 1);
    }

    #[test]
    fn permanently_unallocatable_builds_are_dropped_so_the_queue_drains() {
        let mesh = AssetId::default();
        let mut manager = BlasManager::default();
        manager.build_queue.push_back(QueuedBuild {
            mesh,
            vertex_stride: 32,
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
        assert_eq!(raytracing_vertex_stride(&empty), None);
    }

    #[test]
    fn failed_compaction_completion_is_observable_and_does_not_remain_pending() {
        let mesh = AssetId::default();
        let state = Arc::new(AtomicU8::new(COMPACTION_PENDING));
        record_compaction_result(&state, false);
        assert_eq!(state.load(Ordering::Acquire), COMPACTION_FAILED);

        let mut manager = BlasManager::default();
        manager.compaction_queue.push_back(PendingCompaction {
            mesh,
            vertex_count: 3,
            state,
        });
        manager.disable_compaction(mesh);

        assert!(manager.compaction_disabled);
        assert!(manager.compaction_queue.is_empty());
        assert!(manager.failed_compactions.contains(&mesh));
    }

    #[test]
    fn shared_status_reports_settled_uncompacted_fallback() {
        let mesh = AssetId::default();
        let mut manager = BlasManager::default();
        manager.compaction_disabled = true;
        manager.failed_compactions.insert(mesh);

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
        manager.remove(mesh);
        assert_eq!(manager.changed_meshes(), &[mesh]);
        assert!(manager.pending_deletions.is_empty());

        // With nothing retired, TLAS builds must not accumulate empty batches.
        manager.note_tlas_build();
        assert!(manager.pending_deletions.is_empty());
    }
}
