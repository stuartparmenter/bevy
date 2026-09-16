use super::{
    RaytracingGeometry, RaytracingGeometryBuffers, RaytracingGeometryUpdateMode,
    RaytracingProducerEncoder,
};
use alloc::collections::VecDeque;
use bevy_asset::AssetId;
use bevy_ecs::{
    entity::{Entity, EntityHashMap},
    query::With,
    resource::Resource,
    system::{Query, Res, ResMut},
};
use bevy_mesh::{Indices, Mesh};
use bevy_platform::collections::HashMap;
use bevy_render::{
    diagnostic::{DiagnosticsRecorder, RecordDiagnostics},
    mesh::{
        allocator::{MeshAllocator, MeshBufferSlice},
        RenderMesh,
    },
    render_asset::ExtractedAssets,
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
};

/// After compacting this many vertices worth of meshes per frame, no further BLAS will be compacted.
/// Lower this number to distribute the work across more frames.
const MAX_COMPACTION_VERTICES_PER_FRAME: u32 = 400_000;

/// Under the `wgpu_hal` build path, we need to manage BLAS lifetimes ourselves.
/// Since solari keeps both current and previous frame TLAS's around, only after
/// two TLAS builds since we marked it for deletion is it safe to delete a BLAS.
const TLAS_BUILDS_BEFORE_DELETION_ALLOWED: usize = 2;

#[derive(Resource, Default)]
pub struct BlasManager {
    blas: HashMap<AssetId<Mesh>, Blas>,
    compaction_queue: VecDeque<(AssetId<Mesh>, u32, bool)>,
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

    fn retire(&mut self, blas: Blas) {
        match self.pending_deletions.back_mut() {
            Some(batch) => batch.push(blas),
            None => self.pending_deletions.push_back(vec![blas]),
        }
    }
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

    // Delete BLAS for deleted or modified meshes
    for asset_id in extracted_meshes
        .removed
        .iter()
        .chain(extracted_meshes.modified.iter())
    {
        blas_manager.remove(*asset_id);
    }

    if extracted_meshes.extracted.is_empty() {
        return;
    }

    // Create new BLAS for added or changed meshes
    let blas_resources = extracted_meshes
        .extracted
        .iter()
        .filter(|(_, mesh)| is_mesh_raytracing_compatible(mesh))
        .map(|(asset_id, _)| {
            let vertex_slice = mesh_allocator.mesh_vertex_slice(asset_id).unwrap();
            let index_slice = mesh_allocator.mesh_index_slice(asset_id).unwrap();

            let (blas, blas_size) =
                allocate_blas(&vertex_slice, &index_slice, asset_id, &render_device);

            blas_manager.insert(*asset_id, blas);
            blas_manager
                .compaction_queue
                .push_back((*asset_id, blas_size.vertex_count, false));

            (*asset_id, vertex_slice, index_slice, blas_size)
        })
        .collect::<Vec<_>>();

    // Build geometry into each BLAS
    let build_entries = blas_resources
        .iter()
        .map(|(asset_id, vertex_slice, index_slice, blas_size)| {
            let geometry = BlasTriangleGeometry {
                size: blas_size,
                vertex_buffer: vertex_slice.buffer,
                first_vertex: vertex_slice.range.start,
                vertex_stride: 48,
                index_buffer: Some(index_slice.buffer),
                first_index: Some(index_slice.range.start),
                transform_buffer: None,
                transform_buffer_offset: None,
            };
            BlasBuildEntry {
                blas: &blas_manager.blas[asset_id],
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

pub fn compact_raytracing_blas(
    mut blas_manager: ResMut<BlasManager>,
    render_queue: Res<RenderQueue>,
) {
    let queue_size = blas_manager.compaction_queue.len();
    let mut meshes_processed = 0;
    let mut vertices_compacted = 0;

    while !blas_manager.compaction_queue.is_empty()
        && vertices_compacted < MAX_COMPACTION_VERTICES_PER_FRAME
        && meshes_processed < queue_size
    {
        meshes_processed += 1;

        let (mesh, vertex_count, compaction_started) =
            blas_manager.compaction_queue.pop_front().unwrap();

        let Some(blas) = blas_manager.get(&mesh) else {
            continue;
        };

        if !compaction_started {
            blas.prepare_compaction_async(|_| {});
        }

        if blas.ready_for_compaction() {
            let compacted_blas = render_queue.compact_blas(blas);
            blas_manager.insert(mesh, compacted_blas);

            vertices_compacted += vertex_count;
            continue;
        }

        // BLAS not ready for compaction, put back in queue
        blas_manager
            .compaction_queue
            .push_back((mesh, vertex_count, true));
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

fn allocate_blas(
    vertex_slice: &MeshBufferSlice,
    index_slice: &MeshBufferSlice,
    asset_id: &AssetId<Mesh>,
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

/// BLASes for [`RaytracingGeometry`], keyed by render entity.
///
/// Retired BLASes use [`BlasManager`]'s deferred deletion because the raw TLAS
/// build path does not retain references to them.
#[derive(Resource, Default)]
pub struct GeometryBlasManager {
    blas: EntityHashMap<GeometryBlasEntry>,
    /// Entities whose instance slots need to reference a different BLAS.
    changed: Vec<Entity>,
}

struct GeometryBlasEntry {
    blas: Blas,
    /// The previous frame's BLAS, used only for `RebuildEveryFrame`.
    /// Swap before building to preserve the BLAS referenced by the previous TLAS.
    /// Rebuilding it in place would fail wgpu validation (`BlasNewerThenTlas`).
    previous_blas: Option<Blas>,
    size: BlasTriangleGeometrySizeDescriptor,
    /// Buffer IDs used to detect replacements, even when the counts are unchanged.
    buffer_ids: (BufferId, BufferId),
    /// The mode used to allocate the BLAS. A mode change requires reallocation
    /// because the build flags differ and `RebuildEveryFrame` needs two BLASes.
    update_mode: RaytracingGeometryUpdateMode,
    /// Whether this BLAS still needs to be built.
    /// Keep it pending if the producer buffers are unavailable.
    pending_build: bool,
}

impl GeometryBlasManager {
    pub fn get(&self, entity: &Entity) -> Option<&Blas> {
        self.blas.get(entity).map(|entry| &entry.blas)
    }

    pub fn device_address(&self, entity: &Entity) -> Option<u64> {
        self.get(entity)?.handle()
    }

    pub fn changed_entities(&self) -> &[Entity] {
        &self.changed
    }
}

impl GeometryBlasEntry {
    /// Defers deletion of both BLASes until the retained TLASes no longer use them.
    fn retire(self, blas_manager: &mut BlasManager) {
        blas_manager.retire(self.blas);
        if let Some(previous_blas) = self.previous_blas {
            blas_manager.retire(previous_blas);
        }
    }
}

/// Allocates BLASes from producer buffers before the scene binder references them.
/// Marks new or changed geometry for building after the producers run.
pub fn prepare_raytracing_geometry_blas(
    mut geometry_blas_manager: ResMut<GeometryBlasManager>,
    mut blas_manager: ResMut<BlasManager>,
    geometry: Query<(Entity, &RaytracingGeometryBuffers), With<RaytracingGeometry>>,
    render_device: Res<RenderDevice>,
) {
    let GeometryBlasManager {
        blas: entries,
        changed,
    } = &mut *geometry_blas_manager;
    changed.clear();

    // Drop BLASes for entities that despawned or lost the component.
    let stale: Vec<Entity> = entries
        .keys()
        .filter(|entity| !geometry.contains(**entity))
        .copied()
        .collect();
    for entity in stale {
        if let Some(entry) = entries.remove(&entity) {
            entry.retire(&mut blas_manager);
            changed.push(entity);
        }
    }

    for (entity, buffers) in &geometry {
        if buffers.vertex_count == 0 || buffers.index_count == 0 {
            // Drop the entry so the stale BLAS stops occluding rays.
            if let Some(entry) = entries.remove(&entity) {
                entry.retire(&mut blas_manager);
                changed.push(entity);
            }
            continue;
        }

        let buffer_ids = (buffers.vertex_buffer.id(), buffers.index_buffer.id());
        let rebuild_every_frame = matches!(
            buffers.update_mode,
            RaytracingGeometryUpdateMode::RebuildEveryFrame
        );
        match entries.get_mut(&entity) {
            // Reuse unchanged geometry. For per-frame builds, swap BLASes to preserve
            // the geometry referenced by the previous TLAS.
            Some(entry)
                if entry.size.vertex_count == buffers.vertex_count
                    && entry.size.index_count == Some(buffers.index_count)
                    && entry.buffer_ids == buffer_ids
                    && entry.update_mode == buffers.update_mode =>
            {
                // A pending build already swapped BLASes. Swapping again would build
                // into the BLAS referenced by the previous TLAS.
                if rebuild_every_frame && !entry.pending_build {
                    if let Some(previous_blas) = &mut entry.previous_blas {
                        core::mem::swap(&mut entry.blas, previous_blas);
                        changed.push(entity);
                    }
                    entry.pending_build = true;
                }
            }
            // Allocate for changed geometry and defer deletion of the old BLASes
            // until the retained TLASes no longer use them.
            _ => {
                let (blas, size) = allocate_geometry_blas(buffers, &render_device);
                let previous_blas =
                    rebuild_every_frame.then(|| allocate_geometry_blas(buffers, &render_device).0);
                let old = entries.insert(
                    entity,
                    GeometryBlasEntry {
                        blas,
                        previous_blas,
                        size,
                        buffer_ids,
                        update_mode: buffers.update_mode,
                        pending_build: true,
                    },
                );
                if let Some(old) = old {
                    old.retire(&mut blas_manager);
                }
                changed.push(entity);
            }
        }
    }
}

/// Records pending BLAS builds in the producer encoder after the fill passes.
/// [`submit_raytracing_producers`](super::producer::submit_raytracing_producers)
/// submits the encoder before the render graph builds the TLAS.
pub fn build_raytracing_geometry_blas(
    mut geometry_blas_manager: ResMut<GeometryBlasManager>,
    geometry: Query<(Entity, &RaytracingGeometryBuffers), With<RaytracingGeometry>>,
    render_device: Res<RenderDevice>,
    mut producer_encoder: ResMut<RaytracingProducerEncoder>,
    mut diagnostics: Option<ResMut<DiagnosticsRecorder>>,
) {
    let mut built = Vec::new();
    let build_entries: Vec<BlasBuildEntry> = geometry
        .iter()
        .filter_map(|(entity, buffers)| {
            let entry = geometry_blas_manager.blas.get(&entity)?;
            if !entry.pending_build {
                return None;
            }
            // Producers may replace or resize buffers after prepare. Skip mismatched
            // builds to avoid invalidating the shared command buffer. Leave the entry
            // pending so the next prepare reallocates it.
            if entry.buffer_ids != (buffers.vertex_buffer.id(), buffers.index_buffer.id())
                || entry.size.vertex_count != buffers.vertex_count
                || entry.size.index_count != Some(buffers.index_count)
            {
                return None;
            }
            built.push(entity);
            Some(BlasBuildEntry {
                blas: &entry.blas,
                geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                    size: &entry.size,
                    vertex_buffer: &buffers.vertex_buffer,
                    first_vertex: 0,
                    vertex_stride: RaytracingGeometryBuffers::VERTEX_STRIDE,
                    index_buffer: Some(&buffers.index_buffer),
                    first_index: Some(0),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            })
        })
        .collect();
    if build_entries.is_empty() {
        return;
    }

    let command_encoder = producer_encoder.encoder(&render_device);
    let time_span = diagnostics
        .as_mut()
        .map(|diagnostics| diagnostics.time_span(command_encoder, "geometry_blas_build"));
    command_encoder.build_acceleration_structures(&build_entries, &[]);
    if let Some(time_span) = time_span {
        time_span.end(command_encoder);
    }

    drop(build_entries);
    for entity in built {
        if let Some(entry) = geometry_blas_manager.blas.get_mut(&entity) {
            entry.pending_build = false;
        }
    }
}

fn allocate_geometry_blas(
    buffers: &RaytracingGeometryBuffers,
    render_device: &RenderDevice,
) -> (Blas, BlasTriangleGeometrySizeDescriptor) {
    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: Mesh::ATTRIBUTE_POSITION.format,
        vertex_count: buffers.vertex_count,
        index_format: Some(IndexFormat::Uint32),
        index_count: Some(buffers.index_count),
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };

    // Skip compaction for these few, large BLASes.
    let build_flag = match buffers.update_mode {
        RaytracingGeometryUpdateMode::BuildOnce => AccelerationStructureFlags::PREFER_FAST_TRACE,
        RaytracingGeometryUpdateMode::RebuildEveryFrame => {
            AccelerationStructureFlags::PREFER_FAST_BUILD
        }
    };

    let blas = render_device.wgpu_device().create_blas(
        &CreateBlasDescriptor {
            label: Some("raytracing_geometry_blas"),
            flags: build_flag,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );

    (blas, blas_size)
}
fn is_mesh_raytracing_compatible(mesh: &Mesh) -> bool {
    let triangle_list = mesh.primitive_topology() == PrimitiveTopology::TriangleList;
    let vertex_attributes = mesh
        .attributes()
        .map(|(attribute, _)| (attribute.id, attribute.format))
        .eq([
            (Mesh::ATTRIBUTE_POSITION.id, Mesh::ATTRIBUTE_POSITION.format),
            (Mesh::ATTRIBUTE_NORMAL.id, Mesh::ATTRIBUTE_NORMAL.format),
            (Mesh::ATTRIBUTE_UV_0.id, Mesh::ATTRIBUTE_UV_0.format),
            (Mesh::ATTRIBUTE_TANGENT.id, Mesh::ATTRIBUTE_TANGENT.format),
        ]);
    let indexed_32 = matches!(mesh.indices(), Some(Indices::U32(..)));
    mesh.enable_raytracing && triangle_list && vertex_attributes && indexed_32
}
