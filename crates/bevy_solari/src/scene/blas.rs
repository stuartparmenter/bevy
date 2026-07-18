use super::{RaytracingGeometry, RaytracingGeometryBuffers, RaytracingGeometryUpdateMode};
use alloc::collections::VecDeque;
use bevy_asset::AssetId;
use bevy_ecs::{
    entity::Entity,
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

#[derive(Resource, Default)]
pub struct BlasManager {
    blas: HashMap<AssetId<Mesh>, Blas>,
    compaction_queue: VecDeque<(AssetId<Mesh>, u32, bool)>,
}

impl BlasManager {
    pub fn get(&self, mesh: &AssetId<Mesh>) -> Option<&Blas> {
        self.blas.get(mesh)
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
    // Delete BLAS for deleted or modified meshes
    for asset_id in extracted_meshes
        .removed
        .iter()
        .chain(extracted_meshes.modified.iter())
    {
        blas_manager.blas.remove(asset_id);
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

            blas_manager.blas.insert(*asset_id, blas);
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
            blas_manager.blas.insert(mesh, compacted_blas);

            vertices_compacted += vertex_count;
            continue;
        }

        // BLAS not ready for compaction, put back in queue
        blas_manager
            .compaction_queue
            .push_back((mesh, vertex_count, true));
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

/// BLASes for GPU-authored [`RaytracingGeometry`] entities, keyed by their
/// render-world entity (there is no `Mesh` asset id to key on).
#[derive(Resource, Default)]
pub struct GeometryBlasManager {
    blas: HashMap<Entity, GeometryBlasEntry>,
}

struct GeometryBlasEntry {
    blas: Blas,
    /// The BLAS built last frame (`RebuildEveryFrame` only; `None` for
    /// `BuildOnce`). Swapped with `blas` before each build so last frame's
    /// geometry stays intact for the retained previous-frame TLAS, which
    /// still references it — rebuilding in place would fail wgpu validation
    /// (`BlasNewerThenTlas`).
    previous_blas: Option<Blas>,
    size: BlasTriangleGeometrySizeDescriptor,
    /// Which buffers the BLAS was built from. Swapping in new buffers (even
    /// with the same counts) must trigger a fresh build.
    buffer_ids: (BufferId, BufferId),
    /// Whether the binder should build this BLAS this frame. Recomputed here
    /// every frame; the binder folds the builds into its TLAS build
    /// submission, saving a separate encoder + submit.
    pending_build: bool,
}

impl GeometryBlasManager {
    /// The BLAS plus, if it needs building this frame, its size descriptor.
    pub(crate) fn get_with_pending_build(
        &self,
        entity: &Entity,
    ) -> Option<(&Blas, Option<&BlasTriangleGeometrySizeDescriptor>)> {
        self.blas
            .get(entity)
            .map(|entry| (&entry.blas, entry.pending_build.then_some(&entry.size)))
    }
}

/// Maintains BLASes for [`RaytracingGeometry`] entities from the
/// producer-supplied [`RaytracingGeometryBuffers`]: allocates them, and marks
/// which ones the binder must build this frame. Producers must submit their
/// buffer-filling compute before the binder runs (earlier queue submissions
/// execute first on the GPU).
pub fn prepare_raytracing_geometry_blas(
    mut blas_manager: ResMut<GeometryBlasManager>,
    geometry: Query<(Entity, &RaytracingGeometryBuffers), With<RaytracingGeometry>>,
    render_device: Res<RenderDevice>,
) {
    // Drop BLASes for entities that despawned or lost the component.
    blas_manager
        .blas
        .retain(|entity, _| geometry.contains(*entity));

    for (entity, buffers) in &geometry {
        if buffers.vertex_count == 0 || buffers.index_count == 0 {
            // Drop the entry so the stale BLAS stops occluding rays.
            blas_manager.blas.remove(&entity);
            continue;
        }

        let buffer_ids = (buffers.vertex_buffer.id(), buffers.index_buffer.id());
        let rebuild_every_frame = matches!(
            buffers.update_mode,
            RaytracingGeometryUpdateMode::RebuildEveryFrame
        );
        match blas_manager.blas.get_mut(&entity) {
            // Existing BLAS with unchanged geometry: BuildOnce is done;
            // RebuildEveryFrame swaps so this frame's build lands in the
            // BLAS the retained previous-frame TLAS does not reference.
            Some(entry)
                if entry.size.vertex_count == buffers.vertex_count
                    && entry.size.index_count == Some(buffers.index_count)
                    && entry.buffer_ids == buffer_ids =>
            {
                entry.pending_build = rebuild_every_frame;
                if rebuild_every_frame && let Some(previous_blas) = &mut entry.previous_blas {
                    core::mem::swap(&mut entry.blas, previous_blas);
                }
            }
            // New, resized, or re-buffered geometry: allocate fresh.
            _ => {
                let (blas, size) = allocate_geometry_blas(buffers, &render_device);
                let previous_blas =
                    rebuild_every_frame.then(|| allocate_geometry_blas(buffers, &render_device).0);
                blas_manager.blas.insert(
                    entity,
                    GeometryBlasEntry {
                        blas,
                        previous_blas,
                        size,
                        buffer_ids,
                        pending_build: true,
                    },
                );
            }
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

    // Compaction is left off for both modes: these are few, large BLASes and
    // the bookkeeping isn't worth it.
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
