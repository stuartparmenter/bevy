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
use bevy_mesh::{
    DecodedPositions, Indices, Mesh, MeshAttributeCompressionFlags, MeshRaytracingFlags,
    MeshVertexAttribute, VertexFormat,
};
use bevy_platform::collections::HashMap;
use bevy_render::settings::WgpuFeatures;
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
use bevy_utils::once;
use tracing::warn;

/// After compacting this many vertices worth of meshes per frame, no further BLAS will be compacted.
/// Lower this number to distribute the work across more frames.
const MAX_COMPACTION_VERTICES_PER_FRAME: u32 = 400_000;

/// Under the `wgpu_hal` build path, we need to manage BLAS lifetimes ourselves.
/// Since solari keeps both current and previous frame TLAS's around, only after
/// two TLAS builds since we marked it for deletion is it safe to delete a BLAS.
const TLAS_BUILDS_BEFORE_DELETION_ALLOWED: usize = 2;

#[derive(Resource, Default)]
pub struct BlasManager {
    blas: HashMap<AssetId<Mesh>, MeshBlas>,
    compaction_queue: VecDeque<(BlasKey, u32, bool)>,
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

    fn require(&mut self, mesh: AssetId<Mesh>, flags: MeshRaytracingFlags) {
        self.blas.entry(mesh).or_default().requires = flags;
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
        if let Some(old) = slot.replace(blas) {
            self.retire(old);
        }

        self.changed.push(key.mesh);
    }

    fn remove_mesh(&mut self, mesh: AssetId<Mesh>) {
        self.changed.push(mesh);
        self.compaction_queue.retain(|(key, ..)| key.mesh != mesh);

        if let Some(removed) = self.blas.remove(&mesh) {
            for blas in removed.into_iter() {
                self.retire(blas);
            }
        }
    }

    fn retire(&mut self, blas: Blas) {
        match self.pending_deletions.back_mut() {
            Some(batch) => batch.push(blas),
            None => self.pending_deletions.push_back(vec![blas]),
        }
    }
}

struct MeshBlas {
    requires: MeshRaytracingFlags,
    opaque: Option<Blas>,
    non_opaque: Option<Blas>,
}

impl Default for MeshBlas {
    fn default() -> Self {
        Self {
            requires: MeshRaytracingFlags::empty(),
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

    // Delete BLAS for deleted or modified meshes
    for asset_id in extracted_meshes
        .removed
        .iter()
        .chain(extracted_meshes.modified.iter())
    {
        blas_manager.remove_mesh(*asset_id);
    }

    if extracted_meshes.extracted.is_empty() {
        return;
    }

    // Record which BLAS added or changed meshes declare, even if none
    let features = render_device.features();
    for (asset_id, mesh) in &extracted_meshes.extracted {
        if is_mesh_raytracing_compatible(mesh, features) {
            blas_manager.require(*asset_id, mesh.raytracing);
        }
    }

    // Create a BLAS for each opacity that added or changed meshes declare. Compressed positions
    // are relative to the mesh's AABB, so their geometry is built with a transform back to mesh
    // space.
    let mut transforms = Vec::<[f32; 12]>::new();
    let blas_inputs = extracted_meshes
        .extracted
        .iter()
        .filter(|(_, mesh)| is_mesh_raytracing_compatible(mesh, features))
        .flat_map(|(asset_id, mesh)| {
            BlasOpacity::declared(mesh.raytracing).map(move |opacity| {
                (
                    BlasKey {
                        mesh: *asset_id,
                        opacity,
                    },
                    mesh,
                )
            })
        })
        .map(|(key, mesh)| {
            let vertex_slice = mesh_allocator.mesh_vertex_slice(&key.mesh).unwrap();
            let index_slice = mesh_allocator.mesh_index_slice(&key.mesh).unwrap();
            let compressed_positions = match mesh.decoded_positions() {
                Some(DecodedPositions::Compressed {
                    center, half_size, ..
                }) => Some((center, half_size)),
                _ => None,
            };
            let transform_offset = compressed_positions.map(|(center, half_size)| {
                transforms.push([
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
                ]);
                ((transforms.len() - 1) * size_of::<[f32; 12]>()) as BufferAddress
            });
            let position_format = if compressed_positions.is_some() {
                VertexFormat::Snorm16x4
            } else {
                Mesh::ATTRIBUTE_POSITION.format
            };

            let (blas, size) = allocate_blas(
                &vertex_slice,
                &index_slice,
                position_format,
                key,
                &render_device,
            );

            blas_manager.insert(key, blas);
            blas_manager
                .compaction_queue
                .push_back((key, size.vertex_count, false));

            BlasInput {
                key,
                vertex_stride: mesh.get_vertex_size(),
                vertex_slice,
                index_slice,
                size,
                transform_offset,
            }
        })
        .collect::<Vec<_>>();

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

        let (key, vertex_count, compaction_started) =
            blas_manager.compaction_queue.pop_front().unwrap();

        let Some(blas) = blas_manager.get(&key) else {
            continue;
        };

        if !compaction_started {
            blas.prepare_compaction_async(|_| {});
        }

        if blas.ready_for_compaction() {
            let compacted_blas = render_queue.compact_blas(blas);
            blas_manager.insert(key, compacted_blas);

            vertices_compacted += vertex_count;
            continue;
        }

        // BLAS not ready for compaction, put back in queue
        blas_manager
            .compaction_queue
            .push_back((key, vertex_count, true));
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
    position_format: VertexFormat,
    key: BlasKey,
    render_device: &RenderDevice,
) -> (Blas, BlasTriangleGeometrySizeDescriptor) {
    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: position_format,
        vertex_count: vertex_slice.range.len() as u32,
        index_format: Some(IndexFormat::Uint32),
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

fn is_mesh_raytracing_compatible(mesh: &Mesh, features: WgpuFeatures) -> bool {
    const ATTRIBUTES: [MeshVertexAttribute; 4] = [
        Mesh::ATTRIBUTE_POSITION,
        Mesh::ATTRIBUTE_NORMAL,
        Mesh::ATTRIBUTE_UV_0,
        Mesh::ATTRIBUTE_TANGENT,
    ];
    let triangle_list = mesh.primitive_topology() == PrimitiveTopology::TriangleList;
    // Each attribute is either uncompressed or in the format its compression flag gives it.
    let compression = mesh.attribute_compression();
    let vertex_attributes = mesh
        .attributes()
        .map(|(attribute, _)| (attribute.id, attribute.format))
        .eq(ATTRIBUTES.map(|attribute| {
            let format = match MeshAttributeCompressionFlags::for_attribute(attribute.id) {
                Some((flag, format)) if compression.contains(flag) => format,
                _ => attribute.format,
            };
            (attribute.id, format)
        }));
    let indexed_32 = matches!(mesh.indices(), Some(Indices::U32(..)));
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
    triangle_list && vertex_attributes && indexed_32 && position_format_supported
}
