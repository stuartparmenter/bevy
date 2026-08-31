use super::{asset::MeshletAabb, MeshletMesh};
use alloc::{
    collections::BTreeMap,
    format,
    string::{String, ToString},
    sync::Arc,
    vec,
    vec::Vec,
};
use bevy_asset::{AssetId, Assets};
use bevy_ecs::{
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_platform::collections::HashMap;
use bevy_render::{
    render_resource::{
        BindingResource, Buffer, BufferAddress, BufferBinding, BufferDescriptor, BufferUsages,
        ShaderType, StorageBuffer,
    },
    renderer::{RenderDevice, RenderQueue},
};
use bytemuck::{Pod, Zeroable};
use core::{mem::size_of_val, ops::Range};
use tracing::error;

/// Size of one independently allocated meshlet data page.
pub const MESHLET_PAGE_SIZE: BufferAddress = 64 * 1024 * 1024;
/// Maximum number of pages exposed to meshlet shaders.
pub const MESHLET_MAX_PAGES: usize = 128;
const UPLOAD_CHUNK_SIZE: usize = 4 * 1024 * 1024;
const SECTION_ALIGNMENT: usize = 16;

/// Per-instance address translation for data stored in the paged meshlet heap.
///
/// Every offset is a `u32` word offset within `page_id`. Meshlet and BVH offsets stored in the
/// serialized asset remain asset-local; shaders add the appropriate base from this descriptor.
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, ShaderType, PartialEq, Eq)]
#[repr(C)]
pub struct MeshletGpuDescriptor {
    pub page_id: u32,
    pub vertex_positions_base: u32,
    pub vertex_normals_base: u32,
    pub vertex_uvs_base: u32,
    pub indices_base: u32,
    pub bvh_nodes_base: u32,
    pub meshlets_base: u32,
    pub meshlet_cull_data_base: u32,
}

#[derive(Clone)]
struct MeshletAllocation {
    page_id: u32,
    page_generation: u64,
    range: Range<BufferAddress>,
    descriptor: MeshletGpuDescriptor,
    asset_index: u32,
    bvh_depth: u32,
}

struct PendingUpload {
    asset_id: AssetId<MeshletMesh>,
    page_id: u32,
    page_generation: u64,
    offset: BufferAddress,
    bytes: Arc<[u8]>,
}

#[derive(Clone, Debug)]
struct PageAllocator {
    free: BTreeMap<BufferAddress, BufferAddress>,
    allocations: usize,
}

impl PageAllocator {
    fn new() -> Self {
        Self {
            free: BTreeMap::from([(0, MESHLET_PAGE_SIZE)]),
            allocations: 0,
        }
    }

    fn best_fit(&self, size: BufferAddress) -> Option<(BufferAddress, BufferAddress)> {
        if size == 0 {
            return None;
        }
        self.free
            .iter()
            .filter(|(_, length)| **length >= size)
            .min_by_key(|(start, length)| (**length, **start))
            .map(|(start, length)| (*start, *length))
    }

    fn allocate(&mut self, size: BufferAddress) -> Option<Range<BufferAddress>> {
        if size == 0 {
            return None;
        }
        let (start, length) = self.best_fit(size)?;
        self.free.remove(&start);
        if length != size {
            self.free.insert(start + size, length - size);
        }
        self.allocations += 1;
        Some(start..start + size)
    }

    fn free(&mut self, range: Range<BufferAddress>) {
        debug_assert!(range.start < range.end);
        let mut start = range.start;
        let mut end = range.end;

        if let Some((&previous_start, &previous_length)) = self.free.range(..start).next_back()
            && previous_start + previous_length == start
        {
            start = previous_start;
            self.free.remove(&previous_start);
        }
        if let Some((&next_start, &next_length)) = self.free.range(end..).next()
            && next_start == end
        {
            end = next_start + next_length;
            self.free.remove(&next_start);
        }
        assert!(self.free.insert(start, end - start).is_none());
        self.allocations -= 1;
    }

    fn is_empty(&self) -> bool {
        self.allocations == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AllocationPlanError {
    Empty,
    Oversized,
    Exhausted,
}

/// Why an asset was not uploaded, and whether uploading it again could ever work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UploadFailure {
    /// The asset itself is unusable, so it is skipped without being revalidated every frame.
    Permanent,
    /// The heap had no room. Later frames retry, as removals free space.
    Transient,
}

fn plan_page<'a>(
    pages: impl IntoIterator<Item = Option<&'a PageAllocator>>,
    size: BufferAddress,
) -> Result<(usize, bool), AllocationPlanError> {
    if size == 0 {
        return Err(AllocationPlanError::Empty);
    }
    if size > MESHLET_PAGE_SIZE {
        return Err(AllocationPlanError::Oversized);
    }
    let mut first_vacant = None;
    let existing = pages
        .into_iter()
        .enumerate()
        .filter_map(|(id, page)| {
            let Some(page) = page else {
                first_vacant.get_or_insert(id);
                return None;
            };
            page.best_fit(size).map(|fit| (id, fit))
        })
        .min_by_key(|(id, (start, length))| (*length, *id, *start))
        .map(|(id, _)| id);
    existing
        .map(|id| (id, false))
        .or_else(|| first_vacant.map(|id| (id, true)))
        .ok_or(AllocationPlanError::Exhausted)
}

struct MeshletPage {
    buffer: Buffer,
    generation: u64,
    allocator: PageAllocator,
}

/// Manages immutable [`MeshletMesh`] data in fixed-size, independently allocated GPU pages.
#[derive(Resource)]
pub struct MeshletMeshManager {
    pages: Vec<Option<MeshletPage>>,
    dummy_page: Buffer,
    next_page_generation: u64,
    allocations: HashMap<AssetId<MeshletMesh>, MeshletAllocation>,
    pending_uploads: Vec<PendingUpload>,
    failed_uploads: HashMap<AssetId<MeshletMesh>, UploadFailure>,
    pub asset_aabbs: AssetAabbs,
}

/// Model-space AABB of each resident asset, in a slot that is stable for the asset's residency.
///
/// An AABB belongs to the asset, so it is held once here and reached through a per-instance slot
/// rather than replicated across the instances that share the asset. Slots outlive a frame, so
/// unlike the per-instance buffers this uploads only when the resident set changes.
#[derive(Default)]
pub struct AssetAabbs {
    aabbs: StorageBuffer<Vec<MeshletAabb>>,
    /// Slots whose asset has been dropped, available for the next one uploaded.
    free: Vec<u32>,
    /// Whether `aabbs` has changed since it was last uploaded.
    dirty: bool,
}

impl AssetAabbs {
    /// Reserve a slot for an asset, reusing one freed by a dropped asset.
    fn claim(&mut self, aabb: MeshletAabb) -> u32 {
        self.dirty = true;
        match self.free.pop() {
            Some(slot) => {
                self.aabbs.get_mut()[slot as usize] = aabb;
                slot
            }
            None => {
                self.aabbs.get_mut().push(aabb);
                self.aabbs.get().len() as u32 - 1
            }
        }
    }

    /// Return a dropped asset's slot. Its stale AABB is overwritten by whoever claims the slot next.
    fn release(&mut self, slot: u32) {
        self.free.push(slot);
    }

    /// Upload the table if an asset has claimed a slot since the last call.
    fn write_buffer(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        if !self.dirty {
            return;
        }
        self.aabbs.write_buffer(render_device, render_queue);
        self.dirty = false;
    }

    pub fn binding(&self) -> Option<BindingResource<'_>> {
        self.aabbs.binding()
    }
}

pub fn init_meshlet_mesh_manager(mut commands: Commands, render_device: Res<RenderDevice>) {
    let limits = render_device.limits();
    assert!(
        limits.max_buffer_size >= MESHLET_PAGE_SIZE,
        "MeshletPlugin requires max_buffer_size >= {MESHLET_PAGE_SIZE}, got {}",
        limits.max_buffer_size
    );
    assert!(
        limits.max_storage_buffer_binding_size >= MESHLET_PAGE_SIZE,
        "MeshletPlugin requires max_storage_buffer_binding_size >= {MESHLET_PAGE_SIZE}, got {}",
        limits.max_storage_buffer_binding_size
    );
    assert!(
        limits.max_binding_array_elements_per_shader_stage >= MESHLET_MAX_PAGES as u32,
        "MeshletPlugin requires at least {MESHLET_MAX_PAGES} binding-array elements per shader stage, got {}",
        limits.max_binding_array_elements_per_shader_stage
    );

    commands.insert_resource(MeshletMeshManager {
        pages: (0..MESHLET_MAX_PAGES).map(|_| None).collect(),
        dummy_page: render_device.create_buffer(&BufferDescriptor {
            label: Some("meshlet_page_dummy"),
            size: 4,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }),
        next_page_generation: 1,
        allocations: HashMap::default(),
        pending_uploads: Vec::new(),
        failed_uploads: HashMap::default(),
        asset_aabbs: AssetAabbs::default(),
    });
}

impl MeshletMeshManager {
    /// The GPU descriptor, AABB slot and BVH depth of an asset already resident in the page heap.
    ///
    /// An allocation exists only for an asset that loaded and has not since been modified or
    /// dropped, so a hit here answers "is this loaded?" without consulting the asset server - which
    /// matters because [`Self::queue_upload_if_needed`] takes the asset out of `Assets` once it is
    /// resident, leaving loaded and resident as disjoint phases of one lifecycle.
    fn descriptor(
        &self,
        asset_id: AssetId<MeshletMesh>,
    ) -> Option<(MeshletGpuDescriptor, u32, u32)> {
        let allocation = self.allocations.get(&asset_id)?;
        Some((
            allocation.descriptor,
            allocation.asset_index,
            allocation.bvh_depth,
        ))
    }

    /// Queue an asset for upload if needed and return what an instance of it needs: its GPU
    /// descriptor, its slot in [`Self::asset_aabbs`], and its BVH depth.
    ///
    /// `is_loaded` is consulted only for an asset that is not already resident, so a caller with
    /// one instance per frame per asset pays for its load check once rather than every frame.
    ///
    /// Returns `None` for an asset that is still loading, and for one that cannot be uploaded: both
    /// are skipped instead of rendered, and the latter is logged once.
    pub fn queue_upload_if_needed(
        &mut self,
        asset_id: AssetId<MeshletMesh>,
        is_loaded: impl FnOnce() -> bool,
        assets: &mut Assets<MeshletMesh>,
        render_device: &RenderDevice,
    ) -> Option<(MeshletGpuDescriptor, u32, u32)> {
        if let Some(resident) = self.descriptor(asset_id) {
            return Some(resident);
        }
        // Still loading is not a failure - the asset gets another attempt next frame.
        if !is_loaded() {
            return None;
        }
        if self.failed_uploads.get(&asset_id) == Some(&UploadFailure::Permanent) {
            return None;
        }

        let Some(mesh) = assets.get(asset_id) else {
            self.record_failed_upload(
                asset_id,
                UploadFailure::Permanent,
                "asset was unloaded before it could be uploaded".to_string(),
            );
            return None;
        };
        if let Err(reason) = validate_meshlet_mesh(mesh) {
            self.record_failed_upload(
                asset_id,
                UploadFailure::Permanent,
                format!("asset is structurally invalid: {reason}"),
            );
            return None;
        }
        let size = packed_meshlet_mesh_len(mesh) as BufferAddress;

        // Planning only reads allocator state, so every failure remains atomic without cloning all
        // 128 free-range maps.
        let plan = plan_page(
            self.pages
                .iter()
                .map(|page| page.as_ref().map(|page| &page.allocator)),
            size,
        );
        let (page_id, needs_page) = match plan {
            Ok(plan) => plan,
            Err(error) => {
                let (failure, reason) = match error {
                    AllocationPlanError::Empty => {
                        (UploadFailure::Permanent, "asset packed to zero bytes".to_string())
                    }
                    AllocationPlanError::Oversized => (
                        UploadFailure::Permanent,
                        format!("asset requires {size} bytes, exceeding the {MESHLET_PAGE_SIZE}-byte page limit"),
                    ),
                    AllocationPlanError::Exhausted => (
                        UploadFailure::Transient,
                        format!("the {MESHLET_MAX_PAGES} meshlet data pages have no free run of {size} bytes"),
                    ),
                };
                self.record_failed_upload(asset_id, failure, reason);
                return None;
            }
        };
        let (bytes, mut descriptor) = pack_meshlet_mesh(mesh);

        if needs_page {
            let generation = self.next_page_generation;
            self.next_page_generation = self.next_page_generation.wrapping_add(1).max(1);
            self.pages[page_id] = Some(MeshletPage {
                buffer: render_device.create_buffer(&BufferDescriptor {
                    label: Some("meshlet_data_page"),
                    size: MESHLET_PAGE_SIZE,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                generation,
                allocator: PageAllocator::new(),
            });
        }

        let page = self.pages[page_id].as_mut().unwrap();
        let range = page.allocator.allocate(size).unwrap();
        descriptor.page_id = page_id as u32;
        add_allocation_base(&mut descriptor, range.start);

        let mesh = assets.remove_untracked(asset_id).unwrap();
        let asset_index = self.asset_aabbs.claim(mesh.aabb);
        let page = self.pages[page_id].as_mut().unwrap();
        let allocation = MeshletAllocation {
            page_id: page_id as u32,
            page_generation: page.generation,
            range: range.clone(),
            descriptor,
            asset_index,
            bvh_depth: mesh.bvh_depth,
        };
        self.pending_uploads.push(PendingUpload {
            asset_id,
            page_id: page_id as u32,
            page_generation: page.generation,
            offset: range.start,
            bytes,
        });
        self.allocations.insert(asset_id, allocation.clone());
        self.failed_uploads.remove(&asset_id);
        Some((descriptor, allocation.asset_index, allocation.bvh_depth))
    }

    fn record_failed_upload(
        &mut self,
        asset_id: AssetId<MeshletMesh>,
        failure: UploadFailure,
        reason: String,
    ) {
        debug_assert!(false, "MeshletMesh {asset_id:?} was not uploaded: {reason}");
        if self.failed_uploads.insert(asset_id, failure).is_none() {
            error!("MeshletMesh {asset_id:?} was not uploaded and will not be rendered: {reason}");
        }
    }

    pub fn remove(&mut self, asset_id: &AssetId<MeshletMesh>) {
        // A modified asset gets a fresh attempt, including a fresh error report. Dropping the
        // allocation is also what returns the asset to the `is_loaded`-checked path in
        // [`Self::queue_upload_if_needed`], which relies on residency to mean loaded.
        self.failed_uploads.remove(asset_id);
        let Some(allocation) = self.allocations.remove(asset_id) else {
            return;
        };
        self.pending_uploads.retain(|upload| {
            upload.asset_id != *asset_id || upload.page_generation != allocation.page_generation
        });
        let page_id = allocation.page_id as usize;
        let page = self.pages[page_id]
            .as_mut()
            .expect("live meshlet allocation referenced a vacant page");
        assert_eq!(page.generation, allocation.page_generation);
        page.allocator.free(allocation.range);
        self.asset_aabbs.release(allocation.asset_index);
        // Reclamation is deferred to the post-write sweep. A Modified asset removed and re-added
        // in the same frame can therefore reuse this page and its GPU buffer, while an Unused
        // asset still releases its empty page at the end of the prepare-assets phase.
    }

    /// Upload the per-asset AABB table if it changed.
    pub fn write_asset_aabbs(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        self.asset_aabbs.write_buffer(render_device, render_queue);
    }

    pub fn page_bindings(&self) -> Vec<BufferBinding<'_>> {
        assert_eq!(self.pages.len(), MESHLET_MAX_PAGES);
        let bindings: Vec<_> = self
            .pages
            .iter()
            .map(|page| BufferBinding {
                buffer: page.as_ref().map_or(&self.dummy_page, |page| &page.buffer),
                offset: 0,
                size: None,
            })
            .collect();
        assert_eq!(bindings.len(), MESHLET_MAX_PAGES);
        bindings
    }

    fn reclaim_page_if_unused(&mut self, page_id: usize) {
        let Some(page) = self.pages[page_id].as_ref() else {
            return;
        };
        let has_pending_upload = self.pending_uploads.iter().any(|upload| {
            upload.page_id as usize == page_id && upload.page_generation == page.generation
        });
        if page_is_reclaimable(&page.allocator, has_pending_upload) {
            self.pages[page_id] = None;
        }
    }

    fn perform_writes(&mut self, render_queue: &RenderQueue) {
        let uploads = core::mem::take(&mut self.pending_uploads);
        for upload in uploads {
            let Some(page) = self.pages[upload.page_id as usize].as_ref() else {
                continue;
            };
            let allocation = self.allocations.get(&upload.asset_id);
            if !upload_is_current(
                upload.page_id,
                upload.page_generation,
                allocation.map(|allocation| (allocation.page_id, allocation.page_generation)),
                Some(page.generation),
            ) {
                continue;
            }
            for (chunk_index, chunk) in upload.bytes.chunks(UPLOAD_CHUNK_SIZE).enumerate() {
                render_queue.write_buffer(
                    &page.buffer,
                    upload.offset + (chunk_index * UPLOAD_CHUNK_SIZE) as u64,
                    chunk,
                );
            }
        }

        for page_id in 0..self.pages.len() {
            self.reclaim_page_if_unused(page_id);
        }
    }
}

fn page_is_reclaimable(allocator: &PageAllocator, has_pending_upload: bool) -> bool {
    allocator.is_empty() && !has_pending_upload
}

fn upload_is_current(
    upload_page_id: u32,
    upload_generation: u64,
    allocation: Option<(u32, u64)>,
    page_generation: Option<u64>,
) -> bool {
    allocation == Some((upload_page_id, upload_generation))
        && page_generation == Some(upload_generation)
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

fn append_section<T: Pod>(bytes: &mut Vec<u8>, values: &[T]) -> u32 {
    let start = align_up(bytes.len(), SECTION_ALIGNMENT);
    bytes.resize(start, 0);
    bytes.extend_from_slice(bytemuck::cast_slice(values));
    u32::try_from(start / 4).expect("meshlet page section base exceeded u32 words")
}

/// The upload size of a mesh: its streams at `SECTION_ALIGNMENT`, as `pack_meshlet_mesh` lays
/// them out. Exposed to the asset as `MeshletMesh::packed_byte_len` for budgeting.
pub(super) fn packed_meshlet_mesh_len(mesh: &MeshletMesh) -> usize {
    let mut length = 0usize;
    for section_length in [
        size_of_val(mesh.vertex_positions.as_ref()),
        size_of_val(mesh.vertex_normals.as_ref()),
        size_of_val(mesh.vertex_uvs.as_ref()),
        size_of_val(mesh.indices.as_ref()),
        size_of_val(mesh.bvh.as_ref()),
        size_of_val(mesh.meshlets.as_ref()),
        size_of_val(mesh.meshlet_cull_data.as_ref()),
    ] {
        length = align_up(length, SECTION_ALIGNMENT)
            .checked_add(section_length)
            .expect("meshlet packed byte length overflowed usize");
    }
    align_up(length, SECTION_ALIGNMENT)
}

/// Check that an asset only addresses its own data before it is packed into a shared page.
///
/// Meshlet and BVH offsets are asset-local and are resolved against the allocation base, so an
/// out-of-range offset reads whatever else shares the 64 MiB page instead of faulting. A truncated,
/// corrupted, or version-skewed asset must therefore be rejected here rather than rendered.
fn validate_meshlet_mesh(mesh: &MeshletMesh) -> Result<(), String> {
    if mesh.vertex_positions.is_empty() {
        return Err("vertex position stream is empty".to_string());
    }
    if mesh.vertex_normals.is_empty() {
        return Err("vertex normal stream is empty".to_string());
    }
    if mesh.vertex_uvs.is_empty() {
        return Err("vertex UV stream is empty".to_string());
    }
    if mesh.indices.is_empty() {
        return Err("index stream is empty".to_string());
    }
    if mesh.bvh.is_empty() {
        return Err("BVH is empty".to_string());
    }
    if mesh.meshlets.is_empty() {
        return Err("meshlet stream is empty".to_string());
    }
    if mesh.meshlet_cull_data.is_empty() {
        return Err("meshlet cull-data stream is empty".to_string());
    }
    if mesh.vertex_normals.len() != mesh.vertex_uvs.len() {
        return Err("vertex normal and UV streams have different lengths".to_string());
    }
    if mesh.meshlets.len() != mesh.meshlet_cull_data.len() {
        return Err("meshlet and meshlet cull-data streams have different lengths".to_string());
    }
    if mesh.bvh_depth == 0 {
        return Err("BVH depth is zero, so no meshlet would ever be traversed".to_string());
    }

    let vertex_position_bits = mesh.vertex_positions.len() as u64 * 32;
    for (meshlet_id, meshlet) in mesh.meshlets.iter().enumerate() {
        let vertex_count = meshlet.vertex_count_minus_one as u64 + 1;
        let attribute_end = meshlet.start_vertex_attribute_id as u64 + vertex_count;
        if attribute_end > mesh.vertex_normals.len() as u64 {
            return Err(format!(
                "meshlet {meshlet_id} reads vertex attributes up to {attribute_end} of {}",
                mesh.vertex_normals.len()
            ));
        }
        let index_end = meshlet.start_index_id as u64 + meshlet.triangle_count as u64 * 3;
        if index_end > mesh.indices.len() as u64 {
            return Err(format!(
                "meshlet {meshlet_id} reads indices up to {index_end} of {}",
                mesh.indices.len()
            ));
        }
        let bits_per_vertex = meshlet.bits_per_vertex_position_channel_x as u64
            + meshlet.bits_per_vertex_position_channel_y as u64
            + meshlet.bits_per_vertex_position_channel_z as u64;
        let position_bit_end =
            meshlet.start_vertex_position_bit as u64 + vertex_count * bits_per_vertex;
        if position_bit_end > vertex_position_bits {
            return Err(format!(
                "meshlet {meshlet_id} reads position bits up to {position_bit_end} of {vertex_position_bits}"
            ));
        }
    }

    // Culling dispatches exactly `bvh_depth` BVH passes, so an understated depth silently drops
    // subtrees. Walking the tree bounds every child offset at the same time, and the visited set
    // keeps a malformed asset from looping forever here.
    let mut visited = vec![false; mesh.bvh.len()];
    let mut current = vec![0u32];
    let mut next = Vec::new();
    visited[0] = true;
    let mut depth = 0;
    while !current.is_empty() {
        depth += 1;
        if depth > mesh.bvh_depth {
            return Err(format!(
                "BVH is deeper than its recorded depth of {}",
                mesh.bvh_depth
            ));
        }
        for node_id in current.drain(..) {
            let node = &mesh.bvh[node_id as usize];
            for child in 0..8 {
                let child_count = node.child_counts[child];
                if child_count == 0 {
                    continue;
                }
                let child_offset = node.aabbs[child].child_offset;
                if child_count == u8::MAX {
                    if child_offset as usize >= mesh.bvh.len() {
                        return Err(format!(
                            "BVH node {node_id} child {child} is node {child_offset} of {}",
                            mesh.bvh.len()
                        ));
                    }
                    if core::mem::replace(&mut visited[child_offset as usize], true) {
                        return Err(format!("BVH node {child_offset} has multiple parents"));
                    }
                    next.push(child_offset);
                } else {
                    let meshlet_end = child_offset as u64 + child_count as u64;
                    if meshlet_end > mesh.meshlets.len() as u64 {
                        return Err(format!(
                            "BVH node {node_id} child {child} covers meshlets up to {meshlet_end} of {}",
                            mesh.meshlets.len()
                        ));
                    }
                }
            }
        }
        core::mem::swap(&mut current, &mut next);
    }

    Ok(())
}

fn pack_meshlet_mesh(mesh: &MeshletMesh) -> (Arc<[u8]>, MeshletGpuDescriptor) {
    let packed_length = packed_meshlet_mesh_len(mesh);
    let mut bytes = Vec::with_capacity(packed_length);
    let vertex_positions_base = append_section(&mut bytes, &mesh.vertex_positions);
    let vertex_normals_base = append_section(&mut bytes, &mesh.vertex_normals);
    let vertex_uvs_base = append_section(&mut bytes, &mesh.vertex_uvs);
    let indices_base = append_section(&mut bytes, &mesh.indices);
    let bvh_nodes_base = append_section(&mut bytes, &mesh.bvh);
    let meshlets_base = append_section(&mut bytes, &mesh.meshlets);
    let meshlet_cull_data_base = append_section(&mut bytes, &mesh.meshlet_cull_data);
    bytes.resize(packed_length, 0);
    debug_assert_eq!(bytes.len(), packed_length);

    (
        bytes.into(),
        MeshletGpuDescriptor {
            page_id: 0,
            vertex_positions_base,
            vertex_normals_base,
            vertex_uvs_base,
            indices_base,
            bvh_nodes_base,
            meshlets_base,
            meshlet_cull_data_base,
        },
    )
}

fn add_allocation_base(descriptor: &mut MeshletGpuDescriptor, allocation_start: BufferAddress) {
    assert!(allocation_start.is_multiple_of(4));
    let allocation_base = u32::try_from(allocation_start / 4).unwrap();
    for base in [
        &mut descriptor.vertex_positions_base,
        &mut descriptor.vertex_normals_base,
        &mut descriptor.vertex_uvs_base,
        &mut descriptor.indices_base,
        &mut descriptor.bvh_nodes_base,
        &mut descriptor.meshlets_base,
        &mut descriptor.meshlet_cull_data_base,
    ] {
        *base = base.checked_add(allocation_base).unwrap();
    }
}

/// Upload all newly queued [`MeshletMesh`] asset data to GPU pages.
pub fn perform_pending_meshlet_mesh_writes(
    mut meshlet_mesh_manager: ResMut<MeshletMeshManager>,
    render_queue: Res<RenderQueue>,
) {
    meshlet_mesh_manager.perform_writes(&render_queue);
}

#[cfg(test)]
mod tests {

    fn aabb(center: f32) -> MeshletAabb {
        MeshletAabb {
            center: Vec3::splat(center),
            half_extent: Vec3::ONE,
        }
    }

    #[test]
    fn asset_aabb_slots_are_dense_until_one_is_released() {
        let mut table = AssetAabbs::default();

        assert_eq!(table.claim(aabb(1.0)), 0);
        assert_eq!(table.claim(aabb(2.0)), 1);
        assert_eq!(table.claim(aabb(3.0)), 2);
        assert_eq!(table.aabbs.get().len(), 3);
    }

    #[test]
    fn a_released_slot_is_reused_and_overwritten() {
        let mut table = AssetAabbs::default();
        table.claim(aabb(1.0));
        let released = table.claim(aabb(2.0));
        table.claim(aabb(3.0));

        table.release(released);

        // The slot comes back rather than growing the table, carrying the new asset's AABB - a
        // stale one here would cull the new asset against the dropped one's bounds.
        assert_eq!(table.claim(aabb(4.0)), released);
        assert_eq!(table.aabbs.get().len(), 3);
        assert_eq!(
            table.aabbs.get()[released as usize].center,
            Vec3::splat(4.0)
        );
    }

    #[test]
    fn the_table_uploads_only_after_a_claim() {
        let mut table = AssetAabbs::default();
        assert!(!table.dirty);

        table.claim(aabb(1.0));
        assert!(table.dirty);

        // Releasing leaves the buffer untouched; the slot's contents only matter once reclaimed.
        table.dirty = false;
        table.release(0);
        assert!(!table.dirty);
    }
    use super::*;
    use crate::meshlet::asset::{BvhNode, Meshlet, MeshletCullData};
    use alloc::vec;
    use bevy_math::Vec3;

    #[test]
    fn descriptor_layout_is_exactly_eight_words() {
        assert_eq!(size_of::<MeshletGpuDescriptor>(), 32);
        assert_eq!(MeshletGpuDescriptor::min_size().get(), 32);
    }

    #[test]
    fn allocator_uses_best_fit_and_merges_holes() {
        let mut allocator = PageAllocator::new();
        let a = allocator.allocate(32).unwrap();
        let b = allocator.allocate(16).unwrap();
        let c = allocator.allocate(48).unwrap();
        allocator.free(b.clone());
        assert_eq!(allocator.allocate(8).unwrap(), b.start..b.start + 8);
        allocator.free(a);
        allocator.free(c);
        allocator.free(b.start..b.start + 8);
        assert!(allocator.is_empty());
    }

    #[test]
    fn allocator_chooses_smallest_of_multiple_holes() {
        let mut allocator = PageAllocator::new();
        let _a = allocator.allocate(32).unwrap();
        let b = allocator.allocate(128).unwrap();
        let _c = allocator.allocate(48).unwrap();
        let d = allocator.allocate(96).unwrap();
        let _e = allocator.allocate(16).unwrap();
        allocator.free(b);
        allocator.free(d.clone());
        assert_eq!(allocator.allocate(80).unwrap(), d.start..d.start + 80);
    }

    #[test]
    fn empty_and_oversized_plans_are_atomic() {
        let pages = vec![Some(PageAllocator::new()), None];
        let before = pages.clone();
        assert_eq!(
            plan_page(pages.iter().map(Option::as_ref), 0),
            Err(AllocationPlanError::Empty)
        );
        assert_eq!(
            plan_page(pages.iter().map(Option::as_ref), MESHLET_PAGE_SIZE + 4),
            Err(AllocationPlanError::Oversized)
        );
        for (actual, expected) in pages.iter().zip(before.iter()) {
            assert_eq!(
                actual.as_ref().map(|page| (&page.free, page.allocations)),
                expected.as_ref().map(|page| (&page.free, page.allocations))
            );
        }

        let mut allocator = PageAllocator::new();
        let before = allocator.clone();
        assert_eq!(allocator.allocate(0), None);
        assert_eq!(allocator.free, before.free);
        assert_eq!(allocator.allocations, before.allocations);
    }

    #[test]
    fn structurally_empty_mesh_is_rejected_before_packing() {
        let mesh = MeshletMesh {
            vertex_positions: Arc::from([]),
            vertex_normals: Arc::from([]),
            vertex_uvs: Arc::from([]),
            indices: Arc::from([]),
            bvh: Arc::from([]),
            meshlets: Arc::from([]),
            meshlet_cull_data: Arc::from([]),
            aabb: MeshletAabb::default(),
            bvh_depth: 0,
        };
        assert_eq!(
            validate_meshlet_mesh(&mesh),
            Err("vertex position stream is empty".to_string())
        );
    }

    /// Two meshlets addressing the first and second half of every stream, under a root BVH node
    /// pointing at both of them.
    fn valid_mesh() -> MeshletMesh {
        let meshlet =
            |start_vertex_position_bit, start_vertex_attribute_id, start_index_id| Meshlet {
                start_vertex_position_bit,
                start_vertex_attribute_id,
                start_index_id,
                vertex_count_minus_one: 1,
                triangle_count: 1,
                bits_per_vertex_position_channel_x: 8,
                bits_per_vertex_position_channel_y: 8,
                bits_per_vertex_position_channel_z: 8,
                ..Zeroable::zeroed()
            };
        let mut bvh = BvhNode::zeroed();
        bvh.child_counts[0] = 2;
        MeshletMesh {
            vertex_positions: Arc::from([0u32, 0, 0]),
            vertex_normals: Arc::from([0u32; 4]),
            vertex_uvs: Arc::from([0u32; 4]),
            indices: Arc::from([0u8; 6]),
            bvh: Arc::from([bvh]),
            meshlets: Arc::from([meshlet(0, 0, 0), meshlet(48, 2, 3)]),
            meshlet_cull_data: Arc::from([MeshletCullData::zeroed(); 2]),
            aabb: MeshletAabb::default(),
            bvh_depth: 1,
        }
    }

    #[test]
    fn out_of_range_meshlet_offsets_are_rejected() {
        assert_eq!(validate_meshlet_mesh(&valid_mesh()), Ok(()));

        let mut mesh = valid_mesh();
        Arc::get_mut(&mut mesh.meshlets).unwrap()[1].start_vertex_attribute_id = 3;
        assert!(validate_meshlet_mesh(&mesh)
            .unwrap_err()
            .contains("vertex attributes"));

        let mut mesh = valid_mesh();
        Arc::get_mut(&mut mesh.meshlets).unwrap()[1].start_index_id = 4;
        assert!(validate_meshlet_mesh(&mesh)
            .unwrap_err()
            .contains("indices"));

        let mut mesh = valid_mesh();
        Arc::get_mut(&mut mesh.meshlets).unwrap()[1].start_vertex_position_bit = 49;
        assert!(validate_meshlet_mesh(&mesh)
            .unwrap_err()
            .contains("position bits"));
    }

    #[test]
    fn out_of_range_bvh_children_are_rejected() {
        let mut mesh = valid_mesh();
        Arc::get_mut(&mut mesh.bvh).unwrap()[0].child_counts[0] = 3;
        assert!(validate_meshlet_mesh(&mesh)
            .unwrap_err()
            .contains("meshlets"));

        let mut mesh = valid_mesh();
        let bvh = Arc::get_mut(&mut mesh.bvh).unwrap();
        bvh[0].child_counts[0] = u8::MAX;
        bvh[0].aabbs[0].child_offset = 1;
        assert!(validate_meshlet_mesh(&mesh).unwrap_err().contains("node 1"));
    }

    #[test]
    fn zero_and_understated_bvh_depths_are_rejected() {
        let mut mesh = valid_mesh();
        mesh.bvh_depth = 0;
        assert_eq!(
            validate_meshlet_mesh(&mesh),
            Err("BVH depth is zero, so no meshlet would ever be traversed".to_string())
        );

        // A root whose only child is a second node needs a depth of two.
        let mut mesh = valid_mesh();
        let mut root = BvhNode::zeroed();
        root.child_counts[0] = u8::MAX;
        root.aabbs[0].child_offset = 1;
        let leaf = mesh.bvh[0];
        mesh.bvh = Arc::from([root, leaf]);
        assert_eq!(
            validate_meshlet_mesh(&mesh),
            Err("BVH is deeper than its recorded depth of 1".to_string())
        );
        mesh.bvh_depth = 2;
        assert_eq!(validate_meshlet_mesh(&mesh), Ok(()));
    }

    #[test]
    fn bvh_nodes_with_multiple_parents_are_rejected() {
        let mut mesh = valid_mesh();
        let mut root = BvhNode::zeroed();
        root.child_counts[0] = u8::MAX;
        root.child_counts[1] = u8::MAX;
        root.aabbs[0].child_offset = 1;
        root.aabbs[1].child_offset = 1;
        let leaf = mesh.bvh[0];
        mesh.bvh = Arc::from([root, leaf]);
        mesh.bvh_depth = 2;
        assert_eq!(
            validate_meshlet_mesh(&mesh),
            Err("BVH node 1 has multiple parents".to_string())
        );
    }

    #[test]
    fn allocation_base_preserves_asset_local_offsets() {
        let mut descriptor = MeshletGpuDescriptor {
            vertex_positions_base: 0,
            vertex_normals_base: 4,
            vertex_uvs_base: 8,
            indices_base: 12,
            bvh_nodes_base: 16,
            meshlets_base: 20,
            meshlet_cull_data_base: 24,
            ..Default::default()
        };
        add_allocation_base(&mut descriptor, 64);
        assert_eq!(descriptor.vertex_positions_base, 16);
        assert_eq!(descriptor.meshlet_cull_data_base, 40);
    }

    #[test]
    fn packing_aligns_sections_and_preserves_serialized_bytes() {
        let meshlet = Meshlet {
            start_vertex_position_bit: 37,
            start_vertex_attribute_id: 2,
            start_index_id: 3,
            ..Zeroable::zeroed()
        };
        let mesh = MeshletMesh {
            vertex_positions: Arc::from([11u32, 22]),
            vertex_normals: Arc::from([33u32, 44, 55]),
            vertex_uvs: Arc::from([66u32, 77, 88]),
            indices: Arc::from([1u8, 2, 3, 4, 5]),
            bvh: Arc::from([BvhNode::zeroed()]),
            meshlets: Arc::from([meshlet]),
            meshlet_cull_data: Arc::from([MeshletCullData::zeroed()]),
            aabb: MeshletAabb::default(),
            bvh_depth: 1,
        };
        assert_eq!(validate_meshlet_mesh(&mesh), Ok(()));
        assert_eq!(packed_meshlet_mesh_len(&mesh), 560);
        assert_eq!(mesh.packed_byte_len(), 560);
        let (bytes, descriptor) = pack_meshlet_mesh(&mesh);
        assert_eq!(bytes.len(), packed_meshlet_mesh_len(&mesh));
        for base in [
            descriptor.vertex_positions_base,
            descriptor.vertex_normals_base,
            descriptor.vertex_uvs_base,
            descriptor.indices_base,
            descriptor.bvh_nodes_base,
            descriptor.meshlets_base,
            descriptor.meshlet_cull_data_base,
        ] {
            assert_eq!(base % (SECTION_ALIGNMENT as u32 / 4), 0);
        }

        let meshlet_offset = descriptor.meshlets_base as usize * 4;
        assert_eq!(
            &bytes[meshlet_offset..meshlet_offset + size_of::<Meshlet>()],
            bytemuck::bytes_of(&meshlet)
        );
        let packed_indices = &bytes[descriptor.indices_base as usize * 4..];
        assert_eq!(&packed_indices[..5], &[1, 2, 3, 4, 5]);
        assert_eq!(
            descriptor.vertex_positions_base + meshlet.start_vertex_position_bit / 32,
            1
        );
        assert_eq!(
            descriptor.vertex_normals_base + meshlet.start_vertex_attribute_id,
            6
        );
    }

    #[test]
    fn pages_do_not_need_copy_source_usage() {
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        assert!(!usage.contains(BufferUsages::COPY_SRC));
    }

    #[test]
    fn page_selection_reuses_holes_before_vacant_slots() {
        let mut first = PageAllocator::new();
        let allocation = first.allocate(64).unwrap();
        let mut second = PageAllocator::new();
        second.allocate(MESHLET_PAGE_SIZE).unwrap();
        let pages = vec![Some(first), Some(second), None];
        assert_eq!(
            plan_page(pages.iter().map(Option::as_ref), 32),
            Ok((0, false))
        );

        let mut first = pages[0].clone().unwrap();
        first.free(allocation);
        assert!(first.is_empty());
        let reclaimed = vec![None, pages[1].clone(), None];
        assert_eq!(
            plan_page(reclaimed.iter().map(Option::as_ref), MESHLET_PAGE_SIZE),
            Ok((0, true))
        );
    }

    #[test]
    fn page_exhaustion_planning_does_not_mutate_allocators() {
        let mut full = PageAllocator::new();
        full.allocate(MESHLET_PAGE_SIZE).unwrap();
        let pages = vec![Some(full); MESHLET_MAX_PAGES];
        let before = pages.clone();
        assert_eq!(
            plan_page(pages.iter().map(Option::as_ref), 16),
            Err(AllocationPlanError::Exhausted)
        );
        for (actual, expected) in pages.iter().zip(before.iter()) {
            assert_eq!(
                actual.as_ref().unwrap().free,
                expected.as_ref().unwrap().free
            );
            assert_eq!(
                actual.as_ref().unwrap().allocations,
                expected.as_ref().unwrap().allocations
            );
        }
    }

    #[test]
    fn exactly_128_pages_are_addressable_then_exhaustion_is_atomic() {
        let mut pages = vec![None; MESHLET_MAX_PAGES];
        for page_id in 0..MESHLET_MAX_PAGES {
            assert_eq!(
                plan_page(pages.iter().map(Option::as_ref), MESHLET_PAGE_SIZE),
                Ok((page_id, true))
            );
            let mut full = PageAllocator::new();
            full.allocate(MESHLET_PAGE_SIZE).unwrap();
            pages[page_id] = Some(full);
        }
        let before = pages.clone();
        assert_eq!(
            plan_page(pages.iter().map(Option::as_ref), 4),
            Err(AllocationPlanError::Exhausted)
        );
        assert_eq!(pages.len(), MESHLET_MAX_PAGES);
        for (actual, expected) in pages.iter().zip(before.iter()) {
            assert_eq!(
                actual.as_ref().unwrap().free,
                expected.as_ref().unwrap().free
            );
            assert_eq!(
                actual.as_ref().unwrap().allocations,
                expected.as_ref().unwrap().allocations
            );
        }
    }

    #[test]
    fn page_reclamation_waits_for_all_allocations_and_uploads() {
        let mut allocator = PageAllocator::new();
        let allocation = allocator.allocate(32).unwrap();
        assert!(!page_is_reclaimable(&allocator, false));
        allocator.free(allocation);
        assert!(!page_is_reclaimable(&allocator, true));
        assert!(page_is_reclaimable(&allocator, false));
    }

    #[test]
    fn empty_page_is_reused_until_post_write_reclamation() {
        let mut allocator = PageAllocator::new();
        let allocation = allocator.allocate(32).unwrap();
        allocator.free(allocation);
        let pages = vec![Some(allocator)];
        assert_eq!(
            plan_page(pages.iter().map(Option::as_ref), 32),
            Ok((0, false))
        );

        let reclaimed = vec![None];
        assert_eq!(
            plan_page(reclaimed.iter().map(Option::as_ref), 32),
            Ok((0, true))
        );
    }

    #[test]
    fn stale_or_cancelled_upload_generations_are_rejected() {
        assert!(upload_is_current(3, 7, Some((3, 7)), Some(7)));
        assert!(!upload_is_current(3, 7, None, Some(7)));
        assert!(!upload_is_current(3, 7, Some((3, 8)), Some(8)));
        assert!(!upload_is_current(3, 7, Some((4, 7)), Some(7)));
        assert!(!upload_is_current(3, 7, Some((3, 7)), None));
    }
}
