use super::{
    allocator::{IndexAllocator, RetainedBindingArray},
    assets::AssetState,
    lights::{GpuLightSource, LightSourceId, LightState},
    BlasManager, RaytracingGeometry, RaytracingGeometryBuffers, RaytracingMesh3d,
    RaytracingSceneBindings,
};
use crate::scene::{
    blas::GeometryBlasManager, RaytracingGeometryPreviousVertices,
    RaytracingGeometryTopologyGeneration,
};
use bevy_asset::AssetId;
use bevy_ecs::{
    entity::{Entity, EntityHashMap, EntityHashSet},
    query::{Changed, Has, Or, QueryItem, With},
    system::Query,
};
use bevy_math::{Affine3, Affine3Ext, Vec4};
use bevy_mesh::Mesh;
use bevy_pbr::{MeshMaterial3d, PreviousGlobalTransform, StandardMaterial};
use bevy_platform::collections::HashMap;
use bevy_render::{
    impl_atomic_pod,
    mesh::allocator::MeshAllocator,
    render_resource::{AtomicPod, AtomicSparseBufferVec, Buffer, BufferId, BufferUsages},
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::once;
use bytemuck::{Pod, Zeroable};
use core::{hash::Hash, num::NonZeroU32};
use tracing::{info_span, warn};

pub const MAX_MESH_SLAB_COUNT: NonZeroU32 = NonZeroU32::new(500).unwrap();

/// Translation for an instance that is no longer present in its previous slot.
const INSTANCE_NOT_PRESENT_THIS_FRAME: u32 = u32::MAX;

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuInstanceGeometryIds {
    vertex_buffer_id: u32,
    vertex_buffer_offset: u32,
    previous_vertex_buffer_id: u32,
    index_buffer_id: u32,
    index_buffer_offset: u32,
    triangle_count: u32,
    ray_mask: u32,
    diagnostic_primitives_per_group: u32,
}

/// A world-from-local affine transform, stored transposed as three rows.
#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuTransform([Vec4; 3]);

impl GpuTransform {
    /// The three rows as the flat row-major 3x4 a [`TlasInstance`] wants.
    ///
    /// [`TlasInstance`]: bevy_render::render_resource::TlasInstance
    fn rows(self) -> [f32; 12] {
        bytemuck::cast(self)
    }
}

/// The device address of a slot's acceleration structure. Zero marks an inactive slot.
#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuBlasRef {
    address: u64,
    mask: u32,
    _padding: u32,
}

impl GpuBlasRef {
    const NONE: Self = Self { address: 0, mask: 0, _padding: 0 };

    fn new(address: u64, mask: u8) -> Self {
        Self { address, mask: u32::from(mask), _padding: 0 }
    }
}

impl_atomic_pod!(GpuInstanceGeometryIds, GpuInstanceGeometryIdsBlob);
impl_atomic_pod!(GpuTransform, GpuTransformBlob);
impl_atomic_pod!(GpuBlasRef, GpuBlasRefBlob);

fn storage_buffer<T: AtomicPod>(label: &str) -> AtomicSparseBufferVec<T> {
    AtomicSparseBufferVec::new(BufferUsages::STORAGE, label.into())
}

/// Where an instance's triangles and acceleration structure come from.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum InstanceSource {
    /// Mesh allocator buffer slices and an asset-keyed BLAS.
    Mesh(AssetId<Mesh>),
    /// Producer-owned buffers and an entity-keyed BLAS.
    Geometry,
}

impl InstanceSource {
    fn mesh(self) -> Option<AssetId<Mesh>> {
        match self {
            Self::Mesh(mesh) => Some(mesh),
            Self::Geometry => None,
        }
    }
}

/// Buffer locations whose triangles can be reused by temporal geometry anchors.
#[derive(Clone, Copy, PartialEq, Eq)]
struct GeometryKey {
    topology_generation: u64,
    source: InstanceSource,
    vertex_buffer: BufferId,
    vertex_buffer_offset: u32,
    index_buffer: BufferId,
    index_buffer_offset: u32,
    triangle_count: u32,
}

/// Everything tracked per raytracing instance.
#[derive(Clone, Copy)]
struct Instance {
    slot: u32,
    source: InstanceSource,
    material: AssetId<StandardMaterial>,
    buffers: Option<(BufferId, BufferId, BufferId)>,
    geometry: Option<GeometryKey>,
}

/// Stable slots, reverse dependency indices and GPU data owned by raytracing instances.
pub struct InstanceState {
    pub vertex_buffers: RetainedBindingArray<BufferId, Buffer>,
    pub index_buffers: RetainedBindingArray<BufferId, Buffer>,
    pub transforms: AtomicSparseBufferVec<GpuTransform>,
    pub previous_frame_transforms: AtomicSparseBufferVec<GpuTransform>,
    pub geometry_ids: AtomicSparseBufferVec<GpuInstanceGeometryIds>,
    pub material_ids: AtomicSparseBufferVec<u32>,
    pub blas_refs: AtomicSparseBufferVec<GpuBlasRef>,
    /// Maps instance slots from the last consumed frame to the current frame.
    /// Removed instances map to [`INSTANCE_NOT_PRESENT_THIS_FRAME`].
    pub previous_frame_id_translations: AtomicSparseBufferVec<u32>,
    pub slots: IndexAllocator,
    records: EntityHashMap<Instance>,
    pub live_count: u32,
    pub pending_refresh: EntityHashSet,
    mesh_instances: HashMap<AssetId<Mesh>, EntityHashSet>,
    pub material_instances: HashMap<AssetId<StandardMaterial>, EntityHashSet>,
    /// Live instance slots from the last consumed frame.
    previous_slots: EntityHashMap<u32>,
    /// Instances whose liveness changed since the last consumed frame.
    liveness_changed: EntityHashSet,
    /// Instances whose old triangle anchors must be discarded until the next consumed frame.
    invalidated_geometry: EntityHashSet,
    nonidentity_translations: Vec<u32>,
}

impl InstanceState {
    pub fn new() -> Self {
        Self {
            vertex_buffers: RetainedBindingArray::new(),
            index_buffers: RetainedBindingArray::new(),
            transforms: storage_buffer("solari_transforms"),
            previous_frame_transforms: storage_buffer("solari_previous_frame_transforms"),
            geometry_ids: storage_buffer("solari_geometry_ids"),
            material_ids: storage_buffer("solari_material_ids"),
            blas_refs: storage_buffer("solari_blas_refs"),
            previous_frame_id_translations: storage_buffer(
                "solari_previous_frame_instance_id_translations",
            ),
            slots: IndexAllocator::new(),
            records: EntityHashMap::default(),
            live_count: 0,
            pending_refresh: EntityHashSet::default(),
            mesh_instances: HashMap::default(),
            material_instances: HashMap::default(),
            previous_slots: EntityHashMap::default(),
            liveness_changed: EntityHashSet::default(),
            invalidated_geometry: EntityHashSet::default(),
            nonidentity_translations: Vec::new(),
        }
    }

    /// The slot of `entity` while it is drawable.
    fn live_slot(&self, entity: Entity) -> Option<u32> {
        let slot = self.records.get(&entity)?.slot;
        (self.blas_refs.get(slot) != GpuBlasRef::NONE).then_some(slot)
    }

    /// Starts a new translation frame. Advance the snapshot only after the previous
    /// table was consumed, matching [`LightState::begin_frame`].
    pub fn begin_frame(&mut self, translations_consumed: bool) {
        for index in core::mem::take(&mut self.nonidentity_translations) {
            self.previous_frame_id_translations
                .grow_and_set(index, index);
        }

        if translations_consumed {
            self.invalidated_geometry.clear();
            for entity in core::mem::take(&mut self.liveness_changed) {
                match self.live_slot(entity) {
                    Some(slot) => self.previous_slots.insert(entity, slot),
                    None => self.previous_slots.remove(&entity),
                };
            }
        }
    }

    /// Writes translations for instances that moved to another slot or became inactive.
    pub fn write_instance_id_translations(&mut self) {
        let changed: Vec<Entity> = self.liveness_changed.iter().copied().collect();
        for entity in changed {
            // Instances that first appeared since the last read table have no previous slot
            let Some(&previous) = self.previous_slots.get(&entity) else {
                continue;
            };
            let current = if self.invalidated_geometry.contains(&entity) {
                INSTANCE_NOT_PRESENT_THIS_FRAME
            } else {
                self.live_slot(entity)
                    .unwrap_or(INSTANCE_NOT_PRESENT_THIS_FRAME)
            };

            if current != previous {
                self.previous_frame_id_translations
                    .grow_and_set(previous, current);
                self.nonidentity_translations.push(previous);
            }
        }

        // Initialize new slots with identity translations
        let slot_count = self.slots.high_water_mark();
        let translations = &mut self.previous_frame_id_translations;
        if translations.len() < slot_count {
            let start = translations.len();
            translations.grow(slot_count);
            for index in start..slot_count {
                translations.set(index, index);
            }
        }
    }

    /// Every drawable instance's entity, slot, source and world-from-local transform.
    ///
    /// Only the `wgpu-core` TLAS build path needs this, to fill in the instance descriptors that
    /// the raw path sets up on the GPU. Slots with a null acceleration structure reference are not
    /// currently drawable, and are left out.
    pub fn drawable(&self) -> impl Iterator<Item = (Entity, u32, InstanceSource, [f32; 12], u8)> + '_ {
        self.records.iter().filter_map(|(entity, instance)| {
            let slot = instance.slot;
            (self.blas_refs.get(slot) != GpuBlasRef::NONE).then(|| {
                (
                    *entity,
                    slot,
                    instance.source,
                    self.transforms.get(slot).rows(),
                    self.blas_refs.get(slot).mask as u8,
                )
            })
        })
    }

    /// Queues every instance using `material_id` to be re-resolved.
    pub fn invalidate_material(&mut self, material_id: AssetId<StandardMaterial>) {
        if let Some(instances) = self.material_instances.get(&material_id) {
            self.pending_refresh.extend(instances.iter().copied());
        }
    }
}

pub type InstanceQueryData<'w> = (
    Option<&'w RaytracingMesh3d>,
    Has<RaytracingGeometry>,
    Option<&'w RaytracingGeometryBuffers>,
    Option<&'w RaytracingGeometryPreviousVertices>,
    Option<&'w RaytracingGeometryTopologyGeneration>,
    &'w MeshMaterial3d<StandardMaterial>,
    &'w GlobalTransform,
    &'w PreviousGlobalTransform,
);

/// Filter matching both kinds of raytracing instance.
pub type InstanceQueryFilter = Or<(With<RaytracingMesh3d>, With<RaytracingGeometry>)>;

pub type ChangedInstanceFilter = (
    InstanceQueryFilter,
    Or<(
        Changed<RaytracingMesh3d>,
        Changed<RaytracingGeometry>,
        Changed<RaytracingGeometryBuffers>,
        Changed<RaytracingGeometryPreviousVertices>,
        Changed<RaytracingGeometryTopologyGeneration>,
        Changed<MeshMaterial3d<StandardMaterial>>,
    )>,
);

/// The scene state an instance resolves its GPU data against.
pub struct InstanceInputs<'a> {
    pub assets: &'a AssetState,
    pub blas_manager: &'a BlasManager,
    pub geometry_blas_manager: &'a GeometryBlasManager,
    pub mesh_allocator: &'a MeshAllocator,
}

/// An instance's triangles and acceleration structure, resolved against this frame's scene state.
struct ResolvedGeometry<'a> {
    vertex_buffer: &'a Buffer,
    vertex_buffer_offset: u32,
    previous_vertex_buffer: &'a Buffer,
    index_buffer: &'a Buffer,
    index_buffer_offset: u32,
    triangle_count: u32,
    blas_address: u64,
    ray_mask: u8,
    diagnostic_primitives_per_group: u32,
}

fn unlink<K: Eq + Hash>(map: &mut HashMap<K, EntityHashSet>, key: &K, entity: Entity) {
    let now_empty = map.get_mut(key).is_some_and(|instances| {
        instances.remove(&entity);
        instances.is_empty()
    });
    if now_empty {
        map.remove(key);
    }
}

fn relink<K: Copy + Eq + Hash>(
    map: &mut HashMap<K, EntityHashSet>,
    entity: Entity,
    previous: Option<K>,
    key: Option<K>,
) {
    if previous == key {
        return;
    }
    if let Some(previous) = previous {
        unlink(map, &previous, entity);
    }
    if let Some(key) = key {
        map.entry(key).or_default().insert(entity);
    }
}

impl InstanceState {
    pub fn remove_instances(
        &mut self,
        lights: &mut LightState,
        removed: impl IntoIterator<Item = Entity>,
    ) {
        let _span = info_span!("remove_instances").entered();
        for entity in removed {
            self.remove_instance(lights, entity);
        }
    }

    pub fn refresh_instances(
        &mut self,
        inputs: &InstanceInputs,
        lights: &mut LightState,
        instances: &Query<InstanceQueryData, InstanceQueryFilter>,
        changed_instances: &Query<Entity, ChangedInstanceFilter>,
    ) {
        let _span = info_span!("refresh_instances").entered();

        let mut refresh = core::mem::take(&mut self.pending_refresh);
        refresh.extend(changed_instances.iter());
        refresh.extend(inputs.geometry_blas_manager.changed_entities());

        let moved_meshes = inputs.mesh_allocator.meshes_displaced_by_slab_growth();
        for mesh_id in inputs
            .blas_manager
            .changed_meshes()
            .iter()
            .copied()
            .chain(moved_meshes)
        {
            if let Some(mesh_instances) = self.mesh_instances.get(&mesh_id) {
                refresh.extend(mesh_instances.iter().copied());
            }
        }

        for entity in refresh {
            match instances.get(entity) {
                Ok(data) => self.refresh_instance(inputs, lights, entity, data),
                Err(_) => self.remove_instance(lights, entity),
            }
        }
    }

    fn reserve_slot(&mut self, slot: u32) {
        let len = slot + 1;
        self.transforms.grow(len);
        self.previous_frame_transforms.grow(len);
        self.blas_refs.grow(len);
    }

    fn refresh_instance(
        &mut self,
        inputs: &InstanceInputs,
        lights: &mut LightState,
        entity: Entity,
        (
            mesh,
            has_geometry,
            geometry,
            previous_vertices,
            topology_generation,
            material,
            transform,
            previous_frame_transform,
        ): QueryItem<'_, '_, InstanceQueryData>,
    ) {
        let source = match (has_geometry, mesh) {
            (false, Some(mesh)) => InstanceSource::Mesh(mesh.id()),
            _ => InstanceSource::Geometry,
        };
        let material_id = material.id();
        let previous = self.records.get(&entity).copied();

        relink(
            &mut self.mesh_instances,
            entity,
            previous.and_then(|instance| instance.source.mesh()),
            source.mesh(),
        );
        relink(
            &mut self.material_instances,
            entity,
            previous.map(|instance| instance.material),
            Some(material_id),
        );

        let slot = match previous {
            Some(previous) => previous.slot,
            None => self.slots.allocate(),
        };
        self.reserve_slot(slot);

        // Seed only once. Later refreshes must not overwrite transforms written by extraction.
        if previous.is_none() {
            self.write_transforms(slot, transform, previous_frame_transform);
        }

        let mut instance = Instance {
            slot,
            source,
            material: material_id,
            buffers: previous.and_then(|instance| instance.buffers),
            geometry: previous.and_then(|instance| instance.geometry),
        };
        let resolved = self.resolve_instance(
            inputs,
            lights,
            entity,
            &mut instance,
            geometry,
            previous_vertices,
            topology_generation.map_or(0, |generation| generation.0),
        );

        self.records.insert(entity, instance);
        if !resolved {
            self.pending_refresh.insert(entity);
        }
    }

    fn resolve_geometry<'a>(
        inputs: &InstanceInputs<'a>,
        entity: Entity,
        source: InstanceSource,
        geometry: Option<&'a RaytracingGeometryBuffers>,
        previous_vertices: Option<&'a RaytracingGeometryPreviousVertices>,
    ) -> Option<ResolvedGeometry<'a>> {
        match source {
            InstanceSource::Mesh(mesh) => {
                let vertex_slice = inputs.mesh_allocator.mesh_vertex_slice(&mesh)?;
                let index_slice = inputs.mesh_allocator.mesh_index_slice(&mesh)?;
                Some(ResolvedGeometry {
                    vertex_buffer: vertex_slice.buffer,
                    vertex_buffer_offset: vertex_slice.range.start,
                    previous_vertex_buffer: vertex_slice.buffer,
                    index_buffer: index_slice.buffer,
                    index_buffer_offset: index_slice.range.start,
                    triangle_count: (index_slice.range.len() / 3) as u32,
                    blas_address: inputs.blas_manager.device_address(&mesh)?,
                    ray_mask: 0xFF,
                    diagnostic_primitives_per_group: 0,
                })
            }
            // Wait for producer buffers, then bind them with the entity's BLAS.
            InstanceSource::Geometry => {
                let buffers = geometry?;
                Some(ResolvedGeometry {
                    vertex_buffer: &buffers.vertex_buffer,
                    vertex_buffer_offset: 0,
                    previous_vertex_buffer: previous_vertices
                        .map_or(&buffers.vertex_buffer, |previous| &previous.0),
                    index_buffer: &buffers.index_buffer,
                    index_buffer_offset: 0,
                    triangle_count: buffers.index_count / 3,
                    blas_address: inputs.geometry_blas_manager.device_address(&entity)?,
                    ray_mask: buffers.ray_mask,
                    diagnostic_primitives_per_group: buffers.diagnostic_primitives_per_group,
                })
            }
        }
    }

    fn resolve_instance(
        &mut self,
        inputs: &InstanceInputs,
        lights: &mut LightState,
        entity: Entity,
        instance: &mut Instance,
        geometry: Option<&RaytracingGeometryBuffers>,
        previous_vertices: Option<&RaytracingGeometryPreviousVertices>,
        topology_generation: u64,
    ) -> bool {
        let slot = instance.slot;
        let (Some(resolved), Some(material_slot)) = (
            Self::resolve_geometry(inputs, entity, instance.source, geometry, previous_vertices),
            inputs.assets.material_slots.get(&instance.material),
        ) else {
            self.deactivate_instance(lights, entity, instance);
            return false;
        };

        let vertex_buffer_key = resolved.vertex_buffer.id();
        let previous_vertex_buffer_key = resolved.previous_vertex_buffer.id();
        let index_buffer_key = resolved.index_buffer.id();
        let capacity = MAX_MESH_SLAB_COUNT.get();
        let has_room = |state: &Self| {
            let required_vertex_slots =
                u32::from(!state.vertex_buffers.contains(&vertex_buffer_key))
                    + u32::from(
                        previous_vertex_buffer_key != vertex_buffer_key
                            && !state.vertex_buffers.contains(&previous_vertex_buffer_key),
                    );
            state.vertex_buffers.vacancies(capacity) >= required_vertex_slots
                && state.index_buffers.has_room(&index_buffer_key, capacity)
        };
        if !has_room(self) {
            // Replacing this instance's buffers may free the slots it needs.
            self.release_buffers(instance.buffers.take());
            if !has_room(self) {
                once!(warn!(
                    "Solari scene needs more than {} mesh slabs. Instances past that limit will \
                     not be rendered.",
                    MAX_MESH_SLAB_COUNT.get()
                ));
                self.deactivate_instance(lights, entity, instance);
                return false;
            }
        }

        let previous_buffers = instance.buffers.take();
        let vertex_buffer_id = self
            .vertex_buffers
            .acquire(vertex_buffer_key, capacity, || {
                resolved.vertex_buffer.clone()
            })
            .expect("vertex slab binding array had room but handed out no slot");
        let previous_vertex_buffer_id = self
            .vertex_buffers
            .acquire(previous_vertex_buffer_key, capacity, || {
                resolved.previous_vertex_buffer.clone()
            })
            .expect("vertex slab binding array had room but handed out no slot");
        let index_buffer_id = self
            .index_buffers
            .acquire(index_buffer_key, capacity, || resolved.index_buffer.clone())
            .expect("index slab binding array had room but handed out no slot");
        instance.buffers = Some((
            vertex_buffer_key,
            previous_vertex_buffer_key,
            index_buffer_key,
        ));
        self.release_buffers(previous_buffers);

        let triangle_count = resolved.triangle_count;
        let geometry = GeometryKey {
            topology_generation,
            source: instance.source,
            vertex_buffer: vertex_buffer_key,
            vertex_buffer_offset: resolved.vertex_buffer_offset,
            index_buffer: index_buffer_key,
            index_buffer_offset: resolved.index_buffer_offset,
            triangle_count,
        };
        self.update_geometry(entity, instance, geometry);
        self.geometry_ids.grow_and_set(
            slot,
            GpuInstanceGeometryIds {
                vertex_buffer_id,
                vertex_buffer_offset: resolved.vertex_buffer_offset,
                previous_vertex_buffer_id,
                index_buffer_id,
                index_buffer_offset: resolved.index_buffer_offset,
                triangle_count,
                ray_mask: u32::from(resolved.ray_mask),
                diagnostic_primitives_per_group: resolved.diagnostic_primitives_per_group,
            },
        );
        self.material_ids.grow_and_set(slot, material_slot);
        self.set_blas_ref(entity, slot, GpuBlasRef::new(resolved.blas_address, resolved.ray_mask));

        let is_emissive = inputs
            .assets
            .emissive_materials
            .contains(&instance.material);
        if is_emissive {
            lights.add_light(
                LightSourceId::EmissiveMesh(entity),
                GpuLightSource::new_emissive_mesh_light(slot, triangle_count),
            );
        } else {
            lights.remove_light(LightSourceId::EmissiveMesh(entity));
        }
        true
    }

    fn update_geometry(&mut self, entity: Entity, instance: &mut Instance, geometry: GeometryKey) {
        if instance
            .geometry
            .replace(geometry)
            .is_some_and(|previous| previous != geometry)
        {
            self.invalidated_geometry.insert(entity);
            self.liveness_changed.insert(entity);
        }
    }

    fn write_transforms(
        &self,
        slot: u32,
        transform: &GlobalTransform,
        previous_frame_transform: &PreviousGlobalTransform,
    ) {
        self.transforms.set_if_changed(
            slot,
            GpuTransform(Affine3::from(transform.affine()).to_transpose()),
        );
        self.previous_frame_transforms.set_if_changed(
            slot,
            GpuTransform(Affine3::from(previous_frame_transform.0).to_transpose()),
        );
    }

    /// Updates a slot's BLAS reference, live count, and instance translation state.
    fn set_blas_ref(&mut self, entity: Entity, slot: u32, reference: GpuBlasRef) {
        self.blas_refs.grow(slot + 1);
        let previous = self.blas_refs.get(slot);
        if previous == reference {
            return;
        }
        self.blas_refs.set(slot, reference);

        if previous == GpuBlasRef::NONE {
            self.live_count += 1;
            self.liveness_changed.insert(entity);
        } else if reference == GpuBlasRef::NONE {
            self.invalidated_geometry.insert(entity);
            self.live_count -= 1;
            self.liveness_changed.insert(entity);
        }
    }

    fn deactivate_instance(
        &mut self,
        lights: &mut LightState,
        entity: Entity,
        instance: &mut Instance,
    ) {
        self.set_blas_ref(entity, instance.slot, GpuBlasRef::NONE);
        lights.remove_light(LightSourceId::EmissiveMesh(entity));
        self.release_buffers(instance.buffers.take());
    }

    fn release_buffers(&mut self, buffers: Option<(BufferId, BufferId, BufferId)>) {
        if let Some((vertex_key, previous_vertex_key, index_key)) = buffers {
            self.vertex_buffers.release(&vertex_key);
            self.vertex_buffers.release(&previous_vertex_key);
            self.index_buffers.release(&index_key);
        }
    }

    fn remove_instance(&mut self, lights: &mut LightState, entity: Entity) {
        let Some(mut instance) = self.records.remove(&entity) else {
            return;
        };

        self.deactivate_instance(lights, entity, &mut instance);
        self.slots.release(instance.slot);
        self.pending_refresh.remove(&entity);
        if let Some(mesh) = instance.source.mesh() {
            unlink(&mut self.mesh_instances, &mesh, entity);
        }
        unlink(&mut self.material_instances, &instance.material, entity);
    }
}

impl RaytracingSceneBindings {
    /// Parallel hot path: one entity lookup, then two allocation-free sparse writes.
    pub fn move_instance(
        &self,
        entity: Entity,
        transform: &GlobalTransform,
        previous_frame_transform: &PreviousGlobalTransform,
    ) {
        if let Some(instance) = self.instances.records.get(&entity) {
            self.instances
                .write_transforms(instance.slot, transform, previous_frame_transform);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn live_instance(state: &mut InstanceState) -> (Entity, Instance) {
        let entity = Entity::from_raw_u32(1).unwrap();
        let slot = state.slots.allocate();
        let instance = Instance {
            slot,
            source: InstanceSource::Geometry,
            material: AssetId::default(),
            buffers: None,
            geometry: Some(GeometryKey {
                topology_generation: 0,
                source: InstanceSource::Geometry,
                vertex_buffer: BufferId::new(),
                vertex_buffer_offset: 0,
                index_buffer: BufferId::new(),
                index_buffer_offset: 0,
                triangle_count: 4,
            }),
        };
        state.records.insert(entity, instance);
        state.set_blas_ref(entity, slot, GpuBlasRef::new(1, 0xFF));
        state.write_instance_id_translations();
        state.begin_frame(true);
        (entity, instance)
    }

    #[test]
    fn diagnostic_geometry_metadata_matches_shader_layout() {
        let ids = GpuInstanceGeometryIds {
            vertex_buffer_id: 1,
            vertex_buffer_offset: 2,
            previous_vertex_buffer_id: 3,
            index_buffer_id: 4,
            index_buffer_offset: 5,
            triangle_count: 27,
            ray_mask: 2,
            diagnostic_primitives_per_group: 9,
        };
        assert_eq!(core::mem::size_of_val(&ids), 32);
        let words: [u32; 8] = bytemuck::cast(ids);
        assert_eq!(words, [1, 2, 3, 4, 5, 27, 2, 9]);
    }

    #[test]
    fn tlas_mask_layout_and_cpu_descriptor_agree() {
        let reference = GpuBlasRef::new(0x0123456789ABCDEF, 0x02);
        let words: [u32; 4] = bytemuck::cast(reference);
        assert_eq!(words, [0x89ABCDEF, 0x01234567, 0x02, 0]);
        let mut state = InstanceState::new();
        let (entity, instance) = live_instance(&mut state);
        state.reserve_slot(instance.slot);
        state.set_blas_ref(entity, instance.slot, reference);
        let (_, _, _, _, mask) = state.drawable().next().unwrap();
        assert_eq!(mask, 0x02);
        assert_eq!(mask & 0xFD, 0, "GI rays exclude grass-only geometry");
        assert_ne!(mask & 0xFF, 0, "shadow rays retain grass geometry");
        assert_ne!(0xFFu8 & 0xFD, 0, "GI rays retain ordinary geometry");
    }

    #[test]
    fn replaced_geometry_invalidates_anchors_until_consumed() {
        let mut state = InstanceState::new();
        let (entity, mut instance) = live_instance(&mut state);
        let geometry = GeometryKey {
            index_buffer: BufferId::new(),
            ..instance.geometry.unwrap()
        };
        state.update_geometry(entity, &mut instance, geometry);
        for _ in 0..2 {
            state.write_instance_id_translations();
            assert_eq!(
                state.previous_frame_id_translations.get(instance.slot),
                INSTANCE_NOT_PRESENT_THIS_FRAME,
            );
            state.begin_frame(false);
        }
        state.begin_frame(true);
        state.write_instance_id_translations();
        assert_eq!(
            state.previous_frame_id_translations.get(instance.slot),
            instance.slot,
        );
    }

    #[test]
    fn blas_rotation_preserves_anchors_but_reactivation_invalidates_them() {
        let mut state = InstanceState::new();
        let (entity, mut instance) = live_instance(&mut state);
        let geometry = instance.geometry.unwrap();
        state.update_geometry(entity, &mut instance, geometry);
        state.set_blas_ref(entity, instance.slot, GpuBlasRef::new(2, 0xFF));
        state.write_instance_id_translations();
        assert_eq!(
            state.previous_frame_id_translations.get(instance.slot),
            instance.slot,
        );

        state.set_blas_ref(entity, instance.slot, GpuBlasRef::NONE);
        state.set_blas_ref(entity, instance.slot, GpuBlasRef::new(3, 0xFF));
        state.write_instance_id_translations();
        assert_eq!(
            state.previous_frame_id_translations.get(instance.slot),
            INSTANCE_NOT_PRESENT_THIS_FRAME,
        );
    }

    #[test]
    fn topology_generation_invalidates_anchors_with_unchanged_buffers() {
        let mut state = InstanceState::new();
        let (entity, mut instance) = live_instance(&mut state);
        let original = instance.geometry.unwrap();
        state.update_geometry(entity, &mut instance, original);
        state.write_instance_id_translations();
        assert_eq!(
            state.previous_frame_id_translations.get(instance.slot),
            instance.slot
        );

        let changed = GeometryKey {
            topology_generation: 1,
            ..original
        };
        state.update_geometry(entity, &mut instance, changed);
        state.write_instance_id_translations();
        assert_eq!(
            state.previous_frame_id_translations.get(instance.slot),
            INSTANCE_NOT_PRESENT_THIS_FRAME
        );
        // Generation invalidation leaves geometry and TLAS instance allocation intact.
        assert_eq!(
            instance.geometry.unwrap().vertex_buffer,
            original.vertex_buffer
        );
        assert_eq!(
            instance.geometry.unwrap().index_buffer,
            original.index_buffer
        );
        assert_eq!(state.live_slot(entity), Some(instance.slot));

        state.begin_frame(true);
        state.update_geometry(entity, &mut instance, changed);
        state.write_instance_id_translations();
        assert_eq!(
            state.previous_frame_id_translations.get(instance.slot),
            instance.slot
        );
    }

    #[test]
    fn source_switch_invalidates_anchors_with_shared_buffer_locations() {
        let mut state = InstanceState::new();
        let (entity, mut instance) = live_instance(&mut state);
        let geometry = GeometryKey {
            source: InstanceSource::Mesh(AssetId::default()),
            ..instance.geometry.unwrap()
        };
        state.update_geometry(entity, &mut instance, geometry);
        state.write_instance_id_translations();
        assert_eq!(
            state.previous_frame_id_translations.get(instance.slot),
            INSTANCE_NOT_PRESENT_THIS_FRAME,
        );
    }
}
