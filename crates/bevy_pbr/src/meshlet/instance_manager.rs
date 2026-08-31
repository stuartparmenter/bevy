use super::{
    meshlet_mesh_manager::{MeshletGpuDescriptor, MeshletMeshManager},
    MeshletMesh, MeshletMesh3d,
};
use crate::{
    MeshFlags, MeshGeometryError, MeshTransforms, MeshUniform, PreviousGlobalTransform,
    RenderMaterialInstances,
};
use bevy_asset::{AssetEvent, AssetServer, Assets, UntypedAssetId};
use bevy_camera::visibility::RenderLayers;
use bevy_ecs::{
    change_detection::DetectChanges,
    entity::{Entities, Entity, EntityHashMap},
    lifecycle::{RemovedComponents, RemovedIter},
    message::MessageReader,
    query::{Changed, Has, Or, With},
    resource::Resource,
    system::{Local, Query, Res, ResMut, SystemState},
};
use bevy_light::{NotShadowCaster, NotShadowReceiver};
#[cfg(debug_assertions)]
use bevy_math::Vec4;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    material_bind_groups::{MaterialBindingId, RenderMaterialBindings},
    render_resource::StorageBuffer,
    renderer::RenderDevice,
    sync_world::MainEntity,
    MainWorld,
};
use bevy_transform::components::GlobalTransform;
use core::ops::DerefMut;
use tracing::error;

/// Manages data for each entity with a [`MeshletMesh`].
#[derive(Resource)]
pub struct InstanceManager {
    /// Amount of instances in the scene.
    pub scene_instance_count: u32,
    /// The max BVH depth of any instance in the scene. This is used to control the number of
    /// dependent dispatches emitted for BVH traversal.
    pub max_bvh_depth: u32,

    /// Per-instance [`MainEntity`], [`RenderLayers`], and [`NotShadowCaster`].
    pub instances: Vec<(MainEntity, RenderLayers, bool)>,
    /// Per-instance [`MeshUniform`].
    ///
    /// This and the three buffers below are written only by [`Self::add_instance`] and
    /// `queue_material_meshlet_meshes`. Any other writer must set [`Self::instance_data_dirty`],
    /// or its write never reaches the GPU.
    pub instance_uniforms: StorageBuffer<Vec<MeshUniform>>,
    /// Per-instance slot in [`MeshletMeshManager::asset_aabbs`].
    pub instance_asset_indices: StorageBuffer<Vec<u32>>,
    /// Per-instance material ID.
    pub instance_material_ids: StorageBuffer<Vec<u32>>,
    /// Per-instance page and asset-local section bases in the paged meshlet data heap.
    ///
    /// Kept per instance, unlike the AABB, even though the descriptor is also a property of the
    /// asset: `resolve_vertex_output` indexes it per pixel from every material and prepass fragment
    /// shader, where an extra dependent load would sit ahead of the descriptor read it feeds.
    pub instance_meshlet_descriptors: StorageBuffer<Vec<MeshletGpuDescriptor>>,
    /// Per-view per-instance visibility bit. Used for [`RenderLayers`] and [`NotShadowCaster`] support.
    pub view_instance_visibility: EntityHashMap<StorageBuffer<Vec<u32>>>,

    /// The material assets used by instances in the scene.
    ///
    /// Collected during extraction, where the material of every instance is looked up anyway, so
    /// that the prepare systems never walk the whole scene's material instance list.
    scene_material_assets: HashSet<UntypedAssetId>,
    /// Next material ID available.
    next_material_id: u32,
    /// Map of material asset to material ID.
    material_id_lookup: HashMap<UntypedAssetId, u32>,
    /// Set of material IDs used in the scene.
    material_ids_present_in_scene: HashSet<u32>,

    /// Whether the per-instance buffers have changed and must be re-uploaded.
    ///
    /// Set by `extract_meshlet_mesh_entities` and `queue_material_meshlet_meshes`, cleared by
    /// `prepare_meshlet_per_frame_resources` once it has uploaded.
    pub instance_data_dirty: bool,
    /// Whether an instance was skipped for an asset that may yet upload.
    instances_awaiting_assets: bool,
    /// Residency revision the current instance data was built from.
    ///
    /// Starts past any real revision so that the first extraction rebuilds.
    built_residency_revision: u64,
    /// `next_material_id` when `instance_material_ids` was last written.
    material_ids_written_at: u32,
    /// Frames skipped since the debug rebuild audit last ran.
    #[cfg(debug_assertions)]
    skipped_frames: u32,
}

impl InstanceManager {
    pub fn new() -> Self {
        Self {
            scene_instance_count: 0,
            max_bvh_depth: 0,

            instances: Vec::new(),
            instance_uniforms: {
                let mut buffer = StorageBuffer::default();
                buffer.set_label(Some("meshlet_instance_uniforms"));
                buffer
            },
            instance_asset_indices: {
                let mut buffer = StorageBuffer::default();
                buffer.set_label(Some("meshlet_instance_asset_indices"));
                buffer
            },
            instance_material_ids: {
                let mut buffer = StorageBuffer::default();
                buffer.set_label(Some("meshlet_instance_material_ids"));
                buffer
            },
            instance_meshlet_descriptors: {
                let mut buffer = StorageBuffer::default();
                buffer.set_label(Some("meshlet_instance_descriptors"));
                buffer
            },
            view_instance_visibility: EntityHashMap::default(),

            scene_material_assets: HashSet::default(),
            next_material_id: 0,
            material_id_lookup: HashMap::default(),
            material_ids_present_in_scene: HashSet::default(),

            instance_data_dirty: true,
            instances_awaiting_assets: false,
            built_residency_revision: u64::MAX,
            material_ids_written_at: 0,
            #[cfg(debug_assertions)]
            skipped_frames: 0,
        }
    }

    pub fn add_instance(
        &mut self,
        instance: MainEntity,
        meshlet_descriptor: MeshletGpuDescriptor,
        asset_index: u32,
        bvh_depth: u32,
        transform: &GlobalTransform,
        previous_transform: Option<&PreviousGlobalTransform>,
        render_layers: Option<&RenderLayers>,
        mesh_material_ids: &RenderMaterialInstances,
        render_material_bindings: &RenderMaterialBindings,
        not_shadow_receiver: bool,
        not_shadow_caster: bool,
        geometry_error: Option<&MeshGeometryError>,
    ) {
        // Build a MeshUniform for the instance
        let transform = transform.affine();
        let previous_transform = previous_transform.map(|t| t.0).unwrap_or(transform);
        let mut flags = if not_shadow_receiver {
            MeshFlags::empty()
        } else {
            MeshFlags::SHADOW_RECEIVER
        };
        if transform.matrix3.determinant().is_sign_positive() {
            flags |= MeshFlags::SIGN_DETERMINANT_MODEL_3X3;
        }
        flags |= MeshFlags::from_geometry_error(geometry_error, &transform);
        let transforms = MeshTransforms {
            world_from_local: transform.into(),
            previous_world_from_local: previous_transform.into(),
            flags: flags.bits(),
        };

        let mesh_material = mesh_material_ids.mesh_material(instance);
        let mesh_material_binding_id = if let Some(mesh_material) = mesh_material {
            self.scene_material_assets.insert(mesh_material);
            render_material_bindings
                .get(&mesh_material)
                .cloned()
                .unwrap_or_default()
        } else {
            // Use a dummy binding ID if the mesh has no material
            MaterialBindingId::default()
        };

        let mesh_uniform = MeshUniform::new(
            &transforms,
            0,
            mesh_material_binding_id.slot,
            None,
            None,
            None,
            None,
            None,
        );

        // Append instance data
        self.instances.push((
            instance,
            render_layers.cloned().unwrap_or(RenderLayers::default()),
            not_shadow_caster,
        ));
        self.instance_uniforms.get_mut().push(mesh_uniform);
        self.instance_asset_indices.get_mut().push(asset_index);
        self.instance_material_ids.get_mut().push(0);
        self.instance_meshlet_descriptors
            .get_mut()
            .push(meshlet_descriptor);

        self.scene_instance_count += 1;
        self.max_bvh_depth = self.max_bvh_depth.max(bvh_depth);
    }

    /// Get the material ID for a [`crate::Material`].
    ///
    /// Ids run from 1, with 0 reserved for an instance whose material is not yet known. The
    /// shaders round-trip the id through a `Depth16Unorm` target as `f32(id) / 65535.0`
    /// (`resolve_render_targets.wesl`), so ids past 65535 alias onto other materials.
    pub fn get_material_id(&mut self, material_asset_id: UntypedAssetId) -> u32 {
        *self
            .material_id_lookup
            .entry(material_asset_id)
            .or_insert_with(|| {
                self.next_material_id += 1;
                if self.next_material_id == u32::from(u16::MAX) + 1 {
                    error!(
                        "More than {} meshlet materials are visible. Materials past that shade \
                         as whichever material the id aliases onto.",
                        u16::MAX
                    );
                }
                self.next_material_id
            })
    }

    /// The material assets used by instances in the scene, deduplicated.
    pub fn scene_material_assets(&self) -> &HashSet<UntypedAssetId> {
        &self.scene_material_assets
    }

    pub fn material_present_in_scene(&self, material_id: &u32) -> bool {
        self.material_ids_present_in_scene.contains(material_id)
    }

    /// Drop the per-view visibility bitsets, which are rebuilt for every view every frame.
    ///
    /// Separate from [`Self::reset_instances`] because a view's [`RenderLayers`] changes
    /// independently of the instances, so this runs even on a frame that keeps the instance data.
    fn reset_view_visibility(&mut self, entities: &Entities) {
        self.view_instance_visibility
            .retain(|view_entity, _| entities.contains(*view_entity));
        self.view_instance_visibility
            .values_mut()
            .for_each(|b| b.get_mut().clear());
    }

    /// Drop everything derived from the scene's instances, ready for a full rebuild.
    ///
    /// Material ids are reassigned from zero by the rebuild that follows, so this must not run
    /// without one: the ids baked into `instance_material_ids` and into the material depth target
    /// would outlive the lookup that produced them.
    fn reset_instances(&mut self) {
        self.scene_instance_count = 0;
        self.max_bvh_depth = 0;

        self.instances.clear();
        self.instance_uniforms.get_mut().clear();
        self.instance_asset_indices.get_mut().clear();
        self.instance_material_ids.get_mut().clear();
        self.instance_meshlet_descriptors.get_mut().clear();

        self.scene_material_assets.clear();
        self.instances_awaiting_assets = false;
        self.next_material_id = 0;
        self.material_id_lookup.clear();
        self.material_ids_present_in_scene.clear();
        // Past anything `next_material_id` can reach, so the ids are rewritten after a rebuild.
        self.material_ids_written_at = u32::MAX;
    }

    /// Whether the assets the instance data was built from have moved under it.
    ///
    /// An instance skipped for an asset that has not finished uploading has to be retried, and the
    /// walk itself is what queues that upload - so waiting on the revision alone would deadlock.
    fn asset_residency_changed(&self, residency_revision: u64) -> bool {
        self.instances_awaiting_assets || residency_revision != self.built_residency_revision
    }
}

/// A rebuild's output, for the debug audit to compare a skipped frame against.
///
/// Material ids are left out: they are written later by `queue_material_meshlet_meshes`, so a
/// freshly rebuilt frame always has them at the 0 sentinel.
#[cfg(debug_assertions)]
#[derive(PartialEq)]
struct InstanceSnapshot {
    max_bvh_depth: u32,
    // MeshUniform is not PartialEq; these are the fields extraction derives per instance.
    uniforms: Vec<([Vec4; 3], [Vec4; 3], u32, u32)>,
    asset_indices: Vec<u32>,
    descriptors: Vec<MeshletGpuDescriptor>,
}

#[cfg(debug_assertions)]
impl InstanceManager {
    /// How often the audit forces a rebuild it expects to be redundant.
    const AUDIT_INTERVAL: u32 = 600;

    /// Whether to rebuild a frame the gate was about to skip, to check that skipping was right.
    ///
    /// The gate's failure mode is silent - stale geometry, not a crash - so a debug build
    /// periodically rebuilds anyway and asserts nothing moved. A persistently missed invalidation
    /// source then fails within `AUDIT_INTERVAL` skipped frames; one that misses only
    /// intermittently may still slip through.
    fn audit_due(&mut self) -> bool {
        self.skipped_frames = self.skipped_frames.wrapping_add(1);
        self.skipped_frames.is_multiple_of(Self::AUDIT_INTERVAL)
    }

    fn snapshot(&self) -> InstanceSnapshot {
        InstanceSnapshot {
            max_bvh_depth: self.max_bvh_depth,
            uniforms: self
                .instance_uniforms
                .get()
                .iter()
                .map(|uniform| {
                    (
                        uniform.world_from_local,
                        uniform.previous_world_from_local,
                        uniform.flags,
                        uniform.material_and_lightmap_bind_group_slot,
                    )
                })
                .collect(),
            asset_indices: self.instance_asset_indices.get().clone(),
            descriptors: self.instance_meshlet_descriptors.get().clone(),
        }
    }
}

pub fn extract_meshlet_mesh_entities(
    mut meshlet_mesh_manager: ResMut<MeshletMeshManager>,
    mut instance_manager: ResMut<InstanceManager>,
    // TODO: Replace main_world and system_state when Extract<ResMut<Assets<MeshletMesh>>> is possible
    mut main_world: ResMut<MainWorld>,
    mesh_material_ids: Res<RenderMaterialInstances>,
    render_material_bindings: Res<RenderMaterialBindings>,
    render_device: Res<RenderDevice>,
    mut system_state: Local<
        Option<
            SystemState<(
                Query<(
                    Entity,
                    &MeshletMesh3d,
                    &GlobalTransform,
                    Option<&PreviousGlobalTransform>,
                    Option<&RenderLayers>,
                    Has<NotShadowReceiver>,
                    Has<NotShadowCaster>,
                    Option<&MeshGeometryError>,
                )>,
                // Whether any instance changed. `Changed` covers insertion, so a spawn lands here
                // too; only removals need their own readers.
                Query<
                    (),
                    (
                        With<MeshletMesh3d>,
                        Or<(
                            Changed<GlobalTransform>,
                            Changed<PreviousGlobalTransform>,
                            Changed<MeshletMesh3d>,
                            Changed<RenderLayers>,
                            Changed<NotShadowReceiver>,
                            Changed<NotShadowCaster>,
                            Changed<MeshGeometryError>,
                        )>,
                    ),
                >,
                // A despawn emits a removal for every component, so the unfiltered
                // `MeshletMesh3d` reader alone catches it; the rest are filtered to meshlet
                // entities so an unrelated entity losing a transform costs nothing.
                (
                    RemovedComponents<MeshletMesh3d>,
                    RemovedComponents<GlobalTransform>,
                    RemovedComponents<PreviousGlobalTransform>,
                    RemovedComponents<RenderLayers>,
                    RemovedComponents<NotShadowReceiver>,
                    RemovedComponents<NotShadowCaster>,
                    RemovedComponents<MeshGeometryError>,
                ),
                Query<(), With<MeshletMesh3d>>,
                Res<AssetServer>,
                ResMut<Assets<MeshletMesh>>,
                MessageReader<AssetEvent<MeshletMesh>>,
            )>,
        >,
    >,
    render_entities: &Entities,
) {
    // Get instances query
    if system_state.is_none() {
        *system_state = Some(SystemState::new(&mut main_world));
    }
    let system_state = system_state.as_mut().unwrap();
    let (
        instances_query,
        changed_instances,
        mut removed,
        meshlet_entities,
        asset_server,
        mut assets,
        mut asset_events,
    ) = system_state.get_mut(&mut main_world).unwrap();

    instance_manager.reset_view_visibility(render_entities);

    // Free GPU buffer space for any modified or dropped MeshletMesh assets. This has to precede
    // both the rebuild test and the instance walk: it bumps the residency revision the test reads,
    // and the walk treats residency as proof that an asset is loaded.
    for asset_event in asset_events.read() {
        if let AssetEvent::Unused { id } | AssetEvent::Modified { id } = asset_event {
            meshlet_mesh_manager.remove(id);
        }
    }

    // Every reader drains whether or not it decides the outcome, because an unread removal is lost
    // rather than deferred. Testing membership stops once one removal has counted: tearing a whole
    // scene down otherwise costs a query lookup per component per entity to reach the same answer.
    let mut removed_instance = removed.0.read().count() > 0;
    let drain = |removed: RemovedIter, removed_instance: &mut bool| {
        for entity in removed {
            *removed_instance |= !*removed_instance && meshlet_entities.contains(entity);
        }
    };
    drain(removed.1.read(), &mut removed_instance);
    drain(removed.2.read(), &mut removed_instance);
    drain(removed.3.read(), &mut removed_instance);
    drain(removed.4.read(), &mut removed_instance);
    drain(removed.5.read(), &mut removed_instance);
    drain(removed.6.read(), &mut removed_instance);

    // Ordered cheapest first: the change-filtered query walks every meshlet entity, while the
    // rest are a handful of integer and tick comparisons.
    let rebuild = instance_manager
        .asset_residency_changed(meshlet_mesh_manager.residency_revision())
        // An entity changing material, or a material moving bind group slot - the slot is baked
        // into every MeshUniform.
        || mesh_material_ids.is_changed()
        || render_material_bindings.is_changed()
        || removed_instance
        || !changed_instances.is_empty();
    #[cfg(debug_assertions)]
    if !rebuild && !instance_manager.audit_due() {
        return;
    }
    #[cfg(not(debug_assertions))]
    if !rebuild {
        return;
    }

    // Past here `!rebuild` means the audit forced this rebuild to check that skipping was right.
    #[cfg(debug_assertions)]
    let audited = (!rebuild).then(|| instance_manager.snapshot());

    instance_manager.reset_instances();

    // Iterate over every instance
    for (
        instance,
        meshlet_mesh,
        transform,
        previous_transform,
        render_layers,
        not_shadow_receiver,
        not_shadow_caster,
        geometry_error,
    ) in &instances_query
    {
        // Upload the instance's MeshletMesh asset data if not done already. The load check is a
        // closure because the manager only needs it for an asset it has not already uploaded, and
        // it costs two AssetServer lock acquisitions.
        let Some((meshlet_descriptor, asset_index, bvh_depth)) = meshlet_mesh_manager
            .queue_upload_if_needed(
                meshlet_mesh.id(),
                || {
                    !asset_server.is_managed(meshlet_mesh.id())
                        || asset_server.is_loaded_with_dependencies(meshlet_mesh.id())
                },
                &mut assets,
                &render_device,
            )
        else {
            // The asset is still loading, unusable, or the heap is full. The manager logs why,
            // once. Only a retryable failure keeps the rebuild open; a permanent one would hold it
            // open for the life of the process.
            instance_manager.instances_awaiting_assets |=
                meshlet_mesh_manager.upload_may_retry(meshlet_mesh.id());
            continue;
        };

        // Add the instance's data to the instance manager
        instance_manager.add_instance(
            instance.into(),
            meshlet_descriptor,
            asset_index,
            bvh_depth,
            transform,
            previous_transform,
            render_layers,
            &mesh_material_ids,
            &render_material_bindings,
            not_shadow_receiver,
            not_shadow_caster,
            geometry_error,
        );
    }

    #[cfg(debug_assertions)]
    if let Some(before) = audited {
        assert!(
            instance_manager.snapshot() == before,
            "meshlet instance extraction skipped a frame that changed the instance data - some \
             invalidation source is not covered by the rebuild test",
        );
    }

    // Read after the walk: uploading an asset bumps the residency revision.
    instance_manager.built_residency_revision = meshlet_mesh_manager.residency_revision();
    instance_manager.instance_data_dirty = true;
}

/// For each entity in the scene, record what material ID its material was assigned in the `prepare_material_meshlet_meshes` systems,
/// and note that the material is used by at least one entity in the scene.
pub fn queue_material_meshlet_meshes(
    mut instance_manager: ResMut<InstanceManager>,
    render_material_instances: Res<RenderMaterialInstances>,
) {
    // A rebuild parks the watermark past anything `next_material_id` can reach, so this covers
    // both a rebuilt instance list and a material that became drawable on a skipped frame. Both
    // prepare systems mint their ids before this runs.
    if instance_manager.next_material_id == instance_manager.material_ids_written_at {
        return;
    }

    let instance_manager = instance_manager.deref_mut();
    instance_manager.material_ids_written_at = instance_manager.next_material_id;
    instance_manager.instance_data_dirty = true;

    for (i, (instance, _, _)) in instance_manager.instances.iter().enumerate() {
        if let Some(material_instance) = render_material_instances.instances.get(instance)
            && let Some(material_id) = instance_manager
                .material_id_lookup
                .get(&material_instance.asset_id)
        {
            instance_manager
                .material_ids_present_in_scene
                .insert(*material_id);
            instance_manager.instance_material_ids.get_mut()[i] = *material_id;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_extraction_rebuilds() {
        // The built revision starts past anything the mesh manager can report.
        assert!(InstanceManager::new().asset_residency_changed(0));
    }

    #[test]
    fn a_new_or_dropped_asset_rebuilds() {
        let mut manager = InstanceManager::new();
        manager.built_residency_revision = 7;

        assert!(!manager.asset_residency_changed(7));
        assert!(manager.asset_residency_changed(8));
    }

    #[test]
    fn an_asset_that_may_still_upload_rebuilds() {
        let mut manager = InstanceManager::new();
        manager.built_residency_revision = 7;
        manager.instances_awaiting_assets = true;

        // The walk is what queues the upload, so waiting on the revision alone would deadlock.
        assert!(manager.asset_residency_changed(7));
    }

    #[test]
    fn a_rebuild_forces_the_material_ids_to_be_rewritten() {
        let mut manager = InstanceManager::new();
        manager.material_ids_written_at = manager.next_material_id;

        manager.reset_instances();

        assert_ne!(manager.next_material_id, manager.material_ids_written_at);
    }
}
