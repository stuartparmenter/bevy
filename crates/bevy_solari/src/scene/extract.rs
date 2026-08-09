use super::{RaytracingGeometry, RaytracingGeometryBuffers, RaytracingMesh3d};
use bevy_asset::{AssetEvent, AssetId, Assets};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    change_detection::{DetectChanges, Ref},
    entity::Entity,
    lifecycle::RemovedComponents,
    message::MessageReader,
    query::{Added, Changed, Or, With, Without},
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_mesh::Mesh3d;
use bevy_pbr::{MeshMaterial3d, PreviousGlobalTransform, StandardMaterial};
use bevy_platform::collections::HashMap;
use bevy_render::{sync_world::RenderEntity, Extract};
use bevy_transform::components::GlobalTransform;

/// Filter matching both kinds of raytracing instance.
type RaytracingInstanceFilter = Or<(With<RaytracingMesh3d>, With<RaytracingGeometry>)>;

/// Maintains [`PreviousGlobalTransform`] for raytracing instances without a
/// [`Mesh3d`]: `bevy_pbr` only maintains it for rasterized meshes, but solari
/// needs last frame's transform for motion vectors and temporal reuse on
/// raytracing-only entities (notably [`RaytracingGeometry`]).
///
/// Runs in `PreUpdate` alongside `update_mesh_previous_global_transforms`:
/// at the start of each frame, last frame's [`GlobalTransform`] becomes this
/// frame's [`PreviousGlobalTransform`].
pub fn update_raytracing_previous_global_transforms(
    mut commands: Commands,
    new_instances: Query<
        (Entity, &GlobalTransform),
        (
            RaytracingInstanceFilter,
            Without<Mesh3d>,
            Without<PreviousGlobalTransform>,
        ),
    >,
    mut instances: Query<
        (Ref<GlobalTransform>, &mut PreviousGlobalTransform),
        (RaytracingInstanceFilter, Without<Mesh3d>),
    >,
) {
    for (entity, transform) in &new_instances {
        commands
            .entity(entity)
            .try_insert(PreviousGlobalTransform(transform.affine()));
    }
    for (transform, mut previous) in &mut instances {
        if transform.is_changed_after(previous.last_changed()) {
            *previous = PreviousGlobalTransform(transform.affine());
        }
    }
}

/// Creates or removes components in the render world related to raytracing instances.
pub fn extract_raytracing_scene_structural(
    new_instances: Extract<
        Query<
            (
                RenderEntity,
                &RaytracingMesh3d,
                &MeshMaterial3d<StandardMaterial>,
                &GlobalTransform,
            ),
            Added<RaytracingMesh3d>,
        >,
    >,
    new_geometry_instances: Extract<
        Query<
            (RenderEntity, &MeshMaterial3d<StandardMaterial>, &GlobalTransform),
            Added<RaytracingGeometry>,
        >,
    >,
    mut removed_raytracing_meshes: Extract<RemovedComponents<RaytracingMesh3d>>,
    mut removed_raytracing_geometry: Extract<RemovedComponents<RaytracingGeometry>>,
    render_entities: Extract<Query<RenderEntity>>,
    mut commands: Commands,
) {
    // Process removed components before additions, that way it properly handles same-frame removal->insertion.
    for main_entity in removed_raytracing_meshes.read() {
        if let Ok(render_entity) = render_entities.get(main_entity) {
            commands.entity(render_entity).remove::<RaytracingMesh3d>();
        }
    }

    for main_entity in removed_raytracing_geometry.read() {
        if let Ok(render_entity) = render_entities.get(main_entity) {
            // Also drop the producer-inserted buffers, so re-adding the
            // marker later can't resurrect stale geometry.
            commands
                .entity(render_entity)
                .remove::<(RaytracingGeometry, RaytracingGeometryBuffers)>();
        }
    }

    // Both paths seed a zero-motion previous transform rather than reading
    // the main world's: a newly appearing instance was absent from last
    // frame's TLAS, and a main-world `PreviousGlobalTransform` can hold a
    // stale value from before the component was (re-)added. The `PreUpdate`
    // maintainers take over from the next frame.
    for (render_entity, mesh, material, transform) in &new_instances {
        commands.entity(render_entity).insert((
            mesh.clone(),
            material.clone(),
            *transform,
            PreviousGlobalTransform(transform.affine()),
        ));
    }

    // GPU-authored geometry: extract only the marker + material + transform.
    // The vertex/index buffers live in `RaytracingGeometryBuffers`, inserted
    // separately on the render entity by the producer. Transform and material
    // updates ride the retained update systems below, like mesh instances.
    for (render_entity, material, transform) in &new_geometry_instances {
        commands.entity(render_entity).insert((
            RaytracingGeometry,
            material.clone(),
            *transform,
            PreviousGlobalTransform(transform.affine()),
        ));
    }
}

/// Updates the transforms of existing raytracing instances in the render world.
pub fn extract_raytracing_scene_transforms(
    main_instances: Extract<
        Query<
            (
                RenderEntity,
                &GlobalTransform,
                Option<&PreviousGlobalTransform>,
            ),
            (
                Or<(Changed<GlobalTransform>, Changed<PreviousGlobalTransform>)>,
                RaytracingInstanceFilter,
            ),
        >,
    >,
    mut render_instances: Query<
        (&mut GlobalTransform, Option<&mut PreviousGlobalTransform>),
        RaytracingInstanceFilter,
    >,
) {
    for (render_entity, new_transform, new_previous_frame_transform) in &main_instances {
        if let Ok((mut transform, mut previous_frame_transform)) =
            render_instances.get_mut(render_entity)
        {
            *transform = *new_transform;

            if let Some(previous_frame_transform) = previous_frame_transform.as_deref_mut() {
                *previous_frame_transform = new_previous_frame_transform
                    .cloned()
                    .unwrap_or(PreviousGlobalTransform(new_transform.affine()));
            }
        }
    }
}

/// Updates the mesh and material of existing raytracing instances in the render world.
pub fn extract_raytracing_scene_meshes_and_materials(
    main_instances: Extract<
        Query<
            (
                RenderEntity,
                Option<&RaytracingMesh3d>,
                &MeshMaterial3d<StandardMaterial>,
            ),
            (
                Or<(
                    Changed<RaytracingMesh3d>,
                    Changed<MeshMaterial3d<StandardMaterial>>,
                )>,
                RaytracingInstanceFilter,
            ),
        >,
    >,
    mut render_instances: Query<
        (
            Option<&mut RaytracingMesh3d>,
            &mut MeshMaterial3d<StandardMaterial>,
        ),
        RaytracingInstanceFilter,
    >,
) {
    for (render_entity, new_mesh, new_material) in &main_instances {
        if let Ok((mesh, mut material)) = render_instances.get_mut(render_entity) {
            if let (Some(mut mesh), Some(new_mesh)) = (mesh, new_mesh) {
                *mesh = new_mesh.clone();
            }
            *material = new_material.clone();
        }
    }
}

#[derive(Resource, Deref, DerefMut, Default)]
pub struct StandardMaterialAssets(HashMap<AssetId<StandardMaterial>, StandardMaterial>);

/// Keeps [`StandardMaterialAssets`] up to date in the render world.
pub fn extract_raytracing_material_assets(
    main_materials: Extract<Res<Assets<StandardMaterial>>>,
    mut render_materials: ResMut<StandardMaterialAssets>,
    mut events: Extract<MessageReader<AssetEvent<StandardMaterial>>>,
) {
    for event in events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                if let Some(material) = main_materials.get(*id) {
                    render_materials.insert(*id, material.clone());
                }
            }
            AssetEvent::Removed { id } => {
                render_materials.remove(id);
            }
            AssetEvent::Unused { .. } | AssetEvent::LoadedWithDependencies { .. } => {}
        }
    }
}
