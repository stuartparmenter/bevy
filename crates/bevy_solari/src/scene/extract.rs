use super::{
    environment::ExtractedSolariEnvironmentMap, RaytracingMesh3d, RaytracingSceneBindings,
};
use crate::{pathtracer::Pathtracer, realtime::SolariLighting};
use bevy_asset::{AssetEvent, AssetId, Assets};
use bevy_camera::Camera;
use bevy_ecs::{
    entity::Entity,
    lifecycle::RemovedComponents,
    message::MessageReader,
    query::{Added, Changed, Has, Or, With},
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_light::{EnvironmentMapLight, GeneratedEnvironmentMapLight};
use bevy_pbr::{MeshGeometryError, MeshMaterial3d, PreviousGlobalTransform, StandardMaterial};
use bevy_platform::collections::HashMap;
use bevy_render::{sync_world::RenderEntity, Extract};
use bevy_transform::components::GlobalTransform;

/// Creates or removes components in the render world related to raytracing instances.
pub fn extract_raytracing_scene_structural(
    new_instances: Extract<
        Query<
            (
                RenderEntity,
                &RaytracingMesh3d,
                &MeshGeometryError,
                &MeshMaterial3d<StandardMaterial>,
                &GlobalTransform,
                Option<&PreviousGlobalTransform>,
            ),
            Added<RaytracingMesh3d>,
        >,
    >,
    mut removed_raytracing_meshes: Extract<RemovedComponents<RaytracingMesh3d>>,
    render_entities: Extract<Query<RenderEntity>>,
    mut commands: Commands,
) {
    // Process removed components before additions, that way it properly handles same-frame removal->insertion
    for main_entity in removed_raytracing_meshes.read() {
        if let Ok(render_entity) = render_entities.get(main_entity) {
            commands.entity(render_entity).remove::<RaytracingMesh3d>();
        }
    }

    for (render_entity, mesh, geometry_error, material, transform, previous_frame_transform) in
        &new_instances
    {
        commands.entity(render_entity).insert((
            mesh.clone(),
            *geometry_error,
            material.clone(),
            *transform,
            previous_frame_transform
                .cloned()
                .unwrap_or(PreviousGlobalTransform(transform.affine())),
        ));
    }
}

/// Copies the transforms of moved raytracing instances from the main world
/// straight into their GPU buffers.
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
                With<RaytracingMesh3d>,
            ),
        >,
    >,
    bindings: Res<RaytracingSceneBindings>,
) {
    main_instances
        .par_iter()
        .for_each(|(render_entity, transform, previous_frame_transform)| {
            let previous_frame_transform = previous_frame_transform
                .cloned()
                .unwrap_or(PreviousGlobalTransform(transform.affine()));

            bindings.move_instance(render_entity, transform, &previous_frame_transform);
        });
}

/// Updates the mesh, material and geometry error of existing raytracing instances in the render
/// world.
pub fn extract_raytracing_scene_meshes_and_materials(
    main_instances: Extract<
        Query<
            (
                RenderEntity,
                &RaytracingMesh3d,
                &MeshMaterial3d<StandardMaterial>,
                &MeshGeometryError,
            ),
            Or<(
                Changed<RaytracingMesh3d>,
                Changed<MeshMaterial3d<StandardMaterial>>,
                Changed<MeshGeometryError>,
            )>,
        >,
    >,
    mut render_instances: Query<(
        &mut RaytracingMesh3d,
        &mut MeshMaterial3d<StandardMaterial>,
        &mut MeshGeometryError,
    )>,
) {
    for (render_entity, new_mesh, new_material, new_geometry_error) in &main_instances {
        if let Ok((mut mesh, mut material, mut geometry_error)) =
            render_instances.get_mut(render_entity)
        {
            *mesh = new_mesh.clone();
            *material = new_material.clone();
            *geometry_error = *new_geometry_error;
        }
    }
}

/// Mirrors the `EnvironmentMapLight` of every active Solari camera into the render world.
///
/// Only cameras whose relevant state changed are revisited. The mirrored component is present
/// exactly while the camera is active, has `SolariLighting` or `Pathtracer`, and has an
/// `EnvironmentMapLight`; losing any of those removes it, so the binder can never pick a stale one.
///
/// A `GeneratedEnvironmentMapLight` on the camera (what `AtmosphereEnvironmentMapLight` turns
/// into) marks the map as regenerated every frame: its `EnvironmentMapLight` holds placeholder
/// images whose GPU contents `bevy_pbr`'s generation nodes rewrite each frame, so the importance
/// pyramid has to follow. Both systems that insert those components run in `Update`, so the flag
/// is already correct on the frame the `EnvironmentMapLight` first appears.
pub fn extract_solari_environment_maps(
    changed_cameras: Extract<
        Query<
            Entity,
            (
                With<Camera>,
                Or<(
                    Changed<EnvironmentMapLight>,
                    Changed<GeneratedEnvironmentMapLight>,
                    Changed<Camera>,
                    Added<SolariLighting>,
                    Added<Pathtracer>,
                )>,
            ),
        >,
    >,
    cameras: Extract<
        Query<(
            RenderEntity,
            &Camera,
            Option<&EnvironmentMapLight>,
            Has<GeneratedEnvironmentMapLight>,
            Has<SolariLighting>,
            Has<Pathtracer>,
        )>,
    >,
    mut removed_environment_maps: Extract<RemovedComponents<EnvironmentMapLight>>,
    mut removed_generated_maps: Extract<RemovedComponents<GeneratedEnvironmentMapLight>>,
    mut removed_lighting: Extract<RemovedComponents<SolariLighting>>,
    mut removed_pathtracers: Extract<RemovedComponents<Pathtracer>>,
    mut commands: Commands,
) {
    let removed = removed_environment_maps
        .read()
        .chain(removed_generated_maps.read())
        .chain(removed_lighting.read())
        .chain(removed_pathtracers.read());
    for main_entity in changed_cameras.iter().chain(removed) {
        let Ok((
            render_entity,
            camera,
            environment_map,
            is_generated,
            has_lighting,
            has_pathtracer,
        )) = cameras.get(main_entity)
        else {
            continue;
        };
        let Ok(mut entity_commands) = commands.get_entity(render_entity) else {
            continue;
        };
        match environment_map {
            Some(environment_map) if camera.is_active && (has_lighting || has_pathtracer) => {
                entity_commands.insert(ExtractedSolariEnvironmentMap {
                    specular_map: environment_map.specular_map.id(),
                    intensity: environment_map.intensity,
                    rotation: environment_map.rotation,
                    contents_change_every_frame: is_generated,
                });
            }
            _ => {
                entity_commands.remove::<ExtractedSolariEnvironmentMap>();
            }
        }
    }
}

/// The set of [`StandardMaterial`] in the scene, mirrored into the render world.
#[derive(Resource, Default)]
pub struct StandardMaterialAssets {
    materials: HashMap<AssetId<StandardMaterial>, StandardMaterial>,
    /// Materials added or modified this frame.
    pub changed: Vec<AssetId<StandardMaterial>>,
    /// Materials removed this frame.
    pub removed: Vec<AssetId<StandardMaterial>>,
}

impl StandardMaterialAssets {
    pub fn get(&self, id: &AssetId<StandardMaterial>) -> Option<&StandardMaterial> {
        self.materials.get(id)
    }
}

/// Keeps [`StandardMaterialAssets`] up to date in the render world.
pub fn extract_raytracing_material_assets(
    main_materials: Extract<Res<Assets<StandardMaterial>>>,
    mut render_materials: ResMut<StandardMaterialAssets>,
    mut events: Extract<MessageReader<AssetEvent<StandardMaterial>>>,
) {
    let render_materials = &mut *render_materials;

    render_materials.changed.clear();
    render_materials.removed.clear();

    for event in events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                if let Some(material) = main_materials.get(*id) {
                    render_materials.materials.insert(*id, material.clone());
                    render_materials.changed.push(*id);
                }
            }
            AssetEvent::Removed { id } => {
                render_materials.materials.remove(id);
                render_materials.removed.push(*id);
            }
            AssetEvent::Unused { .. } | AssetEvent::LoadedWithDependencies { .. } => {}
        }
    }
}
