use super::{
    RaytracingGeometry, RaytracingGeometryBuffers, RaytracingGeometryPreviousVertices,
    RaytracingMesh3d, RaytracingSceneBindings,
};
use bevy_asset::{AssetEvent, AssetId, Assets, Handle};
use bevy_camera::Camera;
use bevy_ecs::{
    change_detection::{DetectChanges, Ref},
    entity::Entity,
    lifecycle::RemovedComponents,
    message::MessageReader,
    query::{Added, Changed, Or, With, Without},
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_image::Image;
use bevy_light::{EnvironmentMapLight, GeneratedEnvironmentMapLight};
use bevy_math::Quat;
use bevy_mesh::Mesh3d;
use bevy_pbr::{MeshMaterial3d, PreviousGlobalTransform, StandardMaterial};
use bevy_platform::collections::HashMap;
use bevy_render::{sync_world::RenderEntity, Extract};
use bevy_transform::components::GlobalTransform;
use bevy_utils::once;
use tracing::warn;

/// Filter matching both kinds of raytracing instance.
type RaytracingInstanceFilter = Or<(With<RaytracingMesh3d>, With<RaytracingGeometry>)>;

/// Maintains previous transforms for raytracing instances without [`Mesh3d`].
///
/// Runs in `PreUpdate`, before this frame's transforms change. The mesh transform
/// updater handles rasterized meshes separately.
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
            (
                RenderEntity,
                &MeshMaterial3d<StandardMaterial>,
                &GlobalTransform,
            ),
            Added<RaytracingGeometry>,
        >,
    >,
    mut removed_raytracing_meshes: Extract<RemovedComponents<RaytracingMesh3d>>,
    mut removed_raytracing_geometry: Extract<RemovedComponents<RaytracingGeometry>>,
    render_entities: Extract<Query<RenderEntity>>,
    mut commands: Commands,
) {
    // Process removed components before additions, that way it properly handles same-frame removal->insertion
    for main_entity in removed_raytracing_meshes.read() {
        if let Ok(render_entity) = render_entities.get(main_entity) {
            commands.entity(render_entity).remove::<RaytracingMesh3d>();
        }
    }

    for main_entity in removed_raytracing_geometry.read() {
        if let Ok(render_entity) = render_entities.get(main_entity) {
            // Remove the buffers too, so re-adding the marker cannot restore stale geometry.
            commands.entity(render_entity).remove::<(
                RaytracingGeometry,
                RaytracingGeometryBuffers,
                RaytracingGeometryPreviousVertices,
            )>();
        }
    }

    // New instances were absent from the previous TLAS. Start with zero motion
    // instead of a potentially stale main-world previous transform.
    for (render_entity, mesh, material, transform) in &new_instances {
        commands.entity(render_entity).insert((
            mesh.clone(),
            material.clone(),
            *transform,
            PreviousGlobalTransform(transform.affine()),
        ));
    }

    // The producer inserts the geometry buffers separately on the render entity.
    for (render_entity, material, transform) in &new_geometry_instances {
        commands.entity(render_entity).insert((
            RaytracingGeometry,
            material.clone(),
            *transform,
            PreviousGlobalTransform(transform.affine()),
        ));
    }
}

/// Copies the transforms of moved raytracing instances from the main world
/// straight into their GPU buffers.
///
/// Also updates [`GlobalTransform`] on [`RaytracingGeometry`] render entities so
/// producers converting vertices to local space use the same transform as the TLAS.
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
    moved_geometry: Extract<
        Query<
            (RenderEntity, &GlobalTransform),
            (Changed<GlobalTransform>, With<RaytracingGeometry>),
        >,
    >,
    mut render_geometry: Query<&mut GlobalTransform, With<RaytracingGeometry>>,
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

    // Structural extraction sets the transform for newly added markers.
    for (render_entity, transform) in &moved_geometry {
        if let Ok(mut render_transform) = render_geometry.get_mut(render_entity) {
            *render_transform = *transform;
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

#[derive(Resource, Default, Clone, PartialEq)]
pub struct ExtractedEnvironmentMapLight {
    pub cubemap: Option<Handle<Image>>,
    pub intensity: f32,
    pub rotation: Quat,
}

/// Finds the environment map light to use for the raytraced scene, if any.
pub fn extract_raytracing_environment_map_light(
    cameras: Extract<
        Query<(
            &Camera,
            Option<&GeneratedEnvironmentMapLight>,
            Option<&EnvironmentMapLight>,
        )>,
    >,
    mut environment_map_light: ResMut<ExtractedEnvironmentMapLight>,
) {
    let mut extracted_env_map_light = ExtractedEnvironmentMapLight::default();

    for (camera, generated, pregenerated) in &cameras {
        if !camera.is_active {
            continue;
        }

        let env_map_light = match (generated, pregenerated) {
            (Some(generated), _) => ExtractedEnvironmentMapLight {
                cubemap: Some(generated.environment_map.clone()),
                intensity: generated.intensity,
                rotation: generated.rotation,
            },
            (None, Some(pregenerated)) => ExtractedEnvironmentMapLight {
                cubemap: Some(pregenerated.specular_map.clone()),
                intensity: pregenerated.intensity,
                rotation: pregenerated.rotation,
            },
            (None, None) => continue,
        };

        if extracted_env_map_light.cubemap.is_none() {
            extracted_env_map_light = env_map_light;
        } else if extracted_env_map_light != env_map_light {
            once!(warn!(
                "bevy_solari only supports a single environment light for the whole scene, but \
                 multiple cameras have differing environment lights. Using the first one found."
            ));
        }
    }

    *environment_map_light = extracted_env_map_light;
}
