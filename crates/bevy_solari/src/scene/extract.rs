use super::{RaytracingMesh3d, RaytracingMesh3dGeometryError, SolariEnvironmentLight};
use bevy_asset::{AssetId, Assets};
use bevy_derive::Deref;
use bevy_ecs::{
    lifecycle::RemovedComponents,
    resource::Resource,
    system::{Commands, Query},
};
use bevy_pbr::{MeshMaterial3d, PreviousGlobalTransform, StandardMaterial};
use bevy_platform::collections::HashMap;
use bevy_render::{
    extract_resource::ExtractResource, sync_world::RenderEntity, Extract, RenderApp,
};
use bevy_transform::components::GlobalTransform;

pub fn extract_raytracing_scene(
    instances: Extract<
        Query<(
            RenderEntity,
            &RaytracingMesh3d,
            &RaytracingMesh3dGeometryError,
            &MeshMaterial3d<StandardMaterial>,
            &GlobalTransform,
            Option<&PreviousGlobalTransform>,
        )>,
    >,
    mut removed_raytracing_meshes: Extract<RemovedComponents<RaytracingMesh3d>>,
    environment_lights: Extract<Query<(RenderEntity, &SolariEnvironmentLight)>>,
    mut removed_environment_lights: Extract<RemovedComponents<SolariEnvironmentLight>>,
    render_entities: Extract<Query<RenderEntity>>,
    mut commands: Commands,
) {
    for main_entity in removed_raytracing_meshes.read() {
        if let Ok(render_entity) = render_entities.get(main_entity) {
            commands.entity(render_entity).remove::<RaytracingMesh3d>();
        }
    }
    for main_entity in removed_environment_lights.read() {
        if let Ok(render_entity) = render_entities.get(main_entity) {
            commands
                .entity(render_entity)
                .remove::<SolariEnvironmentLight>();
        }
    }

    for (render_entity, mesh, geometry_error, material, transform, previous_frame_transform) in
        &instances
    {
        let mut commands = commands.entity(render_entity);

        match previous_frame_transform.cloned() {
            Some(previous_frame_transform) => commands.insert((
                mesh.clone(),
                *geometry_error,
                material.clone(),
                *transform,
                previous_frame_transform,
            )),
            None => commands.insert((mesh.clone(), *geometry_error, material.clone(), *transform)),
        };
    }

    for (render_entity, environment_light) in &environment_lights {
        commands.entity(render_entity).insert(*environment_light);
    }
}

#[derive(Resource, Deref, Default)]
pub struct StandardMaterialAssets(HashMap<AssetId<StandardMaterial>, StandardMaterial>);

impl ExtractResource<RenderApp> for StandardMaterialAssets {
    type Source = Assets<StandardMaterial>;

    fn extract_resource(source: &Self::Source) -> Self {
        Self(
            source
                .iter()
                .map(|(asset_id, material)| (asset_id, material.clone()))
                .collect(),
        )
    }
}
