use bevy_ecs::prelude::*;
use bevy_render::sync_world::{MainEntity, MainEntityHashSet};

/// Mesh instances whose skin and morph inputs are needed independently of raster visibility.
///
/// Renderers add requests during [`MeshDeformationSystems::Collect`] in
/// `ExtractSchedule`. Requests are cleared before collection each frame.
#[derive(Resource, Default)]
pub struct MeshDeformationRequests {
    entities: MainEntityHashSet,
}

impl MeshDeformationRequests {
    /// Requests this mesh instance's skin and morph inputs for the current frame.
    pub fn insert(&mut self, entity: MainEntity) {
        self.entities.insert(entity);
    }

    /// Whether this mesh instance's inputs were requested for the current frame.
    pub fn contains(&self, entity: MainEntity) -> bool {
        self.entities.contains(&entity)
    }
}

/// Systems that collect requests for mesh deformation inputs in `ExtractSchedule`.
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MeshDeformationSystems {
    /// Runs after requests are cleared and before skin and morph extraction.
    Collect,
}

pub(super) fn clear_mesh_deformation_requests(mut requests: ResMut<MeshDeformationRequests>) {
    requests.entities.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Resource)]
    struct Request {
        entity: MainEntity,
        enabled: bool,
    }

    #[test]
    fn requests_expire_before_collection() {
        let mut world = World::new();
        let entity = world.spawn_empty().id().into();
        world.init_resource::<MeshDeformationRequests>();
        world.insert_resource(Request {
            entity,
            enabled: true,
        });
        let mut schedule = Schedule::default();
        schedule.add_systems((
            clear_mesh_deformation_requests.before(MeshDeformationSystems::Collect),
            (|request: Res<Request>, mut requests: ResMut<MeshDeformationRequests>| {
                if request.enabled {
                    requests.insert(request.entity);
                }
            })
            .in_set(MeshDeformationSystems::Collect),
            (|request: Res<Request>, requests: Res<MeshDeformationRequests>| {
                assert_eq!(requests.contains(request.entity), request.enabled);
            })
            .after(MeshDeformationSystems::Collect),
        ));
        schedule.run(&mut world);
        world.resource_mut::<Request>().enabled = false;
        schedule.run(&mut world);
    }
}
