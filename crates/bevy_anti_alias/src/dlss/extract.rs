use super::{prepare::DlssRenderContext, Dlss, DlssFeature};
use bevy_camera::{Camera, Hdr, MainPassResolutionOverride, Projection};
use bevy_ecs::{
    query::{Has, With},
    system::{Commands, Query, ResMut},
};
use bevy_render::{sync_world::RenderEntity, MainWorld};

/// Extracts `F` while preserving the shared resolution override if `F2` is still active.
/// The other feature must belong to an active perspective camera to keep the override.
pub fn extract_dlss<F: DlssFeature, F2: DlssFeature>(
    mut commands: Commands,
    mut main_world: ResMut<MainWorld>,
    cleanup_query: Query<Has<Dlss<F>>>,
) {
    let mut cameras_3d = main_world.query_filtered::<(
        RenderEntity,
        &Camera,
        &Projection,
        Option<&mut Dlss<F>>,
        Has<Dlss<F2>>,
    ), With<Hdr>>();

    for (entity, camera, camera_projection, mut dlss, other_dlss_feature) in
        cameras_3d.iter_mut(&mut main_world)
    {
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");
        if dlss.is_some() && camera.is_active && camera_projection.is_perspective() {
            entity_commands.insert(dlss.as_deref().unwrap().clone());
            dlss.as_mut().unwrap().reset = false;
        } else if cleanup_query.get(entity) == Ok(true) {
            entity_commands.remove::<(Dlss<F>, DlssRenderContext<F>)>();
            let other_still_extracting =
                other_dlss_feature && camera.is_active && camera_projection.is_perspective();
            if !other_still_extracting {
                entity_commands.remove::<MainPassResolutionOverride>();
            }
        }
    }
}
