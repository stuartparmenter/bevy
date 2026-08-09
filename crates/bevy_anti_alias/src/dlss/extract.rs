use super::{prepare::DlssRenderContext, Dlss, DlssFeature};
use bevy_camera::{Camera, Hdr, MainPassResolutionOverride, Projection};
use bevy_ecs::{
    query::{Has, With},
    system::{Commands, Query, ResMut},
};
use bevy_render::{sync_world::RenderEntity, MainWorld};

/// `F2` is the other DLSS feature. `MainPassResolutionOverride` is shared
/// between the two, so cleanup only removes it while `F2` remains extractable
/// (present on the main-world camera AND the camera is an active perspective
/// one) — removing it while `F2`'s context survives would leave `F2`'s node
/// skipping forever (it requires the override in its `ViewQuery`), while
/// keeping it after `F2` also stops extracting would strand the camera at the
/// stale render resolution with no upscaler running.
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
