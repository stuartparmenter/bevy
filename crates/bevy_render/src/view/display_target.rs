//! Resolves the [`DisplayTarget`] each view encodes for.
//!
//! Windows have it as a required component. Other render targets register
//! one in [`ManualDisplayTargets`]. [`prepare_view_display_targets`] writes
//! the result to [`ViewDisplayTarget`] every frame.

use bevy_camera::NormalizedRenderTarget;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::prelude::*;
use bevy_extract_macros::ExtractResource;
use bevy_platform::collections::HashMap;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_window::{DisplayTarget, DisplayTransfer};

use super::{window::ExtractedWindow, ExtractedView};
use crate::{camera::ExtractedCamera, sync_world::MainEntity, RenderApp};

/// Resource that stores the [`DisplayTarget`] of each render target that is
/// not a [`Window`](bevy_window::Window), keyed by [`NormalizedRenderTarget`].
///
/// Insert into it from the main world. Views use the value as is, with no
/// negotiation.
#[derive(Default, Clone, Debug, PartialEq, Resource, ExtractResource, Reflect, Deref, DerefMut)]
#[reflect(Resource, Default, Debug, PartialEq, Clone)]
#[extract_app(RenderApp)]
pub struct ManualDisplayTargets(HashMap<NormalizedRenderTarget, DisplayTarget>);

/// The [`DisplayTarget`] a view encodes for, after surface negotiation.
///
/// Required by [`ExtractedCamera`]. [`prepare_view_display_targets`] updates
/// it every frame. A target that cannot be resolved gets
/// [`DisplayTarget::SDR_SRGB`], which is also the default.
#[derive(Component, Debug, Clone, Copy, PartialEq, Deref, Default)]
pub struct ViewDisplayTarget(pub DisplayTarget);

impl ViewDisplayTarget {
    /// Returns `true` if the transfer has high dynamic range. See
    /// [`DisplayTransfer::is_hdr`].
    pub fn is_hdr_transfer(&self) -> bool {
        self.0.transfer.is_hdr()
    }
}

/// Applies the transfer the surface uses to the requested [`DisplayTarget`].
///
/// A downgrade to sRGB replaces the whole target with
/// [`DisplayTarget::SDR_SRGB`], so the view behaves like an SDR view. Any
/// other differing transfer replaces only the transfer field. `None` means
/// the surface is not configured yet.
fn resolve_window_display_target(
    requested: DisplayTarget,
    surface_transfer: Option<DisplayTransfer>,
) -> DisplayTarget {
    match surface_transfer {
        Some(DisplayTransfer::Srgb) if requested.transfer != DisplayTransfer::Srgb => {
            DisplayTarget::SDR_SRGB
        }
        Some(transfer) if transfer != requested.transfer => DisplayTarget {
            transfer,
            ..requested
        },
        _ => requested,
    }
}

/// Resolves the [`ViewDisplayTarget`] for a render target.
///
/// A window uses [`ExtractedWindow::display_target`] with the negotiated
/// transfer applied. An image or texture view looks up
/// [`ManualDisplayTargets`] by the whole [`NormalizedRenderTarget`], so an
/// image entry must also match the scale factor. Anything else, including a
/// missing entry, is [`DisplayTarget::SDR_SRGB`].
pub fn resolve_view_display_target<'a>(
    target: Option<&NormalizedRenderTarget>,
    windows: impl IntoIterator<Item = (Entity, &'a ExtractedWindow)>,
    manual_display_targets: &ManualDisplayTargets,
) -> ViewDisplayTarget {
    let display_target = match target {
        Some(NormalizedRenderTarget::Window(window_ref)) => windows
            .into_iter()
            .find(|(entity, _)| *entity == window_ref.entity())
            .map(|(_, window)| {
                resolve_window_display_target(window.display_target, window.resolved_transfer)
            })
            .unwrap_or_default(),
        Some(
            target @ (NormalizedRenderTarget::Image(_) | NormalizedRenderTarget::TextureView(_)),
        ) => manual_display_targets
            .get(target)
            .copied()
            .unwrap_or_default(),
        Some(NormalizedRenderTarget::None { .. }) | None => DisplayTarget::SDR_SRGB,
    };
    ViewDisplayTarget(display_target)
}

/// Resolves the [`ViewDisplayTarget`] of every view with an
/// [`ExtractedCamera`].
///
/// Runs after [`create_surfaces`](super::window::create_surfaces), so it sees
/// this frame's negotiated transfer.
pub fn prepare_view_display_targets(
    windows: Query<(MainEntity, &ExtractedWindow)>,
    manual_display_targets: Res<ManualDisplayTargets>,
    mut views: Query<(&ExtractedCamera, &mut ViewDisplayTarget), With<ExtractedView>>,
) {
    for (camera, mut target) in &mut views {
        target.set_if_neq(resolve_view_display_target(
            camera.target.as_ref(),
            windows.iter(),
            &manual_display_targets,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::Main;
    use bevy_asset::Handle;
    use bevy_camera::{
        CameraOutputMode, ClearColorConfig, ImageRenderTarget, ManualTextureViewHandle,
        MsaaWriteback,
    };
    use bevy_ecs::{schedule::ScheduleLabel, system::RunSystemOnce, world::World};
    use bevy_image::Image;
    use bevy_math::{Mat4, UVec4};
    use bevy_transform::components::GlobalTransform;
    use bevy_window::DisplayGamut;
    use wgpu::TextureFormat;

    use crate::view::{ColorGrading, RetainedViewEntity};

    fn extracted_camera(target: NormalizedRenderTarget) -> ExtractedCamera {
        ExtractedCamera {
            target: Some(target),
            physical_viewport_size: None,
            physical_target_size: None,
            viewport: None,
            schedule: Main.intern(),
            order: 0,
            output_mode: CameraOutputMode::default(),
            msaa_writeback: MsaaWriteback::default(),
            clear_color: ClearColorConfig::Default,
            sorted_camera_index_for_target: 0,
            exposure: 1.0,
            hdr: false,
        }
    }

    fn extracted_view() -> ExtractedView {
        ExtractedView {
            retained_view_entity: RetainedViewEntity::new(Entity::PLACEHOLDER.into(), None, 0),
            clip_from_view: Mat4::IDENTITY,
            world_from_view: GlobalTransform::default(),
            clip_from_world: None,
            target_format: TextureFormat::Rgba8UnormSrgb,
            viewport: UVec4::ZERO,
            color_grading: ColorGrading::default(),
            invert_culling: false,
        }
    }

    fn image_target(scale_factor: f32) -> NormalizedRenderTarget {
        NormalizedRenderTarget::Image(ImageRenderTarget {
            handle: Handle::<Image>::default(),
            scale_factor,
        })
    }

    #[test]
    fn view_display_target_resolved_per_view() {
        let mut world = World::new();

        let sdr_target = NormalizedRenderTarget::TextureView(ManualTextureViewHandle(0));
        let hdr_target = NormalizedRenderTarget::TextureView(ManualTextureViewHandle(1));
        let pq = DisplayTarget {
            transfer: DisplayTransfer::Pq,
            ..DisplayTarget::SDR_SRGB
        };
        let mut manual_targets = ManualDisplayTargets::default();
        manual_targets.insert(hdr_target.clone(), pq);
        world.insert_resource(manual_targets);

        let sdr = world
            .spawn((extracted_camera(sdr_target), extracted_view()))
            .id();
        let hdr = world
            .spawn((extracted_camera(hdr_target), extracted_view()))
            .id();

        world.run_system_once(prepare_view_display_targets).unwrap();

        let sdr_resolved = world.entity(sdr).get::<ViewDisplayTarget>().copied();
        assert_eq!(
            sdr_resolved,
            Some(ViewDisplayTarget(DisplayTarget::SDR_SRGB))
        );
        assert!(!sdr_resolved.unwrap().is_hdr_transfer());

        let hdr_resolved = world.entity(hdr).get::<ViewDisplayTarget>().copied();
        assert_eq!(hdr_resolved, Some(ViewDisplayTarget(pq)));
        assert!(hdr_resolved.unwrap().is_hdr_transfer());
    }

    #[test]
    fn registered_manual_targets_resolve_to_the_authored_value() {
        let pq = DisplayTarget::SDR_SRGB
            .with_transfer(DisplayTransfer::Pq)
            .with_gamut(DisplayGamut::Rec2020)
            .with_peak_luminance(1000.0)
            .with_paper_white(203.0)
            .with_min_luminance(0.005);
        let scrgb = DisplayTarget::SDR_SRGB.with_transfer(DisplayTransfer::ScRgbLinear);

        let image = image_target(1.0);
        let texture_view = NormalizedRenderTarget::TextureView(ManualTextureViewHandle(7));
        let mut manual = ManualDisplayTargets::default();
        manual.insert(image.clone(), pq);
        manual.insert(texture_view.clone(), scrgb);

        assert_eq!(
            resolve_view_display_target(Some(&image), core::iter::empty(), &manual).0,
            pq
        );
        assert_eq!(
            resolve_view_display_target(Some(&texture_view), core::iter::empty(), &manual).0,
            scrgb
        );
    }

    #[test]
    fn misses_fall_back_to_sdr_srgb() {
        let mut manual = ManualDisplayTargets::default();
        manual.insert(
            image_target(1.0),
            DisplayTarget::SDR_SRGB.with_transfer(DisplayTransfer::Pq),
        );

        // Same image handle, different scale factor: the whole key must match.
        assert_eq!(
            resolve_view_display_target(Some(&image_target(2.0)), core::iter::empty(), &manual).0,
            DisplayTarget::SDR_SRGB
        );
        assert_eq!(
            resolve_view_display_target(
                Some(&NormalizedRenderTarget::TextureView(
                    ManualTextureViewHandle(7)
                )),
                core::iter::empty(),
                &manual
            )
            .0,
            DisplayTarget::SDR_SRGB
        );
        assert_eq!(
            resolve_view_display_target(
                Some(&NormalizedRenderTarget::None {
                    width: 64,
                    height: 64
                }),
                core::iter::empty(),
                &manual
            )
            .0,
            DisplayTarget::SDR_SRGB
        );
        assert_eq!(
            resolve_view_display_target(None, core::iter::empty(), &manual).0,
            DisplayTarget::SDR_SRGB
        );
    }

    #[test]
    fn window_target_resolution() {
        let requested = DisplayTarget {
            paper_white_nits: 200.0,
            peak_luminance_nits: 1000.0,
            gamut: DisplayGamut::Rec2020,
            transfer: DisplayTransfer::Pq,
            ..DisplayTarget::SDR_SRGB
        };

        // A matching transfer or an unconfigured surface passes the request
        // through unchanged.
        assert_eq!(
            resolve_window_display_target(requested, Some(DisplayTransfer::Pq)),
            requested
        );
        assert_eq!(resolve_window_display_target(requested, None), requested);

        assert_eq!(
            resolve_window_display_target(requested, Some(DisplayTransfer::Srgb)),
            DisplayTarget::SDR_SRGB
        );

        // A different HDR transfer replaces only the transfer field.
        assert_eq!(
            resolve_window_display_target(requested, Some(DisplayTransfer::ScRgbLinear)),
            DisplayTarget {
                transfer: DisplayTransfer::ScRgbLinear,
                ..requested
            }
        );

        // An sRGB request against a surface still using an HDR transfer
        // keeps the surface's transfer.
        let sdr = DisplayTarget::SDR_SRGB;
        assert_eq!(
            resolve_window_display_target(sdr, Some(DisplayTransfer::Pq)),
            DisplayTarget {
                transfer: DisplayTransfer::Pq,
                ..sdr
            }
        );
    }
}
