//! Resolves the [`DisplayTarget`] each view encodes for.
//!
//! Windows have it as a required component, and
//! [`resolve_display_targets`](super::window::resolve_display_targets)
//! resolves it in the main world with what the display reports. Other render
//! targets register one in [`ManualDisplayTargets`].
//! [`extract_cameras`](crate::camera::extract_cameras) writes the result to
//! [`ViewDisplayTarget`] every frame.

use bevy_camera::NormalizedRenderTarget;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::prelude::*;
use bevy_extract_macros::ExtractResource;
use bevy_platform::collections::HashMap;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_window::{DisplayTarget, ResolvedDisplayTarget, SurfaceColorSpace};

use super::window::ExtractedWindow;
use crate::RenderApp;

/// Resource that stores the [`DisplayTarget`] of each render target that is
/// not a [`Window`](bevy_window::Window), keyed by [`NormalizedRenderTarget`].
///
/// Insert into it from the main world. These targets have no surface to
/// negotiate with, so [`DisplayTarget::color_space_override`] is used as is
/// and [`DisplayTarget::hdr`] alone cannot be honored: a target with `hdr`
/// and no override resolves to [`SurfaceColorSpace::Srgb`].
#[derive(Default, Clone, Debug, PartialEq, Resource, ExtractResource, Reflect, Deref, DerefMut)]
#[reflect(Resource, Default, Debug, PartialEq, Clone)]
#[extract_app(RenderApp)]
pub struct ManualDisplayTargets(HashMap<NormalizedRenderTarget, DisplayTarget>);

/// The [`ResolvedDisplayTarget`] a view encodes for, after surface
/// negotiation.
///
/// Required by [`ExtractedCamera`](crate::camera::ExtractedCamera).
/// [`extract_cameras`](crate::camera::extract_cameras) inserts it every frame
/// next to [`ExtractedView::target_format`](super::ExtractedView::target_format),
/// and both describe the color space the window surface negotiated in the
/// previous frame. A target that cannot be resolved gets the default: SDR
/// sRGB at 100 nits.
#[derive(Component, Debug, Clone, Copy, PartialEq, Deref, Default)]
pub struct ViewDisplayTarget(pub ResolvedDisplayTarget);

impl ViewDisplayTarget {
    /// Returns `true` if the color space has high dynamic range. See
    /// [`SurfaceColorSpace::is_hdr`].
    pub fn is_hdr(&self) -> bool {
        self.0.color_space.is_hdr()
    }
}

/// Resolves a [`DisplayTarget`] that has no surface to negotiate with.
fn resolve_manual_display_target(display_target: &DisplayTarget) -> ResolvedDisplayTarget {
    display_target.resolve(
        display_target
            .color_space_override
            .unwrap_or(SurfaceColorSpace::Srgb),
    )
}

/// Resolves the [`ViewDisplayTarget`] for a render target.
///
/// A window takes [`ExtractedWindow::resolved_display_target`] when its color
/// space is the one the surface negotiated in the previous frame. When the
/// two differ, the surface wins: the request is resolved again for the
/// surface's color space with [`DisplayTarget::resolve`], so calibrated
/// values are kept and the rest take the defaults of that color space. The
/// main world resolves one frame behind the surface, and its result for
/// another color space would give a PQ frame the SDR peak, or an SDR frame
/// the HDR peak. An image or texture view looks up
/// [`ManualDisplayTargets`] by the whole [`NormalizedRenderTarget`], so an
/// image entry must also match the scale factor. Anything else, including a
/// missing entry, is [`DisplayTarget::default`] resolved as SDR sRGB.
pub fn resolve_view_display_target<'a>(
    target: Option<&NormalizedRenderTarget>,
    windows: impl IntoIterator<Item = (Entity, &'a ExtractedWindow)>,
    manual_display_targets: &ManualDisplayTargets,
) -> ViewDisplayTarget {
    let resolved = match target {
        Some(NormalizedRenderTarget::Window(window_ref)) => windows
            .into_iter()
            .find(|(entity, _)| *entity == window_ref.entity())
            .map(|(_, window)| {
                let resolved = window.resolved_display_target;
                match window.resolved_color_space {
                    Some(color_space) if color_space != resolved.color_space => {
                        window.display_target.resolve(color_space)
                    }
                    _ => resolved,
                }
            }),
        Some(
            target @ (NormalizedRenderTarget::Image(_) | NormalizedRenderTarget::TextureView(_)),
        ) => manual_display_targets
            .get(target)
            .map(resolve_manual_display_target),
        Some(NormalizedRenderTarget::None { .. }) | None => None,
    };
    ViewDisplayTarget(resolved.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::Handle;
    use bevy_camera::{ImageRenderTarget, ManualTextureViewHandle};
    use bevy_image::Image;
    use bevy_window::{
        CompositeAlphaMode, DisplayProvenance, FieldProvenance, PresentMode, WindowRef,
    };

    fn extracted_window(
        display_target: DisplayTarget,
        resolved_display_target: ResolvedDisplayTarget,
        resolved_color_space: Option<SurfaceColorSpace>,
    ) -> ExtractedWindow {
        ExtractedWindow {
            physical_width: 1,
            physical_height: 1,
            present_mode: PresentMode::AutoVsync,
            desired_maximum_frame_latency: None,
            swap_chain_texture_view: None,
            swap_chain_texture: None,
            swap_chain_texture_format: None,
            swap_chain_texture_view_format: None,
            size_changed: false,
            present_mode_changed: false,
            alpha_mode: CompositeAlphaMode::Auto,
            display_target,
            resolved_display_target,
            display_reports_sdr: false,
            color_space_request_changed: false,
            resolved_color_space,
            request_display_requery: false,
            needs_initial_present: false,
        }
    }

    fn image_target(scale_factor: f32) -> NormalizedRenderTarget {
        NormalizedRenderTarget::Image(ImageRenderTarget {
            handle: Handle::<Image>::default(),
            scale_factor,
        })
    }

    fn pq_override() -> DisplayTarget {
        DisplayTarget {
            color_space_override: Some(SurfaceColorSpace::Pq),
            ..Default::default()
        }
    }

    #[test]
    fn view_display_target_resolved_per_target() {
        let sdr_target = NormalizedRenderTarget::TextureView(ManualTextureViewHandle(0));
        let hdr_target = NormalizedRenderTarget::TextureView(ManualTextureViewHandle(1));
        let mut manual = ManualDisplayTargets::default();
        manual.insert(hdr_target.clone(), pq_override());

        let sdr = resolve_view_display_target(Some(&sdr_target), core::iter::empty(), &manual);
        assert_eq!(sdr, ViewDisplayTarget::default());
        assert!(!sdr.is_hdr());

        let hdr = resolve_view_display_target(Some(&hdr_target), core::iter::empty(), &manual);
        assert_eq!(
            hdr,
            ViewDisplayTarget(pq_override().resolve(SurfaceColorSpace::Pq))
        );
        assert!(hdr.is_hdr());
    }

    #[test]
    fn manual_targets_resolve_with_the_override_and_ignore_hdr() {
        let calibrated_pq = DisplayTarget {
            color_space_override: Some(SurfaceColorSpace::Pq),
            paper_white_nits: Some(203.0),
            peak_luminance_nits: Some(1000.0),
            min_luminance_nits: Some(0.005),
            ..Default::default()
        };
        let hdr_only = DisplayTarget {
            hdr: true,
            ..Default::default()
        };

        let image = image_target(1.0);
        let texture_view = NormalizedRenderTarget::TextureView(ManualTextureViewHandle(7));
        let mut manual = ManualDisplayTargets::default();
        manual.insert(image.clone(), calibrated_pq);
        manual.insert(texture_view.clone(), hdr_only);

        assert_eq!(
            resolve_view_display_target(Some(&image), core::iter::empty(), &manual).0,
            ResolvedDisplayTarget {
                color_space: SurfaceColorSpace::Pq,
                paper_white_nits: 203.0,
                peak_luminance_nits: 1000.0,
                min_luminance_nits: 0.005,
                provenance: DisplayProvenance::USER,
            }
        );
        // There is no surface to negotiate an HDR color space with.
        assert_eq!(
            resolve_view_display_target(Some(&texture_view), core::iter::empty(), &manual).0,
            ResolvedDisplayTarget::default()
        );
    }

    #[test]
    fn misses_fall_back_to_the_default() {
        let mut manual = ManualDisplayTargets::default();
        manual.insert(image_target(1.0), pq_override());

        // Same image handle, different scale factor: the whole key must match.
        assert_eq!(
            resolve_view_display_target(Some(&image_target(2.0)), core::iter::empty(), &manual),
            ViewDisplayTarget::default()
        );
        assert_eq!(
            resolve_view_display_target(
                Some(&NormalizedRenderTarget::TextureView(
                    ManualTextureViewHandle(7)
                )),
                core::iter::empty(),
                &manual
            ),
            ViewDisplayTarget::default()
        );
        assert_eq!(
            resolve_view_display_target(
                Some(&NormalizedRenderTarget::None {
                    width: 64,
                    height: 64
                }),
                core::iter::empty(),
                &manual
            ),
            ViewDisplayTarget::default()
        );
        assert_eq!(
            resolve_view_display_target(None, core::iter::empty(), &manual),
            ViewDisplayTarget::default()
        );
        // A window the render world has not extracted.
        assert_eq!(
            resolve_view_display_target(
                Some(&NormalizedRenderTarget::Window(
                    WindowRef::Primary
                        .normalize(Some(Entity::PLACEHOLDER))
                        .unwrap()
                )),
                core::iter::empty(),
                &manual
            ),
            ViewDisplayTarget::default()
        );
    }

    #[test]
    fn window_targets_take_the_main_world_result_with_the_surface_color_space() {
        let window_entity = Entity::from_raw_u32(3).unwrap();
        let target = NormalizedRenderTarget::Window(
            WindowRef::Entity(window_entity).normalize(None).unwrap(),
        );
        let manual = ManualDisplayTargets::default();
        let requested = DisplayTarget {
            hdr: true,
            paper_white_nits: Some(300.0),
            ..Default::default()
        };
        // The main world resolved the request with what the display reports.
        let resolved = ResolvedDisplayTarget {
            color_space: SurfaceColorSpace::Pq,
            paper_white_nits: 300.0,
            peak_luminance_nits: 1500.0,
            min_luminance_nits: 0.01,
            provenance: DisplayProvenance {
                paper_white: FieldProvenance::User,
                peak_luminance: FieldProvenance::Os,
                min_luminance: FieldProvenance::Os,
            },
        };

        // The surface agrees: the result passes through.
        let window = extracted_window(requested, resolved, Some(SurfaceColorSpace::Pq));
        assert_eq!(
            resolve_view_display_target(Some(&target), [(window_entity, &window)], &manual).0,
            resolved
        );

        // The surface fell back to SDR after the main world resolved: the
        // surface wins, and the request is resolved again for its color
        // space. The calibrated paper white is kept, and the sensed values
        // take the SDR defaults until the main world catches up.
        let window = extracted_window(requested, resolved, Some(SurfaceColorSpace::Srgb));
        let view = resolve_view_display_target(Some(&target), [(window_entity, &window)], &manual);
        assert!(!view.is_hdr());
        assert_eq!(view.0, requested.resolve(SurfaceColorSpace::Srgb));
        assert_eq!(view.0.paper_white_nits, 300.0);
        assert_eq!(view.0.peak_luminance_nits, 300.0);
        assert_eq!(view.0.provenance.peak_luminance, FieldProvenance::Default);

        // Before the surface is configured, the main world result is used as
        // is.
        let unconfigured = requested.resolve(SurfaceColorSpace::Srgb);
        let window = extracted_window(requested, unconfigured, None);
        assert_eq!(
            resolve_view_display_target(Some(&target), [(window_entity, &window)], &manual).0,
            unconfigured
        );
    }
}
