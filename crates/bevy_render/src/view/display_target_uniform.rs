//! Per-view display-target plumbing: the render-world [`ViewDisplayTarget`]
//! component and the [`DisplayTargetUniform`] GPU uniform.
//!
//! [`DisplayTarget`] is authored in the main world, as a required component of
//! `Window` or via [`ManualDisplayTargets`] for non-window targets. This module
//! resolves it per view each frame:
//!
//! 1. [`prepare_view_display_targets`] runs in
//!    [`RenderSystems::PrepareViews`](crate::RenderSystems::PrepareViews) and
//!    inserts a [`ViewDisplayTarget`] on every extracted camera view, as
//!    `resolve_view_display_target` describes. All cameras rendering to the
//!    same surface resolve to the same value by construction.
//! 2. The same system inserts the [`DisplayTargetUniform`] built from the
//!    resolved target, but only on views whose resolved transfer is HDR.
//!    [`UniformComponentPlugin`](crate::uniform::UniformComponentPlugin),
//!    registered for it in [`ViewPlugin`](super::ViewPlugin), packs those
//!    components into the [`ComponentUniforms<DisplayTargetUniform>`](crate::uniform::ComponentUniforms)
//!    dynamic uniform buffer during
//!    [`RenderSystems::PrepareResources`](crate::RenderSystems::PrepareResources)
//!    and gives each view a
//!    [`DynamicUniformIndex<DisplayTargetUniform>`](crate::uniform::DynamicUniformIndex)
//!    to address its entry with.
//!
//! Only the display-encoding pass binds the uniform, to drive the transfer
//! OETF. The tone-mapping pass does not read display calibration. Plain SDR
//! pipelines carry no display-target binding, and an SDR-only app never creates
//! the uniform buffer.
//!
//! The matching WGSL struct lives in `display_target.wgsl` and is importable as
//! `bevy_render::display_target`.

use bevy_camera::NormalizedRenderTarget;
use bevy_derive::Deref;
use bevy_ecs::prelude::*;
use bevy_log::warn_once;
use bevy_window::{DisplayTarget, DisplayTransfer};

use super::{
    window::{
        display_target::{resolve_display_target, ManualDisplayTargets},
        ExtractedWindow,
    },
    ExtractedView,
};
use crate::{camera::ExtractedCamera, render_resource::ShaderType, sync_world::MainEntity};

/// Render-world component holding the post-negotiation [`DisplayTarget`] of
/// the surface (window, image, or manual texture view) a view renders to.
///
/// Required by [`ExtractedCamera`], defaulting to [`DisplayTarget::SDR_SRGB`].
/// [`prepare_view_display_targets`] overwrites it every frame with the value
/// `resolve_view_display_target` produces. Views whose target cannot be
/// resolved fall back to [`DisplayTarget::SDR_SRGB`].
///
/// Prepare-time systems (tonemapping pipeline specialization, operator uniform
/// preparation, the display-encoding pass, and the upscaling blit) read this
/// target instead of re-resolving the render target themselves, so they key on
/// what the surface can show rather than on an unfulfilled request. The
/// requested-versus-granted downgrade warnings fire at negotiation time in
/// `negotiate_surface_format` in `view::window`.
#[derive(Component, Debug, Clone, Copy, PartialEq, Deref, Default)]
pub struct ViewDisplayTarget(pub DisplayTarget);

impl ViewDisplayTarget {
    /// Returns `true` if the transfer function is a high dynamic range
    /// transfer (see [`DisplayTransfer::is_hdr`]).
    ///
    /// This gates the display-encoding pass and the upscaling blit's
    /// pass-through mode. HDR-capable operators such as
    /// `Tonemapping::GranTurismo7` also use it to pick their HDR mode at
    /// prepare time.
    pub fn is_hdr_transfer(&self) -> bool {
        self.0.transfer.is_hdr()
    }
}

/// GPU uniform carrying a view's resolved [`DisplayTarget`] calibration.
///
/// Must stay field-for-field in sync with the WGSL `DisplayTargetUniform` in
/// `display_target.wgsl`.
///
/// Paper white is the only calibration value the GPU reads at runtime. The
/// display gamut and transfer select compile-time shader defs in the
/// display-encoding pipeline, and the GT7 operator's peak-luminance parameters
/// are baked on the CPU into its own per-camera uniform.
#[derive(Component, Clone, Copy, Debug, PartialEq, ShaderType)]
pub struct DisplayTargetUniform {
    /// [`DisplayTarget::paper_white_nits`]: the luminance, in nits, that
    /// `1.0` at the tone-map operator output corresponds to.
    pub paper_white_nits: f32,
}

/// Resolves a window view's display target from the transfer the configured
/// surface carries
/// ([`ExtractedWindow::resolved_transfer`](super::window::ExtractedWindow::resolved_transfer)).
///
/// - `Some(`[`DisplayTransfer::Srgb`]`)` while an HDR transfer was requested,
///   such as scRGB-linear on a backend with no `Rgba16Float` surfaces. The
///   whole target degrades to [`DisplayTarget::SDR_SRGB`], not just the
///   transfer field, so the view takes the same plain SDR path as a natively
///   SDR view. The warning is emitted at negotiation time in `create_surfaces`.
/// - Any other `Some(transfer)` differing from the requested transfer, such as
///   PQ downgraded to scRGB-linear when HDR10 is unavailable. The user's
///   calibration (paper white, peak, gamut) is kept and only the transfer is
///   replaced, so the encoder keys on what the surface carries.
/// - Equal transfers, a transient `None` (the surface is not configured yet),
///   or a non-window target (`surface_transfer` is `None`): the request passes
///   through unchanged.
pub(crate) fn resolve_window_display_target(
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

/// Resolves the [`ViewDisplayTarget`] for a camera's normalized render target:
/// the target's [`DisplayTarget`] (a window's `EffectiveDisplayTarget`, or a
/// manual target's [`ManualDisplayTargets`] entry, see
/// [`resolve_display_target`]) with the window surface's negotiated transfer
/// folded in, see `resolve_window_display_target`.
///
/// Shared by [`prepare_view_display_targets`], which runs after surface
/// negotiation and so sees this frame's resolved transfer, and by camera
/// extraction (`extract_cameras`), which picks the main-texture format. In
/// extraction the surface transfer is the previous frame's negotiation result,
/// as fresh as the swapchain format extraction already reads for the output
/// format.
///
/// Non-window targets have no surface to negotiate with, so their requested
/// value passes through unchanged. The user owns the texture and its format.
pub(crate) fn resolve_view_display_target<'a>(
    target: Option<&NormalizedRenderTarget>,
    windows: impl IntoIterator<Item = (Entity, &'a ExtractedWindow)>,
    manual_display_targets: &ManualDisplayTargets,
) -> ViewDisplayTarget {
    // Look the window up once and reuse it for both the requested target and
    // the negotiated surface transfer. Other target kinds have no surface, so
    // `resolve_display_target` needs no window access.
    let window = match target {
        Some(NormalizedRenderTarget::Window(window_ref)) => windows
            .into_iter()
            .find(|(entity, _)| *entity == window_ref.entity())
            .map(|(_, window)| window),
        _ => None,
    };
    let requested = match target {
        Some(NormalizedRenderTarget::Window(_)) => window
            .map(|window| window.display_target)
            .unwrap_or_default(),
        _ => resolve_display_target(target, core::iter::empty(), manual_display_targets),
    };
    let surface_transfer = window.and_then(|window| window.resolved_transfer);
    ViewDisplayTarget(resolve_window_display_target(requested, surface_transfer))
}

/// Resolves and inserts a [`ViewDisplayTarget`] on every extracted view that
/// has an [`ExtractedCamera`], plus the matching [`DisplayTargetUniform`] on
/// views whose resolved transfer is HDR.
///
/// `resolve_group_encode_parameters` in `bevy_core_pipeline` gates the
/// display-encoding pass on the same [`ViewDisplayTarget::is_hdr_transfer`]
/// predicate. A view whose target drops from HDR to SDR has its uniform
/// removed. The [`DynamicUniformIndex`](crate::uniform::DynamicUniformIndex)
/// left behind goes stale, which is harmless because the display-encoding pass
/// bails on its missing pipeline before reading the index.
///
/// Runs in [`RenderSystems::PrepareViews`](crate::RenderSystems::PrepareViews),
/// after `create_surfaces` so the window surface's negotiated transfer is
/// fresh, and before the prepare systems that specialize pipelines or pack
/// uniforms. The [`DisplayTargetUniform`] in particular must be inserted before
/// [`UniformComponentPlugin`](crate::uniform::UniformComponentPlugin) packs it
/// in [`RenderSystems::PrepareResources`](crate::RenderSystems::PrepareResources).
///
/// [`DisplayTargetUniform::paper_white_nits`] is sanitized through
/// [`DisplayTarget::sanitized_paper_white_nits`], which maps non-finite and
/// non-positive values to 100 nits and clamps to the 10000-nit PQ ceiling,
/// passing valid values through bit-for-bit. A `warn_once!` fires when the
/// authored value had to be replaced. This keeps the GPU-side paper white equal
/// to the value the tone-mapping operators fold at their own prepare step, so
/// the two factors cancel: the operator scales its output by
/// `100 / paper_white`, and the encoder scales by `paper_white / 80` (scRGB)
/// or by `paper_white` (PQ).
pub fn prepare_view_display_targets(
    mut commands: Commands,
    windows: Query<(MainEntity, &ExtractedWindow)>,
    manual_display_targets: Res<ManualDisplayTargets>,
    views: Query<(Entity, &ExtractedCamera, Has<DisplayTargetUniform>), With<ExtractedView>>,
) {
    for (entity, camera, has_uniform) in &views {
        let view_display_target = resolve_view_display_target(
            camera.target.as_ref(),
            windows.iter(),
            &manual_display_targets,
        );
        let authored = view_display_target.paper_white_nits;
        let sanitized = view_display_target.sanitized_paper_white_nits();
        // Sanitize and warn for every view, not just HDR ones. SDR consumers
        // such as bloom's nits-denominated threshold fold the sanitized value
        // too, and this is the only diagnostic a plain-SDR project gets.
        // Compare bits so a NaN input, never equal to itself, still counts as
        // replaced.
        if sanitized.to_bits() != authored.to_bits() {
            warn_once!(
                "DisplayTarget::paper_white_nits ({}) is non-finite, non-positive, or above \
                 the 10000-nit PQ ceiling; the display pipeline is using {} nits instead",
                authored,
                sanitized
            );
        }
        if view_display_target.is_hdr_transfer() {
            commands.entity(entity).insert((
                view_display_target,
                DisplayTargetUniform {
                    paper_white_nits: sanitized,
                },
            ));
        } else {
            let mut entity_commands = commands.entity(entity);
            entity_commands.insert(view_display_target);
            // Only queue the removal when there is a uniform to remove. A
            // blanket remove would cost one command per camera per frame on
            // SDR projects.
            if has_uniform {
                entity_commands.remove::<DisplayTargetUniform>();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::Main;
    use bevy_camera::{CameraOutputMode, ClearColorConfig, ManualTextureViewHandle, MsaaWriteback};
    use bevy_ecs::{schedule::ScheduleLabel, system::RunSystemOnce, world::World};
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
            compositing_space: None,
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

    /// SDR views get the [`ViewDisplayTarget`] alone, and a stale uniform from
    /// a previous HDR resolution is removed.
    #[test]
    fn uniform_inserted_only_for_hdr_transfer_views() {
        let mut world = World::new();

        let sdr_target = NormalizedRenderTarget::TextureView(ManualTextureViewHandle(0));
        let hdr_target = NormalizedRenderTarget::TextureView(ManualTextureViewHandle(1));
        let mut manual_targets = ManualDisplayTargets::default();
        manual_targets.insert(
            hdr_target.clone(),
            DisplayTarget {
                transfer: DisplayTransfer::Pq,
                ..DisplayTarget::SDR_SRGB
            },
        );
        world.insert_resource(manual_targets);

        let sdr = world
            .spawn((extracted_camera(sdr_target.clone()), extracted_view()))
            .id();
        let hdr = world
            .spawn((extracted_camera(hdr_target), extracted_view()))
            .id();
        // An SDR view still carrying the uniform from when its target
        // resolved as HDR.
        let downgraded = world
            .spawn((
                extracted_camera(sdr_target),
                extracted_view(),
                DisplayTargetUniform {
                    paper_white_nits: 100.0,
                },
            ))
            .id();

        world.run_system_once(prepare_view_display_targets).unwrap();

        for entity in [sdr, hdr, downgraded] {
            assert!(world.entity(entity).contains::<ViewDisplayTarget>());
        }
        assert!(!world.entity(sdr).contains::<DisplayTargetUniform>());
        assert_eq!(
            world.entity(hdr).get::<DisplayTargetUniform>(),
            Some(&DisplayTargetUniform {
                paper_white_nits: 100.0
            })
        );
        assert!(!world.entity(downgraded).contains::<DisplayTargetUniform>());
    }

    #[test]
    fn view_display_target_is_hdr_transfer() {
        let sdr = ViewDisplayTarget(DisplayTarget::SDR_SRGB);
        assert!(!sdr.is_hdr_transfer());

        // A non-transfer field change does not flip the HDR predicate.
        let brighter = ViewDisplayTarget(DisplayTarget {
            paper_white_nits: 203.0,
            ..DisplayTarget::SDR_SRGB
        });
        assert!(!brighter.is_hdr_transfer());

        for transfer in [
            DisplayTransfer::ScRgbLinear,
            DisplayTransfer::Pq,
            DisplayTransfer::ExtendedSrgb,
        ] {
            let hdr = ViewDisplayTarget(DisplayTarget {
                transfer,
                ..DisplayTarget::SDR_SRGB
            });
            assert!(hdr.is_hdr_transfer());
        }
    }

    #[test]
    fn window_target_resolution_policy() {
        let requested = DisplayTarget {
            paper_white_nits: 200.0,
            peak_luminance_nits: 1000.0,
            gamut: DisplayGamut::Rec2020,
            transfer: DisplayTransfer::Pq,
            ..DisplayTarget::SDR_SRGB
        };

        // Fulfilled, unconfigured, and non-window targets pass the requested
        // value through unchanged.
        assert_eq!(
            resolve_window_display_target(requested, Some(DisplayTransfer::Pq)),
            requested
        );
        assert_eq!(resolve_window_display_target(requested, None), requested);

        // Full SDR downgrade: the whole target degrades to SDR_SRGB.
        assert_eq!(
            resolve_window_display_target(requested, Some(DisplayTransfer::Srgb)),
            DisplayTarget::SDR_SRGB
        );

        // Cross-HDR downgrade (PQ fell back to scRGB-linear): the user's
        // calibration is kept and only the transfer is replaced.
        assert_eq!(
            resolve_window_display_target(requested, Some(DisplayTransfer::ScRgbLinear)),
            DisplayTarget {
                transfer: DisplayTransfer::ScRgbLinear,
                ..requested
            }
        );

        // Only the negotiation's empty-capability arm upgrades an SDR
        // request. The resolved transfer must still flow through so the
        // encoder matches the surface.
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
