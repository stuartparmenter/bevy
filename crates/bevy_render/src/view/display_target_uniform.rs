//! Per-view display-target plumbing: the render-world [`ViewDisplayTarget`]
//! component and the [`DisplayTargetUniform`] GPU uniform.
//!
//! [`DisplayTarget`] is authored in the main world (as a required component of
//! `Window`, or via [`ManualDisplayTargets`] for non-window targets). This
//! module resolves it per view each frame:
//!
//! 1. [`prepare_view_display_targets`] runs in
//!    [`RenderSystems::PrepareViews`](crate::RenderSystems::PrepareViews) and
//!    inserts a [`ViewDisplayTarget`] on every extracted camera view, using
//!    [`resolve_display_target`] for the *requested* target and the window
//!    surface's negotiated transfer
//!    ([`ExtractedWindow::resolved_transfer`](super::window::ExtractedWindow::resolved_transfer))
//!    for the *resolved* one: when the surface could not fulfil the requested
//!    transfer (e.g. scRGB-linear on a backend without `Rgba16Float`
//!    surfaces), the resolved target degrades to
//!    [`DisplayTarget::SDR_SRGB`], so downgraded views take the same plain
//!    SDR path as a natively-SDR view. Views whose target cannot be resolved
//!    fall back to
//!    [`DisplayTarget::SDR_SRGB`]. All cameras rendering to the same surface
//!    resolve to the same value by construction.
//! 2. The same system also inserts the [`DisplayTargetUniform`] built from the
//!    resolved target — but only on views whose resolved transfer is HDR.
//!    [`UniformComponentPlugin`](crate::uniform::UniformComponentPlugin),
//!    registered for it in [`ViewPlugin`](super::ViewPlugin), packs those
//!    components into the [`ComponentUniforms<DisplayTargetUniform>`](crate::uniform::ComponentUniforms)
//!    dynamic uniform buffer during
//!    [`RenderSystems::PrepareResources`](crate::RenderSystems::PrepareResources)
//!    and gives each view a
//!    [`DynamicUniformIndex<DisplayTargetUniform>`](crate::uniform::DynamicUniformIndex)
//!    to address its entry with.
//!
//! Only the display-encoding pass binds the uniform and reads it to drive the
//! transfer OETF, and that pass is scheduled only for HDR-transfer targets.
//! The tone-mapping pass does not read display calibration — GT7's HDR
//! parameters are baked on the CPU into a separate per-camera uniform — so
//! plain-SDR pipelines carry no display-target binding, and SDR-only apps
//! carry no display-target uniform buffer at all.
//!
//! The matching WGSL struct lives in `display_target.wgsl` and is importable as
//! `bevy_render::display_target`.

use bevy_camera::NormalizedRenderTarget;
use bevy_ecs::prelude::*;
use bevy_log::warn_once;
use bevy_window::{DisplayGamut, DisplayTarget, DisplayTransfer};

use super::{
    window::{
        display_target::{resolve_display_target, ManualDisplayTargets},
        ExtractedWindows,
    },
    ExtractedView,
};
use crate::{camera::ExtractedCamera, render_resource::ShaderType};

/// Render-world component holding the [`DisplayTarget`] of the surface
/// (window, image, or manual texture view) a view renders to, in both its
/// *requested* and *resolved* forms.
///
/// Inserted by [`prepare_view_display_targets`] on every extracted view that
/// has an [`ExtractedCamera`]; falls back to [`DisplayTarget::SDR_SRGB`] when
/// the camera's render target has no explicit display target. Views without a
/// camera (e.g. shadow views) do not receive this component; consumers should
/// treat a missing component as [`DisplayTarget::SDR_SRGB`].
///
/// Prepare-time systems (tonemapping pipeline specialization, operator uniform
/// preparation, the display-encoding pass, and the upscaling blit) read the
/// [`resolved`](Self::resolved) target instead of re-resolving the render
/// target themselves, so they always agree on whether a view takes the HDR
/// path — and they key on what the surface can actually show, never on an
/// unfulfilled request.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct ViewDisplayTarget {
    /// The display target this view's render target resolves to (a window's
    /// `EffectiveDisplayTarget`, or a manual target's `ManualDisplayTargets`
    /// entry), before surface negotiation.
    ///
    /// Useful for diagnostics and for re-resolution logic; rendering systems
    /// should use [`resolved`](Self::resolved).
    pub requested: DisplayTarget,
    /// The display target after surface negotiation.
    ///
    /// Equal to [`requested`](Self::requested) when the surface fulfils the
    /// requested transfer (or the target is not a window surface, where the
    /// user owns the texture format). When the requested transfer had to be
    /// downgraded (see `negotiate_surface_format` in `view::window`), this
    /// is [`DisplayTarget::SDR_SRGB`] for a full SDR downgrade (so the
    /// downgraded view takes the same plain SDR path as a natively-SDR view),
    /// or the requested target with only the transfer replaced when the surface
    /// carries a *different HDR* transfer (PQ downgraded to scRGB-linear) —
    /// the user's calibration still applies.
    /// See `resolve_window_display_target` in this module.
    pub resolved: DisplayTarget,
}

impl ViewDisplayTarget {
    /// Creates a `ViewDisplayTarget` whose resolved target equals the
    /// requested one (no surface-side downgrade).
    pub fn fulfilled(target: DisplayTarget) -> Self {
        Self {
            requested: target,
            resolved: target,
        }
    }

    /// Returns `true` if the **resolved** transfer function is a high dynamic
    /// range transfer (see [`DisplayTransfer::is_hdr`]).
    ///
    /// This gates the display-encoding pass and the upscaling blit's
    /// pass-through mode, and HDR-capable operators (e.g.
    /// `Tonemapping::GranTurismo7`) use it to pick their HDR mode at prepare
    /// time. Because it reads the resolved transfer, a view whose HDR request
    /// was downgraded behaves exactly like a plain SDR view.
    pub fn is_hdr_transfer(&self) -> bool {
        self.resolved.transfer.is_hdr()
    }
}

/// GPU uniform carrying a view's resolved [`DisplayTarget`] calibration.
///
/// The WGSL counterpart is `DisplayTargetUniform` in
/// `bevy_render::display_target` (`display_target.wgsl`); the two must stay
/// field-for-field in sync.
///
/// The [`gamut`](Self::gamut) and [`transfer`](Self::transfer) enums are
/// encoded as `u32` indices:
///
/// | `gamut` | meaning | | `transfer` | meaning |
/// |---|---|---|---|---|
/// | 0 | Rec.709 | | 0 | sRGB |
/// | 1 | Display P3 | | 1 | scRGB linear |
/// | 2 | Rec.2020 | | 2 | PQ (ST 2084) |
/// | | | | 3 | (reserved for HLG) |
/// | | | | 4 | extended sRGB (encoded) |
///
/// Gamut conversion matrices are deliberately **not** part of this uniform;
/// the gamut-transform stage of the display-encoding pass derives them per
/// pipeline.
///
/// The [`From<DisplayTarget>`] conversion copies values verbatim, but
/// [`prepare_view_display_targets`] sanitizes
/// [`paper_white_nits`](Self::paper_white_nits) through
/// [`DisplayTarget::sanitized_paper_white_nits`] before writing — every GPU
/// consumer that multiplies by paper white (the display encoder's transfer
/// encoding in particular) must fold the *same* value the tone-mapping
/// operators fold into their seam renormalization, or the two factors stop
/// cancelling (a `NaN`/zero/negative paper white would otherwise `NaN` or
/// black out the encoded frame, and a value above the 10000-nit PQ ceiling
/// would silently disagree with GT7's clamped renormalization). The remaining
/// fields are unvalidated; consumers with hard numeric requirements (e.g. the
/// GT7 operator's HDR peak range) sanitize at their own prepare step.
#[derive(Component, Clone, Copy, Debug, PartialEq, ShaderType)]
pub struct DisplayTargetUniform {
    /// [`DisplayTarget::paper_white_nits`]: the luminance, in nits, that
    /// `1.0` at the tone-map operator output corresponds to.
    pub paper_white_nits: f32,
    /// [`DisplayTarget::peak_luminance_nits`]: the display's maximum
    /// luminance in nits.
    pub peak_luminance_nits: f32,
    /// [`DisplayTarget::min_luminance_nits`]: the display's black level in
    /// nits.
    pub min_luminance_nits: f32,
    /// The display gamut as a `u32` index (see the type docs).
    pub gamut: u32,
    /// The (resolved) transfer function as a `u32` index (see the type
    /// docs).
    pub transfer: u32,
}

impl From<DisplayTarget> for DisplayTargetUniform {
    fn from(target: DisplayTarget) -> Self {
        Self {
            paper_white_nits: target.paper_white_nits,
            peak_luminance_nits: target.peak_luminance_nits,
            min_luminance_nits: target.min_luminance_nits,
            gamut: match target.gamut {
                DisplayGamut::Rec709 => 0,
                DisplayGamut::DisplayP3 => 1,
                DisplayGamut::Rec2020 => 2,
            },
            // Index 3 is reserved for a future HLG transfer, so
            // `ExtendedSrgb` keeps index 4.
            transfer: match target.transfer {
                DisplayTransfer::Srgb => 0,
                DisplayTransfer::ScRgbLinear => 1,
                DisplayTransfer::Pq => 2,
                DisplayTransfer::ExtendedSrgb => 4,
            },
        }
    }
}

/// Resolves a window view's display target from the transfer the configured
/// surface actually carries
/// ([`ExtractedWindow::resolved_transfer`](super::window::ExtractedWindow::resolved_transfer)).
///
/// - `Some(`[`DisplayTransfer::Srgb`]`)` while an HDR transfer was requested
///   (a full SDR downgrade): the **whole** target degrades to
///   [`DisplayTarget::SDR_SRGB`], not just the transfer field, so the view
///   takes the same plain SDR path as a natively-SDR view (warn + degrade;
///   the warning is emitted at negotiation time in `create_surfaces`).
/// - `Some(transfer)` differing from the requested transfer while both are
///   HDR (the negotiation fulfilled the request with a different HDR
///   encoding: PQ downgraded to scRGB-linear when HDR10 is unavailable): the
///   user's calibration (paper white, peak, gamut) is kept and only the
///   transfer is replaced, so the encoder keys on what the surface actually
///   carries.
/// - Equal transfers, `None` (surface not configured yet — transient), or a
///   non-window target (`surface_transfer` = `None`): resolved == requested.
pub(crate) fn resolve_window_display_target(
    requested: DisplayTarget,
    surface_transfer: Option<DisplayTransfer>,
) -> DisplayTarget {
    match surface_transfer {
        // The surface negotiation downgraded the request all the way to SDR:
        // degrade the whole target to the plain SDR default so the view
        // takes the same SDR path as a natively-SDR view.
        Some(DisplayTransfer::Srgb) if requested.transfer != DisplayTransfer::Srgb => {
            DisplayTarget::SDR_SRGB
        }
        // The surface carries a different transfer than requested: keep the
        // calibration, swap the transfer.
        Some(transfer) if transfer != requested.transfer => DisplayTarget {
            transfer,
            ..requested
        },
        // Surface fulfils the request, the target is not a window, or the
        // surface is not configured yet (transient): no change.
        _ => requested,
    }
}

/// Resolves the [`ViewDisplayTarget`] for a camera's normalized render
/// target: the target's [`DisplayTarget`] (a window's
/// `EffectiveDisplayTarget`, or a manual target's [`ManualDisplayTargets`]
/// entry; see [`resolve_display_target`]) with the window surface's negotiated
/// transfer folded in (see `resolve_window_display_target` in this module).
///
/// Single source shared by [`prepare_view_display_targets`] (which inserts
/// the per-view component after surface negotiation, so the resolved
/// transfer is this frame's) and camera extraction (`extract_cameras`, which
/// picks the main-texture format — there the surface transfer is the
/// *previous* frame's negotiation result, exactly as fresh as the swapchain
/// format the extraction already reads for the output format).
pub(crate) fn resolve_view_display_target(
    target: Option<&NormalizedRenderTarget>,
    extracted_windows: &ExtractedWindows,
    manual_display_targets: &ManualDisplayTargets,
) -> ViewDisplayTarget {
    let requested = resolve_display_target(target, extracted_windows, manual_display_targets);

    let surface_transfer = match target {
        Some(NormalizedRenderTarget::Window(window_ref)) => extracted_windows
            .get(&window_ref.entity())
            .and_then(|window| window.resolved_transfer),
        _ => None,
    };
    let resolved = resolve_window_display_target(requested, surface_transfer);

    ViewDisplayTarget {
        requested,
        resolved,
    }
}

/// Resolves and inserts a [`ViewDisplayTarget`] on every extracted view that
/// has an [`ExtractedCamera`], plus the matching [`DisplayTargetUniform`] on
/// views whose resolved transfer is HDR.
///
/// Only the display-encoding pass binds the uniform, and that pass runs only
/// for HDR-transfer views (`resolve_group_encode_parameters` in
/// `bevy_core_pipeline` gates on the same [`ViewDisplayTarget::is_hdr_transfer`]
/// predicate), so SDR views skip the insert and contribute no entry to the
/// [`ComponentUniforms<DisplayTargetUniform>`](crate::uniform::ComponentUniforms)
/// buffer — a plain-SDR app never creates it at all. A view whose target
/// drops from HDR to SDR has its uniform removed; the
/// [`DynamicUniformIndex`](crate::uniform::DynamicUniformIndex) left behind
/// goes stale, which is harmless because the display-encoding pass bails on
/// its missing pipeline before reading the index.
///
/// Runs in [`RenderSystems::PrepareViews`](crate::RenderSystems::PrepareViews)
/// — after `create_surfaces`, so the window surface's negotiated transfer is
/// fresh — so later prepare systems (pipeline specialization, uniform
/// packing) can rely on the components being present. The
/// [`DisplayTargetUniform`] in particular must be inserted before
/// [`UniformComponentPlugin`](crate::uniform::UniformComponentPlugin) packs it
/// in [`RenderSystems::PrepareResources`](crate::RenderSystems::PrepareResources).
///
/// Resolution policy (`resolve_view_display_target`):
/// - **Window targets** go through surface negotiation; see
///   `resolve_window_display_target` (this module) for how the surface's
///   resolved transfer maps onto the requested target.
/// - **Image / manual-texture-view targets** resolve to the requested value
///   unchanged: there is no surface negotiation, the user owns the texture
///   and its format.
///
/// [`DisplayTargetUniform::paper_white_nits`] is sanitized through
/// [`DisplayTarget::sanitized_paper_white_nits`] (non-finite / non-positive →
/// 100 nits, clamped to the 10000-nit PQ ceiling; valid values pass through
/// bit-for-bit) with a `warn_once!` when the authored value had to be
/// replaced. This keeps the GPU-side paper white single-sourced with the
/// value the tone-mapping operators (e.g. GT7's `sdr_correction_factor`)
/// fold at their own prepare step, so the paper-white factors — operator
/// output × `100 / paper_white`, encoder × `paper_white / 80` (scRGB) or
/// `× paper_white` (PQ) — cancel for every authored input.
pub fn prepare_view_display_targets(
    mut commands: Commands,
    extracted_windows: Res<ExtractedWindows>,
    manual_display_targets: Res<ManualDisplayTargets>,
    views: Query<(Entity, &ExtractedCamera, Has<DisplayTargetUniform>), With<ExtractedView>>,
) {
    for (entity, camera, has_uniform) in &views {
        let view_display_target = resolve_view_display_target(
            camera.target.as_ref(),
            &extracted_windows,
            &manual_display_targets,
        );
        let authored = view_display_target.resolved.paper_white_nits;
        let sanitized = view_display_target.resolved.sanitized_paper_white_nits();
        // Sanitize and warn for every view, not just HDR ones: SDR consumers
        // (bloom's nits-denominated threshold) fold the sanitized value too,
        // and this is the only diagnostic a plain-SDR project gets.
        // Bit comparison: valid values pass through bit-for-bit, and a NaN
        // input (NaN != NaN) is still detected as replaced.
        if sanitized.to_bits() != authored.to_bits() {
            warn_once!(
                "DisplayTarget::paper_white_nits ({}) is non-finite, non-positive, or above \
                 the 10000-nit PQ ceiling; the display pipeline is using {} nits instead",
                authored,
                sanitized
            );
        }
        if view_display_target.is_hdr_transfer() {
            let uniform = DisplayTargetUniform {
                paper_white_nits: sanitized,
                ..DisplayTargetUniform::from(view_display_target.resolved)
            };
            commands
                .entity(entity)
                .insert((view_display_target, uniform));
        } else {
            let mut entity_commands = commands.entity(entity);
            entity_commands.insert(view_display_target);
            // Only queue the removal when there is a uniform to remove
            // (an HDR view whose target dropped to SDR); a blanket remove
            // would cost one command per camera per frame on SDR projects.
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

    /// The uniform is inserted only on views whose resolved transfer is HDR;
    /// SDR views get the [`ViewDisplayTarget`] alone, and a stale uniform
    /// from a previous HDR resolution is removed.
    #[test]
    fn uniform_inserted_only_for_hdr_transfer_views() {
        let mut world = World::new();
        world.init_resource::<ExtractedWindows>();

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
                DisplayTargetUniform::from(DisplayTarget::SDR_SRGB),
            ))
            .id();

        world.run_system_once(prepare_view_display_targets).unwrap();

        for entity in [sdr, hdr, downgraded] {
            assert!(world.entity(entity).contains::<ViewDisplayTarget>());
        }
        assert!(!world.entity(sdr).contains::<DisplayTargetUniform>());
        assert!(world.entity(hdr).contains::<DisplayTargetUniform>());
        assert!(!world.entity(downgraded).contains::<DisplayTargetUniform>());
    }

    #[test]
    fn uniform_copies_display_target_verbatim() {
        let target = DisplayTarget {
            paper_white_nits: 203.0,
            peak_luminance_nits: 1000.0,
            min_luminance_nits: 0.005,
            gamut: DisplayGamut::Rec2020,
            transfer: DisplayTransfer::ScRgbLinear,
        };
        let uniform = DisplayTargetUniform::from(target);
        assert_eq!(uniform.paper_white_nits, 203.0);
        assert_eq!(uniform.peak_luminance_nits, 1000.0);
        assert_eq!(uniform.min_luminance_nits, 0.005);
        assert_eq!(uniform.gamut, 2);
        assert_eq!(uniform.transfer, 1);
    }

    #[test]
    fn default_target_uniform_is_sdr_srgb() {
        let uniform = DisplayTargetUniform::from(DisplayTarget::SDR_SRGB);
        assert_eq!(uniform.paper_white_nits, 100.0);
        assert_eq!(uniform.peak_luminance_nits, 100.0);
        assert_eq!(uniform.min_luminance_nits, 0.0);
        assert_eq!(uniform.gamut, 0);
        assert_eq!(uniform.transfer, 0);
    }

    #[test]
    fn view_display_target_is_hdr_transfer() {
        let sdr = ViewDisplayTarget::fulfilled(DisplayTarget::SDR_SRGB);
        assert!(!sdr.is_hdr_transfer());

        // A non-transfer field change does not flip the HDR predicate.
        let brighter = ViewDisplayTarget::fulfilled(DisplayTarget {
            paper_white_nits: 203.0,
            ..DisplayTarget::SDR_SRGB
        });
        assert!(!brighter.is_hdr_transfer());

        for transfer in [
            DisplayTransfer::ScRgbLinear,
            DisplayTransfer::Pq,
            DisplayTransfer::ExtendedSrgb,
        ] {
            let hdr = ViewDisplayTarget::fulfilled(DisplayTarget {
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

        // Fulfilled, unconfigured (transient), and non-window targets: the
        // requested value passes through unchanged.
        assert_eq!(
            resolve_window_display_target(requested, Some(DisplayTransfer::Pq)),
            requested
        );
        assert_eq!(resolve_window_display_target(requested, None), requested);

        // Full SDR downgrade: the WHOLE target degrades to SDR_SRGB.
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

        // An SDR request is never "upgraded" silently except by the
        // defensive empty-capability arm of the negotiation, in which case
        // the resolved transfer must still flow through so the encoder
        // matches the surface.
        let sdr = DisplayTarget::SDR_SRGB;
        assert_eq!(
            resolve_window_display_target(sdr, Some(DisplayTransfer::Pq)),
            DisplayTarget {
                transfer: DisplayTransfer::Pq,
                ..sdr
            }
        );
    }

    #[test]
    fn downgraded_target_takes_the_plain_sdr_path() {
        // A view whose HDR request was downgraded at surface negotiation
        // (resolved = SDR_SRGB) must be indistinguishable from a plain SDR
        // view to every predicate, regardless of what was requested.
        let downgraded = ViewDisplayTarget {
            requested: DisplayTarget {
                paper_white_nits: 200.0,
                peak_luminance_nits: 1000.0,
                transfer: DisplayTransfer::ScRgbLinear,
                ..DisplayTarget::SDR_SRGB
            },
            resolved: DisplayTarget::SDR_SRGB,
        };
        assert!(!downgraded.is_hdr_transfer());
        assert_eq!(
            DisplayTargetUniform::from(downgraded.resolved),
            DisplayTargetUniform::from(DisplayTarget::SDR_SRGB)
        );
    }
}
