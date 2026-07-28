//! Render-world plumbing for [`DisplayTarget`].
//!
//! [`DisplayTarget`] lives on [`Window`] entities as a
//! required component (see `bevy_window`). This module covers the two cases a
//! window component cannot:
//!
//! - non-entity render targets ([`RenderTarget::Image`] and
//!   [`RenderTarget::TextureView`]), which are described by the
//!   [`ManualDisplayTargets`] resource, and
//! - the render world, where [`resolve_display_target`] provides a single
//!   lookup that view-preparation systems can use to find the
//!   [`DisplayTarget`] for any [`NormalizedRenderTarget`].
//!
//! [`RenderTarget::Image`]: bevy_camera::RenderTarget::Image
//! [`RenderTarget::TextureView`]: bevy_camera::RenderTarget::TextureView

use bevy_camera::NormalizedRenderTarget;
use bevy_ecs::{entity::ContainsEntity, prelude::*};
use bevy_platform::collections::HashMap;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render_macros::ExtractResource;
use bevy_window::{
    DisplayCalibrationPolicy, DisplayTarget, EffectiveDisplayTarget, MonitorDisplayCapability,
    OnMonitor, Window, WindowDisplayState,
};

use super::ExtractedWindows;
use crate::RenderApp;

/// Resource that stores the [`DisplayTarget`] for render targets that are not
/// backed by a [`Window`] entity.
///
/// [`RenderTarget::Window`] targets carry their [`DisplayTarget`] as a
/// (required) component on the window entity; [`RenderTarget::Image`] and
/// [`RenderTarget::TextureView`] targets have no entity to host the
/// component, so consumers look them up here instead, keyed by their
/// [`NormalizedRenderTarget`] (the same keying used by `ViewTargetAttachments`).
/// Targets without an entry — including [`RenderTarget::None`] — use the
/// default of [`DisplayTarget::SDR_SRGB`].
///
/// This type dereferences to a `HashMap<NormalizedRenderTarget, DisplayTarget>`.
/// Insert into it from the main world; an [`ExtractResourcePlugin`] clones it
/// into the render world on change, where [`resolve_display_target`] consults
/// it. The render world sees the authored value: manual targets have no surface
/// and no monitor, so calibration has nothing to sense or merge for them.
///
/// This is the "resource sidecar" half of the hybrid `DisplayTarget`
/// placement: most users only ever touch the window component, while
/// offscreen and XR-style texture-view targets opt in through this map.
///
/// [`ExtractResourcePlugin`]: crate::extract_resource::ExtractResourcePlugin
/// [`RenderTarget::Window`]: bevy_camera::RenderTarget::Window
/// [`RenderTarget::Image`]: bevy_camera::RenderTarget::Image
/// [`RenderTarget::TextureView`]: bevy_camera::RenderTarget::TextureView
/// [`RenderTarget::None`]: bevy_camera::RenderTarget::None
#[derive(Default, Clone, Debug, PartialEq, Resource, ExtractResource, Reflect)]
#[reflect(Resource, Default, Debug, PartialEq, Clone)]
#[extract_app(RenderApp)]
pub struct ManualDisplayTargets(HashMap<NormalizedRenderTarget, DisplayTarget>);

impl core::ops::Deref for ManualDisplayTargets {
    type Target = HashMap<NormalizedRenderTarget, DisplayTarget>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for ManualDisplayTargets {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Resolves the [`DisplayTarget`] for a render target in the render world.
///
/// This is the seam view-preparation code should use to parameterize
/// per-display work (tone mapping, gamut mapping, transfer encoding) for a
/// view: pass the view's [`NormalizedRenderTarget`] (e.g.
/// `ExtractedCamera::target`) together with the [`ExtractedWindows`] and
/// [`ManualDisplayTargets`] resources. All cameras rendering to the same
/// target resolve to the same `DisplayTarget`.
///
/// Resolution rules:
/// - [`NormalizedRenderTarget::Window`]: the window's extracted
///   `DisplayTarget`. `extract_windows` feeds this from the window's resolved
///   [`EffectiveDisplayTarget`], so it carries the calibration the renderer
///   actually encodes for (the authored target merged with engine policy and
///   sensed display info), not the raw authored value.
/// - [`NormalizedRenderTarget::Image`] / [`NormalizedRenderTarget::TextureView`]:
///   looked up in [`ManualDisplayTargets`] by the full
///   [`NormalizedRenderTarget`] key. The match is exact: for an image target,
///   the key includes `ImageRenderTarget::scale_factor`, so a registered
///   entry only resolves for views whose target has the same `scale_factor`
///   (and image handle) — a different `scale_factor` misses and falls back to
///   the default.
/// - [`NormalizedRenderTarget::None`], `target == None`, or any missing
///   entry: [`DisplayTarget::SDR_SRGB`].
pub fn resolve_display_target(
    target: Option<&NormalizedRenderTarget>,
    extracted_windows: &ExtractedWindows,
    manual_display_targets: &ManualDisplayTargets,
) -> DisplayTarget {
    match target {
        Some(NormalizedRenderTarget::Window(window_ref)) => extracted_windows
            .get(&window_ref.entity())
            .map(|window| window.display_target)
            .unwrap_or_default(),
        Some(
            target @ (NormalizedRenderTarget::Image(_) | NormalizedRenderTarget::TextureView(_)),
        ) => manual_display_targets
            .get(target)
            .copied()
            .unwrap_or_default(),
        Some(NormalizedRenderTarget::None { .. }) | None => DisplayTarget::SDR_SRGB,
    }
}

/// Pure calibration resolution: merges the user's authored [`DisplayTarget`]
/// with engine policy and sensed display information into the derived
/// [`EffectiveDisplayTarget`], one field at a time. No ECS, no GPU — unit-tested
/// in isolation.
pub(crate) mod policy {
    use bevy_window::{
        DisplayCalibrationPolicy, DisplayProvenance, DisplayTarget, EffectiveDisplayTarget,
        FieldProvenance, MonitorDisplayCapability, WindowDisplayState,
    };

    /// The sensed inputs the resolver may draw on for one target.
    #[derive(Default, Clone, Copy)]
    pub(crate) struct SensedInputs<'a> {
        pub capability: Option<&'a MonitorDisplayCapability>,
        pub live: Option<&'a WindowDisplayState>,
    }

    /// Resolves one [`EffectiveDisplayTarget`] from the authored target, its
    /// policy, and sensed inputs.
    ///
    /// Per field, the highest applicable source wins:
    /// 1. the authored value when the field does not opt into auto-resolution
    ///    (manual is absolute and never overridden);
    /// 2. otherwise, for an auto field: the OS-sensed value, else the authored
    ///    value (the SDR default for a default project), tagged
    ///    [`FieldProvenance::Default`].
    ///
    /// A platform that pushes calibration at the app would rank an
    /// HDR-game-interface (HGIG) override, then an engine policy, between
    /// manual and OS-sensed. Neither wgpu nor winit exposes a channel that
    /// could fill them — an HGIG push is console-SDK territory — so the ladder
    /// carries only the rungs something can reach.
    ///
    /// The transfer is never auto-resolved: it is copied verbatim and carries
    /// no provenance entry.
    pub(crate) fn resolve(
        target: DisplayTarget,
        policy: DisplayCalibrationPolicy,
        sensed: SensedInputs,
    ) -> EffectiveDisplayTarget {
        let mut out = target;
        // All `User` (the byte-identity provenance).
        let mut prov = DisplayProvenance::default();

        // Paper white resolves first: on platforms without absolute nits (Apple)
        // the OS-sensed peak is reconstructed as `paper_white * headroom`, so the
        // resolved paper white must be known before peak.
        resolve_f32(
            policy.auto_paper_white,
            &mut out.paper_white_nits,
            &mut prov.paper_white,
            target.paper_white_nits,
            sensed.live.and_then(|l| l.sdr_white_nits),
        );
        resolve_f32(
            policy.auto_peak_luminance,
            &mut out.peak_luminance_nits,
            &mut prov.peak_luminance,
            target.peak_luminance_nits,
            // The HDR-vs-SDR decision is the transfer's `is_hdr` (a capability
            // question), never the live headroom value. The transfer is never
            // auto-resolved, so `out.transfer` is the authored request.
            os_peak(&sensed, out.paper_white_nits, out.transfer.is_hdr()),
        );
        resolve_f32(
            policy.auto_min_luminance,
            &mut out.min_luminance_nits,
            &mut prov.min_luminance,
            target.min_luminance_nits,
            sensed.capability.and_then(|c| c.min_nits),
        );
        if policy.auto_gamut {
            if let Some(g) = sensed.capability.and_then(|c| c.gamut_hint) {
                out.gamut = g;
                prov.gamut = FieldProvenance::Os;
            } else {
                prov.gamut = FieldProvenance::Default;
            }
        }

        EffectiveDisplayTarget {
            target: out,
            provenance: prov,
        }
    }

    /// The OS-sensed peak luminance in nits, for an HDR target.
    ///
    /// `surface_is_hdr` is the HDR-vs-SDR decision — the transfer's
    /// [`is_hdr`](bevy_window::DisplayTransfer::is_hdr), a capability question
    /// kept separate from the live headroom *value*. On an SDR target there is no
    /// HDR peak to auto-resolve, so peak falls through the ladder to the authored
    /// value.
    ///
    /// For an HDR target the absolute peak is the platform's measured small-patch
    /// peak where it reports one (Windows DXGI `max_nits`), else reconstructed
    /// from the resolved paper white and the live
    /// [`tone_map_headroom`](WindowDisplayState::tone_map_headroom) multiplier
    /// (the Apple EDR path, which reports no absolute nits). Either way GT7's
    /// `peak / paper_white` ceiling resolves to the live headroom — and when the
    /// display has no headroom right now (`tone_map_headroom == 1.0`, e.g. macOS
    /// at full brightness) the Apple estimate is `paper_white * 1.0 == paper_white`,
    /// so highlights correctly do not exceed paper white.
    fn os_peak(sensed: &SensedInputs, paper_white_nits: f32, surface_is_hdr: bool) -> Option<f32> {
        use crate::view::window::display_state::finite_positive;
        if !surface_is_hdr {
            return None;
        }
        finite_positive(sensed.capability.and_then(|c| c.max_nits)).or_else(|| {
            let headroom = finite_positive(sensed.live.and_then(|l| l.tone_map_headroom))?;
            let paper_white = finite_positive(Some(paper_white_nits))?;
            // `finite_positive` on the product also rejects an overflow to
            // infinity from two large finite-positive factors.
            finite_positive(Some(paper_white * headroom))
        })
    }

    /// Resolves one `f32` field through the precedence ladder, writing both the
    /// resolved value and its provenance.
    fn resolve_f32(
        auto: bool,
        out: &mut f32,
        prov: &mut FieldProvenance,
        authored: f32,
        os: Option<f32>,
    ) {
        if !auto {
            *out = authored;
            *prov = FieldProvenance::User;
        } else if let Some(v) = os {
            *out = v;
            *prov = FieldProvenance::Os;
        } else {
            *out = authored;
            *prov = FieldProvenance::Default;
        }
    }
}

/// Resolves every window's [`EffectiveDisplayTarget`] in the main world,
/// *before* extraction, so the identity case has zero frame lag (a
/// user-set-HDR project shows HDR on its first frame with no SDR pop).
///
/// For each window it merges the authored [`DisplayTarget`] with its
/// [`DisplayCalibrationPolicy`] (defaulting to all-manual), the
/// [`MonitorDisplayCapability`] of the monitor it is on (resolved through
/// [`OnMonitor`]), and its live [`WindowDisplayState`]. Windows are the only
/// targets with anything to sense; [`ManualDisplayTargets`] entries reach the
/// render world as authored. The write is
/// [`set_if_neq`](bevy_ecs::change_detection::DetectChangesMut::set_if_neq), so
/// [`Changed<EffectiveDisplayTarget>`](bevy_ecs::prelude::Changed) stays a
/// usable signal and a default project is not dirtied every frame.
pub fn resolve_calibration(
    mut windows: Query<
        (
            Option<&DisplayTarget>,
            Option<&DisplayCalibrationPolicy>,
            Option<&WindowDisplayState>,
            Option<&OnMonitor>,
            &mut EffectiveDisplayTarget,
        ),
        With<Window>,
    >,
    monitors: Query<&MonitorDisplayCapability>,
) {
    for (target, policy, live, on_monitor, mut effective) in &mut windows {
        let target = target.copied().unwrap_or_default();
        let policy = policy.copied().unwrap_or_default();
        let capability = on_monitor.and_then(|m| monitors.get(m.0).ok());
        let sensed = policy::SensedInputs { capability, live };
        effective.set_if_neq(policy::resolve(target, policy, sensed));
    }
}

#[cfg(test)]
mod resolve_display_target_tests {
    use super::*;
    use bevy_asset::Handle;
    use bevy_camera::{ImageRenderTarget, ManualTextureViewHandle};
    use bevy_image::Image;
    use bevy_window::{DisplayGamut, DisplayTransfer};

    fn image_target(scale_factor: f32) -> NormalizedRenderTarget {
        NormalizedRenderTarget::Image(ImageRenderTarget {
            handle: Handle::<Image>::default(),
            scale_factor,
        })
    }

    /// A registered manual target resolves to its authored value field for
    /// field — there is no policy merge on this path.
    #[test]
    fn registered_manual_targets_resolve_to_the_authored_value() {
        let pq = DisplayTarget::SDR_SRGB
            .with_transfer(DisplayTransfer::Pq)
            .with_gamut(DisplayGamut::Rec2020)
            .with_peak(1000.0)
            .with_paper_white(203.0)
            .with_min_luminance(0.005);
        let scrgb = DisplayTarget::SDR_SRGB.with_transfer(DisplayTransfer::ScRgbLinear);

        let image = image_target(1.0);
        let texture_view = NormalizedRenderTarget::TextureView(ManualTextureViewHandle(7));
        let mut manual = ManualDisplayTargets::default();
        manual.insert(image.clone(), pq);
        manual.insert(texture_view.clone(), scrgb);

        let windows = ExtractedWindows::default();
        assert_eq!(resolve_display_target(Some(&image), &windows, &manual), pq);
        assert_eq!(
            resolve_display_target(Some(&texture_view), &windows, &manual),
            scrgb
        );
    }

    /// Every miss — an unregistered key, a key that differs only in
    /// `scale_factor`, `None` targets, and no target at all — falls back to the
    /// SDR default.
    #[test]
    fn misses_fall_back_to_sdr_srgb() {
        let mut manual = ManualDisplayTargets::default();
        manual.insert(
            image_target(1.0),
            DisplayTarget::SDR_SRGB.with_transfer(DisplayTransfer::Pq),
        );

        let windows = ExtractedWindows::default();
        // Same image handle, different scale factor: the key is the whole
        // `NormalizedRenderTarget`, so this misses.
        assert_eq!(
            resolve_display_target(Some(&image_target(2.0)), &windows, &manual),
            DisplayTarget::SDR_SRGB
        );
        assert_eq!(
            resolve_display_target(
                Some(&NormalizedRenderTarget::TextureView(
                    ManualTextureViewHandle(7)
                )),
                &windows,
                &manual
            ),
            DisplayTarget::SDR_SRGB
        );
        assert_eq!(
            resolve_display_target(
                Some(&NormalizedRenderTarget::None {
                    width: 64,
                    height: 64
                }),
                &windows,
                &manual
            ),
            DisplayTarget::SDR_SRGB
        );
        assert_eq!(
            resolve_display_target(None, &windows, &manual),
            DisplayTarget::SDR_SRGB
        );
    }
}

#[cfg(test)]
mod policy_tests {
    use super::policy::*;
    use bevy_window::{
        DisplayCalibrationPolicy, DisplayProvenance, DisplayTarget, DisplayTransfer,
        FieldProvenance, MonitorDisplayCapability, WindowDisplayState,
    };

    fn cap_with_peak(max_nits: f32) -> MonitorDisplayCapability {
        MonitorDisplayCapability {
            max_nits: Some(max_nits),
            ..Default::default()
        }
    }

    #[test]
    fn default_policy_is_identity_pass_byte_for_byte() {
        // A non-default authored target, with a capability that WOULD override
        // an auto field, under the default all-manual policy.
        let target = DisplayTarget::SDR_SRGB
            .with_peak(1000.0)
            .with_paper_white(200.0)
            .with_transfer(DisplayTransfer::Pq);
        let cap = cap_with_peak(4000.0);
        let e = resolve(
            target,
            DisplayCalibrationPolicy::default(),
            SensedInputs {
                capability: Some(&cap),
                ..Default::default()
            },
        );
        // Every field is unchanged and provenance is all-User.
        assert_eq!(e.target, target);
        assert_eq!(e.provenance, DisplayProvenance::default());
    }

    #[test]
    fn auto_peak_takes_os_when_no_higher_source() {
        // An HDR target (the `is_hdr` gate `os_peak` requires) with a measured
        // capability peak: the absolute peak wins over any headroom estimate.
        let target = DisplayTarget::SDR_SRGB
            .with_peak(1000.0)
            .with_transfer(DisplayTransfer::Pq);
        let policy = DisplayCalibrationPolicy {
            auto_peak_luminance: true,
            ..Default::default()
        };
        let cap = cap_with_peak(4000.0);
        let e = resolve(
            target,
            policy,
            SensedInputs {
                capability: Some(&cap),
                ..Default::default()
            },
        );
        assert_eq!(e.target.peak_luminance_nits, 4000.0);
        assert_eq!(e.provenance.peak_luminance, FieldProvenance::Os);
    }

    #[test]
    fn auto_peak_skipped_on_sdr_target() {
        // The HDR-vs-SDR decision is the transfer's `is_hdr`, not the sensed
        // values: on an SDR target a reported capability peak (the EDID panel
        // peak) must NOT surface as a phantom HDR peak; peak falls through to the
        // authored value.
        let target = DisplayTarget::SDR_SRGB.with_peak(1000.0); // Srgb transfer
        let policy = DisplayCalibrationPolicy {
            auto_peak_luminance: true,
            ..Default::default()
        };
        let cap = cap_with_peak(270.0);
        let e = resolve(
            target,
            policy,
            SensedInputs {
                capability: Some(&cap),
                ..Default::default()
            },
        );
        assert_eq!(e.target.peak_luminance_nits, 1000.0);
        assert_eq!(e.provenance.peak_luminance, FieldProvenance::Default);
    }

    #[test]
    fn auto_with_nothing_sensed_falls_back_to_authored_tagged_default() {
        let target = DisplayTarget::SDR_SRGB.with_peak(1000.0);
        let policy = DisplayCalibrationPolicy {
            auto_peak_luminance: true,
            ..Default::default()
        };
        let e = resolve(target, policy, SensedInputs::default());
        assert_eq!(e.target.peak_luminance_nits, 1000.0);
        assert_eq!(e.provenance.peak_luminance, FieldProvenance::Default);
    }

    #[test]
    fn transfer_is_never_resolved() {
        let target = DisplayTarget::SDR_SRGB.with_transfer(DisplayTransfer::Pq);
        // Every field auto.
        let policy = DisplayCalibrationPolicy {
            auto_paper_white: true,
            auto_peak_luminance: true,
            auto_min_luminance: true,
            auto_gamut: true,
        };
        let cap = cap_with_peak(4000.0);
        let e = resolve(
            target,
            policy,
            SensedInputs {
                capability: Some(&cap),
                ..Default::default()
            },
        );
        // The transfer is unchanged.
        assert_eq!(e.target.transfer, DisplayTransfer::Pq);
    }

    #[test]
    fn auto_paper_white_anchors_on_live_sdr_white() {
        let target = DisplayTarget::SDR_SRGB.with_paper_white(203.0);
        let policy = DisplayCalibrationPolicy {
            auto_paper_white: true,
            ..Default::default()
        };
        let live = WindowDisplayState {
            // sdr_white_nits is reported only on the absolute-nits (Windows) path.
            sdr_white_nits: Some(80.0),
            ..Default::default()
        };
        let e = resolve(
            target,
            policy,
            SensedInputs {
                live: Some(&live),
                ..Default::default()
            },
        );
        assert_eq!(e.target.paper_white_nits, 80.0);
        assert_eq!(e.provenance.paper_white, FieldProvenance::Os);
    }

    #[test]
    fn auto_peak_reconstructs_from_headroom_when_no_absolute_peak() {
        // The Apple path: no capability max_nits (Apple reports no absolute
        // nits), only a live headroom multiplier. The OS peak is reconstructed as
        // `paper_white * headroom`. Paper white is manual here, so it is the
        // authored SDR_SRGB default (100 nits): 100 * 5 = 500.
        let target = DisplayTarget::SDR_SRGB
            .with_peak(1000.0)
            .with_transfer(DisplayTransfer::ScRgbLinear);
        let policy = DisplayCalibrationPolicy {
            auto_peak_luminance: true,
            ..Default::default()
        };
        let live = WindowDisplayState {
            tone_map_headroom: Some(5.0),
            ..Default::default()
        };
        let e = resolve(
            target,
            policy,
            SensedInputs {
                live: Some(&live),
                ..Default::default()
            },
        );
        assert_eq!(e.target.peak_luminance_nits, 500.0);
        assert_eq!(e.provenance.peak_luminance, FieldProvenance::Os);
    }

    #[test]
    fn auto_peak_and_paper_white_together_on_apple_path() {
        // Both auto on the Apple path: paper white has no absolute SDR white to
        // draw from (Apple reports none), so it falls back to the authored
        // default (100, tagged Default); peak then reconstructs as 100 * 4 = 400.
        let target = DisplayTarget::SDR_SRGB.with_transfer(DisplayTransfer::ScRgbLinear);
        let policy = DisplayCalibrationPolicy {
            auto_paper_white: true,
            auto_peak_luminance: true,
            ..Default::default()
        };
        let live = WindowDisplayState {
            tone_map_headroom: Some(4.0),
            ..Default::default()
        };
        let e = resolve(
            target,
            policy,
            SensedInputs {
                live: Some(&live),
                ..Default::default()
            },
        );
        assert_eq!(
            e.target.paper_white_nits,
            DisplayTarget::SDR_SRGB.paper_white_nits
        );
        assert_eq!(e.provenance.paper_white, FieldProvenance::Default);
        assert_eq!(
            e.target.peak_luminance_nits,
            DisplayTarget::SDR_SRGB.paper_white_nits * 4.0
        );
        assert_eq!(e.provenance.peak_luminance, FieldProvenance::Os);
    }

    #[test]
    fn peak_uses_the_resolved_paper_white_not_the_authored_one() {
        // Both auto: paper white auto-resolves to the sensed SDR white (120),
        // which must then anchor the peak estimate (120 * 3 = 360). If peak
        // resolved BEFORE paper white it would use the authored 100 (-> 300), so
        // this fixture pins the ordering. Synthetic: a live SDR white with no
        // capability max_nits forces the headroom-estimate branch.
        let target = DisplayTarget::SDR_SRGB.with_transfer(DisplayTransfer::ScRgbLinear);
        let policy = DisplayCalibrationPolicy {
            auto_paper_white: true,
            auto_peak_luminance: true,
            ..Default::default()
        };
        let live = WindowDisplayState {
            sdr_white_nits: Some(120.0),
            tone_map_headroom: Some(3.0),
        };
        let e = resolve(
            target,
            policy,
            SensedInputs {
                live: Some(&live),
                ..Default::default()
            },
        );
        assert_eq!(e.target.paper_white_nits, 120.0);
        assert_eq!(e.target.peak_luminance_nits, 360.0);
        assert_eq!(e.provenance.peak_luminance, FieldProvenance::Os);
    }
}
