//! Render-world plumbing for [`DisplayTarget`].
//!
//! [`DisplayTarget`] is a required component on [`Window`] entities. This module
//! covers the two cases that component cannot: non-entity render targets, which
//! [`ManualDisplayTargets`] describes, and the render world, where
//! [`resolve_display_target`] looks up the [`DisplayTarget`] for any
//! [`NormalizedRenderTarget`].

use bevy_camera::NormalizedRenderTarget;
use bevy_ecs::{entity::ContainsEntity, prelude::*};
use bevy_platform::collections::HashMap;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render_macros::ExtractResource;
use bevy_window::{
    DisplayCalibrationPolicy, DisplayTarget, EffectiveDisplayTarget, MonitorDisplayCapability,
    OnMonitor, Window, WindowDisplayState,
};

use super::ExtractedWindow;
use crate::RenderApp;

/// Resource that stores the [`DisplayTarget`] for render targets that are not
/// backed by a [`Window`] entity, keyed by [`NormalizedRenderTarget`].
///
/// Insert into it from the main world. The render world sees the authored value:
/// manual targets have no surface and no monitor, so calibration has nothing to
/// sense or merge for them.
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
/// `target` is the view's [`NormalizedRenderTarget`], for example
/// `ExtractedCamera::target`.
///
/// A window resolves to its extracted `DisplayTarget`, which `extract_windows`
/// feeds from the window's resolved [`EffectiveDisplayTarget`], so it carries
/// the calibration the renderer encodes for, not the authored value. An image or
/// texture view resolves through [`ManualDisplayTargets`]; the key is the whole
/// [`NormalizedRenderTarget`], so an image entry only matches views with the
/// same image handle and `ImageRenderTarget::scale_factor`. Everything else,
/// including a missing entry, resolves to [`DisplayTarget::SDR_SRGB`].
pub fn resolve_display_target<'a>(
    target: Option<&NormalizedRenderTarget>,
    windows: impl IntoIterator<Item = (Entity, &'a ExtractedWindow)>,
    manual_display_targets: &ManualDisplayTargets,
) -> DisplayTarget {
    match target {
        Some(NormalizedRenderTarget::Window(window_ref)) => windows
            .into_iter()
            .find(|(entity, _)| *entity == window_ref.entity())
            .map(|(_, window)| window.display_target)
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

/// Calibration resolution. No ECS and no GPU, so it can be unit tested on its
/// own.
pub(crate) mod policy {
    use bevy_window::{
        DisplayCalibrationPolicy, DisplayProvenance, DisplayTarget, EffectiveDisplayTarget,
        FieldProvenance, MonitorDisplayCapability, WindowDisplayState,
    };

    #[derive(Default, Clone, Copy)]
    pub(crate) struct SensedInputs<'a> {
        pub capability: Option<&'a MonitorDisplayCapability>,
        pub live: Option<&'a WindowDisplayState>,
    }

    /// Resolves one [`EffectiveDisplayTarget`] from the authored target, its
    /// policy, and sensed inputs.
    ///
    /// Per field: the authored value when the field is not auto, else the
    /// OS-sensed value, else the authored value tagged
    /// [`FieldProvenance::Default`]. wgpu and winit expose no channel for a
    /// platform-pushed calibration override (HGIG), so there is no rung for one.
    ///
    /// The transfer is never auto-resolved. It is copied verbatim and has no
    /// provenance entry.
    pub(crate) fn resolve(
        target: DisplayTarget,
        policy: DisplayCalibrationPolicy,
        sensed: SensedInputs,
    ) -> EffectiveDisplayTarget {
        let mut out = target;
        let mut prov = DisplayProvenance::default();

        // Paper white resolves first: on platforms with no absolute nits (Apple)
        // the OS-sensed peak is `paper_white * headroom`.
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
    /// An SDR target has no HDR peak to auto-resolve, so peak falls through to
    /// the authored value.
    ///
    /// For an HDR target the peak is the platform's measured small-patch peak
    /// where it reports one (Windows DXGI `max_nits`), else the resolved paper
    /// white times the live
    /// [`tone_map_headroom`](WindowDisplayState::tone_map_headroom) (the Apple
    /// EDR path, which reports no absolute nits).
    fn os_peak(sensed: &SensedInputs, paper_white_nits: f32, surface_is_hdr: bool) -> Option<f32> {
        use crate::view::window::display_state::finite_positive;
        if !surface_is_hdr {
            return None;
        }
        finite_positive(sensed.capability.and_then(|c| c.max_nits)).or_else(|| {
            let headroom = finite_positive(sensed.live.and_then(|l| l.tone_map_headroom))?;
            let paper_white = finite_positive(Some(paper_white_nits))?;
            // `finite_positive` on the product also catches an overflow to
            // infinity from two large factors.
            finite_positive(Some(paper_white * headroom))
        })
    }

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

/// Resolves every window's [`EffectiveDisplayTarget`] in the main world, before
/// extraction, so a window that authored HDR shows HDR on its first frame, with
/// no SDR pop.
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

    fn no_windows() -> core::iter::Empty<(Entity, &'static ExtractedWindow)> {
        core::iter::empty()
    }

    fn image_target(scale_factor: f32) -> NormalizedRenderTarget {
        NormalizedRenderTarget::Image(ImageRenderTarget {
            handle: Handle::<Image>::default(),
            scale_factor,
        })
    }

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

        assert_eq!(
            resolve_display_target(Some(&image), no_windows(), &manual),
            pq
        );
        assert_eq!(
            resolve_display_target(Some(&texture_view), no_windows(), &manual),
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
            resolve_display_target(Some(&image_target(2.0)), no_windows(), &manual),
            DisplayTarget::SDR_SRGB
        );
        assert_eq!(
            resolve_display_target(
                Some(&NormalizedRenderTarget::TextureView(
                    ManualTextureViewHandle(7)
                )),
                no_windows(),
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
                no_windows(),
                &manual
            ),
            DisplayTarget::SDR_SRGB
        );
        assert_eq!(
            resolve_display_target(None, no_windows(), &manual),
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
        // On an SDR target a reported capability peak (the EDID panel peak) must
        // NOT surface as a phantom HDR peak.
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
        // The Apple path: no capability max_nits, only a live headroom
        // multiplier. Paper white is manual here, so it stays at the authored
        // SDR_SRGB default of 100 nits: 100 * 5 = 500.
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
        // Apple reports no absolute SDR white, so paper white falls back to the
        // authored default, tagged Default, and anchors the peak.
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
        // Both auto: paper white resolves to the sensed SDR white (120), which
        // must then anchor the peak estimate (120 * 3 = 360). If peak resolved
        // BEFORE paper white it would use the authored 100 and give 300, so this
        // fixture pins the ordering. A live SDR white with no capability
        // max_nits is synthetic; it forces the headroom-estimate branch.
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
