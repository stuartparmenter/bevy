//! Display-calibration provenance carriers that sit alongside the
//! user-authoritative [`DisplayTarget`](crate::DisplayTarget).
//!
//! [`DisplayTarget`](crate::DisplayTarget) is the intent: the calibration the
//! renderer encodes for. These types carry the other three provenances merged
//! with that intent: what the display can do ([`MonitorDisplayCapability`]),
//! what it is doing right now ([`WindowDisplayState`]), and which fields the
//! engine may fill in ([`DisplayCalibrationPolicy`]). The merged result is
//! [`EffectiveDisplayTarget`], which the render pipeline consumes.
//!
//! All are plain data with no renderer types, and the defaults reproduce Bevy's
//! SDR behavior exactly: [`DisplayCalibrationPolicy`] defaults to all-manual,
//! under which the renderer never overwrites a single [`DisplayTarget`] field.

use bevy_ecs::prelude::Component;

use crate::DisplayTarget;

#[cfg(feature = "bevy_reflect")]
use {
    bevy_ecs::prelude::ReflectComponent,
    bevy_reflect::{std_traits::ReflectDefault, Reflect},
};

#[cfg(all(feature = "serialize", feature = "bevy_reflect"))]
use bevy_reflect::{ReflectDeserialize, ReflectSerialize};

/// Static-per-display capability of the [`Monitor`](crate::Monitor) a window is
/// presented on: how bright it can get and how wide a gamut it covers.
///
/// Absence means "can't tell", never "SDR". When present, every field is still
/// `Option`: a platform reports whatever subset it can, and the all-`None`
/// default likewise reads as "nothing sensed". All luminance fields are
/// absolute nits (candela per square meter), achromatic CIE *Y*.
///
/// # Placement
///
/// Attached to the [`Monitor`](crate::Monitor) entity (resolved from a window
/// through its [`OnMonitor`](crate::OnMonitor) relationship), so every window
/// on the same physical display shares one capability record.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Component, Default, Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct MonitorDisplayCapability {
    /// The maximum luminance, in nits, the display can show on a small window
    /// of the panel (its peak).
    pub max_nits: Option<f32>,
    /// The maximum luminance, in nits, the display can sustain across the full
    /// panel (its full-frame ceiling), typically well below [`max_nits`]:
    /// automatic brightness limiting pulls a large bright region down.
    ///
    /// Author [`DisplayTarget::peak_luminance_nits`] from this rather than from
    /// [`max_nits`] when content fills the frame with highlights (a sky, a
    /// snowfield), so tone mapping targets a level the panel can hold. Only
    /// DXGI reports it, so it is `None` under Metal and Vulkan even on a
    /// display that has a full-frame ceiling; fall back to [`max_nits`] there.
    ///
    /// [`max_nits`]: Self::max_nits
    pub max_full_frame_nits: Option<f32>,
    /// The minimum luminance, in nits, the display can show (its black level).
    pub min_nits: Option<f32>,
    /// The coarse gamut bucket the display covers. Only DXGI (matched against
    /// the EDID primaries) and the web (CSS `color-gamut`) report one; `None`
    /// elsewhere, including on displays that do cover a wide gamut.
    pub gamut_hint: Option<crate::DisplayGamut>,
}

/// The live, drifting display state of a window's surface: the HDR headroom the
/// display can drive right now, and the nit level it maps SDR white to.
///
/// Unlike [`MonitorDisplayCapability`] (static per display) this changes while
/// the window is open: the user drags the window to another monitor or moves
/// the SDR-brightness slider, or, on the Apple EDR path, system headroom shifts
/// with ambient light, brightness, and thermal conditions. The renderer commits
/// a fresh reading only once it moves past a relative epsilon and mirrors the
/// committed value back insert-on-change, so
/// [`Changed<WindowDisplayState>`](bevy_ecs::prelude::Changed) marks a real
/// transition rather than raw read jitter.
///
/// Treat it as read-only diagnostics: writing it has no effect on the surface.
/// It is absent until the window's first successful read, and never appears at
/// all on a platform that reports nothing (Vulkan outside Win32, GLES/WebGL,
/// Android).
#[derive(Component, Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Component, Default, Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct WindowDisplayState {
    /// The linear multiplier of SDR (paper) white the display can drive before
    /// clipping, right now: wgpu's `DisplayHdrInfo::tone_map_headroom()`.
    ///
    /// The one cross-platform live HDR value, folded from whatever the backend
    /// reports: Apple's live EDR headroom, `max_nits / sdr_white_nits` on
    /// Windows, or `1.0` for a definitively-SDR display. `None` means "can't
    /// tell here", never "SDR". It is what peak-aware tone mapping targets:
    /// GT7's HDR ceiling is `peak / paper_white`, which auto-resolves to this
    /// multiplier.
    pub tone_map_headroom: Option<f32>,
    /// The luminance, in nits, of SDR reference white on this surface right now.
    ///
    /// Reported only where the platform exposes absolute nits (Windows, via
    /// `DISPLAYCONFIG_SDR_WHITE_LEVEL`; it moves with the SDR-content brightness
    /// slider). `None` on the Apple EDR and web paths, which report no absolute
    /// nits. Feeds the `paper_white` auto-calibration.
    pub sdr_white_nits: Option<f32>,
}

/// Per-field policy companion to [`DisplayTarget`]: which calibration fields
/// the engine may auto-resolve from sensed display information.
///
/// [`DisplayTarget`] stays user-authoritative; the engine never writes it. This
/// component tells the resolver which fields of the derived
/// [`EffectiveDisplayTarget`] may diverge from the authored target when the OS
/// or display reports something. A `true` field takes the sensed value when one
/// is available and the authored value when nothing is sensed; a `false` field
/// keeps the authored value verbatim. The default is all-`false`, so the
/// effective target equals the authored target field-for-field.
///
/// [`DisplayTarget::transfer`] is deliberately absent: auto-resolving the
/// transfer function would force a swapchain renegotiation. Sensing fills
/// luminance and gamut only; an OS-reported gamut mismatch may warn but never
/// rewrites the transfer.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Component, Default, Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct DisplayCalibrationPolicy {
    /// Whether to auto-resolve [`DisplayTarget::paper_white_nits`].
    pub auto_paper_white: bool,
    /// Whether to auto-resolve [`DisplayTarget::peak_luminance_nits`].
    pub auto_peak_luminance: bool,
    /// Whether to auto-resolve [`DisplayTarget::min_luminance_nits`].
    pub auto_min_luminance: bool,
    /// Whether to auto-resolve [`DisplayTarget::gamut`].
    pub auto_gamut: bool,
}

impl DisplayCalibrationPolicy {
    /// Whether any field opts into auto-resolution.
    ///
    /// When `false` the resolver is a pure identity pass, so the render-side
    /// poll uses this to skip continuous display sensing for the window.
    pub const fn has_auto(&self) -> bool {
        self.auto_paper_white
            || self.auto_peak_luminance
            || self.auto_min_luminance
            || self.auto_gamut
    }
}

/// Which provenance won the precedence ladder for one [`DisplayTarget`] field
/// in [`EffectiveDisplayTarget`].
///
/// Calibration UIs read this to show why a value is what it is: a user
/// override, an OS-sensed value, or the SDR default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Default, Debug, PartialEq, Hash, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub enum FieldProvenance {
    /// The authored [`DisplayTarget`] value: the field is not auto-resolved.
    #[default]
    User,
    /// A value sensed from the operating system / display took precedence.
    Os,
    /// The SDR sRGB fallback, used when an auto-resolved field had nothing to
    /// resolve from.
    Default,
}

/// Which provenance won for each resolvable field of an
/// [`EffectiveDisplayTarget`].
///
/// [`DisplayTarget::transfer`] has no entry: it is never auto-resolved, so its
/// provenance is always the authored value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Default, Debug, PartialEq, Hash, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct DisplayProvenance {
    /// Provenance of [`DisplayTarget::paper_white_nits`].
    pub paper_white: FieldProvenance,
    /// Provenance of [`DisplayTarget::peak_luminance_nits`].
    pub peak_luminance: FieldProvenance,
    /// Provenance of [`DisplayTarget::min_luminance_nits`].
    pub min_luminance: FieldProvenance,
    /// Provenance of [`DisplayTarget::gamut`].
    pub gamut: FieldProvenance,
}

/// The derived display target the render pipeline consumes: the
/// [`DisplayTarget`] after the resolver merges the user's intent with engine
/// policy and sensed display information, plus the per-field
/// [`DisplayProvenance`] of how each value was chosen.
///
/// A required component of [`Window`](crate::Window): every window carries one
/// from spawn (the SDR default), and the resolver rewrites it in place only
/// when the resolved value changes. Resolution happens in the main world,
/// before extraction, so the identity case (an all-manual policy, or a user-set
/// target with no sensing) has zero frame lag and a user-set-HDR project shows
/// HDR on its first frame with no SDR pop. The render pipeline reads
/// [`target`](Self::target) in place of the raw [`DisplayTarget`]; if the
/// component is removed it falls back to [`DisplayTarget::SDR_SRGB`], as it
/// does for a missing [`DisplayTarget`].
///
/// The default is the SDR sRGB target with all-[`User`](FieldProvenance::User)
/// provenance.
#[derive(Component, Debug, Default, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Component, Default, Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct EffectiveDisplayTarget {
    /// The resolved calibration the renderer encodes for.
    pub target: DisplayTarget,
    /// Per-field provenance of [`target`](Self::target).
    pub provenance: DisplayProvenance,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_default_is_all_manual() {
        let p = DisplayCalibrationPolicy::default();
        assert!(!p.auto_paper_white);
        assert!(!p.auto_peak_luminance);
        assert!(!p.auto_min_luminance);
        assert!(!p.auto_gamut);
    }

    #[test]
    fn has_auto_is_false_for_all_manual_true_for_any_auto() {
        assert!(!DisplayCalibrationPolicy::default().has_auto());
        assert!(DisplayCalibrationPolicy {
            auto_peak_luminance: true,
            ..Default::default()
        }
        .has_auto());
        assert!(DisplayCalibrationPolicy {
            auto_gamut: true,
            ..Default::default()
        }
        .has_auto());
    }

    #[test]
    fn effective_default_is_sdr_with_user_provenance() {
        let e = EffectiveDisplayTarget::default();
        assert_eq!(e.target, DisplayTarget::SDR_SRGB);
        assert_eq!(e.provenance, DisplayProvenance::default());
        assert_eq!(e.provenance.peak_luminance, FieldProvenance::User);
    }

    #[test]
    fn live_and_capability_defaults_are_all_none_not_sdr() {
        assert_eq!(WindowDisplayState::default().tone_map_headroom, None);
        assert_eq!(WindowDisplayState::default().sdr_white_nits, None);
        assert_eq!(MonitorDisplayCapability::default().max_nits, None);
        assert_eq!(
            MonitorDisplayCapability::default().max_full_frame_nits,
            None
        );
        assert_eq!(MonitorDisplayCapability::default().min_nits, None);
        assert_eq!(MonitorDisplayCapability::default().gamut_hint, None);
    }
}
