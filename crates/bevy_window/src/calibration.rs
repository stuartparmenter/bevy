//! Display-calibration provenance carriers that sit alongside the
//! user-authoritative [`DisplayTarget`](crate::DisplayTarget).
//!
//! [`DisplayTarget`](crate::DisplayTarget) is the *intent*: the calibration the
//! renderer encodes for. These types carry the other three provenances the
//! renderer merges with that intent — what the display can do
//! ([`MonitorDisplayCapability`]), what it is doing right now
//! ([`WindowDisplayState`]), and which fields the engine is allowed to fill in
//! ([`DisplayCalibrationPolicy`]) — and the merged result
//! ([`EffectiveDisplayTarget`]) the render pipeline actually consumes.
//!
//! All of these are plain data: they carry no renderer types, and the defaults
//! reproduce Bevy's SDR behavior exactly. A project that touches none of them
//! renders byte-identically to one that does not know they exist:
//! [`DisplayCalibrationPolicy`] defaults to all-[`Keep`](AutoField::Keep), under
//! which the renderer never overwrites a single [`DisplayTarget`] field.

use bevy_ecs::prelude::Component;

use crate::DisplayTarget;

#[cfg(feature = "bevy_reflect")]
use {
    bevy_ecs::prelude::ReflectComponent,
    bevy_reflect::{std_traits::ReflectDefault, Reflect},
};

#[cfg(all(feature = "serialize", feature = "bevy_reflect"))]
use bevy_reflect::{ReflectDeserialize, ReflectSerialize};

/// Where a piece of display information came from.
///
/// Sensed values flow from the windowing system or the graphics backend; the
/// renderer keeps the source purely as provenance, so calibration UIs can
/// distinguish a value the OS reported in absolute nits (Windows) from one
/// derived from a relative EDR headroom (Apple) or a coarse web capability. It
/// does not affect how a value commits — every continuous field is smoothed the
/// same way.
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
pub enum DisplayInfoSource {
    /// No information is available: the platform or the moment cannot tell us.
    /// This is the default and reads as "not sensed", never as "SDR".
    #[default]
    Unknown,
    /// The operating system / windowing system reported the value directly
    /// (for example a Windows HDR enablement flag or absolute luminance from
    /// display metadata).
    Os,
    /// Derived from a relative EDR-headroom signal (the Apple path), which
    /// reports a unitless multiplier over SDR white and no absolute nits.
    DerivedFromHeadroom,
    /// Reported by the web platform's coarse `dynamic-range` / `color-gamut`
    /// capability query (a capability, not a live mode).
    Web,
}

/// Static-per-display capability of the [`Monitor`](crate::Monitor) a window is
/// presented on: how bright it can get and how wide a gamut it covers.
///
/// Its **absence** means "can't tell" — the renderer never treats a missing
/// component as "SDR". When present, every field is still `Option`: a platform
/// reports whatever subset it can. All luminance fields are absolute nits
/// (candela per square meter), achromatic CIE *Y*.
///
/// # Placement
///
/// Attached to the [`Monitor`](crate::Monitor) entity (resolved from a window
/// through its [`OnMonitor`](crate::OnMonitor) relationship), so every window
/// on the same physical display shares one capability record.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
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
    /// [`max_nits`] when the content fills the frame with highlights — a sky, a
    /// snowfield — so tone mapping targets a level the panel can hold. Only
    /// DXGI reports it, so it is `None` under Metal and Vulkan even on a display
    /// that has a full-frame ceiling; fall back to [`max_nits`] there.
    ///
    /// [`max_nits`]: Self::max_nits
    pub max_full_frame_nits: Option<f32>,
    /// The minimum luminance, in nits, the display can show (its black level).
    pub min_nits: Option<f32>,
    /// The coarse gamut bucket the display covers. Only DXGI (matched against
    /// the EDID primaries) and the web (CSS `color-gamut`) report one; `None`
    /// elsewhere, including on displays that do cover a wide gamut.
    pub gamut_hint: Option<crate::DisplayGamut>,
    /// Where this capability record came from.
    pub source: DisplayInfoSource,
}

impl Default for MonitorDisplayCapability {
    /// An all-`None` capability with an [`Unknown`](DisplayInfoSource::Unknown)
    /// source: "nothing sensed yet", never "SDR".
    fn default() -> Self {
        Self {
            max_nits: None,
            max_full_frame_nits: None,
            min_nits: None,
            gamut_hint: None,
            source: DisplayInfoSource::Unknown,
        }
    }
}

/// The live, drifting display state of a window's surface: how much HDR headroom
/// the display can drive right now, and the nit level it maps SDR white to.
///
/// Unlike [`MonitorDisplayCapability`] (static per display) this changes while
/// the window is open — the user drags the window to another monitor, moves the
/// SDR-brightness slider, or (on the Apple EDR path) the system headroom shifts
/// with ambient light, brightness, and thermal conditions. The renderer mirrors
/// it back to the main world insert-on-change, so
/// [`Changed<WindowDisplayState>`](bevy_ecs::prelude::Changed) is a real signal:
/// [`generation`](Self::generation) is bumped only when a value actually commits
/// past the renderer's epsilon, never on raw read jitter.
///
/// Treat it as read-only diagnostics: writing it has no effect on the surface.
/// It is absent until the window's first successful read.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
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
    /// clipping, **right now** — wgpu's `DisplayHdrInfo::tone_map_headroom()`.
    ///
    /// This is the one cross-platform live HDR value, folded from whatever the
    /// backend reports: Apple's live EDR headroom, or `max_nits / sdr_white_nits`
    /// on Windows, or `1.0` for a definitively-SDR display. `None` means "can't
    /// tell here", never "SDR". It is exactly what peak-aware tone mapping
    /// targets — GT7's HDR ceiling is `peak / paper_white`, which auto-resolves to
    /// this multiplier.
    pub tone_map_headroom: Option<f32>,
    /// The luminance, in nits, of SDR reference white on this surface right now.
    ///
    /// Reported only where the platform exposes absolute nits (Windows, via
    /// `DISPLAYCONFIG_SDR_WHITE_LEVEL`; it moves with the SDR-content brightness
    /// slider). `None` on the Apple EDR and web paths, which report no absolute
    /// nits. Feeds the `paper_white` auto-calibration.
    pub sdr_white_nits: Option<f32>,
    /// Where this live state came from.
    pub source: DisplayInfoSource,
    /// Bumped each time a field commits past the renderer's epsilon, so
    /// [`Changed<WindowDisplayState>`](bevy_ecs::prelude::Changed) signals a
    /// real transition rather than read jitter.
    pub generation: u32,
}

impl Default for WindowDisplayState {
    /// An all-`None` state with an [`Unknown`](DisplayInfoSource::Unknown)
    /// source and generation `0`: "nothing sensed yet", never "SDR".
    fn default() -> Self {
        Self {
            tone_map_headroom: None,
            sdr_white_nits: None,
            source: DisplayInfoSource::Unknown,
            generation: 0,
        }
    }
}

/// Whether the engine may auto-resolve a [`DisplayTarget`] field from sensed
/// display information, or must keep the user's authored value.
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
pub enum AutoField {
    /// Keep the user's authored [`DisplayTarget`] value verbatim; the engine
    /// never overwrites it. This is the default, under which calibration
    /// resolution is an identity pass and output is byte-identical to a project
    /// that never sensed the display.
    #[default]
    Keep,
    /// Let the engine fill this field from sensed display information when a
    /// value is available, falling back through the precedence ladder to the
    /// authored value when nothing is sensed.
    Auto,
}

/// Per-field policy companion to [`DisplayTarget`]: which calibration fields the
/// engine may auto-resolve from sensed display information.
///
/// [`DisplayTarget`] stays user-authoritative — the engine never writes it.
/// This component instead tells the *resolver* which fields of the derived
/// [`EffectiveDisplayTarget`] may diverge from the authored target when the OS
/// or display reports something. The default is all-[`Keep`](AutoField::Keep):
/// the effective target equals the authored target field-for-field, so a
/// project that adds this component (or never does) renders identically.
///
/// [`DisplayTarget::transfer`] is **deliberately absent**: the transfer
/// function is never auto-resolved, because changing it would force a swapchain
/// renegotiation. Sensing fills luminance and gamut only; an OS-reported gamut
/// mismatch may warn but never rewrites the transfer.
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
    pub paper_white: AutoField,
    /// Whether to auto-resolve [`DisplayTarget::peak_luminance_nits`].
    pub peak_luminance: AutoField,
    /// Whether to auto-resolve [`DisplayTarget::min_luminance_nits`].
    pub min_luminance: AutoField,
    /// Whether to auto-resolve [`DisplayTarget::gamut`].
    pub gamut: AutoField,
}

impl DisplayCalibrationPolicy {
    /// Whether any field opts into [`Auto`](AutoField::Auto).
    ///
    /// When this is `false` the resolver is a pure identity pass, so the renderer
    /// has no reason to keep re-reading the live display state for this window —
    /// the gate the render-side poll uses to skip continuous sensing on
    /// all-[`Keep`](AutoField::Keep) projects.
    pub const fn has_auto(&self) -> bool {
        matches!(self.paper_white, AutoField::Auto)
            || matches!(self.peak_luminance, AutoField::Auto)
            || matches!(self.min_luminance, AutoField::Auto)
            || matches!(self.gamut, AutoField::Auto)
    }
}

/// For each [`DisplayTarget`] field, which provenance won the precedence ladder
/// in [`EffectiveDisplayTarget`].
///
/// Calibration UIs read this to show *why* a value is what it is — a user
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
    /// The authored [`DisplayTarget`] value (the field's policy is
    /// [`Keep`](AutoField::Keep), or no higher source applied). This is the
    /// default and the byte-identity case.
    #[default]
    User,
    /// A value sensed from the operating system / display took precedence.
    Os,
    /// The SDR sRGB fallback, used when an [`Auto`](AutoField::Auto) field had
    /// nothing to resolve from.
    Default,
}

/// Which provenance won for each field of an [`EffectiveDisplayTarget`].
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
    /// Provenance of [`DisplayTarget::transfer`]. Always
    /// [`User`](FieldProvenance::User): the transfer is never auto-resolved.
    pub transfer: FieldProvenance,
}

/// The derived display target the render pipeline consumes: the
/// [`DisplayTarget`] after the resolver merges the user's intent with engine
/// policy and sensed display information, plus the per-field
/// [`DisplayProvenance`] of how each value was chosen.
///
/// Computed in the main world (before extraction) so that the identity case —
/// all-[`Keep`](AutoField::Keep) policy, or a user-set target with no sensing —
/// has zero frame lag: a user-set-HDR project shows HDR on its first frame with
/// no SDR pop. The render pipeline reads [`target`](Self::target) in place of
/// the raw [`DisplayTarget`]; when this component is absent (pre-resolve or
/// removed) the pipeline falls back to [`DisplayTarget::SDR_SRGB`], exactly as
/// it falls back for a missing [`DisplayTarget`].
#[derive(Component, Debug, Clone, Copy, PartialEq)]
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

impl Default for EffectiveDisplayTarget {
    /// The SDR sRGB target with all-[`User`](FieldProvenance::User) provenance,
    /// identical to the [`DisplayTarget`] default.
    fn default() -> Self {
        Self {
            target: DisplayTarget::SDR_SRGB,
            provenance: DisplayProvenance::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_default_is_all_keep() {
        let p = DisplayCalibrationPolicy::default();
        assert_eq!(p.paper_white, AutoField::Keep);
        assert_eq!(p.peak_luminance, AutoField::Keep);
        assert_eq!(p.min_luminance, AutoField::Keep);
        assert_eq!(p.gamut, AutoField::Keep);
    }

    #[test]
    fn has_auto_is_false_for_all_keep_true_for_any_auto() {
        assert!(!DisplayCalibrationPolicy::default().has_auto());
        assert!(DisplayCalibrationPolicy {
            peak_luminance: AutoField::Auto,
            ..Default::default()
        }
        .has_auto());
        assert!(DisplayCalibrationPolicy {
            gamut: AutoField::Auto,
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
    fn live_and_capability_defaults_are_unknown_not_sdr() {
        assert_eq!(
            WindowDisplayState::default().source,
            DisplayInfoSource::Unknown
        );
        assert_eq!(WindowDisplayState::default().tone_map_headroom, None);
        assert_eq!(WindowDisplayState::default().generation, 0);
        assert_eq!(
            MonitorDisplayCapability::default().source,
            DisplayInfoSource::Unknown
        );
        assert_eq!(MonitorDisplayCapability::default().max_nits, None);
    }
}
