use bevy_ecs::prelude::Component;

use crate::DisplayTarget;

#[cfg(feature = "bevy_reflect")]
use {
    bevy_ecs::prelude::ReflectComponent,
    bevy_reflect::{std_traits::ReflectDefault, Reflect},
};

#[cfg(all(feature = "serialize", feature = "bevy_reflect"))]
use bevy_reflect::{ReflectDeserialize, ReflectSerialize};

/// The luminance and gamut the platform reports for a
/// [`Monitor`](crate::Monitor).
///
/// The values come from wgpu's [`DisplayHdrInfo`]. `bevy_window` does not
/// depend on wgpu, so the renderer inserts and updates this component. It goes
/// on the monitor entity, so every window on that display shares it. The
/// renderer overwrites it with each new reading. A field is `None` when the
/// platform does not report it. `None` never means SDR.
///
/// [`DisplayHdrInfo`]: https://docs.rs/wgpu/30/wgpu/struct.DisplayHdrInfo.html
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
    /// The peak luminance, in nits, of a small area of the display.
    pub max_nits: Option<f32>,
    /// The luminance, in nits, the display can sustain across the whole panel.
    ///
    /// Power and thermal limits can keep it below [`max_nits`](Self::max_nits).
    /// Only Windows reports it. [`DisplayCalibrationPolicy::auto_peak_luminance`]
    /// uses [`max_nits`](Self::max_nits). For content that is bright across
    /// the whole frame, leave that field off and set
    /// [`DisplayTarget::peak_luminance_nits`] from this value yourself.
    pub max_full_frame_nits: Option<f32>,
    /// The lowest luminance the display can show, in nits.
    pub min_nits: Option<f32>,
    /// The gamut the display covers. Reported on Windows and the web. `None`
    /// elsewhere, even on a wide gamut display.
    pub gamut_hint: Option<crate::DisplayGamut>,
}

/// The HDR headroom and SDR white luminance the display behind a window's
/// surface reports right now.
///
/// The values change while the window is open, for example when it moves to
/// another monitor or the user changes the display brightness.
///
/// The values come from wgpu's [`DisplayHdrInfo`]. `bevy_window` does not
/// depend on wgpu, so the renderer inserts and updates this component. The
/// renderer overwrites it with each new reading. It is one frame behind the
/// surface and is absent until a read reports something. The renderer updates
/// it only when a value changes by more than a small tolerance, so read noise
/// does not trigger change detection.
///
/// [`DisplayHdrInfo`]: https://docs.rs/wgpu/30/wgpu/struct.DisplayHdrInfo.html
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
    /// How many times brighter than SDR white the display can show right now,
    /// from wgpu's [`DisplayHdrInfo::tone_map_headroom`].
    ///
    /// `1.0` means no headroom. `None` means the platform did not report it,
    /// not that the display is SDR.
    ///
    /// [`DisplayHdrInfo::tone_map_headroom`]: https://docs.rs/wgpu/30/wgpu/struct.DisplayHdrInfo.html#method.tone_map_headroom
    pub tone_map_headroom: Option<f32>,
    /// The luminance, in nits, the platform maps SDR white to right now.
    ///
    /// Only Windows reports it. It changes with the SDR brightness setting.
    pub sdr_white_nits: Option<f32>,
}

/// Which fields of a window's [`DisplayTarget`] take the value the display
/// reports.
///
/// Bevy never changes [`DisplayTarget`]. The result goes in
/// [`EffectiveDisplayTarget`]. A field set to `true` takes the reported value
/// when there is one and keeps the [`DisplayTarget`] value otherwise. The
/// default leaves every field `false`.
///
/// [`DisplayTarget::transfer`] has no entry. Changing the transfer would
/// reconfigure the surface.
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
    /// Whether [`DisplayTarget::paper_white_nits`] takes
    /// [`WindowDisplayState::sdr_white_nits`].
    pub auto_paper_white: bool,
    /// Whether [`DisplayTarget::peak_luminance_nits`] takes
    /// [`MonitorDisplayCapability::max_nits`], or the resolved paper white
    /// times [`WindowDisplayState::tone_map_headroom`] when no peak in nits is
    /// reported. Only applies to HDR transfers.
    pub auto_peak_luminance: bool,
    /// Whether [`DisplayTarget::min_luminance_nits`] takes
    /// [`MonitorDisplayCapability::min_nits`].
    pub auto_min_luminance: bool,
    /// Whether [`DisplayTarget::gamut`] takes
    /// [`MonitorDisplayCapability::gamut_hint`].
    pub auto_gamut: bool,
}

impl DisplayCalibrationPolicy {
    /// Returns `true` if any field is `true`.
    pub const fn has_auto(&self) -> bool {
        self.auto_paper_white
            || self.auto_peak_luminance
            || self.auto_min_luminance
            || self.auto_gamut
    }
}

/// Where one field of an [`EffectiveDisplayTarget`] came from.
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
    /// The [`DisplayTarget`] value. [`DisplayCalibrationPolicy`] does not
    /// enable this field.
    #[default]
    User,
    /// The value the display reported.
    Os,
    /// The [`DisplayTarget`] value. [`DisplayCalibrationPolicy`] enables this
    /// field, but the display reported nothing.
    Fallback,
}

/// Where each field of an [`EffectiveDisplayTarget`] came from.
///
/// [`DisplayTarget::transfer`] has no entry because it always comes from
/// [`DisplayTarget`].
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

/// The [`DisplayTarget`] the renderer encodes for, after
/// [`DisplayCalibrationPolicy`] applies what the display reports.
///
/// A required component of [`Window`](crate::Window). Bevy updates it every
/// frame, so writing to it has no effect. With the default policy its
/// [`target`](Self::target) equals the window's [`DisplayTarget`]. Its
/// transfer is still the request. The transfer the surface uses is in
/// [`WindowSurfaceTransfers`](crate::WindowSurfaceTransfers).
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
    /// The resolved [`DisplayTarget`].
    pub target: DisplayTarget,
    /// Where each field of [`target`](Self::target) came from.
    pub provenance: DisplayProvenance,
}
