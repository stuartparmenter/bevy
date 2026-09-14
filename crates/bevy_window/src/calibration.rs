use bevy_color::RgbPrimaries;
use bevy_ecs::prelude::Component;

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
/// A luminance field of [`DisplayTarget`](crate::DisplayTarget) that is `None`
/// takes the value reported here. See
/// [`DisplayTarget::resolve_with_display`](crate::DisplayTarget::resolve_with_display).
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
    ///
    /// An uncalibrated
    /// [`DisplayTarget::peak_luminance_nits`](crate::DisplayTarget::peak_luminance_nits)
    /// takes this value on an HDR surface.
    pub max_nits: Option<f32>,
    /// The luminance, in nits, the display can sustain across the whole panel.
    ///
    /// Power and thermal limits can keep it below [`max_nits`](Self::max_nits).
    /// Only Windows reports it. Nothing resolves from it. For content that is
    /// bright across the whole frame, set
    /// [`DisplayTarget::peak_luminance_nits`](crate::DisplayTarget::peak_luminance_nits)
    /// from this value yourself.
    pub max_full_frame_nits: Option<f32>,
    /// The lowest luminance the display can show, in nits.
    ///
    /// An uncalibrated
    /// [`DisplayTarget::min_luminance_nits`](crate::DisplayTarget::min_luminance_nits)
    /// takes this value.
    pub min_nits: Option<f32>,
    /// The primaries of the widest gamut the display covers, as a coarse
    /// bucket: BT.709, Display P3 or BT.2020. Reported on Windows and the web.
    /// `None` elsewhere, even on a wide gamut display.
    ///
    /// Nothing resolves from it. The primaries the renderer encodes for are
    /// fixed by the surface color space.
    pub gamut_hint: Option<RgbPrimaries>,
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
/// A luminance field of [`DisplayTarget`](crate::DisplayTarget) that is `None`
/// takes a value derived from this component. See
/// [`DisplayTarget::resolve_with_display`](crate::DisplayTarget::resolve_with_display).
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
    /// On Windows a value of `1.0` or less means the OS HDR setting is off,
    /// and negotiation resolves a
    /// [`DisplayTarget::hdr`](crate::DisplayTarget::hdr) request to SDR even
    /// when the surface lists an HDR color space. A
    /// [`color_space_override`](crate::DisplayTarget::color_space_override) is
    /// still used as requested.
    ///
    /// [`DisplayHdrInfo::tone_map_headroom`]: https://docs.rs/wgpu/30/wgpu/struct.DisplayHdrInfo.html#method.tone_map_headroom
    pub tone_map_headroom: Option<f32>,
    /// The luminance, in nits, the platform maps SDR white to right now.
    ///
    /// Only Windows reports it. It changes with the SDR brightness setting.
    pub sdr_white_nits: Option<f32>,
}

/// Where one luminance field of a
/// [`ResolvedDisplayTarget`](crate::ResolvedDisplayTarget) comes from.
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
    /// The app calibrated the field with a valid luminance: the
    /// [`DisplayTarget`](crate::DisplayTarget) field is `Some`.
    User,
    /// The field is `None` and the display reported a value.
    Os,
    /// The field takes the default for the color space: it is `None` and the
    /// display reported nothing, or it is `Some` but not a valid luminance.
    #[default]
    Default,
}

/// Where each luminance field of a
/// [`ResolvedDisplayTarget`](crate::ResolvedDisplayTarget) comes from.
///
/// The color space has no entry. It is the argument to
/// [`DisplayTarget::resolve`](crate::DisplayTarget::resolve), which for a
/// window is the color space the surface negotiated.
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
    /// Provenance of
    /// [`ResolvedDisplayTarget::paper_white_nits`](crate::ResolvedDisplayTarget::paper_white_nits).
    pub paper_white: FieldProvenance,
    /// Provenance of
    /// [`ResolvedDisplayTarget::peak_luminance_nits`](crate::ResolvedDisplayTarget::peak_luminance_nits).
    pub peak_luminance: FieldProvenance,
    /// Provenance of
    /// [`ResolvedDisplayTarget::min_luminance_nits`](crate::ResolvedDisplayTarget::min_luminance_nits).
    pub min_luminance: FieldProvenance,
}

impl DisplayProvenance {
    /// Every field from the app.
    pub const USER: Self = Self {
        paper_white: FieldProvenance::User,
        peak_luminance: FieldProvenance::User,
        min_luminance: FieldProvenance::User,
    };

    /// Every field from the color space default.
    pub const DEFAULT: Self = Self {
        paper_white: FieldProvenance::Default,
        peak_luminance: FieldProvenance::Default,
        min_luminance: FieldProvenance::Default,
    };
}
