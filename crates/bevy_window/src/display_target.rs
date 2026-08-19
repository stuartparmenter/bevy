//! Types describing the display a window (or other render target) is presented on.
//!
//! The central type is [`DisplayTarget`], a required component of
//! [`Window`](crate::Window) holding the calibration of the display the
//! window's swapchain feeds: paper-white and peak luminance, color gamut, and
//! transfer function.

use bevy_ecs::prelude::Component;

#[cfg(feature = "bevy_reflect")]
use {
    bevy_ecs::prelude::ReflectComponent,
    bevy_reflect::{std_traits::ReflectDefault, Reflect},
};

#[cfg(all(feature = "serialize", feature = "bevy_reflect"))]
use bevy_reflect::{ReflectDeserialize, ReflectSerialize};

/// Describes the display device that a [`Window`](crate::Window) (or other
/// render target) is presented on, so the renderer can produce a correctly
/// tone-mapped, gamut-mapped, and transfer-encoded signal for it.
///
/// This component is user-authoritative: Bevy never overwrites values you set,
/// even when the window moves to a different monitor. A move retargets the
/// window's [`OnMonitor`](crate::OnMonitor) relationship; watch
/// `Changed<OnMonitor>` to decide whether to update this component.
///
/// A required component of [`Window`](crate::Window), defaulting to
/// [`DisplayTarget::SDR_SRGB`]. Every camera rendering to the same window
/// shares that one `DisplayTarget`.
///
/// Render targets that are not windows (`RenderTarget::Image`,
/// `RenderTarget::TextureView`) have no window entity to host this component.
/// They are looked up in `bevy_render`'s `ManualDisplayTargets` resource
/// instead, and fall back to [`DisplayTarget::SDR_SRGB`] when absent.
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
pub struct DisplayTarget {
    /// The luminance, in nits, of "paper white" (also called reference or
    /// diffuse white): a full-white UI element or a 100%-diffuse-reflective
    /// surface. Emissive highlights may go brighter, up to
    /// [`peak_luminance_nits`].
    ///
    /// Tone-mapping output is renormalized at the encoder seam so that `1.0`
    /// corresponds to this luminance. On SDR displays it is signal-level
    /// white, nominally `100.0` nits. HDR values run 100-300 nits depending on
    /// viewing environment; ITU-R BT.2408 recommends 203 nits for HDR
    /// broadcast.
    ///
    /// [`peak_luminance_nits`]: Self::peak_luminance_nits
    pub paper_white_nits: f32,
    /// The maximum luminance, in nits, that the display can show.
    ///
    /// Peak-aware tone-mapping operators compress scene highlights into
    /// `[0, peak_luminance_nits]` rather than clipping them. On SDR displays
    /// peak and paper white coincide (nominally `100.0` nits); on HDR displays
    /// the peak is higher (commonly 400-4000 nits), leaving headroom above
    /// paper white for emissive highlights.
    ///
    /// For displays that cannot sustain their peak over the full panel, use the
    /// OS metadata or HGIG-style calibration value (`MaxTML`), not the
    /// marketing peak.
    pub peak_luminance_nits: f32,
    /// The minimum luminance, in nits, that the display can show (its black
    /// level).
    ///
    /// OLED panels reach true zero; backlit panels bottom out between 0.01 and
    /// 0.1 nits.
    ///
    /// No engine stage consumes this value yet. It is carried for calibration
    /// UIs and for HDR10 mastering metadata (SMPTE ST 2086) once wgpu exposes
    /// an API for it.
    pub min_luminance_nits: f32,
    /// The color gamut (set of primaries) of the display target.
    ///
    /// The gamut transform stage converts rendered colors from the working
    /// color space to these primaries, with perceptual gamut compression for
    /// out-of-gamut colors, before transfer encoding.
    pub gamut: DisplayGamut,
    /// The transfer function to encode the signal with.
    ///
    /// A request the backend and OS may not be able to fulfil. See
    /// [`DisplayTransfer`] for per-variant support and degrade behavior.
    pub transfer: DisplayTransfer,
}

impl DisplayTarget {
    /// The standard-dynamic-range sRGB display target, and the [`Default`]
    /// value.
    ///
    /// Renders with no display-encoding pass and hardware sRGB encode on
    /// writeback.
    pub const SDR_SRGB: Self = Self {
        paper_white_nits: 100.0,
        peak_luminance_nits: 100.0,
        min_luminance_nits: 0.0,
        gamut: DisplayGamut::Rec709,
        transfer: DisplayTransfer::Srgb,
    };

    /// Returns `self` with [`paper_white_nits`](Self::paper_white_nits) set to
    /// `nits`.
    pub const fn with_paper_white(mut self, nits: f32) -> Self {
        self.paper_white_nits = nits;
        self
    }

    /// Returns `self` with [`peak_luminance_nits`](Self::peak_luminance_nits)
    /// set to `nits`.
    pub const fn with_peak(mut self, nits: f32) -> Self {
        self.peak_luminance_nits = nits;
        self
    }

    /// Returns `self` with [`min_luminance_nits`](Self::min_luminance_nits)
    /// set to `nits`.
    pub const fn with_min_luminance(mut self, nits: f32) -> Self {
        self.min_luminance_nits = nits;
        self
    }

    /// Returns `self` with [`gamut`](Self::gamut) set to `gamut`.
    pub const fn with_gamut(mut self, gamut: DisplayGamut) -> Self {
        self.gamut = gamut;
        self
    }

    /// Returns `self` with [`transfer`](Self::transfer) set to `transfer`.
    pub const fn with_transfer(mut self, transfer: DisplayTransfer) -> Self {
        self.transfer = transfer;
        self
    }

    /// The ceiling [`sanitized_paper_white_nits`] clamps to: the PQ (SMPTE ST
    /// 2084) coding ceiling of 10000 nits, the brightest luminance any
    /// supported transfer function can represent.
    ///
    /// [`sanitized_paper_white_nits`]: Self::sanitized_paper_white_nits
    pub const MAX_PAPER_WHITE_NITS: f32 = 10000.0;

    /// Returns [`paper_white_nits`](Self::paper_white_nits) sanitized for use
    /// in luminance math:
    ///
    /// - non-finite or non-positive values fall back to
    ///   [`DisplayTarget::SDR_SRGB`]'s 100 nits (a zero or `NaN` paper white
    ///   would black out or `NaN` the whole frame);
    /// - values above [`MAX_PAPER_WHITE_NITS`](Self::MAX_PAPER_WHITE_NITS) are
    ///   clamped to it.
    ///
    /// Valid values are returned bit-for-bit unchanged.
    ///
    /// Every renderer stage that folds `paper_white_nits` into its math must go
    /// through this method, so the tone-map seam renormalization and the
    /// display encoder's scale factor cancel exactly. This method is pure;
    /// callers warn about the fallback.
    pub fn sanitized_paper_white_nits(&self) -> f32 {
        if !self.paper_white_nits.is_finite() || self.paper_white_nits <= 0.0 {
            Self::SDR_SRGB.paper_white_nits
        } else {
            self.paper_white_nits.min(Self::MAX_PAPER_WHITE_NITS)
        }
    }
}

impl Default for DisplayTarget {
    fn default() -> Self {
        Self::SDR_SRGB
    }
}

/// The color gamut (primary chromaticities) of a display target.
///
/// All variants assume a D65 white point.
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
pub enum DisplayGamut {
    /// ITU-R BT.709 primaries, shared by sRGB. The standard-dynamic-range
    /// gamut every display can show.
    #[default]
    Rec709,
    /// Display P3 primaries (DCI-P3 with a D65 white point), used by most Apple
    /// displays and many wide-gamut monitors. Wider than Rec.709, narrower than
    /// Rec.2020.
    ///
    /// Reaches wide-gamut HDR output only paired with
    /// [`DisplayTransfer::ExtendedSrgb`].
    DisplayP3,
    /// ITU-R BT.2020 primaries, the wide gamut used by HDR10 and most HDR
    /// video standards. Physical displays typically cover only part of this
    /// gamut and apply their own gamut mapping.
    Rec2020,
}

/// The transfer function used to encode the final signal for a display target.
///
/// The last stage of the display pipeline: it converts tone-mapped,
/// gamut-mapped display-linear color into the signal values the display
/// expects.
///
/// Backend support follows wgpu's surface color-space API:
///
/// - [`Srgb`][]: everywhere.
/// - [`ScRgbLinear`] (linear scRGB): macOS/iOS (Metal), Windows (Vulkan/DX12),
///   Wayland (Vulkan). Native-only, since browser WebGPU cannot express a
///   linear-transfer canvas.
/// - [`ExtendedSrgb`] (encoded extended-range sRGB): Metal, Vulkan (Rec.709
///   gamut only), and browser WebGPU on HDR-capable displays. The web HDR path.
/// - [`Pq`] (HDR10): Vulkan, DX12, and Metal when the OS has HDR output
///   enabled.
///
/// Unfulfillable requests degrade with a warning.
///
/// [`Srgb`]: DisplayTransfer::Srgb
/// [`ScRgbLinear`]: DisplayTransfer::ScRgbLinear
/// [`Pq`]: DisplayTransfer::Pq
/// [`ExtendedSrgb`]: DisplayTransfer::ExtendedSrgb
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
pub enum DisplayTransfer {
    /// The sRGB transfer function (IEC 61966-2-1), the standard-dynamic-range
    /// default. Applied in hardware via `*UnormSrgb` surface formats.
    #[default]
    Srgb,
    /// scRGB linear (IEC 61966-2-2): a linear, extended-range encoding where
    /// signal value `1.0` is 80 nits and values above `1.0` (and below `0.0`)
    /// are valid. Used with `Rgba16Float` surfaces. The encoder scales by
    /// `paper_white_nits / 80` so that scene paper white lands on the display's
    /// configured paper white.
    ///
    /// scRGB signals are always in (extended) Rec.709/sRGB coordinates,
    /// whatever the physical gamut of the panel: the OS compositor maps to the
    /// panel's primaries, and wide gamut is carried by out-of-range (including
    /// negative) component values. The renderer therefore ignores
    /// [`DisplayTarget::gamut`] for this transfer, with a log notice.
    ScRgbLinear,
    /// The Perceptual Quantizer (SMPTE ST 2084, ITU-R BT.2100), the absolute
    /// transfer function used by HDR10. Encodes absolute luminance normalized
    /// to 10000 nits. Canonically paired with [`DisplayGamut::Rec2020`]: the
    /// renderer coerces the encode to Rec.2020, since HDR10 is Rec.2020.
    ///
    /// Negotiated as an HDR10 swapchain (typically `Rgb10a2Unorm`) where the
    /// backend and OS advertise it. When unavailable, the request downgrades to
    /// [`ScRgbLinear`](Self::ScRgbLinear) if possible, else to SDR sRGB,
    /// warning at each step.
    Pq,
    /// Extended-range sRGB (IEC 61966-2-2, encoded form): the sRGB transfer
    /// function continued past `[0, 1]` by mirroring the curve through the
    /// origin (odd-symmetric, sign-preserving). `1.0` is SDR reference white;
    /// values above `1.0` (and below `0.0`) carry brighter-than-SDR and
    /// out-of-gamut color.
    ///
    /// The encoded (gamma) sibling of [`ScRgbLinear`](Self::ScRgbLinear): the
    /// renderer applies the same `paper_white_nits / 80` scRGB normalization,
    /// then the extended sRGB OETF (`srgb_oetf_extended` in
    /// `bevy_render::transfer_functions`) instead of leaving the signal linear.
    /// An 80-nit paper white therefore round-trips SDR through this transfer,
    /// since the OETF matches the plain sRGB curve on `[0, 1]`.
    ///
    /// Unlike `ScRgbLinear` this transfer is not gamut-agnostic: it pairs with
    /// [`DisplayTarget::gamut`] to select the surface color space.
    ///
    /// - [`DisplayGamut::Rec709`] -> wgpu `ExtendedSrgb` (Vulkan
    ///   `EXTENDED_SRGB_NONLINEAR_EXT`, Metal `kCGColorSpaceExtendedSRGB`,
    ///   browser WebGPU `srgb` canvas + `toneMapping: "extended"`).
    /// - [`DisplayGamut::DisplayP3`] -> wgpu `ExtendedDisplayP3` (Metal
    ///   `kCGColorSpaceExtendedDisplayP3`, browser WebGPU `display-p3` canvas +
    ///   `toneMapping: "extended"`); the encoder converts Rec.709/Rec.2020
    ///   tone-map output into P3 primaries before the OETF.
    /// - [`DisplayGamut::Rec2020`] has no encoded-extended surface and is
    ///   coerced to Rec.709.
    ExtendedSrgb,
}

impl DisplayTransfer {
    /// Returns `true` if this is a high-dynamic-range transfer function.
    ///
    /// The one predicate the display pipeline uses to choose the HDR path
    /// (shader-side transfer encoding, HDR operator modes) or the SDR path
    /// (hardware sRGB encode).
    pub const fn is_hdr(&self) -> bool {
        matches!(self, Self::ScRgbLinear | Self::Pq | Self::ExtendedSrgb)
    }
}

/// A set of [`DisplayTransfer`]s, held as a bitset so it can be copied rather
/// than allocated.
///
/// [`iter`](Self::iter) yields members in the stable cycle order apps step
/// through: `Srgb`, `ScRgbLinear`, `ExtendedSrgb`, `Pq`. That is neither the
/// declaration order of [`DisplayTransfer`] nor the bit order.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
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
pub struct DisplayTransfers(u8);

impl DisplayTransfers {
    /// The empty set.
    pub const EMPTY: Self = Self(0);

    /// The order [`iter`](Self::iter) walks.
    const CYCLE_ORDER: [DisplayTransfer; 4] = [
        DisplayTransfer::Srgb,
        DisplayTransfer::ScRgbLinear,
        DisplayTransfer::ExtendedSrgb,
        DisplayTransfer::Pq,
    ];

    /// The bit a transfer occupies. An explicit match, not a cast or a slice
    /// index, so that adding a variant cannot renumber the existing bits.
    const fn bit(transfer: DisplayTransfer) -> u8 {
        match transfer {
            DisplayTransfer::Srgb => 0b0001,
            DisplayTransfer::ScRgbLinear => 0b0010,
            DisplayTransfer::Pq => 0b0100,
            DisplayTransfer::ExtendedSrgb => 0b1000,
        }
    }

    /// Returns this set with `transfer` added.
    pub const fn with(self, transfer: DisplayTransfer) -> Self {
        Self(self.0 | Self::bit(transfer))
    }

    /// Returns `true` if `transfer` is a member.
    pub const fn contains(self, transfer: DisplayTransfer) -> bool {
        self.0 & Self::bit(transfer) != 0
    }

    /// Iterates the members in the stable cycle order.
    pub fn iter(self) -> impl Iterator<Item = DisplayTransfer> {
        Self::CYCLE_ORDER
            .into_iter()
            .filter(move |&transfer| self.contains(transfer))
    }
}

impl core::fmt::Debug for DisplayTransfers {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// What a window's surface negotiated: the [`DisplayTransfer`] it currently
/// presents, and every transfer it could present right now.
///
/// [`DisplayTarget::transfer`] is the request; surface negotiation can
/// downgrade it when the backend or OS cannot fulfil it. Read
/// [`supported`](Self::supported) to offer only the modes that will not
/// downgrade.
///
/// The renderer inserts and updates this once a window's surface is configured,
/// so the values lag the negotiation by one frame. Treat it as read-only:
/// writing it has no effect on the surface. It is absent until the first
/// surface configuration, and on windows that never get a surface (headless).
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Component, Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct WindowSurfaceTransfers {
    /// The transfer the configured surface currently carries: the negotiated
    /// outcome of the [`DisplayTarget::transfer`] request.
    pub resolved: DisplayTransfer,
    /// The transfers this surface can present, derived from the color spaces it
    /// advertises. [`DisplayTransfer::Srgb`] (the SDR fallback) is always a
    /// member.
    pub supported: DisplayTransfers,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdr_srgb_constant_values() {
        let sdr = DisplayTarget::SDR_SRGB;
        assert_eq!(sdr.paper_white_nits, 100.0);
        assert_eq!(sdr.peak_luminance_nits, 100.0);
        assert_eq!(sdr.min_luminance_nits, 0.0);
        assert_eq!(sdr.gamut, DisplayGamut::Rec709);
        assert_eq!(sdr.transfer, DisplayTransfer::Srgb);
    }

    #[test]
    fn sanitized_paper_white_passes_valid_values_through_bit_for_bit() {
        for nits in [0.001, 80.0, 100.0, 203.0, 1000.0, 10000.0] {
            let target = DisplayTarget {
                paper_white_nits: nits,
                ..DisplayTarget::SDR_SRGB
            };
            assert_eq!(
                target.sanitized_paper_white_nits().to_bits(),
                nits.to_bits()
            );
        }
    }

    #[test]
    fn sanitized_paper_white_replaces_degenerate_values() {
        for nits in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, -50.0] {
            let target = DisplayTarget {
                paper_white_nits: nits,
                ..DisplayTarget::SDR_SRGB
            };
            assert_eq!(target.sanitized_paper_white_nits(), 100.0);
        }
    }

    #[test]
    fn sanitized_paper_white_clamps_to_pq_ceiling() {
        let target = DisplayTarget {
            paper_white_nits: 20000.0,
            ..DisplayTarget::SDR_SRGB
        };
        assert_eq!(
            target.sanitized_paper_white_nits(),
            DisplayTarget::MAX_PAPER_WHITE_NITS
        );
    }

    #[test]
    fn builder_helpers_set_exactly_one_field() {
        let base = DisplayTarget::SDR_SRGB;
        assert_eq!(
            base.with_paper_white(200.0),
            DisplayTarget {
                paper_white_nits: 200.0,
                ..base
            }
        );
        assert_eq!(
            base.with_peak(1000.0),
            DisplayTarget {
                peak_luminance_nits: 1000.0,
                ..base
            }
        );
        assert_eq!(
            base.with_min_luminance(0.05),
            DisplayTarget {
                min_luminance_nits: 0.05,
                ..base
            }
        );
        assert_eq!(
            base.with_gamut(DisplayGamut::Rec2020),
            DisplayTarget {
                gamut: DisplayGamut::Rec2020,
                ..base
            }
        );
        assert_eq!(
            base.with_transfer(DisplayTransfer::ScRgbLinear),
            DisplayTarget {
                transfer: DisplayTransfer::ScRgbLinear,
                ..base
            }
        );
    }

    #[test]
    fn hdr_transfer_predicate() {
        assert!(!DisplayTransfer::Srgb.is_hdr());
        assert!(DisplayTransfer::ScRgbLinear.is_hdr());
        assert!(DisplayTransfer::Pq.is_hdr());
        assert!(DisplayTransfer::ExtendedSrgb.is_hdr());
    }

    #[test]
    fn transfer_set_membership() {
        let set = DisplayTransfers::EMPTY
            .with(DisplayTransfer::Srgb)
            .with(DisplayTransfer::Pq);
        assert!(set.contains(DisplayTransfer::Srgb));
        assert!(set.contains(DisplayTransfer::Pq));
        assert!(!set.contains(DisplayTransfer::ScRgbLinear));
        assert!(!set.contains(DisplayTransfer::ExtendedSrgb));
        assert!(!DisplayTransfers::EMPTY.contains(DisplayTransfer::Srgb));
        assert_eq!(set.with(DisplayTransfer::Pq), set);
    }

    #[test]
    fn transfer_set_iterates_in_cycle_order() {
        // `Pq` is declared before `ExtendedSrgb` but cycles after it, so a
        // bit-index walk would yield these two the wrong way round.
        let set = DisplayTransfers::EMPTY
            .with(DisplayTransfer::Pq)
            .with(DisplayTransfer::ExtendedSrgb)
            .with(DisplayTransfer::Srgb);
        assert!(set.iter().eq([
            DisplayTransfer::Srgb,
            DisplayTransfer::ExtendedSrgb,
            DisplayTransfer::Pq,
        ]));
    }

    #[test]
    fn transfer_set_bits_are_distinct() {
        let all = DisplayTransfers::EMPTY
            .with(DisplayTransfer::Srgb)
            .with(DisplayTransfer::ScRgbLinear)
            .with(DisplayTransfer::Pq)
            .with(DisplayTransfer::ExtendedSrgb);
        assert_eq!(all.iter().count(), DisplayTransfers::CYCLE_ORDER.len());
        // Two transfers sharing a bit survive that count (both still report as
        // members) but not a singleton set.
        for transfer in DisplayTransfers::CYCLE_ORDER {
            assert!(DisplayTransfers::EMPTY.with(transfer).iter().eq([transfer]));
        }
        assert_eq!(DisplayTransfers::EMPTY.iter().count(), 0);
    }
}
