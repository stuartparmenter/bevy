//! Opt-in HDR setup for examples: request the best HDR output the surface
//! supports, falling back to SDR when none is available.
//!
//! Unlike most examples, which demonstrate an application, this is a small
//! reusable plugin library. [`HdrPlugin`] keeps the primary window's
//! [`DisplayTarget`] on the best entry of a preference list the surface can
//! actually present (read from [`WindowSupportedTransfers`]), rather than
//! hardcoding one transfer that might silently downgrade to SDR.
//!
//! On Windows the surface advertises the PQ color space even when the OS HDR
//! toggle is off, so `WindowSupportedTransfers` alone can't detect SDR desktop
//! mode; there the plugin also forces SDR when the live tone-map headroom
//! ([`WindowDisplayState`]) reports 1.0.
//!
//! It writes only [`DisplayTarget::transfer`] and [`DisplayTarget::gamut`]; tone
//! mapping, the per-camera [`Hdr`](bevy::camera::Hdr) component, and
//! [`DisplayCalibrationPolicy`](bevy::window::DisplayCalibrationPolicy) stay the
//! app's job. Pair an HDR transfer with a tone-mapping operator (e.g.
//! [`Tonemapping::GranTurismo7`](bevy::core_pipeline::tonemapping::Tonemapping)),
//! or the renderer warns that HDR output is written without tone mapping.
//!
//! ```ignore
//! // PQ/HDR10 first, then scRGB-linear, then encoded-extended sRGB, else SDR:
//! app.add_plugins(HdrPlugin::default());
//! ```

use bevy::{
    prelude::*,
    window::{
        DisplayGamut, DisplayTarget, DisplayTransfer, PrimaryWindow, WindowDisplayState,
        WindowSupportedTransfers,
    },
};

/// Requests the best supported HDR output for the primary window, falling back
/// to SDR.
///
/// `preference` lists `(transfer, gamut)` candidates best-first; the first whose
/// transfer the surface advertises wins. Each transfer is paired with its
/// canonical gamut (PQ with [`DisplayGamut::Rec2020`], the sRGB-family transfers
/// with [`DisplayGamut::Rec709`]) so the encoder never has to coerce it.
/// Selection re-runs when the surface's capabilities change (first
/// configuration, a monitor move, an OS HDR toggle); set
/// [`HdrPreference::manual_override`] to hand control to the user.
pub struct HdrPlugin {
    /// `(transfer, gamut)` candidates, best first. The first whose transfer the
    /// surface supports wins; if none do, the window stays SDR sRGB.
    pub preference: Vec<(DisplayTransfer, DisplayGamut)>,
}

impl Default for HdrPlugin {
    /// PQ/HDR10 first (absolute, wide-gamut Rec.2020), then scRGB-linear, then
    /// encoded extended-range sRGB (the web HDR path); SDR sRGB if none are
    /// advertised.
    fn default() -> Self {
        Self {
            preference: vec![
                (DisplayTransfer::Pq, DisplayGamut::Rec2020),
                (DisplayTransfer::ScRgbLinear, DisplayGamut::Rec709),
                (DisplayTransfer::ExtendedSrgb, DisplayGamut::Rec709),
            ],
        }
    }
}

impl Plugin for HdrPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(HdrPreference {
            order: self.preference.clone(),
            manual_override: false,
        })
        .add_systems(Update, apply_hdr_preference);
    }
}

/// Runtime state for [`HdrPlugin`]: the preference order plus a switch that
/// hands transfer control back to the user.
#[derive(Resource)]
pub struct HdrPreference {
    /// `(transfer, gamut)` candidates, best first (see [`HdrPlugin::preference`]).
    pub order: Vec<(DisplayTransfer, DisplayGamut)>,
    /// When `true`, the plugin stops auto-selecting. Set it once the user picks
    /// a transfer by hand so a later capability change does not overwrite their
    /// choice; clear it to resume auto-selection on the next frame.
    pub manual_override: bool,
}

/// Selects the best supported `(transfer, gamut)` and writes it to the primary
/// window's [`DisplayTarget`].
///
/// Runs every frame so clearing [`HdrPreference::manual_override`] resumes
/// auto-selection at once, but acts only on a capability, preference, or (Windows)
/// live-HDR change. [`Single`] skips it until [`WindowSupportedTransfers`] exists
/// (first surface configuration; never on headless), leaving the authored SDR
/// default.
fn apply_hdr_preference(
    preference: Res<HdrPreference>,
    window: Single<
        (
            &mut DisplayTarget,
            Ref<WindowSupportedTransfers>,
            Option<Ref<WindowDisplayState>>,
        ),
        With<PrimaryWindow>,
    >,
) {
    // The user took manual control; leave their transfer alone.
    if preference.manual_override {
        return;
    }
    let (mut target, supported, display_state) = window.into_inner();

    // Windows advertises the PQ color space even when the OS HDR toggle is off, so
    // the supported set alone can't tell SDR desktop mode from HDR. The live
    // tone-map headroom can (it reads 1.0 for a definitively-SDR display); force
    // SDR when it does. A no-op off Windows (see `display_reports_sdr`).
    let force_sdr = display_reports_sdr(display_state.as_deref());

    // Act on a capability transition, the app clearing the override, or -- on
    // Windows -- a live HDR enable/disable, which flips the headroom while leaving
    // the supported set unchanged. Writing the transfer renegotiates the surface
    // but leaves all three inputs unchanged, so this never re-triggers itself.
    if !supported.is_changed()
        && !preference.is_changed()
        && !hdr_state_changed(display_state.as_ref())
    {
        return;
    }

    let (transfer, gamut) = if force_sdr {
        (DisplayTransfer::Srgb, DisplayGamut::Rec709)
    } else {
        preference
            .order
            .iter()
            .copied()
            .find(|(transfer, _)| supported.contains(*transfer))
            .unwrap_or((DisplayTransfer::Srgb, DisplayGamut::Rec709))
    };

    if target.transfer != transfer || target.gamut != gamut {
        target.transfer = transfer;
        target.gamut = gamut;
    }

    if transfer.is_hdr() {
        info!(
            "HdrPlugin: selected {transfer:?} / {gamut:?} (surface supports {:?})",
            supported.0
        );
    } else if force_sdr {
        info!(
            "HdrPlugin: display reports SDR (tone-map headroom 1.0); staying on SDR sRGB \
             even though the surface advertises {:?}",
            supported.0
        );
    } else {
        info!(
            "HdrPlugin: no requested HDR transfer is available; staying on SDR sRGB \
             (surface supports {:?})",
            supported.0
        );
    }
}

/// Whether the plugin should force SDR despite an HDR transfer being "supported".
///
/// Windows advertises the PQ (HDR10) color space on the surface even when the OS
/// HDR toggle is off -- DXGI presents a PQ swapchain on an SDR desktop and the
/// compositor tone-maps it down -- so [`WindowSupportedTransfers`] can't tell the
/// two apart. The live tone-map headroom does: it is `1.0` for a definitively-SDR
/// display. Other platforms report their supported transfers accurately, so off
/// Windows this is always `false` and selection is unchanged.
#[cfg(target_os = "windows")]
fn display_reports_sdr(state: Option<&WindowDisplayState>) -> bool {
    state
        .and_then(|state| state.tone_map_headroom)
        .is_some_and(|headroom| headroom <= 1.0)
}

#[cfg(not(target_os = "windows"))]
fn display_reports_sdr(_state: Option<&WindowDisplayState>) -> bool {
    false
}

/// Whether the live display state changed this frame, so the Windows SDR gate
/// re-runs on a runtime HDR toggle (which leaves [`WindowSupportedTransfers`]
/// unchanged). Off Windows the supported set already tracks HDR availability, so
/// this stays `false` and the system keeps its original change detection.
#[cfg(target_os = "windows")]
fn hdr_state_changed(state: Option<&Ref<WindowDisplayState>>) -> bool {
    state.is_some_and(|state| state.is_changed())
}

#[cfg(not(target_os = "windows"))]
fn hdr_state_changed(_state: Option<&Ref<WindowDisplayState>>) -> bool {
    false
}
