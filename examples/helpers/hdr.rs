//! Opt-in HDR setup for examples. [`HdrPlugin`] points the primary window's
//! [`DisplayTarget`] at the best transfer the surface reports in
//! [`WindowSurfaceTransfers`], and falls back to SDR when there is none.
//!
//! It writes only [`DisplayTarget::transfer`] and [`DisplayTarget::gamut`]. Tone
//! mapping, the per-camera [`Hdr`](bevy::camera::Hdr) component, and
//! [`DisplayCalibrationPolicy`](bevy::window::DisplayCalibrationPolicy) stay the
//! app's job. Pair an HDR transfer with a tone-mapping operator such as
//! [`Tonemapping::GranTurismo7`](bevy::core_pipeline::tonemapping::Tonemapping),
//! or the renderer warns that HDR output is written without tone mapping.

use bevy::{
    prelude::*,
    window::{
        DisplayGamut, DisplayTarget, DisplayTransfer, PrimaryWindow, WindowDisplayState,
        WindowSurfaceTransfers,
    },
};

const SDR_FALLBACK: (DisplayTransfer, DisplayGamut) = (
    DisplayTarget::SDR_SRGB.transfer,
    DisplayTarget::SDR_SRGB.gamut,
);

/// Requests the best supported HDR output for the primary window, falling back
/// to SDR.
///
/// Each default candidate pairs a transfer with its canonical gamut, so the
/// encoder does not have to coerce it.
pub struct HdrPlugin {
    /// `(transfer, gamut)` candidates, best first. The first transfer the surface
    /// supports wins; if none do, the window stays SDR sRGB.
    pub preference: Vec<(DisplayTransfer, DisplayGamut)>,
}

impl Default for HdrPlugin {
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

/// Runtime state for [`HdrPlugin`].
#[derive(Resource)]
pub struct HdrPreference {
    /// `(transfer, gamut)` candidates, best first (see [`HdrPlugin::preference`]).
    pub order: Vec<(DisplayTransfer, DisplayGamut)>,
    /// When `true`, the plugin stops auto-selecting, so a later capability change
    /// does not overwrite a transfer the user picked. Clear it to resume.
    pub manual_override: bool,
}

/// Selects the best supported `(transfer, gamut)` and writes it to the primary
/// window's [`DisplayTarget`].
///
/// Runs every frame so clearing [`HdrPreference::manual_override`] takes effect
/// at once. [`WindowSurfaceTransfers`] appears at first surface configuration,
/// and [`Single`] skips this system until then, leaving the app's authored
/// target alone.
fn apply_hdr_preference(
    preference: Res<HdrPreference>,
    window: Single<
        (
            &mut DisplayTarget,
            Ref<WindowSurfaceTransfers>,
            Option<Ref<WindowDisplayState>>,
        ),
        With<PrimaryWindow>,
    >,
) {
    if preference.manual_override {
        return;
    }
    let (mut target, surface, display_state) = window.into_inner();

    if !surface.is_changed()
        && !preference.is_changed()
        && !hdr_state_changed(display_state.as_ref())
    {
        return;
    }

    let force_sdr = display_reports_sdr(display_state.as_deref());
    let (transfer, gamut) = if force_sdr {
        SDR_FALLBACK
    } else {
        preference
            .order
            .iter()
            .copied()
            .find(|(transfer, _)| surface.supported.contains(*transfer))
            .unwrap_or(SDR_FALLBACK)
    };

    let selection_changed = target.transfer != transfer || target.gamut != gamut;
    if selection_changed {
        target.transfer = transfer;
        target.gamut = gamut;
    }

    // Writing the transfer wakes this system again with the same answer. Log on a
    // change, plus the first surface sighting so an SDR-only machine says so once.
    if !selection_changed && !surface.is_added() {
        return;
    }

    if transfer.is_hdr() {
        info!(
            "HdrPlugin: selected {transfer:?} / {gamut:?} (surface supports {:?})",
            surface.supported
        );
    } else if force_sdr {
        info!(
            "HdrPlugin: display reports SDR (tone-map headroom 1.0); staying on SDR sRGB \
             even though the surface advertises {:?}",
            surface.supported
        );
    } else {
        info!(
            "HdrPlugin: no requested HDR transfer is available; staying on SDR sRGB \
             (surface supports {:?})",
            surface.supported
        );
    }
}

/// Whether the plugin should force SDR despite an HDR transfer being "supported".
///
/// Windows advertises the PQ (HDR10) color space on the surface even when the OS
/// HDR toggle is off, so [`WindowSurfaceTransfers::supported`] over-reports
/// there. The live tone-map headroom disambiguates: it is `1.0` for a
/// definitively-SDR display. Other platforms report accurately, so the gate is
/// Windows-only.
fn display_reports_sdr(state: Option<&WindowDisplayState>) -> bool {
    cfg!(target_os = "windows")
        && state
            .and_then(|state| state.tone_map_headroom)
            .is_some_and(|headroom| headroom <= 1.0)
}

/// Whether the live display state changed this frame, so the Windows SDR gate
/// re-runs on a runtime HDR toggle, which leaves [`WindowSurfaceTransfers`]
/// unchanged. Off Windows the supported set already tracks HDR availability.
fn hdr_state_changed(state: Option<&Ref<WindowDisplayState>>) -> bool {
    cfg!(target_os = "windows") && state.is_some_and(|state| state.is_changed())
}
