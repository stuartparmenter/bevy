//! Opt-in HDR setup for examples: request the best HDR output the surface
//! supports, falling back to SDR when none is available.
//!
//! Unlike most examples, which demonstrate an application, this is a small
//! reusable plugin library. [`HdrPlugin`] keeps the primary window's
//! [`DisplayTarget`] on the best entry of a preference list the surface can
//! actually present (read from [`WindowSurfaceTransfers`]), rather than
//! hardcoding one transfer that might silently downgrade to SDR.
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
    window::{DisplayGamut, DisplayTarget, DisplayTransfer, PrimaryWindow, WindowSurfaceTransfers},
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
/// auto-selection at once, but acts only on a capability or preference change.
/// [`Single`] skips it until [`WindowSurfaceTransfers`] exists (first surface
/// configuration; never on headless), leaving the authored SDR default.
fn apply_hdr_preference(
    preference: Res<HdrPreference>,
    window: Single<(&mut DisplayTarget, Ref<WindowSurfaceTransfers>), With<PrimaryWindow>>,
) {
    // The user took manual control; leave their transfer alone.
    if preference.manual_override {
        return;
    }
    let (mut target, surface) = window.into_inner();
    // Act only on a real surface transition, or when the app cleared the
    // override.
    if !surface.is_changed() && !preference.is_changed() {
        return;
    }

    let (transfer, gamut) = preference
        .order
        .iter()
        .copied()
        .find(|(transfer, _)| surface.supported.contains(*transfer))
        .unwrap_or((DisplayTransfer::Srgb, DisplayGamut::Rec709));

    let selection_changed = target.transfer != transfer || target.gamut != gamut;
    if selection_changed {
        target.transfer = transfer;
        target.gamut = gamut;
    }

    // Report the selection, not every wake-up: writing the transfer
    // renegotiates the surface, whose new `resolved` lands back on this same
    // component next frame and wakes this system again with an unchanged
    // answer. First sight of the surface still reports, so an SDR-only machine
    // — where the selection matches the authored default and never changes —
    // says so once.
    if !selection_changed && !surface.is_added() {
        return;
    }

    if transfer.is_hdr() {
        info!(
            "HdrPlugin: selected {transfer:?} / {gamut:?} (surface supports {:?})",
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
