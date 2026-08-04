//! Opt-in HDR setup for examples. [`HdrPlugin`] points the primary window's
//! [`DisplayTarget`] at the best transfer the surface reports in
//! [`WindowSurfaceTransfers`], and falls back to SDR when there is none. A
//! hardcoded transfer would downgrade to SDR instead.
//!
//! It writes only [`DisplayTarget::transfer`] and [`DisplayTarget::gamut`]. Tone
//! mapping, the per-camera [`Hdr`](bevy::camera::Hdr) component, and
//! [`DisplayCalibrationPolicy`](bevy::window::DisplayCalibrationPolicy) stay the
//! app's job. Pair an HDR transfer with a tone-mapping operator such as
//! [`Tonemapping::GranTurismo7`](bevy::core_pipeline::tonemapping::Tonemapping),
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
/// The default list pairs each transfer with its canonical gamut (PQ with
/// [`DisplayGamut::Rec2020`], the sRGB-family transfers with [`DisplayGamut::Rec709`])
/// so the encoder does not have to coerce it. Selection re-runs whenever the surface's
/// capabilities change, such as a monitor move or an OS HDR toggle. Set
/// [`HdrPreference::manual_override`] to hand control to the user.
pub struct HdrPlugin {
    /// `(transfer, gamut)` candidates, best first. The first whose transfer the
    /// surface supports wins. If none do, the window stays SDR sRGB.
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
    /// When `true`, the plugin stops auto-selecting. Set it after the user picks a
    /// transfer by hand, so a later capability change does not overwrite it. Clear
    /// it to resume on the next frame.
    pub manual_override: bool,
}

/// Selects the best supported `(transfer, gamut)` and writes it to the primary
/// window's [`DisplayTarget`].
///
/// Runs every frame so clearing [`HdrPreference::manual_override`] takes effect at
/// once, but acts only on a capability or preference change. [`WindowSurfaceTransfers`]
/// appears at first surface configuration, and [`Single`] skips this system until then,
/// leaving the app's authored target alone.
fn apply_hdr_preference(
    preference: Res<HdrPreference>,
    window: Single<(&mut DisplayTarget, Ref<WindowSurfaceTransfers>), With<PrimaryWindow>>,
) {
    if preference.manual_override {
        return;
    }
    let (mut target, surface) = window.into_inner();
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

    // Writing the transfer renegotiates the surface, which wakes this system again
    // next frame with the same answer. Log only on a change, plus the first sight of
    // the surface so an SDR-only machine still says so once.
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
