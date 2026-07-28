//! Render-side display sensing: re-reads each surface's live HDR state when
//! something can actually have changed — first sight, a surface reconfigure, a
//! window event (move / focus / monitor change), or, for the one signal that
//! drifts with no event (the Apple EDR headroom), every frame while an HDR
//! surface is auto-calibrating. It folds every platform's reporting asymmetry
//! into one cross-platform value (`read_display_state`), suppresses
//! sub-threshold jitter ([`DisplayStateStore`]), and mirrors the result back to
//! the main world as [`WindowDisplayState`] / [`MonitorDisplayCapability`].
//!
//! The single live value the tone mapper consumes is
//! [`DisplayHdrInfo::tone_map_headroom`] — the linear multiplier of SDR white the
//! display can drive right now. wgpu folds the platform-specific reporting
//! (Apple's relative EDR headroom, Windows' absolute `max_nits / sdr_white_nits`
//! ratio, the coarse SDR flag) into it, so this module never reconstructs an
//! absolute peak from platform-specific fields.
//!
//! `bevy_window` stays wgpu-free: it holds only the plain-data result types.
//! Everything that touches a wgpu [`DisplayHdrInfo`] lives here.

use bevy_ecs::prelude::*;
use bevy_platform::collections::HashMap;
use bevy_window::{
    DisplayGamut, DisplayTransfer, MonitorDisplayCapability, OnMonitor, WindowDisplayState,
    WindowSurfaceTransfers,
};
use wgpu::{DisplayGamut as WgpuDisplayGamut, DisplayHdrInfo};

use crate::renderer::RenderAdapter;
use crate::MainWorld;

use super::{ExtractedWindows, WindowSurfaces};

/// Maps a wgpu coarse [`DisplayGamut`](WgpuDisplayGamut) onto the plain-data
/// [`DisplayGamut`] the window crate carries. `Srgb` and any unrecognized
/// future wgpu variant map to the narrowest known gamut
/// ([`DisplayGamut::Rec709`]), the conservative choice for a capability hint.
fn map_gamut(g: WgpuDisplayGamut) -> DisplayGamut {
    match g {
        WgpuDisplayGamut::DisplayP3 => DisplayGamut::DisplayP3,
        WgpuDisplayGamut::Rec2020 => DisplayGamut::Rec2020,
        // `Srgb` and any unrecognized future variant.
        _ => DisplayGamut::Rec709,
    }
}

/// Returns `Some` only for finite, strictly-positive values; folds wgpu's
/// "reported but garbage" (NaN, infinity, zero, negative) into "not reported".
pub(crate) fn finite_positive(v: Option<f32>) -> Option<f32> {
    v.filter(|x| x.is_finite() && *x > 0.0)
}

/// Whether a [`DisplayHdrInfo`] carries any usable reporting model at all:
/// absolute luminance (Windows), a relative headroom (Apple), or a coarse
/// capability (web). `false` is "this platform or this moment can't tell us
/// anything" — never "SDR".
fn reports_anything(info: &DisplayHdrInfo) -> bool {
    info.luminance
        .is_some_and(|l| l.max_nits.is_some() || l.sdr_white_nits.is_some())
        || info.headroom.is_some()
        || info.coarse.is_some()
}

/// Collapses a wgpu [`DisplayHdrInfo`] into the two plain-data carriers, or
/// `None` when the platform reported nothing usable.
///
/// The live half is wgpu's already-folded
/// [`tone_map_headroom`](DisplayHdrInfo::tone_map_headroom): the one
/// cross-platform value, the linear multiplier of SDR white the display can
/// drive right now. wgpu resolves it from whichever model the backend reports —
/// Apple's relative EDR headroom (`current`), Windows' absolute
/// `max_nits / sdr_white_nits`, or `1.0` for a display that reports itself SDR —
/// so this function never reconstructs a peak from platform-specific fields.
/// `sdr_white_nits` is carried alongside (Windows only) to anchor the
/// `paper_white` auto-calibration.
///
/// The capability half — peak / full-frame / min nits and the coarse gamut
/// bucket the backend reports (nearest-primaries match on DXGI, the CSS
/// `color-gamut` query on web) — is copied through for the
/// [`MonitorDisplayCapability`] mirror.
fn read_display_state(
    info: &DisplayHdrInfo,
) -> Option<(WindowDisplayState, MonitorDisplayCapability)> {
    if !reports_anything(info) {
        return None;
    }

    let luminance = info.luminance;
    let state = WindowDisplayState {
        // The single cross-platform live HDR value, already folded by wgpu.
        tone_map_headroom: finite_positive(info.tone_map_headroom()),
        sdr_white_nits: luminance.and_then(|l| finite_positive(l.sdr_white_nits)),
    };
    let capability = MonitorDisplayCapability {
        max_nits: finite_positive(luminance.and_then(|l| l.max_nits)),
        max_full_frame_nits: finite_positive(luminance.and_then(|l| l.max_full_frame_nits)),
        min_nits: finite_positive(luminance.and_then(|l| l.min_nits)),
        gamut_hint: info.coarse.and_then(|c| c.gamut).map(map_gamut),
    };
    Some((state, capability))
}

/// Relative change below which a continuous field is treated as unchanged, so
/// sub-threshold read jitter never commits a new value / fires
/// [`Changed`](bevy_ecs::prelude::Changed). Small enough to track the Apple EDR
/// ramp smoothly (it climbs over ~1–2 s), large enough to swallow float noise.
const EPSILON_REL: f32 = 0.01;

/// Per-surface committed live state and last capability, keyed by window entity.
/// Render-world only.
#[derive(Resource, Default)]
pub struct DisplayStateStore {
    /// The committed (post-epsilon) [`WindowDisplayState`] per surface. A
    /// present-but-`None` entry marks a surface that has been read on a platform
    /// that reports nothing: it stops the per-frame re-read without minting a
    /// live-state component, so the carrier stays absent until a real read
    /// lands.
    states: HashMap<Entity, Option<WindowDisplayState>>,
    /// The last committed capability per surface (so the [`Monitor`] write-back
    /// is insert-on-change).
    ///
    /// [`Monitor`]: bevy_window::Monitor
    capabilities: HashMap<Entity, MonitorDisplayCapability>,
    /// The resolved transfer last seen per surface, so a renegotiation that
    /// changes it (an OS HDR enable/disable drives the surface `Outdated` and
    /// re-picks the color space without any authored-transfer change) forces a
    /// fresh read — otherwise the live state would lag until an unrelated window
    /// event.
    last_resolved: HashMap<Entity, DisplayTransfer>,
}

/// Whether two optional continuous values differ by more than the relative
/// epsilon (a `None`→`Some` or `Some`→`None` transition always counts).
fn rel_changed(old: Option<f32>, new: Option<f32>) -> bool {
    match (old, new) {
        (Some(a), Some(b)) => (a - b).abs() > EPSILON_REL * a.abs().max(f32::MIN_POSITIVE),
        (None, None) => false,
        _ => true,
    }
}

/// Folds a fresh read into the committed live state for `entity`. A field
/// commits only when it moves past the relative [`EPSILON_REL`], so the
/// insert-on-change write-back — and therefore
/// [`Changed<WindowDisplayState>`](bevy_ecs::prelude::Changed) — signals a
/// genuine transition rather than read jitter. The capability half passes
/// through unsmoothed (insert-on-change is handled at write-back).
fn commit(
    store: &mut DisplayStateStore,
    entity: Entity,
    candidate: WindowDisplayState,
    capability: MonitorDisplayCapability,
) {
    let committed = store.states.entry(entity).or_default();

    let changed = committed.is_none_or(|committed| {
        rel_changed(committed.tone_map_headroom, candidate.tone_map_headroom)
            || rel_changed(committed.sdr_white_nits, candidate.sdr_white_nits)
    });

    if changed {
        *committed = Some(candidate);
    }

    store.capabilities.insert(entity, capability);
}

/// Re-reads each configured surface's live HDR state when it can have changed,
/// smooths it, and stores the committed result for write-back.
///
/// Main-thread-pinned on Apple platforms (the relative-headroom query returns
/// `None` off the main thread). A surface is read when:
///
/// - it is seen for the first time (seed the mirror), or
/// - its surface was just (re)configured
///   ([`display_target_transfer_changed`](super::ExtractedWindow::display_target_transfer_changed)),
///   or
/// - a window event flagged it for re-query
///   ([`request_display_requery`](super::ExtractedWindow::request_display_requery)
///   — a move, focus regain, or monitor change), or
/// - it is auto-calibrating an HDR surface on a platform whose live value drifts
///   with no event (Apple EDR headroom), in which case it is read every frame.
///
/// The per-frame branch is gated on
/// [`display_calibration_auto`](super::ExtractedWindow::display_calibration_auto)
/// and an HDR resolved transfer, so SDR and all-[`Keep`](bevy_window::AutoField::Keep)
/// projects never pay it. A read on a platform that reports nothing marks the
/// surface seen (so it is not re-read every frame) but commits nothing — `None`
/// never means "SDR".
pub fn poll_display_state(
    // Apple's relative-headroom query gates on the main thread; pin the system
    // there, matching `create_surfaces`.
    #[cfg(any(target_os = "macos", target_os = "ios"))] _marker: bevy_ecs::system::NonSendMarker,
    window_surfaces: Res<WindowSurfaces>,
    extracted_windows: Res<ExtractedWindows>,
    render_adapter: Res<RenderAdapter>,
    mut store: ResMut<DisplayStateStore>,
) {
    // Drop bookkeeping for surfaces that went away.
    store
        .states
        .retain(|e, _| window_surfaces.surfaces.contains_key(e));
    store
        .capabilities
        .retain(|e, _| window_surfaces.surfaces.contains_key(e));
    store
        .last_resolved
        .retain(|e, _| window_surfaces.surfaces.contains_key(e));

    for (&entity, surface_data) in window_surfaces.surfaces.iter() {
        let extracted = extracted_windows.get(&entity);
        let first_time = !store.states.contains_key(&entity);
        let reconfigured = extracted.is_some_and(|w| w.display_target_transfer_changed);
        let event_requery = extracted.is_some_and(|w| w.request_display_requery);

        // A renegotiation that changed the resolved transfer with no authored
        // change (OS HDR enable/disable) — re-read so the live state reflects it.
        let resolved = surface_data.resolved_transfer;
        let resolved_changed = store.last_resolved.get(&entity) != Some(&resolved);
        if resolved_changed {
            store.last_resolved.insert(entity, resolved);
        }

        // Continuous live re-read: only where a signal drifts with no event, and
        // only while an HDR surface auto-calibrates — so SDR and all-`Keep`
        // projects never pay the per-frame read. Today that signal is the Apple
        // EDR headroom; the OS gate stands in for "this surface's headroom drifts
        // without an event" until wgpu exposes it as a surface capability.
        let continuous = cfg!(any(target_os = "macos", target_os = "ios"))
            && resolved.is_hdr()
            && extracted.is_some_and(|w| w.display_calibration_auto);

        if !(first_time || reconfigured || event_requery || resolved_changed || continuous) {
            continue;
        }

        let info = surface_data.surface.display_hdr_info(&render_adapter);
        let Some((state, capability)) = read_display_state(&info) else {
            // "Can't tell": mark the surface seen so it is not re-read every
            // frame on a no-HDR platform, but never clobber a committed value.
            // The capability is left untouched: a display that can't report now
            // keeps its last-known capability rather than a spurious all-`None`
            // (a stale capability after a switch to an unreporting monitor is the
            // accepted trade — `None` never means "SDR").
            store.states.entry(entity).or_default();
            continue;
        };

        commit(&mut store, entity, state, capability);
    }
}

/// Mirrors everything the render world learned about each window surface back to
/// the main world: the negotiated and supported transfers
/// ([`WindowSurfaceTransfers`], so apps can detect a downgraded request and
/// offer only the modes that will actually work), the surface's committed
/// [`WindowDisplayState`], and its [`MonitorDisplayCapability`] on the
/// [`Monitor`] entity the window is on (resolved through [`OnMonitor`], so every
/// window on a display shares one record).
///
/// Runs during extraction — the render world's only window into the main world —
/// so every value lags the frame that produced it by one. Insert-on-change, so
/// [`Changed`] stays a usable signal.
///
/// [`Monitor`]: bevy_window::Monitor
pub fn write_back_display_state(
    mut main_world: ResMut<MainWorld>,
    window_surfaces: Res<WindowSurfaces>,
    store: Res<DisplayStateStore>,
) {
    for (&entity, surface_data) in window_surfaces.surfaces.iter() {
        super::insert_on_change(
            &mut main_world,
            entity,
            WindowSurfaceTransfers {
                resolved: surface_data.resolved_transfer,
                supported: surface_data.supported_transfers,
            },
        );

        // Absent until this surface's first successful read: a platform that
        // reports nothing leaves the carrier off the window entirely.
        if let Some(state) = store.states.get(&entity).copied().flatten() {
            super::insert_on_change(&mut main_world, entity, state);
        }

        let Some(&capability) = store.capabilities.get(&entity) else {
            continue;
        };
        let Some(monitor_entity) = main_world
            .get_entity(entity)
            .ok()
            .and_then(|w| w.get::<OnMonitor>())
            .map(|on_monitor| on_monitor.0)
        else {
            continue;
        };
        super::insert_on_change(&mut main_world, monitor_entity, capability);
    }
}

#[cfg(test)]
mod read_tests {
    use super::*;
    use wgpu::{DisplayCoarseRange as WgpuCoarseRange, DisplayHeadroom, DisplayLuminance};

    /// Builds a [`DisplayLuminance`] by named field, for readability at the call
    /// sites below.
    fn luminance(
        max_nits: Option<f32>,
        max_full_frame_nits: Option<f32>,
        min_nits: Option<f32>,
        sdr_white_nits: Option<f32>,
    ) -> DisplayLuminance {
        DisplayLuminance {
            max_nits,
            max_full_frame_nits,
            min_nits,
            sdr_white_nits,
        }
    }

    fn headroom(
        current: Option<f32>,
        potential: Option<f32>,
        reference: Option<f32>,
    ) -> DisplayHeadroom {
        DisplayHeadroom {
            current,
            potential,
            reference,
        }
    }

    /// Builds a coarse-range report with the given HDR-capability flag.
    fn coarse(high_dynamic_range: Option<bool>) -> WgpuCoarseRange {
        WgpuCoarseRange {
            high_dynamic_range,
            gamut: None,
        }
    }

    #[test]
    fn macos_relative_uses_folded_headroom() {
        // Apple path: no absolute nits, only the EDR headroom. wgpu folds it to
        // the live `current` multiplier; we never reconstruct a peak in nits.
        let info = DisplayHdrInfo {
            headroom: Some(headroom(Some(4.0), Some(5.0), None)),
            ..Default::default()
        };
        let (state, capability) = read_display_state(&info).unwrap();
        // `tone_map_headroom()` returns Apple's live `current`, not `potential`.
        assert_eq!(state.tone_map_headroom, Some(4.0));
        // Apple reports no absolute nits.
        assert_eq!(state.sdr_white_nits, None);
        assert_eq!(capability.max_nits, None);
    }

    #[test]
    fn windows_absolute_folds_to_nit_ratio() {
        let info = DisplayHdrInfo {
            luminance: Some(luminance(
                Some(1000.0),
                Some(600.0),
                Some(0.01),
                Some(200.0),
            )),
            ..Default::default()
        };
        let (state, capability) = read_display_state(&info).unwrap();
        // Folded multiplier is the nit ratio 1000 / 200.
        assert_eq!(state.tone_map_headroom, Some(5.0));
        assert_eq!(state.sdr_white_nits, Some(200.0));
        assert_eq!(capability.max_nits, Some(1000.0));
        assert_eq!(capability.max_full_frame_nits, Some(600.0));
        assert_eq!(capability.min_nits, Some(0.01));
    }

    #[test]
    fn windows_sdr_collapses_headroom_to_one() {
        // An SDR-mode Windows output still reports its EDID peak (270) against an
        // 80-nit SDR white, but the coarse flag marks it SDR. The folded value
        // collapses to 1.0 rather than the phantom 270 / 80, while the capability
        // half still carries the panel's physical peak.
        let info = DisplayHdrInfo {
            luminance: Some(luminance(Some(270.0), None, None, Some(80.0))),
            coarse: Some(coarse(Some(false))),
            ..Default::default()
        };
        let (state, capability) = read_display_state(&info).unwrap();
        assert_eq!(state.tone_map_headroom, Some(1.0));
        assert_eq!(capability.max_nits, Some(270.0));
    }

    #[test]
    fn none_stays_none_never_sdr() {
        assert!(read_display_state(&DisplayHdrInfo::default()).is_none());
    }

    #[test]
    fn non_finite_filtered_out() {
        let info = DisplayHdrInfo {
            luminance: Some(luminance(Some(f32::NAN), None, None, Some(0.0))),
            ..Default::default()
        };
        let (state, capability) = read_display_state(&info).unwrap();
        // NaN peak and the 0-nit white both fold to "not reported"; with no
        // usable nits and no headroom, the folded multiplier is unknown. The read
        // still counts as a read: the platform named the fields, it just filled
        // them with garbage.
        assert_eq!(capability.max_nits, None);
        assert_eq!(state.sdr_white_nits, None);
        assert_eq!(state.tone_map_headroom, None);
    }

    #[test]
    fn reports_anything_covers_every_reporting_model() {
        let absolute = DisplayHdrInfo {
            luminance: Some(luminance(Some(1000.0), None, None, None)),
            headroom: Some(headroom(Some(4.0), Some(5.0), None)),
            ..Default::default()
        };
        assert!(reports_anything(&absolute));

        let relative = DisplayHdrInfo {
            headroom: Some(headroom(Some(4.0), Some(5.0), None)),
            ..Default::default()
        };
        assert!(reports_anything(&relative));

        let coarse_only = DisplayHdrInfo {
            coarse: Some(coarse(Some(true))),
            ..Default::default()
        };
        assert!(reports_anything(&coarse_only));

        // A luminance record whose every absolute field is absent reports
        // nothing: the backend named the struct but filled in no value.
        let empty_luminance = DisplayHdrInfo {
            luminance: Some(luminance(None, Some(600.0), Some(0.01), None)),
            ..Default::default()
        };
        assert!(!reports_anything(&empty_luminance));

        assert!(!reports_anything(&DisplayHdrInfo::default()));
    }

    #[test]
    fn rel_changed_transitions() {
        // Presence transitions always count; equal-both-absent never does.
        assert!(rel_changed(Some(5.0), None));
        assert!(rel_changed(None, Some(5.0)));
        assert!(!rel_changed(None, None));
        // Sub-epsilon vs supra-epsilon while both present.
        assert!(!rel_changed(Some(5.0), Some(5.02)));
        assert!(rel_changed(Some(5.0), Some(5.5)));
    }
}

#[cfg(test)]
mod commit_tests {
    use super::*;

    fn state(tone_map_headroom: Option<f32>, sdr_white_nits: Option<f32>) -> WindowDisplayState {
        WindowDisplayState {
            tone_map_headroom,
            sdr_white_nits,
        }
    }

    fn committed(store: &DisplayStateStore, entity: Entity) -> Option<WindowDisplayState> {
        store.states.get(&entity).copied().flatten()
    }

    #[test]
    fn first_read_commits() {
        let mut store = DisplayStateStore::default();
        let entity = Entity::from_raw_u32(1).unwrap();

        commit(
            &mut store,
            entity,
            state(Some(5.0), None),
            MonitorDisplayCapability::default(),
        );

        assert_eq!(committed(&store, entity), Some(state(Some(5.0), None)));
    }

    #[test]
    fn sub_epsilon_change_does_not_commit() {
        let mut store = DisplayStateStore::default();
        let entity = Entity::from_raw_u32(2).unwrap();

        let capability = MonitorDisplayCapability::default();
        commit(&mut store, entity, state(Some(5.0), None), capability);

        // A 0.4% change is below the 1% relative epsilon: never commits, so the
        // insert-on-change write-back never fires.
        commit(&mut store, entity, state(Some(5.02), None), capability);
        assert_eq!(committed(&store, entity), Some(state(Some(5.0), None)));
    }

    #[test]
    fn supra_epsilon_change_commits() {
        let mut store = DisplayStateStore::default();
        let entity = Entity::from_raw_u32(3).unwrap();

        let capability = MonitorDisplayCapability::default();
        commit(&mut store, entity, state(Some(5.0), None), capability);

        // A 10% change is above the epsilon: commits.
        commit(&mut store, entity, state(Some(5.5), None), capability);
        assert_eq!(committed(&store, entity), Some(state(Some(5.5), None)));
    }

    #[test]
    fn sdr_white_change_commits_on_its_own() {
        // The SDR-white anchor is smoothed independently of the headroom: the
        // Windows brightness slider moves it while the headroom holds still.
        let mut store = DisplayStateStore::default();
        let entity = Entity::from_raw_u32(4).unwrap();

        let capability = MonitorDisplayCapability::default();
        commit(&mut store, entity, state(Some(2.0), Some(80.0)), capability);

        commit(
            &mut store,
            entity,
            state(Some(2.0), Some(200.0)),
            capability,
        );
        assert_eq!(
            committed(&store, entity),
            Some(state(Some(2.0), Some(200.0)))
        );
    }
}
