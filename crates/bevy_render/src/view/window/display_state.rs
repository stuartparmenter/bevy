//! Render-side display sensing. [`poll_display_state`] re-reads each surface's
//! live HDR state when it can have changed, [`DisplayStateStore`] suppresses
//! sub-threshold jitter, and [`write_back_display_state`] mirrors the result to
//! the main world as [`WindowDisplayState`] and [`MonitorDisplayCapability`].
//!
//! The single live value the tone mapper consumes is
//! [`DisplayHdrInfo::tone_map_headroom`], the linear multiplier of SDR white
//! the display can drive right now. wgpu folds the platform-specific
//! reporting into it (Apple's relative EDR headroom, Windows' absolute
//! `max_nits / sdr_white_nits` ratio, the coarse SDR flag), so this module
//! never reconstructs an absolute peak from platform-specific fields.
//!
//! `bevy_window` stays wgpu-free and holds only the plain-data result types.
//! Everything that touches a wgpu [`DisplayHdrInfo`] lives here.

use bevy_ecs::prelude::*;
use bevy_platform::collections::HashMap;
use bevy_window::{
    DisplayGamut, MonitorDisplayCapability, OnMonitor, WindowDisplayState, WindowSurfaceTransfers,
};
use wgpu::{DisplayGamut as WgpuDisplayGamut, DisplayHdrInfo};

use crate::renderer::RenderAdapter;
use crate::sync_world::MainEntity;
use crate::MainWorld;

use super::{ExtractedWindow, SurfaceData};

/// Maps a wgpu coarse [`DisplayGamut`](WgpuDisplayGamut) onto the plain-data
/// [`DisplayGamut`] the window crate carries. `Srgb` and any unrecognized future
/// wgpu variant map to [`DisplayGamut::Rec709`], the narrowest known gamut and
/// the conservative choice for a capability hint.
fn map_gamut(g: WgpuDisplayGamut) -> DisplayGamut {
    match g {
        WgpuDisplayGamut::DisplayP3 => DisplayGamut::DisplayP3,
        WgpuDisplayGamut::Rec2020 => DisplayGamut::Rec2020,
        _ => DisplayGamut::Rec709,
    }
}

/// Returns `Some` only for finite, strictly positive values. A value wgpu
/// reported as NaN, infinity, zero, or negative becomes "not reported".
pub(crate) fn finite_positive(v: Option<f32>) -> Option<f32> {
    v.filter(|x| x.is_finite() && *x > 0.0)
}

/// Whether a [`DisplayHdrInfo`] carries any usable reporting model: absolute
/// luminance (Windows), a relative headroom (Apple), or a coarse capability
/// (web). `false` means this platform or this moment reports nothing. It never
/// means "SDR".
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
/// [`tone_map_headroom`](DisplayHdrInfo::tone_map_headroom), which reads `1.0`
/// for a display that reports itself SDR. `sdr_white_nits` is carried alongside
/// (Windows only) to anchor the `paper_white` auto-calibration.
///
/// The capability half is copied through for the [`MonitorDisplayCapability`]
/// mirror, including the coarse gamut bucket the backend reports (a
/// nearest-primaries match on DXGI, the CSS `color-gamut` query on web).
fn read_display_state(
    info: &DisplayHdrInfo,
) -> Option<(WindowDisplayState, MonitorDisplayCapability)> {
    if !reports_anything(info) {
        return None;
    }

    let luminance = info.luminance;
    let state = WindowDisplayState {
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

/// Relative change below which a continuous field counts as unchanged. Small
/// enough to follow the Apple EDR ramp (it climbs over 1-2 seconds), large
/// enough to swallow float noise.
const EPSILON_REL: f32 = 0.01;

/// Per-surface committed live state and last capability, keyed by window entity.
/// Render-world only.
#[derive(Resource, Default)]
pub(crate) struct DisplayStateStore {
    /// The committed (post-epsilon) [`WindowDisplayState`] per surface. A
    /// present-but-`None` entry marks a surface that was read on a platform that
    /// reports nothing: it stops the per-frame re-read, and the component stays
    /// absent until a real read lands.
    states: HashMap<Entity, Option<WindowDisplayState>>,
    /// The last committed capability per surface (so the [`Monitor`] write-back
    /// is insert-on-change).
    ///
    /// [`Monitor`]: bevy_window::Monitor
    capabilities: HashMap<Entity, MonitorDisplayCapability>,
}

/// Whether two optional continuous values differ by more than the relative
/// epsilon. A `None` to `Some` or `Some` to `None` transition always counts.
fn rel_changed(old: Option<f32>, new: Option<f32>) -> bool {
    match (old, new) {
        (Some(a), Some(b)) => (a - b).abs() > EPSILON_REL * a.abs().max(f32::MIN_POSITIVE),
        (None, None) => false,
        _ => true,
    }
}

/// Folds a fresh read into the committed live state for `entity`. A read
/// commits only when a field moves past the relative [`EPSILON_REL`], so
/// [`Changed<WindowDisplayState>`](bevy_ecs::prelude::Changed) signals a real
/// transition rather than read jitter. The capability half is not smoothed;
/// write-back handles it insert-on-change.
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
/// Main-thread-pinned on Apple platforms: the relative-headroom query returns
/// `None` off the main thread. A surface is read when:
///
/// - it is seen for the first time, to seed the mirror, or
/// - its surface was just (re)configured
///   ([`display_target_transfer_changed`](super::ExtractedWindow::display_target_transfer_changed)),
///   or
/// - a window event flagged it for re-query
///   ([`request_display_requery`](super::ExtractedWindow::request_display_requery)):
///   a move, a focus regain, a monitor change, or a surface renegotiation that
///   changed the resolved transfer, or
/// - it is auto-calibrating
///   ([`display_calibration_auto`](super::ExtractedWindow::display_calibration_auto))
///   an HDR surface on a platform whose live value drifts with no event (Apple
///   EDR headroom), in which case it is read every frame.
///
/// A read on a platform that reports nothing commits nothing.
pub(crate) fn poll_display_state(
    #[cfg(any(target_os = "macos", target_os = "ios"))] _marker: bevy_ecs::system::NonSendMarker,
    windows: Query<(MainEntity, &ExtractedWindow, &SurfaceData)>,
    render_adapter: Res<RenderAdapter>,
    mut store: ResMut<DisplayStateStore>,
) {
    // Drop bookkeeping for surfaces that went away.
    let live: bevy_ecs::entity::EntityHashSet = windows
        .iter()
        .map(|(main_entity, ..)| main_entity)
        .collect();
    store.states.retain(|e, _| live.contains(e));
    store.capabilities.retain(|e, _| live.contains(e));

    for (entity, extracted, surface_data) in windows.iter() {
        let first_time = !store.states.contains_key(&entity);
        let reconfigured = extracted.display_target_transfer_changed;
        let event_requery = extracted.request_display_requery;

        let resolved = surface_data.resolved_transfer;

        // The OS gate stands in for "this surface's headroom drifts without an
        // event" until wgpu exposes that as a surface capability. Today the
        // Apple EDR headroom is the only such signal.
        let continuous = cfg!(any(target_os = "macos", target_os = "ios"))
            && resolved.is_hdr()
            && extracted.display_calibration_auto;

        if !(first_time || reconfigured || event_requery || continuous) {
            continue;
        }

        let info = surface_data.surface.display_hdr_info(&render_adapter);
        let Some((state, capability)) = read_display_state(&info) else {
            // Nothing was reported. Mark the surface seen so it is not re-read
            // every frame on a no-HDR platform, but never clobber a committed
            // value. The capability is left untouched too, so a display that
            // cannot report keeps its last-known one rather than a spurious
            // all-`None`.
            store.states.entry(entity).or_default();
            continue;
        };

        commit(&mut store, entity, state, capability);
    }
}

/// Mirrors what the render world learned about each window surface back to the
/// main world: the negotiated and supported transfers
/// ([`WindowSurfaceTransfers`], so apps can spot a downgraded request and offer
/// only the modes that will work), the surface's committed
/// [`WindowDisplayState`], and its [`MonitorDisplayCapability`]. The capability
/// goes on the [`Monitor`] entity the window is on (resolved through
/// [`OnMonitor`]), so every window on a display shares one record.
///
/// Runs during extraction, the render world's only window into the main world,
/// so every value lags the frame that produced it by one. Writes are
/// insert-on-change, so [`Changed`] stays a usable signal.
///
/// [`Monitor`]: bevy_window::Monitor
pub(crate) fn write_back_display_state(
    mut main_world: ResMut<MainWorld>,
    windows: Query<(MainEntity, &SurfaceData)>,
    store: Res<DisplayStateStore>,
) {
    for (entity, surface_data) in windows.iter() {
        super::insert_on_change(
            &mut main_world,
            entity,
            WindowSurfaceTransfers {
                resolved: surface_data.resolved_transfer,
                supported: surface_data.supported_transfers,
            },
        );

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

    fn coarse(high_dynamic_range: Option<bool>) -> WgpuCoarseRange {
        WgpuCoarseRange {
            high_dynamic_range,
            gamut: None,
        }
    }

    #[test]
    fn macos_relative_uses_folded_headroom() {
        // Apple path: no absolute nits, only the EDR headroom.
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
        // An SDR-mode Windows output still reports its EDID peak (270) against
        // an 80-nit SDR white, but the coarse flag marks it SDR, so the folded
        // value collapses to 1.0 rather than the phantom 270 / 80.
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
        // NaN peak and the 0-nit white both fold to "not reported", and with no
        // usable nits and no headroom the folded multiplier is unknown. The read
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

        // Only `max_nits` and `sdr_white_nits` are consulted, so a record that
        // fills in neither reports nothing.
        let empty_luminance = DisplayHdrInfo {
            luminance: Some(luminance(None, Some(600.0), Some(0.01), None)),
            ..Default::default()
        };
        assert!(!reports_anything(&empty_luminance));

        assert!(!reports_anything(&DisplayHdrInfo::default()));
    }

    #[test]
    fn rel_changed_transitions() {
        // Presence transitions always count. Both absent never does.
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
    fn sub_epsilon_change_does_not_commit() {
        let mut store = DisplayStateStore::default();
        let entity = Entity::from_raw_u32(2).unwrap();

        let capability = MonitorDisplayCapability::default();
        commit(&mut store, entity, state(Some(5.0), None), capability);

        // A 0.4% change is below the 1% relative epsilon, so it never commits
        // and the insert-on-change write-back never fires.
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
        // The SDR-white anchor is smoothed on its own, not with the headroom:
        // the Windows brightness slider moves it while the headroom holds still.
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
