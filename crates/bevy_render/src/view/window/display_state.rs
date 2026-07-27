//! Render-side display sensing: re-reads each surface's live HDR state when
//! something can actually have changed — first sight, a surface reconfigure, a
//! window event (move / focus / monitor change), or, for the one signal that
//! drifts with no event (the Apple EDR headroom), every frame while an HDR
//! surface is auto-calibrating. It folds every platform's reporting asymmetry
//! into one cross-platform value (`normalize`), suppresses sub-threshold
//! jitter ([`DisplayStateStore`]), and mirrors the result back to the main world
//! as [`WindowDisplayState`] / [`MonitorDisplayCapability`].
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
    DisplayGamut, DisplayInfoSource, DisplayTransfer, MonitorDisplayCapability, OnMonitor,
    WindowDisplayState,
};
use wgpu::{DisplayGamut as WgpuDisplayGamut, DisplayHdrInfo};

use crate::renderer::RenderAdapter;
use crate::MainWorld;

use super::{ExtractedWindows, WindowSurfaces};

/// A platform-agnostic snapshot of one surface's display state, produced by
/// [`normalize`]. All cross-platform asymmetry collapses here: the live half is
/// a single [`tone_map_headroom`](Self::tone_map_headroom) multiplier plus the
/// SDR-white anchor, and callers downstream never see per-backend `Option`
/// shapes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DisplaySnapshot {
    // Live-state fields (mirrored to `WindowDisplayState`).
    pub tone_map_headroom: Option<f32>,
    pub sdr_white_nits: Option<f32>,
    // Capability fields (mirrored to `MonitorDisplayCapability`).
    pub max_nits: Option<f32>,
    pub max_full_frame_nits: Option<f32>,
    pub min_nits: Option<f32>,
    pub gamut_hint: Option<DisplayGamut>,
    /// Provenance of this snapshot, shared by the live-state and capability
    /// halves (a single read has one source).
    pub source: DisplayInfoSource,
}

impl DisplaySnapshot {
    /// The capability half of this snapshot as a [`MonitorDisplayCapability`]
    /// (a field-for-field copy).
    fn capability(&self) -> MonitorDisplayCapability {
        MonitorDisplayCapability {
            max_nits: self.max_nits,
            max_full_frame_nits: self.max_full_frame_nits,
            min_nits: self.min_nits,
            gamut_hint: self.gamut_hint,
            source: self.source,
        }
    }

    /// The live-state half of this snapshot as a [`WindowDisplayState`]
    /// candidate. `generation` starts at zero; [`commit`] bumps it when the
    /// candidate actually commits.
    fn live_state(&self) -> WindowDisplayState {
        WindowDisplayState {
            tone_map_headroom: self.tone_map_headroom,
            sdr_white_nits: self.sdr_white_nits,
            source: self.source,
            generation: 0,
        }
    }
}

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

/// Collapses a wgpu [`DisplayHdrInfo`] into a [`DisplaySnapshot`].
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
/// [`MonitorDisplayCapability`] mirror. `source` is the caller-classified
/// provenance ([`classify_source`]); it tags the mirror for calibration UIs and
/// never affects how a value commits.
pub(crate) fn normalize(info: &DisplayHdrInfo, source: DisplayInfoSource) -> DisplaySnapshot {
    let luminance = info.luminance;

    // The single cross-platform live HDR value, already folded by wgpu.
    let tone_map_headroom = finite_positive(info.tone_map_headroom());
    let sdr_white_nits = luminance.and_then(|l| finite_positive(l.sdr_white_nits));

    let max_nits = finite_positive(luminance.and_then(|l| l.max_nits));
    let max_full_frame_nits = finite_positive(luminance.and_then(|l| l.max_full_frame_nits));
    let min_nits = finite_positive(luminance.and_then(|l| l.min_nits));

    let gamut_hint = info.coarse.and_then(|c| c.gamut).map(map_gamut);

    DisplaySnapshot {
        tone_map_headroom,
        sdr_white_nits,
        max_nits,
        max_full_frame_nits,
        min_nits,
        gamut_hint,
        source,
    }
}

/// Classifies which reporting model a [`DisplayHdrInfo`] represents, so callers
/// pass the right [`DisplayInfoSource`] to [`normalize`]. Absolute luminance →
/// [`Os`](DisplayInfoSource::Os); relative headroom →
/// [`DerivedFromHeadroom`](DisplayInfoSource::DerivedFromHeadroom); coarse-only
/// → [`Web`](DisplayInfoSource::Web); nothing usable →
/// [`Unknown`](DisplayInfoSource::Unknown). This is pure provenance: it tags the
/// mirror for calibration UIs and does not change how a value commits.
pub(crate) fn classify_source(info: &DisplayHdrInfo) -> DisplayInfoSource {
    if info
        .luminance
        .is_some_and(|l| l.max_nits.is_some() || l.sdr_white_nits.is_some())
    {
        DisplayInfoSource::Os
    } else if info.headroom.is_some() {
        DisplayInfoSource::DerivedFromHeadroom
    } else if info.coarse.is_some() {
        DisplayInfoSource::Web
    } else {
        DisplayInfoSource::Unknown
    }
}

/// Relative change below which a continuous field is treated as unchanged, so
/// sub-threshold read jitter never bumps `generation` / fires
/// [`Changed`](bevy_ecs::prelude::Changed). Small enough to track the Apple EDR
/// ramp smoothly (it climbs over ~1–2 s), large enough to swallow float noise.
const EPSILON_REL: f32 = 0.01;

/// Per-surface committed live state and last capability, keyed by window entity.
/// Render-world only.
#[derive(Resource, Default)]
pub struct DisplayStateStore {
    /// The committed (post-epsilon) [`WindowDisplayState`] per surface.
    states: HashMap<Entity, WindowDisplayState>,
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

/// Folds a fresh snapshot into the committed live state for `entity`. A field
/// commits only when it moves past the relative [`EPSILON_REL`] (or the source
/// changes), and `generation` bumps only on a real commit — so
/// [`Changed<WindowDisplayState>`](bevy_ecs::prelude::Changed) signals a genuine
/// transition rather than read jitter. The capability half passes through
/// unsmoothed (insert-on-change is handled at write-back).
fn commit(store: &mut DisplayStateStore, entity: Entity, snap: &DisplaySnapshot) {
    let candidate = snap.live_state();
    let committed = store.states.entry(entity).or_default();

    let changed = rel_changed(committed.tone_map_headroom, candidate.tone_map_headroom)
        || rel_changed(committed.sdr_white_nits, candidate.sdr_white_nits)
        || committed.source != candidate.source;

    if changed {
        let generation = committed.generation.wrapping_add(1);
        *committed = candidate;
        committed.generation = generation;
    }

    store.capabilities.insert(entity, snap.capability());
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
/// projects never pay it. An [`Unknown`](DisplayInfoSource::Unknown) read marks
/// the surface seen (so it is not re-read every frame on a platform that reports
/// nothing) but never overwrites a committed value — `None` never means "SDR".
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
        let source = classify_source(&info);
        if source == DisplayInfoSource::Unknown {
            // "Can't tell": mark the surface seen so it is not re-read every
            // frame on a no-HDR platform, but never clobber a committed value.
            // The capability is left untouched: a display that can't report now
            // keeps its last-known capability rather than a spurious all-`None`
            // (a stale capability after a switch to an unreporting monitor is the
            // accepted trade — `None` never means "SDR").
            store.states.entry(entity).or_default();
            continue;
        }

        commit(&mut store, entity, &normalize(&info, source));
    }
}

/// Mirrors each surface's committed [`WindowDisplayState`] back to its window
/// entity, and its [`MonitorDisplayCapability`] back to the [`Monitor`] entity
/// the window is on (resolved through [`OnMonitor`]). Runs during extraction —
/// the render world's only window into the main world — so the value lags the
/// read by one frame. Insert-on-change, so
/// [`Changed`] stays a usable signal.
///
/// [`Monitor`]: bevy_window::Monitor
pub fn write_back_display_state(mut main_world: ResMut<MainWorld>, store: Res<DisplayStateStore>) {
    for (&entity, state) in store.states.iter() {
        super::insert_on_change(&mut main_world, entity, *state);
    }

    // Resolve each window's Monitor entity through `OnMonitor`, then mirror the
    // capability there (every window on a display shares one record).
    for (&entity, capability) in store.capabilities.iter() {
        let Some(monitor_entity) = main_world
            .get_entity(entity)
            .ok()
            .and_then(|w| w.get::<OnMonitor>())
            .map(|on_monitor| on_monitor.0)
        else {
            continue;
        };
        super::insert_on_change(&mut main_world, monitor_entity, *capability);
    }
}

#[cfg(test)]
mod normalize_tests {
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
        let snap = normalize(&info, DisplayInfoSource::DerivedFromHeadroom);
        // `tone_map_headroom()` returns Apple's live `current`, not `potential`.
        assert_eq!(snap.tone_map_headroom, Some(4.0));
        // Apple reports no absolute nits.
        assert_eq!(snap.sdr_white_nits, None);
        assert_eq!(snap.max_nits, None);
        assert_eq!(snap.source, DisplayInfoSource::DerivedFromHeadroom);
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
        let snap = normalize(&info, DisplayInfoSource::Os);
        // Folded multiplier is the nit ratio 1000 / 200.
        assert_eq!(snap.tone_map_headroom, Some(5.0));
        assert_eq!(snap.max_nits, Some(1000.0));
        assert_eq!(snap.max_full_frame_nits, Some(600.0));
        assert_eq!(snap.sdr_white_nits, Some(200.0));
        assert_eq!(snap.min_nits, Some(0.01));
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
        let snap = normalize(&info, DisplayInfoSource::Os);
        assert_eq!(snap.tone_map_headroom, Some(1.0));
        assert_eq!(snap.max_nits, Some(270.0));
    }

    #[test]
    fn none_stays_none_never_sdr() {
        let info = DisplayHdrInfo::default();
        assert_eq!(classify_source(&info), DisplayInfoSource::Unknown);
        let snap = normalize(&info, DisplayInfoSource::Unknown);
        assert_eq!(snap.tone_map_headroom, None);
        assert_eq!(snap.max_nits, None);
        assert_eq!(snap.sdr_white_nits, None);
    }

    #[test]
    fn non_finite_filtered_out() {
        let info = DisplayHdrInfo {
            luminance: Some(luminance(Some(f32::NAN), None, None, Some(0.0))),
            ..Default::default()
        };
        let snap = normalize(&info, DisplayInfoSource::Os);
        // NaN peak and the 0-nit white both fold to "not reported"; with no
        // usable nits and no headroom, the folded multiplier is unknown.
        assert_eq!(snap.max_nits, None);
        assert_eq!(snap.sdr_white_nits, None);
        assert_eq!(snap.tone_map_headroom, None);
    }

    #[test]
    fn classify_prefers_absolute_then_headroom_then_coarse() {
        let absolute = DisplayHdrInfo {
            luminance: Some(luminance(Some(1000.0), None, None, None)),
            headroom: Some(headroom(Some(4.0), Some(5.0), None)),
            ..Default::default()
        };
        assert_eq!(classify_source(&absolute), DisplayInfoSource::Os);

        let relative = DisplayHdrInfo {
            headroom: Some(headroom(Some(4.0), Some(5.0), None)),
            ..Default::default()
        };
        assert_eq!(
            classify_source(&relative),
            DisplayInfoSource::DerivedFromHeadroom
        );

        let coarse_only = DisplayHdrInfo {
            coarse: Some(coarse(Some(true))),
            ..Default::default()
        };
        assert_eq!(classify_source(&coarse_only), DisplayInfoSource::Web);
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

    fn snapshot(source: DisplayInfoSource) -> DisplaySnapshot {
        DisplaySnapshot {
            tone_map_headroom: None,
            sdr_white_nits: None,
            max_nits: None,
            max_full_frame_nits: None,
            min_nits: None,
            gamut_hint: None,
            source,
        }
    }

    fn committed(store: &DisplayStateStore, entity: Entity) -> WindowDisplayState {
        *store.states.get(&entity).unwrap()
    }

    #[test]
    fn commit_bumps_generation_on_change() {
        let mut store = DisplayStateStore::default();
        let entity = Entity::from_raw_u32(1).unwrap();

        let mut snap = snapshot(DisplayInfoSource::Os);
        snap.tone_map_headroom = Some(5.0);
        commit(&mut store, entity, &snap);

        let state = committed(&store, entity);
        assert_eq!(state.tone_map_headroom, Some(5.0));
        assert_eq!(state.generation, 1);
        assert_eq!(state.source, DisplayInfoSource::Os);
    }

    #[test]
    fn sub_epsilon_change_does_not_bump() {
        let mut store = DisplayStateStore::default();
        let entity = Entity::from_raw_u32(2).unwrap();

        let mut snap = snapshot(DisplayInfoSource::DerivedFromHeadroom);
        snap.tone_map_headroom = Some(5.0);
        commit(&mut store, entity, &snap);
        let generation = committed(&store, entity).generation;

        // A 0.4% change is below the 1% relative epsilon: never commits.
        snap.tone_map_headroom = Some(5.02);
        commit(&mut store, entity, &snap);
        assert_eq!(committed(&store, entity).tone_map_headroom, Some(5.0));
        assert_eq!(committed(&store, entity).generation, generation);
    }

    #[test]
    fn supra_epsilon_change_commits_and_bumps() {
        let mut store = DisplayStateStore::default();
        let entity = Entity::from_raw_u32(3).unwrap();

        let mut snap = snapshot(DisplayInfoSource::DerivedFromHeadroom);
        snap.tone_map_headroom = Some(5.0);
        commit(&mut store, entity, &snap);
        let generation = committed(&store, entity).generation;

        // A 10% change is above the epsilon: commits and bumps once.
        snap.tone_map_headroom = Some(5.5);
        commit(&mut store, entity, &snap);
        assert_eq!(committed(&store, entity).tone_map_headroom, Some(5.5));
        assert_eq!(committed(&store, entity).generation, generation + 1);
    }

    #[test]
    fn source_change_commits() {
        // A provenance change commits even when the numeric value is identical,
        // so the mirrored source stays truthful.
        let mut store = DisplayStateStore::default();
        let entity = Entity::from_raw_u32(4).unwrap();

        let mut snap = snapshot(DisplayInfoSource::Os);
        snap.tone_map_headroom = Some(2.0);
        commit(&mut store, entity, &snap);
        let generation = committed(&store, entity).generation;

        snap.source = DisplayInfoSource::DerivedFromHeadroom;
        commit(&mut store, entity, &snap);
        assert_eq!(
            committed(&store, entity).source,
            DisplayInfoSource::DerivedFromHeadroom
        );
        assert_eq!(committed(&store, entity).generation, generation + 1);
    }
}
