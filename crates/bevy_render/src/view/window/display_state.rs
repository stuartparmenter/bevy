//! Reads what the display behind each window surface reports, writes it back
//! to the main world, and resolves each window's [`EffectiveDisplayTarget`].
//!
//! `bevy_window` does not depend on wgpu, so everything that reads a
//! [`DisplayHdrInfo`] is here.

use bevy_ecs::entity::EntityHashMap;
use bevy_ecs::prelude::*;
use bevy_window::{
    DisplayCalibrationPolicy, DisplayGamut, DisplayProvenance, DisplayTarget,
    EffectiveDisplayTarget, FieldProvenance, MonitorDisplayCapability, OnMonitor, Window,
    WindowDisplayState, WindowSurfaceTransfers,
};
use wgpu::{DisplayGamut as WgpuDisplayGamut, DisplayHdrInfo};

use crate::renderer::RenderAdapter;
use crate::sync_world::MainEntity;
use crate::MainWorld;

use super::{ExtractedWindow, SurfaceData};

/// [`Srgb`](WgpuDisplayGamut::Srgb) and any variant wgpu adds later map to
/// [`DisplayGamut::Rec709`], the narrowest gamut, so the result is never wider
/// than what the display covers.
fn map_gamut(g: WgpuDisplayGamut) -> DisplayGamut {
    match g {
        WgpuDisplayGamut::DisplayP3 => DisplayGamut::DisplayP3,
        WgpuDisplayGamut::Rec2020 => DisplayGamut::Rec2020,
        _ => DisplayGamut::Rec709,
    }
}

/// Returns `Some` only for finite, positive values, so an invalid value from
/// the platform counts as not reported.
fn finite_positive(v: f32) -> Option<f32> {
    (v.is_finite() && v > 0.0).then_some(v)
}

/// Whether the platform reported a value this module uses. `false` means
/// unknown, not SDR.
fn reports_anything(info: &DisplayHdrInfo) -> bool {
    info.luminance
        .is_some_and(|l| l.max_nits.is_some() || l.sdr_white_nits.is_some())
        || info.headroom.is_some()
        || info.coarse.is_some()
}

/// Splits a [`DisplayHdrInfo`] into the two `bevy_window` components, or
/// returns `None` when the platform reported nothing.
fn read_display_state(
    info: &DisplayHdrInfo,
) -> Option<(WindowDisplayState, MonitorDisplayCapability)> {
    if !reports_anything(info) {
        return None;
    }

    let luminance = info.luminance;
    let state = WindowDisplayState {
        tone_map_headroom: info.tone_map_headroom().and_then(finite_positive),
        sdr_white_nits: luminance
            .and_then(|l| l.sdr_white_nits)
            .and_then(finite_positive),
    };
    let capability = MonitorDisplayCapability {
        max_nits: luminance.and_then(|l| l.max_nits).and_then(finite_positive),
        max_full_frame_nits: luminance
            .and_then(|l| l.max_full_frame_nits)
            .and_then(finite_positive),
        min_nits: luminance.and_then(|l| l.min_nits).and_then(finite_positive),
        gamut_hint: info.coarse.and_then(|c| c.gamut).map(map_gamut),
    };
    Some((state, capability))
}

/// Relative change below which a new reading counts as unchanged, so read
/// noise does not trigger change detection in the main world.
const EPSILON_REL: f32 = 0.01;

/// What one window surface has reported. A surface that reported nothing
/// still gets an entry, so it is not read again every frame.
#[derive(Default)]
struct SurfaceDisplayState {
    /// The last state stored by [`commit`].
    state: Option<WindowDisplayState>,
    /// The last capability read. [`write_back_display_state`] runs every frame
    /// and the poll does not, so the value is kept here.
    capability: Option<MonitorDisplayCapability>,
}

/// The last display state read for each window, keyed by render world entity.
#[derive(Resource, Default)]
pub struct DisplayStateStore(EntityHashMap<SurfaceDisplayState>);

fn rel_changed(old: Option<f32>, new: Option<f32>) -> bool {
    match (old, new) {
        (Some(a), Some(b)) => (a - b).abs() > EPSILON_REL * a.abs().max(f32::MIN_POSITIVE),
        (None, None) => false,
        _ => true,
    }
}

/// Stores `candidate` for `entity` only when a field changed by more than
/// [`EPSILON_REL`], so only real changes trigger change detection in the main
/// world. `capability` is always stored.
fn commit(
    store: &mut DisplayStateStore,
    entity: Entity,
    candidate: WindowDisplayState,
    capability: MonitorDisplayCapability,
) {
    let entry = store.0.entry(entity).or_default();

    let changed = entry.state.is_none_or(|committed| {
        rel_changed(committed.tone_map_headroom, candidate.tone_map_headroom)
            || rel_changed(committed.sdr_white_nits, candidate.sdr_white_nits)
    });

    if changed {
        entry.state = Some(candidate);
    }
    entry.capability = Some(capability);
}

/// Reads the display state behind each window surface when it may have
/// changed, and stores it for [`write_back_display_state`].
///
/// On Apple platforms this runs on the main thread. The Metal backend returns
/// nothing from any other thread.
pub fn poll_display_state(
    #[cfg(any(target_os = "macos", target_os = "ios"))] _marker: bevy_ecs::system::NonSendMarker,
    windows: Query<(Entity, &ExtractedWindow, &SurfaceData)>,
    render_adapter: Res<RenderAdapter>,
    mut store: ResMut<DisplayStateStore>,
) {
    store.0.retain(|e, _| windows.contains(*e));

    for (entity, extracted, surface_data) in windows.iter() {
        let first_time = !store.0.contains_key(&entity);
        let reconfigured = extracted.display_target_transfer_changed;
        let event_requery = extracted.request_display_requery;

        let resolved = surface_data.resolved_transfer;

        // On macOS the headroom changes with brightness, ambient light and
        // battery, so it is read every frame there.
        let continuous =
            cfg!(target_os = "macos") && resolved.is_hdr() && extracted.display_calibration_auto;

        if !(first_time || reconfigured || event_requery || continuous) {
            continue;
        }

        let info = surface_data.surface.display_hdr_info(&render_adapter);
        let Some((state, capability)) = read_display_state(&info) else {
            // Nothing was reported. Record the surface so it is not read again
            // every frame, and keep any earlier values.
            store.0.entry(entity).or_default();
            continue;
        };

        commit(&mut store, entity, state, capability);
    }
}

/// Writes each window's [`WindowSurfaceTransfers`], [`WindowDisplayState`],
/// and [`MonitorDisplayCapability`] back to the main world. The capability goes
/// on the window's monitor entity.
///
/// It runs during extraction, so the main world sees the previous frame's
/// result.
pub fn write_back_display_state(
    mut main_world: ResMut<MainWorld>,
    windows: Query<(Entity, MainEntity, &SurfaceData)>,
    store: Res<DisplayStateStore>,
) {
    for (entity, main_entity, surface_data) in windows.iter() {
        super::insert_on_change(
            &mut main_world,
            main_entity,
            WindowSurfaceTransfers {
                resolved: surface_data.resolved_transfer,
                supported: surface_data.supported_transfers,
            },
        );

        if let Some(state) = store.0.get(&entity).and_then(|s| s.state) {
            super::insert_on_change(&mut main_world, main_entity, state);
        }

        let Some(capability) = store.0.get(&entity).and_then(|s| s.capability) else {
            continue;
        };
        let Some(monitor_entity) = main_world
            .get::<OnMonitor>(main_entity)
            .map(|on_monitor| on_monitor.0)
        else {
            continue;
        };
        super::insert_on_change(&mut main_world, monitor_entity, capability);
    }
}

/// Resolves each window's [`EffectiveDisplayTarget`] from its
/// [`DisplayTarget`], its [`DisplayCalibrationPolicy`], and what the display
/// reports.
///
/// It runs in the main world so the first extracted frame already has the
/// resolved target.
pub fn resolve_calibration(
    mut windows: Query<
        (
            Option<&DisplayTarget>,
            Option<&DisplayCalibrationPolicy>,
            Option<&WindowDisplayState>,
            Option<&OnMonitor>,
            &mut EffectiveDisplayTarget,
        ),
        With<Window>,
    >,
    monitors: Query<&MonitorDisplayCapability>,
) {
    for (target, policy, live, on_monitor, mut effective) in &mut windows {
        let target = target.copied().unwrap_or_default();
        let policy = policy.copied().unwrap_or_default();
        let capability = on_monitor.and_then(|m| monitors.get(m.0).ok());
        let sensed = SensedInputs { capability, live };
        effective.set_if_neq(resolve_effective_target(target, policy, sensed));
    }
}

#[derive(Default, Clone, Copy)]
struct SensedInputs<'a> {
    capability: Option<&'a MonitorDisplayCapability>,
    live: Option<&'a WindowDisplayState>,
}

/// Resolves one [`EffectiveDisplayTarget`].
fn resolve_effective_target(
    target: DisplayTarget,
    policy: DisplayCalibrationPolicy,
    sensed: SensedInputs,
) -> EffectiveDisplayTarget {
    let mut out = target;
    let mut prov = DisplayProvenance::default();

    // Paper white resolves first because the peak may be derived from it.
    (out.paper_white_nits, prov.paper_white) = resolve_field(
        policy.auto_paper_white,
        target.paper_white_nits,
        sensed.live.and_then(|l| l.sdr_white_nits),
    );
    (out.peak_luminance_nits, prov.peak_luminance) = resolve_field(
        policy.auto_peak_luminance,
        target.peak_luminance_nits,
        os_peak(&sensed, out.paper_white_nits, out.transfer.is_hdr()),
    );
    (out.min_luminance_nits, prov.min_luminance) = resolve_field(
        policy.auto_min_luminance,
        target.min_luminance_nits,
        sensed.capability.and_then(|c| c.min_nits),
    );
    (out.gamut, prov.gamut) = resolve_field(
        policy.auto_gamut,
        target.gamut,
        sensed.capability.and_then(|c| c.gamut_hint),
    );

    EffectiveDisplayTarget {
        target: out,
        provenance: prov,
    }
}

/// The peak luminance the display reports, in nits. `None` for an SDR target,
/// which has no HDR peak.
///
/// Uses [`MonitorDisplayCapability::max_nits`] when reported, else
/// `paper_white_nits` times [`WindowDisplayState::tone_map_headroom`].
fn os_peak(sensed: &SensedInputs, paper_white_nits: f32, surface_is_hdr: bool) -> Option<f32> {
    if !surface_is_hdr {
        return None;
    }
    sensed
        .capability
        .and_then(|c| c.max_nits)
        .and_then(finite_positive)
        .or_else(|| {
            let headroom = sensed
                .live
                .and_then(|l| l.tone_map_headroom)
                .and_then(finite_positive)?;
            let paper_white = finite_positive(paper_white_nits)?;
            // The product can overflow to infinity.
            finite_positive(paper_white * headroom)
        })
}

fn resolve_field<T>(auto: bool, authored: T, sensed: Option<T>) -> (T, FieldProvenance) {
    match (auto, sensed) {
        (false, _) => (authored, FieldProvenance::User),
        (true, Some(v)) => (v, FieldProvenance::Os),
        (true, None) => (authored, FieldProvenance::Fallback),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_window::DisplayTransfer;
    use wgpu::{DisplayHeadroom, DisplayLuminance};

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

    #[test]
    fn macos_headroom_uses_current_not_potential() {
        let info = DisplayHdrInfo {
            headroom: Some(headroom(Some(4.0), Some(5.0), None)),
            ..Default::default()
        };
        let (state, capability) = read_display_state(&info).unwrap();
        // `tone_map_headroom()` uses `current`, not `potential`.
        assert_eq!(state.tone_map_headroom, Some(4.0));
        assert_eq!(state.sdr_white_nits, None);
        assert_eq!(capability.max_nits, None);
    }

    #[test]
    fn windows_headroom_is_max_nits_over_sdr_white() {
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
        // max_nits / sdr_white_nits.
        assert_eq!(state.tone_map_headroom, Some(5.0));
        assert_eq!(state.sdr_white_nits, Some(200.0));
        assert_eq!(capability.max_nits, Some(1000.0));
        assert_eq!(capability.max_full_frame_nits, Some(600.0));
        assert_eq!(capability.min_nits, Some(0.01));
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
        // NaN and zero count as not reported, but the read itself still counts.
        assert_eq!(capability.max_nits, None);
        assert_eq!(state.sdr_white_nits, None);
        assert_eq!(state.tone_map_headroom, None);
    }

    #[test]
    fn rel_changed_transitions() {
        assert!(rel_changed(Some(5.0), None));
        assert!(rel_changed(None, Some(5.0)));
        assert!(!rel_changed(None, None));
        assert!(!rel_changed(Some(5.0), Some(5.02)));
        assert!(rel_changed(Some(5.0), Some(5.5)));
    }

    fn state(tone_map_headroom: Option<f32>, sdr_white_nits: Option<f32>) -> WindowDisplayState {
        WindowDisplayState {
            tone_map_headroom,
            sdr_white_nits,
        }
    }

    fn committed(store: &DisplayStateStore, entity: Entity) -> Option<WindowDisplayState> {
        store.0.get(&entity).and_then(|s| s.state)
    }

    #[test]
    fn sub_epsilon_change_does_not_commit() {
        let mut store = DisplayStateStore::default();
        let entity = Entity::from_raw_u32(2).unwrap();

        let capability = MonitorDisplayCapability::default();
        commit(&mut store, entity, state(Some(5.0), None), capability);

        // A 0.4% change is below `EPSILON_REL`.
        commit(&mut store, entity, state(Some(5.02), None), capability);
        assert_eq!(committed(&store, entity), Some(state(Some(5.0), None)));
    }

    #[test]
    fn supra_epsilon_change_commits() {
        let mut store = DisplayStateStore::default();
        let entity = Entity::from_raw_u32(3).unwrap();

        let capability = MonitorDisplayCapability::default();
        commit(&mut store, entity, state(Some(5.0), None), capability);

        commit(&mut store, entity, state(Some(5.5), None), capability);
        assert_eq!(committed(&store, entity), Some(state(Some(5.5), None)));
    }

    #[test]
    fn sdr_white_change_commits_on_its_own() {
        // Each field is compared on its own.
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

    fn cap_with_peak(max_nits: f32) -> MonitorDisplayCapability {
        MonitorDisplayCapability {
            max_nits: Some(max_nits),
            ..Default::default()
        }
    }

    #[test]
    fn default_policy_returns_target_unchanged() {
        // The capability would override the peak if that field were enabled.
        let target = DisplayTarget::SDR_SRGB
            .with_peak_luminance(1000.0)
            .with_paper_white(200.0)
            .with_transfer(DisplayTransfer::Pq);
        let cap = cap_with_peak(4000.0);
        let e = resolve_effective_target(
            target,
            DisplayCalibrationPolicy::default(),
            SensedInputs {
                capability: Some(&cap),
                ..Default::default()
            },
        );
        assert_eq!(e.target, target);
        assert_eq!(e.provenance, DisplayProvenance::default());
    }

    #[test]
    fn auto_peak_takes_reported_max_nits() {
        // `os_peak` resolves only for an HDR transfer.
        let target = DisplayTarget::SDR_SRGB
            .with_peak_luminance(1000.0)
            .with_transfer(DisplayTransfer::Pq);
        let policy = DisplayCalibrationPolicy {
            auto_peak_luminance: true,
            ..Default::default()
        };
        let cap = cap_with_peak(4000.0);
        let e = resolve_effective_target(
            target,
            policy,
            SensedInputs {
                capability: Some(&cap),
                ..Default::default()
            },
        );
        assert_eq!(e.target.peak_luminance_nits, 4000.0);
        assert_eq!(e.provenance.peak_luminance, FieldProvenance::Os);
    }

    #[test]
    fn auto_peak_skipped_on_sdr_target() {
        // An SDR target keeps its own peak even when the display reports one.
        let target = DisplayTarget::SDR_SRGB.with_peak_luminance(1000.0); // Srgb transfer
        let policy = DisplayCalibrationPolicy {
            auto_peak_luminance: true,
            ..Default::default()
        };
        let cap = cap_with_peak(270.0);
        let e = resolve_effective_target(
            target,
            policy,
            SensedInputs {
                capability: Some(&cap),
                ..Default::default()
            },
        );
        assert_eq!(e.target.peak_luminance_nits, 1000.0);
        assert_eq!(e.provenance.peak_luminance, FieldProvenance::Fallback);
    }

    #[test]
    fn auto_with_nothing_sensed_falls_back_to_authored_tagged_fallback() {
        let target = DisplayTarget::SDR_SRGB.with_peak_luminance(1000.0);
        let policy = DisplayCalibrationPolicy {
            auto_peak_luminance: true,
            ..Default::default()
        };
        let e = resolve_effective_target(target, policy, SensedInputs::default());
        assert_eq!(e.target.peak_luminance_nits, 1000.0);
        assert_eq!(e.provenance.peak_luminance, FieldProvenance::Fallback);
    }

    #[test]
    fn transfer_is_never_resolved() {
        let target = DisplayTarget::SDR_SRGB.with_transfer(DisplayTransfer::Pq);
        let policy = DisplayCalibrationPolicy {
            auto_paper_white: true,
            auto_peak_luminance: true,
            auto_min_luminance: true,
            auto_gamut: true,
        };
        let cap = cap_with_peak(4000.0);
        let e = resolve_effective_target(
            target,
            policy,
            SensedInputs {
                capability: Some(&cap),
                ..Default::default()
            },
        );
        assert_eq!(e.target.transfer, DisplayTransfer::Pq);
    }

    #[test]
    fn auto_paper_white_takes_live_sdr_white() {
        let target = DisplayTarget::SDR_SRGB.with_paper_white(203.0);
        let policy = DisplayCalibrationPolicy {
            auto_paper_white: true,
            ..Default::default()
        };
        let live = WindowDisplayState {
            sdr_white_nits: Some(80.0),
            ..Default::default()
        };
        let e = resolve_effective_target(
            target,
            policy,
            SensedInputs {
                live: Some(&live),
                ..Default::default()
            },
        );
        assert_eq!(e.target.paper_white_nits, 80.0);
        assert_eq!(e.provenance.paper_white, FieldProvenance::Os);
    }

    #[test]
    fn auto_peak_is_paper_white_times_headroom_when_no_max_nits() {
        // No peak in nits, so the peak is paper white times headroom, 100 * 5.
        let target = DisplayTarget::SDR_SRGB
            .with_peak_luminance(1000.0)
            .with_transfer(DisplayTransfer::ScRgbLinear);
        let policy = DisplayCalibrationPolicy {
            auto_peak_luminance: true,
            ..Default::default()
        };
        let live = WindowDisplayState {
            tone_map_headroom: Some(5.0),
            ..Default::default()
        };
        let e = resolve_effective_target(
            target,
            policy,
            SensedInputs {
                live: Some(&live),
                ..Default::default()
            },
        );
        assert_eq!(e.target.peak_luminance_nits, 500.0);
        assert_eq!(e.provenance.peak_luminance, FieldProvenance::Os);
    }

    #[test]
    fn auto_peak_uses_fallback_paper_white_when_no_sdr_white() {
        // No SDR white reported, so paper white falls back to the `DisplayTarget`
        // value, and the peak is still derived from it.
        let target = DisplayTarget::SDR_SRGB.with_transfer(DisplayTransfer::ScRgbLinear);
        let policy = DisplayCalibrationPolicy {
            auto_paper_white: true,
            auto_peak_luminance: true,
            ..Default::default()
        };
        let live = WindowDisplayState {
            tone_map_headroom: Some(4.0),
            ..Default::default()
        };
        let e = resolve_effective_target(
            target,
            policy,
            SensedInputs {
                live: Some(&live),
                ..Default::default()
            },
        );
        assert_eq!(
            e.target.paper_white_nits,
            DisplayTarget::SDR_SRGB.paper_white_nits
        );
        assert_eq!(e.provenance.paper_white, FieldProvenance::Fallback);
        assert_eq!(
            e.target.peak_luminance_nits,
            DisplayTarget::SDR_SRGB.paper_white_nits * 4.0
        );
        assert_eq!(e.provenance.peak_luminance, FieldProvenance::Os);
    }

    #[test]
    fn peak_uses_the_resolved_paper_white_not_the_authored_one() {
        // The peak must use the resolved paper white, 120 * 3, not the
        // `DisplayTarget` value of 100. The fixture omits the peak in nits to
        // force the headroom branch.
        let target = DisplayTarget::SDR_SRGB.with_transfer(DisplayTransfer::ScRgbLinear);
        let policy = DisplayCalibrationPolicy {
            auto_paper_white: true,
            auto_peak_luminance: true,
            ..Default::default()
        };
        let live = WindowDisplayState {
            sdr_white_nits: Some(120.0),
            tone_map_headroom: Some(3.0),
        };
        let e = resolve_effective_target(
            target,
            policy,
            SensedInputs {
                live: Some(&live),
                ..Default::default()
            },
        );
        assert_eq!(e.target.paper_white_nits, 120.0);
        assert_eq!(e.target.peak_luminance_nits, 360.0);
        assert_eq!(e.provenance.peak_luminance, FieldProvenance::Os);
    }
}
