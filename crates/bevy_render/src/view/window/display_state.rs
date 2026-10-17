//! Reads what the display behind each window surface reports, writes it back
//! to the main world, and resolves each window's [`ResolvedDisplayTarget`].
//!
//! `bevy_window` does not depend on wgpu, so everything that reads a
//! [`DisplayHdrInfo`] is here.

use bevy_color::RgbPrimaries;
use bevy_ecs::entity::EntityHashMap;
use bevy_ecs::prelude::*;
use bevy_window::{
    DisplayTarget, MonitorDisplayCapability, OnMonitor, ResolvedDisplayTarget, SurfaceColorSpace,
    Window, WindowDisplayState, WindowSurfaceColorSpaces,
};
use wgpu::{DisplayGamut, DisplayHdrInfo};

use crate::renderer::RenderAdapter;
use crate::sync_world::MainEntity;
use crate::MainWorld;

use super::{ExtractedWindow, SurfaceData};

/// Maps a coarse wgpu gamut bucket to its primaries.
/// [`Srgb`](DisplayGamut::Srgb) and any variant wgpu adds later map to
/// [`RgbPrimaries::BT709`], the narrowest set, so the result is never wider
/// than what the display covers.
fn map_gamut(gamut: DisplayGamut) -> RgbPrimaries {
    match gamut {
        DisplayGamut::DisplayP3 => RgbPrimaries::DISPLAY_P3,
        DisplayGamut::Rec2020 => RgbPrimaries::BT2020,
        _ => RgbPrimaries::BT709,
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
    info.luminance.is_some_and(|l| {
        l.max_nits.is_some()
            || l.max_full_frame_nits.is_some()
            || l.min_nits.is_some()
            || l.sdr_white_nits.is_some()
    }) || info.headroom.is_some()
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

/// Returns `true` if a headroom the display reports means the OS HDR setting
/// is off.
///
/// Windows lists the PQ color space on the surface even when the OS HDR
/// toggle is off, so the surface capabilities over-report there. The live
/// tone-map headroom disambiguates: it is `1.0` for a display in SDR mode.
/// Other platforms report the color spaces accurately, so the gate is Windows
/// only.
pub(super) fn display_reports_sdr(state: Option<&WindowDisplayState>) -> bool {
    cfg!(target_os = "windows") && headroom_reports_sdr(state.and_then(|s| s.tone_map_headroom))
}

/// [`display_reports_sdr`] without the platform check.
fn headroom_reports_sdr(tone_map_headroom: Option<f32>) -> bool {
    tone_map_headroom.is_some_and(|headroom| headroom <= 1.0)
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

impl DisplayStateStore {
    /// The last [`WindowDisplayState`] stored for `entity`, if the display
    /// behind its surface has reported anything.
    pub(super) fn state(&self, entity: Entity) -> Option<&WindowDisplayState> {
        self.0.get(&entity).and_then(|s| s.state.as_ref())
    }

    /// Reads the display state behind `surface` and stores it for `entity`,
    /// so [`write_back_display_state`] and
    /// [`display_reports_sdr`] see it. A surface that reports nothing still
    /// gets an entry, so it is not read again every frame, and keeps any
    /// earlier values.
    ///
    /// On Apple platforms this must run on the main thread. The Metal backend
    /// returns nothing from any other thread.
    pub(super) fn read(
        &mut self,
        entity: Entity,
        surface: &wgpu::Surface,
        adapter: &RenderAdapter,
    ) {
        let info = surface.display_hdr_info(adapter);
        match read_display_state(&info) {
            Some((state, capability)) => commit(self, entity, state, capability),
            None => {
                self.0.entry(entity).or_default();
            }
        }
    }
}

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

/// Returns `true` if any luminance field of `target` is `None`, so a new
/// reading can change the window's [`ResolvedDisplayTarget`].
fn has_uncalibrated_luminance(target: &DisplayTarget) -> bool {
    target.paper_white_nits.is_none()
        || target.peak_luminance_nits.is_none()
        || target.min_luminance_nits.is_none()
}

/// Returns `true` if [`poll_display_state`] reads the display behind a window
/// surface this frame.
///
/// It reads a surface the store has no entry for, a surface whose color
/// space request changed, and a surface whose window has
/// [`ExtractedWindow::request_display_requery`] set. On a platform where the
/// headroom changes from frame to frame (`continuous_platform`), it also
/// reads an HDR surface whose [`DisplayTarget`] leaves any luminance field
/// uncalibrated.
fn should_poll(
    continuous_platform: bool,
    first_time: bool,
    extracted: &ExtractedWindow,
    resolved_color_space: SurfaceColorSpace,
) -> bool {
    let continuous = continuous_platform
        && resolved_color_space.is_hdr()
        && has_uncalibrated_luminance(&extracted.display_target);
    first_time
        || extracted.color_space_request_changed
        || extracted.request_display_requery
        || continuous
}

/// Reads the display state behind each window surface when it may have
/// changed, and stores it for [`write_back_display_state`].
///
/// It reads on surface creation, on a surface reconfiguration or
/// renegotiation, and when [`ExtractedWindow::request_display_requery`] is
/// set. On macOS the headroom changes with brightness, ambient light and
/// battery, so an HDR surface whose [`DisplayTarget`] leaves any luminance
/// field uncalibrated is read every frame there. See [`should_poll`].
///
/// [`create_surfaces`](super::create_surfaces) reads a new surface before
/// it is configured, so the entry for a new window is usually there already.
///
/// On Apple platforms this runs on the main thread. The Metal backend returns
/// nothing from any other thread.
pub fn poll_display_state(
    #[cfg(any(target_os = "macos", target_os = "ios"))] _marker: bevy_ecs::system::NonSendMarker,
    windows: Query<(Entity, &ExtractedWindow, &SurfaceData)>,
    extracted_windows: Query<(), With<ExtractedWindow>>,
    render_adapter: Res<RenderAdapter>,
    mut store: ResMut<DisplayStateStore>,
) {
    // `SurfaceData` is inserted through commands, so a window whose surface
    // was created this frame may not match `windows` yet. Its entry is kept.
    store.0.retain(|e, _| extracted_windows.contains(*e));

    for (entity, extracted, surface_data) in windows.iter() {
        let first_time = !store.0.contains_key(&entity);
        if !should_poll(
            cfg!(target_os = "macos"),
            first_time,
            extracted,
            surface_data.resolved_color_space,
        ) {
            continue;
        }
        store.read(entity, &surface_data.surface, &render_adapter);
    }
}

/// Writes each window's [`WindowSurfaceColorSpaces`], [`WindowDisplayState`],
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
            WindowSurfaceColorSpaces {
                resolved: surface_data.resolved_color_space,
                supported: surface_data.supported_color_spaces,
            },
        );

        if let Some(state) = store.state(entity) {
            super::insert_on_change(&mut main_world, main_entity, *state);
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

/// Resolves each window's [`ResolvedDisplayTarget`] from its
/// [`DisplayTarget`], the color space in its [`WindowSurfaceColorSpaces`],
/// and what the display reports in [`WindowDisplayState`] and the monitor's
/// [`MonitorDisplayCapability`].
///
/// Before the surface is configured the color space is
/// [`SurfaceColorSpace::Srgb`]. The component is inserted on the first run
/// and written only when the result changes, so change detection fires only
/// on real changes.
///
/// It runs in the main world so the first extracted frame already has the
/// resolved target.
pub fn resolve_display_targets(
    mut commands: Commands,
    mut windows: Query<
        (
            Entity,
            Option<&DisplayTarget>,
            Option<&WindowSurfaceColorSpaces>,
            Option<&WindowDisplayState>,
            Option<&OnMonitor>,
            Option<&mut ResolvedDisplayTarget>,
        ),
        With<Window>,
    >,
    monitors: Query<&MonitorDisplayCapability>,
) {
    for (entity, target, surface, state, on_monitor, resolved) in &mut windows {
        // A required component can still be removed. Fall back to the default
        // rather than skip the window.
        let target = target.copied().unwrap_or_default();
        let color_space = surface.map_or(SurfaceColorSpace::Srgb, |surface| surface.resolved);
        let capability = on_monitor.and_then(|m| monitors.get(m.0).ok());
        let next = target.resolve_with_display(color_space, capability, state);
        match resolved {
            Some(mut resolved) => {
                resolved.set_if_neq(next);
            }
            None => {
                commands.entity(entity).insert(next);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::{DisplayCoarseRange, DisplayHeadroom, DisplayLuminance};

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
    fn gamut_hint_maps_to_primaries() {
        for (gamut, primaries) in [
            (DisplayGamut::Srgb, RgbPrimaries::BT709),
            (DisplayGamut::DisplayP3, RgbPrimaries::DISPLAY_P3),
            (DisplayGamut::Rec2020, RgbPrimaries::BT2020),
        ] {
            let info = DisplayHdrInfo {
                coarse: Some(DisplayCoarseRange {
                    high_dynamic_range: None,
                    gamut: Some(gamut),
                }),
                ..Default::default()
            };
            let (_, capability) = read_display_state(&info).unwrap();
            assert_eq!(capability.gamut_hint, Some(primaries), "{gamut:?}");
        }
    }

    #[test]
    fn none_stays_none_never_sdr() {
        assert!(read_display_state(&DisplayHdrInfo::default()).is_none());
    }

    #[test]
    fn a_minimum_alone_counts_as_reported() {
        let info = DisplayHdrInfo {
            luminance: Some(luminance(None, None, Some(0.02), None)),
            ..Default::default()
        };
        let (state, capability) = read_display_state(&info).unwrap();
        assert_eq!(capability.min_nits, Some(0.02));
        assert_eq!(capability.max_nits, None);
        assert_eq!(state, WindowDisplayState::default());

        let info = DisplayHdrInfo {
            luminance: Some(luminance(None, Some(600.0), None, None)),
            ..Default::default()
        };
        let (_, capability) = read_display_state(&info).unwrap();
        assert_eq!(capability.max_full_frame_nits, Some(600.0));
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
    fn a_headroom_of_one_or_less_reports_sdr() {
        assert!(headroom_reports_sdr(Some(1.0)));
        assert!(headroom_reports_sdr(Some(0.5)));
        assert!(!headroom_reports_sdr(Some(1.01)));
        assert!(!headroom_reports_sdr(Some(4.0)));
        // Not reported never means SDR.
        assert!(!headroom_reports_sdr(None));
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

    fn extracted_window(
        display_target: DisplayTarget,
        color_space_request_changed: bool,
        request_display_requery: bool,
    ) -> ExtractedWindow {
        ExtractedWindow {
            physical_width: 1,
            physical_height: 1,
            present_mode: bevy_window::PresentMode::AutoVsync,
            desired_maximum_frame_latency: None,
            swap_chain_texture_view: None,
            swap_chain_texture: None,
            swap_chain_texture_format: None,
            swap_chain_texture_view_format: None,
            size_changed: false,
            present_mode_changed: false,
            alpha_mode: bevy_window::CompositeAlphaMode::Auto,
            display_target,
            resolved_display_target: display_target.resolve(SurfaceColorSpace::Srgb),
            display_reports_sdr: false,
            color_space_request_changed,
            resolved_color_space: None,
            request_display_requery,
            needs_initial_present: false,
        }
    }

    #[test]
    fn poll_triggers() {
        let hdr = DisplayTarget {
            hdr: true,
            ..Default::default()
        };
        let idle = extracted_window(hdr, false, false);

        // The first read, a reconfiguration, and a requery each trigger a
        // read on every platform.
        for continuous_platform in [false, true] {
            assert!(should_poll(
                continuous_platform,
                true,
                &idle,
                SurfaceColorSpace::Srgb
            ));
            assert!(should_poll(
                continuous_platform,
                false,
                &extracted_window(hdr, true, false),
                SurfaceColorSpace::Srgb
            ));
            assert!(should_poll(
                continuous_platform,
                false,
                &extracted_window(hdr, false, true),
                SurfaceColorSpace::Srgb
            ));
        }

        // Nothing happened: no read, except on a platform whose headroom
        // changes from frame to frame, for an HDR surface with an
        // uncalibrated luminance.
        assert!(!should_poll(false, false, &idle, SurfaceColorSpace::Pq));
        assert!(should_poll(true, false, &idle, SurfaceColorSpace::Pq));
        assert!(!should_poll(true, false, &idle, SurfaceColorSpace::Srgb));

        let calibrated = extracted_window(
            DisplayTarget {
                hdr: true,
                paper_white_nits: Some(200.0),
                peak_luminance_nits: Some(1000.0),
                min_luminance_nits: Some(0.0),
                ..Default::default()
            },
            false,
            false,
        );
        assert!(!should_poll(
            true,
            false,
            &calibrated,
            SurfaceColorSpace::Pq
        ));
    }

    #[test]
    fn resolve_display_targets_inserts_once_and_writes_only_on_change() {
        use bevy_ecs::change_detection::DetectChanges;
        use bevy_ecs::system::RunSystemOnce;

        let mut world = World::new();
        let monitor = world
            .spawn(MonitorDisplayCapability {
                max_nits: Some(1000.0),
                min_nits: Some(0.01),
                ..Default::default()
            })
            .id();
        let requested = DisplayTarget {
            hdr: true,
            paper_white_nits: Some(300.0),
            ..Default::default()
        };
        let on_monitor = world
            .spawn((Window::default(), requested, OnMonitor(monitor)))
            .id();
        let alone = world.spawn((Window::default(), requested)).id();

        assert!(world.get::<ResolvedDisplayTarget>(on_monitor).is_none());
        world.run_system_once(resolve_display_targets).unwrap();

        // Before the surface is configured the color space is SDR sRGB. The
        // capability is found through `OnMonitor`, and only the minimum
        // applies on SDR.
        assert_eq!(
            *world.get::<ResolvedDisplayTarget>(on_monitor).unwrap(),
            requested.resolve_with_display(
                SurfaceColorSpace::Srgb,
                Some(&MonitorDisplayCapability {
                    max_nits: Some(1000.0),
                    min_nits: Some(0.01),
                    ..Default::default()
                }),
                None
            )
        );
        assert_eq!(
            world
                .get::<ResolvedDisplayTarget>(on_monitor)
                .unwrap()
                .min_luminance_nits,
            0.01
        );
        // A window without a monitor resolves with no capability.
        assert_eq!(
            *world.get::<ResolvedDisplayTarget>(alone).unwrap(),
            requested.resolve(SurfaceColorSpace::Srgb)
        );

        let first_change = world
            .entity(on_monitor)
            .get_ref::<ResolvedDisplayTarget>()
            .unwrap()
            .last_changed();
        world.increment_change_tick();

        // An identical result leaves the change tick alone.
        world.run_system_once(resolve_display_targets).unwrap();
        assert_eq!(
            world
                .entity(on_monitor)
                .get_ref::<ResolvedDisplayTarget>()
                .unwrap()
                .last_changed(),
            first_change
        );

        // The surface negotiates PQ: the result changes and is written.
        world
            .entity_mut(on_monitor)
            .insert(WindowSurfaceColorSpaces {
                resolved: SurfaceColorSpace::Pq,
                supported: bevy_window::SurfaceColorSpaces::EMPTY.with(SurfaceColorSpace::Pq),
            });
        world.increment_change_tick();
        world.run_system_once(resolve_display_targets).unwrap();
        let resolved = world
            .entity(on_monitor)
            .get_ref::<ResolvedDisplayTarget>()
            .unwrap();
        assert_ne!(resolved.last_changed(), first_change);
        assert_eq!(resolved.color_space, SurfaceColorSpace::Pq);
        assert_eq!(resolved.peak_luminance_nits, 1000.0);
    }

    #[test]
    fn any_none_luminance_field_is_uncalibrated() {
        assert!(has_uncalibrated_luminance(&DisplayTarget::default()));
        let calibrated = DisplayTarget {
            paper_white_nits: Some(200.0),
            peak_luminance_nits: Some(1000.0),
            min_luminance_nits: Some(0.0),
            ..Default::default()
        };
        assert!(!has_uncalibrated_luminance(&calibrated));
        assert!(has_uncalibrated_luminance(&DisplayTarget {
            min_luminance_nits: None,
            ..calibrated
        }));
    }
}
