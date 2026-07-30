//! Drives the reusable HDR calibration screen
//! ([`HdrCalibrationPlugin`](hdr_calibration::HdrCalibrationPlugin)) as a tiny
//! app, the way a game would: enter a calibration [`State`](AppState), let the
//! player tune their display, then act on [`CalibrationComplete`].
//!
//! The calibration screen, its measurement patches, and its player UI all live
//! in `examples/helpers/hdr_calibration.rs` - copy that plugin into your own app.
//! This file is only the harness around it: it picks an HDR transfer with the
//! `hdr_helper` [`HdrPlugin`](hdr::HdrPlugin), enters `AppState::Calibrating` at
//! startup, and on confirm prints the result and moves to `AppState::Done`.
//!
//! Press the backtick key (`` ` ``) for an engine-telemetry overlay: the
//! intent / effective / sensed comparison your real UI would never show, plus
//! `T` to cycle the requested transfer, `G` for a Gran Turismo 7 tone-mapping
//! preview, and saturated swatches to spot-check gamut. Pass `--rec2020` to
//! render the scene in the wide working color space.
//!
//! **An HDR display is required to calibrate anything real.** On the SDR
//! fallback everything above paper white clips, so the peak step looks flat; the
//! example still runs (this is the path CI exercises headless).

use bevy::{
    color::palettes::css,
    core_pipeline::tonemapping::Tonemapping,
    prelude::*,
    window::{
        DisplayCalibrationPolicy, DisplayGamut, DisplayProvenance, DisplayTarget, DisplayTransfer,
        EffectiveDisplayTarget, FieldProvenance, MonitorDisplayCapability, OnMonitor,
        PrimaryWindow, WindowDisplayState, WindowSurfaceTransfers,
    },
};

#[path = "../helpers/hdr.rs"]
mod hdr;
#[path = "../helpers/hdr_calibration.rs"]
mod hdr_calibration;

use hdr_calibration::{CalibrationComplete, HdrCalibrationPlugin};

/// The app's states. The calibration screen runs in `Calibrating`, which is the
/// default so the example (and the headless CI screenshot) shows the wizard at
/// startup. Confirming moves to `Done`.
#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
enum AppState {
    #[default]
    Calibrating,
    Done,
}

fn main() {
    // Working color space is a startup-time `RenderPlugin` setting. Grayscale
    // calibration patches are gamut-invariant, so `--rec2020` only affects the
    // saturated debug swatches and gizmos.
    let working_color_space = if std::env::args().any(|arg| arg == "--rec2020") {
        bevy::render::WorkingColorSpace::Rec2020
    } else {
        bevy::render::WorkingColorSpace::Rec709
    };

    App::new()
        .add_plugins(DefaultPlugins.set(bevy::render::RenderPlugin {
            working_color_space,
            ..default()
        }))
        .init_state::<AppState>()
        // Request the best HDR output the surface can present (PQ/HDR10 first,
        // then scRGB-linear, then encoded extended-range sRGB; SDR otherwise).
        .add_plugins(hdr::HdrPlugin {
            preference: vec![
                (DisplayTransfer::Pq, DisplayGamut::Rec2020),
                (DisplayTransfer::ScRgbLinear, DisplayGamut::Rec709),
                (DisplayTransfer::ExtendedSrgb, DisplayGamut::Rec709),
            ],
        })
        .add_plugins(HdrCalibrationPlugin {
            state: AppState::Calibrating,
        })
        .insert_resource(ClearColor(Color::BLACK))
        .init_resource::<DebugOverlay>()
        .add_observer(on_calibration_complete)
        .add_systems(OnEnter(AppState::Calibrating), spawn_debug_overlay)
        .add_systems(OnEnter(AppState::Done), spawn_done_screen)
        .add_systems(Update, leave_done.run_if(in_state(AppState::Done)))
        .add_systems(
            Update,
            (
                toggle_debug_overlay,
                toggle_transfer,
                toggle_gt7,
                update_data_panel,
                draw_gamut_gizmos,
            )
                .run_if(in_state(AppState::Calibrating)),
        )
        .run();
}

/// Saves the confirmed calibration (the plugin already persisted it) and advances
/// out of the calibration screen, the way a real app reacts to
/// [`CalibrationComplete`].
fn on_calibration_complete(
    complete: On<CalibrationComplete>,
    mut next: ResMut<NextState<AppState>>,
) {
    let target = complete.target;
    let peak_source = if complete.policy.auto_peak_luminance {
        "OS-sensed"
    } else {
        "manual"
    };
    info!(
        "Calibration confirmed: paper {:.0} / peak {:.0} ({peak_source}) / min {:.2} nits",
        target.paper_white_nits, target.peak_luminance_nits, target.min_luminance_nits,
    );
    next.set(AppState::Done);
}

/// A 2D card shown after calibration; press any key to recalibrate.
fn spawn_done_screen(mut commands: Commands) {
    commands.spawn((Camera2d, DespawnOnExit(AppState::Done)));
    commands.spawn((
        Text::new("Calibration saved.\nPress any key to calibrate again."),
        TextFont::from_font_size(28.0),
        TextLayout::justify(Justify::Center),
        Node {
            position_type: PositionType::Absolute,
            top: percent(40),
            left: px(0),
            right: px(0),
            ..default()
        },
        DespawnOnExit(AppState::Done),
    ));
}

/// Returns to the calibration screen from `Done` on any key.
fn leave_done(keys: Res<ButtonInput<KeyCode>>, mut next: ResMut<NextState<AppState>>) {
    if keys.get_just_pressed().next().is_some() {
        next.set(AppState::Calibrating);
    }
}

// --- Engine-telemetry overlay (debug only) ---------------------------------

/// Whether the engine-telemetry overlay (data panel + gamut demo) is shown. Off
/// by default; the backtick key toggles it. A real calibration UI shows none of
/// this.
#[derive(Resource, Default)]
struct DebugOverlay(bool);

/// Tags the right-hand data-panel text.
#[derive(Component)]
struct DataPanelText;

/// Tags the saturated-swatch group.
#[derive(Component)]
struct GamutSwatches;

/// Spawns the (hidden) telemetry overlay for the calibration screen.
fn spawn_debug_overlay(mut commands: Commands) {
    commands.spawn((
        DataPanelText,
        Text::default(),
        TextFont::from_font_size(14.0),
        Visibility::Hidden,
        Node {
            position_type: PositionType::Absolute,
            top: px(150),
            right: px(12),
            width: px(440),
            ..default()
        },
        DespawnOnExit(AppState::Calibrating),
    ));

    commands
        .spawn((
            GamutSwatches,
            Visibility::Hidden,
            Node {
                position_type: PositionType::Absolute,
                top: px(12),
                left: px(12),
                flex_direction: FlexDirection::Column,
                row_gap: px(4),
                ..default()
            },
            DespawnOnExit(AppState::Calibrating),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("saturated UI (gamut check)"),
                TextFont::from_font_size(12.0),
                TextColor(css::GRAY.into()),
            ));
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: px(4),
                    ..default()
                })
                .with_children(|row| {
                    for color in [
                        css::RED,
                        css::LIME,
                        css::BLUE,
                        css::AQUA,
                        css::MAGENTA,
                        css::YELLOW,
                        css::WHITE,
                    ] {
                        row.spawn((
                            Node {
                                width: px(28),
                                height: px(28),
                                ..default()
                            },
                            BackgroundColor(color.into()),
                        ));
                    }
                });
        });
}

/// Toggles the telemetry overlay with the backtick key.
fn toggle_debug_overlay(
    keys: Res<ButtonInput<KeyCode>>,
    mut overlay: ResMut<DebugOverlay>,
    mut panel: Single<&mut Visibility, (With<DataPanelText>, Without<GamutSwatches>)>,
    mut swatches: Single<&mut Visibility, (With<GamutSwatches>, Without<DataPanelText>)>,
) {
    if keys.just_pressed(KeyCode::Backquote) {
        overlay.0 = !overlay.0;
        let visibility = if overlay.0 {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        **panel = visibility;
        **swatches = visibility;
    }
}

/// The transfers `toggle_transfer` steps through: sRGB -> scRGB-linear ->
/// extended-sRGB -> PQ.
const TRANSFER_CYCLE: [DisplayTransfer; 4] = [
    DisplayTransfer::Srgb,
    DisplayTransfer::ScRgbLinear,
    DisplayTransfer::ExtendedSrgb,
    DisplayTransfer::Pq,
];

/// Cycles the requested transfer with `T` through only the transfers the surface
/// advertises, and hands transfer control to the user (so `HdrPlugin` stops
/// auto-selecting).
fn toggle_transfer(
    keys: Res<ButtonInput<KeyCode>>,
    window: Single<(&mut DisplayTarget, Option<&WindowSurfaceTransfers>), With<PrimaryWindow>>,
    mut hdr_preference: ResMut<hdr::HdrPreference>,
) {
    if !keys.just_pressed(KeyCode::KeyT) {
        return;
    }
    hdr_preference.manual_override = true;
    let (mut display_target, surface) = window.into_inner();
    let current = display_target.transfer;
    let start = TRANSFER_CYCLE
        .iter()
        .position(|&t| t == current)
        .unwrap_or(0);
    let next = (1..=TRANSFER_CYCLE.len())
        .map(|offset| TRANSFER_CYCLE[(start + offset) % TRANSFER_CYCLE.len()])
        .find(|&t| surface.is_none_or(|s| s.supported.contains(t)))
        .unwrap_or(current);
    display_target.transfer = next;
}

/// Toggles a Gran Turismo 7 tone-mapping preview on the calibration camera. The
/// patches are no longer exact while it is on; calibrate with it off.
fn toggle_gt7(
    keys: Res<ButtonInput<KeyCode>>,
    tonemapping: Single<&mut Tonemapping, With<Camera3d>>,
) {
    if keys.just_pressed(KeyCode::KeyG) {
        let mut tonemapping = tonemapping.into_inner();
        *tonemapping = if *tonemapping == Tonemapping::GranTurismo7 {
            Tonemapping::None
        } else {
            Tonemapping::GranTurismo7
        };
    }
}

/// Draws saturated demo gizmo lines along the top edge while the overlay is on.
fn draw_gamut_gizmos(overlay: Res<DebugOverlay>, mut gizmos: Gizmos) {
    if !overlay.0 {
        return;
    }
    let colors = [
        css::RED,
        css::LIME,
        css::BLUE,
        css::AQUA,
        css::MAGENTA,
        css::YELLOW,
    ];
    let count = colors.len() as f32;
    for (index, color) in colors.into_iter().enumerate() {
        let x = (index as f32 - (count - 1.0) / 2.0) * 0.7;
        gizmos.line(Vec3::new(x, 3.2, 1.0), Vec3::new(x, 3.8, 1.0), color);
    }
}

/// Writes the right-hand telemetry panel: requested vs negotiated transfer, the
/// intent / effective[prov] / sensed comparison table, the policy line, and the
/// live sensed footers. Every sensing field the example surfaces lives here.
fn update_data_panel(
    overlay: Res<DebugOverlay>,
    text: Single<&mut Text, With<DataPanelText>>,
    window: Single<
        (
            &DisplayTarget,
            &DisplayCalibrationPolicy,
            &EffectiveDisplayTarget,
            Option<&WindowDisplayState>,
            Option<&OnMonitor>,
            Option<&WindowSurfaceTransfers>,
        ),
        With<PrimaryWindow>,
    >,
    capabilities: Query<&MonitorDisplayCapability>,
) {
    if !overlay.0 {
        return;
    }
    let mut text = text.into_inner();
    let (display_target, policy, effective, live, on_monitor, surface) = window.into_inner();
    let resolved = &effective.target;
    let provenance: &DisplayProvenance = &effective.provenance;
    let capability = on_monitor.and_then(|on_monitor| capabilities.get(on_monitor.0).ok());

    let auto = |auto: bool| if auto { "auto" } else { "manual" };

    let sensed_peak = capability.and_then(|c| c.max_nits);
    let sensed_min = capability.and_then(|c| c.min_nits);
    let sensed_gamut = capability
        .and_then(|c| c.gamut_hint)
        .map_or_else(|| "      -".into(), |gamut| format!("{gamut:>7?}"));

    let transfers_available = match surface {
        Some(surface) => surface
            .supported
            .iter()
            .map(transfer_short_name)
            .collect::<Vec<_>>()
            .join(", "),
        None => "sensing...".into(),
    };

    let mut ui = String::new();
    ui.push_str("ENGINE TELEMETRY (your UI shows none of this)\n");
    ui.push_str(&format!("transfer (req): {:?}\n", display_target.transfer));
    match surface {
        Some(surface) => ui.push_str(&format!(
            "transfer (got): {:?} ({})\n",
            surface.resolved,
            if surface.resolved.is_hdr() {
                "HDR"
            } else {
                "SDR"
            },
        )),
        None => ui.push_str("transfer (got): negotiating...\n"),
    }
    ui.push_str(&format!("transfers available: {transfers_available}\n\n"));

    ui.push_str("field      intent    effective[prov]      sensed\n");
    ui.push_str(&format!(
        "paper   {:7.1}    {:7.1} [{}]{}{}\n",
        display_target.paper_white_nits,
        resolved.paper_white_nits,
        provenance_tag(provenance.paper_white),
        pad(provenance_tag(provenance.paper_white)),
        "      -",
    ));
    ui.push_str(&format!(
        "peak    {:7.1}    {:7.1} [{}]{}{}\n",
        display_target.peak_luminance_nits,
        resolved.peak_luminance_nits,
        provenance_tag(provenance.peak_luminance),
        pad(provenance_tag(provenance.peak_luminance)),
        fmt_nits(sensed_peak),
    ));
    ui.push_str(&format!(
        "min     {:7.3}    {:7.3} [{}]{}{}\n",
        display_target.min_luminance_nits,
        resolved.min_luminance_nits,
        provenance_tag(provenance.min_luminance),
        pad(provenance_tag(provenance.min_luminance)),
        fmt_nits(sensed_min),
    ));
    ui.push_str(&format!(
        "gamut   {:>7?}    {:>7?} [{}]{}{}\n\n",
        display_target.gamut,
        resolved.gamut,
        provenance_tag(provenance.gamut),
        pad(provenance_tag(provenance.gamut)),
        sensed_gamut,
    ));

    ui.push_str(&format!(
        "policy: paper {} | peak {} | min {} | gamut {}\n\n",
        auto(policy.auto_paper_white),
        auto(policy.auto_peak_luminance),
        auto(policy.auto_min_luminance),
        auto(policy.auto_gamut),
    ));

    match live {
        Some(live) => {
            ui.push_str(&format!(
                "Window state (live): headroom {} | src {}\n",
                live.tone_map_headroom
                    .map_or_else(|| "unknown".into(), |h| format!("{h:.2}x")),
                source_name(live, sensed_peak),
            ));
            ui.push_str(&format!("  SDR white {}\n", fmt_nits(live.sdr_white_nits)));
        }
        None => ui.push_str("Window state (live): not sensed yet\n"),
    }

    match capability {
        Some(capability) => {
            ui.push_str(&format!(
                "Monitor cap (static): peak {} | full-frame {} | black {}\n",
                fmt_nits(capability.max_nits),
                fmt_nits(capability.max_full_frame_nits),
                fmt_nits(capability.min_nits),
            ));
        }
        None => ui.push_str("Monitor cap (static): not sensed yet\n"),
    }

    text.0 = ui;
}

/// One-word tag for a field's resolved [`FieldProvenance`].
fn provenance_tag(provenance: FieldProvenance) -> &'static str {
    match provenance {
        FieldProvenance::User => "user",
        FieldProvenance::Os => "OS-sensed",
        FieldProvenance::Default => "default",
    }
}

/// Which reporting model the sensed display values came from, inferred from
/// which of them the platform filled in.
///
/// Absolute nits — the SDR-white anchor on the live state, or the monitor's
/// peak — mean the OS reported real luminance (the Windows path). A headroom
/// multiplier with no absolute nits anywhere is the Apple EDR path, which
/// reports a unitless ratio and nothing else.
fn source_name(live: &WindowDisplayState, sensed_peak: Option<f32>) -> &'static str {
    if live.sdr_white_nits.is_some() || sensed_peak.is_some() {
        "OS / windowing system"
    } else if live.tone_map_headroom.is_some() {
        "derived from EDR headroom"
    } else {
        "unknown (nothing sensed)"
    }
}

/// Renders one optional sensed nits value, or "-" when the platform did not
/// report it.
fn fmt_nits(value: Option<f32>) -> String {
    value.map_or_else(|| "      -".into(), |nits| format!("{nits:7.1}"))
}

/// Pads a provenance tag to a fixed width so the `sensed` column lines up.
fn pad(tag: &str) -> String {
    " ".repeat("OS-sensed".len().saturating_sub(tag.len()) + 3)
}

/// A short, human-readable name for a [`DisplayTransfer`].
fn transfer_short_name(transfer: DisplayTransfer) -> &'static str {
    match transfer {
        DisplayTransfer::Srgb => "sRGB",
        DisplayTransfer::ScRgbLinear => "scRGB",
        DisplayTransfer::ExtendedSrgb => "ext-sRGB",
        DisplayTransfer::Pq => "PQ",
    }
}
