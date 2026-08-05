//! A guided HDR display-calibration screen: three steps that tune the primary
//! window's [`DisplayTarget`] and [`DisplayCalibrationPolicy`], persist the
//! result to the app's settings directory ([`SettingsPlugin`]), and reload it on
//! later runs.
//!
//! Every step reuses one pattern, the way most games calibrate: a probe square
//! on a reference card over a background field, with per-step luminances. Peak
//! raises the card until the max-signal probe disappears into it; paper white
//! tunes the card - and with it every white UI element - to comfortable white
//! paper; black level lowers the probe until it disappears into the black
//! background. Peak measures sharpest with the display in HGIG / Game mode
//! (dynamic tone mapping off); with tone mapping on, the probe fades instead of
//! vanishing, and the right stop is where it is faintest.
//!
//! The calibration camera uses [`Tonemapping::Linear`] (no tone curve) with an
//! fp16 intermediate, so the pattern reaches the display encoder unmodified.
//! [`HdrPlugin`](hdr::HdrPlugin) picks the HDR transfer; when the surface has
//! none there is nothing to calibrate, so a notice shows instead. The
//! calibration itself covers only the three luminance numbers and the per-field
//! auto/manual [`DisplayCalibrationPolicy`] - transfer and gamut stay with
//! `HdrPlugin`.
//!
//! Controls: Up/Down pick a section in the left-hand menu, Left/Right adjust it,
//! `M` flips between manual (HGIG) and trust-the-OS calibration, Enter saves,
//! Esc cancels.

use bevy::{
    camera::{Hdr, ScalingMode},
    core_pipeline::tonemapping::Tonemapping,
    prelude::*,
    settings::{ReflectSettingsGroup, SaveSettingsSync, SettingsGroup, SettingsPlugin},
    window::{
        DisplayCalibrationPolicy, DisplayTarget, EffectiveDisplayTarget, Monitor, OnMonitor,
        PrimaryWindow, WindowSurfaceTransfers,
    },
};

#[path = "../helpers/hdr.rs"]
mod hdr;

/// The app's states: calibrate, then show a done card until a key returns to
/// calibrating.
#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
enum AppState {
    #[default]
    Calibrating,
    Done,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(bevy::render::RenderPlugin {
            // The calibration pattern is grayscale, so the working color space
            // does not change it; Rec.2020 matches the PQ display target
            // `HdrPlugin` prefers, so the renderer does not warn about rendering
            // to a wide target from the narrow default.
            working_color_space: bevy::render::WorkingColorSpace::Rec2020,
            ..default()
        }))
        .init_state::<AppState>()
        // The saved calibration: registered before `SettingsPlugin` so the
        // plugin finds the group, and initialized so it exists (with defaults)
        // even before any settings file has been written.
        .register_type::<CalibrationSettings>()
        .init_resource::<CalibrationSettings>()
        .add_plugins(SettingsPlugin::new("org.bevy.examples.hdr_calibration"))
        // Request the best HDR output the surface can present.
        .add_plugins(hdr::HdrPlugin::default())
        .insert_resource(ClearColor(Color::BLACK))
        .init_resource::<CalibrationStep>()
        .init_resource::<MonitorChangeNotice>()
        .add_systems(Startup, seed_window)
        .add_systems(OnEnter(AppState::Calibrating), spawn_calibration)
        .add_systems(OnEnter(AppState::Done), spawn_done_screen)
        .add_systems(
            Update,
            (
                // The input systems run in a fixed order; the display systems
                // after them are independent of each other.
                (
                    change_step,
                    toggle_mode,
                    adjust_value,
                    confirm_or_cancel,
                    watch_monitor_changes,
                )
                    .chain(),
                (
                    update_visibility,
                    update_pattern,
                    update_banner,
                    update_value_bar,
                    update_menu,
                ),
            )
                .chain()
                .run_if(in_state(AppState::Calibrating)),
        )
        .add_systems(Update, leave_done.run_if(in_state(AppState::Done)))
        .run();
}

/// Which of the three wizard steps is shown.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
enum CalibrationStep {
    /// Raise peak until the max-signal probe disappears into the card.
    #[default]
    PeakLuminance,
    /// Adjust paper white until the card looks like comfortable paper.
    PaperWhite,
    /// Lower black level until the probe disappears into the black background.
    BlackLevel,
}

impl CalibrationStep {
    /// The next step, clamped at black level (no wrap).
    fn next(self) -> Self {
        match self {
            Self::PeakLuminance => Self::PaperWhite,
            Self::PaperWhite | Self::BlackLevel => Self::BlackLevel,
        }
    }

    /// The previous step, clamped at peak luminance (no wrap).
    fn prev(self) -> Self {
        match self {
            Self::BlackLevel => Self::PaperWhite,
            Self::PaperWhite | Self::PeakLuminance => Self::PeakLuminance,
        }
    }

    /// The menu label for this step.
    fn label(self) -> &'static str {
        match self {
            Self::PeakLuminance => "Peak luminance",
            Self::PaperWhite => "Paper white",
            Self::BlackLevel => "Black level",
        }
    }
}

/// The [`DisplayCalibrationPolicy`] for a calibration mode. Manual (HGIG) keeps
/// every authored value; trust-OS lets the engine resolve peak, black level, and
/// gamut from sensed display information. Paper white is a viewing preference
/// the display cannot sense, so it is always manual. The policy component on the
/// window is the single owner of this choice; see [`is_trust_os`].
fn policy_for(trust_os: bool) -> DisplayCalibrationPolicy {
    DisplayCalibrationPolicy {
        auto_paper_white: false,
        auto_peak_luminance: trust_os,
        auto_min_luminance: trust_os,
        auto_gamut: trust_os,
    }
}

/// Whether the window's policy is the trust-OS mode from [`policy_for`].
fn is_trust_os(policy: &DisplayCalibrationPolicy) -> bool {
    policy.auto_peak_luminance
}

/// Countdown for the "window moved monitors" banner notice; raised by
/// [`watch_monitor_changes`], read by [`update_banner`].
#[derive(Resource, Default)]
struct MonitorChangeNotice(Timer);

/// One quad of the calibration pattern; [`update_pattern`] assigns each a
/// luminance per step.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum PatternQuad {
    Background,
    Card,
    Probe,
}

/// Elements gated on the surface's HDR state: shown when `0` matches whether an
/// HDR transfer is active. The wizard and pattern carry `true`, the "no HDR
/// output" notice `false`.
#[derive(Component, Clone, Copy)]
struct VisibleWhenHdr(bool);

/// Tags the banner at the top of the screen.
#[derive(Component, Default, Clone, FromTemplate)]
struct BannerText;

/// Tags the value bar at the bottom of the screen.
#[derive(Component, Default, Clone, FromTemplate)]
struct ValueBarText;

/// Tags one left-hand menu entry by the step it names.
#[derive(Component, Clone, Copy)]
struct StepMenuLabel(CalibrationStep);

/// Tags the menu's manual/trust-OS mode line.
#[derive(Component, Default, Clone, FromTemplate)]
struct ModeText;

/// Height of the orthographic view volume in world units.
const VIEW_HEIGHT: f32 = 8.0;

/// Shifts the pattern right so the left-hand step menu never overlaps it.
const STAGE_SHIFT: f32 = 0.8;

/// The signal the probe carries on the peak step: the PQ coding ceiling, so it
/// stays at whatever the display chain clips to while the card rises.
const MAX_SIGNAL_NITS: f32 = DisplayTarget::MAX_PAPER_WHITE_NITS;

/// The probe's paper-white-relative gray on the paper-white step, like ink on
/// the white "sheet" behind it.
const PAPER_PROBE_LEVEL: f32 = 0.35;

/// The background's paper-white-relative gray on the paper-white step: a dim
/// surround so the card reads as a lit sheet of paper.
const PAPER_SURROUND_LEVEL: f32 = 0.15;

/// The background field, sized to fill most of the view (the view is
/// [`VIEW_HEIGHT`] world units tall).
const BACKGROUND_SIZE: Vec2 = Vec2::new(9.0, 6.0);

/// The reference card: roughly a tenth of the frame. HDR panels are brighter
/// over small areas than over the full frame, so the peak step compares the
/// probe against this similarly-sized card, never against the large background -
/// a fullscreen comparison patch could not match a small one at any setting.
const CARD_SIZE: Vec2 = Vec2::new(4.6, 1.8);

/// The probe square, small enough to sit inside the card.
const PROBE_SIZE: Vec2 = Vec2::splat(1.2);

/// Nits per second the peak adjustment moves at (before the Shift multiplier).
const PEAK_ADJUST_NITS_PER_SEC: f32 = 250.0;

/// Nits per second the paper-white adjustment moves at.
const PAPER_ADJUST_NITS_PER_SEC: f32 = 50.0;

/// Paper-white bounds: below ~80 nits white UI reads dim even in a dark room,
/// and above ~500 a persistent white level is fatiguing.
const PAPER_WHITE_MIN_NITS: f32 = 80.0;
const PAPER_WHITE_MAX_NITS: f32 = 500.0;

/// Exponential rate of the black-level adjustment per second held:
/// multiplicative, because equal luminance ratios look like equal steps near
/// black.
const BLACK_ADJUST_RATE: f32 = 1.5;

/// Black-level bounds, in nits. The floor is nonzero so the multiplicative
/// adjustment can always climb back up.
const BLACK_LEVEL_MIN_NITS: f32 = 0.001;
const BLACK_LEVEL_MAX_NITS: f32 = 5.0;

/// Adjustment speed multiplier while Shift is held.
const FAST_MULTIPLIER: f32 = 4.0;

/// How long the monitor-change notice stays in the banner.
const MONITOR_NOTICE_SECONDS: f32 = 8.0;

/// Menu text colors for the active and inactive sections.
const MENU_ACTIVE: Color = Color::WHITE;
const MENU_DIM: Color = Color::srgb(0.5, 0.5, 0.5);

/// Seeds the primary window's calibration intent at startup from
/// [`CalibrationSettings`]: the saved values when a settings file existed,
/// otherwise the defaults.
fn seed_window(
    mut commands: Commands,
    primary_window: Single<Entity, With<PrimaryWindow>>,
    settings: Res<CalibrationSettings>,
) {
    let mut target = DisplayTarget::SDR_SRGB;
    settings.apply(&mut target);
    commands
        .entity(*primary_window)
        .insert((target, policy_for(settings.trust_os)));
}

/// Spawns the calibration camera, the pattern, and the player UI on entering the
/// calibration state.
fn spawn_calibration(mut commands: Commands, mut step: ResMut<CalibrationStep>) {
    // Start a fresh wizard each entry.
    *step = CalibrationStep::PeakLuminance;
    let camera = commands
        .spawn_scene(calibration_camera())
        .insert(DespawnOnExit(AppState::Calibrating))
        .id();

    commands
        .spawn_scene(calibration_pattern())
        .insert(DespawnOnExit(AppState::Calibrating));

    commands
        .spawn_scene(player_ui())
        .insert((DespawnOnExit(AppState::Calibrating), UiTargetCamera(camera)));
    commands
        .spawn_scene(sdr_notice())
        .insert((DespawnOnExit(AppState::Calibrating), UiTargetCamera(camera)));
}

/// A 2D card shown after calibration.
fn spawn_done_screen(mut commands: Commands) {
    commands.spawn((
        Camera2d,
        // Encode with no tone curve; the card is plain SDR text.
        Tonemapping::Linear,
        DespawnOnExit(AppState::Done),
    ));
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

// --- BSN scene functions ---

/// The [`Tonemapping::Linear`] fp16 orthographic camera the pattern renders
/// through. Linear applies no tone curve, so pattern values reach the display
/// encoder unmodified.
fn calibration_camera() -> impl Scene {
    bsn! {
        Camera3d
        // fp16 intermediate, so pattern values above paper white survive to the
        // display-encoding pass.
        Hdr
        template_value(Tonemapping::Linear)
        template_value(Projection::from(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical { viewport_height: VIEW_HEIGHT },
            ..OrthographicProjection::default_3d()
        }))
        template_value(Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y))
    }
}

/// The one pattern every step reuses: a probe square on a reference card over a
/// background field. [`update_pattern`] drives the three luminances from the
/// current step, so nothing here carries a brightness of its own.
fn calibration_pattern() -> impl Scene {
    bsn! {
        template_value(Transform::from_xyz(STAGE_SHIFT, 0.0, 0.0))
        Visibility::Hidden
        template_value(VisibleWhenHdr(true))
        Children [
            pattern_quad(PatternQuad::Background, BACKGROUND_SIZE, 0.0),
            pattern_quad(PatternQuad::Card, CARD_SIZE, 0.05),
            pattern_quad(PatternQuad::Probe, PROBE_SIZE, 0.1),
        ]
    }
}

/// One unlit pattern quad. Each gets its own material handle because
/// [`update_pattern`] mutates `base_color` per quad; a shared handle would tie
/// them together.
fn pattern_quad(quad: PatternQuad, size: Vec2, z: f32) -> impl Scene {
    bsn! {
        template_value(quad)
        template(move |ctx| Ok(Mesh3d(
            ctx.resource_mut::<Assets<Mesh>>().add(Rectangle::from_size(size))
        )))
        template(|ctx| Ok(MeshMaterial3d(
            ctx.resource_mut::<Assets<StandardMaterial>>().add(StandardMaterial {
                base_color: Color::BLACK,
                unlit: true,
                ..default()
            })
        )))
        template_value(Transform::from_xyz(0.0, 0.0, z))
    }
}

/// The player UI: the left-hand step menu, a one-line banner, the value bar, and
/// the key legend. Hidden while the surface has no HDR transfer. All plain SDR
/// text.
fn player_ui() -> impl Scene {
    bsn! {
        template_value(VisibleWhenHdr(true))
        Node {
            position_type: PositionType::Absolute,
            top: px(0),
            left: px(0),
            width: percent(100),
            height: percent(100),
        }
        Visibility::Visible
        Children [
            step_menu(),
            (
                BannerText
                Text("")
                template(|_| Ok(TextFont::from_font_size(16.0)))
                TextLayout { justify: Justify::Center }
                Node {
                    position_type: PositionType::Absolute,
                    top: px(12),
                    left: px(0),
                    right: px(0),
                }
            ),
            (
                ValueBarText
                Text("")
                template(|_| Ok(TextFont::from_font_size(20.0)))
                TextLayout { justify: Justify::Center }
                Node {
                    position_type: PositionType::Absolute,
                    bottom: px(56),
                    left: px(0),
                    right: px(0),
                }
            ),
            (
                Text("Left/Right adjust   Up/Down section   M mode   Enter save   Esc cancel")
                template(|_| Ok(TextFont::from_font_size(14.0)))
                TextLayout { justify: Justify::Center }
                Node {
                    position_type: PositionType::Absolute,
                    bottom: px(16),
                    left: px(0),
                    right: px(0),
                }
            ),
        ]
    }
}

/// The fullscreen "no HDR output" notice shown instead of the wizard.
fn sdr_notice() -> impl Scene {
    bsn! {
        template_value(VisibleWhenHdr(false))
        Text("No HDR output.\nThe surface has no HDR transfer, so there is nothing to calibrate.\nEnable HDR in your system display settings, then restart.")
        template(|_| Ok(TextFont::from_font_size(22.0)))
        TextLayout { justify: Justify::Center }
        Visibility::Hidden
        Node {
            position_type: PositionType::Absolute,
            top: percent(40),
            left: px(0),
            right: px(0),
        }
    }
}

/// The left-hand menu: the three calibration sections, with the active one
/// highlighted by [`update_menu`], and the manual/trust-OS mode line.
fn step_menu() -> impl Scene {
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            top: px(12),
            left: px(12),
            flex_direction: FlexDirection::Column,
            row_gap: px(6),
        }
        Children [
            (
                Text("CALIBRATE HDR")
                template(|_| Ok(TextFont::from_font_size(13.0)))
                template_value(TextColor(MENU_DIM))
            ),
            menu_entry(CalibrationStep::PeakLuminance),
            menu_entry(CalibrationStep::PaperWhite),
            menu_entry(CalibrationStep::BlackLevel),
            (
                ModeText
                Text("")
                template(|_| Ok(TextFont::from_font_size(13.0)))
                template_value(TextColor(MENU_DIM))
            ),
        ]
    }
}

/// One menu entry; [`update_menu`] recolors it when the step changes.
fn menu_entry(step: CalibrationStep) -> impl Scene {
    bsn! {
        template_value(StepMenuLabel(step))
        template_value(Text::new(step.label()))
        template(|_| Ok(TextFont::from_font_size(16.0)))
        template_value(TextColor(MENU_DIM))
    }
}

// --- Per-frame systems ---

/// Whether any connected gamepad just pressed `button`.
fn any_gamepad_just_pressed(gamepads: &Query<&Gamepad>, button: GamepadButton) -> bool {
    gamepads.iter().any(|gamepad| gamepad.just_pressed(button))
}

/// Walks the menu: Down / d-pad down next section, Up / d-pad up previous.
/// Clamped at the ends.
fn change_step(
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    mut step: ResMut<CalibrationStep>,
) {
    if keys.just_pressed(KeyCode::ArrowDown)
        || any_gamepad_just_pressed(&gamepads, GamepadButton::DPadDown)
    {
        *step = step.next();
    }
    if keys.just_pressed(KeyCode::ArrowUp)
        || any_gamepad_just_pressed(&gamepads, GamepadButton::DPadUp)
    {
        *step = step.prev();
    }
}

/// Flips between manual and trust-OS calibration with `M` (or gamepad north) by
/// rewriting the window's [`DisplayCalibrationPolicy`].
fn toggle_mode(
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    mut policy: Single<&mut DisplayCalibrationPolicy, With<PrimaryWindow>>,
) {
    if keys.just_pressed(KeyCode::KeyM) || any_gamepad_just_pressed(&gamepads, GamepadButton::North)
    {
        **policy = policy_for(!is_trust_os(&policy));
    }
}

/// Adjusts the current step's value while Left/Right (or the d-pad) is held.
/// Under trust-OS the engine resolves peak and black level, so only paper white
/// responds. Peak is clamped at or above paper white.
fn adjust_value(
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    time: Res<Time>,
    step: Res<CalibrationStep>,
    window: Single<(&mut DisplayTarget, &DisplayCalibrationPolicy), With<PrimaryWindow>>,
) {
    let keyboard = keys.pressed(KeyCode::ArrowRight) as i32 as f32
        - keys.pressed(KeyCode::ArrowLeft) as i32 as f32;
    let dpad: f32 = gamepads.iter().map(|gamepad| gamepad.dpad().x).sum();
    let direction = keyboard + dpad;
    if direction == 0.0 {
        return;
    }
    let fast = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let held = direction.signum() * time.delta_secs() * if fast { FAST_MULTIPLIER } else { 1.0 };

    let (mut display_target, policy) = window.into_inner();
    let trust_os = is_trust_os(policy);
    match *step {
        CalibrationStep::PeakLuminance if !trust_os => {
            let paper = display_target.paper_white_nits;
            display_target.peak_luminance_nits = (display_target.peak_luminance_nits
                + held * PEAK_ADJUST_NITS_PER_SEC)
                .clamp(paper, MAX_SIGNAL_NITS);
        }
        CalibrationStep::PaperWhite => {
            display_target.paper_white_nits = (display_target.paper_white_nits
                + held * PAPER_ADJUST_NITS_PER_SEC)
                .clamp(PAPER_WHITE_MIN_NITS, PAPER_WHITE_MAX_NITS);
            // Keep peak at or above the new paper white.
            display_target.peak_luminance_nits = display_target
                .peak_luminance_nits
                .max(display_target.paper_white_nits);
        }
        CalibrationStep::BlackLevel if !trust_os => {
            display_target.min_luminance_nits = (display_target.min_luminance_nits
                * (held * BLACK_ADJUST_RATE).exp())
            .clamp(BLACK_LEVEL_MIN_NITS, BLACK_LEVEL_MAX_NITS);
        }
        _ => {}
    }
}

/// Confirms (`Enter` / gamepad south) -> persist the calibration and show the
/// done screen. Cancels (`Esc` / gamepad east) -> restore the last-saved (or
/// default) calibration and stay on the wizard.
fn confirm_or_cancel(
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    window: Single<(&mut DisplayTarget, &DisplayCalibrationPolicy), With<PrimaryWindow>>,
    mut settings: ResMut<CalibrationSettings>,
    mut commands: Commands,
    mut next: ResMut<NextState<AppState>>,
) {
    let confirm = keys.just_pressed(KeyCode::Enter)
        || any_gamepad_just_pressed(&gamepads, GamepadButton::South);
    let cancel = keys.just_pressed(KeyCode::Escape)
        || any_gamepad_just_pressed(&gamepads, GamepadButton::East);

    let (mut display_target, policy) = window.into_inner();

    if confirm {
        let target = *display_target;
        settings.set_if_neq(CalibrationSettings {
            trust_os: is_trust_os(policy),
            paper_white_nits: target.paper_white_nits,
            peak_luminance_nits: target.peak_luminance_nits,
            min_luminance_nits: target.min_luminance_nits,
        });
        commands.queue(SaveSettingsSync::IfChanged);
        let peak_source = if policy.auto_peak_luminance {
            "OS-sensed"
        } else {
            "manual"
        };
        info!(
            "Calibration confirmed: paper {:.0} / peak {:.0} ({peak_source}) / min {:.3} nits",
            target.paper_white_nits, target.peak_luminance_nits, target.min_luminance_nits,
        );
        next.set(AppState::Done);
    } else if cancel {
        // The settings resource only changes on confirm, so it always holds the
        // last-saved (or default) calibration.
        settings.apply(&mut display_target);
    }
}

/// Shows every [`VisibleWhenHdr`] element whose polarity matches whether an HDR
/// transfer is active, so the wizard swaps for the "no HDR output" notice
/// (including while the surface is still negotiating).
fn update_visibility(
    surface: Option<Single<&WindowSurfaceTransfers, With<PrimaryWindow>>>,
    mut gated: Query<(&VisibleWhenHdr, &mut Visibility)>,
) {
    let is_hdr = surface.is_some_and(|s| s.resolved.is_hdr());
    for (gate, mut visibility) in &mut gated {
        visibility.set_if_neq(if gate.0 == is_hdr {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        });
    }
}

/// The background, card, and probe luminances for one step, in
/// paper-white-relative units, from the resolved target. Reading the resolved
/// [`EffectiveDisplayTarget`], not the authored [`DisplayTarget`], lets a sensed
/// peak or black level show through under trust-OS.
fn pattern_levels(step: CalibrationStep, target: &DisplayTarget) -> (f32, f32, f32) {
    let paper_white = target.sanitized_paper_white_nits();
    match step {
        // Black screen; the card carries the candidate peak and the probe on it
        // holds max signal. Raising the candidate to the display chain's real
        // peak makes the two clip together and the probe vanish.
        CalibrationStep::PeakLuminance => (
            0.0,
            target.peak_luminance_nits / paper_white,
            MAX_SIGNAL_NITS / paper_white,
        ),
        // The card is the "sheet of paper" being judged, over a dim surround;
        // the probe is dim ink on it.
        CalibrationStep::PaperWhite => (PAPER_SURROUND_LEVEL, 1.0, PAPER_PROBE_LEVEL),
        // The probe carries the near-black candidate over true black.
        CalibrationStep::BlackLevel => (0.0, 0.0, target.min_luminance_nits / paper_white),
    }
}

/// Writes the current step's luminances into the three pattern materials,
/// skipping frames where nothing changed so the material assets are not dirtied.
fn update_pattern(
    step: Res<CalibrationStep>,
    effective: Single<&EffectiveDisplayTarget, With<PrimaryWindow>>,
    quads: Query<(Ref<PatternQuad>, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut last_levels: Local<Option<(f32, f32, f32)>>,
) {
    let levels = pattern_levels(*step, &effective.target);
    // Freshly spawned quads (state re-entry) start black and need painting even
    // when the levels match the previous wizard's.
    let any_added = quads.iter().any(|(quad, _)| quad.is_added());
    if !any_added && *last_levels == Some(levels) {
        return;
    }
    *last_levels = Some(levels);

    let (background, card, probe) = levels;
    for (quad, material) in &quads {
        let value = match *quad {
            PatternQuad::Background => background,
            PatternQuad::Card => card,
            PatternQuad::Probe => probe,
        };
        if let Some(mut material) = materials.get_mut(&material.0) {
            material.base_color = Color::linear_rgb(value, value, value);
        }
    }
}

/// Writes the banner: one short instruction for the current step, plus a
/// monitor-change notice while one is active.
fn update_banner(
    mut text: Single<&mut Text, With<BannerText>>,
    step: Res<CalibrationStep>,
    monitor_notice: Res<MonitorChangeNotice>,
) {
    let instruction = match *step {
        CalibrationStep::PeakLuminance => {
            "Raise peak until the center square just disappears into the card."
        }
        CalibrationStep::PaperWhite => {
            "Set paper white so white UI text is comfortable to read (~200 nits is typical)."
        }
        CalibrationStep::BlackLevel => {
            "Lower black level until the square disappears, then back up until barely visible."
        }
    };

    let mut banner = instruction.to_string();
    if !monitor_notice.0.is_finished() {
        banner.push_str("\nWindow moved to a different monitor - recalibration recommended.");
    }
    text.set_if_neq(Text(banner));
}

/// Writes the value bar: the single value the current step edits, reading the
/// authored [`DisplayTarget`] (the value the keys move).
fn update_value_bar(
    mut text: Single<&mut Text, With<ValueBarText>>,
    window: Single<(&DisplayTarget, &DisplayCalibrationPolicy), With<PrimaryWindow>>,
    step: Res<CalibrationStep>,
) {
    let (display_target, policy) = *window;
    let trust_os = is_trust_os(policy);
    let bar = match *step {
        CalibrationStep::PeakLuminance if trust_os => {
            "PEAK is Auto (from the display). Press M for manual.".to_string()
        }
        CalibrationStep::PeakLuminance => format!(
            "PEAK  {:.0} nits   (Left/Right, Shift = 4x)",
            display_target.peak_luminance_nits,
        ),
        CalibrationStep::PaperWhite => format!(
            "PAPER WHITE  {:.0} nits   (Left/Right, Shift = 4x)",
            display_target.paper_white_nits,
        ),
        CalibrationStep::BlackLevel if trust_os => {
            "BLACK LEVEL is Auto (from the display). Press M for manual.".to_string()
        }
        CalibrationStep::BlackLevel => format!(
            "BLACK LEVEL  {:.3} nits   (Left/Right, Shift = 4x)",
            display_target.min_luminance_nits,
        ),
    };
    text.set_if_neq(Text(bar));
}

/// Highlights the active section in the left-hand menu and keeps the mode line
/// current.
fn update_menu(
    step: Res<CalibrationStep>,
    policy: Single<&DisplayCalibrationPolicy, With<PrimaryWindow>>,
    mut labels: Query<(&StepMenuLabel, &mut TextColor)>,
    mut mode: Single<&mut Text, With<ModeText>>,
) {
    for (label, mut color) in &mut labels {
        color.set_if_neq(TextColor(if label.0 == *step {
            MENU_ACTIVE
        } else {
            MENU_DIM
        }));
    }

    let mode_text = if is_trust_os(&policy) {
        "Mode: Trust OS\n(peak/black from display)\nM: manual"
    } else {
        "Mode: Manual (HGIG)\nM: trust OS values"
    };
    if mode.0 != mode_text {
        mode.0 = mode_text.to_string();
    }
}

/// Raises a recalibration notice when the window moves to a different monitor, by
/// watching the [`OnMonitor`] relationship `bevy_winit` maintains. The first
/// insertion per window is the monitor becoming known at startup, so it is logged
/// rather than raised.
fn watch_monitor_changes(
    changed: Query<(Entity, Ref<OnMonitor>), Changed<OnMonitor>>,
    mut removed: RemovedComponents<OnMonitor>,
    windows: Query<(), With<Window>>,
    monitors: Query<&Monitor>,
    mut notice: ResMut<MonitorChangeNotice>,
    time: Res<Time>,
) {
    for (window, on_monitor) in &changed {
        let name = monitors
            .get(on_monitor.0)
            .ok()
            .and_then(|monitor| monitor.name.clone())
            .unwrap_or_else(|| "<unknown>".into());
        if on_monitor.is_added() {
            info!("Window {window} is on monitor {name}.");
        } else {
            info!("Window {window} moved to monitor {name}; recalibration recommended.");
            notice.0 = Timer::from_seconds(MONITOR_NOTICE_SECONDS, TimerMode::Once);
        }
    }
    // Removal means the monitor is no longer known. Skip windows whose relationship
    // went away because the window itself despawned.
    for window in removed.read() {
        if windows.contains(window) {
            info!("Window {window} moved to monitor <unknown>; recalibration recommended.");
            notice.0 = Timer::from_seconds(MONITOR_NOTICE_SECONDS, TimerMode::Once);
        }
    }
    if !notice.0.is_finished() {
        notice.0.tick(time.delta());
    }
}

// --- Persistence ---

/// The persisted calibration, saved by [`SettingsPlugin`] to `settings.toml` in
/// the app's settings directory. Only authored values are stored, never a sensed
/// or effective value.
#[derive(Resource, SettingsGroup, Reflect, Clone, PartialEq)]
#[reflect(Resource, SettingsGroup, Default)]
#[settings_group(group = "calibration")]
struct CalibrationSettings {
    trust_os: bool,
    paper_white_nits: f32,
    peak_luminance_nits: f32,
    min_luminance_nits: f32,
}

impl Default for CalibrationSettings {
    /// The calibration before any save exists, and what Cancel restores on a
    /// first run. The 1000-nit peak candidate exceeds most consumer panels, so
    /// the peak step is two-directional.
    fn default() -> Self {
        Self {
            trust_os: false,
            paper_white_nits: 200.0,
            peak_luminance_nits: 1000.0,
            min_luminance_nits: 0.1,
        }
    }
}

impl CalibrationSettings {
    /// Writes the three luminance numbers onto `target`. Transfer and gamut stay
    /// as they are.
    fn apply(&self, target: &mut DisplayTarget) {
        target.paper_white_nits = self.paper_white_nits;
        target.peak_luminance_nits = self.peak_luminance_nits;
        target.min_luminance_nits = self.min_luminance_nits;
    }
}
