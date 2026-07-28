//! A reusable HDR display-calibration screen, packaged as a plugin an app drops
//! into one of its own [`States`].
//!
//! Unlike most examples, which demonstrate an application, this is a small
//! reusable library: [`HdrCalibrationPlugin<S>`] spawns a guided three-step
//! calibration wizard on [`OnEnter(state)`](OnEnter), despawns it on
//! [`OnExit`](DespawnOnExit), and writes the player's choices into the primary
//! window's [`DisplayTarget`] / [`DisplayCalibrationPolicy`]. On confirm it
//! persists the result next to the executable and emits [`CalibrationComplete`]
//! so the app can react (save, advance a menu, ...).
//!
//! The wizard owns a dedicated [`Tonemapping::None`] fp16 camera on its own
//! [`RenderLayers`], so the measurement patches reach the display encoder
//! unmodified and never fight a gameplay camera. Its UI is pinned to that camera
//! with [`UiTargetCamera`]. The patches are the only luminance-bearing elements;
//! the Bevy wordmark is decoration that tracks paper white.
//!
//! Transfer and gamut are deliberately *not* this plugin's concern: pair it with
//! something that selects an HDR transfer (such as the `hdr_helper` example's
//! `HdrPlugin`), which keeps owning [`DisplayTarget::transfer`] and `.gamut`.
//! This screen only calibrates the three luminance numbers and the per-field
//! auto/manual [`DisplayCalibrationPolicy`].
//!
//! ```ignore
//! #[derive(States, Default, Clone, PartialEq, Eq, Hash, Debug)]
//! enum AppState { #[default] Calibrating, Playing }
//!
//! app.init_state::<AppState>()
//!     .add_plugins(HdrCalibrationPlugin { state: AppState::Calibrating })
//!     .add_observer(|done: On<CalibrationComplete>, mut next: ResMut<NextState<AppState>>| {
//!         next.set(AppState::Playing);
//!     });
//! ```

use std::{fs, path::PathBuf};

use bevy::{
    camera::{visibility::RenderLayers, Hdr, ScalingMode},
    core_pipeline::tonemapping::Tonemapping,
    platform::collections::HashSet,
    prelude::*,
    window::{
        DisplayCalibrationPolicy, DisplayTarget, EffectiveDisplayTarget, Monitor, PrimaryWindow,
        WindowMonitorChanged, WindowSurfaceTransfers,
    },
};
use serde::{Deserialize, Serialize};

/// Adds a guided HDR calibration screen that runs while the app is in `state`.
///
/// Generic over the app's own [`States`] type, so the screen lives inside
/// whatever state the app already uses (a menu state, a dedicated
/// `Calibrating` state, ...).
pub struct HdrCalibrationPlugin<S: States> {
    /// The state during which the calibration screen is shown.
    pub state: S,
}

impl<S: States> Plugin for HdrCalibrationPlugin<S> {
    fn build(&self, app: &mut App) {
        let state = self.state.clone();
        app.init_resource::<CalibrationStep>()
            .init_resource::<CalibrationStrategy>()
            .init_resource::<BlackLevelChoice>()
            .init_resource::<MonitorChangeNotice>()
            .add_systems(Startup, seed_window)
            .add_systems(OnEnter(state.clone()), spawn_calibration::<S>)
            .add_systems(
                Update,
                (
                    change_step,
                    toggle_strategy,
                    adjust_value,
                    confirm_or_cancel,
                    watch_monitor_changes,
                    update_pattern_visibility,
                    update_patch_levels,
                    update_banner,
                    update_value_bar,
                    update_black_labels,
                )
                    .chain()
                    .run_if(in_state(state)),
            );
    }
}

/// Emitted (as a global observer [`Event`]) when the player confirms
/// calibration. Carries the authored target and policy the app should keep; the
/// plugin has already persisted them.
#[derive(Event, Clone)]
pub struct CalibrationComplete {
    /// The user-authored calibration numbers (never sensed values).
    pub target: DisplayTarget,
    /// Which fields the app should let the engine auto-resolve.
    pub policy: DisplayCalibrationPolicy,
}

/// Which of the three wizard steps is shown.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
enum CalibrationStep {
    /// Adjust peak luminance until the center square merges with the surround.
    #[default]
    PeakLuminance,
    /// Adjust paper white until the reference card looks like comfortable paper.
    PaperWhite,
    /// Pick the dimmest near-black step still distinct from black.
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

    /// The step's 1-based number, for the banner.
    fn index(self) -> u8 {
        match self {
            Self::PeakLuminance => 1,
            Self::PaperWhite => 2,
            Self::BlackLevel => 3,
        }
    }
}

/// How the player wants their calibration resolved: tune it by hand, or trust
/// the operating system's sensed values. Chosen up front (`M`) so the auto/manual
/// distinction is explicit rather than a per-field surprise.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum CalibrationStrategy {
    /// The player tunes peak / paper white / black level; the engine keeps every
    /// authored value verbatim (HGIG).
    #[default]
    ManualHgig,
    /// Peak / black level / gamut are filled from sensed display information; the
    /// player only sets paper white (a viewing preference the display can't sense).
    TrustOs,
}

impl CalibrationStrategy {
    /// The [`DisplayCalibrationPolicy`] this strategy implies.
    fn policy(self) -> DisplayCalibrationPolicy {
        let auto = matches!(self, Self::TrustOs);
        DisplayCalibrationPolicy {
            // Paper white is a viewing preference, always manual.
            auto_paper_white: false,
            auto_peak_luminance: auto,
            auto_min_luminance: auto,
            auto_gamut: auto,
        }
    }
}

/// The currently selected near-black step on the black-level screen (an index
/// into [`BLACK_STEP_NITS`]); its nit value is written to `min_luminance_nits`.
#[derive(Resource, Default)]
struct BlackLevelChoice(usize);

/// A short "window moved monitor" recalibration notice with a countdown, owned
/// solely by [`watch_monitor_changes`].
#[derive(Resource, Default)]
struct MonitorChangeNotice {
    seconds_remaining: f32,
}

/// How bright a calibration patch should be; [`update_patch_levels`] converts it
/// to a paper-white-relative linear gray whenever [`EffectiveDisplayTarget`]
/// changes.
#[derive(Component, Clone, Copy)]
enum Patch {
    /// An absolute luminance in nits (drawn at `nits / paper_white_nits`).
    Nits(f32),
    /// A fraction of the resolved `peak_luminance_nits`.
    PeakFraction(f32),
    /// A value in paper-white-relative units (`1.0` = paper white).
    PaperWhiteRelative(f32),
}

/// Marks one pattern's root; visibility follows [`CalibrationStep`].
#[derive(Component, Clone, Copy)]
struct PatternRoot(CalibrationStep);

/// Tags the banner (top): step title, instruction, HGIG / SDR / strategy notes.
#[derive(Component, Default, Clone, FromTemplate)]
struct BannerText;

/// Tags the value bar (bottom): the single value the step edits and its keys.
#[derive(Component, Default, Clone, FromTemplate)]
struct ValueBarText;

/// Tags the black-level legend row, shown only on step 3.
#[derive(Component, Default, Clone, FromTemplate)]
struct BlackLevelRow;

/// Tags one black-level nit label by its [`BLACK_STEP_NITS`] index.
#[derive(Component, Clone, Copy, FromTemplate)]
struct BlackLevelLabel(usize);

/// Calibration applied at startup when no saved settings exist (and what Cancel
/// restores): 200-nit paper white on a 1000-nit display. The 1000-nit candidate
/// exceeds most consumer panels, so the peak step is two-directional.
/// Transfer/gamut stay at the SDR default for whatever HDR plugin owns them to
/// overwrite.
const DEFAULT_TARGET: DisplayTarget = DisplayTarget::SDR_SRGB
    .with_paper_white(200.0)
    .with_peak(1000.0);

/// Height of the orthographic view volume in world units.
const VIEW_HEIGHT: f32 = 8.0;

/// Absolute luminances (nits) of the black-level steps, dim to bright.
const BLACK_STEP_NITS: [f32; 7] = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0];

/// Shifts the patch cluster left so it never renders under a right-hand overlay.
const STAGE_SHIFT: f32 = -1.5;

/// The render layer the calibration camera and patches share, so an embedding
/// app's gameplay meshes (layer 0) are not drawn by the calibration camera.
const CALIBRATION_LAYER: usize = 24;

/// Draw order for the calibration camera: above a typical gameplay camera so the
/// fullscreen screen composites on top.
const CAMERA_ORDER: isize = 100;

/// Seeds the primary window's calibration intent at startup: saved settings if
/// present, otherwise [`DEFAULT_TARGET`]. Loads the strategy too.
fn seed_window(
    mut commands: Commands,
    primary_window: Single<(Entity, Option<&DisplayTarget>), With<PrimaryWindow>>,
    mut strategy: ResMut<CalibrationStrategy>,
    mut choice: ResMut<BlackLevelChoice>,
) {
    let (window, existing) = *primary_window;
    // Keep whatever transfer/gamut the window already has; only the three
    // calibration numbers and the strategy are ours to restore.
    let mut target = existing.copied().unwrap_or(DEFAULT_TARGET);
    if let Some(saved_strategy) = restore_calibration(&mut target) {
        *strategy = saved_strategy;
        info!(
            "HdrCalibration: loaded saved settings from {:?}",
            settings_path()
        );
    }

    choice.0 = nearest_black_step(target.min_luminance_nits);
    commands.entity(window).insert((target, strategy.policy()));
}

/// Spawns the calibration camera, the three pattern groups, the Bevy wordmark,
/// and the player UI on entering the calibration state. Everything carries
/// [`DespawnOnExit`] so leaving the state tears it down.
fn spawn_calibration<S: States>(
    mut commands: Commands,
    state: Res<State<S>>,
    mut step: ResMut<CalibrationStep>,
) {
    let state = state.get().clone();
    // Start a fresh wizard each entry; the write also re-marks the step changed,
    // so `update_pattern_visibility` repaints the just-spawned pattern roots.
    *step = CalibrationStep::PeakLuminance;
    let camera = commands
        .spawn_scene(calibration_camera())
        .insert((
            DespawnOnExit(state.clone()),
            Camera {
                order: CAMERA_ORDER,
                ..default()
            },
        ))
        .id();

    commands
        .spawn_scene(patterns())
        .insert(DespawnOnExit(state.clone()));

    commands
        .spawn_scene(player_ui())
        .insert((DespawnOnExit(state), UiTargetCamera(camera)));
}

// --- BSN scene functions ---------------------------------------------------

/// The [`Tonemapping::None`] fp16 orthographic camera the patches render through.
fn calibration_camera() -> impl Scene {
    bsn! {
        Camera3d
        // fp16 intermediate, so patch values above paper white survive to the
        // display-encoding pass.
        Hdr
        // Patches must reach the encoder unmodified; an operator would bend them.
        template(|_| Ok(Tonemapping::None))
        template_value(Projection::from(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical { viewport_height: VIEW_HEIGHT },
            ..OrthographicProjection::default_3d()
        }))
        template_value(Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y))
        template_value(RenderLayers::layer(CALIBRATION_LAYER))
    }
}

/// One calibration patch: a flat unlit rectangle with its OWN mesh and material
/// handle. The fresh per-entity material is load-bearing: [`update_patch_levels`]
/// mutates `base_color` per patch via `materials.get_mut`, so a shared handle
/// would corrupt every patch. Never hoist the `materials.add(...)` outside and
/// clone the handle.
fn patch(size: Vec2, position: Vec3, level: Patch) -> impl Scene {
    bsn! {
        template(move |ctx| Ok(Mesh3d(
            ctx.resource_mut::<Assets<Mesh>>().add(Rectangle::from_size(size))
        )))
        template(move |ctx| Ok(MeshMaterial3d(
            ctx.resource_mut::<Assets<StandardMaterial>>().add(StandardMaterial {
                base_color: Color::BLACK,
                unlit: true,
                ..default()
            })
        )))
        template_value(Transform::from_translation(position))
        template(move |_| Ok(level))
        template_value(RenderLayers::layer(CALIBRATION_LAYER))
    }
}

/// The Bevy wordmark: a textured unlit quad that tracks paper white
/// (`PaperWhiteRelative(1.0)` keeps its tint at white; the encoder scales it).
/// Reference imagery only, never a measurement element.
fn wordmark(size: Vec2, position: Vec3) -> impl Scene {
    bsn! {
        template(move |ctx| Ok(Mesh3d(
            ctx.resource_mut::<Assets<Mesh>>().add(Rectangle::from_size(size))
        )))
        template(move |ctx| {
            let logo = ctx.resource::<AssetServer>().load("branding/bevy_logo_dark.png");
            Ok(MeshMaterial3d(ctx.resource_mut::<Assets<StandardMaterial>>().add(
                StandardMaterial {
                    base_color: Color::WHITE,
                    base_color_texture: Some(logo),
                    unlit: true,
                    alpha_mode: AlphaMode::Blend,
                    ..default()
                },
            )))
        })
        template_value(Transform::from_translation(position))
        template(move |_| Ok(Patch::PaperWhiteRelative(1.0)))
        template_value(RenderLayers::layer(CALIBRATION_LAYER))
    }
}

/// All three pattern groups under one root, each hidden until its step is active.
fn patterns() -> impl Scene {
    bsn! {
        template_value(Transform::default())
        Visibility::Visible
        Children [
            peak_pattern(),
            paper_pattern(),
            black_pattern(),
        ]
    }
}

/// Step 1: a clipped near-peak surround, a true-black separating frame, and a
/// center square at the candidate peak. The square merges into the surround
/// across the black gap at the display's real peak.
fn peak_pattern() -> impl Scene {
    bsn! {
        template_value(Transform::from_xyz(STAGE_SHIFT, 0.0, 0.0))
        Visibility::Hidden
        template(|_| Ok(PatternRoot(CalibrationStep::PeakLuminance)))
        Children [
            patch(Vec2::new(9.0, 6.0), Vec3::new(0.0, 0.0, 0.0), Patch::Nits(10000.0)),
            patch(Vec2::splat(2.6), Vec3::new(0.0, 0.0, 0.05), Patch::Nits(0.0)),
            patch(Vec2::splat(2.0), Vec3::new(0.0, 0.0, 0.1), Patch::PeakFraction(1.0)),
        ]
    }
}

/// Step 2: a reference white card at paper white over a dim surround, a BT.2408
/// 203-nit strip, two comfort swatches, and the wordmark.
fn paper_pattern() -> impl Scene {
    bsn! {
        template_value(Transform::from_xyz(STAGE_SHIFT, 0.0, 0.0))
        Visibility::Hidden
        template(|_| Ok(PatternRoot(CalibrationStep::PaperWhite)))
        Children [
            patch(Vec2::new(9.0, VIEW_HEIGHT), Vec3::ZERO, Patch::PaperWhiteRelative(0.15)),
            patch(Vec2::new(4.5, 2.6), Vec3::new(0.0, 0.5, 0.1), Patch::PaperWhiteRelative(1.0)),
            patch(Vec2::new(4.5, 0.7), Vec3::new(0.0, -2.2, 0.1), Patch::Nits(203.0)),
            patch(Vec2::splat(0.8), Vec3::new(-1.4, -1.0, 0.1), Patch::PaperWhiteRelative(0.5)),
            patch(Vec2::splat(0.8), Vec3::new(1.4, -1.0, 0.1), Patch::PaperWhiteRelative(0.75)),
            wordmark(Vec2::new(4.0, 1.0), Vec3::new(0.0, 2.6, 0.1)),
        ]
    }
}

/// Step 3: near-black steps at fixed absolute nit levels over true black.
fn black_pattern() -> impl Scene {
    bsn! {
        template_value(Transform::from_xyz(STAGE_SHIFT, 0.0, 0.0))
        Visibility::Hidden
        template(|_| Ok(PatternRoot(CalibrationStep::BlackLevel)))
        Children [
            patch(Vec2::new(9.0, 6.0), Vec3::ZERO, Patch::Nits(0.0)),
            { BLACK_STEP_NITS
                .into_iter()
                .enumerate()
                .map(|(index, nits)| patch(
                    Vec2::new(1.0, 2.4),
                    Vec3::new((index as f32 - 3.0) * 1.3, 0.0, 0.1),
                    Patch::Nits(nits),
                ))
                .collect::<Vec<_>>() }
        ]
    }
}

/// The player UI: banner, value bar, key legend, and the black-level legend row.
/// All plain SDR text; nothing here is a measurement element.
fn player_ui() -> impl Scene {
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            top: px(0),
            left: px(0),
            width: percent(100),
            height: percent(100),
        }
        Visibility::Visible
        Children [
            (
                BannerText
                Text("")
                template(|_| Ok(TextFont::from_font_size(24.0)))
                template(|_| Ok(TextLayout::justify(Justify::Center)))
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
                template(|_| Ok(TextLayout::justify(Justify::Center)))
                Node {
                    position_type: PositionType::Absolute,
                    bottom: px(56),
                    left: px(0),
                    right: px(0),
                }
            ),
            (
                Text("<-/-> adjust   |   N next   P prev   |   M manual/auto   |   Enter save   Esc cancel")
                template(|_| Ok(TextFont::from_font_size(14.0)))
                template(|_| Ok(TextLayout::justify(Justify::Center)))
                Node {
                    position_type: PositionType::Absolute,
                    bottom: px(16),
                    left: px(0),
                    right: px(0),
                }
            ),
            black_level_row(),
        ]
    }
}

/// The black-level legend (step 3 only): a label plus one entry per near-black
/// step, dim to bright. The selected entry brightens in [`update_black_labels`].
fn black_level_row() -> impl Scene {
    bsn! {
        BlackLevelRow
        Visibility::Hidden
        Node {
            position_type: PositionType::Absolute,
            bottom: px(96),
            left: px(0),
            right: px(0),
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::Center,
            column_gap: px(14),
        }
        Children [
            (
                Text("dimmest step you can see (nits):")
                template(|_| Ok(TextFont::from_font_size(13.0)))
                template(|_| Ok(TextColor(Color::srgb(0.5, 0.5, 0.5))))
            ),
            { BLACK_STEP_NITS
                .into_iter()
                .enumerate()
                .map(|(index, nits)| bsn! {
                    BlackLevelLabel({ index })
                    Text({ format!("{nits}") })
                    template(|_| Ok(TextFont::from_font_size(13.0)))
                    template(|_| Ok(TextColor(Color::srgb(0.5, 0.5, 0.5))))
                })
                .collect::<Vec<_>>() }
        ]
    }
}

// --- Per-frame systems -----------------------------------------------------

/// Walks the wizard: `N` / `Space` / right-shoulder next, `P` / left-shoulder
/// prev, `1`/`2`/`3` jump. Clamped at the ends.
fn change_step(
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    mut step: ResMut<CalibrationStep>,
) {
    let gamepad_next = gamepads
        .iter()
        .any(|g| g.just_pressed(GamepadButton::RightTrigger));
    let gamepad_prev = gamepads
        .iter()
        .any(|g| g.just_pressed(GamepadButton::LeftTrigger));

    if keys.any_just_pressed([KeyCode::KeyN, KeyCode::Space]) || gamepad_next {
        *step = step.next();
    }
    if keys.just_pressed(KeyCode::KeyP) || gamepad_prev {
        *step = step.prev();
    }
    for (key, target) in [
        (KeyCode::Digit1, CalibrationStep::PeakLuminance),
        (KeyCode::Digit2, CalibrationStep::PaperWhite),
        (KeyCode::Digit3, CalibrationStep::BlackLevel),
    ] {
        if keys.just_pressed(key) {
            *step = target;
        }
    }
}

/// Flips the calibration strategy with `M` (or gamepad north) and re-applies the
/// implied policy. Making auto/manual an explicit mode is what keeps the peak
/// edit from silently going inert: under [`CalibrationStrategy::TrustOs`] the
/// edit is disabled with an on-screen hint instead.
fn toggle_strategy(
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    mut strategy: ResMut<CalibrationStrategy>,
    mut policy: Single<&mut DisplayCalibrationPolicy, With<PrimaryWindow>>,
) {
    let gamepad = gamepads
        .iter()
        .any(|g| g.just_pressed(GamepadButton::North));
    if keys.just_pressed(KeyCode::KeyM) || gamepad {
        *strategy = match *strategy {
            CalibrationStrategy::ManualHgig => CalibrationStrategy::TrustOs,
            CalibrationStrategy::TrustOs => CalibrationStrategy::ManualHgig,
        };
        **policy = strategy.policy();
    }
}

/// Adjusts the current step's value with Left/Right (or the gamepad d-pad).
/// Peak and black level are inert under [`CalibrationStrategy::TrustOs`] (the
/// engine resolves them), so only paper white responds there. Peak is clamped at
/// or above paper white.
fn adjust_value(
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    time: Res<Time>,
    step: Res<CalibrationStep>,
    strategy: Res<CalibrationStrategy>,
    mut display_target: Single<&mut DisplayTarget, With<PrimaryWindow>>,
    mut choice: ResMut<BlackLevelChoice>,
) {
    let dpad_right = gamepads
        .iter()
        .any(|g| g.just_pressed(GamepadButton::DPadRight));
    let dpad_left = gamepads
        .iter()
        .any(|g| g.just_pressed(GamepadButton::DPadLeft));

    // Discrete direction for the black-level selection.
    let step_dir = (keys.just_pressed(KeyCode::ArrowRight) || dpad_right) as i32
        - (keys.just_pressed(KeyCode::ArrowLeft) || dpad_left) as i32;

    // Continuous rate while a key is held, for the luminance steps.
    let fast = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let held = (keys.pressed(KeyCode::ArrowRight) as i32 - keys.pressed(KeyCode::ArrowLeft) as i32)
        as f32
        * time.delta_secs()
        * if fast { 4.0 } else { 1.0 };

    let trust_os = *strategy == CalibrationStrategy::TrustOs;
    match *step {
        CalibrationStep::PeakLuminance if !trust_os => {
            if held != 0.0 {
                let paper = display_target.paper_white_nits;
                display_target.peak_luminance_nits =
                    (display_target.peak_luminance_nits + held * 250.0).clamp(paper, 10000.0);
            }
        }
        CalibrationStep::PaperWhite => {
            if held != 0.0 {
                display_target.paper_white_nits =
                    (display_target.paper_white_nits + held * 50.0).clamp(80.0, 500.0);
                // Keep peak at or above the new paper white.
                display_target.peak_luminance_nits = display_target
                    .peak_luminance_nits
                    .max(display_target.paper_white_nits);
            }
        }
        CalibrationStep::BlackLevel if !trust_os && step_dir != 0 => {
            let last = BLACK_STEP_NITS.len() - 1;
            choice.0 = (choice.0 as i32 + step_dir).clamp(0, last as i32) as usize;
            display_target.min_luminance_nits = BLACK_STEP_NITS[choice.0];
        }
        _ => {}
    }
}

/// Confirms (`Enter` / gamepad south) -> persist, emit [`CalibrationComplete`].
/// Cancels (`Esc` / gamepad east) -> restore the saved-or-default calibration.
/// Both leave the calibration state to the app's `CalibrationComplete` handler.
fn confirm_or_cancel(
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    strategy: Res<CalibrationStrategy>,
    window: Single<(&mut DisplayTarget, &DisplayCalibrationPolicy), With<PrimaryWindow>>,
    mut choice: ResMut<BlackLevelChoice>,
    mut commands: Commands,
) {
    let confirm = keys.just_pressed(KeyCode::Enter)
        || gamepads
            .iter()
            .any(|g| g.just_pressed(GamepadButton::South));
    let cancel = keys.just_pressed(KeyCode::Escape)
        || gamepads.iter().any(|g| g.just_pressed(GamepadButton::East));

    let (mut display_target, policy) = window.into_inner();

    if confirm {
        let target = *display_target;
        save_settings(&Settings {
            strategy: *strategy,
            paper_white_nits: target.paper_white_nits,
            peak_luminance_nits: target.peak_luminance_nits,
            min_luminance_nits: target.min_luminance_nits,
        });
        commands.trigger(CalibrationComplete {
            target,
            policy: *policy,
        });
    } else if cancel {
        let mut target = *display_target;
        restore_calibration(&mut target);
        choice.0 = nearest_black_step(target.min_luminance_nits);
        *display_target = target;
    }
}

/// Shows the current step's patches and the step-specific UI, hiding the rest.
fn update_pattern_visibility(
    step: Res<CalibrationStep>,
    mut roots: Query<(&PatternRoot, &mut Visibility), Without<BlackLevelRow>>,
    mut black_row: Single<&mut Visibility, With<BlackLevelRow>>,
) {
    if !step.is_changed() {
        return;
    }
    for (root, mut visibility) in &mut roots {
        *visibility = if root.0 == *step {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
    **black_row = if *step == CalibrationStep::BlackLevel {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
}

/// Recolors patches from the resolved [`EffectiveDisplayTarget`] when it changes
/// or a patch is freshly spawned. Reading the resolved target (not the authored
/// [`DisplayTarget`]) is what lets a sensed peak or paper white show through under
/// [`CalibrationStrategy::TrustOs`].
fn update_patch_levels(
    effective: Single<Ref<EffectiveDisplayTarget>, With<PrimaryWindow>>,
    patches: Query<(Ref<Patch>, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Repaint on a resolved-target change (sensing/edits) or a just-spawned patch
    // (startup, or re-entering the calibration state).
    let target_changed = effective.is_changed();
    let target = effective.target;
    let paper_white = target.sanitized_paper_white_nits();
    for (patch, material) in &patches {
        if !target_changed && !patch.is_added() {
            continue;
        }
        let value = match *patch {
            Patch::Nits(nits) => nits / paper_white,
            Patch::PeakFraction(fraction) => fraction * target.peak_luminance_nits / paper_white,
            Patch::PaperWhiteRelative(value) => value,
        };
        if let Some(mut material) = materials.get_mut(&material.0) {
            material.base_color = Color::linear_rgb(value, value, value);
        }
    }
}

/// Writes the banner: step title and instruction, plus the prominent HGIG note on
/// step 1, the SDR-inert note on step 2, the strategy line, and a monitor-change
/// recalibration notice.
fn update_banner(
    mut text: Single<&mut Text, With<BannerText>>,
    step: Res<CalibrationStep>,
    strategy: Res<CalibrationStrategy>,
    surface: Option<Single<&WindowSurfaceTransfers, With<PrimaryWindow>>>,
    monitor_notice: Res<MonitorChangeNotice>,
) {
    let is_sdr = surface.is_none_or(|s| !s.resolved.is_hdr());
    let (name, instruction) = match *step {
        CalibrationStep::PeakLuminance => (
            "PEAK LUMINANCE",
            "Raise PEAK until the centre square just vanishes into the bright surround, then lower \
             one tap. If it is already invisible, lower PEAK until it reappears first.",
        ),
        CalibrationStep::PaperWhite => (
            "PAPER WHITE",
            "Adjust PAPER WHITE until the card looks like comfortable white paper for your room \
             (dark ~100, normal ~200, bright ~250 nits).",
        ),
        CalibrationStep::BlackLevel => (
            "BLACK LEVEL",
            "Select the dimmest step you can still tell apart from black; its value becomes the \
             recorded black level (min luminance).",
        ),
    };

    let mut banner = format!("Step {} of 3 - {name}\n{instruction}", step.index());

    match *strategy {
        CalibrationStrategy::ManualHgig => {
            banner.push_str("\nStrategy: MANUAL (HGIG) - press M to trust the OS instead.");
        }
        CalibrationStrategy::TrustOs => {
            banner.push_str(
                "\nStrategy: TRUST OS - peak/black come from the display; only paper white is manual \
                 (press M for manual).",
            );
        }
    }

    if *step == CalibrationStep::PeakLuminance {
        banner.push_str(
            "\nFirst put your display in HGIG / Game / passthrough mode (disable its dynamic tone \
             mapping), or you measure its roll-off, not the panel peak.",
        );
    }
    if *step == CalibrationStep::PaperWhite && is_sdr {
        banner.push_str(
            "\nDisplay is SDR: this step has no visible effect until an HDR transfer is active.",
        );
    }

    if monitor_notice.seconds_remaining > 0.0 {
        banner.push_str("\nWindow moved to a different monitor - recalibration recommended.");
    }

    text.0 = banner;
}

/// Writes the value bar: the single value the current step edits, reading the
/// authored [`DisplayTarget`] (the value the keys move).
fn update_value_bar(
    mut text: Single<&mut Text, With<ValueBarText>>,
    display_target: Single<&DisplayTarget, With<PrimaryWindow>>,
    step: Res<CalibrationStep>,
    strategy: Res<CalibrationStrategy>,
) {
    let trust_os = *strategy == CalibrationStrategy::TrustOs;
    text.0 = match *step {
        CalibrationStep::PeakLuminance if trust_os => {
            "PEAK is Auto (from the display). Press M for manual.".to_string()
        }
        CalibrationStep::PeakLuminance => format!(
            "PEAK  {:.0} nits   (<- / ->,  Shift = 4x)",
            display_target.peak_luminance_nits,
        ),
        CalibrationStep::PaperWhite => format!(
            "PAPER WHITE  {:.0} nits   (<- / ->,  Shift = 4x)",
            display_target.paper_white_nits,
        ),
        CalibrationStep::BlackLevel if trust_os => {
            "BLACK LEVEL is Auto (from the display). Press M for manual.".to_string()
        }
        CalibrationStep::BlackLevel => format!(
            "BLACK LEVEL  {:.2} nits   (<- / -> to choose)",
            display_target.min_luminance_nits,
        ),
    };
}

/// Brightens the selected near-black label and dims the rest, on a selection
/// change or a freshly-spawned label (re-entry).
fn update_black_labels(
    choice: Res<BlackLevelChoice>,
    mut labels: Query<(Ref<BlackLevelLabel>, &mut TextColor)>,
) {
    let changed = choice.is_changed();
    for (label, mut color) in &mut labels {
        if !changed && !label.is_added() {
            continue;
        }
        color.0 = if label.0 == choice.0 {
            Color::WHITE
        } else {
            Color::srgb(0.5, 0.5, 0.5)
        };
    }
}

/// Raises a recalibration notice when the window moves to a different monitor.
/// Sole owner of the notice countdown. The first event per window only reports
/// the monitor becoming known at startup, so it is logged, not raised.
fn watch_monitor_changes(
    mut events: MessageReader<WindowMonitorChanged>,
    monitors: Query<&Monitor>,
    mut notice: ResMut<MonitorChangeNotice>,
    time: Res<Time>,
    mut known_windows: Local<HashSet<Entity>>,
) {
    for event in events.read() {
        let name = event
            .monitor
            .and_then(|monitor| monitors.get(monitor).ok())
            .and_then(|monitor| monitor.name.clone())
            .unwrap_or_else(|| "<unknown>".into());
        if known_windows.insert(event.window) {
            info!("Window {} is on monitor {name}.", event.window);
        } else {
            info!(
                "Window {} moved to monitor {name}; recalibration recommended.",
                event.window
            );
            notice.seconds_remaining = 8.0;
        }
    }
    notice.seconds_remaining = (notice.seconds_remaining - time.delta_secs()).max(0.0);
}

// --- Persistence -----------------------------------------------------------

/// The persisted calibration: the strategy plus the three authored numbers.
/// Only *authored* values are written - never a sensed/effective value.
#[derive(Serialize, Deserialize, Clone, Copy)]
struct Settings {
    strategy: CalibrationStrategy,
    paper_white_nits: f32,
    peak_luminance_nits: f32,
    min_luminance_nits: f32,
}

/// Applies the saved-or-default calibration numbers to `target`, returning the
/// saved strategy when settings existed. Only the three luminance numbers are
/// touched; transfer/gamut stay as they are.
fn restore_calibration(target: &mut DisplayTarget) -> Option<CalibrationStrategy> {
    let saved = load_settings();
    let numbers = saved.unwrap_or(Settings {
        strategy: CalibrationStrategy::ManualHgig,
        paper_white_nits: DEFAULT_TARGET.paper_white_nits,
        peak_luminance_nits: DEFAULT_TARGET.peak_luminance_nits,
        min_luminance_nits: DEFAULT_TARGET.min_luminance_nits,
    });
    target.paper_white_nits = numbers.paper_white_nits;
    target.peak_luminance_nits = numbers.peak_luminance_nits;
    target.min_luminance_nits = numbers.min_luminance_nits;
    saved.map(|s| s.strategy)
}

/// Where calibration settings are stored: next to the executable (an
/// example-appropriate location; a real app would use the OS preferences dir).
fn settings_path() -> PathBuf {
    let dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."));
    dir.join("hdr_calibration_settings.ron")
}

/// Reads saved settings, or `None` if absent or unparseable.
fn load_settings() -> Option<Settings> {
    let text = fs::read_to_string(settings_path()).ok()?;
    ron::from_str(&text).ok()
}

/// Writes settings, logging the path (or a failure).
fn save_settings(settings: &Settings) {
    let path = settings_path();
    match ron::ser::to_string_pretty(settings, ron::ser::PrettyConfig::default()) {
        Ok(text) => match fs::write(&path, text) {
            Ok(()) => info!("HdrCalibration: saved settings to {path:?}"),
            Err(error) => warn!("HdrCalibration: could not write {path:?}: {error}"),
        },
        Err(error) => warn!("HdrCalibration: could not serialize settings: {error}"),
    }
}

/// The [`BLACK_STEP_NITS`] index nearest a min-luminance value.
fn nearest_black_step(min_nits: f32) -> usize {
    BLACK_STEP_NITS
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - min_nits).abs().total_cmp(&(*b - min_nits).abs()))
        .map(|(index, _)| index)
        .unwrap_or(0)
}
