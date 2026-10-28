//! This example showcases pbr atmospheric scattering
//!
//! Environment variables (all optional; used by the cloud validation scripts):
//! - Scene: `ATMOSPHERE_VIEW=hero|sun|up|under|backlit`, `ATMOSPHERE_CAM`, `ATMOSPHERE_SUN_ELEV_DEG`,
//!   `ATMOSPHERE_SUNSET`, `ATMOSPHERE_TERRAIN`, `ATMOSPHERE_NO_CLOUDS`, `ATMOSPHERE_NO_CIRRUS`,
//!   `ATMOSPHERE_NO_CUMULUS` (cirrus only), `ATMOSPHERE_WIND=slow|opposing`, `ATMOSPHERE_NO_TAA`,
//!   `ATMOSPHERE_MESH_PLANE`, `ATMOSPHERE_MSAA=2|4|8`, `ATMOSPHERE_WINDOW=w,h`,
//!   `ATMOSPHERE_CLOUD_QUALITY=low|medium|high|cinematic`, `ATMOSPHERE_SECOND_THIN=1` (cirrostratus layer),
//!   `ATMOSPHERE_INFLUENCE=hole|coast` (a world-anchored clear region; a maritime/continental coast), `ATMOSPHERE_CLOUD_STEP_SAMPLES=<n>`,
//!   `ATMOSPHERE_CLOUD_PROFILE=nubis|adiabatic|nubis_envelope`,
//!   `ATMOSPHERE_CLOUD_LWC=<g/m3>`,
//!   `ATMOSPHERE_CLOUD_DENSITY=<x>`,
//!   `ATMOSPHERE_LUT_MODE`.
//! - Time: `ATMOSPHERE_CAM_STEP`, `ATMOSPHERE_FREEZE`, `ATMOSPHERE_FIXED_DT`,
//!   `ATMOSPHERE_CLOUD_TIMESCALE=<x>` (cloud clock multiplier; `T` toggles a 60x time-lapse),
//!   `ATMOSPHERE_SUN_SWEEP=<deg/frame>,<first>,<last>` (turns the sun by a fixed angle per frame),
//!   `ATMOSPHERE_WIND_SCALE=<x>` (wind profile multiplier, 0..2; `-` / `=`).
//! - Debug view: `ATMOSPHERE_DEBUG_VIEW=1` starts with the LUT / buffer previews shown (Tab).
//! - Cloud shadows: `ATMOSPHERE_SHADOW_INTERLEAVE`, `ATMOSPHERE_SHADOW_SAMPLES`,
//!   `ATMOSPHERE_SHADOW_TEMPORAL_ALPHA`.
//! - Capture / runs: `ATMOSPHERE_BACKGROUND`, `ATMOSPHERE_SCREENSHOT`, `ATMOSPHERE_SCREENSHOT_SECS`,
//!   `ATMOSPHERE_SCREENSHOT_FRAMES`, `ATMOSPHERE_EXIT_SECS`, `ATMOSPHERE_EXIT_FRAME`,
//!   `ATMOSPHERE_DIAGNOSTICS`, `ATMOSPHERE_CLEAR`.
//! - Read by `bevy_pbr`: `ATMOSPHERE_DBG` (comma-separated `ATM_DBG_*` shader defs, see
//!   `atmosphere/resources.rs`), `ATMOSPHERE_GPU_TIMING` (per-pass CSV).
#[cfg(feature = "free_camera")]
use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState};
use std::f32::consts::PI;

use bevy::{
    anti_alias::taa::TemporalAntiAliasing,
    camera::Exposure,
    color::palettes::css::BLACK,
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    image::{
        ImageAddressMode, ImageFilterMode, ImageLoaderSettings, ImageSampler,
        ImageSamplerDescriptor,
    },
    input::keyboard::KeyCode,
    light::{
        atmosphere::{Aerosol, ScatteringMedium},
        light_consts::lux,
        Atmosphere, AtmosphereEnvironmentMapLight, FogVolume, SunDisk, VolumetricFog,
        VolumetricLight,
    },
    pbr::{
        AtmosphereMode, AtmosphereSettings, AtmosphereWind, CloudTimeScale, CloudWeather,
        CloudBlend, CloudInfluence, CloudInfluenceShape, CloudInfluenceValues, CloudLayerOf,
        CloudProfiles, ThinCloudLayer, VolumetricCloudLayer,
        AtmosphereDebugView, AtmosphereDebugViewPlugin,
        DefaultOpaqueRendererMethod,
        VolumetricCloudSettings,
        ExtendedMaterial, MaterialExtension, ScreenSpaceReflections,
    },
    post_process::bloom::Bloom,
    prelude::*,
    render::{
        diagnostic::RenderDiagnosticsPlugin,
        render_resource::{AsBindGroup, ShaderType},
        view::screenshot::{save_to_disk, Screenshot},
    },
    shader::ShaderRef,
};

#[derive(Resource)]
struct GameState {
    paused: bool,
    planet_scale_view: bool,
}

impl Default for GameState {
    fn default() -> Self {
        Self {
            paused: true,
            planet_scale_view: false,
        }
    }
}

#[derive(Resource)]
struct AtmospherePresets {
    earth: Handle<ScatteringMedium>,
    mars: Handle<ScatteringMedium>,
}

fn main() {
    let mut app = App::new();
    // Continuous mode waits for the OS redraw callback, which macOS stops delivering for occluded
    // windows (the app then freezes in the background). In background test runs drive updates
    // from the event loop instead (reactive with ~0 wait => ControlFlow::Poll).
    let background = std::env::var("ATMOSPHERE_BACKGROUND").is_ok();
    let update_mode = if background {
        bevy::winit::UpdateMode::reactive(core::time::Duration::from_micros(100))
    } else {
        bevy::winit::UpdateMode::Continuous
    };
    app.insert_resource(bevy::winit::WinitSettings {
        focused_mode: update_mode,
        unfocused_mode: update_mode,
    })
    .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(ClearColor(
            // Debug: `ATMOSPHERE_CLEAR=r,g,b` (linear) to test background leakage through the
            // sky's dual-source transmittance blend.
            std::env::var("ATMOSPHERE_CLEAR")
                .ok()
                .and_then(|v| {
                    let c: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                    (c.len() == 3).then(|| Color::linear_rgb(c[0], c[1], c[2]))
                })
                .unwrap_or(Color::BLACK),
        ))
        .insert_resource(GameState::default())
        .insert_resource(GlobalAmbientLight::NONE)
        .add_plugins((
            // `ATMOSPHERE_BACKGROUND=1` (used by the test scripts): start unfocused so macOS does
            // not activate the app / steal focus; rendering stays continuous when unfocused.
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    focused: !background,
                    visible: !background,
                    // No vsync in background runs: an occluded window's drawables are not
                    // display-synced, so acquisition doesn't stall.
                    present_mode: if background {
                        bevy::window::PresentMode::AutoNoVsync
                    } else {
                        bevy::window::PresentMode::AutoVsync
                    },
                    // Fixed 1280x720 physical pixels so dumps match across displays (retina = 2x);
                    // `ATMOSPHERE_WINDOW=w,h` for another size.
                    resolution: {
                        let (w, h) = window_size();
                        bevy::window::WindowResolution::new(w, h).with_scale_factor_override(1.0)
                    },
                    ..default()
                }),
                ..default()
            }),
            #[cfg(feature = "free_camera")]
            FreeCameraPlugin,
        ))
        .add_plugins(MaterialPlugin::<ExtendedMaterial<StandardMaterial, Water>>::default())
        .add_plugins(AtmosphereDebugViewPlugin)
        .add_systems(
            Startup,
            (
                setup_camera_fog,
                setup_terrain_scene,
                setup_debug_overlay.after(setup_camera_fog),
                print_controls,
            ),
        )
        .add_systems(
            Update,
            (
                dynamic_scene,
                sun_drag,
                sun_sweep,
                atmosphere_controls,
                debug_overlay_controls,
                debug_overlay_scale,
                wind_speed_controls,
                cycle_cloud_profiles,
                visibility_toggle.after(atmosphere_controls),
                camera_pose_title,
            ),
        );

    // Optional benchmarking hooks (used for profiling the atmosphere/cloud passes):
    // - `ATMOSPHERE_DIAGNOSTICS=1` logs frame time and per-pass render timings once per second.
    // - `ATMOSPHERE_SCREENSHOT=<path.png>` saves a screenshot after `ATMOSPHERE_SCREENSHOT_SECS` (default 8 s).
    // - `ATMOSPHERE_EXIT_SECS=<secs>` exits the app after the given number of seconds.
    // - `ATMOSPHERE_EXIT_FRAME=<n>` exits after n frames (e.g. just past the last dump frame).
    let bench = BenchSettings::from_env();
    if bench.diagnostics {
        app.add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            RenderDiagnosticsPlugin,
            LogDiagnosticsPlugin::default(),
        ));
    }
    // `ATMOSPHERE_FREEZE=1`: pause virtual time (sun motion, wind/noise offsets, animations) so
    // realtime and reference renders see identical clouds.
    if std::env::var("ATMOSPHERE_FREEZE").is_ok() {
        app.add_systems(Startup, |mut time: ResMut<Time<Virtual>>| time.pause());
    }
    if let Some(scale) = std::env::var("ATMOSPHERE_CLOUD_TIMESCALE").ok().and_then(|v| v.parse().ok()) {
        app.insert_resource(CloudTimeScale(scale));
    }
    // `ATMOSPHERE_FIXED_DT=1`: advance time by exactly 1/60 s per frame, so animated captures at
    // a given frame are reproducible regardless of GPU speed.
    if std::env::var("ATMOSPHERE_FIXED_DT").is_ok() {
        app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
            core::time::Duration::from_secs_f64(1.0 / 60.0),
        ));
    }
    // `ATMOSPHERE_CAM_STEP=x,y,z`: move the camera by this many metres every frame (independent
    // of time, so frozen runs still move); used to test shadow-map / field recentering.
    if let Some(step) = std::env::var("ATMOSPHERE_CAM_STEP").ok().and_then(|v| {
        let c: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (c.len() == 3).then(|| Vec3::new(c[0], c[1], c[2]))
    }) {
        app.add_systems(Update, move |mut cams: Query<&mut Transform, With<Camera3d>>| {
            for mut t in &mut cams {
                t.translation += step;
            }
        });
    }
    // `ATMOSPHERE_CLOUD_LIFETIME_OFF_SECS=<t>`: at t (real time), turn the cloud life cycle off
    // (static weather); used to test that the weather map re-bakes.
    if let Some(secs) = env_f32("ATMOSPHERE_CLOUD_LIFETIME_OFF_SECS") {
        app.add_systems(
            Update,
            move |time: Res<Time<Real>>, mut layers: Query<&mut VolumetricCloudLayer>| {
                if time.elapsed_secs() >= secs {
                    for mut layer in &mut layers {
                        if layer.lifetime != 0.0 {
                            layer.lifetime = 0.0;
                        }
                    }
                }
            },
        );
    }
    // `ATMOSPHERE_AEROSOL_EDIT_SECS=<t>`: at t, switch the Earth medium's aerosol to
    // `Aerosol::URBAN` by editing the asset in place (same handle); used to test that the
    // atmosphere LUTs follow in-place medium edits.
    if let Some(secs) = env_f32("ATMOSPHERE_AEROSOL_EDIT_SECS") {
        app.add_systems(
            Update,
            // Real time: `ATMOSPHERE_FREEZE` pauses virtual time.
            move |time: Res<Time<Real>>,
                  presets: Res<AtmospherePresets>,
                  mut media: ResMut<Assets<ScatteringMedium>>,
                  mut done: Local<bool>| {
                if !*done && time.elapsed_secs() >= secs {
                    *done = true;
                    if let Some(mut medium) = media.get_mut(&presets.earth) {
                        *medium = ScatteringMedium::earth_with_aerosol(256, 256, Aerosol::URBAN);
                    }
                }
            },
        );
    }
    if bench.screenshot.is_some() || bench.exit_secs.is_some() || bench.exit_frame.is_some() {
        app.insert_resource(bench).add_systems(Update, bench_hooks);
    }
    app.run();
}

#[derive(Resource, Default)]
struct BenchSettings {
    diagnostics: bool,
    screenshot: Option<String>,
    screenshot_secs: f32,
    exit_secs: Option<f32>,
    exit_frame: Option<u32>,
    no_clouds: bool,
    sunset: bool,
    screenshot_taken: bool,
    startup_frames: Option<u32>,
    frame: u32,
}

impl BenchSettings {
    fn from_env() -> Self {
        let var = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
        Self {
            diagnostics: var("ATMOSPHERE_DIAGNOSTICS").is_some(),
            screenshot: var("ATMOSPHERE_SCREENSHOT"),
            screenshot_secs: var("ATMOSPHERE_SCREENSHOT_SECS")
                .and_then(|v| v.parse().ok())
                .unwrap_or(8.0),
            exit_secs: var("ATMOSPHERE_EXIT_SECS").and_then(|v| v.parse().ok()),
            exit_frame: var("ATMOSPHERE_EXIT_FRAME").and_then(|v| v.parse().ok()),
            no_clouds: var("ATMOSPHERE_NO_CLOUDS").is_some(),
            sunset: var("ATMOSPHERE_SUNSET").is_some(),
            screenshot_taken: false,
            startup_frames: var("ATMOSPHERE_SCREENSHOT_FRAMES").and_then(|v| v.parse().ok()),
            frame: 0,
        }
    }
}

fn bench_hooks(
    mut commands: Commands,
    time: Res<Time<Real>>,
    mut bench: ResMut<BenchSettings>,
    mut exit: MessageWriter<AppExit>,
    cloud_layers: Query<Entity, With<VolumetricCloudLayer>>,
    offscreen: Option<Res<OffscreenTarget>>,
    second: Option<Res<SecondCameraTarget>>,
) {
    let offscreen = OffscreenTarget(offscreen.and_then(|o| o.0.clone()));
    let _sunset = bench.sunset; // sun tilt applied at spawn via ATMOSPHERE_SUNSET
    if bench.no_clouds {
        for entity in &cloud_layers {
            commands
                .entity(entity)
                .remove::<(VolumetricCloudLayer, ThinCloudLayer)>();
        }
    }
    let elapsed = time.elapsed_secs();
    if !bench.screenshot_taken
        && elapsed >= bench.screenshot_secs
        && let Some(path) = bench.screenshot.clone()
    {
        bench.screenshot_taken = true;
        commands
            .spawn(offscreen.0.clone().map_or_else(Screenshot::primary_window, Screenshot::image))
            .observe(save_to_disk(path));
        if let (Some(second), Ok(path)) = (second.as_ref(), std::env::var("ATMOSPHERE_SECOND_SCREENSHOT")) {
            commands
                .spawn(Screenshot::image(second.0.clone()))
                .observe(save_to_disk(path));
        }
    }
    // `ATMOSPHERE_SCREENSHOT_FRAMES=<n>` + `ATMOSPHERE_SCREENSHOT=<dir/prefix>`: also capture each
    // of the first n frames (startup convergence debugging) to `<prefix>_fNN.png`.
    if let (Some(n), Some(prefix)) = (bench.startup_frames, bench.screenshot.clone())
        && bench.frame < n
    {
        commands
            .spawn(offscreen.0.clone().map_or_else(Screenshot::primary_window, Screenshot::image))
            .observe(save_to_disk(format!("{prefix}_f{:02}.png", bench.frame)));
    }
    bench.frame += 1;
    if bench.exit_secs.is_some_and(|secs| elapsed >= secs)
        || bench.exit_frame.is_some_and(|f| bench.frame >= f)
    {
        exit.write(AppExit::Success);
    }
}

fn print_controls() {
    println!("Atmosphere Example Controls:");
    println!("    1          - Switch to lookup texture rendering method");
    println!("    2          - Switch to raymarched rendering method");
    println!("    3          - Switch to Earth atmosphere");
    println!("    4          - Switch to Mars atmosphere");
    println!("    5          - Switch between normal and planet scale (space) view");
    println!("    C          - Toggle cloud layer");
    println!("    Left/Right - Decrease/Increase cloud coverage");
    println!("    P          - Cycle cloud profiles (physical, Nubis, adiabatic, Nubis envelope)");
    println!("    Enter      - Pause/Resume sun motion");
    println!("    T          - Toggle 60x cloud time-lapse (wind, rising thermals, cloud life cycle)");
    println!("    Right drag - Slide the time of day (sun position)");
    println!("    Up/Down    - Increase/Decrease exposure");
    println!("    - / =      - Decrease/Increase the wind speed (whole profile: drift and lean), 0x .. 2x");
    println!("    Tab        - Toggle the debug view (LUTs, cloud buffers, cloud field, weather, noise)");
    println!("    [ / ]      - Debug view: move the slice of the 3D textures down/up");
}

/// The debug overlay's root node (Tab).
#[derive(Component)]
struct DebugOverlay;

/// The debug overlay's slice readout.
#[derive(Component)]
struct DebugSliceText;

/// Wind speed multiplier (- / =) on the atmosphere's wind profile as spawned.
struct WindScale {
    base: Option<AtmosphereWind>,
    scale: f32,
}

impl Default for WindScale {
    fn default() -> Self {
        Self { base: None, scale: 1.0 }
    }
}

/// Scales the whole wind profile (- / =): the clouds' drift and, through the shear, their lean.
/// Up to 2x: with stronger shear the lean changes across air masses faster than the field can
/// follow (`VolumetricCloudLayer::wind_shear`). `ATMOSPHERE_WIND_SCALE=<x>` sets it at startup.
fn wind_speed_controls(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut state: Local<WindScale>,
    mut winds: Query<&mut AtmosphereWind>,
) {
    let initial = (state.base.is_none() && !winds.is_empty())
        .then(|| env_f32("ATMOSPHERE_WIND_SCALE"))
        .flatten();
    if let Some(scale) = initial {
        state.scale = scale.clamp(0.0, 2.0);
    } else if keyboard_input.just_pressed(KeyCode::Equal) {
        state.scale = (state.scale + 0.25).min(2.0);
    } else if keyboard_input.just_pressed(KeyCode::Minus) {
        state.scale = (state.scale - 0.25).max(0.0);
    } else {
        if state.base.is_none() {
            for wind in &winds {
                state.base = Some(wind.clone());
            }
        }
        return;
    }
    let scale = state.scale;
    for mut wind in &mut winds {
        let base = state.base.get_or_insert_with(|| wind.clone());
        wind.levels = base.levels.iter().map(|&(z, v)| (z, v * scale)).collect();
    }
    let (bottom, top) = bevy::pbr::physical_profile_bounds();
    let shear = state.base.as_ref().map_or(0.0, |b| b.mean_shear(bottom, top).length()) * scale;
    println!("Wind x{scale:.2} (cloud-layer shear {:.1} m/s per km)", shear * 1e3);
}

/// Edge of the debug grid at UI scale 1 (logical px): 5 x 4 tiles with labels.
const DEBUG_GRID: Vec2 = Vec2::new(5.0 * (DEBUG_TILE + 4.0) + 8.0, 4.0 * (DEBUG_TILE + 26.0) + 24.0);

/// The debug overlay at the display's scale: the window is pinned to scale factor 1 (dumps match
/// across displays), so on a HiDPI display the overlay would be half size. The UI scale is the
/// display's own scale factor (2 on HiDPI), at most what keeps the grid within 60 % of the
/// window's width and its height.
fn debug_overlay_scale(windows: Query<&Window, With<bevy::window::PrimaryWindow>>, mut ui_scale: ResMut<UiScale>) {
    let Ok(window) = windows.single() else {
        return;
    };
    let fit = (0.6 * window.physical_width() as f32 / DEBUG_GRID.x)
        .min(0.98 * window.physical_height() as f32 / DEBUG_GRID.y);
    let scale = window.resolution.base_scale_factor().min(fit).max(0.5);
    if (ui_scale.0 - scale).abs() > 1e-3 {
        ui_scale.0 = scale;
    }
}

/// Edge of a debug preview in the overlay (logical px).
const DEBUG_TILE: f32 = 136.0;

/// The debug view (`AtmosphereDebugView`): a grid of live previews of the atmosphere's lookup
/// tables, cloud buffers, cloud field, weather and noise, drawn by the engine into images each
/// frame. Hidden until Tab; `ATMOSPHERE_DEBUG_VIEW=1` starts with it shown.
fn setup_debug_overlay(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<(Entity, &Camera), With<Camera3d>>,
) {
    let mut view = AtmosphereDebugView::all(&mut images);
    view.enabled = std::env::var("ATMOSPHERE_DEBUG_VIEW").is_ok();
    let columns = 5.0;
    let mut overlay = commands.spawn_empty();
    // On the main camera, which renders offscreen in background runs.
    if let Some((camera, _)) = cameras.iter().min_by_key(|(_, c)| c.order) {
        overlay.insert(UiTargetCamera(camera));
    }
    overlay
        .insert((
            DebugOverlay,
            Node {
                position_type: PositionType::Absolute,
                top: px(8),
                right: px(8),
                width: px(columns * (DEBUG_TILE + 4.0) + 8.0),
                flex_wrap: FlexWrap::Wrap,
                column_gap: px(4),
                row_gap: px(2),
                padding: UiRect::all(px(4)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.7)),
            if view.enabled { Visibility::Inherited } else { Visibility::Hidden },
        ))
        .with_children(|parent| {
            for panel in &view.panels {
                parent
                    .spawn(Node {
                        flex_direction: FlexDirection::Column,
                        width: px(DEBUG_TILE),
                        ..default()
                    })
                    .with_children(|tile| {
                        tile.spawn((
                            ImageNode::new(panel.image.clone()),
                            Node {
                                width: px(DEBUG_TILE),
                                height: px(DEBUG_TILE),
                                ..default()
                            },
                        ));
                        tile.spawn((
                            Text::new(panel.label.clone()),
                            TextFont {
                                font_size: FontSize::Px(10.0),
                                ..default()
                            },
                            TextColor(Color::srgb(0.85, 0.85, 0.85)),
                        ));
                    });
            }
            parent.spawn((
                DebugSliceText,
                Text::new("3D slice 0.50 ([ / ])"),
                TextFont {
                    font_size: FontSize::Px(10.0),
                    ..default()
                },
                TextColor(Color::srgb(0.85, 0.85, 0.85)),
            ));
        });
    commands.insert_resource(view);
}

/// Tab shows/hides the debug view (the engine skips the previews while hidden); [ and ] move
/// the slice of every 3D texture (cloud field and aerial perspective: height / distance).
fn debug_overlay_controls(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut view: ResMut<AtmosphereDebugView>,
    mut overlay: Query<&mut Visibility, With<DebugOverlay>>,
    mut slice_text: Query<&mut Text, With<DebugSliceText>>,
) {
    if keyboard_input.just_pressed(KeyCode::Tab) {
        view.enabled = !view.enabled;
        for mut visibility in &mut overlay {
            *visibility = if view.enabled { Visibility::Inherited } else { Visibility::Hidden };
        }
    }
    let step = if keyboard_input.just_pressed(KeyCode::BracketRight) {
        1.0 / 32.0
    } else if keyboard_input.just_pressed(KeyCode::BracketLeft) {
        -1.0 / 32.0
    } else {
        return;
    };
    let mut slice = 0.5;
    for panel in &mut view.panels {
        panel.slice = (panel.slice + step).clamp(0.0, 1.0 - 1e-3);
        slice = panel.slice;
    }
    for mut text in &mut slice_text {
        text.0 = format!("3D slice {slice:.2} ([ / ])");
    }
}

fn atmosphere_controls(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut planet_atmosphere: Query<(&mut Atmosphere, &mut GlobalTransform)>,
    mut camera_settings: Query<&mut AtmosphereSettings, With<Camera3d>>,
    mut sun_disks: Query<&mut SunDisk, With<DirectionalLight>>,
    mut atmosphere_entities: Query<(Entity, Option<&mut VolumetricCloudLayer>), With<Atmosphere>>,
    mut commands: Commands,
    atmosphere_presets: Res<AtmospherePresets>,
    mut game_state: ResMut<GameState>,
    mut camera_exposure: Query<&mut Exposure, With<Camera3d>>,
    mut cloud_time_scale: ResMut<CloudTimeScale>,
    time: Res<Time>,
) {
    if keyboard_input.just_pressed(KeyCode::Digit3) {
        for (mut atmosphere, mut transform) in &mut planet_atmosphere {
            *atmosphere = Atmosphere::earth(atmosphere_presets.earth.clone());
            *transform = GlobalTransform::from_translation(-Vec3::Y * atmosphere.inner_radius);
        }
        for mut sun_disk in &mut sun_disks {
            sun_disk.angular_size = SunDisk::EARTH.angular_size;
        }
        println!("Switched to Earth atmosphere");
    }

    if keyboard_input.just_pressed(KeyCode::Digit4) {
        for (mut atmosphere, mut transform) in &mut planet_atmosphere {
            *atmosphere = Atmosphere::mars(atmosphere_presets.mars.clone());
            *transform = GlobalTransform::from_translation(-Vec3::Y * atmosphere.inner_radius);
        }
        for mut sun_disk in &mut sun_disks {
            sun_disk.angular_size = SunDisk::MARS.angular_size;
        }
        println!("Switched to Mars atmosphere");
    }

    if keyboard_input.just_pressed(KeyCode::Digit1) {
        for mut settings in &mut camera_settings {
            settings.rendering_method = AtmosphereMode::LookupTexture;
            println!("Switched to lookup texture rendering method");
        }
    }

    if keyboard_input.just_pressed(KeyCode::Digit2) {
        for mut settings in &mut camera_settings {
            settings.rendering_method = AtmosphereMode::Raymarched;
            println!("Switched to raymarched rendering method");
        }
    }

    if keyboard_input.just_pressed(KeyCode::Digit5) {
        game_state.planet_scale_view = !game_state.planet_scale_view;
        if game_state.planet_scale_view {
            for mut settings in &mut camera_settings {
                settings.aerial_view_lut_size = UVec3::new(160, 88, 64);
                settings.aerial_view_lut_max_distance = 6.2e4;
            }
            println!("Switched to planet scale view");
        } else {
            for mut settings in &mut camera_settings {
                settings.aerial_view_lut_size = UVec3::new(32, 32, 32);
                settings.aerial_view_lut_max_distance = 3.2e4;
            }
            println!("Switched to normal view");
        }
    }

    if keyboard_input.just_pressed(KeyCode::KeyC) {
        let has_clouds = atmosphere_entities
            .iter()
            .any(|(_, cloud_layer)| cloud_layer.is_some());
        let entities: Vec<Entity> = atmosphere_entities
            .iter()
            .map(|(entity, _)| entity)
            .collect();
        if has_clouds {
            for entity in entities {
                commands.entity(entity).remove::<VolumetricCloudLayer>();
            }
            println!("Cloud layer disabled");
        } else {
            for entity in entities {
                commands.entity(entity).insert(VolumetricCloudLayer::default());
            }
            println!("Cloud layer enabled");
        }
    }

    if keyboard_input.pressed(KeyCode::ArrowLeft) {
        for (_, cloud_layer) in atmosphere_entities.iter_mut() {
            if let Some(mut cloud_layer) = cloud_layer {
                cloud_layer.coverage = (cloud_layer.coverage - time.delta_secs()).max(0.0);
                println!("Cloud coverage scale: {:.2}", cloud_layer.coverage);
            }
        }
    }

    if keyboard_input.pressed(KeyCode::ArrowRight) {
        for (_, cloud_layer) in atmosphere_entities.iter_mut() {
            if let Some(mut cloud_layer) = cloud_layer {
                cloud_layer.coverage = (cloud_layer.coverage + time.delta_secs()).min(5.0);
                println!("Cloud coverage scale: {:.2}", cloud_layer.coverage);
            }
        }
    }

    if keyboard_input.just_pressed(KeyCode::Enter) {
        game_state.paused = !game_state.paused;
    }

    if keyboard_input.just_pressed(KeyCode::KeyT) {
        cloud_time_scale.0 = if cloud_time_scale.0 > 1.0 { 1.0 } else { 60.0 };
    }

    if keyboard_input.pressed(KeyCode::ArrowUp) {
        for mut exposure in &mut camera_exposure {
            exposure.ev100 -= time.delta_secs() * 2.0;
        }
    }

    if keyboard_input.pressed(KeyCode::ArrowDown) {
        for mut exposure in &mut camera_exposure {
            exposure.ev100 += time.delta_secs() * 2.0;
        }
    }

    // Animate clouds by updating noise offset
    for (_, cloud_layer) in atmosphere_entities.iter_mut() {
        if let Some(_cloud_layer) = cloud_layer {
            // _cloud_layer.noise_offset.x += time.delta_secs() * 100.0;
        }
    }
}

/// Cycles the vertical cloud profiles of the cloud layer: the baked physical set, Nubis's
/// reconstructed lookup, and the two analytic envelopes.
fn cycle_cloud_profiles(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut profiles: Query<&mut CloudProfiles, With<VolumetricCloudLayer>>,
) {
    if !keyboard_input.just_pressed(KeyCode::KeyP) {
        return;
    }
    for mut profile in &mut profiles {
        let (next, name) = match *profile {
            CloudProfiles::Physical | CloudProfiles::Image(_) => (CloudProfiles::Nubis, "Nubis"),
            CloudProfiles::Nubis => (CloudProfiles::ADIABATIC, "adiabatic envelope"),
            CloudProfiles::Envelope { bottom, .. } if bottom == 1.0 => {
                (CloudProfiles::NUBIS_ENVELOPE, "Nubis envelope")
            }
            CloudProfiles::Envelope { .. } => (CloudProfiles::Physical, "physical"),
        };
        *profile = next;
        println!("Cloud profiles: {name}");
    }
}

fn visibility_toggle(
    game_state: Res<GameState>,
    mut visibility_query: Query<&mut Visibility, Or<(With<WaterPlane>, With<FogVolume>)>>,
) {
    let visibility = if game_state.planet_scale_view {
        Visibility::Hidden
    } else {
        Visibility::Inherited
    };
    for mut v in &mut visibility_query {
        *v = visibility;
    }
}

fn setup_camera_fog(
    mut commands: Commands,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
    mut images: ResMut<Assets<Image>>,
    asset_server: Res<AssetServer>,
) {
    // `ATMOSPHERE_BACKGROUND=1`: render the camera into an offscreen image of the window's size
    // instead of the window. macOS stops redraws/drawables for occluded windows, so a
    // window-backed run freezes when it is behind other windows; the window itself stays hidden
    // and unfocused.
    let (target_w, target_h) = window_size();
    let offscreen = std::env::var("ATMOSPHERE_BACKGROUND").is_ok().then(|| {
        images.add(Image::new_target_texture(
            target_w,
            target_h,
            bevy::render::render_resource::TextureFormat::Rgba8Unorm,
            Some(bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb),
        ))
    });
    commands.insert_resource(OffscreenTarget(offscreen.clone()));
    // The default aerosol (Bruneton 2017, τ≈0.005) is very clear air: its clear-sky anti-sun
    // L/Lz is 1.5–2.3× steeper than CIE S 011/E:2003 type 12 and Poly Haven HDRs.
    // `ATMOSPHERE_AEROSOL=urban` starts with `Aerosol::URBAN`.
    let aerosol = match std::env::var("ATMOSPHERE_AEROSOL").as_deref() {
        Ok("urban") => Aerosol::URBAN,
        _ => Aerosol::CONTINENTAL_CLEAN,
    };
    let earth_medium =
        scattering_mediums.add(ScatteringMedium::earth_with_aerosol(256, 256, aerosol));
    let mars_phase = asset_server.load("textures/mars_mie_phase.ktx2");
    let mars_medium = scattering_mediums.add(ScatteringMedium::mars(256, 256, mars_phase));

    commands.insert_resource(AtmospherePresets {
        earth: earth_medium.clone(),
        mars: mars_medium.clone(),
    });

    // Spawn earth atmosphere
    commands.spawn((
        // Ground albedo 0.28 to match the cloud ground bounce (CLOUD_GROUND_ALBEDO) and the reference.
        Atmosphere {
            ground_albedo: Vec3::splat(0.28),
            ..Atmosphere::earth(earth_medium)
        },
        // Volumetric cloud layer. Scattering/absorption come from droplet microphysics
        // (`VolumetricCloudLayer::with_liquid_water`): peak cumulus liquid water content 1 g/m^3 at density 1,
        // effective radius 10 um.
        VolumetricCloudLayer {
            cloud_layer_start: 6_361_500.0, // cloud base 1.5 km above the surface
            cloud_layer_end: 6_364_000.0,   // tops 4 km (2.5 km thick layer)
            // Used for enabling/disabling, actual density comes from noise (normalized [0, 1]);
            // `ATMOSPHERE_NO_CUMULUS` leaves only the cirrus (its history moves with its own wind).
            // `ATMOSPHERE_CLOUD_DENSITY=<x>` scales the optical depth (debug).
            cloud_density: if std::env::var("ATMOSPHERE_NO_CUMULUS").is_ok() {
                0.0
            } else {
                std::env::var("ATMOSPHERE_CLOUD_DENSITY").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0)
            },
            // Larger scale = larger cloud features (more “big cumulus”, less “small puffs”)
            noise_scale: 32_000.0, // weather tile (shape noise: shape_noise_scale, 6.4 km)
            noise_offset: Vec3::ZERO,
            // Detail noise tile; `ATMOSPHERE_CLOUD_DETAIL_SCALE=<m>` overrides (validation).
            detail_noise_scale: std::env::var("ATMOSPHERE_CLOUD_DETAIL_SCALE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(600.0),
            // `ATMOSPHERE_CLOUD_LIFETIME=<s>` sets the cloud life time (0 = static weather).
            lifetime: env_f32("ATMOSPHERE_CLOUD_LIFETIME").unwrap_or(1200.0),
            // `ATMOSPHERE_CLOUD_EVOLUTION=<s>` starts the cloud clock (life-cycle phase) there.
            evolution_time: env_f32("ATMOSPHERE_CLOUD_EVOLUTION").unwrap_or(0.0),
            // `ATMOSPHERE_CLOUD_COVERAGE=<x>` sets the starting coverage scale (Left/Right change it).
            coverage: std::env::var("ATMOSPHERE_CLOUD_COVERAGE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0),
            // `ATMOSPHERE_CLOUD_DETAIL=<x>` overrides the detail-erosion strength (validation).
            detail_strength: std::env::var("ATMOSPHERE_CLOUD_DETAIL")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0),
            ..default()
        }
        .with_liquid_water(
            std::env::var("ATMOSPHERE_CLOUD_LWC").ok().and_then(|v| v.parse::<f32>().ok()).map_or(VolumetricCloudLayer::CUMULUS_LIQUID_WATER, |g| g * 1e-3),
            VolumetricCloudLayer::CUMULUS_EFFECTIVE_RADIUS,
        ),
        // Vertical cloud profiles: physically based by default (baked parcel ensemble);
        // `ATMOSPHERE_CLOUD_PROFILE=nubis` selects Nubis's vertical profile, `adiabatic` or
        // `nubis_envelope` an analytic envelope.
        match std::env::var("ATMOSPHERE_CLOUD_PROFILE").as_deref() {
            Ok("nubis") => CloudProfiles::Nubis,
            Ok("adiabatic") => CloudProfiles::ADIABATIC,
            Ok("nubis_envelope") => CloudProfiles::NUBIS_ENVELOPE,
            _ => CloudProfiles::Physical,
        },
        CloudWeather::default(),
        // High-altitude cirrus (thin ice cloud at 9 km); `ATMOSPHERE_NO_CIRRUS` removes it.
        ThinCloudLayer {
            optical_depth: if std::env::var("ATMOSPHERE_NO_CIRRUS").is_ok() { 0.0 } else { 0.3 },
            ..ThinCloudLayer::cirrus()
        },
        // One wind profile for both layers (each takes its wind and shear at its altitude).
        // `ATMOSPHERE_WIND=slow`: 2 m/s everywhere (sub-texel shadow motion per frame);
        // `opposing`: cumulus and cirrus winds in opposite directions (per-layer reprojection).
        match std::env::var("ATMOSPHERE_WIND").as_deref() {
            Ok("slow") => AtmosphereWind::constant(Vec3::X * 2.0),
            Ok("opposing") => AtmosphereWind {
                levels: vec![(0.0, Vec3::X * 10.0), (4000.0, Vec3::X * 15.0), (8000.0, Vec3::X * -30.0)],
            },
            _ => AtmosphereWind::fair_weather(),
        },
    ))
    // Further layers are entities related to the atmosphere (`CloudLayerOf`):
    // `ATMOSPHERE_SECOND_THIN=1` adds a cirrostratus veil at 11 km above the cirrus.
    .with_related_entities::<CloudLayerOf>(|layers| {
        if std::env::var("ATMOSPHERE_SECOND_THIN").is_ok() {
            layers.spawn(ThinCloudLayer {
                top: 6_360_000.0 + 11_500.0,
                ..ThinCloudLayer::cirrostratus()
            });
        }
    });

    // World-anchored weather edits (`CloudInfluence`): `ATMOSPHERE_INFLUENCE=hole` clears a hole
    // 6 km across, 6 km ahead of the default camera (the weather drifts, the hole stays).
    // `ATMOSPHERE_INFLUENCE=coast`: a coastline 8 km ahead, maritime air (few, large droplets:
    // N_d 80 cm^-3, more translucent clouds) before it, continental (N_d 800) beyond.
    if std::env::var("ATMOSPHERE_INFLUENCE").as_deref() == Ok("coast") {
        for (x, droplet_number) in [(-42_000.0, 80.0), (58_000.0, 800.0)] {
            commands.spawn((
                Transform::from_xyz(x, 0.0, 0.0),
                CloudInfluence {
                    shape: CloudInfluenceShape::Rectangle(Vec2::splat(50_000.0)),
                    feather: 1_500.0,
                    values: CloudInfluenceValues { droplet_number: Some(droplet_number), ..default() },
                    ..default()
                },
            ));
        }
    }
    if std::env::var("ATMOSPHERE_INFLUENCE").as_deref() == Ok("hole") {
        commands.spawn((
            Transform::from_xyz(6_000.0, 0.0, 0.0),
            CloudInfluence {
                shape: CloudInfluenceShape::Ellipse(Vec2::splat(3_000.0)),
                feather: 800.0,
                blend: CloudBlend::Subtract,
                values: CloudInfluenceValues { coverage: Some(1.0), ..default() },
                ..default()
            },
        ));
    }

    let mut camera = commands.spawn_empty();
    if let Some(image) = offscreen {
        camera.insert(bevy::camera::RenderTarget::Image(image.into()));
    }
    camera.insert((
        Camera3d::default(),
        // `ATMOSPHERE_VIEW=up|wide` picks an alternative camera for cloud screenshots.
        // `ATMOSPHERE_CAM=x,y,z,tx,ty,tz` overrides the camera position and look-at target.
        match std::env::var("ATMOSPHERE_CAM")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.trim().parse::<f32>().ok()).collect::<Vec<_>>())
            .filter(|v| v.len() == 6)
        {
            Some(v) => Transform::from_xyz(v[0], v[1], v[2])
                .looking_at(Vec3::new(v[3], v[4], v[5]), Vec3::Y),
            None => match std::env::var("ATMOSPHERE_VIEW").as_deref() {
            Ok("up") => Transform::from_xyz(-2.8, 0.045, 0.0)
                .looking_at(Vec3::new(-1.8, 1.2, 0.3), Vec3::Y),
            // Toward the sun (+X), tilted up; use with ATMOSPHERE_SUNSET=1 for a low sun.
            Ok("sun") => Transform::from_xyz(-2.8, 0.045, 0.0)
                .looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
            // Sunset undersides: steeply up toward the sun azimuth (cloud bases over the camera).
            Ok("under") => Transform::from_xyz(-2.8, 0.045, 0.0)
                .looking_at(Vec3::new(-2.2, 1.6, 0.0), Vec3::Y),
            // Inside the cloud slab (2.6 km; world units are metres) looking along it: near-field / step-budget clipping.
            Ok("inslab") => Transform::from_xyz(-2.8, 2600.0, 0.0)
                .looking_at(Vec3::new(5000.0, 2700.0, 0.0), Vec3::Y),
            // Grazing toward the horizon from the ground: far-distance precision / convergence.
            Ok("horizon") => Transform::from_xyz(-2.8, 0.045, 0.0)
                .looking_at(Vec3::new(5.0, 0.12, 0.0), Vec3::Y),
            // Down at the ground from 1 km: cloud shadow map stability.
            Ok("ground") => Transform::from_xyz(-2.8, 1000.0, 0.0)
                .looking_at(Vec3::new(0.0, 0.0, 1500.0), Vec3::Y),
            // Sun behind a cumulus top (silver lining, crepuscular rays), framed like a photo
            // reference. Aimed at the sun for ATMOSPHERE_SUN_ELEV_DEG=20; the camera position was
            // picked by a grid search for an occluded sun pixel.
            Ok("backlit") => Transform::from_xyz(-6000.0, 2.0, 9000.0).looking_at(
                Vec3::new(-6000.0 + 20f32.to_radians().cos(), 2.0 + 20f32.to_radians().sin(), 9000.0),
                Vec3::Y,
            ),
            Ok("wide") => Transform::from_xyz(-2.8, 0.045, 0.0)
                .looking_at(Vec3::new(-2.8, 0.25, -3.0), Vec3::Y),
            // Default ("hero"): eye height, pitched ~15 deg up so the sky fills most of the frame.
            _ => Transform::from_xyz(-2.8, 1.8, 0.0).looking_at(Vec3::new(0.0, 2.55, 0.0), Vec3::Y),
            },
        },
        // Can be adjusted to change the scene scale and rendering quality
        AtmosphereSettings {
            // Profiling override: `ATMOSPHERE_LUT_MODE=1` starts in `AtmosphereMode::LookupTexture`.
            rendering_method: if std::env::var("ATMOSPHERE_LUT_MODE").is_ok() {
                AtmosphereMode::LookupTexture
            } else {
                AtmosphereMode::Raymarched
            },
            // Profiling override: `ATMOSPHERE_SHADOW_SAMPLES=<n>` density samples per shadow ray.
            cloud_shadow_map_samples: std::env::var("ATMOSPHERE_SHADOW_SAMPLES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(AtmosphereSettings::default().cloud_shadow_map_samples),
            ..default()
        },
        // The directional light illuminance used in this scene
        // (the one recommended for use with this feature) is
        // quite bright, so raising the exposure compensation helps
        // bring the scene to a nicer brightness range.
        Exposure { ev100: 13.0 },
        // Tonemapper chosen just because it looked good with the scene, any
        // tonemapper would be fine :)
        Tonemapping::AcesFitted,
        // Bloom gives the sun a much more natural look.
        Bloom::NATURAL,
        // Left-drag looks around; right-drag is the time-of-day slider (`sun_drag`).
        #[cfg(feature = "free_camera")]
        FreeCamera {
            mouse_key_cursor_grab: MouseButton::Left,
            ..default()
        },
        // Planet-scale flight: 100x the default walk/run speed.
        #[cfg(feature = "free_camera")]
        {
            // `FreeCameraState` has private fields, so no struct-update syntax.
            let mut state = FreeCameraState::default();
            state.speed_multiplier = 100.0;
            state
        },
        VolumetricFog {
            ambient_intensity: 0.0,
            ..default()
        },
        // `ATMOSPHERE_MSAA=2|4|8` (the clouds trace multisampled depth; the sky marches at full
        // resolution instead of the half-res haze prepass).
        match std::env::var("ATMOSPHERE_MSAA").as_deref() {
            Ok("2") => Msaa::Sample2,
            Ok("4") => Msaa::Sample4,
            Ok("8") => Msaa::Sample8,
            _ => Msaa::Off,
        },
        // `VolumetricCloudSettings` owns the cloud quality and the shadow update/temporal
        // settings (it overwrites `AtmosphereSettings` every frame), so the overrides go here:
        // `ATMOSPHERE_CLOUD_QUALITY=low|medium|high|cinematic` picks the preset,
        // `ATMOSPHERE_SHADOW_INTERLEAVE=1` traces the whole shadow map every frame instead of
        // time-slicing it, `ATMOSPHERE_SHADOW_TEMPORAL_ALPHA=<a>` sets the history blend
        // (1 = no history), `ATMOSPHERE_CLOUD_STEP_SAMPLES=<n>` the density samples per long
        // in-cloud view step.
        {
            let preset = match std::env::var("ATMOSPHERE_CLOUD_QUALITY").as_deref() {
                Ok("low") => VolumetricCloudSettings::LOW,
                Ok("high") => VolumetricCloudSettings::HIGH,
                Ok("cinematic") => VolumetricCloudSettings::CINEMATIC,
                _ => VolumetricCloudSettings::MEDIUM,
            };
            VolumetricCloudSettings {
                shadow_update_interleave: std::env::var("ATMOSPHERE_SHADOW_INTERLEAVE")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(preset.shadow_update_interleave),
                shadow_temporal_alpha: std::env::var("ATMOSPHERE_SHADOW_TEMPORAL_ALPHA")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(preset.shadow_temporal_alpha),
                step_samples: std::env::var("ATMOSPHERE_CLOUD_STEP_SAMPLES")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(preset.step_samples),
                // `ATMOSPHERE_CLOUD_MAX_RENDER=w,h`: the cloud/haze buffer size cap.
                max_render_size: std::env::var("ATMOSPHERE_CLOUD_MAX_RENDER")
                    .ok()
                    .and_then(|v| {
                        let (w, h) = v.split_once(',')?;
                        Some(UVec2::new(w.trim().parse().ok()?, h.trim().parse().ok()?))
                    })
                    .unwrap_or(preset.max_render_size),
                ..preset
            }
        },
        // ScreenSpaceReflections {
        //     min_perceptual_roughness: 0.0..0.0,
        //     ..default()
        // },
    ));
    // `ATMOSPHERE_VIEWPORT=x,y,w,h`: render the camera into that physical-pixel sub-rectangle of
    // its target (split-screen / inset cameras).
    if let Some(v) = std::env::var("ATMOSPHERE_VIEWPORT")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse::<u32>().ok()).collect::<Vec<_>>())
        .filter(|v| v.len() == 4)
    {
        camera.insert(Camera {
            viewport: Some(bevy::camera::Viewport {
                physical_position: UVec2::new(v[0], v[1]),
                physical_size: UVec2::new(v[2], v[3]),
                ..default()
            }),
            ..default()
        });
    }
    // `ATMOSPHERE_NO_TAA`: no TAA (and so no jitter), to look at the clouds' and haze's own
    // temporal filters alone.
    if std::env::var("ATMOSPHERE_NO_TAA").is_err() {
        camera.insert(TemporalAntiAliasing::default());
    }
    // `ATMOSPHERE_IBL=1`: the atmosphere drives reflections and ambient lighting (IBL) for this
    // view (off by default while debugging the clouds).
    let ibl = std::env::var("ATMOSPHERE_IBL").is_ok();
    if ibl {
        camera.insert(AtmosphereEnvironmentMapLight::default());
    }
    // `ATMOSPHERE_SECOND_CAMERA=x,y,z,tx,ty,tz`: a second camera with the same settings, into an
    // offscreen image of the window's size (equal-sized views), to test per-view cloud
    // state. `ATMOSPHERE_SECOND_SCREENSHOT=<path>` saves its image with the main screenshot.
    let main_camera = camera.id();
    if let Some(v) = std::env::var("ATMOSPHERE_SECOND_CAMERA")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse::<f32>().ok()).collect::<Vec<_>>())
        .filter(|v| v.len() == 6)
    {
        let image = images.add(Image::new_target_texture(
            target_w,
            target_h,
            bevy::render::render_resource::TextureFormat::Rgba8Unorm,
            Some(bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb),
        ));
        commands.insert_resource(SecondCameraTarget(image.clone()));
        commands
            .entity(main_camera)
            .clone_and_spawn_with_opt_out(|builder| {
                // Its own environment map (inserted below), not the main camera's image.
                builder.deny::<(
                    AtmosphereEnvironmentMapLight,
                    bevy::light::GeneratedEnvironmentMapLight,
                    bevy::light::EnvironmentMapLight,
                )>();
                #[cfg(feature = "free_camera")]
                builder.deny::<(FreeCamera, FreeCameraState)>();
            })
            .insert_if(AtmosphereEnvironmentMapLight::default(), || ibl)
            .insert((
                Transform::from_xyz(v[0], v[1], v[2]).looking_at(Vec3::new(v[3], v[4], v[5]), Vec3::Y),
                bevy::camera::RenderTarget::Image(image.into()),
                Camera {
                    order: 1,
                    ..default()
                },
            ));
    }
}

/// The second camera's image (`ATMOSPHERE_SECOND_CAMERA`).
#[derive(Resource)]
struct SecondCameraTarget(Handle<Image>);

#[derive(Component)]
struct Terrain;

#[derive(Component)]
struct WaterPlane;

/// A custom [`ExtendedMaterial`] that creates animated water ripples.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct Water {
    /// The normal map image.
    ///
    /// Note that, like all normal maps, this must not be loaded as sRGB.
    #[texture(100)]
    #[sampler(101)]
    normals: Handle<Image>,

    // Parameters to the water shader.
    #[uniform(102)]
    settings: WaterSettings,
}

/// Parameters to the water shader.
#[derive(ShaderType, Debug, Clone)]
struct WaterSettings {
    /// How much to displace each octave each frame, in the u and v directions.
    /// Two octaves are packed into each `vec4`.
    octave_vectors: [Vec4; 2],
    /// How wide the waves are in each octave.
    octave_scales: Vec4,
    /// How high the waves are in each octave.
    octave_strengths: Vec4,
}

impl MaterialExtension for Water {
    fn deferred_fragment_shader() -> ShaderRef {
        "shaders/water_material.wesl".into()
    }
}

fn setup_terrain_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut water_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, Water>>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // Sun
    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            // lux::RAW_SUNLIGHT is recommended for use with this feature, since
            // other values approximate sunlight *post-scattering* in various
            // conditions. RAW_SUNLIGHT in comparison is the illuminance of the
            // sun unfiltered by the atmosphere, so it is the proper input for
            // sunlight to be filtered by the atmosphere.
            // `ATMOSPHERE_SUN_SCALE=<x>` scales the sun's illuminance.
            illuminance: lux::RAW_SUNLIGHT * env_f32("ATMOSPHERE_SUN_SCALE").unwrap_or(1.0),
            ..default()
        },
        // `ATMOSPHERE_SUN_ELEV_DEG=<deg>` fixes the sun elevation (overrides ATMOSPHERE_SUNSET),
        // `ATMOSPHERE_SUN_AZ_DEG=<deg>` turns it about the vertical from +X toward -Z.
        match env_f32("ATMOSPHERE_SUN_ELEV_DEG") {
            Some(deg) => Transform::from_translation(sun_direction(
                deg,
                env_f32("ATMOSPHERE_SUN_AZ_DEG").unwrap_or(0.0),
            )),
            None => Transform::from_xyz(
                1.0,
                if std::env::var("ATMOSPHERE_SUNSET").is_ok() { 0.06 } else { 0.4 },
                0.0,
            ),
        }
        .looking_at(Vec3::ZERO, Vec3::Y),
        VolumetricLight,
        SunDisk::EARTH,
    ));
    // `ATMOSPHERE_SECOND_SUN=<elev_deg>,<az_deg>`: a second sun (no shadow maps, not volumetric,
    // so it sorts after the first and the cloud shadow map follows the first), to test
    // multiple-light cloud shadows.
    if let Some(v) = std::env::var("ATMOSPHERE_SECOND_SUN")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse::<f32>().ok()).collect::<Vec<_>>())
        .filter(|v| v.len() == 2)
    {
        commands.spawn((
            DirectionalLight {
                illuminance: lux::RAW_SUNLIGHT,
                ..default()
            },
            Transform::from_translation(sun_direction(v[0], v[1])).looking_at(Vec3::ZERO, Vec3::Y),
        ));
    }

    // // spawn the fog volume
    // commands.spawn((
    //     FogVolume::default(),
    //     Transform::from_scale(Vec3::new(10.0, 1.0, 10.0)).with_translation(Vec3::Y * 0.5),
    // ));

    // `ATMOSPHERE_MESH_PLANE=1`: a 40 km grey mesh plane over x < 0, 2 m above the ground, to
    // compare mesh (PBR) cloud shadows with the planet ground's across its edge at x = 0.
    if std::env::var("ATMOSPHERE_MESH_PLANE").is_ok() {
        commands.spawn((
            Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(20_000.0)))),
            MeshMaterial3d(std_materials.add(StandardMaterial {
                base_color: Color::srgb(0.3, 0.3, 0.3),
                perceptual_roughness: 1.0,
                // The example's default is deferred; the camera has no deferred prepass.
                opaque_render_method: bevy::material::OpaqueRendererMethod::Forward,
                ..default()
            })),
            Transform::from_xyz(-20_000.0, 2.0, 0.0),
        ));
    }

    // Terrain: temporarily off by default while tuning clouds + atmosphere (the planet ground
    // sphere still renders); `ATMOSPHERE_TERRAIN=1` brings it back.
    if std::env::var("ATMOSPHERE_TERRAIN").is_err() {
        return;
    }
    commands.spawn((
        Terrain,
        WorldAssetRoot(
            asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/terrain/terrain.glb")),
        ),
        Transform::from_xyz(-1.0, 0.0, -0.5)
            .with_scale(Vec3::splat(0.5))
            .with_rotation(Quat::from_rotation_y(PI / 2.0)),
    ));

    // spawn_water(
    //     &mut commands,
    //     &asset_server,
    //     &mut meshes,
    //     &mut water_materials,
    // );
}

// Spawns the water plane.
fn spawn_water(
    commands: &mut Commands,
    asset_server: &AssetServer,
    meshes: &mut Assets<Mesh>,
    water_materials: &mut Assets<ExtendedMaterial<StandardMaterial, Water>>,
) {
    commands.spawn((
        WaterPlane,
        Visibility::default(),
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(1.0)))),
        MeshMaterial3d(
            water_materials.add(ExtendedMaterial {
                base: StandardMaterial {
                    base_color: BLACK.into(),
                    perceptual_roughness: 0.0,
                    ..default()
                },
                extension: Water {
                    normals: asset_server
                        .load_builder()
                        .with_settings(|settings: &mut ImageLoaderSettings| {
                            settings.is_srgb = false;
                            settings.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                                address_mode_u: ImageAddressMode::Repeat,
                                address_mode_v: ImageAddressMode::Repeat,
                                mag_filter: ImageFilterMode::Linear,
                                min_filter: ImageFilterMode::Linear,
                                ..default()
                            });
                        })
                        .load("textures/water_normals.png"),
                    // These water settings are just random values to create some
                    // variety.
                    settings: WaterSettings {
                        octave_vectors: [
                            vec4(0.080, 0.059, 0.073, -0.062),
                            vec4(0.153, 0.138, -0.149, -0.195),
                        ],
                        octave_scales: vec4(1.0, 2.1, 7.9, 14.9) * 500.0,
                        octave_strengths: vec4(0.16, 0.18, 0.093, 0.044) * 0.2,
                    },
                },
            }),
        ),
        Transform::from_scale(Vec3::splat(100.0)),
    ));
}

/// Puts the camera position (whole numbers) and facing direction in the window title.
fn camera_pose_title(
    cameras: Query<&Transform, With<Camera3d>>,
    mut windows: Query<&mut Window, With<bevy::window::PrimaryWindow>>,
) {
    let Ok(transform) = cameras.single() else {
        return;
    };
    let Ok(mut window) = windows.single_mut() else {
        return;
    };
    let p = transform.translation;
    let f = transform.forward();
    let title = format!(
        "{}, {}, {}  facing {:.2}, {:.2}, {:.2}",
        p.x.round() as i32,
        p.y.round() as i32,
        p.z.round() as i32,
        f.x,
        f.y,
        f.z,
    );
    if window.title != title {
        window.title = title;
    }
}

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// Direction toward a sun at `elev_deg` above the horizon, turned `az_deg` about the vertical
/// from +X toward -Z.
fn sun_direction(elev_deg: f32, az_deg: f32) -> Vec3 {
    let (e, a) = (elev_deg.to_radians(), az_deg.to_radians());
    Vec3::new(e.cos() * a.cos(), e.sin(), -e.cos() * a.sin())
}

fn dynamic_scene(
    mut suns: Query<&mut Transform, With<DirectionalLight>>,
    time: Res<Time>,
    sun_motion_state: Res<GameState>,
) {
    // Only rotate the sun if motion is not paused
    if !sun_motion_state.paused {
        suns.iter_mut()
            .for_each(|mut tf| tf.rotate_x(-time.delta_secs() * PI / 40.0));
    }
}

/// `ATMOSPHERE_SUN_SWEEP=<deg per frame>,<first frame>,<last frame>`: swings the sun along its
/// arc (as right-drag) by a fixed angle per frame, independent of time, so fast lighting changes
/// are reproducible in background runs and dumps.
fn sun_sweep(mut frame: Local<u32>, mut suns: Query<&mut Transform, With<DirectionalLight>>) {
    let Some(v) = std::env::var("ATMOSPHERE_SUN_SWEEP").ok() else {
        return;
    };
    let p: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    let n = *frame;
    *frame += 1;
    if p.len() != 3 || (n as f32) < p[1] || (n as f32) > p[2] {
        return;
    }
    let rotation = Quat::from_rotation_z(-p[0].to_radians());
    for mut tf in &mut suns {
        tf.rotation = rotation * tf.rotation;
    }
}

/// Right-drag slides the time of day: horizontal mouse motion swings the sun along its arc
/// (rotation about world Z, the same plane as `ATMOSPHERE_SUN_ELEV_DEG`), 0.15 deg per pixel.
fn sun_drag(
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<bevy::input::mouse::AccumulatedMouseMotion>,
    mut suns: Query<&mut Transform, With<DirectionalLight>>,
) {
    if !buttons.pressed(MouseButton::Right) || motion.delta.x == 0.0 {
        return;
    }
    let rotation = Quat::from_rotation_z(-motion.delta.x * 0.15f32.to_radians());
    for mut tf in &mut suns {
        tf.rotation = rotation * tf.rotation;
    }
}

/// Offscreen render target used by `ATMOSPHERE_BACKGROUND=1` runs.
#[derive(Resource)]
struct OffscreenTarget(Option<Handle<Image>>);

/// Window (and offscreen target) size in physical pixels: 1280x720 so dumps match across displays,
/// or `ATMOSPHERE_WINDOW=w,h`.
fn window_size() -> (u32, u32) {
    std::env::var("ATMOSPHERE_WINDOW")
        .ok()
        .and_then(|v| {
            let (w, h) = v.split_once(',')?;
            Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
        })
        .unwrap_or((1280, 720))
}
