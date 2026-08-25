//! Headless repro for the interleaved HDR/SDR shared-target trails report.
//!
//! One `Rgba8UnormSrgb` image target, two `Camera3d` views with MSAA 4x:
//! an SDR camera at order 0 (left viewport) and an `Hdr` camera at order 1
//! (right viewport). With per-target camera indices, the HDR camera has
//! `sorted_camera_index_for_target == 1`, so `MsaaWriteback::Auto` enables
//! writeback. Its writeback source is the Rgba16Float main texture, which no
//! camera wrote this frame (the SDR camera uses the Rgba8 texture), so the
//! HDR view starts each frame from last frame's content instead of its clear
//! color.
//!
//! The test clears the HDR camera to red, later switches the clear to blue,
//! and captures the target after each phase. Broken: the right half never
//! shows either clear color (it stays at the stale/zero-initialized buffer).
//! Fixed: the right half tracks the current clear color.
//!
//! Run:
//! ```sh
//! VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json WGPU_BACKEND=vulkan \
//!   cargo test --test gpu_mixed_hdr_repro -- --nocapture
//! ```

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Mutex};
use std::time::Duration;

use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    audio::AudioPlugin,
    camera::{Camera3d, ClearColorConfig, Hdr, RenderTarget, Viewport},
    prelude::*,
    render::{render_resource::TextureFormat, RenderApp},
    window::ExitCondition,
    winit::WinitPlugin,
};

const SIZE: u32 = 64;
const CAPTURE_1: u32 = 40;
const SWITCH_AT: u32 = 60;
const CAPTURE_2: u32 = 100;
const MAX_FRAMES: u32 = 2000;

static CAPTURES: Mutex<Vec<(String, Vec<u8>, u32)>> = Mutex::new(Vec::new());

#[derive(Resource, Default)]
struct Frame(u32);

#[derive(Resource)]
struct TargetHandle(Handle<Image>);

#[derive(Component)]
struct HdrCam;

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let img = Image::new_target_texture(SIZE, SIZE, TextureFormat::Rgba8UnormSrgb, None);
    let handle = images.add(img);

    // SDR camera, bottom of the target: left half, green clear.
    commands.spawn((
        Camera3d::default(),
        RenderTarget::Image(handle.clone().into()),
        Camera {
            order: 0,
            clear_color: ClearColorConfig::Custom(Color::linear_rgb(0.0, 1.0, 0.0)),
            viewport: Some(Viewport {
                physical_position: UVec2::new(0, 0),
                physical_size: UVec2::new(SIZE / 2, SIZE),
                ..default()
            }),
            ..default()
        },
        Msaa::Sample4,
        Transform::from_xyz(0.0, 0.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // HDR camera, top of the target: right half, red clear (switched to blue later).
    commands.spawn((
        Camera3d::default(),
        Hdr,
        HdrCam,
        RenderTarget::Image(handle.clone().into()),
        Camera {
            order: 1,
            clear_color: ClearColorConfig::Custom(Color::linear_rgb(1.0, 0.0, 0.0)),
            viewport: Some(Viewport {
                physical_position: UVec2::new(SIZE / 2, 0),
                physical_size: UVec2::new(SIZE / 2, SIZE),
                ..default()
            }),
            ..default()
        },
        Msaa::Sample4,
        Transform::from_xyz(0.0, 0.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.insert_resource(TargetHandle(handle));
}

fn capture(commands: &mut Commands, handle: &Handle<Image>, label: &'static str) {
    let mut entity = commands.spawn(Screenshot::image(handle.clone()));
    entity.observe(move |captured: On<ScreenshotCaptured>| {
        let image = &captured.image;
        CAPTURES.lock().unwrap().push((
            label.to_string(),
            image.data.clone().unwrap_or_default(),
            image.texture_descriptor.size.width,
        ));
    });
}

fn driver(
    mut frame: ResMut<Frame>,
    handle: Res<TargetHandle>,
    mut commands: Commands,
    mut hdr_cam: Query<&mut Camera, With<HdrCam>>,
    mut exit: MessageWriter<AppExit>,
) {
    frame.0 += 1;
    match frame.0 {
        CAPTURE_1 => capture(&mut commands, &handle.0, "phase_red"),
        SWITCH_AT => {
            for mut cam in &mut hdr_cam {
                cam.clear_color = ClearColorConfig::Custom(Color::linear_rgb(0.0, 0.0, 1.0));
            }
        }
        CAPTURE_2 => capture(&mut commands, &handle.0, "phase_blue"),
        _ => {}
    }

    if CAPTURES.lock().unwrap().len() >= 2 {
        exit.write(AppExit::Success);
    } else if frame.0 > MAX_FRAMES {
        eprintln!("TIMEOUT waiting for captures");
        exit.write(AppExit::error());
    }
}

fn pixel(data: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * width + x) * 4) as usize;
    [data[i], data[i + 1], data[i + 2], data[i + 3]]
}

#[test]
fn gpu_mixed_hdr_repro() {
    let (tx, rx) = mpsc::channel::<Result<(), String>>();
    let handle = std::thread::spawn(move || {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut app = App::new();
            app.add_plugins(
                DefaultPlugins
                    .set(WindowPlugin {
                        primary_window: None,
                        exit_condition: ExitCondition::DontExit,
                        ..default()
                    })
                    .disable::<WinitPlugin>()
                    .disable::<AudioPlugin>(),
            )
            .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::ZERO))
            .init_resource::<Frame>()
            .add_systems(Startup, setup)
            .add_systems(Update, driver);
            if app.get_sub_app_mut(RenderApp).is_none() {
                tx.send(Err("no render sub-app".into())).ok();
                return;
            }
            let exit = app.run();
            tx.send(if exit == AppExit::Success {
                Ok(())
            } else {
                Err(format!("app exited with {exit:?}"))
            })
            .ok();
        }));
        if let Err(payload) = result {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic".into());
            eprintln!("app thread panicked: {msg}");
        }
    });

    match rx.recv_timeout(Duration::from_secs(480)) {
        Ok(Ok(())) => {}
        Ok(Err(msg)) => {
            eprintln!("SKIP: {msg}");
            return;
        }
        Err(_) => panic!("timed out waiting for the app"),
    }
    handle.join().ok();

    let captures = CAPTURES.lock().unwrap();
    let mut failures = Vec::new();
    for (label, data, width) in captures.iter() {
        let sdr = pixel(data, *width, SIZE / 4, SIZE / 2);
        let hdr = pixel(data, *width, 3 * SIZE / 4, SIZE / 2);
        println!("{label}: sdr(left)={sdr:?} hdr(right)={hdr:?}");
        if !(sdr[1] > 128 && sdr[0] < 64 && sdr[2] < 64) {
            failures.push(format!("{label}: SDR half is not green: {sdr:?}"));
        }
        let (want, chan) = match label.as_str() {
            "phase_red" => ("red", 0usize),
            _ => ("blue", 2usize),
        };
        let other: u8 = (0..3usize)
            .filter(|c| *c != chan)
            .map(|c| hdr[c])
            .max()
            .unwrap();
        if !(hdr[chan] > 100 && hdr[chan] > other.saturating_add(40)) {
            failures.push(format!(
                "{label}: HDR half is not {want}-dominant: {hdr:?} (stale/never-cleared buffer?)"
            ));
        }
    }
    assert!(
        captures.len() == 2 && failures.is_empty(),
        "mixed HDR/SDR shared-target repro failed:\n{}",
        failures.join("\n")
    );
}
