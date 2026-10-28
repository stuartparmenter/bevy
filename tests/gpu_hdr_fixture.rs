//! Renders HDR display output headlessly and checks the decoded screenshots.
//!
//! Run it on a software Vulkan driver (lavapipe), with no display server:
//!
//! ```sh
//! VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json WGPU_BACKEND=vulkan \
//!   cargo test --test gpu_hdr_fixture -- --nocapture
//! ```
//!
//! Without a usable wgpu adapter, the test prints a `SKIP` line and passes. With
//! `--features bevy_solari`, it also renders Solari scenes.
//!
//! Each camera renders to an offscreen image with a manual display target. One app renders
//! in the Rec.709 working color space and one in Rec.2020. The checks compare the captures
//! with CPU copies of the display encoding, gamut compression and GT7, or the Rec.2020
//! captures with the Rec.709 ones. They also cover the HDR PNG, anti-aliasing and bloom on
//! HDR targets, and auto exposure and auto white balance.

use std::collections::HashMap;
use std::io::Read;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Mutex};
use std::time::Duration;

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    asset::RenderAssetUsages,
    audio::AudioPlugin,
    camera::{
        CameraOutputMode, ClearColorConfig, CompositingSpace, Exposure, MsaaWriteback,
        RenderTarget, Viewport,
    },
    color::RgbPrimaries,
    core_pipeline::tonemapping::{DebandDither, Tonemapping},
    image::{CompressedImageFormats, ImageSampler, ImageType, SourceColorPrimaries},
    log::LogPlugin,
    math::{ops, UVec2, Vec3},
    post_process::auto_exposure::{
        AutoExposure, AutoExposurePlugin, AutoExposureState, AutoWhiteBalance,
        PhysiologicalAdaptation,
    },
    prelude::*,
    render::{
        render_resource::TextureFormat,
        transfer_functions::{pq_inverse_eotf_from_nits, srgb_oetf_extended},
        view::{
            screenshot::{save_to_disk, Screenshot, ScreenshotCaptured},
            ManualDisplayTargets,
        },
        RenderPlugin, WorkingColorSpace,
    },
    window::{
        DisplayTarget, ExitCondition, SurfaceColorSpace, MAX_LUMINANCE_NITS,
        SCRGB_REFERENCE_WHITE_NITS,
    },
    winit::WinitPlugin,
};

const SIZE: u32 = 64;
/// Frame for the capture round, past pipeline compilation and enough path tracing samples.
const CAPTURE_AT: u32 = 60;
const MAX_FRAMES: u32 = 2000;

/// Main texture clear for every HDR camera, in linear BT.709. Above 1.0 to use HDR range.
const MAIN_CLEAR: Vec3 = Vec3::new(2.0, 0.5, 0.05);
/// Out texture clear of the viewport camera.
const OUT_CLEAR: Vec3 = Vec3::new(0.1, 0.2, 0.3);
/// A clear with a negative channel and one in the curve's linear segment.
const SIGNED_CLEAR: Vec3 = Vec3::new(-0.25, 0.002, 1.5);
/// A clear above the 10000 nits PQ can encode at the default paper white of 203 nits.
const PQ_OVER_RANGE_CLEAR: Vec3 = Vec3::splat(60.0);
/// Main texture clear of the SDR control camera.
const SDR_CLEAR: Vec3 = Vec3::new(0.25, 0.5, 0.75);
/// A saturated clear that uses HDR range.
const SATURATED_CLEAR: Vec3 = Vec3::new(1.5, 0.0, 0.1);
/// A clear outside the Rec.709 gamut but inside Rec.2020.
const WIDE_CLEAR: Vec3 = Vec3::new(-0.05, 1.0, 0.1);

/// The case whose capture is also saved as a PNG. Its paper white is not the default.
const PNG_CASE: &str = "pq_pw100";
const PNG_PATH: &str = concat!(env!("CARGO_TARGET_TMPDIR"), "/gpu_hdr_fixture.png");
const REC2100PQ_ICC: &[u8] =
    include_bytes!("../crates/bevy_render/src/view/window/screenshot/rec2100pq.icc");

#[derive(Clone, Copy)]
struct Case {
    label: &'static str,
    color_space: Option<SurfaceColorSpace>,
    paper_white_nits: Option<f32>,
    viewport: bool,
    msaa_writeback: bool,
    /// The color of a UI node that covers the whole target.
    ui_color: Option<Vec3>,
    compositing_space: Option<CompositingSpace>,
    main_clear: Vec3,
    format: TextureFormat,
    tonemapping: Tonemapping,
}

const CASES: &[Case] = &[
    Case::hdr("pq", SurfaceColorSpace::Pq),
    Case::hdr("scrgb", SurfaceColorSpace::ScRgbLinear),
    Case::hdr("extended_p3", SurfaceColorSpace::ExtendedDisplayP3),
    Case {
        main_clear: SIGNED_CLEAR,
        ..Case::hdr("extended_srgb_signed", SurfaceColorSpace::ExtendedSrgb)
    },
    Case {
        paper_white_nits: Some(100.0),
        ..Case::hdr("pq_pw100", SurfaceColorSpace::Pq)
    },
    Case {
        format: TextureFormat::Rgb10a2Unorm,
        ..Case::hdr("pq_rgb10a2", SurfaceColorSpace::Pq)
    },
    Case {
        main_clear: PQ_OVER_RANGE_CLEAR,
        ..Case::hdr("pq_over_range", SurfaceColorSpace::Pq)
    },
    Case {
        viewport: true,
        ..Case::hdr("pq_viewport", SurfaceColorSpace::Pq)
    },
    Case {
        msaa_writeback: true,
        ..Case::hdr("pq_msaa_writeback", SurfaceColorSpace::Pq)
    },
    Case {
        compositing_space: Some(CompositingSpace::Srgb),
        ..Case::hdr("pq_srgb_compositing", SurfaceColorSpace::Pq)
    },
    Case {
        compositing_space: Some(CompositingSpace::Oklab),
        ..Case::hdr("pq_oklab_compositing", SurfaceColorSpace::Pq)
    },
    Case {
        tonemapping: Tonemapping::Linear,
        ..Case::hdr("pq_linear", SurfaceColorSpace::Pq)
    },
    Case::sdr("sdr"),
    Case {
        main_clear: SATURATED_CLEAR,
        tonemapping: Tonemapping::GranTurismo7,
        ..Case::hdr("scrgb_gt7", SurfaceColorSpace::ScRgbLinear)
    },
    Case {
        tonemapping: Tonemapping::KhronosPbrNeutral,
        ..Case::hdr("pq_sdr_only_becomes_gt7", SurfaceColorSpace::Pq)
    },
];

const REC2020_CASES: &[Case] = &[
    Case {
        main_clear: SATURATED_CLEAR,
        ..Case::hdr("rec2020_pq_none", SurfaceColorSpace::Pq)
    },
    Case {
        main_clear: SATURATED_CLEAR,
        tonemapping: Tonemapping::Linear,
        ..Case::hdr("rec2020_pq_linear", SurfaceColorSpace::Pq)
    },
    Case {
        main_clear: WIDE_CLEAR,
        tonemapping: Tonemapping::Linear,
        ..Case::hdr("rec2020_scrgb_linear_wide", SurfaceColorSpace::ScRgbLinear)
    },
    Case::sdr("rec2020_sdr_none"),
    Case {
        ui_color: Some(SATURATED_CLEAR),
        ..Case::hdr("rec2020_pq_none_ui", SurfaceColorSpace::Pq)
    },
    Case {
        main_clear: WIDE_CLEAR,
        tonemapping: Tonemapping::GranTurismo7,
        ..Case::hdr(GT7_WIDE_CASE, SurfaceColorSpace::Pq)
    },
    Case {
        main_clear: WIDE_CLEAR,
        tonemapping: Tonemapping::GranTurismo7,
        ..Case::hdr("rec2020_scrgb_gt7_wide", SurfaceColorSpace::ScRgbLinear)
    },
    Case {
        main_clear: WIDE_CLEAR,
        tonemapping: Tonemapping::GranTurismo7,
        ..Case::hdr(
            "rec2020_extended_p3_gt7_wide",
            SurfaceColorSpace::ExtendedDisplayP3,
        )
    },
    Case {
        main_clear: WIDE_CLEAR,
        ..Case::sdr("rec2020_sdr_none_wide")
    },
    // An image target makes this SDR camera without `Hdr` run the tonemapping pass.
    Case {
        main_clear: WIDE_CLEAR,
        tonemapping: Tonemapping::Linear,
        ..Case::sdr("rec2020_sdr_linear_wide")
    },
];

/// Rec.2020 cases whose GT7 output the blit compresses into the target's gamut.
const COMPRESSED_GT7_CASES: [&str; 2] = ["rec2020_scrgb_gt7_wide", "rec2020_extended_p3_gt7_wide"];

/// A Rec.2020 case whose GT7 output has a negative Rec.709 channel.
const GT7_WIDE_CASE: &str = "rec2020_pq_gt7_wide";

impl Case {
    const fn hdr(label: &'static str, color_space: SurfaceColorSpace) -> Self {
        Self {
            label,
            color_space: Some(color_space),
            paper_white_nits: None,
            viewport: false,
            msaa_writeback: false,
            ui_color: None,
            compositing_space: None,
            main_clear: MAIN_CLEAR,
            format: TextureFormat::Rgba16Float,
            tonemapping: Tonemapping::None,
        }
    }

    const fn sdr(label: &'static str) -> Self {
        Self {
            label,
            color_space: None,
            paper_white_nits: None,
            viewport: false,
            msaa_writeback: false,
            ui_color: None,
            compositing_space: None,
            main_clear: SDR_CLEAR,
            format: TextureFormat::Rgba8UnormSrgb,
            tonemapping: Tonemapping::None,
        }
    }

    fn display_target(&self) -> DisplayTarget {
        DisplayTarget {
            color_space_override: self.color_space,
            paper_white_nits: self.paper_white_nits,
            ..default()
        }
    }
}

static CAPTURES: Mutex<Vec<(&'static str, Image)>> = Mutex::new(Vec::new());

#[derive(Resource, Default)]
struct Frame(u32);

#[derive(Resource)]
struct Targets(Vec<(&'static str, Handle<Image>)>);

#[derive(Resource)]
struct Cases(&'static [Case]);

/// Captures that modules take themselves, outside [`Targets`].
#[derive(Resource, Default)]
struct ExtraCaptures(usize);

impl Cases {
    fn saves_png(&self) -> bool {
        self.0.iter().any(|case| case.label == PNG_CASE)
    }
}

fn driver(
    mut frame: ResMut<Frame>,
    cases: Res<Cases>,
    targets: Res<Targets>,
    extra: Res<ExtraCaptures>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    frame.0 += 1;
    if frame.0 == CAPTURE_AT {
        for (label, handle) in &targets.0 {
            let label = *label;
            let mut screenshot = commands.spawn(Screenshot::image(handle.clone()));
            screenshot.observe(move |captured: On<ScreenshotCaptured>| {
                CAPTURES
                    .lock()
                    .unwrap()
                    .push((label, captured.image.clone()));
            });
            if label == PNG_CASE {
                screenshot.observe(save_to_disk(PNG_PATH));
            }
        }
    }

    let captured = CAPTURES.lock().unwrap().len();
    let expected = targets.0.len() + extra.0;
    if captured >= expected && (!cases.saves_png() || png_written()) {
        exit.write(AppExit::Success);
    } else if frame.0 > MAX_FRAMES {
        eprintln!("TIMEOUT: only {captured}/{expected} captures after {MAX_FRAMES} frames");
        exit.write(AppExit::error());
    }
}

/// Whether the PNG save has finished. It runs on a background task.
fn png_written() -> bool {
    std::fs::read(PNG_PATH).is_ok_and(|bytes| bytes.ends_with(&[73, 69, 78, 68, 174, 66, 96, 130]))
}

/// The chunks of a PNG file, as name and payload.
fn png_chunks(bytes: &[u8]) -> Vec<([u8; 4], &[u8])> {
    let mut chunks = Vec::new();
    let mut rest = &bytes[8..];
    while rest.len() >= 12 {
        let len = u32::from_be_bytes(rest[..4].try_into().unwrap()) as usize;
        chunks.push((rest[4..8].try_into().unwrap(), &rest[8..8 + len]));
        rest = &rest[12 + len..];
    }
    chunks
}

fn setup(
    mut commands: Commands,
    cases: Res<Cases>,
    mut images: ResMut<Assets<Image>>,
    mut manual: ResMut<ManualDisplayTargets>,
) {
    let mut targets = Vec::new();
    for (order, case) in cases.0.iter().enumerate() {
        let handle = images.add(Image::new_target_texture(SIZE, SIZE, case.format, None));
        let render_target = RenderTarget::Image(handle.clone().into());
        if case.color_space.is_some() {
            let key = render_target
                .normalize(None)
                .expect("image render target normalizes without a window");
            manual.insert(key, case.display_target());
        }

        let mut camera = Camera {
            clear_color: ClearColorConfig::Custom(linear(case.main_clear)),
            order: order as isize * 2,
            ..default()
        };
        if case.viewport {
            camera.viewport = Some(Viewport {
                physical_position: UVec2::splat(SIZE / 4),
                physical_size: UVec2::splat(SIZE / 2),
                ..default()
            });
            camera.output_mode = CameraOutputMode::Write {
                blend_state: None,
                clear_color: ClearColorConfig::Custom(linear(OUT_CLEAR)),
            };
        }
        let components = (
            camera,
            render_target.clone(),
            case.tonemapping,
            DebandDither::Disabled,
            // Cameras share a main texture only when their MSAA matches, and the writeback
            // camera reads this camera's output from it.
            if case.msaa_writeback {
                Msaa::Sample4
            } else {
                Msaa::Off
            },
        );
        // Only 2D views encode into a compositing space.
        let camera = match case.compositing_space {
            Some(compositing_space) => commands.spawn((Camera2d, compositing_space, components)),
            None => commands.spawn((Camera3d::default(), components)),
        }
        .id();
        if let Some(ui_color) = case.ui_color {
            commands.spawn((
                Node {
                    width: percent(100),
                    height: percent(100),
                    ..default()
                },
                BackgroundColor(linear(ui_color)),
                UiTargetCamera(camera),
            ));
        }

        // A second camera that writes back the first camera's output and draws nothing over it.
        if case.msaa_writeback {
            commands.spawn((
                Camera3d::default(),
                Camera {
                    clear_color: ClearColorConfig::None,
                    msaa_writeback: MsaaWriteback::Auto,
                    order: order as isize * 2 + 1,
                    ..default()
                },
                render_target,
                Tonemapping::None,
                DebandDither::Disabled,
                Msaa::Sample4,
            ));
        }

        targets.push((case.label, handle));
    }
    commands.insert_resource(Targets(targets));
}

fn linear(color: Vec3) -> Color {
    Color::linear_rgb(color.x, color.y, color.z)
}

/// Decodes one channel of a `color_space` signal to linear light where 1.0 is paper white,
/// written out from the specifications.
fn decode_channel(signal: f32, color_space: SurfaceColorSpace, paper_white_nits: f32) -> f32 {
    let scrgb_scale = SCRGB_REFERENCE_WHITE_NITS / paper_white_nits;
    match color_space {
        SurfaceColorSpace::Srgb => unreachable!(),
        SurfaceColorSpace::ScRgbLinear => signal * scrgb_scale,
        SurfaceColorSpace::Pq => {
            let (m1, m2) = (2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0);
            let (c1, c2, c3) = (
                3424.0 / 4096.0,
                2413.0 / 4096.0 * 32.0,
                2392.0 / 4096.0 * 32.0,
            );
            let e = ops::powf(signal.max(0.0), 1.0 / m2);
            let y = ops::powf((e - c1).max(0.0) / (c2 - c3 * e), 1.0 / m1);
            y * MAX_LUMINANCE_NITS / paper_white_nits
        }
        SurfaceColorSpace::ExtendedSrgb | SurfaceColorSpace::ExtendedDisplayP3 => {
            let a = signal.abs();
            let linear = if a <= 0.04045 {
                a / 12.92
            } else {
                ops::powf((a + 0.055) / 1.055, 2.4)
            };
            linear.copysign(signal) * scrgb_scale
        }
    }
}

/// How far a decoded screenshot of `color` may be from [`decoded_color`]: one code value
/// of the target format on the encoded signal, plus one f16 step of the main texture.
/// Under Rec.2020, two more f16 steps of the Rec.2020 color.
fn tolerance(
    case: &Case,
    color: Vec3,
    color_space: SurfaceColorSpace,
    paper_white: f32,
    working_color_space: WorkingColorSpace,
) -> Vec3 {
    let to_709 = color_space.primaries().matrix_to(RgbPrimaries::BT709);
    let scrgb_scale = SCRGB_REFERENCE_WHITE_NITS / paper_white;
    let encode_channel = |linear: f32| match color_space {
        SurfaceColorSpace::Pq => pq_inverse_eotf_from_nits(linear * paper_white),
        SurfaceColorSpace::ScRgbLinear => linear / scrgb_scale,
        _ => srgb_oetf_extended(linear / scrgb_scale),
    };
    let surface = surface_color(color, color_space, paper_white);
    let signal_error = Vec3::from_array(std::array::from_fn(|i| {
        let signal = encode_channel(surface[i]);
        let step = match case.format {
            TextureFormat::Rgb10a2Unorm => 1.0 / 1023.0,
            TextureFormat::Rgba16Float => signal.abs().max(6.1e-5) * ops::exp2(-10.0),
            other => panic!("no tolerance for {other:?}"),
        };
        [signal - step, signal + step]
            .map(|s| (decode_channel(s, color_space, paper_white) - surface[i]).abs())
            .into_iter()
            .fold(0.0, f32::max)
    }));
    let working_error = match working_color_space {
        WorkingColorSpace::Rec709 => Vec3::ZERO,
        WorkingColorSpace::Rec2020 => {
            let rec2020 = RgbPrimaries::BT709.matrix_to(RgbPrimaries::BT2020) * color;
            RgbPrimaries::BT2020.matrix_to(RgbPrimaries::BT709).abs()
                * rec2020.abs()
                * ops::exp2(-9.0)
        }
    };
    to_709.abs() * signal_error + color.abs() * ops::exp2(-10.0) + working_error + 1e-5
}

fn to_709_abs(color_space: SurfaceColorSpace) -> Mat3 {
    color_space.primaries().matrix_to(RgbPrimaries::BT709).abs()
}

/// `color` in the surface primaries, clamped to what PQ can encode.
fn surface_color(color: Vec3, color_space: SurfaceColorSpace, paper_white: f32) -> Vec3 {
    let surface = RgbPrimaries::BT709.matrix_to(color_space.primaries()) * color;
    if color_space == SurfaceColorSpace::Pq {
        surface.min(Vec3::splat(MAX_LUMINANCE_NITS / paper_white))
    } else {
        surface
    }
}

/// Whether the blit compresses the colors of `case` into its target's gamut.
fn compresses(case: &Case, working_color_space: WorkingColorSpace) -> bool {
    match (case.color_space, case.tonemapping) {
        (Some(SurfaceColorSpace::Pq), _) => false,
        (None | Some(_), Tonemapping::None) | (Some(_), Tonemapping::Linear) => {
            working_color_space == WorkingColorSpace::Rec2020
        }
        (Some(_), _) => true,
        (None, _) => false,
    }
}

/// A CPU copy of the blit's gamut compression, on a color in the target's primaries.
fn gamut_compress(rgb: Vec3) -> Vec3 {
    const THRESHOLD: Vec3 = Vec3::new(0.815, 0.803, 0.880);
    const LIMIT: Vec3 = Vec3::new(1.62, 1.10, 1.13);
    const POWER: f32 = 1.2;
    let achromatic = rgb.max_element();
    if achromatic <= 0.0 {
        return rgb;
    }
    let scale = (LIMIT - THRESHOLD)
        / (((Vec3::ONE - THRESHOLD) / (LIMIT - THRESHOLD)).powf(-POWER) - 1.0).powf(1.0 / POWER);
    let distance = (Vec3::splat(achromatic) - rgb) / achromatic;
    let excess = (distance - THRESHOLD).max(Vec3::ZERO) / scale;
    let compressed = THRESHOLD + scale * excess / (1.0 + excess.powf(POWER)).powf(1.0 / POWER);
    Vec3::select(
        distance.cmplt(THRESHOLD),
        rgb,
        Vec3::splat(achromatic) - compressed * achromatic,
    )
}

/// The linear BT.709 color that `color` becomes after gamut compression for `color_space`.
fn compressed_color(color: Vec3, color_space: SurfaceColorSpace) -> Vec3 {
    let primaries = color_space.primaries();
    primaries.matrix_to(RgbPrimaries::BT709)
        * gamut_compress(RgbPrimaries::BT709.matrix_to(primaries) * color)
}

/// What a screenshot of `color` decodes to.
fn decoded_color(color: Vec3, color_space: SurfaceColorSpace, paper_white: f32) -> Vec3 {
    color_space.primaries().matrix_to(RgbPrimaries::BT709)
        * surface_color(color, color_space, paper_white)
}

fn decoded_texel(image: &Image, x: u32, y: u32) -> Vec3 {
    assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba32Float);
    let color = image.get_color_at(x, y).unwrap().to_linear();
    Vec3::new(color.red, color.green, color.blue)
}

struct Checker {
    failed: Vec<String>,
}

impl Checker {
    fn check(&mut self, name: &str, ok: bool, detail: String) {
        if ok {
            println!("PASS: {name}: {detail}");
        } else {
            println!("FAIL: {name}: {detail}");
            self.failed.push(name.to_string());
        }
    }
}

fn build_app(working_color_space: WorkingColorSpace, cases: &'static [Case]) -> App {
    let mut plugins = DefaultPlugins
        .set(WindowPlugin {
            primary_window: None,
            exit_condition: ExitCondition::DontExit,
            ..default()
        })
        .set(RenderPlugin {
            working_color_space,
            ..default()
        })
        .disable::<WinitPlugin>()
        .disable::<AudioPlugin>();
    // The first app sets the global logger.
    if working_color_space != WorkingColorSpace::Rec709 {
        plugins = plugins.disable::<LogPlugin>();
    }
    let mut app = App::new();
    app.add_plugins(plugins)
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::ZERO))
        .insert_resource(Cases(cases))
        .init_resource::<Frame>()
        .init_resource::<ExtraCaptures>()
        .add_plugins(AutoExposurePlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, driver);
    app.add_systems(Startup, atmosphere::setup.after(setup));
    app.add_systems(Startup, gt7::setup.after(setup));
    app.add_systems(Startup, aa_bloom::setup.after(setup));
    auto_exposure::add(&mut app);
    #[cfg(feature = "bevy_solari")]
    solari::add(&mut app);

    if app.get_sub_app(bevy::render::RenderApp).is_none() {
        panic!("no render sub-app (no wgpu backend available)");
    }
    app
}

enum Phase {
    InitFailed(String),
    InitOk,
    Finished(AppExit),
    RunPanicked(String),
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

const INIT_TIMEOUT: Duration = Duration::from_secs(120);
const RUN_TIMEOUT: Duration = Duration::from_secs(480);

/// Runs an app that renders `cases`, and returns its captures by label. Returns `None` when
/// the renderer can't start.
fn run_app(
    working_color_space: WorkingColorSpace,
    cases: &'static [Case],
) -> Option<HashMap<&'static str, Image>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let built = catch_unwind(AssertUnwindSafe(|| build_app(working_color_space, cases)));
        std::panic::set_hook(default_hook);
        let mut app = match built {
            Ok(app) => app,
            Err(payload) => {
                let _ = tx.send(Phase::InitFailed(panic_message(payload)));
                return;
            }
        };
        let _ = tx.send(Phase::InitOk);
        match catch_unwind(AssertUnwindSafe(move || app.run())) {
            Ok(exit) => {
                let _ = tx.send(Phase::Finished(exit));
            }
            Err(payload) => {
                let _ = tx.send(Phase::RunPanicked(panic_message(payload)));
            }
        }
    });

    match rx.recv_timeout(INIT_TIMEOUT) {
        Ok(Phase::InitOk) => {}
        Ok(Phase::InitFailed(reason)) => {
            eprintln!("SKIP gpu_hdr_fixture: no usable wgpu adapter ({reason})");
            return None;
        }
        Err(_) => {
            eprintln!(
                "SKIP gpu_hdr_fixture: renderer init did not complete within {INIT_TIMEOUT:?}"
            );
            return None;
        }
        Ok(Phase::Finished(_) | Phase::RunPanicked(_)) => {
            unreachable!("app finished before reporting init")
        }
    }

    let exit = match rx.recv_timeout(RUN_TIMEOUT) {
        Ok(Phase::Finished(exit)) => exit,
        Ok(Phase::RunPanicked(reason)) => panic!("app panicked while running: {reason}"),
        Err(_) => panic!("app did not finish within {RUN_TIMEOUT:?}"),
        Ok(Phase::InitOk | Phase::InitFailed(_)) => unreachable!("init reported twice"),
    };
    assert_eq!(exit, AppExit::Success, "app did not exit cleanly");
    Some(CAPTURES.lock().unwrap().drain(..).collect())
}

#[test]
fn gpu_hdr_fixture() {
    let _ = std::fs::remove_file(PNG_PATH);
    let Some(captures) = run_app(WorkingColorSpace::Rec709, CASES) else {
        return;
    };
    let readings = auto_exposure::take_readings();
    let rec2020_captures =
        run_app(WorkingColorSpace::Rec2020, REC2020_CASES).expect("renderer started once");
    let rec2020_readings = auto_exposure::take_readings();

    let capture = |label: &str| {
        captures
            .get(label)
            .unwrap_or_else(|| panic!("no capture for {label}"))
    };
    let center = SIZE / 2;
    let mut c = Checker { failed: Vec::new() };

    {
        let case = CASES.iter().find(|case| case.label == PNG_CASE).unwrap();
        let paper_white = case.paper_white_nits.unwrap();
        let bytes = std::fs::read(PNG_PATH).unwrap();
        let chunks = png_chunks(&bytes);
        let chunk = |name: &[u8; 4]| chunks.iter().find(|(n, _)| n == name).map(|(_, d)| *d);

        let mut names: Vec<&[u8]> = chunks.iter().map(|(name, _)| name.as_slice()).collect();
        names.dedup();
        let want: [&[u8]; 8] = [
            b"IHDR", b"cHRM", b"iCCP", b"cICP", b"cLLI", b"sBIT", b"IDAT", b"IEND",
        ];
        c.check(
            "HDR PNG has the HDR chunks in order",
            names == want,
            format!(
                "{:?}",
                names
                    .iter()
                    .map(|n| String::from_utf8_lossy(n))
                    .collect::<Vec<_>>()
            ),
        );
        let ihdr = chunk(b"IHDR").unwrap();
        c.check(
            "HDR PNG is 16-bit RGB",
            ihdr[8..10] == [16, 2],
            format!("bit depth and color type {:?}", &ihdr[8..10]),
        );
        c.check(
            "HDR PNG cICP and sBIT",
            chunk(b"cICP") == Some(&[9, 16, 0, 1]) && chunk(b"sBIT") == Some(&[10, 10, 10]),
            format!("cICP {:?}, sBIT {:?}", chunk(b"cICP"), chunk(b"sBIT")),
        );
        let mut icc = Vec::new();
        let iccp = chunk(b"iCCP").unwrap_or_default();
        let icc_ok = iccp.starts_with(b"_\0\0")
            && flate2::read::ZlibDecoder::new(&iccp[3..])
                .read_to_end(&mut icc)
                .is_ok()
            && icc == REC2100PQ_ICC;
        c.check(
            "HDR PNG iCCP holds the Rec2100PQ profile",
            icc_ok,
            format!("{} profile bytes", icc.len()),
        );

        let png = Image::from_buffer(
            &bytes,
            ImageType::Extension("png"),
            CompressedImageFormats::NONE,
            false,
            ImageSampler::Default,
            RenderAssetUsages::MAIN_WORLD,
            None,
        )
        .unwrap();
        c.check(
            "HDR PNG reads back with BT.2020 primaries",
            png.source_color_primaries == SourceColorPrimaries::Bt2020,
            format!("{:?}", png.source_color_primaries),
        );

        // Compare in BT.2020 nits, within one 10-bit PQ code value.
        let texel = decoded_texel(capture(PNG_CASE), center, center);
        let want = (RgbPrimaries::BT709.matrix_to(RgbPrimaries::BT2020) * texel).max(Vec3::ZERO)
            * paper_white;
        let index = ((center * SIZE + center) * 4 * 2) as usize;
        let data = png.data.as_ref().unwrap();
        let signal = |i: usize| {
            f32::from(u16::from_le_bytes([data[index + 2 * i], data[index + 2 * i + 1]]) >> 6)
                / 1023.0
        };
        let nits = |s: f32| decode_channel(s, SurfaceColorSpace::Pq, 1.0);
        let got = Vec3::from_array(std::array::from_fn(|i| nits(signal(i))));
        let tolerance = Vec3::from_array(std::array::from_fn(|i| {
            nits(signal(i) + 1.0 / 1023.0) - nits(signal(i) - 1.0 / 1023.0)
        }));
        c.check(
            "HDR PNG pixel decodes to the capture",
            (got - want).abs().cmple(tolerance).all(),
            format!("got {got:?} nits, want {want:?}, tolerance {tolerance:?}"),
        );

        let clli = chunk(b"cLLI").unwrap();
        let max_cll = u32::from_be_bytes(clli[..4].try_into().unwrap()) as f32 / 10000.0;
        let max_fall = u32::from_be_bytes(clli[4..].try_into().unwrap()) as f32 / 10000.0;
        let luminance = Vec3::new(0.2627, 0.6780, 0.0593).dot(want);
        c.check(
            "HDR PNG cLLI matches the uniform frame",
            (max_cll - luminance).abs() <= luminance * 1e-3
                && (max_fall - luminance).abs() <= luminance * 1e-3,
            format!("MaxCLL {max_cll}, MaxFALL {max_fall}, luminance {luminance}"),
        );
    }

    let runs = [
        (WorkingColorSpace::Rec709, CASES, &captures),
        (WorkingColorSpace::Rec2020, REC2020_CASES, &rec2020_captures),
    ];
    for (working_color_space, case, captures) in
        runs.iter().flat_map(|(working, cases, captures)| {
            cases
                .iter()
                .filter(|case| !case.viewport)
                .map(move |case| (*working, case, *captures))
        })
    {
        let Some(color_space) = case.color_space else {
            continue;
        };
        let paper_white = case.display_target().resolve(color_space).paper_white_nits;
        let got = decoded_texel(&captures[case.label], center, center);
        let (color, gt7_error) = match case.tonemapping {
            Tonemapping::None | Tonemapping::Linear => {
                (case.ui_color.unwrap_or(case.main_clear), Vec3::ZERO)
            }
            _ => {
                let color = gt7::tonemap(
                    case.main_clear,
                    working_color_space,
                    &case.display_target().resolve(color_space),
                );
                (color, color.abs() * gt7::TOLERANCE + gt7::TOLERANCE)
            }
        };
        let color = if compresses(case, working_color_space) {
            compressed_color(color, color_space)
        } else {
            color
        };
        let want = decoded_color(color, color_space, paper_white);
        let tolerance = tolerance(case, color, color_space, paper_white, working_color_space)
            + to_709_abs(color_space) * gt7_error;
        c.check(
            &format!("{} decodes to its color", case.label),
            (got - want).abs().cmple(tolerance).all(),
            format!(
                "got {got:?}, want {want:?}, error {:?}, tolerance {tolerance:?}",
                (got - want).abs()
            ),
        );
    }

    {
        let case = CASES.iter().find(|case| case.viewport).unwrap();
        let image = capture(case.label);
        for (name, (x, y), want) in [
            (
                "letterbox decodes to the out texture clear",
                (2, 2),
                OUT_CLEAR,
            ),
            (
                "viewport decodes to the main clear",
                (center, center),
                MAIN_CLEAR,
            ),
        ] {
            let got = decoded_texel(image, x, y);
            let tolerance = tolerance(
                case,
                want,
                SurfaceColorSpace::Pq,
                203.0,
                WorkingColorSpace::Rec709,
            );
            c.check(
                name,
                (got - want).abs().cmple(tolerance).all(),
                format!("got {got:?}, want {want:?}, tolerance {tolerance:?}"),
            );
        }
    }

    // Under Rec.2020 the 8-bit main texture holds Rec.2020 values, which costs precision.
    for (name, image, color, max_error) in [
        ("SDR control is unchanged", capture("sdr"), SDR_CLEAR, 2),
        (
            "Rec.2020 SDR decodes to the main clear",
            &rec2020_captures["rec2020_sdr_none"],
            SDR_CLEAR,
            3,
        ),
        (
            "Rec.2020 SDR compresses a color outside Rec.709",
            &rec2020_captures["rec2020_sdr_none_wide"],
            gamut_compress(WIDE_CLEAR).max(Vec3::ZERO),
            4,
        ),
        (
            "Rec.2020 SDR Linear compresses a color outside Rec.709",
            &rec2020_captures["rec2020_sdr_linear_wide"],
            gamut_compress(WIDE_CLEAR).max(Vec3::ZERO),
            2,
        ),
    ] {
        assert_eq!(
            image.texture_descriptor.format,
            TextureFormat::Rgba8UnormSrgb
        );
        let index = ((center * SIZE + center) * 4) as usize;
        let got = &image.data.as_ref().unwrap()[index..index + 4];
        let want = linear(color).to_srgba().to_u8_array();
        c.check(
            name,
            (0..4).all(|i| (i32::from(got[i]) - i32::from(want[i])).abs() <= max_error),
            format!("bytes {got:?}, want {want:?} (within {max_error})"),
        );
    }

    {
        let got = decoded_texel(&rec2020_captures[GT7_WIDE_CASE], center, center);
        c.check(
            "Rec.2020 GT7 keeps a color outside Rec.709",
            got.x < -0.01,
            format!("got {got:?}"),
        );
    }

    for label in COMPRESSED_GT7_CASES {
        let case = REC2020_CASES
            .iter()
            .find(|case| case.label == label)
            .unwrap();
        let color_space = case.color_space.unwrap();
        let got = decoded_texel(&rec2020_captures[label], center, center);
        let in_surface = RgbPrimaries::BT709.matrix_to(color_space.primaries()) * got;
        c.check(
            &format!("{label} is inside the target's gamut"),
            in_surface.min_element() > -1e-3,
            format!("got {in_surface:?} in the target's primaries"),
        );
        let tonemapped = gt7::tonemap(
            case.main_clear,
            WorkingColorSpace::Rec2020,
            &case.display_target().resolve(color_space),
        );
        let hue = |color: Vec3| Oklcha::from(LinearRgba::from_vec3(color)).hue;
        let hue_shift = (hue(got) - hue(tonemapped) + 540.0) % 360.0 - 180.0;
        c.check(
            &format!("{label} keeps its hue"),
            hue_shift.abs() < 10.0,
            format!("hue shift {hue_shift} degrees, got {got:?}, tonemapped {tonemapped:?}"),
        );
    }

    atmosphere::check(&mut c, &captures, &rec2020_captures);
    gt7::check(&mut c, &captures, &rec2020_captures);
    aa_bloom::check(&mut c, &captures);
    auto_exposure::check(&mut c, &captures, &readings, "Rec.709");
    auto_exposure::check(&mut c, &rec2020_captures, &rec2020_readings, "Rec.2020");
    #[cfg(feature = "bevy_solari")]
    solari::check(&mut c, &captures, &rec2020_captures);

    assert!(
        c.failed.is_empty(),
        "{} check(s) FAILED: {:?}",
        c.failed.len(),
        c.failed
    );
    println!("ALL CHECKS PASSED");
}

/// The mean color of a 6x6 block of a linear capture, which averages out noise.
fn block_mean(image: &Image, x: u32, y: u32) -> Vec3 {
    let mut sum = Vec3::ZERO;
    for dy in 0..6 {
        for dx in 0..6 {
            let color = image
                .get_color_at(x - 3 + dx, y - 3 + dy)
                .unwrap()
                .to_linear();
            sum += Vec3::new(color.red, color.green, color.blue);
        }
    }
    sum / 36.0
}

/// An atmosphere renders in Rec.709, so its sky must not depend on the working color space.
mod atmosphere {
    use super::*;
    use bevy::{
        light::{atmosphere::ScatteringMedium, Atmosphere},
        pbr::AtmosphereSettings,
    };

    const LABEL: &str = "atmosphere";

    pub fn setup(
        mut commands: Commands,
        mut images: ResMut<Assets<Image>>,
        mut mediums: ResMut<Assets<ScatteringMedium>>,
        mut targets: ResMut<Targets>,
    ) {
        let handle = images.add(Image::new_target_texture(
            32,
            32,
            TextureFormat::Rgba16Float,
            None,
        ));
        commands.spawn(Atmosphere::earth(
            mediums.add(ScatteringMedium::earth(64, 64)),
        ));
        commands.spawn((
            Camera3d::default(),
            Camera {
                order: 2000,
                ..default()
            },
            RenderTarget::Image(handle.clone().into()),
            Transform::from_xyz(0.0, 1.0, 0.0).looking_at(Vec3::new(10.0, 1.5, 0.0), Vec3::Y),
            AtmosphereSettings::default(),
            // Bright enough for the 1 lux sun.
            Exposure { ev100: -3.5 },
            Tonemapping::None,
            DebandDither::Disabled,
            Msaa::Off,
        ));
        // The only directional light, which the Solari scene also uses.
        commands.spawn((
            DirectionalLight {
                illuminance: 1.0,
                color: Color::linear_rgb(1.0, 0.6, 0.2),
                shadow_maps_enabled: false,
                ..default()
            },
            Transform::from_xyz(1.0, 0.5, 0.2).looking_at(Vec3::ZERO, Vec3::Y),
        ));
        targets.0.push((LABEL, handle));
    }

    pub fn check(
        c: &mut Checker,
        rec709: &HashMap<&'static str, Image>,
        rec2020: &HashMap<&'static str, Image>,
    ) {
        for (x, y) in [(16, 8), (16, 24)] {
            let want = block_mean(&rec709[LABEL], x, y);
            let got = block_mean(&rec2020[LABEL], x, y);
            let tolerance = want.abs() * 0.01 + 1e-3;
            c.check(
                &format!("atmosphere ({x}, {y}) matches under Rec.2020"),
                (got - want).abs().cmple(tolerance).all() && want.max_element() > 0.01,
                format!("got {got:?}, want {want:?}, tolerance {tolerance:?}"),
            );
        }
    }
}

/// A Solari scene whose colors don't depend on the working color space: a colored emissive
/// quad, and a gray quad lit by the colored directional light and a uniform colored
/// environment map. A gray albedo scales light the same way in every color space.
#[cfg(feature = "bevy_solari")]
mod solari {
    use super::*;
    use bevy::{
        camera::{visibility::RenderLayers, CameraMainTextureUsages},
        render::render_resource::{
            Extent3d, TextureDimension, TextureUsages, TextureViewDescriptor, TextureViewDimension,
        },
        solari::{
            pathtracer::{Pathtracer, PathtracingPlugin},
            prelude::{RaytracingMesh3d, SolariLighting, SolariPlugins},
        },
    };

    const SOLARI_SIZE: u32 = 32;
    const LABELS: [&str; 2] = ["solari_pathtracer", "solari_realtime"];
    const LAYER: usize = 1;

    pub fn add(app: &mut App) {
        app.add_plugins((SolariPlugins, PathtracingPlugin))
            .add_systems(Startup, setup.after(super::setup));
    }

    fn setup(
        mut commands: Commands,
        mut images: ResMut<Assets<Image>>,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        mut targets: ResMut<Targets>,
    ) {
        let sky: Vec<u8> = [0.2f32, 0.4, 0.8, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut cubemap = Image::new_fill(
            Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            &sky,
            TextureFormat::Rgba32Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        cubemap.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..default()
        });
        let cubemap = images.add(cubemap);

        for (order, label) in LABELS.into_iter().enumerate() {
            let handle = images.add(Image::new_target_texture(
                SOLARI_SIZE,
                SOLARI_SIZE,
                TextureFormat::Rgba16Float,
                None,
            ));
            let mut camera = commands.spawn((
                Camera3d::default(),
                Camera {
                    order: 1000 + order as isize,
                    clear_color: ClearColorConfig::Custom(Color::BLACK),
                    ..default()
                },
                RenderTarget::Image(handle.clone().into()),
                Transform::from_xyz(0.0, 3.0, 0.0).looking_at(Vec3::ZERO, Vec3::Z),
                CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
                Msaa::Off,
                Exposure { ev100: 0.0 },
                EnvironmentMapLight {
                    diffuse_map: cubemap.clone(),
                    specular_map: cubemap.clone(),
                    intensity: 1.0,
                    ..default()
                },
                RenderLayers::layer(LAYER),
            ));
            if label == "solari_pathtracer" {
                camera.insert((
                    Pathtracer::default(),
                    Tonemapping::None,
                    DebandDither::Disabled,
                ));
            } else {
                camera.insert((SolariLighting::default(), Tonemapping::Linear));
            }
            targets.0.push((label, handle));
        }

        let quad = meshes.add(
            Plane3d::default()
                .mesh()
                .size(1.0, 1.0)
                .build()
                .with_generated_tangents()
                .unwrap(),
        );
        for (x, material) in [
            (
                0.4,
                StandardMaterial {
                    base_color: Color::linear_rgb(0.5, 0.5, 0.5),
                    perceptual_roughness: 1.0,
                    ..default()
                },
            ),
            (
                -0.4,
                StandardMaterial {
                    base_color: Color::BLACK,
                    reflectance: 0.0,
                    emissive: LinearRgba::rgb(1.0, 0.05, 0.1),
                    ..default()
                },
            ),
        ] {
            commands.spawn((
                RaytracingMesh3d(quad.clone()),
                Mesh3d(quad.clone()),
                MeshMaterial3d(materials.add(material)),
                Transform::from_xyz(x, 0.0, 0.0),
                RenderLayers::layer(LAYER),
            ));
        }
    }

    pub fn check(
        c: &mut Checker,
        rec709: &HashMap<&'static str, Image>,
        rec2020: &HashMap<&'static str, Image>,
    ) {
        for label in LABELS {
            // The gray quad, the emissive quad, and the sky, which only the pathtracer draws.
            let mut blocks = vec![("gray", 8, 16), ("emissive", 24, 16)];
            if label == "solari_pathtracer" {
                blocks.push(("sky", 4, 4));
            }
            for (name, x, y) in blocks {
                let rec709_mean = block_mean(&rec709[label], x, y);
                // Both cameras can output colors outside Rec.709 under Rec.2020, so the blit
                // compresses them into the sRGB target's gamut. Saturated colors inside the
                // gamut move too.
                let want = compressed_color(rec709_mean, SurfaceColorSpace::Srgb);
                let got = block_mean(&rec2020[label], x, y);
                // Path tracing noise scales with the brightest channel.
                let tolerance = rec709_mean.abs() * 0.03 + rec709_mean.max_element() * 0.01;
                c.check(
                    &format!("{label} {name} matches under Rec.2020"),
                    (got - want).abs().cmple(tolerance).all() && want.max_element() > 0.01,
                    format!("got {got:?}, want {want:?}, tolerance {tolerance:?}"),
                );
            }
        }
    }
}

/// A CPU port of `Tonemapping::GranTurismo7` at its default parameters, and SDR cameras
/// that run it.
mod gt7 {
    use super::*;
    use bevy::{
        camera::visibility::RenderLayers,
        core_pipeline::prepass::{DeferredPrepass, DepthPrepass},
        material::OpaqueRendererMethod,
        window::ResolvedDisplayTarget,
    };

    /// Relative error allowed between the GPU and this port.
    pub const TOLERANCE: f32 = 2e-3;
    /// Label, whether the camera shares its target so that it tonemaps in its shaders, and
    /// whether it renders deferred.
    const CAMERAS: [(&str, bool, bool); 4] = [
        ("gt7_sdr_pass", false, false),
        ("gt7_sdr_in_shader", true, false),
        ("gt7_sdr_deferred_pass", false, true),
        ("gt7_sdr_deferred_in_shader", true, true),
    ];
    const COLOR: Vec3 = Vec3::new(0.8, 0.3, 0.05);
    const FORWARD_LAYER: usize = 7;
    const DEFERRED_LAYER: usize = 8;

    struct Params {
        peak: f32,
        k_a: f32,
        k_b: f32,
        k_c: f32,
        peak_ucs: f32,
        output_scale: f32,
        clamp: bool,
    }

    const MID_POINT: f32 = 0.538;
    const LINEAR_SECTION: f32 = 0.444;
    const TOE_STRENGTH: f32 = 1.28;
    const ALPHA: f32 = 0.25;
    const BLEND_RATIO: f32 = 0.6;
    const FADE_START: f32 = 0.98;
    const FADE_END: f32 = 1.16;

    fn params(target: &ResolvedDisplayTarget) -> Params {
        let (peak_nits, paper_white, clamp) = if target.color_space.is_hdr() {
            (
                target.peak_luminance_nits.max(250.0),
                target.paper_white_nits,
                false,
            )
        } else {
            (250.0, 250.0, true)
        };
        let peak = peak_nits / 100.0;
        let k = (LINEAR_SECTION - 1.0) / (ALPHA - 1.0);
        Params {
            peak,
            k_a: peak * LINEAR_SECTION + peak * k,
            k_b: -peak * k * ops::exp(LINEAR_SECTION / k),
            k_c: -1.0 / (k * peak),
            peak_ucs: ictcp(Vec3::splat(peak)).x,
            output_scale: 100.0 / paper_white,
            clamp,
        }
    }

    fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    fn pq_eotf(n: f32) -> f32 {
        let (m1, m2) = (2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0);
        let (c1, c2, c3) = (
            3424.0 / 4096.0,
            2413.0 / 4096.0 * 32.0,
            2392.0 / 4096.0 * 32.0,
        );
        let np = ops::powf(n.clamp(0.0, 1.0), 1.0 / m2);
        ops::powf((np - c1).max(0.0) / (c2 - c3 * np), 1.0 / m1) * 100.0
    }

    fn ictcp(rgb: Vec3) -> Vec3 {
        let lms = Vec3::new(
            rgb.dot(Vec3::new(1688.0, 2146.0, 262.0)),
            rgb.dot(Vec3::new(683.0, 2951.0, 462.0)),
            rgb.dot(Vec3::new(99.0, 309.0, 3688.0)),
        ) / 4096.0;
        let pq = lms.map(|v| pq_inverse_eotf_from_nits(v * 100.0));
        Vec3::new(
            pq.dot(Vec3::new(2048.0, 2048.0, 0.0)),
            pq.dot(Vec3::new(6610.0, -13613.0, 7003.0)),
            pq.dot(Vec3::new(17933.0, -17390.0, -543.0)),
        ) / 4096.0
    }

    fn rgb(ictcp: Vec3) -> Vec3 {
        let lms = Vec3::new(
            ictcp.dot(Vec3::new(1.0, 0.00860904, 0.11103)),
            ictcp.dot(Vec3::new(1.0, -0.00860904, -0.11103)),
            ictcp.dot(Vec3::new(1.0, 0.560031, -0.320627)),
        )
        .map(pq_eotf);
        Vec3::new(
            lms.dot(Vec3::new(3.43661, -2.50645, 0.0698454)),
            lms.dot(Vec3::new(-0.79133, 1.9836, -0.192271)),
            lms.dot(Vec3::new(-0.0259499, -0.0989137, 1.12486)),
        )
        .max(Vec3::ZERO)
    }

    fn curve(x: f32, p: &Params) -> f32 {
        if x < 0.0 {
            return 0.0;
        }
        let weight = smoothstep(0.0, MID_POINT, x);
        if x < LINEAR_SECTION * p.peak {
            let toe = MID_POINT * ops::powf(x / MID_POINT, TOE_STRENGTH);
            (1.0 - weight) * toe + weight * x
        } else {
            p.k_a + p.k_b * ops::exp(x * p.k_c)
        }
    }

    /// What the tonemapping pass outputs for a linear Rec.709 `color` held in the working
    /// color space: linear Rec.709.
    pub fn tonemap(
        color: Vec3,
        working_color_space: WorkingColorSpace,
        target: &ResolvedDisplayTarget,
    ) -> Vec3 {
        let to_2020 = RgbPrimaries::BT709.matrix_to(RgbPrimaries::BT2020);
        let rec2020 = match working_color_space {
            WorkingColorSpace::Rec709 => to_2020 * color.max(Vec3::ZERO),
            WorkingColorSpace::Rec2020 => (to_2020 * color).max(Vec3::ZERO),
        };
        let p = params(target);
        let fb = rec2020 * 2.5;
        let ucs = ictcp(fb);
        let skewed = fb.map(|x| curve(x, &p));
        let skewed_ucs = ictcp(skewed);
        let chroma = 1.0 - smoothstep(FADE_START, FADE_END, ucs.x / p.peak_ucs);
        let scaled = rgb(Vec3::new(skewed_ucs.x, ucs.y * chroma, ucs.z * chroma));
        let blended = skewed.lerp(scaled, BLEND_RATIO);
        let mapped = RgbPrimaries::BT2020.matrix_to(RgbPrimaries::BT709)
            * (p.output_scale * blended.min(Vec3::splat(p.peak)));
        if p.clamp {
            mapped.clamp(Vec3::ZERO, Vec3::ONE)
        } else {
            mapped
        }
    }

    /// Two SDR cameras. One runs the tonemapping pass. The other shares its target with a
    /// second camera, so both tonemap in their material shaders.
    pub fn setup(
        mut commands: Commands,
        mut images: ResMut<Assets<Image>>,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        mut targets: ResMut<Targets>,
    ) {
        let quad = meshes.add(Rectangle::new(10.0, 10.0));
        for (layer, opaque_render_method) in [
            (FORWARD_LAYER, OpaqueRendererMethod::Forward),
            (DEFERRED_LAYER, OpaqueRendererMethod::Deferred),
        ] {
            commands.spawn((
                Mesh3d(quad.clone()),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: linear(COLOR),
                    unlit: true,
                    opaque_render_method,
                    ..default()
                })),
                Transform::from_xyz(0.0, 0.0, -1.0),
                RenderLayers::layer(layer),
            ));
        }
        for (index, (label, shares_target, deferred)) in CAMERAS.into_iter().enumerate() {
            let handle = images.add(Image::new_target_texture(
                SIZE,
                SIZE,
                TextureFormat::Rgba8UnormSrgb,
                None,
            ));
            let layer = RenderLayers::layer(if deferred {
                DEFERRED_LAYER
            } else {
                FORWARD_LAYER
            });
            for i in 0..1 + isize::from(shares_target) {
                let mut camera = commands.spawn((
                    Camera3d::default(),
                    Camera {
                        order: 3000 + index as isize * 2 + i,
                        clear_color: if i == 0 {
                            ClearColorConfig::Custom(Color::BLACK)
                        } else {
                            ClearColorConfig::None
                        },
                        ..default()
                    },
                    RenderTarget::Image(handle.clone().into()),
                    Tonemapping::GranTurismo7,
                    DebandDither::Disabled,
                    Msaa::Off,
                    layer.clone(),
                ));
                if deferred {
                    camera.insert((DepthPrepass, DeferredPrepass));
                }
            }
            targets.0.push((label, handle));
        }
    }

    pub fn check(
        c: &mut Checker,
        rec709: &HashMap<&'static str, Image>,
        rec2020: &HashMap<&'static str, Image>,
    ) {
        let want = Color::from(LinearRgba::from_vec3(tonemap(
            COLOR,
            WorkingColorSpace::Rec709,
            &ResolvedDisplayTarget::default(),
        )))
        .to_srgba()
        .to_u8_array();
        for (working, captures) in [("Rec.709", rec709), ("Rec.2020", rec2020)] {
            for (label, ..) in CAMERAS {
                let image = &captures[label];
                let index = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
                let got = &image.data.as_ref().unwrap()[index..index + 4];
                c.check(
                    &format!("{label} under {working} matches the CPU port"),
                    (0..3).all(|i| (i32::from(got[i]) - i32::from(want[i])).abs() <= 2),
                    format!("bytes {got:?}, want {want:?} (within 2)"),
                );
            }
        }
    }
}

/// Post-tonemap effects on HDR display targets, where values go above 1.0 and below 0.0.
mod aa_bloom {
    use super::*;
    use bevy::{
        anti_alias::{
            contrast_adaptive_sharpening::ContrastAdaptiveSharpening, fxaa::Fxaa, smaa::Smaa,
        },
        camera::visibility::RenderLayers,
        material::OpaqueRendererMethod,
        post_process::bloom::{Bloom, BloomCompositeMode, BloomPrefilter},
    };

    /// Four times paper white in red.
    const BRIGHT: Vec3 = Vec3::new(4.0, 2.0, 0.5);
    /// Negative luma, which has no square root.
    const NEGATIVE: Vec3 = Vec3::new(-1.0, 0.1, 0.1);
    /// 150 nits at a paper white of 100 nits.
    const BLOOM_CLEAR: Vec3 = Vec3::splat(1.5);
    const EFFECTS: [&str; 4] = ["control", "fxaa", "smaa", "cas"];
    /// Rec.2020 green at paper white has negative red and blue in Rec.709.
    const EDGE_CLEAR: Vec3 = Vec3::new(-0.6, 1.0, -0.1);
    /// A darker color with the same signs, on the left half of the target.
    const EDGE_QUAD: Vec3 = Vec3::new(-0.1, 0.2, -0.02);
    const EDGE_LAYER: usize = 9;
    const EDGE_LABELS: [&str; 2] = ["cas_edge_control", "cas_edge"];

    #[derive(Clone, Copy)]
    enum Effect {
        None,
        Fxaa,
        Smaa,
        Cas,
        Bloom { threshold_nits: f32 },
    }

    /// Label, effect, color space, paper white and clear color of each camera.
    fn cameras() -> Vec<(String, Effect, SurfaceColorSpace, Option<f32>, Vec3)> {
        let mut cameras = Vec::new();
        for (target, color_space, clear) in [
            ("pq", SurfaceColorSpace::Pq, BRIGHT),
            ("scrgb", SurfaceColorSpace::ScRgbLinear, NEGATIVE),
        ] {
            for (name, effect) in
                EFFECTS
                    .into_iter()
                    .zip([Effect::None, Effect::Fxaa, Effect::Smaa, Effect::Cas])
            {
                cameras.push((format!("{name}_{target}"), effect, color_space, None, clear));
            }
        }
        for (name, effect) in [
            ("bloom_control", Effect::None),
            (
                "bloom_below",
                Effect::Bloom {
                    threshold_nits: 200.0,
                },
            ),
            (
                "bloom_above",
                Effect::Bloom {
                    threshold_nits: 100.0,
                },
            ),
        ] {
            cameras.push((
                name.to_string(),
                effect,
                SurfaceColorSpace::Pq,
                Some(100.0),
                BLOOM_CLEAR,
            ));
        }
        cameras
    }

    pub fn setup(
        mut commands: Commands,
        mut images: ResMut<Assets<Image>>,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        mut manual: ResMut<ManualDisplayTargets>,
        mut targets: ResMut<Targets>,
    ) {
        commands.spawn((
            Mesh3d(meshes.add(Rectangle::new(10.0, 10.0))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: linear(EDGE_QUAD),
                unlit: true,
                // Solari makes deferred the default, and these cameras have no deferred prepass.
                opaque_render_method: OpaqueRendererMethod::Forward,
                ..default()
            })),
            Transform::from_xyz(-5.0, 0.0, -1.0),
            RenderLayers::layer(EDGE_LAYER),
        ));
        for (index, label) in EDGE_LABELS.into_iter().enumerate() {
            let handle = images.add(Image::new_target_texture(
                SIZE,
                SIZE,
                TextureFormat::Rgba16Float,
                None,
            ));
            let render_target = RenderTarget::Image(handle.clone().into());
            manual.insert(
                render_target.normalize(None).unwrap(),
                DisplayTarget {
                    color_space_override: Some(SurfaceColorSpace::ScRgbLinear),
                    ..default()
                },
            );
            let mut camera = commands.spawn((
                Camera3d::default(),
                Camera {
                    order: 3900 + index as isize,
                    clear_color: ClearColorConfig::Custom(linear(EDGE_CLEAR)),
                    ..default()
                },
                render_target,
                Tonemapping::None,
                DebandDither::Disabled,
                Msaa::Off,
                RenderLayers::layer(EDGE_LAYER),
            ));
            if label == "cas_edge" {
                camera.insert(ContrastAdaptiveSharpening {
                    sharpening_strength: 1.0,
                    ..default()
                });
            }
            targets.0.push((label, handle));
        }

        for (index, (label, effect, color_space, paper_white_nits, clear)) in
            cameras().into_iter().enumerate()
        {
            let handle = images.add(Image::new_target_texture(
                SIZE,
                SIZE,
                TextureFormat::Rgba16Float,
                None,
            ));
            let render_target = RenderTarget::Image(handle.clone().into());
            manual.insert(
                render_target.normalize(None).unwrap(),
                DisplayTarget {
                    color_space_override: Some(color_space),
                    paper_white_nits,
                    ..default()
                },
            );
            let mut camera = commands.spawn((
                Camera3d::default(),
                Camera {
                    order: 4000 + index as isize,
                    clear_color: ClearColorConfig::Custom(linear(clear)),
                    ..default()
                },
                render_target,
                Tonemapping::None,
                DebandDither::Disabled,
                Msaa::Off,
            ));
            match effect {
                Effect::None => {}
                Effect::Fxaa => {
                    camera.insert(Fxaa::default());
                }
                Effect::Smaa => {
                    camera.insert(Smaa::default());
                }
                Effect::Cas => {
                    camera.insert(ContrastAdaptiveSharpening::default());
                }
                Effect::Bloom { threshold_nits } => {
                    camera.insert(Bloom {
                        intensity: 0.5,
                        composite_mode: BloomCompositeMode::Additive,
                        prefilter: BloomPrefilter {
                            threshold_nits: Some(threshold_nits),
                            ..default()
                        },
                        ..Bloom::NATURAL
                    });
                }
            }
            targets.0.push((label.leak(), handle));
        }
    }

    pub fn check(c: &mut Checker, captures: &HashMap<&'static str, Image>) {
        let center = SIZE / 2;
        let texel = |label: &str| decoded_texel(&captures[label], center, center);
        // One f16 step of the main texture and of the encoded signal.
        let close = |got: Vec3, want: Vec3| {
            (got - want)
                .abs()
                .cmple(want.abs() * ops::exp2(-9.0) + 1e-5)
                .all()
        };
        for target in ["pq", "scrgb"] {
            let want = texel(&format!("control_{target}"));
            for name in &EFFECTS[1..] {
                let got = texel(&format!("{name}_{target}"));
                c.check(
                    &format!("{name} on {target} leaves a flat color unchanged"),
                    !got.is_nan() && close(got, want),
                    format!("got {got:?}, want {want:?}"),
                );
            }
        }
        {
            let texels = |label: &str| {
                (0..SIZE)
                    .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
                    .map(|(x, y)| decoded_texel(&captures[label], x, y))
                    .collect::<Vec<_>>()
            };
            let input = texels(EDGE_LABELS[0]);
            let output = texels(EDGE_LABELS[1]);
            let in_min = input.iter().copied().fold(Vec3::INFINITY, Vec3::min);
            let in_max = input.iter().copied().fold(Vec3::NEG_INFINITY, Vec3::max);
            // The input range, widened to 0, and to paper white for a positive channel.
            let lower = in_min.min(Vec3::ZERO) - 1e-3;
            let upper =
                Vec3::select(in_max.cmpgt(Vec3::ZERO), in_max.max(Vec3::ONE), Vec3::ZERO) + 1e-3;
            let out_min = output.iter().copied().fold(Vec3::INFINITY, Vec3::min);
            let out_max = output.iter().copied().fold(Vec3::NEG_INFINITY, Vec3::max);
            c.check(
                "cas across an edge with a negative channel stays inside the input range",
                output.iter().all(|t| t.is_finite())
                    && out_min.cmpge(lower).all()
                    && out_max.cmple(upper).all(),
                format!("output {out_min:?} to {out_max:?}, bounds {lower:?} to {upper:?}"),
            );
            let sharpened = input
                .iter()
                .zip(&output)
                .filter(|(i, o)| (**i - **o).abs().max_element() > 0.01)
                .count();
            c.check(
                "cas sharpens the edge",
                sharpened > 0 && in_min.x < -0.5 && in_max.x < 0.0,
                format!("{sharpened} texels changed, input {in_min:?} to {in_max:?}"),
            );
        }
        let want = texel("bloom_control");
        let below = texel("bloom_below");
        c.check(
            "bloom below threshold_nits at paper white 100 adds nothing",
            close(below, want),
            format!("got {below:?}, want {want:?}"),
        );
        let above = texel("bloom_above");
        c.check(
            "bloom above threshold_nits at paper white 100 adds light",
            (above - want).min_element() > 0.05,
            format!("got {above:?}, control {want:?}"),
        );
    }
}

/// Auto exposure with physiological adaptation, and auto white balance.
///
/// Each camera meters a flat clear color with `pre_exposure` off, so the scene luminance doesn't
/// depend on the adapted exposure. Once all have adapted, the clear color becomes `BRIGHT_CLEAR`,
/// three stops brighter with a different tint. Then `ae_reset` resets.
mod auto_exposure {
    use super::*;

    /// A tungsten tint, about 3700 K.
    const DIM_CLEAR: Vec3 = Vec3::new(0.25, 0.15, 0.075);
    /// About 5700 K, with three stops more luminance than `DIM_CLEAR`.
    const BRIGHT_CLEAR: Vec3 = Vec3::new(1.451, 1.306, 1.161);
    const CAMERA_EV100: f32 = 12.0;
    /// Half a histogram bin of the default metering range.
    const EV100_TOLERANCE: f32 = 0.12;
    /// Frames to wait for the adapted exposure to settle and reach the readback.
    const SETTLE_FRAMES: u32 = 20;
    /// Neutralizes the dim tint, and keeps adapting.
    const WHITE_BALANCE: &str = "awb_neutral";
    /// Envelope fast enough to never hold the exposure.
    const PHYSIOLOGICAL: &str = "ae_physiological";
    /// Envelope that never moves, and white balance that never adapts, until the reset.
    const RESET: &str = "ae_reset";

    static READINGS: Mutex<Vec<(&'static str, f32)>> = Mutex::new(Vec::new());

    #[derive(Resource, Default)]
    enum Phase {
        #[default]
        Adapting,
        Brightened {
            at: u32,
        },
        Reset {
            seen_none: bool,
            settled_at: Option<u32>,
        },
        Done,
    }

    #[derive(Component)]
    struct Label(&'static str);

    #[derive(Resource)]
    struct Handles(Vec<(&'static str, Handle<Image>)>);

    pub fn add(app: &mut App) {
        app.init_resource::<Phase>()
            .add_systems(Startup, setup.after(super::setup))
            .add_systems(Update, drive.after(driver));
    }

    pub fn take_readings() -> HashMap<&'static str, f32> {
        READINGS.lock().unwrap().drain(..).collect()
    }

    /// The target EV100 for a clear color, which tonemapping displays at luminance 1.
    fn target_ev100(color: Vec3) -> f32 {
        ops::log2(Vec3::new(0.2125, 0.7154, 0.0721).dot(color)) + CAMERA_EV100
    }

    fn setup(
        mut commands: Commands,
        mut images: ResMut<Assets<Image>>,
        mut manual: ResMut<ManualDisplayTargets>,
        mut extra: ResMut<ExtraCaptures>,
    ) {
        let fast = PhysiologicalAdaptation {
            speed_brighten: 1000.0,
            speed_darken: 1000.0,
            bound_brighten: 1.0,
            bound_darken: 1.0,
        };
        let held = PhysiologicalAdaptation {
            speed_brighten: 0.0,
            speed_darken: 0.0,
            ..fast
        };
        let mut handles = Vec::new();
        for (index, (label, physiological, white_balance_speed)) in [
            (WHITE_BALANCE, None, Some(10.0)),
            (PHYSIOLOGICAL, Some(fast), None),
            (RESET, Some(held), Some(0.0)),
        ]
        .into_iter()
        .enumerate()
        {
            let handle = images.add(Image::new_target_texture(
                SIZE,
                SIZE,
                TextureFormat::Rgba16Float,
                None,
            ));
            let render_target = RenderTarget::Image(handle.clone().into());
            manual.insert(
                render_target.normalize(None).unwrap(),
                DisplayTarget {
                    color_space_override: Some(SurfaceColorSpace::ScRgbLinear),
                    ..default()
                },
            );
            let mut camera = commands.spawn((
                Camera3d::default(),
                Camera {
                    order: 5000 + index as isize,
                    clear_color: ClearColorConfig::Custom(linear(DIM_CLEAR)),
                    ..default()
                },
                render_target,
                Exposure {
                    ev100: CAMERA_EV100,
                },
                AutoExposure {
                    speed_brighten: 1000.0,
                    speed_darken: 1000.0,
                    pre_exposure: false,
                    physiological,
                    ..default()
                },
                Tonemapping::Linear,
                DebandDither::Disabled,
                Msaa::Off,
                Label(label),
            ));
            if let Some(speed) = white_balance_speed {
                camera.insert(AutoWhiteBalance { speed, ..default() });
                handles.push((label, handle));
            }
        }
        extra.0 += handles.len();
        commands.insert_resource(Handles(handles));
    }

    fn drive(
        frame: Res<Frame>,
        mut phase: ResMut<Phase>,
        handles: Res<Handles>,
        mut commands: Commands,
        mut cameras: Query<(&Label, &mut Camera, &mut AutoExposure, &AutoExposureState)>,
    ) {
        let ev100 = |label: &str| {
            cameras
                .iter()
                .find(|(l, ..)| l.0 == label)
                .and_then(|(.., state)| state.ev100)
        };
        let record = |name: &'static str, value: Option<f32>| {
            READINGS
                .lock()
                .unwrap()
                .push((name, value.unwrap_or(f32::NAN)));
        };
        match *phase {
            Phase::Adapting => {
                if cameras.iter().all(|(.., state)| state.ev100.is_some()) {
                    record("reset_dim", ev100(RESET));
                    for (_, mut camera, ..) in &mut cameras {
                        camera.clear_color = ClearColorConfig::Custom(linear(BRIGHT_CLEAR));
                    }
                    *phase = Phase::Brightened { at: frame.0 };
                }
            }
            Phase::Brightened { at } => {
                if frame.0 >= at + SETTLE_FRAMES {
                    record("physiological_bright", ev100(PHYSIOLOGICAL));
                    record("reset_held", ev100(RESET));
                    for (label, _, mut auto_exposure, _) in &mut cameras {
                        auto_exposure.reset = label.0 == RESET;
                    }
                    *phase = Phase::Reset {
                        seen_none: false,
                        settled_at: None,
                    };
                }
            }
            Phase::Reset {
                ref mut seen_none,
                ref mut settled_at,
            } => {
                let value = ev100(RESET);
                *seen_none |= value.is_none();
                if *seen_none && value.is_some() && settled_at.is_none() {
                    *settled_at = Some(frame.0);
                }
                if settled_at.is_some_and(|at| frame.0 >= at + SETTLE_FRAMES) {
                    record("reset_bright", value);
                    for (label, handle) in &handles.0 {
                        let label = *label;
                        commands.spawn(Screenshot::image(handle.clone())).observe(
                            move |captured: On<ScreenshotCaptured>| {
                                CAPTURES
                                    .lock()
                                    .unwrap()
                                    .push((label, captured.image.clone()));
                            },
                        );
                    }
                    *phase = Phase::Done;
                }
            }
            Phase::Done => {}
        }
    }

    pub fn check(
        c: &mut Checker,
        captures: &HashMap<&'static str, Image>,
        readings: &HashMap<&'static str, f32>,
        working_color_space: &str,
    ) {
        let dim = target_ev100(DIM_CLEAR);
        let bright = target_ev100(BRIGHT_CLEAR);
        let near = |got: f32, want: f32, tolerance: f32| (got - want).abs() <= tolerance;
        let reset_dim = readings["reset_dim"];
        for (name, got, want, tolerance) in [
            ("adapts to the dim target", reset_dim, dim, EV100_TOLERANCE),
            (
                "with a fast long-term adaptation converges to the bright target",
                readings["physiological_bright"],
                bright,
                EV100_TOLERANCE,
            ),
            (
                "holds one stop above a fixed long-term adaptation",
                readings["reset_held"],
                reset_dim + 1.0,
                1e-3,
            ),
            (
                "converges to the bright target after a reset",
                readings["reset_bright"],
                bright,
                EV100_TOLERANCE,
            ),
        ] {
            c.check(
                &format!("{working_color_space} auto exposure {name}"),
                near(got, want, tolerance),
                format!("EV100 {got}, want {want} within {tolerance}"),
            );
        }
        for (label, name) in [
            (WHITE_BALANCE, "neutralizes a tinted gray"),
            (RESET, "neutralizes a new tint after a reset"),
        ] {
            let got = decoded_texel(&captures[label], SIZE / 2, SIZE / 2);
            let spread = (got.max_element() - got.min_element()) / got.max_element();
            c.check(
                &format!("{working_color_space} auto white balance {name}"),
                spread < 0.01,
                format!("got {got:?}, channel spread {spread}"),
            );
        }
    }
}
