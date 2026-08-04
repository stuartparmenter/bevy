//! Headless GPU test for the HDR display-output path.
//!
//! Verifies, on a real wgpu device:
//!
//! - the populated `ManualDisplayTargets` map reaches the render world, each view
//!   resolves its registered target, and screenshot decode keys off the same map;
//! - the display-target uniform is HDR-gated: registered PQ views carry
//!   `DisplayTargetUniform` with a per-view dynamic offset, while a plain SDR view
//!   renders correctly without it;
//! - each GT7 view's `Gt7ParamsUniform` matches the CPU-computed uniform, an
//!   in-place `GranTurismo7Params` mutation refreshes it, and every GT7 view
//!   tone-maps node-side (the `TonemappingPass` veto);
//! - the PQ encode/decode round trip matches the CPU reference chain in
//!   `expected_decoded`;
//! - auto exposure and auto white balance render and move exposure over time.
//!
//! Run on a software Vulkan driver (lavapipe), no display server needed:
//!
//! ```sh
//! VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json WGPU_BACKEND=vulkan \
//!   cargo test --test gpu_hdr_fixture -- --nocapture
//! ```
//!
//! Without a usable wgpu adapter, or when renderer init fails, the test prints a `SKIP`
//! line and passes.
//!
//! Scene: five cameras, each with a uniform clear color, each rendering to
//! its own offscreen `Image` target. Four targets are registered in
//! `ManualDisplayTargets` with `DisplayTransfer::Pq` + Rec.2020. One is a plain SDR
//! `Rgba8UnormSrgb` target with no registration. Screenshots are captured mid-run,
//! decoded by `decode_pq_screenshot`, then compared against the CPU prediction.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::{mpsc, Mutex, OnceLock};
use std::time::Duration;

use bevy::render::view::screenshot::{save_to_disk, Screenshot, ScreenshotCaptured};
use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    asset::AssetId,
    audio::AudioPlugin,
    camera::{Camera3d, ClearColorConfig, Hdr, NormalizedRenderTarget, RenderTarget},
    core_pipeline::tonemapping::{
        DebandDither, GranTurismo7Params, Gt7ParamsUniform, Tonemapping, ViewTonemappingPipeline,
    },
    math::{ops, Vec3},
    post_process::auto_exposure::{AutoExposure, AutoExposurePlugin, AutoWhiteBalance},
    prelude::*,
    render::{
        camera::ExtractedCamera,
        render_resource::TextureFormat,
        transfer_functions::{
            pq_eotf, pq_inverse_eotf_from_nits, srgb_oetf, PQ_MAX_LUMINANCE_NITS,
            SCRGB_REFERENCE_WHITE_NITS,
        },
        uniform::DynamicUniformIndex,
        view::{
            window::display_target::ManualDisplayTargets, DisplayTargetUniform, ViewDisplayTarget,
        },
        working_color_space::{REC2020_TO_REC709, REC709_TO_REC2020},
        Render, RenderApp, RenderSystems,
    },
    window::{DisplayGamut, DisplayTarget, DisplayTransfer, ExitCondition},
    winit::WinitPlugin,
};

const SIZE: u32 = 128;
/// Frame for the first capture round (past async pipeline compilation).
const CAPTURE_1: u32 = 40;
/// Frame at which camera B's `GranTurismo7Params` is mutated in place.
const MUTATE_AT: u32 = 70;
/// Frame for the second capture round (mutated B + late auto-exposure).
const CAPTURE_2: u32 = 110;
const MAX_FRAMES: u32 = 2000;

// --- Display targets registered in ManualDisplayTargets ---

/// PQ / Rec.2020, 1000-nit peak, 203-nit paper white (targets A, B, E).
const T_PQ_203: DisplayTarget = DisplayTarget::SDR_SRGB
    .with_transfer(DisplayTransfer::Pq)
    .with_gamut(DisplayGamut::Rec2020)
    .with_peak(1000.0)
    .with_paper_white(203.0)
    .with_min_luminance(0.005);

/// PQ / Rec.2020 for target C. A second, distinct calibration, so the display-target
/// uniform buffer holds more than one slice and a wrong dynamic offset shows up.
const T_PQ_100: DisplayTarget = DisplayTarget::SDR_SRGB
    .with_transfer(DisplayTransfer::Pq)
    .with_gamut(DisplayGamut::Rec2020)
    .with_peak(4000.0)
    .with_paper_white(100.0);

// --- GT7 params ---

/// Camera B's initial params: a full UCS blend with the chroma fade band pulled to
/// near zero, so the output is achromatic.
fn params_b_initial() -> GranTurismo7Params {
    GranTurismo7Params {
        blend_ratio: 1.0,
        fade_start: 0.0,
        fade_end: 0.05,
        ..Default::default()
    }
}

/// Camera B's params after the in-place mutation at `MUTATE_AT`: pure
/// per-channel curve with a linear toe (colorful again).
fn params_b_mutated() -> GranTurismo7Params {
    GranTurismo7Params {
        blend_ratio: 0.0,
        alpha: 0.5,
        mid_point: 0.4,
        linear_section: 0.3,
        toe_strength: 1.0,
        ..Default::default()
    }
}

// --- Clear colors (scene-linear Rec.709 working space) ---

const C_AB: Vec3 = Vec3::new(1.0, 2.0, 4.0); // spans > 1.0, hits the GT7 shoulder
const C_C: Vec3 = Vec3::new(4.0, 1.0, 0.05); // > 1.0 red, toe-region blue
const C_D: Vec3 = Vec3::new(0.25, 0.5, 0.75); // plain SDR
const C_E: Vec3 = Vec3::new(0.02, 0.02, 0.02); // dark: auto exposure must brighten

// --- Cross-world plumbing ---
// Statics. The render world and observers write, the test body reads after the app exits.

#[derive(Clone, Debug)]
struct CapturedImage {
    format: TextureFormat,
    width: u32,
    height: u32,
    data: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
struct ViewObs {
    label: String,
    display_target: Option<DisplayTarget>,
    dt_uniform_paper_white: Option<f32>,
    has_dt_index: bool,
    gt7_uniform: Option<Gt7ParamsUniform>,
    has_gt7_index: bool,
    has_tonemap_pipeline: bool,
}

#[derive(Clone, Debug, Default)]
struct RenderObs {
    manual_len: usize,
    manual_map: Vec<(String, DisplayTarget)>,
    views: Vec<ViewObs>,
}

static LABELS: OnceLock<Vec<(AssetId<Image>, &'static str)>> = OnceLock::new();
static RENDER_OBS: Mutex<Option<RenderObs>> = Mutex::new(None);
static CAPTURES: Mutex<Vec<(String, CapturedImage)>> = Mutex::new(Vec::new());
/// Total captures the driver waits for before exiting.
const EXPECTED_CAPTURES: usize = 7;

#[derive(Resource, Default)]
struct Frame(u32);

#[derive(Resource)]
struct TargetHandles {
    a: Handle<Image>,
    b: Handle<Image>,
    c: Handle<Image>,
    d: Handle<Image>,
    e: Handle<Image>,
}

// --- App setup ---

fn hdr_image(images: &mut Assets<Image>) -> Handle<Image> {
    let img = Image::new_target_texture(SIZE, SIZE, TextureFormat::Rgba16Float, None);
    images.add(img)
}

/// Directory for the screenshots written by `save_to_disk`: an explicit
/// override, or cargo's per-target scratch space (never the repo tree).
fn out_dir() -> PathBuf {
    std::env::var_os("GPU_FIXTURE_OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")))
}

fn capture(commands: &mut Commands, handle: &Handle<Image>, label: &'static str) {
    let mut entity = commands.spawn(Screenshot::image(handle.clone()));
    entity.observe(move |captured: On<ScreenshotCaptured>| {
        let image = &captured.image;
        CAPTURES.lock().unwrap().push((
            label.to_string(),
            CapturedImage {
                format: image.texture_descriptor.format,
                width: image.texture_descriptor.size.width,
                height: image.texture_descriptor.size.height,
                data: image.data.clone().unwrap_or_default(),
            },
        ));
    });
    // Exercise the save-to-disk path for one target. `.exr` needs the opt-in `exr`
    // cargo feature, `.hdr` ships by default. A missing codec only logs an error.
    if label == "gt7_default" {
        entity
            .observe(save_to_disk(out_dir().join("gt7_default.exr")))
            .observe(save_to_disk(out_dir().join("gt7_default.hdr")));
    }
}

fn driver(
    mut frame: ResMut<Frame>,
    handles: Res<TargetHandles>,
    mut commands: Commands,
    mut params: Query<&mut GranTurismo7Params>,
    mut exit: MessageWriter<AppExit>,
) {
    frame.0 += 1;
    match frame.0 {
        CAPTURE_1 => {
            capture(&mut commands, &handles.a, "gt7_default");
            capture(&mut commands, &handles.b, "gt7_custom_t1");
            capture(&mut commands, &handles.c, "gt7_pw100");
            capture(&mut commands, &handles.d, "sdr_none");
            capture(&mut commands, &handles.e, "auto_exposure_t1");
        }
        MUTATE_AT => {
            // In-place mutation (no remove/insert): the per-frame uniform
            // refresh must pick it up.
            for mut p in &mut params {
                *p = params_b_mutated();
            }
        }
        CAPTURE_2 => {
            capture(&mut commands, &handles.b, "gt7_custom_t2");
            capture(&mut commands, &handles.e, "auto_exposure_t2");
        }
        _ => {}
    }

    let captured = CAPTURES.lock().unwrap().len();
    if captured >= EXPECTED_CAPTURES {
        exit.write(AppExit::Success);
    } else if frame.0 > MAX_FRAMES {
        eprintln!(
            "TIMEOUT: only {captured}/{EXPECTED_CAPTURES} captures after {MAX_FRAMES} frames"
        );
        exit.write(AppExit::error());
    }
}

/// Records, every frame, what the render world resolved. The last frame's
/// observation is what the test asserts on.
fn observe_render_world(
    manual: Res<ManualDisplayTargets>,
    views: Query<(
        &ExtractedCamera,
        &ViewDisplayTarget,
        Option<&DisplayTargetUniform>,
        Has<DynamicUniformIndex<DisplayTargetUniform>>,
        Option<&Gt7ParamsUniform>,
        Has<DynamicUniformIndex<Gt7ParamsUniform>>,
        Has<ViewTonemappingPipeline>,
    )>,
) {
    let Some(labels) = LABELS.get() else {
        return;
    };
    let label_of = |id: AssetId<Image>| {
        labels
            .iter()
            .find(|(known, _)| *known == id)
            .map(|(_, l)| (*l).to_string())
    };

    let mut obs = RenderObs {
        manual_len: manual.len(),
        ..Default::default()
    };
    for (key, dt) in manual.iter() {
        if let NormalizedRenderTarget::Image(image) = key
            && let Some(label) = label_of(image.handle.id())
        {
            obs.manual_map.push((label, *dt));
        }
    }
    for (cam, vdt, dtu, has_dt_index, gt7, has_gt7_index, has_tonemap_pipeline) in &views {
        let Some(NormalizedRenderTarget::Image(image)) = &cam.target else {
            continue;
        };
        let Some(label) = label_of(image.handle.id()) else {
            continue;
        };
        obs.views.push(ViewObs {
            label,
            display_target: Some(**vdt),
            dt_uniform_paper_white: dtu.map(|u| u.paper_white_nits),
            has_dt_index,
            gt7_uniform: gt7.copied(),
            has_gt7_index,
            has_tonemap_pipeline,
        });
    }
    *RENDER_OBS.lock().unwrap() = Some(obs);
}

// --- CPU reference for the GT7 operator ---
// A port of the test-only `cpu_reference` module in bevy_core_pipeline's gt7.rs, driven
// by the same `Gt7ParamsUniform` the shader binds.

const REFERENCE_LUMINANCE: f32 = 100.0;
const GT7_PQ_M1: f32 = 0.159_301_76;
const GT7_PQ_M2: f32 = 78.84375;
const GT7_PQ_C1: f32 = 0.835_937_5;
const GT7_PQ_C2: f32 = 18.851_563;
const GT7_PQ_C3: f32 = 18.6875;
const GT7_PQ_C: f32 = 10000.0;

fn smooth_step(x: f32, edge0: f32, edge1: f32) -> f32 {
    let t = (x - edge0) / (edge1 - edge0);
    if x < edge0 {
        return 0.0;
    }
    if x > edge1 {
        return 1.0;
    }
    t * t * (3.0 - 2.0 * t)
}

fn inverse_eotf_st2084(v: f32) -> f32 {
    let physical = v * REFERENCE_LUMINANCE;
    let y = physical / GT7_PQ_C;
    let ym = ops::powf(y, GT7_PQ_M1);
    ops::exp2(GT7_PQ_M2 * (ops::log2(GT7_PQ_C1 + GT7_PQ_C2 * ym) - ops::log2(1.0 + GT7_PQ_C3 * ym)))
}

fn eotf_st2084(n: f32) -> f32 {
    let n = n.clamp(0.0, 1.0);
    let np = ops::powf(n, 1.0 / GT7_PQ_M2);
    let mut l = np - GT7_PQ_C1;
    if l < 0.0 {
        l = 0.0;
    }
    l /= GT7_PQ_C2 - GT7_PQ_C3 * np;
    l = ops::powf(l, 1.0 / GT7_PQ_M1);
    l * GT7_PQ_C / REFERENCE_LUMINANCE
}

fn rgb_to_ictcp(rgb: [f32; 3]) -> [f32; 3] {
    let l = (rgb[0] * 1688.0 + rgb[1] * 2146.0 + rgb[2] * 262.0) / 4096.0;
    let m = (rgb[0] * 683.0 + rgb[1] * 2951.0 + rgb[2] * 462.0) / 4096.0;
    let s = (rgb[0] * 99.0 + rgb[1] * 309.0 + rgb[2] * 3688.0) / 4096.0;
    let l_pq = inverse_eotf_st2084(l.max(0.0));
    let m_pq = inverse_eotf_st2084(m.max(0.0));
    let s_pq = inverse_eotf_st2084(s.max(0.0));
    [
        (2048.0 * l_pq + 2048.0 * m_pq) / 4096.0,
        (6610.0 * l_pq - 13613.0 * m_pq + 7003.0 * s_pq) / 4096.0,
        (17933.0 * l_pq - 17390.0 * m_pq - 543.0 * s_pq) / 4096.0,
    ]
}

fn ictcp_to_rgb(ictcp: [f32; 3]) -> [f32; 3] {
    let l = ictcp[0] + 0.00860904 * ictcp[1] + 0.11103 * ictcp[2];
    let m = ictcp[0] - 0.00860904 * ictcp[1] - 0.11103 * ictcp[2];
    let s = ictcp[0] + 0.560031 * ictcp[1] - 0.320627 * ictcp[2];
    let l_lin = eotf_st2084(l);
    let m_lin = eotf_st2084(m);
    let s_lin = eotf_st2084(s);
    [
        (3.43661 * l_lin - 2.50645 * m_lin + 0.0698454 * s_lin).max(0.0),
        (-0.79133 * l_lin + 1.9836 * m_lin - 0.192271 * s_lin).max(0.0),
        (-0.0259499 * l_lin - 0.0989137 * m_lin + 1.12486 * s_lin).max(0.0),
    ]
}

fn evaluate_curve(params: &Gt7ParamsUniform, x: f32) -> f32 {
    if x < 0.0 {
        return 0.0;
    }
    let weight_linear = smooth_step(x, 0.0, params.mid_point);
    let weight_toe = 1.0 - weight_linear;
    let shoulder = params.k_a + params.k_b * ops::exp(x * params.k_c);
    if x < params.linear_section * params.peak {
        let toe_mapped = params.mid_point * ops::powf(x / params.mid_point, params.toe_strength);
        weight_toe * toe_mapped + weight_linear * x
    } else {
        shoulder
    }
}

fn gt7_apply(params: &Gt7ParamsUniform, rgb: [f32; 3]) -> [f32; 3] {
    let ucs = rgb_to_ictcp(rgb);
    let skewed_rgb = [
        evaluate_curve(params, rgb[0]),
        evaluate_curve(params, rgb[1]),
        evaluate_curve(params, rgb[2]),
    ];
    let skewed_ucs = rgb_to_ictcp(skewed_rgb);
    let chroma_scale =
        1.0 - smooth_step(ucs[0] / params.peak_ucs, params.fade_start, params.fade_end);
    let scaled_ucs = [skewed_ucs[0], ucs[1] * chroma_scale, ucs[2] * chroma_scale];
    let scaled_rgb = ictcp_to_rgb(scaled_ucs);
    let mut out = [0.0; 3];
    for i in 0..3 {
        let blended =
            (1.0 - params.blend_ratio) * skewed_rgb[i] + params.blend_ratio * scaled_rgb[i];
        out[i] = params.sdr_correction_factor * blended.min(params.peak);
    }
    out
}

/// CPU prediction of one decoded screenshot pixel for a GT7 camera on a PQ/Rec.2020
/// manual target. The tonemapping pass expands Rec.709 to Rec.2020, applies the
/// x2.5 GT7 paper-white seam, then runs the operator in HDR mode. The display
/// encoding pass is an identity gamut stage for PQ/Rec.2020, a `max(0)` clip, and
/// the PQ inverse EOTF at the target's paper white. Screenshot decode runs the PQ
/// EOTF at 1.0 = 80 nits and converts Rec.2020 back to Rec.709.
fn expected_decoded(clear: Vec3, uniform: &Gt7ParamsUniform, paper_white_nits: f32) -> Vec3 {
    let rec2020 = REC709_TO_REC2020 * clear;
    let fb = rec2020 * 2.5;
    let mapped = gt7_apply(uniform, [fb.x, fb.y, fb.z]);
    let signal = [
        pq_inverse_eotf_from_nits(mapped[0].max(0.0) * paper_white_nits),
        pq_inverse_eotf_from_nits(mapped[1].max(0.0) * paper_white_nits),
        pq_inverse_eotf_from_nits(mapped[2].max(0.0) * paper_white_nits),
    ];
    let scale = PQ_MAX_LUMINANCE_NITS / SCRGB_REFERENCE_WHITE_NITS;
    let linear = Vec3::new(
        pq_eotf(signal[0]) * scale,
        pq_eotf(signal[1]) * scale,
        pq_eotf(signal[2]) * scale,
    );
    REC2020_TO_REC709 * linear
}

// --- Capture inspection ---

fn pixel_f32(cap: &CapturedImage, x: u32, y: u32) -> [f32; 4] {
    assert_eq!(cap.format, TextureFormat::Rgba32Float);
    let idx = ((y * cap.width + x) * 16) as usize;
    let mut out = [0.0; 4];
    for (i, value) in out.iter_mut().enumerate() {
        *value = f32::from_le_bytes(cap.data[idx + i * 4..idx + i * 4 + 4].try_into().unwrap());
    }
    out
}

fn pixel_u8(cap: &CapturedImage, x: u32, y: u32) -> [u8; 4] {
    let idx = ((y * cap.width + x) * 4) as usize;
    cap.data[idx..idx + 4].try_into().unwrap()
}

/// Per-channel comparison: |actual - expected| <= max(rel * |expected|, abs).
fn close(actual: Vec3, expected: Vec3, rel: f32, abs: f32) -> bool {
    (0..3).all(|i| (actual[i] - expected[i]).abs() <= f32::max(rel * expected[i].abs(), abs))
}

// --- The test ---

struct Checker {
    failed: Vec<String>,
}

impl Checker {
    fn check(&mut self, name: &str, ok: bool, detail: String) {
        if ok {
            println!("PASS: {name} — {detail}");
        } else {
            println!("FAIL: {name} — {detail}");
            self.failed.push(name.to_string());
        }
    }
}

/// Builds the app. `RenderCreation::Automatic` blocks on renderer init inside
/// `RenderPlugin::build`, so the wgpu device exists by the time this returns. With
/// no usable adapter it panics inside `add_plugins` instead.
fn build_app() -> App {
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
    .add_plugins(AutoExposurePlugin)
    .init_resource::<Frame>()
    .add_systems(Startup, setup)
    .add_systems(Update, driver);

    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        panic!("no render sub-app (no wgpu backend available)");
    };
    render_app.add_systems(Render, observe_render_world.after(RenderSystems::Render));
    app
}

/// Progress reports from the app thread back to the test thread.
enum Phase {
    /// Renderer init panicked (no adapter, device creation failed, ...).
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

/// Bound on renderer init (instance + adapter + device creation).
const INIT_TIMEOUT: Duration = Duration::from_secs(120);
/// Bound on the render loop (pipeline compilation + `MAX_FRAMES` frames).
const RUN_TIMEOUT: Duration = Duration::from_secs(480);

#[test]
fn gpu_hdr_fixture() {
    // Build and run on a helper thread so a wedged driver cannot hang the test. The
    // test thread only ever waits with a timeout. Winit is disabled, so nothing
    // needs the main thread.
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        // Silence the default panic hook while building. A missing adapter panics
        // inside `add_plugins`, which is the expected skip path, not a failure.
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let built = catch_unwind(AssertUnwindSafe(build_app));
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
            return;
        }
        Err(_) => {
            eprintln!(
                "SKIP gpu_hdr_fixture: renderer init did not complete within {INIT_TIMEOUT:?}"
            );
            return;
        }
        Ok(Phase::Finished(_) | Phase::RunPanicked(_)) => {
            unreachable!("app finished before reporting init")
        }
    }

    // Renderer init succeeded: from here on, problems are real failures.
    let exit = match rx.recv_timeout(RUN_TIMEOUT) {
        Ok(Phase::Finished(exit)) => exit,
        Ok(Phase::RunPanicked(reason)) => panic!("app panicked while running: {reason}"),
        Err(_) => panic!("app did not finish within {RUN_TIMEOUT:?}"),
        Ok(Phase::InitOk | Phase::InitFailed(_)) => unreachable!("init reported twice"),
    };
    assert_eq!(exit, AppExit::Success, "app did not exit cleanly");

    let obs = RENDER_OBS
        .lock()
        .unwrap()
        .clone()
        .expect("render-world observation recorded");
    let captures: HashMap<String, CapturedImage> = CAPTURES.lock().unwrap().drain(..).collect();

    let view = |label: &str| -> ViewObs {
        obs.views
            .iter()
            .find(|v| v.label == label)
            .cloned()
            .unwrap_or_else(|| panic!("no render-world view observed for {label}"))
    };
    let cap = |label: &str| -> &CapturedImage {
        captures
            .get(label)
            .unwrap_or_else(|| panic!("no capture for {label}"))
    };
    // Uniform image: sample three pixels and require agreement, return center.
    let decoded_px = |label: &str| -> (Vec3, f32) {
        let c = cap(label);
        let center = pixel_f32(c, c.width / 2, c.height / 2);
        let corner_a = pixel_f32(c, 2, 2);
        let corner_b = pixel_f32(c, c.width - 3, c.height - 3);
        for other in [corner_a, corner_b] {
            for i in 0..4 {
                assert!(
                    (center[i] - other[i]).abs() < 1e-4,
                    "{label}: image not uniform: center {center:?} vs {other:?}"
                );
            }
        }
        (Vec3::new(center[0], center[1], center[2]), center[3])
    };

    let mut c = Checker { failed: Vec::new() };

    // --- Check 1: manual display targets reach the render world ---
    c.check(
        "1a extracted ManualDisplayTargets map reaches the render world",
        obs.manual_len == 4
            && [
                ("gt7_default", T_PQ_203),
                ("gt7_custom", T_PQ_203),
                ("gt7_pw100", T_PQ_100),
                ("auto_exposure", T_PQ_203),
            ]
            .iter()
            .all(|(label, want)| {
                obs.manual_map
                    .iter()
                    .any(|(seen, dt)| seen == label && dt == want)
            }),
        format!(
            "render-world map has {} entries: {:?}",
            obs.manual_len, obs.manual_map
        ),
    );
    for label in ["gt7_default", "gt7_custom", "auto_exposure"] {
        c.check(
            &format!("1b {label} view resolves the registered PQ target"),
            view(label).display_target == Some(T_PQ_203),
            format!("ViewDisplayTarget = {:?}", view(label).display_target),
        );
    }
    c.check(
        "1b gt7_pw100 view resolves the registered PQ target",
        view("gt7_pw100").display_target == Some(T_PQ_100),
        format!("ViewDisplayTarget = {:?}", view("gt7_pw100").display_target),
    );
    // A PQ target decoding to Rgba32Float is the evidence the map entry was found. The
    // unregistered SDR target passes through as Rgba8UnormSrgb.
    c.check(
        "1c manual_target_decode keys off the registered map entries",
        [
            "gt7_default",
            "gt7_custom_t1",
            "gt7_pw100",
            "auto_exposure_t1",
        ]
        .iter()
        .all(|l| cap(l).format == TextureFormat::Rgba32Float)
            && cap("sdr_none").format == TextureFormat::Rgba8UnormSrgb,
        format!(
            "gt7_default={:?} sdr_none={:?}",
            cap("gt7_default").format,
            cap("sdr_none").format
        ),
    );

    // --- Check 2: HDR-gated display-target uniform ---
    for (label, want_pw) in [
        ("gt7_default", 203.0),
        ("gt7_custom", 203.0),
        ("gt7_pw100", 100.0),
        ("auto_exposure", 203.0),
    ] {
        let v = view(label);
        c.check(
            &format!("2a {label} carries DisplayTargetUniform + dynamic index"),
            v.dt_uniform_paper_white == Some(want_pw) && v.has_dt_index,
            format!(
                "paper_white={:?} has_index={}",
                v.dt_uniform_paper_white, v.has_dt_index
            ),
        );
    }
    let d = view("sdr_none");
    c.check(
        "2b plain SDR view has NO display-target uniform and resolves SDR_SRGB",
        d.display_target == Some(DisplayTarget::SDR_SRGB)
            && d.dt_uniform_paper_white.is_none()
            && d.gt7_uniform.is_none()
            && !d.has_tonemap_pipeline,
        format!("{d:?}"),
    );
    // The SDR camera renders fine without the uniform: Tonemapping::None +
    // hardware sRGB encode => bytes are srgb_oetf(clear) quantized.
    {
        let sdr = cap("sdr_none");
        let px = pixel_u8(sdr, sdr.width / 2, sdr.height / 2);
        let want = [
            (srgb_oetf(C_D.x) * 255.0).round() as i32,
            (srgb_oetf(C_D.y) * 255.0).round() as i32,
            (srgb_oetf(C_D.z) * 255.0).round() as i32,
            255,
        ];
        let ok = (0..4).all(|i| (px[i] as i32 - want[i]).abs() <= 2);
        c.check(
            "2b SDR camera renders correctly without the uniform",
            ok,
            format!("bytes {px:?} vs expected {want:?} (±2)"),
        );
    }

    // --- Check 3: GT7 uniform and node-side tonemapping veto ---
    let defaults = GranTurismo7Params::default();
    for (label, target, params) in [
        ("gt7_default", T_PQ_203, defaults),
        ("gt7_custom", T_PQ_203, params_b_mutated()), // post-mutation frame
        ("gt7_pw100", T_PQ_100, defaults),
        ("auto_exposure", T_PQ_203, defaults),
    ] {
        let want = Gt7ParamsUniform::new(&target, &params);
        let v = view(label);
        c.check(
            &format!("3a {label} Gt7ParamsUniform matches CPU-computed uniform"),
            v.gt7_uniform == Some(want) && v.has_gt7_index,
            format!("got {:?}\n        want {:?}", v.gt7_uniform, Some(want)),
        );
        c.check(
            &format!("3b {label} tone-maps node-side (TonemappingPass veto path)"),
            v.has_tonemap_pipeline,
            format!("has ViewTonemappingPipeline = {}", v.has_tonemap_pipeline),
        );
    }
    // Custom params visibly affect output: A (defaults) vs B (custom), same
    // target calibration, same clear color.
    {
        let (a, _) = decoded_px("gt7_default");
        let (b1, _) = decoded_px("gt7_custom_t1");
        let (b2, _) = decoded_px("gt7_custom_t2");
        let rel_diff = |x: Vec3, y: Vec3| {
            (0..3)
                .map(|i| (x[i] - y[i]).abs() / f32::max(x[i].abs().max(y[i].abs()), 1e-3))
                .fold(0.0f32, f32::max)
        };
        c.check(
            "3c custom GranTurismo7Params visibly change the output (A vs B)",
            rel_diff(a, b1) > 0.05,
            format!("a={a:?} b1={b1:?} max rel diff {:.4}", rel_diff(a, b1)),
        );
        c.check(
            "3d in-place params mutation refreshes the uniform (B t1 vs t2)",
            rel_diff(b1, b2) > 0.05,
            format!("b1={b1:?} b2={b2:?} max rel diff {:.4}", rel_diff(b1, b2)),
        );
    }

    // --- Check 4: PQ encode/decode round trip vs the CPU reference chain ---
    let round_trip = |name: &str,
                      label: &str,
                      clear: Vec3,
                      target: DisplayTarget,
                      params: &GranTurismo7Params,
                      checker: &mut Checker| {
        let uniform = Gt7ParamsUniform::new(&target, params);
        let want = expected_decoded(clear, &uniform, target.paper_white_nits);
        let (got, alpha) = decoded_px(label);
        let max_dev = (0..3)
            .map(|i| (got[i] - want[i]).abs() / f32::max(want[i].abs(), 1e-3))
            .fold(0.0f32, f32::max);
        checker.check(
            name,
            close(got, want, 0.015, 0.01) && (alpha - 1.0).abs() < 1e-3,
            format!(
                "decoded {got:?} vs CPU chain {want:?} (max rel dev {max_dev:.5}, alpha {alpha})"
            ),
        );
    };
    round_trip(
        "4a PQ round trip, defaults @ pw 203 / peak 1000 (shoulder region)",
        "gt7_default",
        C_AB,
        T_PQ_203,
        &defaults,
        &mut c,
    );
    round_trip(
        "4b PQ round trip, custom params t1 (achromatic fade)",
        "gt7_custom_t1",
        C_AB,
        T_PQ_203,
        &params_b_initial(),
        &mut c,
    );
    round_trip(
        "4c PQ round trip, custom params t2 (post-mutation)",
        "gt7_custom_t2",
        C_AB,
        T_PQ_203,
        &params_b_mutated(),
        &mut c,
    );
    round_trip(
        "4d PQ round trip, defaults @ pw 100 / peak 4000 (x2.5 seam, >1 input)",
        "gt7_pw100",
        C_C,
        T_PQ_100,
        &defaults,
        &mut c,
    );

    // --- Check 5: auto-exposure smoke ---
    {
        let (e1, _) = decoded_px("auto_exposure_t1");
        let (e2, _) = decoded_px("auto_exposure_t2");
        let finite = e1.is_finite() && e2.is_finite();
        let moved = (0..3)
            .any(|i| (e1[i] - e2[i]).abs() / f32::max(e1[i].abs().max(e2[i].abs()), 1e-4) > 0.02);
        c.check(
            "5 auto-exposure + auto-white-balance: renders, exposure moves",
            finite && moved,
            format!("t1={e1:?} t2={e2:?}"),
        );
    }

    assert!(
        c.failed.is_empty(),
        "{} check(s) FAILED: {:?}",
        c.failed.len(),
        c.failed
    );
    println!("ALL CHECKS PASSED");
}

/// Spawns the scene. The unregistered target must fall back to `DisplayTarget::SDR_SRGB`.
fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut manual: ResMut<ManualDisplayTargets>,
) {
    let images = &mut *images;
    let commands = &mut commands;
    let a = hdr_image(images);
    let b = hdr_image(images);
    let c = hdr_image(images);
    let e = hdr_image(images);
    let d = images.add(Image::new_target_texture(
        SIZE,
        SIZE,
        TextureFormat::Rgba8UnormSrgb,
        None,
    ));

    let key = |h: &Handle<Image>| {
        RenderTarget::Image(h.clone().into())
            .normalize(None)
            .expect("image render target normalizes without a window")
    };
    manual.insert(key(&a), T_PQ_203);
    manual.insert(key(&b), T_PQ_203);
    manual.insert(key(&c), T_PQ_100);
    manual.insert(key(&e), T_PQ_203);

    LABELS
        .set(vec![
            (a.id(), "gt7_default"),
            (b.id(), "gt7_custom"),
            (c.id(), "gt7_pw100"),
            (d.id(), "sdr_none"),
            (e.id(), "auto_exposure"),
        ])
        .expect("labels set once");

    let camera = |target: &Handle<Image>, order: isize, clear: Vec3| {
        (
            Camera {
                clear_color: ClearColorConfig::Custom(Color::linear_rgb(clear.x, clear.y, clear.z)),
                order,
                ..default()
            },
            RenderTarget::Image(target.clone().into()),
        )
    };

    commands.spawn((
        Camera3d::default(),
        camera(&a, 0, C_AB),
        Hdr,
        Tonemapping::GranTurismo7,
        DebandDither::Disabled,
        Msaa::Off,
    ));
    commands.spawn((
        Camera3d::default(),
        camera(&b, 1, C_AB),
        Hdr,
        Tonemapping::GranTurismo7,
        params_b_initial(),
        DebandDither::Disabled,
        Msaa::Off,
    ));
    commands.spawn((
        Camera3d::default(),
        camera(&c, 2, C_C),
        Hdr,
        Tonemapping::GranTurismo7,
        DebandDither::Disabled,
        Msaa::Off,
    ));
    commands.spawn((
        Camera3d::default(),
        camera(&d, 3, C_D),
        Tonemapping::None,
        DebandDither::Disabled,
        Msaa::Off,
    ));
    commands.spawn((
        Camera3d::default(),
        camera(&e, 4, C_E),
        Hdr,
        Tonemapping::GranTurismo7,
        AutoExposure::default(),
        AutoWhiteBalance::default(),
        DebandDither::Disabled,
        Msaa::Off,
    ));

    commands.insert_resource(TargetHandles { a, b, c, d, e });
}
