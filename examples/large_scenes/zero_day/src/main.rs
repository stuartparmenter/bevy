//! Beeple's "Zero-Day" sci-fi corridor (NVIDIA ORCA), path-traced with Bevy Solari.
//!
//! Zero-Day is authored to be lit entirely by ~10,000 emissive triangles with no
//! punctual lights -- the way NVIDIA's original real-time ["Measure 1"](https://www.youtube.com/watch?v=0WE7CgJMuVc)
//! demo renders it. That only works with a path tracer, so this example requires Bevy
//! Solari: the emissive meshes become real area lights with global illumination. The
//! HDR result rides the Gran Turismo 7 tonemapper (`Tonemapping::GranTurismo7` +
//! `Bloom::GT7_GLARE`, Rec.2020 working space, HDR swapchain via the shared
//! `HdrPlugin`).
//!
//! It plays the film's take -- ~550 animated objects (baked to ~2337 glTF clips) plus
//! the film's camera flythrough -- and drives the render camera from that camera. No
//! ORCA measure ships animated *lights*, though: the film's emissive pulsing was
//! procedural in Octane, isn't in any exported asset, and couldn't come through glTF
//! anyway (Bevy doesn't support `KHR_animation_pointer`). So `animate_emissive` fakes
//! it -- a wave of light sweeping the corridor's emissive panels as a stand-in.
//!
//! `--scene` picks the ORCA measure (`measure_one` by default; also `measure_seven` and
//! `measure_seven_colored_lights`). Each is a separate `.glb` produced by `convert.py`;
//! they differ in geometry and emissive palette, not in whether the lights animate.
//!
//! Requires a ray-tracing capable GPU (Solari currently needs the Vulkan backend in
//! wgpu). DLSS Ray Reconstruction denoises the output when the `dlss` feature (the
//! default) is enabled on a supported NVIDIA GPU.
//!
//! Controls: `C` toggles the film flythrough vs. free-fly (WASD + mouse), `N` toggles
//! DLSS Ray Reconstruction, and `B` runs a short benchmark (printed to the console).

// The engine allows these workspace-wide; standalone example crates don't inherit that
// lint config (they rely on `println!`, which the workspace lints forbid), so allow them
// here -- ECS systems naturally take many args and complex query types.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::collections::{HashMap, HashSet};
use std::f32::consts::{PI, TAU};
use std::time::Instant;

use argh::FromArgs;
use bevy::{
    asset::LoadState,
    camera::{CameraMainTextureUsages, Hdr},
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    core_pipeline::tonemapping::{GranTurismo7Params, Tonemapping},
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    gltf::Gltf,
    math::ops,
    mesh::Indices,
    post_process::bloom::Bloom,
    prelude::*,
    render::{
        render_resource::TextureUsages, working_color_space::WorkingColorSpace, RenderPlugin,
    },
    solari::prelude::{RaytracingMesh3d, SolariLighting, SolariPlugins},
    window::{
        AutoField, DisplayCalibrationPolicy, DisplayTarget, PresentMode, PrimaryWindow,
        WindowResolution,
    },
    winit::WinitSettings,
    world_serialization::WorldInstanceReady,
};

// DLSS Ray Reconstruction denoises the Solari output when the `dlss` feature is on
// (the default). Needs an NVIDIA RTX GPU; without it the path tracer still runs.
// `TemporalJitter`/`MipBias` are inserted by DLSS and removed alongside it when the
// `N` key turns the denoiser off.
#[cfg(feature = "dlss")]
use bevy::{
    anti_alias::dlss::{
        Dlss, DlssPerfQualityMode, DlssProjectId, DlssRayReconstructionFeature,
        DlssRayReconstructionSupported,
    },
    render::camera::{MipBias, TemporalJitter},
};

// Opt-in HDR setup shared with the other HDR examples: keeps the primary window's
// `DisplayTarget` on the best transfer the surface advertises, else SDR.
#[path = "../../../helpers/hdr.rs"]
mod hdr;

/// Config
#[derive(FromArgs, Resource, Clone)]
pub struct Args {
    /// which ORCA measure to load: measure_one (default), measure_seven, or
    /// measure_seven_colored_lights. Each is a separate `.glb` built by `convert.py`.
    #[argh(option, default = "Scene::MeasureOne")]
    scene: Scene,

    /// emissive multiplier for the accent panels (they are the scene's only lights, so
    /// they must be bright to illuminate the space). Defaults per measure (measure_seven is
    /// a much larger, more open space, so its default is higher); override to taste.
    #[argh(option)]
    emissive: Option<f32>,

    /// disable the synthetic emissive pulse. By default a wave of light sweeps the panels
    /// to evoke the film's animated lights (that animation isn't in the exported asset).
    #[argh(switch)]
    no_pulse: bool,

    /// render resolution as `WxH` (default 1920x1080). Solari cost scales with pixel count,
    /// so lower it (e.g. `1280x720`) to trade sharpness for framerate on the heavy measures.
    #[argh(option)]
    resolution: Option<String>,

    /// DLSS quality mode: auto (default), dlaa, quality, balanced, performance, or
    /// ultra_performance. Lower renders at a smaller internal resolution for more framerate.
    #[cfg(feature = "dlss")]
    #[argh(option)]
    dlss_quality: Option<String>,
}

/// Which ORCA "Zero-Day" measure to load. Each converts to its own self-contained `.glb`
/// (see `convert.py` and the README); the flythrough camera and emissive handling are the
/// same across all three, so only the asset filename changes.
// The shared `Measure` prefix mirrors the ORCA asset names, so keep it.
#[allow(clippy::enum_variant_names)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scene {
    MeasureOne,
    MeasureSeven,
    MeasureSevenColoredLights,
}

impl Scene {
    /// The `.glb` file this measure loads from `assets/` (all `.gitignore`d).
    fn glb(self) -> &'static str {
        match self {
            Scene::MeasureOne => "zero_day_measure_one.glb",
            Scene::MeasureSeven => "zero_day_measure_seven.glb",
            Scene::MeasureSevenColoredLights => "zero_day_measure_seven_colored_lights.glb",
        }
    }

    /// Default emissive multiplier (overridable with `--emissive`). Measure One is a tight
    /// corridor; Measure Seven is a much larger, more open shaft whose panels must be far
    /// brighter to carry the space.
    fn default_emissive(self) -> f32 {
        match self {
            Scene::MeasureOne => 150_000.0,
            Scene::MeasureSeven | Scene::MeasureSevenColoredLights => 600_000.0,
        }
    }
}

impl Args {
    /// Emissive multiplier: the `--emissive` override, else the measure's default.
    fn emissive(&self) -> f32 {
        self.emissive.unwrap_or(self.scene.default_emissive())
    }

    /// Parsed `--resolution WxH`, else 1920x1080. Falls back to the default (with a warning)
    /// if the value can't be parsed.
    fn resolution(&self) -> (u32, u32) {
        let Some(s) = &self.resolution else {
            return (1920, 1080);
        };
        let parsed = s
            .split_once(['x', 'X'])
            .and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?)));
        parsed.unwrap_or_else(|| {
            warn!("zero_day: could not parse --resolution `{s}`; using 1920x1080");
            (1920, 1080)
        })
    }

    /// Parsed `--dlss-quality`, else `Auto`. Lower modes render at a smaller internal
    /// resolution (faster, softer).
    #[cfg(feature = "dlss")]
    fn dlss_perf_quality(&self) -> DlssPerfQualityMode {
        match self.dlss_quality.as_deref() {
            None | Some("auto") => DlssPerfQualityMode::Auto,
            Some("dlaa") => DlssPerfQualityMode::Dlaa,
            Some("quality") => DlssPerfQualityMode::Quality,
            Some("balanced") => DlssPerfQualityMode::Balanced,
            Some("performance") => DlssPerfQualityMode::Performance,
            Some("ultra_performance") => DlssPerfQualityMode::UltraPerformance,
            Some(other) => {
                warn!("zero_day: unknown --dlss-quality `{other}`; using auto");
                DlssPerfQualityMode::Auto
            }
        }
    }
}

impl argh::FromArgValue for Scene {
    fn from_arg_value(value: &str) -> Result<Self, String> {
        match value {
            "measure_one" => Ok(Scene::MeasureOne),
            "measure_seven" => Ok(Scene::MeasureSeven),
            "measure_seven_colored_lights" => Ok(Scene::MeasureSevenColoredLights),
            other => Err(format!(
                "unknown scene `{other}`; expected measure_one, measure_seven, or \
                 measure_seven_colored_lights"
            )),
        }
    }
}

// Synthetic-pulse tuning (see `animate_emissive`). The real film sequences its lights
// procedurally; this fakes that with a wave travelling along the corridor. All tunable.
/// Temporal rate of the wave (rad/s).
const PULSE_FREQ: f32 = 2.0;
/// Spatial frequency along the corridor's Z axis (rad/world-unit): sets the wavelength,
/// so panels at different depths flare at different times instead of all together.
const PULSE_WAVE_NUMBER: f32 = 0.05;
/// Exponent that sharpens the sine into discrete flares (higher = snappier "pops").
const PULSE_SHARPNESS: f32 = 2.0;
/// Dim/bright bounds (as a fraction of each panel's base emissive). The floor keeps the
/// corridor lit between flares; the peak overshoots 1.0 so passing panels visibly bloom.
const PULSE_FLOOR: f32 = 0.4;
const PULSE_PEAK: f32 = 1.8;
/// Golden angle (rad): scatters a stable per-panel phase so same-depth panels don't flare
/// in lockstep.
const PULSE_PHASE_STRIDE: f32 = 2.399_963_2;

fn main() {
    let args: Args = argh::from_env();
    let (win_w, win_h) = args.resolution();

    let mut app = App::new();

    // DLSS reads its project id during renderer init, so set it before the plugins.
    #[cfg(feature = "dlss")]
    app.insert_resource(DlssProjectId(bevy::asset::uuid::uuid!(
        "b1a7c0de-4d2f-4e6a-9b3c-0d1e2f3a4b5c"
    )));

    app.insert_resource(ClearColor(Color::BLACK))
        // All light comes from the emissive meshes (via Solari); no ambient fill.
        .insert_resource(GlobalAmbientLight::NONE)
        .insert_resource(args)
        .insert_resource(WinitSettings::continuous())
        .init_resource::<Cinematic>()
        .init_resource::<FilmLength>()
        .init_resource::<FrameStats>()
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: PresentMode::Immediate,
                        resolution: WindowResolution::new(win_w, win_h)
                            .with_scale_factor_override(1.0),
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    // GT7's native working space.
                    working_color_space: WorkingColorSpace::Rec2020,
                    ..default()
                }),
            SolariPlugins,
            FreeCameraPlugin,
            // Auto-select the best HDR output the surface can present, else SDR.
            hdr::HdrPlugin::default(),
            // Longer history so the HUD/benchmark can report 1% lows.
            FrameTimeDiagnosticsPlugin {
                max_history_length: 1000,
                ..default()
            },
        ))
        .add_systems(Startup, (setup_hdr_calibration, setup))
        // Waits for the glTF to fully load, then spawns the scene (see the fn doc).
        .add_systems(Update, spawn_scene_when_ready)
        .add_systems(
            Update,
            (
                toggle_flythrough,
                drive_flythrough,
                animate_emissive,
                frame_stats,
                benchmark,
                update_hud,
            )
                .chain(),
        );

    // Runtime DLSS on/off (`N`); only meaningful when compiled with the `dlss` feature.
    #[cfg(feature = "dlss")]
    app.add_systems(Update, toggle_denoiser);

    app.run();
}

/// Trusts the calibrated monitor for HDR luminance: hands peak and black level to the
/// OS so GT7 tone maps against the panel's real headroom, and seeds a 200-nit HDR
/// reference paper white. Gamut stays paired with the `HdrPlugin`-chosen transfer.
fn setup_hdr_calibration(
    window: Single<(Entity, &mut DisplayTarget), With<PrimaryWindow>>,
    mut commands: Commands,
) {
    let (window, mut display_target) = window.into_inner();
    display_target.paper_white_nits = 200.0;
    commands.entity(window).insert(DisplayCalibrationPolicy {
        paper_white: AutoField::Keep,
        peak_luminance: AutoField::Auto,
        min_luminance: AutoField::Auto,
        gamut: AutoField::Keep,
    });
}

// --- Components / resources --------------------------------------------------------

/// Our render camera; owns the HDR/GT7/Solari stack.
#[derive(Component)]
struct RenderCamera;

/// The film's imported camera (named per measure -- `DynamicCamera2`, `DynamicCamera`,
/// ...), stripped to a transform source that `drive_flythrough` follows.
#[derive(Component)]
struct FilmCamera;

/// The on-screen readout.
#[derive(Component)]
struct HudText;

/// Keeps the whole glTF loaded so its animation clips stay reachable.
#[derive(Resource)]
struct SceneGltf(Handle<Gltf>);

/// One emissive panel instance. `proc_scene` gives every emissive instance its own
/// material clone (they otherwise share a handful of materials) so `animate_emissive`
/// can drive each independently -- keyed to its world position for a corridor-length
/// wave, offset by a stable per-panel `phase` so neighbours don't flare in lockstep.
#[derive(Component)]
struct EmissivePanel {
    /// The boosted base emissive (`proc_scene`'s output); the pulse scales this.
    base: LinearRgba,
    /// Stable per-panel phase offset (radians).
    phase: f32,
}

/// Length of the loaded film take (seconds), from the longest animation clip. Set in
/// `start_animation`; used to size the `B` benchmark so it covers exactly one loop
/// regardless of which measure is loaded (they run to different frame counts).
#[derive(Resource, Default)]
struct FilmLength(f32);

/// Rolling frame-time stats for the HUD and the `B` benchmark (Solari is heavy, so
/// these are worth watching). `one_percent_high_ms` is the worst 1% of frames.
#[derive(Resource, Default)]
struct FrameStats {
    avg_ms: f64,
    one_percent_low_ms: f64,
    one_percent_high_ms: f64,
}

/// Whether the camera follows the film's animated camera (the default) or free-flies.
/// Toggled with `C`.
#[derive(Resource)]
struct Cinematic {
    active: bool,
}

impl Default for Cinematic {
    fn default() -> Self {
        Self { active: true }
    }
}

// --- Setup -------------------------------------------------------------------------

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    args: Res<Args>,
    #[cfg(feature = "dlss")] dlss_rr_supported: Option<Res<DlssRayReconstructionSupported>>,
) {
    let glb = args.scene.glb();
    println!("Loading Zero-Day `{glb}` (this is a large scene; give it a moment)");

    // Load the whole glTF (not just the scene); `spawn_scene_when_ready` spawns the scene
    // once this finishes loading. Holding the whole `Gltf` also keeps its animation clips
    // reachable for `start_animation`.
    commands.insert_resource(SceneGltf(asset_server.load(glb.to_string())));

    // Camera. Its field of view and near plane are overwritten with the film camera's in
    // `setup_flythrough_camera` (far is kept); the transform is the fallback view held until
    // the flythrough is actually driving.
    let mut cam = commands.spawn((
        Camera3d::default(),
        // The imported film camera also spawns as an active camera (order 0). A
        // higher order makes ours always win until it is stripped, so we never flash
        // the film camera's un-animated rest pose.
        Camera {
            order: 1,
            ..default()
        },
        Hdr,
        // Solari (and DLSS) require MSAA off.
        Msaa::Off,
        Transform::from_xyz(-27.0, 8.0, 70.0).looking_at(Vec3::new(-27.0, 8.0, -150.0), Vec3::Y),
        Projection::Perspective(PerspectiveProjection {
            fov: PI / 3.0,
            near: 0.1,
            far: 2000.0,
            ..default()
        }),
        // Solari writes its result into the main texture via a storage binding.
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        SolariLighting::default(),
        RenderCamera,
    ));
    cam.insert((
        // GT7 tonemapping + physically based glare drive the HDR output.
        Tonemapping::GranTurismo7,
        GranTurismo7Params::default(),
        Bloom {
            intensity: 0.15,
            ..Bloom::GT7_GLARE
        },
        FreeCamera {
            walk_speed: 20.0,
            run_speed: 60.0,
            ..default()
        },
    ));
    // DLSS Ray Reconstruction denoises the path-traced output when supported.
    #[cfg(feature = "dlss")]
    if dlss_rr_supported.is_some() {
        cam.insert(dlss_rr(args.dlss_perf_quality()));
    }

    // HUD: rides the single GT7-tonemapped 3D camera (a second 2D camera with
    // `Tonemapping::None` would desaturate under Rec.2020).
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(14.0),
            ..default()
        },
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.8)),
        Node {
            position_type: PositionType::Absolute,
            top: px(8.0),
            left: px(8.0),
            ..default()
        },
        HudText,
    ));
}

/// One-shot latch for `spawn_scene_when_ready`.
#[derive(Default)]
struct SceneSpawn {
    spawned: bool,
    reported_error: bool,
}

/// Spawns the scene once its glTF is loaded with all dependencies (materials, meshes,
/// animation clips), then latches so it runs once. The `WorldInstanceReady` observers read
/// those sub-assets directly, so the scene must not spawn before they are all present: an
/// unboosted emissive leaves the emissive-only corridor black, and a missing clip leaves it
/// frozen. A load failure (usually a not-yet-converted `.glb`) is logged once.
fn spawn_scene_when_ready(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    args: Res<Args>,
    scene_gltf: Res<SceneGltf>,
    mut state: Local<SceneSpawn>,
) {
    if state.spawned {
        return;
    }
    if let LoadState::Failed(err) = asset_server.load_state(&scene_gltf.0) {
        if !state.reported_error {
            state.reported_error = true;
            error!(
                "zero_day: failed to load `{}` ({err}). Convert it first with convert.py \
                 (see the example README).",
                args.scene.glb()
            );
        }
        return;
    }
    if !asset_server.is_loaded_with_dependencies(&scene_gltf.0) {
        return;
    }
    state.spawned = true;
    commands
        .spawn(WorldAssetRoot(
            asset_server.load(format!("{}#Scene0", args.scene.glb())),
        ))
        // Repairs + boosts the emissive materials, one clone per emissive instance.
        .observe(proc_scene)
        // Turns the film camera into the flythrough source.
        .observe(setup_flythrough_camera)
        // Tags every mesh `RaytracingMesh3d` so Solari can trace against it.
        .observe(setup_raytracing_meshes)
        // Plays every animation clip once the scene (and its player) exist.
        .observe(start_animation);
}

// --- Scene processing (on load) ----------------------------------------------------

/// Repairs the imported materials and boosts the emissive ones so the panels act as
/// bright Solari light sources, drops any imported lights, then gives each emissive
/// *instance* its own material so the pulse can animate them independently.
///
/// `convert.py` handles most of the FBX quirks at the asset level, but deliberately
/// leaves the normals DirectX-convention (green flipped vs glTF) for the engine to
/// flip -- matching how the `bistro` example handles Sponza.
///
/// Two passes over the scene:
/// 1. Repair + boost each unique material once (instanced meshes share a material handle,
///    so a per-entity boost would multiply a shared emissive many times, to infinity).
/// 2. For each *emissive* instance, swap in a clone of its (now boosted) material and tag
///    it `EmissivePanel`. Only ~230 instances are emissive, so this is cheap -- and Solari
///    reads emissive per material asset, so distinct clones are what let neighbouring
///    panels flare at different times (a shared handle can only pulse in lockstep).
fn proc_scene(
    scene_ready: On<WorldInstanceReady>,
    mut commands: Commands,
    has_std_mat: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    lights: Query<Entity, Or<(With<PointLight>, With<DirectionalLight>, With<SpotLight>)>>,
    children: Query<&Children>,
    args: Res<Args>,
    mut processed: Local<HashSet<AssetId<StandardMaterial>>>,
) {
    // Pass 1: repair + boost each unique material; remember the boosted emissive per
    // material id so pass 2 can seed each instance's `EmissivePanel`.
    let mut emissive_bases: HashMap<AssetId<StandardMaterial>, LinearRgba> = HashMap::new();
    for entity in children.iter_descendants(scene_ready.entity) {
        if let Ok(mat_h) = has_std_mat.get(entity)
            && processed.insert(mat_h.id())
            && let Some(mut mat) = materials.get_mut(mat_h)
        {
            // DirectX-convention normals (see the fn doc).
            mat.flip_normal_map_y = true;

            // A material with an emissive texture but a black factor emits nothing;
            // promote it to white so the texture lights up.
            if mat.emissive == LinearRgba::BLACK && mat.emissive_texture.is_some() {
                mat.emissive = LinearRgba::WHITE;
            }
            if mat.emissive != LinearRgba::BLACK {
                mat.emissive *= args.emissive();
                emissive_bases.insert(mat_h.id(), mat.emissive);
            }
        }

        // Zero-Day is lit purely by its emissives; drop any imported lights.
        if lights.get(entity).is_ok() {
            commands.entity(entity).despawn();
        }
    }

    // Pass 2: per-instance material clone + `EmissivePanel` for the emissive instances.
    for entity in children.iter_descendants(scene_ready.entity) {
        let Ok(mat_h) = has_std_mat.get(entity) else {
            continue;
        };
        let Some(base) = emissive_bases.get(&mat_h.id()).copied() else {
            continue;
        };
        let Some(material) = materials.get(mat_h.id()).cloned() else {
            continue;
        };
        let handle = materials.add(material);
        // Stable per-panel seed from the entity bits (its `index()` isn't a primitive).
        let phase = ((entity.to_bits() & 0xffff) as f32 * PULSE_PHASE_STRIDE) % TAU;
        commands
            .entity(entity)
            .insert((MeshMaterial3d(handle), EmissivePanel { base, phase }));
    }
}

/// Repurposes the imported film camera as the flythrough transform source: adopts its
/// field of view and near plane on the render camera, then strips its render components so
/// only our camera draws (it keeps its animated transform, which `drive_flythrough`
/// follows). The camera entity is present at `WorldInstanceReady`, so this is reliable.
///
/// FOV and near are copied, but not the film's far plane. The film ships near=0.001/far=100;
/// we keep our own far=2000 because the measures extend far past 100 units from the camera
/// (measure_seven's shaft is ~700 units deep) and far=100 would cut the far end short. The
/// tiny near, though, is essential and must be copied: Solari's realtime path resolves
/// primary visibility from a *rasterized* depth/G-buffer prepass, which clips anything
/// closer than the near plane. The flythrough grazes geometry within ~0.02 units (measure
/// seven threads through packed machinery), so a coarse near=0.1 clips those surfaces to
/// empty depth and the camera appears to see straight through them. The film's 0.001 is
/// authored to exactly clear that geometry, and Bevy's reverse-z depth keeps precision fine
/// at 0.001/2000.
///
/// Every measure has exactly one camera, but we still tag only the first as `FilmCamera`
/// (and strip the rest) so `drive_flythrough`'s `single()` can never go ambiguous.
fn setup_flythrough_camera(
    scene_ready: On<WorldInstanceReady>,
    children: Query<&Children>,
    film_cameras: Query<(Entity, &Projection), (With<Camera>, Without<RenderCamera>)>,
    mut render_projection: Query<&mut Projection, With<RenderCamera>>,
    mut commands: Commands,
) {
    let mut film_optics = None;
    let mut tagged = false;
    for entity in children.iter_descendants(scene_ready.entity) {
        if let Ok((camera, projection)) = film_cameras.get(entity) {
            let mut camera = commands.entity(camera);
            camera.remove::<(Camera3d, Camera, Projection)>();
            if !tagged {
                tagged = true;
                camera.insert(FilmCamera);
                if let Projection::Perspective(p) = projection {
                    film_optics = Some((p.fov, p.near));
                }
            }
        }
    }
    if let (Some((fov, near)), Ok(mut render_projection)) =
        (film_optics, render_projection.single_mut())
        && let Projection::Perspective(p) = &mut *render_projection
    {
        p.fov = fov;
        p.near = near;
    }
}

/// Makes the scene's meshes ray-traceable for Solari: tags each with `RaytracingMesh3d`
/// and ensures the UV0 / tangent / U32-index layout the tracer needs.
fn setup_raytracing_meshes(
    scene_ready: On<WorldInstanceReady>,
    children: Query<&Children>,
    mesh_query: Query<&Mesh3d>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut commands: Commands,
) {
    for descendant in children.iter_descendants(scene_ready.entity) {
        let Ok(Mesh3d(mesh_handle)) = mesh_query.get(descendant) else {
            continue;
        };
        commands
            .entity(descendant)
            .insert(RaytracingMesh3d(mesh_handle.clone()));

        let Some(mut mesh) = meshes.get_mut(mesh_handle) else {
            continue;
        };
        if !mesh.contains_attribute(Mesh::ATTRIBUTE_UV_0) {
            let n = mesh.count_vertices();
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; n]);
        }
        if mesh.contains_attribute(Mesh::ATTRIBUTE_UV_1) {
            mesh.remove_attribute(Mesh::ATTRIBUTE_UV_1);
        }
        if !mesh.contains_attribute(Mesh::ATTRIBUTE_TANGENT) && mesh.generate_tangents().is_err() {
            let n = mesh.count_vertices();
            mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[0.0, 0.0, 0.0, 0.0]; n]);
        }
        if let Some(indices) = mesh.indices_mut()
            && let Indices::U16(_) = indices
        {
            *indices = Indices::U32(indices.iter().map(|i| i as u32).collect());
        }
    }
}

// --- Runtime -----------------------------------------------------------------------

/// Merges every imported clip into one and plays that single clip (the ~550 animated
/// objects plus the film camera) on the scene's animation player, looping.
///
/// `convert.py` exports with glTF `SCENE` animation mode intending one baked clip, but
/// Blender's exporter still emits one clip *per object* (2337 for measure_one, 5272 for
/// measure_seven). Playing thousands of separate clips is the example's dominant CPU cost:
/// every frame the `AnimationPlayer` advances thousands of `ActiveAnimation`s and evaluates
/// a thousands-wide blend graph, all pure per-clip overhead. So we merge their curves into a
/// single clip up front. Each object's curves are keyed by a distinct `AnimationTargetId`,
/// so they never collide; playback is identical, but there is one active animation instead of
/// thousands. (The clips carry no events -- FBX animation is rigid TRS -- so none are lost.)
///
/// Runs on `WorldInstanceReady` like `animated_mesh.rs`: by the time the scene has
/// spawned, its parent glTF (and `animations`) is loaded and the player is a
/// descendant -- so this is reliable without polling.
fn start_animation(
    scene_ready: On<WorldInstanceReady>,
    scene_gltf: Res<SceneGltf>,
    gltfs: Res<Assets<Gltf>>,
    mut clips: ResMut<Assets<AnimationClip>>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    mut film_length: ResMut<FilmLength>,
    children: Query<&Children>,
    mut players: Query<&mut AnimationPlayer>,
    mut commands: Commands,
) {
    let Some(gltf) = gltfs.get(&scene_gltf.0) else {
        warn!("zero_day: glTF asset not ready; animations will not play");
        return;
    };

    // Fold all imported clips into one. The take's length is the longest source clip (every
    // object rides the film's shared timeline), which is just the merged clip's duration.
    let mut merged = AnimationClip::default();
    let mut source_clips = 0;
    for handle in &gltf.animations {
        let Some(clip) = clips.get(handle) else {
            continue;
        };
        for (target_id, curves) in clip.curves() {
            merged
                .curves_mut()
                .entry(*target_id)
                .or_default()
                .extend(curves.iter().cloned());
        }
        merged.set_duration(merged.duration().max(clip.duration()));
        source_clips += 1;
    }
    film_length.0 = merged.duration();
    let merged = clips.add(merged);

    let (graph, node) = AnimationGraph::from_clip(merged);
    let graph = graphs.add(graph);
    for entity in children.iter_descendants(scene_ready.entity) {
        if let Ok(mut player) = players.get_mut(entity) {
            player.play(node).repeat();
            commands
                .entity(entity)
                .insert(AnimationGraphHandle(graph.clone()));
        }
    }
    info!(
        "zero_day: merged {source_clips} clips into one ({:.1}s take)",
        film_length.0
    );
}

/// Toggles the film flythrough vs. free-fly.
fn toggle_flythrough(input: Res<ButtonInput<KeyCode>>, mut cinematic: ResMut<Cinematic>) {
    if input.just_pressed(KeyCode::KeyC) {
        cinematic.active = !cinematic.active;
    }
}

/// The DLSS Ray Reconstruction component at the requested quality (`--dlss-quality`).
#[cfg(feature = "dlss")]
fn dlss_rr(perf_quality_mode: DlssPerfQualityMode) -> Dlss<DlssRayReconstructionFeature> {
    Dlss::<DlssRayReconstructionFeature> {
        perf_quality_mode,
        reset: Default::default(),
        _phantom_data: Default::default(),
    }
}

/// Turns DLSS Ray Reconstruction on/off at runtime (`N`), mirroring the `solari`
/// example. DLSS also owns `TemporalJitter`/`MipBias`, so they come off with it.
#[cfg(feature = "dlss")]
fn toggle_denoiser(
    input: Res<ButtonInput<KeyCode>>,
    args: Res<Args>,
    camera: Single<(Entity, Has<Dlss<DlssRayReconstructionFeature>>), With<RenderCamera>>,
    dlss_rr_supported: Option<Res<DlssRayReconstructionSupported>>,
    mut commands: Commands,
) {
    if !input.just_pressed(KeyCode::KeyN) || dlss_rr_supported.is_none() {
        return;
    }
    let (entity, has_dlss) = *camera;
    if has_dlss {
        commands
            .entity(entity)
            .remove::<(Dlss<DlssRayReconstructionFeature>, TemporalJitter, MipBias)>();
    } else {
        commands
            .entity(entity)
            .insert(dlss_rr(args.dlss_perf_quality()));
    }
}

/// Rolling average + 1%-low/high frame times from the diagnostics history.
fn frame_stats(
    diagnostics: Res<DiagnosticsStore>,
    mut stats: ResMut<FrameStats>,
    mut scratch: Local<Vec<f64>>,
) {
    let Some(frame_time) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FRAME_TIME) else {
        return;
    };
    stats.avg_ms = frame_time.average().unwrap_or_default();
    if frame_time.history_len() >= 100 {
        scratch.clear();
        scratch.extend(frame_time.measurements().map(|m| m.value));
        scratch.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let count = (scratch.len() / 100).max(1);
        stats.one_percent_low_ms = scratch.iter().take(count).sum::<f64>() / count as f64;
        stats.one_percent_high_ms = scratch.iter().rev().take(count).sum::<f64>() / count as f64;
    }
}

/// `B` restarts the flythrough from the beginning and benchmarks one loop (~13.7 s),
/// printing a summary. Rewinding + forcing cinematic mode makes runs comparable (the
/// same camera path and object motion every time).
fn benchmark(
    input: Res<ButtonInput<KeyCode>>,
    stats: Res<FrameStats>,
    film_length: Res<FilmLength>,
    meshes: Res<Assets<Mesh>>,
    materials: Res<Assets<StandardMaterial>>,
    mesh_instances: Query<&Mesh3d>,
    mut players: Query<&mut AnimationPlayer>,
    mut cinematic: ResMut<Cinematic>,
    mut running: Local<Option<Instant>>,
    mut frames: Local<u32>,
    mut low_sum: Local<f64>,
    mut high_sum: Local<f64>,
) {
    // One full flythrough loop; falls back to MEASURE_ONE's length until the take loads.
    let target = if film_length.0 > 0.0 {
        film_length.0 as f64
    } else {
        13.7
    };
    if input.just_pressed(KeyCode::KeyB) && running.is_none() {
        // Restart the take from frame 0 and follow the film camera so the measured
        // segment is identical each run.
        cinematic.active = true;
        for mut player in &mut players {
            player.rewind_all();
        }
        *running = Some(Instant::now());
        *frames = 0;
        *low_sum = 0.0;
        *high_sum = 0.0;
        println!("zero_day: benchmarking one flythrough loop (~{target:.1}s)...");
    }
    let Some(start) = *running else {
        return;
    };
    *frames += 1;
    *low_sum += stats.one_percent_low_ms;
    *high_sum += stats.one_percent_high_ms;
    let elapsed = start.elapsed().as_secs_f64();
    if elapsed >= target {
        let f = *frames as f64;
        println!(
            "  {:.2} ms/frame avg  ({:.0} fps)  over {} frames",
            elapsed * 1000.0 / f,
            f / elapsed,
            *frames
        );
        println!(
            "  1% low {:.2} ms | 1% high {:.2} ms",
            *low_sum / f,
            *high_sum / f
        );
        println!(
            "  {} meshes | {} instances | {} materials",
            meshes.len(),
            mesh_instances.iter().count(),
            materials.len()
        );
        *running = None;
    }
}

/// While cinematic mode is active, drives the render camera from the film camera's
/// animated transform -- the original Zero-Day flythrough.
fn drive_flythrough(
    cinematic: Res<Cinematic>,
    film: Query<&GlobalTransform, With<FilmCamera>>,
    mut render: Query<&mut Transform, With<RenderCamera>>,
) {
    if !cinematic.active {
        return;
    }
    let Ok(film) = film.single() else {
        return;
    };
    let Ok(mut render) = render.single_mut() else {
        return;
    };
    // Position/orientation only -- the film camera's global transform can carry parent
    // scale we don't want on the camera itself.
    let film = film.compute_transform();
    render.translation = film.translation;
    render.rotation = film.rotation;
}

/// Fakes the film's animated lights: a wave of brightness travelling down the corridor,
/// sharpened into discrete flares, with a stable per-panel phase so neighbours don't pulse
/// in lockstep. Under Solari this modulates the real illumination -- each panel is its own
/// area light, so passing flares light the corridor as they go. The film's actual
/// sequencing was procedural in Octane and isn't in the asset, so this only evokes it;
/// `--no-pulse` holds the panels at their steady boosted emissive instead.
fn animate_emissive(
    args: Res<Args>,
    time: Res<Time>,
    panels: Query<(&GlobalTransform, &EmissivePanel, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if args.no_pulse {
        return;
    }
    let t = time.elapsed_secs();
    for (transform, panel, material) in &panels {
        // A wave travelling along the corridor's long (Z) axis, offset per panel so the
        // whole scene shimmers rather than blinking as one.
        let z = transform.translation().z;
        let wave = ops::sin(t * PULSE_FREQ - z * PULSE_WAVE_NUMBER + panel.phase);
        // Sharpen the sine so each panel sits dim, then pops.
        let flare = ops::powf(0.5 + 0.5 * wave, PULSE_SHARPNESS);
        let level = PULSE_FLOOR + (PULSE_PEAK - PULSE_FLOOR) * flare;
        if let Some(mut mat) = materials.get_mut(material.id()) {
            mat.emissive = panel.base * level;
        }
    }
}

fn update_hud(
    stats: Res<FrameStats>,
    diagnostics: Res<DiagnosticsStore>,
    cinematic: Res<Cinematic>,
    #[cfg(feature = "dlss")] denoiser: Single<
        Has<Dlss<DlssRayReconstructionFeature>>,
        With<RenderCamera>,
    >,
    mut text: Single<&mut Text, With<HudText>>,
) {
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or_default();
    let mode = if cinematic.active {
        "flythrough  (C: free-fly)"
    } else {
        "free-fly  (C: flythrough)"
    };

    #[cfg(feature = "dlss")]
    let dlss_line = format!("\nDLSS-RR: {}  (N)", if *denoiser { "on" } else { "off" });
    #[cfg(not(feature = "dlss"))]
    let dlss_line = "";

    text.0 = format!(
        "Zero-Day (Solari)\n{fps:>5.0} fps | {:.1} ms avg | {:.1} ms 1%-worst\n{mode}\nB: benchmark{dlss_line}",
        stats.avg_ms, stats.one_percent_high_ms,
    );
}
