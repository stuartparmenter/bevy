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
//! It plays the film's ~13.7 s take -- ~550 animated objects plus the original
//! `DynamicCamera2` flythrough -- and drives the render camera from that camera. The
//! film's emissive pulsing was procedural in Octane and isn't in the exported asset,
//! so `animate_emissive` breathes the panels as a stand-in.
//!
//! Requires a ray-tracing capable GPU (Solari currently needs the Vulkan backend in
//! wgpu). DLSS Ray Reconstruction denoises the output when the `dlss` feature (the
//! default) is enabled on a supported NVIDIA GPU.
//!
//! Controls: `C` toggles the film flythrough vs. free-fly (WASD + mouse), `N` toggles
//! DLSS Ray Reconstruction, and `B` runs a short benchmark (printed to the console).

use std::collections::HashSet;
use std::f32::consts::PI;
use std::time::Instant;

use argh::FromArgs;
use bevy::{
    camera::{CameraMainTextureUsages, Hdr},
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    core_pipeline::tonemapping::{GranTurismo7Params, Tonemapping},
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    gltf::Gltf,
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
        Dlss, DlssProjectId, DlssRayReconstructionFeature, DlssRayReconstructionSupported,
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
    /// emissive multiplier for the accent panels (they are the scene's only lights, so
    /// they must be bright to illuminate the corridor). ~150000 reads about right;
    /// lower it if the scene is blown out.
    #[argh(option, default = "150000.0")]
    emissive: f32,

    /// disable the synthetic emissive pulse. By default the panels breathe out of sync
    /// to evoke the film's animated lights (that animation isn't in the exported asset).
    #[argh(switch)]
    no_pulse: bool,
}

/// Depth (fraction of base) and rate (rad/s) of the synthetic emissive pulse.
const PULSE_AMPLITUDE: f32 = 0.5;
const PULSE_FREQ: f32 = 2.5;

fn main() {
    let args: Args = argh::from_env();

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
        .init_resource::<EmissiveMaterials>()
        .init_resource::<FrameStats>()
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: PresentMode::Immediate,
                        resolution: WindowResolution::new(1920, 1080)
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

/// The film's imported camera (`DynamicCamera2`), stripped to a transform source that
/// `drive_flythrough` follows.
#[derive(Component)]
struct FilmCamera;

/// The on-screen readout.
#[derive(Component)]
struct HudText;

/// Keeps the whole glTF loaded so its animation clips stay reachable.
#[derive(Resource)]
struct SceneGltf(Handle<Gltf>);

/// Emissive materials collected in `proc_scene` (handle + boosted base emissive),
/// modulated by `animate_emissive`.
#[derive(Resource, Default)]
struct EmissiveMaterials(Vec<(Handle<StandardMaterial>, LinearRgba)>);

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
    #[cfg(feature = "dlss")] dlss_rr_supported: Option<Res<DlssRayReconstructionSupported>>,
) {
    println!("Loading Zero-Day (this is a large scene; give it a moment)");

    // The whole glTF (not just the scene) so its ~550 animation clips -- and the film
    // camera's -- stay loaded and reachable in `start_animation`.
    commands.insert_resource(SceneGltf(asset_server.load("zero_day.glb")));

    commands
        .spawn(WorldAssetRoot(asset_server.load("zero_day.glb#Scene0")))
        // Repairs + boosts the emissive materials.
        .observe(proc_scene)
        // Turns the film camera into the flythrough source.
        .observe(setup_flythrough_camera)
        // Tags every mesh `RaytracingMesh3d` so Solari can trace against it.
        .observe(setup_raytracing_meshes)
        // Plays every animation clip once the scene (and its player) exist.
        .observe(start_animation);

    // Camera. Its projection is overwritten with the film camera's in
    // `setup_flythrough_camera`; the transform is the fallback view held until the
    // flythrough is actually driving.
    let mut cam = commands.spawn((
        Camera3d::default(),
        // The imported `DynamicCamera2` also spawns as an active camera (order 0). A
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
        cam.insert(dlss_rr());
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

// --- Scene processing (on load) ----------------------------------------------------

/// Repairs the imported materials and boosts the emissive ones so the panels act as
/// bright Solari light sources, then drops any imported lights.
///
/// `convert.py` handles most of the FBX quirks at the asset level, but deliberately
/// leaves the normals DirectX-convention (green flipped vs glTF) for the engine to
/// flip -- matching how the `bistro` example handles Sponza. Instanced meshes share a
/// material handle, so each material is processed once (a per-entity pass would
/// multiply a shared emissive many times, to infinity).
fn proc_scene(
    scene_ready: On<WorldInstanceReady>,
    mut commands: Commands,
    has_std_mat: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    lights: Query<Entity, Or<(With<PointLight>, With<DirectionalLight>, With<SpotLight>)>>,
    children: Query<&Children>,
    args: Res<Args>,
    mut emissive_materials: ResMut<EmissiveMaterials>,
    mut processed: Local<HashSet<AssetId<StandardMaterial>>>,
) {
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
                mat.emissive = mat.emissive * args.emissive;
                emissive_materials.0.push((mat_h.0.clone(), mat.emissive));
            }
        }

        // Zero-Day is lit purely by its emissives; drop any imported lights.
        if lights.get(entity).is_ok() {
            commands.entity(entity).despawn();
        }
    }
}

/// Repurposes the imported `DynamicCamera2` as the flythrough transform source: copies
/// its projection to the render camera, then strips its render components so only our
/// camera draws (it keeps its animated transform, which `drive_flythrough` follows).
/// The camera entity is present at `WorldInstanceReady`, so this is reliable.
fn setup_flythrough_camera(
    scene_ready: On<WorldInstanceReady>,
    children: Query<&Children>,
    film_cameras: Query<(Entity, &Projection), (With<Camera>, Without<RenderCamera>)>,
    mut render_projection: Query<&mut Projection, With<RenderCamera>>,
    mut commands: Commands,
) {
    let mut film_projection = None;
    for entity in children.iter_descendants(scene_ready.entity) {
        if let Ok((camera, projection)) = film_cameras.get(entity) {
            film_projection = Some(projection.clone());
            commands
                .entity(camera)
                .remove::<(Camera3d, Camera, Projection)>()
                .insert(FilmCamera);
        }
    }
    if let (Some(projection), Ok(mut render_projection)) =
        (film_projection, render_projection.single_mut())
    {
        *render_projection = projection;
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

/// Plays every animation clip (the ~550 animated objects plus the film camera) on the
/// scene's animation player, looping.
///
/// Runs on `WorldInstanceReady` like `animated_mesh.rs`: by the time the scene has
/// spawned, its parent glTF (and `animations`) is loaded and the player is a
/// descendant -- so this is reliable without polling.
fn start_animation(
    scene_ready: On<WorldInstanceReady>,
    scene_gltf: Res<SceneGltf>,
    gltfs: Res<Assets<Gltf>>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    children: Query<&Children>,
    mut players: Query<&mut AnimationPlayer>,
    mut commands: Commands,
) {
    let Some(gltf) = gltfs.get(&scene_gltf.0) else {
        warn!("zero_day: glTF asset not ready; animations will not play");
        return;
    };
    let (graph, nodes) = AnimationGraph::from_clips(gltf.animations.iter().cloned());
    let graph = graphs.add(graph);
    for entity in children.iter_descendants(scene_ready.entity) {
        if let Ok(mut player) = players.get_mut(entity) {
            for node in &nodes {
                player.play(*node).repeat();
            }
            commands
                .entity(entity)
                .insert(AnimationGraphHandle(graph.clone()));
        }
    }
    info!("zero_day: started {} animation clips", nodes.len());
}

/// Toggles the film flythrough vs. free-fly.
fn toggle_flythrough(input: Res<ButtonInput<KeyCode>>, mut cinematic: ResMut<Cinematic>) {
    if input.just_pressed(KeyCode::KeyC) {
        cinematic.active = !cinematic.active;
    }
}

/// The DLSS Ray Reconstruction component, at default quality.
#[cfg(feature = "dlss")]
fn dlss_rr() -> Dlss<DlssRayReconstructionFeature> {
    Dlss::<DlssRayReconstructionFeature> {
        perf_quality_mode: Default::default(),
        reset: Default::default(),
        _phantom_data: Default::default(),
    }
}

/// Turns DLSS Ray Reconstruction on/off at runtime (`N`), mirroring the `solari`
/// example. DLSS also owns `TemporalJitter`/`MipBias`, so they come off with it.
#[cfg(feature = "dlss")]
fn toggle_denoiser(
    input: Res<ButtonInput<KeyCode>>,
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
        commands.entity(entity).insert(dlss_rr());
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
#[allow(clippy::too_many_arguments)]
fn benchmark(
    input: Res<ButtonInput<KeyCode>>,
    stats: Res<FrameStats>,
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
        println!("zero_day: benchmarking one flythrough loop (~13.7s)...");
    }
    let Some(start) = *running else {
        return;
    };
    *frames += 1;
    *low_sum += stats.one_percent_low_ms;
    *high_sum += stats.one_percent_high_ms;
    let elapsed = start.elapsed().as_secs_f64();
    if elapsed >= 13.7 {
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

/// Breathes the emissive panels in and out of sync -- a synthetic stand-in for the
/// film's animated lights. Under Solari this pulses the actual illumination.
fn animate_emissive(
    args: Res<Args>,
    time: Res<Time>,
    emissive: Res<EmissiveMaterials>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if args.no_pulse {
        return;
    }
    let t = time.elapsed_secs();
    for (i, (handle, base)) in emissive.0.iter().enumerate() {
        if let Some(mut mat) = materials.get_mut(handle) {
            let phase = i as f32 * 1.7;
            let pulse = 1.0 + PULSE_AMPLITUDE * (t * PULSE_FREQ + phase).sin();
            mat.emissive = *base * pulse;
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
