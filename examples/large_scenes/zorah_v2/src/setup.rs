//! The camera, the stand-in lights, and the dev keys.

use std::{f32::consts::PI, time::Instant};

use bevy::{
    app::AppExit,
    camera::{CameraMainTextureUsages, Exposure, Hdr},
    camera_controller::free_camera::FreeCamera,
    core_pipeline::{
        prepass::{DeferredPrepass, DepthPrepass},
        tonemapping::Tonemapping,
    },
    diagnostic::{Diagnostic, DiagnosticPath, DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    curve::cubic_splines::LinearSpline,
    math::ops,
    post_process::auto_exposure::{AutoExposure, AutoExposureCompensationCurve},
    prelude::*,
    render::{
        render_resource::TextureUsages,
        view::screenshot::{save_to_disk, Screenshot, ScreenshotCaptured},
    },
};

use crate::{
    bake::{self, BakeSettings, MeshJob},
    environment::EnvironmentMapLoad,
    runner::{PendingRaytracingInstance, PendingScene, ZorahState},
    scene::{SceneEnvironmentMap, SceneInstance, SceneView},
};

/// Node name fragments of the fire props that get an emissive proxy.
const FIRE_NODE_NAMES: &[&str] = &["FirePot", "FireGrate", "Firewood_Coal"];
/// The proxy sphere sits this far above the prop's origin, at this radius.
const FIRE_PROXY_HEIGHT: f32 = 0.3;
const FIRE_PROXY_RADIUS: f32 = 0.15;
/// A wood fire's colour temperature.
const FIRE_TEMPERATURE_K: f32 = 1900.0;
/// The reference renderer's metering (donut's histogram eye adaptation, which
/// RTXMG runs with its defaults): the mean of the 80th..95th luminance
/// percentiles, so the scene is exposed for its bright end rather than its
/// median, is scaled to `exp2(-0.5)` before the ACES curve. Bevy's meter
/// scales its mean to 1.0 and adds the compensation, so the same target is a
/// flat -0.5 EV over that same percentile band. The export has no exposure
/// data of its own; this is where the reference's look comes from.
const REFERENCE_METERING_FILTER: std::ops::RangeInclusive<f32> = 0.80..=0.95;
const REFERENCE_EXPOSURE_COMPENSATION_EV: f32 = -0.5;
/// donut's `eyeAdaptationSpeedUp`/`eyeAdaptationSpeedDown`, in EV per second.
/// Bevy's defaults are 3.0 and 1.0, so the reference adapts markedly slower in
/// both directions.
const REFERENCE_ADAPTATION_SPEED_BRIGHTEN: f32 = 1.0;
const REFERENCE_ADAPTATION_SPEED_DARKEN: f32 = 0.5;

#[derive(Component)]
pub struct ZorahCamera;

/// What setup needs from the command line.
#[derive(Resource)]
pub struct SetupOptions {
    pub view: SceneView,
    /// The sidecar's HDR: the sky light and the backdrop.
    pub environment_map: SceneEnvironmentMap,
    pub camera_position: Option<Vec3>,
    pub camera_target: Option<Vec3>,
    pub exposure_ev100: Option<f32>,
    pub auto_exposure: bool,
    pub exposure_bias: f32,
    pub fire_lumens: f32,
    pub solari_albedo: bool,
    /// `--disable-dlss`: leave Ray Reconstruction off even where it is
    /// supported, so the main pass renders at native resolution and nothing
    /// denoises Solari's output. Nothing reads it in a build without DLSS,
    /// where there is no Ray Reconstruction to leave off in the first place.
    #[cfg_attr(not(feature = "dlss"), allow(dead_code))]
    pub disable_dlss: bool,
    pub bake: BakeSettings,
    pub jobs: Vec<MeshJob>,
}

impl SetupOptions {
    /// The fixed base exposure the traced radiance is packed against;
    /// `--exposure-bias` brightens for positive values, so it lowers the EV100.
    pub fn base_ev100(&self) -> f32 {
        self.exposure_ev100.unwrap_or(Exposure::BLENDER.ev100) - self.exposure_bias
    }
}

/// Spawns the camera and the lights, then starts the bake. The bake starts
/// here rather than in `main` so `MeshletMesh::from_mesh` finds the task pool
/// the app sized, not a default one.
pub fn setup(
    mut commands: Commands,
    mut options: ResMut<SetupOptions>,
    scene: Res<crate::runner::SceneData>,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut compensation_curves: ResMut<Assets<AutoExposureCompensationCurve>>,
) {
    let mut raytracing_instances = Vec::new();
    if options.fire_lumens > 0.0 {
        raytracing_instances = spawn_fire_proxies(
            &mut commands,
            &scene.instances,
            &mut meshes,
            &mut materials,
            options.fire_lumens,
        );
    }

    let camera_position = options.camera_position.unwrap_or(options.view.position);
    let camera_target = options.camera_target.unwrap_or(options.view.target);
    let base_ev100 = options.base_ev100();
    // No bloom: the reference renderer's blit is ACES and the sRGB transfer and
    // nothing else, so its highlights carry no glare. `Bloom::NATURAL` is also
    // stronger than its name suggests here - its prefilter threshold is 0, so
    // every pixel contributes, making it a 15% blurred copy of the whole frame
    // rather than a highlight effect. An albedo capture wants the texture
    // values themselves on screen, so it skips the filmic curve as well.
    let tonemapping = if options.solari_albedo {
        // `Linear` rather than `None`: the capture still wants the exposure
        // correction and the sRGB transfer the tonemapping pass applies.
        Tonemapping::Linear
    } else {
        Tonemapping::AcesFitted
    };
    let mut camera = commands.spawn((
        Name::new("Camera"),
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        // Solari requires the main texture to be storage-bindable. Add `Hdr`
        // from the camera's first frame instead of waiting for the deferred
        // `SolariLighting` insertion to add it as a required component.
        Hdr,
        DepthPrepass,
        DeferredPrepass,
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        Exposure { ev100: base_ev100 },
        tonemapping,
        Projection::Perspective(PerspectiveProjection {
            fov: options.view.fov_degrees.to_radians(),
            ..default()
        }),
        ZorahCamera,
        FreeCamera {
            walk_speed: 3.0,
            run_speed: 20.0,
            ..default()
        },
        Transform::from_translation(camera_position).looking_at(camera_target, Vec3::Y),
    ));
    // The fixed `Exposure` stays as the base the traced radiance is packed
    // against; the histogram adds its correction in the tonemapping pass, so
    // Solari's light-tile precision is unaffected by adaptation.
    let auto_exposure = options.auto_exposure && options.exposure_ev100.is_none();
    if auto_exposure {
        // The metered target is independent of the base, so the reference's
        // compensation and the bias enter the meter as a flat exposure
        // compensation over the whole metered range; positive brightens.
        let range = AutoExposure::default().range;
        let compensation_ev = REFERENCE_EXPOSURE_COMPENSATION_EV + options.exposure_bias;
        let compensation_curve = AutoExposureCompensationCurve::from_curve(LinearSpline::new([
            Vec2::new(*range.start(), compensation_ev),
            Vec2::new(*range.end(), compensation_ev),
        ]))
        .expect("a flat two-point curve is monotonic and continuous");
        camera.insert(AutoExposure {
            range,
            filter: REFERENCE_METERING_FILTER,
            speed_brighten: REFERENCE_ADAPTATION_SPEED_BRIGHTEN,
            speed_darken: REFERENCE_ADAPTATION_SPEED_DARKEN,
            compensation_curve: compensation_curves.add(compensation_curve),
            ..default()
        });
    }
    // The map goes on the camera once `install_environment_map` has
    // converted it; until then the preview is unlit (a second or two).
    commands.insert_resource(EnvironmentMapLoad::start(
        &asset_server,
        options.environment_map.clone(),
    ));
    info!(
        camera_position = %camera_position,
        camera_target = %camera_target,
        fov_degrees = options.view.fov_degrees,
        base_ev100,
        auto_exposure,
        environment_map = options.environment_map.path.as_str(),
        fire_proxies = raytracing_instances.len(),
        "Zorah view and lighting set up"
    );

    let jobs = std::mem::take(&mut options.jobs);
    let bake = bake::start(options.bake.clone(), jobs);
    commands.insert_resource(PendingScene::new(bake, raytracing_instances));
}

/// An emissive sphere over every fire prop, for Solari only: it has no
/// `Mesh3d`, so the raster image never shows the synthetic ball.
fn spawn_fire_proxies(
    commands: &mut Commands,
    instances: &[SceneInstance],
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    lumens: f32,
) -> Vec<PendingRaytracingInstance> {
    let mut raytracing_instances = Vec::new();
    let fires = instances
        .iter()
        .filter(|instance| {
            FIRE_NODE_NAMES
                .iter()
                .any(|name| instance.name.contains(name))
        })
        .collect::<Vec<_>>();
    if fires.is_empty() {
        return raytracing_instances;
    }
    // POSITION/NORMAL/UV_0 with u32 indices: Solari's compact vertex.
    let sphere = meshes.add(Sphere::new(1.0).mesh().uv(16, 8));
    let color = blackbody_srgb(FIRE_TEMPERATURE_K).to_linear();
    // Flux spread over the sphere as v1 does for point lights: lm / (4 pi^2 r^2).
    let radiance = lumens / (4.0 * ops::powf(PI * FIRE_PROXY_RADIUS, 2.0)).max(0.0001);
    let material = materials.add(StandardMaterial {
        base_color: Color::BLACK,
        emissive: LinearRgba::rgb(
            color.red * radiance,
            color.green * radiance,
            color.blue * radiance,
        ),
        perceptual_roughness: 1.0,
        ..default()
    });
    for instance in fires {
        let entity = commands
            .spawn((
                Name::new(format!("{} Solari emitter", instance.name)),
                MeshMaterial3d(material.clone()),
                Transform::from_translation(
                    instance.transform.translation + Vec3::Y * FIRE_PROXY_HEIGHT,
                )
                .with_scale(Vec3::splat(FIRE_PROXY_RADIUS)),
            ))
            .id();
        raytracing_instances.push(PendingRaytracingInstance {
            entity,
            mesh: sphere.clone(),
            geometry_error: 0.0,
        });
    }
    raytracing_instances
}

/// A compact correlated-colour-temperature approximation, as v1 uses.
fn blackbody_srgb(temperature: f32) -> Color {
    let temperature = temperature.clamp(1000.0, 40_000.0) / 100.0;
    let red = if temperature <= 66.0 {
        255.0
    } else {
        329.698_73 * ops::powf(temperature - 60.0, -0.133_204_76)
    };
    let green = if temperature <= 66.0 {
        99.470_8 * ops::ln(temperature) - 161.119_57
    } else {
        288.122_16 * ops::powf(temperature - 60.0, -0.075_514_846)
    };
    let blue = if temperature >= 66.0 {
        255.0
    } else if temperature <= 19.0 {
        0.0
    } else {
        138.517_73 * ops::ln(temperature - 10.0) - 305.044_8
    };
    Color::srgb(
        (red / 255.0).clamp(0.0, 1.0),
        (green / 255.0).clamp(0.0, 1.0),
        (blue / 255.0).clamp(0.0, 1.0),
    )
}

fn screenshot_path() -> String {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    format!("zorah_v2-{stamp}.png")
}

/// `P` logs the current view as the command-line fragment that reproduces
/// it; `F12` saves the frame in the working directory.
pub fn dump_camera_and_screenshot(
    keys: Res<ButtonInput<KeyCode>>,
    camera: Single<&Transform, With<ZorahCamera>>,
    mut commands: Commands,
) {
    if keys.just_pressed(KeyCode::KeyP) {
        let position = camera.translation;
        // Any point along the view direction reproduces the look; a few metres
        // out keeps the printed target readable and away from the position.
        let target = position + camera.forward() * 5.0;
        info!(
            "camera: --camera-position {:.3},{:.3},{:.3} --camera-target {:.3},{:.3},{:.3}",
            position.x, position.y, position.z, target.x, target.y, target.z
        );
    }
    if keys.just_pressed(KeyCode::F12) {
        let path = screenshot_path();
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path.clone()));
        info!(path, "saving screenshot");
    }
}

/// `--screenshot-after`: one F12 this long after Solari comes on, then exit.
#[derive(Resource)]
pub struct ScreenshotAfter {
    pub delay_seconds: f32,
    running_since: Option<Instant>,
    requested: bool,
    captured: bool,
}

impl ScreenshotAfter {
    pub fn new(delay_seconds: f32) -> Self {
        Self {
            delay_seconds,
            running_since: None,
            requested: false,
            captured: false,
        }
    }
}

pub fn screenshot_after(
    mut commands: Commands,
    mut schedule: ResMut<ScreenshotAfter>,
    state: Res<State<ZorahState>>,
    mut exit: MessageWriter<AppExit>,
) {
    if schedule.captured {
        // The observer saved the file synchronously before this frame.
        info!("screenshot written; exiting");
        exit.write(AppExit::Success);
        schedule.captured = false;
        return;
    }
    if *state != ZorahState::Running || schedule.requested {
        return;
    }
    let since = *schedule.running_since.get_or_insert_with(Instant::now);
    if since.elapsed().as_secs_f32() < schedule.delay_seconds {
        return;
    }
    schedule.requested = true;
    let path = screenshot_path();
    info!(path, "saving the scheduled screenshot");
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path))
        .observe(
            |_: On<ScreenshotCaptured>, mut schedule: ResMut<ScreenshotAfter>| {
                schedule.captured = true;
            },
        );
}

/// Marks the performance overlay's text.
#[derive(Component)]
pub struct HudText;

/// Spawns the corner performance overlay, styled after the `solari` example's.
pub fn spawn_hud(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            right: px(0.0),
            padding: px(4.0).all(),
            border_radius: BorderRadius::bottom_left(px(4.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.10, 0.10, 0.10, 0.8)),
        children![(
            HudText,
            Text::default(),
            TextFont {
                font_size: FontSize::Px(8.0),
                ..default()
            },
        )],
    ));
}

/// Rewrites the overlay from the diagnostics store each frame: the frame rate,
/// then whichever GPU spans report on the current path.
pub fn update_hud(mut text: Single<&mut Text, With<HudText>>, diagnostics: Res<DiagnosticsStore>) {
    text.0.clear();
    if let Some(fps) = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(Diagnostic::smoothed)
    {
        text.push_str(&format!("{:17}  {fps:.1}\n", "FPS"));
    }
    if let Some(frame_ms) = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(Diagnostic::smoothed)
    {
        text.push_str(&format!("{:17}  {frame_ms:.2} ms\n", "Frame"));
    }

    let mut total = 0.0;
    let mut add_span = |name: &str, path: &'static str| {
        let path = DiagnosticPath::new(path);
        if let Some(value) = diagnostics.get(&path).and_then(Diagnostic::smoothed) {
            text.push_str(&format!("{name:17}  {value:.2} ms\n"));
            total += value;
        }
    };
    (add_span)(
        "Meshlet raster",
        "render/meshlet_visibility_buffer_raster/elapsed_gpu",
    );
    (add_span)(
        "Light tiles",
        "render/solari_lighting/presample_light_tiles/elapsed_gpu",
    );
    (add_span)(
        "World cache",
        "render/solari_lighting/world_cache/elapsed_gpu",
    );
    (add_span)("Lighting", "render/solari_lighting/lighting/elapsed_gpu");
    (add_span)("BLAS build", "render/blas_build/elapsed_gpu");
    (add_span)("TLAS build", "render/tlas_build/elapsed_gpu");
    (add_span)("DLSS-RR", "render/dlss_ray_reconstruction/elapsed_gpu");
    (add_span)("DLSS-SR", "render/dlss_super_resolution/elapsed_gpu");
    if total > 0.0 {
        text.push_str(&format!("{:17}  {total:.2} ms\n", "GPU measured"));
    }
    add_world_cache_occupancy(&mut text, &diagnostics);
}

/// The world cache's live cells as a share of its fixed capacity. Solari reads
/// the count back off the GPU itself, so this is a plain diagnostic lookup.
///
/// Once the cache is full, a query that collides past its few probe steps
/// resolves to no indirect light at all, so this is the number that says
/// whether a scene has outgrown the cache.
pub fn add_world_cache_occupancy(text: &mut Text, diagnostics: &DiagnosticsStore) {
    // `bevy_solari`'s `WORLD_CACHE_SIZE` is private to the crate, so the
    // capacity is restated here, as the `solari` example does.
    const WORLD_CACHE_CELLS: f64 = 1_048_576.0;
    let Some(active_cells) = diagnostics
        .get(&DiagnosticPath::new(
            "render/solari_lighting/world_cache_active_cells_count",
        ))
        .and_then(Diagnostic::smoothed)
    else {
        return;
    };
    text.push_str(&format!(
        "{:17}  {:.1}% ({:.0})\n",
        "World cache full",
        active_cells * 100.0 / WORLD_CACHE_CELLS,
        active_cells,
    ));
}
