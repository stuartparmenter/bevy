//! The camera, the stand-in lights, and the dev keys.

use std::time::Instant;

use bevy::{
    app::AppExit,
    camera::{CameraMainTextureUsages, Exposure, Hdr},
    camera_controller::free_camera::FreeCamera,
    core_pipeline::{
        prepass::{DeferredPrepass, DepthPrepass},
        tonemapping::Tonemapping,
    },
    light::SunDisk,
    curve::cubic_splines::LinearSpline,
    math::ops,
    post_process::{
        auto_exposure::{AutoExposure, AutoExposureCompensationCurve},
        bloom::Bloom,
    },
    prelude::*,
    render::{
        render_resource::TextureUsages,
        view::screenshot::{save_to_disk, Screenshot, ScreenshotCaptured},
    },
};

use crate::{
    bake::{self, BakeSettings, MeshJob},
    runner::{PendingRaytracingInstance, PendingScene, ZorahState},
    scene::{SceneInstance, SceneView},
};

/// Node name fragments of the fire props that get an emissive proxy.
const FIRE_NODE_NAMES: &[&str] = &["FirePot", "FireGrate", "Firewood_Coal"];
/// The proxy sphere sits this far above the prop's origin, at this radius.
const FIRE_PROXY_HEIGHT: f32 = 0.3;
const FIRE_PROXY_RADIUS: f32 = 0.15;
/// A wood fire's colour temperature.
const FIRE_TEMPERATURE_K: f32 = 1900.0;
/// `vk_lod_clusters`' `--sundirection` (towards the sun) and `--suncolor`
/// from the export's .cfg.
const SUN_DIRECTION: Vec3 = Vec3::new(0.6, 0.7, 0.36);
/// A renderer parameter, so linear rather than encoded.
const SUN_COLOR: Color = Color::linear_rgb(1.0, 0.8, 0.5);
/// The reference renderer's metering (donut's histogram eye adaptation, which
/// RTXMG runs with its defaults): the mean of the 80th..95th luminance
/// percentiles, so the scene is exposed for its bright end rather than its
/// median, is scaled to `exp2(-0.5)` before the ACES curve. Bevy's meter
/// scales its mean to 1.0 and adds the compensation, so the same target is a
/// flat -0.5 EV over that same percentile band. The export has no exposure
/// data of its own; this is where the reference's look comes from.
const REFERENCE_METERING_FILTER: std::ops::RangeInclusive<f32> = 0.80..=0.95;
const REFERENCE_EXPOSURE_COMPENSATION_EV: f32 = -0.5;

#[derive(Component)]
pub struct ZorahCamera;

/// What setup needs from the command line.
#[derive(Resource)]
pub struct SetupOptions {
    pub view: SceneView,
    pub camera_position: Option<Vec3>,
    pub camera_target: Option<Vec3>,
    pub exposure_ev100: Option<f32>,
    pub auto_exposure: bool,
    pub exposure_bias: f32,
    pub sky_illuminance: f32,
    pub sky_color: Vec3,
    pub sun_illuminance: f32,
    pub fire_lumens: f32,
    pub solari_albedo: bool,
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
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut compensation_curves: ResMut<Assets<AutoExposureCompensationCurve>>,
    mut ambient_light: ResMut<GlobalAmbientLight>,
) {
    let sky_color = LinearRgba::rgb(
        options.sky_color.x,
        options.sky_color.y,
        options.sky_color.z,
    );
    // Overrides the `GlobalAmbientLight::NONE` main inserts for the preview
    // only: it keeps the raster image lit while the BLASes build instead of
    // showing a black frame, and the warm-up zeroes it again when Solari
    // takes over. Brightness is radiance; a uniform sky of illuminance E has
    // radiance E / pi, so the preview matches the traced sky it stands in for.
    ambient_light.color = Color::LinearRgba(sky_color);
    ambient_light.brightness = options.sky_illuminance / std::f32::consts::PI;

    if options.sun_illuminance > 0.0 {
        commands.spawn((
            Name::new("Sun"),
            DirectionalLight {
                color: SUN_COLOR,
                illuminance: options.sun_illuminance,
                // Solari provides the shadows; a cascade pass would only cost.
                shadow_maps_enabled: false,
                ..default()
            },
            SunDisk::EARTH,
            Transform::from_translation(Vec3::ZERO).looking_to(-SUN_DIRECTION.normalize(), Vec3::Y),
        ));
    }

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
    // An albedo capture wants the texture values themselves on screen, so it
    // skips the filmic curve and the glare that would reshape them.
    let (tonemapping, bloom) = if options.solari_albedo {
        (
            // `Linear` rather than `None`: the capture still wants the exposure
            // correction and the sRGB transfer the tonemapping pass applies.
            Tonemapping::Linear,
            Bloom {
                intensity: 0.0,
                ..Bloom::NATURAL
            },
        )
    } else {
        (Tonemapping::AcesFitted, Bloom::NATURAL)
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
        (tonemapping, bloom),
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
            compensation_curve: compensation_curves.add(compensation_curve),
            ..default()
        });
    }
    info!(
        camera_position = %camera_position,
        camera_target = %camera_target,
        fov_degrees = options.view.fov_degrees,
        base_ev100,
        auto_exposure,
        sky_illuminance = options.sky_illuminance,
        sun_illuminance = options.sun_illuminance,
        fire_proxies = raytracing_instances.len(),
        "Zorah view and stand-in lighting set up"
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
    let radiance =
        lumens / (4.0 * ops::powf(std::f32::consts::PI * FIRE_PROXY_RADIUS, 2.0)).max(0.0001);
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
