//! NVIDIA's Zorah glTF export (`zorah_textured_public.v1`) in bevy meshlets
//! plus Solari, baked straight from the export on first run.

// An example binary always has std; the alloc split is for the engine crates.
#![expect(
    clippy::std_instead_of_alloc,
    reason = "an example binary always has std"
)]

mod bake;
mod geometry;
mod lod;
mod materials;
mod report;
mod runner;
mod scene;
mod setup;

// Opt-in HDR setup shared with the other HDR examples: keeps the primary
// window's `DisplayTarget` on the best transfer the surface can present
// (PQ/HDR10, then scRGB-linear, then extended sRGB), else stays SDR.
#[path = "../../../helpers/hdr.rs"]
mod hdr;

use std::{
    collections::{BTreeSet, HashMap},
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant},
};

use argh::FromArgs;
use bevy::{
    asset::{io::AssetSourceBuilder, AssetMetaCheck},
    camera_controller::free_camera::FreeCameraPlugin,
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    pbr::{experimental::meshlet::MeshletPlugin, DefaultOpaqueRendererMethod},
    post_process::auto_exposure::AutoExposurePlugin,
    prelude::*,
    render::{working_color_space::WorkingColorSpace, RenderPlugin},
    solari::prelude::SolariPlugins,
};

#[cfg(feature = "dlss")]
use bevy::anti_alias::dlss::DlssProjectId;

use bake::{BakeEvent, BakeSettings, MeshJob};
use lod::{LodSettings, ZorahPart, ZorahPartLoader};
use materials::{MaterialCache, MaterialOptions, MaterialSpec};
use runner::{SceneData, SpawnOptions, ZorahState};
use scene::SceneView;
use setup::{ScreenshotAfter, SetupOptions};

const DOWNLOAD_HINT: &str =
    "download NVIDIA's zorah_textured_public.v1 export (see README.md) and \
pass its directory with --scene-root or ZORAH_ROOT";
const ROOT_GLTF: &str = "zorah_textured_public.v1.gltf";
const CACHE_DIR_NAME: &str = ".bevy_zorah_cache";
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

// TODO(lighting): the export carries no lights at all. RTXMG, the reference
// renderer, lights it with the sidecar's equirectangular HDR alone, which
// bevy_solari cannot bind yet (its environment light is a uniform +Y
// hemisphere), and lets sky light through the throne room's stained glass as
// thin-wall transmission, which it has no notion of either. The sky, sun,
// emissive-boost and fire flags below are stand-ins until a textured
// environment light and thin-wall transmission exist in bevy_solari.

// Doc comments here are argh's help text, where backticks would print literally.
#[expect(clippy::doc_markdown, reason = "argh prints these as help text")]
#[derive(FromArgs)]
/// Render NVIDIA's Zorah glTF export with meshlets and Solari.
struct Args {
    /// the export's directory or its root .gltf (else the ZORAH_ROOT environment variable)
    #[argh(option)]
    scene_root: Option<PathBuf>,

    /// where baked meshlet meshes live (default <scene root>/.bevy_zorah_cache)
    #[argh(option)]
    cache_dir: Option<PathBuf>,

    /// threads baking meshes (or measuring them under --report-lod-budget) at once (default clamp((RAM - 8 GiB) / 2 GiB, 1, cores))
    #[argh(option)]
    bake_workers: Option<usize>,

    /// cut primitives above this many triangles into spatial partitions before the meshlet build
    #[argh(option, default = "500_000")]
    partition_triangles: usize,

    /// metres of geometric error the finest meshlet LOD kept resident may carry; every part is pruned to it as it loads
    #[argh(option, default = "0.004")]
    raster_error: f32,

    /// metres of geometric error the raytracing LOD cut may carry (never finer than --raster-error)
    #[argh(option, default = "0.05")]
    raytracing_error: f32,

    /// measure what the cache costs at a ladder of --raster-error and --raytracing-error values, then exit
    #[argh(switch)]
    report_lod_budget: bool,

    /// meshlet vertex position quantization factor (positions snap to 1/2^n cm)
    #[argh(option, default = "4")]
    quantization: u8,

    /// drop KTX2 mips above this edge length at load; 0 keeps the full 4k textures
    #[argh(option, default = "1024")]
    max_texture_size: u32,

    /// dev: bake and spawn only the first N meshes by glTF index (fire proxies follow the selected props)
    #[argh(option)]
    limit_meshes: Option<usize>,

    /// exit once the bake is complete instead of rendering
    #[argh(switch)]
    bake_only: bool,

    /// dev: take the F12 screenshot this many seconds after the scene settles, then exit
    #[argh(option)]
    screenshot_after: Option<f32>,

    /// print Bevy's periodic frame-time diagnostics
    #[argh(switch)]
    diagnostics: bool,

    /// camera position x,y,z in the export's metres (default the sidecar scene.json view)
    #[argh(option)]
    camera_position: Option<String>,

    /// camera look target x,y,z (default the sidecar scene.json view)
    #[argh(option)]
    camera_target: Option<String>,

    /// fixed camera exposure in EV100 instead of auto exposure
    #[argh(option)]
    exposure_ev100: Option<f32>,

    /// fixed exposure at the base EV100 (Blender's default, minus the bias) instead of histogram auto exposure
    #[argh(switch)]
    no_auto_exposure: bool,

    /// extra exposure compensation in EV, positive brighter (default 0)
    #[argh(option, default = "0.0")]
    exposure_bias: f32,

    /// skip every node whose name contains this substring (repeatable)
    #[argh(option)]
    hide_nodes: Vec<String>,

    /// keep MASK and BLEND materials as alpha-tested Mesh3d instances instead of forcing them opaque
    #[argh(switch)]
    preserve_alpha: bool,

    /// render every material double-sided, as the reference renderer does
    #[argh(switch)]
    double_sided_all: bool,

    /// honour KHR_materials_specular (off: the export's 0.498 is UE's default 0.5 and would halve F0)
    #[argh(switch)]
    gltf_specular: bool,

    /// include transmissive (stained glass) materials in the BLAS; off lets sky light through them
    #[argh(switch)]
    glass_in_blas: bool,

    /// illuminance in lux of the uniform sky Solari lights the scene with
    #[argh(option, default = "15000.0")]
    sky_illuminance: f32,

    /// sky colour r,g,b in linear 0..1
    #[argh(option, default = "String::from(\"0.78,0.86,1.0\")")]
    sky_color: String,

    /// illuminance in lux of a directional sun from the .cfg direction; 0 = no sun
    #[argh(option, default = "0.0")]
    sun_illuminance: f32,

    /// multiplier on every emissive material so lamp glass and coals act as light sources
    #[argh(option, default = "4.0")]
    emissive_boost: f32,

    /// luminous power of the emissive proxy sphere spawned at every fire node; 0 = none
    #[argh(option, default = "800.0")]
    fire_lumens: f32,

    /// render every surface as flat grey clay (normal maps and emission kept)
    #[argh(switch)]
    clay: bool,

    /// show base colour through the traced path: surfaces emit their albedo and reflect nothing
    #[argh(switch)]
    solari_albedo: bool,
}

impl Args {
    fn lod(&self) -> Result<LodSettings, String> {
        let bound = |name: &str, value: f32| {
            (value.is_finite() && value >= 0.0)
                .then_some(value)
                .ok_or_else(|| format!("{name}: {value} is not a finite non-negative distance"))
        };
        Ok(LodSettings {
            raster_error: bound("--raster-error", self.raster_error)?,
            raytracing_error: bound("--raytracing-error", self.raytracing_error)?,
        })
    }

    fn sky_color(&self) -> Result<Vec3, String> {
        parse_vec3(&self.sky_color).map_err(|error| format!("--sky-color: {error}"))
    }

    fn camera_position(&self) -> Result<Option<Vec3>, String> {
        self.camera_position
            .as_deref()
            .map(parse_vec3)
            .transpose()
            .map_err(|error| format!("--camera-position: {error}"))
    }

    fn camera_target(&self) -> Result<Option<Vec3>, String> {
        self.camera_target
            .as_deref()
            .map(parse_vec3)
            .transpose()
            .map_err(|error| format!("--camera-target: {error}"))
    }
}

fn parse_vec3(text: &str) -> Result<Vec3, String> {
    let values = text
        .split(',')
        .map(|value| value.trim().parse::<f32>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("{text:?}: {error}"))?;
    match values[..] {
        [x, y, z] => Ok(Vec3::new(x, y, z)),
        _ => Err(format!("{text:?} is not three comma-separated numbers")),
    }
}

/// Where the export and its cache are.
struct SceneRoot {
    dir: PathBuf,
    gltf: PathBuf,
    cache_dir: PathBuf,
}

fn resolve_scene_root(args: &Args) -> Result<SceneRoot, String> {
    let given = match &args.scene_root {
        Some(path) => path.clone(),
        None => match std::env::var_os("ZORAH_ROOT") {
            Some(root) => PathBuf::from(root),
            None => return Err(format!("no scene root: {DOWNLOAD_HINT}")),
        },
    };
    let (dir, gltf) = if given.is_dir() {
        let named = given.join(ROOT_GLTF);
        let gltf = if named.is_file() {
            named
        } else {
            single_gltf_in(&given).ok_or_else(|| {
                format!(
                    "{} holds no {ROOT_GLTF} and no single .gltf: {DOWNLOAD_HINT}",
                    given.display()
                )
            })?
        };
        (given, gltf)
    } else if given.is_file() {
        let dir = given
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        (dir, given)
    } else {
        return Err(format!(
            "{} does not exist: {DOWNLOAD_HINT}",
            given.display()
        ));
    };
    let cache_dir = args
        .cache_dir
        .clone()
        .unwrap_or_else(|| dir.join(CACHE_DIR_NAME));
    Ok(SceneRoot {
        dir,
        gltf,
        cache_dir,
    })
}

fn single_gltf_in(dir: &Path) -> Option<PathBuf> {
    let mut found = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "gltf")
        });
    let first = found.next()?;
    found.next().is_none().then_some(first)
}

/// One worker per 2 GiB beyond an 8 GiB floor for the app and the OS: a
/// worker holds a decoded mesh plus its meshlet build, and the largest
/// meshes here run to tens of millions of triangles.
fn default_bake_workers() -> usize {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let total = system.total_memory();
    let spare = total.saturating_sub(8 << 30);
    ((spare / (2 << 30)) as usize).clamp(1, cores)
}

fn main() -> ExitCode {
    let args: Args = argh::from_env();
    let prepared = match prepare(&args) {
        Ok(prepared) => prepared,
        Err(message) => {
            // No app yet, so no LogPlugin; a plain subscriber carries the error.
            init_plain_logging();
            error!("{message}");
            return ExitCode::FAILURE;
        }
    };
    if args.report_lod_budget {
        init_plain_logging();
        info!("{}", prepared.summary);
        report::run(
            &prepared.root.cache_dir,
            &prepared.settings.stamp(),
            &prepared.jobs,
            prepared.settings.workers,
        );
        return ExitCode::SUCCESS;
    }
    // Only now, so the report leaves a scene without a cache untouched.
    if let Err(message) = create_cache_dir(&prepared) {
        init_plain_logging();
        error!("{message}");
        return ExitCode::FAILURE;
    }
    if args.bake_only {
        init_plain_logging();
        return match bake_headless(prepared) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                error!("{message}");
                ExitCode::FAILURE
            }
        };
    }
    run_app(&args, prepared)
}

fn create_cache_dir(prepared: &Prepared) -> Result<(), String> {
    let cache_dir = &prepared.root.cache_dir;
    std::fs::create_dir_all(cache_dir)
        .map_err(|error| format!("creating {}: {error}", cache_dir.display()))
}

/// The subscriber for the paths that never build an `App`.
fn init_plain_logging() {
    let _ = bevy::log::tracing_subscriber::fmt()
        .with_max_level(bevy::log::tracing::Level::INFO)
        .try_init();
}

/// Everything parsed before the app exists: the flags, the root glTF, and
/// the bake plan.
struct Prepared {
    root: SceneRoot,
    settings: BakeSettings,
    lod: LodSettings,
    jobs: Vec<MeshJob>,
    scene: SceneData,
    materials: Vec<MaterialSpec>,
    view: SceneView,
    sky_color: Vec3,
    camera_position: Option<Vec3>,
    camera_target: Option<Vec3>,
    /// Logged once the app has a subscriber.
    summary: String,
    /// Warnings raised while parsing, logged alongside the summary: nothing
    /// subscribes to `warn!` before the app's `LogPlugin` exists.
    warnings: Vec<String>,
}

fn prepare(args: &Args) -> Result<Prepared, String> {
    // Validated up front so a typo fails before the bake, not after it.
    let lod = args.lod()?;
    let sky_color = args.sky_color()?;
    let camera_position = args.camera_position()?;
    let camera_target = args.camera_target()?;
    let root = resolve_scene_root(args)?;
    let started = Instant::now();
    let gltf_bytes = std::fs::read(&root.gltf)
        .map_err(|error| format!("reading {}: {error}", root.gltf.display()))?;
    let document = gltf::Gltf::from_slice_without_validation(&gltf_bytes)
        .map_err(|error| format!("parsing {}: {error}", root.gltf.display()))?
        .document;
    let mut instances = scene::walk_scene(&document, &root.dir)
        .map_err(|error| format!("walking {}: {error}", root.gltf.display()))?;
    let placed = instances.len();
    if !args.hide_nodes.is_empty() {
        instances.retain(|instance| {
            !args
                .hide_nodes
                .iter()
                .any(|hidden| instance.name.contains(hidden))
        });
    }
    let referenced = instances
        .iter()
        .map(|instance| instance.mesh)
        .collect::<BTreeSet<_>>();
    let selected = referenced
        .iter()
        .copied()
        .take(args.limit_meshes.unwrap_or(usize::MAX))
        .collect::<BTreeSet<_>>();
    let jobs =
        bake::plan_jobs(&document, selected.iter().copied()).map_err(|error| error.to_string())?;
    let materials = materials::read_materials(&document);
    let primitive_materials = document
        .meshes()
        .map(|mesh| {
            mesh.primitives()
                .map(|primitive| primitive.material().index())
                .collect()
        })
        .collect();
    let (view, view_warning) = scene::read_view(&root.gltf);
    let warnings = view_warning.into_iter().collect::<Vec<_>>();

    // Only the selected meshes' instances spawn; the rest are dropped here so
    // the spawner never consults them. Under `--limit-meshes` that includes
    // the fire props, so their proxies only appear when their meshes are in.
    instances.retain(|instance| selected.contains(&instance.mesh));
    let mut instances_by_mesh: HashMap<usize, Vec<usize>> = HashMap::new();
    for (index, instance) in instances.iter().enumerate() {
        instances_by_mesh
            .entry(instance.mesh)
            .or_default()
            .push(index);
    }
    let meshes_of_job = jobs
        .iter()
        .map(|plan| plan.mesh_indices.clone())
        .collect::<Vec<_>>();
    let workers = args
        .bake_workers
        .unwrap_or_else(default_bake_workers)
        .max(1);
    let settings = BakeSettings {
        scene_root: root.dir.clone(),
        cache_dir: root.cache_dir.clone(),
        workers,
        partition_triangles: args.partition_triangles.max(1),
        quantization: args.quantization,
    };
    let triangles = jobs.iter().map(MeshJob::triangles).sum::<u64>();
    let summary = format!(
        "{}: {} nodes, {} meshes, {} materials, {placed} placed instances; \
         {} instances of {} selected meshes spawn; {} geometry files ({triangles} triangles) \
         bake with {workers} workers into {}; parts load pruned to {} m with BLAS cuts at {} m; \
         parsed in {:.1?}",
        root.gltf.display(),
        document.nodes().count(),
        document.meshes().count(),
        materials.len(),
        instances.len(),
        selected.len(),
        jobs.len(),
        root.cache_dir.display(),
        lod.raster_error,
        lod.blas_error(),
        started.elapsed()
    );
    Ok(Prepared {
        root,
        settings,
        lod,
        jobs,
        scene: SceneData {
            instances,
            instances_by_mesh,
            meshes_of_job,
            primitive_materials,
        },
        materials,
        view,
        sky_color,
        camera_position,
        camera_target,
        summary,
        warnings,
    })
}

/// `--bake-only`: the bake alone, polled from this thread, with no window.
fn bake_headless(prepared: Prepared) -> Result<(), String> {
    for warning in &prepared.warnings {
        warn!("{warning}");
    }
    info!("{}", prepared.summary);
    let total = prepared.jobs.len();
    let handle = bake::start(prepared.settings, prepared.jobs);
    let mut last_report = Instant::now();
    loop {
        // Counters first, then the channel: see `BakeHandle::try_recv`.
        let progress = handle.progress();
        while let Some(event) = handle.try_recv() {
            match event {
                BakeEvent::Complete {
                    manifest, reused, ..
                } if !reused => info!(
                    "baked {} ({}): {} parts, {} triangles",
                    manifest.mesh_name,
                    manifest.stem,
                    manifest.parts.len(),
                    manifest.source_triangles
                ),
                BakeEvent::Complete { .. } => {}
                BakeEvent::Failed { job, error } => warn!("bake job {job} failed: {error}"),
            }
        }
        if progress.is_finished() {
            break;
        }
        if last_report.elapsed() >= PROGRESS_INTERVAL {
            info!(
                "baked {}/{} meshes ({} reused, {} failed), {} partitions written, elapsed {:.0?}",
                progress.finished,
                progress.total,
                progress.reused,
                progress.failed,
                progress.partitions,
                progress.elapsed
            );
            last_report = Instant::now();
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let progress = handle.progress();
    info!(
        "bake complete: {} baked, {} reused, {} failed of {total} geometry files in {:.1?}",
        progress.baked, progress.reused, progress.failed, progress.elapsed
    );
    handle.stop();
    if progress.failed > 0 {
        return Err(format!("{} meshes failed to bake", progress.failed));
    }
    Ok(())
}

fn run_app(args: &Args, prepared: Prepared) -> ExitCode {
    let Prepared {
        root,
        settings,
        lod,
        jobs,
        scene,
        materials,
        view,
        sky_color,
        camera_position,
        camera_target,
        summary,
        warnings,
    } = prepared;
    let setup_options = SetupOptions {
        view,
        camera_position,
        camera_target,
        exposure_ev100: args.exposure_ev100.filter(|ev100| ev100.is_finite()),
        auto_exposure: !args.no_auto_exposure,
        exposure_bias: if args.exposure_bias.is_finite() {
            args.exposure_bias
        } else {
            0.0
        },
        sky_illuminance: args.sky_illuminance.max(0.0),
        sky_color,
        sun_illuminance: args.sun_illuminance.max(0.0),
        fire_lumens: args.fire_lumens.max(0.0),
        solari_albedo: args.solari_albedo,
        bake: settings,
        jobs,
    };
    let material_options = MaterialOptions {
        max_texture_size: (args.max_texture_size > 0).then_some(args.max_texture_size),
        preserve_alpha: args.preserve_alpha,
        double_sided_all: args.double_sided_all,
        gltf_specular: args.gltf_specular,
        emissive_boost: args.emissive_boost.max(0.0),
        clay: args.clay,
        albedo_emission_scale: args
            .solari_albedo
            .then(|| materials::albedo_emission_scale(setup_options.base_ev100())),
    };

    let mut app = App::new();
    // DLSS configures Vulkan instance/device creation, so its project ID must
    // exist before `DefaultPlugins` installs the renderer.
    #[cfg(feature = "dlss")]
    app.insert_resource(DlssProjectId(bevy::asset::uuid::uuid!(
        "69d895fa-5bc2-4fb3-b52d-7b77343be702"
    )));
    app.insert_resource(GlobalAmbientLight::NONE)
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(setup_options)
        .insert_resource(scene)
        .insert_resource(MaterialCache::new(materials, material_options))
        .insert_resource(SpawnOptions {
            glass_in_blas: args.glass_in_blas,
            cache_dir: root.cache_dir.clone(),
        })
        // Baked parts load as `cache://<stem>/<file>`; registered before
        // `AssetPlugin` builds its sources.
        .register_asset_source(
            "cache",
            AssetSourceBuilder::platform_default(&root.cache_dir.to_string_lossy(), None),
        )
        .add_plugins((
            DefaultPlugins
                .set(AssetPlugin {
                    // Textures load by the root glTF's own relative URIs.
                    file_path: root.dir.to_string_lossy().into_owned(),
                    // Thousands of textures and cache parts, none with a
                    // .meta file: skip the probe for each.
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                })
                .set(RenderPlugin {
                    // GT7's native working space.
                    working_color_space: WorkingColorSpace::Rec2020,
                    ..default()
                }),
            // Auto-select the best HDR output the surface can present, else SDR.
            hdr::HdrPlugin::default(),
            MeshletPlugin {
                // Zorah's instanced scene exceeds eight million leaf meshlets
                // before hierarchy and candidate pressure; an undersized cull
                // queue presents as blinking geometry, so use the maximum.
                cluster_buffer_slots: 1 << 25,
            },
            FreeCameraPlugin,
            AutoExposurePlugin,
            SolariPlugins,
        ))
        // Cache parts load as `ZorahPart`s, pruned to the raster bound with
        // their BLAS cut alongside; bevy's own `.meshlet_mesh` loader stays
        // registered for the asset type it serves.
        .init_asset::<ZorahPart>()
        .register_asset_loader(ZorahPartLoader { settings: lod });
    if args.diagnostics {
        app.add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            LogDiagnosticsPlugin::default(),
        ));
    }
    if let Some(delay) = args.screenshot_after {
        app.insert_resource(ScreenshotAfter::new(delay.max(0.0)))
            .add_systems(Update, setup::screenshot_after);
    }
    app.init_state::<ZorahState>()
        .add_systems(
            Startup,
            (
                move || {
                    for warning in &warnings {
                        warn!("{warning}");
                    }
                    info!("{summary}");
                },
                setup::setup_hdr_calibration,
                setup::setup,
            ),
        )
        .add_systems(
            Update,
            runner::stream_scene.run_if(|state: Res<State<ZorahState>>| {
                matches!(state.get(), ZorahState::Baking | ZorahState::LoadingScene)
            }),
        )
        .add_systems(
            Update,
            runner::warm_up_raytracing.run_if(in_state(ZorahState::WarmingRaytracing)),
        )
        .add_systems(Update, setup::dump_camera_and_screenshot)
        .run();
    ExitCode::SUCCESS
}
