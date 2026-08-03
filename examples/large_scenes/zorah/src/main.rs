//! Zorah's source geometry rendered as Bevy meshlets and traced by Solari.

#![allow(clippy::too_many_arguments)]

mod zorah_bundle;

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};

use argh::FromArgs;
use bevy::{
    asset::LoadState,
    camera::{CameraMainTextureUsages, Exposure, Hdr},
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    core_pipeline::{
        prepass::{DeferredPrepass, DepthPrepass},
        tonemapping::{GranTurismo7Params, Tonemapping},
    },
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    light::{
        atmosphere::{Falloff, PhaseFunction, ScatteringMedium, ScatteringTerm},
        Atmosphere, AtmosphereEnvironmentMapLight, SunDisk,
    },
    math::Affine2,
    pbr::experimental::meshlet::{MeshletMesh, MeshletMesh3d, MeshletPlugin},
    pbr::{AtmosphereSettings, DefaultOpaqueRendererMethod, MeshMaterial3d},
    post_process::bloom::{Bloom, BloomScatterModel},
    prelude::*,
    render::{
        render_resource::{Face, TextureUsages},
        renderer::RenderDevice,
        working_color_space::WorkingColorSpace,
        RenderPlugin,
    },
    solari::prelude::{
        RaytracingMesh3d, RaytracingMesh3dGeometryError, RaytracingSceneStatus,
        RaytracingSceneStatusSnapshot, SolariEnvironmentLight, SolariLighting, SolariPlugins,
    },
    window::{AutoField, DisplayCalibrationPolicy, DisplayTarget, PrimaryWindow},
};
use serde::Deserialize;

use zorah_bundle::{ZorahBundle, ZorahBundlePlugin};

#[cfg(feature = "dlss")]
use bevy::anti_alias::dlss::{
    Dlss, DlssPerfQualityMode, DlssProjectId, DlssRayReconstructionFeature,
    DlssRayReconstructionSupported,
};

// Opt-in HDR setup shared with the other HDR examples: keeps the primary
// window's `DisplayTarget` on the best transfer the surface can present
// (PQ/HDR10, then scRGB-linear, then extended sRGB), else stays SDR.
#[path = "../../../helpers/hdr.rs"]
mod hdr;

const ASSET_ROOT: &str = "assets";
const MAX_NEW_PARTITION_LOADS_PER_FRAME: usize = 32;
const MAX_NEW_GEOMETRY_VERTICES_PER_FRAME: usize = 2_000_000;
const MAX_RASTER_INSTANCES_PER_FRAME: usize = 512;
const MAX_RAYTRACING_INSTANCES_PER_FRAME: usize = 512;
const BLAS_BUILD_VERTICES_PER_FRAME: u64 = 2_000_000;
const BLAS_WARMUP_MARGIN_FRAMES: u64 = 60;
const BLAS_PROGRESS_LOG_INTERVAL_FRAMES: u64 = 120;
// UE's legacy non-inverse-square lights use an artistic, unitless intensity.
// Treating one unit as 100,000 lumens made ordinary Zorah values such as 2,000
// into tiny 200-million-lumen emitters, which showed up as firefly noise in
// Solari. This scale keeps the converted fixtures in a plausible range while
// preserving their authored relative brightness.
const UE_UNITLESS_LIGHT_LUMENS: f32 = 100.0;
// UE SkyLight intensity is a multiplier over a captured environment rather
// than a physical unit. Until Solari supports environment maps directly, use
// the same source-unit bridge for raster ambient and traced environment light.
// Keeping two different scales caused Greenhouse to jump from a plausible
// preview to a 12.5x brighter environment as soon as Solari became active.
const UE_SKY_LIGHT_LUX_PER_UNIT: f32 = 80.0;

fn ue_sky_light_illuminance(intensity: f32) -> f32 {
    intensity.max(0.0) * UE_SKY_LIGHT_LUX_PER_UNIT
}
const UE_BLACK_UNLIT_MATERIAL: &str =
    "/Engine/EngineDebugMaterials/BlackUnlitMaterial.BlackUnlitMaterial";
const UE_WORLD_GRID_MATERIAL: &str = "/Engine/EngineMaterials/WorldGridMaterial.WorldGridMaterial";
/// `MaterialRecord::kind` the converter writes for a material the mesh
/// references but the project download does not contain.
const MISSING_SOURCE_MATERIAL: &str = "MissingSourceMaterial";
const BASE_COLOR_TEXTURE_NAMES: &[&str] = &[
    "basecolortexture",
    "diffusetexture",
    "basecolor",
    "albedo",
    "diffuse",
    "marblebasecolor",
    "goldbasecolor",
];
const NORMAL_TEXTURE_NAMES: &[&str] = &[
    "normal",
    "normalmap",
    "normaltexture",
    "marblenormal",
    "marblechippingnormal",
    "goldbasenormal",
];
const ORM_TEXTURE_NAMES: &[&str] = &[
    "orm",
    "ors",
    "occlusionroughnessmetallic",
    "packedorm",
    "marbleorm",
    "goldbaseorm",
];
const EMISSIVE_TEXTURE_NAMES: &[&str] = &["emissive", "emissivemask", "emissivetexture", "extra"];
const EMISSIVE_INTENSITY_NAMES: &[&str] = &[
    "globalemissionintensity",
    "emissiveintensity",
    "emissionintensity",
];

#[derive(FromArgs)]
/// Render a converted Zorah World Partition level with meshlets plus Solari.
struct Args {
    /// scene manifest relative to the Zorah assets directory
    #[argh(option, default = "String::from(\"scenes/ThroneRoom_Level.json\")")]
    scene: String,

    /// geometry manifest relative to the Zorah assets directory
    #[argh(option, default = "String::from(\"geometry.json\")")]
    geometry: String,

    /// material manifest relative to the Zorah assets directory
    #[argh(option, default = "String::from(\"materials.json\")")]
    materials: String,

    /// exported texture manifest relative to the Zorah assets directory
    #[argh(option, default = "String::from(\"textures.exported.json\")")]
    textures: String,

    /// print Bevy's periodic frame-time diagnostics
    #[argh(switch)]
    diagnostics: bool,

    /// render the conventional deferred meshlet scene without building Solari BLASes
    #[argh(switch)]
    raster_only: bool,

    /// preserve UE masked/translucent blend modes instead of forcing raster/Solari parity
    #[argh(switch)]
    preserve_alpha: bool,

    /// show unlit base-color textures; missing material assignments are magenta
    #[argh(switch)]
    unlit_textures: bool,

    /// camera position as Bevy-space x,y,z (overrides the selected level preset)
    #[argh(option)]
    camera_position: Option<String>,

    /// camera look target as Bevy-space x,y,z (overrides the selected level preset)
    #[argh(option)]
    camera_target: Option<String>,

    /// fixed camera exposure in EV100 (overrides the selected level's UE post-process volume)
    #[argh(option)]
    exposure_ev100: Option<f32>,
}

#[derive(Resource)]
struct RuntimeOptions {
    raster_only: bool,
    preserve_alpha: bool,
    unlit_textures: bool,
    camera_position: Option<Vec3>,
    camera_target: Option<Vec3>,
    exposure_ev100: Option<f32>,
}

#[derive(Resource)]
struct ConvertedWorld {
    level: String,
    actors: Vec<ActorRecord>,
    post_process: Option<PostProcessRecord>,
    geometry: HashMap<String, ConvertedMesh>,
    materials: HashMap<String, MaterialRecord>,
    textures: HashMap<String, TextureExportRecord>,
}

struct ConvertedMesh {
    partitions: Vec<PartitionRecord>,
    material_slots: Vec<Option<String>>,
}

#[derive(Resource)]
struct PendingScene {
    partitions: Vec<PendingPartition>,
    loaded_assets: HashMap<String, LoadedPartitionAssets>,
    bundle_roots: Vec<String>,
    bundle_cursor: usize,
    active_bundle: Option<(String, Handle<ZorahBundle>, Instant)>,
    loaded_bundle_roots: HashSet<String>,
    loaded_bundles: Vec<Handle<ZorahBundle>>,
    prepared_meshes: HashSet<AssetId<Mesh>>,
    failed_meshes: HashSet<AssetId<Mesh>>,
    raytracing_instances: Vec<PendingRaytracingInstance>,
    raytracing_cursor: usize,
    expected_blas: usize,
    warmup_frames_remaining: u64,
    warmup_timeout_reported: bool,
    warmup_progress_log_frames_remaining: u64,
    warmup_started_at: Option<Instant>,
    unique_blas_vertices: u64,
    spawned: usize,
    failed: usize,
    reported_done: bool,
}

#[derive(States, Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ZorahState {
    #[default]
    LoadingTextureBundles,
    ReleasingUnusedTextures,
    LoadingScene,
    WarmingRaytracing,
    Running,
}

struct PendingPartition {
    geometry: String,
    mesh_path: String,
    meshlet_path: String,
    assets: Option<LoadedPartitionAssets>,
    material: Handle<StandardMaterial>,
    transform: Transform,
    vertices: usize,
    blas_vertices: usize,
    blas_achieved_error: f32,
    raytracing_only: bool,
    spawned: bool,
}

#[derive(Clone)]
struct LoadedPartitionAssets {
    mesh: Handle<Mesh>,
    meshlet: Handle<MeshletMesh>,
}

struct PendingRaytracingInstance {
    entity: Entity,
    mesh: Handle<Mesh>,
    geometry_error: f32,
}

#[derive(Resource)]
struct TextureBundlePreload {
    roots: Vec<String>,
    cursor: usize,
    active: Option<(String, Handle<ZorahBundle>)>,
    loaded: Vec<Handle<ZorahBundle>>,
}

/// Texture bundle roots that failed to load. Materials referencing images inside
/// them are built untextured; binding a handle that will never resolve makes the
/// material stall in bind-group preparation forever, which silently removes
/// every instance using it from the meshlet shading passes.
#[derive(Resource, Default)]
struct FailedTextureBundles(HashSet<String>);

#[derive(Component)]
struct ZorahCamera;

#[derive(Deserialize)]
struct SceneManifest {
    level: String,
    actors: Vec<ActorRecord>,
}

#[derive(Deserialize)]
struct ActorRecord {
    #[serde(rename = "name")]
    _name: String,
    #[serde(rename = "label")]
    _label: Option<String>,
    #[serde(rename = "type")]
    kind: String,
    transform: UeTransform,
    #[serde(default)]
    hidden: bool,
    components: Vec<ComponentRecord>,
    #[serde(default)]
    lights: Vec<LightRecord>,
    #[serde(default)]
    atmosphere: Option<SkyAtmosphereRecord>,
    #[serde(default)]
    height_fog: Option<HeightFogRecord>,
    #[serde(default)]
    post_process: Option<PostProcessRecord>,
}

#[derive(Default, Deserialize)]
struct SkyAtmosphereRecord {
    #[serde(rename = "name")]
    _name: String,
    #[serde(default = "default_true")]
    visible: bool,
    #[serde(default)]
    hidden_in_game: bool,
    #[serde(default)]
    transform_mode: Option<String>,
    #[serde(default)]
    bottom_radius_km: Option<f32>,
    #[serde(default)]
    atmosphere_height_km: Option<f32>,
    #[serde(default)]
    ground_albedo: Option<UeColor>,
    #[serde(default)]
    rayleigh_scattering_scale: Option<f32>,
    #[serde(default)]
    rayleigh_scattering_per_km: Option<UeLinearColor>,
    #[serde(default)]
    rayleigh_exponential_distribution_km: Option<f32>,
    #[serde(default)]
    mie_scattering_scale: Option<f32>,
    #[serde(default)]
    mie_scattering_per_km: Option<UeLinearColor>,
    #[serde(default)]
    mie_absorption_scale: Option<f32>,
    #[serde(default)]
    mie_absorption_per_km: Option<UeLinearColor>,
    #[serde(default)]
    mie_anisotropy: Option<f32>,
    #[serde(default)]
    mie_exponential_distribution_km: Option<f32>,
    #[serde(default)]
    other_absorption_scale: Option<f32>,
    #[serde(default)]
    other_absorption_per_km: Option<UeLinearColor>,
    #[serde(default)]
    sky_luminance_factor: Option<UeLinearColor>,
    #[serde(default)]
    sky_and_aerial_perspective_luminance_factor: Option<UeLinearColor>,
}

#[derive(Default, Deserialize)]
struct HeightFogRecord {
    #[serde(rename = "name")]
    _name: String,
    #[serde(default = "default_true")]
    visible: bool,
    #[serde(default)]
    hidden_in_game: bool,
    #[serde(default)]
    fog_density: Option<f32>,
    #[serde(default)]
    fog_height_falloff: Option<f32>,
    #[serde(default)]
    fog_inscattering_color: Option<UeLinearColor>,
    #[serde(default)]
    enable_volumetric_fog: Option<bool>,
    #[serde(default)]
    volumetric_fog_albedo: Option<UeColor>,
    #[serde(default)]
    volumetric_fog_extinction_scale: Option<f32>,
    #[serde(default)]
    volumetric_fog_scattering_distribution: Option<f32>,
}

#[derive(Clone, Deserialize)]
struct PostProcessRecord {
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default)]
    unbound: bool,
    #[serde(default)]
    priority: f32,
    #[serde(default = "default_blend_weight")]
    blend_weight: f32,
    #[serde(default)]
    bloom_method: Option<String>,
    #[serde(default)]
    bloom_intensity: Option<f32>,
    // UE's ACES film curve knobs, exported so the authored curve shape is
    // visible in the manifest. Bevy's tone-mapping operators are not
    // parameterized by slope/toe/shoulder, so nothing reads them yet; see the
    // tonemapper choice in `setup`.
    #[serde(default, rename = "film_slope")]
    _film_slope: Option<f32>,
    #[serde(default, rename = "film_toe")]
    _film_toe: Option<f32>,
    #[serde(default, rename = "film_shoulder")]
    _film_shoulder: Option<f32>,
    #[serde(default, rename = "film_black_clip")]
    _film_black_clip: Option<f32>,
    #[serde(default, rename = "film_white_clip")]
    _film_white_clip: Option<f32>,
    #[serde(default, rename = "auto_exposure_method")]
    _auto_exposure_method: Option<String>,
    #[serde(default)]
    auto_exposure_min_ev100: Option<f32>,
    #[serde(default)]
    auto_exposure_max_ev100: Option<f32>,
    #[serde(default)]
    auto_exposure_bias: Option<f32>,
}

#[derive(Deserialize)]
struct LightRecord {
    name: String,
    #[serde(rename = "type")]
    kind: String,
    transform: UeTransform,
    #[serde(default = "default_true")]
    visible: bool,
    #[serde(default)]
    hidden_in_game: bool,
    #[serde(default = "default_true")]
    affects_world: bool,
    #[serde(default = "default_true")]
    #[serde(rename = "cast_shadows")]
    _cast_shadows: bool,
    intensity: f32,
    intensity_units: String,
    color: UeColor,
    #[serde(default)]
    use_temperature: bool,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_attenuation_radius")]
    attenuation_radius: f32,
    #[serde(default)]
    source_radius: f32,
    #[serde(default)]
    soft_source_radius: f32,
    #[serde(default)]
    inner_cone_angle: f32,
    #[serde(default = "default_outer_cone_angle")]
    outer_cone_angle: f32,
    #[serde(default = "default_light_source_angle")]
    light_source_angle: f32,
    #[serde(default)]
    ies_texture: Option<String>,
    #[serde(default)]
    light_function_material: Option<String>,
    #[serde(default)]
    real_time_capture: bool,
}

#[derive(Deserialize)]
struct UeColor {
    r: u8,
    g: u8,
    b: u8,
    #[serde(default = "default_alpha")]
    a: u8,
}

#[derive(Clone, Copy, Default, Deserialize)]
struct UeLinearColor {
    r: f32,
    g: f32,
    b: f32,
    #[serde(rename = "a")]
    _a: f32,
}

#[derive(Deserialize)]
struct ComponentRecord {
    mesh: Option<String>,
    transform: UeTransform,
    #[serde(default = "default_true")]
    visible: bool,
    #[serde(default)]
    hidden_in_game: bool,
    instances: Option<Vec<UeTransform>>,
    #[serde(default)]
    override_materials: Vec<String>,
}

#[derive(Clone, Deserialize)]
struct UeTransform {
    translation: UeVec3,
    rotation: UeQuat,
    scale: UeVec3,
}

#[derive(Clone, Deserialize)]
struct UeVec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Deserialize)]
struct UeQuat {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

#[derive(Deserialize)]
struct GeometryManifest {
    meshes: Vec<GeometryRecord>,
}

#[derive(Deserialize)]
struct GeometryRecord {
    object: String,
    #[serde(default)]
    parts_manifest: Option<String>,
    #[serde(default)]
    partitions: Vec<PartitionRecord>,
    #[serde(default)]
    material_slots: Vec<GeometryMaterialSlot>,
}

#[derive(Deserialize)]
struct GeometryMaterialSlot {
    material: Option<String>,
}

#[derive(Clone, Deserialize)]
struct PartitionRecord {
    geometry: String,
    #[serde(default)]
    mesh: Option<String>,
    meshlet: String,
    material_slot: usize,
    /// UE `StaticMaterials` slot for `material_slot`, resolved by the converter
    /// through the mesh's `FMeshSectionInfoMap`. Component `override_materials`
    /// arrays are indexed by this, not by the section index. Manifests written
    /// before the field existed fall back to the section index.
    #[serde(default)]
    material_index: Option<usize>,
    #[serde(default)]
    vertices: usize,
    #[serde(default)]
    blas_vertices: usize,
    #[serde(default)]
    blas_achieved_error: f32,
    #[serde(default)]
    uv_min: Option<[f32; 2]>,
    #[serde(default)]
    uv_max: Option<[f32; 2]>,
    #[serde(default)]
    aabb_min: Option<[f32; 3]>,
    #[serde(default)]
    aabb_max: Option<[f32; 3]>,
}

#[derive(Deserialize)]
struct PartitionManifest {
    partitions: Vec<PartitionRecord>,
}

#[derive(Deserialize)]
struct MaterialManifest {
    materials: Vec<MaterialRecord>,
}

#[derive(Clone, Deserialize)]
struct MaterialRecord {
    object: String,
    #[serde(rename = "type", default)]
    kind: Option<String>,
    parent: Option<String>,
    #[serde(default)]
    emissive: Option<bool>,
    #[serde(default)]
    scalars: Vec<ScalarParameter>,
    #[serde(default)]
    vectors: Vec<VectorParameter>,
    #[serde(default)]
    textures: Vec<TextureParameter>,
    #[serde(default)]
    base_overrides: BaseMaterialOverrides,
}

#[derive(Clone, Default, Deserialize)]
struct BaseMaterialOverrides {
    #[serde(rename = "BlendMode")]
    blend_mode: Option<String>,
    #[serde(rename = "TwoSided")]
    two_sided: Option<bool>,
    #[serde(rename = "OpacityMaskClipValue")]
    opacity_mask_clip_value: Option<f32>,
    #[serde(rename = "bOverride_BlendMode")]
    override_blend_mode: Option<bool>,
    #[serde(rename = "bOverride_TwoSided")]
    override_two_sided: Option<bool>,
}

#[derive(Clone, Deserialize)]
struct ScalarParameter {
    name: String,
    association: String,
    index: i32,
    value: f32,
}

#[derive(Clone, Deserialize)]
struct VectorParameter {
    name: String,
    association: String,
    index: i32,
    value: String,
}

#[derive(Clone, Deserialize)]
struct TextureParameter {
    name: String,
    association: String,
    index: i32,
    value: Option<String>,
}

#[derive(Deserialize)]
struct TextureExportManifest {
    exported: Vec<TextureExportRecord>,
}

#[derive(Clone, Deserialize)]
struct TextureExportRecord {
    object: String,
    output: String,
    #[serde(default = "default_one")]
    source_grid_columns: u32,
    #[serde(default = "default_one")]
    source_grid_rows: u32,
}

#[derive(Clone)]
struct EffectiveMaterial {
    scalars: Vec<ScalarParameter>,
    vectors: Vec<VectorParameter>,
    textures: Vec<TextureParameter>,
    emissive: bool,
    blend_mode: SourceBlendMode,
    two_sided: bool,
    opacity_mask_clip_value: f32,
}

impl Default for EffectiveMaterial {
    fn default() -> Self {
        Self {
            scalars: Vec::new(),
            vectors: Vec::new(),
            textures: Vec::new(),
            emissive: false,
            blend_mode: SourceBlendMode::Opaque,
            two_sided: false,
            opacity_mask_clip_value: 0.3333,
        }
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
enum SourceBlendMode {
    #[default]
    Opaque,
    Masked,
    Translucent,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SourceMaterialRenderProperties {
    alpha_mode: AlphaMode,
    double_sided: bool,
    cull_mode: Option<Face>,
}

fn default_true() -> bool {
    true
}

fn default_blend_weight() -> f32 {
    1.0
}

fn default_temperature() -> f32 {
    6500.0
}

fn default_attenuation_radius() -> f32 {
    1000.0
}

fn default_outer_cone_angle() -> f32 {
    44.0
}

fn default_light_source_angle() -> f32 {
    0.5357
}

fn default_alpha() -> u8 {
    255
}

fn default_one() -> u32 {
    1
}

fn read_json<T: for<'de> Deserialize<'de>>(
    path: &Path,
    read_asset: &dyn Fn(&Path) -> std::io::Result<Vec<u8>>,
) -> T {
    let bytes = read_asset(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}

fn load_converted_world(
    args: &Args,
    read_asset: &dyn Fn(&Path) -> std::io::Result<Vec<u8>>,
) -> ConvertedWorld {
    let scene_path = Path::new(&args.scene);
    let geometry_path = Path::new(&args.geometry);
    let scene: SceneManifest = read_json(scene_path, read_asset);
    let geometry_manifest: GeometryManifest = read_json(geometry_path, read_asset);
    let mut geometry = HashMap::new();
    for mesh in geometry_manifest.meshes {
        let GeometryRecord {
            object,
            parts_manifest,
            partitions,
            material_slots,
        } = mesh;
        let (parent, partitions) = if let Some(parts_manifest) = parts_manifest {
            let partition_path = PathBuf::from(parts_manifest);
            let partition_manifest: PartitionManifest = read_json(&partition_path, read_asset);
            let parent = partition_path
                .parent()
                .unwrap_or_else(|| Path::new(""))
                .to_path_buf();
            (parent, partition_manifest.partitions)
        } else {
            (PathBuf::new(), partitions)
        };
        let partitions = partitions
            .into_iter()
            .map(|partition| PartitionRecord {
                geometry: asset_path(&parent.join(partition.geometry)),
                mesh: partition.mesh.map(|mesh| asset_path(&parent.join(mesh))),
                meshlet: asset_path(&parent.join(partition.meshlet)),
                material_slot: partition.material_slot,
                material_index: partition.material_index,
                vertices: partition.vertices,
                blas_vertices: partition.blas_vertices,
                blas_achieved_error: partition.blas_achieved_error,
                uv_min: partition.uv_min,
                uv_max: partition.uv_max,
                aabb_min: partition.aabb_min,
                aabb_max: partition.aabb_max,
            })
            .collect();
        geometry.insert(
            object,
            ConvertedMesh {
                partitions,
                material_slots: material_slots
                    .into_iter()
                    .map(|slot| slot.material)
                    .collect(),
            },
        );
    }
    let material_path = Path::new(&args.materials);
    let materials = if read_asset(material_path).is_ok() {
        let manifest: MaterialManifest = read_json(material_path, read_asset);
        manifest
            .materials
            .into_iter()
            .map(|material| (material.object.clone(), material))
            .collect()
    } else {
        warn!(
            "material manifest {} is absent; using the fallback material",
            material_path.display()
        );
        HashMap::new()
    };
    let texture_path = Path::new(&args.textures);
    let textures = if read_asset(texture_path).is_ok() {
        let manifest: TextureExportManifest = read_json(texture_path, read_asset);
        manifest
            .exported
            .into_iter()
            .map(|texture| (texture.object.clone(), texture))
            .collect()
    } else {
        warn!(
            "texture manifest {} is absent; using scalar-only materials",
            texture_path.display()
        );
        HashMap::new()
    };
    let post_process = active_unbound_post_process(&scene.actors)
        .cloned()
        .or_else(|| legacy_zorah_post_process(&scene.level));
    ConvertedWorld {
        level: scene.level,
        actors: scene.actors,
        post_process,
        geometry,
        materials,
        textures,
    }
}

fn asset_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn asset_base_path() -> PathBuf {
    env::var_os("BEVY_ASSET_ROOT")
        .or_else(|| env::var_os("CARGO_MANIFEST_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            env::current_exe()
                .expect("executable path must be available")
                .parent()
                .expect("executable must have a parent directory")
                .to_path_buf()
        })
}

fn parse_vec3_arg(name: &str, value: Option<&str>) -> Option<Vec3> {
    let value = value?;
    let components: Vec<_> = value.split(',').map(str::trim).collect();
    assert!(
        components.len() == 3,
        "{name} must contain exactly three comma-separated numbers"
    );
    Some(Vec3::new(
        components[0]
            .parse()
            .unwrap_or_else(|_| panic!("{name} has an invalid x component")),
        components[1]
            .parse()
            .unwrap_or_else(|_| panic!("{name} has an invalid y component")),
        components[2]
            .parse()
            .unwrap_or_else(|_| panic!("{name} has an invalid z component")),
    ))
}

fn level_camera_placement(level: &str, center: Vec3, extent: f32) -> (Vec3, Vec3) {
    match level {
        // These presets face the authored light groups from unobstructed,
        // human-scale positions. The manifest-wide origin bounds include
        // distant courtyards and light-blocker shells, so they are not a
        // useful camera framing heuristic for these World Partition levels.
        "ThroneRoom_Level" => (Vec3::new(0.0, 4.5, 48.0), Vec3::new(0.0, 4.0, 40.0)),
        "Restir_Level" => (Vec3::new(0.0, 5.0, 18.0), Vec3::new(0.0, 4.0, 2.0)),
        "GreenHouse_Level" => (Vec3::new(-2.0, 3.0, 14.0), Vec3::new(-2.0, 2.0, 6.0)),
        _ => (center + Vec3::new(0.0, extent * 0.15, extent * 0.5), center),
    }
}

fn parameter_key(name: &str, association: &str, index: i32) -> (String, String, i32) {
    (name.to_ascii_lowercase(), association.to_string(), index)
}

fn merge_effective(parent: &mut EffectiveMaterial, child: &MaterialRecord) {
    if let Some(emissive) = child.emissive {
        parent.emissive = emissive;
    }
    for parameter in &child.scalars {
        let key = parameter_key(&parameter.name, &parameter.association, parameter.index);
        if let Some(existing) = parent.scalars.iter_mut().find(|existing| {
            parameter_key(&existing.name, &existing.association, existing.index) == key
        }) {
            *existing = parameter.clone();
        } else {
            parent.scalars.push(parameter.clone());
        }
    }
    for parameter in &child.textures {
        let key = parameter_key(&parameter.name, &parameter.association, parameter.index);
        if let Some(existing) = parent.textures.iter_mut().find(|existing| {
            parameter_key(&existing.name, &existing.association, existing.index) == key
        }) {
            *existing = parameter.clone();
        } else {
            parent.textures.push(parameter.clone());
        }
    }
    for parameter in &child.vectors {
        let key = parameter_key(&parameter.name, &parameter.association, parameter.index);
        if let Some(existing) = parent.vectors.iter_mut().find(|existing| {
            parameter_key(&existing.name, &existing.association, existing.index) == key
        }) {
            *existing = parameter.clone();
        } else {
            parent.vectors.push(parameter.clone());
        }
    }
    let overrides = &child.base_overrides;
    if overrides.override_blend_mode == Some(true) && overrides.blend_mode.is_none() {
        // UE omits enum values equal to the default from JSON. An explicit
        // override with no serialized value therefore means opaque.
        parent.blend_mode = SourceBlendMode::Opaque;
    }
    if let Some(blend_mode) = overrides.blend_mode.as_deref() {
        parent.blend_mode = match blend_mode.rsplit("::").next().unwrap_or(blend_mode) {
            "BLEND_Masked" => SourceBlendMode::Masked,
            "BLEND_Translucent" | "BLEND_Additive" | "BLEND_Modulate" => {
                SourceBlendMode::Translucent
            }
            _ => SourceBlendMode::Opaque,
        };
    }
    if overrides.override_two_sided == Some(true) && overrides.two_sided.is_none() {
        // As above, an omitted overridden bool is false.
        parent.two_sided = false;
    }
    if let Some(two_sided) = overrides.two_sided {
        parent.two_sided = two_sided;
    }
    if let Some(clip) = overrides.opacity_mask_clip_value {
        parent.opacity_mask_clip_value = clip.clamp(0.0, 1.0);
    }
}

fn resolve_effective_material(
    object: &str,
    records: &HashMap<String, MaterialRecord>,
    cache: &mut HashMap<String, EffectiveMaterial>,
    stack: &mut Vec<String>,
) -> EffectiveMaterial {
    if let Some(cached) = cache.get(object) {
        return cached.clone();
    }
    if stack.iter().any(|entry| entry == object) {
        warn!("material inheritance cycle at {object}");
        return EffectiveMaterial::default();
    }
    let Some(record) = records.get(object) else {
        return EffectiveMaterial::default();
    };
    stack.push(object.to_string());
    let mut result = record
        .parent
        .as_deref()
        .map_or_else(EffectiveMaterial::default, |parent| {
            resolve_effective_material(parent, records, cache, stack)
        });
    merge_effective(&mut result, record);
    stack.pop();
    cache.insert(object.to_string(), result.clone());
    result
}

fn normalized_parameter_name(name: &str) -> String {
    name.chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn source_material_render_properties(
    material: &EffectiveMaterial,
) -> SourceMaterialRenderProperties {
    let alpha_mode = match material.blend_mode {
        SourceBlendMode::Opaque => AlphaMode::Opaque,
        SourceBlendMode::Masked => AlphaMode::Mask(material.opacity_mask_clip_value),
        SourceBlendMode::Translucent => AlphaMode::Blend,
    };
    SourceMaterialRenderProperties {
        alpha_mode,
        double_sided: material.two_sided,
        cull_mode: (!material.two_sided).then_some(Face::Back),
    }
}

fn runtime_alpha_mode(material: &EffectiveMaterial, preserve_alpha: bool) -> AlphaMode {
    if preserve_alpha {
        source_material_render_properties(material).alpha_mode
    } else {
        // Solari currently traces the same geometry as opaque and does not
        // evaluate texture alpha. Keeping meshlet raster opaque as well avoids
        // disappearing cards and raster/raytracing disagreement while the
        // alpha-tested ray-query path is still missing.
        AlphaMode::Opaque
    }
}

fn select_texture<'a>(material: &'a EffectiveMaterial, desired_names: &[&str]) -> Option<&'a str> {
    select_texture_parameter(material, desired_names)
        .and_then(|parameter| parameter.value.as_deref())
}

fn select_texture_parameter<'a>(
    material: &'a EffectiveMaterial,
    desired_names: &[&str],
) -> Option<&'a TextureParameter> {
    material
        .textures
        .iter()
        .filter_map(|parameter| {
            let normalized = normalized_parameter_name(&parameter.name);
            let name_rank = desired_names.iter().position(|name| *name == normalized)?;
            let scope_rank = match parameter.index {
                -1 => 0,
                0 => 1,
                value if value > 0 => 2 + value as usize,
                _ => usize::MAX,
            };
            parameter.value.as_ref()?;
            Some(((name_rank, scope_rank), parameter))
        })
        .min_by_key(|(rank, _)| *rank)
        .map(|(_, parameter)| parameter)
}

fn texture_carries_metallic(parameter: Option<&TextureParameter>) -> bool {
    parameter.is_some_and(|parameter| {
        matches!(
            normalized_parameter_name(&parameter.name).as_str(),
            "orm" | "occlusionroughnessmetallic" | "packedorm" | "marbleorm" | "goldbaseorm"
        )
    })
}

fn select_scalar(material: &EffectiveMaterial, desired_names: &[&str]) -> Option<f32> {
    material
        .scalars
        .iter()
        .filter_map(|parameter| {
            let normalized = normalized_parameter_name(&parameter.name);
            let name_rank = desired_names.iter().position(|name| *name == normalized)?;
            let scope_rank = match parameter.index {
                -1 => 0,
                0 => 1,
                value if value > 0 => 2 + value as usize,
                _ => usize::MAX,
            };
            Some(((name_rank, scope_rank), parameter.value))
        })
        .min_by_key(|(rank, _)| *rank)
        .map(|(_, value)| value)
}

fn select_color(material: &EffectiveMaterial, desired_names: &[&str]) -> Option<Color> {
    let value = material
        .vectors
        .iter()
        .filter_map(|parameter| {
            let normalized = normalized_parameter_name(&parameter.name);
            let name_rank = desired_names.iter().position(|name| *name == normalized)?;
            let scope_rank = match parameter.index {
                -1 => 0,
                0 => 1,
                value if value > 0 => 2 + value as usize,
                _ => usize::MAX,
            };
            Some(((name_rank, scope_rank), parameter.value.as_str()))
        })
        .min_by_key(|(rank, _)| *rank)?
        .1
        .split_whitespace()
        .next()?;
    // The byte slicing below is only valid for the ASCII hex every producer of
    // this field emits.
    if !value.is_ascii() {
        return None;
    }
    let (red, green, blue, alpha) = match value.len() {
        6 => (
            u8::from_str_radix(&value[0..2], 16).ok()?,
            u8::from_str_radix(&value[2..4], 16).ok()?,
            u8::from_str_radix(&value[4..6], 16).ok()?,
            255,
        ),
        8 => (
            u8::from_str_radix(&value[0..2], 16).ok()?,
            u8::from_str_radix(&value[2..4], 16).ok()?,
            u8::from_str_radix(&value[4..6], 16).ok()?,
            u8::from_str_radix(&value[6..8], 16).ok()?,
        ),
        _ => return None,
    };
    Some(Color::srgba_u8(red, green, blue, alpha))
}

fn material_base_color(material: &EffectiveMaterial) -> Color {
    let tint = select_color(
        material,
        &[
            "basecolortint",
            "diffusetint",
            "tint",
            "marblebasecolortinta",
            "waterbasecolor",
        ],
    )
    .unwrap_or(Color::WHITE)
    .to_linear();
    let luminance = select_scalar(material, &["luminance"])
        .filter(|value| value.is_finite())
        .unwrap_or(1.0)
        .max(0.0);
    Color::LinearRgba(LinearRgba::new(
        (tint.red * luminance).clamp(0.0, 1.0),
        (tint.green * luminance).clamp(0.0, 1.0),
        (tint.blue * luminance).clamp(0.0, 1.0),
        tint.alpha,
    ))
}

/// Resolves the material a partition renders with. UE indexes a component's
/// `OverrideMaterials` by static-mesh material slot while a partition's
/// `material_slot` is its LOD0 section index, and the two differ whenever the
/// mesh carries a non-identity `FMeshSectionInfoMap`.
fn partition_material<'a>(
    partition: &PartitionRecord,
    material_slots: &'a [Option<String>],
    overrides: &'a [String],
) -> Option<&'a str> {
    overrides
        .get(partition.material_index.unwrap_or(partition.material_slot))
        .filter(|object| !object.is_empty())
        .map(String::as_str)
        .or_else(|| {
            material_slots
                .get(partition.material_slot)
                .and_then(Option::as_deref)
        })
}

/// Every material a visible partition renders with, mapped to the mesh slots
/// that ask for it as `<mesh object> slot <LOD0 section index>`. The slots name
/// the requester whether the assignment came from the mesh or from a component
/// override, which is what a diagnostic material has to report.
fn used_material_objects(converted: &ConvertedWorld) -> BTreeMap<String, BTreeSet<String>> {
    let mut used: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for actor in &converted.actors {
        if actor.hidden {
            continue;
        }
        for component in &actor.components {
            if !component.visible || component.hidden_in_game {
                continue;
            }
            let Some(mesh_name) = component.mesh.as_ref() else {
                continue;
            };
            let Some(mesh) = converted.geometry.get(mesh_name) else {
                continue;
            };
            for partition in &mesh.partitions {
                if let Some(material) = partition_material(
                    partition,
                    &mesh.material_slots,
                    &component.override_materials,
                ) {
                    used.entry(material.to_string())
                        .or_default()
                        .insert(format!("{mesh_name} slot {}", partition.material_slot));
                }
            }
        }
    }
    used
}

/// Why a material renders as the magenta diagnostic: the project download has
/// nothing to reproduce. `convert.py`'s `report_diagnostic_materials` prints the
/// same reasons at conversion time, with the authored slot names.
fn diagnostic_material_reason(
    object: &str,
    materials: &HashMap<String, MaterialRecord>,
) -> Option<&'static str> {
    if object == UE_WORLD_GRID_MATERIAL {
        return Some("UE renders this slot with its unassigned-slot fallback");
    }
    let kind = materials.get(object)?.kind.as_deref()?;
    (kind == MISSING_SOURCE_MATERIAL)
        .then_some("the material package is absent from the project download")
}

fn selected_texture_bundle_roots(converted: &ConvertedWorld) -> Vec<String> {
    let mut roots = HashSet::new();
    let mut effective_cache = HashMap::new();
    for object in used_material_objects(converted).into_keys() {
        let effective = resolve_effective_material(
            &object,
            &converted.materials,
            &mut effective_cache,
            &mut Vec::new(),
        );
        let mut references = vec![
            select_texture(&effective, BASE_COLOR_TEXTURE_NAMES),
            select_texture(&effective, NORMAL_TEXTURE_NAMES),
            select_texture(&effective, ORM_TEXTURE_NAMES),
        ];
        let emissive_intensity = select_scalar(&effective, EMISSIVE_INTENSITY_NAMES).unwrap_or(0.0);
        if effective.emissive && emissive_intensity > 0.0 {
            references.push(select_texture(&effective, EMISSIVE_TEXTURE_NAMES));
        }
        for reference in references.into_iter().flatten() {
            if let Some(root) = converted
                .textures
                .get(reference)
                .and_then(|texture| bundle_root(&texture.output))
            {
                roots.insert(root.to_string());
            }
        }
    }
    let mut roots: Vec<_> = roots.into_iter().collect();
    roots.sort();
    roots
}

fn extend_material_uv_bounds(
    bounds: &mut HashMap<String, (Vec2, Vec2)>,
    material: &str,
    partition: &PartitionRecord,
) {
    let (Some(min), Some(max)) = (partition.uv_min, partition.uv_max) else {
        return;
    };
    let min = Vec2::from_array(min);
    let max = Vec2::from_array(max);
    bounds
        .entry(material.to_string())
        .and_modify(|bounds| {
            bounds.0 = bounds.0.min(min);
            bounds.1 = bounds.1.max(max);
        })
        .or_insert((min, max));
}

/// UDIM atlas addressing, derived only from the exported grid.
///
/// The exporter pastes block `(bx, by)` at pixel
/// `(bx * tile_width, (rows - 1 - by) * tile_height)`, so block row 0 is the
/// bottom row of the image. UE's V flip puts UDIM row `k` at mesh `v` in
/// `[-k, -k + 1]`, which makes `u/columns` and `(v + rows - 1)/rows` exact for
/// every tile regardless of which tiles a given material uses.
fn udim_uv_transform(columns: u32, rows: u32) -> Affine2 {
    let (columns, rows) = (columns.max(1), rows.max(1));
    if columns == 1 && rows == 1 {
        return Affine2::IDENTITY;
    }
    Affine2::from_scale_angle_translation(
        Vec2::new(1.0 / columns as f32, 1.0 / rows as f32),
        0.0,
        Vec2::new(0.0, (rows - 1) as f32 / rows as f32),
    )
}

/// Whether authored UVs stay inside the atlas the exporter wrote: `u` in
/// `[0, columns]` and `v` in `[-(rows - 1), 1]`. Anything outside still renders,
/// wrapped by the `Repeat` sampler onto another tile of the same atlas.
fn udim_uv_bounds_are_addressable(bounds: (Vec2, Vec2), columns: u32, rows: u32) -> bool {
    // Absorbs seam noise such as -0.0001 without hiding a whole-tile excursion.
    const TOLERANCE: f32 = 0.01;
    let (min, max) = bounds;
    min.x >= -TOLERANCE
        && max.x <= columns.max(1) as f32 + TOLERANCE
        && min.y >= -((rows.max(1) - 1) as f32) - TOLERANCE
        && max.y <= 1.0 + TOLERANCE
}

fn build_material_handles(
    converted: &ConvertedWorld,
    asset_server: &AssetServer,
    materials: &mut Assets<StandardMaterial>,
    preserve_alpha: bool,
    unlit_textures: bool,
    failed_texture_bundles: &HashSet<String>,
) -> HashMap<String, Handle<StandardMaterial>> {
    let mut result = HashMap::new();
    let mut cache = HashMap::new();
    let used = used_material_objects(converted);
    let mut masked_materials = 0usize;
    let mut translucent_materials = 0usize;
    let mut dropped_textures = 0usize;
    let mut wrapped_uv_materials = Vec::new();
    // Diagnostic only: UDIM addressing comes from the exported texture grid, so
    // these bounds just report materials authored outside their own atlas.
    let mut uv_bounds = HashMap::new();
    for mesh in converted.geometry.values() {
        for partition in &mesh.partitions {
            if let Some(Some(material)) = mesh.material_slots.get(partition.material_slot) {
                extend_material_uv_bounds(&mut uv_bounds, material, partition);
            }
        }
    }
    for actor in &converted.actors {
        if actor.hidden {
            continue;
        }
        for component in &actor.components {
            if !component.visible || component.hidden_in_game {
                continue;
            }
            let Some(mesh_name) = component.mesh.as_ref() else {
                continue;
            };
            let Some(mesh) = converted.geometry.get(mesh_name) else {
                continue;
            };
            for partition in &mesh.partitions {
                if let Some(material) = partition_material(
                    partition,
                    &mesh.material_slots,
                    &component.override_materials,
                ) {
                    extend_material_uv_bounds(&mut uv_bounds, material, partition);
                }
            }
        }
    }
    for (object, requesters) in used {
        if let Some(reason) = diagnostic_material_reason(&object, &converted.materials) {
            let requesters: Vec<_> = requesters.iter().map(String::as_str).collect();
            warn!(
                material = %object,
                requested_by = %requesters.join(", "),
                "{reason}; rendering the magenta diagnostic material"
            );
        }
        if object == UE_BLACK_UNLIT_MATERIAL {
            result.insert(
                object,
                materials.add(StandardMaterial {
                    base_color: Color::BLACK,
                    unlit: true,
                    ..default()
                }),
            );
            continue;
        }
        if object == UE_WORLD_GRID_MATERIAL {
            // Never substitute a scene material whose name merely resembles the
            // authored slot name: the slot's assignment really is the engine
            // fallback, and engine content is not part of the download.
            result.insert(
                object,
                materials.add(StandardMaterial {
                    base_color: Color::srgb(1.0, 0.0, 1.0),
                    perceptual_roughness: 1.0,
                    unlit: unlit_textures,
                    ..default()
                }),
            );
            continue;
        }
        let effective =
            resolve_effective_material(&object, &converted.materials, &mut cache, &mut Vec::new());
        let mut image = |reference: Option<&str>| {
            let texture = reference.and_then(|reference| converted.textures.get(reference))?;
            if bundle_root(&texture.output)
                .is_some_and(|root| failed_texture_bundles.contains(root))
            {
                dropped_textures += 1;
                return None;
            }
            Some(asset_server.load::<Image>(texture.output.clone()))
        };
        let base_color_reference = select_texture(&effective, BASE_COLOR_TEXTURE_NAMES);
        let normal_reference = select_texture(&effective, NORMAL_TEXTURE_NAMES);
        let orm_parameter = select_texture_parameter(&effective, ORM_TEXTURE_NAMES);
        let orm_reference = orm_parameter.and_then(|parameter| parameter.value.as_deref());
        let emissive_reference = select_texture(&effective, EMISSIVE_TEXTURE_NAMES);
        let base_color_texture = image(base_color_reference);
        let normal_map_texture = image(normal_reference);
        let orm = image(orm_reference);
        let emissive_intensity = select_scalar(&effective, EMISSIVE_INTENSITY_NAMES)
            .unwrap_or(0.0)
            .max(0.0);
        let emissive = if effective.emissive && emissive_intensity > 0.0 {
            LinearRgba::from(
                select_color(&effective, &["emissivecolor", "emissioncolor"])
                    .unwrap_or(Color::WHITE),
            ) * emissive_intensity
        } else {
            LinearRgba::BLACK
        };
        let emissive_texture = if effective.emissive && emissive_intensity > 0.0 {
            image(emissive_reference)
        } else {
            None
        };
        let texture_grid = |reference: Option<&str>| {
            let texture = converted.textures.get(reference?)?;
            Some((
                texture.source_grid_columns.max(1),
                texture.source_grid_rows.max(1),
            ))
        };
        let base_color_grid = texture_grid(base_color_reference);
        let normal_grid = texture_grid(normal_reference);
        let orm_grid = texture_grid(orm_reference);
        // A `StandardMaterial` has one `uv_transform` for every map, so maps
        // that disagree on grid size can only be addressed by one of them.
        let (columns, rows) = base_color_grid
            .or(normal_grid)
            .or(orm_grid)
            .unwrap_or((1, 1));
        if [normal_grid, orm_grid]
            .into_iter()
            .flatten()
            .any(|grid| grid != (columns, rows))
        {
            warn!(
                material = %object,
                base_color = ?base_color_grid,
                normal = ?normal_grid,
                orm = ?orm_grid,
                "material maps disagree on their UDIM grid; addressing every map with the base color grid"
            );
        }
        if (columns > 1 || rows > 1)
            && uv_bounds
                .get(&object)
                .is_some_and(|bounds| !udim_uv_bounds_are_addressable(*bounds, columns, rows))
        {
            wrapped_uv_materials.push(object.clone());
        }
        let uv_transform = udim_uv_transform(columns, rows);
        let metallic = select_scalar(&effective, &["metallic", "metalness"]).unwrap_or(
            if texture_carries_metallic(orm_parameter) {
                1.0
            } else {
                0.0
            },
        );
        let perceptual_roughness = select_scalar(&effective, &["roughness"])
            .unwrap_or(if orm.is_some() { 1.0 } else { 0.5 });
        let render_properties = source_material_render_properties(&effective);
        match effective.blend_mode {
            SourceBlendMode::Opaque => {}
            SourceBlendMode::Masked => masked_materials += 1,
            SourceBlendMode::Translucent => translucent_materials += 1,
        }
        let material = StandardMaterial {
            base_color: if unlit_textures {
                Color::WHITE
            } else {
                material_base_color(&effective)
            },
            base_color_texture,
            normal_map_texture: (!unlit_textures).then_some(normal_map_texture).flatten(),
            // Unreal normal maps use the DirectX convention.
            flip_normal_map_y: true,
            emissive: if unlit_textures {
                LinearRgba::BLACK
            } else {
                emissive
            },
            emissive_texture: (!unlit_textures).then_some(emissive_texture).flatten(),
            metallic_roughness_texture: (!unlit_textures).then_some(orm.clone()).flatten(),
            occlusion_texture: (!unlit_textures).then_some(orm).flatten(),
            metallic: if unlit_textures { 0.0 } else { metallic },
            perceptual_roughness: if unlit_textures {
                1.0
            } else {
                perceptual_roughness
            },
            uv_transform,
            unlit: unlit_textures,
            // Meshlets currently draw only opaque materials. Retaining UE's
            // alpha mode deliberately omits unsupported foliage/translucency
            // instead of turning their cards into solid scene occluders.
            alpha_mode: if unlit_textures {
                AlphaMode::Opaque
            } else {
                runtime_alpha_mode(&effective, preserve_alpha)
            },
            double_sided: render_properties.double_sided,
            cull_mode: render_properties.cull_mode,
            ..default()
        };
        result.insert(object, materials.add(material));
    }
    if !wrapped_uv_materials.is_empty() {
        warn!(
            materials = ?wrapped_uv_materials,
            "partition UVs reach outside their UDIM atlas grid; the Repeat sampler wraps them back onto tiles of the same atlas"
        );
    }
    if dropped_textures != 0 {
        warn!(
            dropped_textures,
            failed_bundles = failed_texture_bundles.len(),
            "building materials without images from failed texture bundles; the geometry renders untextured"
        );
    }
    if masked_materials != 0 || translucent_materials != 0 {
        if preserve_alpha {
            warn!(
                masked_materials,
                translucent_materials,
                "preserving source alpha modes; meshlet raster and Solari cannot yet render these materials consistently"
            );
        } else {
            info!(
                masked_materials,
                translucent_materials,
                "temporarily forcing source alpha materials opaque for meshlet raster/Solari parity"
            );
        }
    }
    result
}

fn main() {
    let args: Args = argh::from_env();
    let diagnostics = args.diagnostics;
    let raster_only = args.raster_only;
    let camera_position = parse_vec3_arg("--camera-position", args.camera_position.as_deref());
    let camera_target = parse_vec3_arg("--camera-target", args.camera_target.as_deref());
    let base_path = asset_base_path();
    let asset_root = base_path.join(ASSET_ROOT);
    let converted_world = load_converted_world(&args, &|path| fs::read(asset_root.join(path)));

    let mut app = App::new();
    // DLSS configures Vulkan instance/device creation, so its project ID must
    // exist before `DefaultPlugins` installs the renderer.
    #[cfg(feature = "dlss")]
    app.insert_resource(DlssProjectId(bevy::asset::uuid::uuid!(
        "69d895fa-5bc2-4fb3-b52d-7b77343be702"
    )));
    app.insert_resource(GlobalAmbientLight::NONE)
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(RuntimeOptions {
            raster_only,
            preserve_alpha: args.preserve_alpha,
            unlit_textures: args.unlit_textures,
            camera_position,
            camera_target,
            exposure_ev100: args.exposure_ev100,
        })
        .insert_resource(converted_world)
        .init_resource::<FailedTextureBundles>()
        .add_plugins((
            // Unprocessed on purpose: the converter already emits
            // runtime-ready bundles, so the asset processor only re-reads and
            // hashes every multi-GiB shard on each launch before concluding
            // nothing changed.
            DefaultPlugins.set(RenderPlugin {
                // GT7's native working space.
                working_color_space: WorkingColorSpace::Rec2020,
                ..default()
            }),
            // Auto-select the best HDR output the surface can present, else SDR.
            hdr::HdrPlugin::default(),
            MeshletPlugin {
                // Zorah's instanced scenes exceed eight million leaf meshlets even
                // before accounting for hierarchy/candidate pressure. An undersized
                // two-ended cull queue silently drops or corrupts visibility work and
                // presents as blinking geometry, so use Bevy's supported maximum on
                // the target RTX 5090 (roughly 1 GiB of queue storage).
                cluster_buffer_slots: 1 << 25,
            },
            FreeCameraPlugin,
            ZorahBundlePlugin,
        ));
    // Solari builds a BLAS for every compatible extracted `Mesh` asset, not for
    // entities carrying `RaytracingMesh3d`. Omitting its plugins is the only way
    // to keep `--raster-only` from paying for the whole level's BLAS set.
    if !raster_only {
        app.add_plugins(SolariPlugins);
    }
    if diagnostics {
        app.add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            LogDiagnosticsPlugin::default(),
        ));
    }
    app.init_state::<ZorahState>();
    app.add_systems(Startup, setup_hdr_calibration);
    app.add_systems(
        OnEnter(ZorahState::LoadingTextureBundles),
        begin_texture_bundle_preload,
    )
    .add_systems(OnEnter(ZorahState::ReleasingUnusedTextures), setup)
    .add_systems(
        Update,
        preload_texture_bundles.run_if(in_state(ZorahState::LoadingTextureBundles)),
    )
    .add_systems(
        Update,
        release_unused_texture_bundles.run_if(in_state(ZorahState::ReleasingUnusedTextures)),
    )
    .add_systems(
        Update,
        spawn_partitions_when_ready.run_if(in_state(ZorahState::LoadingScene)),
    )
    .add_systems(
        Update,
        warm_up_raytracing.run_if(in_state(ZorahState::WarmingRaytracing)),
    )
    .run();
}

/// Trusts the calibrated monitor for HDR luminance: hands peak and black level
/// to the OS so GT7 tone maps against the panel's real headroom, and seeds a
/// 200-nit HDR reference paper white. Gamut stays paired with the
/// `HdrPlugin`-chosen transfer.
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

fn begin_texture_bundle_preload(
    mut commands: Commands,
    converted: Res<ConvertedWorld>,
    mut next_state: ResMut<NextState<ZorahState>>,
) {
    let roots = selected_texture_bundle_roots(&converted);
    if roots.is_empty() {
        next_state.set(ZorahState::ReleasingUnusedTextures);
        return;
    }
    info!(
        bundles = roots.len(),
        "preloading Zorah texture bundles sequentially"
    );
    commands.insert_resource(TextureBundlePreload {
        roots,
        cursor: 0,
        active: None,
        loaded: Vec::new(),
    });
}

fn preload_texture_bundles(
    mut preload: ResMut<TextureBundlePreload>,
    mut failed: ResMut<FailedTextureBundles>,
    asset_server: Res<AssetServer>,
    mut next_state: ResMut<NextState<ZorahState>>,
) {
    if let Some((root, handle)) = preload.active.as_ref() {
        match asset_server.load_state(handle) {
            LoadState::Loaded => {
                info!(bundle = %root, "loaded Zorah texture bundle");
                let handle = handle.clone();
                preload.loaded.push(handle);
                preload.active = None;
            }
            LoadState::Failed(_) => {
                error!(bundle = %root, "failed to load Zorah texture bundle");
                failed.0.insert(root.clone());
                preload.active = None;
            }
            _ => return,
        }
    }
    if preload.cursor < preload.roots.len() {
        let root = preload.roots[preload.cursor].clone();
        preload.cursor += 1;
        preload.active = Some((root.clone(), asset_server.load(root)));
        return;
    }
    if failed.0.is_empty() {
        info!(
            loaded = preload.loaded.len(),
            "all Zorah texture bundles loaded; creating scene materials"
        );
    } else {
        error!(
            loaded = preload.loaded.len(),
            failed = failed.0.len(),
            "some Zorah texture bundles failed; creating their materials untextured"
        );
    }
    next_state.set(ZorahState::ReleasingUnusedTextures);
}

/// Frames spent in [`ZorahState::ReleasingUnusedTextures`] after the bundle
/// roots are dropped: one for `Assets<ZorahBundle>` to process the handle
/// drops, one for the released image handles to reach `Assets<Image>`, one
/// for the render world to free the GPU textures.
const TEXTURE_RELEASE_SETTLE_FRAMES: u32 = 3;

fn release_unused_texture_bundles(
    mut commands: Commands,
    preload: Option<Res<TextureBundlePreload>>,
    mut frames_waited: Local<u32>,
    mut next_state: ResMut<NextState<ZorahState>>,
) {
    // `setup` ran on state entry, so the selected level's materials now hold
    // direct handles to every image they need. Dropping the bundle roots
    // releases the rest of the preloaded texture set.
    if let Some(preload) = preload {
        info!(
            bundles = preload.loaded.len(),
            "releasing texture bundle roots; unreferenced images will be freed"
        );
        commands.remove_resource::<TextureBundlePreload>();
        return;
    }
    *frames_waited += 1;
    if *frames_waited >= TEXTURE_RELEASE_SETTLE_FRAMES {
        next_state.set(ZorahState::LoadingScene);
    }
}

fn make_light_proxy_meshes(meshes: &mut Assets<Mesh>) -> (Handle<Mesh>, Handle<Mesh>) {
    let mut sphere = Sphere::new(1.0).mesh().uv(16, 8);
    sphere
        .generate_tangents()
        .expect("the generated point-light sphere must have valid tangent frames");
    let mut disk = Circle::new(1.0).mesh().resolution(24).build();
    disk.generate_tangents()
        .expect("the generated spotlight disk must have valid tangent frames");
    (meshes.add(sphere), meshes.add(disk))
}

fn spawn_exported_lights(
    commands: &mut Commands,
    converted: &ConvertedWorld,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    ambient_light: &mut GlobalAmbientLight,
) -> Vec<PendingRaytracingInstance> {
    let has_atmosphere = has_sky_atmosphere(converted);
    let (point_proxy_mesh, spot_proxy_mesh) = make_light_proxy_meshes(meshes);
    let mut raytracing_instances = Vec::new();
    let mut point_count = 0usize;
    let mut spot_count = 0usize;
    let mut directional_count = 0usize;
    let mut sky_count = 0usize;
    let mut environment_count = 0usize;
    let mut unsupported_profiles = 0usize;
    let mut legacy_unit_count = 0usize;
    let mut sky_brightness = 0.0f32;
    for actor in &converted.actors {
        if actor.hidden {
            continue;
        }
        let actor_matrix = ue_matrix(&actor.transform);
        let has_concrete_lights = actor
            .lights
            .iter()
            .any(|light| !light.name.ends_with("_GEN_VARIABLE"));
        for light in &actor.lights {
            // These records are blueprint-template fallbacks. Prefer actual
            // component exports when the actor contains them, otherwise both
            // versions occupy the same transform and emit twice.
            if has_concrete_lights && light.name.ends_with("_GEN_VARIABLE") {
                continue;
            }
            if !light.visible
                || light.hidden_in_game
                || !light.affects_world
                || light.intensity <= 0.0
            {
                continue;
            }
            let has_unsupported_profile =
                light.ies_texture.is_some() || light.light_function_material.is_some();
            if has_unsupported_profile {
                unsupported_profiles += 1;
            }
            if normalized_light_units(&light.intensity_units).eq_ignore_ascii_case("Unitless") {
                legacy_unit_count += 1;
            }
            let world_matrix = actor_matrix * ue_matrix(&light.transform);
            let world_transform = ue_world_to_bevy(world_matrix);
            let color = light_color(light);
            let color_value = Color::LinearRgba(color);
            let outer_angle = light
                .outer_cone_angle
                .to_radians()
                .clamp(0.001, std::f32::consts::FRAC_PI_2 - 0.001);

            match light.kind.as_str() {
                "directional" => {
                    directional_count += 1;
                    commands.spawn((
                        Name::new(light.name.clone()),
                        DirectionalLight {
                            color: color_value,
                            illuminance: light.intensity.max(0.0),
                            // Solari provides the production shadows. Avoid an
                            // expensive duplicate cascade pass during startup.
                            shadow_maps_enabled: false,
                            ..default()
                        },
                        SunDisk {
                            angular_size: light.light_source_angle.to_radians().max(0.0001),
                            intensity: 1.0,
                        },
                        world_transform,
                    ));
                }
                "point" | "spot" => {
                    let bevy_lumens = bevy_light_lumens(light, outer_angle);
                    let range = (light.attenuation_radius * 0.01).max(0.01);
                    let source_radius =
                        (light.source_radius.max(light.soft_source_radius) * 0.01).max(0.05);
                    if light.kind == "point" {
                        point_count += 1;
                        commands.spawn((
                            Name::new(light.name.clone()),
                            PointLight {
                                color: color_value,
                                intensity: bevy_lumens,
                                range,
                                radius: source_radius,
                                shadow_maps_enabled: false,
                                ..default()
                            },
                            world_transform,
                        ));
                    } else {
                        spot_count += 1;
                        commands.spawn((
                            Name::new(light.name.clone()),
                            SpotLight {
                                color: color_value,
                                intensity: bevy_lumens,
                                range,
                                radius: source_radius,
                                inner_angle: light.inner_cone_angle.to_radians().min(outer_angle),
                                outer_angle,
                                shadow_maps_enabled: false,
                                ..default()
                            },
                            world_transform,
                        ));
                    }

                    // Solari currently binds directional and emissive-mesh
                    // sources. Model UE point lights as tiny emissive spheres
                    // and spotlights as forward-facing Lambertian apertures.
                    // Both proxy meshes are shared, so this adds two BLAS total.
                    // IES and light-function profiles are not evaluated yet,
                    // but omitting those emitters makes the traced scene lose
                    // authored key lights entirely. Keep their total flux as a
                    // uniform proxy until Solari can bind the profiles.
                    let flux = emitted_light_flux(light, outer_angle);
                    let area_denominator = if light.kind == "point" {
                        4.0 * std::f32::consts::PI.powi(2) * source_radius.powi(2)
                    } else {
                        std::f32::consts::PI.powi(2) * source_radius.powi(2)
                    };
                    let emissive = color * (flux / area_denominator.max(0.0001));
                    let material = materials.add(StandardMaterial {
                        base_color: Color::BLACK,
                        emissive,
                        perceptual_roughness: 1.0,
                        ..default()
                    });
                    let mesh = if light.kind == "point" {
                        point_proxy_mesh.clone()
                    } else {
                        spot_proxy_mesh.clone()
                    };
                    let local_rotation = if light.kind == "spot" {
                        Mat4::from_rotation_y(std::f32::consts::PI)
                    } else {
                        Mat4::IDENTITY
                    };
                    let proxy_transform = Transform::from_matrix(
                        world_transform.to_matrix()
                            * local_rotation
                            * Mat4::from_scale(Vec3::splat(source_radius)),
                    );
                    // This geometry exists only to give Solari an emissive BLAS. The
                    // corresponding Bevy PointLight/SpotLight already supplies raster
                    // lighting, so attaching `Mesh3d` here would expose our synthetic
                    // sphere/disk in the raster image.
                    let entity = commands
                        .spawn((
                            Name::new(format!("{} Solari emitter", light.name)),
                            MeshMaterial3d(material),
                            proxy_transform,
                        ))
                        .id();
                    raytracing_instances.push(PendingRaytracingInstance {
                        entity,
                        mesh,
                        geometry_error: 0.0,
                    });
                }
                "sky" => {
                    sky_count += 1;
                    let illuminance = ue_sky_light_illuminance(light.intensity);
                    sky_brightness += illuminance;
                    commands.spawn((
                        Name::new(format!("{} Solari environment", light.name)),
                        SolariEnvironmentLight { color, illuminance },
                    ));
                    environment_count += 1;
                    if light.real_time_capture {
                        debug!(light = %light.name, "approximating UE real-time SkyLight as raster ambient light");
                    }
                }
                kind => warn!(light = %light.name, %kind, "unsupported Zorah light component"),
            }
        }
    }

    // This is a responsive raster preview while BLASes warm up. Solari does
    // not currently consume Bevy environment maps, so the exported
    // directional and emissive proxy sources remain its traced lighting.
    // AtmosphereEnvironmentMapLight supplies the raster ambient/specular
    // environment when a UE SkyAtmosphere is present. Retain the old scalar
    // ambient fallback only for levels without an atmosphere.
    ambient_light.brightness = if has_atmosphere { 0.0 } else { sky_brightness };
    info!(
        point_count,
        spot_count,
        directional_count,
        sky_count,
        environment_count,
        unsupported_profiles,
        legacy_unit_count,
        solari_emitters = raytracing_instances.len(),
        "spawned exported Zorah lighting"
    );
    raytracing_instances
}

fn has_sky_atmosphere(converted: &ConvertedWorld) -> bool {
    converted.actors.iter().any(|actor| {
        !actor.hidden
            && actor.kind == "SkyAtmosphere"
            && actor
                .atmosphere
                .as_ref()
                .is_none_or(|atmosphere| atmosphere.visible && !atmosphere.hidden_in_game)
    })
}

fn active_sky_atmosphere_actor(converted: &ConvertedWorld) -> Option<&ActorRecord> {
    converted
        .actors
        .iter()
        .filter(|actor| !actor.hidden && actor.kind == "SkyAtmosphere")
        .find(|actor| {
            actor
                .atmosphere
                .as_ref()
                .is_none_or(|atmosphere| atmosphere.visible && !atmosphere.hidden_in_game)
        })
}

fn active_height_fog(converted: &ConvertedWorld) -> Option<&HeightFogRecord> {
    converted
        .actors
        .iter()
        .filter(|actor| !actor.hidden && actor.kind == "ExponentialHeightFog")
        .filter_map(|actor| actor.height_fog.as_ref())
        .find(|fog| fog.visible && !fog.hidden_in_game)
}

fn ue_linear_rgb(color: UeLinearColor) -> Vec3 {
    Vec3::new(color.r, color.g, color.b)
}

fn ue_color_linear_rgb(color: &UeColor) -> Vec3 {
    let linear = Color::srgba_u8(color.r, color.g, color.b, color.a).to_linear();
    Vec3::new(linear.red, linear.green, linear.blue)
}

fn atmosphere_planet_center(actor: Option<&ActorRecord>, inner_radius: f32) -> Vec3 {
    let default_center = -Vec3::Y * inner_radius;
    let Some(actor) = actor else {
        return default_center;
    };
    let Some(transform_mode) = actor
        .atmosphere
        .as_ref()
        .and_then(|atmosphere| atmosphere.transform_mode.as_deref())
        .and_then(|mode| mode.rsplit("::").next())
    else {
        // UE's default is PlanetTopAtAbsoluteWorldOrigin. The component's
        // transform is deliberately ignored in this mode.
        return default_center;
    };
    let component_position = ue_world_to_bevy(ue_matrix(&actor.transform)).translation;
    match transform_mode {
        "PlanetCenterAtComponentTransform" => component_position,
        "PlanetTopAtComponentTransform" => component_position - Vec3::Y * inner_radius,
        "PlanetTopAtAbsoluteWorldOrigin" => default_center,
        other => {
            warn!(%other, "unknown UE SkyAtmosphere transform mode; using absolute world origin");
            default_center
        }
    }
}

fn configured_atmosphere(
    record: Option<&SkyAtmosphereRecord>,
    height_fog: Option<&HeightFogRecord>,
) -> (ScatteringMedium, f32, f32, Vec3, f32) {
    const UE_DEFAULT_BOTTOM_RADIUS_KM: f32 = 6_360.0;
    const UE_DEFAULT_ATMOSPHERE_HEIGHT_KM: f32 = 60.0;

    let bottom_radius_km = record
        .and_then(|record| record.bottom_radius_km)
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(UE_DEFAULT_BOTTOM_RADIUS_KM);
    let atmosphere_height_km = record
        .and_then(|record| record.atmosphere_height_km)
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(UE_DEFAULT_ATMOSPHERE_HEIGHT_KM);
    let mut medium = ScatteringMedium::earth(256, 256).with_label("zorah_ue_atmosphere");

    if let Some(record) = record {
        let rayleigh_scale = record
            .rayleigh_scattering_scale
            .filter(|value| value.is_finite())
            .unwrap_or(1.0)
            .max(0.0);
        if let Some(scattering) = record.rayleigh_scattering_per_km {
            medium.terms[0].scattering = ue_linear_rgb(scattering).max(Vec3::ZERO) * 0.001;
        }
        medium.terms[0].scattering *= rayleigh_scale;
        if let Some(height_km) = record
            .rayleigh_exponential_distribution_km
            .filter(|value| value.is_finite() && *value > 0.0)
        {
            medium.terms[0].falloff = Falloff::Exponential {
                scale: height_km / atmosphere_height_km,
            };
        }

        let mie_scattering_scale = record
            .mie_scattering_scale
            .filter(|value| value.is_finite())
            .unwrap_or(1.0)
            .max(0.0);
        if let Some(scattering) = record.mie_scattering_per_km {
            medium.terms[1].scattering = ue_linear_rgb(scattering).max(Vec3::ZERO) * 0.001;
        }
        medium.terms[1].scattering *= mie_scattering_scale;

        let mie_absorption_scale = record
            .mie_absorption_scale
            .filter(|value| value.is_finite())
            .unwrap_or(1.0)
            .max(0.0);
        if let Some(absorption) = record.mie_absorption_per_km {
            medium.terms[1].absorption = ue_linear_rgb(absorption).max(Vec3::ZERO) * 0.001;
        }
        medium.terms[1].absorption *= mie_absorption_scale;
        if let Some(asymmetry) = record.mie_anisotropy.filter(|value| value.is_finite()) {
            medium.terms[1].phase = PhaseFunction::Mie {
                asymmetry: asymmetry.clamp(-0.999, 0.999),
            };
        }
        if let Some(height_km) = record
            .mie_exponential_distribution_km
            .filter(|value| value.is_finite() && *value > 0.0)
        {
            medium.terms[1].falloff = Falloff::Exponential {
                scale: height_km / atmosphere_height_km,
            };
        }

        let other_absorption_scale = record
            .other_absorption_scale
            .filter(|value| value.is_finite())
            .unwrap_or(1.0)
            .max(0.0);
        if let Some(absorption) = record.other_absorption_per_km {
            medium.terms[2].absorption = ue_linear_rgb(absorption).max(Vec3::ZERO) * 0.001;
        }
        medium.terms[2].absorption *= other_absorption_scale;
    }

    if let Some(fog) = height_fog {
        // UE's exponential-height-fog density is expressed per kilometer.
        // Convert it to Bevy's per-meter optical coefficients and split total
        // extinction according to the authored single-scattering albedo.
        let density_per_m = fog
            .fog_density
            .filter(|value| value.is_finite())
            .unwrap_or(0.02)
            .max(0.0)
            * 0.001
            * fog
                .volumetric_fog_extinction_scale
                .filter(|value| value.is_finite())
                .unwrap_or(1.0)
                .max(0.0);
        let mut albedo = fog
            .volumetric_fog_albedo
            .as_ref()
            .map(ue_color_linear_rgb)
            .unwrap_or(Vec3::ONE);
        if let Some(color) = fog.fog_inscattering_color {
            albedo *= ue_linear_rgb(color).max(Vec3::ZERO);
        }
        albedo = albedo.clamp(Vec3::ZERO, Vec3::ONE);
        let height_falloff_per_km = fog
            .fog_height_falloff
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(0.2);
        let height_scale_km = 1.0 / height_falloff_per_km;
        let asymmetry = fog
            .volumetric_fog_scattering_distribution
            .filter(|value| value.is_finite())
            .unwrap_or(0.2)
            .clamp(-0.9, 0.9);
        medium.terms.push(ScatteringTerm {
            absorption: Vec3::splat(density_per_m) * (Vec3::ONE - albedo),
            scattering: Vec3::splat(density_per_m) * albedo,
            falloff: Falloff::Exponential {
                scale: height_scale_km / atmosphere_height_km,
            },
            phase: PhaseFunction::Mie { asymmetry },
        });
        if fog.enable_volumetric_fog == Some(false) {
            debug!("mapping non-volumetric UE height fog into the shared Bevy atmosphere medium");
        }
    }

    let ground_albedo = record
        .and_then(|record| record.ground_albedo.as_ref())
        .map(ue_color_linear_rgb)
        .unwrap_or(Vec3::splat(0.3));
    let environment_intensity = record
        .and_then(|record| {
            record
                .sky_luminance_factor
                .or(record.sky_and_aerial_perspective_luminance_factor)
        })
        .map(ue_linear_rgb)
        .map(|factor| factor.max(Vec3::ZERO).element_sum() / 3.0)
        .filter(|value| value.is_finite())
        .unwrap_or(1.0);

    (
        medium,
        bottom_radius_km * 1_000.0,
        (bottom_radius_km + atmosphere_height_km) * 1_000.0,
        ground_albedo,
        environment_intensity,
    )
}

fn active_unbound_post_process(actors: &[ActorRecord]) -> Option<&PostProcessRecord> {
    actors
        .iter()
        .filter(|actor| !actor.hidden)
        .filter_map(|actor| actor.post_process.as_ref())
        .filter(|post_process| {
            post_process.enabled && post_process.unbound && post_process.blend_weight > 0.0
        })
        .max_by(|left, right| left.priority.total_cmp(&right.priority))
}

fn legacy_zorah_post_process(level: &str) -> Option<PostProcessRecord> {
    // Older packed Zorah manifests predate post-process export. These values
    // are the authored unbound volumes in the immutable 1.1.0 source sample;
    // newly converted manifests carry the same data directly on their actors.
    // ThroneRoom is also the only level whose volume ticks the bloom overrides.
    let (min_ev100, max_ev100, bias, bloom_method, bloom_intensity) = match level {
        "GreenHouse_Level" => (Some(4.0), Some(5.0), Some(0.99), None, None),
        "ThroneRoom_Level" => (Some(8.0), Some(8.0), None, Some("BM_FFT"), Some(0.003)),
        "Restir_Level" => (None, None, Some(-2.5), None, None),
        _ => return None,
    };
    Some(PostProcessRecord {
        enabled: true,
        unbound: true,
        priority: 0.0,
        blend_weight: 1.0,
        bloom_method: bloom_method.map(String::from),
        bloom_intensity,
        _film_slope: None,
        _film_toe: None,
        _film_shoulder: None,
        _film_black_clip: None,
        _film_white_clip: None,
        _auto_exposure_method: Some("AEM_Histogram".into()),
        auto_exposure_min_ev100: min_ev100,
        auto_exposure_max_ev100: max_ev100,
        auto_exposure_bias: bias,
    })
}

fn resolved_exposure_ev100(
    exposure_override: Option<f32>,
    post_process: Option<&PostProcessRecord>,
) -> f32 {
    if let Some(exposure) = exposure_override.filter(|exposure| exposure.is_finite()) {
        return exposure;
    }

    let Some(post_process) = post_process else {
        return Exposure::BLENDER.ev100;
    };
    let min_ev100 = post_process
        .auto_exposure_min_ev100
        .filter(|value| value.is_finite());
    let max_ev100 = post_process
        .auto_exposure_max_ev100
        .filter(|value| value.is_finite());
    let metered_ev100 = match (min_ev100, max_ev100) {
        (Some(minimum), Some(maximum)) => (minimum + maximum) * 0.5,
        (Some(minimum), None) => minimum,
        (None, Some(maximum)) => maximum,
        (None, None) => Exposure::BLENDER.ev100,
    };
    // UE exposure compensation is additive brightness in stops. Bevy's
    // physical exposure runs in the opposite direction: lowering EV100 makes
    // the result brighter.
    let compensation = post_process
        .auto_exposure_bias
        .filter(|value| value.is_finite())
        .unwrap_or(0.0);
    (metered_ev100 - compensation).clamp(-16.0, 32.0)
}

fn resolved_bloom(post_process: Option<&PostProcessRecord>) -> Bloom {
    // Where the volume authors nothing, UE falls back to its engine-wide bloom
    // defaults, which live in C++ and not in the shipped project -- so the
    // baseline stays this example's own preset rather than a guessed number.
    let mut bloom = Bloom::GT7_GLARE;
    let Some(post_process) = post_process else {
        return bloom;
    };
    // UE's BloomIntensity and `Bloom::intensity` are both a linear "how much of
    // the blurred image survives" dial anchored at 0 = no bloom, but they scale
    // different convolutions, so this is a direct transfer of the authored
    // number, not a calibrated match.
    if let Some(intensity) = post_process
        .bloom_intensity
        .filter(|intensity| intensity.is_finite() && *intensity >= 0.0)
    {
        bloom.intensity = intensity;
    }
    // BM_FFT is UE's aperture-convolution bloom, which `Gt7Glare` reproduces by
    // weighting the mip chain with a diffraction pattern; BM_SOG is its
    // sum-of-Gaussians approximation, closest to the parametric curve.
    match post_process.bloom_method.as_deref() {
        Some("BM_FFT") => {
            bloom.scatter = BloomScatterModel::Gt7Glare {
                f_number: BloomScatterModel::DEFAULT_F_NUMBER,
            }
        }
        Some("BM_SOG") => bloom.scatter = BloomScatterModel::Aesthetic,
        _ => {}
    }
    bloom
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    converted: Res<ConvertedWorld>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering_media: ResMut<Assets<ScatteringMedium>>,
    mut ambient_light: ResMut<GlobalAmbientLight>,
    options: Res<RuntimeOptions>,
    failed_texture_bundles: Res<FailedTextureBundles>,
) {
    let has_atmosphere = has_sky_atmosphere(&converted);
    let mut atmosphere_environment_intensity = 1.0;
    if has_atmosphere {
        // Zorah authors a UE SkyAtmosphere in every shipped level. Apply its
        // serialized coefficient overrides on top of the common Earth preset;
        // old manifests without component data retain the UE defaults.
        let atmosphere_actor = active_sky_atmosphere_actor(&converted);
        let atmosphere_record = atmosphere_actor.and_then(|actor| actor.atmosphere.as_ref());
        let (medium, inner_radius, outer_radius, ground_albedo, environment_intensity) =
            configured_atmosphere(atmosphere_record, active_height_fog(&converted));
        atmosphere_environment_intensity = environment_intensity;
        commands.spawn((
            Atmosphere {
                inner_radius,
                outer_radius,
                ground_albedo,
                medium: scattering_media.add(medium),
            },
            GlobalTransform::from_translation(atmosphere_planet_center(
                atmosphere_actor,
                inner_radius,
            )),
        ));
    }
    let default_material = materials.add(StandardMaterial {
        base_color: if options.unlit_textures {
            Color::srgb(1.0, 0.0, 1.0)
        } else {
            Color::srgb(0.55, 0.52, 0.48)
        },
        perceptual_roughness: 0.65,
        unlit: options.unlit_textures,
        ..default()
    });
    let material_handles = build_material_handles(
        &converted,
        &asset_server,
        &mut materials,
        options.preserve_alpha,
        options.unlit_textures,
        &failed_texture_bundles.0,
    );
    let mut pending = Vec::new();
    let mut skipped_components = 0usize;
    let mut world_min = Vec3::splat(f32::INFINITY);
    let mut world_max = Vec3::splat(f32::NEG_INFINITY);

    for actor in &converted.actors {
        if actor.hidden {
            continue;
        }
        let actor_matrix = ue_matrix(&actor.transform);
        for component in &actor.components {
            if !component.visible || component.hidden_in_game {
                continue;
            }
            let Some(mesh_name) = component.mesh.as_ref() else {
                continue;
            };
            let Some(converted_mesh) = converted.geometry.get(mesh_name) else {
                skipped_components += 1;
                continue;
            };
            let component_matrix = actor_matrix * ue_matrix(&component.transform);
            match component.instances.as_deref() {
                Some([]) => {}
                Some(instances) => {
                    for instance in instances {
                        queue_partitions(
                            &mut pending,
                            &converted_mesh.partitions,
                            &converted_mesh.material_slots,
                            &component.override_materials,
                            component_matrix * ue_matrix(instance),
                            &material_handles,
                            &default_material,
                            &mut world_min,
                            &mut world_max,
                            is_lightblocker_mesh(mesh_name),
                        );
                    }
                }
                None => queue_partitions(
                    &mut pending,
                    &converted_mesh.partitions,
                    &converted_mesh.material_slots,
                    &component.override_materials,
                    component_matrix,
                    &material_handles,
                    &default_material,
                    &mut world_min,
                    &mut world_max,
                    is_lightblocker_mesh(mesh_name),
                ),
            }
        }
    }
    let raytracing_light_instances = spawn_exported_lights(
        &mut commands,
        &converted,
        &mut meshes,
        &mut materials,
        &mut ambient_light,
    );

    let center = if world_min.is_finite() && world_max.is_finite() {
        (world_min + world_max) * 0.5
    } else {
        Vec3::ZERO
    };
    let extent = if world_min.is_finite() && world_max.is_finite() {
        (world_max - world_min).length().max(10.0)
    } else {
        10.0
    };
    let (preset_position, preset_target) = level_camera_placement(&converted.level, center, extent);
    let camera_position = options.camera_position.unwrap_or(preset_position);
    let camera_target = options.camera_target.unwrap_or(preset_target);
    let exposure_ev100 =
        resolved_exposure_ev100(options.exposure_ev100, converted.post_process.as_ref());
    let bloom = resolved_bloom(converted.post_process.as_ref());
    let queued_partitions = pending.len();
    let mut seen_bundle_roots = HashSet::new();
    let bundle_roots = pending
        .iter()
        .filter_map(|partition| bundle_root(&partition.mesh_path))
        .filter(|root| seen_bundle_roots.insert((*root).to_string()))
        .map(str::to_string)
        .collect();
    let mut unique_blas_vertices = HashMap::<&str, usize>::new();
    for partition in &pending {
        unique_blas_vertices
            .entry(&partition.geometry)
            .or_insert(partition.blas_vertices);
    }
    let unique_blas_vertices = unique_blas_vertices
        .into_values()
        .map(|vertices| vertices as u64)
        .sum();
    commands.insert_resource(PendingScene {
        partitions: pending,
        loaded_assets: HashMap::new(),
        bundle_roots,
        bundle_cursor: 0,
        active_bundle: None,
        loaded_bundle_roots: HashSet::new(),
        loaded_bundles: Vec::new(),
        prepared_meshes: HashSet::new(),
        failed_meshes: HashSet::new(),
        raytracing_instances: raytracing_light_instances,
        raytracing_cursor: 0,
        expected_blas: 0,
        warmup_frames_remaining: 0,
        warmup_timeout_reported: false,
        warmup_progress_log_frames_remaining: 0,
        warmup_started_at: None,
        unique_blas_vertices,
        spawned: 0,
        failed: 0,
        reported_done: false,
    });
    let mut camera = commands.spawn((
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
        Exposure {
            ev100: exposure_ev100,
        },
        // Zorah renders through UE's stock ACES filmic curve: Restir ticks all
        // five Film* overrides (Slope/Toe/Shoulder/BlackClip/WhiteClip) yet
        // serializes no value for any of them, so each sits at the engine
        // default, and GreenHouse/ThroneRoom leave the curve untouched.
        // `Tonemapping::AcesFitted` is the closest operator by curve shape, but
        // it is SDR-only -- its output is capped at paper white, which would
        // defeat the HDR display target this example configures. GT7 is kept as
        // the one peak-luminance-aware filmic operator; the authored Film*
        // parameters ride along in the manifest so the choice can be revisited
        // if Bevy grows a parameterizable ACES curve.
        Tonemapping::GranTurismo7,
        GranTurismo7Params::default(),
        bloom,
        ZorahCamera,
        FreeCamera {
            walk_speed: (extent * 0.02).max(3.0),
            run_speed: (extent * 0.15).max(20.0),
            ..default()
        },
        Transform::from_translation(camera_position).looking_at(camera_target, Vec3::Y),
    ));
    if has_atmosphere {
        camera.insert((
            AtmosphereSettings {
                // Keep useful aerial-perspective precision across these
                // building-scale levels instead of spending the LUT's depth
                // range on the default 32 km.
                aerial_view_lut_max_distance: extent.max(1_000.0).min(16_000.0),
                ..default()
            },
            // Generate raster diffuse/specular IBL from the same atmosphere
            // and imported directional sun/moon.
            AtmosphereEnvironmentMapLight {
                intensity: atmosphere_environment_intensity,
                ..default()
            },
        ));
    }
    info!(
        "queued Zorah level={} partitions={} skipped_components_without_converted_geometry={} camera_position={} camera_target={} exposure_ev100={}",
        converted.level,
        queued_partitions,
        skipped_components,
        camera_position,
        camera_target,
        exposure_ev100,
    );
}

fn is_lightblocker_mesh(mesh: &str) -> bool {
    mesh.to_ascii_lowercase().contains("/lightblockers/")
}

fn queue_partitions(
    pending: &mut Vec<PendingPartition>,
    partitions: &[PartitionRecord],
    material_slots: &[Option<String>],
    overrides: &[String],
    ue_world: Mat4,
    material_handles: &HashMap<String, Handle<StandardMaterial>>,
    default_material: &Handle<StandardMaterial>,
    world_min: &mut Vec3,
    world_max: &mut Vec3,
    raytracing_only: bool,
) {
    let transform = ue_world_to_bevy(ue_world);
    *world_min = world_min.min(transform.translation);
    *world_max = world_max.max(transform.translation);
    for partition in partitions {
        let mesh_path = partition
            .mesh
            .clone()
            .unwrap_or_else(|| format!("{}#Mesh0/Primitive0", partition.geometry));
        let material_object = partition_material(partition, material_slots, overrides);
        let material = match material_object {
            Some(object) => material_handles
                .get(object)
                .unwrap_or_else(|| panic!("converted material has no runtime handle: {object}")),
            None => default_material,
        };
        pending.push(PendingPartition {
            geometry: partition.geometry.clone(),
            mesh_path,
            meshlet_path: partition.meshlet.clone(),
            assets: None,
            material: material.clone(),
            transform,
            vertices: partition.vertices,
            blas_vertices: partition.blas_vertices,
            blas_achieved_error: partition.blas_achieved_error,
            raytracing_only,
            spawned: false,
        });
    }
}

fn ue_matrix(transform: &UeTransform) -> Mat4 {
    Mat4::from_scale_rotation_translation(
        Vec3::new(transform.scale.x, transform.scale.y, transform.scale.z),
        Quat::from_xyzw(
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ),
        Vec3::new(
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ),
    )
}

fn ue_world_to_bevy(ue_world: Mat4) -> Transform {
    // (UE X, Y, Z) -> (Bevy Y, Z, -X). Conjugating also converts rotation
    // and non-uniform scale; only the translated centimetres need /100.
    let basis = Mat4::from_cols(
        Vec4::new(0.0, 0.0, -1.0, 0.0),
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::W,
    );
    let mut bevy_world = basis * ue_world * basis.inverse();
    bevy_world.w_axis.x *= 0.01;
    bevy_world.w_axis.y *= 0.01;
    bevy_world.w_axis.z *= 0.01;
    Transform::from_matrix(bevy_world)
}

fn normalized_light_units(units: &str) -> &str {
    units.rsplit("::").next().unwrap_or(units)
}

fn spot_solid_angle(outer_angle_radians: f32) -> f32 {
    2.0 * std::f32::consts::PI * (1.0 - outer_angle_radians.cos())
}

fn bevy_light_lumens(light: &LightRecord, outer_angle_radians: f32) -> f32 {
    match normalized_light_units(&light.intensity_units)
        .to_ascii_lowercase()
        .as_str()
    {
        "candelas" | "candela" => light.intensity * std::f32::consts::TAU * 2.0,
        // UE divides spot lumens by the cone solid angle to reach candela while
        // Bevy divides `SpotLight::intensity` by 4pi like a point light, so the
        // authored flux has to be restated in Bevy's convention or the cone
        // renders 7-15x too dim next to its Solari emissive proxy.
        "lumens" | "lumen" if light.kind == "spot" => {
            light.intensity * 2.0 * std::f32::consts::TAU
                / spot_solid_angle(outer_angle_radians).max(f32::MIN_POSITIVE)
        }
        "lumens" | "lumen" => light.intensity,
        "unitless" => light.intensity * UE_UNITLESS_LIGHT_LUMENS,
        _ => light.intensity,
    }
    .max(0.0)
}

fn emitted_light_flux(light: &LightRecord, outer_angle_radians: f32) -> f32 {
    match normalized_light_units(&light.intensity_units)
        .to_ascii_lowercase()
        .as_str()
    {
        "candelas" | "candela" if light.kind == "spot" => {
            light.intensity * spot_solid_angle(outer_angle_radians)
        }
        "candelas" | "candela" => light.intensity * std::f32::consts::TAU * 2.0,
        "unitless" => light.intensity * UE_UNITLESS_LIGHT_LUMENS,
        _ => light.intensity,
    }
    .max(0.0)
}

fn blackbody_srgb(temperature: f32) -> Color {
    // A compact correlated-color-temperature approximation. UE applies its
    // temperature tint on top of LightColor; doing the multiplication in
    // linear space below preserves that behavior closely enough for export.
    let temperature = temperature.clamp(1000.0, 40_000.0) / 100.0;
    let red = if temperature <= 66.0 {
        255.0
    } else {
        329.698_73 * (temperature - 60.0).powf(-0.133_204_76)
    };
    let green = if temperature <= 66.0 {
        99.470_8 * temperature.ln() - 161.119_57
    } else {
        288.122_16 * (temperature - 60.0).powf(-0.075_514_846)
    };
    let blue = if temperature >= 66.0 {
        255.0
    } else if temperature <= 19.0 {
        0.0
    } else {
        138.517_73 * (temperature - 10.0).ln() - 305.044_8
    };
    Color::srgb(
        (red / 255.0).clamp(0.0, 1.0),
        (green / 255.0).clamp(0.0, 1.0),
        (blue / 255.0).clamp(0.0, 1.0),
    )
}

fn light_color(light: &LightRecord) -> LinearRgba {
    let mut color =
        Color::srgba_u8(light.color.r, light.color.g, light.color.b, light.color.a).to_linear();
    if light.use_temperature {
        let temperature = blackbody_srgb(light.temperature).to_linear();
        color.red *= temperature.red;
        color.green *= temperature.green;
        color.blue *= temperature.blue;
    }
    color
}

fn bundle_root(path: &str) -> Option<&str> {
    let (root, _) = path.split_once('#')?;
    root.ends_with(".zorah_bundle").then_some(root)
}

fn spawn_partitions_when_ready(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut pending: ResMut<PendingScene>,
    meshes: Res<Assets<Mesh>>,
    options: Res<RuntimeOptions>,
    render_device: Res<RenderDevice>,
    mut next_state: ResMut<NextState<ZorahState>>,
) {
    let PendingScene {
        partitions,
        loaded_assets,
        bundle_roots,
        bundle_cursor,
        active_bundle,
        loaded_bundle_roots,
        loaded_bundles,
        prepared_meshes,
        failed_meshes,
        raytracing_instances,
        spawned: total_spawned,
        failed: total_failed,
        reported_done,
        expected_blas,
        warmup_frames_remaining,
        warmup_timeout_reported,
        warmup_progress_log_frames_remaining,
        warmup_started_at,
        unique_blas_vertices,
        ..
    } = &mut *pending;
    if let Some((root, handle, started_at)) = active_bundle.as_ref() {
        match asset_server.load_state(handle) {
            LoadState::Loaded => {
                info!(
                    bundle = %root,
                    elapsed_seconds = started_at.elapsed().as_secs_f32(),
                    completed = loaded_bundle_roots.len() + 1,
                    total = bundle_roots.len(),
                    "loaded Zorah runtime bundle"
                );
                loaded_bundle_roots.insert(root.clone());
                loaded_bundles.push(handle.clone());
                *active_bundle = None;
            }
            LoadState::Failed(_) => {
                error!(
                    bundle = %root,
                    elapsed_seconds = started_at.elapsed().as_secs_f32(),
                    "failed to load Zorah runtime bundle"
                );
                loaded_bundle_roots.insert(root.clone());
                *active_bundle = None;
            }
            _ => {}
        }
    }
    let mut loads_started = 0usize;
    let mut geometry_vertices = 0usize;
    let mut spawned = 0usize;
    let mut failed = 0usize;
    for partition in partitions.iter_mut() {
        if partition.spawned {
            continue;
        }
        if let Some(root) = bundle_root(&partition.mesh_path) {
            if !loaded_bundle_roots.contains(root) {
                continue;
            }
        }
        if partition.assets.is_none() {
            if let Some(assets) = loaded_assets.get(&partition.mesh_path) {
                partition.assets = Some(assets.clone());
            } else if loads_started < MAX_NEW_PARTITION_LOADS_PER_FRAME {
                let assets = LoadedPartitionAssets {
                    mesh: asset_server.load(partition.mesh_path.clone()),
                    meshlet: asset_server.load(partition.meshlet_path.clone()),
                };
                loaded_assets.insert(partition.mesh_path.clone(), assets.clone());
                partition.assets = Some(assets);
                loads_started += 1;
            } else {
                continue;
            }
        }
        let Some(assets) = partition.assets.as_ref() else {
            continue;
        };
        let mesh_state = asset_server.load_state(&assets.mesh);
        let meshlet_state = asset_server.load_state(&assets.meshlet);
        if matches!(mesh_state, LoadState::Failed(_))
            || matches!(meshlet_state, LoadState::Failed(_))
        {
            partition.spawned = true;
            failed += 1;
            if failed_meshes.insert(assets.mesh.id()) {
                error!(
                    geometry = %partition.geometry,
                    meshlet = %partition.meshlet_path,
                    "failed to load converted Zorah partition"
                );
            }
            continue;
        }
        if !matches!(mesh_state, LoadState::Loaded) || !matches!(meshlet_state, LoadState::Loaded) {
            continue;
        }
        let Some(mesh) = meshes.get(&assets.mesh) else {
            continue;
        };
        let compact_meshlet_blas = mesh.contains_attribute(Mesh::ATTRIBUTE_POSITION)
            && mesh.contains_attribute(Mesh::ATTRIBUTE_NORMAL)
            && mesh.contains_attribute(Mesh::ATTRIBUTE_UV_0)
            && !mesh.contains_attribute(Mesh::ATTRIBUTE_TANGENT)
            && mesh.get_vertex_size() == 32;
        if !compact_meshlet_blas {
            partition.spawned = true;
            failed += 1;
            if failed_meshes.insert(assets.mesh.id()) {
                error!(
                    geometry = %partition.geometry,
                    vertex_bytes = mesh.get_vertex_size(),
                    "converted Zorah BLAS does not have compact POSITION/NORMAL/UV geometry; rerun the converter"
                );
            }
            continue;
        }
        let is_new_geometry = !prepared_meshes.contains(&assets.mesh.id());
        let partition_vertices = partition.vertices.max(partition.blas_vertices);
        if is_new_geometry
            && geometry_vertices != 0
            && geometry_vertices.saturating_add(partition_vertices)
                > MAX_NEW_GEOMETRY_VERTICES_PER_FRAME
        {
            continue;
        }
        if spawned >= MAX_RASTER_INSTANCES_PER_FRAME {
            continue;
        }
        let entity = if partition.raytracing_only {
            commands
                .spawn((
                    Name::new(format!("{} (raytracing-only)", partition.geometry)),
                    MeshMaterial3d(partition.material.clone()),
                    partition.transform,
                ))
                .id()
        } else {
            commands
                .spawn((
                    MeshletMesh3d(assets.meshlet.clone()),
                    MeshMaterial3d(partition.material.clone()),
                    partition.transform,
                ))
                .id()
        };
        raytracing_instances.push(PendingRaytracingInstance {
            entity,
            mesh: assets.mesh.clone(),
            geometry_error: partition.blas_achieved_error,
        });
        if is_new_geometry {
            prepared_meshes.insert(assets.mesh.id());
            geometry_vertices = geometry_vertices.saturating_add(partition_vertices);
        }
        partition.spawned = true;
        spawned += 1;
    }
    *total_spawned += spawned;
    *total_failed += failed;
    // Do not overlap another archive decode with the entity creation and GPU
    // upload burst from the bundle that just completed. In particular, a
    // multi-gigabyte meshlet bundle can otherwise monopolize memory bandwidth
    // while the render world is ingesting the previous multi-gigabyte bundle.
    let mut all_spawned = true;
    let mut loaded_geometry_drained = true;
    for partition in partitions.iter() {
        if partition.spawned {
            continue;
        }
        all_spawned = false;
        if bundle_root(&partition.mesh_path).is_some_and(|root| loaded_bundle_roots.contains(root))
        {
            loaded_geometry_drained = false;
            break;
        }
    }
    if active_bundle.is_none() && loaded_geometry_drained && *bundle_cursor < bundle_roots.len() {
        let root = bundle_roots[*bundle_cursor].clone();
        *bundle_cursor += 1;
        let handle = asset_server.load::<ZorahBundle>(root.clone());
        info!(
            bundle = %root,
            started = *bundle_cursor,
            total = bundle_roots.len(),
            "starting Zorah runtime bundle load"
        );
        *active_bundle = Some((root, handle, Instant::now()));
    }
    if !*reported_done && all_spawned {
        *reported_done = true;
        // Solari's plugins silently register nothing when the adapter lacks ray
        // queries, leaving the readiness counters at zero forever. Report the
        // raster scene as final instead of waiting on BLASes nobody will build.
        let raytracing_supported = render_device
            .features()
            .contains(SolariPlugins::required_wgpu_features());
        if options.raster_only || !raytracing_supported {
            if !options.raster_only {
                warn!(
                    missing_features = ?SolariPlugins::required_wgpu_features()
                        .difference(render_device.features()),
                    "this GPU or backend cannot run Solari; rendering meshlet raster only"
                );
            }
            info!(
                "Zorah raster-only level ready: spawned={} failed={}",
                *total_spawned, *total_failed,
            );
            next_state.set(ZorahState::Running);
            return;
        }
        *expected_blas = raytracing_instances
            .iter()
            .map(|instance| instance.mesh.id())
            .collect::<HashSet<_>>()
            .len();
        *warmup_frames_remaining = unique_blas_vertices
            .div_ceil(BLAS_BUILD_VERTICES_PER_FRAME)
            .saturating_add(BLAS_WARMUP_MARGIN_FRAMES);
        *warmup_timeout_reported = false;
        *warmup_progress_log_frames_remaining = 0;
        *warmup_started_at = Some(Instant::now());
        info!(
            spawned = *total_spawned,
            failed = *total_failed,
            expected_blas = *expected_blas,
            diagnostic_timeout_frames = *warmup_frames_remaining,
            "Zorah raster scene submitted; waiting for measured BLAS readiness",
        );
        next_state.set(ZorahState::WarmingRaytracing);
    }
}

fn raytracing_scene_ready(snapshot: &RaytracingSceneStatusSnapshot, expected_blas: usize) -> bool {
    snapshot.is_settled_for(expected_blas)
}

fn warm_up_raytracing(
    mut commands: Commands,
    mut pending: ResMut<PendingScene>,
    raytracing_status: Res<RaytracingSceneStatus>,
    camera: Single<Entity, With<ZorahCamera>>,
    mut next_state: ResMut<NextState<ZorahState>>,
    #[cfg(feature = "dlss")] dlss_rr_supported: Option<Res<DlssRayReconstructionSupported>>,
) {
    // Solari enqueues a BLAS build per extracted compatible `Mesh` asset, so
    // that work already started as the geometry bundles loaded. Attaching
    // `RaytracingMesh3d` only registers TLAS instances; batch it anyway to keep
    // the per-frame instance extraction bounded, then wait on the render-world
    // readiness counters.
    let end = pending
        .raytracing_cursor
        .saturating_add(MAX_RAYTRACING_INSTANCES_PER_FRAME)
        .min(pending.raytracing_instances.len());
    for instance in &pending.raytracing_instances[pending.raytracing_cursor..end] {
        commands.entity(instance.entity).insert((
            RaytracingMesh3d(instance.mesh.clone()),
            RaytracingMesh3dGeometryError(instance.geometry_error),
        ));
    }
    pending.raytracing_cursor = end;
    if end != pending.raytracing_instances.len() {
        return;
    }

    let snapshot = raytracing_status.snapshot();
    let settled = raytracing_scene_ready(&snapshot, pending.expected_blas);
    if pending.warmup_progress_log_frames_remaining == 0 {
        info!(
            expected_blas = pending.expected_blas,
            available_blas = snapshot.available_blas,
            queued_builds = snapshot.queued_builds,
            allocator_waiting = snapshot.allocator_waiting,
            pending_compactions = snapshot.pending_compactions,
            compacted_blas = snapshot.compacted_blas,
            failed_compactions = snapshot.failed_compactions,
            compaction_disabled = snapshot.compaction_disabled,
            elapsed_seconds = pending
                .warmup_started_at
                .map_or(0.0, |started| started.elapsed().as_secs_f32()),
            settled,
            "Zorah BLAS preparation progress",
        );
        pending.warmup_progress_log_frames_remaining = BLAS_PROGRESS_LOG_INTERVAL_FRAMES;
    } else {
        pending.warmup_progress_log_frames_remaining -= 1;
    }

    if !settled {
        if pending.warmup_frames_remaining != 0 {
            pending.warmup_frames_remaining -= 1;
        } else if !pending.warmup_timeout_reported {
            pending.warmup_timeout_reported = true;
            error!(
                expected_blas = pending.expected_blas,
                available_blas = snapshot.available_blas,
                queued_builds = snapshot.queued_builds,
                allocator_waiting = snapshot.allocator_waiting,
                pending_compactions = snapshot.pending_compactions,
                compacted_blas = snapshot.compacted_blas,
                failed_compactions = snapshot.failed_compactions,
                compaction_disabled = snapshot.compaction_disabled,
                elapsed_seconds = pending
                    .warmup_started_at
                    .map_or(0.0, |started| started.elapsed().as_secs_f32()),
                "Zorah BLAS preparation exceeded its conservative diagnostic estimate; continuing to wait rather than enabling Solari early",
            );
        }
        return;
    }

    let mut camera = commands.entity(*camera);
    camera.insert(SolariLighting {
        // Scene tuning: Zorah's throne room and courtyard have sightlines well
        // past the 50 m default, at which a world-cache GI ray is truncated and
        // contributes nothing, leaving the large interiors short of bounce
        // energy.
        world_cache_max_gi_ray_distance: 200.0,
        ..default()
    });
    // Ray Reconstruction consumes guide buffers produced by Solari, so enable
    // it at the same measured-ready transition rather than during BLAS warmup.
    #[cfg(feature = "dlss")]
    if dlss_rr_supported.is_some() {
        camera.insert(Dlss::<DlssRayReconstructionFeature> {
            perf_quality_mode: DlssPerfQualityMode::Auto,
            reset: Default::default(),
            _phantom_data: Default::default(),
        });
        info!("DLSS Ray Reconstruction enabled for Zorah");
    }
    info!(
        spawned = pending.spawned,
        failed = pending.failed,
        expected_blas = pending.expected_blas,
        available_blas = snapshot.available_blas,
        compacted_blas = snapshot.compacted_blas,
        failed_compactions = snapshot.failed_compactions,
        elapsed_seconds = pending
            .warmup_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f32()),
        "Zorah level ready: meshlet raster + measured-ready Solari BLAS instances",
    );
    next_state.set(ZorahState::Running);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_transform() -> UeTransform {
        UeTransform {
            translation: UeVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            rotation: UeQuat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            scale: UeVec3 {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            },
        }
    }

    fn test_atmosphere_actor(transform_mode: &str, translation: UeVec3) -> ActorRecord {
        let mut transform = test_transform();
        transform.translation = translation;
        ActorRecord {
            _name: "atmosphere".into(),
            _label: None,
            kind: "SkyAtmosphere".into(),
            transform,
            hidden: false,
            components: vec![],
            lights: vec![],
            atmosphere: Some(SkyAtmosphereRecord {
                _name: "atmosphere component".into(),
                transform_mode: Some(transform_mode.into()),
                ..default()
            }),
            height_fog: None,
            post_process: None,
        }
    }

    fn test_light(kind: &str, intensity: f32, units: &str) -> LightRecord {
        LightRecord {
            name: "test".into(),
            kind: kind.into(),
            transform: test_transform(),
            visible: true,
            hidden_in_game: false,
            affects_world: true,
            _cast_shadows: true,
            intensity,
            intensity_units: units.into(),
            color: UeColor {
                r: 255,
                g: 255,
                b: 255,
                a: 255,
            },
            use_temperature: false,
            temperature: 6500.0,
            attenuation_radius: 1000.0,
            source_radius: 0.0,
            soft_source_radius: 0.0,
            inner_cone_angle: 0.0,
            outer_cone_angle: 60.0,
            light_source_angle: 0.5357,
            ies_texture: None,
            light_function_material: None,
            real_time_capture: false,
        }
    }

    #[test]
    fn converts_ue_candelas_to_bevy_lumens() {
        let light = test_light("point", 250_000.0, "ELightUnits::Candelas");
        let expected = 250_000.0 * std::f32::consts::PI * 4.0;
        assert!((bevy_light_lumens(&light, 1.0) - expected).abs() < 0.5);
    }

    #[test]
    fn spot_emitter_flux_respects_the_ue_cone() {
        let light = test_light("spot", 10_000.0, "ELightUnits::Candelas");
        let outer = 60.0_f32.to_radians();
        let expected = 10_000.0 * std::f32::consts::PI;
        assert!((emitted_light_flux(&light, outer) - expected).abs() < 0.1);
    }

    #[test]
    fn lumens_spot_lights_keep_their_ue_cone_candela() {
        let outer = 60.0_f32.to_radians();
        let light = test_light("spot", 10_000.0, "ELightUnits::Lumens");
        // Bevy derives candela as `intensity / 4pi`; UE derives it from the
        // cone solid angle, so both engines must agree after the conversion.
        let bevy_candela = bevy_light_lumens(&light, outer) / (4.0 * std::f32::consts::PI);
        let ue_candela = 10_000.0 / spot_solid_angle(outer);
        assert!((bevy_candela - ue_candela).abs() < 0.5);
        // The emissive proxy still carries the authored total flux.
        assert_eq!(emitted_light_flux(&light, outer), 10_000.0);
        // Point lights already share Bevy's 4pi convention.
        let point = test_light("point", 10_000.0, "ELightUnits::Lumens");
        assert_eq!(bevy_light_lumens(&point, outer), 10_000.0);
    }

    #[test]
    fn legacy_unitless_lights_do_not_become_firefly_emitters() {
        let light = test_light("spot", 2_000.0, "Unitless");
        assert_eq!(bevy_light_lumens(&light, 60.0_f32.to_radians()), 200_000.0);
        assert_eq!(emitted_light_flux(&light, 60.0_f32.to_radians()), 200_000.0);
    }

    #[test]
    fn candle_temperature_is_warm() {
        let color = blackbody_srgb(1800.0).to_linear();
        assert!(color.red > color.green);
        assert!(color.green > color.blue);
    }

    #[test]
    fn parses_camera_override_and_uses_throne_room_preset() {
        assert_eq!(
            parse_vec3_arg("--camera-position", Some("1.5, 2, -3")),
            Some(Vec3::new(1.5, 2.0, -3.0))
        );
        assert_eq!(
            level_camera_placement("ThroneRoom_Level", Vec3::ZERO, 100.0),
            (Vec3::new(0.0, 4.5, 48.0), Vec3::new(0.0, 4.0, 40.0))
        );
    }

    #[test]
    fn classifies_only_lightblocker_asset_paths_as_raytracing_only() {
        assert!(is_lightblocker_mesh(
            "/Game/Assets/Environment/Lightblockers/Meshes/SM_ThroneRoom_Room_A1"
        ));
        assert!(!is_lightblocker_mesh(
            "/Game/Assets/Environment/ThroneRoom/Meshes/SM_ThroneRoom_Room_A1"
        ));
    }

    #[test]
    fn profiled_lights_keep_flux_for_the_uniform_solari_proxy() {
        let mut light = test_light("point", 250_000.0, "ELightUnits::Candelas");
        light.light_function_material = Some("/Game/VFX/LF_Caustics".into());
        assert!(emitted_light_flux(&light, 1.0) > 3_000_000.0);
    }

    #[test]
    fn does_not_enable_solari_while_expected_blas_work_is_pending() {
        let mut snapshot = RaytracingSceneStatusSnapshot {
            available_blas: 3,
            pending_compactions: 1,
            ..default()
        };
        assert!(!raytracing_scene_ready(&snapshot, 3));

        snapshot.pending_compactions = 0;
        assert!(raytracing_scene_ready(&snapshot, 3));
        assert!(!raytracing_scene_ready(&snapshot, 4));
    }

    #[test]
    fn udim_addressing_comes_only_from_the_texture_grid() {
        assert_eq!(udim_uv_transform(1, 1), Affine2::IDENTITY);

        // MI_ThroneRoom_MainGate_Frame_A: 10 columns, 3 rows.
        let transform = udim_uv_transform(10, 3);
        assert_eq!(transform.matrix2.x_axis, Vec2::new(0.1, 0.0));
        assert_eq!(transform.matrix2.y_axis, Vec2::new(0.0, 1.0 / 3.0));
        assert_eq!(transform.translation, Vec2::new(0.0, 2.0 / 3.0));
        // UDIM row 0 (tiles 1001-1010, mesh v in [0, 1]) is the bottom atlas row.
        assert!(transform
            .transform_point2(Vec2::new(0.0, 0.0))
            .abs_diff_eq(Vec2::new(0.0, 2.0 / 3.0), 1e-6));
        assert!(transform
            .transform_point2(Vec2::new(10.0, 1.0))
            .abs_diff_eq(Vec2::new(1.0, 1.0), 1e-6));
        // UDIM row 2 (mesh v in [-2, -1]) is the top atlas row.
        assert!(transform
            .transform_point2(Vec2::new(7.0, -2.0))
            .abs_diff_eq(Vec2::new(0.7, 0.0), 1e-6));
    }

    #[test]
    fn udim_addressing_keeps_columns_a_material_never_uses() {
        // MI_Courtyard_Stairs_Steps_Steps_Gray occupies columns 2-4 of a 5x1
        // atlas; deriving the transform from the grid leaves it there.
        assert!(udim_uv_transform(5, 1)
            .transform_point2(Vec2::new(2.0, 0.5))
            .abs_diff_eq(Vec2::new(0.4, 0.5), 1e-6));
    }

    #[test]
    fn reports_uvs_authored_outside_their_udim_atlas() {
        assert!(udim_uv_bounds_are_addressable(
            (Vec2::new(0.00294, -1.99581), Vec2::new(9.99623, 0.99858)),
            10,
            3,
        ));
        // MI_ThroneRoom_CupolaInterior_A1 is authored from u -8.53 on a 2x2 atlas.
        assert!(!udim_uv_bounds_are_addressable(
            (Vec2::new(-8.53, -0.5), Vec2::new(-7.5, 0.9)),
            2,
            2,
        ));
        assert!(!udim_uv_bounds_are_addressable(
            (Vec2::new(0.0, -13.5), Vec2::new(1.9, 0.9)),
            2,
            2,
        ));
    }

    #[test]
    fn preloads_only_texture_bundles_used_by_the_selected_scene() {
        let mesh_name = "/Game/Test/SM_Used".to_string();
        let material_name = "/Game/Test/MI_Used".to_string();
        let base_color_name = "/Game/Test/T_Used_Base".to_string();
        let normal_name = "/Game/Test/T_Used_Normal".to_string();
        let unused_name = "/Game/Test/T_Unused".to_string();
        let converted = ConvertedWorld {
            level: "Test".into(),
            post_process: None,
            actors: vec![ActorRecord {
                _name: "actor".into(),
                _label: None,
                kind: "StaticMeshActor".into(),
                transform: test_transform(),
                hidden: false,
                components: vec![ComponentRecord {
                    mesh: Some(mesh_name.clone()),
                    transform: test_transform(),
                    visible: true,
                    hidden_in_game: false,
                    instances: None,
                    override_materials: vec![],
                }],
                lights: vec![],
                atmosphere: None,
                height_fog: None,
                post_process: None,
            }],
            geometry: HashMap::from([(
                mesh_name,
                ConvertedMesh {
                    partitions: vec![PartitionRecord {
                        geometry: "unused".into(),
                        mesh: None,
                        meshlet: "unused".into(),
                        material_slot: 0,
                        material_index: None,
                        vertices: 0,
                        blas_vertices: 0,
                        blas_achieved_error: 0.0,
                        uv_min: None,
                        uv_max: None,
                        aabb_min: None,
                        aabb_max: None,
                    }],
                    material_slots: vec![Some(material_name.clone())],
                },
            )]),
            materials: HashMap::from([(
                material_name.clone(),
                MaterialRecord {
                    object: material_name,
                    kind: None,
                    parent: None,
                    emissive: None,
                    scalars: vec![],
                    vectors: vec![],
                    textures: vec![
                        TextureParameter {
                            name: "Base Color".into(),
                            association: "GlobalParameter".into(),
                            index: -1,
                            value: Some(base_color_name.clone()),
                        },
                        TextureParameter {
                            name: "Normal".into(),
                            association: "GlobalParameter".into(),
                            index: -1,
                            value: Some(normal_name.clone()),
                        },
                    ],
                    base_overrides: BaseMaterialOverrides::default(),
                },
            )]),
            textures: HashMap::from([
                (
                    base_color_name.clone(),
                    TextureExportRecord {
                        object: base_color_name,
                        output: "bundles/used-a.zorah_bundle#base".into(),
                        source_grid_columns: 1,
                        source_grid_rows: 1,
                    },
                ),
                (
                    normal_name.clone(),
                    TextureExportRecord {
                        object: normal_name,
                        output: "bundles/used-b.zorah_bundle#normal".into(),
                        source_grid_columns: 1,
                        source_grid_rows: 1,
                    },
                ),
                (
                    unused_name.clone(),
                    TextureExportRecord {
                        object: unused_name,
                        output: "bundles/unused.zorah_bundle#image".into(),
                        source_grid_columns: 1,
                        source_grid_rows: 1,
                    },
                ),
            ]),
        };

        assert_eq!(
            selected_texture_bundle_roots(&converted),
            vec![
                "bundles/used-a.zorah_bundle".to_string(),
                "bundles/used-b.zorah_bundle".to_string()
            ]
        );
    }

    #[test]
    fn ue_post_process_exposure_maps_to_bevy_ev100() {
        let post_process = PostProcessRecord {
            auto_exposure_min_ev100: Some(4.0),
            auto_exposure_max_ev100: Some(5.0),
            auto_exposure_bias: Some(0.99),
            ..legacy_zorah_post_process("GreenHouse_Level").unwrap()
        };

        let exposure = resolved_exposure_ev100(None, Some(&post_process));
        assert!((exposure - 3.51).abs() < 0.0001);
        assert_eq!(resolved_exposure_ev100(Some(7.0), Some(&post_process)), 7.0);
    }

    #[test]
    fn ue_post_process_bloom_drives_bevy_bloom() {
        // ThroneRoom is the one level that ticks bOverride_BloomMethod and
        // bOverride_BloomIntensity: BM_FFT at 0.003.
        let throne_room = resolved_bloom(legacy_zorah_post_process("ThroneRoom_Level").as_ref());
        assert_eq!(throne_room.intensity, 0.003);
        assert!(matches!(
            throne_room.scatter,
            BloomScatterModel::Gt7Glare { .. }
        ));

        // GreenHouse and Restir author no bloom override, so the example's own
        // preset stands in for UE's engine-wide defaults.
        let green_house = resolved_bloom(legacy_zorah_post_process("GreenHouse_Level").as_ref());
        assert_eq!(green_house.intensity, Bloom::GT7_GLARE.intensity);
        assert!(matches!(
            green_house.scatter,
            BloomScatterModel::Gt7Glare { .. }
        ));
        assert_eq!(resolved_bloom(None).intensity, Bloom::GT7_GLARE.intensity);
    }

    #[test]
    fn ue_sum_of_gaussians_bloom_maps_to_the_parametric_curve() {
        let post_process = PostProcessRecord {
            bloom_method: Some("BM_SOG".into()),
            bloom_intensity: Some(0.675),
            ..legacy_zorah_post_process("Restir_Level").unwrap()
        };

        let bloom = resolved_bloom(Some(&post_process));
        assert_eq!(bloom.intensity, 0.675);
        assert!(matches!(bloom.scatter, BloomScatterModel::Aesthetic));
    }

    #[test]
    fn non_finite_bloom_intensity_falls_back_to_the_preset() {
        let post_process = PostProcessRecord {
            bloom_intensity: Some(f32::NAN),
            ..legacy_zorah_post_process("ThroneRoom_Level").unwrap()
        };

        assert_eq!(
            resolved_bloom(Some(&post_process)).intensity,
            Bloom::GT7_GLARE.intensity
        );
    }

    #[test]
    fn ue_atmosphere_coefficients_are_converted_from_inverse_kilometers() {
        let record = SkyAtmosphereRecord {
            _name: "atmosphere".into(),
            atmosphere_height_km: Some(50.0),
            rayleigh_scattering_scale: Some(2.0),
            rayleigh_scattering_per_km: Some(UeLinearColor {
                r: 0.01,
                g: 0.02,
                b: 0.03,
                _a: 1.0,
            }),
            mie_exponential_distribution_km: Some(2.0),
            sky_luminance_factor: Some(UeLinearColor {
                r: 1.0,
                g: 2.0,
                b: 3.0,
                _a: 1.0,
            }),
            ..default()
        };
        let (medium, inner_radius, outer_radius, _, environment_intensity) =
            configured_atmosphere(Some(&record), None);

        assert_eq!(inner_radius, 6_360_000.0);
        assert_eq!(outer_radius, 6_410_000.0);
        assert!(medium.terms[0]
            .scattering
            .abs_diff_eq(Vec3::new(0.00002, 0.00004, 0.00006), 1e-9));
        assert!(matches!(
            medium.terms[1].falloff,
            Falloff::Exponential { scale } if (scale - 0.04).abs() < 1e-6
        ));
        assert_eq!(environment_intensity, 2.0);
    }

    #[test]
    fn ue_atmosphere_transform_modes_map_to_bevy_planet_centers() {
        let radius = 6_360_000.0;
        let top_actor = test_atmosphere_actor(
            "ESkyAtmosphereTransformMode::PlanetTopAtComponentTransform",
            UeVec3 {
                x: 100.0,
                y: 200.0,
                z: 300.0,
            },
        );
        let center_actor = test_atmosphere_actor(
            "PlanetCenterAtComponentTransform",
            UeVec3 {
                x: 100.0,
                y: 200.0,
                z: 300.0,
            },
        );
        // UE centimeters become Bevy meters and UE (X, Y, Z) becomes
        // Bevy (Y, Z, -X).
        let component_position = Vec3::new(2.0, 3.0, -1.0);
        assert_eq!(
            atmosphere_planet_center(Some(&top_actor), radius),
            component_position - Vec3::Y * radius
        );
        assert_eq!(
            atmosphere_planet_center(Some(&center_actor), radius),
            component_position
        );
        assert_eq!(atmosphere_planet_center(None, radius), -Vec3::Y * radius);
    }

    #[test]
    fn ue_height_fog_adds_a_low_altitude_scattering_term() {
        let fog = HeightFogRecord {
            _name: "fog".into(),
            fog_density: Some(0.04),
            fog_height_falloff: Some(0.25),
            volumetric_fog_extinction_scale: Some(2.0),
            volumetric_fog_scattering_distribution: Some(0.5),
            ..default()
        };
        let (medium, _, _, _, _) = configured_atmosphere(None, Some(&fog));
        let fog_term = &medium.terms[3];

        assert!(fog_term.scattering.abs_diff_eq(Vec3::splat(0.00008), 1e-9));
        assert_eq!(fog_term.absorption, Vec3::ZERO);
        assert!(matches!(
            fog_term.falloff,
            Falloff::Exponential { scale } if (scale - (4.0 / 60.0)).abs() < 1e-6
        ));
        assert!(matches!(
            fog_term.phase,
            PhaseFunction::Mie { asymmetry } if (asymmetry - 0.5).abs() < 1e-6
        ));
    }

    #[test]
    fn material_inheritance_preserves_mask_and_two_sided_intent() {
        let parent: MaterialRecord = serde_json::from_str(
            r#"{
                "object": "parent",
                "parent": null,
                "base_overrides": {
                    "BlendMode": "BLEND_Masked",
                    "TwoSided": true,
                    "OpacityMaskClipValue": 0.42
                }
            }"#,
        )
        .unwrap();
        let child: MaterialRecord =
            serde_json::from_str(r#"{"object": "child", "parent": "parent"}"#).unwrap();
        let records = HashMap::from([
            (parent.object.clone(), parent),
            (child.object.clone(), child),
        ]);
        let effective =
            resolve_effective_material("child", &records, &mut HashMap::new(), &mut Vec::new());
        let properties = source_material_render_properties(&effective);

        assert_eq!(effective.blend_mode, SourceBlendMode::Masked);
        assert!(matches!(properties.alpha_mode, AlphaMode::Mask(value) if value == 0.42));
        assert_eq!(runtime_alpha_mode(&effective, false), AlphaMode::Opaque);
        assert!(matches!(
            runtime_alpha_mode(&effective, true),
            AlphaMode::Mask(value) if value == 0.42
        ));
        assert!(properties.double_sided);
        assert_eq!(properties.cull_mode, None);
    }

    #[test]
    fn explicit_default_base_overrides_clear_inherited_values() {
        let parent: MaterialRecord = serde_json::from_str(
            r#"{
                "object": "parent",
                "parent": null,
                "base_overrides": {
                    "BlendMode": "BLEND_Masked",
                    "TwoSided": true
                }
            }"#,
        )
        .unwrap();
        let child: MaterialRecord = serde_json::from_str(
            r#"{
                "object": "child",
                "parent": "parent",
                "base_overrides": {
                    "bOverride_BlendMode": true,
                    "bOverride_TwoSided": true
                }
            }"#,
        )
        .unwrap();
        let records = HashMap::from([
            (parent.object.clone(), parent),
            (child.object.clone(), child),
        ]);
        let effective =
            resolve_effective_material("child", &records, &mut HashMap::new(), &mut Vec::new());
        let properties = source_material_render_properties(&effective);

        assert_eq!(properties.alpha_mode, AlphaMode::Opaque);
        assert!(!properties.double_sided);
        assert_eq!(properties.cull_mode, Some(Face::Back));
    }

    #[test]
    fn foliage_diffuse_texture_alias_is_selected_as_base_color() {
        let material = EffectiveMaterial {
            textures: vec![TextureParameter {
                name: "Diffuse Texture".into(),
                association: "GlobalParameter".into(),
                index: -1,
                value: Some("foliage-albedo".into()),
            }],
            ..default()
        };
        assert_eq!(
            select_texture(&material, BASE_COLOR_TEXTURE_NAMES),
            Some("foliage-albedo")
        );
    }

    #[test]
    fn explicit_foliage_texture_beats_inherited_legacy_basecolor_alias() {
        let material = EffectiveMaterial {
            textures: vec![
                TextureParameter {
                    name: "BaseColor".into(),
                    association: "GlobalParameter".into(),
                    index: -1,
                    value: Some("inherited-placeholder".into()),
                },
                TextureParameter {
                    name: "Base Color Texture".into(),
                    association: "GlobalParameter".into(),
                    index: -1,
                    value: Some("authored-albedo".into()),
                },
            ],
            ..default()
        };
        assert_eq!(
            select_texture(&material, BASE_COLOR_TEXTURE_NAMES),
            Some("authored-albedo")
        );
    }

    #[test]
    fn ors_specular_channel_is_not_interpreted_as_metallic() {
        let ors = TextureParameter {
            name: "ORS".into(),
            association: "GlobalParameter".into(),
            index: -1,
            value: Some("foliage-ors".into()),
        };
        let orm = TextureParameter {
            name: "ORM".into(),
            association: "GlobalParameter".into(),
            index: -1,
            value: Some("metal-orm".into()),
        };
        assert!(!texture_carries_metallic(Some(&ors)));
        assert!(texture_carries_metallic(Some(&orm)));
    }

    #[test]
    fn ue_material_luminance_modulates_the_selected_tint() {
        let material = EffectiveMaterial {
            scalars: vec![ScalarParameter {
                name: "Luminance".into(),
                association: "GlobalParameter".into(),
                index: -1,
                value: 0.5,
            }],
            vectors: vec![VectorParameter {
                name: "Tint".into(),
                association: "GlobalParameter".into(),
                index: -1,
                value: "FFFFFF (FLinearColor)".into(),
            }],
            ..default()
        };
        let color = material_base_color(&material).to_linear();
        assert!((color.red - 0.5).abs() < 1e-6);
        assert!((color.green - 0.5).abs() < 1e-6);
        assert!((color.blue - 0.5).abs() < 1e-6);
    }

    #[test]
    fn base_material_expression_texture_aliases_are_selected() {
        let material = EffectiveMaterial {
            textures: vec![
                TextureParameter {
                    name: "Gold Base Color".into(),
                    association: "GlobalParameter".into(),
                    index: -1,
                    value: Some("gold-albedo".into()),
                },
                TextureParameter {
                    name: "Gold Base Normal".into(),
                    association: "GlobalParameter".into(),
                    index: -1,
                    value: Some("gold-normal".into()),
                },
                TextureParameter {
                    name: "Gold Base ORM".into(),
                    association: "GlobalParameter".into(),
                    index: -1,
                    value: Some("gold-orm".into()),
                },
            ],
            ..default()
        };
        assert_eq!(
            select_texture(&material, BASE_COLOR_TEXTURE_NAMES),
            Some("gold-albedo")
        );
        assert_eq!(
            select_texture(&material, NORMAL_TEXTURE_NAMES),
            Some("gold-normal")
        );
        assert_eq!(
            select_texture(&material, ORM_TEXTURE_NAMES),
            Some("gold-orm")
        );
    }

    #[test]
    fn override_materials_are_indexed_by_the_ue_material_slot() {
        // A mesh whose LOD0 section 0 draws with StaticMaterials slot 1.
        let partition: PartitionRecord = serde_json::from_str(
            r#"{
                "geometry": "part.glb",
                "meshlet": "part.meshlet",
                "material_slot": 0,
                "material_index": 1
            }"#,
        )
        .unwrap();
        let slots = vec![Some("base-for-section-0".to_string())];
        let overrides = vec![String::new(), "MI_Override".to_string()];

        assert_eq!(
            partition_material(&partition, &slots, &overrides),
            Some("MI_Override")
        );
        assert_eq!(
            partition_material(&partition, &slots, &[]),
            Some("base-for-section-0")
        );

        // Manifests written before the field existed keep the section index.
        let legacy: PartitionRecord = serde_json::from_str(
            r#"{"geometry": "part.glb", "meshlet": "part.meshlet", "material_slot": 0}"#,
        )
        .unwrap();
        assert_eq!(legacy.material_index, None);
        assert_eq!(
            partition_material(&legacy, &slots, &overrides),
            Some("base-for-section-0")
        );
    }

    #[test]
    fn ue_sky_light_uses_the_same_scale_for_raster_and_solari() {
        assert!((ue_sky_light_illuminance(std::f32::consts::PI) - 251.32742).abs() < 1e-4);
        assert_eq!(ue_sky_light_illuminance(-1.0), 0.0);
    }
}
