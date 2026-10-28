//! 3D FBM Noise Generation for Atmospheric Effects
//!
//! This module generates a 3D texture filled with Fractional Brownian Motion (FBM) noise,
//! which can be used for atmospheric effects like clouds, fog, and volumetric effects.

use bevy_asset::{load_embedded_asset, AssetId, Assets};
use bevy_image::Image;
use bevy_ecs::{
    resource::Resource,
    system::{Local, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_math::{UVec2, UVec3};
use bevy_render::{
    render_resource::{binding_types::*, *},
    renderer::{RenderAdapter, RenderDevice, RenderQueue},
    settings::WgpuFeatures,
    texture::CachedTexture,
    Extract,
};
use bevy_utils::default;
use tracing::warn;

use super::{
    cloud_profile::{
        constant_lean_texels, f32_to_f16, nubis_profile_texels, physical_lean_texels,
        physical_profile_texels, CloudProfiles,
        CLOUD_PROFILE_HEIGHT_RES, CLOUD_PROFILE_TYPE_RES,
    },
    detail_noise::{DetailNoiseGenerated, DetailNoiseTexture},
    perlin_worley_noise::{PerlinWorleyNoiseGenerated, PerlinWorleyNoiseTexture},
    VolumetricCloudLayer,
};

/// Storage format of the cirrus generating field: R16Float is a storage format only with the
/// adapter-specific format features, not in WebGPU's guaranteed set; elsewhere the field falls
/// back to Rgba16Float (a core storage format), which the generator declares through
/// `CIRRUS_STORAGE_RGBA16`. R16Float is kept where possible (2.8 MB with mips vs 11 MB).
#[derive(Resource, Clone, Copy)]
pub struct CloudNoiseFormats {
    pub cirrus: TextureFormat,
}

impl CloudNoiseFormats {
    pub fn cirrus_shader_defs(&self) -> Vec<bevy_shader::ShaderDefVal> {
        if self.cirrus == TextureFormat::Rgba16Float {
            vec!["CIRRUS_STORAGE_RGBA16".into()]
        } else {
            Vec::new()
        }
    }
}

impl FromWorld for CloudNoiseFormats {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let adapter = world.resource::<RenderAdapter>();
        let adapter_formats = device
            .features()
            .contains(WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES);
        let storage = |format: TextureFormat| {
            if adapter_formats
                && adapter
                    .get_texture_format_features(format)
                    .allowed_usages
                    .contains(TextureUsages::STORAGE_BINDING)
            {
                format
            } else {
                TextureFormat::Rgba16Float
            }
        };
        Self {
            cirrus: storage(TextureFormat::R16Float),
        }
    }
}

/// Parameters for controlling FBM noise generation
#[derive(Clone, Copy, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct FbmNoiseParams {
    /// Number of octaves for the FBM noise (more octaves = more detail)
    pub octaves: u32,
    /// Base frequency of the noise
    pub frequency: f32,
    /// Base amplitude of the noise
    pub amplitude: f32,
    /// Frequency multiplier for each octave (typically 2.0)
    pub lacunarity: f32,
    /// Amplitude multiplier for each octave (typically 0.5)
    pub persistence: f32,
    /// Cloud life-cycle phase, `evolution_time / lifetime + 0.5` ([`crate::VolumetricCloudLayer`]).
    pub life_t: f32,
    /// 1 when the life cycle is on (`lifetime > 0`).
    pub life_enabled: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl Default for FbmNoiseParams {
    fn default() -> Self {
        Self {
            octaves: 4,
            frequency: 1.0,
            amplitude: 1.0,
            lacunarity: 2.0,
            persistence: 0.5,
            life_t: 0.5,
            life_enabled: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Size of the 2D noise texture
#[derive(Clone, Copy)]
pub struct NoiseTextureSize {
    pub size: UVec2,
}

impl Default for NoiseTextureSize {
    fn default() -> Self {
        Self {
            // Procedural weather map (design.md §1.3 / §3.2): 512² RGBA.
            // R coverage, G type, B density scale, A spare.
            size: UVec2::new(512, 512),
        }
    }
}

/// Resource containing the bind group layout for the FBM noise pass
#[derive(Resource)]
pub struct FbmNoiseBindGroupLayout {
    pub layout: BindGroupLayout,
    pub descriptor: BindGroupLayoutDescriptor,
}

impl FromWorld for FbmNoiseBindGroupLayout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let descriptor = BindGroupLayoutDescriptor::new(
            "fbm_noise_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (
                        // 2D noise texture storage
                        13,
                        texture_storage_2d(
                            // Weather map (coverage, type, density scale, spare).
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    (
                        // FBM parameters uniform buffer
                        14,
                        uniform_buffer::<FbmNoiseParams>(false),
                    ),
                    (
                        // Cirrus generating field (`cirrus` entry point).
                        15,
                        texture_storage_2d(
                            world.resource::<CloudNoiseFormats>().cirrus,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let layout =
            render_device.create_bind_group_layout(descriptor.label.as_ref(), &descriptor.entries);

        Self { layout, descriptor }
    }
}

/// Resource containing the compute pipeline for FBM noise generation
#[derive(Resource)]
pub struct FbmNoisePipeline {
    pub pipeline: CachedComputePipelineId,
    pub cirrus_pipeline: CachedComputePipelineId,
}

impl FromWorld for FbmNoisePipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layout = world.resource::<FbmNoiseBindGroupLayout>();

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("fbm_noise_2d_pipeline".into()),
            layout: vec![layout.descriptor.clone()],
            shader: load_embedded_asset!(world, "fbm_noise_3d.wesl"),
            entry_point: Some("main".into()),
            ..default()
        });
        let cirrus_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cirrus_field_pipeline".into()),
            layout: vec![layout.descriptor.clone()],
            shader: load_embedded_asset!(world, "fbm_noise_3d.wesl"),
            entry_point: Some("cirrus".into()),
            shader_defs: world.resource::<CloudNoiseFormats>().cirrus_shader_defs(),
            ..default()
        });

        Self { pipeline, cirrus_pipeline }
    }
}

/// Resource containing the generated 2D noise texture
#[derive(Resource)]
pub struct FbmNoiseTexture {
    pub texture: CachedTexture,
    pub size: UVec2,
    /// Cirrus generating field (R16F, unit Gaussian, -5/3 spectrum; tile = cirrus outer scale),
    /// with a mip chain; each mip is generated as the band-limited field (its resolvable octaves).
    pub cirrus: CachedTexture,
    /// Single-mip views of [`FbmNoiseTexture::cirrus`] for generation.
    pub cirrus_mips: Vec<TextureView>,
    /// Vertical profile lookups (Rgba16Float, height x type; `cloud_profile.rs`).
    pub profile: CachedTexture,
    /// Moisture fluctuation of the physical cloud scheme (`cloud_noise_quantiles.wesl` bake,
    /// `clouds.wesl` cloud_scheme_density): Rgba16Float 128^3 on the shape-noise tile, R the
    /// Gaussianized shape noise g, G its variance within the texel, B the thermal bubbles b
    /// (three octaves), A its variance; 0 at mip 0 (the texels are the field), and in each mip
    /// the variance over the block it averages: what a footprint of that size leaves unresolved.
    pub scheme_noise: CachedTexture,
    /// Single-mip views of [`FbmNoiseTexture::scheme_noise`] (bake and mip generation).
    pub scheme_noise_mips: Vec<TextureView>,
}

/// Edge of the physical cloud scheme's fluctuation texture ([`FbmNoiseTexture::scheme_noise`]).
pub const CLOUD_SCHEME_NOISE_SIZE: u32 = 128;

/// Weight of the thermal bubbles against the shape noise in the physical cloud scheme: the first
/// bubble octave (cells of 1/8 of the shape tile) continues the shape noise's spectrum (base
/// frequency 4) as (1/2)^H, H = 2/3, times the layer's detail strength (0 = no bubbles). Must
/// match `clouds.wesl` cloud_bubble_weight.
pub fn cloud_bubble_weight(detail_strength: f32) -> f32 {
    0.5f32.powf(2.0 / 3.0) * detail_strength.max(0.0)
}

/// Size of the cirrus generating-field texture.
pub const CIRRUS_TEXTURE_SIZE: u32 = 1024;

/// Resource containing the bind group for the noise generation pass
#[derive(Resource)]
pub struct FbmNoiseBindGroup {
    pub bind_group: BindGroup,
    /// One per cirrus mip (binding 15 = that mip).
    pub cirrus_bind_groups: Vec<BindGroup>,
}

/// Resource containing the uniform buffer for FBM parameters
#[derive(Resource)]
pub struct FbmNoiseParamsBuffer {
    pub buffer: Buffer,
}

/// System to initialize the FBM noise texture.
///
/// Allocates GPU memory directly instead of the render [`TextureCache`](bevy_render::texture::TextureCache)
/// so this `RenderStartup` system does not need to run after `init_gpu_resource::<TextureCache>`.
pub fn init_fbm_noise_texture(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    formats: Res<CloudNoiseFormats>,
    mut commands: bevy_ecs::system::Commands,
) {
    let size = NoiseTextureSize::default();

    let texture_descriptor = TextureDescriptor {
        label: Some("cloud_weather_map_texture"),
        size: Extent3d {
            width: size.size.x,
            height: size.size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        // Weather map (coverage, type, density scale, spare).
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    };

    let texture = render_device.create_texture(&texture_descriptor);
    let default_view = texture.create_view(&TextureViewDescriptor::default());

    let cirrus_texture = render_device.create_texture(&TextureDescriptor {
        label: Some("cirrus_generating_field_texture"),
        size: Extent3d {
            width: CIRRUS_TEXTURE_SIZE,
            height: CIRRUS_TEXTURE_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: CIRRUS_TEXTURE_SIZE.ilog2() + 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: formats.cirrus,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let cirrus_view = cirrus_texture.create_view(&TextureViewDescriptor::default());
    let cirrus_mips = (0..CIRRUS_TEXTURE_SIZE.ilog2() + 1)
        .map(|mip| {
            cirrus_texture.create_view(&TextureViewDescriptor {
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..TextureViewDescriptor::default()
            })
        })
        .collect();

    let profile_texture = render_device.create_texture(&TextureDescriptor {
        label: Some("cloud_profile_lut_texture"),
        size: CLOUD_PROFILE_TEXTURE_EXTENT,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST
            | TextureUsages::COPY_SRC
            | TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    write_cloud_profile(&render_queue, &profile_texture, physical_profile_texels());
    write_cloud_lean(&render_queue, &profile_texture, physical_lean_texels());
    // Noise distribution rows: uniform until measured; scheme row: threshold 0, mean excess and
    // magnitude 1 until computed (`cloud_noise_quantiles.wesl`).
    let identity: Vec<u8> = (0..(CLOUD_NOISE_QUANTILE_ROWS + 1) * CLOUD_PROFILE_HEIGHT_RES as u32)
        .flat_map(|i| {
            let row = i / CLOUD_PROFILE_HEIGHT_RES as u32;
            let q = (i as usize % CLOUD_PROFILE_HEIGHT_RES) as f32 / (CLOUD_PROFILE_HEIGHT_RES - 1) as f32;
            if row < CLOUD_NOISE_QUANTILE_ROWS {
                [q, q, q, 0.0]
            } else {
                [0.0, 1.0, 1.0, 1.0]
            }
            .map(f32_to_f16)
        })
        .flat_map(u16::to_le_bytes)
        .collect();
    render_queue.write_texture(
        TexelCopyTextureInfo {
            texture: &profile_texture,
            mip_level: 0,
            origin: Origin3d {
                x: 0,
                y: CLOUD_PROFILE_TYPE_RES as u32,
                z: 0,
            },
            aspect: TextureAspect::All,
        },
        &identity,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8 * CLOUD_PROFILE_HEIGHT_RES as u32),
            rows_per_image: None,
        },
        Extent3d {
            width: CLOUD_PROFILE_HEIGHT_RES as u32,
            height: CLOUD_NOISE_QUANTILE_ROWS + 1,
            depth_or_array_layers: 1,
        },
    );
    let profile_view = profile_texture.create_view(&TextureViewDescriptor::default());

    let scheme_mip_count = CLOUD_SCHEME_NOISE_SIZE.ilog2() + 1;
    let scheme_noise = render_device.create_texture(&TextureDescriptor {
        label: Some("cloud_scheme_noise_texture"),
        size: Extent3d {
            width: CLOUD_SCHEME_NOISE_SIZE,
            height: CLOUD_SCHEME_NOISE_SIZE,
            depth_or_array_layers: CLOUD_SCHEME_NOISE_SIZE,
        },
        mip_level_count: scheme_mip_count,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let scheme_noise_view = scheme_noise.create_view(&TextureViewDescriptor::default());
    let scheme_noise_mips = (0..scheme_mip_count)
        .map(|mip| {
            scheme_noise.create_view(&TextureViewDescriptor {
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..TextureViewDescriptor::default()
            })
        })
        .collect();

    commands.insert_resource(FbmNoiseTexture {
        texture: CachedTexture {
            texture,
            default_view,
        },
        size: size.size,
        cirrus: CachedTexture {
            texture: cirrus_texture,
            default_view: cirrus_view,
        },
        cirrus_mips,
        profile: CachedTexture {
            texture: profile_texture,
            default_view: profile_view,
        },
        scheme_noise: CachedTexture {
            texture: scheme_noise,
            default_view: scheme_noise_view,
        },
        scheme_noise_mips,
    });
}

/// The profile rows (one per cloud type).
const CLOUD_PROFILE_EXTENT: Extent3d = Extent3d {
    width: CLOUD_PROFILE_HEIGHT_RES as u32,
    height: CLOUD_PROFILE_TYPE_RES as u32,
    depth_or_array_layers: 1,
};
/// Noise distribution rows after the profile rows: one per shape-noise mip (128^3 -> 8 levels);
/// then one cloud-scheme row. Must match `clouds.wesl` and `cloud_noise_quantiles.wesl`, which
/// also take the profile row count.
const CLOUD_NOISE_QUANTILE_ROWS: u32 = 8;
/// First of the lean rows (one per cloud type, `cloud_profile.rs`), after the scheme row. Must
/// match `clouds.wesl` CLOUD_LEAN_FIRST_ROW.
const CLOUD_LEAN_FIRST_ROW: u32 = CLOUD_PROFILE_TYPE_RES as u32 + CLOUD_NOISE_QUANTILE_ROWS + 1;
/// The profile rows, the noise distribution rows, the scheme row and the lean rows.
const CLOUD_PROFILE_TEXTURE_EXTENT: Extent3d = Extent3d {
    width: CLOUD_PROFILE_HEIGHT_RES as u32,
    height: CLOUD_LEAN_FIRST_ROW + CLOUD_PROFILE_TYPE_RES as u32,
    depth_or_array_layers: 1,
};

fn write_cloud_lean(queue: &RenderQueue, texture: &Texture, rgba16f: &[u8]) {
    queue.write_texture(
        TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d {
                x: 0,
                y: CLOUD_LEAN_FIRST_ROW,
                z: 0,
            },
            aspect: TextureAspect::All,
        },
        rgba16f,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8 * CLOUD_PROFILE_HEIGHT_RES as u32),
            rows_per_image: None,
        },
        CLOUD_PROFILE_EXTENT,
    );
}

fn write_cloud_profile(queue: &RenderQueue, texture: &Texture, rgba16f: &[u8]) {
    queue.write_texture(
        texture.as_image_copy(),
        rgba16f,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8 * CLOUD_PROFILE_HEIGHT_RES as u32),
            rows_per_image: None,
        },
        CLOUD_PROFILE_EXTENT,
    );
}

/// Profile lookup texels and lean rows (Rgba16Float) chosen in the extract schedule, uploaded by
/// [`upload_cloud_profiles`].
#[derive(Resource, Default)]
pub struct PendingCloudProfiles(pub(crate) Option<Vec<u8>>, pub(crate) Option<Vec<u8>>);

/// What the lookup holds.
#[derive(PartialEq)]
pub(crate) enum ProfileSource {
    Physical,
    Nubis,
    Image(AssetId<Image>),
}

/// Picks the lookups of the first cloud layer's [`CloudProfiles`] and queues an upload when they
/// change: a loaded image of the lookup format, else the physical set (the analytic envelope is
/// evaluated in the shader and leaves the lookups unused). One set for all atmospheres, like the
/// weather map.
pub(crate) fn extract_cloud_profiles(
    layers: Extract<Query<(&CloudProfiles, &VolumetricCloudLayer)>>,
    images: Extract<Res<Assets<Image>>>,
    mut last: Local<Option<ProfileSource>>,
    mut last_lean: Local<Option<Option<u32>>>,
    mut pending: ResMut<PendingCloudProfiles>,
) {
    let Some((profiles, layer)) = layers.iter().next() else {
        return;
    };
    // Lean rows: the physical set's own, else a constant updraft at the layer's speed (an image
    // carries no updraft; the Nubis and envelope modes lean analytically in the shader).
    let lean = (!matches!(profiles, CloudProfiles::Physical)).then(|| layer.updraft_speed.to_bits());
    if last_lean.as_ref() != Some(&lean) {
        pending.1 = Some(match lean {
            None => physical_lean_texels().to_vec(),
            Some(bits) => constant_lean_texels(f32::from_bits(bits)),
        });
        *last_lean = Some(lean);
    }
    let expected = (
        TextureFormat::Rgba16Float,
        CLOUD_PROFILE_EXTENT.width,
        CLOUD_PROFILE_EXTENT.height,
    );
    if let CloudProfiles::Nubis = profiles {
        if last.as_ref() != Some(&ProfileSource::Nubis) {
            pending.0 = Some(nubis_profile_texels().to_vec());
            *last = Some(ProfileSource::Nubis);
        }
        return;
    }
    let image = match profiles {
        CloudProfiles::Image(handle) => images.get(handle).map(|image| (handle.id(), image)),
        _ => None,
    };
    let usable = image.filter(|(_, image)| {
        let d = &image.texture_descriptor;
        (d.format, d.size.width, d.size.height) == expected && image.data.is_some()
    });
    let source = usable.map_or(ProfileSource::Physical, |(id, _)| ProfileSource::Image(id));
    if last.as_ref() == Some(&source) {
        return;
    }
    if let (Some((_, image)), None) = (image, usable) {
        let d = &image.texture_descriptor;
        warn!(
            "CloudProfiles image is {:?} {}x{}, expected Rgba16Float {}x{}: using the physical profiles",
            d.format, d.size.width, d.size.height, expected.1, expected.2
        );
    }
    pending.0 = Some(match usable {
        Some((_, image)) => image.data.clone().unwrap_or_default(),
        None => physical_profile_texels().to_vec(),
    });
    *last = Some(source);
}

pub fn upload_cloud_profiles(
    mut pending: ResMut<PendingCloudProfiles>,
    texture: Option<Res<FbmNoiseTexture>>,
    queue: Res<RenderQueue>,
) {
    let Some(texture) = texture else {
        return;
    };
    if let Some(texels) = pending.0.take() {
        write_cloud_profile(&queue, &texture.profile.texture, &texels);
    }
    if let Some(texels) = pending.1.take() {
        write_cloud_lean(&queue, &texture.profile.texture, &texels);
    }
}

/// Measures the shape- and billow-noise distributions of each noise mip once both noises exist,
/// and tabulates the thermal cloud scheme, into the last rows of the profile texture
/// (`cloud_noise_quantiles.wesl`), so the scheme exceeds its threshold with probability p (the
/// profile is the cloud fraction) at every level of detail.
#[derive(Resource)]
pub struct CloudNoiseQuantilePipeline {
    layout: BindGroupLayoutDescriptor,
    histogram_marginals: CachedComputePipelineId,
    prefix: CachedComputePipelineId,
    bake: CachedComputePipelineId,
    distributions: CachedComputePipelineId,
    scheme: CachedComputePipelineId,
}

/// `cloud_noise_quantiles.wesl` QuantileParams.
#[derive(Clone, Copy, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct CloudNoiseQuantileParams {
    _pad0: f32,
    bubble_weight: f32,
    _pad1: f32,
    _pad2: f32,
}

impl FromWorld for CloudNoiseQuantilePipeline {
    fn from_world(world: &mut World) -> Self {
        let layout = BindGroupLayoutDescriptor::new(
            "cloud_noise_quantiles_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_3d(TextureSampleType::Float { filterable: false }),
                    storage_buffer_sized(false, None),
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                    texture_3d(TextureSampleType::Float { filterable: false }),
                    uniform_buffer::<CloudNoiseQuantileParams>(false),
                    texture_storage_3d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );
        let shader = load_embedded_asset!(world, "cloud_noise_quantiles.wesl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = |entry: &'static str| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(format!("cloud_noise_quantiles_{entry}").into()),
                layout: vec![layout.clone()],
                shader: shader.clone(),
                entry_point: Some(entry.into()),
                ..default()
            })
        };
        Self {
            histogram_marginals: pipeline("histogram_marginals"),
            prefix: pipeline("prefix"),
            bake: pipeline("bake"),
            distributions: pipeline("distributions"),
            scheme: pipeline("scheme"),
            layout,
        }
    }
}

/// Bakes the physical cloud scheme's fluctuation texture (g, g^2, b, b^2 with a box-filtered mip
/// chain) and tabulates the scheme (`cloud_noise_quantiles.wesl`) once the shape noise exists,
/// and again when the first cloud layer's bubble weight changes (must match `clouds.wesl`
/// cloud_bubble_weight).
pub fn compute_cloud_noise_quantiles(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<CloudNoiseQuantilePipeline>,
    profile: Option<Res<FbmNoiseTexture>>,
    noise: Option<Res<PerlinWorleyNoiseTexture>>,
    detail: Option<Res<DetailNoiseTexture>>,
    generated: Res<PerlinWorleyNoiseGenerated>,
    detail_generated: Res<DetailNoiseGenerated>,
    mip_pipeline: Res<super::perlin_worley_noise::NoiseMipDownsamplePipeline>,
    mip_layout: Res<super::perlin_worley_noise::NoiseMipDownsampleBindGroupLayout>,
    layers: Query<&crate::GpuCloudLayers>,
    mut last: Local<Option<u32>>,
) {
    let (Some(profile), Some(noise), Some(detail)) = (profile, noise, detail) else {
        return;
    };
    let Some(layer) = layers.iter().next() else {
        return;
    };
    if generated.0 < noise.mip_level_count || !detail_generated.0 {
        return;
    }
    let bubble_weight = cloud_bubble_weight(layer.detail_strength);
    let key = bubble_weight.to_bits();
    if last.as_ref() == Some(&key) {
        return;
    }
    let pipelines = [
        pipeline.histogram_marginals,
        pipeline.prefix,
        pipeline.bake,
        pipeline.distributions,
        pipeline.scheme,
    ]
    .map(|id| pipeline_cache.get_compute_pipeline(id));
    let [Some(histogram_marginals), Some(prefix), Some(bake), Some(distributions), Some(scheme)] =
        pipelines
    else {
        return;
    };
    let Some(mip_compute) = pipeline_cache.get_compute_pipeline(mip_pipeline.moments) else {
        return;
    };
    // Bin blocks of `cloud_noise_quantiles.wesl` (4096 counters), zero-initialized.
    let bins = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloud_noise_histogram"),
        size: 4096 * 4,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let params = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("cloud_noise_quantile_params"),
        contents: bytemuck::bytes_of(&CloudNoiseQuantileParams {
            _pad0: 0.0,
            bubble_weight,
            _pad1: 0.0,
            _pad2: 0.0,
        }),
        usage: BufferUsages::UNIFORM,
    });
    let bind_group = render_device.create_bind_group(
        "cloud_noise_quantiles_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            &noise.texture.default_view,
            bins.as_entire_binding(),
            &profile.profile.default_view,
            &detail.texture.default_view,
            params.as_entire_binding(),
            &profile.scheme_noise_mips[0],
        )),
    );
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloud_noise_quantiles"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloud_noise_quantiles"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bind_group, &[]);
        let n = (noise.size.max(UVec3::splat(CLOUD_SCHEME_NOISE_SIZE)) + UVec3::splat(3)) / 4;
        pass.set_pipeline(histogram_marginals);
        pass.dispatch_workgroups(n.x, n.y, n.z);
        pass.set_pipeline(prefix);
        pass.dispatch_workgroups(1, 1, 1);
        let b = CLOUD_SCHEME_NOISE_SIZE.div_ceil(4);
        pass.set_pipeline(bake);
        pass.dispatch_workgroups(b, b, b);
        let row = CLOUD_PROFILE_HEIGHT_RES as u32;
        pass.set_pipeline(distributions);
        pass.dispatch_workgroups(row * CLOUD_NOISE_QUANTILE_ROWS / 64, 1, 1);
        pass.set_pipeline(scheme);
        pass.dispatch_workgroups(row / 64, 1, 1);
    }
    // Mips: footprint mean and variance of g and b.
    let mut size = CLOUD_SCHEME_NOISE_SIZE;
    for mip in 1..profile.scheme_noise_mips.len() {
        size = (size / 2).max(1);
        let mip_bind = render_device.create_bind_group(
            "cloud_scheme_noise_mip",
            &mip_layout.layout,
            &BindGroupEntries::with_indices((
                (13, &profile.scheme_noise_mips[mip - 1]),
                (14, &profile.scheme_noise_mips[mip]),
            )),
        );
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloud_scheme_noise_mip"),
            timestamp_writes: None,
        });
        pass.set_pipeline(mip_compute);
        pass.set_bind_group(0, &mip_bind, &[]);
        let d = size.div_ceil(4);
        pass.dispatch_workgroups(d, d, d);
    }
    render_queue.submit([encoder.finish()]);
    *last = Some(key);
}

/// System to prepare the FBM noise bind group
pub fn prepare_fbm_noise_bind_group(
    mut commands: bevy_ecs::system::Commands,
    render_device: Res<RenderDevice>,
    layout: Res<FbmNoiseBindGroupLayout>,
    texture: Res<FbmNoiseTexture>,
    params_buffer: Res<FbmNoiseParamsBuffer>,
) {
    let bind_group = render_device.create_bind_group(
        "fbm_noise_bind_group",
        &layout.layout,
        &BindGroupEntries::with_indices((
            (13, &texture.texture.default_view),
            (14, params_buffer.buffer.as_entire_binding()),
            (15, &texture.cirrus_mips[0]),
        )),
    );
    let cirrus_bind_groups = texture
        .cirrus_mips
        .iter()
        .map(|view| {
            render_device.create_bind_group(
                "cirrus_field_bind_group",
                &layout.layout,
                &BindGroupEntries::with_indices((
                    (13, &texture.texture.default_view),
                    (14, params_buffer.buffer.as_entire_binding()),
                    (15, view),
                )),
            )
        })
        .collect();

    commands.insert_resource(FbmNoiseBindGroup {
        bind_group,
        cirrus_bind_groups,
    });
}

/// System to initialize the FBM noise parameters buffer
pub fn init_fbm_noise_params_buffer(
    mut commands: bevy_ecs::system::Commands,
    render_device: Res<RenderDevice>,
) {
    let params = FbmNoiseParams::default();
    let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("fbm_noise_params_buffer"),
        contents: bytemuck::cast_slice(&[params]),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    commands.insert_resource(FbmNoiseParamsBuffer { buffer });
}

/// Returns the dispatch workgroup counts for the 2D noise texture based on its size
pub fn get_noise_dispatch_size(texture_size: UVec2) -> UVec2 {
    const WORKGROUP_SIZE: u32 = 8;
    UVec2::new(
        texture_size.x.div_ceil(WORKGROUP_SIZE),
        texture_size.y.div_ceil(WORKGROUP_SIZE),
    )
}

/// Resource to track if noise has been generated
#[derive(Resource, Default)]
pub struct NoiseGenerated(pub bool);

/// System to generate the weather map: once, and again whenever the cloud life-cycle clock
/// ([`crate::VolumetricCloudLayer::evolution_time`]) advances (the map holds the blended generations).
///
/// There is one weather map for all views, so one life cycle drives it: that of the cumulus
/// layer with a life cycle on the camera with the lowest order (then the lowest entity), so the
/// choice does not depend on query order and a cirrus-only or static layer does not hold it
/// (audit 38). Turning every life cycle off re-bakes the static map.
pub fn generate_fbm_noise_once(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<FbmNoisePipeline>,
    bind_group: Option<Res<FbmNoiseBindGroup>>,
    texture: Res<FbmNoiseTexture>,
    params_buffer: Option<Res<FbmNoiseParamsBuffer>>,
    layers: Query<(
        &crate::GpuCloudLayers,
        &bevy_render::view::ExtractedView,
        Option<&bevy_render::camera::ExtractedCamera>,
    )>,
    mut last_life_t: Local<Option<f32>>,
    mut cirrus_done: Local<bool>,
    mut noise_generated: ResMut<NoiseGenerated>,
) {
    let life = layers
        .iter()
        .filter(|(layer, ..)| layer.cloud_density > 0.0 && layer.lifetime > 0.0)
        .min_by_key(|(_, view, camera)| {
            (camera.map_or(isize::MAX, |c| c.order), view.retained_view_entity.main_entity.id())
        })
        .map(|(layer, ..)| layer.evolution_time / layer.lifetime + 0.5);
    // Re-bake only when the phase moved by > 1/2000 of a cycle (a weight change far below
    // visibility; at 1x with a 20 min life time that is every ~0.6 s), or the life cycle was
    // turned on or off.
    let unchanged = match (life, *last_life_t) {
        (Some(t), Some(last)) => (t - last).abs() < 5e-4,
        (None, None) => true,
        _ => false,
    };
    if noise_generated.0 && unchanged && *cirrus_done {
        return;
    }
    if let Some(params_buffer) = params_buffer.as_ref() {
        let params = FbmNoiseParams {
            life_t: life.unwrap_or(0.5),
            life_enabled: life.is_some() as u32,
            ..FbmNoiseParams::default()
        };
        render_queue.write_buffer(&params_buffer.buffer, 0, bytemuck::cast_slice(&[params]));
    }

    let Some(bind_group) = bind_group else {
        return;
    };

    let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("fbm_noise_generation"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("fbm_noise_2d_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group.bind_group, &[]);

        let dispatch_size = get_noise_dispatch_size(texture.size);
        compute_pass.dispatch_workgroups(dispatch_size.x, dispatch_size.y, 1);

        // The cirrus generating field does not evolve: generate it once.
        if !*cirrus_done {
            if let Some(cirrus_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.cirrus_pipeline) {
                compute_pass.set_pipeline(cirrus_pipeline);
                for (mip, mip_bind_group) in bind_group.cirrus_bind_groups.iter().enumerate() {
                    compute_pass.set_bind_group(0, mip_bind_group, &[]);
                    let n = get_noise_dispatch_size(UVec2::splat((CIRRUS_TEXTURE_SIZE >> mip).max(1)));
                    compute_pass.dispatch_workgroups(n.x, n.y, 1);
                }
                *cirrus_done = true;
            }
        }
    }

    let command_buffer = encoder.finish();
    render_queue.submit([command_buffer]);

    noise_generated.0 = true;
    *last_life_t = life;
}
