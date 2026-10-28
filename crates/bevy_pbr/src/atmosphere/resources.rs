use crate::{
    detail_noise::DetailNoiseTexture, fbm_noise::FbmNoiseTexture,
    perlin_worley_noise::PerlinWorleyNoiseTexture, Bluenoise, ExtractedAtmosphere,
    ExtractedDirectionalLight, GpuCloudLayers, GpuLights, GpuScatteringMedium, LightMeta,
    LinearSampler, ScatteringMediumSampler, ShadowSamplers, ViewShadowBindings,
};
use bevy_asset::{load_embedded_asset, AssetId, Handle};
use bevy_camera::{Camera, Camera3d, MainPassResolutionOverride};
use bevy_core_pipeline::FullscreenShader;
use bevy_derive::Deref;
use bevy_diagnostic::FrameCount;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    error::BevyError,
    query::With,
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_image::ToExtents;
use bevy_light::atmosphere::ScatteringMedium;
use bevy_math::{Affine3A, Mat4, UVec2, Vec3, Vec3A, Vec4};
use bevy_platform::collections::HashMap;
use bevy_render::camera::ExtractedCamera;
use bevy_render::render_resource::UniformBuffer;
use bevy_render::view::RetainedViewEntity;
use bevy_render::{
    extract_component::ComponentUniforms,
    render_asset::RenderAssets,
    render_resource::{binding_types::*, TextureViewDescriptor, TextureViewDimension, *},
    renderer::{RenderDevice, RenderQueue},
    texture::{CachedTexture, FallbackImage, GpuImage},
    view::{ExtractedView, Msaa, ViewDepthStencilTexture, ViewUniform, ViewUniforms},
};
use bevy_shader::Shader;
use bevy_utils::default;

use super::GpuAtmosphereSettings;

/// Cloud SDF grid (must match `clouds.wesl` CLOUD_FIELD_RES_*).
pub(crate) const CLOUD_FIELD_RES_XY: u32 = 128;
/// Edge of the weather window around the camera, in weather-map texels (`weather_window.wesl`).
pub const WEATHER_WINDOW_SIZE: u32 = 1024;
pub(crate) const CLOUD_FIELD_RES_Z: u32 = 32;
/// Horizontal extent of the cloud field grid (m; `clouds.wesl` CLOUD_FIELD_EXTENT_M).
pub(crate) const CLOUD_FIELD_EXTENT_M: f32 = 24_000.0;

/// Shadow map bind group layout entries
fn shadow_map_layout_entries() -> impl IntoIndexedBindGroupLayoutEntryBuilderArray<4> {
    (
        (
            14,
            #[cfg(all(
                not(target_abi = "sim"),
                any(
                    not(feature = "webgl"),
                    not(target_arch = "wasm32"),
                    feature = "webgpu"
                )
            ))]
            texture_cube_array(TextureSampleType::Depth),
            #[cfg(any(
                target_abi = "sim",
                all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu"))
            ))]
            texture_cube(TextureSampleType::Depth),
        ),
        (15, sampler(SamplerBindingType::Comparison)),
        (
            16,
            #[cfg(any(
                not(feature = "webgl"),
                not(target_arch = "wasm32"),
                feature = "webgpu"
            ))]
            texture_2d_array(TextureSampleType::Depth),
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            texture_2d(TextureSampleType::Depth),
        ),
        (17, sampler(SamplerBindingType::Comparison)),
    )
}

/// The directional half of [`shadow_map_layout_entries`]: `render_sky` reads no point-light
/// shadows, and the slot keeps its passes within 16 sampled textures.
fn directional_shadow_layout_entries() -> impl IntoIndexedBindGroupLayoutEntryBuilderArray<2> {
    (
        (
            16,
            #[cfg(any(
                not(feature = "webgl"),
                not(target_arch = "wasm32"),
                feature = "webgpu"
            ))]
            texture_2d_array(TextureSampleType::Depth),
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            texture_2d(TextureSampleType::Depth),
        ),
        (17, sampler(SamplerBindingType::Comparison)),
    )
}

/// Cloud bind group layout entries (cloud layer uniform, noise textures, cloud shadow map, STBN).
/// Sampled outputs of the `clouds_raymarch` pass, read by `render_sky` (26: luminance and
/// transmittance, 27: transmittance-weighted cloud depth).
fn cloud_pass_output_entries() -> impl IntoIndexedBindGroupLayoutEntryBuilderArray<2> {
    (
        (
            26,
            texture_2d(TextureSampleType::Float { filterable: true }),
        ),
        (
            27,
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    )
}

/// Half-res haze from the `haze` prepass (render_sky.wesl): in-scattering, transmittance (the
/// prepass reads the previous frame's as its history), and their reprojection parameters.
fn haze_entries() -> impl IntoIndexedBindGroupLayoutEntryBuilderArray<3> {
    (
        (
            50,
            texture_2d(TextureSampleType::Float { filterable: true }),
        ),
        (
            51,
            texture_2d(TextureSampleType::Float { filterable: true }),
        ),
        (52, uniform_buffer::<CloudsTemporalParams>(false)),
    )
}

/// The cloud entries the full-resolution `render_sky` pass reads when the atmosphere march runs
/// in the half-res haze prepass: it only upsamples the haze and composites the clouds.
fn render_sky_composite_cloud_entries() -> impl IntoIndexedBindGroupLayoutEntryBuilderArray<3> {
    (
        (20, uniform_buffer::<GpuCloudLayers>(true)),
        (22, sampler(SamplerBindingType::Filtering)),
        (
            23,
            texture_2d(TextureSampleType::Float { filterable: true }),
        ),
    )
}

/// Placed at 20..=25, 33 so they don't collide with the shadow map entries (14..=17)
/// or the clouds_reconstruct history/storage bindings (28..=32).
fn cloud_layout_entries() -> impl IntoIndexedBindGroupLayoutEntryBuilderArray<11> {
    (
        (20, uniform_buffer::<GpuCloudLayers>(true)), // cloud layer parameters
        // 2D noise texture for cloud coverage / weather
        (
            21,
            texture_2d(TextureSampleType::Float { filterable: true }),
        ),
        (22, sampler(SamplerBindingType::Filtering)), // noise sampler
        // cloud shadow map
        (
            23,
            texture_2d(TextureSampleType::Float { filterable: true }),
        ),
        // 3D Perlin–Worley shape noise (128³ + mips)
        (
            24,
            texture_3d(TextureSampleType::Float { filterable: true }),
        ),
        // STBN for temporal stratification (different layer per frame)
        (
            25,
            texture_2d_array(TextureSampleType::Float { filterable: true }),
        ),
        // 3D Worley detail / erosion noise (32³ + mips)
        (
            33,
            texture_3d(TextureSampleType::Float { filterable: true }),
        ),
        // Vertical profile lookups, height x type (`cloud_profile.rs`).
        (
            34,
            texture_2d(TextureSampleType::Float { filterable: true }),
        ),
        // Physical cloud scheme fluctuation (g, g^2, b, b^2), shape tile + mips (`fbm_noise.rs`).
        (
            35,
            texture_3d(TextureSampleType::Float { filterable: true }),
        ),
        // Weather window around the camera: art (36) and physical (37) channels (`weather_window.wesl`).
        (
            36,
            texture_2d(TextureSampleType::Float { filterable: true }),
        ),
        (
            37,
            texture_2d(TextureSampleType::Float { filterable: true }),
        ),
    )
}

#[derive(Resource)]
pub(crate) struct AtmosphereBindGroupLayouts {
    pub transmittance_lut: BindGroupLayoutDescriptor,
    pub multiscattering_lut: BindGroupLayoutDescriptor,
    pub sky_view_lut: BindGroupLayoutDescriptor,
    pub aerial_view_lut: BindGroupLayoutDescriptor,
    pub sky_view_lut_clouds: BindGroupLayoutDescriptor,
    pub aerial_view_lut_clouds: BindGroupLayoutDescriptor,
    pub cloud_shadow_map: BindGroupLayoutDescriptor,
    // Accessed from the render-world graph; some tooling can miss the cross-module usage.
    #[allow(dead_code)]
    pub cloud_shadow_filter: BindGroupLayoutDescriptor,
    pub cloud_shadow_temporal: BindGroupLayoutDescriptor,
    pub clouds_raymarch: BindGroupLayoutDescriptor,
    pub clouds_raymarch_msaa: BindGroupLayoutDescriptor,
    pub clouds_reconstruct: BindGroupLayoutDescriptor,
    /// Cloud SDF separable distance-transform passes (`cloud_field.wesl`).
    pub cloud_field_edt: BindGroupLayoutDescriptor,
    pub cloud_field_final: BindGroupLayoutDescriptor,
    /// Escape-probability pass (`cloud_field.wesl` escape).
    pub cloud_field_escape: BindGroupLayoutDescriptor,
    /// Weather window compose pass (`weather_window.wesl`).
    pub weather_window: BindGroupLayoutDescriptor,
    /// One level of the cloud shadow map's mip chain from the level above (`cloud_shadow_mips.wesl`).
    pub cloud_shadow_mips: BindGroupLayoutDescriptor,
    /// Diffusion solve of the cloud field (`cloud_field.wesl` diffusion_*, sor_*).
    pub cloud_field_diffusion: BindGroupLayoutDescriptor,
}

#[derive(Resource)]
pub(crate) struct RenderSkyBindGroupLayouts {
    // Two variants:
    // - *_clouds: includes VolumetricCloudLayer + cloud noise + cloud shadow map bindings
    // - *_no_clouds: omits cloud-related bindings (cheaper, and works when VolumetricCloudLayer is removed)
    pub render_sky_clouds: BindGroupLayoutDescriptor,
    /// The half-res haze prepass (`haze` entry point) of views with clouds.
    pub render_sky_haze: BindGroupLayoutDescriptor,
    pub render_sky_msaa_clouds: BindGroupLayoutDescriptor,
    pub render_sky_no_clouds: BindGroupLayoutDescriptor,
    pub render_sky_msaa_no_clouds: BindGroupLayoutDescriptor,
    pub fullscreen_shader: FullscreenShader,
    pub fragment_shader: Handle<Shader>,
}

impl AtmosphereBindGroupLayouts {
    pub fn new() -> Self {
        let transmittance_lut = BindGroupLayoutDescriptor::new(
            "transmittance_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuAtmosphere>(true)),
                    (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                    // scattering medium luts and sampler
                    (5, texture_2d(TextureSampleType::default())),
                    (6, texture_2d(TextureSampleType::default())),
                    (7, sampler(SamplerBindingType::Filtering)),
                    // transmittance lut storage texture
                    (
                        13,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let multiscattering_lut = BindGroupLayoutDescriptor::new(
            "multiscattering_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuAtmosphere>(true)),
                    (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                    // scattering medium luts and sampler
                    (5, texture_2d(TextureSampleType::default())),
                    (6, texture_2d(TextureSampleType::default())),
                    (7, sampler(SamplerBindingType::Filtering)),
                    // atmosphere luts and sampler
                    (8, texture_2d(TextureSampleType::default())), // transmittance
                    (12, sampler(SamplerBindingType::Filtering)),
                    // multiscattering lut storage texture
                    (
                        13,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let sky_view_lut_entries = || {
            DynamicBindGroupLayoutEntries::new_with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuAtmosphere>(true)),
                    (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                    (2, uniform_buffer::<AtmosphereTransform>(true)),
                    (3, uniform_buffer::<ViewUniform>(true)),
                    (4, uniform_buffer::<GpuLights>(true)),
                    // scattering medium luts and sampler
                    (5, texture_2d(TextureSampleType::default())),
                    (6, texture_2d(TextureSampleType::default())),
                    (7, sampler(SamplerBindingType::Filtering)),
                    // atmosphere luts and sampler
                    (8, texture_2d(TextureSampleType::default())), // transmittance
                    (9, texture_2d(TextureSampleType::default())), // multiscattering
                    (12, sampler(SamplerBindingType::Filtering)),
                    // sky view lut storage texture
                    (
                        13,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            )
            .extend_with_indices(shadow_map_layout_entries())
        };
        // Cloud-shadowed variants (Hillaire 2020 / UE5: the volumetric cloud shadow map darkens
        // the sky-view and aerial-perspective LUT in-scattering).
        let sky_view_lut_clouds = BindGroupLayoutDescriptor::new(
            "sky_view_lut_clouds_bind_group_layout",
            &sky_view_lut_entries().extend_with_indices(cloud_layout_entries()),
        );
        let sky_view_lut = BindGroupLayoutDescriptor::new(
            "sky_view_lut_bind_group_layout",
            &sky_view_lut_entries(),
        );

        let aerial_view_lut_entries = || {
            DynamicBindGroupLayoutEntries::new_with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuAtmosphere>(true)),
                    (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                    (3, uniform_buffer::<ViewUniform>(true)),
                    (4, uniform_buffer::<GpuLights>(true)),
                    // scattering medium luts and sampler
                    (5, texture_2d(TextureSampleType::default())),
                    (6, texture_2d(TextureSampleType::default())),
                    (7, sampler(SamplerBindingType::Filtering)),
                    // atmosphere luts and sampler
                    (8, texture_2d(TextureSampleType::default())), // transmittance
                    (9, texture_2d(TextureSampleType::default())), // multiscattering
                    (12, sampler(SamplerBindingType::Filtering)),
                    // aerial view lut storage texture
                    (
                        13,
                        texture_storage_3d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            )
            .extend_with_indices(shadow_map_layout_entries())
        };
        let aerial_view_lut_clouds = BindGroupLayoutDescriptor::new(
            "aerial_view_lut_clouds_bind_group_layout",
            &aerial_view_lut_entries().extend_with_indices(cloud_layout_entries()),
        );
        let aerial_view_lut = BindGroupLayoutDescriptor::new(
            "aerial_view_lut_bind_group_layout",
            &aerial_view_lut_entries(),
        );

        let cloud_shadow_map = BindGroupLayoutDescriptor::new(
            "cloud_shadow_map_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuAtmosphere>(true)),
                    (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                    (3, uniform_buffer::<ViewUniform>(true)),
                    (4, uniform_buffer::<GpuLights>(true)),
                    (20, uniform_buffer::<GpuCloudLayers>(true)),
                    (
                        21,
                        texture_2d(TextureSampleType::Float { filterable: true }),
                    ),
                    (22, sampler(SamplerBindingType::Filtering)),
                    // 3D Perlin–Worley shape noise
                    (
                        24,
                        texture_3d(TextureSampleType::Float { filterable: true }),
                    ),
                    // STBN for temporal stratification (different layer per frame)
                    (
                        25,
                        texture_2d_array(TextureSampleType::Float { filterable: true }),
                    ),
                    // detail Worley
                    (
                        33,
                        texture_3d(TextureSampleType::Float { filterable: true }),
                    ),
                    // Vertical profile lookups (`cloud_profile.rs`).
                    (
                        34,
                        texture_2d(TextureSampleType::Float { filterable: true }),
                    ),
                    // Physical cloud scheme fluctuation (g, g^2, b, b^2), shape tile + mips (`fbm_noise.rs`).
                    (
                        35,
                        texture_3d(TextureSampleType::Float { filterable: true }),
                    ),
                    // Weather window around the camera: art (36) and physical (37) channels (`weather_window.wesl`).
                    (
                        36,
                        texture_2d(TextureSampleType::Float { filterable: true }),
                    ),
                    (
                        37,
                        texture_2d(TextureSampleType::Float { filterable: true }),
                    ),
                    // Cloud SDF seeds (`cloud_field.wesl` init entry point).
                    (
                        40,
                        texture_storage_3d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    // Cloud field extinction (`cloud_field.wesl` init), read by `escape`.
                    (
                        45,
                        texture_storage_3d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    // Cirrus generating field (thin-layer shadow in the map's alpha).
                    (
                        49,
                        texture_2d(TextureSampleType::Float { filterable: true }),
                    ),
                    (
                        13,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        // Cloud shadow spatial filter (ping-pong).
        // Reads: sampled shadow map (rgba16f)
        // Writes: storage shadow map (rgba16f)
        let cloud_shadow_filter = BindGroupLayoutDescriptor::new(
            "cloud_shadow_filter_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                    (12, sampler(SamplerBindingType::Filtering)),
                    (
                        23,
                        texture_2d(TextureSampleType::Float { filterable: true }),
                    ),
                    (
                        13,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        // Cloud shadow temporal filter.
        // Reads: traced output (storage), history (sampled)
        // Writes: output (storage)
        let cloud_shadow_temporal = BindGroupLayoutDescriptor::new(
            "cloud_shadow_temporal_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<CloudShadowTemporalParams>(false)),
                    (
                        1,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::ReadOnly,
                        ),
                    ),
                    (2, texture_2d(TextureSampleType::Float { filterable: true })),
                    (3, sampler(SamplerBindingType::Filtering)),
                    (
                        4,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        // Dedicated low-res cloud raymarch (`clouds_raymarch.wesl`).
        // Same bindings as `render_sky` (clouds variant, non-MSAA depth, no shadow maps) plus the
        // storage outputs at 26 (rgba16f luminance/transmittance) and 27 (r32f weighted depth).
        let clouds_raymarch_layout = |label: &'static str, depth: BindGroupLayoutEntryBuilder| {
            BindGroupLayoutDescriptor::new(
                label,
                &DynamicBindGroupLayoutEntries::new_with_indices(
                    ShaderStages::COMPUTE,
                    (
                        (0, uniform_buffer::<GpuAtmosphere>(true)),
                        (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                        (2, uniform_buffer::<AtmosphereTransform>(true)),
                        (3, uniform_buffer::<ViewUniform>(true)),
                        (4, uniform_buffer::<GpuLights>(true)),
                        (5, texture_2d(TextureSampleType::default())),
                        (6, texture_2d(TextureSampleType::default())),
                        (7, sampler(SamplerBindingType::Filtering)),
                        (8, texture_2d(TextureSampleType::default())),
                        (9, texture_2d(TextureSampleType::default())),
                        (10, texture_2d(TextureSampleType::default())),
                        (11, texture_3d(TextureSampleType::default())),
                        (12, sampler(SamplerBindingType::Filtering)),
                        (13, depth),
                    ),
                )
                .extend_with_indices(cloud_layout_entries())
                .extend_with_indices((
                    (
                        26,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    (
                        27,
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    // Cloud signed-distance field (`cloud_field.wesl`).
                    (
                        44,
                        texture_3d(TextureSampleType::Float { filterable: true }),
                    ),
                    // Escape probability of multiply scattered light (`cloud_field.wesl` escape).
                    (
                        48,
                        texture_3d(TextureSampleType::Float { filterable: true }),
                    ),
                    // Cirrus generating field (`fbm_noise_3d.wesl` cirrus).
                    (
                        49,
                        texture_2d(TextureSampleType::Float { filterable: true }),
                    ),
                    // Diffuse sunlight of the cloud field (`cloud_field.wesl` diffusion solve).
                    (
                        59,
                        texture_3d(TextureSampleType::Float { filterable: true }),
                    ),
                )),
            )
        };
        let clouds_raymarch = clouds_raymarch_layout(
            "clouds_raymarch_bind_group_layout",
            texture_2d(TextureSampleType::Depth),
        );
        // MSAA views: the scene depth is multisampled (the trace takes the farthest sample).
        let clouds_raymarch_msaa = clouds_raymarch_layout(
            "clouds_raymarch_msaa_bind_group_layout",
            texture_2d_multisampled(TextureSampleType::Depth),
        );

        // Checkerboard temporal reconstruction (`clouds_reconstruct.wesl`).
        let clouds_reconstruct = BindGroupLayoutDescriptor::new(
            "clouds_reconstruct_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuAtmosphere>(true)),
                    (3, uniform_buffer::<ViewUniform>(true)),
                    (12, sampler(SamplerBindingType::Filtering)),
                    (
                        26,
                        texture_2d(TextureSampleType::Float { filterable: true }),
                    ),
                    (
                        27,
                        texture_2d(TextureSampleType::Float { filterable: false }),
                    ),
                    (
                        28,
                        texture_2d(TextureSampleType::Float { filterable: true }),
                    ),
                    (
                        29,
                        texture_2d(TextureSampleType::Float { filterable: false }),
                    ),
                    (30, uniform_buffer::<CloudsTemporalParams>(false)),
                    (
                        31,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    (
                        32,
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    // CLOUD_REFERENCE progressive accumulator (f32 sums + sample counts).
                    (33, storage_buffer_sized(false, None)),
                ),
            ),
        );

        Self {
            transmittance_lut,
            multiscattering_lut,
            sky_view_lut,
            aerial_view_lut,
            sky_view_lut_clouds,
            aerial_view_lut_clouds,
            cloud_shadow_map,
            cloud_shadow_filter,
            cloud_shadow_temporal,
            clouds_raymarch,
            clouds_raymarch_msaa,
            clouds_reconstruct,
            cloud_field_edt: BindGroupLayoutDescriptor::new(
                "cloud_field_edt_bind_group_layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::COMPUTE,
                    (
                        (
                            41,
                            texture_3d(TextureSampleType::Float { filterable: false }),
                        ),
                        (
                            42,
                            texture_storage_3d(
                                TextureFormat::Rgba16Float,
                                StorageTextureAccess::WriteOnly,
                            ),
                        ),
                    ),
                ),
            ),
            cloud_field_final: BindGroupLayoutDescriptor::new(
                "cloud_field_final_bind_group_layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::COMPUTE,
                    (
                        (
                            41,
                            texture_3d(TextureSampleType::Float { filterable: false }),
                        ),
                        (
                            43,
                            texture_storage_3d(
                                TextureFormat::Rgba16Float,
                                StorageTextureAccess::WriteOnly,
                            ),
                        ),
                    ),
                ),
            ),
            cloud_field_escape: BindGroupLayoutDescriptor::new(
                "cloud_field_escape_bind_group_layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::COMPUTE,
                    (
                        (
                            47,
                            texture_3d(TextureSampleType::Float { filterable: false }),
                        ),
                        (
                            46,
                            texture_storage_3d(
                                TextureFormat::Rgba16Float,
                                StorageTextureAccess::WriteOnly,
                            ),
                        ),
                    ),
                ),
            ),
            cloud_field_diffusion: BindGroupLayoutDescriptor::new(
                "cloud_field_diffusion_bind_group_layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::COMPUTE,
                    (
                        (0, uniform_buffer::<GpuAtmosphere>(true)),
                        (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                        (3, uniform_buffer::<ViewUniform>(true)),
                        (4, uniform_buffer::<GpuLights>(true)),
                        (20, uniform_buffer::<GpuCloudLayers>(true)),
                        (
                            23,
                            texture_2d(TextureSampleType::Float { filterable: true }),
                        ),
                        (
                            44,
                            texture_3d(TextureSampleType::Float { filterable: false }),
                        ),
                        (
                            47,
                            texture_3d(TextureSampleType::Float { filterable: false }),
                        ),
                        (
                            55,
                            texture_storage_3d(
                                TextureFormat::R32Float,
                                StorageTextureAccess::ReadWrite,
                            ),
                        ),
                        (
                            56,
                            texture_storage_3d(
                                TextureFormat::R32Float,
                                StorageTextureAccess::ReadWrite,
                            ),
                        ),
                        (
                            57,
                            texture_storage_3d(
                                TextureFormat::Rgba16Float,
                                StorageTextureAccess::WriteOnly,
                            ),
                        ),
                        (
                            58,
                            texture_storage_3d(
                                TextureFormat::R32Float,
                                StorageTextureAccess::ReadWrite,
                            ),
                        ),
                        (
                            60,
                            texture_storage_3d(
                                TextureFormat::R32Float,
                                StorageTextureAccess::ReadWrite,
                            ),
                        ),
                    ),
                ),
            ),
            cloud_shadow_mips: BindGroupLayoutDescriptor::new(
                "cloud_shadow_mips_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        texture_2d(TextureSampleType::Float { filterable: false }),
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
            weather_window: BindGroupLayoutDescriptor::new(
                "weather_window_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        texture_2d(TextureSampleType::Float { filterable: true }),
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                        uniform_buffer_sized(false, None),
                    ),
                ),
            ),
        }
    }
}

/// Exponential history weight of a freshly traced cloud pixel (each pixel is traced every 4th
/// frame). Hillaire 2016 (Frostbite) blends 5 % per frame, HZD (Schneider 2015) reprojects over
/// 16 frames; 0.5 averaged almost nothing, so far/horizon clouds never converged. Ghosting is
/// bounded by variance clipping in `clouds_reconstruct.wesl`.
const CLOUD_RECON_NEW_SAMPLE_WEIGHT: f32 = 0.1;

/// Reprojection parameters for the checkerboarded cloud reconstruction pass.
#[derive(Clone, Copy, ShaderType)]
pub struct CloudsTemporalParams {
    pub prev_clip_from_world: Mat4,
    /// Horizontal displacement of the cloud field since the previous frame (atmosphere space, m):
    /// a cloud point at `p` was at `p - cloud_motion` (wind advection).
    pub cloud_motion: Vec3,
    pub history_valid: u32,
    /// Horizontal displacement of the thin (cirrus) layer since the previous frame: it has its
    /// own wind, so its pixels reproject with this instead of `cloud_motion`.
    pub cirrus_motion: Vec3,
    pub new_sample_weight: f32,
    /// Radius above which a pixel's representative cloud depth is taken as cirrus (midway
    /// between the cumulus tops and the cirrus base; 0 without cumulus).
    pub cirrus_min_radius: f32,
    /// Current / previous `view.exposure`: the cloud history stores exposed radiance, so it is
    /// rescaled before clamping and blending when the camera's exposure changes.
    pub history_exposure_scale: f32,
}

/// Per-view uniform for [`CloudsTemporalParams`].
#[derive(Component)]
pub struct CloudsTemporalParamsUniform(pub UniformBuffer<CloudsTemporalParams>);

/// Writes [`CloudsTemporalParams`] for every atmosphere view, remembering each view's
/// previous `clip_from_world` (keyed by retained view entity) for history reprojection.
///
/// The remembered transform includes the view's TAA jitter, as `prepare_view_uniforms` applies
/// it: the cloud and haze rays are generated from the jittered `view_from_clip`, so each history
/// texel holds a jittered sample, and reprojecting it with the unjittered previous projection
/// put it up to half a full-resolution pixel (a quarter of a half-resolution texel) off.
pub(super) fn prepare_clouds_temporal_params(
    views: Query<
        (
            Entity,
            &ExtractedView,
            &AtmosphereTextures,
            Option<&GpuCloudLayers>,
            Option<&bevy_render::camera::TemporalJitter>,
            Option<&ExtractedCamera>,
            Option<&MainPassResolutionOverride>,
        ),
        With<ExtractedAtmosphere>,
    >,
    mut previous: bevy_ecs::system::Local<HashMap<RetainedViewEntity, PreviousCloudFrame>>,
    writes: Res<CloudHistoryWrites>,
    frame_count: Res<FrameCount>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut commands: Commands,
) {
    let mut seen = HashMap::default();
    for (entity, view, textures, cloud_layer, temporal_jitter, camera, resolution_override) in
        &views
    {
        let offset = cloud_layer.map_or(Vec3::ZERO, |c| c.noise_offset);
        // Thin-layer pixels reproject with the lowest thin layer's motion.
        let cirrus_offset = cloud_layer
            .and_then(|c| c.lowest_thin())
            .map_or(Vec3::ZERO, |t| t.offset);
        let cirrus_min_radius = cloud_layer.map_or(0.0, |c| match c.lowest_thin() {
            Some(t) if c.cloud_density > 0.0 => 0.5 * (c.cloud_layer_end + t.mid_radius()),
            _ => 0.0,
        });
        let exposure = camera.map_or(1.0, |c| c.exposure);
        let clip_from_world = match temporal_jitter {
            Some(jitter) => {
                // The jitter is in render pixels, like the view uniform's (`prepare_view_uniforms`).
                let render_size = resolution_override
                    .map_or(UVec2::new(view.viewport.z, view.viewport.w), |size| **size);
                let mut clip_from_view = view.clip_from_view;
                jitter.jitter_projection(&mut clip_from_view, render_size.as_vec2());
                clip_from_view * view.world_from_view.to_matrix().inverse()
            }
            None => view.clip_from_world.unwrap_or_else(|| {
                view.clip_from_view * view.world_from_view.to_matrix().inverse()
            }),
        };
        // The history is what this view wrote last frame: the ping-pong textures it reads now
        // must be the ones `render_sky` actually wrote then (a new view, a reallocation, a resize
        // or a skipped frame breaks that).
        let history_written = writes.wrote_last_frame(
            view.retained_view_entity,
            frame_count.0,
            textures.clouds_history.texture.id(),
            textures.haze_history_inscattering.texture.id(),
        );
        let (prev_clip_from_world, history_valid, prev_offset, prev_cirrus_offset, prev_exposure) =
            match previous.get(&view.retained_view_entity) {
                Some(prev) if history_written => (
                    prev.clip_from_world,
                    1,
                    prev.offset,
                    prev.cirrus_offset,
                    prev.exposure,
                ),
                _ => (clip_from_world, 0, offset, cirrus_offset, exposure),
            };
        let motion = offset - prev_offset;
        let cirrus_motion = cirrus_offset - prev_cirrus_offset;
        let mut buffer = UniformBuffer::from(CloudsTemporalParams {
            prev_clip_from_world,
            cloud_motion: Vec3::new(motion.x, 0.0, motion.z),
            history_valid,
            cirrus_motion: Vec3::new(cirrus_motion.x, 0.0, cirrus_motion.z),
            new_sample_weight: CLOUD_RECON_NEW_SAMPLE_WEIGHT,
            cirrus_min_radius,
            history_exposure_scale: exposure / prev_exposure.max(f32::MIN_POSITIVE),
        });
        buffer.write_buffer(&render_device, &render_queue);
        commands
            .entity(entity)
            .insert(CloudsTemporalParamsUniform(buffer));
        seen.insert(
            view.retained_view_entity,
            PreviousCloudFrame {
                clip_from_world,
                offset,
                cirrus_offset,
                exposure,
            },
        );
    }
    *previous = seen;
}

/// What a view's cloud and haze passes wrote in the previous frame ([`prepare_clouds_temporal_params`]).
pub struct PreviousCloudFrame {
    clip_from_world: Mat4,
    offset: Vec3,
    cirrus_offset: Vec3,
    exposure: f32,
}

/// The cloud and haze targets each view's `render_sky` wrote, and in which frame: the next
/// frame's histories. Prepared targets are not necessarily written: `render_sky` skips views
/// whose pipelines or cloud field are not ready (audit 5, 20).
#[derive(Resource, Default)]
pub struct CloudHistoryWrites(
    HashMap<RetainedViewEntity, (u32, Option<TextureId>, Option<TextureId>)>,
);

impl CloudHistoryWrites {
    /// Records this frame's writes of a view: the cloud target if the reconstruction ran, the
    /// haze target if the haze prepass ran (or the view has none, so nothing reads its history).
    pub fn record(
        &mut self,
        view: RetainedViewEntity,
        frame: u32,
        clouds: Option<TextureId>,
        haze: Option<TextureId>,
    ) {
        self.0
            .retain(|_, (written, ..)| frame.wrapping_sub(*written) <= 1);
        self.0.insert(view, (frame, clouds, haze));
    }

    fn wrote_last_frame(
        &self,
        view: RetainedViewEntity,
        frame: u32,
        clouds: TextureId,
        haze: TextureId,
    ) -> bool {
        self.0.get(&view).is_some_and(|&(written, c, h)| {
            written == frame.wrapping_sub(1) && c == Some(clouds) && h == Some(haze)
        })
    }
}

/// Parameters for the cloud shadow temporal filter pass.
/// Used to reproject history from the previous frame.
#[derive(Clone, Default, ShaderType)]
pub struct CloudShadowTemporalParams {
    pub curr_anchor: Vec3,
    pub curr_light_dir: Vec3,
    pub curr_basis_x: Vec3,
    pub curr_basis_y: Vec3,
    pub prev_anchor: Vec3,
    pub prev_light_dir: Vec3,
    pub prev_basis_x: Vec3,
    pub prev_basis_y: Vec3,
    pub extent: f32,
    pub temporal_alpha: f32,
    pub history_valid: u32,
    pub anchor_moved: u32,
    pub size: UVec2,
    /// Time-slicing tile size (1 = every texel is traced each frame).
    pub interleave: u32,
    /// Frame index used to select the traced texel in each tile (matches `view.frame_count`).
    pub frame_index: u32,
    /// Horizontal displacement of the cloud field since the previous frame (atmosphere space, m).
    pub cloud_motion: Vec3,
    /// Horizontal displacement of the thin (cirrus) layer since the previous frame: transports
    /// the thin-layer optical depth in alpha.
    pub cirrus_motion: Vec3,
    /// |cos(light zenith)| / cloud layer thickness (1/m): a texel's history stays valid while
    /// the light turns by less than its size times this (`cloud_shadow_temporal.wesl`).
    pub history_angle_scale: f32,
    /// Angle the light turned (rad) over the last k + 1 frames, k = 0..16 (component k % 4 of
    /// element k / 4): what a texel traced k + 1 frames ago has missed.
    pub rotation_since: [Vec4; 4],
}

/// Per-view state for cloud shadow temporal filtering.
/// Persists between frames to enable history reprojection.
#[derive(Default, Clone)]
pub struct CloudShadowTemporalState {
    pub anchor: Vec3,
    pub light_dir: Vec3,
    pub basis_x: Vec3,
    pub basis_y: Vec3,
    pub initialized: bool,
    /// Cloud field displacement ([`crate::VolumetricCloudLayer::noise_offset`]) when the map was last filtered.
    pub cloud_offset: Vec3,
    /// Thin-layer displacement ([`crate::ThinCloudLayer::offset`]) when the map was last filtered.
    pub cirrus_offset: Vec3,
    /// Map size and extent the history was written with: a resized map is a new, cleared
    /// texture, and another extent maps its texels to other places.
    pub size: UVec2,
    pub extent: f32,
    /// The history texture the map was copied into.
    pub history_texture: Option<TextureId>,
    /// Angle the light turned (rad) in each of the last 16 frames, the latest first.
    pub rotations: [f32; 16],
}

impl FromWorld for RenderSkyBindGroupLayouts {
    fn from_world(world: &mut World) -> Self {
        // Base render-sky entries (0..=13) + directional shadow map entries (16, 17).
        let base_entries = |depth: BindGroupLayoutEntryBuilder| {
            DynamicBindGroupLayoutEntries::new_with_indices(
                ShaderStages::FRAGMENT,
                (
                    (0, uniform_buffer::<GpuAtmosphere>(true)),
                    (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                    (2, uniform_buffer::<AtmosphereTransform>(true)),
                    (3, uniform_buffer::<ViewUniform>(true)),
                    (4, uniform_buffer::<GpuLights>(true)),
                    // scattering medium luts and sampler
                    (5, texture_2d(TextureSampleType::default())),
                    (6, texture_2d(TextureSampleType::default())),
                    (7, sampler(SamplerBindingType::Filtering)),
                    // atmosphere luts and sampler
                    (8, texture_2d(TextureSampleType::default())), // transmittance
                    (9, texture_2d(TextureSampleType::default())), // multiscattering
                    (10, texture_2d(TextureSampleType::default())), // sky view
                    (11, texture_3d(TextureSampleType::default())), // aerial view
                    (12, sampler(SamplerBindingType::Filtering)),
                    // view depth texture
                    (13, depth),
                ),
            )
            .extend_with_indices(directional_shadow_layout_entries())
        };

        // Two variants:
        // - *_clouds: base + VolumetricCloudLayer + cloud noise + cloud shadow map + STBN (20..=25)
        // - *_no_clouds: base only
        let render_sky_no_clouds = BindGroupLayoutDescriptor::new(
            "render_sky_bind_group_layout_no_clouds",
            &base_entries(texture_2d(TextureSampleType::Depth)),
        );
        let render_sky_msaa_no_clouds = BindGroupLayoutDescriptor::new(
            "render_sky_msaa_bind_group_layout_no_clouds",
            &base_entries(texture_2d_multisampled(TextureSampleType::Depth)),
        );
        // Each pass binds only what it reads: WebGPU allows 16 sampled textures per shader stage,
        // and one layout for the prepass and the composite declared 20.
        // - haze prepass: base (8 textures) + cloud inputs (6) + haze history (2) = 16
        // - composite with half-res haze: base + cloud shadow map + clouds (2) + haze (2) = 13
        // - MSAA composite (full-res march, no haze prepass): base + cloud inputs + clouds = 16
        let render_sky_clouds = BindGroupLayoutDescriptor::new(
            "render_sky_bind_group_layout_clouds",
            &base_entries(texture_2d(TextureSampleType::Depth))
                .extend_with_indices(render_sky_composite_cloud_entries())
                .extend_with_indices(cloud_pass_output_entries())
                .extend_with_indices(haze_entries()),
        );
        let render_sky_haze = BindGroupLayoutDescriptor::new(
            "render_sky_haze_bind_group_layout",
            &base_entries(texture_2d(TextureSampleType::Depth))
                .extend_with_indices(cloud_layout_entries())
                .extend_with_indices(haze_entries()),
        );
        let render_sky_msaa_clouds = BindGroupLayoutDescriptor::new(
            "render_sky_msaa_bind_group_layout_clouds",
            &base_entries(texture_2d_multisampled(TextureSampleType::Depth))
                .extend_with_indices(cloud_layout_entries())
                .extend_with_indices(cloud_pass_output_entries()),
        );

        Self {
            render_sky_clouds,
            render_sky_haze,
            render_sky_msaa_clouds,
            render_sky_no_clouds,
            render_sky_msaa_no_clouds,
            fullscreen_shader: world.resource::<FullscreenShader>().clone(),
            fragment_shader: load_embedded_asset!(world, "render_sky.wesl"),
        }
    }
}

#[derive(Resource, Deref)]
pub struct CloudNoiseSampler(Sampler);

impl FromWorld for CloudNoiseSampler {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let sampler = render_device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self(sampler)
    }
}

#[derive(Resource)]
pub(crate) struct AtmosphereLutPipelines {
    pub transmittance_lut: CachedComputePipelineId,
    pub multiscattering_lut: CachedComputePipelineId,
    pub sky_view_lut: CachedComputePipelineId,
    pub aerial_view_lut: CachedComputePipelineId,
    pub sky_view_lut_clouds: CachedComputePipelineId,
    pub aerial_view_lut_clouds: CachedComputePipelineId,
    pub cloud_shadow_map: CachedComputePipelineId,
    // Accessed from the render-world graph; some tooling can miss the cross-module usage.
    #[allow(dead_code)]
    pub cloud_shadow_filter: CachedComputePipelineId,
    pub cloud_shadow_temporal: CachedComputePipelineId,
    pub clouds_raymarch: CachedComputePipelineId,
    /// `clouds_raymarch` for MSAA views (multisampled scene depth).
    pub clouds_raymarch_msaa: CachedComputePipelineId,
    pub clouds_reconstruct: CachedComputePipelineId,
    pub cloud_field_init: CachedComputePipelineId,
    pub cloud_field_edt_x: CachedComputePipelineId,
    pub cloud_field_edt_y: CachedComputePipelineId,
    pub cloud_field_edt_z: CachedComputePipelineId,
    pub cloud_field_escape: CachedComputePipelineId,
    pub weather_window: CachedComputePipelineId,
    pub cloud_shadow_mips: CachedComputePipelineId,
    /// Diffusion solve: source, red and black SOR sweeps, publish, reset (`cloud_field.wesl`).
    pub cloud_field_diffusion: [CachedComputePipelineId; 8],
}

impl FromWorld for AtmosphereLutPipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layouts = world.resource::<AtmosphereBindGroupLayouts>();

        let transmittance_lut = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("transmittance_lut_pipeline".into()),
            layout: vec![layouts.transmittance_lut.clone()],
            shader: load_embedded_asset!(world, "transmittance_lut.wesl"),
            shader_defs: atm_dbg_defs(),
            ..default()
        });

        let multiscattering_lut =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("multi_scattering_lut_pipeline".into()),
                layout: vec![layouts.multiscattering_lut.clone()],
                shader: load_embedded_asset!(world, "multiscattering_lut.wesl"),
                shader_defs: atm_dbg_defs(),
                ..default()
            });

        let sky_view_lut = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("sky_view_lut_pipeline".into()),
            layout: vec![layouts.sky_view_lut.clone()],
            shader: load_embedded_asset!(world, "sky_view_lut.wesl"),
            shader_defs: [atm_dbg_defs(), vec!["SKY_VIEW_LUT_PASS".into()]].concat(),
            // Note: Clouds disabled for sky_view_lut - too complex for precomputation
            // Clouds are rendered in real-time raymarching instead
            ..default()
        });

        let aerial_view_lut = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("aerial_view_lut_pipeline".into()),
            layout: vec![layouts.aerial_view_lut.clone()],
            shader: load_embedded_asset!(world, "aerial_view_lut.wesl"),
            shader_defs: atm_dbg_defs(),
            ..default()
        });

        let lut_cloud_defs = || -> Vec<bevy_shader::ShaderDefVal> {
            vec![
                "CLOUDS_ENABLED".into(),
                "CLOUD_PASS".into(),
                "ATMOSPHERE_LUT_CLOUD_SHADOW".into(),
            ]
        };
        let sky_view_lut_clouds =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("sky_view_lut_clouds_pipeline".into()),
                layout: vec![layouts.sky_view_lut_clouds.clone()],
                shader: load_embedded_asset!(world, "sky_view_lut.wesl"),
                shader_defs: [
                    atm_dbg_defs(),
                    vec!["SKY_VIEW_LUT_PASS".into()],
                    lut_cloud_defs(),
                ]
                .concat(),
                ..default()
            });
        let aerial_view_lut_clouds =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("aerial_view_lut_clouds_pipeline".into()),
                layout: vec![layouts.aerial_view_lut_clouds.clone()],
                shader: load_embedded_asset!(world, "aerial_view_lut.wesl"),
                shader_defs: [atm_dbg_defs(), lut_cloud_defs()].concat(),
                ..default()
            });

        let cloud_shadow_map = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_shadow_map_pipeline".into()),
            layout: vec![layouts.cloud_shadow_map.clone()],
            shader: load_embedded_asset!(world, "cloud_shadow_map.wesl"),
            shader_defs: atm_dbg_defs(),
            ..default()
        });

        let cloud_shadow_filter =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("cloud_shadow_filter_pipeline".into()),
                layout: vec![layouts.cloud_shadow_filter.clone()],
                shader: load_embedded_asset!(world, "cloud_shadow_filter.wesl"),
                ..default()
            });

        let cloud_shadow_temporal =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("cloud_shadow_temporal_pipeline".into()),
                layout: vec![layouts.cloud_shadow_temporal.clone()],
                shader: load_embedded_asset!(world, "cloud_shadow_temporal.wesl"),
                shader_defs: atm_dbg_defs(),
                ..default()
            });

        let clouds_raymarch = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("clouds_raymarch_pipeline".into()),
            layout: vec![layouts.clouds_raymarch.clone()],
            shader: load_embedded_asset!(world, "clouds_raymarch.wesl"),
            shader_defs: vec![
                "CLOUDS_ENABLED".into(),
                "CLOUD_SHADOW_SIMPLE_SAMPLING".into(),
                "ATMOSPHERE_NO_SHADOW_MAPS".into(),
                "CLOUD_FIELD_BOUND".into(),
            ]
            .into_iter()
            .chain(cloud_reference_def())
            .chain(cloud_term_defs())
            .chain(atm_dbg_defs())
            .collect(),
            ..default()
        });
        let clouds_raymarch_msaa =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("clouds_raymarch_msaa_pipeline".into()),
                layout: vec![layouts.clouds_raymarch_msaa.clone()],
                shader: load_embedded_asset!(world, "clouds_raymarch.wesl"),
                shader_defs: vec![
                    "CLOUDS_ENABLED".into(),
                    "CLOUD_SHADOW_SIMPLE_SAMPLING".into(),
                    "ATMOSPHERE_NO_SHADOW_MAPS".into(),
                    "CLOUD_FIELD_BOUND".into(),
                    "MULTISAMPLED_DEPTH".into(),
                ]
                .into_iter()
                .chain(cloud_reference_def())
                .chain(cloud_term_defs())
                .chain(atm_dbg_defs())
                .collect(),
                ..default()
            });

        let clouds_reconstruct = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("clouds_reconstruct_pipeline".into()),
            layout: vec![layouts.clouds_reconstruct.clone()],
            shader: load_embedded_asset!(world, "clouds_reconstruct.wesl"),
            shader_defs: vec!["ATMOSPHERE_NO_SHADOW_MAPS".into()]
                .into_iter()
                .chain(cloud_reference_def())
                .collect(),
            ..default()
        });

        let field =
            |label: &'static str, layout: &BindGroupLayoutDescriptor, entry: &'static str| {
                pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some(label.into()),
                    layout: vec![layout.clone()],
                    shader: load_embedded_asset!(world, "cloud_field.wesl"),
                    entry_point: Some(entry.into()),
                    ..default()
                })
            };

        Self {
            transmittance_lut,
            multiscattering_lut,
            sky_view_lut,
            aerial_view_lut,
            sky_view_lut_clouds,
            aerial_view_lut_clouds,
            cloud_shadow_map,
            cloud_shadow_filter,
            cloud_shadow_temporal,
            clouds_raymarch,
            clouds_raymarch_msaa,
            clouds_reconstruct,
            cloud_field_init: field(
                "cloud_field_init_pipeline",
                &layouts.cloud_shadow_map,
                "init",
            ),
            cloud_field_edt_x: field(
                "cloud_field_edt_x_pipeline",
                &layouts.cloud_field_edt,
                "edt_x",
            ),
            cloud_field_edt_y: field(
                "cloud_field_edt_y_pipeline",
                &layouts.cloud_field_edt,
                "edt_y",
            ),
            cloud_field_edt_z: field(
                "cloud_field_edt_z_pipeline",
                &layouts.cloud_field_final,
                "edt_z",
            ),
            cloud_field_escape: field(
                "cloud_field_escape_pipeline",
                &layouts.cloud_field_escape,
                "escape",
            ),
            // The source reads the cloud shadow map (`functions.wesl` under CLOUDS_ENABLED).
            cloud_field_diffusion: [
                "diffusion_source",
                "sor_red",
                "sor_black",
                "diffusion_publish",
                "diffusion_reset",
                "diffusion_source_slice",
                "diffusion_shift",
                "diffusion_shift_commit",
            ]
            .map(|entry| {
                pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some(format!("cloud_{entry}_pipeline").into()),
                    layout: vec![layouts.cloud_field_diffusion.clone()],
                    shader: load_embedded_asset!(world, "cloud_field.wesl"),
                    shader_defs: [atm_dbg_defs(), vec!["CLOUDS_ENABLED".into()]].concat(),
                    entry_point: Some(entry.into()),
                    ..default()
                })
            }),
            cloud_shadow_mips: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("cloud_shadow_mips_pipeline".into()),
                layout: vec![layouts.cloud_shadow_mips.clone()],
                shader: load_embedded_asset!(world, "cloud_shadow_mips.wesl"),
                entry_point: Some("main".into()),
                ..default()
            }),
            weather_window: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("weather_window_pipeline".into()),
                layout: vec![layouts.weather_window.clone()],
                shader: load_embedded_asset!(world, "weather_window.wesl"),
                entry_point: Some("compose".into()),
                ..default()
            }),
        }
    }
}

#[derive(Component)]
pub(crate) struct RenderSkyPipelineId(pub CachedRenderPipelineId);

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub(crate) struct RenderSkyPipelineKey {
    pub msaa_samples: u32,
    pub dual_source_blending: bool,
    pub clouds_enabled: bool,
    /// The half-resolution haze prepass (`haze` entry point) instead of the full-res pass.
    pub haze_prepass: bool,
}

impl SpecializedRenderPipeline for RenderSkyBindGroupLayouts {
    type Key = RenderSkyPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();

        if key.msaa_samples > 1 {
            shader_defs.push("MULTISAMPLED".into());
        }
        if key.dual_source_blending {
            shader_defs.push("DUAL_SOURCE_BLENDING".into());
        }

        // Enable cloud rendering only when a VolumetricCloudLayer component is present.
        if key.clouds_enabled {
            shader_defs.push("CLOUDS_ENABLED".into());
            shader_defs.push("CLOUD_SHADOW_SIMPLE_SAMPLING".into());
            // Clouds come from the dedicated `clouds_raymarch` pass, not `raymarch_atmosphere`.
            shader_defs.push("CLOUD_PASS".into());
            shader_defs.extend(cloud_reference_def());
            // The atmosphere march at half resolution (prepass) + depth-aware upsample.
            if key.msaa_samples == 1 {
                shader_defs.push("SKY_HAZE_HALF_RES".into());
            }
        } else {
            shader_defs.extend(atm_dbg_defs());
        }

        let dst_factor = if key.dual_source_blending {
            BlendFactor::Src1
        } else {
            BlendFactor::SrcAlpha
        };

        if key.haze_prepass {
            let target = Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            });
            return RenderPipelineDescriptor {
                label: Some("render_sky_haze_pipeline".into()),
                layout: vec![self.render_sky_haze.clone()],
                vertex: self.fullscreen_shader.to_vertex_state(),
                fragment: Some(FragmentState {
                    shader: self.fragment_shader.clone(),
                    shader_defs,
                    entry_point: Some("haze".into()),
                    targets: vec![target.clone(), target],
                    ..default()
                }),
                ..default()
            };
        }

        RenderPipelineDescriptor {
            label: Some(format!("render_sky_pipeline_{}", key.msaa_samples).into()),
            layout: vec![match (key.msaa_samples == 1, key.clouds_enabled) {
                (true, true) => self.render_sky_clouds.clone(),
                (false, true) => self.render_sky_msaa_clouds.clone(),
                (true, false) => self.render_sky_no_clouds.clone(),
                (false, false) => self.render_sky_msaa_no_clouds.clone(),
            }],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::Zero,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
                entry_point: Some("main".into()),
                ..default()
            }),
            multisample: MultisampleState {
                count: key.msaa_samples,
                ..default()
            },
            ..default()
        }
    }
}

pub(super) fn queue_render_sky_pipelines(
    views: Query<
        (Entity, &Msaa, Option<&GpuCloudLayers>),
        (With<Camera>, With<ExtractedAtmosphere>),
    >,
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<RenderSkyBindGroupLayouts>,
    mut specializer: ResMut<SpecializedRenderPipelines<RenderSkyBindGroupLayouts>>,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    for (entity, msaa, cloud_layer) in &views {
        let id = specializer.specialize(
            &pipeline_cache,
            &layouts,
            RenderSkyPipelineKey {
                msaa_samples: msaa.samples(),
                dual_source_blending: render_device
                    .features()
                    .contains(WgpuFeatures::DUAL_SOURCE_BLENDING),
                clouds_enabled: cloud_layer.is_some(),
                haze_prepass: false,
            },
        );
        commands.entity(entity).insert(RenderSkyPipelineId(id));
        if cloud_layer.is_some() && msaa.samples() == 1 {
            let haze = specializer.specialize(
                &pipeline_cache,
                &layouts,
                RenderSkyPipelineKey {
                    msaa_samples: 1,
                    dual_source_blending: false,
                    clouds_enabled: true,
                    haze_prepass: true,
                },
            );
            commands
                .entity(entity)
                .insert(RenderSkyHazePipelineId(haze));
        } else {
            commands.entity(entity).remove::<RenderSkyHazePipelineId>();
        }
    }
}

/// The render_sky half-resolution haze prepass pipeline (views with clouds, no MSAA).
#[derive(Component)]
pub struct RenderSkyHazePipelineId(pub CachedRenderPipelineId);

#[derive(Component)]
pub struct AtmosphereTextures {
    pub transmittance_lut: CachedTexture,
    pub multiscattering_lut: CachedTexture,
    pub sky_view_lut: CachedTexture,
    pub aerial_view_lut: CachedTexture,
    pub cloud_shadow_map: CachedTexture,
    pub cloud_shadow_map_tmp: CachedTexture,
    pub cloud_shadow_map_history: CachedTexture,
    /// Low-res cloud raymarch output: rgb = pre-exposed luminance, a = transmittance.
    pub clouds: CachedTexture,
    /// Half-res haze (render_sky `haze` prepass), this frame: pre-exposure in-scattering (rgb).
    pub haze_inscattering: CachedTexture,
    /// Half-res haze, this frame: transmittance (rgb), ray length in km (a).
    pub haze_transmittance: CachedTexture,
    /// Previous frame's haze (ping-pong), the prepass's temporal history.
    pub haze_history_inscattering: CachedTexture,
    pub haze_history_transmittance: CachedTexture,
    /// Low-res transmittance-weighted cloud distance (m).
    pub clouds_depth: CachedTexture,
    /// Previous frame's [`Self::clouds`] / [`Self::clouds_depth`] (ping-pong by frame parity).
    pub clouds_history: CachedTexture,
    pub clouds_depth_history: CachedTexture,
    /// Checkerboard trace targets (1 of each 2x2 half-res pixels per frame).
    pub clouds_trace: CachedTexture,
    pub clouds_trace_depth: CachedTexture,
    /// Cloud SDF ping-pong (rgba32f: squared distances + voxel height) and result (r = metres).
    pub cloud_field_a: CachedTexture,
    pub cloud_field_b: CachedTexture,
    pub cloud_field_sdf: CachedTexture,
    pub cloud_field_ext: CachedTexture,
    pub cloud_field_esc: CachedTexture,
    /// Weather window around the camera (`weather_window.wesl`): A the art channels (the tiled
    /// map's), B the physical channels. [`WEATHER_WINDOW_SIZE`]^2, recomposed every frame.
    pub weather_window_art: CachedTexture,
    pub weather_window_physical: CachedTexture,
    /// Diffusion solve of the cloud field (`cloud_field.wesl`): source and solution (R32Float,
    /// read-write) and the published, filterable solution the cloud pass samples.
    pub cloud_field_diff_src: CachedTexture,
    pub cloud_field_diff_u: CachedTexture,
    pub cloud_field_diff: CachedTexture,
    /// The diffusion solution and source moved with the grid (`cloud_field.wesl` diffusion_shift).
    pub cloud_field_diff_tmp: CachedTexture,
    /// The diffusion solve's build cell and pending shift (4 x 1 x 1).
    pub cloud_field_diff_state: CachedTexture,
}

pub(super) fn prepare_atmosphere_textures(
    views: Query<
        (
            Entity,
            &ExtractedView,
            &GpuAtmosphereSettings,
            Option<&ExtractedCamera>,
            Option<&MainPassResolutionOverride>,
            bevy_ecs::query::Has<GpuCloudLayers>,
        ),
        With<ExtractedAtmosphere>,
    >,
    render_device: Res<RenderDevice>,
    mut store: bevy_ecs::system::Local<HashMap<RetainedViewEntity, ViewTextureStore>>,
    frame_count: Res<FrameCount>,
    mut commands: Commands,
) {
    // Each view owns its textures (keyed by retained view entity): the LUTs, the cloud field,
    // the shadow map and the ping-pong histories persist across frames and must stay with
    // their view. `TextureCache` hands out the first free texture of a descriptor, so
    // equal-sized views could exchange them when the render-world query order changed (audit
    // 5). A texture is only reallocated when its descriptor changes, and a reallocation is a
    // new texture id, which the history validity checks compare.
    let mut seen: HashMap<RetainedViewEntity, ViewTextureStore> = HashMap::default();
    for (entity, view, lut_settings, camera, resolution_override, has_clouds) in &views {
        let mut textures = store.remove(&view.retained_view_entity).unwrap_or_default();
        textures.device = Some(render_device.clone());
        // Views without cloud layers bind 1x1 placeholders for every cloud texture (their
        // passes do not run, and PBR skips the cloud shadow map at extent 0): the full set is
        // ~52 MiB per view at 1080p (cloud field, shadow map + history, half-res cloud and haze
        // ping-pong targets).
        let cloud_extent = |full: Extent3d| {
            if has_clouds {
                full
            } else {
                Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                }
            }
        };
        let cloud_shadow_map_extent = cloud_extent(lut_settings.cloud_shadow_map_size.to_extents());
        let transmittance_lut = textures.get(TextureDescriptor {
            label: Some("transmittance_lut"),
            size: lut_settings.transmittance_lut_size.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let multiscattering_lut = textures.get(TextureDescriptor {
            label: Some("multiscattering_lut"),
            size: lut_settings.multiscattering_lut_size.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let sky_view_lut = textures.get(TextureDescriptor {
            label: Some("sky_view_lut"),
            size: lut_settings.sky_view_lut_size.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let aerial_view_lut = textures.get(TextureDescriptor {
            label: Some("aerial_view_lut"),
            size: lut_settings.aerial_view_lut_size.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let mut field_tex = |label: &'static str, format: TextureFormat| {
            textures.get(TextureDescriptor {
                label: Some(label),
                size: cloud_extent(Extent3d {
                    width: CLOUD_FIELD_RES_XY,
                    height: CLOUD_FIELD_RES_XY,
                    depth_or_array_layers: CLOUD_FIELD_RES_Z,
                }),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format,
                // COPY_SRC: debug dumps (`CLOUD_DUMP_DIR`).
                usage: TextureUsages::STORAGE_BINDING
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let cloud_field_a = field_tex("cloud_field_a", TextureFormat::Rgba16Float);
        let cloud_field_b = field_tex("cloud_field_b", TextureFormat::Rgba16Float);
        let cloud_field_sdf = field_tex("cloud_field_sdf", TextureFormat::Rgba16Float);
        let cloud_field_ext = field_tex("cloud_field_ext", TextureFormat::Rgba16Float);
        // Escape probability at half the horizontal resolution (`cloud_field.wesl` escape).
        let cloud_field_esc = textures.get(TextureDescriptor {
            label: Some("cloud_field_esc"),
            size: cloud_extent(Extent3d {
                width: CLOUD_FIELD_RES_XY / 2,
                height: CLOUD_FIELD_RES_XY / 2,
                depth_or_array_layers: CLOUD_FIELD_RES_Z,
            }),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let mut field_volume =
            |label: &'static str, format: TextureFormat, usage: TextureUsages| {
                textures.get(TextureDescriptor {
                    label: Some(label),
                    size: cloud_extent(Extent3d {
                        width: CLOUD_FIELD_RES_XY,
                        height: CLOUD_FIELD_RES_XY,
                        depth_or_array_layers: CLOUD_FIELD_RES_Z,
                    }),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D3,
                    format,
                    usage,
                    view_formats: &[],
                })
            };
        let cloud_field_diff_src = field_volume(
            "cloud_field_diff_src",
            TextureFormat::R32Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        );
        let cloud_field_diff_u = field_volume(
            "cloud_field_diff_u",
            TextureFormat::R32Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        );
        let cloud_field_diff = field_volume(
            "cloud_field_diff",
            TextureFormat::Rgba16Float,
            TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
        );
        let mut diff_aux = |label: &'static str, size: Extent3d| {
            textures.get(TextureDescriptor {
                label: Some(label),
                size: cloud_extent(size),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::R32Float,
                usage: TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            })
        };
        let cloud_field_diff_tmp = diff_aux(
            "cloud_field_diff_tmp",
            Extent3d {
                width: CLOUD_FIELD_RES_XY,
                height: CLOUD_FIELD_RES_XY,
                depth_or_array_layers: 2 * CLOUD_FIELD_RES_Z,
            },
        );
        let cloud_field_diff_state = diff_aux(
            "cloud_field_diff_state",
            Extent3d {
                width: 4,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let mut window = |label: &'static str| {
            textures.get(TextureDescriptor {
                label: Some(label),
                size: cloud_extent(Extent3d {
                    width: WEATHER_WINDOW_SIZE,
                    height: WEATHER_WINDOW_SIZE,
                    depth_or_array_layers: 1,
                }),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let weather_window_art = window("weather_window_art");
        let weather_window_physical = window("weather_window_physical");

        // Mip chain: lookups read the map at their pixel footprint (`cloud_shadow_mips.wesl`).
        let cloud_shadow_map = textures.get(TextureDescriptor {
            label: Some("cloud_shadow_map"),
            size: cloud_shadow_map_extent,
            mip_level_count: cloud_shadow_map_extent.max_mips(TextureDimension::D2),
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let cloud_shadow_map_tmp = textures.get(TextureDescriptor {
            label: Some("cloud_shadow_map_tmp"),
            size: cloud_shadow_map_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let cloud_shadow_map_history = textures.get(TextureDescriptor {
            label: Some("cloud_shadow_map_history"),
            size: cloud_shadow_map_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Half-resolution cloud raymarch targets (design.md §1.2), relative to the main pass's
        // render resolution (below the output resolution under an upscaler such as DLSS).
        let render_size = resolution_override
            .map(|size| **size)
            .or(camera.and_then(|camera| camera.physical_viewport_size))
            .filter(|_| has_clouds)
            .unwrap_or(UVec2::ONE);
        let clouds_size = cloud_render_size(render_size, lut_settings.cloud_max_render_size);
        let trace_size = (clouds_size + UVec2::ONE) / 2;
        let clouds_texture =
            |label: &'static str, format: TextureFormat, size: UVec2, copy: TextureUsages| {
                TextureDescriptor {
                    label: Some(label),
                    size: size.to_extents(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    usage: TextureUsages::STORAGE_BINDING
                        | TextureUsages::TEXTURE_BINDING
                        | TextureUsages::COPY_SRC
                        | copy,
                    view_formats: &[],
                }
            };
        let mut get =
            |label, format, size, copy| textures.get(clouds_texture(label, format, size, copy));
        // Ping-pong reconstruction targets: distinct labels keep the cached textures stable
        // across frames, and the frame parity picks which one is written this frame.
        let none = TextureUsages::empty();
        let clouds_a = get("clouds_a", TextureFormat::Rgba16Float, clouds_size, none);
        let clouds_b = get("clouds_b", TextureFormat::Rgba16Float, clouds_size, none);
        let clouds_depth_a = get("clouds_depth_a", TextureFormat::R32Float, clouds_size, none);
        let clouds_depth_b = get("clouds_depth_b", TextureFormat::R32Float, clouds_size, none);
        let (clouds, clouds_history, clouds_depth, clouds_depth_history) = if frame_count.0 % 2 == 0
        {
            (clouds_a, clouds_b, clouds_depth_a, clouds_depth_b)
        } else {
            (clouds_b, clouds_a, clouds_depth_b, clouds_depth_a)
        };
        let haze_usage = TextureUsages::RENDER_ATTACHMENT;
        let haze_l_a = get(
            "haze_inscattering_a",
            TextureFormat::Rgba16Float,
            clouds_size,
            haze_usage,
        );
        let haze_l_b = get(
            "haze_inscattering_b",
            TextureFormat::Rgba16Float,
            clouds_size,
            haze_usage,
        );
        let haze_t_a = get(
            "haze_transmittance_a",
            TextureFormat::Rgba16Float,
            clouds_size,
            haze_usage,
        );
        let haze_t_b = get(
            "haze_transmittance_b",
            TextureFormat::Rgba16Float,
            clouds_size,
            haze_usage,
        );
        let (
            haze_inscattering,
            haze_history_inscattering,
            haze_transmittance,
            haze_history_transmittance,
        ) = if frame_count.0 % 2 == 0 {
            (haze_l_a, haze_l_b, haze_t_a, haze_t_b)
        } else {
            (haze_l_b, haze_l_a, haze_t_b, haze_t_a)
        };
        let clouds_trace = get(
            "clouds_trace",
            TextureFormat::Rgba16Float,
            trace_size,
            TextureUsages::empty(),
        );
        let clouds_trace_depth = get(
            "clouds_trace_depth",
            TextureFormat::R32Float,
            trace_size,
            TextureUsages::empty(),
        );

        commands.entity(entity).insert({
            AtmosphereTextures {
                transmittance_lut,
                multiscattering_lut,
                sky_view_lut,
                aerial_view_lut,
                cloud_shadow_map,
                cloud_shadow_map_tmp,
                cloud_shadow_map_history,
                clouds,
                haze_inscattering,
                haze_transmittance,
                haze_history_inscattering,
                haze_history_transmittance,
                clouds_depth,
                clouds_history,
                clouds_depth_history,
                clouds_trace,
                clouds_trace_depth,
                cloud_field_a,
                cloud_field_b,
                cloud_field_sdf,
                cloud_field_ext,
                cloud_field_esc,
                weather_window_art,
                weather_window_physical,
                cloud_field_diff_src,
                cloud_field_diff_u,
                cloud_field_diff,
                cloud_field_diff_tmp,
                cloud_field_diff_state,
            }
        });
        seen.insert(view.retained_view_entity, textures);
    }
    *store = seen;
}

/// Textures owned by one atmosphere view ([`prepare_atmosphere_textures`]), by label.
#[derive(Default)]
pub struct ViewTextureStore {
    device: Option<RenderDevice>,
    textures: HashMap<&'static str, (TextureKey, CachedTexture)>,
}

/// The parts of a [`TextureDescriptor`] that require a new texture when they change.
type TextureKey = (
    Extent3d,
    u32,
    u32,
    TextureDimension,
    TextureFormat,
    TextureUsages,
);

impl ViewTextureStore {
    /// This view's texture with the descriptor's label, reallocated when the descriptor changes.
    fn get(&mut self, descriptor: TextureDescriptor<'static>) -> CachedTexture {
        let label = descriptor.label.unwrap_or("atmosphere_texture");
        let key = (
            descriptor.size,
            descriptor.mip_level_count,
            descriptor.sample_count,
            descriptor.dimension,
            descriptor.format,
            descriptor.usage,
        );
        if let Some((k, texture)) = self.textures.get(label) {
            if *k == key {
                return texture.clone();
            }
        }
        let device = self
            .device
            .as_ref()
            .expect("ViewTextureStore without a device");
        let texture = device.create_texture(&descriptor);
        let default_view = texture.create_view(&TextureViewDescriptor::default());
        let cached = CachedTexture {
            texture,
            default_view,
        };
        self.textures.insert(label, (key, cached.clone()));
        cached
    }
}

#[derive(Copy, Clone, Debug, thiserror::Error)]
#[error("ScatteringMedium missing with id {0:?}: make sure the asset was not removed.")]
struct ScatteringMediumMissingError(AssetId<ScatteringMedium>);

/// The shader-uniform representation of an Atmosphere.
#[derive(Clone, Component, ShaderType)]
pub struct GpuAtmosphere {
    pub ground_albedo: Vec3,
    pub inner_radius: f32,
    pub outer_radius: f32,
    pub world_to_atmosphere: Mat4,
    pub atmosphere_to_world: Mat4,
}

pub fn prepare_atmosphere_uniforms(
    mut commands: Commands,
    atmospheres: Query<(Entity, &ExtractedAtmosphere)>,
) -> Result<(), BevyError> {
    for (entity, atmosphere) in atmospheres {
        commands.entity(entity).insert(GpuAtmosphere {
            ground_albedo: atmosphere.ground_albedo,
            inner_radius: atmosphere.inner_radius,
            outer_radius: atmosphere.outer_radius,
            world_to_atmosphere: atmosphere.world_to_atmosphere,
            atmosphere_to_world: atmosphere.atmosphere_to_world,
        });
    }
    Ok(())
}

#[derive(Resource, Default)]
pub struct AtmosphereTransforms {
    uniforms: DynamicUniformBuffer<AtmosphereTransform>,
}

impl AtmosphereTransforms {
    #[inline]
    pub fn uniforms(&self) -> &DynamicUniformBuffer<AtmosphereTransform> {
        &self.uniforms
    }
}

/// Transforms between world space and atmosphere space.
///
/// Up is the local planet surface normal, so the horizon stays along the x-z
/// plane for horizon-detail parameterization. Back is chosen from a constant
/// world-horizontal direction such as `Vec3A::NEG_Z`, then projected orthogonal
/// to up. It may drift slightly from world-horizontal but stays camera-independent.
#[derive(ShaderType)]
pub struct AtmosphereTransform {
    world_from_atmosphere: Mat4,
    atmosphere_from_world: Mat4,
}

#[derive(Component)]
pub struct AtmosphereTransformsOffset {
    index: u32,
}

impl AtmosphereTransformsOffset {
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }
}

pub(super) fn prepare_atmosphere_transforms(
    views: Query<
        (Entity, &ExtractedView, &GpuAtmosphere),
        (With<ExtractedAtmosphere>, With<Camera3d>),
    >,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut atmo_uniforms: ResMut<AtmosphereTransforms>,
    mut commands: Commands,
) {
    let atmo_count = views.iter().len();
    let Some(mut writer) =
        atmo_uniforms
            .uniforms
            .get_writer(atmo_count, &render_device, &render_queue)
    else {
        return;
    };

    for (entity, view, gpu_atmosphere) in &views {
        // Camera position in atmosphere space
        let cam_world = view.world_from_view.translation();
        let cam_pos = Vec3A::from(
            gpu_atmosphere
                .world_to_atmosphere
                .transform_point3(cam_world),
        );

        // Up is the local planet surface normal.
        let atmo_y = cam_pos.try_normalize().unwrap_or(Vec3A::Y);

        // World-horizontal reference for back, projected orthogonal to atmo_y.
        let world_ref = Vec3A::NEG_Z;
        let ref_horizontal = world_ref - atmo_y * atmo_y.dot(world_ref);
        let atmo_z = ref_horizontal.try_normalize().unwrap_or_else(|| {
            // `NEG_Z` is degenerate at the poles of a Z-up world.
            let fallback_ref = Vec3A::NEG_Y;
            (fallback_ref - atmo_y * atmo_y.dot(fallback_ref)).normalize()
        });
        let atmo_x = atmo_y.cross(atmo_z).normalize();

        let world_from_atmosphere = Mat4::from(Affine3A::from_cols(
            atmo_x,
            atmo_y,
            atmo_z,
            view.world_from_view.translation_vec3a(),
        ));
        // The shader only uses the upper-left 3x3 block, where transpose equals inverse for
        // orthonormal matrices and is cheaper than computing the full inverse.
        let atmosphere_from_world = world_from_atmosphere.transpose();

        commands.entity(entity).insert(AtmosphereTransformsOffset {
            index: writer.write(&AtmosphereTransform {
                world_from_atmosphere,
                atmosphere_from_world,
            }),
        });
    }
}

#[derive(Component)]
pub(crate) struct AtmosphereBindGroups {
    pub transmittance_lut: BindGroup,
    pub multiscattering_lut: BindGroup,
    pub sky_view_lut: BindGroup,
    pub aerial_view_lut: BindGroup,
    /// Cloud-shadowed LUT bind groups (views with a `VolumetricCloudLayer`).
    pub sky_view_lut_clouds: Option<BindGroup>,
    pub aerial_view_lut_clouds: Option<BindGroup>,
    pub cloud_shadow_map: Option<BindGroup>,
    // Accessed from the render-world graph; some tooling can miss the cross-module usage.
    #[allow(dead_code)]
    pub cloud_shadow_filter_a_to_b: Option<BindGroup>,
    // Accessed from the render-world graph; some tooling can miss the cross-module usage.
    #[allow(dead_code)]
    pub cloud_shadow_filter_b_to_a: Option<BindGroup>,
    /// Temporal filter bind group; present when cloud_shadow_temporal_enabled.
    pub cloud_shadow_temporal: Option<BindGroup>,
    pub render_sky_clouds: Option<BindGroup>,
    /// `render_sky_clouds` with placeholders in the haze slots, for the haze prepass.
    pub render_sky_haze: Option<BindGroup>,
    pub render_sky_no_clouds: BindGroup,
    /// `clouds_raymarch` pass (views with a `VolumetricCloudLayer`).
    pub clouds_raymarch: Option<BindGroup>,
    /// Whether `clouds_raymarch` was made for the multisampled-depth layout (MSAA view).
    pub clouds_raymarch_multisampled: bool,
    /// `clouds_reconstruct` pass (same conditions as `clouds_raymarch`).
    pub clouds_reconstruct: Option<BindGroup>,
    /// Cloud SDF passes: x (a->b), y (b->a), z (a->sdf).
    /// edt_x, edt_y, edt_z, escape.
    pub cloud_field_edt: Option<[BindGroup; 4]>,
    /// Diffusion solve (`cloud_field.wesl`), dynamic offsets: atmosphere, lights.
    pub cloud_field_diffusion: Option<BindGroup>,
}

#[derive(Copy, Clone, Debug, thiserror::Error)]
enum AtmosphereBindGroupError {
    #[error("Failed to prepare atmosphere bind groups. Atmosphere uniform buffer missing")]
    Atmosphere,
    #[error(
        "Failed to prepare atmosphere bind groups. AtmosphereTransforms uniform buffer missing"
    )]
    Transforms,
    #[error("Failed to prepare atmosphere bind groups. AtmosphereSettings uniform buffer missing")]
    Settings,
    #[error("Failed to prepare atmosphere bind groups. View uniform buffer missing")]
    ViewUniforms,
    #[error("Failed to prepare atmosphere bind groups. Light uniform buffer missing")]
    LightUniforms,
}

pub(super) fn prepare_cloud_shadow_temporal_params(
    views: Query<
        (
            Entity,
            &ExtractedView,
            &GpuAtmosphereSettings,
            &AtmosphereTextures,
            Option<&GpuCloudLayers>,
            Option<&bevy_camera::visibility::RenderLayers>,
        ),
        (With<Camera3d>, With<ExtractedAtmosphere>),
    >,
    directional_lights: Query<(
        Entity,
        &ExtractedDirectionalLight,
        &bevy_camera::visibility::RenderLayers,
    )>,
    mut states: bevy_ecs::system::Local<HashMap<RetainedViewEntity, CloudShadowTemporalState>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    frame_count: Res<FrameCount>,
    mut commands: Commands,
) {
    // The shadow map, its history and therefore its temporal state belong to each view (audit
    // 10): one global state reprojected secondary views with the first view's anchor, extent
    // and motion.
    let mut lights: Vec<_> = directional_lights.iter().collect();
    // The order of the GPU's directional lights (render/light.rs `prepare_lights`): sorted by
    // volumetric first, then shadow maps enabled, then entity, and filtered by the view's render
    // layers. The cloud shadow map follows light 0 of its view (`lights.directional_lights[0]`).
    lights.sort_by_key(|(entity, light, _)| {
        (
            core::cmp::Reverse(light.volumetric),
            core::cmp::Reverse(light.shadow_maps_enabled),
            *entity,
        )
    });
    let mut seen = HashMap::default();
    for (entity, view, settings, textures, cloud_layer, view_layers) in &views {
        if !settings.cloud_shadow_uses_temporal_pass() || cloud_layer.is_none() {
            // No temporal pass, or no cloud layer (1x1 placeholder shadow textures): no history
            // is kept, so it is invalid when they come back.
            continue;
        }
        let view_layers = view_layers.cloned().unwrap_or_default();
        let light_dir = lights
            .iter()
            .find(|(_, _, layers)| layers.intersects(&view_layers))
            .map(|(_, light, _)| Vec3::from(light.transform.back()))
            .unwrap_or(Vec3::NEG_Y);
        let previous = states.remove(&view.retained_view_entity);
        let (state, params) = cloud_shadow_temporal_params(
            previous.as_ref(),
            settings,
            cloud_layer,
            light_dir,
            textures.cloud_shadow_map_history.texture.id(),
            frame_count.0,
        );
        let turn = CloudLightTurn {
            recent: params.rotation_since[3].w,
            field_tolerance: params.history_angle_scale
                * (CLOUD_FIELD_EXTENT_M / CLOUD_FIELD_RES_XY as f32)
                / 8.0,
        };
        let mut buffer = UniformBuffer::from(params);
        buffer.write_buffer(&render_device, &render_queue);
        commands
            .entity(entity)
            .insert((CloudShadowTemporalParamsUniform(buffer), turn));
        seen.insert(view.retained_view_entity, state);
    }
    *states = seen;
}

/// How far the view's light turned recently, for the cloud field's diffusion solve (`node.rs`):
/// the angle over the last 16 frames (rad), and the angle at which the solve's source goes stale
/// by 1/8 of a field voxel (rad): voxel x |mu| / layer thickness / 8, the 8-frame slice cycle's
/// share of the voxel (see [`CloudShadowTemporalParams::history_angle_scale`]).
#[derive(Component, Clone, Copy, Default)]
pub struct CloudLightTurn {
    pub recent: f32,
    pub field_tolerance: f32,
}

/// Per-view uniform for [`CloudShadowTemporalParams`].
#[derive(Component)]
pub struct CloudShadowTemporalParamsUniform(pub UniformBuffer<CloudShadowTemporalParams>);

/// One view's shadow temporal parameters and its new state.
fn cloud_shadow_temporal_params(
    previous: Option<&CloudShadowTemporalState>,
    settings: &GpuAtmosphereSettings,
    cloud_layer: Option<&GpuCloudLayers>,
    light_dir: Vec3,
    history_texture: TextureId,
    frame_index: u32,
) -> (CloudShadowTemporalState, CloudShadowTemporalParams) {
    let temporal_state = previous.cloned().unwrap_or_default();
    let extent = settings.cloud_shadow_map_extent;
    // When only time-slicing (no temporal filtering), traced texels replace history outright.
    let temporal_alpha = if settings.cloud_shadow_temporal_enabled != 0 {
        settings.cloud_shadow_temporal_alpha
    } else {
        1.0
    };
    let light_rotation_cut_deg = settings.cloud_shadow_temporal_light_rotation_cut_deg;

    // The map's centre, with hysteresis (`cloud_shadow_anchor`), as every shader reads it.
    let anchor = settings.cloud_shadow_anchor;
    let trace_dir = (-light_dir).normalize_or_zero();
    let world_up = Vec3::Y;
    let a = if trace_dir.dot(world_up).abs() > 0.999 {
        Vec3::X
    } else {
        world_up
    };
    let basis_x = a.cross(trace_dir).normalize_or_zero();
    let basis_y = trace_dir.cross(basis_x);

    let size = settings.cloud_shadow_map_size;
    let prev_anchor = temporal_state.anchor;
    let prev_light_dir = temporal_state.light_dir;
    let anchor_moved = (anchor - prev_anchor).length_squared() > 1e-6;
    let light_rot_cos = (light_rotation_cut_deg * std::f32::consts::PI / 180.0).cos();
    let light_rotated = prev_light_dir.dot(light_dir.normalize_or_zero()) < light_rot_cos;
    // The history must be the texture this view copied its map into last frame.
    let history_valid = temporal_state.initialized
        && !light_rotated
        && temporal_state.size == size
        && temporal_state.extent == extent
        && temporal_state.history_texture == Some(history_texture);

    let cloud_offset = cloud_layer.map_or(Vec3::ZERO, |c| c.noise_offset);
    let cirrus_offset = cloud_layer
        .and_then(|c| c.lowest_thin())
        .map_or(Vec3::ZERO, |t| t.offset);
    let (cloud_motion, cirrus_motion) = if history_valid {
        (
            cloud_offset - temporal_state.cloud_offset,
            cirrus_offset - temporal_state.cirrus_offset,
        )
    } else {
        (Vec3::ZERO, Vec3::ZERO)
    };
    // The light's turn this frame, and over the last frames (a time-sliced texel was traced up
    // to 16 frames ago, and its history holds whatever the turning light left in it since).
    let light_rotation = if history_valid {
        prev_light_dir
            .dot(light_dir.normalize_or_zero())
            .clamp(-1.0, 1.0)
            .acos()
    } else {
        0.0
    };
    let mut rotations = [0.0; 16];
    if history_valid {
        rotations[1..].copy_from_slice(&temporal_state.rotations[..15]);
    }
    rotations[0] = light_rotation;
    let mut rotation_since = [Vec4::ZERO; 4];
    let mut sum = 0.0;
    for (k, r) in rotations.iter().enumerate() {
        sum += r;
        rotation_since[k / 4][k % 4] = sum;
    }
    let new_state = CloudShadowTemporalState {
        anchor,
        light_dir: light_dir.normalize_or_zero(),
        basis_x,
        basis_y,
        initialized: true,
        cloud_offset,
        cirrus_offset,
        size,
        extent,
        history_texture: Some(history_texture),
        rotations,
    };

    let prev_anchor = if history_valid { prev_anchor } else { anchor };
    let prev_light_dir_to_light = if history_valid {
        prev_light_dir
    } else {
        light_dir.normalize_or_zero()
    };
    let prev_trace_dir = -prev_light_dir_to_light;
    // A ray through the cloud layer at zenith cosine mu crosses it over thickness / mu; turning
    // the light by d moves the ray's far end sideways by d thickness / mu.
    let mu = trace_dir.dot(anchor.normalize_or_zero()).abs().max(1e-3);
    let thickness = cloud_layer.map_or(1.0, |c| (c.cloud_layer_end - c.cloud_layer_start).max(1.0));
    let history_angle_scale = mu / thickness;
    // Build the previous basis from the trace direction exactly like the current one (and like
    // `build_light_basis` in the shaders); building it from the direction *to* the light mirrors
    // the X axis and reprojects the history to the wrong texels.
    let prev_a = if prev_trace_dir.dot(Vec3::Y).abs() > 0.999 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let prev_basis_x = prev_a.cross(prev_trace_dir).normalize_or_zero();
    let prev_basis_y = prev_trace_dir.cross(prev_basis_x);

    (
        new_state,
        CloudShadowTemporalParams {
            curr_anchor: anchor,
            curr_light_dir: trace_dir,
            curr_basis_x: basis_x,
            curr_basis_y: basis_y,
            prev_anchor,
            prev_light_dir: prev_trace_dir,
            prev_basis_x,
            prev_basis_y,
            extent,
            temporal_alpha,
            history_valid: history_valid as u32,
            anchor_moved: anchor_moved as u32,
            size,
            interleave: settings.cloud_shadow_map_update_interleave.max(1),
            frame_index,
            cloud_motion: Vec3::new(cloud_motion.x, 0.0, cloud_motion.z),
            cirrus_motion: Vec3::new(cirrus_motion.x, 0.0, cirrus_motion.z),
            history_angle_scale,
            rotation_since,
        },
    )
}

pub(super) fn prepare_atmosphere_bind_groups(
    views: Query<
        (
            Entity,
            &ExtractedAtmosphere,
            &AtmosphereTextures,
            &ViewDepthStencilTexture,
            &ViewShadowBindings,
            &Msaa,
            Option<&GpuCloudLayers>,
            Option<&GpuAtmosphereSettings>,
            Option<&CloudsTemporalParamsUniform>,
            Option<&CloudShadowTemporalParamsUniform>,
            Option<&CloudReferenceAccumulator>,
        ),
        (With<Camera3d>, With<ExtractedAtmosphere>),
    >,
    render_device: Res<RenderDevice>,
    layouts: Res<AtmosphereBindGroupLayouts>,
    render_sky_layouts: Res<RenderSkyBindGroupLayouts>,
    (atmosphere_sampler, cloud_noise_sampler, shadow_samplers): (
        Res<LinearSampler>,
        Res<CloudNoiseSampler>,
        Res<ShadowSamplers>,
    ),
    (bluenoise, render_images, fallback_image): (
        Res<Bluenoise>,
        Res<RenderAssets<GpuImage>>,
        Res<FallbackImage>,
    ),
    view_uniforms: Res<ViewUniforms>,
    lights_uniforms: Res<LightMeta>,
    atmosphere_transforms: Res<AtmosphereTransforms>,
    atmosphere_uniforms: Res<ComponentUniforms<GpuAtmosphere>>,
    settings_uniforms: Res<ComponentUniforms<GpuAtmosphereSettings>>,
    gpu_media: Res<RenderAssets<GpuScatteringMedium>>,
    medium_sampler: Res<ScatteringMediumSampler>,
    pipeline_cache: Res<PipelineCache>,
    (cloud_layer_uniforms, fbm_noise_texture, perlin_worley_noise_texture, detail_noise_texture): (
        Res<ComponentUniforms<GpuCloudLayers>>,
        Res<FbmNoiseTexture>,
        Res<PerlinWorleyNoiseTexture>,
        Res<DetailNoiseTexture>,
    ),
    mut commands: Commands,
) -> Result<(), BevyError> {
    if views.iter().len() == 0 {
        return Ok(());
    }

    let atmosphere_binding = atmosphere_uniforms
        .binding()
        .ok_or(AtmosphereBindGroupError::Atmosphere)?;

    let transforms_binding = atmosphere_transforms
        .uniforms()
        .binding()
        .ok_or(AtmosphereBindGroupError::Transforms)?;

    let settings_binding = settings_uniforms
        .binding()
        .ok_or(AtmosphereBindGroupError::Settings)?;

    let view_binding = view_uniforms
        .uniforms
        .binding()
        .ok_or(AtmosphereBindGroupError::ViewUniforms)?;

    let lights_binding = lights_uniforms
        .view_gpu_lights
        .binding()
        .ok_or(AtmosphereBindGroupError::LightUniforms)?;

    // Optional: when no view has a VolumetricCloudLayer component, the uniform buffer can be absent.
    // We build no-cloud bind groups in that case.
    let cloud_layer_binding = cloud_layer_uniforms.binding();

    // STBN texture for temporal stratification; use fallback when not yet loaded.
    let stbn_view: std::borrow::Cow<'_, TextureView> = match render_images.get(&bluenoise.texture) {
        Some(gpu) => std::borrow::Cow::Owned(gpu.texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        })),
        None => std::borrow::Cow::Borrowed(&fallback_image.d2_array.texture_view),
    };

    for (
        entity,
        atmosphere,
        textures,
        view_depth_texture,
        shadow_bindings,
        msaa,
        cloud_layer,
        settings,
        temporal_uniform,
        shadow_temporal_uniform,
        reference_accumulator,
    ) in &views
    {
        let Some(depth_view) = view_depth_texture
            .attachment
            .depth_stencil_views()
            .depth_only_view()
        else {
            continue;
        };
        let temporal_enabled = settings
            .map(GpuAtmosphereSettings::cloud_shadow_uses_temporal_pass)
            .unwrap_or(false);
        let gpu_medium = gpu_media
            .get(atmosphere.medium)
            .ok_or(ScatteringMediumMissingError(atmosphere.medium))?;
        let transmittance_lut = render_device.create_bind_group(
            "transmittance_lut_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.transmittance_lut),
            &BindGroupEntries::with_indices((
                // uniforms
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                // scattering medium luts and sampler
                (5, &gpu_medium.density_lut_view),
                (6, &gpu_medium.scattering_lut_view),
                (7, medium_sampler.sampler()),
                // transmittance lut storage texture
                (13, &textures.transmittance_lut.default_view),
            )),
        );

        let multiscattering_lut = render_device.create_bind_group(
            "multiscattering_lut_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.multiscattering_lut),
            &BindGroupEntries::with_indices((
                // uniforms
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                // scattering medium luts and sampler
                (5, &gpu_medium.density_lut_view),
                (6, &gpu_medium.scattering_lut_view),
                (7, medium_sampler.sampler()),
                // atmosphere luts and sampler
                (8, &textures.transmittance_lut.default_view),
                (12, &atmosphere_sampler.0),
                // multiscattering lut storage texture
                (13, &textures.multiscattering_lut.default_view),
            )),
        );

        let sky_view_lut = render_device.create_bind_group(
            "sky_view_lut_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.sky_view_lut),
            &BindGroupEntries::with_indices((
                // uniforms
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                (2, transforms_binding.clone()),
                (3, view_binding.clone()),
                (4, lights_binding.clone()),
                // scattering medium luts and sampler
                (5, &gpu_medium.density_lut_view),
                (6, &gpu_medium.scattering_lut_view),
                (7, medium_sampler.sampler()),
                // atmosphere luts and sampler
                (8, &textures.transmittance_lut.default_view),
                (9, &textures.multiscattering_lut.default_view),
                (12, &atmosphere_sampler.0),
                // sky view lut storage texture
                (13, &textures.sky_view_lut.default_view),
                // shadow maps
                (14, &shadow_bindings.point_light_depth_texture_view),
                (15, &shadow_samplers.shadow_comparison_sampler),
                (16, &shadow_bindings.directional_light_depth_texture_view),
                (17, &shadow_samplers.shadow_comparison_sampler),
            )),
        );

        let aerial_view_lut = render_device.create_bind_group(
            "aerial_view_lut_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.aerial_view_lut),
            &BindGroupEntries::with_indices((
                // uniforms
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                (3, view_binding.clone()),
                (4, lights_binding.clone()),
                // scattering medium luts and sampler
                (5, &gpu_medium.density_lut_view),
                (6, &gpu_medium.scattering_lut_view),
                (7, medium_sampler.sampler()),
                // atmosphere luts and sampler
                (8, &textures.transmittance_lut.default_view),
                (9, &textures.multiscattering_lut.default_view),
                (12, &atmosphere_sampler.0),
                // aerial view lut storage texture
                (13, &textures.aerial_view_lut.default_view),
                // shadow maps
                (14, &shadow_bindings.point_light_depth_texture_view),
                (15, &shadow_samplers.shadow_comparison_sampler),
                (16, &shadow_bindings.directional_light_depth_texture_view),
                (17, &shadow_samplers.shadow_comparison_sampler),
            )),
        );

        // The map's passes write and read its top level as storage (a storage view holds one
        // mip); every lookup samples the whole mip chain through `default_view`.
        let shadow_mip0 = textures.cloud_shadow_map.texture.create_view(&TextureViewDescriptor {
            label: Some("cloud_shadow_map_mip0"),
            base_mip_level: 0,
            mip_level_count: Some(1),
            ..default()
        });
        // When temporal enabled: trace writes to tmp; filter reads tmp first.
        // When temporal disabled: trace writes to cloud_shadow_map; filter reads cloud_shadow_map first.
        let trace_storage = if temporal_enabled {
            &textures.cloud_shadow_map_tmp.default_view
        } else {
            &shadow_mip0
        };
        let (filter_read_a, filter_write_a) = if temporal_enabled {
            (
                &textures.cloud_shadow_map_tmp.default_view,
                &shadow_mip0,
            )
        } else {
            (
                &shadow_mip0,
                &textures.cloud_shadow_map_tmp.default_view,
            )
        };
        let (filter_read_b, filter_write_b) = if temporal_enabled {
            (
                &shadow_mip0,
                &textures.cloud_shadow_map_tmp.default_view,
            )
        } else {
            (
                &textures.cloud_shadow_map_tmp.default_view,
                &shadow_mip0,
            )
        };

        let cloud_shadow_map = cloud_layer_binding.as_ref().map(|cloud_layer_binding| {
            render_device.create_bind_group(
                "cloud_shadow_map_bind_group",
                &pipeline_cache.get_bind_group_layout(&layouts.cloud_shadow_map),
                &BindGroupEntries::with_indices((
                    (0, atmosphere_binding.clone()),
                    (1, settings_binding.clone()),
                    (3, view_binding.clone()),
                    (4, lights_binding.clone()),
                    (20, cloud_layer_binding.clone()),
                    (21, &fbm_noise_texture.texture.default_view),
                    (22, &**cloud_noise_sampler),
                    (24, &perlin_worley_noise_texture.texture.default_view),
                    (25, stbn_view.as_ref()),
                    (33, &detail_noise_texture.texture.default_view),
                    (34, &fbm_noise_texture.profile.default_view),
                    (35, &fbm_noise_texture.scheme_noise.default_view),
                    (36, &textures.weather_window_art.default_view),
                    (37, &textures.weather_window_physical.default_view),
                    (13, trace_storage),
                    (40, &textures.cloud_field_a.default_view),
                    (45, &textures.cloud_field_ext.default_view),
                    (49, &fbm_noise_texture.cirrus.default_view),
                )),
            )
        });

        // Cloud shadow spatial filter ping-pong bind groups.
        // When temporal: A=tmp (trace output), B=cloud_shadow_map. When not: A=cloud_shadow_map, B=tmp.
        let cloud_shadow_filter_a_to_b = cloud_shadow_map.as_ref().map(|_| {
            render_device.create_bind_group(
                "cloud_shadow_filter_a_to_b_bind_group",
                &pipeline_cache.get_bind_group_layout(&layouts.cloud_shadow_filter),
                &BindGroupEntries::with_indices((
                    (1, settings_binding.clone()),
                    (12, &atmosphere_sampler.0),
                    (23, filter_read_a),
                    (13, filter_write_a),
                )),
            )
        });

        let cloud_shadow_filter_b_to_a = cloud_shadow_map.as_ref().map(|_| {
            render_device.create_bind_group(
                "cloud_shadow_filter_b_to_a_bind_group",
                &pipeline_cache.get_bind_group_layout(&layouts.cloud_shadow_filter),
                &BindGroupEntries::with_indices((
                    (1, settings_binding.clone()),
                    (12, &atmosphere_sampler.0),
                    (23, filter_read_b),
                    (13, filter_write_b),
                )),
            )
        });

        // Temporal filter: curr=tmp (or cloud_shadow_map when no filter), prev=history, out=cloud_shadow_map.
        let cloud_shadow_temporal = (cloud_shadow_map.is_some() && temporal_enabled)
            .then(|| {
                shadow_temporal_uniform
                    .and_then(|u| u.0.binding())
                    .map(|temporal_params_binding| {
                        render_device.create_bind_group(
                            "cloud_shadow_temporal_bind_group",
                            &pipeline_cache.get_bind_group_layout(&layouts.cloud_shadow_temporal),
                            &BindGroupEntries::with_indices((
                                (0, temporal_params_binding),
                                (1, &textures.cloud_shadow_map_tmp.default_view),
                                (2, &textures.cloud_shadow_map_history.default_view),
                                (3, &atmosphere_sampler.0),
                                (4, &shadow_mip0),
                            )),
                        )
                    })
            })
            .flatten();

        let render_sky_no_clouds = render_device.create_bind_group(
            "render_sky_bind_group_no_clouds",
            &pipeline_cache.get_bind_group_layout(if *msaa == Msaa::Off {
                &render_sky_layouts.render_sky_no_clouds
            } else {
                &render_sky_layouts.render_sky_msaa_no_clouds
            }),
            &BindGroupEntries::with_indices((
                // uniforms
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                (2, transforms_binding.clone()),
                (3, view_binding.clone()),
                (4, lights_binding.clone()),
                // scattering medium luts and sampler
                (5, &gpu_medium.density_lut_view),
                (6, &gpu_medium.scattering_lut_view),
                (7, medium_sampler.sampler()),
                // atmosphere luts and sampler
                (8, &textures.transmittance_lut.default_view),
                (9, &textures.multiscattering_lut.default_view),
                (10, &textures.sky_view_lut.default_view),
                (11, &textures.aerial_view_lut.default_view),
                (12, &atmosphere_sampler.0),
                // view depth texture
                (13, depth_view),
                // directional shadow maps
                (16, &shadow_bindings.directional_light_depth_texture_view),
                (17, &shadow_samplers.shadow_comparison_sampler),
            )),
        );

        // Only create clouds bind group for views that have VolumetricCloudLayer; otherwise use no_clouds
        // to avoid passing clouds layout to no_clouds pipeline when disabling clouds.
        let base_sky_entries = || {
            DynamicBindGroupEntries::new_with_indices((
                // uniforms
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                (2, transforms_binding.clone()),
                (3, view_binding.clone()),
                (4, lights_binding.clone()),
                // scattering medium luts and sampler
                (5, &gpu_medium.density_lut_view),
                (6, &gpu_medium.scattering_lut_view),
                (7, medium_sampler.sampler()),
                // atmosphere luts and sampler
                (8, &textures.transmittance_lut.default_view),
                (9, &textures.multiscattering_lut.default_view),
                (10, &textures.sky_view_lut.default_view),
                (11, &textures.aerial_view_lut.default_view),
                (12, &atmosphere_sampler.0),
                // view depth texture
                (13, depth_view),
                // directional shadow maps
                (16, &shadow_bindings.directional_light_depth_texture_view),
                (17, &shadow_samplers.shadow_comparison_sampler),
            ))
        };
        let cloud_inputs = cloud_layer_binding
            .as_ref()
            .filter(|_| cloud_layer.is_some());
        let haze_params = temporal_uniform.and_then(|u| u.0.binding());
        // Composite pass (full resolution). Without MSAA the atmosphere march runs in the half-res
        // haze prepass, and this pass only upsamples the haze and composites the clouds.
        let render_sky_clouds = cloud_inputs.and_then(|cloud_layer_binding| {
            if *msaa == Msaa::Off {
                let haze_params = haze_params.clone()?;
                Some(render_device.create_bind_group(
                    "render_sky_bind_group_clouds",
                    &pipeline_cache.get_bind_group_layout(&render_sky_layouts.render_sky_clouds),
                    &base_sky_entries().extend_with_indices((
                        (20, cloud_layer_binding.clone()),
                        (22, &**cloud_noise_sampler),
                        (23, &textures.cloud_shadow_map.default_view),
                        (26, &textures.clouds.default_view),
                        (27, &textures.clouds_depth.default_view),
                        (50, &textures.haze_inscattering.default_view),
                        (51, &textures.haze_transmittance.default_view),
                        (52, haze_params),
                    )),
                ))
            } else {
                Some(
                    render_device.create_bind_group(
                        "render_sky_msaa_bind_group_clouds",
                        &pipeline_cache
                            .get_bind_group_layout(&render_sky_layouts.render_sky_msaa_clouds),
                        &base_sky_entries().extend_with_indices((
                            (20, cloud_layer_binding.clone()),
                            (21, &fbm_noise_texture.texture.default_view),
                            (22, &**cloud_noise_sampler),
                            (23, &textures.cloud_shadow_map.default_view),
                            (24, &perlin_worley_noise_texture.texture.default_view),
                            (25, stbn_view.as_ref()),
                            (33, &detail_noise_texture.texture.default_view),
                            (34, &fbm_noise_texture.profile.default_view),
                            (35, &fbm_noise_texture.scheme_noise.default_view),
                            (36, &textures.weather_window_art.default_view),
                            (37, &textures.weather_window_physical.default_view),
                            (26, &textures.clouds.default_view),
                            (27, &textures.clouds_depth.default_view),
                        )),
                    ),
                )
            }
        });
        // The half-res haze prepass writes this frame's haze, so it reads the previous frame's
        // (its temporal history) in the haze slots.
        let render_sky_haze = cloud_inputs
            .filter(|_| *msaa == Msaa::Off)
            .zip(haze_params.clone())
            .map(|(cloud_layer_binding, haze_params)| {
                render_device.create_bind_group(
                    "render_sky_haze_bind_group",
                    &pipeline_cache.get_bind_group_layout(&render_sky_layouts.render_sky_haze),
                    &base_sky_entries().extend_with_indices((
                        (20, cloud_layer_binding.clone()),
                        (21, &fbm_noise_texture.texture.default_view),
                        (22, &**cloud_noise_sampler),
                        (23, &textures.cloud_shadow_map.default_view),
                        (24, &perlin_worley_noise_texture.texture.default_view),
                        (25, stbn_view.as_ref()),
                        (33, &detail_noise_texture.texture.default_view),
                        (34, &fbm_noise_texture.profile.default_view),
                        (35, &fbm_noise_texture.scheme_noise.default_view),
                        (36, &textures.weather_window_art.default_view),
                        (37, &textures.weather_window_physical.default_view),
                        (50, &textures.haze_history_inscattering.default_view),
                        (51, &textures.haze_history_transmittance.default_view),
                        (52, haze_params),
                    )),
                )
            });

        let clouds_raymarch = cloud_layer_binding
            .as_ref()
            .filter(|_| cloud_layer.is_some())
            .map(|cloud_layer_binding| {
                render_device.create_bind_group(
                    "clouds_raymarch_bind_group",
                    &pipeline_cache.get_bind_group_layout(if *msaa == Msaa::Off {
                        &layouts.clouds_raymarch
                    } else {
                        &layouts.clouds_raymarch_msaa
                    }),
                    &BindGroupEntries::with_indices((
                        (0, atmosphere_binding.clone()),
                        (1, settings_binding.clone()),
                        (2, transforms_binding.clone()),
                        (3, view_binding.clone()),
                        (4, lights_binding.clone()),
                        (5, &gpu_medium.density_lut_view),
                        (6, &gpu_medium.scattering_lut_view),
                        (7, medium_sampler.sampler()),
                        (8, &textures.transmittance_lut.default_view),
                        (9, &textures.multiscattering_lut.default_view),
                        (10, &textures.sky_view_lut.default_view),
                        (11, &textures.aerial_view_lut.default_view),
                        (12, &atmosphere_sampler.0),
                        (13, depth_view),
                        (20, cloud_layer_binding.clone()),
                        (21, &fbm_noise_texture.texture.default_view),
                        (22, &**cloud_noise_sampler),
                        (23, &textures.cloud_shadow_map.default_view),
                        (24, &perlin_worley_noise_texture.texture.default_view),
                        (25, stbn_view.as_ref()),
                        (33, &detail_noise_texture.texture.default_view),
                        (34, &fbm_noise_texture.profile.default_view),
                        (35, &fbm_noise_texture.scheme_noise.default_view),
                        (36, &textures.weather_window_art.default_view),
                        (37, &textures.weather_window_physical.default_view),
                        (26, &textures.clouds_trace.default_view),
                        (27, &textures.clouds_trace_depth.default_view),
                        (44, &textures.cloud_field_sdf.default_view),
                        (48, &textures.cloud_field_esc.default_view),
                        (49, &fbm_noise_texture.cirrus.default_view),
                        (59, &textures.cloud_field_diff.default_view),
                    )),
                )
            });

        let clouds_reconstruct = clouds_raymarch
            .as_ref()
            .zip(temporal_uniform.and_then(|u| u.0.binding()))
            .zip(reference_accumulator)
            .map(|((_, params_binding), reference_accumulator)| {
                render_device.create_bind_group(
                    "clouds_reconstruct_bind_group",
                    &pipeline_cache.get_bind_group_layout(&layouts.clouds_reconstruct),
                    &BindGroupEntries::with_indices((
                        (0, atmosphere_binding.clone()),
                        (3, view_binding.clone()),
                        (12, &atmosphere_sampler.0),
                        (26, &textures.clouds_trace.default_view),
                        (27, &textures.clouds_trace_depth.default_view),
                        (28, &textures.clouds_history.default_view),
                        (29, &textures.clouds_depth_history.default_view),
                        (30, params_binding),
                        (31, &textures.clouds.default_view),
                        (32, &textures.clouds_depth.default_view),
                        (33, reference_accumulator.0.as_entire_binding()),
                    )),
                )
            });

        let cloud_field_edt = cloud_shadow_map.as_ref().map(|_| {
            let edt = |label: &'static str, src: &TextureView, dst: &TextureView| {
                render_device.create_bind_group(
                    label,
                    &pipeline_cache.get_bind_group_layout(&layouts.cloud_field_edt),
                    &BindGroupEntries::with_indices(((41, src), (42, dst))),
                )
            };
            [
                edt(
                    "cloud_field_edt_x_bind_group",
                    &textures.cloud_field_a.default_view,
                    &textures.cloud_field_b.default_view,
                ),
                edt(
                    "cloud_field_edt_y_bind_group",
                    &textures.cloud_field_b.default_view,
                    &textures.cloud_field_a.default_view,
                ),
                render_device.create_bind_group(
                    "cloud_field_edt_z_bind_group",
                    &pipeline_cache.get_bind_group_layout(&layouts.cloud_field_final),
                    &BindGroupEntries::with_indices((
                        (41, &textures.cloud_field_a.default_view),
                        (43, &textures.cloud_field_sdf.default_view),
                    )),
                ),
                render_device.create_bind_group(
                    "cloud_field_escape_bind_group",
                    &pipeline_cache.get_bind_group_layout(&layouts.cloud_field_escape),
                    &BindGroupEntries::with_indices((
                        (47, &textures.cloud_field_ext.default_view),
                        (46, &textures.cloud_field_esc.default_view),
                    )),
                ),
            ]
        });

        let cloud_field_diffusion = cloud_shadow_map
            .as_ref()
            .zip(cloud_layer_binding.clone())
            .map(|(_, cloud_layer_binding)| {
                render_device.create_bind_group(
                    "cloud_field_diffusion_bind_group",
                    &pipeline_cache.get_bind_group_layout(&layouts.cloud_field_diffusion),
                    &BindGroupEntries::with_indices((
                        (0, atmosphere_binding.clone()),
                        (1, settings_binding.clone()),
                        (3, view_binding.clone()),
                        (4, lights_binding.clone()),
                        (20, cloud_layer_binding),
                        (23, &textures.cloud_shadow_map.default_view),
                        (44, &textures.cloud_field_sdf.default_view),
                        (47, &textures.cloud_field_ext.default_view),
                        (55, &textures.cloud_field_diff_src.default_view),
                        (56, &textures.cloud_field_diff_u.default_view),
                        (57, &textures.cloud_field_diff.default_view),
                        (58, &textures.cloud_field_diff_tmp.default_view),
                        (60, &textures.cloud_field_diff_state.default_view),
                    )),
                )
            });

        let sky_view_lut_clouds = cloud_layer_binding
            .as_ref()
            .filter(|_| cloud_layer.is_some())
            .map(|cloud_layer_binding| {
                render_device.create_bind_group(
                    "sky_view_lut_clouds_bind_group",
                    &pipeline_cache.get_bind_group_layout(&layouts.sky_view_lut_clouds),
                    &BindGroupEntries::with_indices((
                        (0, atmosphere_binding.clone()),
                        (1, settings_binding.clone()),
                        (2, transforms_binding.clone()),
                        (3, view_binding.clone()),
                        (4, lights_binding.clone()),
                        (5, &gpu_medium.density_lut_view),
                        (6, &gpu_medium.scattering_lut_view),
                        (7, medium_sampler.sampler()),
                        (8, &textures.transmittance_lut.default_view),
                        (9, &textures.multiscattering_lut.default_view),
                        (12, &atmosphere_sampler.0),
                        (13, &textures.sky_view_lut.default_view),
                        (14, &shadow_bindings.point_light_depth_texture_view),
                        (15, &shadow_samplers.shadow_comparison_sampler),
                        (16, &shadow_bindings.directional_light_depth_texture_view),
                        (17, &shadow_samplers.shadow_comparison_sampler),
                        (20, cloud_layer_binding.clone()),
                        (21, &fbm_noise_texture.texture.default_view),
                        (22, &**cloud_noise_sampler),
                        (23, &textures.cloud_shadow_map.default_view),
                        (24, &perlin_worley_noise_texture.texture.default_view),
                        (25, stbn_view.as_ref()),
                        (33, &detail_noise_texture.texture.default_view),
                        (34, &fbm_noise_texture.profile.default_view),
                        (35, &fbm_noise_texture.scheme_noise.default_view),
                        (36, &textures.weather_window_art.default_view),
                        (37, &textures.weather_window_physical.default_view),
                    )),
                )
            });
        let aerial_view_lut_clouds = cloud_layer_binding
            .as_ref()
            .filter(|_| cloud_layer.is_some())
            .map(|cloud_layer_binding| {
                render_device.create_bind_group(
                    "aerial_view_lut_clouds_bind_group",
                    &pipeline_cache.get_bind_group_layout(&layouts.aerial_view_lut_clouds),
                    &BindGroupEntries::with_indices((
                        (0, atmosphere_binding.clone()),
                        (1, settings_binding.clone()),
                        (3, view_binding.clone()),
                        (4, lights_binding.clone()),
                        (5, &gpu_medium.density_lut_view),
                        (6, &gpu_medium.scattering_lut_view),
                        (7, medium_sampler.sampler()),
                        (8, &textures.transmittance_lut.default_view),
                        (9, &textures.multiscattering_lut.default_view),
                        (12, &atmosphere_sampler.0),
                        (13, &textures.aerial_view_lut.default_view),
                        (14, &shadow_bindings.point_light_depth_texture_view),
                        (15, &shadow_samplers.shadow_comparison_sampler),
                        (16, &shadow_bindings.directional_light_depth_texture_view),
                        (17, &shadow_samplers.shadow_comparison_sampler),
                        (20, cloud_layer_binding.clone()),
                        (21, &fbm_noise_texture.texture.default_view),
                        (22, &**cloud_noise_sampler),
                        (23, &textures.cloud_shadow_map.default_view),
                        (24, &perlin_worley_noise_texture.texture.default_view),
                        (25, stbn_view.as_ref()),
                        (33, &detail_noise_texture.texture.default_view),
                        (34, &fbm_noise_texture.profile.default_view),
                        (35, &fbm_noise_texture.scheme_noise.default_view),
                        (36, &textures.weather_window_art.default_view),
                        (37, &textures.weather_window_physical.default_view),
                    )),
                )
            });

        commands.entity(entity).insert(AtmosphereBindGroups {
            transmittance_lut,
            multiscattering_lut,
            sky_view_lut,
            aerial_view_lut,
            sky_view_lut_clouds,
            aerial_view_lut_clouds,
            cloud_shadow_map,
            cloud_shadow_filter_a_to_b,
            cloud_shadow_filter_b_to_a,
            cloud_shadow_temporal,
            render_sky_clouds,
            render_sky_haze,
            render_sky_no_clouds,
            clouds_raymarch,
            clouds_raymarch_multisampled: *msaa != Msaa::Off,
            clouds_reconstruct,
            cloud_field_edt,
            cloud_field_diffusion,
        });
    }

    Ok(())
}

#[derive(ShaderType)]
#[repr(C)]
pub(crate) struct AtmosphereData {
    pub atmosphere: GpuAtmosphere,
    pub settings: GpuAtmosphereSettings,
}

#[derive(Component)]
pub struct AtmosphereBuffer {
    pub(crate) buffer: StorageBuffer<AtmosphereData>,
}

pub(crate) fn prepare_atmosphere_buffers(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut views: Query<
        (
            Entity,
            &GpuAtmosphere,
            &GpuAtmosphereSettings,
            Option<&mut AtmosphereBuffer>,
            bevy_ecs::query::Has<GpuCloudLayers>,
        ),
        With<ExtractedAtmosphere>,
    >,
    mut commands: Commands,
) {
    for (entity, atmosphere, settings, existing_buffer, has_clouds) in &mut views {
        let mut settings = settings.clone();
        if !has_clouds {
            // PBR shading skips the cloud shadow map lookup when the extent is zero, so a
            // stale map from a removed `VolumetricCloudLayer` never darkens surfaces.
            settings.cloud_shadow_map_extent = 0.0;
        }
        let data = AtmosphereData {
            atmosphere: atmosphere.clone(),
            settings,
        };
        if let Some(mut atmosphere_buffer) = existing_buffer {
            atmosphere_buffer.buffer.set(data);
            atmosphere_buffer.buffer.write_buffer(&device, &queue);
        } else {
            let mut buffer = StorageBuffer::from(data);
            buffer.write_buffer(&device, &queue);
            commands.entity(entity).insert(AtmosphereBuffer { buffer });
        }
    }
}

/// Debug: `CLOUD_REFERENCE=1` swaps the realtime cloud march for a progressive volumetric path
/// tracer (delta/ratio tracking, full multiple scattering) over the same density field, for
/// ground-truth comparisons. Read once at pipeline creation.
///
/// Its scope (audit 34): the cumulus layer, the air and the planet's ground lit by the first
/// directional light, from a camera inside or outside the atmosphere. Not the thin cloud layer
/// (compare with `ATMOSPHERE_NO_CIRRUS=1`), not the sky light from other lights. Of the term
/// flags it honours only `CLOUD_TERMS=s1`, as single scattering of any medium (the first event
/// can be air or ground as well as cloud): mask non-cloud pixels before attributing differences
/// to the realtime cloud single scatter. `CLOUD_TERMS=sun|amb` and `CLOUD_NO_HAZE` have no
/// path-space definition here and are ignored, with a warning.
fn cloud_reference_def() -> Vec<bevy_shader::ShaderDefVal> {
    let mut defs: Vec<bevy_shader::ShaderDefVal> = Vec::new();
    if std::env::var("CLOUD_REFERENCE").is_ok() {
        defs.push("CLOUD_REFERENCE".into());
        if matches!(std::env::var("CLOUD_TERMS").as_deref(), Ok("sun" | "amb"))
            || std::env::var_os("CLOUD_NO_HAZE").is_some()
        {
            bevy_log::warn_once!(
                "CLOUD_REFERENCE ignores CLOUD_TERMS=sun|amb and CLOUD_NO_HAZE: the reference is \
                 the full image (or single scattering with CLOUD_TERMS=s1)"
            );
        }
    }
    // Clear-sky validation of the reference against the LUT sky (realtime: ATMOSPHERE_NO_CLOUDS).
    if std::env::var("CLOUD_REF_SEED2").is_ok() {
        defs.push("CLOUD_REF_SEED2".into());
    }
    if std::env::var("CLOUD_REF_QUAD_T").is_ok() {
        defs.push("CLOUD_REF_QUAD_T".into());
    }
    if std::env::var("CLOUD_REF_NO_CLOUDS").is_ok() {
        defs.push("CLOUD_REF_NO_CLOUDS".into());
    }
    defs.extend(atm_dbg_defs());
    defs
}

/// Debug: extra shader defs for the atmosphere shaders, comma separated
/// (`ATMOSPHERE_DBG=ATM_DBG_NOMS,ATM_DBG_TAP_ONE`). Supported debug defs:
/// - `ATM_DBG_NOSHADOW`: no cloud shadow on the atmosphere in-scatter.
/// - `ATM_DBG_NOMS`: no atmosphere multiple scattering (single scatter only).
/// - `ATM_DBG_NO_HAZE_HISTORY`: no temporal accumulation of the half-resolution haze.
/// - `ATM_DBG_GROUND_SKY_FIXED`: one fixed set of ground-skylight directions (no per-pixel/frame
///   jitter), to measure the jitter's temporal noise.
/// - `ATM_DBG_AP_BAL`: ballistic (non-similarity) aerial-perspective transmittance.
/// - `ATM_DBG_TAP_ONE`: no aerial-perspective transmittance on cloud luminance (term split).
/// - `ATM_DBG_NO_FAR_SHADOW`: drop the cloud shadow map beyond the local light march.
/// - `ATM_DBG_MIP0`: full-resolution cloud noise at any distance.
/// - `ATM_DBG_OUT_H`: write height-in-layer / profile / depth instead of cloud colour.
/// - `ATM_DBG_NO_SDF_SKIP`: disable cloud-SDF empty-space skipping.
/// - `ATM_DBG_NO_CLOUD_MOTION`: reproject the cloud history without the wind displacement.
/// - `ATM_DBG_NO_THIN_SHADOW`: no thin-layer (cirrus) shadow in the cloud shadow map's alpha.
/// - `ATM_DBG_ESC_FALLBACK`: the out-of-field escape/column estimate everywhere (validation).
/// - `ATM_DBG_ESC_FIELD_ONLY`: the cloud field's escape probability only (no local estimate).
/// - `ATM_DBG_ESC_ZERO`: zero escape probability (the infinitely thick multiple-scattering limit).
/// - `ATM_DBG_SHADOW_JITTER`: rotate the cloud shadow map's per-step sample phases per texel and
///   frame instead of the fixed phases, and accumulate optical depth (audit 1 validation: after
///   convergence half the mean error and 40 % of the edge error of the fixed phases, with
///   frame-to-frame noise of 0.02 in edge transmittance).
/// - `ATM_DBG_STEP_DIV2`, `ATM_DBG_STEP_DIV4`, `ATM_DBG_STEP_DIV16`: a 2x / 4x / 16x shorter
///   cloud view step (with `ATM_DBG_NO_COARSE`: the step-size convergence of the view march,
///   audit 24).
/// - `ATM_DBG_MS_DIFFUSION`: add the delta-Eddington vertical-slab diffusion term to the multiple
///   scattering in thick cloud (experiment: shaded sides right, lit sides still short).
/// - `ATM_DBG_MS_SERIES_ONLY`: the escape series alone for multiple scattering (without the cloud
///   field's diffusion solve).
/// - `ATM_DBG_LIGHT_BRUTE`: a dense uniform light march (256 x 30 m) instead of the growing taps.
/// - `ATM_DBG_OUT_SUN_OD`: write (local sun optical depth, the cloud field's diffusion solution,
///   its escape probability, density) at each pixel's first in-cloud sample instead of cloud colour.
/// - `ATM_DBG_OUT_SUN_TAU`: write (sun optical depth of the light taps plus the map beyond, the
///   cloud shadow map's alone at the sample, the diffusion solution, density) there instead.
/// - `ATM_DBG_REF_CAPS`: with `CLOUD_REFERENCE`, the fraction of samples that reached the
///   tracer's bounce (r), flight segment (g) and sun-ray (b) budgets, and that continued past a
///   segment's null-collision budget (a, exact).
/// - `ATM_DBG_NO_COARSE`: fine samples everywhere from the ray start, no step budget (with `ATM_DBG_NO_SDF_SKIP`:
///   the ground-truth march for validating empty-space skipping).
/// See also `CLOUD_TERMS=s1|sun|amb` and `CLOUD_REFERENCE=1`.
fn atm_dbg_defs() -> Vec<bevy_shader::ShaderDefVal> {
    std::env::var("ATMOSPHERE_DBG")
        .map(|v| {
            v.split(',')
                .filter(|d| !d.is_empty())
                .map(|d| d.to_string().into())
                .collect()
        })
        .unwrap_or_default()
}

/// Debug: lighting-term isolation for validation (realtime and reference alike).
/// `CLOUD_TERMS=s1` sun single scatter only, `sun` sun incl. multiple scattering, `amb` sky+ground
/// only; `CLOUD_NO_HAZE=1` skips the aerial perspective in front of the clouds.
fn cloud_term_defs() -> Vec<bevy_shader::ShaderDefVal> {
    let mut defs: Vec<bevy_shader::ShaderDefVal> = Vec::new();
    match std::env::var("CLOUD_TERMS").as_deref() {
        Ok("s1") => defs.extend(["CLOUD_TERM_S1".into(), "CLOUD_TERM_NO_AMB".into()]),
        Ok("sun") => defs.push("CLOUD_TERM_NO_AMB".into()),
        Ok("amb") => defs.push("CLOUD_TERM_NO_SUN".into()),
        _ => {}
    }
    if std::env::var("CLOUD_NO_HAZE").is_ok() {
        defs.push("CLOUD_NO_HAZE".into());
    }
    defs
}

/// f32 running-sum accumulator of a view for the reference path tracer (`CLOUD_REFERENCE`):
/// a header vec4 (x: the frame it was started, as u32 bits), then 2 x vec4 per half-res pixel:
/// sum(rgb, T), (count, first-hit distance sum, 0, 0). The 16-bit cloud history cannot hold a
/// true running mean (increments drop below the f16 ulp after ~1-2k samples). 32 bytes when
/// `CLOUD_REFERENCE` is off.
#[derive(Component)]
pub struct CloudReferenceAccumulator(pub Buffer);

/// One accumulator per view, restarted (a new, zeroed buffer) whenever anything in the integrand
/// changes: the view, its size, the lights, exposure, atmosphere, medium, cloud layers or
/// settings (audit 32). The average was process-global and keyed only by its byte size, so views
/// of equal size shared sums, and a changed scene kept averaging into the old one.
pub(super) fn prepare_cloud_reference_accumulators(
    views: Query<(
        Entity,
        &ExtractedView,
        Option<&ExtractedCamera>,
        &AtmosphereTextures,
        &ExtractedAtmosphere,
        Option<&GpuCloudLayers>,
        Option<&GpuAtmosphereSettings>,
    )>,
    lights: Query<&ExtractedDirectionalLight>,
    gpu_media: Res<RenderAssets<GpuScatteringMedium>>,
    frame_count: Res<FrameCount>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut accumulators: bevy_ecs::system::Local<HashMap<RetainedViewEntity, (u64, Buffer)>>,
    mut placeholder: bevy_ecs::system::Local<Option<Buffer>>,
    mut commands: Commands,
) {
    use core::hash::{Hash, Hasher};
    let create = |bytes: u64| {
        render_device.create_buffer(&BufferDescriptor {
            label: Some("cloud_reference_accumulator"),
            size: bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    if std::env::var_os("CLOUD_REFERENCE").is_none() {
        let buffer = placeholder.get_or_insert_with(|| create(32)).clone();
        for (entity, ..) in &views {
            commands
                .entity(entity)
                .insert(CloudReferenceAccumulator(buffer.clone()));
        }
        return;
    }
    fn shader_bytes<T: ShaderType + encase::internal::WriteInto>(value: &T) -> Vec<u8> {
        let mut buffer = encase::UniformBuffer::new(Vec::new());
        buffer.write(value).ok();
        buffer.into_inner()
    }
    // The lights in any order.
    let lights_key = lights.iter().fold(0u64, |acc, light| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for f in light.transform.to_matrix().to_cols_array() {
            f.to_bits().hash(&mut h);
        }
        for f in [
            light.color.red,
            light.color.green,
            light.color.blue,
            light.color.alpha,
        ] {
            f.to_bits().hash(&mut h);
        }
        light.illuminance.to_bits().hash(&mut h);
        acc ^ h.finish()
    });
    let mut seen = HashMap::default();
    for (entity, view, camera, textures, atmosphere, cloud_layers, settings) in &views {
        let size = textures.clouds.texture.size();
        let bytes = 16 + size.width as u64 * size.height as u64 * 32;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        lights_key.hash(&mut h);
        bytes.hash(&mut h);
        for f in view
            .world_from_view
            .to_matrix()
            .to_cols_array()
            .into_iter()
            .chain(view.clip_from_view.to_cols_array())
            .chain([camera.map_or(1.0, |c| c.exposure)])
            .chain([atmosphere.inner_radius, atmosphere.outer_radius])
            .chain(atmosphere.ground_albedo.to_array())
            .chain(atmosphere.world_to_atmosphere.to_cols_array())
        {
            f.to_bits().hash(&mut h);
        }
        atmosphere.medium.hash(&mut h);
        if let Some(medium) = gpu_media.get(atmosphere.medium) {
            medium.density_lut.id().hash(&mut h);
            medium.scattering_lut.id().hash(&mut h);
        }
        cloud_layers.map(shader_bytes).hash(&mut h);
        settings.map(shader_bytes).hash(&mut h);
        let key = h.finish();
        let buffer = match accumulators.get(&view.retained_view_entity) {
            Some((k, buffer)) if *k == key => buffer.clone(),
            _ => {
                let buffer = create(bytes);
                render_queue.write_buffer(
                    &buffer,
                    0,
                    bytemuck::cast_slice(&[frame_count.0, 0, 0, 0]),
                );
                super::debug_dump::note(&format!(
                    "reference accumulator (re)started at frame {} for view {:?}",
                    frame_count.0, view.retained_view_entity
                ));
                buffer
            }
        };
        if cloud_layers.is_some_and(|c| c.thin_count > 0) {
            bevy_log::warn_once!(
                "CLOUD_REFERENCE traces the cumulus layer and air only: the thin cloud layer \
                 (cirrus) is not in the reference image; compare without it"
            );
        }
        commands
            .entity(entity)
            .insert(CloudReferenceAccumulator(buffer.clone()));
        seen.insert(view.retained_view_entity, (key, buffer));
    }
    *accumulators = seen;
}

/// Size of the cloud and haze buffers for a viewport: half of it, scaled down (same aspect) to
/// fit within `max` (`VolumetricCloudSettings::max_render_size`, compared by pixel count so
/// portrait and landscape views get the same budget).
pub fn cloud_render_size(full: UVec2, max: UVec2) -> UVec2 {
    let half = (full / 2).max(UVec2::ONE);
    let budget = (max.x as f64) * (max.y as f64);
    let pixels = (half.x as f64) * (half.y as f64);
    if pixels <= budget {
        return half;
    }
    // Even, so the checkerboard trace is exactly half of it (`clouds_raymarch.wesl` recovers
    // the size as min(viewport / 2, 2 x trace)).
    let s = (budget / pixels).sqrt();
    UVec2::new(
        (((half.x as f64 * s).floor() as u32) & !1).max(2),
        (((half.y as f64 * s).floor() as u32) & !1).max(2),
    )
}
