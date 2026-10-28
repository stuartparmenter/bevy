//! Atmosphere debug view: live previews of the atmosphere's lookup tables, cloud buffers, cloud
//! field and noise, for a debug overlay ([`AtmosphereDebugViewPlugin`]).
//!
//! Each [`AtmosphereDebugPanel`] names a source texture and how to display it, and owns an
//! [`Image`] (Rgba8Unorm) that a compute pass (`debug_view.wesl`) rewrites every frame from the
//! render world's texture: put the image in a UI `ImageNode` to see it. No readback; the images
//! show the frame before the one they are drawn in. The view of the lowest-order camera with an
//! atmosphere is shown.

use super::{
    detail_noise::DetailNoiseTexture, fbm_noise::FbmNoiseTexture,
    perlin_worley_noise::PerlinWorleyNoiseTexture, resources::AtmosphereTextures,
};
use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, Assets, Handle, RenderAssetUsages};
use bevy_ecs::{
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Query, Res},
    world::World,
};
use bevy_image::Image;
use bevy_render::{
    camera::ExtractedCamera,
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{render_system, RenderDevice, RenderQueue},
    texture::GpuImage,
    Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::ShaderDefVal;
use bevy_utils::default;

/// Edge of the debug images (texels); sources are fitted into them with their aspect kept.
pub const ATMOSPHERE_DEBUG_IMAGE_SIZE: u32 = 256;

/// A texture the debug view can show.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtmosphereDebugSource {
    /// Transmittance LUT: RGB optical depth (decode with [`DebugDecode::ExpNeg`] for the
    /// transmittance), A the reducible Mie optical depth. x: view zenith, y: altitude.
    TransmittanceLut,
    /// Multiple-scattering LUT: log of psi_ms (RGB). x: sun zenith, y: altitude.
    MultiscatteringLut,
    /// Sky-view LUT: pre-exposed in-scattered radiance (RGB). x: azimuth, y: zenith.
    SkyViewLut,
    /// Aerial-perspective volume: log of pre-exposed in-scattering (RGB), z = distance slice.
    /// Built only in [`AtmosphereMode::LookupTexture`](crate::AtmosphereMode) (raymarched mode
    /// marches every pixel's haze instead), so it is stale in raymarched mode.
    AerialViewLut,
    /// Cloud shadow map: R front depth (km), G mean extinction (1/m), B max optical depth, A the
    /// thin layers' optical depth.
    CloudShadowMap,
    /// Reconstructed half-resolution clouds: RGB pre-exposed radiance, A transmittance.
    Clouds,
    /// Transmittance-weighted cloud distance (m).
    CloudsDepth,
    /// This frame's checkerboard trace (a quarter of the half-res pixels).
    CloudsTrace,
    /// Half-resolution haze: in-scattering before exposure (RGB, physical units).
    HazeInscattering,
    /// Half-resolution haze: transmittance (RGB), ray length in km (A).
    HazeTransmittance,
    /// Cloud field signed distance (R, m); z = height in the layer.
    CloudFieldSdf,
    /// Cloud field extinction (R, 1/m), voxel height (G).
    CloudFieldExtinction,
    /// Escape probability of multiply scattered light (R), vertical optical depths to the layer
    /// top (G) and bottom (B); half horizontal resolution.
    CloudFieldEscape,
    /// Diffuse sunlight of the cloud field per unit sun irradiance (R).
    CloudFieldDiffusion,
    /// Procedural weather map tile: coverage, cloud type, density scale, mesoscale fraction.
    WeatherMap,
    /// Weather window around the camera, art channels (as [`Self::WeatherMap`]).
    WeatherWindowArt,
    /// Weather window, physical channels: cloud-fraction scale, air mass, droplet number,
    /// base offset.
    WeatherWindowPhysical,
    /// Vertical profile lookups: x height above the base, y cloud type (R, G presence factors,
    /// B liquid water, A base), then the noise-distribution and cloud-scheme rows.
    CloudProfiles,
    /// Shape noise (Perlin-Worley; R the base shape), z = the tile's third axis.
    ShapeNoise,
    /// Detail noise (Worley octaves).
    DetailNoise,
    /// Physical cloud scheme fluctuation: R the Gaussianized shape noise g, B the bubbles b.
    SchemeNoise,
    /// Cirrus generating field (unit Gaussian, R).
    CirrusField,
}

/// The channels a panel shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DebugChannels {
    #[default]
    Rgb,
    R,
    G,
    B,
    A,
    /// R x G (the profile lookups' presence).
    RTimesG,
}

/// How the stored value is decoded before display.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DebugDecode {
    #[default]
    Identity,
    /// The texture stores a log: show exp(x).
    Exp,
    /// The texture stores an optical depth: show the transmittance exp(-x).
    ExpNeg,
}

/// How the decoded value maps to brightness.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DebugMapping {
    /// value x scale + bias, clamped to [0, 1].
    #[default]
    Linear,
    /// log2(value x scale) over 12 stops, white at 2^2: for values over decades (optical
    /// depth, extinction).
    Log,
    /// x / (1 + x), x = value x scale: for (pre-exposed) radiance.
    Tonemap,
}

/// One preview: a source texture, its display mapping, and the image it is drawn into.
#[derive(Clone, Debug)]
pub struct AtmosphereDebugPanel {
    pub label: String,
    pub source: AtmosphereDebugSource,
    pub channels: DebugChannels,
    pub decode: DebugDecode,
    pub mapping: DebugMapping,
    pub scale: f32,
    pub bias: f32,
    /// 3D sources: the z slice as a fraction of the depth.
    pub slice: f32,
    pub mip: u32,
    pub image: Handle<Image>,
}

impl AtmosphereDebugPanel {
    /// A panel with an identity, linear, unit-scale RGB display of `source`, drawing into a new
    /// image; adjust the display fields after.
    pub fn new(
        images: &mut Assets<Image>,
        label: impl Into<String>,
        source: AtmosphereDebugSource,
    ) -> Self {
        Self {
            label: label.into(),
            source,
            channels: DebugChannels::Rgb,
            decode: DebugDecode::Identity,
            mapping: DebugMapping::Linear,
            scale: 1.0,
            bias: 0.0,
            slice: 0.5,
            mip: 0,
            image: images.add(debug_image()),
        }
    }

    pub fn channels(mut self, channels: DebugChannels) -> Self {
        self.channels = channels;
        self
    }

    pub fn decode(mut self, decode: DebugDecode) -> Self {
        self.decode = decode;
        self
    }

    /// Log display (see [`DebugMapping::Log`]) with `scale` applied first.
    pub fn log(mut self, scale: f32) -> Self {
        self.mapping = DebugMapping::Log;
        self.scale = scale;
        self
    }

    /// Tonemapped display (see [`DebugMapping::Tonemap`]) with `scale` applied first.
    pub fn tonemap(mut self, scale: f32) -> Self {
        self.mapping = DebugMapping::Tonemap;
        self.scale = scale;
        self
    }

    /// Linear display, value x scale + bias.
    pub fn linear(mut self, scale: f32, bias: f32) -> Self {
        self.mapping = DebugMapping::Linear;
        self.scale = scale;
        self.bias = bias;
        self
    }
}

/// The panels to draw. Nothing runs while `enabled` is false.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct AtmosphereDebugView {
    pub enabled: bool,
    pub panels: Vec<AtmosphereDebugPanel>,
}

impl AtmosphereDebugView {
    /// Every texture with a display suited to its contents.
    pub fn all(images: &mut Assets<Image>) -> Self {
        use AtmosphereDebugSource as S;
        use DebugChannels as C;
        let mut p = |label: &str, source| AtmosphereDebugPanel::new(images, label, source);
        let panels = vec![
            p("transmittance LUT", S::TransmittanceLut).decode(DebugDecode::ExpNeg),
            p("multiple scattering LUT", S::MultiscatteringLut)
                .decode(DebugDecode::Exp)
                .log(64.0),
            p("sky-view LUT", S::SkyViewLut).tonemap(1.0),
            p("aerial persp. (slice; LUT mode)", S::AerialViewLut)
                .decode(DebugDecode::Exp)
                .tonemap(1.0),
            p("shadow map: optical depth", S::CloudShadowMap)
                .channels(C::B)
                .log(1.0 / 16.0),
            p("shadow map: front depth (km)", S::CloudShadowMap)
                .channels(C::R)
                .linear(1.0 / 256.0, 0.0),
            p("clouds (half-res)", S::Clouds).tonemap(1.0),
            p("clouds: transmittance", S::Clouds).channels(C::A),
            p("clouds: trace (this frame)", S::CloudsTrace).tonemap(1.0),
            p("haze in-scattering (unexposed)", S::HazeInscattering).log(1.0 / 4096.0),
            p("field SDF (slice)", S::CloudFieldSdf)
                .channels(C::R)
                .linear(1.0 / 3000.0, 0.0),
            p("field extinction (slice)", S::CloudFieldExtinction)
                .channels(C::R)
                .log(20.0),
            p("field escape prob. (slice)", S::CloudFieldEscape).channels(C::R),
            p("field diffusion (slice)", S::CloudFieldDiffusion)
                .channels(C::R)
                .log(1.0),
            p("weather: coverage", S::WeatherMap).channels(C::R),
            p("weather: air mass", S::WeatherWindowPhysical).channels(C::G),
            p("profiles: fraction", S::CloudProfiles).channels(C::RTimesG),
            p("shape noise (slice)", S::ShapeNoise).channels(C::R),
            p("scheme noise g (slice)", S::SchemeNoise)
                .channels(C::R)
                .linear(0.25, 0.5),
            p("cirrus field", S::CirrusField)
                .channels(C::R)
                .linear(0.25, 0.5),
        ];
        Self {
            enabled: false,
            panels,
        }
    }
}

/// A blank debug image, writable by the debug pass and sampled by the UI.
pub fn debug_image() -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: ATMOSPHERE_DEBUG_IMAGE_SIZE,
            height: ATMOSPHERE_DEBUG_IMAGE_SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[13, 13, 13, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    image
}

/// Draws the [`AtmosphereDebugView`] panels each frame (add with the atmosphere plugin).
pub struct AtmosphereDebugViewPlugin;

impl Plugin for AtmosphereDebugViewPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "debug_view.wesl");
        app.init_resource::<AtmosphereDebugView>()
            .add_plugins(ExtractResourcePlugin::<AtmosphereDebugView>::default());
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_debug_view_pipelines)
            .add_systems(
                Render,
                draw_debug_view
                    .in_set(RenderSystems::Render)
                    .after(render_system),
            );
    }
}

#[derive(Resource)]
struct DebugViewPipelines {
    layout_2d: BindGroupLayoutDescriptor,
    layout_3d: BindGroupLayoutDescriptor,
    pipeline_2d: CachedComputePipelineId,
    pipeline_3d: CachedComputePipelineId,
}

#[derive(Clone, Copy, ShaderType)]
struct DebugParams {
    channels: u32,
    decode: u32,
    mapping: u32,
    mip: u32,
    scale: f32,
    bias: f32,
    slice: f32,
    _pad: f32,
}

fn init_debug_view_pipelines(world: &mut World) {
    let layout = |name: &'static str, source: BindGroupLayoutEntryBuilder| {
        BindGroupLayoutDescriptor::new(
            name,
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    source,
                    texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
                    uniform_buffer::<DebugParams>(false),
                ),
            ),
        )
    };
    let unfilterable = TextureSampleType::Float { filterable: false };
    let layout_2d = layout("atmosphere_debug_view_2d_layout", texture_2d(unfilterable));
    let layout_3d = layout("atmosphere_debug_view_3d_layout", texture_3d(unfilterable));
    let shader = load_embedded_asset!(world, "debug_view.wesl");
    let pipeline_cache = world.resource::<PipelineCache>();
    let queue =
        |label: &'static str, layout: &BindGroupLayoutDescriptor, defs: Vec<ShaderDefVal>| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(label.into()),
                layout: vec![layout.clone()],
                shader: shader.clone(),
                shader_defs: defs,
                entry_point: Some("blit".into()),
                ..default()
            })
        };
    let pipeline_2d = queue("atmosphere_debug_view_2d", &layout_2d, vec![]);
    let pipeline_3d = queue(
        "atmosphere_debug_view_3d",
        &layout_3d,
        vec!["SOURCE_3D".into()],
    );
    world.insert_resource(DebugViewPipelines {
        layout_2d,
        layout_3d,
        pipeline_2d,
        pipeline_3d,
    });
}

fn draw_debug_view(
    view: Option<Res<AtmosphereDebugView>>,
    pipelines: Option<Res<DebugViewPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    images: Res<RenderAssets<GpuImage>>,
    views: Query<(&AtmosphereTextures, Option<&ExtractedCamera>)>,
    fbm: Option<Res<FbmNoiseTexture>>,
    shape: Option<Res<PerlinWorleyNoiseTexture>>,
    detail: Option<Res<DetailNoiseTexture>>,
) {
    let (Some(view), Some(pipelines)) = (view, pipelines) else {
        return;
    };
    if !view.enabled || view.panels.is_empty() {
        return;
    }
    let (Some(pipeline_2d), Some(pipeline_3d)) = (
        pipeline_cache.get_compute_pipeline(pipelines.pipeline_2d),
        pipeline_cache.get_compute_pipeline(pipelines.pipeline_3d),
    ) else {
        return;
    };
    let textures = views
        .iter()
        .min_by_key(|(_, camera)| camera.map_or(isize::MAX, |c| c.order))
        .map(|(textures, _)| textures);
    let layout_2d = pipeline_cache.get_bind_group_layout(&pipelines.layout_2d);
    let layout_3d = pipeline_cache.get_bind_group_layout(&pipelines.layout_3d);

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("atmosphere_debug_view"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("atmosphere_debug_view"),
            timestamp_writes: None,
        });
        for panel in &view.panels {
            use AtmosphereDebugSource as S;
            // (view, is 3D)
            let source: Option<(&TextureView, bool)> = match panel.source {
                S::TransmittanceLut => textures.map(|t| (&t.transmittance_lut.default_view, false)),
                S::MultiscatteringLut => {
                    textures.map(|t| (&t.multiscattering_lut.default_view, false))
                }
                S::SkyViewLut => textures.map(|t| (&t.sky_view_lut.default_view, false)),
                S::AerialViewLut => textures.map(|t| (&t.aerial_view_lut.default_view, true)),
                S::CloudShadowMap => textures.map(|t| (&t.cloud_shadow_map.default_view, false)),
                S::Clouds => textures.map(|t| (&t.clouds.default_view, false)),
                S::CloudsDepth => textures.map(|t| (&t.clouds_depth.default_view, false)),
                S::CloudsTrace => textures.map(|t| (&t.clouds_trace.default_view, false)),
                S::HazeInscattering => textures.map(|t| (&t.haze_inscattering.default_view, false)),
                S::HazeTransmittance => {
                    textures.map(|t| (&t.haze_transmittance.default_view, false))
                }
                S::CloudFieldSdf => textures.map(|t| (&t.cloud_field_sdf.default_view, true)),
                S::CloudFieldExtinction => {
                    textures.map(|t| (&t.cloud_field_ext.default_view, true))
                }
                S::CloudFieldEscape => textures.map(|t| (&t.cloud_field_esc.default_view, true)),
                S::CloudFieldDiffusion => {
                    textures.map(|t| (&t.cloud_field_diff.default_view, true))
                }
                S::WeatherWindowArt => {
                    textures.map(|t| (&t.weather_window_art.default_view, false))
                }
                S::WeatherWindowPhysical => {
                    textures.map(|t| (&t.weather_window_physical.default_view, false))
                }
                S::WeatherMap => fbm.as_ref().map(|f| (&f.texture.default_view, false)),
                S::CloudProfiles => fbm.as_ref().map(|f| (&f.profile.default_view, false)),
                S::SchemeNoise => fbm.as_ref().map(|f| (&f.scheme_noise.default_view, true)),
                S::CirrusField => fbm.as_ref().map(|f| (&f.cirrus.default_view, false)),
                S::ShapeNoise => shape.as_ref().map(|s| (&s.texture.default_view, true)),
                S::DetailNoise => detail.as_ref().map(|d| (&d.texture.default_view, true)),
            };
            let (Some((source, is_3d)), Some(target)) = (source, images.get(&panel.image)) else {
                continue;
            };
            let params = DebugParams {
                channels: panel.channels as u32,
                decode: panel.decode as u32,
                mapping: panel.mapping as u32,
                mip: panel.mip,
                scale: panel.scale,
                bias: panel.bias,
                slice: panel.slice.clamp(0.0, 1.0),
                _pad: 0.0,
            };
            let mut uniform = UniformBuffer::from(params);
            uniform.write_buffer(&render_device, &render_queue);
            let Some(params_binding) = uniform.binding() else {
                continue;
            };
            let bind_group = render_device.create_bind_group(
                "atmosphere_debug_view_bind_group",
                if is_3d { &layout_3d } else { &layout_2d },
                &BindGroupEntries::sequential((source, &target.texture_view, params_binding)),
            );
            pass.set_pipeline(if is_3d { pipeline_3d } else { pipeline_2d });
            pass.set_bind_group(0, &bind_group, &[]);
            let n = ATMOSPHERE_DEBUG_IMAGE_SIZE.div_ceil(8);
            pass.dispatch_workgroups(n, n, 1);
        }
    }
    render_queue.submit([encoder.finish()]);
}
