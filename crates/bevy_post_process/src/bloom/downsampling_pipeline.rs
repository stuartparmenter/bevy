use bevy_core_pipeline::FullscreenShader;

use super::{bloom_texture_format, Bloom};
use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_ecs::{
    prelude::{Component, Entity},
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_math::{Vec2, Vec4};
use bevy_render::{
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    renderer::RenderDevice,
    view::ViewDisplayTarget,
};
use bevy_shader::Shader;
use bevy_utils::{default, once};
use tracing::warn;

#[derive(Component)]
pub struct BloomDownsamplingPipelineIds {
    pub main: CachedRenderPipelineId,
    pub first: CachedRenderPipelineId,
}

#[derive(Resource)]
pub struct BloomDownsamplingPipeline {
    /// Layout with a texture, a sampler, and uniforms
    pub bind_group_layout: BindGroupLayoutDescriptor,
    pub sampler: Sampler,
    /// The asset handle for the fullscreen vertex shader.
    pub fullscreen_shader: FullscreenShader,
    /// The fragment shader asset handle.
    pub fragment_shader: Handle<Shader>,
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct BloomDownsamplingPipelineKeys {
    prefilter: bool,
    first_downsample: bool,
    uniform_scale: bool,
    /// The bloom pyramid format the pass renders into: `Rg11b10Ufloat` for
    /// views on SDR display targets, `Rgba16Float` for views whose resolved
    /// display target transfer is HDR (see [`bloom_texture_format`]).
    texture_format: TextureFormat,
}

/// The uniform struct built from [`Bloom`] attached to a Camera by
/// `prepare_bloom_uniforms`. Will be available for use in the Bloom shader.
#[derive(Component, ShaderType, Clone)]
pub struct BloomUniforms {
    // Precomputed values used when thresholding, see https://catlikecoding.com/unity/tutorials/advanced-rendering/bloom/#3.4
    pub threshold_precomputations: Vec4,
    pub viewport: Vec4,
    pub scale: Vec2,
    pub aspect: f32,
}

impl BloomUniforms {
    /// Packs the soft-knee threshold curve parameters consumed by
    /// `soft_threshold` in `bloom.wgsl`
    /// (see <https://catlikecoding.com/unity/tutorials/advanced-rendering/bloom/#3.4>).
    ///
    /// `threshold` is in scene-linear framebuffer units; `threshold_softness`
    /// is clamped to `[0, 1]`. This is the single source of the packing math,
    /// consumed by `prepare_bloom_uniforms`.
    pub(crate) fn threshold_precomputations(threshold: f32, threshold_softness: f32) -> Vec4 {
        let knee = threshold * threshold_softness.clamp(0.0, 1.0);
        Vec4::new(
            threshold,
            threshold - knee,
            2.0 * knee,
            0.25 / (knee + 0.00001),
        )
    }
}

pub fn init_bloom_downsampling_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
) {
    // Bind group layout
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "bloom_downsampling_bind_group_layout_with_settings",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                // Input texture binding
                texture_2d(TextureSampleType::Float { filterable: true }),
                // Sampler binding
                sampler(SamplerBindingType::Filtering),
                // Downsampling settings binding
                uniform_buffer::<BloomUniforms>(true),
            ),
        ),
    );

    // Sampler
    let sampler = render_device.create_sampler(&SamplerDescriptor {
        min_filter: FilterMode::Linear,
        mag_filter: FilterMode::Linear,
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        ..Default::default()
    });

    commands.insert_resource(BloomDownsamplingPipeline {
        bind_group_layout,
        sampler,
        fullscreen_shader: fullscreen_shader.clone(),
        fragment_shader: load_embedded_asset!(asset_server.as_ref(), "bloom.wgsl"),
    });
}

impl SpecializedRenderPipeline for BloomDownsamplingPipeline {
    type Key = BloomDownsamplingPipelineKeys;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let layout = vec![self.bind_group_layout.clone()];

        let entry_point = if key.first_downsample {
            "downsample_first".into()
        } else {
            "downsample".into()
        };

        let mut shader_defs = vec![];

        if key.first_downsample {
            shader_defs.push("FIRST_DOWNSAMPLE".into());
        }

        if key.prefilter {
            shader_defs.push("USE_THRESHOLD".into());
        }

        if key.uniform_scale {
            shader_defs.push("UNIFORM_SCALE".into());
        }

        RenderPipelineDescriptor {
            label: Some(
                if key.first_downsample {
                    "bloom_downsampling_pipeline_first"
                } else {
                    "bloom_downsampling_pipeline"
                }
                .into(),
            ),
            layout,
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                shader_defs,
                entry_point: Some(entry_point),
                targets: vec![Some(ColorTargetState {
                    format: key.texture_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                constants: vec![],
            }),
            ..default()
        }
    }
}

pub fn prepare_downsampling_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<BloomDownsamplingPipeline>>,
    pipeline: Res<BloomDownsamplingPipeline>,
    views: Query<(Entity, &Bloom, Option<&ViewDisplayTarget>)>,
) {
    for (entity, bloom, display_target) in &views {
        // `Bloom::thresholding_active` (not `BloomPrefilter::is_active`):
        // the GT7 glare scatter model is threshold-free by construction.
        let prefilter = bloom.thresholding_active();
        if !prefilter && bloom.prefilter.is_active() {
            once!(warn!(
                "Bloom prefilter thresholds are ignored under BloomScatterModel::Gt7Glare: \
                a physical glare PSF integrates the total scene energy"
            ));
        }
        let texture_format = bloom_texture_format(display_target);

        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            BloomDownsamplingPipelineKeys {
                prefilter,
                first_downsample: false,
                uniform_scale: bloom.scale == Vec2::ONE,
                texture_format,
            },
        );

        let pipeline_first_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            BloomDownsamplingPipelineKeys {
                prefilter,
                first_downsample: true,
                uniform_scale: bloom.scale == Vec2::ONE,
                texture_format,
            },
        );

        commands
            .entity(entity)
            .insert(BloomDownsamplingPipelineIds {
                first: pipeline_first_id,
                main: pipeline_id,
            });
    }
}
