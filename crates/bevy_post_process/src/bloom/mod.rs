mod downsampling_pipeline;
mod glare;
mod settings;
mod upsampling_pipeline;

use bevy_image::ToExtents;
pub use settings::{Bloom, BloomCompositeMode, BloomPrefilter, BloomScatterModel};

use crate::bloom::{
    downsampling_pipeline::init_bloom_downsampling_pipeline,
    upsampling_pipeline::init_bloom_upscaling_pipeline,
};
use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_color::{Gray, LinearRgba};
use bevy_core_pipeline::{
    schedule::{Core2d, Core2dSystems, Core3d, Core3dSystems},
    tonemapping::tonemapping,
};
use bevy_ecs::prelude::*;
use bevy_math::{ops, AspectRatio, UVec2, UVec4};
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    extract_component::{
        ComponentUniforms, DynamicUniformIndex, ExtractComponentPlugin, UniformComponentPlugin,
    },
    render_resource::*,
    renderer::{RenderContext, RenderDevice, ViewQuery},
    texture::{CachedTexture, TextureCache},
    view::{ExtractedView, ViewDisplayTarget, ViewTarget},
    GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use downsampling_pipeline::{
    prepare_downsampling_pipeline, BloomDownsamplingPipeline, BloomDownsamplingPipelineIds,
    BloomUniforms,
};
use upsampling_pipeline::{
    prepare_upsampling_pipeline, BloomUpsamplingPipeline, UpsamplingPipelineIds,
};

/// The bloom pyramid format for views on SDR display targets. `Rg11b10Ufloat`
/// halves the memory and bandwidth of `Rgba16Float`, and its range
/// (~`[6.1e-5, 65024]`, no sign bit, no alpha) covers scene-linear input.
const BLOOM_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rg11b10Ufloat;

/// The bloom pyramid format for views whose display target has an HDR transfer.
///
/// Above-paper-white content reaches an HDR display, where `Rg11b10Ufloat`'s
/// coarse mantissa above 1.0 bands visibly in the Karis-averaged downsample
/// sums. `Rgba16Float` has uniform precision there at twice the memory cost:
/// ~5 MB/frame vs ~2.5 MB/frame at 4K with the default 8-level chain.
const BLOOM_TEXTURE_FORMAT_HDR: TextureFormat = TextureFormat::Rgba16Float;

/// Returns the bloom pyramid texture format for a view, keyed on its display
/// target transfer. Used by [`prepare_bloom_textures`] and both pipeline
/// specializations so they cannot disagree about the format. A missing
/// [`ViewDisplayTarget`] (a view never extracted as a camera) means SDR.
pub(crate) fn bloom_texture_format(display_target: Option<&ViewDisplayTarget>) -> TextureFormat {
    if display_target.is_some_and(ViewDisplayTarget::is_hdr_transfer) {
        BLOOM_TEXTURE_FORMAT_HDR
    } else {
        BLOOM_TEXTURE_FORMAT
    }
}

#[derive(Default)]
pub struct BloomPlugin;

impl Plugin for BloomPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "bloom.wesl");

        app.add_plugins((
            ExtractComponentPlugin::<Bloom>::default(),
            UniformComponentPlugin::<BloomUniforms>::default(),
        ));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_gpu_resource::<SpecializedRenderPipelines<BloomDownsamplingPipeline>>()
            .init_gpu_resource::<SpecializedRenderPipelines<BloomUpsamplingPipeline>>()
            .add_systems(
                RenderStartup,
                (
                    init_bloom_downsampling_pipeline,
                    init_bloom_upscaling_pipeline,
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_bloom_uniforms
                        .in_set(RenderSystems::Prepare)
                        .before(RenderSystems::PrepareResources),
                    prepare_downsampling_pipeline.in_set(RenderSystems::Prepare),
                    prepare_upsampling_pipeline.in_set(RenderSystems::Prepare),
                    prepare_bloom_textures.in_set(RenderSystems::PrepareResources),
                    prepare_bloom_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                Core3d,
                bloom.before(tonemapping).in_set(Core3dSystems::PostProcess),
            )
            .add_systems(
                Core2d,
                bloom.before(tonemapping).in_set(Core2dSystems::PostProcess),
            );
    }
}

pub fn bloom(
    view: ViewQuery<(
        &ExtractedCamera,
        &ViewTarget,
        &BloomTexture,
        &BloomBindGroups,
        &DynamicUniformIndex<BloomUniforms>,
        &Bloom,
        &UpsamplingPipelineIds,
        &BloomDownsamplingPipelineIds,
    )>,
    downsampling_pipeline_res: Res<BloomDownsamplingPipeline>,
    pipeline_cache: Res<PipelineCache>,
    uniforms: Res<ComponentUniforms<BloomUniforms>>,
    mut ctx: RenderContext,
) {
    let (
        camera,
        view_target,
        bloom_texture,
        bind_groups,
        uniform_index,
        bloom_settings,
        upsampling_pipeline_ids,
        downsampling_pipeline_ids,
    ) = view.into_inner();

    if bloom_settings.intensity == 0.0 || !camera.hdr {
        return;
    }

    let (
        Some(uniforms_binding),
        Some(downsampling_first_pipeline),
        Some(downsampling_pipeline),
        Some(upsampling_pipeline),
        Some(upsampling_final_pipeline),
    ) = (
        uniforms.binding(),
        pipeline_cache.get_render_pipeline(downsampling_pipeline_ids.first),
        pipeline_cache.get_render_pipeline(downsampling_pipeline_ids.main),
        pipeline_cache.get_render_pipeline(upsampling_pipeline_ids.id_main),
        pipeline_cache.get_render_pipeline(upsampling_pipeline_ids.id_final),
    )
    else {
        return;
    };

    let view_texture = view_target.main_texture_view();
    let view_texture_unsampled = view_target.get_unsampled_color_attachment();

    // Create the first downsampling bind group (reads from main texture)
    let downsampling_first_bind_group = ctx.render_device().create_bind_group(
        "bloom_downsampling_first_bind_group",
        &pipeline_cache.get_bind_group_layout(&downsampling_pipeline_res.bind_group_layout),
        &BindGroupEntries::sequential((
            view_texture,
            &bind_groups.sampler,
            uniforms_binding.clone(),
        )),
    );

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(ctx.command_encoder(), "bloom");

    let command_encoder = ctx.command_encoder();
    command_encoder.push_debug_group("bloom");

    // First downsample pass
    {
        let view = &bloom_texture.view(0);
        let mut downsampling_first_pass =
            command_encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("bloom_downsampling_first_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations::default(),
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        downsampling_first_pass.set_pipeline(downsampling_first_pipeline);
        downsampling_first_pass.set_bind_group(
            0,
            &downsampling_first_bind_group,
            &[uniform_index.index()],
        );
        downsampling_first_pass.draw(0..3, 0..1);
    }

    // Other downsample passes
    for mip in 1..bloom_texture.mip_count {
        let view = &bloom_texture.view(mip);
        let mut downsampling_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("bloom_downsampling_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        downsampling_pass.set_pipeline(downsampling_pipeline);
        downsampling_pass.set_bind_group(
            0,
            &bind_groups.downsampling_bind_groups[mip as usize - 1],
            &[uniform_index.index()],
        );
        downsampling_pass.draw(0..3, 0..1);
    }

    // Upsample passes except the final one
    for mip in (1..bloom_texture.mip_count).rev() {
        let view = &bloom_texture.view(mip - 1);
        let mut upsampling_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("bloom_upsampling_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        upsampling_pass.set_pipeline(upsampling_pipeline);
        upsampling_pass.set_bind_group(
            0,
            &bind_groups.upsampling_bind_groups[(bloom_texture.mip_count - mip - 1) as usize],
            &[uniform_index.index()],
        );
        let blend = compute_blend_factor(
            bloom_settings,
            mip as f32,
            (bloom_texture.mip_count - 1) as f32,
        );
        upsampling_pass.set_blend_constant(LinearRgba::gray(blend).into());
        upsampling_pass.draw(0..3, 0..1);
    }

    // Final upsample pass
    {
        let mut upsampling_final_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("bloom_upsampling_final_pass"),
            color_attachments: &[Some(view_texture_unsampled)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        upsampling_final_pass.set_pipeline(upsampling_final_pipeline);
        upsampling_final_pass.set_bind_group(
            0,
            &bind_groups.upsampling_bind_groups[(bloom_texture.mip_count - 1) as usize],
            &[uniform_index.index()],
        );
        if let Some(viewport) = camera.viewport.as_ref() {
            upsampling_final_pass.set_viewport(
                viewport.physical_position.x as f32,
                viewport.physical_position.y as f32,
                viewport.physical_size.x as f32,
                viewport.physical_size.y as f32,
                viewport.depth.start,
                viewport.depth.end,
            );
        }
        let blend = compute_blend_factor(bloom_settings, 0.0, (bloom_texture.mip_count - 1) as f32);
        upsampling_final_pass.set_blend_constant(LinearRgba::gray(blend).into());
        upsampling_final_pass.draw(0..3, 0..1);
    }

    command_encoder.pop_debug_group();
    time_span.end(ctx.command_encoder());
}

#[derive(Component)]
pub struct BloomTexture {
    // First mip is half the screen resolution, successive mips are half the previous
    #[cfg(any(
        not(feature = "webgl"),
        not(target_arch = "wasm32"),
        feature = "webgpu"
    ))]
    texture: CachedTexture,
    // WebGL does not support binding specific mip levels for sampling, fallback to separate textures instead
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    texture: Vec<CachedTexture>,
    mip_count: u32,
}

impl BloomTexture {
    #[cfg(any(
        not(feature = "webgl"),
        not(target_arch = "wasm32"),
        feature = "webgpu"
    ))]
    fn view(&self, base_mip_level: u32) -> TextureView {
        self.texture.texture.create_view(&TextureViewDescriptor {
            base_mip_level,
            mip_level_count: Some(1u32),
            ..Default::default()
        })
    }
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    fn view(&self, base_mip_level: u32) -> TextureView {
        self.texture[base_mip_level as usize]
            .texture
            .create_view(&TextureViewDescriptor {
                base_mip_level: 0,
                mip_level_count: Some(1u32),
                ..Default::default()
            })
    }
}

/// Builds each bloom view's [`BloomUniforms`] from the extracted view geometry
/// and the paper white of the view's resolved display target.
///
/// Runs after `RenderSystems::PrepareViews` has resolved [`ViewDisplayTarget`]
/// and before `RenderSystems::PrepareResources`, where `UniformComponentPlugin`
/// uploads the components.
fn prepare_bloom_uniforms(
    mut commands: Commands,
    views: Query<(
        Entity,
        &Bloom,
        &ExtractedCamera,
        &ExtractedView,
        &ViewDisplayTarget,
    )>,
) {
    let uniforms = views
        .iter()
        .filter_map(|(entity, bloom, camera, view, display_target)| {
            let target_size = camera.physical_target_size?;
            // `UVec4(origin.x, origin.y, size.x, size.y)` in physical pixels. The
            // size is non-zero: `Bloom` is only extracted for drawable viewports.
            let viewport = view.viewport;
            let uniform = BloomUniforms {
                threshold_precomputations: BloomUniforms::threshold_precomputations(
                    bloom
                        .prefilter
                        .resolve_threshold(display_target.sanitized_paper_white_nits()),
                    bloom.prefilter.threshold_softness,
                ),
                viewport: viewport.as_vec4()
                    / UVec4::new(target_size.x, target_size.y, target_size.x, target_size.y)
                        .as_vec4(),
                aspect: AspectRatio::try_from_pixels(viewport.z, viewport.w)
                    .expect("Valid screen size values for Bloom settings")
                    .ratio(),
                scale: bloom.scale,
            };
            Some((entity, uniform))
        })
        .collect::<Vec<_>>();
    commands.try_insert_batch(uniforms);
}

fn prepare_bloom_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera, &Bloom, &ViewDisplayTarget)>,
) {
    for (entity, camera, bloom, display_target) in &views {
        if let Some(viewport) = camera.physical_viewport_size {
            // How many times we can halve the resolution minus one so we don't go unnecessarily low
            let mip_count = bloom.max_mip_dimension.ilog2().max(2) - 1;
            let mip_height_ratio = if viewport.y != 0 {
                bloom.max_mip_dimension as f32 / viewport.y as f32
            } else {
                0.
            };

            let texture_descriptor = TextureDescriptor {
                label: Some("bloom_texture"),
                size: (viewport.as_vec2() * mip_height_ratio)
                    .round()
                    .as_uvec2()
                    .max(UVec2::ONE)
                    .to_extents(),
                mip_level_count: mip_count,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: bloom_texture_format(Some(display_target)),
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            };

            #[cfg(any(
                not(feature = "webgl"),
                not(target_arch = "wasm32"),
                feature = "webgpu"
            ))]
            let texture = texture_cache.get(&render_device, texture_descriptor);
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            let texture: Vec<CachedTexture> = (0..mip_count)
                .map(|mip| {
                    texture_cache.get(
                        &render_device,
                        TextureDescriptor {
                            size: Extent3d {
                                width: (texture_descriptor.size.width >> mip).max(1),
                                height: (texture_descriptor.size.height >> mip).max(1),
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            ..texture_descriptor.clone()
                        },
                    )
                })
                .collect();

            commands
                .entity(entity)
                .insert(BloomTexture { texture, mip_count });
        }
    }
}

#[derive(Component)]
pub struct BloomBindGroups {
    #[cfg(any(
        not(feature = "webgl"),
        not(target_arch = "wasm32"),
        feature = "webgpu"
    ))]
    cache_key: (TextureId, BufferId),
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    cache_key: (Vec<TextureId>, BufferId),
    downsampling_bind_groups: Box<[BindGroup]>,
    upsampling_bind_groups: Box<[BindGroup]>,
    sampler: Sampler,
}

fn prepare_bloom_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    downsampling_pipeline: Res<BloomDownsamplingPipeline>,
    upsampling_pipeline: Res<BloomUpsamplingPipeline>,
    views: Query<(Entity, &BloomTexture, Option<&BloomBindGroups>)>,
    uniforms: Res<ComponentUniforms<BloomUniforms>>,
    pipeline_cache: Res<PipelineCache>,
) {
    let sampler = &downsampling_pipeline.sampler;

    for (entity, bloom_texture, bloom_bind_groups) in &views {
        #[cfg(any(
            not(feature = "webgl"),
            not(target_arch = "wasm32"),
            feature = "webgpu"
        ))]
        let cache_key = (
            bloom_texture.texture.texture.id(),
            uniforms.buffer().unwrap().id(),
        );
        #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
        let cache_key = (
            bloom_texture
                .texture
                .iter()
                .map(|tex| tex.texture.id())
                .collect(),
            uniforms.buffer().unwrap().id(),
        );

        if let Some(b) = bloom_bind_groups
            && b.cache_key == cache_key
        {
            continue;
        }

        let bind_group_count = bloom_texture.mip_count as usize - 1;

        let mut downsampling_bind_groups = Vec::with_capacity(bind_group_count);
        for mip in 1..bloom_texture.mip_count {
            downsampling_bind_groups.push(render_device.create_bind_group(
                "bloom_downsampling_bind_group",
                &pipeline_cache.get_bind_group_layout(&downsampling_pipeline.bind_group_layout),
                &BindGroupEntries::sequential((
                    &bloom_texture.view(mip - 1),
                    sampler,
                    uniforms.binding().unwrap(),
                )),
            ));
        }

        let mut upsampling_bind_groups = Vec::with_capacity(bind_group_count);
        for mip in (0..bloom_texture.mip_count).rev() {
            upsampling_bind_groups.push(render_device.create_bind_group(
                "bloom_upsampling_bind_group",
                &pipeline_cache.get_bind_group_layout(&upsampling_pipeline.bind_group_layout),
                &BindGroupEntries::sequential((
                    &bloom_texture.view(mip),
                    sampler,
                    uniforms.binding().unwrap(),
                )),
            ));
        }

        commands.entity(entity).insert(BloomBindGroups {
            cache_key,
            downsampling_bind_groups: downsampling_bind_groups.into_boxed_slice(),
            upsampling_bind_groups: upsampling_bind_groups.into_boxed_slice(),
            sampler: sampler.clone(),
        });
    }
}

/// Calculates blend intensities of blur pyramid levels
/// during the upsampling + compositing stage.
///
/// The function assumes all pyramid levels are upsampled and
/// blended into higher frequency ones using this function to
/// calculate blend levels every time. The final (highest frequency)
/// pyramid level in not blended into anything therefore this function
/// is not applied to it. As a result, the *mip* parameter of 0 indicates
/// the second-highest frequency pyramid level (in our case that is the
/// 0th mip of the bloom texture with the original image being the
/// actual highest frequency level).
///
/// Parameters:
/// * `mip` - the index of the lower frequency pyramid level (0 - `max_mip`, where 0 indicates highest frequency mip but not the highest frequency image).
/// * `max_mip` - the index of the lowest frequency pyramid level.
///
/// This function can be visually previewed for all values of *mip* (normalized) with tweakable
/// [`Bloom`] parameters on [Desmos graphing calculator](https://www.desmos.com/calculator/ncc8xbhzzl).
///
/// [`BloomScatterModel::Gt7Glare`] instead uses the diffraction weights of
/// [`glare::blend_factor`]. Those are tied to the absolute texel scale of the
/// pyramid levels, not to the chain depth, so that branch ignores `max_mip`.
fn compute_blend_factor(bloom: &Bloom, mip: f32, max_mip: f32) -> f32 {
    match bloom.scatter {
        BloomScatterModel::Aesthetic => {
            let mut lf_boost =
                (1.0 - ops::powf(
                    1.0 - (mip / max_mip),
                    1.0 / (1.0 - bloom.low_frequency_boost_curvature),
                )) * bloom.low_frequency_boost;
            let high_pass_lq = 1.0
                - (((mip / max_mip) - bloom.high_pass_frequency) / bloom.high_pass_frequency)
                    .clamp(0.0, 1.0);
            lf_boost *= match bloom.composite_mode {
                BloomCompositeMode::EnergyConserving => 1.0 - bloom.intensity,
                BloomCompositeMode::Additive => 1.0,
            };

            (bloom.intensity + lf_boost) * high_pass_lq
        }
        BloomScatterModel::Gt7Glare { f_number } => {
            glare::blend_factor(f_number, bloom.intensity, mip as u32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::{schedule::ScheduleLabel, system::RunSystemOnce};
    use bevy_math::{Mat4, Vec2};
    use bevy_render::view::RetainedViewEntity;

    #[test]
    fn prepare_bloom_uniforms_packs_reference_uniform() {
        let mut world = World::new();

        let bloom = Bloom {
            prefilter: BloomPrefilter {
                threshold: 0.0,
                threshold_nits: Some(406.0),
                threshold_softness: 0.25,
            },
            scale: Vec2::new(2.0, 1.0),
            ..Bloom::NATURAL
        };
        let mut display_target = ViewDisplayTarget::default();
        display_target.0.paper_white_nits = 203.0;

        let entity = world
            .spawn((
                bloom,
                ExtractedCamera {
                    target: None,
                    physical_viewport_size: Some(UVec2::new(320, 240)),
                    physical_target_size: Some(UVec2::new(640, 480)),
                    viewport: None,
                    schedule: Core3d.intern(),
                    order: 0,
                    output_mode: Default::default(),
                    msaa_writeback: Default::default(),
                    clear_color: Default::default(),
                    sorted_camera_index_for_target: 0,
                    exposure: 1.0,
                    hdr: true,
                    compositing_space: None,
                },
                ExtractedView {
                    retained_view_entity: RetainedViewEntity::new(
                        Entity::PLACEHOLDER.into(),
                        None,
                        0,
                    ),
                    clip_from_view: Mat4::IDENTITY,
                    world_from_view: Default::default(),
                    clip_from_world: None,
                    target_format: TextureFormat::Rgba16Float,
                    viewport: UVec4::new(8, 16, 320, 240),
                    color_grading: Default::default(),
                    invert_culling: false,
                },
                display_target,
            ))
            .id();

        world.run_system_once(prepare_bloom_uniforms).unwrap();

        let uniforms = world.get::<BloomUniforms>(entity).unwrap();
        assert_eq!(
            uniforms.threshold_precomputations,
            // 406 nits against a 203-nit paper white is 2.0 in framebuffer units.
            BloomUniforms::threshold_precomputations(2.0, 0.25)
        );
        assert_eq!(
            uniforms.viewport,
            bevy_math::Vec4::new(8.0 / 640.0, 16.0 / 480.0, 320.0 / 640.0, 240.0 / 480.0)
        );
        assert_eq!(
            uniforms.aspect,
            AspectRatio::try_from_pixels(320, 240).unwrap().ratio()
        );
        assert_eq!(uniforms.scale, Vec2::new(2.0, 1.0));
    }

    /// An independent copy of the parametric curve, to lock the
    /// [`BloomScatterModel::Aesthetic`] path bit for bit.
    fn legacy_compute_blend_factor(bloom: &Bloom, mip: f32, max_mip: f32) -> f32 {
        let mut lf_boost =
            (1.0 - ops::powf(
                1.0 - (mip / max_mip),
                1.0 / (1.0 - bloom.low_frequency_boost_curvature),
            )) * bloom.low_frequency_boost;
        let high_pass_lq = 1.0
            - (((mip / max_mip) - bloom.high_pass_frequency) / bloom.high_pass_frequency)
                .clamp(0.0, 1.0);
        lf_boost *= match bloom.composite_mode {
            BloomCompositeMode::EnergyConserving => 1.0 - bloom.intensity,
            BloomCompositeMode::Additive => 1.0,
        };

        (bloom.intensity + lf_boost) * high_pass_lq
    }

    #[test]
    fn aesthetic_blend_factors_match_dedicated_implementation() {
        let presets = [
            Bloom::NATURAL,
            Bloom::ANAMORPHIC,
            Bloom::OLD_SCHOOL,
            Bloom::SCREEN_BLUR,
            Bloom {
                intensity: 0.37,
                low_frequency_boost: 0.21,
                low_frequency_boost_curvature: 0.5,
                high_pass_frequency: 0.66,
                composite_mode: BloomCompositeMode::Additive,
                scale: Vec2::new(2.0, 1.0),
                ..Bloom::NATURAL
            },
        ];
        for bloom in &presets {
            for max_mip in [3.0f32, 7.0, 9.0] {
                for mip in 0..=(max_mip as u32) {
                    let mip = mip as f32;
                    assert_eq!(
                        compute_blend_factor(bloom, mip, max_mip).to_bits(),
                        legacy_compute_blend_factor(bloom, mip, max_mip).to_bits(),
                        "mismatch at mip {mip}/{max_mip}"
                    );
                }
            }
        }
    }

    #[test]
    fn glare_overrides_threshold_and_composite_mode() {
        let mut bloom = Bloom {
            composite_mode: BloomCompositeMode::Additive,
            ..Bloom::OLD_SCHOOL
        };
        assert!(bloom.thresholding_active());
        assert_eq!(
            bloom.effective_composite_mode(),
            BloomCompositeMode::Additive
        );

        bloom.scatter = BloomScatterModel::Gt7Glare { f_number: 8.0 };
        assert!(!bloom.thresholding_active());
        assert_eq!(
            bloom.effective_composite_mode(),
            BloomCompositeMode::EnergyConserving
        );
    }

    #[test]
    fn glare_blend_factor_uses_psf_not_parametric_curve() {
        let glare = Bloom {
            scatter: BloomScatterModel::Gt7Glare { f_number: 5.6 },
            ..Bloom::NATURAL
        };
        assert_eq!(compute_blend_factor(&glare, 0.0, 7.0), glare.intensity);

        let tweaked = Bloom {
            low_frequency_boost: 0.0,
            low_frequency_boost_curvature: 0.1,
            high_pass_frequency: 0.5,
            ..glare.clone()
        };
        for mip in 0..8 {
            assert_eq!(
                compute_blend_factor(&glare, mip as f32, 7.0).to_bits(),
                compute_blend_factor(&tweaked, mip as f32, 7.0).to_bits()
            );
        }
    }
}
