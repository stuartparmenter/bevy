use crate::{
    blit::{BlitPipeline, BlitPipelineKey, DisplayTargetUniform},
    tonemapping::outputs_wide_gamut,
};
use bevy_app::prelude::*;
use bevy_camera::CameraOutputMode;
use bevy_color::RgbPrimaries;
use bevy_ecs::prelude::*;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::*,
    transfer_functions::gamut_matrix,
    view::{ResolvedCompositingSpace, Tonemapping, ViewDisplayTarget, ViewTarget},
    Render, RenderApp, RenderStartup, RenderSystems, WorkingColorSpace,
};

mod node;

pub use node::upscaling;

pub struct UpscalingPlugin;

impl Plugin for UpscalingPlugin {
    fn build(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(
                Render,
                // This system should probably technically be run *after* all of the other systems
                // that might modify `PipelineCache` via interior mutability, but for now,
                // we've chosen to simply ignore the ambiguities out of a desire for a better refactor
                // and aversion to extensive and intrusive system ordering.
                // See https://github.com/bevyengine/bevy/issues/14770 for more context.
                prepare_view_upscaling_pipelines
                    .in_set(RenderSystems::Prepare)
                    .before(RenderSystems::PrepareResources)
                    .ambiguous_with_all(),
            );
            render_app.add_systems(RenderStartup, clear_view_upscaling_pipelines);
        }
    }
}

#[derive(Component)]
pub struct ViewUpscalingPipeline(CachedRenderPipelineId, BlitPipelineKey);

/// This is not required on first startup but is required during render recovery
fn clear_view_upscaling_pipelines(
    mut commands: Commands,
    views: Query<Entity, With<ViewUpscalingPipeline>>,
) {
    for entity in &views {
        commands.entity(entity).remove::<ViewUpscalingPipeline>();
    }
}

fn prepare_view_upscaling_pipelines(
    mut commands: Commands,
    mut pipeline_cache: ResMut<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<BlitPipeline>>,
    blit_pipeline: Res<BlitPipeline>,
    working_color_space: Res<WorkingColorSpace>,
    view_targets: Query<(
        Entity,
        &ViewTarget,
        Option<&ExtractedCamera>,
        Option<&Tonemapping>,
        Option<&ViewUpscalingPipeline>,
        Option<&ResolvedCompositingSpace>,
        Option<&ViewDisplayTarget>,
        Option<&DisplayTargetUniform>,
    )>,
) {
    for (
        entity,
        view_target,
        camera,
        tonemapping,
        maybe_pipeline,
        resolved_space,
        display_target,
        display_target_uniform,
    ) in view_targets.iter()
    {
        let blend_state = if let Some(extracted_camera) = camera {
            match extracted_camera.output_mode {
                CameraOutputMode::Skip => None,
                CameraOutputMode::Write { blend_state, .. } => {
                    match blend_state {
                        None => {
                            // Auto-detect: the first camera to render to this output
                            // (sorted_camera_index_for_target == 0) uses replace mode;
                            // subsequent cameras default to alpha blending so they don't
                            // accidentally overwrite earlier cameras' output.
                            if extracted_camera.sorted_camera_index_for_target > 0 {
                                Some(BlendState::ALPHA_BLENDING)
                            } else {
                                None
                            }
                        }
                        _ => blend_state,
                    }
                }
            }
        } else {
            None
        };

        let Some(target_format) = view_target.out_texture_view_format() else {
            continue;
        };

        let display_target = display_target.map(|target| target.0).unwrap_or_default();
        // Tonemapping outputs Rec.709, so only a camera without it keeps the
        // working color space.
        let source_rec2020 = *working_color_space == WorkingColorSpace::Rec2020
            && camera.is_some_and(|camera| !camera.tonemaps());
        let source_primaries = if source_rec2020 {
            RgbPrimaries::BT2020
        } else {
            RgbPrimaries::BT709
        };
        let wide_source = source_rec2020
            || camera.is_some_and(|camera| {
                outputs_wide_gamut(camera, tonemapping, &display_target, *working_color_space)
            });
        let key = BlitPipelineKey {
            target_format,
            blend_state,
            samples: 1,
            source_space: ResolvedCompositingSpace::space(resolved_space),
            display_color_space: display_target.color_space,
            gamut_compression: wide_source
                && display_target.color_space.primaries() != RgbPrimaries::BT2020,
        };

        let uniform = DisplayTargetUniform {
            gamut: gamut_matrix(source_primaries, display_target.color_space.primaries()),
            paper_white_nits: display_target.paper_white_nits,
        };
        if display_target_uniform != Some(&uniform) {
            commands.entity(entity).insert(uniform);
        }

        if maybe_pipeline.is_none_or(|ViewUpscalingPipeline(_, cached_key)| *cached_key != key) {
            let pipeline = pipelines.specialize(&pipeline_cache, &blit_pipeline, key);

            // Ensure the pipeline is loaded before continuing the frame to prevent frames without
            // any GPU work submitted
            pipeline_cache.block_on_render_pipeline(pipeline);

            commands
                .entity(entity)
                .insert(ViewUpscalingPipeline(pipeline, key));
        }
    }
}
