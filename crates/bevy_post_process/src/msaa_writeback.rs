use bevy_app::{App, Plugin};
use bevy_camera::{ClearColorConfig, MsaaWriteback};
use bevy_color::LinearRgba;
use bevy_core_pipeline::{
    blit::{BlitPipeline, BlitPipelineKey},
    schedule::{Core2d, Core2dSystems, Core3d, Core3dSystems},
};
use bevy_ecs::prelude::*;
use bevy_platform::collections::HashMap;
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    render_resource::*,
    renderer::{RenderContext, ViewQuery},
    view::{Msaa, ViewTarget},
    Render, RenderApp, RenderSystems,
};

/// This enables "msaa writeback" support for the `core_2d` and `core_3d` pipelines, which can be enabled on cameras
/// using [`bevy_camera::Camera::msaa_writeback`]. See the docs on that field for more information.
#[derive(Default)]
pub struct MsaaWritebackPlugin;

impl Plugin for MsaaWritebackPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(
            Render,
            prepare_msaa_writeback_pipelines.in_set(RenderSystems::Prepare),
        );
        render_app.add_systems(Core3d, msaa_writeback.before(Core3dSystems::MainPass));
        render_app.add_systems(Core2d, msaa_writeback.before(Core2dSystems::MainPass));
    }
}

pub(crate) fn msaa_writeback(
    view: ViewQuery<(&ViewTarget, &MsaaWritebackBlitPipeline, &Msaa)>,
    blit_pipeline: Res<BlitPipeline>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (target, blit_pipeline_id, msaa) = view.into_inner();

    if *msaa == Msaa::Off {
        return;
    }

    let Some(pipeline) = pipeline_cache.get_render_pipeline(blit_pipeline_id.0) else {
        return;
    };

    // The current "main texture" needs to be bound as an input resource, and we need the "other"
    // unused target to be the "resolve target" for the MSAA write. Therefore this is the same
    // as a post process write!
    let post_process = target.post_process_write();

    let pass_descriptor = RenderPassDescriptor {
        label: Some("msaa_writeback"),
        // The target's "resolve target" is the "destination" in post_process.
        // We will indirectly write the results to the "destination" using
        // the MSAA resolve step.
        color_attachments: &[Some(RenderPassColorAttachment {
            // If MSAA is enabled, then the sampled texture will always exist
            view: target.sampled_main_texture_view().unwrap(),
            depth_slice: None,
            resolve_target: Some(post_process.destination),
            ops: Operations {
                load: LoadOp::Clear(LinearRgba::BLACK.into()),
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    };

    let bind_group =
        blit_pipeline.create_bind_group(ctx.render_device(), post_process.source, &pipeline_cache);

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(ctx.command_encoder(), "msaa_writeback");

    {
        let mut render_pass = ctx.command_encoder().begin_render_pass(&pass_descriptor);

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    time_span.end(ctx.command_encoder());
}

#[derive(Component)]
pub struct MsaaWritebackBlitPipeline(CachedRenderPipelineId);

fn prepare_msaa_writeback_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<BlitPipeline>>,
    blit_pipeline: Res<BlitPipeline>,
    view_targets: Query<(Entity, &ViewTarget, &ExtractedCamera, &Msaa)>,
) {
    // The lowest sorted camera index writing to each main texture this frame.
    // Cameras on one render target can still render into different main
    // textures (an Hdr camera's fp16 next to an SDR camera's 8-bit one), so
    // the writeback decision groups by the texture, not the target.
    let mut first_writer_index = <HashMap<TextureId, usize>>::default();
    for (_, view_target, camera, _) in view_targets.iter() {
        let index = first_writer_index
            .entry(view_target.main_texture().id())
            .or_insert(usize::MAX);
        *index = (*index).min(camera.sorted_camera_index_for_target);
    }

    for (entity, view_target, camera, msaa) in view_targets.iter() {
        // Determine if we should do MSAA writeback based on the camera's setting
        let should_writeback = match camera.msaa_writeback {
            MsaaWriteback::Off => false,
            // Writeback fills the sampled texture with the content the main
            // pass would otherwise load: last frame's, for a camera that
            // preserves content across frames via `ClearColorConfig::None`, or
            // a lower camera's, for a fullscreen camera compositing over an
            // earlier camera that rendered into the same main texture this
            // frame. A camera that clears a texture no earlier camera wrote
            // has nothing to load; writeback would feed it stale pixels and
            // suppress its clear.
            MsaaWriteback::Auto => {
                matches!(camera.clear_color, ClearColorConfig::None)
                    || (camera.viewport.is_none()
                        && first_writer_index
                            .get(&view_target.main_texture().id())
                            .is_some_and(|first| *first < camera.sorted_camera_index_for_target))
            }
            MsaaWriteback::Always => true,
        };

        if msaa.samples() > 1 && should_writeback {
            let key = BlitPipelineKey {
                target_format: view_target.main_texture_format(),
                samples: msaa.samples(),
                blend_state: None,
                source_space: None,
            };

            let pipeline = pipelines.specialize(&pipeline_cache, &blit_pipeline, key);
            commands
                .entity(entity)
                .insert(MsaaWritebackBlitPipeline(pipeline));
        } else {
            // This isn't strictly necessary now, but if we move to retained render entity state I don't
            // want this to silently break
            commands
                .entity(entity)
                .remove::<MsaaWritebackBlitPipeline>();
        }
    }
}
