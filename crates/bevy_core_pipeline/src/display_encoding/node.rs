use bevy_ecs::prelude::*;
use bevy_render::{
    diagnostic::RecordDiagnostics,
    extract_component::{ComponentUniforms, DynamicUniformIndex},
    render_resource::{
        BindGroup, BindGroupEntries, BufferId, LoadOp, Operations, PipelineCache,
        RenderPassColorAttachment, RenderPassDescriptor, StoreOp, TextureViewId,
    },
    renderer::{RenderContext, ViewQuery},
    view::{DisplayTargetUniform, ViewTarget},
};

use super::{DisplayEncodingPipeline, ViewDisplayEncodingPipeline};

/// Cached bind group state for the display-encoding pass.
#[derive(Default)]
pub struct DisplayEncodingBindGroupCache {
    cached: Option<CachedBindGroup>,
}

/// The inputs a cached display-encoding bind group was created from.
struct CachedBindGroup {
    source: TextureViewId,
    display_target_uniforms: BufferId,
    bind_group: BindGroup,
}

/// Render node (a `Core2d` / `Core3d` schedule system) performing the
/// gamut-transform + transfer-encoding pass.
///
/// Scheduled after the UI pass (UI composites in paper-white-relative
/// display-linear space, which this pass consumes) and before the upscaling
/// blit (which passes the encoded signal through unchanged; see
/// `prepare_view_upscaling_pipelines`).
///
/// Views without a [`ViewDisplayEncodingPipeline`] — every view on a plain
/// SDR sRGB display target, i.e. all of them by default — return immediately:
/// no pass, no ping-pong flip, zero overhead.
pub fn display_encoding(
    view: ViewQuery<(
        &ViewTarget,
        Option<&ViewDisplayEncodingPipeline>,
        Option<&DynamicUniformIndex<DisplayTargetUniform>>,
    )>,
    pipeline_cache: Res<PipelineCache>,
    encoding_pipeline: Res<DisplayEncodingPipeline>,
    display_target_uniforms: Res<ComponentUniforms<DisplayTargetUniform>>,
    mut cache: Local<DisplayEncodingBindGroupCache>,
    mut ctx: RenderContext,
) {
    let (target, view_encoding_pipeline, display_target_index) = view.into_inner();

    // Plain SDR targets (the default for every existing user) never get the
    // component; hardware sRGB on the upscaling blit remains their encoder.
    let Some(view_encoding_pipeline) = view_encoding_pipeline else {
        return;
    };

    let Some(pipeline) = pipeline_cache.get_render_pipeline(view_encoding_pipeline.pipeline_id)
    else {
        return;
    };

    // Defensive: every view that gets a pipeline also carries a
    // `DisplayTargetUniform` and so gets an index; bail before flipping the
    // ping-pong if not.
    let Some(display_target_index) = display_target_index else {
        return;
    };
    let Some(uniforms_buffer) = display_target_uniforms.buffer() else {
        return;
    };
    let uniforms_buffer_id = uniforms_buffer.id();

    let post_process = target.post_process_write();
    let source = post_process.source;
    let destination = post_process.destination;

    let bind_group = match &mut cache.cached {
        Some(cached)
            if source.id() == cached.source
                && uniforms_buffer_id == cached.display_target_uniforms =>
        {
            &cached.bind_group
        }
        cached => {
            let layout = pipeline_cache.get_bind_group_layout(&encoding_pipeline.layout);
            let bind_group = ctx.render_device().create_bind_group(
                Some("display_encoding_bind_group"),
                &layout,
                &BindGroupEntries::sequential((
                    source,
                    &encoding_pipeline.sampler,
                    display_target_uniforms.uniforms(),
                )),
            );
            let cached = cached.insert(CachedBindGroup {
                source: source.id(),
                display_target_uniforms: uniforms_buffer_id,
                bind_group,
            });
            &cached.bind_group
        }
    };

    let pass_descriptor = RenderPassDescriptor {
        label: Some("display_encoding"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: destination,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Default::default()),
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(ctx.command_encoder(), "display_encoding");

    {
        let mut render_pass = ctx.command_encoder().begin_render_pass(&pass_descriptor);

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[display_target_index.index()]);
        render_pass.draw(0..3, 0..1);
    }

    time_span.end(ctx.command_encoder());
}
