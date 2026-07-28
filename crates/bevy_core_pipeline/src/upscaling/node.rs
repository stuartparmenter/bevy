use crate::{
    blit::BlitPipeline, display_encoding::encode_out_texture_clear_color,
    upscaling::ViewUpscalingPipeline,
};
use bevy_camera::{CameraOutputMode, ClearColor, ClearColorConfig};
use bevy_ecs::prelude::*;
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    render_resource::{BindGroup, PipelineCache, RenderPassDescriptor, TextureViewId},
    renderer::{RenderContext, ViewQuery},
    view::{ViewDisplayTarget, ViewTarget},
};

use crate::camera_stack::ViewStackContract;

#[derive(Default)]
pub struct UpscalingBindGroupCache {
    cached: Option<(TextureViewId, BindGroup)>,
}

pub fn upscaling(
    view: ViewQuery<(
        &ViewTarget,
        &ViewUpscalingPipeline,
        Option<&ExtractedCamera>,
        Option<&ViewStackContract>,
        Option<&ViewDisplayTarget>,
    )>,
    pipeline_cache: Res<PipelineCache>,
    blit_pipeline: Res<BlitPipeline>,
    clear_color_global: Res<ClearColor>,
    mut cache: Local<UpscalingBindGroupCache>,
    mut ctx: RenderContext,
) {
    let (target, upscaling_target, camera, contract, view_display_target) = view.into_inner();

    let clear_color = if let Some(camera) = camera {
        match camera.output_mode {
            CameraOutputMode::Write { clear_color, .. } => clear_color,
            CameraOutputMode::Skip => return,
        }
    } else {
        ClearColorConfig::Default
    };
    let clear_color = match clear_color {
        ClearColorConfig::Default => Some(clear_color_global.0),
        ClearColorConfig::Custom(color) => Some(color),
        ClearColorConfig::None => None,
    };
    // On an HDR-transfer target the out texture stores encoded signal, so the
    // `LoadOp::Clear` value must be encoded CPU-side to match the rendered
    // pixels the finalizer's blit composites over (regions no blit covers
    // would otherwise present raw display-linear values as HDR signal). For
    // SDR targets (`encoding: None`) the path is unchanged: the hardware sRGB
    // view encodes the linear clear on store. The sanitized paper white must
    // match the value the GPU encoder folds (`sanitized_paper_white_nits`), so
    // the encoded clear and the rendered pixels agree even for degenerate
    // authored paper whites. `encoding` is `Some` only on an HDR-transfer
    // group, which always carries a `ViewDisplayTarget`.
    let converted_clear_color = clear_color.map(|color| {
        match contract
            .and_then(|contract| contract.encoding)
            .zip(view_display_target)
        {
            Some((encoding, view_display_target)) => encode_out_texture_clear_color(
                color.into(),
                &encoding,
                view_display_target.sanitized_paper_white_nits(),
            ),
            None => color.into(),
        }
    });

    // texture to be upscaled to the output texture
    let main_texture_view = target.main_texture_view();

    let bind_group = match &mut cache.cached {
        Some((id, bind_group)) if main_texture_view.id() == *id => bind_group,
        cached => {
            let bind_group = blit_pipeline.create_bind_group(
                ctx.render_device(),
                main_texture_view,
                &pipeline_cache,
            );

            let (_, bind_group) = cached.insert((main_texture_view.id(), bind_group));
            bind_group
        }
    };

    let Some(out_attachment) = target.out_texture_color_attachment(converted_clear_color) else {
        return;
    };

    let pass_descriptor = RenderPassDescriptor {
        label: Some("upscaling"),
        color_attachments: &[Some(out_attachment)],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    };

    let Some(pipeline) = pipeline_cache.get_render_pipeline(upscaling_target.0) else {
        // we need to do some work on the swapchain to avoid pink screen uninit on macos
        #[cfg(target_os = "macos")]
        ctx.command_encoder().begin_render_pass(&pass_descriptor);
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(ctx.command_encoder(), "upscaling");

    {
        let mut render_pass = ctx.command_encoder().begin_render_pass(&pass_descriptor);

        if let Some(camera) = camera
            && let Some(viewport) = &camera.viewport
        {
            let size = viewport.physical_size;
            let position = viewport.physical_position;
            render_pass.set_scissor_rect(position.x, position.y, size.x, size.y);
        }

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    time_span.end(ctx.command_encoder());
}
