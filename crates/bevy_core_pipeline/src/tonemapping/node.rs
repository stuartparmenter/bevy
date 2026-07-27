use crate::tonemapping::{
    Gt7ParamsUniforms, TonemappingLuts, TonemappingPipeline, TonemappingPipelineKeyFlags,
    ViewGt7ParamsUniformOffset, ViewTonemappingPipeline,
};

use bevy_ecs::prelude::*;
use bevy_render::{
    diagnostic::RecordDiagnostics,
    render_asset::RenderAssets,
    render_resource::{
        BindGroup, BindGroupEntries, BufferId, LoadOp, Operations, PipelineCache,
        RenderPassColorAttachment, RenderPassDescriptor, StoreOp, TextureFormat, TextureViewId,
    },
    renderer::{RenderContext, ViewQuery},
    texture::{FallbackImage, GpuImage},
    view::{ViewTarget, ViewUniformOffset, ViewUniforms},
};

use super::{get_lut_bindings, Tonemapping};

/// Cached bind group state for tonemapping.
#[derive(Default)]
pub struct TonemappingBindGroupCache {
    cached: Option<CachedBindGroup>,
    last_tonemapping: Option<Tonemapping>,
}

/// The inputs a cached tonemapping bind group was created from.
struct CachedBindGroup {
    view_uniforms: BufferId,
    source: TextureViewId,
    lut: TextureViewId,
    /// `Some` iff the pipeline binds the per-view GT7 params uniform.
    gt7_params_uniforms: Option<BufferId>,
    bind_group: BindGroup,
}

pub fn tonemapping(
    view: ViewQuery<(
        &ViewUniformOffset,
        &ViewTarget,
        &ViewTonemappingPipeline,
        &Tonemapping,
        Option<&ViewGt7ParamsUniformOffset>,
    )>,
    pipeline_cache: Res<PipelineCache>,
    tonemapping_pipeline: Res<TonemappingPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    fallback_image: Res<FallbackImage>,
    view_uniforms: Res<ViewUniforms>,
    gt7_params_uniforms: Res<Gt7ParamsUniforms>,
    tonemapping_luts: Res<TonemappingLuts>,
    mut cache: Local<TonemappingBindGroupCache>,
    mut ctx: RenderContext,
) {
    let (view_uniform_offset, target, view_tonemapping_pipeline, tonemapping, gt7_params_offset) =
        view.into_inner();

    // `Tonemapping::None` is a true opt-out: the pass does not run, the
    // camera keeps its pre-existing main-texture format, and no color
    // grading / exposure from `ColorGrading` is applied.
    if *tonemapping == Tonemapping::None {
        return;
    }

    // Eligible SDR cameras fold tone mapping into their material shaders (the
    // `TONEMAP_IN_SHADER` path selected in camera extraction) and keep an 8-bit
    // main texture instead of the fp16 intermediate. The 8-bit format is the
    // carried signal: running this node for such a camera would tone-map a
    // second time, so skip it. (Every camera that runs the node-side operator
    // is on an `Rgba16Float` intermediate.)
    if matches!(
        target.main_texture_format(),
        TextureFormat::Rgba8UnormSrgb | TextureFormat::Rgba8Unorm
    ) {
        return;
    }

    let Some(pipeline) = pipeline_cache.get_render_pipeline(view_tonemapping_pipeline.pipeline_id)
    else {
        return;
    };

    let view_uniforms_buffer = &view_uniforms.uniforms;
    let view_uniforms_id = view_uniforms_buffer.buffer().unwrap().id();

    // Collect the optional GT7 params binding the pipeline was specialized
    // with. If its buffer or offset is missing (which should not happen — the
    // same predicate drives specialization and preparation), skip the pass
    // rather than binding with a mismatched layout.
    let needs_gt7_params = view_tonemapping_pipeline
        .flags
        .contains(TonemappingPipelineKeyFlags::GT7_PARAMS_UNIFORM);

    let gt7_params_binding = if needs_gt7_params {
        let (Some(_), Some(offset)) = (gt7_params_uniforms.uniforms.buffer(), gt7_params_offset)
        else {
            return;
        };
        Some((&gt7_params_uniforms.uniforms, offset.offset))
    } else {
        None
    };

    let gt7_params_uniforms_id =
        gt7_params_binding.map(|(uniforms, _)| uniforms.buffer().unwrap().id());

    let post_process = target.post_process_write();
    let source = post_process.source;
    let destination = post_process.destination;

    let tonemapping_changed = cache.last_tonemapping != Some(*tonemapping);
    if tonemapping_changed {
        cache.last_tonemapping = Some(*tonemapping);
    }

    let bind_group = match &mut cache.cached {
        Some(cached)
            if view_uniforms_id == cached.view_uniforms
                && source.id() == cached.source
                && cached.lut != fallback_image.d3.texture_view.id()
                && gt7_params_uniforms_id == cached.gt7_params_uniforms
                && !tonemapping_changed =>
        {
            &cached.bind_group
        }
        cached => {
            // LUT selection keys on the *authored* operator. For views whose
            // SDR-only operator was D6-substituted with GT7 (see
            // `effective_tonemapping`) this binds the authored operator's
            // LUT, which is harmless: the GT7 shader path never samples the
            // LUT, and every LUT satisfies the same 3D-texture layout entry.
            let lut_bindings =
                get_lut_bindings(&gpu_images, &tonemapping_luts, tonemapping, &fallback_image);

            let layout = pipeline_cache.get_bind_group_layout(
                view_tonemapping_pipeline.bind_group_layout(&tonemapping_pipeline),
            );
            let render_device = ctx.render_device();
            let bind_group = match gt7_params_binding {
                None => render_device.create_bind_group(
                    None,
                    &layout,
                    &BindGroupEntries::sequential((
                        view_uniforms_buffer,
                        source,
                        &tonemapping_pipeline.sampler,
                        lut_bindings.0,
                        lut_bindings.1,
                    )),
                ),
                Some((gt7_params_uniforms, _)) => render_device.create_bind_group(
                    None,
                    &layout,
                    &BindGroupEntries::sequential((
                        view_uniforms_buffer,
                        source,
                        &tonemapping_pipeline.sampler,
                        lut_bindings.0,
                        lut_bindings.1,
                        gt7_params_uniforms,
                    )),
                ),
            };

            let cached = cached.insert(CachedBindGroup {
                view_uniforms: view_uniforms_id,
                source: source.id(),
                lut: lut_bindings.0.id(),
                gt7_params_uniforms: gt7_params_uniforms_id,
                bind_group,
            });
            &cached.bind_group
        }
    };

    // Dynamic offsets in increasing binding order: view (0), GT7 params (5, if
    // bound).
    let mut dynamic_offsets = [0u32; 2];
    let mut dynamic_offset_count = 0;
    dynamic_offsets[dynamic_offset_count] = view_uniform_offset.offset;
    dynamic_offset_count += 1;
    if let Some((_, offset)) = gt7_params_binding {
        dynamic_offsets[dynamic_offset_count] = offset;
        dynamic_offset_count += 1;
    }

    let pass_descriptor = RenderPassDescriptor {
        label: Some("tonemapping"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: destination,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Default::default()), // TODO shouldn't need to be cleared
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
    let time_span = diagnostics.time_span(ctx.command_encoder(), "tonemapping");

    {
        let mut render_pass = ctx.command_encoder().begin_render_pass(&pass_descriptor);

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &dynamic_offsets[..dynamic_offset_count]);
        render_pass.draw(0..3, 0..1);
    }

    time_span.end(ctx.command_encoder());
}
