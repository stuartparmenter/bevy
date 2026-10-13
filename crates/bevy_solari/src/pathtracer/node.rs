use super::{prepare::PathtracerAccumulationTexture, Pathtracer};
use crate::scene::{FarFieldVisibility, FarFieldVisibilityBindGroup, RaytracingSceneBindings};
use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_ecs::{prelude::*, resource::Resource, system::Commands};
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        binding_types::{texture_storage_2d, uniform_buffer},
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
        ImageSubresourceRange, PipelineCache, ShaderStages, StorageTextureAccess, TextureFormat,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::{ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
};
use bevy_utils::default;

/// Resource holding the pathtracer pipeline configuration.
#[derive(Resource)]
pub struct PathtracerPipelines {
    bind_group_layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

/// Initializes the pathtracer pipelines at render startup.
pub fn init_pathtracer_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    scene_bindings: Res<RaytracingSceneBindings>,
    asset_server: Res<AssetServer>,
    far_field: Option<Res<FarFieldVisibility>>,
) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "pathtracer_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<ViewUniform>(true),
            ),
        ),
    );

    let mut layout = vec![
        scene_bindings.bind_group_layout.clone(),
        bind_group_layout.clone(),
    ];
    let mut shader_defs = Vec::new();
    if let Some(far_field) = &far_field {
        layout.push(far_field.bind_group_layout.clone());
        shader_defs.push("FAR_FIELD_VISIBILITY".into());
    }
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("pathtracer_pipeline".into()),
        layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "pathtracer.wesl"),
        shader_defs,
        ..default()
    });

    commands.insert_resource(PathtracerPipelines {
        bind_group_layout,
        pipeline,
    });
}

pub fn pathtracer(
    view: ViewQuery<(
        &Pathtracer,
        &PathtracerAccumulationTexture,
        &ExtractedCamera,
        &ViewTarget,
        &ViewUniformOffset,
    )>,
    pathtracer_pipelines: Option<Res<PathtracerPipelines>>,
    far_field: Option<Res<FarFieldVisibility>>,
    far_field_bind_group: Option<Res<FarFieldVisibilityBindGroup>>,
    pipeline_cache: Res<PipelineCache>,
    scene_bindings: Res<RaytracingSceneBindings>,
    view_uniforms: Res<ViewUniforms>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (pathtracer_settings, accumulation_texture, camera, view_target, view_uniform_offset) =
        view.into_inner();

    let Some(pathtracer_pipelines) = pathtracer_pipelines else {
        return;
    };

    let (Some(pipeline), Some(scene_bind_group), Some(viewport), Some(view_uniforms_binding)) = (
        pipeline_cache.get_compute_pipeline(pathtracer_pipelines.pipeline),
        &scene_bindings.bind_group,
        camera.physical_viewport_size,
        view_uniforms.uniforms.binding(),
    ) else {
        return;
    };

    let bind_group = render_device.create_bind_group(
        "pathtracer_bind_group",
        &pipeline_cache.get_bind_group_layout(&pathtracer_pipelines.bind_group_layout),
        &BindGroupEntries::sequential((
            &accumulation_texture.0.default_view,
            view_target.get_unsampled_color_attachment().view,
            view_uniforms_binding,
        )),
    );

    let command_encoder = ctx.command_encoder();

    if pathtracer_settings.reset {
        command_encoder.clear_texture(
            &accumulation_texture.0.texture,
            &ImageSubresourceRange::default(),
        );
    }

    let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("pathtracer"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, scene_bind_group, &[]);
    pass.set_bind_group(1, &bind_group, &[view_uniform_offset.offset]);
    if far_field.is_some() {
        let Some(far_field_bind_group) = far_field_bind_group else {
            return;
        };
        pass.set_bind_group(FarFieldVisibility::BIND_GROUP, &far_field_bind_group.0, &[]);
    }
    pass.dispatch_workgroups(viewport.x.div_ceil(8), viewport.y.div_ceil(8), 1);
}
