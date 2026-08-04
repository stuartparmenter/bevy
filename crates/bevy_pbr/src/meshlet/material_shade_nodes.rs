use super::{
    material_pipeline_prepare::{
        MeshletViewMaterial, MeshletViewMaterialsDeferredGBufferPrepass,
        MeshletViewMaterialsMainOpaquePass, MeshletViewMaterialsPrepass,
    },
    resource_manager::{MeshletViewBindGroups, MeshletViewResources},
    InstanceManager,
};
use crate::{MeshViewBindGroup, PrepassViewBindGroup};
use bevy_camera::MainPassResolutionOverride;
use bevy_camera::Viewport;
use bevy_core_pipeline::prepass::{
    MotionVectorPrepass, PreviousViewUniformOffset, ViewPrepassTextures,
};
use bevy_ecs::{prelude::*, query::Has};
use bevy_render::{
    camera::ExtractedCamera,
    material_bind_groups::MaterialBindGroupAllocators,
    render_phase::TrackedRenderPass,
    render_resource::{
        BindGroup, LoadOp, Operations, PipelineCache, RenderPassDepthStencilAttachment,
        RenderPassDescriptor, RenderPipeline, StoreOp,
    },
    renderer::{RenderContext, ViewQuery},
    view::{ViewTarget, ViewUniformOffset},
};

/// The pipeline and bind group to draw `material` with, or `None` while it cannot be drawn.
///
/// A material mutated this frame has no bind group until `RenderSystems::PrepareBindGroups` rebuilds
/// its slab, and every material sharing that slab goes with it. So "the list is non-empty" is not
/// the same question as "there is anything to draw", and beginning a pass that draws nothing still
/// costs an attachment load and store.
fn drawable_material<'a>(
    material: &MeshletViewMaterial,
    instance_manager: &InstanceManager,
    pipeline_cache: &'a PipelineCache,
    allocators: &'a MaterialBindGroupAllocators,
) -> Option<(&'a RenderPipeline, &'a BindGroup)> {
    if !instance_manager.material_present_in_scene(&material.material_id) {
        return None;
    }
    Some((
        pipeline_cache.get_render_pipeline(material.pipeline)?,
        material.bind_group(allocators)?,
    ))
}

fn any_material_is_drawable(
    materials: &[MeshletViewMaterial],
    instance_manager: &InstanceManager,
    pipeline_cache: &PipelineCache,
    allocators: &MaterialBindGroupAllocators,
) -> bool {
    materials.iter().any(|material| {
        drawable_material(material, instance_manager, pipeline_cache, allocators).is_some()
    })
}

/// One fullscreen triangle draw per drawable material.
fn draw_meshlet_materials<'a>(
    render_pass: &mut TrackedRenderPass<'a>,
    materials: &[MeshletViewMaterial],
    instance_manager: &InstanceManager,
    pipeline_cache: &'a PipelineCache,
    allocators: &'a MaterialBindGroupAllocators,
) {
    for material in materials {
        if let Some((pipeline, bind_group)) =
            drawable_material(material, instance_manager, pipeline_cache, allocators)
        {
            let x = material.material_id * 3;
            render_pass.set_render_pipeline(pipeline);
            render_pass.set_bind_group(3, bind_group, &[]);
            render_pass.draw(x..(x + 3), 0..1);
        }
    }
}

///
/// Fullscreen shading pass based on the visibility buffer generated from rasterizing meshlets.
pub fn meshlet_main_opaque_pass(
    view: ViewQuery<(
        &ExtractedCamera,
        &ViewTarget,
        &MeshViewBindGroup,
        Option<&MainPassResolutionOverride>,
        &MeshletViewMaterialsMainOpaquePass,
        &MeshletViewBindGroups,
        &MeshletViewResources,
    )>,
    instance_manager: Res<InstanceManager>,
    pipeline_cache: Res<PipelineCache>,
    material_bind_group_allocators: Res<MaterialBindGroupAllocators>,
    mut ctx: RenderContext,
) {
    let (
        camera,
        target,
        mesh_view_bind_group,
        resolution_override,
        meshlet_view_materials,
        meshlet_view_bind_groups,
        meshlet_view_resources,
    ) = view.into_inner();

    if !any_material_is_drawable(
        meshlet_view_materials,
        &instance_manager,
        &pipeline_cache,
        &material_bind_group_allocators,
    ) {
        return;
    }

    let (Some(meshlet_material_depth), Some(meshlet_material_shade_bind_group)) = (
        meshlet_view_resources.material_depth.as_ref(),
        meshlet_view_bind_groups.material_shade.as_ref(),
    ) else {
        return;
    };

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("meshlet_material_opaque_3d_pass"),
        color_attachments: &[Some(target.get_color_attachment())],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: &meshlet_material_depth.default_view,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
    {
        render_pass.set_camera_viewport(&viewport);
    }

    render_pass.set_bind_group(
        0,
        &mesh_view_bind_group.main,
        &mesh_view_bind_group.main_offsets,
    );
    render_pass.set_bind_group(1, &mesh_view_bind_group.binding_array, &[]);
    render_pass.set_bind_group(2, meshlet_material_shade_bind_group, &[]);

    draw_meshlet_materials(
        &mut render_pass,
        meshlet_view_materials,
        &instance_manager,
        &pipeline_cache,
        &material_bind_group_allocators,
    );
}

///
/// Fullscreen pass to generate prepass textures based on the visibility buffer generated from rasterizing meshlets.
pub fn meshlet_prepass(
    view: ViewQuery<(
        &ExtractedCamera,
        &ViewPrepassTextures,
        &ViewUniformOffset,
        &PreviousViewUniformOffset,
        Option<&MainPassResolutionOverride>,
        Has<MotionVectorPrepass>,
        &MeshletViewMaterialsPrepass,
        &MeshletViewBindGroups,
        &MeshletViewResources,
    )>,
    prepass_view_bind_group: Res<PrepassViewBindGroup>,
    instance_manager: Res<InstanceManager>,
    pipeline_cache: Res<PipelineCache>,
    material_bind_group_allocators: Res<MaterialBindGroupAllocators>,
    mut ctx: RenderContext,
) {
    let (
        camera,
        view_prepass_textures,
        view_uniform_offset,
        previous_view_uniform_offset,
        resolution_override,
        view_has_motion_vector_prepass,
        meshlet_view_materials,
        meshlet_view_bind_groups,
        meshlet_view_resources,
    ) = view.into_inner();

    if !any_material_is_drawable(
        meshlet_view_materials,
        &instance_manager,
        &pipeline_cache,
        &material_bind_group_allocators,
    ) {
        return;
    }

    let (Some(meshlet_material_depth), Some(meshlet_material_shade_bind_group)) = (
        meshlet_view_resources.material_depth.as_ref(),
        meshlet_view_bind_groups.material_shade.as_ref(),
    ) else {
        return;
    };

    let color_attachments = vec![
        view_prepass_textures
            .normal
            .as_ref()
            .map(|normals_texture| normals_texture.get_attachment()),
        view_prepass_textures
            .motion_vectors
            .as_ref()
            .map(|motion_vectors_texture| motion_vectors_texture.get_attachment()),
        // Use None in place of Deferred attachments
        None,
        None,
    ];

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("meshlet_material_prepass"),
        color_attachments: &color_attachments,
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: &meshlet_material_depth.default_view,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
    {
        render_pass.set_camera_viewport(&viewport);
    }

    if view_has_motion_vector_prepass {
        render_pass.set_bind_group(
            0,
            prepass_view_bind_group.motion_vectors.as_ref().unwrap(),
            &[
                view_uniform_offset.offset,
                previous_view_uniform_offset.offset,
            ],
        );
    } else {
        render_pass.set_bind_group(
            0,
            prepass_view_bind_group.no_motion_vectors.as_ref().unwrap(),
            &[view_uniform_offset.offset],
        );
    }

    render_pass.set_bind_group(1, &prepass_view_bind_group.empty_bind_group, &[]);
    render_pass.set_bind_group(2, meshlet_material_shade_bind_group, &[]);

    draw_meshlet_materials(
        &mut render_pass,
        meshlet_view_materials,
        &instance_manager,
        &pipeline_cache,
        &material_bind_group_allocators,
    );
}

/// Fullscreen pass to generate a gbuffer based on the visibility buffer generated from rasterizing meshlets.
pub fn meshlet_deferred_gbuffer_prepass(
    view: ViewQuery<(
        &ExtractedCamera,
        &ViewPrepassTextures,
        &ViewUniformOffset,
        &PreviousViewUniformOffset,
        Option<&MainPassResolutionOverride>,
        Has<MotionVectorPrepass>,
        &MeshletViewMaterialsDeferredGBufferPrepass,
        &MeshletViewBindGroups,
        &MeshletViewResources,
    )>,
    prepass_view_bind_group: Res<PrepassViewBindGroup>,
    instance_manager: Res<InstanceManager>,
    pipeline_cache: Res<PipelineCache>,
    material_bind_group_allocators: Res<MaterialBindGroupAllocators>,
    mut ctx: RenderContext,
) {
    let (
        camera,
        view_prepass_textures,
        view_uniform_offset,
        previous_view_uniform_offset,
        resolution_override,
        view_has_motion_vector_prepass,
        meshlet_view_materials,
        meshlet_view_bind_groups,
        meshlet_view_resources,
    ) = view.into_inner();

    if !any_material_is_drawable(
        meshlet_view_materials,
        &instance_manager,
        &pipeline_cache,
        &material_bind_group_allocators,
    ) {
        return;
    }

    let (Some(meshlet_material_depth), Some(meshlet_material_shade_bind_group)) = (
        meshlet_view_resources.material_depth.as_ref(),
        meshlet_view_bind_groups.material_shade.as_ref(),
    ) else {
        return;
    };

    let color_attachments = vec![
        view_prepass_textures
            .normal
            .as_ref()
            .map(|normals_texture| normals_texture.get_attachment()),
        view_prepass_textures
            .motion_vectors
            .as_ref()
            .map(|motion_vectors_texture| motion_vectors_texture.get_attachment()),
        view_prepass_textures
            .deferred
            .as_ref()
            .map(|deferred_texture| deferred_texture.get_attachment()),
        view_prepass_textures
            .deferred_lighting_pass_id
            .as_ref()
            .map(|deferred_lighting_pass_id| deferred_lighting_pass_id.get_attachment()),
    ];

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("meshlet_material_deferred_prepass"),
        color_attachments: &color_attachments,
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: &meshlet_material_depth.default_view,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
    {
        render_pass.set_camera_viewport(&viewport);
    }

    if view_has_motion_vector_prepass {
        render_pass.set_bind_group(
            0,
            prepass_view_bind_group.motion_vectors.as_ref().unwrap(),
            &[
                view_uniform_offset.offset,
                previous_view_uniform_offset.offset,
            ],
        );
    } else {
        render_pass.set_bind_group(
            0,
            prepass_view_bind_group.no_motion_vectors.as_ref().unwrap(),
            &[view_uniform_offset.offset],
        );
    }

    render_pass.set_bind_group(1, &prepass_view_bind_group.empty_bind_group, &[]);
    render_pass.set_bind_group(2, meshlet_material_shade_bind_group, &[]);

    draw_meshlet_materials(
        &mut render_pass,
        meshlet_view_materials,
        &instance_manager,
        &pipeline_cache,
        &material_bind_group_allocators,
    );
}
