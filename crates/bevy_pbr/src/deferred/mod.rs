use crate::{
    contact_shadows::ContactShadows, DistanceFog, ExtractedAtmosphere, MeshPipeline,
    MeshPipelineKey, MeshPipelineSystems, MeshViewBindGroup, RenderViewLightProbes,
    ScreenSpaceAmbientOcclusion, ScreenSpaceReflectionsUniform, ScreenSpaceTransmission,
};
use bevy_app::prelude::*;
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer, Handle};
use bevy_core_pipeline::{
    core_3d::main_opaque_pass_3d,
    deferred::{
        copy_lighting_id::DeferredLightingIdDepthTexture, DEFERRED_LIGHTING_PASS_ID_DEPTH_FORMAT,
    },
    oit::OrderIndependentTransparencySettingsOffset,
    prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::prelude::*;
use bevy_light::{EnvironmentMapLight, IrradianceVolume, ShadowFilteringMethod};
use bevy_render::{
    extract_component::{
        ComponentUniforms, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
    },
    render_resource::{binding_types::uniform_buffer, *},
    renderer::{RenderContext, ViewQuery},
    view::{ExtractedView, ViewTarget},
    Render, RenderApp, RenderSystems,
};
use bevy_render::{GpuResourceAppExt, RenderStartup};
use bevy_shader::Shader;
use bevy_utils::default;

pub struct DeferredPbrLightingPlugin;

pub const DEFAULT_PBR_DEFERRED_LIGHTING_PASS_ID: u8 = 1;

/// Component with a `depth_id` for specifying which corresponding materials should be rendered by this specific PBR deferred lighting pass.
///
/// Will be automatically added to entities with the [`DeferredPrepass`] component that don't already have a [`PbrDeferredLightingDepthId`].
#[derive(Component, Clone, Copy, ExtractComponent, ShaderType)]
pub struct PbrDeferredLightingDepthId {
    depth_id: u32,

    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    _webgl2_padding_0: f32,
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    _webgl2_padding_1: f32,
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    _webgl2_padding_2: f32,
}

impl PbrDeferredLightingDepthId {
    pub fn new(value: u8) -> PbrDeferredLightingDepthId {
        PbrDeferredLightingDepthId {
            depth_id: value as u32,

            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            _webgl2_padding_0: 0.0,
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            _webgl2_padding_1: 0.0,
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            _webgl2_padding_2: 0.0,
        }
    }

    pub fn set(&mut self, value: u8) {
        self.depth_id = value as u32;
    }

    pub fn get(&self) -> u8 {
        self.depth_id as u8
    }
}

impl Default for PbrDeferredLightingDepthId {
    fn default() -> Self {
        PbrDeferredLightingDepthId {
            depth_id: DEFAULT_PBR_DEFERRED_LIGHTING_PASS_ID as u32,

            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            _webgl2_padding_0: 0.0,
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            _webgl2_padding_1: 0.0,
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            _webgl2_padding_2: 0.0,
        }
    }
}

impl Plugin for DeferredPbrLightingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractComponentPlugin::<PbrDeferredLightingDepthId>::default(),
            UniformComponentPlugin::<PbrDeferredLightingDepthId>::default(),
        ))
        .add_systems(PostUpdate, insert_deferred_lighting_pass_id_component);

        embedded_asset!(app, "deferred_lighting.wgsl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_gpu_resource::<SpecializedRenderPipelines<DeferredLightingLayout>>()
            .add_systems(
                RenderStartup,
                init_deferred_lighting_layout.after(MeshPipelineSystems),
            )
            .add_systems(
                Render,
                prepare_deferred_lighting_pipelines.in_set(RenderSystems::Prepare),
            )
            .add_systems(
                Core3d,
                deferred_lighting
                    .before(main_opaque_pass_3d)
                    .in_set(Core3dSystems::MainPass),
            );
    }
}

pub fn deferred_lighting(
    view: ViewQuery<(
        &MeshViewBindGroup,
        &ViewTarget,
        &DeferredLightingIdDepthTexture,
        &DeferredLightingPipeline,
    )>,
    pipeline_cache: Res<PipelineCache>,
    deferred_lighting_layout: Res<DeferredLightingLayout>,
    deferred_lighting_pass_id: Res<ComponentUniforms<PbrDeferredLightingDepthId>>,
    mut ctx: RenderContext,
) {
    let (
        mesh_view_bind_group,
        target,
        deferred_lighting_id_depth_texture,
        deferred_lighting_pipeline,
    ) = view.into_inner();

    let Some(pipeline) = pipeline_cache.get_render_pipeline(deferred_lighting_pipeline.pipeline_id)
    else {
        return;
    };

    let Some(deferred_lighting_pass_id_binding) = deferred_lighting_pass_id.uniforms().binding()
    else {
        return;
    };

    let bind_group_2 = ctx.render_device().create_bind_group(
        "deferred_lighting_layout_group_2",
        &pipeline_cache.get_bind_group_layout(&deferred_lighting_layout.bind_group_layout_2),
        &BindGroupEntries::single(deferred_lighting_pass_id_binding),
    );

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("deferred_lighting"),
        color_attachments: &[Some(target.get_color_attachment())],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: &deferred_lighting_id_depth_texture.texture.default_view,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: StoreOp::Discard,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    render_pass.set_render_pipeline(pipeline);

    render_pass.set_bind_group(
        0,
        &mesh_view_bind_group.main,
        &mesh_view_bind_group.main_offsets,
    );
    render_pass.set_bind_group(1, &mesh_view_bind_group.binding_array, &[]);
    render_pass.set_bind_group(2, &bind_group_2, &[]);
    render_pass.draw(0..3, 0..1);
}

#[derive(Resource)]
pub struct DeferredLightingLayout {
    mesh_pipeline: MeshPipeline,
    bind_group_layout_2: BindGroupLayoutDescriptor,
    deferred_lighting_shader: Handle<Shader>,
}

#[derive(Component)]
pub struct DeferredLightingPipeline {
    pub pipeline_id: CachedRenderPipelineId,
}

impl SpecializedRenderPipeline for DeferredLightingLayout {
    type Key = MeshPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();

        // Let the shader code know that it's running in a deferred pipeline.
        shader_defs.push("DEFERRED_LIGHTING_PIPELINE".into());

        // Project-global working-space axis (mirrors `MeshPipeline`): the
        // deferred lighting pass samples environment maps, which convert
        // into the working space under this def. Material colors in the
        // G-buffer were already converted by the (prepass) material
        // pipelines, so there is no double conversion here. Not pushed for
        // the default Rec.709 working space.
        if self.mesh_pipeline.working_color_space.is_rec2020() {
            shader_defs.push(
                bevy_render::working_color_space::WORKING_COLOR_SPACE_REC2020_SHADER_DEF.into(),
            );
        }

        #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
        shader_defs.push("WEBGL2".into());

        if key.contains(MeshPipelineKey::SCREEN_SPACE_AMBIENT_OCCLUSION) {
            shader_defs.push("SCREEN_SPACE_AMBIENT_OCCLUSION".into());
        }

        if key.contains(MeshPipelineKey::ENVIRONMENT_MAP) {
            shader_defs.push("ENVIRONMENT_MAP".into());
        }

        if key.contains(MeshPipelineKey::IRRADIANCE_VOLUME) {
            shader_defs.push("IRRADIANCE_VOLUME".into());
        }

        if key.contains(MeshPipelineKey::NORMAL_PREPASS) {
            shader_defs.push("NORMAL_PREPASS".into());
        }

        if key.contains(MeshPipelineKey::DEPTH_PREPASS) {
            shader_defs.push("DEPTH_PREPASS".into());
        }

        if key.contains(MeshPipelineKey::MOTION_VECTOR_PREPASS) {
            shader_defs.push("MOTION_VECTOR_PREPASS".into());
        }

        if key.contains(MeshPipelineKey::SCREEN_SPACE_REFLECTIONS) {
            shader_defs.push("SCREEN_SPACE_REFLECTIONS".into());
        }

        if key.contains(MeshPipelineKey::CONTACT_SHADOWS) {
            shader_defs.push("CONTACT_SHADOWS".into());
        }

        if key.contains(MeshPipelineKey::HAS_PREVIOUS_SKIN) {
            shader_defs.push("HAS_PREVIOUS_SKIN".into());
        }

        if key.contains(MeshPipelineKey::HAS_PREVIOUS_MORPH) {
            shader_defs.push("HAS_PREVIOUS_MORPH".into());
        }

        if key.contains(MeshPipelineKey::DISTANCE_FOG) {
            shader_defs.push("DISTANCE_FOG".into());
        }
        if key.contains(MeshPipelineKey::ATMOSPHERE) {
            shader_defs.push("ATMOSPHERE".into());
        }
        shader_defs.push("STANDARD_MATERIAL_CLEARCOAT".into());

        // Always true, since we're in the deferred lighting pipeline
        shader_defs.push("DEFERRED_PREPASS".into());

        let shadow_filter_method =
            key.intersection(MeshPipelineKey::SHADOW_FILTER_METHOD_RESERVED_BITS);
        if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2 {
            shader_defs.push("SHADOW_FILTER_METHOD_HARDWARE_2X2".into());
        } else if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_GAUSSIAN {
            shader_defs.push("SHADOW_FILTER_METHOD_GAUSSIAN".into());
        } else if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_TEMPORAL {
            shader_defs.push("SHADOW_FILTER_METHOD_TEMPORAL".into());
        }
        if self.mesh_pipeline.binding_arrays_are_usable {
            shader_defs.push("MULTIPLE_LIGHT_PROBES_IN_ARRAY".into());
            shader_defs.push("MULTIPLE_LIGHTMAPS_IN_ARRAY".into());
        }

        #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
        shader_defs.push("SIXTEEN_BYTE_ALIGNMENT".into());

        let layout = self.mesh_pipeline.get_view_layout(key.into());
        RenderPipelineDescriptor {
            label: Some("deferred_lighting_pipeline".into()),
            layout: vec![
                layout.main_layout,
                layout.binding_array_layout,
                self.bind_group_layout_2.clone(),
            ],
            vertex: VertexState {
                shader: self.deferred_lighting_shader.clone(),
                shader_defs: shader_defs.clone(),
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.deferred_lighting_shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format(),
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            depth_stencil: Some(DepthStencilState {
                format: DEFERRED_LIGHTING_PASS_ID_DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::Equal),
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            ..default()
        }
    }
}

pub fn init_deferred_lighting_layout(
    mut commands: Commands,
    mesh_pipeline: Res<MeshPipeline>,
    asset_server: Res<AssetServer>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "deferred_lighting_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<PbrDeferredLightingDepthId>(false),
        ),
    );
    commands.insert_resource(DeferredLightingLayout {
        mesh_pipeline: mesh_pipeline.clone(),
        bind_group_layout_2: layout,
        deferred_lighting_shader: load_embedded_asset!(
            asset_server.as_ref(),
            "deferred_lighting.wgsl"
        ),
    });
}

pub fn insert_deferred_lighting_pass_id_component(
    mut commands: Commands,
    views: Query<Entity, (With<DeferredPrepass>, Without<PbrDeferredLightingDepthId>)>,
) {
    for entity in views.iter() {
        commands
            .entity(entity)
            .insert(PbrDeferredLightingDepthId::default());
    }
}

pub fn prepare_deferred_lighting_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<DeferredLightingLayout>>,
    deferred_lighting_layout: Res<DeferredLightingLayout>,
    cameras: Query<(
        Entity,
        &ExtractedView,
        Option<&ShadowFilteringMethod>,
        (
            Has<ScreenSpaceAmbientOcclusion>,
            Has<ScreenSpaceReflectionsUniform>,
            Has<ContactShadows>,
            Has<DistanceFog>,
        ),
        (
            Has<NormalPrepass>,
            Has<DepthPrepass>,
            Has<MotionVectorPrepass>,
            Has<DeferredPrepass>,
        ),
        Has<RenderViewLightProbes<EnvironmentMapLight>>,
        Has<RenderViewLightProbes<IrradianceVolume>>,
        Option<&ScreenSpaceTransmission>,
        Has<OrderIndependentTransparencySettingsOffset>,
        Has<SkipDeferredLighting>,
        Has<ExtractedAtmosphere>,
    )>,
) {
    for (
        entity,
        view,
        shadow_filter_method,
        (ssao, ssr, contact_shadows, distance_fog),
        (normal_prepass, depth_prepass, motion_vector_prepass, deferred_prepass),
        has_environment_maps,
        has_irradiance_volumes,
        transmission,
        has_oit,
        skip_deferred_lighting,
        has_atmosphere,
    ) in &cameras
    {
        // If there is no deferred prepass or we want to skip the deferred lighting pass,
        // remove the old pipeline if there was one. This handles the case in which a
        // view using deferred stops using it.
        if !deferred_prepass || skip_deferred_lighting {
            commands.entity(entity).remove::<DeferredLightingPipeline>();
            continue;
        }

        let mut view_key = MeshPipelineKey::from_target_format(view.target_format);

        if normal_prepass {
            view_key |= MeshPipelineKey::NORMAL_PREPASS;
        }

        if depth_prepass {
            view_key |= MeshPipelineKey::DEPTH_PREPASS;
        }

        if motion_vector_prepass {
            view_key |= MeshPipelineKey::MOTION_VECTOR_PREPASS;
        }

        if has_atmosphere {
            view_key |= MeshPipelineKey::ATMOSPHERE;
        }

        if let Some(transmission) = transmission {
            view_key |= transmission.quality.pipeline_key();
        }

        if has_oit {
            view_key |= MeshPipelineKey::OIT_ENABLED;
        }

        if view.invert_culling {
            view_key |= MeshPipelineKey::INVERT_CULLING;
        }

        // Always true, since we're in the deferred lighting pipeline
        view_key |= MeshPipelineKey::DEFERRED_PREPASS;

        if ssao {
            view_key |= MeshPipelineKey::SCREEN_SPACE_AMBIENT_OCCLUSION;
        }
        if ssr {
            view_key |= MeshPipelineKey::SCREEN_SPACE_REFLECTIONS;
        }
        if contact_shadows {
            view_key |= MeshPipelineKey::CONTACT_SHADOWS;
        }
        if distance_fog {
            view_key |= MeshPipelineKey::DISTANCE_FOG;
        }

        // We don't need to check to see whether the environment map is loaded
        // because [`gather_light_probes`] already checked that for us before
        // adding the [`RenderViewEnvironmentMaps`] component.
        if has_environment_maps {
            view_key |= MeshPipelineKey::ENVIRONMENT_MAP;
        }

        if has_irradiance_volumes {
            view_key |= MeshPipelineKey::IRRADIANCE_VOLUME;
        }

        match shadow_filter_method.unwrap_or(&ShadowFilteringMethod::default()) {
            ShadowFilteringMethod::Hardware2x2 => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2;
            }
            ShadowFilteringMethod::Gaussian => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_GAUSSIAN;
            }
            ShadowFilteringMethod::Temporal => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_TEMPORAL;
            }
        }

        let pipeline_id =
            pipelines.specialize(&pipeline_cache, &deferred_lighting_layout, view_key);

        commands
            .entity(entity)
            .insert(DeferredLightingPipeline { pipeline_id });
    }
}

/// Component to skip running the deferred lighting pass in [`deferred_lighting`] for a specific view.
///
/// This works like [`crate::PbrPlugin::add_default_deferred_lighting_plugin`], but is per-view instead of global.
///
/// Useful for cases where you want to generate a gbuffer, but skip the built-in deferred lighting pass
/// to run your own custom lighting pass instead.
///
/// Insert this component in the render world only.
#[derive(Component, Clone, Copy, Default)]
pub struct SkipDeferredLighting;
