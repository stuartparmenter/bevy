use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_camera::CompositingSpace;
use bevy_ecs::prelude::*;
use bevy_mesh::VertexBufferLayout;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    view::{ResolvedCompositingSpace, ViewUniform},
    WorkingColorSpace,
};
use bevy_shader::{Shader, ShaderDefVal};
use bevy_utils::default;

/// How a UI fragment encodes its output for the view it renders into,
/// shared by every UI pipeline key.
///
/// UI draws into the view's main texture after tonemapping, so each
/// fragment encodes to match that texture's compositing space.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct UiWriterEncodeKey {
    /// The view's resolved [`CompositingSpace`]. Linear views leave this
    /// `None`, since linear output needs no encode.
    pub compositing_space: Option<CompositingSpace>,
    /// Whether the view's main texture holds working color space values when UI draws,
    /// because the camera doesn't tonemap.
    pub output_working_color_space: bool,
}

impl UiWriterEncodeKey {
    /// Builds the key from a camera view's [`ResolvedCompositingSpace`] and [`ExtractedCamera`].
    pub fn new(
        resolved: Option<&ResolvedCompositingSpace>,
        camera: Option<&ExtractedCamera>,
        working_color_space: WorkingColorSpace,
    ) -> Self {
        Self {
            compositing_space: ResolvedCompositingSpace::space(resolved)
                .filter(|space| !space.is_linear()),
            output_working_color_space: working_color_space == WorkingColorSpace::Rec2020
                && camera.is_some_and(|camera| !camera.tonemaps()),
        }
    }

    /// Appends the shader defs for the compositing space and working color space.
    pub fn push_shader_defs(&self, shader_defs: &mut Vec<ShaderDefVal>) {
        if self.output_working_color_space {
            shader_defs.push("OUTPUT_WORKING_COLOR_SPACE".into());
        }
        match self.compositing_space {
            Some(CompositingSpace::Srgb) => shader_defs.push("COMPOSITING_SPACE_SRGB".into()),
            Some(CompositingSpace::Oklab) => shader_defs.push("COMPOSITING_SPACE_OKLAB".into()),
            Some(CompositingSpace::Linear) | None => {}
        }
    }
}

#[derive(Resource)]
pub struct UiPipeline {
    pub view_layout: BindGroupLayoutDescriptor,
    pub image_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
}

pub fn init_ui_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "ui_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );

    let image_layout = BindGroupLayoutDescriptor::new(
        "ui_image_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );

    commands.insert_resource(UiPipeline {
        view_layout,
        image_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "ui.wesl"),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct UiPipelineKey {
    pub target_format: TextureFormat,
    pub anti_alias: bool,
    pub writer_encode: UiWriterEncodeKey,
}

impl SpecializedRenderPipeline for UiPipeline {
    type Key = UiPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let vertex_layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Vertex,
            vec![
                // position
                VertexFormat::Float32x3,
                // uv
                VertexFormat::Float32x2,
                // color
                VertexFormat::Float32x4,
                // mode
                VertexFormat::Uint32,
                // border radius x values (top left, top right, bottom right, bottom left)
                VertexFormat::Float32x4,
                // border radius y values (top left, top right, bottom right, bottom left)
                VertexFormat::Float32x4,
                // border thickness
                VertexFormat::Float32x4,
                // border size
                VertexFormat::Float32x2,
                // position relative to the center
                VertexFormat::Float32x2,
            ],
        );
        let shader_defs = if key.anti_alias {
            vec!["ANTI_ALIAS".into()]
        } else {
            Vec::new()
        };
        let mut fragment_defs = shader_defs.clone();
        key.writer_encode.push_shader_defs(&mut fragment_defs);

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs,
                buffers: vec![vertex_layout],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: fragment_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            layout: vec![self.view_layout.clone(), self.image_layout.clone()],
            label: Some("ui_pipeline".into()),
            ..default()
        }
    }
}
