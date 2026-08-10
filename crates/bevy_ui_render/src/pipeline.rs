use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_camera::CompositingSpace;
use bevy_core_pipeline::camera_stack::ViewStackContract;
use bevy_ecs::prelude::*;
use bevy_mesh::VertexBufferLayout;
use bevy_render::{
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    view::ViewUniform,
};
use bevy_shader::{Shader, ShaderDefVal};
use bevy_utils::default;

/// Writer-side encode state of the view a UI node renders into, shared by
/// every UI pipeline key.
///
/// UI colors are authored in Rec.709 and composite into the post-tonemap
/// buffer, so the fragment converts and encodes them to match it.
#[derive(Clone, Copy, Default, Hash, PartialEq, Eq)]
pub struct UiWriterEncodeKey {
    /// Resolved [`CompositingSpace`] of the view ([`ViewStackContract::compositing_space`]).
    pub compositing_space: Option<CompositingSpace>,
    /// Whether the post-tonemap buffer uses Rec.2020 primaries
    /// ([`ViewStackContract::source_gamut_is_rec2020`]).
    pub buffer_gamut_rec2020: bool,
}

impl UiWriterEncodeKey {
    /// Appends the writer-encode defs: gamut convert first, then compositing space.
    pub fn push_shader_defs(&self, shader_defs: &mut Vec<ShaderDefVal>) {
        if self.buffer_gamut_rec2020 {
            shader_defs.push("OUTPUT_GAMUT_REC2020".into());
        }
        match self.compositing_space {
            Some(CompositingSpace::Srgb) => shader_defs.push("COMPOSITING_SPACE_SRGB".into()),
            Some(CompositingSpace::Oklab) => shader_defs.push("COMPOSITING_SPACE_OKLAB".into()),
            Some(CompositingSpace::Linear) | None => {}
        }
    }
}

impl From<&ViewStackContract> for UiWriterEncodeKey {
    fn from(contract: &ViewStackContract) -> Self {
        Self {
            compositing_space: contract.compositing_space,
            buffer_gamut_rec2020: contract.source_gamut_is_rec2020(),
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
        let mut shader_defs = if key.anti_alias {
            vec!["ANTI_ALIAS".into()]
        } else {
            Vec::new()
        };
        key.writer_encode.push_shader_defs(&mut shader_defs);

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_layout],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn defs_for(key: UiWriterEncodeKey) -> Vec<ShaderDefVal> {
        let mut defs = Vec::new();
        key.push_shader_defs(&mut defs);
        defs
    }

    #[test]
    fn no_writer_encode_for_default_or_linear() {
        let baseline = vec![ShaderDefVal::from("ANTI_ALIAS")];

        let mut none_defs = baseline.clone();
        UiWriterEncodeKey::default().push_shader_defs(&mut none_defs);
        assert_eq!(none_defs, baseline);

        let mut linear_defs = baseline.clone();
        UiWriterEncodeKey {
            compositing_space: Some(CompositingSpace::Linear),
            buffer_gamut_rec2020: false,
        }
        .push_shader_defs(&mut linear_defs);
        assert_eq!(linear_defs, baseline);
    }

    #[test]
    fn compositing_space_appends_exactly_its_def() {
        assert_eq!(
            defs_for(UiWriterEncodeKey {
                compositing_space: Some(CompositingSpace::Srgb),
                buffer_gamut_rec2020: false,
            }),
            vec![ShaderDefVal::from("COMPOSITING_SPACE_SRGB")]
        );
        assert_eq!(
            defs_for(UiWriterEncodeKey {
                compositing_space: Some(CompositingSpace::Oklab),
                buffer_gamut_rec2020: false,
            }),
            vec![ShaderDefVal::from("COMPOSITING_SPACE_OKLAB")]
        );
    }

    #[test]
    fn rec2020_buffer_appends_exactly_the_gamut_def() {
        assert_eq!(
            defs_for(UiWriterEncodeKey {
                compositing_space: None,
                buffer_gamut_rec2020: true,
            }),
            vec![ShaderDefVal::from("OUTPUT_GAMUT_REC2020")]
        );
    }

    /// Def order is part of the shader-cache key, so it must be the same for every pipeline.
    #[test]
    fn combined_key_pushes_gamut_then_space() {
        assert_eq!(
            defs_for(UiWriterEncodeKey {
                compositing_space: Some(CompositingSpace::Srgb),
                buffer_gamut_rec2020: true,
            }),
            vec![
                ShaderDefVal::from("OUTPUT_GAMUT_REC2020"),
                ShaderDefVal::from("COMPOSITING_SPACE_SRGB"),
            ]
        );
    }
}
