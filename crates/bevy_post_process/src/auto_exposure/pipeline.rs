use super::compensation_curve::{
    AutoExposureCompensationCurve, AutoExposureCompensationCurveUniform,
};
use bevy_asset::{load_embedded_asset, prelude::*};
use bevy_ecs::prelude::*;
use bevy_image::Image;
use bevy_render::{
    globals::GlobalsUniform,
    render_resource::{binding_types::*, *},
    view::ViewUniform,
};
use bevy_shader::Shader;
use bevy_utils::default;
use core::num::NonZero;

#[derive(Resource)]
pub struct AutoExposurePipeline {
    pub histogram_layout: BindGroupLayoutDescriptor,
    pub histogram_shader: Handle<Shader>,
}

#[derive(Component)]
pub struct ViewAutoExposurePipeline {
    pub histogram_pipeline: CachedComputePipelineId,
    pub mean_luminance_pipeline: CachedComputePipelineId,
    pub compensation_curve: Handle<AutoExposureCompensationCurve>,
    pub metering_mask: Handle<Image>,
}

#[derive(ShaderType, Clone, Copy)]
pub struct AutoExposureUniform {
    pub(super) min_log_lum: f32,
    pub(super) inv_log_lum_range: f32,
    pub(super) log_lum_range: f32,
    pub(super) low_percent: f32,
    pub(super) high_percent: f32,
    pub(super) speed_up: f32,
    pub(super) speed_down: f32,
    pub(super) exponential_transition_distance: f32,
    /// The adapted exposure correction is clamped to this range, in EV.
    pub(super) correction_min: f32,
    pub(super) correction_max: f32,
    /// Tail padding to 48 bytes. The ten fields above occupy 40, and a WGSL
    /// uniform struct's size is rounded up to a multiple of 16; encase does not
    /// apply that rounding, so the mirror has to carry it explicitly or the
    /// uploaded buffer is smaller than the binding the shader declares.
    pub(super) pad_0: u32,
    pub(super) pad_1: u32,
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum AutoExposurePass {
    Histogram,
    Average,
}

pub const HISTOGRAM_BIN_COUNT: u64 = 64;

pub fn init_auto_exposure_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(AutoExposurePipeline {
        histogram_layout: BindGroupLayoutDescriptor::new(
            "compute histogram bind group",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer::<GlobalsUniform>(false),
                    uniform_buffer::<AutoExposureUniform>(false),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    texture_1d(TextureSampleType::Float { filterable: false }),
                    uniform_buffer::<AutoExposureCompensationCurveUniform>(false),
                    storage_buffer_sized(false, NonZero::<u64>::new(HISTOGRAM_BIN_COUNT * 4)),
                    storage_buffer_sized(false, NonZero::<u64>::new(4)),
                    storage_buffer::<ViewUniform>(true),
                ),
            ),
        ),
        histogram_shader: load_embedded_asset!(asset_server.as_ref(), "auto_exposure.wesl"),
    });
}

impl SpecializedComputePipeline for AutoExposurePipeline {
    type Key = AutoExposurePass;

    fn specialize(&self, pass: AutoExposurePass) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: Some("luminance compute pipeline".into()),
            layout: vec![self.histogram_layout.clone()],
            shader: self.histogram_shader.clone(),
            shader_defs: vec![],
            entry_point: Some(match pass {
                AutoExposurePass::Histogram => "compute_histogram".into(),
                AutoExposurePass::Average => "compute_average".into(),
            }),
            ..default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AutoExposureUniform;
    use bevy_render::render_resource::ShaderType;

    #[test]
    fn uniform_layout_matches_the_wesl_struct() {
        // naga computes span=48 for the WESL `AutoExposure` uniform struct (ten
        // sequential 4-byte scalars plus two words of tail padding); the encase
        // layout must agree or the uploaded buffer is smaller than the binding.
        assert_eq!(AutoExposureUniform::min_size().get(), 48);
    }
}
