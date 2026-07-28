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
    working_color_space::{WorkingColorSpace, WORKING_COLOR_SPACE_REC2020_SHADER_DEF},
};
use bevy_shader::Shader;
use bevy_utils::default;
use core::num::NonZero;

#[derive(Resource)]
pub struct AutoExposurePipeline {
    pub histogram_layout: BindGroupLayoutDescriptor,
    pub histogram_shader: Handle<Shader>,
    /// The project-global working color space, captured at startup. When it
    /// is Rec.2020, the `WORKING_COLOR_SPACE_REC2020` shader def is pushed so
    /// the auto-white-balance chromaticity measurement uses the matching
    /// RGB → XYZ matrix; in the default Rec.709 working space no def is
    /// pushed and the composed shader source is unchanged.
    pub working_color_space: WorkingColorSpace,
}

#[derive(Component)]
pub struct ViewAutoExposurePipeline {
    pub histogram_pipeline: CachedComputePipelineId,
    pub mean_luminance_pipeline: CachedComputePipelineId,
    pub compensation_curve: Handle<AutoExposureCompensationCurve>,
    pub metering_mask: Handle<Image>,
}

/// CPU mirror of the `AutoExposure` settings uniform in `auto_exposure.wgsl`.
/// The field order and types must match the WGSL struct exactly.
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
    pub(super) metering_bias: f32,
    pub(super) long_term_speed_up: f32,
    pub(super) long_term_speed_down: f32,
    pub(super) long_term_bound_up: f32,
    pub(super) long_term_bound_down: f32,
    pub(super) physiological: u32,
    /// Auto white balance chromaticity adaptation speed, per second.
    pub(super) awb_speed: f32,
    /// Auto white balance virtual-light anchor luminance, scene-linear units.
    pub(super) awb_anchor: f32,
    /// Non-zero when the camera has an `AutoWhiteBalance` component. All auto
    /// white balance shader statements are gated on this flag, so the
    /// auto-exposure-only configuration runs no white-balance arithmetic at
    /// all.
    pub(super) awb_enabled: u32,
    /// Tail padding to 80 bytes. The seventeen fields above occupy 68, and a
    /// WGSL uniform struct's size is rounded up to a multiple of 16; encase
    /// does not apply that rounding, so the mirror has to carry it explicitly
    /// or the uploaded buffer is smaller than the binding the shader declares.
    pub(super) pad_0: u32,
    pub(super) pad_1: u32,
    pub(super) pad_2: u32,
}

/// CPU mirror of the per-view `AutoExposureState` storage buffer in `auto_exposure.wgsl`.
/// The field order and types must match the WGSL struct exactly.
#[derive(ShaderType, Clone, Copy, Debug, PartialEq)]
pub struct AutoExposureState {
    /// The smoothed short-term exposure correction, in EV.
    pub(super) exposure: f32,
    /// The long-term physiological adaptation envelope, in EV.
    pub(super) long_term: f32,
    /// The adapted white-point chromaticity (CIE 1931 *x*), used by auto
    /// white balance. Initialized to (and meaningless at) the D65 *x* value.
    pub(super) chroma_x: f32,
    /// The adapted white-point chromaticity (CIE 1931 *y*); see
    /// [`Self::chroma_x`].
    pub(super) chroma_y: f32,
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum AutoExposurePass {
    Histogram,
    Average,
}

pub const HISTOGRAM_BIN_COUNT: u64 = 64;

/// Size in bytes of the global chroma accumulator storage buffer
/// (three fixed-point `atomic<u32>` sums: `x·Y`, `y·Y`, and `Y`).
pub const CHROMA_ACCUMULATOR_SIZE: u64 = 12;

pub fn init_auto_exposure_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    working_color_space: Res<WorkingColorSpace>,
) {
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
                    storage_buffer::<AutoExposureState>(false),
                    storage_buffer::<ViewUniform>(true),
                    storage_buffer_sized(false, NonZero::<u64>::new(CHROMA_ACCUMULATOR_SIZE)),
                ),
            ),
        ),
        histogram_shader: load_embedded_asset!(asset_server.as_ref(), "auto_exposure.wgsl"),
        working_color_space: *working_color_space,
    });
}

impl SpecializedComputePipeline for AutoExposurePipeline {
    type Key = AutoExposurePass;

    fn specialize(&self, pass: AutoExposurePass) -> ComputePipelineDescriptor {
        let mut shader_defs = Vec::new();

        // Project-global working-space axis: pushed for every specialization
        // when (and only when) the app opted into the Rec.2020 working space,
        // so default projects compose byte-identically.
        if self.working_color_space.is_rec2020() {
            shader_defs.push(WORKING_COLOR_SPACE_REC2020_SHADER_DEF.into());
        }

        ComputePipelineDescriptor {
            label: Some("luminance compute pipeline".into()),
            layout: vec![self.histogram_layout.clone()],
            shader: self.histogram_shader.clone(),
            shader_defs,
            entry_point: Some(match pass {
                AutoExposurePass::Histogram => "compute_histogram".into(),
                AutoExposurePass::Average => "compute_average".into(),
            }),
            ..default()
        }
    }
}
