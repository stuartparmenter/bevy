//! NVIDIA Deep Learning Super Sampling (DLSS).
//!
//! DLSS uses machine learning models to upscale and anti-alias images.
//!
//! Requires a NVIDIA RTX GPU, and the Windows/Linux Vulkan rendering backend. Does not work on other platforms.
//!
//! See https://github.com/bevyengine/dlss_wgpu for licensing requirements and setup instructions.
//!
//! # Usage
//! 1. Enable Bevy's `dlss` feature
//! 2. During app setup, insert the `DlssProjectId` resource before `DefaultPlugins`
//! 3. Check for the presence of `Option<Res<DlssSuperResolutionSupported>>` at runtime to see if DLSS is supported on the current machine
//! 4. Add the `Dlss` component to your camera entity, optionally setting a specific `DlssPerfQualityMode` (defaults to `Auto`)
//! 5. Optionally add sharpening via `ContrastAdaptiveSharpening`
//! 6. Custom rendering code, including third party crates, should account for the optional `MainPassResolutionOverride` to work with DLSS (see the `custom_render_phase` example)

mod extract;
mod frame_generation;
mod node;
mod prepare;

pub use dlss_wgpu::DlssPerfQualityMode;

use bevy_app::{App, Plugin};
use bevy_camera::Hdr;
use bevy_core_pipeline::{
    prepass::{DepthPrepass, MotionVectorPrepass},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::prelude::*;
use bevy_math::{UVec2, Vec2};
use bevy_reflect::{prelude::ReflectDefault, reflect_remote, Reflect};
use bevy_render::{
    camera::{MipBias, TemporalJitter},
    renderer::{
        raw_vulkan_init::{AdditionalVulkanFeatures, RawVulkanInitSettings},
        RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue,
    },
    texture::CachedTexture,
    view::{prepare_view_attachments, prepare_view_targets, window::paced_present},
    ExtractSchedule, Render, RenderApp, RenderSystems,
};
use dlss_wgpu::{
    ray_reconstruction::{
        DlssRayReconstruction, DlssRayReconstructionDepthMode, DlssRayReconstructionRoughnessMode,
    },
    super_resolution::DlssSuperResolution,
    FeatureSupport,
};
use std::{
    marker::PhantomData,
    ops::Deref,
    sync::{Arc, Mutex},
};
use tracing::info;
use uuid::Uuid;

/// Initializes DLSS support in the renderer. This must be registered before [`RenderPlugin`](bevy_render::RenderPlugin) because
/// it configures render init code.
#[derive(Default)]
pub struct DlssInitPlugin;

impl Plugin for DlssInitPlugin {
    #[allow(unsafe_code)]
    fn build(&self, app: &mut App) {
        let dlss_project_id = app.world().get_resource::<DlssProjectId>()
                        .expect("The `dlss` feature is enabled, but DlssProjectId was not added to the App before DlssInitPlugin.").0;
        let mut raw_vulkan_settings = app
            .world_mut()
            .get_resource_or_init::<RawVulkanInitSettings>();

        // SAFETY: this does not remove any instance features and only enables features that are supported
        unsafe {
            raw_vulkan_settings.add_create_instance_callback(
                move |mut args, additional_vulkan_features| {
                    let mut feature_support = FeatureSupport::default();
                    match dlss_wgpu::register_instance_extensions(
                        dlss_project_id,
                        &mut args,
                        &mut feature_support,
                    ) {
                        Ok(_) => {
                            if feature_support.super_resolution_supported {
                                additional_vulkan_features.insert::<DlssSuperResolutionSupported>();
                            }
                            if feature_support.ray_reconstruction_supported {
                                additional_vulkan_features
                                    .insert::<DlssRayReconstructionSupported>();
                            }
                            if feature_support.frame_generation_supported {
                                additional_vulkan_features.insert::<DlssFrameGenerationSupported>();
                            }
                        }
                        Err(_) => {}
                    }
                },
            );
        }

        // SAFETY: this does not remove any device features and only enables features that are supported
        unsafe {
            raw_vulkan_settings.add_create_device_callback(
                move |mut args, adapter, additional_vulkan_features| {
                    let mut feature_support = FeatureSupport::default();
                    match dlss_wgpu::register_device_extensions(
                        dlss_project_id,
                        &mut args,
                        adapter,
                        &mut feature_support,
                    ) {
                        Ok(_) => {
                            if feature_support.super_resolution_supported {
                                additional_vulkan_features.insert::<DlssSuperResolutionSupported>();
                            } else {
                                additional_vulkan_features.remove::<DlssSuperResolutionSupported>();
                            }
                            if feature_support.ray_reconstruction_supported {
                                additional_vulkan_features
                                    .insert::<DlssRayReconstructionSupported>();
                            } else {
                                additional_vulkan_features
                                    .remove::<DlssRayReconstructionSupported>();
                            }
                            if feature_support.frame_generation_supported {
                                additional_vulkan_features.insert::<DlssFrameGenerationSupported>();
                            } else {
                                additional_vulkan_features.remove::<DlssFrameGenerationSupported>();
                            }
                        }
                        Err(_) => {}
                    }

                    // Paced presentation of generated frames relies on driver present
                    // metering
                    if dlss_wgpu::present_metering::register_present_metering(&mut args, adapter) {
                        additional_vulkan_features
                            .insert::<paced_present::PresentMeteringSupported>();
                    }
                },
            )
        };
    }
}

/// Enables DLSS support. This requires [`DlssInitPlugin`] to function, which must be manually registered in the correct order
/// prior to registering this plugin.
#[derive(Default)]
pub struct DlssPlugin;

impl Plugin for DlssPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Dlss<DlssSuperResolutionFeature>>()
            .register_type::<Dlss<DlssRayReconstructionFeature>>()
            .register_type::<DlssFrameGeneration>();
    }

    fn finish(&self, app: &mut App) {
        let (
            super_resolution_supported,
            ray_reconstruction_supported,
            frame_generation_supported,
            present_metering_supported,
        ) = {
            let features = app
                .sub_app_mut(RenderApp)
                .world()
                .resource::<AdditionalVulkanFeatures>();
            (
                features.has::<DlssSuperResolutionSupported>(),
                features.has::<DlssRayReconstructionSupported>(),
                features.has::<DlssFrameGenerationSupported>(),
                features.has::<paced_present::PresentMeteringSupported>(),
            )
        };
        if !super_resolution_supported && !frame_generation_supported {
            return;
        }

        let wgpu_device = {
            let render_world = app.sub_app(RenderApp).world();
            let render_device = render_world.resource::<RenderDevice>().wgpu_device();
            render_device.clone()
        };
        let project_id = app.world().get_resource::<DlssProjectId>()
            .expect("The `dlss` feature is enabled, but DlssProjectId was not added to the App before DlssPlugin.");
        let Ok(dlss_sdk) = dlss_wgpu::DlssSdk::new(project_id.0, wgpu_device) else {
            info!("DLSS is not supported on this system");
            return;
        };
        let (
            super_resolution_supported,
            ray_reconstruction_supported,
            frame_generation_supported,
            max_frames_to_generate,
        ) = {
            let sdk = dlss_sdk.lock().unwrap();
            (
                super_resolution_supported
                    && sdk.feature_supported(dlss_wgpu::DlssFeature::SuperResolution),
                ray_reconstruction_supported
                    && sdk.feature_supported(dlss_wgpu::DlssFeature::RayReconstruction),
                frame_generation_supported
                    && sdk.feature_supported(dlss_wgpu::DlssFeature::FrameGeneration)
                    && present_metering_supported,
                sdk.multi_frame_count_max(),
            )
        };

        if super_resolution_supported {
            app.insert_resource(DlssSuperResolutionSupported);
        }
        if super_resolution_supported && ray_reconstruction_supported {
            app.insert_resource(DlssRayReconstructionSupported);
        }
        if frame_generation_supported {
            let supported = DlssFrameGenerationSupported {
                max_frames_to_generate,
            };
            app.insert_resource(supported);
            app.sub_app_mut(RenderApp).insert_resource(supported);
        }

        app.sub_app_mut(RenderApp)
            .insert_resource(DlssSdk(dlss_sdk))
            .add_systems(
                ExtractSchedule,
                (
                    extract::extract_dlss::<DlssSuperResolutionFeature>,
                    extract::extract_dlss::<DlssRayReconstructionFeature>,
                ),
            )
            .add_systems(
                Render,
                (
                    prepare::prepare_dlss::<DlssSuperResolutionFeature>,
                    prepare::prepare_dlss::<DlssRayReconstructionFeature>,
                )
                    .in_set(RenderSystems::PrepareViews)
                    .before(prepare_view_targets),
            );

        if frame_generation_supported {
            app.sub_app_mut(RenderApp)
                .add_systems(
                    ExtractSchedule,
                    frame_generation::extract_frame_generation
                        .after(paced_present::reset_paced_windows)
                        // Reads the normalized render target off ExtractedCamera
                        .after(bevy_render::camera::extract_cameras),
                )
                .add_systems(
                    Render,
                    frame_generation::prepare_frame_generation
                        .in_set(RenderSystems::PrepareViews)
                        .after(prepare_view_attachments)
                        // Window screenshots also override the target attachment; frame
                        // generation deterministically wins (screenshots of paced windows
                        // are unsupported)
                        .after(bevy_render::view::window::screenshot::ScreenshotPreparation)
                        .after(prepare::prepare_dlss::<DlssSuperResolutionFeature>)
                        .after(prepare::prepare_dlss::<DlssRayReconstructionFeature>)
                        .before(prepare_view_targets),
                )
                // Runs after all cameras have rendered, so that overlay cameras targeting
                // the same window are composited into the frame generation input first
                .add_systems(
                    RenderGraph,
                    frame_generation::frame_generation
                        .in_set(RenderGraphSystems::Render)
                        .after(bevy_core_pipeline::schedule::camera_driver),
                );
        }

        app.sub_app_mut(RenderApp).add_systems(
            Core3d,
            (node::dlss_super_resolution, node::dlss_ray_reconstruction)
                .chain()
                .in_set(Core3dSystems::EarlyPostProcess),
        );
    }
}

/// Camera component to enable DLSS.
#[derive(Component, Reflect, Clone)]
#[reflect(Component)]
#[require(TemporalJitter, MipBias, DepthPrepass, MotionVectorPrepass, Hdr)]
pub struct Dlss<F: DlssFeature = DlssSuperResolutionFeature> {
    /// How much upscaling should be applied.
    #[reflect(remote = DlssPerfQualityModeRemoteReflect)]
    pub perf_quality_mode: DlssPerfQualityMode,
    /// Set to true to delete the saved temporal history (past frames).
    ///
    /// Useful for preventing ghosting when the history is no longer
    /// representative of the current frame, such as in sudden camera cuts.
    ///
    /// After setting this to true, it will automatically be toggled
    /// back to false at the end of the frame.
    pub reset: bool,
    #[reflect(ignore)]
    pub _phantom_data: PhantomData<F>,
}

impl Default for Dlss<DlssSuperResolutionFeature> {
    fn default() -> Self {
        Self {
            perf_quality_mode: Default::default(),
            reset: Default::default(),
            _phantom_data: Default::default(),
        }
    }
}

pub trait DlssFeature: Reflect + Clone + Default {
    type Context: Send;

    fn upscaled_resolution(context: &Self::Context) -> UVec2;

    fn render_resolution(context: &Self::Context) -> UVec2;

    fn suggested_jitter(
        context: &Self::Context,
        frame_number: u32,
        render_resolution: UVec2,
    ) -> Vec2;

    fn suggested_mip_bias(context: &Self::Context, render_resolution: UVec2) -> f32;

    fn new_context(
        upscaled_resolution: UVec2,
        perf_quality_mode: DlssPerfQualityMode,
        feature_flags: dlss_wgpu::DlssFeatureFlags,
        sdk: Arc<Mutex<dlss_wgpu::DlssSdk>>,
        device: &RenderDevice,
        queue: &RenderQueue,
    ) -> Result<Self::Context, dlss_wgpu::DlssError>;
}

/// DLSS Super Resolution.
///
/// Only available when the [`DlssSuperResolutionSupported`] resource exists.
#[derive(Reflect, Clone, Default)]
pub struct DlssSuperResolutionFeature;

impl DlssFeature for DlssSuperResolutionFeature {
    type Context = DlssSuperResolution;

    fn upscaled_resolution(context: &Self::Context) -> UVec2 {
        context.upscaled_resolution().into()
    }

    fn render_resolution(context: &Self::Context) -> UVec2 {
        context.render_resolution().into()
    }

    fn suggested_jitter(
        context: &Self::Context,
        frame_number: u32,
        render_resolution: UVec2,
    ) -> Vec2 {
        context
            .suggested_jitter(frame_number, render_resolution.to_array())
            .into()
    }

    fn suggested_mip_bias(context: &Self::Context, render_resolution: UVec2) -> f32 {
        context.suggested_mip_bias(render_resolution.to_array())
    }

    fn new_context(
        upscaled_resolution: UVec2,
        perf_quality_mode: DlssPerfQualityMode,
        feature_flags: dlss_wgpu::DlssFeatureFlags,
        sdk: Arc<Mutex<dlss_wgpu::DlssSdk>>,
        device: &RenderDevice,
        queue: &RenderQueue,
    ) -> Result<Self::Context, dlss_wgpu::DlssError> {
        DlssSuperResolution::new(
            upscaled_resolution.to_array(),
            perf_quality_mode,
            feature_flags,
            sdk,
            device.wgpu_device(),
            queue.deref(),
        )
    }
}

/// DLSS Ray Reconstruction.
///
/// Only available when the [`DlssRayReconstructionSupported`] resource exists.
#[derive(Reflect, Clone, Default)]
pub struct DlssRayReconstructionFeature;

impl DlssFeature for DlssRayReconstructionFeature {
    type Context = DlssRayReconstruction;

    fn upscaled_resolution(context: &Self::Context) -> UVec2 {
        context.upscaled_resolution().into()
    }

    fn render_resolution(context: &Self::Context) -> UVec2 {
        context.render_resolution().into()
    }

    fn suggested_jitter(
        context: &Self::Context,
        frame_number: u32,
        render_resolution: UVec2,
    ) -> Vec2 {
        context
            .suggested_jitter(frame_number, render_resolution.to_array())
            .into()
    }

    fn suggested_mip_bias(context: &Self::Context, render_resolution: UVec2) -> f32 {
        context.suggested_mip_bias(render_resolution.to_array())
    }

    fn new_context(
        upscaled_resolution: UVec2,
        perf_quality_mode: DlssPerfQualityMode,
        feature_flags: dlss_wgpu::DlssFeatureFlags,
        sdk: Arc<Mutex<dlss_wgpu::DlssSdk>>,
        device: &RenderDevice,
        queue: &RenderQueue,
    ) -> Result<Self::Context, dlss_wgpu::DlssError> {
        DlssRayReconstruction::new(
            upscaled_resolution.to_array(),
            perf_quality_mode,
            feature_flags,
            DlssRayReconstructionRoughnessMode::Packed,
            DlssRayReconstructionDepthMode::Hardware,
            sdk,
            device.wgpu_device(),
            queue.deref(),
        )
    }
}

/// Additional textures needed as inputs for [`DlssRayReconstructionFeature`].
#[derive(Component)]
pub struct ViewDlssRayReconstructionTextures {
    pub diffuse_albedo: CachedTexture,
    pub specular_albedo: CachedTexture,
    pub normal_roughness: CachedTexture,
    pub specular_motion_vectors: CachedTexture,
}

#[reflect_remote(DlssPerfQualityMode)]
#[derive(Default)]
enum DlssPerfQualityModeRemoteReflect {
    #[default]
    Auto,
    Dlaa,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

#[derive(Resource)]
struct DlssSdk(Arc<Mutex<dlss_wgpu::DlssSdk>>);

/// Application-specific ID for DLSS.
///
/// See the DLSS programming guide for more info.
#[derive(Resource, Clone)]
pub struct DlssProjectId(pub Uuid);

/// When DLSS Super Resolution is supported by the current system, this resource will exist in the main world.
/// Otherwise this resource will be absent.
#[derive(Resource, Clone, Copy)]
pub struct DlssSuperResolutionSupported;

/// When DLSS Ray Reconstruction is supported by the current system, this resource will exist in the main world.
/// Otherwise this resource will be absent.
#[derive(Resource, Clone, Copy)]
pub struct DlssRayReconstructionSupported;

/// Camera component that enables DLSS Frame Generation.
///
/// Requires a perspective projection, a window render target, and [`Msaa::Off`](bevy_render::view::Msaa::Off).
///
/// Check for the presence of `Option<Res<DlssFrameGenerationSupported>>` at runtime to see if
/// frame generation is supported on the current machine, and use
/// [`DlssFrameGenerationSupported::supports`] to determine which modes are valid.
#[derive(Component, Reflect, Clone, Default)]
#[reflect(Component, Default)]
#[require(DepthPrepass, MotionVectorPrepass, Hdr)]
pub struct DlssFrameGeneration {
    /// Presented frame rate multiplier.
    ///
    /// Modes above what the current machine supports (see [`DlssFrameGenerationSupported::max_mode`])
    /// fall back to the highest supported mode, with a warning.
    pub mode: DlssFrameGenerationMode,
    /// Set to true after a camera cut or other discontinuity.
    ///
    /// This is automatically reset to false after extraction.
    pub reset: bool,
}

/// How many total frames DLSS Frame Generation presents per rendered frame.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum DlssFrameGenerationMode {
    /// One generated frame per rendered frame, doubling the presented frame rate.
    ///
    /// Supported by all frame generation capable GPUs (RTX 40 series and newer).
    #[default]
    X2,
    /// Two generated frames per rendered frame, tripling the presented frame rate.
    ///
    /// Requires multi frame generation support (RTX 50 series and newer).
    X3,
    /// Three generated frames per rendered frame, quadrupling the presented frame rate.
    ///
    /// Requires multi frame generation support (RTX 50 series and newer).
    X4,
}

impl DlssFrameGenerationMode {
    /// The number of frames generated between each pair of rendered frames.
    pub const fn frames_to_generate(self) -> u32 {
        match self {
            Self::X2 => 1,
            Self::X3 => 2,
            Self::X4 => 3,
        }
    }

    /// The presented frame rate multiplier.
    pub const fn multiplier(self) -> u32 {
        self.frames_to_generate() + 1
    }
}

/// Present when DLSS Frame Generation is supported by the active adapter and driver.
#[derive(Resource, Clone, Copy)]
pub struct DlssFrameGenerationSupported {
    max_frames_to_generate: u32,
}

impl DlssFrameGenerationSupported {
    /// The maximum number of frames that can be generated between each pair of rendered frames.
    ///
    /// 1 means only [`DlssFrameGenerationMode::X2`] is supported, 3 means up to
    /// [`DlssFrameGenerationMode::X4`].
    pub fn max_frames_to_generate(&self) -> u32 {
        self.max_frames_to_generate
    }

    /// Whether `mode` is supported by the current machine.
    pub fn supports(&self, mode: DlssFrameGenerationMode) -> bool {
        mode.frames_to_generate() <= self.max_frames_to_generate
    }

    /// The highest supported [`DlssFrameGenerationMode`], for building settings UIs.
    pub fn max_mode(&self) -> DlssFrameGenerationMode {
        match self.max_frames_to_generate {
            1 => DlssFrameGenerationMode::X2,
            2 => DlssFrameGenerationMode::X3,
            _ => DlssFrameGenerationMode::X4,
        }
    }
}
