//! Driver-metered presentation of multiple textures per rendered frame.
//!
//! Frame generation features such as DLSS Frame Generation produce one or more generated
//! frames alongside each real rendered frame, and need to present them evenly spaced in
//! time. The [VK_NV_present_metering] Vulkan extension lets the driver meter the display
//! timing of the batch, so the frames are presented back to back.
//!
//! A producer marks a window as paced by inserting its main world window entity into
//! [`PacedWindows`] during extract, which suppresses the render thread's normal swapchain
//! acquisition, and then submits a [`PacedPresentPlan`] for the window before the end of
//! the [`RenderGraph`](crate::renderer::RenderGraph) schedule. During presentation each
//! plan frame is blitted to the surface with the screenshot fullscreen pipeline and
//! presented, with the driver metering the batch.
//!
//! Driver metering is used when [`PresentMeteringSupported`] is present in
//! [`AdditionalVulkanFeatures`](crate::renderer::raw_vulkan_init::AdditionalVulkanFeatures).
//! Whoever enables VK_NV_present_metering during raw Vulkan device creation, for example
//! the DLSS plugin through `dlss_wgpu::present_metering`, must insert the marker. Without
//! it the frames are presented unmetered, so producers should gate their support on it too.
//!
//! [VK_NV_present_metering]: https://registry.khronos.org/vulkan/specs/latest/man/html/VK_NV_present_metering.html

use super::{screenshot::ScreenshotToScreenPipeline, ExtractedWindow, SurfaceData};
use crate::{
    render_resource::{
        BindGroupEntries, PipelineCache, RenderPipeline, SpecializedRenderPipelines, TextureView,
    },
    renderer::{RenderDevice, RenderQueue},
    sync_world::MainEntity,
};
use bevy_ecs::{
    entity::{EntityHashMap, EntityHashSet},
    prelude::*,
    system::SystemState,
};
use bevy_log::warn;
use bevy_utils::default;
use core::mem;

/// Marker for [`AdditionalVulkanFeatures`](crate::renderer::raw_vulkan_init::AdditionalVulkanFeatures),
/// inserted by whoever enables the VK_NV_present_metering device extension during raw
/// Vulkan device creation. Paced presentation is driver-metered only when it is present.
pub struct PresentMeteringSupported;

/// `VkSetPresentConfigNV` from VK_NV_present_metering, chained onto the `VkPresentInfoKHR`
/// of the first present of a batch to have the driver meter the batch's display timing.
/// Defined locally because `bevy_render` depends on neither `dlss_wgpu` nor ash, and ash
/// does not have the extension yet either.
#[repr(C)]
struct SetPresentConfigNV {
    /// `VK_STRUCTURE_TYPE_SET_PRESENT_CONFIG_NV`
    s_type: i32,
    p_next: *const core::ffi::c_void,
    num_frames_per_batch: u32,
    present_config_feedback: u32,
}

const STRUCTURE_TYPE_SET_PRESENT_CONFIG_NV: i32 = 1000613000;

/// Windows whose swapchain acquisition and presentation are owned by paced presentation
/// this frame, keyed by main world window entity.
///
/// Cleared at the start of every extract by [`reset_paced_windows`]; a producer must
/// re-insert its window every frame it wants paced presentation, so ceasing to write
/// reverts the window to normal presentation. Windows marked here skip normal swapchain
/// acquisition and must receive a [`PacedPresentPlan`] before presentation, or nothing
/// will be presented for the frame.
#[derive(Resource, Default)]
pub struct PacedWindows(pub EntityHashSet);

/// System set containing [`reset_paced_windows`]. Producers writing [`PacedWindows`]
/// during extract must order themselves after this set.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct PacedWindowReset;

/// Clears [`PacedWindows`] each frame, in [`PacedWindowReset`].
pub fn reset_paced_windows(mut paced_windows: ResMut<PacedWindows>) {
    paced_windows.0.clear();
}

/// An ordered list of textures to present to a window, ending with the real rendered
/// frame, displayed at an even cadence by driver present metering.
pub struct PacedPresentPlan {
    /// Frames in presentation order. They must match the window surface size, and sRGB
    /// views should be used for correct encoding.
    pub frames: Vec<TextureView>,
}

/// Presentation plans for the current render frame, consumed during presentation.
#[derive(Resource, Default)]
pub struct PacedPresentPlans {
    plans: EntityHashMap<PacedPresentPlan>,
}

impl PacedPresentPlans {
    /// Queues `plan` for the main world `window` entity, replacing any plan already queued.
    pub fn insert(&mut self, window: Entity, plan: PacedPresentPlan) {
        self.plans.insert(window, plan);
    }
}

/// System parameters for [`present_paced_plans`], threaded in from `render_system`.
pub(crate) type PacedPresentState<'w, 's> = (
    ResMut<'w, SpecializedRenderPipelines<ScreenshotToScreenPipeline>>,
    ResMut<'w, PipelineCache>,
    Res<'w, ScreenshotToScreenPipeline>,
    Res<'w, RenderDevice>,
    Res<'w, RenderQueue>,
    Query<'w, 's, (MainEntity, &'s mut ExtractedWindow, Option<&'s SurfaceData>)>,
);

/// Drains [`PacedPresentPlans`] and presents each plan's frames back to back, metered by
/// the driver when [`PresentMeteringSupported`] is available. Returns the windows that
/// were presented, which must be skipped by normal presentation.
pub(crate) fn present_paced_plans(
    world: &mut World,
    state: &mut SystemState<PacedPresentState>,
) -> EntityHashSet {
    let mut presented = EntityHashSet::default();
    let mut plans = match world.get_resource_mut::<PacedPresentPlans>() {
        Some(mut plans) if !plans.plans.is_empty() => mem::take(&mut plans.plans),
        _ => return presented,
    };

    let metering_supported = world
        .get_resource::<crate::renderer::raw_vulkan_init::AdditionalVulkanFeatures>()
        .is_some_and(|features| features.has::<PresentMeteringSupported>());

    let Ok((
        mut pipelines,
        mut pipeline_cache,
        blit_pipeline,
        render_device,
        render_queue,
        mut windows,
    )) = state.get_mut(world)
    else {
        return presented;
    };

    for (window_entity, mut window, surface_data) in &mut windows {
        let Some(plan) = plans.remove(&window_entity) else {
            continue;
        };
        if plan.frames.is_empty() {
            continue;
        }
        let (Some(surface_data), Some(view_format)) =
            (surface_data, window.swap_chain_texture_view_format)
        else {
            warn!("No surface for paced window {window_entity}");
            continue;
        };

        // Release any swapchain texture acquired before this window became paced, so
        // presentation can acquire from the full pool
        drop(window.swap_chain_texture.take());
        window.swap_chain_texture_view = None;
        window.needs_initial_present = false;

        let pipeline_id = pipelines.specialize(&pipeline_cache, &blit_pipeline, view_format);
        pipeline_cache.block_on_render_pipeline(pipeline_id);
        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id).cloned() else {
            warn!("Failed to compile paced present blit pipeline");
            continue;
        };

        // Read by the driver during the batch's first present below, so it must outlive
        // that present
        let mut present_config = SetPresentConfigNV {
            s_type: STRUCTURE_TYPE_SET_PRESENT_CONFIG_NV,
            p_next: core::ptr::null(),
            num_frames_per_batch: plan.frames.len() as u32,
            present_config_feedback: 0,
        };

        let layout = pipeline_cache.get_bind_group_layout(&blit_pipeline.bind_group_layout);
        for (frame_index, frame) in plan.frames.iter().enumerate() {
            let Some(surface_texture) = acquire(surface_data, &render_device) else {
                break;
            };
            // The driver meters the whole batch starting at the next present. This is
            // armed only after a successful acquire, when a present is guaranteed to
            // follow in this scope, so the stashed pointer is consumed before
            // `present_config` drops.
            if frame_index == 0
                && metering_supported
                && let Some(hal_surface) =
                    unsafe { surface_data.surface.as_hal::<wgpu::hal::api::Vulkan>() }
            {
                // SAFETY: VK_NV_present_metering was enabled at device creation per
                // `PresentMeteringSupported`, and the chain stays valid until the present
                // below consumes it.
                unsafe {
                    hal_surface.set_next_present_chain((&raw mut present_config).cast());
                }
            }
            let bind_group = render_device.create_bind_group(
                "paced_present_bind_group",
                &layout,
                &BindGroupEntries::single(frame),
            );
            blit(
                &pipeline,
                view_format,
                &render_device,
                &render_queue,
                &bind_group,
                &surface_texture,
            );
            render_queue.present(surface_texture);
        }
        presented.insert(window_entity);
    }

    presented
}

fn acquire(
    surface_data: &SurfaceData,
    render_device: &RenderDevice,
) -> Option<wgpu::SurfaceTexture> {
    let mut status = surface_data.surface.get_current_texture();
    if matches!(status, wgpu::CurrentSurfaceTexture::Outdated) {
        render_device.configure_surface(&surface_data.surface, &surface_data.configuration);
        status = surface_data.surface.get_current_texture();
    }
    match status {
        wgpu::CurrentSurfaceTexture::Success(surface_texture)
        | wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => Some(surface_texture),
        // The window is not visible, there is nothing to present to
        wgpu::CurrentSurfaceTexture::Occluded => None,
        status => {
            warn!("Couldn't acquire paced swap chain texture: {status:?}");
            None
        }
    }
}

fn blit(
    pipeline: &RenderPipeline,
    view_format: wgpu::TextureFormat,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    bind_group: &crate::render_resource::BindGroup,
    surface_texture: &wgpu::SurfaceTexture,
) {
    let texture_view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor {
            format: Some(view_format),
            ..default()
        });
    let mut encoder = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("paced_present_blit"),
    });
    super::screenshot::fullscreen_blit_pass(
        &mut encoder,
        "paced_present_blit",
        &texture_view,
        wgpu::LoadOp::Clear(wgpu::Color::BLACK),
        pipeline,
        bind_group,
    );
    render_queue.submit([encoder.finish()]);
}
