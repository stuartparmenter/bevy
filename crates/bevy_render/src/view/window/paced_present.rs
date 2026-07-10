//! Driver-metered presentation of multiple textures per rendered frame.
//!
//! Frame generation features (e.g. DLSS Frame Generation) produce one or more generated
//! frames alongside each real rendered frame, and need to present them evenly spaced in
//! time. The [VK_NV_present_metering] Vulkan extension
//! ([`Features::VULKAN_NV_PRESENT_METERING`]) lets the driver meter the display timing of
//! the batch, so the frames are simply presented back to back.
//!
//! A producer marks a window as paced by inserting its (main world) window entity into
//! [`PacedWindows`] during extract, which suppresses the render thread's normal swapchain
//! acquisition, and then submits a [`PacedPresentPlan`] for the window before the end of
//! the [`RenderGraph`](crate::renderer::RenderGraph) schedule. During presentation each
//! plan frame is blitted to the surface with the screenshot fullscreen pipeline and
//! presented, with the driver metering the batch.
//!
//! Without driver metering support the frames are presented unmetered, so producers
//! should gate their support on [`Features::VULKAN_NV_PRESENT_METERING`].
//!
//! [VK_NV_present_metering]: https://registry.khronos.org/vulkan/specs/latest/man/html/VK_NV_present_metering.html
//! [`Features::VULKAN_NV_PRESENT_METERING`]: wgpu::Features::VULKAN_NV_PRESENT_METERING

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
};
use bevy_log::warn;
use bevy_utils::default;
use core::mem;

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

/// Clears [`PacedWindows`] each frame. Producers writing [`PacedWindows`] during extract
/// must order themselves after this system.
pub fn reset_paced_windows(mut paced_windows: ResMut<PacedWindows>) {
    paced_windows.0.clear();
}

/// An ordered list of textures to present to a window, ending with the real rendered
/// frame, displayed at an even cadence by driver present metering.
pub struct PacedPresentPlan {
    /// Frames in presentation order. Must match the window surface size; sRGB views
    /// should be used for correct encoding.
    pub frames: Vec<TextureView>,
}

/// Presentation plans for the current render frame, consumed during presentation.
#[derive(Resource, Default)]
pub struct PacedPresentPlans {
    plans: EntityHashMap<PacedPresentPlan>,
}

impl PacedPresentPlans {
    /// Queues `plan` for `window` (main world entity), replacing any plan already queued.
    pub fn insert(&mut self, window: Entity, plan: PacedPresentPlan) {
        self.plans.insert(window, plan);
    }
}

/// Drains [`PacedPresentPlans`] and presents each plan's frames back to back, metered by
/// the driver when [`Features::VULKAN_NV_PRESENT_METERING`](wgpu::Features::VULKAN_NV_PRESENT_METERING)
/// is available. Returns the windows that were presented, which must be skipped by normal
/// presentation.
pub(crate) fn present_paced_plans(world: &mut World) -> EntityHashSet {
    let mut presented = EntityHashSet::default();
    let mut plans = match world.get_resource_mut::<PacedPresentPlans>() {
        Some(mut plans) if !plans.plans.is_empty() => mem::take(&mut plans.plans),
        _ => return presented,
    };

    world.resource_scope(
        |world, mut pipelines: Mut<SpecializedRenderPipelines<ScreenshotToScreenPipeline>>| {
            world.resource_scope(|world, mut pipeline_cache: Mut<PipelineCache>| {
                world.resource_scope(|world, blit_pipeline: Mut<ScreenshotToScreenPipeline>| {
                    let render_device = world.resource::<RenderDevice>().clone();
                    let render_queue = world.resource::<RenderQueue>().clone();
                    let metering_supported = render_device
                        .features()
                        .contains(wgpu::Features::VULKAN_NV_PRESENT_METERING);

                    let mut windows =
                        world.query::<(MainEntity, &mut ExtractedWindow, Option<&SurfaceData>)>();
                    for (window_entity, mut window, surface_data) in windows.iter_mut(world) {
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

                        // Release any swapchain texture acquired before this window
                        // became paced, so presentation can acquire from the full pool
                        drop(window.swap_chain_texture.take());
                        window.swap_chain_texture_view = None;
                        window.needs_initial_present = false;

                        let pipeline_id =
                            pipelines.specialize(&pipeline_cache, &blit_pipeline, view_format);
                        pipeline_cache.block_on_render_pipeline(pipeline_id);
                        let Some(pipeline) =
                            pipeline_cache.get_render_pipeline(pipeline_id).cloned()
                        else {
                            warn!("Failed to compile paced present blit pipeline");
                            continue;
                        };

                        // The driver meters the whole batch starting at the next present
                        if metering_supported
                            && let Some(hal_surface) =
                                unsafe { surface_data.surface.as_hal::<wgpu::hal::api::Vulkan>() }
                        {
                            hal_surface.set_next_present_config(plan.frames.len() as u32);
                        }

                        let layout =
                            pipeline_cache.get_bind_group_layout(&blit_pipeline.bind_group_layout);
                        for frame in &plan.frames {
                            let Some(surface_texture) = acquire(surface_data, &render_device)
                            else {
                                break;
                            };
                            blit(
                                &pipeline,
                                view_format,
                                &render_device,
                                &render_queue,
                                &render_device.create_bind_group(
                                    "paced_present_bind_group",
                                    &layout,
                                    &BindGroupEntries::single(frame),
                                ),
                                &surface_texture,
                            );
                            render_queue.present(surface_texture);
                        }
                        presented.insert(window_entity);
                    }
                });
            });
        },
    );

    presented
}

fn acquire(
    surface_data: &SurfaceData,
    render_device: &RenderDevice,
) -> Option<wgpu::SurfaceTexture> {
    match surface_data.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(surface_texture)
        | wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => Some(surface_texture),
        wgpu::CurrentSurfaceTexture::Outdated => {
            render_device.configure_surface(&surface_data.surface, &surface_data.configuration);
            match surface_data.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(surface_texture)
                | wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => Some(surface_texture),
                status => {
                    warn!(
                        "Couldn't acquire paced swap chain texture after reconfiguring: {status:?}"
                    );
                    None
                }
            }
        }
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
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("paced_present_blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    render_queue.submit([encoder.finish()]);
}
