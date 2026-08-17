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
//! presented, with the driver metering the batch. A producer also declares its largest
//! batch in [`PacedSwapchainDepths`] once, so the swapchain is sized for the whole batch.
//!
//! Driver metering is armed when a plan carries a [`PresentChainLink`]. Whoever enables a
//! present pacing extension during device creation, for example the DLSS plugin through
//! `dlss_wgpu::present_metering`, builds the link and installs it on the plan. A plan
//! without a link presents its frames unmetered, back to back as fast as the swapchain
//! allows. The engine makes the single `set_next_present_chain` call per present, so
//! future engine-owned links can merge into one chain.
//!
//! # Ownership
//!
//! Marking a window in [`PacedWindows`] is an ownership claim. For that frame the paced
//! presenter owns swapchain acquisition and presentation for the window, including when
//! no plan arrives. A claimed window that receives no plan presents nothing that frame.
//! It is not handed back to normal presentation until the claim lapses on a later frame.
//!
//! # Producer contract
//!
//! Order producer extract systems after [`PacedWindowReset`]. The reset clears all
//! claims, so a claim written before it is silently dropped. Order them after
//! [`extract_cameras`](crate::camera::extract_cameras) too, because the normalized
//! window render target is only available on `ExtractedCamera` after it runs.
//!
//! Submit a [`PacedPresentPlan`] for every claimed window before the end of the
//! [`RenderGraph`](crate::renderer::RenderGraph) schedule. Plan textures are owned by
//! the pacer from submission until the release signal. See [`PlannedFrame`].
//!
//! Read [`PacedPresentedFrames`] during extract of the next frame to learn how many
//! frames were actually presented.
//!
//! Screenshots of paced windows are not supported. The pacer blits plan frames straight
//! to the surface and ignores the screenshot capture target.
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
use bevy_derive::Deref;
use bevy_ecs::{
    entity::{EntityHashMap, EntityHashSet},
    prelude::*,
    system::SystemState,
};
use bevy_log::{warn, warn_once};
use bevy_utils::default;
use core::{any::Any, ffi::c_void, mem};

/// A producer-owned Vulkan `pNext` chain, installed on a [`PacedPresentPlan`] and chained
/// by the engine onto the first present of the plan's batch.
///
/// The engine owns the single `set_next_present_chain` call per present. Producers never
/// touch the surface, they only hand over this link. The link owns its payload. The
/// engine keeps the plan alive until the window's presents are issued, and drops it after
/// the per-window present loop.
pub struct PresentChainLink {
    /// Points into `_payload`. The allocation is boxed and does not move, so the pointer
    /// stays valid until the payload drops.
    head: *mut c_void,
    /// Owns the allocation `head` points into.
    _payload: Box<dyn Any + Send + Sync>,
}

// SAFETY: `head` points into the boxed payload, which is `Send`. The engine never
// dereferences `head`, it only passes it to the driver.
unsafe impl Send for PresentChainLink {}
// SAFETY: The payload is `Sync`, and `head` is only read during presentation, which
// borrows the plan exclusively.
unsafe impl Sync for PresentChainLink {}

impl PresentChainLink {
    /// Wraps `payload` as the head of a present `pNext` chain.
    ///
    /// # Safety
    /// - `payload` must start with a valid Vulkan structure that can extend
    ///   `VkPresentInfoKHR`, with a correct `sType`.
    /// - Any structs reachable through the payload's `pNext` pointers must stay valid for
    ///   as long as the payload itself.
    /// - The payload must be freshly allocated for each plan. wgpu writes into the tail
    ///   of the chain, so a payload reused across frames retains a stale pointer.
    pub unsafe fn new<T: Any + Send + Sync>(mut payload: Box<T>) -> Self {
        let head = (&raw mut *payload).cast::<c_void>();
        Self {
            head,
            _payload: payload,
        }
    }

    /// The chain head passed to `set_next_present_chain`.
    fn head(&self) -> *mut c_void {
        self.head
    }
}

/// Windows claimed for paced presentation this frame, keyed by main world window entity.
///
/// A claim transfers ownership. While a window is claimed, the paced presenter owns
/// swapchain acquisition and presentation for it, including when no [`PacedPresentPlan`]
/// arrives. A claimed window without a plan presents nothing for the frame.
///
/// Cleared at the start of every extract by [`reset_paced_windows`]. A producer must
/// re-insert its window every frame it wants paced presentation, so ceasing to write
/// reverts the window to normal presentation.
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

/// Swapchain depth declared per window, keyed by main world window entity.
///
/// A producer declares the largest batch it will ever present, generated frames plus the
/// real frame. Surface configuration then sizes the swapchain so a full batch can queue
/// while one image is still on display. A smaller swapchain does not deadlock, but FIFO
/// acquire serializes the batch across vblanks and defeats present metering, which is
/// armed on the first present of a batch only.
///
/// The depth comes from the producer's maximum, never from a plan's length. Plan length
/// drops to one on reset and fallback frames, and reconfiguring the surface drains the
/// device.
///
/// Unlike [`PacedWindows`] this is not a per-frame ownership claim and is not cleared
/// between frames. Entries persist until the window closes, so toggling a producer off
/// does not reconfigure the surface. The cost is a deeper swapchain, and more permitted
/// frame latency, while no producer paces the window.
#[derive(Resource, Default)]
pub struct PacedSwapchainDepths(EntityHashMap<u32>);

impl PacedSwapchainDepths {
    /// Declares that plans for `window` can hold up to `frames_per_batch` frames.
    /// Keeps the largest value ever declared.
    pub fn declare(&mut self, window: Entity, frames_per_batch: u32) {
        let depth = self.0.entry(window).or_default();
        *depth = (*depth).max(frames_per_batch);
    }

    /// The declared depth for `window`, or zero when none was declared.
    pub fn get(&self, window: Entity) -> u32 {
        self.0.get(&window).copied().unwrap_or(0)
    }

    pub(crate) fn remove(&mut self, window: Entity) {
        self.0.remove(&window);
    }
}

/// One frame of a [`PacedPresentPlan`].
///
/// # Texture ownership
///
/// The texture is owned by the pacer from plan submission until an explicit release
/// signal. The release signal is the window's entry in [`PacedPresentedFrames`] for the
/// frame the plan was submitted in. Today the pacer presents and releases synchronously,
/// at the end of the present step of the same render frame. The contract permits the
/// pacer to hold the texture past `render_system`, so a producer must not reuse or free
/// a submitted texture before it observes the release signal. A producer that waits for
/// the release signal stays correct if the pacer later presents on another thread.
#[non_exhaustive]
pub struct PlannedFrame {
    /// The texture to blit to the surface. It must match the window surface size, and an
    /// sRGB view should be used for correct encoding.
    pub texture: TextureView,
}

impl PlannedFrame {
    /// Creates a plan entry that presents `texture`.
    pub fn new(texture: TextureView) -> Self {
        Self { texture }
    }
}

/// An ordered list of frames to present to a window, ending with the real rendered
/// frame, displayed at an even cadence by driver present metering.
pub struct PacedPresentPlan {
    /// Frames in presentation order.
    pub frames: Vec<PlannedFrame>,
    /// Chained onto the first present of the batch. Without it the batch is unmetered.
    present_chain_link: Option<PresentChainLink>,
}

impl PacedPresentPlan {
    /// Creates a plan that presents `frames` with no present chain, so unmetered.
    pub fn new(frames: Vec<PlannedFrame>) -> Self {
        Self {
            frames,
            present_chain_link: None,
        }
    }

    /// Installs the chain link that the engine chains onto the batch's first present.
    ///
    /// The engine keeps the link's payload alive until the plan's presents are issued,
    /// and drops it with the plan after the per-window present loop. The caller must not
    /// keep pointers into the payload after installation, because the driver reads and
    /// writes the chain during presentation. On non-Vulkan backends the link is ignored
    /// and the batch presents unmetered.
    ///
    /// # Safety
    /// Every extension in the link's chain must have been enabled at device creation, for
    /// example through a raw Vulkan device creation callback.
    pub unsafe fn set_present_chain_link(&mut self, link: PresentChainLink) {
        self.present_chain_link = Some(link);
    }
}

/// Presentation plans for the current render frame, consumed during presentation.
#[derive(Resource, Default)]
pub struct PacedPresentPlans {
    plans: EntityHashMap<PacedPresentPlan>,
}

impl PacedPresentPlans {
    /// Queues `plan` for the main world `window` entity, replacing any plan already queued.
    ///
    /// The plan is presented only when the window is also claimed in [`PacedWindows`]
    /// this frame. A plan for an unclaimed window is dropped with a warning.
    pub fn insert(&mut self, window: Entity, plan: PacedPresentPlan) {
        self.plans.insert(window, plan);
    }
}

/// How many frames paced presentation actually presented for each claimed window during
/// the last present step, keyed by main world window entity.
///
/// Every claimed window that was extracted this frame gets an entry. The count can be
/// lower than the plan length, because the present loop stops when a swapchain acquire
/// fails. The count is also one when the blit pipeline was not ready, because only the
/// real frame is presented then. A claimed window with no plan gets an entry of zero.
///
/// This resource does not exist before the first frame in which paced presentation
/// handles a window, so read it with `Option<Res<PacedPresentedFrames>>`. Producers read
/// it during extract of the next frame. This is safe under pipelined rendering because
/// extract of the next frame runs after the previous render frame has finished.
///
/// An entry is also the texture release signal under the [`PlannedFrame`] ownership
/// contract. When the entry for a frame's plan appears, the pacer has released every
/// texture in that plan.
///
/// The Streamline analogue is `numFramesActuallyPresented` from `slDLSSGGetState`.
#[derive(Resource, Default, Deref)]
pub struct PacedPresentedFrames(pub(crate) EntityHashMap<u32>);

/// Specializes the surface blit pipeline for every window with a declared swapchain
/// depth, so [`present_paced_plans`] finds it compiled.
///
/// Producers declare a depth from their first active frame, at least one frame before
/// their first plan, so asynchronous compilation normally finishes in time. The present
/// path falls back to presenting only the real frame when it has not.
pub(crate) fn prewarm_paced_blit_pipelines(
    paced_depths: Res<PacedSwapchainDepths>,
    windows: Query<(MainEntity, &SurfaceData)>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ScreenshotToScreenPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    blit_pipeline: Res<ScreenshotToScreenPipeline>,
) {
    for (main_entity, surface_data) in &windows {
        if paced_depths.get(main_entity) == 0 {
            continue;
        }
        let view_format = surface_data.configuration.format.add_srgb_suffix();
        pipelines.specialize(&pipeline_cache, &blit_pipeline, view_format);
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

/// Presents the queued [`PacedPresentPlan`] for every window claimed in
/// [`PacedWindows`], writes [`PacedPresentedFrames`], and returns the claimed windows.
/// Normal presentation must skip every returned window, even when the pacer presented
/// nothing for it. Each plan's frames are presented back to back, metered by the driver
/// when the plan carries a [`PresentChainLink`].
pub(crate) fn present_paced_plans(
    world: &mut World,
    state: &mut SystemState<PacedPresentState>,
) -> EntityHashSet {
    let claimed = world
        .get_resource::<PacedWindows>()
        .map(|paced| paced.0.clone())
        .unwrap_or_default();
    let mut plans = world
        .get_resource_mut::<PacedPresentPlans>()
        .map(|mut plans| mem::take(&mut plans.plans))
        .unwrap_or_default();
    let mut presented_counts: EntityHashMap<u32> = EntityHashMap::default();

    if !claimed.is_empty()
        && let Ok((
            mut pipelines,
            mut pipeline_cache,
            blit_pipeline,
            render_device,
            render_queue,
            mut windows,
        )) = state.get_mut(world)
    {
        for (window_entity, mut window, surface_data) in &mut windows {
            if !claimed.contains(&window_entity) {
                continue;
            }

            // Release any swapchain texture still held from a frame before the window
            // became paced. Extraction keeps unpresented textures alive across frames,
            // and the pacer owns acquisition now, so a held image would shrink the pool.
            drop(window.swap_chain_texture.take());
            window.swap_chain_texture_view = None;

            let plan = plans
                .remove(&window_entity)
                .filter(|plan| !plan.frames.is_empty());
            let Some(plan) = plan else {
                // The claim stands without a plan. The window stays owned by the pacer
                // and nothing is presented for it this frame.
                warn_once!("Paced window {window_entity} was claimed but has no frames to present");
                // needs_initial_present stays set. A Wayland window is invisible until
                // its first present, which now must come from a later frame.
                presented_counts.insert(window_entity, 0);
                continue;
            };

            let (Some(surface_data), Some(view_format)) =
                (surface_data, window.swap_chain_texture_view_format)
            else {
                warn!("No surface for paced window {window_entity}");
                presented_counts.insert(window_entity, 0);
                continue;
            };

            let pipeline_id = pipelines.specialize(&pipeline_cache, &blit_pipeline, view_format);
            // Normally compiled ahead of time by prewarm_paced_blit_pipelines. When the
            // pipeline is not ready, the wait must not stall a metered batch, so after the
            // wait only the real frame is presented and the generated frames are dropped.
            let pipeline_ready = pipeline_cache.get_render_pipeline(pipeline_id).is_some();
            if !pipeline_ready {
                warn_once!("Paced present blit pipeline not ready, presenting only the real frame");
                pipeline_cache.block_on_render_pipeline(pipeline_id);
            }
            let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id).cloned() else {
                warn!("Failed to compile paced present blit pipeline");
                presented_counts.insert(window_entity, 0);
                continue;
            };
            let frames = if pipeline_ready {
                plan.frames.as_slice()
            } else {
                // The real frame is the last plan entry per the plan contract
                &plan.frames[plan.frames.len() - 1..]
            };

            let layout = pipeline_cache.get_bind_group_layout(&blit_pipeline.bind_group_layout);
            let mut frames_presented: u32 = 0;
            for (frame_index, frame) in frames.iter().enumerate() {
                let Some(surface_texture) = acquire(surface_data, &render_device) else {
                    break;
                };
                // The driver meters the whole batch starting at the next present. The link is
                // armed only after a successful acquire, when a present is guaranteed to
                // follow in this scope, so the stashed pointer is consumed before the plan
                // and its payload drop at the end of this window's iteration.
                // The pipeline fallback never arms the link, because the payload encodes
                // the full batch size and the fallback knowingly presents fewer frames.
                // An acquire failure later in the batch can still present fewer frames
                // after the link is armed. The driver tolerates that as a short batch.
                if pipeline_ready
                    && frame_index == 0
                    && let Some(link) = &plan.present_chain_link
                    && let Some(hal_surface) =
                        // SAFETY: The hal surface is only used to stash the chain pointer,
                        // and it is not kept beyond this statement.
                        unsafe { surface_data.surface.as_hal::<wgpu::hal::api::Vulkan>() }
                {
                    // SAFETY: The link installer promised the chain's extensions were enabled
                    // at device creation, and the chain stays valid until the present below
                    // consumes it.
                    unsafe {
                        hal_surface.set_next_present_chain(link.head());
                    }
                }
                let bind_group = render_device.create_bind_group(
                    "paced_present_bind_group",
                    &layout,
                    &BindGroupEntries::single(&frame.texture),
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
                frames_presented += 1;
            }
            if frames_presented > 0 {
                // At least one present reached the surface, so the Wayland first-present
                // requirement is satisfied
                window.needs_initial_present = false;
            }
            presented_counts.insert(window_entity, frames_presented);
        }
    }

    // A plan for an unclaimed window cannot present. The claim is what suppressed the
    // normal swapchain acquisition, so presenting without it would fight normal
    // presentation for the same surface.
    for window_entity in plans.keys() {
        if !claimed.contains(window_entity) {
            warn_once!(
                "PacedPresentPlan for window {window_entity} was dropped because the window was not claimed in PacedWindows"
            );
        }
    }

    // Rewritten even when empty once it exists, so stale counts never survive a frame
    // where a window stopped being paced
    if !presented_counts.is_empty() || world.contains_resource::<PacedPresentedFrames>() {
        world.insert_resource(PacedPresentedFrames(presented_counts));
    }

    claimed
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
