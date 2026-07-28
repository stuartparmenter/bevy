use super::{display_target::ManualDisplayTargets, ExtractedWindows};
use crate::{
    gpu_readback,
    render_asset::RenderAssets,
    render_resource::{
        BindGroup, BindGroupEntries, Buffer, BufferUsages, PipelineCache,
        SpecializedRenderPipeline, SpecializedRenderPipelines, Texture, TextureUsages, TextureView,
    },
    renderer::RenderDevice,
    texture::{GpuImage, ManualTextureViews, OutputColorAttachment},
    view::{prepare_view_attachments, prepare_view_targets, ViewTargetAttachments, WindowSurfaces},
    working_color_space::{DISPLAYP3_TO_REC709, REC2020_TO_REC709},
    ExtractSchedule, GpuResourceAppExt, MainWorld, Render, RenderApp, RenderStartup, RenderSystems,
};
use alloc::{borrow::Cow, sync::Arc};
use bevy_app::{First, Plugin, Update};
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer, Handle, RenderAssetUsages};
use bevy_camera::{ManualTextureViewHandle, NormalizedRenderTarget, RenderTarget};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    entity::EntityHashMap, message::message_update_system, prelude::*, system::SystemState,
};
use bevy_image::{Image, TextureFormatPixelInfo, ToExtents};
use bevy_log::{error, info, warn};
use bevy_material::{
    bind_group_layout_entries::{binding_types::texture_2d, BindGroupLayoutEntries},
    descriptor::{
        BindGroupLayoutDescriptor, CachedRenderPipelineId, FragmentState, RenderPipelineDescriptor,
        VertexState,
    },
};
use bevy_math::Vec3;
use bevy_platform::collections::HashSet;
use bevy_reflect::Reflect;
use bevy_shader::Shader;
use bevy_tasks::AsyncComputeTaskPool;
use bevy_utils::default;
use bevy_window::{DisplayGamut, DisplayTransfer, PrimaryWindow, WindowRef};
use core::ops::Deref;
use std::{
    path::Path,
    sync::{
        mpsc::{Receiver, Sender},
        Mutex,
    },
};
use wgpu::{CommandEncoder, Extent3d, TextureFormat};

#[derive(EntityEvent, Reflect, Deref, DerefMut, Debug)]
#[reflect(Debug, Event)]
pub struct ScreenshotCaptured {
    pub entity: Entity,
    #[deref]
    pub image: Image,
}

/// A component that signals to the renderer to capture a screenshot this frame.
///
/// This component should be spawned on a new entity with an observer that will trigger
/// with [`ScreenshotCaptured`] when the screenshot is ready.
///
/// Screenshots are captured asynchronously and may not be available immediately after the frame
/// that the component is spawned on. The observer should be used to handle the screenshot when it
/// is ready.
///
/// Note that the screenshot entity will be despawned after the screenshot is captured and the
/// observer is triggered.
///
/// # Usage
///
/// ```
/// # use bevy_ecs::prelude::*;
/// # use bevy_render::view::screenshot::{save_to_disk, Screenshot};
///
/// fn take_screenshot(mut commands: Commands) {
///    commands.spawn(Screenshot::primary_window())
///       .observe(save_to_disk("screenshot.png"));
/// }
/// ```
#[derive(Component, Deref, DerefMut, Reflect, Debug)]
#[reflect(Component, Debug)]
pub struct Screenshot(pub RenderTarget);

/// A marker component that indicates that a screenshot is currently being captured.
#[derive(Component, Default)]
pub struct Capturing;

/// A marker component that indicates that a screenshot has been captured, the image is ready, and
/// the screenshot entity can be despawned.
#[derive(Component, Default)]
pub struct Captured;

impl Screenshot {
    /// Capture a screenshot of the provided window entity.
    pub fn window(window: Entity) -> Self {
        Self(RenderTarget::Window(WindowRef::Entity(window)))
    }

    /// Capture a screenshot of the primary window, if one exists.
    pub fn primary_window() -> Self {
        Self(RenderTarget::Window(WindowRef::Primary))
    }

    /// Capture a screenshot of the provided render target image.
    pub fn image(image: Handle<Image>) -> Self {
        Self(RenderTarget::Image(image.into()))
    }

    /// Capture a screenshot of the provided manual texture view.
    pub fn texture_view(texture_view: ManualTextureViewHandle) -> Self {
        Self(RenderTarget::TextureView(texture_view))
    }
}

struct ScreenshotPreparedState {
    pub texture: Texture,
    pub buffer: Buffer,
    pub bind_group: BindGroup,
    pub pipeline_id: CachedRenderPipelineId,
    pub size: Extent3d,
    /// How the captured texture's signal is decoded back to display-linear
    /// values (at the scRGB scale, `1.0` = 80 nits) before the image is handed
    /// to the main world. See [`ScreenshotDecode`].
    pub decode: ScreenshotDecode,
}

/// How a captured HDR screenshot's signal is decoded back to display-linear
/// values at the scRGB scale (`1.0` = 80 nits), matching the transfer the
/// display-encoding pass applied to the surface it was captured from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScreenshotDecode {
    /// Nothing is decoded. An SDR capture keeps its `*UnormSrgb` format label,
    /// which declares the sRGB encoding its bytes carry, and a
    /// [`DisplayTransfer::ScRgbLinear`] swapchain is display-linear by
    /// construction.
    Linear,
    /// HDR10 ([`DisplayTransfer::Pq`]): decode through the PQ EOTF (see
    /// [`decode_pq_screenshot`]).
    Pq,
    /// Encoded extended-range sRGB ([`DisplayTransfer::ExtendedSrgb`]): decode
    /// through the extended sRGB EOTF (see [`decode_extended_srgb_screenshot`]).
    /// `display_p3` is true for the `ExtendedDisplayP3` surface, whose decoded
    /// linear values are converted back to Rec.709 so every HDR screenshot
    /// shares one Rec.709, scRGB-scale convention.
    ExtendedSrgb { display_p3: bool },
}

impl ScreenshotDecode {
    /// The decode that reverses what the display encoder wrote for a target
    /// resolved to `(transfer, gamut)`.
    ///
    /// Keyed on the encoder's own two inputs (`coerce_display_encode` in
    /// `bevy_core_pipeline::camera_stack`), so decoder and encoder cannot drift:
    /// `ExtendedSrgb` is the one transfer that keeps a Display-P3 gamut (wgpu's
    /// `ExtendedDisplayP3` surface), `Pq` is always written in Rec.2020, and the
    /// rest reach the readback already display-linear.
    fn for_target(transfer: DisplayTransfer, gamut: DisplayGamut) -> Self {
        match transfer {
            DisplayTransfer::Pq => Self::Pq,
            DisplayTransfer::ExtendedSrgb => Self::ExtendedSrgb {
                display_p3: gamut == DisplayGamut::DisplayP3,
            },
            DisplayTransfer::Srgb | DisplayTransfer::ScRgbLinear => Self::Linear,
        }
    }
}

#[derive(Resource, Deref, DerefMut)]
pub struct CapturedScreenshots(pub Arc<Mutex<Receiver<(Entity, Image)>>>);

#[derive(Resource, Deref, DerefMut, Default)]
struct RenderScreenshotTargets(EntityHashMap<NormalizedRenderTarget>);

#[derive(Resource, Deref, DerefMut, Default)]
struct RenderScreenshotsPrepared(EntityHashMap<ScreenshotPreparedState>);

#[derive(Resource, Deref, DerefMut)]
struct RenderScreenshotsSender(Sender<(Entity, Image)>);

/// Saves the captured screenshot to disk at the provided path.
///
/// Screenshots of HDR surfaces decode to floating-point images holding
/// display-linear Rec.709 values at the scRGB scale (`1.0` = 80 nits):
/// `Rgba16Float` scRGB-linear swapchains (`DisplayTransfer::ScRgbLinear`) are
/// read back as-is (their signal is already linear at that scale), HDR10
/// swapchains (`DisplayTransfer::Pq`) are decoded from the PQ signal through the
/// PQ EOTF and converted from the encoder's Rec.2020 primaries to Rec.709 (see
/// `decode_pq_screenshot`), and the encoded extended-range sRGB swapchains
/// (`DisplayTransfer::ExtendedSrgb`, i.e. wgpu `ExtendedSrgb` /
/// `ExtendedDisplayP3`) are decoded through the extended sRGB EOTF — converting
/// Display-P3 back to Rec.709 — to the same scale (see
/// `decode_extended_srgb_screenshot`). These values are display-referred and
/// captured after the encoder's gamut stage, so any out-of-gamut compression
/// it applied is baked in: the decode reverses the primaries, not that
/// compression. Colors outside Rec.709 come back with negative components.
///
/// `.exr` keeps a float capture intact — `f32` channels, alpha included.
/// Radiance `.hdr` stores shared-exponent RGBE: an 8-bit mantissa per channel,
/// no alpha, and negatives clipped, so prefer `.exr` for extended-range
/// captures. Each container needs the matching `image` crate codec: Bevy's
/// `hdr` feature is on by default through `common_api`, `exr` is opt-in. For
/// 8-bit containers such as PNG the display-linear signal is clamped to
/// `[0, 1]` (clipping everything above the 80-nit scRGB reference white),
/// sRGB-encoded, and quantized, with a warning.
pub fn save_to_disk(path: impl AsRef<Path>) -> impl FnMut(On<ScreenshotCaptured>) {
    let path = path.as_ref().to_owned();
    move |screenshot_captured| {
        use image::{DynamicImage, ImageFormat};

        let img = screenshot_captured.image.clone();
        match img.try_into_dynamic() {
            Ok(dyn_img) => match ImageFormat::from_path(&path) {
                Ok(format) => {
                    let img = match (&dyn_img, format) {
                        // Float sources keep their full range in
                        // float-capable containers. The signal is written
                        // as-is, without HDR mastering metadata, PQ
                        // containers, or a paper-white-aware SDR preview.
                        (
                            DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_),
                            ImageFormat::OpenExr,
                        ) => DynamicImage::ImageRgba32F(dyn_img.into_rgba32f()),
                        (
                            DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_),
                            ImageFormat::Hdr,
                        ) => DynamicImage::ImageRgb32F(dyn_img.into_rgb32f()),
                        // Float source into an 8-bit container: the buffer
                        // holds display-linear signal (for scRGB surfaces,
                        // 1.0 = the 80-nit scRGB reference white), so clamp,
                        // apply the sRGB OETF, and quantize.
                        (DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_), _) => {
                            warn!(
                                "Saving an HDR (floating point) screenshot to an 8-bit image \
                                format: values above 1.0 are clipped and the signal is \
                                sRGB-encoded. Use an .exr path to keep the full range."
                            );
                            let mut rgb = dyn_img.into_rgb32f();
                            for value in rgb.iter_mut() {
                                *value =
                                    crate::transfer_functions::srgb_oetf(value.clamp(0.0, 1.0));
                            }
                            DynamicImage::ImageRgb8(DynamicImage::ImageRgb32F(rgb).to_rgb8())
                        }
                        // discard the alpha channel which stores brightness values when HDR is enabled to make sure
                        // the screenshot looks right
                        _ => DynamicImage::ImageRgb8(dyn_img.to_rgb8()),
                    };
                    #[cfg(not(target_arch = "wasm32"))]
                    match img.save_with_format(&path, format) {
                        Ok(_) => info!("Screenshot saved to {}", path.display()),
                        Err(e) => error!("Cannot save screenshot, IO error: {e}"),
                    }

                    #[cfg(target_arch = "wasm32")]
                    {
                        let save_screenshot = || {
                            use image::EncodableLayout;
                            use wasm_bindgen::{JsCast, JsValue};

                            let mut image_buffer = std::io::Cursor::new(Vec::new());
                            img.write_to(&mut image_buffer, format)
                                .map_err(|e| JsValue::from_str(&format!("{e}")))?;

                            let parts = js_sys::Array::of1(
                                &js_sys::Uint8Array::new_from_slice(
                                    image_buffer.into_inner().as_bytes(),
                                )
                                .into(),
                            );
                            let blob = web_sys::Blob::new_with_u8_array_sequence(&parts)?;
                            let url = web_sys::Url::create_object_url_with_blob(&blob)?;
                            let window = web_sys::window().unwrap();
                            let document = window.document().unwrap();
                            let link = document.create_element("a")?;
                            link.set_attribute("href", &url)?;
                            link.set_attribute(
                                "download",
                                path.file_name()
                                    .and_then(|filename| filename.to_str())
                                    .ok_or_else(|| JsValue::from_str("Invalid filename"))?,
                            )?;
                            let html_element = link.dyn_into::<web_sys::HtmlElement>()?;
                            html_element.click();
                            web_sys::Url::revoke_object_url(&url)?;
                            Ok::<(), JsValue>(())
                        };

                        match (save_screenshot)() {
                            Ok(_) => info!("Screenshot saved to {}", path.display()),
                            Err(e) => error!("Cannot save screenshot, error: {e:?}"),
                        };
                    }
                }
                Err(e) => error!("Cannot save screenshot, requested format not recognized: {e}"),
            },
            Err(e) => error!("Cannot save screenshot, screen format cannot be understood: {e}"),
        }
    }
}

fn clear_screenshots(mut commands: Commands, screenshots: Query<Entity, With<Captured>>) {
    for entity in screenshots.iter() {
        commands.entity(entity).despawn();
    }
}

pub fn trigger_screenshots(
    mut commands: Commands,
    captured_screenshots: ResMut<CapturedScreenshots>,
) {
    let captured_screenshots = captured_screenshots.lock().unwrap();
    while let Ok((entity, image)) = captured_screenshots.try_recv() {
        commands.entity(entity).insert(Captured);
        commands.trigger(ScreenshotCaptured { image, entity });
    }
}

fn extract_screenshots(
    mut targets: ResMut<RenderScreenshotTargets>,
    mut main_world: ResMut<MainWorld>,
    mut system_state: Local<
        Option<
            SystemState<(
                Commands,
                Query<Entity, With<PrimaryWindow>>,
                Query<(Entity, &Screenshot), Without<Capturing>>,
            )>,
        >,
    >,
    mut seen_targets: Local<HashSet<NormalizedRenderTarget>>,
) {
    if system_state.is_none() {
        *system_state = Some(SystemState::new(&mut main_world));
    }
    let system_state = system_state.as_mut().unwrap();
    let (mut commands, primary_window, screenshots) =
        system_state.get_mut(&mut main_world).unwrap();

    targets.clear();
    seen_targets.clear();

    let primary_window = primary_window.iter().next();

    for (entity, screenshot) in screenshots.iter() {
        let render_target = screenshot.0.clone();
        let Some(render_target) = render_target.normalize(primary_window) else {
            warn!(
                "Unknown render target for screenshot, skipping: {:?}",
                render_target
            );
            continue;
        };
        if seen_targets.contains(&render_target) {
            warn!(
                "Duplicate render target for screenshot, skipping entity {}: {:?}",
                entity, render_target
            );
            // If we don't despawn the entity here, it will be captured again in the next frame
            commands.entity(entity).despawn();
            continue;
        }
        seen_targets.insert(render_target.clone());
        targets.insert(entity, render_target);
        commands.entity(entity).insert(Capturing);
    }

    system_state.apply(&mut main_world);
}

/// Looks up how a manual (`Image` / `TextureView`) render target's signal must
/// be decoded, from its [`ManualDisplayTargets`] entry. Manual targets resolve
/// to the requested transfer verbatim — there is no surface negotiation to
/// downgrade them — so the registered target is the encoder's input directly.
/// An unregistered target falls back to `DisplayTarget::SDR_SRGB`, exactly as
/// `resolve_display_target` does for the encoder.
fn manual_target_decode(
    manual_display_targets: &ManualDisplayTargets,
    target: &NormalizedRenderTarget,
) -> ScreenshotDecode {
    let display_target = manual_display_targets
        .get(target)
        .copied()
        .unwrap_or_default();
    ScreenshotDecode::for_target(display_target.transfer, display_target.gamut)
}

fn prepare_screenshots(
    targets: Res<RenderScreenshotTargets>,
    mut prepared: ResMut<RenderScreenshotsPrepared>,
    window_surfaces: Res<WindowSurfaces>,
    extracted_windows: Res<ExtractedWindows>,
    render_device: Res<RenderDevice>,
    screenshot_pipeline: Res<ScreenshotToScreenPipeline>,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ScreenshotToScreenPipeline>>,
    images: Res<RenderAssets<GpuImage>>,
    manual_texture_views: Res<ManualTextureViews>,
    manual_display_targets: Res<ManualDisplayTargets>,
    mut view_target_attachments: ResMut<ViewTargetAttachments>,
) {
    prepared.clear();
    for (entity, target) in targets.iter() {
        match target {
            NormalizedRenderTarget::Window(window) => {
                let window = window.entity();
                let Some(surface_data) = window_surfaces.surfaces.get(&window) else {
                    warn!("Unknown window for screenshot, skipping: {}", window);
                    continue;
                };
                // The requested gamut, the encoder's other input. A window with
                // a surface is always extracted, so this never skips in
                // practice.
                let Some(extracted_window) = extracted_windows.get(&window) else {
                    warn!("Unextracted window for screenshot, skipping: {}", window);
                    continue;
                };
                // Single-sourced with `SurfaceData`: the sRGB view of the
                // surface format when one exists, the raw (e.g. fp16 scRGB)
                // surface format otherwise. Must match what
                // `set_swapchain_texture` produces, since
                // `submit_screenshot_commands` blits this texture back to
                // `swap_chain_texture_view`.
                let view_format = surface_data.view_format();
                let size = Extent3d {
                    width: surface_data.configuration.width,
                    height: surface_data.configuration.height,
                    ..default()
                };
                let (texture_view, mut state) = prepare_screenshot_state(
                    size,
                    view_format,
                    &render_device,
                    &screenshot_pipeline,
                    &pipeline_cache,
                    &mut pipelines,
                );
                // HDR surfaces hold an encoded signal (the display encoder's
                // OETF output); record how to decode the readback back to
                // display-linear values.
                //
                // The transfer is the one this frame's pixels were *encoded*
                // with, not the live configuration: `prepare_windows` can
                // renegotiate a surface mid-frame, after the encoder's
                // `ViewDisplayTarget`s were built, and decoding a PQ readback as
                // if it were already display-linear would silently corrupt the
                // saved image. The gamut needs no such care — renegotiation
                // never changes it.
                state.decode = ScreenshotDecode::for_target(
                    surface_data.encoded_transfer(),
                    extracted_window.display_target.gamut,
                );
                prepared.insert(*entity, state);
                view_target_attachments.insert(
                    target.clone(),
                    OutputColorAttachment::new(texture_view.clone(), view_format),
                );
            }
            NormalizedRenderTarget::Image(image) => {
                let Some(gpu_image) = images.get(&image.handle) else {
                    warn!("Unknown image for screenshot, skipping: {:?}", image);
                    continue;
                };
                let view_format = gpu_image.view_format();
                let (texture_view, mut state) = prepare_screenshot_state(
                    gpu_image.texture_descriptor.size,
                    view_format,
                    &render_device,
                    &screenshot_pipeline,
                    &pipeline_cache,
                    &mut pipelines,
                );
                state.decode = manual_target_decode(&manual_display_targets, target);
                prepared.insert(*entity, state);
                view_target_attachments.insert(
                    target.clone(),
                    OutputColorAttachment::new(texture_view.clone(), view_format),
                );
            }
            NormalizedRenderTarget::TextureView(texture_view) => {
                let Some(manual_texture_view) = manual_texture_views.get(texture_view) else {
                    warn!(
                        "Unknown manual texture view for screenshot, skipping: {:?}",
                        texture_view
                    );
                    continue;
                };
                let view_format = manual_texture_view.view_format;
                let size = manual_texture_view.size.to_extents();
                let (texture_view, mut state) = prepare_screenshot_state(
                    size,
                    view_format,
                    &render_device,
                    &screenshot_pipeline,
                    &pipeline_cache,
                    &mut pipelines,
                );
                state.decode = manual_target_decode(&manual_display_targets, target);
                prepared.insert(*entity, state);
                view_target_attachments.insert(
                    target.clone(),
                    OutputColorAttachment::new(texture_view.clone(), view_format),
                );
            }
            NormalizedRenderTarget::None { .. } => {
                // Nothing to screenshot!
            }
        }
    }
}

fn prepare_screenshot_state(
    size: Extent3d,
    format: TextureFormat,
    render_device: &RenderDevice,
    pipeline: &ScreenshotToScreenPipeline,
    pipeline_cache: &PipelineCache,
    pipelines: &mut SpecializedRenderPipelines<ScreenshotToScreenPipeline>,
) -> (TextureView, ScreenshotPreparedState) {
    let texture = render_device.create_texture(&wgpu::TextureDescriptor {
        label: Some("screenshot-capture-rendertarget"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::COPY_SRC
            | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&Default::default());
    let buffer = render_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot-transfer-buffer"),
        size: gpu_readback::get_aligned_size(size, format.pixel_size().unwrap_or(0) as u32) as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = render_device.create_bind_group(
        "screenshot-to-screen-bind-group",
        &pipeline_cache.get_bind_group_layout(&pipeline.bind_group_layout),
        &BindGroupEntries::single(&texture_view),
    );
    let pipeline_id = pipelines.specialize(pipeline_cache, pipeline, format);

    (
        texture_view,
        ScreenshotPreparedState {
            texture,
            buffer,
            bind_group,
            pipeline_id,
            size,
            decode: ScreenshotDecode::Linear,
        },
    )
}

pub struct ScreenshotPlugin;

impl Plugin for ScreenshotPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        embedded_asset!(app, "screenshot.wgsl");

        let (tx, rx) = std::sync::mpsc::channel();
        app.register_type::<Screenshot>()
            .register_type::<ScreenshotCaptured>()
            .insert_resource(CapturedScreenshots(Arc::new(Mutex::new(rx))))
            .add_systems(
                First,
                clear_screenshots
                    .after(message_update_system)
                    .before(ApplyDeferred),
            )
            .add_systems(Update, trigger_screenshots);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .insert_resource(RenderScreenshotsSender(tx))
            .init_resource::<RenderScreenshotTargets>()
            .init_resource::<RenderScreenshotsPrepared>()
            .init_gpu_resource::<SpecializedRenderPipelines<ScreenshotToScreenPipeline>>()
            .add_systems(RenderStartup, init_screenshot_to_screen_pipeline)
            .add_systems(ExtractSchedule, extract_screenshots.ambiguous_with_all())
            .add_systems(
                Render,
                prepare_screenshots
                    .after(prepare_view_attachments)
                    .before(prepare_view_targets)
                    .in_set(RenderSystems::PrepareViews),
            );
    }
}

#[derive(Resource)]
pub struct ScreenshotToScreenPipeline {
    pub bind_group_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
}

pub fn init_screenshot_to_screen_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "screenshot-to-screen-bgl",
        &BindGroupLayoutEntries::single(
            wgpu::ShaderStages::FRAGMENT,
            texture_2d(wgpu::TextureSampleType::Float { filterable: false }),
        ),
    );

    let shader = load_embedded_asset!(asset_server.as_ref(), "screenshot.wgsl");

    commands.insert_resource(ScreenshotToScreenPipeline {
        bind_group_layout,
        shader,
    });
}

impl SpecializedRenderPipeline for ScreenshotToScreenPipeline {
    type Key = TextureFormat;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some(Cow::Borrowed("screenshot-to-screen")),
            layout: vec![self.bind_group_layout.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                ..default()
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            multisample: Default::default(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                targets: vec![Some(wgpu::ColorTargetState {
                    format: key,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

pub(crate) fn submit_screenshot_commands(world: &World, encoder: &mut CommandEncoder) {
    let targets = world.resource::<RenderScreenshotTargets>();
    let prepared = world.resource::<RenderScreenshotsPrepared>();
    let pipelines = world.resource::<PipelineCache>();
    let gpu_images = world.resource::<RenderAssets<GpuImage>>();
    let windows = world.resource::<ExtractedWindows>();
    let manual_texture_views = world.resource::<ManualTextureViews>();

    for (entity, render_target) in targets.iter() {
        match render_target {
            NormalizedRenderTarget::Window(window) => {
                let window = window.entity();
                let Some(window) = windows.get(&window) else {
                    continue;
                };
                let width = window.physical_width;
                let height = window.physical_height;
                let Some(texture_format) = window.swap_chain_texture_view_format else {
                    continue;
                };
                let Some(swap_chain_texture_view) = window.swap_chain_texture_view.as_ref() else {
                    continue;
                };
                render_screenshot(
                    encoder,
                    prepared,
                    pipelines,
                    entity,
                    width,
                    height,
                    texture_format,
                    swap_chain_texture_view,
                );
            }
            NormalizedRenderTarget::Image(image) => {
                let Some(gpu_image) = gpu_images.get(&image.handle) else {
                    warn!("Unknown image for screenshot, skipping: {:?}", image);
                    continue;
                };
                let width = gpu_image.texture_descriptor.size.width;
                let height = gpu_image.texture_descriptor.size.height;
                let texture_format = gpu_image.texture_descriptor.format;
                let texture_view = gpu_image.texture_view.deref();
                render_screenshot(
                    encoder,
                    prepared,
                    pipelines,
                    entity,
                    width,
                    height,
                    texture_format,
                    texture_view,
                );
            }
            NormalizedRenderTarget::TextureView(texture_view) => {
                let Some(texture_view) = manual_texture_views.get(texture_view) else {
                    warn!(
                        "Unknown manual texture view for screenshot, skipping: {:?}",
                        texture_view
                    );
                    continue;
                };
                let width = texture_view.size.x;
                let height = texture_view.size.y;
                let texture_format = texture_view.view_format;
                let texture_view = texture_view.texture_view.deref();
                render_screenshot(
                    encoder,
                    prepared,
                    pipelines,
                    entity,
                    width,
                    height,
                    texture_format,
                    texture_view,
                );
            }
            NormalizedRenderTarget::None { .. } => {
                // Nothing to screenshot!
            }
        };
    }
}

fn render_screenshot(
    encoder: &mut CommandEncoder,
    prepared: &RenderScreenshotsPrepared,
    pipelines: &PipelineCache,
    entity: &Entity,
    width: u32,
    height: u32,
    texture_format: TextureFormat,
    texture_view: &wgpu::TextureView,
) {
    if let Some(prepared_state) = &prepared.get(entity) {
        let extent = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        encoder.copy_texture_to_buffer(
            prepared_state.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &prepared_state.buffer,
                layout: gpu_readback::layout_data(extent, texture_format),
            },
            extent,
        );

        if let Some(pipeline) = pipelines.get_render_pipeline(prepared_state.pipeline_id) {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("screenshot_to_screen_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &prepared_state.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

pub(crate) fn collect_screenshots(world: &mut World) {
    #[cfg(feature = "trace")]
    let _span = bevy_log::info_span!("collect_screenshots").entered();

    let sender = world.resource::<RenderScreenshotsSender>().deref().clone();
    let prepared = world.resource::<RenderScreenshotsPrepared>();

    for (entity, prepared) in prepared.iter() {
        let entity = *entity;
        let sender = sender.clone();
        let width = prepared.size.width;
        let height = prepared.size.height;
        let texture_format = prepared.texture.format();
        let Ok(pixel_size) = texture_format.pixel_size() else {
            continue;
        };
        let decode = prepared.decode;
        let buffer = prepared.buffer.clone();

        let finish = async move {
            let (tx, rx) = async_channel::bounded(1);
            let buffer_slice = buffer.slice(..);
            // The polling for this map call is done every frame when the command queue is submitted.
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                if let Err(err) = result {
                    panic!("{}", err.to_string());
                }
                tx.try_send(()).unwrap();
            });
            rx.recv().await.unwrap();
            let data = buffer_slice
                .get_mapped_range()
                .expect("screenshot buffer should be mapped");
            // we immediately move the data to CPU memory to avoid holding the mapped view for long
            let mut result = Vec::from(&*data);
            drop(data);

            if result.len() != ((width * height) as usize * pixel_size) {
                // Our buffer has been padded because we needed to align to a multiple of 256.
                // We remove this padding here
                let initial_row_bytes = width as usize * pixel_size;
                let buffered_row_bytes =
                    gpu_readback::align_byte_size(width * pixel_size as u32) as usize;

                let mut take_offset = buffered_row_bytes;
                let mut place_offset = initial_row_bytes;
                for _ in 1..height {
                    result.copy_within(take_offset..take_offset + buffered_row_bytes, place_offset);
                    take_offset += buffered_row_bytes;
                    place_offset += initial_row_bytes;
                }
                result.truncate(initial_row_bytes * height as usize);
            }

            let (result, texture_format) = match decode {
                ScreenshotDecode::Linear => (result, texture_format),
                ScreenshotDecode::Pq => decode_pq_screenshot(result, texture_format),
                ScreenshotDecode::ExtendedSrgb { display_p3 } => {
                    decode_extended_srgb_screenshot(result, texture_format, display_p3)
                }
            };

            if let Err(e) = sender.send((
                entity,
                Image::new(
                    Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    wgpu::TextureDimension::D2,
                    result,
                    texture_format,
                    RenderAssetUsages::MAIN_WORLD,
                ),
            )) {
                error!("Failed to send screenshot: {}", e);
            }
        };

        AsyncComputeTaskPool::get().spawn(finish).detach();
    }
}

/// Decodes a PQ-encoded (HDR10) screenshot readback into display-linear
/// `Rgba32Float` data.
///
/// HDR10 swapchains hold the PQ *signal* (the display encoder's SMPTE ST 2084
/// OETF output, `1.0` = 10000 nits); saved as-is, the values would be
/// mislabeled as linear. Each color channel is decoded through the PQ EOTF
/// ([`crate::transfer_functions::pq_eotf`]) to absolute luminance and stored
/// linearly at the **same scale scRGB screenshots use: `1.0` = 80 nits** (the
/// scRGB reference white), so `.exr` output keeps the full range with one
/// consistent scale across all HDR surface kinds, and [`save_to_disk`]'s
/// 8-bit fallback clips at the same reference white for all of them. The
/// encoder writes PQ in Rec.2020 primaries unconditionally, so the decoded
/// values are converted back through [`REC2020_TO_REC709`] to the one Rec.709
/// convention every HDR screenshot shares; colors outside Rec.709 decode to
/// negative components. Alpha is decoded to its plain normalized value.
///
/// `Rgb10a2Unorm` (10-bit PQ channels + 2-bit alpha) and `Rgba16Float` (PQ
/// signal in half floats) are what a window surface normally negotiates for
/// HDR10, but negotiation can fall back to any non-sRGB format the surface
/// advertises, and a manual target arrives in whatever format its user chose.
/// Anything else is passed through undecoded, with a warning.
fn decode_pq_screenshot(data: Vec<u8>, format: TextureFormat) -> (Vec<u8>, TextureFormat) {
    use crate::transfer_functions::{pq_eotf, PQ_MAX_LUMINANCE_NITS, SCRGB_REFERENCE_WHITE_NITS};
    /// What PQ signal 1.0 decodes to over what stored value 1.0 represents.
    const NITS_SCALE: f32 = PQ_MAX_LUMINANCE_NITS / SCRGB_REFERENCE_WHITE_NITS;

    // PQ signal triple to display-linear Rec.709 at the scRGB scale.
    let decode_rgb = |signal: Vec3| {
        REC2020_TO_REC709
            * (Vec3::new(pq_eotf(signal.x), pq_eotf(signal.y), pq_eotf(signal.z)) * NITS_SCALE)
    };

    let decoded: Vec<f32> = match format {
        TextureFormat::Rgb10a2Unorm => data
            .chunks_exact(4)
            .flat_map(|texel| {
                let packed = u32::from_le_bytes(texel.try_into().unwrap());
                // WebGPU packing: red in the least significant bits.
                let channel = |shift: u32| ((packed >> shift) & 0x3ff) as f32 / 1023.0;
                let rgb = decode_rgb(Vec3::new(channel(0), channel(10), channel(20)));
                [rgb.x, rgb.y, rgb.z, (packed >> 30) as f32 / 3.0]
            })
            .collect(),
        TextureFormat::Rgba16Float => data
            .chunks_exact(8)
            .flat_map(|texel| {
                let channel = |i: usize| {
                    f16_bits_to_f32(u16::from_le_bytes([texel[2 * i], texel[2 * i + 1]]))
                };
                let rgb = decode_rgb(Vec3::new(channel(0), channel(1), channel(2)));
                [rgb.x, rgb.y, rgb.z, channel(3)]
            })
            .collect(),
        other => {
            warn!(
                "PQ-encoded screenshot readback in unexpected format {other:?}; \
                saving the raw encoded signal values."
            );
            return (data, format);
        }
    };
    (
        decoded
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        TextureFormat::Rgba32Float,
    )
}

/// Decodes an encoded extended-range sRGB (`ExtendedSrgb` / `ExtendedDisplayP3`)
/// screenshot readback into display-linear `Rgba32Float` data.
///
/// These swapchains hold the gamma-encoded extended sRGB *signal* (the display
/// encoder's odd-symmetric sRGB OETF output); saved as-is, the values would be
/// mislabeled as linear. Each color channel is decoded through the extended
/// sRGB EOTF ([`crate::transfer_functions::srgb_eotf_extended`]), which yields
/// display-linear values at the **same scale scRGB screenshots use: `1.0` = 80
/// nits** (the OETF input is the scRGB-normalized `color × paper_white / 80`).
/// When `display_p3` (the `ExtendedDisplayP3` surface), the decoded P3-primary
/// values are converted back to Rec.709 via [`DISPLAYP3_TO_REC709`] so every
/// HDR screenshot shares one Rec.709 scale. Alpha passes through.
///
/// `Rgba16Float` is what a window surface normally negotiates for these color
/// spaces, but negotiation can fall back to any non-sRGB format the surface
/// advertises, and a manual target arrives in whatever format its user chose.
/// Anything else is passed through undecoded, with a warning.
fn decode_extended_srgb_screenshot(
    data: Vec<u8>,
    format: TextureFormat,
    display_p3: bool,
) -> (Vec<u8>, TextureFormat) {
    use crate::transfer_functions::srgb_eotf_extended;

    let TextureFormat::Rgba16Float = format else {
        warn!(
            "Extended-sRGB-encoded screenshot readback in unexpected format {format:?}; \
            saving the raw encoded signal values."
        );
        return (data, format);
    };

    let decoded: Vec<f32> = data
        .chunks_exact(8)
        .flat_map(|texel| {
            let channel =
                |i: usize| f16_bits_to_f32(u16::from_le_bytes([texel[2 * i], texel[2 * i + 1]]));
            let mut rgb = Vec3::new(
                srgb_eotf_extended(channel(0)),
                srgb_eotf_extended(channel(1)),
                srgb_eotf_extended(channel(2)),
            );
            if display_p3 {
                rgb = DISPLAYP3_TO_REC709 * rgb;
            }
            [rgb.x, rgb.y, rgb.z, channel(3)]
        })
        .collect();

    (
        decoded
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        TextureFormat::Rgba32Float,
    )
}

/// Converts IEEE 754 binary16 bits to an `f32`.
///
/// `bevy_render` has no `half` dependency, and the HDR screenshot decodes are
/// the only place it would be needed, so this is a minimal local conversion
/// (handles normals, subnormals, zeros, infinities, and NaN).
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from(bits >> 10) & 0x1f;
    let mantissa = u32::from(bits) & 0x3ff;
    let bits32 = match (exponent, mantissa) {
        // Signed zero.
        (0, 0) => sign,
        // Subnormal (value = mantissa × 2⁻²⁴): renormalize into the f32
        // exponent range around the mantissa's highest set bit `p`.
        (0, _) => {
            let p = 31 - mantissa.leading_zeros();
            let exponent = 103 + p; // 127 + p - 24
            let mantissa = (mantissa << (23 - p)) & 0x007f_ffff;
            sign | (exponent << 23) | mantissa
        }
        // Infinity / NaN.
        (0x1f, _) => sign | 0x7f80_0000 | (mantissa << 13),
        // Normal.
        _ => sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer_functions::{pq_eotf, pq_inverse_eotf_from_nits};
    use bevy_window::DisplayTarget;

    #[test]
    fn f16_conversion_matches_known_values() {
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        assert!(f16_bits_to_f32(0x8000) == 0.0 && f16_bits_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0xbc00), -1.0);
        assert_eq!(f16_bits_to_f32(0x3800), 0.5);
        assert_eq!(f16_bits_to_f32(0x3555), 0.333_251_95); // closest f16 to 1/3
        assert_eq!(f16_bits_to_f32(0x7bff), 65504.0); // f16::MAX
        assert_eq!(f16_bits_to_f32(0x0001), 5.960_464_5e-8); // smallest subnormal
        assert_eq!(f16_bits_to_f32(0x03ff), 6.097_555e-5); // largest subnormal
        assert_eq!(f16_bits_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_bits_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_bits_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn pq_decode_rgb10a2() {
        // A 100-nit gray texel: PQ signal for 100 nits, quantized to 10 bits,
        // with opaque alpha.
        let signal = pq_inverse_eotf_from_nits(100.0);
        let quantized = (signal * 1023.0).round() as u32;
        let packed: u32 = quantized | (quantized << 10) | (quantized << 20) | (3 << 30);
        let (decoded, format) =
            decode_pq_screenshot(packed.to_le_bytes().to_vec(), TextureFormat::Rgb10a2Unorm);
        assert_eq!(format, TextureFormat::Rgba32Float);
        let values: Vec<f32> = decoded
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // Stored scale: 1.0 = 80 nits, so 100 nits ≈ 1.25 (within 10-bit
        // quantization error).
        for channel in &values[0..3] {
            assert!(
                (channel - 1.25).abs() < 0.01,
                "expected ~1.25, got {channel}"
            );
        }
        assert_eq!(values[3], 1.0);
        // Beyond the EOTF and the rescale, the only processing is the
        // Rec.2020 -> Rec.709 conversion, whose rows sum to 1.0 only to within
        // rounding, so gray survives it to that precision.
        assert!(
            (values[0] - pq_eotf(quantized as f32 / 1023.0) * 125.0).abs() < 1e-5,
            "expected the plain EOTF value, got {}",
            values[0]
        );
    }

    #[test]
    fn pq_decode_rgba16float() {
        // PQ signal 0.5079… (f16 0x3810 ≈ 0.5078) decodes to ≈ 92 nits; use
        // exact f16 values to keep the assertion strict.
        let half_one = 0x3c00u16; // 1.0 → 10000 nits → 125.0 at 80-nit scale
        let half_zero = 0x0000u16;
        let texel: Vec<u8> = [half_one, half_zero, half_one, half_one]
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect();
        let (decoded, format) = decode_pq_screenshot(texel, TextureFormat::Rgba16Float);
        assert_eq!(format, TextureFormat::Rgba32Float);
        let values: Vec<f32> = decoded
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // The decoded magenta is (125, 0, 125) in the Rec.2020 primaries the
        // encoder writes; it leaves as Rec.709, which puts it out of gamut.
        let expected = REC2020_TO_REC709 * Vec3::new(125.0, 0.0, 125.0);
        assert_eq!(values, vec![expected.x, expected.y, expected.z, 1.0]);
        assert!(
            values[1] < 0.0,
            "green should decode negative: {}",
            values[1]
        );
    }

    /// The one keying both the window and manual paths go through. The gamut
    /// reaches only `ExtendedSrgb`, the sole transfer whose surface can carry
    /// Display-P3 primaries.
    #[test]
    fn decode_keys_on_transfer_and_gamut() {
        use ScreenshotDecode as Decode;

        for gamut in [
            DisplayGamut::Rec709,
            DisplayGamut::DisplayP3,
            DisplayGamut::Rec2020,
        ] {
            assert_eq!(
                Decode::for_target(DisplayTransfer::Srgb, gamut),
                Decode::Linear
            );
            assert_eq!(
                Decode::for_target(DisplayTransfer::ScRgbLinear, gamut),
                Decode::Linear
            );
            assert_eq!(Decode::for_target(DisplayTransfer::Pq, gamut), Decode::Pq);
        }

        assert_eq!(
            Decode::for_target(DisplayTransfer::ExtendedSrgb, DisplayGamut::DisplayP3),
            Decode::ExtendedSrgb { display_p3: true }
        );
        // Rec.2020 has no encoded-extended surface and is coerced to Rec.709,
        // so it decodes as the Rec.709 `ExtendedSrgb` surface does.
        for gamut in [DisplayGamut::Rec709, DisplayGamut::Rec2020] {
            assert_eq!(
                Decode::for_target(DisplayTransfer::ExtendedSrgb, gamut),
                Decode::ExtendedSrgb { display_p3: false }
            );
        }
    }

    /// The manual-target decode reads the authored `ManualDisplayTargets`
    /// entry, and an unregistered target decodes as linear.
    #[test]
    fn manual_target_decode_follows_the_registered_transfer() {
        let target = NormalizedRenderTarget::TextureView(ManualTextureViewHandle(3));
        let mut manual = ManualDisplayTargets::default();
        assert!(matches!(
            manual_target_decode(&manual, &target),
            ScreenshotDecode::Linear
        ));

        manual.insert(
            target.clone(),
            DisplayTarget::SDR_SRGB.with_transfer(DisplayTransfer::Pq),
        );
        assert!(matches!(
            manual_target_decode(&manual, &target),
            ScreenshotDecode::Pq
        ));

        manual.insert(
            target.clone(),
            DisplayTarget::SDR_SRGB
                .with_transfer(DisplayTransfer::ExtendedSrgb)
                .with_gamut(DisplayGamut::DisplayP3),
        );
        assert!(matches!(
            manual_target_decode(&manual, &target),
            ScreenshotDecode::ExtendedSrgb { display_p3: true }
        ));
    }

    #[test]
    fn pq_decode_passes_unexpected_formats_through() {
        let data = vec![0u8, 1, 2, 3];
        let (out, format) = decode_pq_screenshot(data.clone(), TextureFormat::Rgba8Unorm);
        assert_eq!(out, data);
        assert_eq!(format, TextureFormat::Rgba8Unorm);
    }

    /// Builds an `Rgba16Float` texel from four f16-as-f32 channel values.
    fn rgba16f_texel(rgba: [u16; 4]) -> Vec<u8> {
        rgba.iter().flat_map(|bits| bits.to_le_bytes()).collect()
    }

    #[test]
    fn extended_srgb_decode_rec709() {
        use crate::transfer_functions::srgb_eotf_extended;
        // Each RGB channel at signal 1.0 (f16 0x3c00), opaque alpha.
        let texel = rgba16f_texel([0x3c00, 0x3c00, 0x3c00, 0x3c00]);
        let (decoded, format) =
            decode_extended_srgb_screenshot(texel, TextureFormat::Rgba16Float, false);
        assert_eq!(format, TextureFormat::Rgba32Float);
        let values: Vec<f32> = decoded
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // Signal 1.0 decodes through the extended sRGB EOTF to linear 1.0
        // (the 80-nit scRGB reference white).
        for channel in &values[0..3] {
            assert!(
                (channel - srgb_eotf_extended(1.0)).abs() < 1e-4,
                "expected ~1.0, got {channel}"
            );
        }
        assert_eq!(values[3], 1.0);
    }

    #[test]
    fn extended_srgb_decode_display_p3_converts_to_rec709() {
        // White is preserved through the P3 -> Rec.709 conversion.
        let white = rgba16f_texel([0x3c00, 0x3c00, 0x3c00, 0x3c00]);
        let (decoded, _) = decode_extended_srgb_screenshot(white, TextureFormat::Rgba16Float, true);
        let values: Vec<f32> = decoded
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for channel in &values[0..3] {
            assert!((channel - 1.0).abs() < 1e-4, "P3 white drifted: {channel}");
        }

        // Pure-red P3 signal: EOTF -> P3 linear (1, 0, 0), then P3 -> Rec.709.
        let red = rgba16f_texel([0x3c00, 0x0000, 0x0000, 0x3c00]);
        let (decoded, _) = decode_extended_srgb_screenshot(red, TextureFormat::Rgba16Float, true);
        let values: Vec<f32> = decoded
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let expected = DISPLAYP3_TO_REC709 * Vec3::new(1.0, 0.0, 0.0);
        assert!((values[0] - expected.x).abs() < 1e-4, "{}", values[0]);
        assert!((values[1] - expected.y).abs() < 1e-4, "{}", values[1]);
        assert!((values[2] - expected.z).abs() < 1e-4, "{}", values[2]);
    }

    #[test]
    fn extended_srgb_decode_passes_unexpected_formats_through() {
        let data = vec![0u8, 1, 2, 3];
        let (out, format) =
            decode_extended_srgb_screenshot(data.clone(), TextureFormat::Rgba8Unorm, false);
        assert_eq!(out, data);
        assert_eq!(format, TextureFormat::Rgba8Unorm);
    }
}
