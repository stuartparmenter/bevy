use super::{DlssFrameGeneration, DlssFrameGenerationSupported, DlssSdk};
extern crate alloc;
use alloc::sync::Arc;
use bevy_camera::{Camera, MainPassResolutionOverride, NormalizedRenderTarget, Projection};
use bevy_core_pipeline::prepass::ViewPrepassTextures;
use bevy_ecs::{prelude::*, query::Has};
use bevy_log::warn_once;
use bevy_math::{Mat4, UVec2, Vec2, Vec4Swizzles};
use bevy_render::{
    camera::{ExtractedCamera, TemporalJitter},
    diagnostic::RecordDiagnostics,
    render_resource::{
        Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
        TextureViewDescriptor,
    },
    renderer::{RenderAdapter, RenderContext, RenderDevice, RenderQueue},
    sync_world::{MainEntity, RenderEntity},
    texture::{CachedTexture, OutputColorAttachment, TextureCache},
    view::{
        window::{
            paced_present::{PacedPresentPlan, PacedPresentPlans, PacedWindows},
            ExtractedWindow,
        },
        ExtractedView, Msaa, ViewTargetAttachments,
    },
    MainWorld,
};
use dlss_wgpu::frame_generation::{
    DlssFrameGeneration as WgpuDlssFrameGeneration, DlssFrameGenerationCamera,
    DlssFrameGenerationRenderParameters,
};
use tracing::warn;

pub(super) fn extract_frame_generation(
    mut commands: Commands,
    mut main_world: ResMut<MainWorld>,
    mut paced_windows: ResMut<PacedWindows>,
    render_state: Query<(
        Has<DlssFrameGeneration>,
        Has<ViewFrameGenerationTextures>,
        Option<&ExtractedCamera>,
    )>,
) {
    let mut cameras = main_world.query::<(
        RenderEntity,
        &Camera,
        &Projection,
        Option<&mut DlssFrameGeneration>,
    )>();

    for (entity, camera, projection, mut frame_generation) in cameras.iter_mut(&mut main_world) {
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");
        let (had_frame_generation, prepared, extracted_camera) =
            render_state.get(entity).unwrap_or((false, false, None));
        // `extract_cameras` ran earlier this frame, so the render target is already normalized
        let window = match extracted_camera.and_then(|camera| camera.target.as_ref()) {
            Some(NormalizedRenderTarget::Window(window_ref)) => Some(window_ref.entity()),
            _ => None,
        };

        if let (Some(frame_generation), Projection::Perspective(_), Some(window)) =
            (frame_generation.as_deref_mut(), projection, window)
            && camera.is_active
        {
            entity_commands.insert(frame_generation.clone());
            frame_generation.reset = false;
            // Only pace the window once a prior prepare succeeded, so that the first frame
            // after enabling (and any prepare failure) presents normally instead of
            // suppressing swapchain acquisition with nothing to present.
            if prepared {
                paced_windows.0.insert(window);
            }
        } else if had_frame_generation {
            if frame_generation.is_some()
                && camera.is_active
                && !matches!(projection, Projection::Perspective(_))
            {
                warn_once!(
                    "DLSS Frame Generation requires a perspective projection, and will be disabled"
                );
            }
            entity_commands.remove::<(
                DlssFrameGeneration,
                FrameGenerationRenderContext,
                ViewFrameGenerationTextures,
            )>();
        }
    }
}

#[derive(Component)]
pub(super) struct FrameGenerationRenderContext {
    context: WgpuDlssFrameGeneration,
    previous_clip_from_world: Option<Mat4>,
    /// Monotonic rendered-frame counter for `DLSSG.BackbufferFrameID`.
    frame_id: u64,
    output_resolution: UVec2,
    render_resolution: UVec2,
    output_format: TextureFormat,
}

#[derive(Component)]
pub(super) struct ViewFrameGenerationTextures {
    /// The camera's redirected final output. Its default (non-sRGB) view is the NGX
    /// backbuffer input; it is retained and presented as the real frame.
    input: CachedTexture,
    /// sRGB view of [`Self::input`], rendered to by the camera and blitted by the pacer.
    input_srgb_view: TextureView,
    interpolated: Vec<InterpolatedTexture>,
}

struct InterpolatedTexture {
    /// NGX interpolated output. The default (non-sRGB) view is written by NGX.
    texture: CachedTexture,
    /// sRGB view blitted to the surface by the pacer.
    srgb_view: TextureView,
}

#[expect(clippy::too_many_arguments, reason = "render preparation resources")]
pub(super) fn prepare_frame_generation(
    mut commands: Commands,
    cameras: Query<(
        Entity,
        &ExtractedCamera,
        &ExtractedView,
        &DlssFrameGeneration,
        Option<&Msaa>,
        Option<&MainPassResolutionOverride>,
        Option<&FrameGenerationRenderContext>,
    )>,
    windows: Query<(MainEntity, &ExtractedWindow)>,
    paced_windows: Res<PacedWindows>,
    supported: Res<DlssFrameGenerationSupported>,
    sdk: Res<DlssSdk>,
    adapter: Res<RenderAdapter>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut texture_cache: ResMut<TextureCache>,
    mut attachments: ResMut<ViewTargetAttachments>,
) {
    for (entity, camera, view, settings, msaa, resolution_override, context) in &cameras {
        let Some(target @ NormalizedRenderTarget::Window(window_ref)) = camera.target.as_ref()
        else {
            continue;
        };
        let window_entity = window_ref.entity();
        let Some((_, window)) = windows.iter().find(|(window, _)| *window == window_entity) else {
            continue;
        };
        let Some(surface_format) = window.swap_chain_texture_format else {
            continue;
        };
        if msaa.is_some_and(|msaa| *msaa != Msaa::Off) {
            warn_once!("DLSS Frame Generation requires Msaa::Off on the camera, and is disabled");
            continue;
        }
        let Some((storage_format, view_format, view_formats)) =
            frame_generation_formats(surface_format)
        else {
            warn_once!("DLSS Frame Generation does not support surface format {surface_format:?}");
            commands
                .entity(entity)
                .remove::<(FrameGenerationRenderContext, ViewFrameGenerationTextures)>();
            continue;
        };

        let output_resolution = UVec2::new(window.physical_width, window.physical_height);
        let render_resolution = resolution_override.map_or(view.viewport.zw(), |size| size.0);
        let frames_to_generate = settings.mode.frames_to_generate();
        let max_frames_to_generate = supported.max_frames_to_generate();
        if frames_to_generate > max_frames_to_generate {
            warn_once!(
                "DlssFrameGenerationMode::{:?} is not supported on this machine (max {}x); falling back",
                settings.mode,
                max_frames_to_generate + 1,
            );
        }
        let frames_to_generate = frames_to_generate.min(max_frames_to_generate);

        let recreate_context = context.is_none_or(|context| {
            context.output_resolution != output_resolution
                || context.render_resolution != render_resolution
                || context.output_format != storage_format
        });
        if recreate_context {
            let context = WgpuDlssFrameGeneration::new(
                output_resolution.to_array(),
                render_resolution.to_array(),
                storage_format,
                false,
                Arc::clone(&sdk.0),
                &adapter,
                render_device.wgpu_device(),
                &render_queue,
            );
            match context {
                Ok(context) => {
                    commands
                        .entity(entity)
                        .insert(FrameGenerationRenderContext {
                            context,
                            previous_clip_from_world: None,
                            frame_id: 0,
                            output_resolution,
                            render_resolution,
                            output_format: storage_format,
                        });
                }
                Err(error) => {
                    warn!("Failed to create DLSS Frame Generation context: {error}");
                    commands
                        .entity(entity)
                        .remove::<(FrameGenerationRenderContext, ViewFrameGenerationTextures)>();
                    continue;
                }
            }
        }

        let descriptor = TextureDescriptor {
            label: None,
            size: Extent3d {
                width: output_resolution.x,
                height: output_resolution.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: storage_format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats,
        };
        let input = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("dlss_frame_generation_input"),
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING,
                ..descriptor
            },
        );
        let input_srgb_view = input.texture.create_view(&TextureViewDescriptor {
            label: Some("dlss_frame_generation_input_srgb_view"),
            format: Some(view_format),
            usage: Some(TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING),
            ..Default::default()
        });
        let interpolated = (0..frames_to_generate)
            .map(|_| {
                let texture = texture_cache.get(
                    &render_device,
                    TextureDescriptor {
                        label: Some("dlss_frame_generation_interpolated"),
                        ..descriptor.clone()
                    },
                );
                let srgb_view = texture.texture.create_view(&TextureViewDescriptor {
                    label: Some("dlss_frame_generation_interpolated_srgb_view"),
                    format: Some(view_format),
                    usage: Some(TextureUsages::TEXTURE_BINDING),
                    ..Default::default()
                });
                InterpolatedTexture { texture, srgb_view }
            })
            .collect();

        // Redirect the camera's final output into the input texture only while the window
        // is actually paced; during warm-up the camera renders to the swapchain as normal.
        if paced_windows.0.contains(&window_entity) {
            attachments.insert(
                target.clone(),
                OutputColorAttachment::new(input_srgb_view.clone(), view_format),
            );
        }
        commands.entity(entity).insert(ViewFrameGenerationTextures {
            input,
            input_srgb_view,
            interpolated,
        });
    }
}

/// Returns the NGX storage format, sRGB view format, and texture view formats to use for a
/// given window surface format.
///
/// NGX consumes and produces non-sRGB UNORM textures holding display-ready (sRGB encoded)
/// values; the camera renders and the pacer blits through sRGB views. The sRGB view format
/// always matches the surface view format the pacer presents with
/// (`surface_format.add_srgb_suffix()`).
fn frame_generation_formats(
    surface_format: TextureFormat,
) -> Option<(TextureFormat, TextureFormat, &'static [TextureFormat])> {
    match surface_format.remove_srgb_suffix() {
        TextureFormat::Bgra8Unorm => Some((
            TextureFormat::Bgra8Unorm,
            TextureFormat::Bgra8UnormSrgb,
            &[TextureFormat::Bgra8UnormSrgb],
        )),
        TextureFormat::Rgba8Unorm => Some((
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb,
            &[TextureFormat::Rgba8UnormSrgb],
        )),
        _ => None,
    }
}

pub(super) fn frame_generation(
    mut views: Query<(
        &DlssFrameGeneration,
        &Projection,
        &mut FrameGenerationRenderContext,
        &ViewFrameGenerationTextures,
        &ExtractedCamera,
        &ExtractedView,
        &ViewPrepassTextures,
        Option<&TemporalJitter>,
    )>,
    windows: Query<(MainEntity, &ExtractedWindow)>,
    paced_windows: Res<PacedWindows>,
    adapter: Res<RenderAdapter>,
    mut plans: ResMut<PacedPresentPlans>,
    mut ctx: RenderContext,
) {
    for (
        settings,
        projection,
        mut state,
        textures,
        camera,
        extracted_view,
        prepass_textures,
        temporal_jitter,
    ) in &mut views
    {
        let Some(NormalizedRenderTarget::Window(window_ref)) = camera.target.as_ref() else {
            continue;
        };
        let window_entity = window_ref.entity();
        if !windows.iter().any(|(window, _)| window == window_entity) {
            continue;
        }
        let Projection::Perspective(projection) = projection else {
            continue;
        };

        // Update per-frame state even during warm-up, so the first paced frame has history
        let world_from_view = extracted_view.world_from_view.to_matrix();
        let view_from_world = world_from_view.inverse();
        let current_clip_from_world = extracted_view.clip_from_view * view_from_world;
        let previous_clip_from_world = state
            .previous_clip_from_world
            .unwrap_or(current_clip_from_world);
        let reset = settings.reset || state.previous_clip_from_world.is_none();
        state.previous_clip_from_world = Some(current_clip_from_world);
        state.frame_id += 1;

        if !paced_windows.0.contains(&window_entity) {
            continue;
        }

        let (Some(depth), Some(motion_vectors)) =
            (&prepass_textures.depth, &prepass_textures.motion_vectors)
        else {
            // A paced window must always receive a plan, or nothing is presented
            plans.insert(
                window_entity,
                PacedPresentPlan {
                    frames: vec![textures.input_srgb_view.clone()],
                },
            );
            continue;
        };

        let render_size = state.render_resolution.as_vec2();
        let jitter_offset = temporal_jitter.map_or(Vec2::ZERO, |jitter| {
            jitter.offset * Vec2::new(-2.0, 2.0) / render_size
        });
        // NGX expects row-major post-multiplication matrices, which have the same memory
        // layout as glam's untransposed column-vector matrices
        let camera_data = DlssFrameGenerationCamera {
            camera_view_to_clip: extracted_view.clip_from_view.to_cols_array_2d(),
            clip_to_camera_view: extracted_view.clip_from_view.inverse().to_cols_array_2d(),
            clip_to_previous_clip: (previous_clip_from_world * current_clip_from_world.inverse())
                .to_cols_array_2d(),
            previous_clip_to_clip: (current_clip_from_world * previous_clip_from_world.inverse())
                .to_cols_array_2d(),
            jitter_offset: jitter_offset.to_array(),
            motion_vector_scale: [-render_size.x, -render_size.y],
            position: extracted_view.world_from_view.translation().to_array(),
            up: extracted_view.world_from_view.up().as_vec3().to_array(),
            right: extracted_view.world_from_view.right().as_vec3().to_array(),
            forward: extracted_view
                .world_from_view
                .forward()
                .as_vec3()
                .to_array(),
            near: projection.near,
            far: projection.far,
            vertical_fov: projection.fov,
            aspect_ratio: projection.aspect_ratio,
            depth_inverted: true,
            camera_motion_included: true,
            motion_vectors_dilated: false,
        };
        let outputs_interpolated = textures
            .interpolated
            .iter()
            .map(|interpolated| &*interpolated.texture.default_view)
            .collect::<Vec<_>>();
        let render_parameters = DlssFrameGenerationRenderParameters {
            backbuffer: &textures.input.default_view,
            depth: &depth.texture.default_view,
            motion_vectors: &motion_vectors.texture.default_view,
            hudless: None,
            ui: None,
            outputs_interpolated: &outputs_interpolated,
            output_real: None,
            camera: camera_data,
            reset,
            not_rendering_game_frames: false,
            partial_texture_size: Some(state.render_resolution.to_array()),
            backbuffer_frame_id: state.frame_id,
        };

        let diagnostics = ctx.diagnostic_recorder();
        let diagnostics = diagnostics.as_deref();
        let time_span = diagnostics.time_span(ctx.command_encoder(), "dlss_frame_generation");
        let result = state
            .context
            .render(render_parameters, ctx.command_encoder(), &adapter);
        let mut frames = Vec::with_capacity(textures.interpolated.len() + 1);
        match result {
            Ok(command_buffer) => {
                ctx.add_command_buffer(command_buffer);
                // Present the generated frames followed by the real frame, metered by the
                // driver. On reset the generated frames are copies of the input and are
                // skipped.
                if !reset {
                    frames.extend(
                        textures
                            .interpolated
                            .iter()
                            .map(|interpolated| interpolated.srgb_view.clone()),
                    );
                }
                frames.push(textures.input_srgb_view.clone());
            }
            Err(error) => {
                state.previous_clip_from_world = None;
                warn!("Failed to render DLSS Frame Generation: {error}");
                frames.push(textures.input_srgb_view.clone());
            }
        }
        time_span.end(ctx.command_encoder());
        plans.insert(window_entity, PacedPresentPlan { frames });
    }
}
