use crate::extract_resource::ExtractResourcePlugin;
use crate::renderer::WgpuWrapper;
use crate::sync_world::{MainEntity, RenderEntity, SyncToRenderWorld};
use crate::{camera::extract_cameras, renderer::RenderQueue};
use crate::{
    render_resource::{SurfaceTexture, TextureView},
    renderer::{RenderAdapter, RenderDevice, RenderInstance},
    Extract, ExtractSchedule, MainWorld, Render, RenderApp, RenderSystems,
};
use bevy_app::{App, Plugin, PostUpdate};
use bevy_ecs::entity::EntityHashSet;
use bevy_ecs::prelude::*;
use bevy_ecs::system::RunSystemOnce;
use bevy_log::{debug, info, warn, warn_once};
use bevy_utils::default;
use bevy_window::{
    CompositeAlphaMode, DisplayCalibrationPolicy, DisplayGamut, DisplayTarget, DisplayTransfer,
    DisplayTransfers, EffectiveDisplayTarget, OnMonitor, PresentMode, PrimaryWindow,
    RawHandleWrapper, Window, WindowClosing, WindowFocused, WindowMoved,
};
use core::num::NonZero;
use wgpu::{
    SurfaceColorSpace, SurfaceColorSpaces, SurfaceConfiguration, SurfaceFormatCapabilities,
    SurfaceTargetUnsafe, TextureFormat, TextureUsages, TextureViewDescriptor,
};

pub(crate) mod display_state;
pub mod display_target;
pub mod screenshot;

pub(crate) use display_state::*;
pub use display_target::*;
use screenshot::ScreenshotPlugin;

pub struct WindowRenderPlugin;

impl Plugin for WindowRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ScreenshotPlugin,
            ExtractResourcePlugin::<ManualDisplayTargets>::default(),
        ))
        .init_resource::<ManualDisplayTargets>()
        // Runs in the main schedule before extraction, so the render world
        // reads an `EffectiveDisplayTarget` from this frame.
        .add_systems(PostUpdate, resolve_calibration);

        // We need to sync the window entity in the render world
        // We can't use [`SyncComponentPlugin`] because it would introduce `bevy_render` as
        // a dependency to `bevy_window`
        {
            app.add_observer(|trigger: On<Add, Window>, mut commands: Commands| {
                commands.entity(trigger.entity).insert(SyncToRenderWorld);
            });

            // The primary window gets added before this plugin so we can't rely on the observer
            let _ = app.world_mut().run_system_once(
                |mut commands: Commands, windows: Query<Entity, With<Window>>| {
                    for entity in &windows {
                        commands.entity(entity).insert(SyncToRenderWorld);
                    }
                },
            );
        }

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                // Also initialized in the main world. Extraction overwrites the
                // resource in place; inserting it through `Commands` would be
                // deferred past `extract_cameras`, a same-frame reader.
                .init_resource::<ManualDisplayTargets>()
                .init_resource::<DisplayStateStore>()
                .add_systems(
                    ExtractSchedule,
                    (
                        extract_windows.before(extract_cameras),
                        write_back_display_state.after(extract_windows),
                    ),
                )
                .add_systems(
                    Render,
                    create_surfaces
                        .run_if(need_surface_configuration)
                        .before(prepare_windows),
                )
                .add_systems(Render, prepare_windows.in_set(RenderSystems::PrepareViews))
                .add_systems(
                    Render,
                    poll_display_state
                        .in_set(RenderSystems::PrepareViews)
                        .after(prepare_windows),
                );
        }
    }
}

#[derive(Component)]
pub struct ExtractedWindow {
    pub physical_width: u32,
    pub physical_height: u32,
    pub present_mode: PresentMode,
    pub desired_maximum_frame_latency: Option<NonZero<u32>>,
    /// Note: this will not always be the swap chain texture view. When taking a screenshot,
    /// this will point to an alternative texture instead to allow for copying the render result
    /// to CPU memory.
    pub swap_chain_texture_view: Option<TextureView>,
    pub swap_chain_texture: Option<SurfaceTexture>,
    pub swap_chain_texture_format: Option<TextureFormat>,
    /// This is an srgb view of [`ExtractedWindow::swap_chain_texture_format`]
    /// so that in shaders we are always in linear space.
    /// Under an HDR transfer this is the surface format itself: there is no
    /// hardware encode, so shaders write the signal directly.
    pub swap_chain_texture_view_format: Option<TextureFormat>,
    pub size_changed: bool,
    pub present_mode_changed: bool,
    pub alpha_mode: CompositeAlphaMode,
    /// The [`DisplayTarget`] extracted from the window entity.
    ///
    /// Every camera rendering to this window shares it.
    /// Use [`resolve_display_target`] to look one up for any render target kind.
    /// [`create_surfaces`] negotiates the surface from its
    /// [`transfer`](DisplayTarget::transfer) and reports the outcome in
    /// [`resolved_transfer`](Self::resolved_transfer).
    pub display_target: DisplayTarget,
    /// Whether the [`DisplayTarget`] changed in a way that needs the surface
    /// renegotiated: a [`DisplayTarget::transfer`] change, or a
    /// [`DisplayTarget::gamut`] change under
    /// [`DisplayTransfer::ExtendedSrgb`], the one transfer whose surface color
    /// space depends on the gamut.
    ///
    /// Works like [`size_changed`](Self::size_changed) and
    /// [`present_mode_changed`](Self::present_mode_changed): the surface is
    /// reconfigured with a fresh format, and the window's
    /// [`ViewTarget`](crate::view::ViewTarget)s are invalidated because the
    /// output format changes.
    /// Paper white and peak changes flow through uniforms instead.
    pub display_target_transfer_changed: bool,
    /// The [`DisplayTransfer`] the configured surface can carry, written by
    /// [`create_surfaces`] after negotiation.
    /// `None` until the surface has been configured.
    ///
    /// This is the requested [`DisplayTarget::transfer`] unless the surface
    /// could not provide it, in which case it is whatever
    /// `negotiate_surface_format` downgraded to.
    /// `prepare_view_display_targets` reads it to build each view's
    /// [`ViewDisplayTarget`](crate::view::ViewDisplayTarget), so the encoding
    /// pass keys on what the surface can show rather than on the request.
    pub resolved_transfer: Option<DisplayTransfer>,
    /// Set for one frame when the backing display may have changed: a window
    /// move, a focus regain, a monitor change, or a renegotiation that changed
    /// the resolved transfer (the OS HDR toggle re-picks the color space with
    /// no authored change).
    /// `poll_display_state` re-reads the live display state in response.
    ///
    /// A focus regain counts because Windows' SDR-content brightness slider
    /// fires no OS message, and the user has to leave the app to reach it.
    pub request_display_requery: bool,
    /// Whether this window's [`DisplayCalibrationPolicy`] opts any field into
    /// auto-resolution.
    ///
    /// Gates the per-frame live re-read in `poll_display_state` on Apple
    /// platforms.
    /// An all-manual project resolves the same either way, so it never pays
    /// for the query.
    pub display_calibration_auto: bool,
    /// Whether this window needs an initial buffer commit.
    ///
    /// On Wayland, windows must present at least once before they are shown.
    /// See <https://wayland.app/protocols/xdg-shell#xdg_surface>
    pub needs_initial_present: bool,
}

impl ExtractedWindow {
    fn set_swapchain_texture(
        &mut self,
        frame: wgpu::SurfaceTexture,
        texture_view_format: Option<TextureFormat>,
    ) {
        // Take the view format the negotiation stored in `SurfaceData` rather
        // than recomputing `add_srgb_suffix`, which would re-attach a hardware
        // sRGB encode the negotiation decided against.
        self.swap_chain_texture_view_format =
            Some(texture_view_format.unwrap_or_else(|| frame.texture.format()));
        let texture_view_descriptor = TextureViewDescriptor {
            format: self.swap_chain_texture_view_format,
            ..default()
        };
        self.swap_chain_texture_view = Some(TextureView::from(
            frame.texture.create_view(&texture_view_descriptor),
        ));
        self.swap_chain_texture = Some(SurfaceTexture::from(frame));
    }

    fn has_swapchain_texture(&self) -> bool {
        self.swap_chain_texture_view.is_some() && self.swap_chain_texture.is_some()
    }

    pub fn present(&mut self, queue: &RenderQueue) {
        if let Some(surface_texture) = self.swap_chain_texture.take() {
            // TODO(clean): winit docs recommends calling pre_present_notify before this.
            // though `present()` doesn't present the frame, it schedules it to be presented
            // by wgpu.
            // https://docs.rs/winit/0.29.9/wasm32-unknown-unknown/winit/window/struct.Window.html#method.pre_present_notify
            surface_texture.present(queue);
        }
    }
}

fn extract_windows(
    mut commands: Commands,
    mut extracted_windows: Query<&mut ExtractedWindow>,
    mut closing: Extract<MessageReader<WindowClosing>>,
    windows: Extract<
        Query<(
            Entity,
            RenderEntity,
            &Window,
            Option<&EffectiveDisplayTarget>,
            Option<&DisplayCalibrationPolicy>,
            &RawHandleWrapper,
            Has<PrimaryWindow>,
        )>,
    >,
    // Signal that the backing display may have changed. See
    // `ExtractedWindow::request_display_requery`.
    mut moved: Extract<MessageReader<WindowMoved>>,
    mut focused: Extract<MessageReader<WindowFocused>>,
    changed_monitor: Extract<Query<Entity, Changed<OnMonitor>>>,
    mut removed_monitor: Extract<RemovedComponents<OnMonitor>>,
    mut removed: Extract<RemovedComponents<RawHandleWrapper>>,
    mut removed_primary: Extract<RemovedComponents<PrimaryWindow>>,
    mapper: Extract<Query<&RenderEntity>>,
) {
    // Collect the windows whose backing display may have changed this frame.
    // `removed_monitor` also fires for despawned windows. Those entities never
    // match the extraction query below, so they sit unused in the set.
    let mut display_requery: EntityHashSet = moved
        .read()
        .map(|moved| moved.window)
        .chain(changed_monitor.iter())
        .chain(removed_monitor.read())
        .collect();
    display_requery.extend(focused.read().filter(|f| f.focused).map(|f| f.window));

    for (
        entity,
        render_entity,
        window,
        effective_display_target,
        calibration_policy,
        handle,
        is_primary,
    ) in windows.iter()
    {
        // `EffectiveDisplayTarget` is a required component, but removing it is
        // legal, so fall back to the SDR default rather than dropping the
        // window from extraction.
        let display_target = effective_display_target
            .map(|effective| effective.target)
            .unwrap_or_default();
        if is_primary {
            commands.entity(render_entity).insert(PrimaryWindow);
        }

        let (new_width, new_height) = (
            window.resolution.physical_width().max(1),
            window.resolution.physical_height().max(1),
        );

        let Ok(mut extracted_window) = extracted_windows.get_mut(render_entity) else {
            commands.entity(render_entity).insert((
                ExtractedWindow {
                    physical_width: new_width,
                    physical_height: new_height,
                    present_mode: window.present_mode,
                    desired_maximum_frame_latency: window.desired_maximum_frame_latency,
                    swap_chain_texture: None,
                    swap_chain_texture_view: None,
                    size_changed: false,
                    swap_chain_texture_format: None,
                    swap_chain_texture_view_format: None,
                    present_mode_changed: false,
                    alpha_mode: window.composite_alpha_mode,
                    display_target,
                    display_target_transfer_changed: false,
                    // This path skips the sync block below, so without this a
                    // window event on the first extracted frame would be lost.
                    request_display_requery: display_requery.contains(&entity),
                    display_calibration_auto: calibration_policy
                        .is_some_and(DisplayCalibrationPolicy::has_auto),
                    resolved_transfer: None,
                    needs_initial_present: true,
                },
                handle.clone(),
            ));
            continue;
        };

        // Diff the fields that affect surface negotiation, for
        // `need_surface_configuration` / `create_surfaces` and the `ViewTarget`
        // invalidation in `cleanup_view_targets_for_resize`. See
        // `ExtractedWindow::display_target_transfer_changed`.
        let previous = extracted_window.display_target;
        let transfer_changed = previous.transfer != display_target.transfer;
        let extended_srgb_gamut_changed = (previous.transfer == DisplayTransfer::ExtendedSrgb
            || display_target.transfer == DisplayTransfer::ExtendedSrgb)
            && previous.gamut != display_target.gamut;
        extracted_window.display_target_transfer_changed =
            transfer_changed || extended_srgb_gamut_changed;
        extracted_window.display_target = display_target;

        extracted_window.request_display_requery = display_requery.contains(&entity);
        extracted_window.display_calibration_auto =
            calibration_policy.is_some_and(DisplayCalibrationPolicy::has_auto);

        if extracted_window.swap_chain_texture.is_none() {
            // If we called present on the previous swap-chain texture last update,
            // then drop the swap chain frame here, otherwise we can keep it for the
            // next update as an optimization. `prepare_windows` will only acquire a new
            // swap chain texture if needed.
            extracted_window.swap_chain_texture_view = None;
        }

        extracted_window.size_changed = new_width != extracted_window.physical_width
            || new_height != extracted_window.physical_height;
        extracted_window.present_mode_changed =
            window.present_mode != extracted_window.present_mode;

        if extracted_window.size_changed {
            debug!(
                "Window size changed from {}x{} to {}x{}",
                extracted_window.physical_width,
                extracted_window.physical_height,
                new_width,
                new_height
            );
            extracted_window.physical_width = new_width;
            extracted_window.physical_height = new_height;
        }

        if extracted_window.present_mode_changed {
            debug!(
                "Window Present Mode changed from {:?} to {:?}",
                extracted_window.present_mode, window.present_mode
            );
            extracted_window.present_mode = window.present_mode;
        }
    }

    for closing_window in closing.read() {
        if let Ok(render_entity) = mapper.get(closing_window.window) {
            commands.entity(render_entity.entity()).despawn();
        }
    }
    for removed_window in removed.read() {
        if let Ok(render_entity) = mapper.get(removed_window) {
            commands.entity(render_entity.entity()).despawn();
        }
    }
    for removed_window in removed_primary.read() {
        if let Ok(render_entity) = mapper.get(removed_window) {
            commands
                .entity(render_entity.entity())
                .remove::<PrimaryWindow>();
        }
    }
}

/// Inserts `value` onto `entity` in the main world, but only when it differs
/// from the component already there, so a `Changed<C>` reader sees a real
/// transition instead of a write every frame. A missing entity is skipped.
pub(super) fn insert_on_change<C: Component + PartialEq>(
    main_world: &mut MainWorld,
    entity: Entity,
    value: C,
) {
    let Ok(mut entity_mut) = main_world.get_entity_mut(entity) else {
        return;
    };
    if entity_mut.get::<C>() != Some(&value) {
        entity_mut.insert(value);
    }
}

#[derive(Component)]
pub struct SurfaceData {
    // TODO: what lifetime should this be?
    surface: WgpuWrapper<wgpu::Surface<'static>>,
    configuration: SurfaceConfiguration,
    texture_view_format: Option<TextureFormat>,
    /// The [`DisplayTransfer`] the configured (format, color space) pair
    /// carries. Mirrored into [`ExtractedWindow::resolved_transfer`] by
    /// [`create_surfaces`].
    resolved_transfer: DisplayTransfer,
    /// The [`DisplayTransfer`]s this surface can present (see
    /// [`supported_transfers`]). Mirrored to the main world as
    /// [`WindowSurfaceTransfers::supported`](bevy_window::WindowSurfaceTransfers::supported)
    /// by [`write_back_display_state`].
    supported_transfers: DisplayTransfers,
    /// The transfer this surface carried before [`prepare_windows`]
    /// renegotiated it mid-frame, cleared at the top of every
    /// [`prepare_windows`] pass.
    ///
    /// A renegotiation on the `Outdated` path lands after
    /// `prepare_view_display_targets` built the encoder's
    /// [`ViewDisplayTarget`](crate::view::ViewDisplayTarget)s, so this frame's
    /// pixels are still encoded for the old transfer.
    /// `prepare_screenshots` reads it through
    /// [`encoded_transfer`](Self::encoded_transfer) when decoding those pixels.
    /// The texture's own format and size come from the live configuration,
    /// which is what was allocated.
    transfer_before_renegotiation: Option<DisplayTransfer>,
}

impl SurfaceData {
    /// The format the renderer's final blit writes through: the sRGB view of
    /// the surface format under the plain sRGB transfer when the format has a
    /// distinct sRGB pair, otherwise the surface format itself.
    fn view_format(&self) -> TextureFormat {
        self.texture_view_format
            .unwrap_or(self.configuration.format)
    }

    /// The transfer this frame's pixels were encoded with: the live resolved
    /// transfer, or
    /// [`transfer_before_renegotiation`](Self::transfer_before_renegotiation)
    /// when [`prepare_windows`] renegotiated the surface mid-frame.
    ///
    /// The gamut is the requested [`DisplayTarget::gamut`], which renegotiation
    /// never touches, so it needs no such record.
    fn encoded_transfer(&self) -> DisplayTransfer {
        self.transfer_before_renegotiation
            .unwrap_or(self.resolved_transfer)
    }

    /// Applies a fresh [`negotiate_surface_format`] outcome to the stored
    /// configuration: format, color space, resolved transfer, and the sRGB view
    /// format.
    ///
    /// Only the plain sRGB transfer gets an sRGB view, and only when the format
    /// has a distinct sRGB pair.
    /// The gate is the resolved transfer rather than the format alone because a
    /// last-resort HDR10 negotiation can land on an 8-bit format, and a hardware
    /// sRGB encode on top of already-PQ-encoded signal would double-encode it.
    fn apply_negotiated(&mut self, negotiated: NegotiatedSurface) {
        let view_format = negotiated.format.add_srgb_suffix();
        self.configuration.format = negotiated.format;
        self.configuration.color_space = negotiated.color_space;
        self.texture_view_format = (negotiated.resolved_transfer == DisplayTransfer::Srgb
            && view_format != negotiated.format)
            .then_some(view_format);
        self.configuration.view_formats = match self.texture_view_format {
            Some(format) => vec![format],
            None => vec![],
        };
        self.resolved_transfer = negotiated.resolved_transfer;
    }

    /// Renegotiates this surface when its configured explicit color space is
    /// gone from `caps`, returning the transfer it replaced, or `None` when it
    /// left the surface alone.
    ///
    /// An explicit (non-`Auto`) color space can disappear at runtime: turning
    /// the OS HDR toggle off makes DX12 stop advertising HDR10 for the output.
    /// Reconfiguring with the stale pair would fail wgpu validation
    /// (`ConfigureSurfaceError::UnsupportedColorSpace`) and bring the renderer
    /// down, so renegotiate from the fresh capabilities instead.
    ///
    /// Callers that do not already hold capabilities should gate on
    /// `configuration.color_space.to_color_spaces().is_some()` first.
    /// The SDR path negotiates `Auto`, which can never go unsupported, so it
    /// should not pay a driver query on every `Outdated` event.
    fn renegotiate_if_color_space_lost(
        &mut self,
        caps: &wgpu::SurfaceCapabilities,
        requested_transfer: DisplayTransfer,
        requested_gamut: DisplayGamut,
    ) -> Option<DisplayTransfer> {
        let flag = self.configuration.color_space.to_color_spaces()?;
        if caps.color_spaces(self.configuration.format).contains(flag) {
            return None;
        }
        warn_once!(
            "The configured surface color space ({:?}) is no longer supported for \
            {:?} (did the OS HDR setting change?); renegotiating the swapchain \
            from the current capabilities.",
            self.configuration.color_space,
            self.configuration.format
        );
        let previous = self.resolved_transfer;
        self.apply_negotiated(negotiate_surface_format(
            &caps.formats,
            &caps.format_capabilities,
            requested_transfer,
            requested_gamut,
        ));
        Some(previous)
    }
}

/// (re)configures window surfaces, and obtains a swapchain texture for rendering.
///
/// NOTE: `get_current_texture` in `prepare_windows` can take a long time if the GPU workload is
/// the performance bottleneck. This can be seen in profiles as multiple prepare-set systems all
/// taking an unusually long time to complete, and all finishing at about the same time as the
/// `prepare_windows` system. Improvements in bevy are planned to avoid this happening when it
/// should not but it will still happen as it is easy for a user to create a large GPU workload
/// relative to the GPU performance and/or CPU workload.
/// This can be caused by many reasons, but several of them are:
/// - GPU workload is more than your current GPU can manage
/// - Error / performance bug in your custom shaders
/// - wgpu was unable to detect a proper GPU hardware-accelerated device given the chosen
///   [`Backends`](crate::settings::Backends), [`WgpuLimits`](crate::settings::WgpuLimits),
///   and/or [`WgpuFeatures`](crate::settings::WgpuFeatures). For example, on Windows currently
///   `DirectX 11` is not supported by wgpu 0.12 and so if your GPU/drivers do not support Vulkan,
///   it may be that a software renderer called "Microsoft Basic Render Driver" using `DirectX 12`
///   will be chosen and performance will be very poor. This is visible in a log message that is
///   output during renderer initialization.
///   Another alternative is to try to use [`ANGLE`](https://github.com/gfx-rs/wgpu#angle) and
///   [`Backends::GL`](crate::settings::Backends::GL) with the `gles` feature enabled if your
///   GPU/drivers support `OpenGL 4.3` / `OpenGL ES 3.0` or later.
pub fn prepare_windows(
    mut windows: Query<(MainEntity, &mut ExtractedWindow, Option<&mut SurfaceData>)>,
    render_device: Res<RenderDevice>,
    render_adapter: Res<RenderAdapter>,
    sorted_cameras: Res<crate::camera::SortedCameras>,
    #[cfg(target_os = "linux")] render_instance: Res<RenderInstance>,
) {
    for (main_entity, mut window, maybe_surface_data) in &mut windows {
        let Some(mut surface_data) = maybe_surface_data else {
            continue;
        };
        // Clear last frame's record before any early exit below, so it only
        // ever describes a renegotiation from this pass.
        surface_data.transfer_before_renegotiation = None;

        // Skip acquiring a swap-chain texture for windows that no camera
        // targets. This avoids a wasted clear pass in
        // `handle_uncovered_swap_chains` that triggers a DMA-fence fd leak on
        // Adreno 740 (Quest 3). The exception is windows that still need their
        // initial present (required on Wayland).
        let is_camera_target = sorted_cameras.0.iter().any(|c| {
            matches!(
                &c.target,
                Some(bevy_camera::NormalizedRenderTarget::Window(w)) if w.entity() == main_entity
            ) && matches!(c.output_mode, bevy_camera::CameraOutputMode::Write { .. })
        });
        if !is_camera_target && !window.needs_initial_present {
            continue;
        }

        // We didn't present the previous frame, so we can keep using our existing swapchain texture.
        if window.has_swapchain_texture()
            && !window.size_changed
            && !window.present_mode_changed
            && !window.display_target_transfer_changed
        {
            continue;
        }

        // A recurring issue is hitting `wgpu::SurfaceError::Timeout` on certain Linux
        // mesa driver implementations. This seems to be a quirk of some drivers.
        // We'd rather keep panicking when not on Linux mesa, because in those case,
        // the `Timeout` is still probably the symptom of a degraded unrecoverable
        // application state.
        // see https://github.com/bevyengine/bevy/pull/5957
        // and https://github.com/gfx-rs/wgpu/issues/1218
        #[cfg(target_os = "linux")]
        let may_erroneously_timeout = || {
            bevy_tasks::IoTaskPool::get().scope(|scope| {
                scope.spawn(async {
                    render_instance
                        .enumerate_adapters(wgpu::Backends::VULKAN)
                        .await
                        .iter()
                        .any(|adapter| {
                            let name = adapter.get_info().name;
                            name.starts_with("Radeon")
                                || name.starts_with("AMD")
                                || name.starts_with("Intel")
                        })
                });
            })[0]
        };

        let surface = &surface_data.surface;
        match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                window.set_swapchain_texture(surface_texture, surface_data.texture_view_format);
            }
            #[cfg(target_os = "linux")]
            wgpu::CurrentSurfaceTexture::Timeout if may_erroneously_timeout() => {
                bevy_log::trace!(
                    "Couldn't get swap chain texture. This is probably a quirk \
                        of your Linux GPU driver, so it can be safely ignored."
                );
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                // A surface can go outdated because the OS color capabilities
                // changed, not just because of a resize.
                if surface_data
                    .configuration
                    .color_space
                    .to_color_spaces()
                    .is_some()
                {
                    let caps = surface_data.surface.get_capabilities(&render_adapter);
                    if let Some(previous) = surface_data.renegotiate_if_color_space_lost(
                        &caps,
                        window.display_target.transfer,
                        window.display_target.gamut,
                    ) {
                        surface_data.transfer_before_renegotiation = Some(previous);
                        window.resolved_transfer = Some(surface_data.resolved_transfer);
                        window.request_display_requery |=
                            previous != surface_data.resolved_transfer;
                    }
                }
                let surface = &surface_data.surface;
                render_device.configure_surface(surface, &surface_data.configuration);
                let frame = match surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(surface_texture)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => surface_texture,
                    variant => {
                        // This is a common occurrence on X11 and Xwayland with NVIDIA drivers
                        // when opening and resizing the window.
                        warn!(
                            "Couldn't get swap chain texture after configuring. Cause: '{variant:?}'"
                        );
                        continue;
                    }
                };
                window.set_swapchain_texture(frame, surface_data.texture_view_format);
            }
            wgpu::CurrentSurfaceTexture::Occluded => {}
            other => {
                bevy_log::error!("Couldn't get swap chain texture: {other:?}");
            }
        }
        window.swap_chain_texture_format = Some(surface_data.configuration.format);
    }
}

pub fn need_surface_configuration(windows: Query<(&ExtractedWindow, Has<SurfaceData>)>) -> bool {
    for (window, has_surface_data) in &windows {
        if !has_surface_data
            || window.size_changed
            || window.present_mode_changed
            || window.display_target_transfer_changed
        {
            return true;
        }
    }
    false
}

/// The outcome of [`negotiate_surface_format`]: the (format, color space) pair
/// [`create_surfaces`] configures the surface with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NegotiatedSurface {
    /// The swapchain texture format.
    format: TextureFormat,
    /// The color space the presentation engine interprets the swapchain in
    /// ([`SurfaceConfiguration::color_space`]).
    ///
    /// Always [`SurfaceColorSpace::Auto`] (the SDR path) or an explicit color
    /// space the surface advertised, so configuring the pair cannot fail wgpu's
    /// `ConfigureSurfaceError::UnsupportedColorSpace` validation against the
    /// capabilities the negotiation ran on.
    color_space: SurfaceColorSpace,
    /// The transfer the configured surface carries, reported back through
    /// [`ExtractedWindow::resolved_transfer`].
    resolved_transfer: DisplayTransfer,
}

/// Mirror of `wgpu::SurfaceCapabilities::color_spaces`, over the raw
/// capability slice so [`negotiate_surface_format`] stays unit-testable
/// without a live surface.
fn advertised_color_spaces(
    format_capabilities: &[SurfaceFormatCapabilities],
    format: TextureFormat,
) -> SurfaceColorSpaces {
    format_capabilities
        .iter()
        .filter(|fc| fc.format == format)
        .fold(SurfaceColorSpaces::empty(), |acc, fc| acc | fc.color_spaces)
}

/// Negotiates an `Rgba16Float` swapchain in the extended-sRGB-linear (scRGB)
/// color space, if the surface advertises the pair.
///
/// scRGB (IEC 61966-2-2) is an extended-range float encoding against
/// Rec.709/sRGB primaries, so only `Rgba16Float` is considered.
fn negotiate_scrgb_linear(
    format_capabilities: &[SurfaceFormatCapabilities],
) -> Option<NegotiatedSurface> {
    advertised_color_spaces(format_capabilities, TextureFormat::Rgba16Float)
        .contains(SurfaceColorSpaces::EXTENDED_SRGB_LINEAR)
        .then_some(NegotiatedSurface {
            format: TextureFormat::Rgba16Float,
            color_space: SurfaceColorSpace::ExtendedSrgbLinear,
            resolved_transfer: DisplayTransfer::ScRgbLinear,
        })
}

/// Negotiates a swapchain in the HDR10 (PQ / SMPTE ST 2084, Rec.2020) color
/// space, if the surface advertises it for any format.
///
/// Format preference follows wgpu's canonical HDR example: `Rgb10a2Unorm`
/// (HDR10's native 10-bit container; what DX12 and most Vulkan drivers
/// expose) first, `Rgba16Float` second (advertised on Metal), then any other
/// format the surface lists with HDR10 support, in capability order.
fn negotiate_hdr10(format_capabilities: &[SurfaceFormatCapabilities]) -> Option<NegotiatedSurface> {
    const PREFERRED: &[TextureFormat] = &[TextureFormat::Rgb10a2Unorm, TextureFormat::Rgba16Float];
    let preferred = PREFERRED.iter().copied().filter(|&format| {
        advertised_color_spaces(format_capabilities, format).contains(SurfaceColorSpaces::BT2100_PQ)
    });
    // sRGB formats are excluded: their stores bake in the sRGB OETF, which
    // would re-encode the already-PQ-encoded signal.
    let any = format_capabilities
        .iter()
        .filter(|fc| {
            fc.color_spaces.contains(SurfaceColorSpaces::BT2100_PQ) && !fc.format.is_srgb()
        })
        .map(|fc| fc.format);
    preferred.chain(any).next().map(|format| NegotiatedSurface {
        format,
        color_space: SurfaceColorSpace::Bt2100Pq,
        resolved_transfer: DisplayTransfer::Pq,
    })
}

/// Negotiates a swapchain in one of the two encoded extended-range sRGB color
/// spaces (`ExtendedSrgb` for Rec.709, `ExtendedDisplayP3` for Display-P3), if
/// the surface advertises `flag` for any format.
///
/// `Rgba16Float` is preferred, then any other non-sRGB format the surface
/// lists with the color space, in capability order. sRGB formats are excluded,
/// as in [`negotiate_hdr10`]. Both color spaces resolve to
/// [`DisplayTransfer::ExtendedSrgb`]; the gamut rides [`DisplayTarget::gamut`].
fn negotiate_extended_srgb_space(
    format_capabilities: &[SurfaceFormatCapabilities],
    flag: SurfaceColorSpaces,
    color_space: SurfaceColorSpace,
) -> Option<NegotiatedSurface> {
    let preferred = core::iter::once(TextureFormat::Rgba16Float)
        .filter(|&format| advertised_color_spaces(format_capabilities, format).contains(flag));
    let any = format_capabilities
        .iter()
        .filter(|fc| fc.color_spaces.contains(flag) && !fc.format.is_srgb())
        .map(|fc| fc.format);
    preferred.chain(any).next().map(|format| NegotiatedSurface {
        format,
        color_space,
        resolved_transfer: DisplayTransfer::ExtendedSrgb,
    })
}

/// Negotiates an encoded extended-range sRGB swapchain in the `ExtendedSrgb`
/// (Rec.709) color space. This is the web's HDR path, since browser WebGPU
/// cannot present a linear-transfer canvas. Metal and Vulkan advertise it too.
fn negotiate_extended_srgb(
    format_capabilities: &[SurfaceFormatCapabilities],
) -> Option<NegotiatedSurface> {
    negotiate_extended_srgb_space(
        format_capabilities,
        SurfaceColorSpaces::EXTENDED_SRGB,
        SurfaceColorSpace::ExtendedSrgb,
    )
}

/// Negotiates an encoded extended-range Display-P3 swapchain in the
/// `ExtendedDisplayP3` color space (wide-gamut HDR), advertised by Metal and
/// browser WebGPU on HDR-capable displays.
fn negotiate_extended_display_p3(
    format_capabilities: &[SurfaceFormatCapabilities],
) -> Option<NegotiatedSurface> {
    negotiate_extended_srgb_space(
        format_capabilities,
        SurfaceColorSpaces::EXTENDED_DISPLAY_P3,
        SurfaceColorSpace::ExtendedDisplayP3,
    )
}

/// The [`DisplayTransfer`]s a surface with these capabilities can present,
/// mirrored to the main world as
/// [`WindowSurfaceTransfers::supported`](bevy_window::WindowSurfaceTransfers::supported).
///
/// [`DisplayTransfer::Srgb`] is always a member; every other transfer is one
/// exactly when its negotiation helper can satisfy a request for it, so the
/// set never offers a transfer that would silently downgrade.
/// [`DisplayTransfer::ExtendedSrgb`] counts either encoded extended-range
/// color space, since the gamut rides [`DisplayTarget::gamut`].
/// This is a set, not a cycle order; that lives in [`DisplayTransfers::iter`].
fn supported_transfers(format_capabilities: &[SurfaceFormatCapabilities]) -> DisplayTransfers {
    let mut transfers = DisplayTransfers::EMPTY.with(DisplayTransfer::Srgb);
    if negotiate_scrgb_linear(format_capabilities).is_some() {
        transfers = transfers.with(DisplayTransfer::ScRgbLinear);
    }
    if negotiate_extended_srgb(format_capabilities).is_some()
        || negotiate_extended_display_p3(format_capabilities).is_some()
    {
        transfers = transfers.with(DisplayTransfer::ExtendedSrgb);
    }
    if negotiate_hdr10(format_capabilities).is_some() {
        transfers = transfers.with(DisplayTransfer::Pq);
    }
    transfers
}

/// Negotiates the (format, color space) pair for a window surface from the
/// surface's capabilities, honoring the requested [`DisplayTransfer`] when
/// possible.
///
/// `auto_formats` is `SurfaceCapabilities::formats`: the formats configurable
/// with [`SurfaceColorSpace::Auto`], in preference order. `format_capabilities`
/// is `SurfaceCapabilities::format_capabilities`: every format with the color
/// spaces the surface supports it in, a superset of `auto_formats`.
/// `requested_gamut` is read only by the [`DisplayTransfer::ExtendedSrgb`] arm,
/// the one transfer whose surface color space depends on the gamut.
/// Policy per requested transfer:
///
/// - [`DisplayTransfer::Srgb`] (the default): the first of `Rgba8UnormSrgb` /
///   `Bgra8UnormSrgb` in capability order, else the surface's first
///   `Auto`-configurable format, paired with [`SurfaceColorSpace::Auto`].
///   `Auto` lets wgpu pick the color space (sRGB for the 8-bit formats,
///   extended-sRGB-linear for a first-listed `Rgba16Float` where supported) and
///   is always valid for formats in `auto_formats`, whereas an explicit
///   [`SurfaceColorSpace::Srgb`] would fail validation on drivers that do not
///   advertise the sRGB color space by name.
/// - [`DisplayTransfer::ScRgbLinear`]: `Rgba16Float` +
///   [`SurfaceColorSpace::ExtendedSrgbLinear`] when advertised (macOS/iOS Metal
///   EDR, Windows Vulkan/DX12, Wayland Vulkan). This one is native only, so the
///   web requests [`DisplayTransfer::ExtendedSrgb`] instead. Otherwise warn and
///   downgrade to SDR.
/// - [`DisplayTransfer::ExtendedSrgb`]: the requested gamut picks the surface
///   color space. [`DisplayGamut::DisplayP3`](bevy_window::DisplayGamut) gets
///   [`negotiate_extended_display_p3`], any other gamut gets
///   [`negotiate_extended_srgb`]. There is no cross-gamut downgrade: a
///   Display-P3 request that cannot get `ExtendedDisplayP3` falls straight to
///   SDR, because a Rec.709 surface would mismatch the gamut the encoder emits
///   (the returned transfer carries no gamut).
/// - [`DisplayTransfer::Pq`]: HDR10 ([`SurfaceColorSpace::Bt2100Pq`]) on the
///   advertised formats (see [`negotiate_hdr10`]), which needs the OS to have
///   HDR output enabled on DX12/Vulkan. When it is unavailable the chain is
///   PQ -> scRGB-linear -> SDR sRGB, each step with its own warning.
///
/// The encoder coerces the gamut to match the surface: PQ targets are encoded
/// in Rec.2020, and scRGB-linear targets in Rec.709 coordinates, where a wide
/// gamut rides out-of-range components.
fn negotiate_surface_format(
    auto_formats: &[TextureFormat],
    format_capabilities: &[SurfaceFormatCapabilities],
    requested_transfer: DisplayTransfer,
    requested_gamut: DisplayGamut,
) -> NegotiatedSurface {
    match requested_transfer {
        DisplayTransfer::Srgb => {}
        DisplayTransfer::ScRgbLinear => {
            if let Some(negotiated) = negotiate_scrgb_linear(format_capabilities) {
                return negotiated;
            }
            warn_once!(
                "DisplayTransfer::ScRgbLinear was requested, but this surface does not \
                support an Rgba16Float swapchain in the extended-sRGB-linear color \
                space. Downgrading to SDR sRGB output. scRGB-linear output requires an \
                HDR-capable display on macOS/iOS (Metal), Windows (Vulkan/DX12), or \
                Wayland (Vulkan); on the web, request DisplayTransfer::ExtendedSrgb \
                (the encoded sibling) instead."
            );
        }
        DisplayTransfer::ExtendedSrgb => {
            // `prepare_screenshots` decodes a window readback from the
            // requested gamut, so an `ExtendedDisplayP3` surface must always
            // mean a Display-P3 gamut.
            if requested_gamut == DisplayGamut::DisplayP3 {
                if let Some(negotiated) = negotiate_extended_display_p3(format_capabilities) {
                    return negotiated;
                }
                warn_once!(
                    "DisplayTransfer::ExtendedSrgb with DisplayGamut::DisplayP3 was \
                    requested, but this surface does not advertise the ExtendedDisplayP3 \
                    color space (wide-gamut HDR, available on Metal and browser WebGPU on \
                    HDR-capable displays). Downgrading to SDR sRGB output."
                );
            } else {
                if let Some(negotiated) = negotiate_extended_srgb(format_capabilities) {
                    return negotiated;
                }
                warn_once!(
                    "DisplayTransfer::ExtendedSrgb was requested, but this surface does \
                    not advertise the encoded extended-range sRGB color space (available \
                    on Metal, Vulkan, and browser WebGPU on HDR-capable displays). \
                    Downgrading to SDR sRGB output."
                );
            }
        }
        DisplayTransfer::Pq => {
            if let Some(negotiated) = negotiate_hdr10(format_capabilities) {
                return negotiated;
            }
            warn_once!(
                "DisplayTransfer::Pq was requested, but this surface does not \
                advertise the HDR10 (PQ) color space — the OS may have HDR output \
                disabled, or the backend lacks support. Downgrading to scRGB-linear \
                if available, else SDR sRGB."
            );
            if let Some(negotiated) = negotiate_scrgb_linear(format_capabilities) {
                return negotiated;
            }
            warn_once!(
                "DisplayTransfer::Pq could not be downgraded to scRGB-linear either \
                (no Rgba16Float extended-sRGB-linear support); downgrading to SDR \
                sRGB output."
            );
        }
    }

    // SDR path: prefer sRGB formats for surfaces, but fall back to the first
    // available format if no sRGB formats are available.
    if let Some(first) = auto_formats.first() {
        let mut format = *first;
        for available_format in auto_formats {
            // Rgba8UnormSrgb and Bgra8UnormSrgb and the only sRGB formats wgpu exposes that we can use for surfaces.
            if *available_format == TextureFormat::Rgba8UnormSrgb
                || *available_format == TextureFormat::Bgra8UnormSrgb
            {
                format = *available_format;
                break;
            }
        }
        return NegotiatedSurface {
            format,
            color_space: SurfaceColorSpace::Auto,
            resolved_transfer: DisplayTransfer::Srgb,
        };
    }

    // Some drivers report formats only in explicit-opt-in (wide-gamut / HDR)
    // color spaces when the OS is in HDR mode, leaving
    // `SurfaceCapabilities::formats` (the `Auto`-configurable set) empty.
    // Configuring such a format with `Auto` fails wgpu validation, so pick the
    // first advertised pair we can drive instead of panicking.
    for (flag, color_space, resolved_transfer) in [
        (
            SurfaceColorSpaces::SRGB,
            SurfaceColorSpace::Srgb,
            DisplayTransfer::Srgb,
        ),
        (
            SurfaceColorSpaces::EXTENDED_DISPLAY_P3,
            SurfaceColorSpace::ExtendedDisplayP3,
            DisplayTransfer::ExtendedSrgb,
        ),
        (
            SurfaceColorSpaces::EXTENDED_SRGB,
            SurfaceColorSpace::ExtendedSrgb,
            DisplayTransfer::ExtendedSrgb,
        ),
        (
            SurfaceColorSpaces::EXTENDED_SRGB_LINEAR,
            SurfaceColorSpace::ExtendedSrgbLinear,
            DisplayTransfer::ScRgbLinear,
        ),
        (
            SurfaceColorSpaces::BT2100_PQ,
            SurfaceColorSpace::Bt2100Pq,
            DisplayTransfer::Pq,
        ),
    ] {
        if color_space == SurfaceColorSpace::ExtendedDisplayP3
            && requested_gamut != DisplayGamut::DisplayP3
        {
            continue;
        }
        if color_space == SurfaceColorSpace::ExtendedSrgb
            && requested_gamut == DisplayGamut::DisplayP3
        {
            continue;
        }
        if let Some(fc) = format_capabilities.iter().find(|fc| {
            fc.color_spaces.contains(flag)
                && (resolved_transfer == DisplayTransfer::Srgb || !fc.format.is_srgb())
        }) {
            warn_once!(
                "This surface advertises no Auto-configurable formats; falling back to \
                {:?} in the {color_space:?} color space (resolved transfer: \
                {resolved_transfer:?}).",
                fc.format
            );
            return NegotiatedSurface {
                format: fc.format,
                color_space,
                resolved_transfer,
            };
        }
    }

    // Nothing in either capability list can be driven.
    panic!("No supported formats for surface");
}

// 2 is wgpu's default/what we've been using so far.
// 1 is the minimum, but may cause lower framerates due to the cpu waiting for the gpu to finish
// all work for the previous frame before starting work on the next frame, which then means the gpu
// has to wait for the cpu to finish to start on the next frame.
const DEFAULT_DESIRED_MAXIMUM_FRAME_LATENCY: u32 = 2;

/// Creates window surfaces.
pub fn create_surfaces(
    mut commands: Commands,
    // By accessing a NonSend resource, we tell the scheduler to put this system on the main thread,
    // which is necessary for some OS's
    #[cfg(any(target_os = "macos", target_os = "ios"))] _marker: bevy_ecs::system::NonSendMarker,
    mut windows: Query<(
        Entity,
        &mut ExtractedWindow,
        &RawHandleWrapper,
        Option<&mut SurfaceData>,
    )>,
    render_instance: Res<RenderInstance>,
    render_adapter: Res<RenderAdapter>,
    render_device: Res<RenderDevice>,
) {
    for (entity, mut window, handle, mut maybe_surface_data) in &mut windows {
        let Some(data) = maybe_surface_data.as_mut() else {
            let surface_target = SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(handle.get_display_handle()),
                raw_window_handle: handle.get_window_handle(),
            };
            // SAFETY: The window handles in ExtractedWindows will always be valid objects to create surfaces on
            let surface = unsafe {
                // NOTE: On some OSes this MUST be called from the main thread.
                // As of wgpu 0.15, only fallible if the given window is a HTML canvas and obtaining a WebGPU or WebGL2 context fails.
                render_instance
                    .create_surface_unsafe(surface_target)
                    .expect("Failed to create wgpu surface")
            };
            let caps = surface.get_capabilities(&render_adapter);
            let present_mode = present_mode(&window, &caps);
            let negotiated = negotiate_surface_format(
                &caps.formats,
                &caps.format_capabilities,
                window.display_target.transfer,
                window.display_target.gamut,
            );
            let supported_transfers = supported_transfers(&caps.format_capabilities);

            // Same sRGB-view rule as `SurfaceData::apply_negotiated`.
            let view_format = negotiated.format.add_srgb_suffix();
            let texture_view_format = (negotiated.resolved_transfer == DisplayTransfer::Srgb
                && view_format != negotiated.format)
                .then_some(view_format);
            let configuration = SurfaceConfiguration {
                format: negotiated.format,
                // The color space is the only display parameter the surface
                // carries. The wgpu surface API exposes no HDR10 mastering
                // metadata (SMPTE ST 2086 primaries/luminance, CTA-861.3
                // MaxCLL/MaxFALL), so drivers use their own defaults.
                // `DisplayTarget` keeps `peak_luminance_nits` and
                // `min_luminance_nits` ready to feed it once wgpu exposes it.
                color_space: negotiated.color_space,
                width: window.physical_width,
                height: window.physical_height,
                usage: TextureUsages::RENDER_ATTACHMENT,
                present_mode,
                desired_maximum_frame_latency: window
                    .desired_maximum_frame_latency
                    .map(NonZero::<u32>::get)
                    .unwrap_or(DEFAULT_DESIRED_MAXIMUM_FRAME_LATENCY),
                alpha_mode: match window.alpha_mode {
                    CompositeAlphaMode::Auto => wgpu::CompositeAlphaMode::Auto,
                    CompositeAlphaMode::Opaque => wgpu::CompositeAlphaMode::Opaque,
                    CompositeAlphaMode::PreMultiplied => wgpu::CompositeAlphaMode::PreMultiplied,
                    CompositeAlphaMode::PostMultiplied => wgpu::CompositeAlphaMode::PostMultiplied,
                    CompositeAlphaMode::Inherit => wgpu::CompositeAlphaMode::Inherit,
                },
                view_formats: match texture_view_format {
                    Some(format) => vec![format],
                    None => vec![],
                },
            };

            render_device.configure_surface(&surface, &configuration);

            // The `SurfaceData` insert is deferred to the sync point before
            // `prepare_windows`, so the extracted window is what carries the
            // resolved transfer to this frame's consumers.
            window.resolved_transfer = Some(negotiated.resolved_transfer);
            commands.entity(entity).insert(SurfaceData {
                surface: WgpuWrapper::new(surface),
                configuration,
                texture_view_format,
                resolved_transfer: negotiated.resolved_transfer,
                supported_transfers,
                transfer_before_renegotiation: None,
            });
            continue;
        };

        if window.size_changed
            || window.present_mode_changed
            || window.display_target_transfer_changed
        {
            // normally this is dropped on present but we double check here to be safe as failure to
            // drop it will cause validation errors in wgpu
            drop(window.swap_chain_texture.take());
            #[cfg_attr(
                target_arch = "wasm32",
                expect(clippy::drop_non_drop, reason = "texture views are not drop on wasm")
            )]
            drop(window.swap_chain_texture_view.take());

            data.configuration.width = window.physical_width;
            data.configuration.height = window.physical_height;
            let caps = data.surface.get_capabilities(&render_adapter);
            data.configuration.present_mode = present_mode(&window, &caps);
            // Refresh the supported-transfer set from the fresh capabilities
            // so it tracks the OS HDR toggle adding or removing HDR10.
            data.supported_transfers = supported_transfers(&caps.format_capabilities);
            if window.display_target_transfer_changed {
                // Re-run negotiation with the new requested transfer.
                // `cleanup_view_targets_for_resize` already invalidated the
                // window's `ViewTarget`s this frame, so pipelines specialized
                // on the old output format are not reused.
                data.apply_negotiated(negotiate_surface_format(
                    &caps.formats,
                    &caps.format_capabilities,
                    window.display_target.transfer,
                    window.display_target.gamut,
                ));
            } else {
                // The capabilities are already in hand, so this does not need
                // the color-space gate. `prepare_view_display_targets` runs
                // after `create_surfaces`, so no consumer sees the
                // pre-renegotiation transfer.
                if let Some(previous) = data.renegotiate_if_color_space_lost(
                    &caps,
                    window.display_target.transfer,
                    window.display_target.gamut,
                ) {
                    window.request_display_requery |= previous != data.resolved_transfer;
                }
            }
            render_device.configure_surface(&data.surface, &data.configuration);
        }

        window.resolved_transfer = Some(data.resolved_transfer);
    }
}

fn present_mode(window: &ExtractedWindow, caps: &wgpu::SurfaceCapabilities) -> wgpu::PresentMode {
    let present_mode = match window.present_mode {
        PresentMode::Fifo => wgpu::PresentMode::Fifo,
        PresentMode::FifoRelaxed => wgpu::PresentMode::FifoRelaxed,
        PresentMode::Mailbox => wgpu::PresentMode::Mailbox,
        PresentMode::Immediate => wgpu::PresentMode::Immediate,
        PresentMode::AutoVsync => wgpu::PresentMode::AutoVsync,
        PresentMode::AutoNoVsync => wgpu::PresentMode::AutoNoVsync,
    };
    let fallbacks = match present_mode {
        wgpu::PresentMode::AutoVsync => {
            &[wgpu::PresentMode::FifoRelaxed, wgpu::PresentMode::Fifo][..]
        }
        wgpu::PresentMode::AutoNoVsync => &[
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Fifo,
        ][..],
        wgpu::PresentMode::Mailbox => &[
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Fifo,
        ][..],
        // Always end in FIFO to make sure it's always supported
        x => &[x, wgpu::PresentMode::Fifo][..],
    };
    let new_present_mode = fallbacks
        .iter()
        .copied()
        .find(|fallback| caps.present_modes.contains(fallback))
        .unwrap_or_else(|| {
            unreachable!(
                "Fallback system failed to choose present mode. \
                            This is a bug. Mode: {:?}, Options: {:?}",
                window.present_mode, &caps.present_modes
            );
        });
    if new_present_mode != present_mode && fallbacks.contains(&present_mode) {
        info!("PresentMode {present_mode:?} requested but not available. Falling back to {new_present_mode:?}");
    }
    new_present_mode
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fc(format: TextureFormat, color_spaces: SurfaceColorSpaces) -> SurfaceFormatCapabilities {
        SurfaceFormatCapabilities {
            format,
            color_spaces,
        }
    }

    /// [`negotiate_surface_format`] with the default Rec.709 gamut. Tests that
    /// exercise the gamut call `negotiate_surface_format` directly.
    fn negotiate(
        auto_formats: &[TextureFormat],
        format_capabilities: &[SurfaceFormatCapabilities],
        requested_transfer: DisplayTransfer,
    ) -> NegotiatedSurface {
        negotiate_surface_format(
            auto_formats,
            format_capabilities,
            requested_transfer,
            DisplayGamut::Rec709,
        )
    }

    /// A Metal-like HDR-capable surface: every format also offers Display P3,
    /// `Rgba16Float` offers everything, and `Rgb10a2Unorm` adds HDR10/HLG.
    /// Order matters: the SDR path picks the first 8-bit sRGB format.
    fn metal_like() -> (Vec<TextureFormat>, Vec<SurfaceFormatCapabilities>) {
        (
            vec![
                TextureFormat::Bgra8UnormSrgb,
                TextureFormat::Bgra8Unorm,
                TextureFormat::Rgba16Float,
                TextureFormat::Rgb10a2Unorm,
            ],
            vec![
                fc(
                    TextureFormat::Bgra8UnormSrgb,
                    SurfaceColorSpaces::SRGB | SurfaceColorSpaces::DISPLAY_P3,
                ),
                fc(
                    TextureFormat::Bgra8Unorm,
                    SurfaceColorSpaces::SRGB | SurfaceColorSpaces::DISPLAY_P3,
                ),
                fc(
                    TextureFormat::Rgba16Float,
                    SurfaceColorSpaces::SRGB
                        | SurfaceColorSpaces::DISPLAY_P3
                        | SurfaceColorSpaces::EXTENDED_SRGB_LINEAR
                        | SurfaceColorSpaces::EXTENDED_SRGB
                        | SurfaceColorSpaces::EXTENDED_DISPLAY_P3
                        | SurfaceColorSpaces::BT2100_PQ
                        | SurfaceColorSpaces::BT2100_HLG,
                ),
                fc(
                    TextureFormat::Rgb10a2Unorm,
                    SurfaceColorSpaces::SRGB
                        | SurfaceColorSpaces::DISPLAY_P3
                        | SurfaceColorSpaces::BT2100_PQ
                        | SurfaceColorSpaces::BT2100_HLG,
                ),
            ],
        )
    }

    /// A browser-WebGPU-like surface on an HDR-capable display: the encoded
    /// extended-range sRGB / Display-P3 color spaces on `Rgba16Float` (the web
    /// HDR path), with no `ExtendedSrgbLinear` (the web cannot present a
    /// linear-transfer canvas) and no HDR10.
    fn web_like() -> (Vec<TextureFormat>, Vec<SurfaceFormatCapabilities>) {
        (
            vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba16Float],
            vec![
                fc(
                    TextureFormat::Bgra8UnormSrgb,
                    SurfaceColorSpaces::SRGB | SurfaceColorSpaces::DISPLAY_P3,
                ),
                fc(
                    TextureFormat::Rgba16Float,
                    SurfaceColorSpaces::SRGB
                        | SurfaceColorSpaces::DISPLAY_P3
                        | SurfaceColorSpaces::EXTENDED_SRGB
                        | SurfaceColorSpaces::EXTENDED_DISPLAY_P3,
                ),
            ],
        )
    }

    /// A Vulkan-like surface on an HDR-enabled display: `Rgb10a2Unorm` is
    /// HDR-only (not `Auto`-configurable, so absent from `formats`).
    fn vulkan_hdr_like() -> (Vec<TextureFormat>, Vec<SurfaceFormatCapabilities>) {
        (
            vec![
                TextureFormat::Bgra8UnormSrgb,
                TextureFormat::Bgra8Unorm,
                TextureFormat::Rgba16Float,
            ],
            vec![
                fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
                fc(TextureFormat::Bgra8Unorm, SurfaceColorSpaces::SRGB),
                fc(
                    TextureFormat::Rgba16Float,
                    SurfaceColorSpaces::EXTENDED_SRGB_LINEAR,
                ),
                fc(
                    TextureFormat::Rgb10a2Unorm,
                    SurfaceColorSpaces::BT2100_PQ | SurfaceColorSpaces::BT2100_HLG,
                ),
            ],
        )
    }

    /// A surface with scRGB but no HDR10 (e.g. a backend/OS combination
    /// without PQ support).
    fn scrgb_only() -> (Vec<TextureFormat>, Vec<SurfaceFormatCapabilities>) {
        (
            vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba16Float],
            vec![
                fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
                fc(
                    TextureFormat::Rgba16Float,
                    SurfaceColorSpaces::EXTENDED_SRGB_LINEAR,
                ),
            ],
        )
    }

    /// A surface without any HDR-capable color spaces (e.g. X11/GLES).
    fn sdr_only() -> (Vec<TextureFormat>, Vec<SurfaceFormatCapabilities>) {
        (
            vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Bgra8Unorm],
            vec![
                fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
                fc(TextureFormat::Bgra8Unorm, SurfaceColorSpaces::SRGB),
            ],
        )
    }

    const SDR_SELECTION: fn(TextureFormat) -> NegotiatedSurface = |format| NegotiatedSurface {
        format,
        color_space: SurfaceColorSpace::Auto,
        resolved_transfer: DisplayTransfer::Srgb,
    };

    #[test]
    fn srgb_default_selects_srgb_format_with_auto() {
        let (formats, caps) = metal_like();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Srgb),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
        let (formats, caps) = sdr_only();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Srgb),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
        // No sRGB format offered: fall back to the first supported format.
        assert_eq!(
            negotiate(
                &[TextureFormat::Bgra8Unorm, TextureFormat::Rgba16Float],
                &[],
                DisplayTransfer::Srgb
            ),
            SDR_SELECTION(TextureFormat::Bgra8Unorm)
        );
        // Rgba8UnormSrgb is picked when it is listed before Bgra8UnormSrgb.
        assert_eq!(
            negotiate(
                &[TextureFormat::Rgba8UnormSrgb, TextureFormat::Bgra8UnormSrgb],
                &[],
                DisplayTransfer::Srgb
            ),
            SDR_SELECTION(TextureFormat::Rgba8UnormSrgb)
        );
    }

    #[test]
    fn scrgb_picks_rgba16float_with_extended_srgb_linear() {
        let expected = NegotiatedSurface {
            format: TextureFormat::Rgba16Float,
            color_space: SurfaceColorSpace::ExtendedSrgbLinear,
            resolved_transfer: DisplayTransfer::ScRgbLinear,
        };
        let (formats, caps) = metal_like();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::ScRgbLinear),
            expected
        );
        let (formats, caps) = vulkan_hdr_like();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::ScRgbLinear),
            expected
        );
    }

    #[test]
    fn scrgb_requires_the_color_space_not_just_the_format() {
        // `Rgba16Float` is offered, but only in the sRGB color space, where
        // linear scRGB values would display incorrectly.
        let formats = vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba16Float];
        let caps = vec![
            fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
            fc(TextureFormat::Rgba16Float, SurfaceColorSpaces::SRGB),
        ];
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::ScRgbLinear),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn pq_negotiates_hdr10_preferring_rgb10a2unorm() {
        let expected = NegotiatedSurface {
            format: TextureFormat::Rgb10a2Unorm,
            color_space: SurfaceColorSpace::Bt2100Pq,
            resolved_transfer: DisplayTransfer::Pq,
        };
        // Vulkan-like: Rgb10a2Unorm is the only HDR10 format.
        let (formats, caps) = vulkan_hdr_like();
        assert_eq!(negotiate(&formats, &caps, DisplayTransfer::Pq), expected);
        // Metal-like: Rgba16Float and Rgb10a2Unorm both advertise HDR10, and
        // Rgb10a2Unorm is preferred even though Rgba16Float is listed first.
        let (formats, caps) = metal_like();
        assert_eq!(negotiate(&formats, &caps, DisplayTransfer::Pq), expected);
    }

    #[test]
    fn pq_uses_rgba16float_when_it_is_the_only_hdr10_format() {
        let formats = vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba16Float];
        let caps = vec![
            fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
            fc(
                TextureFormat::Rgba16Float,
                SurfaceColorSpaces::EXTENDED_SRGB_LINEAR | SurfaceColorSpaces::BT2100_PQ,
            ),
        ];
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Pq),
            NegotiatedSurface {
                format: TextureFormat::Rgba16Float,
                color_space: SurfaceColorSpace::Bt2100Pq,
                resolved_transfer: DisplayTransfer::Pq,
            }
        );
    }

    #[test]
    fn pq_takes_any_hdr10_format_as_a_last_resort() {
        // A driver advertising HDR10 on an 8-bit format only.
        let formats = vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Bgra8Unorm];
        let caps = vec![
            fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
            fc(
                TextureFormat::Bgra8Unorm,
                SurfaceColorSpaces::SRGB | SurfaceColorSpaces::BT2100_PQ,
            ),
        ];
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Pq),
            NegotiatedSurface {
                format: TextureFormat::Bgra8Unorm,
                color_space: SurfaceColorSpace::Bt2100Pq,
                resolved_transfer: DisplayTransfer::Pq,
            }
        );
    }

    #[test]
    fn pq_downgrades_through_scrgb_to_sdr() {
        // Downgrade chain: PQ -> scRGB-linear when HDR10 is unavailable...
        let (formats, caps) = scrgb_only();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Pq),
            NegotiatedSurface {
                format: TextureFormat::Rgba16Float,
                color_space: SurfaceColorSpace::ExtendedSrgbLinear,
                resolved_transfer: DisplayTransfer::ScRgbLinear,
            }
        );
        // ...and all the way to SDR sRGB when scRGB is unavailable too.
        let (formats, caps) = sdr_only();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Pq),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn empty_auto_formats_fall_back_to_an_explicit_color_space() {
        // A driver in OS HDR mode reporting formats only in explicit-opt-in
        // color spaces. Configuring with `Auto` would fail validation.
        let caps = vec![fc(
            TextureFormat::Rgb10a2Unorm,
            SurfaceColorSpaces::BT2100_PQ,
        )];
        assert_eq!(
            negotiate(&[], &caps, DisplayTransfer::Srgb),
            NegotiatedSurface {
                format: TextureFormat::Rgb10a2Unorm,
                color_space: SurfaceColorSpace::Bt2100Pq,
                resolved_transfer: DisplayTransfer::Pq,
            }
        );
        // An explicitly-advertised sRGB pair is preferred when present.
        let caps = vec![
            fc(TextureFormat::Rgb10a2Unorm, SurfaceColorSpaces::BT2100_PQ),
            fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
        ];
        assert_eq!(
            negotiate(&[], &caps, DisplayTransfer::Srgb),
            NegotiatedSurface {
                format: TextureFormat::Bgra8UnormSrgb,
                color_space: SurfaceColorSpace::Srgb,
                resolved_transfer: DisplayTransfer::Srgb,
            }
        );
    }

    #[test]
    fn extended_srgb_rec709_negotiates_extended_srgb() {
        let expected = NegotiatedSurface {
            format: TextureFormat::Rgba16Float,
            color_space: SurfaceColorSpace::ExtendedSrgb,
            resolved_transfer: DisplayTransfer::ExtendedSrgb,
        };
        // The web HDR path: an `Rgba16Float` `ExtendedSrgb` swapchain.
        let (formats, caps) = web_like();
        assert_eq!(
            negotiate_surface_format(
                &formats,
                &caps,
                DisplayTransfer::ExtendedSrgb,
                DisplayGamut::Rec709
            ),
            expected
        );
        // Also advertised on a Metal-like native surface.
        let (formats, caps) = metal_like();
        assert_eq!(
            negotiate_surface_format(
                &formats,
                &caps,
                DisplayTransfer::ExtendedSrgb,
                DisplayGamut::Rec709
            ),
            expected
        );
    }

    #[test]
    fn extended_srgb_displayp3_negotiates_extended_display_p3() {
        // The resolved transfer is still `ExtendedSrgb`: the gamut rides
        // `DisplayTarget::gamut`, not the transfer.
        let expected = NegotiatedSurface {
            format: TextureFormat::Rgba16Float,
            color_space: SurfaceColorSpace::ExtendedDisplayP3,
            resolved_transfer: DisplayTransfer::ExtendedSrgb,
        };
        let (formats, caps) = web_like();
        assert_eq!(
            negotiate_surface_format(
                &formats,
                &caps,
                DisplayTransfer::ExtendedSrgb,
                DisplayGamut::DisplayP3
            ),
            expected
        );
    }

    #[test]
    fn extended_srgb_displayp3_without_p3_support_downgrades_straight_to_sdr() {
        let formats = vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba16Float];
        let caps = vec![
            fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
            fc(
                TextureFormat::Rgba16Float,
                SurfaceColorSpaces::EXTENDED_SRGB,
            ),
        ];
        assert_eq!(
            negotiate_surface_format(
                &formats,
                &caps,
                DisplayTransfer::ExtendedSrgb,
                DisplayGamut::DisplayP3
            ),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn extended_srgb_without_support_downgrades_to_sdr() {
        let (formats, caps) = sdr_only();
        assert_eq!(
            negotiate_surface_format(
                &formats,
                &caps,
                DisplayTransfer::ExtendedSrgb,
                DisplayGamut::Rec709
            ),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn scrgb_linear_does_not_fall_back_to_extended_srgb() {
        // scRGB-linear is native only, and a web-like surface advertises the
        // encoded `ExtendedSrgb` but not `ExtendedSrgbLinear`. Apps target the
        // web HDR path by requesting `ExtendedSrgb`.
        let (formats, caps) = web_like();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::ScRgbLinear),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn pq_does_not_fall_back_to_extended_srgb() {
        // PQ's chain is PQ -> scRGB-linear -> SDR; it never resolves to the
        // encoded extended-sRGB transfer. A web-like surface has neither step.
        let (formats, caps) = web_like();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Pq),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn empty_auto_formats_fall_back_to_extended_srgb_spaces() {
        // A driver reporting only the encoded extended-sRGB space, with no
        // Auto-configurable format.
        let caps = vec![fc(
            TextureFormat::Rgba16Float,
            SurfaceColorSpaces::EXTENDED_SRGB,
        )];
        assert_eq!(
            negotiate_surface_format(&[], &caps, DisplayTransfer::Srgb, DisplayGamut::Rec709),
            NegotiatedSurface {
                format: TextureFormat::Rgba16Float,
                color_space: SurfaceColorSpace::ExtendedSrgb,
                resolved_transfer: DisplayTransfer::ExtendedSrgb,
            }
        );
        let caps = vec![fc(
            TextureFormat::Rgba16Float,
            SurfaceColorSpaces::EXTENDED_DISPLAY_P3,
        )];
        assert_eq!(
            negotiate_surface_format(&[], &caps, DisplayTransfer::Srgb, DisplayGamut::DisplayP3),
            NegotiatedSurface {
                format: TextureFormat::Rgba16Float,
                color_space: SurfaceColorSpace::ExtendedDisplayP3,
                resolved_transfer: DisplayTransfer::ExtendedSrgb,
            }
        );
        // A non-P3 request must not take the extended-P3 fallback. With no
        // other drivable space it would panic, so this surface also offers
        // SRGB, which the fallback table tries first.
        let caps = vec![
            fc(
                TextureFormat::Rgba16Float,
                SurfaceColorSpaces::EXTENDED_DISPLAY_P3,
            ),
            fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
        ];
        assert_eq!(
            negotiate_surface_format(&[], &caps, DisplayTransfer::Srgb, DisplayGamut::Rec709),
            NegotiatedSurface {
                format: TextureFormat::Bgra8UnormSrgb,
                color_space: SurfaceColorSpace::Srgb,
                resolved_transfer: DisplayTransfer::Srgb,
            }
        );
    }

    #[test]
    fn supported_transfers_matches_negotiable_set() {
        use DisplayTransfer::{ExtendedSrgb, Pq, ScRgbLinear, Srgb};

        // Each surface advertises exactly the transfers its negotiation helpers
        // can satisfy, with `Srgb` always present. The literals are also the
        // cycle order `DisplayTransfers::iter` owes: a bit-index walk would
        // swap `ExtendedSrgb` and `Pq` in `metal_like`.
        let cases: [(
            &str,
            fn() -> (Vec<TextureFormat>, Vec<SurfaceFormatCapabilities>),
            Vec<DisplayTransfer>,
        ); 5] = [
            // Metal advertises every color space: the full cycle.
            (
                "metal_like",
                metal_like,
                vec![Srgb, ScRgbLinear, ExtendedSrgb, Pq],
            ),
            // The web HDR path is the encoded extended-sRGB transfer; no
            // linear-transfer canvas (scRGB-linear) and no HDR10.
            ("web_like", web_like, vec![Srgb, ExtendedSrgb]),
            // The reported Windows+NVIDIA Vulkan case: scRGB-linear and HDR10
            // but no encoded extended-sRGB, so the cycle skips `ExtendedSrgb`.
            (
                "vulkan_hdr_like",
                vulkan_hdr_like,
                vec![Srgb, ScRgbLinear, Pq],
            ),
            ("scrgb_only", scrgb_only, vec![Srgb, ScRgbLinear]),
            ("sdr_only", sdr_only, vec![Srgb]),
        ];

        for (name, fixture, expected) in cases {
            let (_, caps) = fixture();
            assert_eq!(
                supported_transfers(&caps).iter().collect::<Vec<_>>(),
                expected,
                "{name}"
            );
        }
    }
}
