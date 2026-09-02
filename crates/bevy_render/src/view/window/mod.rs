use crate::renderer::wgpu_wrapper;
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
    SurfaceColorSpace, SurfaceColorSpaces, SurfaceConfiguration, SurfaceTargetUnsafe,
    TextureFormat, TextureUsages, TextureViewDescriptor,
};

mod display_state;
pub mod screenshot;

pub use display_state::resolve_calibration;
use display_state::{poll_display_state, write_back_display_state, DisplayStateStore};
use screenshot::ScreenshotPlugin;

pub struct WindowRenderPlugin;

impl Plugin for WindowRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ScreenshotPlugin)
            .add_systems(PostUpdate, resolve_calibration);

        // We need to sync the window entity in the render world
        // We can't use [`SyncComponentPlugin`] because it would introduce `bevy_render` as
        // a dependency to `bevy_window`
        {
            app.add_observer(|trigger: On<Add<Window>>, mut commands: Commands| {
                commands
                    .entity(trigger.entity)
                    .insert(SyncToRenderWorld::default());
            });

            // The primary window gets added before this plugin so we can't rely on the observer
            let _ = app.world_mut().run_system_once(
                |mut commands: Commands, windows: Query<Entity, With<Window>>| {
                    for entity in &windows {
                        commands.entity(entity).insert(SyncToRenderWorld::default());
                    }
                },
            );
        }

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
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
                .add_systems(
                    Render,
                    (prepare_windows, poll_display_state)
                        .chain()
                        .in_set(RenderSystems::PrepareViews),
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
    /// Under a negotiated HDR transfer it equals the surface format, with no
    /// sRGB view.
    pub swap_chain_texture_view_format: Option<TextureFormat>,
    pub size_changed: bool,
    pub present_mode_changed: bool,
    pub alpha_mode: CompositeAlphaMode,
    /// The window's [`EffectiveDisplayTarget::target`].
    pub display_target: DisplayTarget,
    /// Whether the surface must be reconfigured for a [`DisplayTarget`] change.
    /// The surface color space depends on [`DisplayTarget::transfer`], and under
    /// [`DisplayTransfer::ExtendedSrgb`] also on [`DisplayTarget::gamut`].
    pub display_target_transfer_changed: bool,
    /// The [`DisplayTransfer`] the configured surface uses. `None` until
    /// [`create_surfaces`] has configured the surface.
    pub resolved_transfer: Option<DisplayTransfer>,
    /// Set for one frame when the display behind the window may have changed,
    /// after a window move, a focus regain, a monitor change, or a surface
    /// renegotiation. `poll_display_state` reads the display state again in
    /// response.
    ///
    /// A focus regain counts because the user may have changed the display's
    /// brightness setting while the window did not have focus.
    pub request_display_requery: bool,
    /// Whether the window's [`DisplayCalibrationPolicy`] enables any field.
    ///
    /// When it is `false`, `poll_display_state` skips the per-frame read on
    /// macOS, since the result could not change the window's
    /// [`EffectiveDisplayTarget`].
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
        // `add_srgb_suffix` would give an sRGB view to a non-sRGB format an HDR
        // negotiation chose.
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
    mut moved: Extract<MessageReader<WindowMoved>>,
    mut focused: Extract<MessageReader<WindowFocused>>,
    changed_monitor: Extract<Query<Entity, Changed<OnMonitor>>>,
    mut removed_monitor: Extract<RemovedComponents<OnMonitor>>,
    mut removed: Extract<RemovedComponents<RawHandleWrapper>>,
    mut removed_primary: Extract<RemovedComponents<PrimaryWindow>>,
    mapper: Extract<Query<&RenderEntity>>,
) {
    let display_requery: EntityHashSet = moved
        .read()
        .map(|moved| moved.window)
        .chain(focused.read().filter(|f| f.focused).map(|f| f.window))
        .chain(changed_monitor.iter())
        .chain(removed_monitor.read())
        .collect();

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
        // A required component can still be removed. Fall back to the default
        // rather than drop the window.
        let display_target = effective_display_target
            .map(|effective| effective.target)
            .unwrap_or_default();
        let request_display_requery = display_requery.contains(&entity);
        let display_calibration_auto =
            calibration_policy.is_some_and(DisplayCalibrationPolicy::has_auto);
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
                    request_display_requery,
                    display_calibration_auto,
                    resolved_transfer: None,
                    needs_initial_present: true,
                },
                handle.clone(),
            ));
            continue;
        };

        let previous = extracted_window.display_target;
        let transfer_changed = previous.transfer != display_target.transfer;
        let extended_srgb_gamut_changed = (previous.transfer == DisplayTransfer::ExtendedSrgb
            || display_target.transfer == DisplayTransfer::ExtendedSrgb)
            && previous.gamut != display_target.gamut;
        extracted_window.display_target_transfer_changed =
            transfer_changed || extended_srgb_gamut_changed;
        extracted_window.display_target = display_target;

        extracted_window.request_display_requery = request_display_requery;
        extracted_window.display_calibration_auto = display_calibration_auto;

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

/// Inserts `value` only if it differs from the current component, so change
/// detection fires only on real changes. Does nothing if `entity` is gone.
fn insert_on_change<C: Component + PartialEq>(
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

// TODO: what lifetime should this be?
wgpu_wrapper!(struct WgpuSurface(wgpu::Surface<'static>));

#[derive(Component)]
pub struct SurfaceData {
    surface: WgpuSurface,
    configuration: SurfaceConfiguration,
    texture_view_format: Option<TextureFormat>,
    /// The [`DisplayTransfer`] the surface is configured for.
    resolved_transfer: DisplayTransfer,
    supported_transfers: DisplayTransfers,
}

impl SurfaceData {
    fn apply_negotiated(&mut self, negotiated: NegotiatedSurface) {
        self.configuration.format = negotiated.format;
        self.configuration.color_space = negotiated.color_space;
        self.texture_view_format = negotiated.texture_view_format();
        self.configuration.view_formats = Vec::from_iter(self.texture_view_format);
        self.resolved_transfer = negotiated.resolved_transfer;
    }

    /// Renegotiates the surface if `caps` no longer lists its color space.
    ///
    /// The capabilities can stop listing an explicit color space at runtime,
    /// for example when the OS HDR setting changes. Configuring with it again
    /// fails wgpu validation with `ConfigureSurfaceError::UnsupportedColorSpace`.
    fn renegotiate_if_color_space_lost(
        &mut self,
        caps: &wgpu::SurfaceCapabilities,
        requested_transfer: DisplayTransfer,
        requested_gamut: DisplayGamut,
    ) -> bool {
        let Some(flag) = self.configuration.color_space.to_color_spaces() else {
            return false;
        };
        if caps.color_spaces(self.configuration.format).contains(flag) {
            return false;
        }
        warn_once!(
            "Surface color space {:?} is no longer supported for {:?}. The OS HDR \
            setting may have changed. Renegotiating the surface.",
            self.configuration.color_space,
            self.configuration.format
        );
        self.apply_negotiated(negotiate_surface_format(
            caps,
            requested_transfer,
            requested_gamut,
        ));
        true
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

        let Some(mut surface_data) = maybe_surface_data else {
            continue;
        };

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
                // wgpu reports `Outdated` when the underlying surface changed, which
                // includes its color spaces.
                if surface_data
                    .configuration
                    .color_space
                    .to_color_spaces()
                    .is_some()
                {
                    let caps = surface_data.surface.get_capabilities(&render_adapter);
                    if surface_data.renegotiate_if_color_space_lost(
                        &caps,
                        window.display_target.transfer,
                        window.display_target.gamut,
                    ) {
                        window.resolved_transfer = Some(surface_data.resolved_transfer);
                        window.request_display_requery = true;
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

/// The format and color space [`negotiate_surface_format`] chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NegotiatedSurface {
    format: TextureFormat,
    /// [`SurfaceColorSpace::Auto`], or a color space the surface supports for
    /// `format`. Anything else fails wgpu validation.
    color_space: SurfaceColorSpace,
    resolved_transfer: DisplayTransfer,
}

impl NegotiatedSurface {
    /// The sRGB view format to render through, or `None` to use the texture as
    /// is.
    ///
    /// Only the sRGB transfer gets an sRGB view. An HDR negotiation can choose
    /// an 8-bit format. Values written to it must already be encoded for its
    /// color space, and an sRGB view would encode them again.
    fn texture_view_format(&self) -> Option<TextureFormat> {
        let view_format = self.format.add_srgb_suffix();
        (self.resolved_transfer == DisplayTransfer::Srgb && view_format != self.format)
            .then_some(view_format)
    }
}

/// Returns the first of `preferred` that supports `flag`, else the first
/// non-sRGB format in `caps` that does.
///
/// Values in an HDR color space are already encoded for it, so the format
/// must not add sRGB encoding on write.
fn first_format_in(
    caps: &wgpu::SurfaceCapabilities,
    flag: SurfaceColorSpaces,
    preferred: &[TextureFormat],
) -> Option<TextureFormat> {
    let preferred = preferred
        .iter()
        .copied()
        .filter(|&format| caps.color_spaces(format).contains(flag));
    let any = caps
        .format_capabilities
        .iter()
        .filter(|fc| fc.color_spaces.contains(flag) && !fc.format.is_srgb())
        .map(|fc| fc.format);
    preferred.chain(any).next()
}

/// Negotiates an `Rgba16Float` surface in
/// [`SurfaceColorSpace::ExtendedSrgbLinear`], linear scRGB, if the surface
/// supports the pair.
///
/// Linear values need float precision, so only `Rgba16Float` is tried.
fn negotiate_scrgb_linear(caps: &wgpu::SurfaceCapabilities) -> Option<NegotiatedSurface> {
    caps.color_spaces(TextureFormat::Rgba16Float)
        .contains(SurfaceColorSpaces::EXTENDED_SRGB_LINEAR)
        .then_some(NegotiatedSurface {
            format: TextureFormat::Rgba16Float,
            color_space: SurfaceColorSpace::ExtendedSrgbLinear,
            resolved_transfer: DisplayTransfer::ScRgbLinear,
        })
}

/// Negotiates a surface in [`SurfaceColorSpace::Bt2100Pq`], the HDR10 color
/// space, if the surface supports it for any format.
///
/// `Rgb10a2Unorm` is preferred, then `Rgba16Float`, then any other format
/// that supports it. wgpu documents `Rgb10a2Unorm` as the typical format for
/// this color space.
fn negotiate_hdr10(caps: &wgpu::SurfaceCapabilities) -> Option<NegotiatedSurface> {
    first_format_in(
        caps,
        SurfaceColorSpaces::BT2100_PQ,
        &[TextureFormat::Rgb10a2Unorm, TextureFormat::Rgba16Float],
    )
    .map(|format| NegotiatedSurface {
        format,
        color_space: SurfaceColorSpace::Bt2100Pq,
        resolved_transfer: DisplayTransfer::Pq,
    })
}

/// Negotiates a surface for [`DisplayTransfer::ExtendedSrgb`], if the surface
/// supports the color space `gamut` needs.
///
/// A `DisplayP3` gamut needs [`SurfaceColorSpace::ExtendedDisplayP3`]. Every
/// other gamut uses [`SurfaceColorSpace::ExtendedSrgb`]. Both resolve to
/// [`DisplayTransfer::ExtendedSrgb`], and the gamut stays in
/// [`DisplayTarget::gamut`]. `Rgba16Float` is preferred.
fn negotiate_extended_srgb(
    caps: &wgpu::SurfaceCapabilities,
    gamut: DisplayGamut,
) -> Option<NegotiatedSurface> {
    let (flag, color_space) = match gamut {
        DisplayGamut::DisplayP3 => (
            SurfaceColorSpaces::EXTENDED_DISPLAY_P3,
            SurfaceColorSpace::ExtendedDisplayP3,
        ),
        _ => (
            SurfaceColorSpaces::EXTENDED_SRGB,
            SurfaceColorSpace::ExtendedSrgb,
        ),
    };
    first_format_in(caps, flag, &[TextureFormat::Rgba16Float]).map(|format| NegotiatedSurface {
        format,
        color_space,
        resolved_transfer: DisplayTransfer::ExtendedSrgb,
    })
}

/// Returns the [`DisplayTransfer`]s a surface with these capabilities can
/// provide. [`DisplayTransfer::Srgb`] is always included. A listed transfer can
/// still be downgraded when the surface lacks the requested gamut for it.
fn supported_transfers(caps: &wgpu::SurfaceCapabilities) -> DisplayTransfers {
    let mut transfers = DisplayTransfers::EMPTY.with(DisplayTransfer::Srgb);
    if negotiate_scrgb_linear(caps).is_some() {
        transfers = transfers.with(DisplayTransfer::ScRgbLinear);
    }
    if negotiate_extended_srgb(caps, DisplayGamut::Rec709).is_some()
        || negotiate_extended_srgb(caps, DisplayGamut::DisplayP3).is_some()
    {
        transfers = transfers.with(DisplayTransfer::ExtendedSrgb);
    }
    if negotiate_hdr10(caps).is_some() {
        transfers = transfers.with(DisplayTransfer::Pq);
    }
    transfers
}

/// Chooses the format and color space for a window surface from the
/// requested [`DisplayTransfer`], downgrading when the surface cannot
/// provide it. Each downgrade logs a warning.
///
/// - [`DisplayTransfer::Pq`] falls back to linear scRGB, then to SDR.
/// - [`DisplayTransfer::ExtendedSrgb`] with a `DisplayP3` gamut falls back
///   to SDR, never to [`SurfaceColorSpace::ExtendedSrgb`].
/// - Every other unmet request falls back to SDR.
/// - If the surface lists no format for [`SurfaceColorSpace::Auto`], an HDR
///   request takes the first explicit color space that fits `requested_gamut`.
///
/// The SDR path uses [`SurfaceColorSpace::Auto`], which keeps wgpu's own
/// color space choice.
///
/// # Panics
///
/// Panics if the surface offers no format the request can use.
fn negotiate_surface_format(
    caps: &wgpu::SurfaceCapabilities,
    requested_transfer: DisplayTransfer,
    requested_gamut: DisplayGamut,
) -> NegotiatedSurface {
    match requested_transfer {
        DisplayTransfer::Srgb => {}
        DisplayTransfer::ScRgbLinear => {
            if let Some(negotiated) = negotiate_scrgb_linear(caps) {
                return negotiated;
            }
            warn_once!(
                "DisplayTransfer::ScRgbLinear was requested, but this surface does not \
                support Rgba16Float in the linear scRGB color space. Downgrading to SDR \
                sRGB. On the web, request DisplayTransfer::ExtendedSrgb instead."
            );
        }
        DisplayTransfer::ExtendedSrgb => {
            if let Some(negotiated) = negotiate_extended_srgb(caps, requested_gamut) {
                return negotiated;
            }
            warn_once!(
                "DisplayTransfer::ExtendedSrgb was requested with the {requested_gamut:?} \
                gamut, but this surface does not support an extended sRGB color space for \
                it. Downgrading to SDR sRGB."
            );
        }
        DisplayTransfer::Pq => {
            if let Some(negotiated) = negotiate_hdr10(caps) {
                return negotiated;
            }
            warn_once!(
                "DisplayTransfer::Pq was requested, but this surface does not support \
                the HDR10 color space. The OS HDR setting may be off, or the backend may \
                not support it. Downgrading to linear scRGB if available, else SDR sRGB."
            );
            if let Some(negotiated) = negotiate_scrgb_linear(caps) {
                return negotiated;
            }
            warn_once!(
                "DisplayTransfer::Pq could not fall back to linear scRGB either. \
                Downgrading to SDR sRGB."
            );
        }
    }

    // SDR path: prefer sRGB formats for surfaces, but fall back to the first
    // available format if no sRGB formats are available.
    if let Some(first) = caps.formats.first() {
        let mut format = *first;
        for available_format in &caps.formats {
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

    if !requested_transfer.is_hdr() {
        panic!("No supported formats for surface");
    }

    // The request's own negotiation failed, and the surface lists formats only
    // in explicit color spaces, which wgpu documents some drivers do in OS HDR
    // mode. Configuring those with `Auto` fails validation, so take the first
    // pair the surface supports.
    for (flag, color_space, resolved_transfer) in [
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
        if let Some(format) = first_format_in(caps, flag, &[]) {
            warn_once!(
                "This surface has no format that works with the default color space. \
                Using {format:?} in the {color_space:?} color space, resolved as \
                {resolved_transfer:?}."
            );
            return NegotiatedSurface {
                format,
                color_space,
                resolved_transfer,
            };
        }
    }

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
                &caps,
                window.display_target.transfer,
                window.display_target.gamut,
            );
            let supported_transfers = supported_transfers(&caps);
            let texture_view_format = negotiated.texture_view_format();
            let configuration = SurfaceConfiguration {
                format: negotiated.format,
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
                view_formats: Vec::from_iter(texture_view_format),
            };

            render_device.configure_surface(&surface, &configuration);

            // `SurfaceData` is inserted through commands, so systems later this
            // frame read the transfer from the window.
            window.resolved_transfer = Some(negotiated.resolved_transfer);
            commands.entity(entity).insert(SurfaceData {
                surface: WgpuSurface::new(surface),
                configuration,
                texture_view_format,
                resolved_transfer: negotiated.resolved_transfer,
                supported_transfers,
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
            // Refreshed on every reconfigure, since the OS HDR setting can change.
            data.supported_transfers = supported_transfers(&caps);
            if window.display_target_transfer_changed {
                data.apply_negotiated(negotiate_surface_format(
                    &caps,
                    window.display_target.transfer,
                    window.display_target.gamut,
                ));
            } else {
                window.request_display_requery |= data.renegotiate_if_color_space_lost(
                    &caps,
                    window.display_target.transfer,
                    window.display_target.gamut,
                );
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
    use wgpu::SurfaceFormatCapabilities;

    fn fc(format: TextureFormat, color_spaces: SurfaceColorSpaces) -> SurfaceFormatCapabilities {
        SurfaceFormatCapabilities {
            format,
            color_spaces,
        }
    }

    fn caps(
        formats: Vec<TextureFormat>,
        format_capabilities: Vec<SurfaceFormatCapabilities>,
    ) -> wgpu::SurfaceCapabilities {
        wgpu::SurfaceCapabilities {
            formats,
            format_capabilities,
            ..Default::default()
        }
    }

    /// [`negotiate_surface_format`] with the `Rec709` gamut.
    fn negotiate(
        caps: &wgpu::SurfaceCapabilities,
        requested_transfer: DisplayTransfer,
    ) -> NegotiatedSurface {
        negotiate_surface_format(caps, requested_transfer, DisplayGamut::Rec709)
    }

    /// A Metal-like HDR-capable surface.
    fn metal_like() -> wgpu::SurfaceCapabilities {
        caps(
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

    /// A browser WebGPU-like surface on an HDR display. `Rgba16Float` supports
    /// the encoded extended sRGB and Display P3 color spaces, but no linear
    /// scRGB and no HDR10.
    fn web_like() -> wgpu::SurfaceCapabilities {
        caps(
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

    /// A Vulkan-like surface on an HDR display. `Rgb10a2Unorm` supports only
    /// HDR color spaces, so it is absent from `formats`.
    fn vulkan_hdr_like() -> wgpu::SurfaceCapabilities {
        caps(
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

    /// A surface with scRGB but no HDR10.
    fn scrgb_only() -> wgpu::SurfaceCapabilities {
        caps(
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

    /// A surface with no HDR color spaces.
    fn sdr_only() -> wgpu::SurfaceCapabilities {
        caps(
            vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Bgra8Unorm],
            vec![
                fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
                fc(TextureFormat::Bgra8Unorm, SurfaceColorSpaces::SRGB),
            ],
        )
    }

    fn sdr_selection(format: TextureFormat) -> NegotiatedSurface {
        NegotiatedSurface {
            format,
            color_space: SurfaceColorSpace::Auto,
            resolved_transfer: DisplayTransfer::Srgb,
        }
    }

    #[test]
    fn srgb_default_selects_srgb_format_with_auto() {
        assert_eq!(
            negotiate(&metal_like(), DisplayTransfer::Srgb),
            sdr_selection(TextureFormat::Bgra8UnormSrgb)
        );
        assert_eq!(
            negotiate(&sdr_only(), DisplayTransfer::Srgb),
            sdr_selection(TextureFormat::Bgra8UnormSrgb)
        );
        assert_eq!(
            negotiate(
                &caps(
                    vec![TextureFormat::Bgra8Unorm, TextureFormat::Rgba16Float],
                    vec![]
                ),
                DisplayTransfer::Srgb
            ),
            sdr_selection(TextureFormat::Bgra8Unorm)
        );
        assert_eq!(
            negotiate(
                &caps(
                    vec![TextureFormat::Rgba8UnormSrgb, TextureFormat::Bgra8UnormSrgb],
                    vec![]
                ),
                DisplayTransfer::Srgb
            ),
            sdr_selection(TextureFormat::Rgba8UnormSrgb)
        );
    }

    #[test]
    fn scrgb_picks_rgba16float_with_extended_srgb_linear() {
        let expected = NegotiatedSurface {
            format: TextureFormat::Rgba16Float,
            color_space: SurfaceColorSpace::ExtendedSrgbLinear,
            resolved_transfer: DisplayTransfer::ScRgbLinear,
        };
        assert_eq!(
            negotiate(&metal_like(), DisplayTransfer::ScRgbLinear),
            expected
        );
        assert_eq!(
            negotiate(&vulkan_hdr_like(), DisplayTransfer::ScRgbLinear),
            expected
        );
    }

    #[test]
    fn scrgb_requires_the_color_space_not_just_the_format() {
        // `Rgba16Float` is listed, but only in the sRGB color space, where
        // linear scRGB values would display incorrectly.
        let caps = caps(
            vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba16Float],
            vec![
                fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
                fc(TextureFormat::Rgba16Float, SurfaceColorSpaces::SRGB),
            ],
        );
        assert_eq!(
            negotiate(&caps, DisplayTransfer::ScRgbLinear),
            sdr_selection(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn pq_negotiates_hdr10_preferring_rgb10a2unorm() {
        let expected = NegotiatedSurface {
            format: TextureFormat::Rgb10a2Unorm,
            color_space: SurfaceColorSpace::Bt2100Pq,
            resolved_transfer: DisplayTransfer::Pq,
        };
        assert_eq!(negotiate(&vulkan_hdr_like(), DisplayTransfer::Pq), expected);
        // Metal-like: both formats advertise HDR10, and Rgb10a2Unorm is chosen
        // even though Rgba16Float is listed first.
        assert_eq!(negotiate(&metal_like(), DisplayTransfer::Pq), expected);
    }

    #[test]
    fn pq_uses_rgba16float_when_it_is_the_only_hdr10_format() {
        let caps = caps(
            vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba16Float],
            vec![
                fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
                fc(
                    TextureFormat::Rgba16Float,
                    SurfaceColorSpaces::EXTENDED_SRGB_LINEAR | SurfaceColorSpaces::BT2100_PQ,
                ),
            ],
        );
        assert_eq!(
            negotiate(&caps, DisplayTransfer::Pq),
            NegotiatedSurface {
                format: TextureFormat::Rgba16Float,
                color_space: SurfaceColorSpace::Bt2100Pq,
                resolved_transfer: DisplayTransfer::Pq,
            }
        );
    }

    #[test]
    fn pq_takes_any_hdr10_format_when_preferred_ones_are_missing() {
        // A driver advertising HDR10 on an 8-bit format only.
        let caps = caps(
            vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Bgra8Unorm],
            vec![
                fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
                fc(
                    TextureFormat::Bgra8Unorm,
                    SurfaceColorSpaces::SRGB | SurfaceColorSpaces::BT2100_PQ,
                ),
            ],
        );
        assert_eq!(
            negotiate(&caps, DisplayTransfer::Pq),
            NegotiatedSurface {
                format: TextureFormat::Bgra8Unorm,
                color_space: SurfaceColorSpace::Bt2100Pq,
                resolved_transfer: DisplayTransfer::Pq,
            }
        );
    }

    #[test]
    fn pq_downgrades_through_scrgb_to_sdr() {
        assert_eq!(
            negotiate(&scrgb_only(), DisplayTransfer::Pq),
            NegotiatedSurface {
                format: TextureFormat::Rgba16Float,
                color_space: SurfaceColorSpace::ExtendedSrgbLinear,
                resolved_transfer: DisplayTransfer::ScRgbLinear,
            }
        );
        assert_eq!(
            negotiate(&sdr_only(), DisplayTransfer::Pq),
            sdr_selection(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn empty_auto_formats_fall_back_to_an_explicit_color_space() {
        // A driver in OS HDR mode that lists formats only in explicit color
        // spaces. Each request's own negotiation fails first.
        let pq_only = caps(
            vec![],
            vec![fc(
                TextureFormat::Rgb10a2Unorm,
                SurfaceColorSpaces::BT2100_PQ,
            )],
        );
        let expected = NegotiatedSurface {
            format: TextureFormat::Rgb10a2Unorm,
            color_space: SurfaceColorSpace::Bt2100Pq,
            resolved_transfer: DisplayTransfer::Pq,
        };
        assert_eq!(negotiate(&pq_only, DisplayTransfer::ScRgbLinear), expected);
        assert_eq!(
            negotiate_surface_format(
                &pq_only,
                DisplayTransfer::ExtendedSrgb,
                DisplayGamut::Rec709
            ),
            expected
        );
    }

    #[test]
    #[should_panic(expected = "No supported formats for surface")]
    fn empty_auto_formats_panic_for_an_srgb_request() {
        let pq_only = caps(
            vec![],
            vec![fc(
                TextureFormat::Rgb10a2Unorm,
                SurfaceColorSpaces::BT2100_PQ,
            )],
        );
        negotiate(&pq_only, DisplayTransfer::Srgb);
    }

    #[test]
    fn extended_srgb_rec709_negotiates_extended_srgb() {
        let expected = NegotiatedSurface {
            format: TextureFormat::Rgba16Float,
            color_space: SurfaceColorSpace::ExtendedSrgb,
            resolved_transfer: DisplayTransfer::ExtendedSrgb,
        };
        assert_eq!(
            negotiate_surface_format(
                &web_like(),
                DisplayTransfer::ExtendedSrgb,
                DisplayGamut::Rec709
            ),
            expected
        );
        assert_eq!(
            negotiate_surface_format(
                &metal_like(),
                DisplayTransfer::ExtendedSrgb,
                DisplayGamut::Rec709
            ),
            expected
        );
    }

    #[test]
    fn extended_srgb_displayp3_negotiates_extended_display_p3() {
        // The resolved transfer is still `ExtendedSrgb`. The gamut stays in
        // `DisplayTarget::gamut`.
        let expected = NegotiatedSurface {
            format: TextureFormat::Rgba16Float,
            color_space: SurfaceColorSpace::ExtendedDisplayP3,
            resolved_transfer: DisplayTransfer::ExtendedSrgb,
        };
        assert_eq!(
            negotiate_surface_format(
                &web_like(),
                DisplayTransfer::ExtendedSrgb,
                DisplayGamut::DisplayP3
            ),
            expected
        );
    }

    #[test]
    fn extended_srgb_displayp3_without_p3_support_downgrades_straight_to_sdr() {
        let caps = caps(
            vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba16Float],
            vec![
                fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
                fc(
                    TextureFormat::Rgba16Float,
                    SurfaceColorSpaces::EXTENDED_SRGB,
                ),
            ],
        );
        assert_eq!(
            negotiate_surface_format(
                &caps,
                DisplayTransfer::ExtendedSrgb,
                DisplayGamut::DisplayP3
            ),
            sdr_selection(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn extended_srgb_without_support_downgrades_to_sdr() {
        assert_eq!(
            negotiate_surface_format(
                &sdr_only(),
                DisplayTransfer::ExtendedSrgb,
                DisplayGamut::Rec709
            ),
            sdr_selection(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn scrgb_linear_does_not_fall_back_to_extended_srgb() {
        // A web-like surface advertises the encoded `ExtendedSrgb` but not
        // `ExtendedSrgbLinear`. On the web, request `ExtendedSrgb` instead.
        assert_eq!(
            negotiate(&web_like(), DisplayTransfer::ScRgbLinear),
            sdr_selection(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn pq_does_not_fall_back_to_extended_srgb() {
        // PQ falls back to linear scRGB, then SDR, never to `ExtendedSrgb`. A
        // web-like surface has neither PQ nor linear scRGB.
        assert_eq!(
            negotiate(&web_like(), DisplayTransfer::Pq),
            sdr_selection(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn empty_auto_formats_fall_back_to_extended_srgb_spaces() {
        // A driver reporting only an encoded extended-sRGB space, no Auto
        // format. PQ and its fallbacks find nothing on either surface.
        let extended_srgb_only = caps(
            vec![],
            vec![fc(
                TextureFormat::Rgba16Float,
                SurfaceColorSpaces::EXTENDED_SRGB,
            )],
        );
        assert_eq!(
            negotiate(&extended_srgb_only, DisplayTransfer::Pq),
            NegotiatedSurface {
                format: TextureFormat::Rgba16Float,
                color_space: SurfaceColorSpace::ExtendedSrgb,
                resolved_transfer: DisplayTransfer::ExtendedSrgb,
            }
        );
        let extended_p3_only = caps(
            vec![],
            vec![fc(
                TextureFormat::Rgba16Float,
                SurfaceColorSpaces::EXTENDED_DISPLAY_P3,
            )],
        );
        assert_eq!(
            negotiate_surface_format(
                &extended_p3_only,
                DisplayTransfer::Pq,
                DisplayGamut::DisplayP3
            ),
            NegotiatedSurface {
                format: TextureFormat::Rgba16Float,
                color_space: SurfaceColorSpace::ExtendedDisplayP3,
                resolved_transfer: DisplayTransfer::ExtendedSrgb,
            }
        );
    }

    #[test]
    #[should_panic(expected = "No supported formats for surface")]
    fn empty_auto_formats_never_give_a_rec709_request_the_extended_p3_fallback() {
        let extended_p3_only = caps(
            vec![],
            vec![fc(
                TextureFormat::Rgba16Float,
                SurfaceColorSpaces::EXTENDED_DISPLAY_P3,
            )],
        );
        negotiate_surface_format(
            &extended_p3_only,
            DisplayTransfer::ExtendedSrgb,
            DisplayGamut::Rec709,
        );
    }

    #[test]
    fn supported_transfers_matches_negotiable_set() {
        use DisplayTransfer::{ExtendedSrgb, Pq, ScRgbLinear, Srgb};

        let cases: [(
            &str,
            fn() -> wgpu::SurfaceCapabilities,
            Vec<DisplayTransfer>,
        ); 5] = [
            (
                "metal_like",
                metal_like,
                vec![Srgb, ScRgbLinear, Pq, ExtendedSrgb],
            ),
            ("web_like", web_like, vec![Srgb, ExtendedSrgb]),
            (
                "vulkan_hdr_like",
                vulkan_hdr_like,
                vec![Srgb, ScRgbLinear, Pq],
            ),
            ("scrgb_only", scrgb_only, vec![Srgb, ScRgbLinear]),
            ("sdr_only", sdr_only, vec![Srgb]),
        ];

        for (name, fixture, expected) in cases {
            assert_eq!(
                supported_transfers(&fixture()).iter().collect::<Vec<_>>(),
                expected,
                "{name}"
            );
        }
    }
}
