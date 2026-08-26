//! Decides which camera in a camera stack runs the tonemapping pass, and
//! writes a [`ViewStackContract`] on every camera view that has a
//! [`ViewTarget`], for prepare systems that depend on the stack.
//!
//! Cameras that render to the same target share main textures when their
//! settings match. `prepare_view_targets` allocates them. When an earlier
//! camera runs a fullscreen pass, every later camera in the stack composites
//! on top of pixels that pass already processed. Running the pass per camera
//! would tonemap a lower camera's output a second time. Instead the stack's
//! last tonemapping camera runs the pass once, over the composited buffer.
//!
//! The display-encoding pass hands off the same way. On an HDR display
//! target the encode must hand off whenever the tonemap does, so a stack
//! shape that forces per-camera encoding also forces per-camera
//! tonemapping.

use bevy_app::{App, Plugin};
use bevy_camera::{CameraOutputMode, ClearColorConfig};
use bevy_ecs::{
    change_detection::DetectChangesMut,
    component::Component,
    entity::{Entity, EntityHashMap},
    schedule::IntoScheduleConfigs,
    system::{Commands, Query, Res},
};
use bevy_log::{info_once, warn_once};
use bevy_platform::collections::HashMap;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::TextureId,
    view::{
        composites_fullscreen, prepare_view_display_targets, prepare_view_targets,
        ViewDisplayTarget, ViewTarget,
    },
    working_color_space::WorkingColorSpace,
    Render, RenderApp, RenderSystems,
};
use bevy_window::{DisplayGamut, DisplayTransfer};
use core::hash::Hash;

use crate::tonemapping::{resolve_tonemapping, Tonemapping};

/// Registers [`resolve_camera_stack_contracts`], which turns each frame's
/// camera stacks into a [`ViewStackContract`] component on every camera
/// view that has a [`ViewTarget`].
pub struct CameraStackPlugin;

impl Plugin for CameraStackPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(
            Render,
            resolve_camera_stack_contracts
                .in_set(RenderSystems::PrepareViews)
                .after(prepare_view_targets)
                .after(prepare_view_display_targets),
        );
    }
}

/// A view's role for a fullscreen pass that can run once for the whole
/// camera stack.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StackRole {
    /// The view runs its own pass.
    Solo,
    /// The view doesn't run the pass. The camera named here runs it once
    /// for the whole stack.
    HandledBy(Entity),
    /// The view runs the pass once for the whole stack.
    Finalizer,
}

/// Whether a view's upscaling blit runs, and with which auto-detected blend.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlitDisposition {
    /// The view's blit runs.
    Run {
        /// Whether the auto-detected alpha blend becomes a replacing blend.
        /// This is true on the finalizer when the lower blits are skipped.
        /// Its blit writes the whole stack's output, and no earlier blit
        /// from the stack has written the out texture, so it must replace
        /// rather than blend.
        force_replace: bool,
    },
    /// The view sits below its stack's finalizer, so its blit would write
    /// pixels the stack hasn't tonemapped yet. The blit is skipped. The
    /// finalizer's blit writes the whole stack's output instead.
    SkipForFinalizer,
}

/// Resolved display-encoding parameters for a view, after the prepare-time
/// transfer and gamut coercion chain (see `coerce_display_encode`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ResolvedEncoding {
    pub transfer: DisplayTransfer,
    /// The gamut the encoder transforms to.
    pub gamut: DisplayGamut,
}

/// A view's resolved composition state.
///
/// The contract is overwritten in place every frame and never removed, so a
/// view whose `ViewTarget` was dropped keeps a stale contract. Systems that
/// read the contract must also require `ViewTarget` in their query.
///
/// The component isn't registered for reflection. It's internal to the
/// render world, and its field types aren't `Reflect`.
#[derive(Component, Clone, Copy, PartialEq, Debug)]
pub struct ViewStackContract {
    /// The view's role for the tonemapping pass.
    pub tonemap: StackRole,
    /// The view's role for the display-encoding pass.
    pub encode: StackRole,
    /// The view's upscaling blit disposition.
    pub blit: BlitDisposition,
    /// Color primaries of the buffer at display-encoding time. A handed-off
    /// encode uses the tonemap output gamut of the last tonemap-enabled
    /// member of the stack. A solo encode uses this view's own, because each
    /// per-camera encode keys for its own region.
    pub source_gamut: DisplayGamut,
    /// Resolved encode parameters. `Some` exactly when the view's resolved
    /// display target requests an HDR transfer.
    pub encoding: Option<ResolvedEncoding>,
}

impl ViewStackContract {
    /// Whether the encoder's input buffer for this view uses Rec.2020
    /// primaries. That is a GT7 pass on an HDR-transfer target, this view's
    /// own or its stack's, which emits native Rec.2020. Post-tonemap writers
    /// such as UI use it to convert their Rec.709-authored colors to the
    /// buffer's primaries.
    ///
    /// A solo `Tonemapping::None` view under a Rec.2020 working space returns
    /// `false`: no pass marks the buffer Rec.2020.
    pub fn source_gamut_is_rec2020(&self) -> bool {
        matches!(self.source_gamut, DisplayGamut::Rec2020)
    }

    /// Whether this view's stack encodes for an HDR transfer. Post-tonemap
    /// consumers that key HDR-input math (FXAA, SMAA, CAS) read it here
    /// instead of the `ViewDisplayTarget`.
    pub fn is_hdr_encode(&self) -> bool {
        self.encoding.is_some()
    }
}

/// One view's input to [`resolve_contracts`].
pub(crate) struct ContractInput<K> {
    pub entity: Entity,
    /// Identity of the main textures the view renders into. Views
    /// resolve together only when they share it.
    pub texture: K,
    /// The camera's position in its render target's sorted camera order.
    /// The resolver and the upscaling auto-blend rely on this bottom-to-top
    /// order.
    pub sorted_index: usize,
    /// See [`composites_fullscreen`].
    pub composites_fullscreen: bool,
    /// Whether the view's display-encoding pass would run (the resolved
    /// display target requests an HDR transfer).
    pub encode_enabled: bool,
    /// Whether the camera writes to its render target. False for
    /// [`CameraOutputMode::Skip`].
    pub output_writes: bool,
    /// This view's own tonemap output gamut
    /// (`resolve_tonemapping(own method, own display target).output_gamut`).
    pub tonemap_output_gamut: DisplayGamut,
    /// Whether the view's main pass loads the previous buffer contents,
    /// true for [`ClearColorConfig::None`]. This tells viewport members
    /// from clearing members in the stack-shape checks.
    pub loads_previous: bool,
    /// The view's resolved tonemapping method
    /// (`resolve_tonemapping(..).method`).
    pub method: Tonemapping,
}

/// A view's tonemapping pass runs when its resolved tonemapping method is
/// enabled. This mirrors the tonemapping pass's own gate, so a stack never
/// relies on a pass that doesn't run.
fn tonemap_pass_runs<K>(input: &ContractInput<K>) -> bool {
    input.method.is_enabled()
}

/// A stack misconfiguration found during contract resolution.
/// [`resolve_camera_stack_contracts`] reports each error as a warning with
/// `warn_once!`. [`resolve_contracts`] returns the errors so tests can
/// check the trigger conditions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum StackResolutionError {
    /// A viewport or clearing member forces the whole stack to tonemap per
    /// camera, because its HDR target's encode pass can't hand off past it.
    EncodeShapeForcesPerCameraTonemap { blocking_camera: Entity },
    /// The stack encodes for an HDR transfer with no tonemapping pass
    /// anywhere in it.
    EncodeWithoutTonemap { first: Entity },
    /// A fullscreen compositing member blits over regions that members
    /// below it tonemap or encode per camera.
    FullscreenBlitOverPerCameraPasses { fullscreen_camera: Entity },
    /// The stack's first member loads the previous buffer contents while the
    /// stack runs a tonemapping or display-encoding pass.
    FrameStartLoadsProcessedOutput { first: Entity },
    /// A `HandledBy` member's tonemapping method differs from its finalizer's.
    TonemappingMismatch {
        member: Entity,
        own: Tonemapping,
        finalizing: Tonemapping,
    },
}

/// Per-view output of [`resolve_contracts`]: every [`ViewStackContract`]
/// field except the encode parameters, which need the `ViewDisplayTarget`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct ContractOutput {
    pub tonemap: StackRole,
    pub encode: StackRole,
    pub blit: BlitDisposition,
    pub source_gamut: DisplayGamut,
}

/// Resolves every view's stack roles, blit disposition, and encoder source
/// gamut, grouped by shared main texture. Returns the per-view outputs and
/// the errors that fired.
///
/// Precondition: `sorted_index` values are unique within a texture group
/// (`sort_cameras` counts indices per render target, so distinct cameras on
/// one target never tie). The resolver defines no tie semantics.
pub(crate) fn resolve_contracts<K: Copy + Eq + Hash>(
    views: Vec<ContractInput<K>>,
) -> (EntityHashMap<ContractOutput>, Vec<StackResolutionError>) {
    let mut groups: HashMap<K, Vec<ContractInput<K>>> = HashMap::default();
    for view in views {
        groups.entry(view.texture).or_default().push(view);
    }

    let mut outputs = EntityHashMap::default();
    let mut errors = Vec::new();
    for group in groups.values_mut() {
        group.sort_unstable_by_key(|view| view.sorted_index);
        debug_assert!(
            group
                .windows(2)
                .all(|pair| pair[0].sorted_index != pair[1].sorted_index),
            "sorted camera indices must be unique within a texture group"
        );
        resolve_group(group, &mut outputs, &mut errors);
    }
    (outputs, errors)
}

/// Returns the index of the member that runs one fullscreen pass for the
/// whole sorted texture group, or `None` when the pass runs per camera.
///
/// The group gets a finalizer only when at least two members are enabled
/// and every enabled member after the first composites fullscreen. The
/// finalizer is the last enabled member. Any other arrangement keeps each
/// camera running its own pass over what it rendered.
fn pass_finalizer<K>(
    members: &[ContractInput<K>],
    enabled: impl Fn(&ContractInput<K>) -> bool,
) -> Option<usize> {
    let mut tail = members
        .iter()
        .enumerate()
        .filter(|(_, member)| enabled(member));
    tail.next()?;
    let mut finalizer = None;
    for (index, member) in tail {
        if !member.composites_fullscreen {
            return None;
        }
        finalizer = Some(index);
    }
    finalizer
}

/// Resolves one texture group of sorted members into [`ContractOutput`]s
/// and errors.
fn resolve_group<K>(
    members: &[ContractInput<K>],
    outputs: &mut EntityHashMap<ContractOutput>,
    errors: &mut Vec<StackResolutionError>,
) {
    let encode_enabled_group = members.iter().any(|member| member.encode_enabled);

    let mut tonemap_finalizer = pass_finalizer(members, tonemap_pass_runs);
    let encode_finalizer = pass_finalizer(members, |member| member.encode_enabled);

    // On an encode-enabled group the encode must hand off whenever the
    // tonemap does, or a handled member's own encode pass would run on the
    // not-yet-tonemapped buffer. Each hand-off test looks only at enabled
    // views, so a pass-disabled viewport member is invisible to the tonemap
    // test but breaks the shape for the encode test. The tonemap hand-off
    // is then cancelled and every member tonemaps per camera.
    if encode_enabled_group && tonemap_finalizer.is_some() && encode_finalizer.is_none() {
        tonemap_finalizer = None;
        let blocking = members
            .iter()
            .filter(|member| member.encode_enabled)
            .skip(1)
            .find(|member| !member.composites_fullscreen);
        if let Some(blocking) = blocking {
            errors.push(StackResolutionError::EncodeShapeForcesPerCameraTonemap {
                blocking_camera: blocking.entity,
            });
        }
    }

    if encode_enabled_group
        && !members.iter().any(tonemap_pass_runs)
        && let Some(first) = members.first()
    {
        errors.push(StackResolutionError::EncodeWithoutTonemap {
            first: first.entity,
        });
    }

    // The finalizer is by construction the last enabled member, so an
    // enabled member below it is handled by it.
    let role = |index: usize, enabled: bool, finalizer: Option<usize>| match finalizer {
        Some(f) if index == f => StackRole::Finalizer,
        Some(f) if enabled && index < f => StackRole::HandledBy(members[f].entity),
        _ => StackRole::Solo,
    };

    // The gamut of the composed buffer a handed-off encode reads: whatever
    // the last tonemap-enabled member in sorted order produced. Not the
    // tonemap finalizer, which does not exist when the tonemap pass runs
    // per camera.
    let group_gamut = members
        .iter()
        .rev()
        .find(|member| tonemap_pass_runs(member))
        .map(|member| member.tonemap_output_gamut)
        .unwrap_or(DisplayGamut::Rec709);

    // The finalizer whose blit writes the whole stack's output: the
    // highest-index finalizer of either pass. A `CameraOutputMode::Skip`
    // finalizer never blits. Lower members skip their blits only because
    // the finalizer's blit writes their output, so without it the group
    // keeps every blit.
    let blitting_finalizer = tonemap_finalizer
        .max(encode_finalizer)
        .filter(|&finalizer| members[finalizer].output_writes);

    // A stack that neither tonemaps nor encodes keeps the buffer
    // scene-referred and accumulates stably, so the warning fires only when
    // a pass runs.
    if (encode_enabled_group || members.iter().any(tonemap_pass_runs))
        && members.first().is_some_and(|first| first.loads_previous)
    {
        errors.push(StackResolutionError::FrameStartLoadsProcessedOutput {
            first: members[0].entity,
        });
    }

    // A member covers partially when its output doesn't composite over the
    // whole target. The first member is expected to clear, so a clearing
    // viewport as the first member counts as covering.
    let covers_partially = |index: usize, member: &ContractInput<K>| {
        if index == 0 {
            !member.composites_fullscreen && member.loads_previous
        } else {
            !member.composites_fullscreen
        }
    };
    // A member whose enabled pass runs per camera. When a pass has a
    // finalizer, every enabled member either hands off to it or is it, so a
    // per-camera pass means the pass has no finalizer at all.
    let runs_own_pass = |member: &ContractInput<K>| {
        (tonemap_pass_runs(member) && tonemap_finalizer.is_none())
            || (member.encode_enabled && encode_finalizer.is_none())
    };
    // A fullscreen compositing member above a partially covering member
    // blits the whole target, so regions tonemapped or encoded per camera
    // below it get written twice. The reverse shape, a viewport member
    // above members that run their own passes, gets no warning. Any trigger
    // for it would also fire on ordinary split screen.
    let flagged = members.iter().enumerate().find(|(index, candidate)| {
        candidate.composites_fullscreen
            && members[..*index]
                .iter()
                .enumerate()
                .any(|(below_index, below)| covers_partially(below_index, below))
            && members[..*index].iter().any(&runs_own_pass)
    });
    if let Some((_, flagged)) = flagged {
        errors.push(StackResolutionError::FullscreenBlitOverPerCameraPasses {
            fullscreen_camera: flagged.entity,
        });
    }

    for (index, member) in members.iter().enumerate() {
        let tonemap = role(index, tonemap_pass_runs(member), tonemap_finalizer);
        let encode = role(index, member.encode_enabled, encode_finalizer);

        let blit = match blitting_finalizer {
            Some(finalizer) if index < finalizer => BlitDisposition::SkipForFinalizer,
            Some(finalizer) if index == finalizer => BlitDisposition::Run {
                force_replace: true,
            },
            // Members above the blitting finalizer composite over what the
            // finalizer blitted. Without one, every blit composites over
            // whatever blitted before it.
            _ => BlitDisposition::Run {
                force_replace: false,
            },
        };

        let source_gamut = match encode {
            StackRole::Solo => member.tonemap_output_gamut,
            StackRole::HandledBy(_) | StackRole::Finalizer => group_gamut,
        };

        if let (StackRole::HandledBy(_), Some(finalizer)) = (tonemap, tonemap_finalizer) {
            let finalizing = members[finalizer].method;
            if member.method != finalizing {
                errors.push(StackResolutionError::TonemappingMismatch {
                    member: member.entity,
                    own: member.method,
                    finalizing,
                });
            }
        }

        outputs.insert(
            member.entity,
            ContractOutput {
                tonemap,
                encode,
                blit,
                source_gamut,
            },
        );
    }
}

/// Resolves every camera view's stack into a [`ViewStackContract`].
///
/// Runs in [`RenderSystems::PrepareViews`] after `prepare_view_targets`,
/// which supplies the `ViewTarget`, and `prepare_view_display_targets`, which
/// supplies the `ViewDisplayTarget`.
///
/// A texture group never spans two [`ResolvedCompositingSpace`](bevy_render::view::ResolvedCompositingSpace) groups: equal
/// main-texture ids imply equal main-texture keys, and `prepare_view_targets`
/// dedups allocations on exactly that key.
///
/// The stack rules live in `resolve_contracts`.
pub fn resolve_camera_stack_contracts(
    mut commands: Commands,
    views: Query<(
        Entity,
        &ExtractedCamera,
        &ViewTarget,
        &ViewDisplayTarget,
        Option<&Tonemapping>,
    )>,
    mut contracts: Query<&mut ViewStackContract>,
    working_color_space: Res<WorkingColorSpace>,
) {
    // Encode parameters resolve once per texture group. Members share one
    // `ViewDisplayTarget`, which resolves per render target, and the target is
    // part of the texture grouping key, so transfer and gamut are uniform
    // across a group. Resolving from the first-iterated member therefore
    // agrees with every member's `encode_enabled`, since both read
    // `is_hdr_transfer()` off that shared target. The `debug_assert_eq!`
    // catches a pathological plugin order that leaves a view holding its
    // default SDR `ViewDisplayTarget`, which would mix an encoded group with
    // an unencoded out-texture clear.
    let mut group_encodings: HashMap<TextureId, Option<ResolvedEncoding>> = HashMap::default();
    let mut inputs = Vec::new();
    for (entity, camera, view_target, view_display_target, tonemapping) in &views {
        let texture = view_target.main_texture().id();
        let encode_enabled = view_display_target.is_hdr_transfer();
        let encoding = *group_encodings.entry(texture).or_insert_with(|| {
            resolve_group_encode_parameters(view_display_target, view_target, *working_color_space)
        });
        debug_assert_eq!(
            encoding.is_some(),
            encode_enabled,
            "every member of a texture group must agree on display-encode enablement"
        );
        let resolved = resolve_tonemapping(tonemapping, view_display_target);
        inputs.push(ContractInput {
            entity,
            texture,
            sorted_index: camera.sorted_camera_index_for_target,
            composites_fullscreen: composites_fullscreen(camera),
            encode_enabled,
            output_writes: !matches!(camera.output_mode, CameraOutputMode::Skip),
            tonemap_output_gamut: resolved.output_gamut,
            loads_previous: matches!(camera.clear_color, ClearColorConfig::None),
            method: resolved.method,
        });
    }

    let (outputs, errors) = resolve_contracts(inputs);
    for error in errors {
        emit_stack_resolution_error(error);
    }

    for (entity, _, view_target, ..) in &views {
        let Some(output) = outputs.get(&entity) else {
            continue;
        };

        let encoding = group_encodings
            .get(&view_target.main_texture().id())
            .copied()
            .flatten();

        let contract = ViewStackContract {
            tonemap: output.tonemap,
            encode: output.encode,
            blit: output.blit,
            source_gamut: output.source_gamut,
            encoding,
        };
        if let Ok(mut existing) = contracts.get_mut(entity) {
            existing.set_if_neq(contract);
        } else {
            commands.entity(entity).insert(contract);
        }
    }
}

/// Resolves a texture group's display transfer and gamut: the group-level
/// display-target warnings plus [`coerce_display_encode`]. Returns `None`
/// when the group's display target does not request an HDR transfer, so no
/// encode pass runs.
fn resolve_group_encode_parameters(
    view_display_target: &ViewDisplayTarget,
    view_target: &ViewTarget,
    working_color_space: WorkingColorSpace,
) -> Option<ResolvedEncoding> {
    if !view_display_target.is_hdr_transfer() {
        return None;
    }

    // Only manual Image/TextureView targets can hit this. Window surfaces
    // negotiate HDR transfers onto formats without a hardware sRGB encode,
    // while a `ManualDisplayTargets` entry resolves verbatim because the user
    // owns the texture.
    if view_target
        .out_texture_view_format()
        .is_some_and(|format| format.is_srgb())
    {
        warn_once!(
            "A render target registered in `ManualDisplayTargets` with an HDR transfer \
            is backed by an sRGB texture format; the hardware sRGB encode will corrupt \
            the encoded HDR signal. Use a non-sRGB format (e.g. `Rgba16Float`) for HDR \
            render targets."
        );
    }

    // Advisory only: a global setting must never flip automatically because
    // one window went HDR.
    if !working_color_space.is_rec2020() {
        warn_once!(
            "A camera is rendering to an HDR display target while the working color \
            space is the default `WorkingColorSpace::Rec709`. Output is correct but \
            limited to the Rec.709 gamut; consider opting into the wide working space \
            with `RenderPlugin {{ working_color_space: WorkingColorSpace::Rec2020, .. }}`."
        );
    }

    let (transfer, gamut) =
        coerce_display_encode(view_display_target.transfer, view_display_target.gamut);
    Some(ResolvedEncoding { transfer, gamut })
}

/// The prepare-time display-encode coercion chain over the resolved transfer
/// and gamut.
///
/// The order is load bearing: the P3 -> Rec709 arm runs before the
/// transfer-specific gamut arms.
pub(crate) fn coerce_display_encode(
    transfer: DisplayTransfer,
    gamut: DisplayGamut,
) -> (DisplayTransfer, DisplayGamut) {
    let mut gamut = gamut;
    if gamut == DisplayGamut::DisplayP3 && transfer != DisplayTransfer::ExtendedSrgb {
        warn_once!(
            "`DisplayGamut::DisplayP3` output is only supported with \
            `DisplayTransfer::ExtendedSrgb` (wgpu's `ExtendedDisplayP3` surface \
            color space); the {transfer:?} transfer ships no P3 surface, \
            so leaving colors in Rec.709 primaries."
        );
        gamut = DisplayGamut::Rec709;
    }
    if transfer == DisplayTransfer::ScRgbLinear && gamut != DisplayGamut::Rec709 {
        // scRGB-linear (IEC 61966-2-2) gets wide gamut from out-of-range and
        // negative components, not from a change of primaries. Rec.2020
        // coordinates would still be read as Rec.709 and desaturate every pixel.
        info_once!(
            "scRGB-linear signals are always expressed in (extended) Rec.709/sRGB \
            coordinates (the OS compositor performs the mapping to the panel's gamut); \
            ignoring `DisplayTarget::gamut` ({gamut:?}) for encoding. The field still \
            correctly describes the panel for luminance/metadata purposes."
        );
        gamut = DisplayGamut::Rec709;
    }
    if transfer == DisplayTransfer::ExtendedSrgb && gamut == DisplayGamut::Rec2020 {
        warn_once!(
            "Encoded extended-range sRGB (`DisplayTransfer::ExtendedSrgb`) has no \
            Rec.2020 surface (only `ExtendedSrgb` / `ExtendedDisplayP3`); coercing \
            `DisplayTarget::gamut` from Rec2020 to Rec709 for encoding."
        );
        gamut = DisplayGamut::Rec709;
    }
    if transfer == DisplayTransfer::Pq && gamut != DisplayGamut::Rec2020 {
        info_once!(
            "PQ display targets are canonically Rec.2020 (ITU-R BT.2100); encoding \
            the {gamut:?} gamut against Rec.2020 primaries (a lossless widening; \
            `DisplayTarget::gamut` is unchanged)."
        );
        gamut = DisplayGamut::Rec2020;
    }
    (transfer, gamut)
}

/// Warns about one resolution error.
fn emit_stack_resolution_error(error: StackResolutionError) {
    match error {
        StackResolutionError::EncodeShapeForcesPerCameraTonemap { blocking_camera } => {
            warn_once!(
                "Viewport or clearing camera {blocking_camera} sits in a camera stack \
                rendering to an HDR display target. The display-encoding pass can't run \
                once for the whole stack with that camera in it, and tonemapping can't \
                hand off without the encode, so each camera tonemaps its own pass and \
                the composited buffer isn't tonemapped as one image. Give that camera \
                fullscreen ClearColorConfig::None composition, or move it to its own \
                render target."
            );
        }
        StackResolutionError::EncodeWithoutTonemap { first } => {
            warn_once!(
                "No camera in the stack of view {first} has an active tonemapping \
                method, and the stack encodes for an HDR display target, so the encoder \
                transfer-encodes untonemapped linear values. Add a method like \
                Tonemapping::GranTurismo7, or Tonemapping::Linear to encode with no \
                tone curve."
            );
        }
        StackResolutionError::FullscreenBlitOverPerCameraPasses { fullscreen_camera } => {
            warn_once!(
                "Fullscreen ClearColorConfig::None camera {fullscreen_camera} composites \
                above viewport or clearing cameras whose tonemapping or display-encoding \
                passes run per camera. Its blit covers the whole target and writes their \
                processed pixels a second time. Give that camera its own render target."
            );
        }
        StackResolutionError::FrameStartLoadsProcessedOutput { first } => {
            warn_once!(
                "The first camera rendering to a target, view {first}, uses \
                ClearColorConfig::None while its stack runs a tonemapping or \
                display-encoding pass. The main texture persists across frames, so each \
                frame reprocesses last frame's tonemapped or encoded output. Feedback \
                and trail effects built this way drift over time. Stable accumulation \
                needs Tonemapping::None on an SDR target, which never tonemaps or \
                encodes the buffer."
            );
        }
        StackResolutionError::TonemappingMismatch {
            member,
            own,
            finalizing,
        } => {
            warn_once!(
                "Stacked cameras rendering to the same target use different tonemapping \
                methods. View {member} uses {own:?}, but the stack composites into one \
                shared buffer and tonemaps it once with {finalizing:?}. The last \
                tonemapping camera's Tonemapping, ColorGrading, GranTurismo7Params, and \
                DebandDither apply to the whole stack. Give every camera in the stack \
                the same Tonemapping."
            );
        }
    }
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    const RUN: BlitDisposition = BlitDisposition::Run {
        force_replace: false,
    };
    const RUN_REPLACE: BlitDisposition = BlitDisposition::Run {
        force_replace: true,
    };

    fn entity(raw: u32) -> Entity {
        Entity::from_raw_u32(raw).unwrap()
    }

    /// An SDR member with an active method that clears its target.
    fn clearing(raw: u32, index: usize) -> ContractInput<u32> {
        ContractInput {
            entity: entity(raw),
            texture: 0,
            sorted_index: index,
            composites_fullscreen: false,
            encode_enabled: false,
            output_writes: true,
            tonemap_output_gamut: DisplayGamut::Rec709,
            loads_previous: false,
            method: Tonemapping::TonyMcMapface,
        }
    }

    /// A fullscreen `ClearColorConfig::None` member.
    fn compositing(raw: u32, index: usize) -> ContractInput<u32> {
        let mut input = clearing(raw, index);
        input.composites_fullscreen = true;
        input.loads_previous = true;
        input
    }

    /// A viewport member that loads previous content.
    fn viewport(raw: u32, index: usize) -> ContractInput<u32> {
        let mut input = clearing(raw, index);
        input.composites_fullscreen = false;
        input.loads_previous = true;
        input
    }

    /// Marks a member as GT7 on an HDR (PQ) target.
    fn gt7_hdr(mut input: ContractInput<u32>) -> ContractInput<u32> {
        input.encode_enabled = true;
        input.method = Tonemapping::GranTurismo7;
        input.tonemap_output_gamut = DisplayGamut::Rec2020;
        input
    }

    /// Marks a member as `Tonemapping::None` on an HDR (PQ) target.
    fn passthrough_hdr(mut input: ContractInput<u32>) -> ContractInput<u32> {
        input.encode_enabled = true;
        input.method = Tonemapping::None;
        input.tonemap_output_gamut = DisplayGamut::Rec709;
        input
    }

    /// Marks a member as `Tonemapping::None` on an SDR target.
    fn disabled(mut input: ContractInput<u32>) -> ContractInput<u32> {
        input.method = Tonemapping::None;
        input
    }

    fn output(outputs: &EntityHashMap<ContractOutput>, raw: u32) -> ContractOutput {
        *outputs
            .get(&entity(raw))
            .expect("view must have a contract")
    }

    #[test]
    fn solo_camera_is_solo_everywhere() {
        let (outputs, errors) = resolve_contracts(vec![clearing(1, 0)]);
        let solo = output(&outputs, 1);
        assert_eq!(solo.tonemap, StackRole::Solo);
        assert_eq!(solo.encode, StackRole::Solo);
        assert_eq!(solo.blit, RUN);
        assert_eq!(solo.source_gamut, DisplayGamut::Rec709);
        assert!(errors.is_empty());
    }

    // The handed-off encode's source gamut is the last tonemap-enabled
    // member's (the base), not the finalizer's own.
    #[test]
    fn gt7_base_with_passthrough_top_hands_off_encode_only() {
        let (outputs, errors) = resolve_contracts(vec![
            gt7_hdr(clearing(1, 0)),
            passthrough_hdr(compositing(2, 1)),
        ]);
        let base = output(&outputs, 1);
        let top = output(&outputs, 2);
        assert_eq!(base.tonemap, StackRole::Solo);
        assert_eq!(top.tonemap, StackRole::Solo);
        assert_eq!(base.encode, StackRole::HandledBy(entity(2)));
        assert_eq!(top.encode, StackRole::Finalizer);
        assert_eq!(base.source_gamut, DisplayGamut::Rec2020);
        assert_eq!(top.source_gamut, DisplayGamut::Rec2020);
        assert_eq!(base.blit, BlitDisposition::SkipForFinalizer);
        assert_eq!(top.blit, RUN_REPLACE);
        assert!(errors.is_empty());
    }

    #[test]
    fn all_passthrough_hdr_stack_flags_encode_without_tonemap() {
        let (outputs, errors) = resolve_contracts(vec![
            passthrough_hdr(clearing(1, 0)),
            passthrough_hdr(compositing(2, 1)),
        ]);
        let base = output(&outputs, 1);
        let top = output(&outputs, 2);
        assert_eq!(base.tonemap, StackRole::Solo);
        assert_eq!(top.tonemap, StackRole::Solo);
        assert_eq!(base.encode, StackRole::HandledBy(entity(2)));
        assert_eq!(top.encode, StackRole::Finalizer);
        assert_eq!(base.source_gamut, DisplayGamut::Rec709);
        assert_eq!(top.source_gamut, DisplayGamut::Rec709);
        assert_eq!(
            errors,
            vec![StackResolutionError::EncodeWithoutTonemap { first: entity(1) }]
        );
    }

    #[test]
    fn viewport_member_forces_per_camera_tonemap() {
        let (outputs, errors) = resolve_contracts(vec![
            gt7_hdr(clearing(1, 0)),
            gt7_hdr(compositing(2, 1)),
            passthrough_hdr(viewport(3, 2)),
        ]);
        for raw in 1..=3 {
            let member = output(&outputs, raw);
            assert_eq!(member.tonemap, StackRole::Solo);
            assert_eq!(member.encode, StackRole::Solo);
            assert_eq!(member.blit, RUN);
        }
        // Solo encodes key for their own pass output.
        assert_eq!(output(&outputs, 1).source_gamut, DisplayGamut::Rec2020);
        assert_eq!(output(&outputs, 2).source_gamut, DisplayGamut::Rec2020);
        assert_eq!(output(&outputs, 3).source_gamut, DisplayGamut::Rec709);
        assert_eq!(
            errors,
            vec![StackResolutionError::EncodeShapeForcesPerCameraTonemap {
                blocking_camera: entity(3)
            }]
        );
    }

    #[test]
    fn coherent_hdr_stack_hands_off_both_passes() {
        let (outputs, errors) =
            resolve_contracts(vec![gt7_hdr(clearing(1, 0)), gt7_hdr(compositing(2, 1))]);
        let base = output(&outputs, 1);
        let top = output(&outputs, 2);
        assert_eq!(base.tonemap, StackRole::HandledBy(entity(2)));
        assert_eq!(base.encode, StackRole::HandledBy(entity(2)));
        assert_eq!(top.tonemap, StackRole::Finalizer);
        assert_eq!(top.encode, StackRole::Finalizer);
        assert_eq!(base.blit, BlitDisposition::SkipForFinalizer);
        assert_eq!(top.blit, RUN_REPLACE);
        assert!(errors.is_empty());
    }

    // The tonemapping pass runs on SDR views too, so an SDR stack hands the
    // pass to its finalizer and skips the lower blits. There is no encode
    // pass to cohere with.
    #[test]
    fn sdr_stack_hands_off_tonemap_and_skips_lower_blits() {
        let (outputs, errors) = resolve_contracts(vec![clearing(1, 0), compositing(2, 1)]);
        let base = output(&outputs, 1);
        let top = output(&outputs, 2);
        assert_eq!(base.tonemap, StackRole::HandledBy(entity(2)));
        assert_eq!(top.tonemap, StackRole::Finalizer);
        assert_eq!(base.encode, StackRole::Solo);
        assert_eq!(top.encode, StackRole::Solo);
        assert_eq!(base.blit, BlitDisposition::SkipForFinalizer);
        assert_eq!(top.blit, RUN_REPLACE);
        assert!(errors.is_empty());
    }

    // A tonemapping base under a disabled compositing camera keeps
    // per-camera behavior. Nothing is handed off or skipped.
    #[test]
    fn single_enabled_member_keeps_per_camera_behavior() {
        let (outputs, errors) =
            resolve_contracts(vec![clearing(1, 0), disabled(compositing(2, 1))]);
        let base = output(&outputs, 1);
        let top = output(&outputs, 2);
        assert_eq!(base.tonemap, StackRole::Solo);
        assert_eq!(top.tonemap, StackRole::Solo);
        assert_eq!(base.blit, RUN);
        assert_eq!(top.blit, RUN);
        assert!(errors.is_empty());
    }

    #[test]
    fn three_member_stack_hands_the_pass_to_the_last() {
        let (outputs, _) =
            resolve_contracts(vec![clearing(1, 0), compositing(2, 1), compositing(3, 2)]);
        assert_eq!(output(&outputs, 1).tonemap, StackRole::HandledBy(entity(3)));
        assert_eq!(output(&outputs, 2).tonemap, StackRole::HandledBy(entity(3)));
        assert_eq!(output(&outputs, 3).tonemap, StackRole::Finalizer);
    }

    // The finalizer rule checks every tonemapping member after the first,
    // not just the last.
    #[test]
    fn partially_covering_middle_member_keeps_per_camera_passes() {
        let (outputs, _) =
            resolve_contracts(vec![compositing(1, 0), viewport(2, 1), compositing(3, 2)]);
        for raw in 1..=3 {
            assert_eq!(output(&outputs, raw).tonemap, StackRole::Solo);
        }
    }

    #[test]
    fn viewport_split_screen_keeps_per_camera_passes() {
        // The first split screen camera clears its target. A
        // `ClearColorConfig::None` first member would trip
        // `FrameStartLoadsProcessedOutput`.
        let mut left = viewport(1, 0);
        left.loads_previous = false;
        left.tonemap_output_gamut = DisplayGamut::Rec709;
        let mut right = viewport(2, 1);
        right.tonemap_output_gamut = DisplayGamut::Rec2020;
        let (outputs, errors) = resolve_contracts(vec![left, right]);
        let left = output(&outputs, 1);
        let right = output(&outputs, 2);
        assert_eq!(left.tonemap, StackRole::Solo);
        assert_eq!(right.tonemap, StackRole::Solo);
        assert_eq!(left.blit, RUN);
        assert_eq!(right.blit, RUN);
        assert_eq!(left.source_gamut, DisplayGamut::Rec709);
        assert_eq!(right.source_gamut, DisplayGamut::Rec2020);
        assert!(errors.is_empty());
    }

    #[test]
    fn sorted_index_orders_roles_not_insertion_order() {
        let (outputs, _) = resolve_contracts(vec![compositing(2, 1), clearing(1, 0)]);
        assert_eq!(output(&outputs, 1).tonemap, StackRole::HandledBy(entity(2)));
        assert_eq!(output(&outputs, 2).tonemap, StackRole::Finalizer);
        assert_eq!(output(&outputs, 1).blit, BlitDisposition::SkipForFinalizer);
        assert_eq!(output(&outputs, 2).blit, RUN_REPLACE);
    }

    // A disabled member below the finalizer is Solo, but its blit is still
    // skipped. The finalizer rule ignores disabled members, so a disabled
    // clearing member doesn't break it.
    #[test]
    fn disabled_member_below_finalizer_skips_blit() {
        let (outputs, _) = resolve_contracts(vec![
            clearing(1, 0),
            disabled(compositing(2, 1)),
            compositing(3, 2),
        ]);
        let base = output(&outputs, 1);
        let middle = output(&outputs, 2);
        let finalizer = output(&outputs, 3);
        assert_eq!(base.tonemap, StackRole::HandledBy(entity(3)));
        assert_eq!(middle.tonemap, StackRole::Solo);
        assert_eq!(finalizer.tonemap, StackRole::Finalizer);
        assert_eq!(base.blit, BlitDisposition::SkipForFinalizer);
        assert_eq!(middle.blit, BlitDisposition::SkipForFinalizer);
        assert_eq!(finalizer.blit, RUN_REPLACE);
    }

    // The camera above composites over the finalizer's blit.
    #[test]
    fn disabled_member_above_finalizer_keeps_alpha_blit() {
        let (outputs, _) = resolve_contracts(vec![
            clearing(1, 0),
            compositing(2, 1),
            disabled(compositing(3, 2)),
        ]);
        let base = output(&outputs, 1);
        let finalizer = output(&outputs, 2);
        let top = output(&outputs, 3);
        assert_eq!(base.tonemap, StackRole::HandledBy(entity(2)));
        assert_eq!(finalizer.tonemap, StackRole::Finalizer);
        assert_eq!(top.tonemap, StackRole::Solo);
        assert_eq!(base.blit, BlitDisposition::SkipForFinalizer);
        assert_eq!(finalizer.blit, RUN_REPLACE);
        assert_eq!(top.blit, RUN);
    }

    // No member skips its blit for a `CameraOutputMode::Skip` finalizer.
    // The pass roles are unaffected.
    #[test]
    fn skip_finalizer_cancels_blit_skipping() {
        let mut finalizer = compositing(2, 1);
        finalizer.output_writes = false;
        let (outputs, _) = resolve_contracts(vec![clearing(1, 0), finalizer]);
        assert_eq!(output(&outputs, 1).tonemap, StackRole::HandledBy(entity(2)));
        assert_eq!(output(&outputs, 2).tonemap, StackRole::Finalizer);
        assert_eq!(output(&outputs, 1).blit, RUN);
        assert_eq!(output(&outputs, 2).blit, RUN);
    }

    // The fullscreen camera's blit rewrites regions already tonemapped
    // per camera. The roles stay Solo.
    #[test]
    fn fullscreen_camera_above_viewport_cameras_is_flagged() {
        let (outputs, errors) = resolve_contracts(vec![
            viewport(1, 0),
            viewport(2, 1),
            disabled(compositing(3, 2)),
        ]);
        for raw in 1..=3 {
            let member = output(&outputs, raw);
            assert_eq!(member.tonemap, StackRole::Solo);
            assert_eq!(member.blit, RUN);
        }
        assert!(
            errors.contains(&StackResolutionError::FullscreenBlitOverPerCameraPasses {
                fullscreen_camera: entity(3)
            })
        );
    }

    // No warning fires here. See the fullscreen-blit check in
    // `resolve_group`.
    #[test]
    fn viewport_above_enabled_members_is_silent() {
        let (outputs, errors) = resolve_contracts(vec![clearing(1, 0), viewport(2, 1)]);
        assert_eq!(output(&outputs, 1).tonemap, StackRole::Solo);
        assert_eq!(output(&outputs, 2).tonemap, StackRole::Solo);
        assert!(errors.is_empty());
    }

    // A disabled clearing member mid-stack does not break the enabled-only
    // hand-off, because its clear is inert. The fullscreen-blit check stays
    // quiet since nothing below the finalizer runs a per-camera pass.
    #[test]
    fn divergent_stack_hands_off_and_skips_blits() {
        let (outputs, errors) = resolve_contracts(vec![
            clearing(1, 0),
            disabled(clearing(2, 1)),
            compositing(3, 2),
        ]);
        let base = output(&outputs, 1);
        let middle = output(&outputs, 2);
        let finalizer = output(&outputs, 3);
        assert_eq!(base.tonemap, StackRole::HandledBy(entity(3)));
        assert_eq!(middle.tonemap, StackRole::Solo);
        assert_eq!(finalizer.tonemap, StackRole::Finalizer);
        assert_eq!(base.blit, BlitDisposition::SkipForFinalizer);
        assert_eq!(middle.blit, BlitDisposition::SkipForFinalizer);
        assert_eq!(finalizer.blit, RUN_REPLACE);
        assert!(errors.is_empty());
    }

    #[test]
    fn tonemapping_mismatch_is_flagged_on_the_handled_member() {
        let mut base = clearing(1, 0);
        base.method = Tonemapping::AcesFitted;
        let (_, errors) = resolve_contracts(vec![base, compositing(2, 1)]);
        assert_eq!(
            errors,
            vec![StackResolutionError::TonemappingMismatch {
                member: entity(1),
                own: Tonemapping::AcesFitted,
                finalizing: Tonemapping::TonyMcMapface,
            }]
        );
    }

    // Not the first member's gamut and not the encode finalizer's own.
    #[test]
    fn last_tonemap_enabled_member_sets_the_handed_off_source_gamut() {
        let mut base = gt7_hdr(clearing(1, 0));
        base.tonemap_output_gamut = DisplayGamut::Rec2020;
        let mut middle = gt7_hdr(compositing(2, 1));
        middle.method = Tonemapping::Linear;
        middle.tonemap_output_gamut = DisplayGamut::Rec709;
        let top = passthrough_hdr(compositing(3, 2));
        let (outputs, _) = resolve_contracts(vec![base, middle, top]);
        for raw in 1..=3 {
            assert_eq!(output(&outputs, raw).source_gamut, DisplayGamut::Rec709);
        }
        assert_eq!(output(&outputs, 3).encode, StackRole::Finalizer);
    }

    #[test]
    fn frame_start_load_with_tonemapping_is_flagged() {
        let (_, errors) = resolve_contracts(vec![compositing(1, 0)]);
        assert_eq!(
            errors,
            vec![StackResolutionError::FrameStartLoadsProcessedOutput { first: entity(1) }]
        );
    }

    #[test]
    fn frame_start_load_with_encode_only_is_flagged() {
        let (_, errors) = resolve_contracts(vec![passthrough_hdr(compositing(1, 0))]);
        assert!(errors
            .contains(&StackResolutionError::FrameStartLoadsProcessedOutput { first: entity(1) }));
    }

    // With no pass over it the buffer stays scene-referred, so feedback
    // accumulates stably.
    #[test]
    fn frame_start_load_without_passes_is_silent() {
        let (_, errors) = resolve_contracts(vec![disabled(compositing(1, 0))]);
        assert!(errors.is_empty());
    }

    #[test]
    fn separate_textures_resolve_independently() {
        let mut other = compositing(2, 0);
        other.texture = 1;
        let (outputs, _) = resolve_contracts(vec![clearing(1, 0), other]);
        assert_eq!(output(&outputs, 1).tonemap, StackRole::Solo);
        assert_eq!(output(&outputs, 2).tonemap, StackRole::Solo);
        assert_eq!(output(&outputs, 1).blit, RUN);
        assert_eq!(output(&outputs, 2).blit, RUN);
    }
}

#[cfg(test)]
mod coercion_tests {
    use super::*;

    #[test]
    fn scrgb_rec709_is_unchanged() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::ScRgbLinear, DisplayGamut::Rec709),
            (DisplayTransfer::ScRgbLinear, DisplayGamut::Rec709)
        );
    }

    #[test]
    fn scrgb_forces_rec709() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::ScRgbLinear, DisplayGamut::Rec2020),
            (DisplayTransfer::ScRgbLinear, DisplayGamut::Rec709)
        );
    }

    #[test]
    fn pq_forces_rec2020() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::Pq, DisplayGamut::Rec709),
            (DisplayTransfer::Pq, DisplayGamut::Rec2020)
        );
    }

    #[test]
    fn pq_rec2020_is_unchanged() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::Pq, DisplayGamut::Rec2020),
            (DisplayTransfer::Pq, DisplayGamut::Rec2020)
        );
    }

    #[test]
    fn display_p3_collapses_to_rec709() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::ScRgbLinear, DisplayGamut::DisplayP3),
            (DisplayTransfer::ScRgbLinear, DisplayGamut::Rec709)
        );
    }

    // The sRGB transfer never reaches the chain from an HDR target, but the
    // coercion still leaves a non-HDR transfer untouched.
    #[test]
    fn srgb_is_unchanged() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::Srgb, DisplayGamut::Rec709),
            (DisplayTransfer::Srgb, DisplayGamut::Rec709)
        );
    }

    #[test]
    fn extended_srgb_keeps_display_p3() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::ExtendedSrgb, DisplayGamut::DisplayP3),
            (DisplayTransfer::ExtendedSrgb, DisplayGamut::DisplayP3)
        );
    }

    #[test]
    fn extended_srgb_keeps_rec709() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::ExtendedSrgb, DisplayGamut::Rec709),
            (DisplayTransfer::ExtendedSrgb, DisplayGamut::Rec709)
        );
    }

    #[test]
    fn extended_srgb_rec2020_falls_back_to_rec709() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::ExtendedSrgb, DisplayGamut::Rec2020),
            (DisplayTransfer::ExtendedSrgb, DisplayGamut::Rec709)
        );
    }
}
