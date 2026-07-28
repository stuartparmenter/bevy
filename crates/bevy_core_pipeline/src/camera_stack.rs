//! Camera-stack analysis: which camera in a stack runs each deferrable
//! fullscreen pass, and the per-view [`ViewStackContract`] every
//! stack-sensitive prepare system reads.
//!
//! Cameras that render to the same target share one main-texture ping-pong
//! (see `prepare_view_targets`), so a fullscreen pass run by an earlier
//! camera feeds already-processed pixels into every later camera that
//! composites on top with [`ClearColorConfig::None`]. Running the pass per
//! camera would then apply it twice to the earlier camera's pixels — e.g.
//! tone mapping the lower camera's output a second time, or alpha-blending
//! PQ-encoded signal. Instead, the pass is deferred to the last enabled
//! camera in the stack, which processes the composed buffer exactly once.
//!
//! [`resolve_camera_stack_contracts`] generalizes that analysis: it runs the
//! deferral once per pass (tone mapping and display encoding), reconciles
//! the two, resolves the display encoder's parameters once per stack, and
//! publishes the result as one [`ViewStackContract`] component per view, so
//! no consumer re-derives stack shape, deferral, buffer space, or encoder
//! inputs on its own.
//!
//! The main-texture ping-pong persists across frames, so a stack whose first
//! camera uses [`ClearColorConfig::None`] starts each frame from last frame's
//! tone-mapped (and, on an HDR target, display-encoded) output and reprocesses
//! it. Feedback/trail effects built that way drift over time; the resolver
//! reports it as a diagnostic but does not change the behavior. Stable
//! feedback accumulation needs [`Tonemapping::None`] on an SDR target so the
//! main buffer stays scene-referred across frames. Keeping a scene-referred
//! main buffer alongside a separate presentation chain is the follow-up
//! posture.
//!
//! [`ClearColorConfig::None`]: bevy_camera::ClearColorConfig
//! [`Tonemapping::None`]: crate::tonemapping::Tonemapping::None

use bevy_app::{App, Plugin};
use bevy_camera::{CameraOutputMode, ClearColorConfig, CompositingSpace};
use bevy_ecs::{
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
        ResolvedCompositingSpace, ViewDisplayTarget, ViewTarget,
    },
    working_color_space::WorkingColorSpace,
    Render, RenderApp, RenderSystems,
};
use bevy_window::{DisplayGamut, DisplayTransfer};
use core::hash::Hash;

use crate::tonemapping::{effective_tonemapping, tonemap_output_gamut, Tonemapping};

/// Registers the phase-2 contract resolver
/// ([`resolve_camera_stack_contracts`]), which turns the per-frame camera
/// stacks into per-view [`ViewStackContract`] components.
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

/// A view's role for one deferrable fullscreen pass within its camera stack.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StackRole {
    /// The view runs its own pass (if enabled). Solo views, members of
    /// non-deferring stack shapes, and pass-disabled views all carry this.
    Solo,
    /// The pass is deferred to the named finalizing view, which processes the
    /// composed buffer once. This view must not run the pass.
    Deferred(Entity),
    /// The view runs the pass once for the whole stack.
    Finalizer,
}

/// Whether a view's upscaling blit runs, and with which auto-detected blend.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlitDisposition {
    /// The blit runs. `force_replace` upgrades the auto-detected
    /// `ALPHA_BLENDING` (sorted index > 0) to replace; it is set for the
    /// first surviving blit of a stack whose earlier blits were skipped, and
    /// never overrides an explicit user `blend_state`.
    Run {
        /// Whether the auto-detected blend is upgraded to replace.
        force_replace: bool,
    },
    /// The view sits below its stack's finalizer (it defers a pass to the
    /// finalizer, or it is pass-disabled mid-stack); presenting the
    /// not-yet-finalized buffer would show un-tonemapped or un-encoded
    /// pixels, so the blit is skipped entirely and the finalizer's blit
    /// carries the composition.
    SkipDeferred,
}

/// Resolved display-encoding parameters for a view, after the prepare-time
/// transfer/gamut coercion chain (P3 -> Rec709 except under `ExtendedSrgb`,
/// scRGB forces Rec709, `ExtendedSrgb` forces Rec2020 -> Rec709, PQ forces
/// Rec2020).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ResolvedEncoding {
    /// The resolved display transfer function.
    pub transfer: DisplayTransfer,
    /// The resolved display gamut the encoder transforms to.
    pub gamut: DisplayGamut,
}

/// Per-view resolved composition state. The single source every
/// stack-sensitive prepare system reads; no consumer re-derives stack shape,
/// deferral, buffer space, or encoder inputs on its own.
///
/// Overwritten in place by [`resolve_camera_stack_contracts`] every frame and
/// never removed, so a view whose `ViewTarget` was dropped keeps a stale
/// contract. Consumers must therefore keep a `ViewTarget` term (or a
/// component gated on it) in their queries as the liveness gate; the
/// resolver's query requires `ViewTarget`, so a live contract always
/// corresponds to a view that holds one.
///
/// Not registered for reflection: the component is render-world internal, and
/// [`StackRole`] and [`BlitDisposition`] have no `Reflect` implementations.
#[derive(Component, Clone, Copy, PartialEq, Debug)]
pub struct ViewStackContract {
    /// Role of this view's tonemapping pass.
    pub tonemap: StackRole,
    /// Role of this view's display-encoding pass.
    pub encode: StackRole,
    /// Whether this view's upscaling blit runs.
    pub blit: BlitDisposition,
    /// The resolved compositing space of the buffer this view renders into
    /// (the phase-1 [`ResolvedCompositingSpace`] value, copied here so
    /// consumers need one component).
    pub compositing_space: Option<CompositingSpace>,
    /// Color primaries of the buffer at display-encoding time: the tonemap
    /// output gamut of the last tonemap-enabled member of the stack for
    /// deferred encodes, this view's own tonemap output gamut for solo
    /// encodes (each per-camera encode keys for its own region).
    pub source_gamut: DisplayGamut,
    /// Resolved encode parameters; `Some` exactly when the view's resolved
    /// display target requests an HDR transfer.
    pub encoding: Option<ResolvedEncoding>,
}

impl ViewStackContract {
    /// Whether the encoder's input buffer for this view uses Rec.2020 primaries —
    /// i.e. a GT7 view on an HDR-transfer target, whose pass emits native
    /// Rec.2020. A post-tonemap writer such as UI uses this to convert its
    /// Rec.709-authored colors to the buffer's primaries; `false` otherwise,
    /// including a `Tonemapping::None` view under a Rec.2020 working space (its
    /// `source_gamut` stays Rec.709 because no pass marks the buffer Rec.2020),
    /// where no post-tonemap conversion runs.
    pub fn source_gamut_is_rec2020(&self) -> bool {
        matches!(self.source_gamut, DisplayGamut::Rec2020)
    }
}

/// Per-view input to [`resolve_contracts`].
pub(crate) struct ContractInput<K> {
    pub entity: Entity,
    /// Identity of the main-texture ping-pong the view renders into; views
    /// resolve together only when they share it.
    pub texture: K,
    /// The camera's position in its render target's sorted camera order.
    pub sorted_index: usize,
    /// See [`composites_fullscreen`].
    pub composites_fullscreen: bool,
    /// Whether the view's display-encoding pass would run (the resolved
    /// display target requests an HDR transfer).
    pub encode_enabled: bool,
    /// Whether the camera writes to its render target at all
    /// (`CameraOutputMode::Skip` views never blit).
    pub output_writes: bool,
    /// Whether the camera carries an explicit user `blend_state`
    /// (`CameraOutputMode::Write { blend_state: Some(_) }`), which the blit
    /// disposition must never override.
    pub explicit_blend: bool,
    /// This view's own tonemap output gamut
    /// (`tonemap_output_gamut(own operator, own display target)`).
    pub tonemap_output_gamut: DisplayGamut,
    /// The phase-1 resolved compositing space, passed through to the
    /// contract.
    pub compositing_space: Option<CompositingSpace>,
    /// Whether the view's main pass loads the previous buffer contents
    /// (`ClearColorConfig::None`); distinguishes viewport members from
    /// clearing members in the stack-shape diagnostics.
    pub loads_previous: bool,
    /// The view's effective tone-mapping operator (`effective_tonemapping(..)`).
    /// [`Tonemapping::is_enabled`] gates the view's tonemapping pass; the
    /// operator is compared against the finalizer's for the operator-mismatch
    /// diagnostic.
    pub operator: Tonemapping,
}

/// Which resolver diagnostics fired for a view. The ECS layer reports each
/// as a `warn_once`; the pure core returns them so the table tests can
/// assert trigger conditions.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct ContractDiagnostics {
    /// The stack's tonemap deferral was cancelled because its HDR target's
    /// encode pass cannot defer past a viewport or clearing member.
    pub coherence_cancelled: bool,
    /// The stack encodes for an HDR transfer with no tonemapping pass
    /// anywhere in it.
    pub encode_without_tonemap: bool,
    /// A fullscreen `ClearColorConfig::None` member blits over regions whose
    /// passes ran per camera below it (double-processed presentation).
    pub fullscreen_blit_over_per_camera_passes: bool,
    /// The stack's first member loads the previous buffer contents
    /// (`ClearColorConfig::None`) while the stack runs a tonemapping or
    /// display-encoding pass, so each frame reprocesses last frame's
    /// already-processed output (feedback apps drift).
    pub frame_start_loads_processed_output: bool,
    /// `Some((own, finalizing))` when this deferred member's operator differs
    /// from its finalizer's.
    pub operator_mismatch: Option<(Tonemapping, Tonemapping)>,
}

/// Per-view output of [`resolve_contracts`]: the [`ViewStackContract`] fields
/// the pure core can decide (everything but the encode parameters, which
/// need the `ViewDisplayTarget`), plus the diagnostics that fired.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct ContractOutput {
    pub tonemap: StackRole,
    pub encode: StackRole,
    pub blit: BlitDisposition,
    pub compositing_space: Option<CompositingSpace>,
    pub source_gamut: DisplayGamut,
    pub stack_tonemaps: bool,
    pub diagnostics: ContractDiagnostics,
}

/// Resolves every view's stack roles, blit disposition, and encoder source
/// gamut, grouped by shared main texture.
///
/// Precondition: `sorted_index` values are unique within a texture group
/// (`sort_cameras` counts indices per render target, so distinct cameras on
/// one target never tie); the resolver defines no tie semantics.
pub(crate) fn resolve_contracts<K: Copy + Eq + Hash>(
    views: Vec<ContractInput<K>>,
) -> EntityHashMap<ContractOutput> {
    let mut groups: HashMap<K, Vec<ContractInput<K>>> = HashMap::default();
    for view in views {
        groups.entry(view.texture).or_default().push(view);
    }

    let mut outputs = EntityHashMap::default();
    for group in groups.values_mut() {
        group.sort_unstable_by_key(|view| view.sorted_index);
        debug_assert!(
            group
                .windows(2)
                .all(|pair| pair[0].sorted_index != pair[1].sorted_index),
            "sorted camera indices must be unique within a texture group"
        );
        resolve_group(group, &mut outputs);
    }
    outputs
}

/// Returns the index of the member that runs one fullscreen pass for the
/// whole sorted texture group, or `None` when the pass runs per camera.
///
/// The pass is deferred to the last enabled member if and only if there are
/// at least two enabled members and every enabled member after the first
/// composites fullscreen (loads the previous content and covers the whole
/// target with its output). Any other arrangement — clearing cameras,
/// viewport-scoped cameras — keeps the per-camera behavior, where each
/// camera's pass only feeds its own region of the final image.
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

/// Resolves one texture group of sorted members into [`ContractOutput`]s.
fn resolve_group<K>(members: &[ContractInput<K>], outputs: &mut EntityHashMap<ContractOutput>) {
    let encode_enabled_group = members.iter().any(|member| member.encode_enabled);
    let stack_tonemaps = members.iter().any(|member| member.operator.is_enabled());

    let mut tonemap_finalizer = pass_finalizer(members, |member| member.operator.is_enabled());
    let encode_finalizer = pass_finalizer(members, |member| member.encode_enabled);

    // Coherence: on an encode-enabled group the encode must defer whenever
    // the tonemap does, or a deferring member's own encode pass would run on
    // the not-yet-tonemapped buffer (encode-before-tonemap). The deferral
    // only checks the shape of ENABLED views, so a pass-disabled viewport
    // member is invisible to the tonemap shape test but shape-breaking for
    // the encode test; when that happens, tonemap deferral is cancelled and
    // every member tone-maps per camera. SDR groups (no encode pass) keep
    // tonemap deferral unconditionally.
    let coherence_cancelled =
        encode_enabled_group && tonemap_finalizer.is_some() && encode_finalizer.is_none();
    if coherence_cancelled {
        tonemap_finalizer = None;
    }

    // The finalizer is by construction the last enabled member, so an
    // enabled member below it defers to it and every other member runs solo.
    let role = |index: usize, enabled: bool, finalizer: Option<usize>| match finalizer {
        Some(f) if index == f => StackRole::Finalizer,
        Some(f) if enabled && index < f => StackRole::Deferred(members[f].entity),
        _ => StackRole::Solo,
    };

    // The gamut of the composed buffer a deferred encode reads: produced by
    // the LAST tonemap-enabled member in sorted order (not the tonemap
    // finalizer, which does not exist when the tonemap pass does not defer),
    // Rec.709 when nothing in the group tone-maps.
    let group_gamut = members
        .iter()
        .rev()
        .find(|member| member.operator.is_enabled())
        .map(|member| member.tonemap_output_gamut)
        .unwrap_or(DisplayGamut::Rec709);

    // The finalizer whose blit presents the whole composition: the
    // highest-index finalizer of either pass. Every member below it skips
    // its blit (presenting the un-finalized buffer would show un-tonemapped
    // or un-encoded pixels, and the lowest surviving blit would steal the
    // finalizer's replace). A `CameraOutputMode::Skip` finalizer never blits,
    // so skipping anyone for it would leave the target unpresented; members
    // then keep their blits.
    let presenting_finalizer = tonemap_finalizer
        .max(encode_finalizer)
        .filter(|&finalizer| members[finalizer].output_writes);

    let encode_without_tonemap = encode_enabled_group && !stack_tonemaps;

    // The ping-pong main texture persists across frames, so a stack whose
    // first member loads the previous buffer (`ClearColorConfig::None`)
    // starts the frame from last frame's tone-mapped (and, on HDR,
    // display-encoded) output and reprocesses it. Feedback/trail apps that
    // depend on this drift; the diagnostic only fires when a pass actually
    // runs over the group (a stack that neither tone-maps nor encodes leaves
    // the buffer scene-referred and accumulates stably).
    let frame_start_loads_processed_output =
        members.first().is_some_and(|first| first.loads_previous)
            && (stack_tonemaps || encode_enabled_group);

    // A member is shape-breaking when its output does not composite over the
    // whole target: any viewport member, or a clearing member that is not
    // the group's first (the first member is expected to clear; viewport-ness
    // is derived as "loads previous content but does not composite
    // fullscreen", so a clearing viewport first member counts as a normal
    // root).
    let shape_breaking = |index: usize, member: &ContractInput<K>| {
        if index == 0 {
            !member.composites_fullscreen && member.loads_previous
        } else {
            !member.composites_fullscreen
        }
    };
    // A member whose enabled pass runs per camera. When a pass has a
    // finalizer every enabled member defers to it or is it, so a member's
    // pass runs per camera exactly when it is enabled and the pass has no
    // finalizer at all.
    let runs_own_pass = |member: &ContractInput<K>| {
        (member.operator.is_enabled() && tonemap_finalizer.is_none())
            || (member.encode_enabled && encode_finalizer.is_none())
    };
    // A fullscreen `ClearColorConfig::None` member above a shape-breaking
    // member blits the WHOLE target, re-presenting regions whose passes ran
    // per camera below it (double-processed). The symmetric arrangement
    // (a viewport member above per-camera members) is a silent documented
    // limitation: any trigger for it would also fire on every ordinary
    // splitscreen, because a non-first member's clear is inert on the shared
    // attachment.
    let fullscreen_blit_over_per_camera_passes =
        members.iter().enumerate().any(|(index, overlay)| {
            overlay.composites_fullscreen
                && members[..index]
                    .iter()
                    .enumerate()
                    .any(|(below_index, below)| shape_breaking(below_index, below))
                && members[..index].iter().any(&runs_own_pass)
        });

    for (index, member) in members.iter().enumerate() {
        let tonemap = role(index, member.operator.is_enabled(), tonemap_finalizer);
        let encode = role(index, member.encode_enabled, encode_finalizer);

        let blit = match presenting_finalizer {
            Some(finalizer) if index < finalizer => BlitDisposition::SkipDeferred,
            Some(finalizer) if index == finalizer => BlitDisposition::Run {
                force_replace: !member.explicit_blend,
            },
            // No finalizer, or a member above it: the blit runs with its auto
            // `ALPHA_BLENDING` and composites over any earlier present.
            _ => BlitDisposition::Run {
                force_replace: false,
            },
        };

        let source_gamut = match encode {
            StackRole::Solo => member.tonemap_output_gamut,
            StackRole::Deferred(_) | StackRole::Finalizer => group_gamut,
        };

        let operator_mismatch = match (tonemap, tonemap_finalizer) {
            (StackRole::Deferred(_), Some(finalizer)) => {
                let finalizing = members[finalizer].operator;
                (member.operator != finalizing).then_some((member.operator, finalizing))
            }
            _ => None,
        };

        outputs.insert(
            member.entity,
            ContractOutput {
                tonemap,
                encode,
                blit,
                compositing_space: member.compositing_space,
                source_gamut,
                stack_tonemaps,
                diagnostics: ContractDiagnostics {
                    coherence_cancelled,
                    encode_without_tonemap,
                    fullscreen_blit_over_per_camera_passes,
                    frame_start_loads_processed_output,
                    operator_mismatch,
                },
            },
        );
    }
}

/// Resolves every camera view's stack into a [`ViewStackContract`].
///
/// Groups views by `ViewTarget::main_texture().id()` and orders each group by
/// `sorted_camera_index_for_target`, exactly as the tonemapping and
/// display-encoding prepare systems group their deferrals, so it runs in
/// [`RenderSystems::PrepareViews`] after `prepare_view_targets` (the
/// `ViewTarget` source) and `prepare_view_display_targets` (the
/// `ViewDisplayTarget` source). Phase-2 texture groups never span phase-1
/// [`ResolvedCompositingSpace`] groups: equal main-texture ids imply equal
/// main-texture keys (`prepare_view_targets` dedups allocations on exactly
/// that key).
///
/// The stack rules live on `resolve_contracts`; this system feeds it,
/// resolves the encode parameters per group (the coercion chain over the
/// group's shared `ViewDisplayTarget`), reports the diagnostics as
/// `warn_once`s, and inserts the contracts. A missing [`ViewDisplayTarget`]
/// counts as a plain SDR target.
pub fn resolve_camera_stack_contracts(
    mut commands: Commands,
    views: Query<(
        Entity,
        &ExtractedCamera,
        &ViewTarget,
        Option<&ViewDisplayTarget>,
        Option<&Tonemapping>,
        Option<&ResolvedCompositingSpace>,
    )>,
    working_color_space: Res<WorkingColorSpace>,
) {
    // Encode parameters resolve once per texture group: members share one
    // `ViewDisplayTarget` (it resolves per render target, and the target is
    // part of the texture grouping key), so transfer and gamut are uniform
    // across a group. Resolving from the first-iterated member's
    // `ViewDisplayTarget` therefore matches the per-member `encode_enabled`
    // used by `resolve_contracts` (both read `is_hdr_transfer()` off the same
    // shared target). The shared-target invariant holds in every supported
    // configuration; it can break only in the pathological plugin order where
    // a camera view momentarily lacks its `ViewDisplayTarget` (treated as
    // plain SDR, out of scope per the spec). The `debug_assert!` makes that
    // divergence loud in debug builds rather than letting it silently mix an
    // encoded group with an unencoded out-texture clear.
    let mut group_encodings: HashMap<TextureId, Option<ResolvedEncoding>> = HashMap::default();
    let mut inputs = Vec::new();
    for (entity, camera, view_target, view_display_target, tonemapping, resolved_space) in &views {
        let texture = view_target.main_texture().id();
        let encode_enabled = view_display_target.is_some_and(ViewDisplayTarget::is_hdr_transfer);
        let encoding = *group_encodings.entry(texture).or_insert_with(|| {
            resolve_group_encode_parameters(view_display_target, view_target, *working_color_space)
        });
        debug_assert_eq!(
            encoding.is_some(),
            encode_enabled,
            "every member of a texture group must agree on display-encode enablement"
        );
        inputs.push(ContractInput {
            entity,
            texture,
            sorted_index: camera.sorted_camera_index_for_target,
            composites_fullscreen: composites_fullscreen(camera),
            encode_enabled,
            output_writes: !matches!(camera.output_mode, CameraOutputMode::Skip),
            explicit_blend: matches!(
                camera.output_mode,
                CameraOutputMode::Write {
                    blend_state: Some(_),
                    ..
                }
            ),
            tonemap_output_gamut: tonemap_output_gamut(tonemapping, view_display_target),
            compositing_space: resolved_space.and_then(|space| space.0),
            loads_previous: matches!(camera.clear_color, ClearColorConfig::None),
            operator: effective_tonemapping(tonemapping, view_display_target),
        });
    }

    let outputs = resolve_contracts(inputs);

    for (entity, _, view_target, ..) in &views {
        let Some(output) = outputs.get(&entity) else {
            continue;
        };
        emit_contract_diagnostics(&output.diagnostics);

        let encoding = group_encodings
            .get(&view_target.main_texture().id())
            .copied()
            .flatten();

        commands.entity(entity).insert(ViewStackContract {
            tonemap: output.tonemap,
            encode: output.encode,
            blit: output.blit,
            compositing_space: output.compositing_space,
            source_gamut: output.source_gamut,
            encoding,
        });
    }
}

/// Resolves a texture group's display transfer and gamut: the group-level
/// display-target diagnostics plus [`coerce_display_encode`]. Returns `None`
/// for groups whose resolved display target does not request an HDR transfer
/// (no encode pass; a missing [`ViewDisplayTarget`] counts as plain SDR).
fn resolve_group_encode_parameters(
    view_display_target: Option<&ViewDisplayTarget>,
    view_target: &ViewTarget,
    working_color_space: WorkingColorSpace,
) -> Option<ResolvedEncoding> {
    let view_display_target = view_display_target?;
    if !view_display_target.is_hdr_transfer() {
        return None;
    }

    // Window surfaces only negotiate HDR transfers onto formats without a
    // hardware sRGB encode, but manual Image/TextureView targets resolve
    // their ManualDisplayTargets entry verbatim — the user owns the texture.
    // Writing the encoded signal through an sRGB view would encode it a
    // second time.
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

    // HDR display output reaches noticeably wider gamuts when the scene is
    // rendered in the Rec.2020 working space. This is advisory only (the
    // Rec.709 working space remains correct, just gamut-limited); a global
    // axis must never flip automatically because one window went HDR.
    if !working_color_space.is_rec2020() {
        warn_once!(
            "A camera is rendering to an HDR display target while the working color \
            space is the default `WorkingColorSpace::Rec709`. Output is correct but \
            limited to the Rec.709 gamut; consider opting into the wide working space \
            with `RenderPlugin {{ working_color_space: WorkingColorSpace::Rec2020, .. }}`."
        );
    }

    let target = view_display_target.resolved;
    let (transfer, gamut) = coerce_display_encode(target.transfer, target.gamut);
    Some(ResolvedEncoding { transfer, gamut })
}

/// The prepare-time display-encode coercion chain over the resolved transfer
/// and gamut: `DisplayP3` -> Rec709 for every transfer except `ExtendedSrgb`
/// (which keeps P3 — wgpu's `ExtendedDisplayP3`), then scRGB forces Rec709,
/// then `ExtendedSrgb` forces Rec2020 -> Rec709 (no extended Rec.2020 surface),
/// then PQ forces Rec2020. The order is load bearing: the P3 -> 709 arm runs
/// before the transfer-specific gamut arms. Each arm reports the coercion it
/// applies as a `warn_once` / `info_once`; the result depends on nothing but
/// the arguments, so the chain is tested directly.
pub(crate) fn coerce_display_encode(
    transfer: DisplayTransfer,
    gamut: DisplayGamut,
) -> (DisplayTransfer, DisplayGamut) {
    let mut gamut = gamut;
    // Display-P3 is a real encoder gamut only for the encoded extended-range
    // sRGB transfer (wgpu's `ExtendedDisplayP3` surface color space); every
    // other transfer ships no P3 surface and collapses it to Rec.709.
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
        // scRGB-linear (IEC 61966-2-2) is *definitionally* encoded against
        // Rec.709/sRGB primaries: every backend that negotiates the
        // Rgba16Float surface declares it as extended-sRGB-linear, and the
        // OS compositor performs the mapping to the panel's physical gamut
        // itself. Wide gamut rides scRGB's out-of-range (including negative)
        // component values, never a change of primaries — re-coordinatizing
        // into Rec.2020 here would be interpreted as Rec.709 by the
        // compositor and desaturate every pixel.
        info_once!(
            "scRGB-linear signals are always expressed in (extended) Rec.709/sRGB \
            coordinates (the OS compositor performs the mapping to the panel's gamut); \
            ignoring `DisplayTarget::gamut` ({gamut:?}) for encoding. The field still \
            correctly describes the panel for luminance/metadata purposes."
        );
        gamut = DisplayGamut::Rec709;
    }
    // Encoded extended-range sRGB has no Rec.2020 surface (only `ExtendedSrgb`
    // / `ExtendedDisplayP3`), so a Rec.2020 gamut falls back to Rec.709; a
    // Display-P3 gamut is kept (handled above).
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

/// Reports the resolver diagnostics that fired for one view.
fn emit_contract_diagnostics(diagnostics: &ContractDiagnostics) {
    if diagnostics.coherence_cancelled {
        warn_once!(
            "Tone mapping cannot be deferred to the last camera of a stack rendering to \
            an HDR display target, because a viewport-scoped or clearing camera prevents \
            deferring the display encoding with it. Each camera tone-maps its own pass \
            instead, so the composed buffer is not tone-mapped as one image. Give every \
            overlay camera fullscreen `ClearColorConfig::None` composition, or move the \
            viewport camera to its own render target."
        );
    }
    if diagnostics.encode_without_tonemap {
        warn_once!(
            "No camera rendering to an HDR display target has an active tone-mapping \
            operator, so scene-linear values are transfer-encoded without tone mapping. \
            Add an operator like `Tonemapping::GranTurismo7`, or `Tonemapping::Linear` to \
            encode with no tone curve; raw pass-through is intentional only for debug or \
            calibration passes."
        );
    }
    if diagnostics.fullscreen_blit_over_per_camera_passes {
        warn_once!(
            "A fullscreen `ClearColorConfig::None` camera composites above viewport-scoped \
            or clearing cameras whose tone-mapping or display-encoding passes run per \
            camera; its full-target blit re-presents their already-processed pixels \
            (double-processed). Give the overlay camera its own render target."
        );
    }
    if diagnostics.frame_start_loads_processed_output {
        warn_once!(
            "The first camera rendering to a target uses `ClearColorConfig::None` while \
            its stack runs a tone-mapping or display-encoding pass. The main texture \
            persists across frames, so each frame starts from last frame's tone-mapped \
            (and, on an HDR target, display-encoded) output and reprocesses it; \
            feedback/trail effects built this way drift over time. Stable feedback \
            accumulation needs `Tonemapping::None` on an SDR target."
        );
    }
    if let Some((own, finalizing)) = diagnostics.operator_mismatch {
        warn_once!(
            "Stacked cameras rendering to the same target use different tone-mapping \
            operators ({own:?} and {finalizing:?}). The stack is composed in scene-linear \
            space and tone-mapped once, by the last camera, so its operator, ColorGrading, \
            GranTurismo7Params, and DebandDither settings apply to the whole stack."
        );
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

    /// An SDR member with an active operator that clears its target; tests
    /// override the fields each case exercises.
    fn clearing(raw: u32, index: usize) -> ContractInput<u32> {
        ContractInput {
            entity: entity(raw),
            texture: 0,
            sorted_index: index,
            composites_fullscreen: false,
            encode_enabled: false,
            output_writes: true,
            explicit_blend: false,
            tonemap_output_gamut: DisplayGamut::Rec709,
            compositing_space: None,
            loads_previous: false,
            operator: Tonemapping::TonyMcMapface,
        }
    }

    /// A fullscreen `ClearColorConfig::None` member.
    fn compositing(raw: u32, index: usize) -> ContractInput<u32> {
        let mut input = clearing(raw, index);
        input.composites_fullscreen = true;
        input.loads_previous = true;
        input
    }

    /// A viewport-scoped member that loads previous content.
    fn viewport(raw: u32, index: usize) -> ContractInput<u32> {
        let mut input = clearing(raw, index);
        input.composites_fullscreen = false;
        input.loads_previous = true;
        input
    }

    /// Marks a member as GT7 on an HDR (PQ) target.
    fn gt7_hdr(mut input: ContractInput<u32>) -> ContractInput<u32> {
        input.encode_enabled = true;
        input.operator = Tonemapping::GranTurismo7;
        input.tonemap_output_gamut = DisplayGamut::Rec2020;
        input
    }

    /// Marks a member as `Tonemapping::None` on an HDR (PQ) target.
    fn passthrough_hdr(mut input: ContractInput<u32>) -> ContractInput<u32> {
        input.encode_enabled = true;
        input.operator = Tonemapping::None;
        input.tonemap_output_gamut = DisplayGamut::Rec709;
        input
    }

    /// Marks a member as `Tonemapping::None` on an SDR target.
    fn disabled(mut input: ContractInput<u32>) -> ContractInput<u32> {
        input.operator = Tonemapping::None;
        input
    }

    fn output(outputs: &EntityHashMap<ContractOutput>, raw: u32) -> ContractOutput {
        *outputs
            .get(&entity(raw))
            .expect("view must have a contract")
    }

    fn assert_silent(output: &ContractOutput) {
        assert_eq!(output.diagnostics, ContractDiagnostics::default());
    }

    // E1: a solo camera runs everything itself with no diagnostics.
    #[test]
    fn solo_camera_is_solo_everywhere() {
        let outputs = resolve_contracts(vec![clearing(1, 0)]);
        let solo = output(&outputs, 1);
        assert_eq!(solo.tonemap, StackRole::Solo);
        assert_eq!(solo.encode, StackRole::Solo);
        assert_eq!(solo.blit, RUN);
        assert_eq!(solo.source_gamut, DisplayGamut::Rec709);
        assert!(solo.stack_tonemaps);
        assert_eq!(solo.compositing_space, None);
        assert_silent(&solo);
    }

    // E3 (canonical S1): a GT7 base with a pass-through HDR overlay defers
    // only the encode; the deferred encode's source gamut is the LAST
    // tonemap-enabled member's (the base), not the finalizer's own.
    #[test]
    fn gt7_base_with_passthrough_overlay_defers_encode_only() {
        let outputs = resolve_contracts(vec![
            gt7_hdr(clearing(1, 0)),
            passthrough_hdr(compositing(2, 1)),
        ]);
        let base = output(&outputs, 1);
        let overlay = output(&outputs, 2);
        assert_eq!(base.tonemap, StackRole::Solo);
        assert_eq!(overlay.tonemap, StackRole::Solo);
        assert_eq!(base.encode, StackRole::Deferred(entity(2)));
        assert_eq!(overlay.encode, StackRole::Finalizer);
        assert_eq!(base.source_gamut, DisplayGamut::Rec2020);
        assert_eq!(overlay.source_gamut, DisplayGamut::Rec2020);
        assert_eq!(base.blit, BlitDisposition::SkipDeferred);
        assert_eq!(overlay.blit, RUN_REPLACE);
        assert!(base.stack_tonemaps);
        assert_silent(&base);
        assert_silent(&overlay);
    }

    // E4: an all-pass-through stack on an HDR target defers the encode with
    // the Rec.709 fallback gamut and flags the missing tone mapping.
    #[test]
    fn all_passthrough_hdr_stack_flags_encode_without_tonemap() {
        let outputs = resolve_contracts(vec![
            passthrough_hdr(clearing(1, 0)),
            passthrough_hdr(compositing(2, 1)),
        ]);
        let base = output(&outputs, 1);
        let overlay = output(&outputs, 2);
        assert_eq!(base.tonemap, StackRole::Solo);
        assert_eq!(overlay.tonemap, StackRole::Solo);
        assert_eq!(base.encode, StackRole::Deferred(entity(2)));
        assert_eq!(overlay.encode, StackRole::Finalizer);
        assert_eq!(base.source_gamut, DisplayGamut::Rec709);
        assert_eq!(overlay.source_gamut, DisplayGamut::Rec709);
        assert!(!base.stack_tonemaps);
        assert!(base.diagnostics.encode_without_tonemap);
        assert!(overlay.diagnostics.encode_without_tonemap);
        assert!(!base.diagnostics.coherence_cancelled);
    }

    // E5: a pass-disabled viewport member is invisible to the tonemap shape
    // test but shape-breaking for the encode test; tonemap deferral is
    // cancelled for the whole group and every member runs per camera.
    #[test]
    fn viewport_member_cancels_tonemap_deferral() {
        let outputs = resolve_contracts(vec![
            gt7_hdr(clearing(1, 0)),
            gt7_hdr(compositing(2, 1)),
            passthrough_hdr(viewport(3, 2)),
        ]);
        for raw in 1..=3 {
            let member = output(&outputs, raw);
            assert_eq!(member.tonemap, StackRole::Solo);
            assert_eq!(member.encode, StackRole::Solo);
            assert_eq!(member.blit, RUN);
            assert!(member.diagnostics.coherence_cancelled);
            assert!(!member.diagnostics.fullscreen_blit_over_per_camera_passes);
        }
        // Solo encodes key for their own pass output.
        assert_eq!(output(&outputs, 1).source_gamut, DisplayGamut::Rec2020);
        assert_eq!(output(&outputs, 2).source_gamut, DisplayGamut::Rec2020);
        assert_eq!(output(&outputs, 3).source_gamut, DisplayGamut::Rec709);
    }

    // Coherence negative control: when both passes can defer, both do, and
    // nothing is cancelled.
    #[test]
    fn coherent_hdr_stack_defers_both_passes() {
        let outputs = resolve_contracts(vec![gt7_hdr(clearing(1, 0)), gt7_hdr(compositing(2, 1))]);
        let base = output(&outputs, 1);
        let overlay = output(&outputs, 2);
        assert_eq!(base.tonemap, StackRole::Deferred(entity(2)));
        assert_eq!(base.encode, StackRole::Deferred(entity(2)));
        assert_eq!(overlay.tonemap, StackRole::Finalizer);
        assert_eq!(overlay.encode, StackRole::Finalizer);
        assert_eq!(base.blit, BlitDisposition::SkipDeferred);
        assert_eq!(overlay.blit, RUN_REPLACE);
        assert_silent(&base);
        assert_silent(&overlay);
    }

    // E6: an SDR stack keeps tonemap deferral (no encode pass exists to
    // cohere with) and the finalizer's blit carries the composition.
    #[test]
    fn sdr_stack_keeps_tonemap_deferral_and_skips_deferred_blit() {
        let outputs = resolve_contracts(vec![clearing(1, 0), compositing(2, 1)]);
        let base = output(&outputs, 1);
        let overlay = output(&outputs, 2);
        assert_eq!(base.tonemap, StackRole::Deferred(entity(2)));
        assert_eq!(overlay.tonemap, StackRole::Finalizer);
        assert_eq!(base.encode, StackRole::Solo);
        assert_eq!(overlay.encode, StackRole::Solo);
        assert_eq!(base.blit, BlitDisposition::SkipDeferred);
        assert_eq!(overlay.blit, RUN_REPLACE);
        assert!(base.stack_tonemaps);
        assert_silent(&base);
        assert_silent(&overlay);
    }

    // Three enabled members over one texture defer to the last.
    #[test]
    fn three_member_stack_defers_to_the_last() {
        let outputs = resolve_contracts(vec![clearing(1, 0), compositing(2, 1), compositing(3, 2)]);
        assert_eq!(output(&outputs, 1).tonemap, StackRole::Deferred(entity(3)));
        assert_eq!(output(&outputs, 2).tonemap, StackRole::Deferred(entity(3)));
        assert_eq!(output(&outputs, 3).tonemap, StackRole::Finalizer);
    }

    // A shape-breaking member in the MIDDLE of the enabled set suppresses
    // deferral for the whole group: the predicate scans every enabled member
    // after the first, not just the last.
    #[test]
    fn shape_breaking_middle_member_suppresses_deferral() {
        let outputs = resolve_contracts(vec![compositing(1, 0), viewport(2, 1), compositing(3, 2)]);
        for raw in 1..=3 {
            assert_eq!(output(&outputs, raw).tonemap, StackRole::Solo);
        }
    }

    // E7: viewport splitscreen keeps per-camera passes and per-view source
    // gamuts, silently.
    #[test]
    fn viewport_splitscreen_keeps_per_camera_passes() {
        // The first split-screen camera establishes the frame by clearing its
        // target; a `ClearColorConfig::None` first member would instead load
        // last frame's processed output and trip the frame-start diagnostic.
        let mut left = viewport(1, 0);
        left.loads_previous = false;
        left.tonemap_output_gamut = DisplayGamut::Rec709;
        let mut right = viewport(2, 1);
        right.tonemap_output_gamut = DisplayGamut::Rec2020;
        let outputs = resolve_contracts(vec![left, right]);
        let left = output(&outputs, 1);
        let right = output(&outputs, 2);
        assert_eq!(left.tonemap, StackRole::Solo);
        assert_eq!(right.tonemap, StackRole::Solo);
        assert_eq!(left.blit, RUN);
        assert_eq!(right.blit, RUN);
        assert_eq!(left.source_gamut, DisplayGamut::Rec709);
        assert_eq!(right.source_gamut, DisplayGamut::Rec2020);
        assert_silent(&left);
        assert_silent(&right);
    }

    // E8: the sorted index orders the stack regardless of input order, so
    // mixed-Hdr stacks with deterministic per-target indices defer and blit
    // deterministically.
    #[test]
    fn sorted_index_orders_roles_not_insertion_order() {
        let outputs = resolve_contracts(vec![compositing(2, 1), clearing(1, 0)]);
        assert_eq!(output(&outputs, 1).tonemap, StackRole::Deferred(entity(2)));
        assert_eq!(output(&outputs, 2).tonemap, StackRole::Finalizer);
        assert_eq!(output(&outputs, 1).blit, BlitDisposition::SkipDeferred);
        assert_eq!(output(&outputs, 2).blit, RUN_REPLACE);
    }

    // E12: a pass-disabled member below the finalizer is Solo (never
    // deferred, never a finalizer) but its blit would present the
    // un-finalized buffer, so it skips too.
    #[test]
    fn disabled_member_below_finalizer_skips_blit() {
        let outputs = resolve_contracts(vec![
            clearing(1, 0),
            disabled(compositing(2, 1)),
            compositing(3, 2),
        ]);
        let base = output(&outputs, 1);
        let middle = output(&outputs, 2);
        let finalizer = output(&outputs, 3);
        assert_eq!(base.tonemap, StackRole::Deferred(entity(3)));
        assert_eq!(middle.tonemap, StackRole::Solo);
        assert_eq!(finalizer.tonemap, StackRole::Finalizer);
        assert_eq!(base.blit, BlitDisposition::SkipDeferred);
        assert_eq!(middle.blit, BlitDisposition::SkipDeferred);
        assert_eq!(finalizer.blit, RUN_REPLACE);
    }

    // A pass-disabled overlay ABOVE the finalizer keeps its auto
    // alpha-blended blit and composites over the finalizer's present.
    #[test]
    fn disabled_member_above_finalizer_keeps_alpha_blit() {
        let outputs = resolve_contracts(vec![
            clearing(1, 0),
            compositing(2, 1),
            disabled(compositing(3, 2)),
        ]);
        let base = output(&outputs, 1);
        let finalizer = output(&outputs, 2);
        let overlay = output(&outputs, 3);
        assert_eq!(base.tonemap, StackRole::Deferred(entity(2)));
        assert_eq!(finalizer.tonemap, StackRole::Finalizer);
        assert_eq!(overlay.tonemap, StackRole::Solo);
        assert_eq!(base.blit, BlitDisposition::SkipDeferred);
        assert_eq!(finalizer.blit, RUN_REPLACE);
        assert_eq!(overlay.blit, RUN);
    }

    // E13: a `CameraOutputMode::Skip` finalizer never blits, so nobody
    // skips for it; deferral roles are unaffected.
    #[test]
    fn skip_finalizer_cancels_blit_skipping() {
        let mut finalizer = compositing(2, 1);
        finalizer.output_writes = false;
        let outputs = resolve_contracts(vec![clearing(1, 0), finalizer]);
        assert_eq!(output(&outputs, 1).tonemap, StackRole::Deferred(entity(2)));
        assert_eq!(output(&outputs, 2).tonemap, StackRole::Finalizer);
        assert_eq!(output(&outputs, 1).blit, RUN);
        assert_eq!(output(&outputs, 2).blit, RUN);
    }

    // An explicit user blend_state is never overridden by force_replace;
    // members below the finalizer still skip.
    #[test]
    fn explicit_blend_is_never_overridden() {
        let mut finalizer = compositing(2, 1);
        finalizer.explicit_blend = true;
        let outputs = resolve_contracts(vec![clearing(1, 0), finalizer]);
        assert_eq!(output(&outputs, 1).blit, BlitDisposition::SkipDeferred);
        assert_eq!(output(&outputs, 2).blit, RUN);
    }

    // E15: a fullscreen None-clear overlay above viewport cameras with
    // enabled per-camera passes re-presents their processed regions; the
    // configuration is diagnosed, behavior unchanged (all Solo).
    #[test]
    fn fullscreen_overlay_above_viewport_cameras_is_flagged() {
        let outputs = resolve_contracts(vec![
            viewport(1, 0),
            viewport(2, 1),
            disabled(compositing(3, 2)),
        ]);
        for raw in 1..=3 {
            let member = output(&outputs, raw);
            assert_eq!(member.tonemap, StackRole::Solo);
            assert_eq!(member.blit, RUN);
            assert!(member.diagnostics.fullscreen_blit_over_per_camera_passes);
        }
    }

    // E17: the symmetric arrangement (an enabled viewport member above
    // enabled members) is a silent documented limitation.
    #[test]
    fn viewport_above_enabled_members_is_silent() {
        let outputs = resolve_contracts(vec![clearing(1, 0), viewport(2, 1)]);
        let base = output(&outputs, 1);
        let pip = output(&outputs, 2);
        assert_eq!(base.tonemap, StackRole::Solo);
        assert_eq!(pip.tonemap, StackRole::Solo);
        assert_silent(&base);
        assert_silent(&pip);
    }

    // E18: a disabled clearing member mid-stack breaks the phase-1 stack
    // shape but not the enabled-only deferral; its inert clear leaves the
    // deferral's intent intact, the blits below the finalizer skip, and the
    // overlay-over-per-camera diagnostic stays quiet (nothing below the
    // finalizer runs a per-camera pass).
    #[test]
    fn divergent_stack_defers_and_skips_blits() {
        let mut top = compositing(3, 2);
        top.compositing_space = Some(CompositingSpace::Oklab);
        let outputs = resolve_contracts(vec![clearing(1, 0), disabled(clearing(2, 1)), top]);
        let base = output(&outputs, 1);
        let middle = output(&outputs, 2);
        let finalizer = output(&outputs, 3);
        assert_eq!(base.tonemap, StackRole::Deferred(entity(3)));
        assert_eq!(middle.tonemap, StackRole::Solo);
        assert_eq!(finalizer.tonemap, StackRole::Finalizer);
        assert_eq!(base.blit, BlitDisposition::SkipDeferred);
        assert_eq!(middle.blit, BlitDisposition::SkipDeferred);
        assert_eq!(finalizer.blit, RUN_REPLACE);
        // The phase-1 resolution passes through per view.
        assert_eq!(base.compositing_space, None);
        assert_eq!(finalizer.compositing_space, Some(CompositingSpace::Oklab));
        assert_silent(&base);
        assert_silent(&middle);
        assert_silent(&finalizer);
    }

    // W13: a deferred member whose operator differs from its finalizer's is
    // flagged with both operators; the finalizer itself is not.
    #[test]
    fn operator_mismatch_is_flagged_on_the_deferred_member() {
        let mut base = clearing(1, 0);
        base.operator = Tonemapping::AcesFitted;
        let outputs = resolve_contracts(vec![base, compositing(2, 1)]);
        assert_eq!(
            output(&outputs, 1).diagnostics.operator_mismatch,
            Some((Tonemapping::AcesFitted, Tonemapping::TonyMcMapface))
        );
        assert_eq!(output(&outputs, 2).diagnostics.operator_mismatch, None);
    }

    // Matching operators stay silent (negative control for the mismatch
    // diagnostic).
    #[test]
    fn matching_stack_members_are_silent() {
        let outputs = resolve_contracts(vec![clearing(1, 0), compositing(2, 1)]);
        assert_silent(&output(&outputs, 1));
        assert_silent(&output(&outputs, 2));
    }

    // The deferred encode keys on the LAST tonemap-enabled member's gamut in
    // sorted order, not the first's and not the encode finalizer's own.
    #[test]
    fn last_tonemap_enabled_member_sets_the_deferred_source_gamut() {
        let mut base = gt7_hdr(clearing(1, 0));
        base.tonemap_output_gamut = DisplayGamut::Rec2020;
        let mut middle = gt7_hdr(compositing(2, 1));
        middle.operator = Tonemapping::Linear;
        middle.tonemap_output_gamut = DisplayGamut::Rec709;
        let top = passthrough_hdr(compositing(3, 2));
        let outputs = resolve_contracts(vec![base, middle, top]);
        for raw in 1..=3 {
            assert_eq!(output(&outputs, raw).source_gamut, DisplayGamut::Rec709);
        }
        assert_eq!(output(&outputs, 3).encode, StackRole::Finalizer);
    }

    // E14: a solo camera that loads the previous buffer
    // (`ClearColorConfig::None`) while running a tone-mapping pass reprocesses
    // last frame's output every frame; the diagnostic fires.
    #[test]
    fn frame_start_load_with_tonemapping_is_flagged() {
        let outputs = resolve_contracts(vec![compositing(1, 0)]);
        let solo = output(&outputs, 1);
        assert!(solo.diagnostics.frame_start_loads_processed_output);
    }

    // The same load-previous solo camera with display encoding (HDR target)
    // but no tone-mapping pass still reprocesses last frame's encoded output.
    #[test]
    fn frame_start_load_with_encode_only_is_flagged() {
        let outputs = resolve_contracts(vec![passthrough_hdr(compositing(1, 0))]);
        let solo = output(&outputs, 1);
        assert!(solo.diagnostics.frame_start_loads_processed_output);
    }

    // E14 negative: a solo camera that CLEARS its target starts each frame
    // fresh, so the diagnostic stays quiet even with tone mapping enabled.
    #[test]
    fn clearing_solo_camera_does_not_flag_frame_start_load() {
        let outputs = resolve_contracts(vec![clearing(1, 0)]);
        let solo = output(&outputs, 1);
        assert!(!solo.diagnostics.frame_start_loads_processed_output);
    }

    // A load-previous stack that neither tone-maps nor encodes leaves the
    // buffer scene-referred, so feedback accumulates stably and the
    // diagnostic stays quiet.
    #[test]
    fn frame_start_load_without_passes_is_silent() {
        let outputs = resolve_contracts(vec![disabled(compositing(1, 0))]);
        let solo = output(&outputs, 1);
        assert!(!solo.diagnostics.frame_start_loads_processed_output);
    }

    // Views on different textures resolve independently.
    #[test]
    fn separate_textures_resolve_independently() {
        let mut other = compositing(2, 0);
        other.texture = 1;
        let outputs = resolve_contracts(vec![clearing(1, 0), other]);
        assert_eq!(output(&outputs, 1).tonemap, StackRole::Solo);
        assert_eq!(output(&outputs, 2).tonemap, StackRole::Solo);
        assert_eq!(output(&outputs, 1).blit, RUN);
        assert_eq!(output(&outputs, 2).blit, RUN);
    }
}

#[cfg(test)]
mod coercion_tests {
    use super::*;

    // scRGB-linear with the canonical Rec.709 gamut passes through unchanged.
    #[test]
    fn scrgb_rec709_is_unchanged() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::ScRgbLinear, DisplayGamut::Rec709),
            (DisplayTransfer::ScRgbLinear, DisplayGamut::Rec709)
        );
    }

    // scRGB-linear forces Rec.709 because the signal is definitionally
    // expressed in extended Rec.709/sRGB coordinates.
    #[test]
    fn scrgb_forces_rec709() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::ScRgbLinear, DisplayGamut::Rec2020),
            (DisplayTransfer::ScRgbLinear, DisplayGamut::Rec709)
        );
    }

    // PQ forces Rec.2020 (canonically Rec.2020 / ITU-R BT.2100).
    #[test]
    fn pq_forces_rec2020() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::Pq, DisplayGamut::Rec709),
            (DisplayTransfer::Pq, DisplayGamut::Rec2020)
        );
    }

    // PQ with its canonical Rec.2020 gamut passes through unchanged.
    #[test]
    fn pq_rec2020_is_unchanged() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::Pq, DisplayGamut::Rec2020),
            (DisplayTransfer::Pq, DisplayGamut::Rec2020)
        );
    }

    // DisplayP3 collapses to Rec.709 (no P3 gamut matrix ships); the scRGB
    // transfer is left intact.
    #[test]
    fn display_p3_collapses_to_rec709() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::ScRgbLinear, DisplayGamut::DisplayP3),
            (DisplayTransfer::ScRgbLinear, DisplayGamut::Rec709)
        );
    }

    // The sRGB transfer never reaches the chain via an HDR target, but the
    // pure coercion leaves a non-HDR transfer untouched.
    #[test]
    fn srgb_is_unchanged() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::Srgb, DisplayGamut::Rec709),
            (DisplayTransfer::Srgb, DisplayGamut::Rec709)
        );
    }

    // The encoded extended-range sRGB transfer is the one transfer that keeps a
    // Display-P3 gamut (wgpu's `ExtendedDisplayP3` surface color space): the
    // P3 -> Rec.709 collapse is gated off for it.
    #[test]
    fn extended_srgb_keeps_display_p3() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::ExtendedSrgb, DisplayGamut::DisplayP3),
            (DisplayTransfer::ExtendedSrgb, DisplayGamut::DisplayP3)
        );
    }

    // Extended-range sRGB at Rec.709 is already canonical and untouched.
    #[test]
    fn extended_srgb_keeps_rec709() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::ExtendedSrgb, DisplayGamut::Rec709),
            (DisplayTransfer::ExtendedSrgb, DisplayGamut::Rec709)
        );
    }

    // There is no encoded-extended Rec.2020 surface, so a Rec.2020 gamut under
    // the extended-sRGB transfer falls back to Rec.709.
    #[test]
    fn extended_srgb_rec2020_falls_back_to_rec709() {
        assert_eq!(
            coerce_display_encode(DisplayTransfer::ExtendedSrgb, DisplayGamut::Rec2020),
            (DisplayTransfer::ExtendedSrgb, DisplayGamut::Rec709)
        );
    }
}
