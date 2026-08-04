//! Phase-1 resolution of per-camera [`CompositingSpace`] requests.
//!
//! [`CompositingSpace`] is a per-camera request, and absent means linear.
//! Cameras rendering to the same target share one main-texture ping-pong (see
//! [`prepare_view_targets`](super::prepare_view_targets)), and that buffer can
//! only hold one space at a time when its cameras composite over each other.
//! [`resolve_composition_spaces`] groups views by the shared-main-texture key
//! and resolves one space per compositing stack into each view's
//! [`ResolvedCompositingSpace`].
//!
//! These phase-1 groups are a superset of the phase-2 texture groups in
//! `bevy_core_pipeline`. `prepare_view_targets` dedups allocations on the
//! main-texture key, so equal main-texture ids imply equal `MainTextureKey`s
//! and a phase-2 group never spans two phase-1 groups. If the phases disagree
//! on shape, the views resolve per view and warn rather than mismatching
//! across groups.

use bevy_camera::{Camera2d, CameraMainTextureUsages, ClearColorConfig, CompositingSpace};
use bevy_ecs::{
    change_detection::DetectChangesMut,
    component::Component,
    entity::{Entity, EntityHashMap},
    query::Has,
    system::Query,
};
use bevy_log::warn_once;
use bevy_platform::collections::HashMap;
use core::hash::Hash;
use wgpu::TextureFormat;

use super::{main_texture_key, ExtractedView, MainTextureKey, Msaa};
use crate::camera::ExtractedCamera;

/// A camera view's per-frame resolved compositing space.
///
/// Camera extraction seeds this with the camera's raw request.
/// [`resolve_composition_spaces`] then overwrites it in
/// [`RenderSystems::CreateViews`](crate::RenderSystems::CreateViews), after
/// `sort_cameras`. Views that share a main-texture ping-pong and form a
/// compositing stack (every later member uses `ClearColorConfig::None` with no
/// viewport) resolve to one shared space. Solo views and other group shapes
/// keep their own request.
///
/// Read this component, or the downstream `ViewStackContract`, rather than
/// [`ExtractedCamera::compositing_space`]. The raw request only feeds the
/// extract-time main-texture format choice.
///
/// Spawning or despawning an overlay camera can flip the base view's space.
/// That dirties its 2d specializations and rebuilds its tonemapping pipeline
/// for one frame, and only for stacks that request a non-default space.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedCompositingSpace(pub Option<CompositingSpace>);

/// Whether a camera composites over the previous camera's output and covers
/// the whole target: [`ClearColorConfig::None`] and no viewport.
///
/// The phase-2 resolver in `bevy_core_pipeline` imports this, so the two
/// phases cannot drift.
pub fn composites_fullscreen(camera: &ExtractedCamera) -> bool {
    matches!(camera.clear_color, ClearColorConfig::None) && camera.viewport.is_none()
}

/// Per-view input to [`resolve_spaces`].
struct SpaceInput<K> {
    entity: Entity,
    /// Identity of the main-texture ping-pong the view renders into. Views
    /// resolve together only when they share it.
    texture: K,
    /// The camera's position in its render target's sorted camera order.
    sorted_index: usize,
    /// The camera's own [`CompositingSpace`] request.
    request: Option<CompositingSpace>,
    /// See [`composites_fullscreen`].
    composites_fullscreen: bool,
    is_camera_2d: bool,
    /// Whether the main-texture format stores signed floats
    /// (`Rgba16Float`/`Rgba32Float`). The format is part of the texture key,
    /// so the value is uniform within a group.
    signed_float_storage: bool,
}

/// A misconfiguration found during space resolution.
/// `resolve_composition_spaces` reports each variant as a `warn_once`.
/// `resolve_spaces` returns them so tests can check the trigger conditions.
#[derive(Debug, PartialEq, Eq)]
enum SpaceDiagnostic {
    /// A compositing stack requests both `Srgb` and `Oklab`.
    ConflictingStackRequests {
        requests: Vec<(Entity, CompositingSpace)>,
    },
    /// Views sharing a main texture without forming a stack mix
    /// Linear-normalized requests, at least one of them `Srgb`/`Oklab`.
    MixedSharedTextureRequests {
        requests: Vec<(Entity, Option<CompositingSpace>)>,
    },
    /// A non-`Camera2d` view, or a stack holding one, requests `Srgb`/`Oklab`.
    NonCamera2dRequest { non_camera_2d: Vec<Entity> },
    /// A resolved `Oklab` lands on a main texture without signed-float storage.
    OklabWithoutSignedFloatStorage { entities: Vec<Entity> },
}

/// Treats `Some(Linear)` as no request. The two produce identical pipelines.
fn normalize(request: Option<CompositingSpace>) -> Option<CompositingSpace> {
    match request {
        Some(CompositingSpace::Linear) => None,
        other => other,
    }
}

/// Resolves compositing-space requests per shared-texture group.
///
/// A group of two or more views is a compositing stack when every member after
/// the first composites fullscreen. That test counts pass-disabled members too:
/// writers always write, and phase 2 filters deferred views separately.
fn resolve_spaces<K: Clone + Eq + Hash>(
    views: impl IntoIterator<Item = SpaceInput<K>>,
) -> (
    EntityHashMap<Option<CompositingSpace>>,
    Vec<SpaceDiagnostic>,
) {
    let mut groups: HashMap<K, Vec<SpaceInput<K>>> = HashMap::default();
    for view in views {
        groups.entry(view.texture.clone()).or_default().push(view);
    }

    let mut resolved = EntityHashMap::default();
    let mut diagnostics = Vec::new();
    for group in groups.values_mut() {
        group.sort_unstable_by_key(|view| view.sorted_index);
        let is_stack = group.len() >= 2 && group[1..].iter().all(|view| view.composites_fullscreen);
        if is_stack {
            resolve_stack(group, &mut resolved, &mut diagnostics);
        } else {
            resolve_per_view(group, &mut resolved, &mut diagnostics);
        }
    }
    (resolved, diagnostics)
}

/// Resolves one space for every member of a compositing stack.
///
/// The result is the single distinct `Srgb`/`Oklab` request among the members,
/// or linear when there is none. Requesting both resolves to linear and reports
/// a conflict. The result is never `Some(Linear)`, because it must not fork the
/// texture key space.
///
/// Two overrides then apply, in order. Any non-`Camera2d` member forces the
/// stack to linear: non-2d render paths do not writer-encode, so honoring the
/// request would mis-decode linear pixels. Resolved `Oklab` degrades to linear
/// when the main texture cannot store signed floats, because UNORM storage
/// clamps the signed a/b channels.
fn resolve_stack<K>(
    members: &[SpaceInput<K>],
    resolved: &mut EntityHashMap<Option<CompositingSpace>>,
    diagnostics: &mut Vec<SpaceDiagnostic>,
) {
    let requests: Vec<(Entity, CompositingSpace)> = members
        .iter()
        .filter_map(|member| match member.request {
            Some(space @ (CompositingSpace::Srgb | CompositingSpace::Oklab)) => {
                Some((member.entity, space))
            }
            _ => None,
        })
        .collect();
    let has_srgb = requests
        .iter()
        .any(|(_, space)| *space == CompositingSpace::Srgb);
    let has_oklab = requests
        .iter()
        .any(|(_, space)| *space == CompositingSpace::Oklab);
    let mut space = match (has_srgb, has_oklab) {
        (false, false) => None,
        (true, false) => Some(CompositingSpace::Srgb),
        (false, true) => Some(CompositingSpace::Oklab),
        (true, true) => {
            diagnostics.push(SpaceDiagnostic::ConflictingStackRequests {
                requests: requests.clone(),
            });
            None
        }
    };

    let non_camera_2d: Vec<Entity> = members
        .iter()
        .filter(|member| !member.is_camera_2d)
        .map(|member| member.entity)
        .collect();
    if !non_camera_2d.is_empty() {
        // The group resolves to `Srgb`/`Oklab` only when some member requests
        // it, so a nonempty request list is exactly the warn condition.
        if !requests.is_empty() {
            diagnostics.push(SpaceDiagnostic::NonCamera2dRequest { non_camera_2d });
        }
        space = None;
    }

    if space == Some(CompositingSpace::Oklab) && !members[0].signed_float_storage {
        diagnostics.push(SpaceDiagnostic::OklabWithoutSignedFloatStorage {
            entities: members.iter().map(|member| member.entity).collect(),
        });
        space = None;
    }

    for member in members {
        resolved.insert(member.entity, space);
    }
}

/// Resolves solo views and non-stack groups, meaning groups with a clearing or
/// viewport-scoped member after the first.
///
/// Each view keeps its own request verbatim, including `Some(Linear)`, so
/// untouched configurations stay byte-identical. The `resolve_stack` overrides
/// then apply in order per view instead of per group: a non-`Camera2d` view
/// loses only its own `Srgb`/`Oklab` request, so a `Camera2d` member of a
/// mixed-type group keeps its request and its splitscreen blend semantics.
/// A kept `Oklab` degrades to linear without signed-float storage.
fn resolve_per_view<K>(
    members: &[SpaceInput<K>],
    resolved: &mut EntityHashMap<Option<CompositingSpace>>,
    diagnostics: &mut Vec<SpaceDiagnostic>,
) {
    if members.len() >= 2 {
        let normalized: Vec<Option<CompositingSpace>> = members
            .iter()
            .map(|member| normalize(member.request))
            .collect();
        let mixed = normalized.iter().any(|request| *request != normalized[0]);
        // Normalized `Some` is always `Srgb`/`Oklab`. A `Some`-vs-no-request
        // mixture is as per-pixel wrong at the clear seam as `Srgb`-vs-`Oklab`.
        let any_space = normalized.iter().any(Option::is_some);
        if mixed && any_space {
            diagnostics.push(SpaceDiagnostic::MixedSharedTextureRequests {
                requests: members
                    .iter()
                    .map(|member| (member.entity, member.request))
                    .collect(),
            });
        }
    }

    for member in members {
        let mut space = member.request;
        if !member.is_camera_2d
            && matches!(
                space,
                Some(CompositingSpace::Srgb | CompositingSpace::Oklab)
            )
        {
            diagnostics.push(SpaceDiagnostic::NonCamera2dRequest {
                non_camera_2d: vec![member.entity],
            });
            space = None;
        }
        if space == Some(CompositingSpace::Oklab) && !member.signed_float_storage {
            diagnostics.push(SpaceDiagnostic::OklabWithoutSignedFloatStorage {
                entities: vec![member.entity],
            });
            space = None;
        }
        resolved.insert(member.entity, space);
    }
}

/// Resolves every camera view's [`CompositingSpace`] request into its
/// [`ResolvedCompositingSpace`].
///
/// Each group's members are ordered by `sorted_camera_index_for_target`, so
/// this runs after `sort_cameras`, in
/// [`RenderSystems::CreateViews`](crate::RenderSystems::CreateViews).
/// The rules live on `resolve_spaces`, `resolve_stack`, and `resolve_per_view`.
/// This system feeds them and reports their diagnostics as `warn_once`s.
///
/// `Has<Camera2d>` reads the render-world marker that `bevy_core_pipeline`
/// extracts with its `ExtractComponentPlugin::<Camera2d>`. Without that plugin
/// every view counts as non-2d.
pub fn resolve_composition_spaces(
    mut views: Query<(
        Entity,
        &ExtractedCamera,
        &ExtractedView,
        &CameraMainTextureUsages,
        &Msaa,
        Has<Camera2d>,
        &mut ResolvedCompositingSpace,
    )>,
) {
    let inputs: Vec<SpaceInput<MainTextureKey>> = views
        .iter()
        .map(
            |(entity, camera, view, texture_usage, msaa, is_camera_2d, _)| SpaceInput {
                entity,
                texture: main_texture_key(camera, view, texture_usage, *msaa),
                sorted_index: camera.sorted_camera_index_for_target,
                request: camera.compositing_space,
                composites_fullscreen: composites_fullscreen(camera),
                is_camera_2d,
                signed_float_storage: matches!(
                    view.target_format,
                    TextureFormat::Rgba16Float | TextureFormat::Rgba32Float
                ),
            },
        )
        .collect();

    let (spaces, diagnostics) = resolve_spaces(inputs);
    // Every queried view fed the resolver, so the lookup always hits.
    for (entity, .., mut resolved) in views.iter_mut() {
        resolved.set_if_neq(ResolvedCompositingSpace(
            spaces.get(&entity).copied().flatten(),
        ));
    }

    for diagnostic in diagnostics {
        match diagnostic {
            SpaceDiagnostic::ConflictingStackRequests { requests } => warn_once!(
                "Cameras stacked on one shared main texture request conflicting compositing \
                spaces: {requests:?}. The stack composites in linear instead; give every \
                camera in the stack the same CompositingSpace."
            ),
            SpaceDiagnostic::MixedSharedTextureRequests { requests } => warn_once!(
                "Cameras sharing a render target mix compositing-space requests: {requests:?}. \
                Blending is per-pixel wrong wherever their regions meet; use one \
                CompositingSpace for every camera on a shared target."
            ),
            SpaceDiagnostic::NonCamera2dRequest { non_camera_2d } => warn_once!(
                "A CompositingSpace::Srgb/Oklab request resolves to linear because \
                non-Camera2d views {non_camera_2d:?} render into the shared buffer and 3d/UI \
                render paths do not encode into compositing spaces. Remove the \
                CompositingSpace component or use a Camera2d."
            ),
            SpaceDiagnostic::OklabWithoutSignedFloatStorage { entities } => warn_once!(
                "CompositingSpace::Oklab on views {entities:?} resolves to linear because the \
                main texture format cannot store the signed Oklab a/b channels. Add the Hdr \
                component to the camera to get a signed-float main texture."
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRGB: Option<CompositingSpace> = Some(CompositingSpace::Srgb);
    const OKLAB: Option<CompositingSpace> = Some(CompositingSpace::Oklab);
    const LINEAR: Option<CompositingSpace> = Some(CompositingSpace::Linear);

    fn entity(raw: u32) -> Entity {
        Entity::from_raw_u32(raw).unwrap()
    }

    /// A `Camera2d` view on signed-float storage. Tests override the fields
    /// each case exercises.
    fn view(
        raw: u32,
        texture: u32,
        index: usize,
        request: Option<CompositingSpace>,
    ) -> SpaceInput<u32> {
        SpaceInput {
            entity: entity(raw),
            texture,
            sorted_index: index,
            request,
            composites_fullscreen: true,
            is_camera_2d: true,
            signed_float_storage: true,
        }
    }

    fn resolved_for(
        output: &EntityHashMap<Option<CompositingSpace>>,
        raw: u32,
    ) -> Option<CompositingSpace> {
        *output.get(&entity(raw)).expect("view must be resolved")
    }

    fn has_conflict(diagnostics: &[SpaceDiagnostic]) -> bool {
        diagnostics
            .iter()
            .any(|d| matches!(d, SpaceDiagnostic::ConflictingStackRequests { .. }))
    }

    fn has_mixed(diagnostics: &[SpaceDiagnostic]) -> bool {
        diagnostics
            .iter()
            .any(|d| matches!(d, SpaceDiagnostic::MixedSharedTextureRequests { .. }))
    }

    fn has_non_camera_2d(diagnostics: &[SpaceDiagnostic]) -> bool {
        diagnostics
            .iter()
            .any(|d| matches!(d, SpaceDiagnostic::NonCamera2dRequest { .. }))
    }

    fn has_oklab_storage(diagnostics: &[SpaceDiagnostic]) -> bool {
        diagnostics
            .iter()
            .any(|d| matches!(d, SpaceDiagnostic::OklabWithoutSignedFloatStorage { .. }))
    }

    #[test]
    fn solo_default_camera_keeps_no_request() {
        let (resolved, diagnostics) = resolve_spaces([view(1, 0, 0, None)]);
        assert_eq!(resolved_for(&resolved, 1), None);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn solo_linear_request_is_kept_verbatim() {
        let (resolved, diagnostics) = resolve_spaces([view(1, 0, 0, LINEAR)]);
        assert_eq!(resolved_for(&resolved, 1), LINEAR);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn stack_of_linear_requests_normalizes_to_none() {
        let (resolved, diagnostics) =
            resolve_spaces([view(1, 0, 0, LINEAR), view(2, 0, 1, LINEAR)]);
        assert_eq!(resolved_for(&resolved, 1), None);
        assert_eq!(resolved_for(&resolved, 2), None);
        assert!(diagnostics.is_empty());
    }

    // Two members asking for the same space is still one distinct request.
    #[test]
    fn stack_with_one_distinct_space_resolves_every_member_to_it() {
        let (resolved, diagnostics) = resolve_spaces([
            view(1, 0, 0, None),
            view(2, 0, 1, SRGB),
            view(3, 0, 2, SRGB),
        ]);
        assert_eq!(resolved_for(&resolved, 1), SRGB);
        assert_eq!(resolved_for(&resolved, 2), SRGB);
        assert_eq!(resolved_for(&resolved, 3), SRGB);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn stack_with_conflicting_spaces_resolves_to_none_and_warns() {
        let (resolved, diagnostics) = resolve_spaces([view(1, 0, 0, SRGB), view(2, 0, 1, OKLAB)]);
        assert_eq!(resolved_for(&resolved, 1), None);
        assert_eq!(resolved_for(&resolved, 2), None);
        assert!(has_conflict(&diagnostics));
        assert!(!has_mixed(&diagnostics));
    }

    #[test]
    fn viewport_splitscreen_keeps_per_view_requests() {
        let mut base = view(1, 0, 0, SRGB);
        base.composites_fullscreen = false;
        let mut pip = view(2, 0, 1, OKLAB);
        pip.composites_fullscreen = false;
        let (resolved, diagnostics) = resolve_spaces([base, pip]);
        assert_eq!(resolved_for(&resolved, 1), SRGB);
        assert_eq!(resolved_for(&resolved, 2), OKLAB);
        assert!(has_mixed(&diagnostics));
        assert!(!has_conflict(&diagnostics));
    }

    #[test]
    fn mixed_request_and_no_request_non_stack_warns() {
        let mut upper = view(2, 0, 1, None);
        upper.composites_fullscreen = false;
        let (resolved, diagnostics) = resolve_spaces([view(1, 0, 0, SRGB), upper]);
        assert_eq!(resolved_for(&resolved, 1), SRGB);
        assert_eq!(resolved_for(&resolved, 2), None);
        assert!(has_mixed(&diagnostics));
    }

    #[test]
    fn same_request_non_stack_does_not_warn() {
        let mut upper = view(2, 0, 1, SRGB);
        upper.composites_fullscreen = false;
        let (resolved, diagnostics) = resolve_spaces([view(1, 0, 0, SRGB), upper]);
        assert_eq!(resolved_for(&resolved, 1), SRGB);
        assert_eq!(resolved_for(&resolved, 2), SRGB);
        assert!(diagnostics.is_empty());
    }

    // `Some(Linear)` normalizes to no-request, so this mixture is not mixed.
    #[test]
    fn linear_vs_no_request_non_stack_does_not_warn() {
        let mut upper = view(2, 0, 1, None);
        upper.composites_fullscreen = false;
        let (resolved, diagnostics) = resolve_spaces([view(1, 0, 0, LINEAR), upper]);
        assert_eq!(resolved_for(&resolved, 1), LINEAR);
        assert_eq!(resolved_for(&resolved, 2), None);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn solo_non_camera_2d_srgb_request_resolves_to_none() {
        let mut camera_3d = view(1, 0, 0, SRGB);
        camera_3d.is_camera_2d = false;
        let (resolved, diagnostics) = resolve_spaces([camera_3d]);
        assert_eq!(resolved_for(&resolved, 1), None);
        assert!(has_non_camera_2d(&diagnostics));
    }

    #[test]
    fn solo_non_camera_2d_linear_request_kept_without_warning() {
        let mut camera_3d = view(1, 0, 0, LINEAR);
        camera_3d.is_camera_2d = false;
        let (resolved, diagnostics) = resolve_spaces([camera_3d]);
        assert_eq!(resolved_for(&resolved, 1), LINEAR);
        assert!(diagnostics.is_empty());
    }

    // The stack warns only because one of its members requested a space.
    #[test]
    fn stack_with_non_camera_2d_member_resolves_to_none() {
        let mut base = view(1, 0, 0, None);
        base.is_camera_2d = false;
        base.composites_fullscreen = false;
        let (resolved, diagnostics) = resolve_spaces([base, view(2, 0, 1, SRGB)]);
        assert_eq!(resolved_for(&resolved, 1), None);
        assert_eq!(resolved_for(&resolved, 2), None);
        assert!(has_non_camera_2d(&diagnostics));
    }

    #[test]
    fn non_camera_2d_stack_without_requests_does_not_warn() {
        let mut base = view(1, 0, 0, None);
        base.is_camera_2d = false;
        base.composites_fullscreen = false;
        let (resolved, diagnostics) = resolve_spaces([base, view(2, 0, 1, None)]);
        assert_eq!(resolved_for(&resolved, 1), None);
        assert_eq!(resolved_for(&resolved, 2), None);
        assert!(diagnostics.is_empty());
    }

    // The non-`Camera2d` member has no request of its own, so it draws no
    // non-2d warning.
    #[test]
    fn camera_2d_member_of_mixed_non_stack_group_keeps_request() {
        let mut camera_2d = view(1, 0, 0, SRGB);
        camera_2d.composites_fullscreen = false;
        let mut camera_3d = view(2, 0, 1, None);
        camera_3d.composites_fullscreen = false;
        camera_3d.is_camera_2d = false;
        let (resolved, diagnostics) = resolve_spaces([camera_2d, camera_3d]);
        assert_eq!(resolved_for(&resolved, 1), SRGB);
        assert_eq!(resolved_for(&resolved, 2), None);
        assert!(has_mixed(&diagnostics));
        assert!(!has_non_camera_2d(&diagnostics));
    }

    #[test]
    fn oklab_without_signed_float_storage_degrades_to_linear() {
        let mut camera = view(1, 0, 0, OKLAB);
        camera.signed_float_storage = false;
        let (resolved, diagnostics) = resolve_spaces([camera]);
        assert_eq!(resolved_for(&resolved, 1), None);
        assert!(has_oklab_storage(&diagnostics));
    }

    #[test]
    fn stack_resolved_oklab_degrades_on_unorm_storage() {
        let mut base = view(1, 0, 0, None);
        base.signed_float_storage = false;
        let mut overlay = view(2, 0, 1, OKLAB);
        overlay.signed_float_storage = false;
        let (resolved, diagnostics) = resolve_spaces([base, overlay]);
        assert_eq!(resolved_for(&resolved, 1), None);
        assert_eq!(resolved_for(&resolved, 2), None);
        assert!(has_oklab_storage(&diagnostics));
    }

    // The non-`Camera2d` rule runs before the storage rule, so a request
    // forced to linear never warns twice.
    #[test]
    fn non_camera_2d_oklab_fires_non_2d_warning_not_storage_warning() {
        let mut camera_3d = view(1, 0, 0, OKLAB);
        camera_3d.is_camera_2d = false;
        camera_3d.signed_float_storage = false;
        let (resolved, diagnostics) = resolve_spaces([camera_3d]);
        assert_eq!(resolved_for(&resolved, 1), None);
        assert!(has_non_camera_2d(&diagnostics));
        assert!(!has_oklab_storage(&diagnostics));
    }

    #[test]
    fn separate_textures_resolve_independently() {
        let (resolved, diagnostics) = resolve_spaces([view(1, 0, 0, SRGB), view(2, 1, 0, OKLAB)]);
        assert_eq!(resolved_for(&resolved, 1), SRGB);
        assert_eq!(resolved_for(&resolved, 2), OKLAB);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn sorted_index_orders_the_group_not_insertion_order() {
        let mut base = view(1, 0, 0, None);
        base.composites_fullscreen = false;
        // Insert the overlay first. The group is still a stack because the
        // clearing member, the only one allowed to clear, sorts to the front.
        let (resolved, diagnostics) = resolve_spaces([view(2, 0, 1, SRGB), base]);
        assert_eq!(resolved_for(&resolved, 1), SRGB);
        assert_eq!(resolved_for(&resolved, 2), SRGB);
        assert!(diagnostics.is_empty());
    }
}
