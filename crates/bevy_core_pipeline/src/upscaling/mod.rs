use crate::blit::{BlitPipeline, BlitPipelineKey};
use crate::camera_stack::{BlitDisposition, ViewStackContract};
use bevy_app::prelude::*;
use bevy_camera::{CameraOutputMode, CompositingSpace};
use bevy_ecs::prelude::*;
use bevy_render::{
    camera::ExtractedCamera, render_resource::*, view::ViewTarget, Render, RenderApp,
    RenderStartup, RenderSystems,
};

mod node;

pub use node::upscaling;

pub struct UpscalingPlugin;

impl Plugin for UpscalingPlugin {
    fn build(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(
                Render,
                // This system should probably technically be run *after* all of the other systems
                // that might modify `PipelineCache` via interior mutability, but for now,
                // we've chosen to simply ignore the ambiguities out of a desire for a better refactor
                // and aversion to extensive and intrusive system ordering.
                // See https://github.com/bevyengine/bevy/issues/14770 for more context.
                prepare_view_upscaling_pipelines
                    .in_set(RenderSystems::Prepare)
                    .ambiguous_with_all(),
            );
            render_app.add_systems(RenderStartup, clear_view_upscaling_pipelines);
        }
    }
}

#[derive(Component)]
pub struct ViewUpscalingPipeline(CachedRenderPipelineId, BlitPipelineKey);

/// This is not required on first startup but is required during render recovery
fn clear_view_upscaling_pipelines(
    mut commands: Commands,
    views: Query<Entity, With<ViewUpscalingPipeline>>,
) {
    for entity in &views {
        commands.entity(entity).remove::<ViewUpscalingPipeline>();
    }
}

/// The compositing space the upscaling blit decodes from, taken from the
/// view's [`ViewStackContract`].
///
/// Returns `None` when the stack resolved encode parameters: the display
/// encoder already did the compositing-space decode and the main texture
/// holds encoded signal, so the blit passes it through. A request downgraded
/// to plain SDR carries no encode parameters and keeps the
/// decode-and-hardware-encode path.
fn blit_source_space(contract: Option<&ViewStackContract>) -> Option<CompositingSpace> {
    let contract = contract?;
    if contract.encoding.is_some() {
        None
    } else {
        contract.compositing_space
    }
}

/// The blend state of a camera whose upscaling blit auto-detects its blend
/// (`CameraOutputMode::Write { blend_state: None }`).
///
/// The first camera to render to an output (`sorted_index == 0`) replaces.
/// Later cameras alpha-blend so they do not overwrite earlier cameras'
/// output. `force_replace` replaces too, because that blit carries the whole
/// composition and must not blend over the cleared out texture.
fn auto_blit_blend_state(force_replace: bool, sorted_index: usize) -> Option<BlendState> {
    if force_replace || sorted_index == 0 {
        None
    } else {
        Some(BlendState::ALPHA_BLENDING)
    }
}

fn prepare_view_upscaling_pipelines(
    mut commands: Commands,
    mut pipeline_cache: ResMut<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<BlitPipeline>>,
    blit_pipeline: Res<BlitPipeline>,
    view_targets: Query<(
        Entity,
        &ViewTarget,
        Option<&ExtractedCamera>,
        Option<&ViewUpscalingPipeline>,
        Option<&ViewStackContract>,
    )>,
) {
    for (entity, view_target, camera, maybe_pipeline, contract) in view_targets.iter() {
        // Removing the pipeline keeps the upscaling node from running, since
        // its `ViewQuery` requires the component, so the out texture stays
        // untouched until the finalizer blits.
        let force_replace = match contract.map(|contract| contract.blit) {
            Some(BlitDisposition::SkipDeferred) => {
                commands.entity(entity).remove::<ViewUpscalingPipeline>();
                continue;
            }
            Some(BlitDisposition::Run { force_replace }) => force_replace,
            None => false,
        };

        let blend_state = if let Some(extracted_camera) = camera {
            match extracted_camera.output_mode {
                CameraOutputMode::Skip => None,
                CameraOutputMode::Write { blend_state, .. } => match blend_state {
                    None => auto_blit_blend_state(
                        force_replace,
                        extracted_camera.sorted_camera_index_for_target,
                    ),
                    _ => blend_state,
                },
            }
        } else {
            None
        };

        let Some(target_format) = view_target.out_texture_view_format() else {
            continue;
        };

        let source_space = blit_source_space(contract);

        let key = BlitPipelineKey {
            target_format,
            blend_state,
            samples: 1,
            source_space,
        };

        if maybe_pipeline.is_none_or(|ViewUpscalingPipeline(_, cached_key)| *cached_key != key) {
            let pipeline = pipelines.specialize(&pipeline_cache, &blit_pipeline, key);

            // Ensure the pipeline is loaded before continuing the frame to prevent frames without
            // any GPU work submitted
            pipeline_cache.block_on_render_pipeline(pipeline);

            commands
                .entity(entity)
                .insert(ViewUpscalingPipeline(pipeline, key));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_stack::{ResolvedEncoding, StackRole};
    use bevy_window::{DisplayGamut, DisplayTransfer};

    /// A solo SDR view's contract, with no encode parameters.
    fn sdr_contract(compositing_space: Option<CompositingSpace>) -> ViewStackContract {
        ViewStackContract {
            tonemap: StackRole::Solo,
            encode: StackRole::Solo,
            blit: BlitDisposition::Run {
                force_replace: false,
            },
            compositing_space,
            source_gamut: DisplayGamut::Rec709,
            encoding: None,
        }
    }

    #[test]
    fn solo_sdr_default_keys_no_source_space() {
        let contract = sdr_contract(None);
        let key = BlitPipelineKey {
            target_format: TextureFormat::Bgra8UnormSrgb,
            blend_state: None,
            samples: 1,
            source_space: blit_source_space(Some(&contract)),
        };
        let expected = BlitPipelineKey {
            target_format: TextureFormat::Bgra8UnormSrgb,
            blend_state: None,
            samples: 1,
            source_space: None,
        };
        assert!(key == expected);
    }

    #[test]
    fn resolved_space_keys_the_blit_decode_on_sdr() {
        assert_eq!(
            blit_source_space(Some(&sdr_contract(Some(CompositingSpace::Oklab)))),
            Some(CompositingSpace::Oklab)
        );
        assert_eq!(
            blit_source_space(Some(&sdr_contract(Some(CompositingSpace::Srgb)))),
            Some(CompositingSpace::Srgb)
        );
    }

    #[test]
    fn encoded_views_blit_without_decode() {
        let mut contract = sdr_contract(Some(CompositingSpace::Srgb));
        contract.encoding = Some(ResolvedEncoding {
            transfer: DisplayTransfer::Pq,
            gamut: DisplayGamut::Rec2020,
        });
        assert_eq!(blit_source_space(Some(&contract)), None);
    }

    #[test]
    fn missing_contract_blits_without_decode() {
        assert_eq!(blit_source_space(None), None);
    }

    #[test]
    fn auto_blend_is_replace_for_first_camera_and_alpha_for_later() {
        assert_eq!(auto_blit_blend_state(false, 0), None);
        assert_eq!(
            auto_blit_blend_state(false, 1),
            Some(BlendState::ALPHA_BLENDING)
        );
        assert_eq!(
            auto_blit_blend_state(false, 2),
            Some(BlendState::ALPHA_BLENDING)
        );
    }

    #[test]
    fn force_replace_upgrades_later_finalizer_to_replace() {
        assert_eq!(auto_blit_blend_state(true, 1), None);
        assert_eq!(auto_blit_blend_state(true, 2), None);
        assert_eq!(auto_blit_blend_state(true, 0), None);
    }
}
