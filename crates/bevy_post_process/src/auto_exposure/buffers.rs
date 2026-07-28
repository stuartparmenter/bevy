use bevy_ecs::{entity::EntityHashMap, prelude::*};
use bevy_platform::collections::hash_map::Entry;
use bevy_render::{
    render_resource::{StorageBuffer, UniformBuffer},
    renderer::{RenderDevice, RenderQueue},
    sync_world::RenderEntity,
    Extract,
};
use bevy_utils::once;
use tracing::warn;

use super::{
    pipeline::{AutoExposureState, AutoExposureUniform},
    settings::PhysiologicalAdaptation,
    AutoExposure, AutoWhiteBalance,
};

/// The CIE 1931 *xy* chromaticity of the D65 white point, matching `D65_XY`
/// in `bevy_render::view` and `AWB_D65_XY` in `auto_exposure.wgsl`; keep them
/// in sync.
const D65_XY: (f32, f32) = (0.31272, 0.32903);

#[derive(Resource, Default)]
pub(super) struct AutoExposureBuffers {
    pub(super) buffers: EntityHashMap<AutoExposureBuffer>,
}

pub(super) struct AutoExposureBuffer {
    pub(super) state: StorageBuffer<AutoExposureState>,
    pub(super) settings: UniformBuffer<AutoExposureUniform>,
}

/// One extracted camera that needs its settings uniform (re)built. Every
/// metering camera carries [`AutoExposure`]; [`AutoWhiteBalance`] rides along
/// in the same pass when present.
type ExtractedCamera = (Entity, AutoExposure, Option<AutoWhiteBalance>);

#[derive(Resource)]
pub(super) struct ExtractedStateBuffers {
    changed: Vec<ExtractedCamera>,
    /// Render-world entities whose camera stopped metering (its
    /// [`AutoExposure`] was removed while the camera itself stays alive).
    /// These are *render* entities because [`AutoExposureBuffers`] is keyed
    /// by render entities; cameras that were despawned outright are handled
    /// by the liveness sweep in [`prepare_buffers`] instead.
    removed: Vec<Entity>,
}

pub(super) fn extract_buffers(
    mut commands: Commands,
    changed: Extract<
        Query<
            (RenderEntity, &AutoExposure, Option<&AutoWhiteBalance>),
            Or<(Changed<AutoExposure>, Changed<AutoWhiteBalance>)>,
        >,
    >,
    mut removed: Extract<RemovedComponents<AutoExposure>>,
    mut removed_white_balance: Extract<RemovedComponents<AutoWhiteBalance>>,
    cameras: Extract<Query<(RenderEntity, &AutoExposure, Option<&AutoWhiteBalance>)>>,
    render_entities: Extract<Query<RenderEntity>>,
) {
    let mut changed: Vec<ExtractedCamera> = changed
        .iter()
        .map(|(entity, settings, white_balance)| (entity, settings.clone(), white_balance.copied()))
        .collect();
    let mut fully_removed = Vec::new();

    // Removing one of the components does not trigger the `Changed` filters
    // above, but the settings uniform must still be rebuilt from whatever is
    // left on the camera. Read the live component state instead of assuming
    // the removed component is gone: a remove + re-insert within the same
    // frame still buffers a removal event, and unconditionally pushing the
    // component as absent here would override the freshly inserted value
    // (the `changed` entries above are processed first, in order). Only when
    // the camera no longer meters at all is the buffer torn down.
    {
        let mut handle_removal = |entity: Entity| {
            if let Ok((render_entity, settings, white_balance)) = cameras.get(entity) {
                changed.push((render_entity, settings.clone(), white_balance.copied()));
            } else if let Ok(render_entity) = render_entities.get(entity) {
                // The camera is still alive but no longer meters: tear its
                // buffer down by its *render*-world key — the buffer map is
                // keyed by render entities, so pushing the main-world entity
                // here would silently leave the buffer (and the metering
                // dispatches consuming it) alive forever.
                fully_removed.push(render_entity);
            }
            // Otherwise the camera was despawned outright; its render entity
            // is torn down by the sync machinery and `prepare_buffers`'
            // liveness sweep drops the buffer.
        };

        for entity in removed.read() {
            handle_removal(entity);
        }
        for entity in removed_white_balance.read() {
            handle_removal(entity);
        }
    }

    commands.insert_resource(ExtractedStateBuffers {
        changed,
        removed: fully_removed,
    });
}

pub(super) fn prepare_buffers(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut extracted: ResMut<ExtractedStateBuffers>,
    mut buffers: ResMut<AutoExposureBuffers>,
    live_entities: Query<()>,
) {
    for (entity, settings, white_balance) in extracted.changed.drain(..) {
        let uniform = build_uniform(&settings, white_balance.as_ref());

        match buffers.buffers.entry(entity) {
            Entry::Occupied(mut entry) => {
                // Update the settings buffer, but skip updating the state buffer.
                // The state buffer is skipped so that the animation stays continuous.
                let value = entry.get_mut();
                value.settings.set(uniform);
                value.settings.write_buffer(&device, &queue);
            }
            Entry::Vacant(entry) => {
                let value = entry.insert(AutoExposureBuffer {
                    state: StorageBuffer::from(initial_state(&settings)),
                    settings: UniformBuffer::from(uniform),
                });

                value.state.write_buffer(&device, &queue);
                value.settings.write_buffer(&device, &queue);
            }
        }
    }

    // Dropping the buffer is enough to stop the metering pass: the render node
    // bails out when the view has no entry in this map, so the stale
    // `ViewAutoExposurePipeline` the queue system left behind is inert.
    for entity in extracted.removed.drain(..) {
        buffers.buffers.remove(&entity);
    }

    // Cameras that are despawned outright never make it into the `removed`
    // list (their main-world entity is gone before the removal events are
    // read), so sweep out buffers whose render-world entity no longer exists.
    buffers
        .buffers
        .retain(|&entity, _| live_entities.contains(entity));
}

/// Builds the settings uniform for one view, sanitizing invalid values.
///
/// When [`AutoExposure::physiological`] is `None`, the long-term envelope parameters are
/// still filled in (with their defaults) because the compute shader keeps the envelope
/// tracking the short-term exposure even while it is disabled; only the
/// `physiological` flag controls whether the envelope actually bounds the exposure.
pub(super) fn build_uniform(
    settings: &AutoExposure,
    white_balance: Option<&AutoWhiteBalance>,
) -> AutoExposureUniform {
    let (min_log_lum, max_log_lum) = settings.range.clone().into_inner();
    let (low_percent, high_percent) = settings.filter.clone().into_inner();

    let metering_bias = if settings.metering_bias.is_finite() {
        settings.metering_bias
    } else {
        once!(warn!(
            "AutoExposure::metering_bias must be finite; ignoring the configured value"
        ));
        0.0
    };

    let adaptation = settings
        .physiological
        .as_ref()
        .map(PhysiologicalAdaptation::sanitized)
        .unwrap_or_default();

    let white_balance = white_balance.map(AutoWhiteBalance::sanitized);

    AutoExposureUniform {
        min_log_lum,
        inv_log_lum_range: 1.0 / (max_log_lum - min_log_lum),
        log_lum_range: max_log_lum - min_log_lum,
        low_percent,
        high_percent,
        speed_up: settings.speed_brighten,
        speed_down: settings.speed_darken,
        exponential_transition_distance: settings.exponential_transition_distance,
        metering_bias,
        long_term_speed_up: adaptation.speed_brighten,
        long_term_speed_down: adaptation.speed_darken,
        long_term_bound_up: adaptation.bound_darken,
        long_term_bound_down: adaptation.bound_brighten,
        physiological: settings.physiological.is_some() as u32,
        awb_speed: white_balance.map_or(0.0, |wb| wb.speed),
        awb_anchor: white_balance.map_or(0.0, |wb| wb.virtual_light_anchor),
        awb_enabled: white_balance.is_some() as u32,
        pad_0: 0,
        pad_1: 0,
        pad_2: 0,
    }
}

/// Builds the initial per-view adaptation state for one view.
///
/// Both the short-term exposure and the long-term envelope start at the neutral value the
/// classic implementation used, clamped into [`AutoExposure::range`]. The adapted
/// white-balance chromaticity always starts at the neutral D65 white point.
pub(super) fn initial_state(settings: &AutoExposure) -> AutoExposureState {
    let (min_log_lum, max_log_lum) = settings.range.clone().into_inner();
    let exposure = 0.0f32.clamp(min_log_lum, max_log_lum);

    AutoExposureState {
        exposure,
        long_term: exposure,
        chroma_x: D65_XY.0,
        chroma_y: D65_XY.1,
    }
}
