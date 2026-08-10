use bevy_ecs::{entity::EntityHashMap, prelude::*};
use bevy_platform::collections::hash_map::Entry;
use bevy_render::{
    render_resource::StorageBuffer,
    renderer::{RenderDevice, RenderQueue},
};
use bevy_utils::once;
use tracing::warn;

use super::{
    pipeline::{AutoExposureState, AutoExposureUniform},
    settings::PhysiologicalAdaptation,
    AutoExposure, AutoWhiteBalance,
};

/// CIE 1931 xy of the D65 white point. Keep in sync with `D65_XY` in `bevy_render::view`
/// and `AWB_D65_XY` in `auto_exposure.wesl`.
const D65_XY: (f32, f32) = (0.31272, 0.32903);

/// Per-view GPU adaptation state, keyed by render-world view entity.
#[derive(Resource, Default)]
pub(super) struct AutoExposureBuffers {
    pub(super) buffers: EntityHashMap<StorageBuffer<AutoExposureState>>,
}

/// An existing buffer is never rewritten, so the adaptation stays continuous across
/// settings changes. Removing and re-adding [`AutoExposure`] resets it.
pub(super) fn prepare_buffers(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut buffers: ResMut<AutoExposureBuffers>,
    views: Query<(Entity, &AutoExposure), With<AutoExposureUniform>>,
) {
    for (entity, settings) in &views {
        if let Entry::Vacant(entry) = buffers.buffers.entry(entity) {
            let state = entry.insert(StorageBuffer::from(initial_state(settings)));
            state.write_buffer(&device, &queue);
        }
    }

    // The render node bails out when the view has no entry here, so dropping the buffer
    // stops the metering pass. `ViewAutoExposurePipeline` is never removed, so keying on
    // it would leak the buffer. `AutoExposureUniform` is the liveness anchor: the sync
    // machinery removes it when the main-world camera stops metering.
    buffers.buffers.retain(|&entity, _| views.contains(entity));
}

/// Builds the settings uniform for one view, sanitizing invalid values. The long-term
/// envelope parameters are filled in even when [`AutoExposure::physiological`] is `None`;
/// see [`PhysiologicalAdaptation`].
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

pub(super) fn initial_state(settings: &AutoExposure) -> AutoExposureState {
    let (min_log_lum, max_log_lum) = settings.range.clone().into_inner();
    let exposure = 0.0f32.clamp(min_log_lum, max_log_lum);

    AutoExposureState {
        exposure,
        long_term: exposure,
        chroma_x: D65_XY.0,
        chroma_y: D65_XY.1,
        // The shader drains these to zero every frame; this is the only CPU write.
        chroma_sums: [0; 3],
    }
}
