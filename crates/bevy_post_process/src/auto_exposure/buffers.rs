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

/// The CIE 1931 *xy* chromaticity of the D65 white point, matching `D65_XY`
/// in `bevy_render::view` and `AWB_D65_XY` in `auto_exposure.wgsl`; keep them
/// in sync.
const D65_XY: (f32, f32) = (0.31272, 0.32903);

/// The per-view GPU adaptation state, keyed by render-world view entity.
#[derive(Resource, Default)]
pub(super) struct AutoExposureBuffers {
    pub(super) buffers: EntityHashMap<StorageBuffer<AutoExposureState>>,
}

/// Creates the adaptation state buffer for views that started metering and
/// drops the buffer of views that stopped.
///
/// An existing state buffer is never rewritten, so the adaptation animation
/// stays continuous across settings changes; removing and re-adding
/// [`AutoExposure`] drops and recreates the buffer, resetting the adaptation.
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

    // Dropping the buffer is enough to stop the metering pass: the render node
    // bails out when the view has no entry in this map, so the stale
    // `ViewAutoExposurePipeline` the queue system left behind is inert.
    // `AutoExposureUniform` is the liveness anchor because the sync machinery
    // removes it from the render entity when the main-world camera stops
    // metering, and a despawned view fails the query as well; the components
    // the node reads are insert-only and would leak the buffer.
    buffers.buffers.retain(|&entity, _| views.contains(entity));
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
