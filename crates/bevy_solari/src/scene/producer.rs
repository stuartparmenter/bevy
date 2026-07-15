use bevy_ecs::{
    resource::Resource,
    system::{Res, ResMut},
};
use bevy_render::{
    render_resource::{CommandEncoder, CommandEncoderDescriptor},
    renderer::{RenderDevice, RenderQueue},
};

/// Shared command encoder for GPU-authored geometry producers (compute
/// skinners, expanders — anything that fills [`RaytracingGeometryBuffers`]
/// on the GPU each frame).
///
/// Producers record their passes here during [`RenderSystems::PrepareResources`]
/// instead of finishing and submitting their own encoders;
/// [`submit_raytracing_producers`] submits the lot once, ahead of this frame's
/// BLAS/TLAS builds. Queue submission order guarantees the compute output is
/// visible to the builds, and `queue.write_buffer` data staged by producers is
/// flushed at the head of this same submit — so per-producer params uploads
/// stay ordered before their dispatches. This replaces one `queue.submit` per
/// producer per frame (each with its own pending-writes flush) with one total.
///
/// [`RaytracingGeometryBuffers`]: super::RaytracingGeometryBuffers
/// [`RenderSystems::PrepareResources`]: bevy_render::RenderSystems::PrepareResources
#[derive(Resource, Default)]
pub struct RaytracingProducerEncoder(Option<CommandEncoder>);

impl RaytracingProducerEncoder {
    /// The shared encoder, created on first use each frame.
    pub fn encoder(&mut self, render_device: &RenderDevice) -> &mut CommandEncoder {
        self.0.get_or_insert_with(|| {
            render_device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("raytracing_producer_encoder"),
            })
        })
    }
}

/// Submits the shared producer encoder (if any producer recorded work),
/// before [`prepare_raytracing_geometry_blas`](super::blas) readies the
/// BLASes the binder builds from.
pub fn submit_raytracing_producers(
    mut producer_encoder: ResMut<RaytracingProducerEncoder>,
    render_queue: Res<RenderQueue>,
) {
    if let Some(encoder) = producer_encoder.0.take() {
        render_queue.submit([encoder.finish()]);
    }
}
