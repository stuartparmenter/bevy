use bevy_ecs::{
    resource::Resource,
    system::{Res, ResMut},
};
use bevy_render::{
    render_resource::{CommandEncoder, CommandEncoderDescriptor},
    renderer::{RenderDevice, RenderQueue},
};

/// A shared command encoder for GPU geometry updates and BLAS builds.
///
/// Producers record compute passes that fill [`RaytracingGeometryBuffers`] during
/// [`RenderSystems::PrepareResources`]. Solari records the BLAS builds afterward
/// and submits the encoder once, before the render graph builds the TLAS.
/// Queue buffer writes are flushed before the recorded commands execute.
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

/// Submits the producer passes and BLAS builds, if any were recorded this frame.
pub fn submit_raytracing_producers(
    mut producer_encoder: ResMut<RaytracingProducerEncoder>,
    render_queue: Res<RenderQueue>,
) {
    if let Some(encoder) = producer_encoder.0.take() {
        render_queue.submit([encoder.finish()]);
    }
}
