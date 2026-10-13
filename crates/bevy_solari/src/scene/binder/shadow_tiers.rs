use crate::scene::ShadowTiers;
use bevy_math::Vec3;
use bevy_render::{
    render_resource::{BindingResource, Buffer, ShaderType, StorageBuffer},
    renderer::{RenderDevice, RenderQueue},
};

/// `TierSphere` in `bindings.wesl`.
#[derive(ShaderType, Clone, Copy, Default, PartialEq)]
struct GpuTierSphere {
    center: Vec3,
    near_radius: f32,
    mid_radius: f32,
}

/// `ShadowTiers` in `bindings.wesl`.
#[derive(ShaderType, Clone, Copy, Default, PartialEq)]
struct GpuShadowTiers {
    spheres: [GpuTierSphere; 2],
    tiered: u32,
}

/// This frame's and the previous frame's [`ShadowTiers`] on the GPU.
pub struct ShadowTiersState {
    buffer: StorageBuffer<GpuShadowTiers>,
}

impl ShadowTiersState {
    pub fn new() -> Self {
        let mut buffer = StorageBuffer::default();
        buffer.set_label(Some("solari_shadow_tiers"));
        Self { buffer }
    }

    /// Uploads `tiers`. Without them the shaders fall back to a single untiered trace.
    pub fn update(
        &mut self,
        tiers: Option<&ShadowTiers>,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
    ) {
        let last = *self.buffer.get();
        let gpu = tiers.map_or(GpuShadowTiers::default(), |tiers| {
            let current = GpuTierSphere {
                center: tiers.center,
                near_radius: tiers.near_radius,
                mid_radius: tiers.mid_radius.max(tiers.near_radius),
            };
            let previous = if last.tiered == 1 {
                last.spheres[0]
            } else {
                current
            };
            GpuShadowTiers {
                spheres: [current, previous],
                tiered: 1,
            }
        });
        if self.buffer.buffer().is_none() || last != gpu {
            self.buffer.set(gpu);
            self.buffer.write_buffer(render_device, render_queue);
        }
    }

    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.buffer()
    }

    pub fn binding(&self) -> BindingResource<'_> {
        self.buffer
            .binding()
            .expect("shadow tiers should have been written before the bind group is prepared")
    }
}
