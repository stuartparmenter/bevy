use crate::scene::ShadowTiers;
use bevy_math::{Vec2, Vec3};
use bevy_render::{
    render_asset::RenderAssets,
    render_resource::{
        BindingResource, Buffer, ShaderType, StorageBuffer, TextureView, TextureViewId,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::{FallbackImage, GpuImage},
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
pub struct GpuShadowTiers {
    spheres: [GpuTierSphere; 2],
    flags: u32,
    height_max: f32,
    origin: Vec2,
    extent: f32,
    canopy_height_max: f32,
    canopy_base: f32,
    canopy_extinction: f32,
    canopy_self_radius: f32,
    ground_slack: f32,
    canopy_surface_height: f32,
}

/// `flags`: directional shadow rays are tiered.
const TIERED: u32 = 1;
/// `flags`: the heightfield textures are bound and marched past the mid radius.
const HEIGHTFIELD: u32 = 2;

/// This frame's and the previous frame's [`ShadowTiers`] on the GPU, with the heightfield
/// textures they march.
pub struct ShadowTiersState {
    buffer: StorageBuffer<GpuShadowTiers>,
    previous: Option<GpuTierSphere>,
    ground: Option<TextureView>,
    canopy: Option<TextureView>,
}

impl ShadowTiersState {
    pub fn new() -> Self {
        let mut buffer = StorageBuffer::default();
        buffer.set_label(Some("solari_shadow_tiers"));
        Self {
            buffer,
            previous: None,
            ground: None,
            canopy: None,
        }
    }

    /// Uploads `tiers`, resolving its heightfield textures. Without `tiers`, or while a texture is
    /// not loaded, the shaders fall back to a single untiered trace.
    pub fn update(
        &mut self,
        tiers: Option<&ShadowTiers>,
        images: &RenderAssets<GpuImage>,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
    ) {
        let mut gpu = GpuShadowTiers::default();
        self.ground = None;
        self.canopy = None;
        if let Some(tiers) = tiers {
            let current = GpuTierSphere {
                center: tiers.center,
                near_radius: tiers.near_radius,
                mid_radius: tiers.mid_radius.max(tiers.near_radius),
            };
            gpu.spheres = [current, self.previous.unwrap_or(current)];
            self.previous = Some(current);
            gpu.flags = TIERED;
            if let Some(heightfield) = &tiers.heightfield
                && let (Some(ground), Some(canopy)) = (
                    images.get(&heightfield.ground),
                    images.get(&heightfield.canopy),
                )
            {
                self.ground = Some(ground.texture_view.clone());
                self.canopy = Some(canopy.texture_view.clone());
                gpu.flags |= HEIGHTFIELD;
                gpu.height_max = heightfield.height_max;
                gpu.origin = heightfield.origin;
                gpu.extent = heightfield.extent;
                gpu.canopy_height_max = heightfield.canopy_height_max;
                gpu.canopy_base = heightfield.canopy_base;
                gpu.canopy_extinction = heightfield.canopy_extinction;
                gpu.canopy_self_radius = heightfield.canopy_self_radius;
                gpu.ground_slack = heightfield.ground_slack;
                gpu.canopy_surface_height = heightfield.canopy_surface_height;
            }
        } else {
            self.previous = None;
        }
        if self.buffer.buffer().is_none() || *self.buffer.get() != gpu {
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

    /// The ground and canopy views to bind, the fallback while a heightfield is not bound.
    pub fn views<'a>(&'a self, fallback: &'a FallbackImage) -> [&'a TextureView; 2] {
        let fallback = &fallback.d2.texture_view;
        [
            self.ground.as_ref().unwrap_or(fallback),
            self.canopy.as_ref().unwrap_or(fallback),
        ]
    }

    /// The ids of the bound heightfield views, `None` for the fallback.
    pub fn texture_ids(&self) -> [Option<TextureViewId>; 2] {
        [
            self.ground.as_ref().map(TextureView::id),
            self.canopy.as_ref().map(TextureView::id),
        ]
    }
}
