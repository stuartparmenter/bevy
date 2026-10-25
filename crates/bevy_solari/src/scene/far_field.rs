use alloc::borrow::Cow;
use bevy_app::App;
use bevy_asset::{Assets, Handle};
use bevy_ecs::resource::Resource;
use bevy_render::{
    render_resource::{BindGroup, BindGroupLayoutDescriptor},
    RenderApp,
};
use bevy_shader::Shader;

/// Directional light visibility past the mid sphere of [`ShadowTiers`](super::ShadowTiers),
/// supplied by the application: a terrain heightfield, a distance field, a shadow map.
/// [`Self::install`] it before the first frame renders; the pipelines read it once, at render
/// startup. Absent, nothing occludes past the mid sphere.
///
/// The module defines
/// `fn far_field_visibility(origin: vec3<f32>, direction: vec3<f32>, near_t: f32, mid_t: f32, previous: bool) -> f32`:
/// the transmittance of the light along the unit `direction` from `origin`. The scene's
/// instances block nothing up to `near_t`, where the ray leaves the near tier, and its partial
/// occluders nothing up to `mid_t`, where it leaves the mid sphere (`near_t <= mid_t`, each 0
/// when the origin is already past it), in the previous frame's world when `previous`. The
/// module's bindings are group [`Self::BIND_GROUP`] of `bind_group_layout`, bound from
/// [`FarFieldVisibilityBindGroup`].
/// Inside the module `package` is `bevy_solari`, so it may import `package::scene::bindings`.
#[derive(Resource)]
pub struct FarFieldVisibility {
    /// Keeps the module's shader asset alive; replacing the asset at this handle recompiles the
    /// pipelines.
    pub module: Handle<Shader>,
    pub bind_group_layout: BindGroupLayoutDescriptor,
}

impl FarFieldVisibility {
    /// The bind group index the module's bindings take in every pipeline that traces visibility.
    pub const BIND_GROUP: u32 = 2;
    /// The module path the tiers shader imports `far_field_visibility` from.
    const MODULE_PATH: &'static str = "embedded://bevy_solari/scene/far_field.wesl";

    /// Registers `source` as the far field module and installs the hook in the render world.
    /// Once per app: a second module at the path replaces the first.
    pub fn install(
        app: &mut App,
        source: impl Into<Cow<'static, str>>,
        bind_group_layout: BindGroupLayoutDescriptor,
    ) {
        let module = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wesl(source, Self::MODULE_PATH));
        app.sub_app_mut(RenderApp).insert_resource(Self {
            module,
            bind_group_layout,
        });
    }
}

/// The bind group of [`FarFieldVisibility::bind_group_layout`] the pipelines bind. The
/// application inserts it in the render world, present whenever Solari renders, and replaces it
/// when its bindings change.
#[derive(Resource)]
pub struct FarFieldVisibilityBindGroup(pub BindGroup);
