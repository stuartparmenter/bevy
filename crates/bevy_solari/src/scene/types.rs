use bevy_asset::Handle;
use bevy_color::LinearRgba;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{component::Component, prelude::ReflectComponent, template::FromTemplate};
use bevy_mesh::Mesh;
use bevy_pbr::{MeshGeometryError, MeshMaterial3d, StandardMaterial};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::sync_world::SyncToRenderWorld;
use bevy_transform::components::Transform;
use derive_more::derive::From;

/// A mesh component used for raytracing.
///
/// The mesh used in this component must have [`Mesh::enable_raytracing`] set to true, use
/// [`bevy_mesh::PrimitiveTopology::TriangleList`], use a non-empty [`bevy_mesh::Indices::U32`]
/// index buffer, and use one of two vertex attribute sets: `{POSITION, NORMAL, UV_0}`, where the
/// tangent frame is reconstructed from the UVs at hit time, or `{POSITION, NORMAL, UV_0, TANGENT}`.
///
/// The material used for this entity must be [`MeshMaterial3d<StandardMaterial>`].
#[derive(
    Component, FromTemplate, Clone, Debug, Default, Deref, DerefMut, Reflect, PartialEq, Eq, From,
)]
#[reflect(Component, Default, Clone, PartialEq)]
#[require(
    MeshMaterial3d<StandardMaterial>,
    Transform,
    SyncToRenderWorld,
    MeshGeometryError
)]
pub struct RaytracingMesh3d(pub Handle<Mesh>);

/// A uniform upper-hemisphere environment source used by Solari.
///
/// `illuminance` is the horizontal-surface illuminance in lux. Solari converts
/// this to constant sky radiance with `L = E / PI` and samples directions
/// uniformly over the world-space `+Y` hemisphere. This provides a compact
/// bridge for captured sky lights until textured environment maps are bound by
/// the raytracing scene.
#[derive(Component, Clone, Copy, Debug, Reflect, PartialEq)]
#[reflect(Component, Clone, PartialEq)]
#[require(SyncToRenderWorld)]
pub struct SolariEnvironmentLight {
    pub color: LinearRgba,
    pub illuminance: f32,
}

impl Default for SolariEnvironmentLight {
    fn default() -> Self {
        Self {
            color: LinearRgba::WHITE,
            illuminance: 1_000.0,
        }
    }
}
