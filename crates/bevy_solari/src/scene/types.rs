use bevy_asset::Handle;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{component::Component, prelude::ReflectComponent, template::FromTemplate};
use bevy_mesh::Mesh;
use bevy_pbr::{MeshMaterial3d, StandardMaterial};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::{render_resource::Buffer, sync_world::SyncToRenderWorld};
use bevy_transform::components::Transform;
use derive_more::derive::From;

/// A mesh component used for raytracing.
///
/// The mesh used in this component must have [`Mesh::enable_raytracing`] set to true,
/// use the following set of vertex attributes: `{POSITION, NORMAL, UV_0, TANGENT}`, use [`bevy_mesh::PrimitiveTopology::TriangleList`],
/// and use [`bevy_mesh::Indices::U32`].
///
/// The material used for this entity must be [`MeshMaterial3d<StandardMaterial>`].
#[derive(
    Component, FromTemplate, Clone, Debug, Default, Deref, DerefMut, Reflect, PartialEq, Eq, From,
)]
#[reflect(Component, Default, Clone, PartialEq)]
#[require(MeshMaterial3d<StandardMaterial>, Transform, SyncToRenderWorld)]
pub struct RaytracingMesh3d(pub Handle<Mesh>);

/// A component for raytracing geometry generated on the GPU.
///
/// The producer inserts [`RaytracingGeometryBuffers`] on the render entity and fills
/// the vertex and index buffers with a compute pass. Record this pass in
/// [`RaytracingProducerEncoder`] during `RenderSystems::PrepareResources`.
/// Solari records the BLAS builds afterward and submits the encoder in
/// `RenderSystems::PrepareBindGroups`.
///
/// [`RaytracingProducerEncoder`]: super::RaytracingProducerEncoder
///
/// The render entity's [`GlobalTransform`](bevy_transform::components::GlobalTransform)
/// is kept in sync with the TLAS instance transform, so producers can use it to
/// convert vertices to entity-local space.
///
/// Remove the component or despawn the entity to remove its geometry.
/// Like [`RaytracingMesh3d`], this component ignores `Visibility::Hidden`.
///
/// The material must be [`MeshMaterial3d<StandardMaterial>`]. An emissive material
/// makes the geometry an area light.
#[derive(Component, Clone, Copy, Debug, Default, Reflect, PartialEq, Eq)]
#[reflect(Component, Default, Clone, PartialEq)]
#[require(MeshMaterial3d<StandardMaterial>, Transform, SyncToRenderWorld)]
pub struct RaytracingGeometry;

/// How Solari maintains the BLAS for a [`RaytracingGeometry`] entity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub enum RaytracingGeometryUpdateMode {
    /// Build the BLAS once using `PREFER_FAST_TRACE`.
    ///
    /// For geometry that does not change after the initial build. Replace the buffers
    /// or use [`RebuildEveryFrame`](Self::RebuildEveryFrame) to update the geometry.
    #[default]
    BuildOnce,
    /// Rebuild the BLAS every frame using `PREFER_FAST_BUILD`.
    /// For geometry whose buffer contents change each frame, such as skinned meshes.
    RebuildEveryFrame,
}

/// The vertex and index buffers for a [`RaytracingGeometry`] render entity.
///
/// The producer creates and fills these buffers. Solari builds the BLAS and adds
/// the TLAS instance. Both buffers must have `STORAGE | BLAS_INPUT` usage.
/// With [`RaytracingGeometryUpdateMode::RebuildEveryFrame`], the producer can update
/// the buffer contents in place each frame.
///
/// Solari does not double-buffer vertex data. Previous positions use the current
/// vertices with the previous transform, so motion vectors capture rigid motion
/// but not deformation.
#[derive(Component, Clone)]
pub struct RaytracingGeometryBuffers {
    /// `array<PackedVertex>`, [`VERTEX_STRIDE`](Self::VERTEX_STRIDE) bytes each.
    pub vertex_buffer: Buffer,
    /// `array<u32>` triangle-list indices.
    pub index_buffer: Buffer,
    /// Number of vertices in `vertex_buffer`.
    pub vertex_count: u32,
    /// Number of indices in `index_buffer` (a multiple of 3).
    pub index_count: u32,
    /// Whether the BLAS is built once or rebuilt every frame.
    pub update_mode: RaytracingGeometryUpdateMode,
}

impl RaytracingGeometryBuffers {
    /// Size of a packed vertex: position `vec3`, normal `vec3`, UV `vec2`, tangent `vec4`.
    pub const VERTEX_STRIDE: u64 = 48;
}
