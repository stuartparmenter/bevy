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
/// The mesh must have [`Mesh::enable_raytracing`] set to true and use
/// [`bevy_mesh::PrimitiveTopology::TriangleList`]. Static meshes must have exactly
/// the vertex attributes `{POSITION, NORMAL, UV_0, TANGENT}` and use
/// [`bevy_mesh::Indices::U32`].
///
/// Entities with [`bevy_mesh::skinning::SkinnedMesh`] or
/// [`bevy_mesh::morph::MeshMorphWeights`] are deformed automatically for raytracing.
/// These meshes support additional vertex attributes, `U16` or `U32` indices, and
/// non-indexed triangles. The standard vertex attributes must use uncompressed
/// floating-point formats; skinning also requires joint indices and weights.
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
/// vertex positions in place each frame. Replace the index buffer when triangle
/// topology changes so Solari discards temporal triangle anchors.
///
/// Add [`RaytracingGeometryPreviousVertices`] to provide deformation motion history.
/// Otherwise, previous positions use the current vertices with the previous transform.
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

/// Previous-frame vertices for a [`RaytracingGeometry`] render entity.
///
/// The buffer must use the same packed vertex layout and vertex order as
/// [`RaytracingGeometryBuffers::vertex_buffer`], contain at least `vertex_count`
/// vertices, and have `STORAGE` usage. Positions are in the entity's previous local
/// space; Solari applies its previous global transform when reconstructing motion.
///
/// The producer must update this buffer before raytracing each frame. Initialize it
/// with current vertices when history is unavailable, including after topology changes.
/// Without this component, motion reconstruction only accounts for rigid transforms.
#[derive(Component, Clone)]
pub struct RaytracingGeometryPreviousVertices(pub Buffer);
