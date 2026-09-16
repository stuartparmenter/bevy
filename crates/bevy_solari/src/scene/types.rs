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

/// Raytracing geometry written directly on the GPU (compute-filled
/// vertex/index buffers) instead of loaded from a [`Mesh`] asset — terrain
/// tiles, expanded grass blades, GPU-skinned meshes.
///
/// This is just the marker. The producer inserts [`RaytracingGeometryBuffers`]
/// on the render entity and fills the buffers with its own compute pass,
/// which must be submitted before solari builds BLASes in
/// `RenderSystems::PrepareBindGroups` — a `PrepareResources` system with its
/// own `queue.submit` works.
///
/// Remove by despawning the entity or removing this component.
/// `Visibility::Hidden` is ignored, like [`RaytracingMesh3d`].
///
/// The material must be [`MeshMaterial3d<StandardMaterial>`]. Keep it
/// emissive-black unless the geometry should be an area light.
#[derive(Component, Clone, Copy, Debug, Default, Reflect, PartialEq, Eq)]
#[reflect(Component, Default, Clone, PartialEq)]
#[require(MeshMaterial3d<StandardMaterial>, Transform, SyncToRenderWorld)]
pub struct RaytracingGeometry;

/// How solari maintains the BLAS for a [`RaytracingGeometry`] entity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub enum RaytracingGeometryUpdateMode {
    /// Build the BLAS once, when the buffers first appear. For geometry that
    /// never changes after the initial fill. Uses `PREFER_FAST_TRACE`.
    ///
    /// Rewriting the buffer contents later does nothing — rays keep hitting
    /// the old geometry. Swap in new buffers, or use
    /// [`RebuildEveryFrame`](Self::RebuildEveryFrame).
    #[default]
    BuildOnce,
    /// Rebuild the BLAS every frame from the current buffer contents. For
    /// geometry the producer re-fills each frame (skinned meshes, grass).
    /// Uses `PREFER_FAST_BUILD`.
    RebuildEveryFrame,
}

/// Render-world component holding a [`RaytracingGeometry`] entity's buffers.
/// The producer creates and fills them; solari builds the BLAS and emits the
/// TLAS instance.
///
/// Both buffers need `STORAGE | BLAS_INPUT` usage. Insert once; for
/// [`RaytracingGeometryUpdateMode::RebuildEveryFrame`] the producer re-fills
/// the contents in place each frame.
///
/// Motion contract: solari does not double-buffer these buffers, so
/// previous-frame positions are reconstructed as the previous-frame transform
/// applied to the *current* vertex data. Consumers (e.g. specular motion
/// vectors) see rigid motion only; in-place re-fills contribute no
/// deformation motion.
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
    /// Bytes per packed vertex — the `PackedVertex` layout the hit shaders
    /// read: position `vec3`, normal `vec3`, uv `vec2`, tangent `vec4`.
    pub const VERTEX_STRIDE: u64 = 48;
}
