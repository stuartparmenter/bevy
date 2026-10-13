use bevy_asset::Handle;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component, prelude::ReflectComponent, resource::Resource, template::FromTemplate,
};
use bevy_image::Image;
use bevy_math::{Vec2, Vec3};
use bevy_mesh::Mesh;
use bevy_pbr::{MeshMaterial3d, StandardMaterial};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::{
    extract_resource::ExtractResource, render_resource::Buffer, sync_world::SyncToRenderWorld,
    RenderApp,
};
use bevy_transform::components::Transform;
use derive_more::derive::From;

/// A mesh component used for raytracing.
///
/// The mesh must declare the BLAS its material needs in [`Mesh::raytracing`] and use
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
/// topology changes so Solari discards temporal triangle anchors. If triangle
/// identities change within existing buffers, update
/// [`RaytracingGeometryTopologyGeneration`] instead.
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

/// Producer-maintained identity generation for triangles in external geometry.
///
/// Insert this component in the render world and change its value whenever a
/// triangle index starts referring to a different surface (for example, after
/// compacting a procedural mesh). This discards temporal geometry anchors while
/// keeping buffer allocations and BLAS allocations intact. Ordinary deformation
/// with stable triangle identities must leave the generation unchanged.
///
/// Absence is equivalent to generation zero. This does not request a BLAS rebuild;
/// use [`RaytracingGeometryUpdateMode::RebuildEveryFrame`] for changing geometry.
#[derive(Component, Clone, Copy, Default, PartialEq, Eq)]
pub struct RaytracingGeometryTopologyGeneration(pub u64);

/// A producer-defined value for a raytracing instance. Solari never interprets it.
///
/// Insert this component on the instance's render entity. Shaders read it as
/// `InstanceGeometryIds.user_tag` from `geometry_ids`, indexed by instance slot.
/// Absence is equivalent to 0.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RaytracingInstanceTag(pub u32);

/// Which rays trace a raytracing instance: the mask of its TLAS instance.
///
/// Bit 0 ([`Self::ALL_RAYS`]) makes the instance visible to every ray. A directional shadow ray
/// past [`ShadowTiers::near_radius`] instead picks one of bits 1 to 7 uniformly at random, so an
/// instance set in `k` of them occludes `k / 7` of that light ([`Self::partial_occluder`]).
/// Insert it beside [`RaytracingMesh3d`] or [`RaytracingGeometry`]; absence is
/// [`Self::ALL_RAYS`].
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Reflect)]
#[reflect(Component, Default, Clone, PartialEq)]
pub struct RaytracingInstanceMask(pub u8);

impl RaytracingInstanceMask {
    /// Visible to every ray and every shadow tier.
    pub const ALL: Self = Self(0xFF);
    /// Visible to every ray that is not a directional shadow ray past the near radius.
    pub const ALL_RAYS: Self = Self(1);
    /// Bits a directional shadow ray past the near radius picks one of.
    pub const PARTIAL_BITS: u8 = 0xFE;
    /// How many it picks among: the shader's `partial_mask` takes `rand % PARTIAL_COUNT`.
    pub const PARTIAL_COUNT: u32 = Self::PARTIAL_BITS.count_ones();

    /// Occludes `opacity` (clamped to `[0, 1]`, in sevenths) of directional light past the near
    /// radius, and nothing else. `salt` rotates which bits are set: a ray through occluders of
    /// different salts is blocked by their union, so they compound like independent ones, while
    /// occluders of one salt block only as much as the most opaque of them.
    pub fn partial_occluder(opacity: f32, salt: u32) -> Self {
        Self::partial_occluder_range(0.0, opacity, salt)
    }

    /// The bits of [`Self::partial_occluder`]`(to, salt)` beyond those of `(from, salt)`.
    /// Consecutive ranges of one salt are disjoint and union to the outermost `to`.
    pub fn partial_occluder_range(from: f32, to: f32, salt: u32) -> Self {
        let count = Self::PARTIAL_COUNT;
        let bits = |opacity: f32| (opacity.clamp(0.0, 1.0) * count as f32).round() as u32;
        let set = ((1u32 << bits(to)) - 1) & !((1u32 << bits(from)) - 1);
        let rotate = salt % count;
        let rotated = ((set << rotate) | (set >> (count - rotate))) & ((1 << count) - 1);
        Self((rotated << Self::PARTIAL_BITS.trailing_zeros()) as u8)
    }
}

#[cfg(test)]
mod instance_mask_tests {
    use super::RaytracingInstanceMask;

    #[test]
    fn partial_occluders_take_sevenths() {
        assert_eq!(RaytracingInstanceMask::PARTIAL_COUNT, 7);
        assert_eq!(RaytracingInstanceMask::partial_occluder(0.0, 5).0, 0);
        assert_eq!(
            RaytracingInstanceMask::partial_occluder(1.0 / 7.0, 0).0,
            0b10
        );
        assert_eq!(RaytracingInstanceMask::partial_occluder(1.0, 3).0, 0xFE);
        // Salts rotate the bits within 1 to 7.
        assert_eq!(
            RaytracingInstanceMask::partial_occluder(2.0 / 7.0, 6).0,
            0x82
        );
        assert_eq!(
            RaytracingInstanceMask::partial_occluder(2.0 / 7.0, 7).0,
            0b110
        );
    }

    #[test]
    fn partial_occluder_ranges_nest() {
        let outer = RaytracingInstanceMask::partial_occluder_range(0.0, 1.0 / 7.0, 3).0;
        let mid = RaytracingInstanceMask::partial_occluder_range(1.0 / 7.0, 3.0 / 7.0, 3).0;
        let core = RaytracingInstanceMask::partial_occluder_range(3.0 / 7.0, 6.0 / 7.0, 3).0;
        assert_eq!(outer.count_ones(), 1);
        assert_eq!(outer & mid, 0);
        assert_eq!(mid & core, 0);
        assert_eq!(
            outer | mid | core,
            RaytracingInstanceMask::partial_occluder(6.0 / 7.0, 3).0
        );
        // A range rounding to the same seventh takes no bits.
        assert_eq!(
            RaytracingInstanceMask::partial_occluder_range(0.52, 0.56, 0).0,
            0
        );
    }
}

impl Default for RaytracingInstanceMask {
    fn default() -> Self {
        Self::ALL_RAYS
    }
}

/// Splits directional light visibility into tiers by distance from a centre, so that near geometry
/// is traced in full while cheaper stand-ins cover the distance.
///
/// A shadow ray from a surface inside the `near_radius` sphere first traces the scene to where it
/// leaves that sphere, against instances with bit 0 of their [`RaytracingInstanceMask`]. From
/// there to where it leaves the `mid_radius` sphere it traces one random bit of 1 to 7, culling
/// back faces so a receiver inside a closed proxy of its own is not shadowed by it. Past that, and
/// from the surface itself for one outside the mid sphere, it marches the [`ShadowHeightfield`].
/// Every ray that is not a directional shadow ray traces bit 0 at any distance.
///
/// Insert and update this resource in the main world. Temporal reuse traces the previous frame's
/// scene with the previous frame's tiers.
#[derive(Resource, Clone, Debug, PartialEq, ExtractResource)]
#[extract_app(RenderApp)]
pub struct ShadowTiers {
    pub center: Vec3,
    pub near_radius: f32,
    /// At least `near_radius`.
    pub mid_radius: f32,
    pub heightfield: Option<ShadowHeightfield>,
}

/// Ground and canopy heightfields that directional shadow rays march past the scene's geometry.
///
/// Both textures cover the same world square, `origin` to `origin + extent` in XZ, texel (0, 0)
/// at `origin`. A ray below the ground by more than `ground_slack` is blocked. A ray crossing a
/// canopy texel's band, from `canopy_base` of its canopy height above the ground to the full
/// height, is attenuated by `coverage * (1 - exp(-canopy_extinction * length))`.
#[derive(Clone, Debug, PartialEq)]
pub struct ShadowHeightfield {
    /// `R32Float` ground height in metres, with a full mip chain of per-texel maxima.
    pub ground: Handle<Image>,
    /// `Rgba8Unorm`: r = share of the texel under canopy, g = canopy height over
    /// `canopy_height_max`, b = the greatest canopy height in the texel over `canopy_height_max`.
    /// Mips average r, average g weighted by r, and take the maximum of b.
    pub canopy: Handle<Image>,
    pub origin: Vec2,
    pub extent: f32,
    /// The greatest height any texel's canopy top reaches, in metres.
    pub height_max: f32,
    /// Metres a canopy height of 1 encodes.
    pub canopy_height_max: f32,
    /// Share of the canopy height below which a texel is clear.
    pub canopy_base: f32,
    /// Extinction per metre of ray inside the canopy band.
    pub canopy_extinction: f32,
    /// Horizontal distance from a surface inside the canopy over which canopy is ignored, so a
    /// surface is not shadowed by the canopy texels it is part of.
    pub canopy_self_radius: f32,
    /// Metres the ground may stand above a ray before it is blocked: the difference between the
    /// ground texture and the surfaces rays leave from.
    pub ground_slack: f32,
    /// Metres above the ground at which a surface counts as inside the canopy.
    pub canopy_surface_height: f32,
}
