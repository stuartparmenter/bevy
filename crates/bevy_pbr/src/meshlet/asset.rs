use alloc::sync::Arc;
use bevy_asset::{
    io::{Reader, Writer},
    saver::{AssetSaver, SavedAsset},
    Asset, AssetLoader, AssetPath, AsyncReadExt, AsyncWriteExt, LoadContext,
};
use bevy_math::{Vec2, Vec3};
use bevy_reflect::TypePath;
use bevy_render::render_resource::ShaderType;
use bevy_tasks::block_on;
use bytemuck::{Pod, Zeroable};
use core::mem::offset_of;
use lz4_flex::frame::{FrameDecoder, FrameEncoder};
use std::io::{Read, Write};
use thiserror::Error;

/// Unique identifier for the [`MeshletMesh`] asset format.
const MESHLET_MESH_ASSET_MAGIC: u64 = 1717551717668;

/// The current version of the [`MeshletMesh`] asset format.
pub const MESHLET_MESH_ASSET_VERSION: u64 = 4;

/// A mesh that has been pre-processed into multiple small clusters of triangles called meshlets.
///
/// A [`bevy_mesh::Mesh`] can be converted to a [`MeshletMesh`] using `MeshletMesh::from_mesh` when the `meshlet_processor` cargo feature is enabled.
/// The conversion step is very slow, and is meant to be ran once ahead of time, and not during runtime. This type of mesh is not suitable for
/// dynamically generated geometry.
///
/// There are restrictions on the [`Material`](`crate::Material`) functionality that can be used with this type of mesh.
/// * Materials have no control over the vertex shader or vertex attributes.
/// * Materials must be opaque. Transparent, alpha masked, and transmissive materials are not supported.
/// * Do not use normal maps baked from higher-poly geometry. Use the high-poly geometry directly and skip the normal map.
///   * If additional detail is needed, a smaller tiling normal map not baked from a mesh is ok.
/// * Material shaders must not use builtin functions that automatically calculate derivatives <https://gpuweb.github.io/gpuweb/wgsl/#derivatives>.
///   * Performing manual arithmetic on texture coordinates (UVs) is forbidden. Use the chain-rule version of arithmetic functions instead (TODO: not yet implemented).
/// * Limited control over [`bevy_render::render_resource::RenderPipelineDescriptor`] attributes.
/// * Materials must use the [`Material::meshlet_mesh_fragment_shader`](`crate::Material::meshlet_mesh_fragment_shader`) method (and similar variants for prepass/deferred shaders)
///   which requires certain shader patterns that differ from the regular material shaders.
///
/// See also [`MeshletMesh3d`](`super::MeshletMesh3d`) and [`MeshletPlugin`](`super::MeshletPlugin`).
#[derive(Asset, TypePath, Clone)]
pub struct MeshletMesh {
    /// Quantized and bitstream-packed vertex positions for meshlet vertices.
    pub(crate) vertex_positions: Arc<[u32]>,
    /// Octahedral-encoded and 2x16snorm packed normals for meshlet vertices.
    pub(crate) vertex_normals: Arc<[u32]>,
    /// Per-meshlet 2x16unorm packed vertex texture coordinates.
    pub(crate) vertex_uvs: Arc<[u32]>,
    /// Triangle indices for meshlets.
    pub(crate) indices: Arc<[u8]>,
    /// The BVH8 used for culling and LOD selection of the meshlets. The root is at index 0.
    pub(crate) bvh: Arc<[BvhNode]>,
    /// The list of meshlets making up this mesh.
    pub(crate) meshlets: Arc<[Meshlet]>,
    /// Spherical bounding volumes.
    pub(crate) meshlet_cull_data: Arc<[MeshletCullData]>,
    /// The tight AABB of the meshlet mesh, used for frustum and occlusion culling at the instance
    /// level.
    pub(crate) aabb: MeshletAabb,
    /// The depth of the culling BVH, used to determine the number of dispatches at runtime.
    pub(crate) bvh_depth: u32,
}

impl MeshletMesh {
    /// The number of meshlets in this mesh, across every LOD.
    pub fn meshlet_count(&self) -> usize {
        self.meshlets.len()
    }

    /// The number of triangles in this mesh across every LOD, which is what its meshlet data
    /// scales with; the finest LOD alone is the source mesh's count.
    pub fn triangle_count(&self) -> usize {
        self.meshlets
            .iter()
            .map(|meshlet| meshlet.triangle_count as usize)
            .sum()
    }

    /// The bytes the meshlet manager uploads for this mesh: its seven packed streams laid out
    /// back to back at the manager's section alignment, exactly as `pack_meshlet_mesh` in
    /// `meshlet_mesh_manager.rs` sizes an allocation. A scene deciding what fits in the manager's
    /// `MESHLET_MAX_PAGES` pages of `MESHLET_PAGE_SIZE` bytes budgets with this.
    pub fn packed_byte_len(&self) -> usize {
        super::meshlet_mesh_manager::packed_meshlet_mesh_len(self)
    }
}

/// A fixed-error meshlet LOD decoded into hardware ray-tracing input geometry.
///
/// Hardware acceleration structures cannot consume [`MeshletMesh`]'s variable-bit packed
/// positions or local `u8` triangle indices directly. This is the minimal companion data they
/// require. Tangents are intentionally omitted: ray-hit shading reconstructs the tangent frame
/// from triangle positions and texture coordinates.
#[cfg(feature = "meshlet_processor")]
pub struct MeshletRaytracingGeometry {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    /// The largest geometric error any selected meshlet carries, which is what this geometry
    /// actually deviates from the full-detail surface by - at most the requested `max_error`, and
    /// usually far less. A consumer biasing rays off a rasterized surface wants this rather than the
    /// request, because a mesh that simplifies to itself deviates by nothing and asking it to answer
    /// for the request inflates every other instance's bias with it. Zero for empty geometry.
    pub achieved_error: f32,
}

#[cfg(feature = "meshlet_processor")]
impl MeshletMesh {
    /// Selects the meshlet LOD whose absolute geometric error is at most `max_error`, then
    /// decodes it into the standard triangle buffers required for a hardware BLAS.
    pub fn raytracing_geometry(&self, max_error: f32) -> MeshletRaytracingGeometry {
        assert!(max_error.is_finite() && max_error >= 0.0);
        let mut selected = Vec::new();
        self.select_raytracing_meshlets(0, max_error, &mut selected);

        let vertex_count = selected
            .iter()
            .map(|&id| self.meshlets[id].vertex_count_minus_one as usize + 1)
            .sum();
        let index_count = selected
            .iter()
            .map(|&id| self.meshlets[id].triangle_count as usize * 3)
            .sum();
        let mut geometry = MeshletRaytracingGeometry {
            positions: Vec::with_capacity(vertex_count),
            normals: Vec::with_capacity(vertex_count),
            uvs: Vec::with_capacity(vertex_count),
            indices: Vec::with_capacity(index_count),
            achieved_error: selected
                .iter()
                .map(|&id| self.meshlet_cull_data[id].aabb.error)
                .fold(0.0, f32::max),
        };

        for meshlet_id in selected {
            let meshlet = &self.meshlets[meshlet_id];
            let base_vertex = geometry.positions.len() as u32;
            let vertex_count = meshlet.vertex_count_minus_one as usize + 1;
            for vertex_id in 0..vertex_count {
                geometry
                    .positions
                    .push(self.decode_position(meshlet, vertex_id as u32).to_array());
                geometry
                    .normals
                    .push(self.decode_normal(meshlet, vertex_id).to_array());
                geometry
                    .uvs
                    .push(self.decode_uv(meshlet, vertex_id).to_array());
            }
            let index_start = meshlet.start_index_id as usize;
            let index_end = index_start + meshlet.triangle_count as usize * 3;
            geometry.indices.extend(
                self.indices[index_start..index_end]
                    .iter()
                    .map(|index| base_vertex + *index as u32),
            );
        }
        geometry
    }

    fn select_raytracing_meshlets(
        &self,
        node_id: usize,
        max_error: f32,
        selected: &mut Vec<usize>,
    ) {
        let node = &self.bvh[node_id];
        for child in 0..8 {
            let child_count = node.child_counts[child];
            if child_count == 0 {
                break;
            }
            let child_data = node.aabbs[child];
            // If the parent approximation is already within the requested error, its finer
            // children are intentionally skipped; the corresponding parent meshlets live in
            // another BVH leaf and will pass the per-meshlet error test below.
            if child_data.error <= max_error {
                continue;
            }
            if child_count == u8::MAX {
                self.select_raytracing_meshlets(
                    child_data.child_offset as usize,
                    max_error,
                    selected,
                );
            } else {
                let start = child_data.child_offset as usize;
                let end = start + child_count as usize;
                selected.extend((start..end).filter(|&meshlet_id| {
                    self.meshlet_cull_data[meshlet_id].aabb.error <= max_error
                }));
            }
        }
    }

    fn decode_position(&self, meshlet: &Meshlet, vertex_id: u32) -> Vec3 {
        let bits = [
            meshlet.bits_per_vertex_position_channel_x,
            meshlet.bits_per_vertex_position_channel_y,
            meshlet.bits_per_vertex_position_channel_z,
        ];
        let bits_per_vertex = bits.iter().map(|bits| *bits as u32).sum::<u32>();
        let mut start_bit = meshlet.start_vertex_position_bit + vertex_id * bits_per_vertex;
        let mut packed = [0u32; 3];
        for channel in 0..3 {
            packed[channel] = read_packed_bits(&self.vertex_positions, start_bit, bits[channel]);
            start_bit += bits[channel] as u32;
        }
        let scale = ((1u32 << meshlet.vertex_position_quantization_factor) as f32) * 100.0;
        Vec3::new(
            packed[0] as f32 + meshlet.min_vertex_position_channel_x,
            packed[1] as f32 + meshlet.min_vertex_position_channel_y,
            packed[2] as f32 + meshlet.min_vertex_position_channel_z,
        ) / scale
    }

    fn decode_normal(&self, meshlet: &Meshlet, vertex_id: usize) -> Vec3 {
        let packed = self.vertex_normals[meshlet.start_vertex_attribute_id as usize + vertex_id];
        let x = (packed as u16 as i16 as f32 / i16::MAX as f32).max(-1.0);
        let y = ((packed >> 16) as u16 as i16 as f32 / i16::MAX as f32).max(-1.0);
        let mut normal = Vec3::new(x, y, 1.0 - x.abs() - y.abs());
        let t = (-normal.z).clamp(0.0, 1.0);
        normal.x += if normal.x >= 0.0 { -t } else { t };
        normal.y += if normal.y >= 0.0 { -t } else { t };
        normal.normalize_or_zero()
    }

    fn decode_uv(&self, meshlet: &Meshlet, vertex_id: usize) -> Vec2 {
        let packed = self.vertex_uvs[meshlet.start_vertex_attribute_id as usize + vertex_id];
        let normalized = Vec2::new(
            (packed as u16) as f32 / u16::MAX as f32,
            ((packed >> 16) as u16) as f32 / u16::MAX as f32,
        );
        meshlet.min_vertex_uv + normalized * meshlet.vertex_uv_extent
    }
}

#[cfg(feature = "meshlet_processor")]
fn read_packed_bits(words: &[u32], start_bit: u32, bit_count: u8) -> u32 {
    if bit_count == 0 {
        return 0;
    }
    let word = start_bit as usize / 32;
    let shift = start_bit & 31;
    let mut value = words[word] >> shift;
    if shift + bit_count as u32 > 32 {
        value |= words[word + 1] << (32 - shift);
    }
    let mask = if bit_count == 32 {
        u32::MAX
    } else {
        (1u32 << bit_count) - 1
    };
    value & mask
}

/// A single BVH8 node in the BVH used for culling and LOD selection of a [`MeshletMesh`].
#[derive(Copy, Clone, Default, Pod, Zeroable)]
#[repr(C)]
pub struct BvhNode {
    /// The tight AABBs of this node's children, used for frustum and occlusion during BVH
    /// traversal.
    pub aabbs: [MeshletAabbErrorOffset; 8],
    /// The LOD bounding spheres of this node's children, used for LOD selection during BVH
    /// traversal.
    pub lod_bounds: [MeshletBoundingSphere; 8],
    /// If `u8::MAX`, it indicates that the child of each children is a BVH node, otherwise it is the number of meshlets in the group.
    pub child_counts: [u8; 8],
    pub _padding: [u32; 2],
}

// `load_bvh_subnode` in `meshlet_bindings.wgsl` hardcodes this layout as a 100-word stride with
// `lod_bounds` at word 64 and `child_counts` at word 96.
const _: () = assert!(size_of::<BvhNode>() == 400, "BvhNode stride changed");
const _: () = assert!(
    offset_of!(BvhNode, lod_bounds) == 256 && offset_of!(BvhNode, child_counts) == 384,
    "BvhNode field offsets changed"
);

/// A single meshlet within a [`MeshletMesh`].
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct Meshlet {
    /// The bit offset within the parent mesh's [`MeshletMesh::vertex_positions`] buffer where the vertex positions for this meshlet begin.
    pub start_vertex_position_bit: u32,
    /// The offset within the parent mesh's [`MeshletMesh::vertex_normals`] and [`MeshletMesh::vertex_uvs`] buffers
    /// where non-position vertex attributes for this meshlet begin.
    pub start_vertex_attribute_id: u32,
    /// The offset within the parent mesh's [`MeshletMesh::indices`] buffer where the indices for this meshlet begin.
    pub start_index_id: u32,
    /// The amount of vertices in this meshlet (minus one to fit 256 in a `u8`).
    pub vertex_count_minus_one: u8,
    /// The amount of triangles in this meshlet.
    pub triangle_count: u8,
    /// Unused.
    pub padding: u16,
    /// Number of bits used to store the X channel of vertex positions within this meshlet.
    pub bits_per_vertex_position_channel_x: u8,
    /// Number of bits used to store the Y channel of vertex positions within this meshlet.
    pub bits_per_vertex_position_channel_y: u8,
    /// Number of bits used to store the Z channel of vertex positions within this meshlet.
    pub bits_per_vertex_position_channel_z: u8,
    /// Power of 2 factor used to quantize vertex positions within this meshlet.
    pub vertex_position_quantization_factor: u8,
    /// Minimum quantized X channel value of vertex positions within this meshlet.
    pub min_vertex_position_channel_x: f32,
    /// Minimum quantized Y channel value of vertex positions within this meshlet.
    pub min_vertex_position_channel_y: f32,
    /// Minimum quantized Z channel value of vertex positions within this meshlet.
    pub min_vertex_position_channel_z: f32,
    /// Minimum texture coordinate used to decode this meshlet's packed UVs.
    pub min_vertex_uv: Vec2,
    /// Texture-coordinate extent used to decode this meshlet's packed UVs.
    pub vertex_uv_extent: Vec2,
}

// `load_meshlet` in `meshlet_bindings.wgsl` hardcodes this layout as a 12-word stride, reading the
// counts and position bit widths as the packed words 3 and 4, and the UV box as words 8 to 11.
const _: () = assert!(size_of::<Meshlet>() == 48, "Meshlet stride changed");
const _: () = assert!(
    offset_of!(Meshlet, vertex_count_minus_one) == 12
        && offset_of!(Meshlet, bits_per_vertex_position_channel_x) == 16
        && offset_of!(Meshlet, min_vertex_uv) == 32,
    "Meshlet field offsets changed"
);

/// Bounding spheres used for culling and choosing level of detail for a [`Meshlet`].
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct MeshletCullData {
    /// Tight bounding box, used for frustum and occlusion culling for this meshlet.
    pub aabb: MeshletAabbErrorOffset,
    /// Bounding sphere used for determining if this meshlet's group is at the correct level of detail for a given view.
    pub lod_group_sphere: MeshletBoundingSphere,
}

// `load_meshlet_cull_data` in `meshlet_bindings.wgsl` hardcodes this layout as a 12-word stride
// with `lod_group_sphere` at word 8.
const _: () = assert!(
    size_of::<MeshletCullData>() == 48 && offset_of!(MeshletCullData, lod_group_sphere) == 32,
    "MeshletCullData layout changed"
);

/// An axis-aligned bounding box used for a [`Meshlet`].
#[derive(Copy, Clone, Default, Pod, Zeroable, ShaderType)]
#[repr(C)]
pub struct MeshletAabb {
    pub center: Vec3,
    pub half_extent: Vec3,
}

// An axis-aligned bounding box used for a [`Meshlet`].
#[derive(Copy, Clone, Default, Pod, Zeroable, ShaderType)]
#[repr(C)]
pub struct MeshletAabbErrorOffset {
    pub center: Vec3,
    pub error: f32,
    pub half_extent: Vec3,
    pub child_offset: u32,
}

// `load_aabb_error_offset` in `meshlet_bindings.wgsl` reads exactly 8 words.
const _: () = assert!(
    size_of::<MeshletAabbErrorOffset>() == 32,
    "MeshletAabbErrorOffset stride changed"
);

/// A spherical bounding volume used for a [`Meshlet`].
#[derive(Copy, Clone, Default, Pod, Zeroable)]
#[repr(C)]
pub struct MeshletBoundingSphere {
    pub center: Vec3,
    pub radius: f32,
}

/// An [`AssetSaver`] for `.meshlet_mesh` [`MeshletMesh`] assets.
#[derive(TypePath)]
pub struct MeshletMeshSaver;

impl AssetSaver for MeshletMeshSaver {
    type Asset = MeshletMesh;
    type Settings = ();
    type OutputLoader = MeshletMeshLoader;
    type Error = MeshletMeshSaveOrLoadError;

    async fn save(
        &self,
        writer: &mut Writer,
        asset: SavedAsset<'_, '_, MeshletMesh>,
        _settings: &(),
        _asset_path: AssetPath<'_>,
    ) -> Result<(), MeshletMeshSaveOrLoadError> {
        asset.write(&mut AsyncWriteSyncAdapter(writer))
    }
}

impl MeshletMesh {
    /// Encodes a `.meshlet_mesh` file, as [`MeshletMeshSaver`] does; the inverse of
    /// [`Self::read`].
    pub fn write(&self, writer: &mut dyn Write) -> Result<(), MeshletMeshSaveOrLoadError> {
        writer.write_all(&MESHLET_MESH_ASSET_MAGIC.to_le_bytes())?;
        writer.write_all(&MESHLET_MESH_ASSET_VERSION.to_le_bytes())?;
        writer.write_all(bytemuck::bytes_of(&self.aabb))?;
        writer.write_all(bytemuck::bytes_of(&self.bvh_depth))?;

        // Compress and write asset data
        let mut writer = FrameEncoder::new(writer);
        write_slice(&self.vertex_positions, &mut writer)?;
        write_slice(&self.vertex_normals, &mut writer)?;
        write_slice(&self.vertex_uvs, &mut writer)?;
        write_slice(&self.indices, &mut writer)?;
        write_slice(&self.bvh, &mut writer)?;
        write_slice(&self.meshlets, &mut writer)?;
        write_slice(&self.meshlet_cull_data, &mut writer)?;
        writer.finish()?;
        Ok(())
    }
}

/// An [`AssetLoader`] for `.meshlet_mesh` [`MeshletMesh`] assets.
#[derive(TypePath)]
pub struct MeshletMeshLoader;

impl AssetLoader for MeshletMeshLoader {
    type Asset = MeshletMesh;
    type Settings = ();
    type Error = MeshletMeshSaveOrLoadError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<MeshletMesh, MeshletMeshSaveOrLoadError> {
        MeshletMesh::read(&mut AsyncReadSyncAdapter(reader))
    }

    fn extensions(&self) -> &[&str] {
        &["meshlet_mesh"]
    }
}

impl MeshletMesh {
    /// Decodes a `.meshlet_mesh` file, as [`MeshletMeshLoader`] does, for a caller without an
    /// asset server: a tool measuring a cache of them, or a loader wrapping the format in one of
    /// its own.
    pub fn read(reader: &mut dyn Read) -> Result<MeshletMesh, MeshletMeshSaveOrLoadError> {
        let magic = read_u64(reader)?;
        if magic != MESHLET_MESH_ASSET_MAGIC {
            return Err(MeshletMeshSaveOrLoadError::WrongFileType);
        }
        let version = read_u64(reader)?;
        if version != MESHLET_MESH_ASSET_VERSION {
            return Err(MeshletMeshSaveOrLoadError::WrongVersion { found: version });
        }

        let mut bytes = [0u8; size_of::<MeshletAabb>()];
        reader.read_exact(&mut bytes)?;
        let aabb = bytemuck::cast(bytes);
        let mut bytes = [0u8; size_of::<u32>()];
        reader.read_exact(&mut bytes)?;
        let bvh_depth = u32::from_le_bytes(bytes);

        let reader = &mut FrameDecoder::new(reader);
        let vertex_positions = read_slice(reader)?;
        let vertex_normals = read_slice(reader)?;
        let vertex_uvs = read_slice(reader)?;
        let indices = read_slice(reader)?;
        let bvh = read_slice(reader)?;
        let meshlets = read_slice(reader)?;
        let meshlet_cull_data = read_slice(reader)?;

        Ok(MeshletMesh {
            vertex_positions,
            vertex_normals,
            vertex_uvs,
            indices,
            bvh,
            meshlets,
            meshlet_cull_data,
            aabb,
            bvh_depth,
        })
    }
}

#[derive(Error, Debug)]
pub enum MeshletMeshSaveOrLoadError {
    #[error("file was not a MeshletMesh asset")]
    WrongFileType,
    #[error("expected asset version {MESHLET_MESH_ASSET_VERSION} but found version {found}")]
    WrongVersion { found: u64 },
    #[error("failed to compress or decompress asset data")]
    CompressionOrDecompression(#[from] lz4_flex::frame::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

fn read_u64(reader: &mut dyn Read) -> Result<u64, std::io::Error> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn write_slice<T: Pod>(
    field: &[T],
    writer: &mut dyn Write,
) -> Result<(), MeshletMeshSaveOrLoadError> {
    writer.write_all(&(field.len() as u64).to_le_bytes())?;
    writer.write_all(bytemuck::cast_slice(field))?;
    Ok(())
}

fn read_slice<T: Pod>(reader: &mut dyn Read) -> Result<Arc<[T]>, std::io::Error> {
    let len = read_u64(reader)? as usize;

    let mut data: Arc<[T]> = core::iter::repeat_with(T::zeroed).take(len).collect();
    let slice = Arc::get_mut(&mut data).unwrap();
    reader.read_exact(bytemuck::cast_slice_mut(slice))?;

    Ok(data)
}

// TODO: Use async for everything and get rid of this adapter
struct AsyncWriteSyncAdapter<'a>(&'a mut Writer);

impl Write for AsyncWriteSyncAdapter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        block_on(self.0.write(buf))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        block_on(self.0.flush())
    }
}

// TODO: Use async for everything and get rid of this adapter
struct AsyncReadSyncAdapter<'a>(&'a mut dyn Reader);

impl Read for AsyncReadSyncAdapter<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        block_on(self.0.read(buf))
    }
}
