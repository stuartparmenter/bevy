//! The `ZBLAS001` companion of a baked partition.
//!
//! Solari builds its BLAS from an ordinary [`Mesh`], but a meshlet mesh's
//! packed vertices and local `u8` triangle indices are nothing an acceleration
//! structure can consume. The bake therefore also writes the meshlet LOD cut
//! it chose for ray tracing as plain 32-byte vertices plus `u32` indices, the
//! same layout v1's bundles carried, and this module reads it back.

use bevy::{
    asset::{io::Reader, AssetLoader, LoadContext, RenderAssetUsages},
    mesh::{Indices, Mesh},
    pbr::experimental::meshlet::MeshletRaytracingGeometry,
    reflect::TypePath,
    render::render_resource::PrimitiveTopology,
};
use thiserror::Error;

pub const ZBLAS_MAGIC: [u8; 8] = *b"ZBLAS001";
pub const ZBLAS_VERSION: u32 = 1;
const HEADER_BYTES: usize = 20;
/// Position, normal and UV as `f32`s: the vertex Solari's BLAS reads.
const VERTEX_BYTES: usize = 32;

#[derive(Debug, Error)]
pub enum ZblasError {
    #[error("invalid ZBLAS data: {0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Serializes a raytracing LOD cut as `ZBLAS001`.
pub fn encode(geometry: &MeshletRaytracingGeometry) -> Result<Vec<u8>, ZblasError> {
    let vertex_count = u32::try_from(geometry.positions.len())
        .map_err(|_| ZblasError::Invalid("more than u32::MAX vertices".into()))?;
    let index_count = u32::try_from(geometry.indices.len())
        .map_err(|_| ZblasError::Invalid("more than u32::MAX indices".into()))?;
    if geometry.normals.len() != geometry.positions.len()
        || geometry.uvs.len() != geometry.positions.len()
    {
        return Err(ZblasError::Invalid(
            "normal and uv counts differ from the position count".into(),
        ));
    }
    if !index_count.is_multiple_of(3) {
        return Err(ZblasError::Invalid(
            "index count is not a multiple of three".into(),
        ));
    }
    if geometry.indices.iter().any(|index| *index >= vertex_count) {
        return Err(ZblasError::Invalid(
            "an index exceeds the vertex count".into(),
        ));
    }
    let mut bytes = Vec::with_capacity(
        HEADER_BYTES + vertex_count as usize * VERTEX_BYTES + index_count as usize * 4,
    );
    bytes.extend_from_slice(&ZBLAS_MAGIC);
    bytes.extend_from_slice(&ZBLAS_VERSION.to_le_bytes());
    bytes.extend_from_slice(&vertex_count.to_le_bytes());
    bytes.extend_from_slice(&index_count.to_le_bytes());
    for ((position, normal), uv) in geometry
        .positions
        .iter()
        .zip(&geometry.normals)
        .zip(&geometry.uvs)
    {
        for value in position.iter().chain(normal).chain(uv) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    for index in &geometry.indices {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    Ok(bytes)
}

/// Parses `ZBLAS001` bytes into the mesh Solari builds a BLAS from.
///
/// The mesh is render-world only: Solari reads BLAS geometry from the render
/// world's `MeshAllocator` slices (`bevy_solari::scene::blas`), and nothing in
/// the main world inspects these vertices, so keeping a CPU copy of every
/// partition's LOD cut resident would only cost memory. The one thing lost is
/// the automatic `Aabb` of a `Mesh3d` entity using this mesh, which leaves the
/// few alpha-tested `Mesh3d` instances unculled - acceptable for their count.
pub fn decode(bytes: &[u8]) -> Result<Mesh, ZblasError> {
    if bytes.len() < HEADER_BYTES || bytes[..8] != ZBLAS_MAGIC {
        return Err(ZblasError::Invalid("wrong magic".into()));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != ZBLAS_VERSION {
        return Err(ZblasError::Invalid(format!(
            "version {version} is unsupported"
        )));
    }
    let vertex_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let index_count = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    let expected = vertex_count
        .checked_mul(VERTEX_BYTES)
        .and_then(|size| size.checked_add(index_count.checked_mul(4)?))
        .and_then(|size| size.checked_add(HEADER_BYTES))
        .ok_or_else(|| ZblasError::Invalid("byte count overflow".into()))?;
    if bytes.len() != expected {
        return Err(ZblasError::Invalid(format!(
            "{} bytes, expected {expected}",
            bytes.len()
        )));
    }
    if !index_count.is_multiple_of(3) {
        return Err(ZblasError::Invalid(
            "index count is not a multiple of three".into(),
        ));
    }

    let mut offset = HEADER_BYTES;
    let mut read_f32 = || {
        let value = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        value
    };
    let mut positions = Vec::with_capacity(vertex_count);
    let mut normals = Vec::with_capacity(vertex_count);
    let mut uvs = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        positions.push([read_f32(), read_f32(), read_f32()]);
        normals.push([read_f32(), read_f32(), read_f32()]);
        uvs.push([read_f32(), read_f32()]);
    }
    let indices = bytes[HEADER_BYTES + vertex_count * VERTEX_BYTES..]
        .chunks_exact(4)
        .map(|index| u32::from_le_bytes(index.try_into().unwrap()))
        .collect::<Vec<_>>();
    if indices.iter().any(|index| *index as usize >= vertex_count) {
        return Err(ZblasError::Invalid(
            "an index exceeds the vertex count".into(),
        ));
    }

    Ok(Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(Indices::U32(indices)))
}

/// Loads `.zblas` files from the bake cache as [`Mesh`] assets.
#[derive(Default, TypePath)]
pub struct ZblasLoader;

impl AssetLoader for ZblasLoader {
    type Asset = Mesh;
    type Settings = ();
    type Error = ZblasError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Mesh, ZblasError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        decode(&bytes)
    }

    fn extensions(&self) -> &[&str] {
        &["zblas"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;

    fn geometry() -> MeshletRaytracingGeometry {
        MeshletRaytracingGeometry {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.5, 1.5, -2.0],
            ],
            normals: vec![
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.25, 0.75]],
            indices: vec![0, 1, 2, 2, 1, 3],
            achieved_error: 0.01,
        }
    }

    #[test]
    fn round_trips() {
        let source = geometry();
        let bytes = encode(&source).unwrap();
        assert_eq!(bytes.len(), HEADER_BYTES + 4 * VERTEX_BYTES + 6 * 4);
        let mesh = decode(&bytes).unwrap();
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions are not float3");
        };
        let Some(VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("normals are not float3");
        };
        let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
        else {
            panic!("uvs are not float2");
        };
        assert_eq!(positions, &source.positions);
        assert_eq!(normals, &source.normals);
        assert_eq!(uvs, &source.uvs);
        assert_eq!(
            mesh.indices()
                .unwrap()
                .iter()
                .map(|index| index as u32)
                .collect::<Vec<_>>(),
            source.indices
        );
    }

    #[test]
    fn rejects_corrupt_data() {
        let bytes = encode(&geometry()).unwrap();
        assert!(decode(&bytes[..bytes.len() - 1]).is_err());
        let mut wrong_magic = bytes.clone();
        wrong_magic[0] = b'X';
        assert!(decode(&wrong_magic).is_err());
        let mut out_of_range = bytes.clone();
        let last = out_of_range.len() - 4;
        out_of_range[last..].copy_from_slice(&7u32.to_le_bytes());
        assert!(decode(&out_of_range).is_err());
    }
}
