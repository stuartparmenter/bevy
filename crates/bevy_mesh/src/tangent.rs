use crate::{Indices, Mesh, MeshAccessError, VertexAttributeValues};
use bevy_reflect::Reflect;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use thiserror::Error;
use wgpu_types::{PrimitiveTopology, VertexFormat};

/// An algorithm for [`Mesh::compute_tangents`](crate::Mesh::compute_tangents).
///
/// Normal maps look correct only with tangents from the algorithm they were baked with.
/// Defaults to `Mikktspace`, which requires the `bevy_mikktspace` feature. Without that
/// feature there is no default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Reflect)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "bevy_mikktspace", derive(Default))]
pub enum TangentAlgorithm {
    /// The `MikkTSpace` algorithm, which most normal map bakers use.
    #[cfg(feature = "bevy_mikktspace")]
    #[default]
    Mikktspace,
    /// meshoptimizer's tangent generator with its default weighting.
    ///
    /// It weights triangles differently from `MikkTSpace`, so use `MeshoptCompatible` for
    /// normal maps baked with `MikkTSpace`. Requires unit-length normals.
    #[cfg(feature = "meshopt")]
    Meshopt,
    /// meshoptimizer's tangent generator with `MikkTSpace`-compatible weighting.
    ///
    /// The results are close to `MikkTSpace` but not identical, mostly near degenerate
    /// triangles. Requires unit-length normals.
    #[cfg(feature = "meshopt")]
    MeshoptCompatible,
}

#[derive(Error, Debug)]
/// Failed to generate tangents for the mesh.
pub enum GenerateTangentsError {
    #[error("cannot generate tangents for {0:?}")]
    UnsupportedTopology(PrimitiveTopology),
    #[error("missing indices")]
    MissingIndices,
    #[error("missing vertex attributes '{0}'")]
    MissingVertexAttribute(&'static str),
    #[error("the '{0}' vertex attribute should have {1:?} format")]
    InvalidVertexAttributeFormat(&'static str, VertexFormat),
    #[error("the '{0}' vertex attribute has fewer values than there are positions")]
    InvalidVertexAttributeLength(&'static str),
    #[error("index {0} is out of bounds for {1} vertices")]
    InvalidIndex(usize, usize),
    #[cfg(feature = "bevy_mikktspace")]
    #[error("mesh not suitable for tangent generation")]
    MikktspaceError(#[from] bevy_mikktspace::GenerateTangentSpaceError),
    #[error("morph target data does not contain a whole number of targets")]
    InvalidMorphTargetLength,
    #[error("Mesh access error: {0}")]
    MeshAccessError(#[from] MeshAccessError),
}

pub(crate) fn generate_tangents_for_mesh(
    mesh: &Mesh,
    algorithm: TangentAlgorithm,
) -> Result<Vec<[f32; 4]>, GenerateTangentsError> {
    match mesh.primitive_topology() {
        PrimitiveTopology::TriangleList => {}
        other => return Err(GenerateTangentsError::UnsupportedTopology(other)),
    };

    let positions = mesh.try_attribute_option(Mesh::ATTRIBUTE_POSITION)?.ok_or(
        GenerateTangentsError::MissingVertexAttribute(Mesh::ATTRIBUTE_POSITION.name),
    )?;
    let VertexAttributeValues::Float32x3(positions) = positions else {
        return Err(GenerateTangentsError::InvalidVertexAttributeFormat(
            Mesh::ATTRIBUTE_POSITION.name,
            VertexFormat::Float32x3,
        ));
    };
    let normals = mesh.try_attribute_option(Mesh::ATTRIBUTE_NORMAL)?.ok_or(
        GenerateTangentsError::MissingVertexAttribute(Mesh::ATTRIBUTE_NORMAL.name),
    )?;
    let VertexAttributeValues::Float32x3(normals) = normals else {
        return Err(GenerateTangentsError::InvalidVertexAttributeFormat(
            Mesh::ATTRIBUTE_NORMAL.name,
            VertexFormat::Float32x3,
        ));
    };
    let uvs = mesh.try_attribute_option(Mesh::ATTRIBUTE_UV_0)?.ok_or(
        GenerateTangentsError::MissingVertexAttribute(Mesh::ATTRIBUTE_UV_0.name),
    )?;
    let VertexAttributeValues::Float32x2(uvs) = uvs else {
        return Err(GenerateTangentsError::InvalidVertexAttributeFormat(
            Mesh::ATTRIBUTE_UV_0.name,
            VertexFormat::Float32x2,
        ));
    };

    let len = positions.len();
    // Extra attribute values and an incomplete trailing triangle are ignored. meshopt
    // doesn't bounds-check indices, so out-of-range indices must be rejected here.
    if normals.len() < len {
        return Err(GenerateTangentsError::InvalidVertexAttributeLength(
            Mesh::ATTRIBUTE_NORMAL.name,
        ));
    }
    if uvs.len() < len {
        return Err(GenerateTangentsError::InvalidVertexAttributeLength(
            Mesh::ATTRIBUTE_UV_0.name,
        ));
    }
    let indices = mesh.try_indices_option()?;
    let corner_count = indices.map_or(len, Indices::len) / 3 * 3;
    if let Some(indices) = indices
        && let Some(index) = indices
            .iter()
            .take(corner_count)
            .find(|&index| index >= len)
    {
        return Err(GenerateTangentsError::InvalidIndex(index, len));
    }
    let mut tangents = match algorithm {
        #[cfg(feature = "bevy_mikktspace")]
        TangentAlgorithm::Mikktspace => {
            crate::mikktspace::generate_tangents(indices, positions, normals, uvs)?
        }
        #[cfg(feature = "meshopt")]
        TangentAlgorithm::Meshopt => crate::meshopt::generate_tangents(
            indices,
            positions,
            normals,
            uvs,
            corner_count,
            meshopt::TangentOptions::None,
        ),
        #[cfg(feature = "meshopt")]
        TangentAlgorithm::MeshoptCompatible => crate::meshopt::generate_tangents(
            indices,
            positions,
            normals,
            uvs,
            corner_count,
            meshopt::TangentOptions::Compatible,
        ),
    };

    // Bevy uses the opposite bitangent sign convention.
    for tangent in &mut tangents {
        tangent[3] = -tangent[3];
    }
    Ok(tangents)
}
