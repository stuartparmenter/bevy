use crate::MeshAccessError;

use super::{Indices, Mesh, VertexAttributeValues};
use thiserror::Error;
use wgpu_types::{PrimitiveTopology, VertexFormat};

struct MikktspaceGeometryHelper<'a> {
    indices: Option<&'a Indices>,
    positions: &'a Vec<[f32; 3]>,
    normals: &'a Vec<[f32; 3]>,
    uvs: &'a Vec<[f32; 2]>,
    tangents: Vec<[f32; 4]>,
}

impl MikktspaceGeometryHelper<'_> {
    fn index(&self, face: usize, vert: usize) -> usize {
        let index_index = face * 3 + vert;

        match self.indices {
            Some(Indices::U16(indices)) => indices[index_index] as usize,
            Some(Indices::U32(indices)) => indices[index_index] as usize,
            None => index_index,
        }
    }
}

impl bevy_mikktspace::Geometry for MikktspaceGeometryHelper<'_> {
    fn num_faces(&self) -> usize {
        self.indices
            .map(Indices::len)
            .unwrap_or_else(|| self.positions.len())
            / 3
    }

    fn num_vertices_of_face(&self, _: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.index(face, vert)]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.index(face, vert)]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.uvs[self.index(face, vert)]
    }

    fn set_tangent(
        &mut self,
        tangent_space: Option<bevy_mikktspace::TangentSpace>,
        face: usize,
        vert: usize,
    ) {
        self.tangents[face * 3 + vert] = tangent_space.unwrap_or_default().tangent_encoded();
    }
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
    #[error("the '{0}' vertex attribute length does not match vertex positions")]
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
    // Validate before passing the streams to native code. Extra attribute values
    // and incomplete trailing triangles are ignored, as by the Rust backend.
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
    let mut tangents = generate_tangents_rust(indices, positions, normals, uvs)?;

    // Bevy uses the opposite bitangent sign convention.
    for tangent in &mut tangents {
        tangent[3] = -tangent[3];
    }
    Ok(tangents)
}

fn generate_tangents_rust(
    indices: Option<&Indices>,
    positions: &Vec<[f32; 3]>,
    normals: &Vec<[f32; 3]>,
    uvs: &Vec<[f32; 2]>,
) -> Result<Vec<[f32; 4]>, GenerateTangentsError> {
    let mut mikktspace_mesh = MikktspaceGeometryHelper {
        indices,
        positions,
        normals,
        uvs,
        tangents: vec![[0.; 4]; indices.map_or(positions.len(), Indices::len) / 3 * 3],
    };
    bevy_mikktspace::generate_tangents(&mut mikktspace_mesh)?;

    Ok(mikktspace_mesh.tangents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::RenderAssetUsages;

    fn quad() -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0., 0., 1.]; 4])
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_UV_0,
            vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
        )
        .with_inserted_indices(Indices::U32(vec![0, 1, 2, 0, 2, 3]))
    }

    fn mirrored_seam() -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., -1., 0.]],
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0., 0., 1.]; 4])
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_UV_0,
            vec![[0., 0.], [1., 0.], [0., 1.], [0., 1.]],
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_JOINT_INDEX,
            VertexAttributeValues::Uint16x4(vec![[0; 4], [1; 4], [2; 4], [3; 4]]),
        )
        .with_inserted_indices(Indices::U16(vec![0, 1, 2, 1, 0, 3, 1, 0, 3]))
    }

    #[test]
    fn splits_mirrored_seams_and_copies_every_attribute() {
        let mut mesh = mirrored_seam();
        let original = mesh.clone();
        mesh.generate_tangents().unwrap();
        assert_eq!(mesh.count_vertices(), 6);
        assert_eq!(
            mesh.indices().unwrap().len(),
            original.indices().unwrap().len()
        );
        for (attribute, values) in original.attributes() {
            let new_values = mesh.attribute(attribute.id).unwrap();
            for (new, old) in mesh
                .indices()
                .unwrap()
                .iter()
                .zip(original.indices().unwrap().iter())
            {
                assert_eq!(new_values.get_bytes_at(new), values.get_bytes_at(old));
            }
        }
        let VertexAttributeValues::Float32x4(tangents) =
            mesh.attribute(Mesh::ATTRIBUTE_TANGENT).unwrap()
        else {
            panic!()
        };
        for (corner, index) in mesh.indices().unwrap().iter().enumerate() {
            assert_eq!(
                tangents[index],
                [1., 0., 0., if corner < 3 { -1. } else { 1. }]
            );
        }
    }

    #[cfg(feature = "morph")]
    #[test]
    fn tangent_seams_preserve_all_morph_targets() {
        use crate::morph::MorphAttributes;
        use bevy_math::Vec3;
        let mut mesh = mirrored_seam();
        let original_indices = mesh.indices().unwrap().clone();
        let targets: Vec<_> = (0..8)
            .map(|i| MorphAttributes::from([Vec3::splat(i as f32); 3]))
            .collect();
        mesh.set_morph_targets(targets.clone());
        mesh.generate_tangents().unwrap();
        let actual = mesh.morph_targets().unwrap();
        assert_eq!(actual.len(), mesh.count_vertices() * 2);
        for target in 0..2 {
            for (new, old) in mesh.indices().unwrap().iter().zip(original_indices.iter()) {
                assert_eq!(
                    actual[target * mesh.count_vertices() + new],
                    targets[target * 4 + old]
                );
            }
        }
    }

    #[test]
    fn tangent_seams_promote_indices_when_u16_is_exhausted() {
        let mut mesh = mirrored_seam();
        mesh.remove_attribute(Mesh::ATTRIBUTE_JOINT_INDEX);
        let vertex_count = usize::from(u16::MAX);
        for attribute in [
            Mesh::ATTRIBUTE_POSITION,
            Mesh::ATTRIBUTE_NORMAL,
            Mesh::ATTRIBUTE_UV_0,
        ] {
            match mesh.attribute_mut(attribute).unwrap() {
                VertexAttributeValues::Float32x3(values) => {
                    values.resize(vertex_count, [0., 0., 1.]);
                }
                VertexAttributeValues::Float32x2(values) => {
                    values.resize(vertex_count, [0.; 2]);
                }
                _ => unreachable!(),
            }
        }
        mesh.generate_tangents().unwrap();
        assert!(matches!(mesh.indices(), Some(Indices::U32(_))));
        assert!(mesh
            .indices()
            .unwrap()
            .iter()
            .any(|index| index > usize::from(u16::MAX)));
        assert_eq!(mesh.count_vertices(), vertex_count + 2);
    }

    #[test]
    fn tangents_for_all_index_formats_and_unindexed_meshes() {
        let mut u32_mesh = quad();
        let mut u16_mesh = quad();
        // Incomplete trailing triangles, including invalid indices, must not reach the backend.
        u32_mesh.insert_indices(Indices::U32(vec![0, 1, 2, 0, 2, 3, u32::MAX]));
        u16_mesh.insert_indices(Indices::U16(vec![0, 1, 2, 0, 2, 3, u16::MAX]));
        let mut unindexed = quad();
        unindexed.remove_indices(); // Four vertices: one triangle and an unused trailing vertex.
        let empty = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new())
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new())
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, Vec::<[f32; 2]>::new());
        for (mut mesh, used_vertices) in [(u32_mesh, 4), (u16_mesh, 4), (unindexed, 3), (empty, 0)]
        {
            let vertex_count = mesh.count_vertices();
            mesh.generate_tangents().unwrap();
            let VertexAttributeValues::Float32x4(tangents) =
                mesh.attribute(Mesh::ATTRIBUTE_TANGENT).unwrap()
            else {
                panic!()
            };
            assert_eq!(tangents.len(), vertex_count);
            assert!(tangents[..used_vertices]
                .iter()
                .all(|t| *t == [1., 0., 0., -1.]));
        }
    }

    #[test]
    fn rejects_out_of_bounds_indices_and_short_input_streams() {
        let mut mesh = quad();
        mesh.insert_indices(Indices::U32(vec![0, 1, 4]));
        assert!(matches!(
            mesh.generate_tangents(),
            Err(GenerateTangentsError::InvalidIndex(4, 4))
        ));
        for (attribute, values) in [
            (
                Mesh::ATTRIBUTE_NORMAL,
                VertexAttributeValues::Float32x3(vec![[0., 0., 1.]; 3]),
            ),
            (
                Mesh::ATTRIBUTE_UV_0,
                VertexAttributeValues::Float32x2(vec![[0.; 2]; 3]),
            ),
        ] {
            let mut mesh = quad();
            mesh.insert_attribute(attribute, values);
            assert!(matches!(
                mesh.generate_tangents(),
                Err(GenerateTangentsError::InvalidVertexAttributeLength(name)) if name == attribute.name
            ));
        }
    }
}
