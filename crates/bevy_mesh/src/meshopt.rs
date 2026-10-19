use crate::Indices;

pub(crate) fn generate_tangents(
    indices: Option<&Indices>,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    corner_count: usize,
    options: meshopt::TangentOptions,
) -> Vec<[f32; 4]> {
    if corner_count == 0 {
        return Vec::new();
    }
    let converted_indices;
    let indices = match indices {
        Some(Indices::U16(indices)) => {
            converted_indices = indices[..corner_count]
                .iter()
                .map(|&i| u32::from(i))
                .collect::<Vec<_>>();
            Some(converted_indices.as_slice())
        }
        Some(Indices::U32(indices)) => Some(&indices[..corner_count]),
        None => None,
    };
    let vertex_count = if indices.is_some() {
        positions.len()
    } else {
        corner_count
    };
    meshopt::generate_tangents(
        indices,
        &positions[..vertex_count],
        &normals[..vertex_count],
        &uvs[..vertex_count],
        options,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Mesh, TangentAlgorithm, VertexAttributeValues};
    use bevy_asset::RenderAssetUsages;
    use wgpu_types::PrimitiveTopology;

    #[test]
    fn meshopt_modes_use_the_requested_weighting_and_bevy_handedness() {
        let positions = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [-2., 0., 0.]];
        let normals = [[0., 0., 1.]; 4];
        let uvs = [[0., 0.], [1., 0.], [0., 1.], [-1., 0.7]];
        let indices = [0, 1, 2, 0, 2, 3];
        let mut outputs = Vec::new();
        for (algorithm, options) in [
            (TangentAlgorithm::Meshopt, meshopt::TangentOptions::None),
            (
                TangentAlgorithm::MeshoptCompatible,
                meshopt::TangentOptions::Compatible,
            ),
        ] {
            let expected =
                meshopt::generate_tangents(Some(&indices), &positions, &normals, &uvs, options);
            let mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions.to_vec())
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals.to_vec())
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs.to_vec())
            .with_inserted_indices(Indices::U32(indices.to_vec()))
            .with_computed_tangents(algorithm)
            .unwrap();
            let VertexAttributeValues::Float32x4(actual) =
                mesh.attribute(Mesh::ATTRIBUTE_TANGENT).unwrap()
            else {
                panic!()
            };
            for (index, &[x, y, z, w]) in mesh.indices().unwrap().iter().zip(&expected) {
                assert_eq!(actual[index], [x, y, z, -w]);
            }
            outputs.push(expected);
        }
        assert_ne!(
            outputs[0], outputs[1],
            "fixture must distinguish the two weighting modes"
        );
    }
}
