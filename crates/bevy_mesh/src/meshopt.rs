use alloc::collections::BTreeMap;

use crate::{Indices, MeshAttributeData, MeshVertexAttributeId, VertexFormat};

pub(crate) fn generate_vertex_remap(
    attributes: &BTreeMap<MeshVertexAttributeId, MeshAttributeData>,
    vertex_count: usize,
) -> Option<(usize, Vec<u32>)> {
    // meshoptimizer accepts at most 16 streams.
    if attributes.is_empty() || attributes.len() > 16 || vertex_count == 0 {
        return None;
    }
    assert!(
        u32::try_from(vertex_count).is_ok(),
        "The number of vertices exceeds u32::MAX"
    );
    let streams: Vec<_> = attributes
        .values()
        .map(|attribute| {
            let mut stream = meshopt::VertexStream::new(attribute.values.get_bytes().as_ptr());
            let size = VertexFormat::from(&attribute.values).size() as usize;
            stream.size = size;
            stream.stride = size;
            stream
        })
        .collect();
    // Each stream points into `attributes` and holds at least `vertex_count` tightly
    // packed values.
    Some(meshopt::generate_vertex_remap_multi(
        vertex_count,
        &streams,
        None,
    ))
}

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
