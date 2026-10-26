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
