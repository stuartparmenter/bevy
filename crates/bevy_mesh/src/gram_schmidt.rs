//! Gram-Schmidt tangents for meshes without `MikkTSpace`-baked normal maps.

use crate::{GenerateTangentsError, Indices, Mesh};
use bevy_math::{DVec2, DVec3};

pub(crate) fn generate_tangents(
    indices: Option<&Indices>,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
) -> Result<Vec<[f32; 4]>, GenerateTangentsError> {
    // The caller validates stream lengths and indices. f64 keeps the UV determinant
    // and the sums in range for any finite f32 input.
    for (attribute, finite) in [
        (
            Mesh::ATTRIBUTE_POSITION,
            positions.iter().flatten().all(|v| v.is_finite()),
        ),
        (
            Mesh::ATTRIBUTE_NORMAL,
            normals[..positions.len()]
                .iter()
                .flatten()
                .all(|v| v.is_finite()),
        ),
        (
            Mesh::ATTRIBUTE_UV_0,
            uvs[..positions.len()]
                .iter()
                .flatten()
                .all(|v| v.is_finite()),
        ),
    ] {
        if !finite {
            return Err(GenerateTangentsError::NonFiniteVertexAttribute(
                attribute.name,
            ));
        }
    }
    let normals = normals[..positions.len()]
        .iter()
        .enumerate()
        .map(|(index, &normal)| {
            DVec3::from_array(normal.map(f64::from))
                .try_normalize()
                .ok_or(GenerateTangentsError::ZeroNormal(index))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let corner_count = indices.map_or(positions.len(), Indices::len) / 3 * 3;
    let index = |corner| match indices {
        Some(Indices::U16(indices)) => usize::from(indices[corner]),
        Some(Indices::U32(indices)) => indices[corner] as usize,
        None => corner,
    };

    // Sum tangents per vertex and handedness. `Mesh::compute_tangents` splits vertices
    // used with both. Vertices are not welded by position.
    let mut sums = vec![[DVec3::ZERO; 2]; positions.len()];
    let mut orientations = vec![0u8; corner_count];
    for (face, orientations) in orientations.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let vertices = [index(face * 3), index(face * 3 + 1), index(face * 3 + 2)];
        let [p0, p1, p2] = vertices.map(|i| DVec3::from_array(positions[i].map(f64::from)));
        let [uv0, uv1, uv2] = vertices.map(|i| DVec2::from_array(uvs[i].map(f64::from)));
        let dp1 = p1 - p0;
        let dp2 = p2 - p0;
        let duv1 = uv1 - uv0;
        let duv2 = uv2 - uv0;
        let determinant = duv1.perp_dot(duv2);
        // An absolute epsilon would incorrectly discard valid, small UV charts.
        if determinant == 0.0 {
            continue;
        }
        let tangent = (dp1 * duv2.y - dp2 * duv1.y) / determinant;
        let bitangent = (dp2 * duv1.x - dp1 * duv2.x) / determinant;
        for (corner, vertex) in vertices.into_iter().enumerate() {
            let orientation = normals[vertex].cross(tangent).dot(bitangent);
            if orientation == 0.0 {
                // The normal is coplanar with the tangent and bitangent.
                continue;
            }
            let group = usize::from(orientation < 0.0);
            orientations[corner] = group as u8;
            sums[vertex][group] += tangent;
        }
    }

    // Normalize first so the tolerance doesn't depend on mesh or UV scale. Sums that are
    // zero or parallel to the normal fall back to an arbitrary perpendicular vector.
    for (normal, groups) in normals.iter().zip(&mut sums) {
        for tangent in groups {
            let direction = tangent.normalize_or_zero();
            let projected = direction - *normal * normal.dot(direction);
            *tangent = if projected.length_squared() > 64.0 * f64::EPSILON * f64::EPSILON {
                projected.normalize()
            } else {
                normal.any_orthonormal_vector()
            };
        }
    }
    Ok(orientations
        .into_iter()
        .enumerate()
        .map(|(corner, group)| {
            let tangent = sums[index(corner)][usize::from(group)].as_vec3();
            // `generate_tangents_for_mesh` flips w to Bevy's convention.
            tangent
                .extend(if group == 0 { 1.0 } else { -1.0 })
                .to_array()
        })
        .collect())
}
