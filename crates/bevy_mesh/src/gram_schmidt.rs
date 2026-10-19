//! An inexpensive approximation for procedural meshes without `MikkTSpace`-baked maps.

use crate::{GenerateTangentsError, Indices, Mesh};
use bevy_math::{DVec2, DVec3};

pub(crate) fn generate_tangents(
    indices: Option<&Indices>,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
) -> Result<Vec<[f32; 4]>, GenerateTangentsError> {
    // The caller has already validated stream lengths and triangle indices.
    // f64 intermediates keep differences, determinants and accumulation in range
    // for finite f32 inputs, including very small UV charts and large positions.
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

    // Average within the existing vertex topology, but never combine opposite
    // handedness. Each group produces one frame; the shared mesh adapter splits
    // vertices when those frames differ. No position-based welding is performed.
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
                // The derivative frame is degenerate relative to this normal.
                continue;
            }
            let group = usize::from(orientation < 0.0);
            orientations[corner] = group as u8;
            sums[vertex][group] += tangent;
        }
    }

    // Normalize before projection to make the parallel-vector tolerance relative
    // to direction, not to geometry size or UV scale. Degenerate/cancelled sums
    // get a deterministic unit vector perpendicular to the supplied normal.
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
            // The shared adapter converts this conventional sign to Bevy's convention.
            tangent
                .extend(if group == 0 { 1.0 } else { -1.0 })
                .to_array()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PrimitiveTopology, TangentAlgorithm, VertexAttributeValues};
    use bevy_asset::RenderAssetUsages;
    use bevy_math::{Vec3, Vec4};

    fn triangle(positions: [[f32; 3]; 3], normal: [f32; 3], uvs: [[f32; 2]; 3]) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions.to_vec())
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![normal; 3])
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs.to_vec())
    }

    fn assert_valid_frames(mesh: &Mesh) {
        let VertexAttributeValues::Float32x4(tangents) =
            mesh.attribute(Mesh::ATTRIBUTE_TANGENT).unwrap()
        else {
            panic!()
        };
        let normals = mesh
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .unwrap()
            .as_float3()
            .unwrap();
        for (&tangent, &normal) in tangents.iter().zip(normals) {
            let tangent = Vec4::from(tangent);
            let normal = DVec3::from_array(normal.map(f64::from))
                .normalize()
                .as_vec3();
            assert!(tangent.is_finite());
            assert!((tangent.truncate().length_squared() - 1.0).abs() < 2e-6);
            assert!(tangent.truncate().dot(normal).abs() < 2e-6);
            assert_eq!(tangent.w.abs(), 1.0);
        }
    }

    #[test]
    fn analytic_rotated_frame_and_mirrored_u() {
        let normal = DVec3::new(1.0, 2.0, 3.0).normalize();
        let tangent = normal.any_orthonormal_vector().as_vec3();
        let bitangent = normal.as_vec3().cross(tangent);
        for mirror in [1.0, -1.0] {
            let mut mesh = triangle(
                [
                    Vec3::ZERO.to_array(),
                    tangent.to_array(),
                    bitangent.to_array(),
                ],
                [2.0, 4.0, 6.0],
                [[0.0, 0.0], [mirror, 0.0], [0.0, 1.0]],
            );
            mesh.compute_tangents(TangentAlgorithm::GramSchmidt)
                .unwrap();
            assert_valid_frames(&mesh);
            let VertexAttributeValues::Float32x4(actual) =
                mesh.attribute(Mesh::ATTRIBUTE_TANGENT).unwrap()
            else {
                panic!()
            };
            for &actual in actual {
                let actual = Vec4::from(actual);
                assert!(actual.truncate().distance(tangent * mirror) < 2e-6);
                // Bevy's bitangent convention is opposite to the usual UV derivative frame.
                assert_eq!(actual.w, -mirror);
            }
        }
    }

    #[test]
    fn small_uvs_and_extreme_scales_preserve_the_frame() {
        // Small valid charts must not be discarded by an absolute determinant epsilon.
        // The other cases exercise f32 underflow/overflow in derivatives and normalization.
        for (geometry_scale, uv_scale, normal_scale) in [
            (1.0, 1e-12, 1.0),
            (1e-30, 1e30, f32::from_bits(1)),
            (1e30, f32::from_bits(1), 1e30),
        ] {
            let mut mesh = triangle(
                [
                    [0.0; 3],
                    [geometry_scale, 0.0, 0.0],
                    [0.0, geometry_scale, 0.0],
                ],
                [0.0, 0.0, normal_scale],
                [[0.0; 2], [uv_scale, 0.0], [0.0, uv_scale]],
            );
            mesh.compute_tangents(TangentAlgorithm::GramSchmidt)
                .unwrap();
            assert_valid_frames(&mesh);
            let VertexAttributeValues::Float32x4(actual) =
                mesh.attribute(Mesh::ATTRIBUTE_TANGENT).unwrap()
            else {
                panic!()
            };
            assert!(actual.iter().all(|&t| t == [1.0, 0.0, 0.0, -1.0]));
        }
    }

    #[test]
    fn degenerate_parallel_and_cancelling_derivatives_have_unit_fallbacks() {
        let positions = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let uvs = [[0.0; 2], [1.0, 0.0], [0.0, 1.0]];
        let cancelling = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![
                [0.0; 3],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 5])
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_UV_0,
            vec![[0.0; 2], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        )
        .with_inserted_indices(Indices::U32(vec![0, 1, 2, 0, 3, 4]));
        for mut mesh in [
            triangle([[0.0; 3]; 3], [1.0, 2.0, 3.0], uvs),
            triangle(positions, [1.0, 2.0, 3.0], [[0.0; 2]; 3]),
            triangle(positions, [1.0, 0.0, 0.0], uvs),
            cancelling,
        ] {
            mesh.compute_tangents(TangentAlgorithm::GramSchmidt)
                .unwrap();
            assert_valid_frames(&mesh);
        }
    }
}
