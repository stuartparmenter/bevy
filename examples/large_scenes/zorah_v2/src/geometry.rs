//! Triangle-level helpers the bake applies to a decoded primitive before the
//! meshlet build: normal synthesis, winding repair, spatial partitioning and
//! per-partition reindexing.

use bevy::math::Vec3;

/// Decoded geometry of one primitive, always positions + normals + UVs.
pub struct Geometry {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

/// Area-weighted vertex normals for a primitive that exports none.
pub fn smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut sums = vec![Vec3::ZERO; positions.len()];
    for triangle in indices.chunks_exact(3) {
        let [a, b, c] =
            [triangle[0], triangle[1], triangle[2]].map(|i| Vec3::from(positions[i as usize]));
        let face = (b - a).cross(c - a);
        for index in triangle {
            sums[*index as usize] += face;
        }
    }
    sums.into_iter()
        .map(|sum| sum.try_normalize().unwrap_or(Vec3::Y).to_array())
        .collect()
}

/// Flips the winding of a primitive whose faces oppose its vertex normals.
///
/// Detected from face and vertex normal agreement over the first triangles,
/// as v1 did for its converter output, rather than flipping unconditionally.
pub fn repair_inverted_winding(geometry: &mut Geometry) -> bool {
    let mut aligned = 0usize;
    let mut opposed = 0usize;
    for triangle in geometry.indices.chunks_exact(3).take(8192) {
        let [i0, i1, i2] = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        let (p0, p1, p2) = (
            Vec3::from(geometry.positions[i0]),
            Vec3::from(geometry.positions[i1]),
            Vec3::from(geometry.positions[i2]),
        );
        let shading = Vec3::from(geometry.normals[i0])
            + Vec3::from(geometry.normals[i1])
            + Vec3::from(geometry.normals[i2]);
        let face = (p1 - p0).cross(p2 - p0);
        let denominator = face.length() * shading.length();
        if denominator <= 1e-12 {
            continue;
        }
        let agreement = face.dot(shading) / denominator;
        if agreement > 0.1 {
            aligned += 1;
        } else if agreement < -0.1 {
            opposed += 1;
        }
    }
    let inverted = opposed != 0 && opposed > aligned.saturating_mul(3);
    if inverted {
        for triangle in geometry.indices.chunks_exact_mut(3) {
            triangle.swap(1, 2);
        }
    }
    inverted
}

/// Cuts a primitive's triangles into spatial groups of at most `threshold`
/// by recursive median bisection on the longest axis of their centroids.
/// Returns triangle ids (index / 3) per partition.
pub fn partition_triangles(geometry: &Geometry, threshold: usize) -> Vec<Vec<u32>> {
    let centroids = geometry
        .indices
        .chunks_exact(3)
        .map(|triangle| {
            let sum = triangle
                .iter()
                .map(|index| Vec3::from(geometry.positions[*index as usize]))
                .sum::<Vec3>();
            sum / 3.0
        })
        .collect::<Vec<_>>();
    let mut partitions = Vec::new();
    bisect(
        &centroids,
        (0..centroids.len() as u32).collect(),
        threshold.max(1),
        &mut partitions,
    );
    partitions
}

fn bisect(centroids: &[Vec3], mut triangles: Vec<u32>, threshold: usize, out: &mut Vec<Vec<u32>>) {
    if triangles.len() <= threshold {
        out.push(triangles);
        return;
    }
    let (min, max) = triangles.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(min, max), id| {
            let centroid = centroids[*id as usize];
            (min.min(centroid), max.max(centroid))
        },
    );
    let extent = max - min;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    let mid = triangles.len() / 2;
    triangles.select_nth_unstable_by(mid, |a, b| {
        centroids[*a as usize][axis].total_cmp(&centroids[*b as usize][axis])
    });
    let upper = triangles.split_off(mid);
    bisect(centroids, triangles, threshold, out);
    bisect(centroids, upper, threshold, out);
}

/// Extracts the vertices `triangle_ids` touch into a compact primitive.
/// `remap` is scratch of one `u32::MAX` per source vertex, restored on return.
pub fn reindex(geometry: &Geometry, triangle_ids: &[u32], remap: &mut [u32]) -> Geometry {
    let mut part = Geometry {
        positions: Vec::new(),
        normals: Vec::new(),
        uvs: Vec::new(),
        indices: Vec::with_capacity(triangle_ids.len() * 3),
    };
    let mut touched = Vec::new();
    for id in triangle_ids {
        let base = *id as usize * 3;
        for index in &geometry.indices[base..base + 3] {
            let source = *index as usize;
            if remap[source] == u32::MAX {
                remap[source] = part.positions.len() as u32;
                part.positions.push(geometry.positions[source]);
                part.normals.push(geometry.normals[source]);
                part.uvs.push(geometry.uvs[source]);
                touched.push(source);
            }
            part.indices.push(remap[source]);
        }
    }
    for source in touched {
        remap[source] = u32::MAX;
    }
    part
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat grid of `n` by `n` quads, two triangles each, on the XZ plane.
    fn grid(n: usize) -> Geometry {
        let mut positions = Vec::new();
        for z in 0..=n {
            for x in 0..=n {
                positions.push([x as f32, 0.0, z as f32]);
            }
        }
        let mut indices = Vec::new();
        for z in 0..n {
            for x in 0..n {
                let a = (z * (n + 1) + x) as u32;
                let b = a + 1;
                let c = a + (n + 1) as u32;
                let d = c + 1;
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        let count = positions.len();
        Geometry {
            positions,
            normals: vec![[0.0, 1.0, 0.0]; count],
            uvs: (0..count).map(|i| [i as f32, 0.0]).collect(),
            indices,
        }
    }

    #[test]
    fn partitions_cover_every_triangle_once_under_the_threshold() {
        let geometry = grid(20);
        let triangles = geometry.indices.len() / 3;
        let threshold = 37;
        let partitions = partition_triangles(&geometry, threshold);
        assert!(partitions.len() > 1);
        let mut seen = partitions.iter().flatten().copied().collect::<Vec<_>>();
        seen.sort_unstable();
        assert_eq!(seen, (0..triangles as u32).collect::<Vec<_>>());
        let mut remap = vec![u32::MAX; geometry.positions.len()];
        for ids in &partitions {
            assert!(ids.len() <= threshold, "{} triangles", ids.len());
            let part = reindex(&geometry, ids, &mut remap);
            assert_eq!(part.indices.len(), ids.len() * 3);
            assert_eq!(part.positions.len(), part.normals.len());
            assert_eq!(part.positions.len(), part.uvs.len());
            assert!(part
                .indices
                .iter()
                .all(|i| (*i as usize) < part.positions.len()));
            // Every reindexed vertex is the source vertex it stands for.
            for (id, triangle) in ids.iter().zip(part.indices.chunks_exact(3)) {
                let source = &geometry.indices[*id as usize * 3..*id as usize * 3 + 3];
                for (local, original) in triangle.iter().zip(source) {
                    assert_eq!(
                        part.positions[*local as usize],
                        geometry.positions[*original as usize]
                    );
                    assert_eq!(part.uvs[*local as usize], geometry.uvs[*original as usize]);
                }
            }
            assert!(remap.iter().all(|entry| *entry == u32::MAX));
        }
    }

    #[test]
    fn small_primitives_stay_whole() {
        let geometry = grid(3);
        let partitions = partition_triangles(&geometry, 1000);
        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0].len(), 18);
    }

    #[test]
    fn inverted_winding_is_repaired() {
        let mut geometry = grid(4);
        let original = geometry.indices.clone();
        assert!(!repair_inverted_winding(&mut geometry));
        assert_eq!(geometry.indices, original);
        for triangle in geometry.indices.chunks_exact_mut(3) {
            triangle.swap(1, 2);
        }
        assert!(repair_inverted_winding(&mut geometry));
        assert_eq!(geometry.indices, original);
    }
}
