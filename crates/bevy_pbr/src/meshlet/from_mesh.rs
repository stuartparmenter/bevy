use crate::meshlet::asset::{MeshletAabb, MeshletAabbErrorOffset, MeshletCullData};

use super::asset::{BvhNode, Meshlet, MeshletBoundingSphere, MeshletMesh};
use alloc::borrow::Cow;
use bevy_math::{ops::log2, IVec3, Isometry3d, Vec2, Vec3, Vec3A, Vec3Swizzles};
use bevy_mesh::{Indices, Mesh, MeshAttributeCompressionFlags, MeshVertexAttribute};
use bevy_platform::collections::HashMap;
use bevy_render::render_resource::PrimitiveTopology;
use bevy_shape::{Aabb3d, BoundingSphere, BoundingVolume};
use bevy_tasks::{AsyncComputeTaskPool, ParallelSlice, TaskPool};
use bitvec::{order::Lsb0, vec::BitVec, view::BitView};
use core::ops::Range;
use meshopt::{
    build_meshlets_flex, ffi::meshopt_Meshlet, generate_position_remap,
    partition_clusters_with_positions, simplify_with_attributes_and_locks,
    spatial_sort_remap_decoder, Meshlets, SimplifyOptions, VertexDataAdapter,
};
use smallvec::SmallVec;
use thiserror::Error;
use tracing::debug_span;

// The visibility buffer addresses a meshlet's triangles in 7 bits, and a meshlet stores its
// vertex count minus one in a byte.
const MESHLET_MAX_TRIANGLES: usize = 128;
const MESHLET_MAX_VERTICES: usize = 256;
// Meshlets between these two sizes are split when their bounds outgrow the expected meshlet
// size by more than the split factor. meshoptimizer 1.x takes any minimum; the "divisible by 4"
// in meshopt-rs's docs is left over from 0.x's padded triangle data.
const MESHLET_MIN_TRIANGLES: usize = MESHLET_MAX_TRIANGLES / 3;
const MESHLET_SPLIT_FACTOR: f32 = 2.0;
// Aim to have 24 meshlets per group; partitions come out at up to a third more.
const TARGET_MESHLETS_PER_GROUP: usize = 24;
// A group that keeps over 85% of its triangles is stuck. Its meshlets go on to the next LOD
// unsimplified, where they are grouped with different neighbours.
const SIMPLIFICATION_FAILURE_PERCENTAGE: f32 = 0.85;
// Simplifier error weights for each vertex's normal and UV. With absolute errors an attribute's
// deviation counts in units of the group's extent, clamped to the scale of its position error, so
// the weights trade attribute fidelity against geometric error in the LOD and ray-bias error.
const ATTRIBUTE_WEIGHTS: [f32; 5] = [0.5; 5];

/// Default vertex position quantization factor for use with [`MeshletMesh::from_mesh`].
///
/// Snaps vertices to the nearest 1/16th of a centimeter (1/2^4).
pub const MESHLET_DEFAULT_VERTEX_POSITION_QUANTIZATION_FACTOR: u8 = 4;

const CENTIMETERS_PER_METER: f32 = 100.0;

/// Grid steps per metre at a vertex position quantization factor: positions
/// snap to 1/2^factor of a centimetre.
pub fn vertex_position_quantization_scale(vertex_position_quantization_factor: u8) -> f32 {
    (1 << vertex_position_quantization_factor) as f32 * CENTIMETERS_PER_METER
}

/// The fixed-point grid position a meshlet mesh stores for `position`, which
/// is also the identity two vertices must share to be one point after the
/// build.
pub fn quantize_vertex_position(position: Vec3, vertex_position_quantization_factor: u8) -> IVec3 {
    (position * vertex_position_quantization_scale(vertex_position_quantization_factor) + 0.5)
        .as_ivec3()
}

impl MeshletMesh {
    /// Process a [`Mesh`] to generate a [`MeshletMesh`].
    ///
    /// This process is very slow, and should be done ahead of time, and not at runtime.
    ///
    /// # Requirements
    ///
    /// This function requires the `meshlet_processor` cargo feature.
    ///
    /// The input mesh must:
    /// 1. Use [`PrimitiveTopology::TriangleList`]
    /// 2. Use indices
    /// 3. Have the exact following set of vertex attributes: `{POSITION, NORMAL, UV_0}` (tangents can be used in material shaders, but are calculated at runtime and are not stored in the mesh)
    ///
    /// # Vertex precision
    ///
    /// `vertex_position_quantization_factor` is the amount of precision to use when quantizing vertex positions.
    ///
    /// Vertices are snapped to the nearest (1/2^x)th of a centimeter, where x = `vertex_position_quantization_factor`.
    /// E.g. if x = 4, then vertices are snapped to the nearest 1/2^4 = 1/16th of a centimeter.
    ///
    /// Use [`MESHLET_DEFAULT_VERTEX_POSITION_QUANTIZATION_FACTOR`] as a default, adjusting lower to save memory and disk space, and higher to prevent artifacts if needed.
    ///
    /// To ensure that two different meshes do not have cracks between them when placed directly next to each other:
    ///   * Use the same quantization factor when converting each mesh to a meshlet mesh
    ///   * Ensure that their [`bevy_transform::components::Transform::translation`]s are a multiple of 1/2^x centimeters (note that translations are in meters)
    ///   * Ensure that their [`bevy_transform::components::Transform::scale`]s are the same
    ///   * Ensure that their [`bevy_transform::components::Transform::rotation`]s are a multiple of 90 degrees
    ///
    /// Simplification runs on [`AsyncComputeTaskPool`], which must be initialized; use
    /// [`MeshletMeshBuilder`] to choose another pool or none.
    pub fn from_mesh(
        mesh: &Mesh,
        vertex_position_quantization_factor: u8,
    ) -> Result<Self, MeshToMeshletMeshConversionError> {
        MeshletMeshBuilder::new(vertex_position_quantization_factor)
            .with_task_pool(AsyncComputeTaskPool::get())
            .build(mesh)
    }

    /// Like [`Self::from_mesh`], but keeps every vertex on an open edge of the
    /// mesh where it is at every LOD.
    ///
    /// Simplification only locks the vertices that neighbouring meshlet groups
    /// share, which keeps a closed surface crack-free. A mesh that is one piece
    /// of a larger surface - a partition of a mesh too large to build at once,
    /// or a tile that meets its neighbours edge to edge - has open borders that
    /// nothing else in the mesh shares, so coarser LODs move them and the seam
    /// with the adjacent piece opens into cracks. Locking the border pins the
    /// seam at every LOD, at the cost of the border never simplifying.
    ///
    /// Every open edge is locked, whether or not anything meets it. A closed
    /// surface cut into partitions has open edges only along the cuts, so the
    /// cost is small there; a mesh that is open everywhere - separate tiles,
    /// loose shells, foliage cards, ribbons - is pinned nearly everywhere and
    /// its coarse LODs stay close to full detail. Such a mesh should use
    /// [`Self::from_mesh_with_locks`] with only the vertices it really shares.
    pub fn from_mesh_with_locked_borders(
        mesh: &Mesh,
        vertex_position_quantization_factor: u8,
    ) -> Result<Self, MeshToMeshletMeshConversionError> {
        MeshletMeshBuilder::new(vertex_position_quantization_factor)
            .with_locked_borders()
            .with_task_pool(AsyncComputeTaskPool::get())
            .build(mesh)
    }

    /// Like [`Self::from_mesh`], but keeps the vertices `locked` marks (one
    /// entry per input vertex) where they are at every LOD.
    ///
    /// This is the precise form of [`Self::from_mesh_with_locked_borders`]:
    /// the caller names the seam instead of every open edge. Locking a vertex
    /// locks its position, so every vertex sharing that position (a UV or
    /// normal seam) is held with it; a locked position persists into the
    /// coarsest LOD and is never collapsed, so lock only what geometry built
    /// separately depends on meeting - a partition cut, the edge a tile
    /// shares with its neighbour - and nothing else.
    pub fn from_mesh_with_locks(
        mesh: &Mesh,
        vertex_position_quantization_factor: u8,
        locked: &[bool],
    ) -> Result<Self, MeshToMeshletMeshConversionError> {
        MeshletMeshBuilder::new(vertex_position_quantization_factor)
            .with_locks(locked)
            .with_task_pool(AsyncComputeTaskPool::get())
            .build(mesh)
    }
}

/// Builds a [`MeshletMesh`] from a [`Mesh`], choosing what [`MeshletMesh::from_mesh`] and its
/// variants fix: which vertices stay put, and where simplification runs.
///
/// ```ignore
/// let meshlet_mesh = MeshletMeshBuilder::new(MESHLET_DEFAULT_VERTEX_POSITION_QUANTIZATION_FACTOR)
///     .with_task_pool(&task_pool)
///     .build(&mesh)?;
/// ```
///
/// The output depends only on the mesh, the quantization factor and the locks, not on the task
/// pool or its size.
#[derive(Clone, Copy, Debug)]
pub struct MeshletMeshBuilder<'a> {
    vertex_position_quantization_factor: u8,
    locks: InputLocks<'a>,
    task_pool: Option<&'a TaskPool>,
}

impl<'a> MeshletMeshBuilder<'a> {
    /// A builder that snaps vertices to `vertex_position_quantization_factor` (see
    /// [`MeshletMesh::from_mesh`]), locks only the borders between meshlet groups, and builds
    /// entirely on the calling thread.
    pub fn new(vertex_position_quantization_factor: u8) -> Self {
        Self {
            vertex_position_quantization_factor,
            locks: InputLocks::None,
            task_pool: None,
        }
    }

    /// Locks every vertex on an open edge of the mesh, as
    /// [`MeshletMesh::from_mesh_with_locked_borders`] does.
    pub fn with_locked_borders(mut self) -> Self {
        self.locks = InputLocks::OpenBorders;
        self
    }

    /// Locks the vertices `locked` marks, one entry per input vertex, as
    /// [`MeshletMesh::from_mesh_with_locks`] does.
    pub fn with_locks(mut self, locked: &'a [bool]) -> Self {
        self.locks = InputLocks::Explicit(locked);
        self
    }

    /// Simplifies each LOD's meshlet groups in parallel on `task_pool`.
    ///
    /// The rest of the build runs on the calling thread. While it waits for a level's groups,
    /// that thread runs any of `task_pool`'s queued tasks, not only its own, so a caller that is
    /// itself a task on `task_pool` can start further builds on its stack. Callers that build
    /// many meshes at once should do so from threads outside the pool.
    pub fn with_task_pool(mut self, task_pool: &'a TaskPool) -> Self {
        self.task_pool = Some(task_pool);
        self
    }

    /// Builds the LOD DAG with the rules of meshoptimizer's `clusterlod.h`, but in its own loop
    /// rather than through `meshopt::clod_build`: groups within a level simplify independently, a
    /// stuck group's meshlets are regrouped in the next level instead of becoming terminal,
    /// permissive simplification is only a fallback, and the output feeds Bevy's own BVH.
    ///
    /// The mesh must meet the requirements listed on [`MeshletMesh::from_mesh`].
    pub fn build(&self, mesh: &Mesh) -> Result<MeshletMesh, MeshToMeshletMeshConversionError> {
        let Self {
            vertex_position_quantization_factor,
            locks,
            task_pool,
        } = *self;
        let s = debug_span!("build meshlet mesh");
        let _e = s.enter();

        // Meshlet vertex data is built from float attributes.
        let mesh = if mesh.attribute_compression().is_empty() {
            Cow::Borrowed(mesh)
        } else {
            let mut mesh = mesh.clone();
            mesh.decompress_attributes(MeshAttributeCompressionFlags::all());
            Cow::Owned(mesh)
        };
        let mesh = mesh.as_ref();

        // Validate mesh format
        let indices = validate_input_mesh(mesh)?;

        // Get meshlet vertices
        let vertex_buffer = mesh.create_packed_vertex_buffer_data();
        let vertex_stride = mesh.get_vertex_size() as usize;
        let vertices = VertexDataAdapter::new(&vertex_buffer, vertex_stride, 0).unwrap();
        // Every vertex's attributes - normal, then UV - `vertex_stride` bytes apart from the
        // first vertex's.
        let vertex_attributes = bytemuck::cast_slice(&vertex_buffer[12..]);

        // Generate a position-only vertex buffer for determining triangle/meshlet connectivity
        let position_only_vertex_remap = generate_position_remap(&vertices);

        let border_locks = match locks {
            InputLocks::None => vec![false; vertices.vertex_count],
            InputLocks::OpenBorders => mesh_border_locks(&indices, &position_only_vertex_remap),
            InputLocks::Explicit(locked) => {
                if locked.len() != vertices.vertex_count {
                    return Err(MeshToMeshletMeshConversionError::WrongLockCount {
                        required: vertices.vertex_count,
                        provided: locked.len(),
                    });
                }
                position_locks(locked, &position_only_vertex_remap)
            }
        };

        // Split the mesh into an initial list of meshlets (LOD 0)
        let (mut meshlets, mut cull_data) = compute_meshlets(&indices, &vertices, None);

        let mut vertex_locks = vec![false; vertices.vertex_count];

        // Build further LODs
        let mut bvh = BvhBuilder::default();
        let mut all_groups = Vec::new();
        let mut simplification_queue: Vec<_> = (0..meshlets.len() as u32).collect();
        let mut stuck = Vec::new();
        while simplification_queue.len() > 1 {
            let s = debug_span!("simplify lod", meshlets = simplification_queue.len());
            let _e = s.enter();

            let groups = group_meshlets(
                &simplification_queue,
                &meshlets,
                &cull_data,
                &vertices,
                &position_only_vertex_remap,
            );
            simplification_queue.clear();

            // Lock borders between groups to prevent cracks when simplifying
            lock_group_borders(
                &mut vertex_locks,
                &border_locks,
                &groups,
                &meshlets,
                &position_only_vertex_remap,
            );

            let simplify = |group: &TempMeshletGroup| {
                let s = debug_span!("simplify group", meshlets = group.meshlets.len());
                let _e = s.enter();

                let (simplified_group_indices, mut group_error) = simplify_meshlet_group(
                    group,
                    &meshlets,
                    &vertices,
                    vertex_attributes,
                    &vertex_locks,
                )?;

                // Force the group error to be atleast as large as all of its constituent meshlet's
                // individual errors.
                for &id in group.meshlets.iter() {
                    group_error = group_error.max(cull_data[id as usize].error);
                }

                // Build new meshlets using the simplified group
                let new_meshlets = compute_meshlets(
                    &simplified_group_indices,
                    &vertices,
                    Some((group.lod_bounds, group_error)),
                );

                Some((group_error, new_meshlets))
            };
            let simplified: Vec<_> = match task_pool {
                Some(task_pool) => {
                    // A few chunks per thread balance groups of uneven cost without flooding the
                    // pool's queue with one task per group.
                    let chunk_size = (groups.len() / (4 * task_pool.thread_num().max(1))).max(1);
                    groups
                        .par_chunk_map(task_pool, chunk_size, |_, chunk| {
                            chunk.iter().map(simplify).collect::<Vec<_>>()
                        })
                        .into_iter()
                        .flatten()
                        .collect()
                }
                None => groups.iter().map(simplify).collect(),
            };

            let first_group = all_groups.len() as u32;
            for (mut group, simplified) in groups.into_iter().zip(simplified) {
                let Some((parent_error, (new_meshlets, new_cull_data))) = simplified else {
                    stuck.push(group);
                    continue;
                };
                group.parent_error = parent_error;

                let start = meshlets.len();
                merge_meshlets(&mut meshlets, new_meshlets);
                cull_data.extend(new_cull_data);
                let end = meshlets.len();

                simplification_queue.extend(start as u32..end as u32);
                all_groups.push(group);
            }

            // A level where no group simplified is the last: its groups are the roots
            if all_groups.len() as u32 == first_group {
                break;
            }
            bvh.add_lod(first_group, &all_groups);

            // Stuck meshlets go on to the next level, to be grouped with their simplified
            // neighbours
            simplification_queue.extend(stuck.drain(..).flat_map(|group| group.meshlets));
        }

        // A lone meshlet is a root of its own
        if !simplification_queue.is_empty() {
            stuck.extend(group_meshlets(
                &simplification_queue,
                &meshlets,
                &cull_data,
                &vertices,
                &position_only_vertex_remap,
            ));
        }

        // The roots get the final LOD, with an unbounded error so that they are never replaced
        let first_group = all_groups.len() as u32;
        all_groups.extend(stuck);
        bvh.add_lod(first_group, &all_groups);

        let (bvh, aabb, depth) = bvh.build(&mut meshlets, all_groups, &mut cull_data);

        // Copy vertex attributes per meshlet and compress
        let mut vertex_positions = BitVec::<u32, Lsb0>::new();
        let mut vertex_normals = Vec::new();
        let mut vertex_uvs = Vec::new();
        let mut bevy_meshlets = Vec::with_capacity(meshlets.len());
        for (i, meshlet) in meshlets.meshlets.iter().enumerate() {
            build_and_compress_per_meshlet_vertex_data(
                meshlet,
                meshlets.get(i).vertices,
                &vertex_buffer,
                vertex_stride,
                &mut vertex_positions,
                &mut vertex_normals,
                &mut vertex_uvs,
                &mut bevy_meshlets,
                vertex_position_quantization_factor,
            );
        }
        vertex_positions.set_uninitialized(false);

        Ok(MeshletMesh {
            vertex_positions: vertex_positions.into_vec().into(),
            vertex_normals: vertex_normals.into(),
            vertex_uvs: vertex_uvs.into(),
            indices: meshlets.triangles.into(),
            bvh: bvh.into(),
            meshlets: bevy_meshlets.into(),
            meshlet_cull_data: cull_data
                .into_iter()
                .map(|cull_data| MeshletCullData {
                    aabb: aabb_to_meshlet(cull_data.aabb, cull_data.error, 0),
                    lod_group_sphere: sphere_to_meshlet(cull_data.lod_group_sphere),
                })
                .collect(),
            aabb,
            bvh_depth: depth,
        })
    }
}

impl MeshletMesh {
    /// Drops every meshlet finer than `min_error` and rebuilds the mesh over the rest, so a scene
    /// can trade detail it cannot afford to keep resident for a bounded geometric error.
    ///
    /// What is dropped: every LOD subtree whose BVH error is strictly below `min_error` - those
    /// meshlets are only ever selected when the coarser meshlets that approximate them to within
    /// that error are perceptibly wrong. The rule is strict, unlike the `<=` of
    /// [`Self::raytracing_geometry`], so that `pruned(0.0)` is an identity on any mesh, including
    /// one whose simplification reached a zero-error level; the cost is that a subtree with an
    /// error of exactly `min_error` survives here where the `min_error` cut would skip it, so
    /// `pruned(e)` holds the geometry of the cut at the next float below `e` and can retain a
    /// little more than the `e` cut. Everything coarser survives with its packed vertex and index
    /// data compacted, and the BVH is rebuilt over the survivors, so the result is a smaller
    /// upload rather than a view of the original.
    ///
    /// Error semantics afterwards: a surviving meshlet whose own error was below `min_error` has
    /// lost the finer children it would otherwise hand over to, so its error becomes `0.0` and
    /// the runtime treats it as full detail - `cull_clusters.wgsl` only rasterizes a meshlet whose
    /// own error is imperceptible, and a positive error would cull it up close and leave a hole
    /// where its children used to be. Its `lod_group_sphere` is kept; with an error of zero it no
    /// longer affects the test. Every other meshlet and every BVH node keeps its error, so the
    /// geometry `raytracing_geometry(e)` selects from the result is the unpruned mesh's for every
    /// `e >= min_error`, while a tighter request can only reach the detail that survived and so
    /// returns the geometry pruning kept. What that call reports as `achieved_error` is not
    /// comparable, though: the rewritten zeros hide how far those meshlets sit from the true
    /// surface, so a consumer that biases by it (a BLAS cut) should select from the unpruned
    /// mesh.
    pub fn pruned(&self, min_error: f32) -> MeshletMesh {
        assert!(min_error.is_finite() && min_error >= 0.0);

        let mut groups = Vec::new();
        self.collect_pruned_groups(0, min_error, &mut groups);
        assert!(
            !groups.is_empty(),
            "the coarsest LOD carries an unbounded error and always survives"
        );

        // Lay the survivors out finest LOD first, as the bake did, so the BVH keeps each LOD in
        // its own subtree and the cull traversal can skip whole LODs at once.
        let levels = pruned_group_levels(&groups, &self.meshlet_cull_data);
        let mut order: Vec<usize> = (0..groups.len()).collect();
        order.sort_by_key(|&i| (levels[i], groups[i].aabb.child_offset));

        let source_bits = self.vertex_positions.view_bits::<Lsb0>();
        let mut vertex_positions = BitVec::<u32, Lsb0>::new();
        let mut vertex_normals = Vec::new();
        let mut vertex_uvs = Vec::new();
        let mut indices = Vec::new();
        let mut meshlets = Vec::new();
        let mut cull_data = Vec::new();
        let mut temp_groups = Vec::with_capacity(groups.len());
        let mut lods: Vec<Range<u32>> = Vec::new();
        for &group_id in &order {
            let group = &groups[group_id];
            let first = meshlets.len() as u32;
            let next = temp_groups.len() as u32;
            match lods.last_mut() {
                Some(lod) if levels[group_id] == levels[order[lod.start as usize]] => lod.end += 1,
                _ => lods.push(next..next + 1),
            }

            let start = group.aabb.child_offset as usize;
            for meshlet_id in start..start + group.count as usize {
                let meshlet = self.meshlets[meshlet_id];
                let vertex_count = meshlet.vertex_count_minus_one as usize + 1;
                let bits_per_vertex = meshlet.bits_per_vertex_position_channel_x as usize
                    + meshlet.bits_per_vertex_position_channel_y as usize
                    + meshlet.bits_per_vertex_position_channel_z as usize;
                let position_start = meshlet.start_vertex_position_bit as usize;
                let attribute_start = meshlet.start_vertex_attribute_id as usize;
                let index_start = meshlet.start_index_id as usize;
                let index_count = meshlet.triangle_count as usize * 3;

                meshlets.push(Meshlet {
                    start_vertex_position_bit: vertex_positions.len() as u32,
                    start_vertex_attribute_id: vertex_normals.len() as u32,
                    start_index_id: indices.len() as u32,
                    ..meshlet
                });
                vertex_positions.extend_from_bitslice(
                    &source_bits[position_start..position_start + vertex_count * bits_per_vertex],
                );
                vertex_normals.extend_from_slice(
                    &self.vertex_normals[attribute_start..attribute_start + vertex_count],
                );
                vertex_uvs.extend_from_slice(
                    &self.vertex_uvs[attribute_start..attribute_start + vertex_count],
                );
                indices.extend_from_slice(&self.indices[index_start..index_start + index_count]);

                let data = self.meshlet_cull_data[meshlet_id];
                let error = if data.aabb.error < min_error {
                    0.0
                } else {
                    data.aabb.error
                };
                cull_data.push(MeshletCullData {
                    aabb: MeshletAabbErrorOffset { error, ..data.aabb },
                    lod_group_sphere: data.lod_group_sphere,
                });
            }

            temp_groups.push(TempMeshletGroup {
                aabb: Aabb3d::new(group.aabb.center, group.aabb.half_extent),
                lod_bounds: BoundingSphere::new(group.lod_bounds.center, group.lod_bounds.radius),
                parent_error: group.aabb.error,
                meshlets: [first, group.count as u32].into_iter().collect(),
            });
        }
        vertex_positions.set_uninitialized(false);

        let mut bvh = BvhBuilder::default();
        for lod in lods {
            bvh.add_lod(lod.start, &temp_groups[..lod.end as usize]);
        }
        let (mut bvh, aabb, bvh_depth) = bvh.build_nodes(&temp_groups);
        // The builder round-trips leaf bounds through `Aabb3d`, which is not bit-exact; the
        // leaves are the bake's own values, so restore them.
        let group_by_first: HashMap<u32, usize> = temp_groups
            .iter()
            .zip(&order)
            .map(|(group, &group_id)| (group.meshlets[0], group_id))
            .collect();
        for node in &mut bvh {
            for child in 0..8 {
                let count = node.child_counts[child];
                if count == 0 {
                    break;
                }
                if count != u8::MAX {
                    let group = &groups[group_by_first[&node.aabbs[child].child_offset]];
                    node.aabbs[child].center = group.aabb.center;
                    node.aabbs[child].half_extent = group.aabb.half_extent;
                }
            }
        }
        verify_bvh_complete(&bvh, &temp_cull_data(&cull_data));

        MeshletMesh {
            vertex_positions: vertex_positions.into_vec().into(),
            vertex_normals: vertex_normals.into(),
            vertex_uvs: vertex_uvs.into(),
            indices: indices.into(),
            bvh: bvh.into(),
            meshlets: meshlets.into(),
            meshlet_cull_data: cull_data.into(),
            aabb,
            bvh_depth,
        }
    }

    /// The BVH leaves at or above the cut: the walk of `select_raytracing_meshlets`, but skipping
    /// only strictly finer subtrees and keeping whole groups, since a group is the unit the BVH
    /// is rebuilt from.
    fn collect_pruned_groups(&self, node_id: usize, min_error: f32, groups: &mut Vec<PrunedGroup>) {
        let node = &self.bvh[node_id];
        for child in 0..8 {
            let child_count = node.child_counts[child];
            if child_count == 0 {
                break;
            }
            let aabb = node.aabbs[child];
            if aabb.error < min_error {
                continue;
            }
            if child_count == u8::MAX {
                self.collect_pruned_groups(aabb.child_offset as usize, min_error, groups);
            } else {
                groups.push(PrunedGroup {
                    aabb,
                    lod_bounds: node.lod_bounds[child],
                    count: child_count,
                });
            }
        }
    }
}

/// The built cull data in the form the BVH builder and its verifier work in.
fn temp_cull_data(cull_data: &[MeshletCullData]) -> Vec<TempMeshletCullData> {
    cull_data
        .iter()
        .map(|data| TempMeshletCullData {
            aabb: Aabb3d::new(data.aabb.center, data.aabb.half_extent),
            lod_group_sphere: BoundingSphere::new(
                data.lod_group_sphere.center,
                data.lod_group_sphere.radius,
            ),
            error: data.aabb.error,
        })
        .collect()
}

/// A BVH leaf of a built mesh: the group it was simplified as, addressing a run of `count`
/// meshlets at `aabb.child_offset`.
struct PrunedGroup {
    aabb: MeshletAabbErrorOffset,
    lod_bounds: MeshletBoundingSphere,
    count: u8,
}

/// Recovers how many simplification generations sit below each surviving group. A meshlet carries
/// the LOD sphere and error of the group it was simplified from, verbatim, so those values name a
/// group's children among the survivors; a group whose children were all pruned is a leaf. This
/// only shapes the rebuilt BVH, so a coincidental match costs tree quality, not correctness.
fn pruned_group_levels(groups: &[PrunedGroup], cull_data: &[MeshletCullData]) -> Vec<u32> {
    fn key(sphere: MeshletBoundingSphere, error: f32) -> ([u32; 3], u32, u32) {
        (
            sphere.center.to_array().map(f32::to_bits),
            sphere.radius.to_bits(),
            error.to_bits(),
        )
    }
    fn level(
        group_id: usize,
        groups: &[PrunedGroup],
        cull_data: &[MeshletCullData],
        by_key: &HashMap<([u32; 3], u32, u32), usize>,
        levels: &mut [Option<u32>],
        visiting: &mut [bool],
    ) -> u32 {
        if let Some(level) = levels[group_id] {
            return level;
        }
        if visiting[group_id] {
            return 0;
        }
        visiting[group_id] = true;
        let group = &groups[group_id];
        let start = group.aabb.child_offset as usize;
        let mut result = 0;
        for data in &cull_data[start..start + group.count as usize] {
            if let Some(&child) = by_key.get(&key(data.lod_group_sphere, data.aabb.error))
                && child != group_id
            {
                result = result.max(1 + level(child, groups, cull_data, by_key, levels, visiting));
            }
        }
        visiting[group_id] = false;
        levels[group_id] = Some(result);
        result
    }

    let by_key: HashMap<_, _> = groups
        .iter()
        .enumerate()
        .map(|(i, group)| (key(group.lod_bounds, group.aabb.error), i))
        .collect();
    let mut levels = vec![None; groups.len()];
    let mut visiting = vec![false; groups.len()];
    (0..groups.len())
        .map(|i| level(i, groups, cull_data, &by_key, &mut levels, &mut visiting))
        .collect()
}

fn validate_input_mesh(mesh: &Mesh) -> Result<Cow<'_, [u32]>, MeshToMeshletMeshConversionError> {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return Err(MeshToMeshletMeshConversionError::WrongMeshPrimitiveTopology);
    }

    let required_attributes = [
        Mesh::ATTRIBUTE_POSITION,
        Mesh::ATTRIBUTE_NORMAL,
        Mesh::ATTRIBUTE_UV_0,
    ];
    if mesh
        .attributes()
        .map(|(attribute, _)| (attribute.id, attribute.format))
        .ne(required_attributes
            .iter()
            .map(|attribute| (attribute.id, attribute.format)))
    {
        return Err(
            MeshToMeshletMeshConversionError::WrongMeshVertexAttributes {
                required: required_attributes,
                provided: mesh.attributes().map(|(attribute, _)| *attribute).collect(),
            },
        );
    }

    let indices: Cow<'_, [u32]> = match mesh.indices() {
        Some(Indices::U32(indices)) => Cow::Borrowed(indices.as_slice()),
        Some(Indices::U16(indices)) => indices.iter().map(|i| *i as u32).collect(),
        _ => return Err(MeshToMeshletMeshConversionError::MeshMissingIndices),
    };
    if indices.is_empty() || !indices.len().is_multiple_of(3) {
        return Err(MeshToMeshletMeshConversionError::WrongIndexCount(
            indices.len(),
        ));
    }
    let vertex_count = mesh.count_vertices();
    if let Some(&index) = indices
        .iter()
        .find(|&&index| index as usize >= vertex_count)
    {
        return Err(MeshToMeshletMeshConversionError::IndexOutOfRange {
            index,
            vertex_count,
        });
    }
    Ok(indices)
}

/// Splits `indices` into meshlets. A simplified group's meshlets all carry the group's LOD sphere
/// and error, `prev_lod_data`, so whatever group they join next merges conservatively; LOD 0
/// meshlets carry their own bounds and no error.
fn compute_meshlets(
    indices: &[u32],
    vertices: &VertexDataAdapter,
    prev_lod_data: Option<(BoundingSphere, f32)>,
) -> (Meshlets, Vec<TempMeshletCullData>) {
    let mut meshlets = build_meshlets_flex(
        indices,
        vertices,
        MESHLET_MAX_VERTICES,
        MESHLET_MIN_TRIANGLES,
        MESHLET_MAX_TRIANGLES,
        0.0,
        MESHLET_SPLIT_FACTOR,
    );
    // meshopt-rs rounds the end of the triangle buffer up to four bytes and keeps the worst-case
    // capacity it reserved; drop both, so levels pack back to back and a level's groups don't
    // hold several times their size until they are merged.
    let used_triangle_bytes = meshlets.meshlets.last().map_or(0, |meshlet| {
        meshlet.triangle_offset as usize + meshlet.triangle_count as usize * 3
    });
    meshlets.triangles.truncate(used_triangle_bytes);
    meshlets.meshlets.shrink_to_fit();
    meshlets.vertices.shrink_to_fit();
    meshlets.triangles.shrink_to_fit();
    let cull_data = meshlets
        .iter()
        .map(|meshlet| {
            let (lod_group_sphere, error) = prev_lod_data.unwrap_or_else(|| {
                let bounds = meshopt::compute_meshlet_bounds(meshlet, vertices);
                (BoundingSphere::new(bounds.center, bounds.radius), 0.0)
            });
            TempMeshletCullData {
                aabb: Aabb3d::from_point_cloud(
                    Isometry3d::IDENTITY,
                    meshlet
                        .vertices
                        .iter()
                        .map(|&vertex| vertex_position(vertices, vertex)),
                ),
                lod_group_sphere,
                error,
            }
        })
        .collect();
    (meshlets, cull_data)
}

/// Partitions the queued meshlets into groups of about [`TARGET_MESHLETS_PER_GROUP`], keeping
/// meshlets that share vertices together and merging the ones that share none - the loose parts
/// of a kitbashed mesh, foliage cards - with their nearest neighbours. Groups come out in
/// spatial order, so nearby groups get nearby meshlets.
fn group_meshlets(
    simplification_queue: &[u32],
    meshlets: &Meshlets,
    meshlet_cull_data: &[TempMeshletCullData],
    vertices: &VertexDataAdapter,
    position_only_vertex_remap: &[u32],
) -> Vec<TempMeshletGroup> {
    let mut group_per_meshlet = vec![0; simplification_queue.len()];
    let mut group_count = 1;
    if simplification_queue.len() > TARGET_MESHLETS_PER_GROUP {
        // Connectivity is by position, so meshlets meeting at a UV or normal seam are neighbours
        let mut cluster_indices = Vec::new();
        let mut cluster_index_counts = Vec::with_capacity(simplification_queue.len());
        for &meshlet_id in simplification_queue {
            let meshlet = meshlets.get(meshlet_id as usize);
            cluster_indices.extend(
                meshlet
                    .vertices
                    .iter()
                    .map(|&vertex| position_only_vertex_remap[vertex as usize]),
            );
            cluster_index_counts.push(meshlet.vertices.len() as u32);
        }
        group_count = partition_clusters_with_positions(
            &mut group_per_meshlet,
            &cluster_indices,
            &cluster_index_counts,
            vertices,
            TARGET_MESHLETS_PER_GROUP,
        );

        // Any of a group's meshlets stands in for the group's position
        let mut group_positions = vec![[0.0; 3]; group_count];
        for (&group, &meshlet_id) in group_per_meshlet.iter().zip(simplification_queue) {
            group_positions[group as usize] =
                meshlet_cull_data[meshlet_id as usize].aabb.center().into();
        }
        let group_order = spatial_sort_remap_decoder(&group_positions);
        for group in &mut group_per_meshlet {
            *group = group_order[*group as usize];
        }
    }

    let mut groups = vec![TempMeshletGroup::default(); group_count];
    for (i, meshlet_group) in group_per_meshlet.into_iter().enumerate() {
        let group = &mut groups[meshlet_group as usize];
        let meshlet_id = simplification_queue[i];

        group.meshlets.push(meshlet_id);
        let data = &meshlet_cull_data[meshlet_id as usize];
        group.aabb = group.aabb.merge(&data.aabb);
        group.lod_bounds = merge_spheres(group.lod_bounds, data.lod_group_sphere);
    }
    groups
}

/// Marks every vertex that lies on an open edge of the mesh - an edge used by
/// exactly one triangle, in position-only terms - so a partition's seam with
/// its neighbours stays put through simplification.
fn mesh_border_locks(indices: &[u32], position_only_vertex_remap: &[u32]) -> Vec<bool> {
    let mut edge_uses: HashMap<(u32, u32), u32> = HashMap::default();
    for triangle in indices.chunks_exact(3) {
        let corners = [
            position_only_vertex_remap[triangle[0] as usize],
            position_only_vertex_remap[triangle[1] as usize],
            position_only_vertex_remap[triangle[2] as usize],
        ];
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let edge = (corners[a].min(corners[b]), corners[a].max(corners[b]));
            if edge.0 != edge.1 {
                *edge_uses.entry(edge).or_default() += 1;
            }
        }
    }
    let mut border_positions = vec![false; position_only_vertex_remap.len()];
    for (&(a, b), &uses) in &edge_uses {
        if uses == 1 {
            border_positions[a as usize] = true;
            border_positions[b as usize] = true;
        }
    }
    spread_over_positions(&border_positions, position_only_vertex_remap)
}

/// Per-vertex flags from per-position flags: a lock belongs to a position, so
/// every vertex sharing one (a UV or normal seam) carries it.
fn spread_over_positions(position_flags: &[bool], position_only_vertex_remap: &[u32]) -> Vec<bool> {
    position_only_vertex_remap
        .iter()
        .map(|position| position_flags[*position as usize])
        .collect()
}

fn lock_group_borders(
    vertex_locks: &mut [bool],
    border_locks: &[bool],
    groups: &[TempMeshletGroup],
    meshlets: &Meshlets,
    position_only_vertex_remap: &[u32],
) {
    let mut position_only_locks = vec![-1; position_only_vertex_remap.len()];

    // Iterate over position-only based vertices of all meshlets in all groups
    for (group_id, group) in groups.iter().enumerate() {
        for &meshlet_id in group.meshlets.iter() {
            for &vertex in meshlets.get(meshlet_id as usize).vertices {
                let vertex_id = position_only_vertex_remap[vertex as usize] as usize;

                // If the vertex is not yet claimed by any group, or was already claimed by this group
                if position_only_locks[vertex_id] == -1
                    || position_only_locks[vertex_id] == group_id as i32
                {
                    position_only_locks[vertex_id] = group_id as i32; // Then claim the vertex for this group
                } else {
                    position_only_locks[vertex_id] = -2; // Else vertex was already claimed by another group or was already locked, lock it
                }
            }
        }
    }

    // Lock vertices used by more than 1 group, and the mesh's own open borders
    // when the caller asked for them.
    for i in 0..vertex_locks.len() {
        let vertex_id = position_only_vertex_remap[i] as usize;
        vertex_locks[i] = border_locks[i] || position_only_locks[vertex_id] == -2;
    }
}

/// Simplifies a group to about half its triangles, returning the new indices and their error, or
/// `None` if the group is stuck.
fn simplify_meshlet_group(
    group: &TempMeshletGroup,
    meshlets: &Meshlets,
    vertices: &VertexDataAdapter<'_>,
    vertex_attributes: &[f32],
    vertex_locks: &[bool],
) -> Option<(Vec<u32>, f32)> {
    // Build a new index buffer into the mesh vertex data by combining all meshlet data in the group
    let group_indices = group
        .meshlets
        .iter()
        .flat_map(|&meshlet_id| {
            let meshlet = meshlets.get(meshlet_id as _);
            meshlet
                .triangles
                .iter()
                .map(|&meshlet_index| meshlet.vertices[meshlet_index as usize])
        })
        .collect::<Vec<_>>();
    let target_count = group_indices.len() / 6 * 3;

    let simplify = |options: SimplifyOptions| {
        let mut error = 0.0;
        let indices = simplify_with_attributes_and_locks(
            &group_indices,
            vertices,
            vertex_attributes,
            &ATTRIBUTE_WEIGHTS,
            vertices.vertex_stride,
            vertex_locks,
            target_count,
            f32::MAX,
            SimplifyOptions::Sparse
                | SimplifyOptions::ErrorAbsolute
                | SimplifyOptions::ErrorClamped
                | options,
            Some(&mut error),
        );
        (indices, error)
    };

    let (mut simplified_group_indices, mut error) = simplify(SimplifyOptions::empty());

    // UV and normal seams can stop the simplifier short of the target. Permissive simplification
    // collapses across them, weighing what that does to the attributes in its error. It is only a
    // fallback: the two sides of double-sided foliage meet at normal seams, and collapsing across
    // them removes leaves wholesale, thinning a hedge long before its error grows perceptible.
    if simplified_group_indices.len() > target_count {
        let permissive = simplify(SimplifyOptions::Permissive);
        if permissive.0.len() < simplified_group_indices.len() {
            (simplified_group_indices, error) = permissive;
        }
    }

    // Keep one triangle of a group that simplifies away entirely, so that every part of the mesh
    // still ends in a root
    if simplified_group_indices.is_empty() {
        simplified_group_indices.extend_from_slice(&group_indices[..3]);
    }

    // Check if we were able to simplify
    if simplified_group_indices.len() as f32 / group_indices.len() as f32
        > SIMPLIFICATION_FAILURE_PERCENTAGE
    {
        return None;
    }

    Some((simplified_group_indices, error))
}

fn vertex_position(vertices: &VertexDataAdapter<'_>, vertex: u32) -> Vec3 {
    *bytemuck::from_bytes::<Vec3>(
        &vertices.reader.get_ref()
            [vertices.position_offset + vertex as usize * vertices.vertex_stride..][..12],
    )
}

fn merge_meshlets(meshlets: &mut Meshlets, merge: Meshlets) {
    let vertex_offset = meshlets.vertices.len() as u32;
    let triangle_offset = meshlets.triangles.len() as u32;
    meshlets.vertices.extend_from_slice(&merge.vertices);
    meshlets.triangles.extend_from_slice(&merge.triangles);
    meshlets
        .meshlets
        .extend(merge.meshlets.into_iter().map(|mut meshlet| {
            meshlet.vertex_offset += vertex_offset;
            meshlet.triangle_offset += triangle_offset;
            meshlet
        }));
}

fn build_and_compress_per_meshlet_vertex_data(
    meshlet: &meshopt_Meshlet,
    meshlet_vertex_ids: &[u32],
    vertex_buffer: &[u8],
    vertex_stride: usize,
    vertex_positions: &mut BitVec<u32, Lsb0>,
    vertex_normals: &mut Vec<u32>,
    vertex_uvs: &mut Vec<u32>,
    meshlets: &mut Vec<Meshlet>,
    vertex_position_quantization_factor: u8,
) {
    let start_vertex_position_bit = vertex_positions.len() as u32;
    let start_vertex_attribute_id = vertex_normals.len() as u32;

    let mut min_quantized_position_channels = IVec3::MAX;
    let mut max_quantized_position_channels = IVec3::MIN;
    let mut min_vertex_uv = Vec2::splat(f32::INFINITY);
    let mut max_vertex_uv = Vec2::splat(f32::NEG_INFINITY);

    // Lossy vertex compression
    let mut quantized_positions = [IVec3::ZERO; MESHLET_MAX_VERTICES];
    let mut uncompressed_uvs = [Vec2::ZERO; MESHLET_MAX_VERTICES];
    for (i, vertex_id) in meshlet_vertex_ids.iter().enumerate() {
        // Load source vertex attributes
        let vertex_id_byte = *vertex_id as usize * vertex_stride;
        let vertex_data = &vertex_buffer[vertex_id_byte..(vertex_id_byte + vertex_stride)];
        let position = Vec3::from_slice(bytemuck::cast_slice(&vertex_data[0..12]));
        let normal = Vec3::from_slice(bytemuck::cast_slice(&vertex_data[12..24]));
        let uv = Vec2::from_slice(bytemuck::cast_slice(&vertex_data[24..32]));

        uncompressed_uvs[i] = uv;
        min_vertex_uv = min_vertex_uv.min(uv);
        max_vertex_uv = max_vertex_uv.max(uv);

        // Compress normal
        vertex_normals.push(pack2x16snorm(octahedral_encode(normal)));

        // Quantize position to a fixed-point IVec3
        let quantized_position =
            quantize_vertex_position(position, vertex_position_quantization_factor);
        quantized_positions[i] = quantized_position;

        // Compute per X/Y/Z-channel quantized position min/max for this meshlet
        min_quantized_position_channels = min_quantized_position_channels.min(quantized_position);
        max_quantized_position_channels = max_quantized_position_channels.max(quantized_position);
    }

    // Normalize UVs to each meshlet's local bounds before packing. This gives substantially more
    // precision than raw f16 UVs for tiled materials with coordinates far outside 0..1.
    let vertex_uv_extent = max_vertex_uv - min_vertex_uv;
    for uv in uncompressed_uvs.iter().take(meshlet_vertex_ids.len()) {
        let normalized_uv = Vec2::new(
            if vertex_uv_extent.x > 0.0 {
                (uv.x - min_vertex_uv.x) / vertex_uv_extent.x
            } else {
                0.0
            },
            if vertex_uv_extent.y > 0.0 {
                (uv.y - min_vertex_uv.y) / vertex_uv_extent.y
            } else {
                0.0
            },
        );
        vertex_uvs.push(pack2x16unorm(normalized_uv));
    }

    // Calculate bits needed to encode each quantized vertex position channel based on the range of each channel
    let range = max_quantized_position_channels - min_quantized_position_channels + 1;
    let bits_per_vertex_position_channel_x = log2(range.x as f32).ceil() as u8;
    let bits_per_vertex_position_channel_y = log2(range.y as f32).ceil() as u8;
    let bits_per_vertex_position_channel_z = log2(range.z as f32).ceil() as u8;

    // Lossless encoding of vertex positions in the minimum number of bits per channel
    for quantized_position in quantized_positions.iter().take(meshlet_vertex_ids.len()) {
        // Remap [range_min, range_max] IVec3 to [0, range_max - range_min] UVec3
        let position = (quantized_position - min_quantized_position_channels).as_uvec3();

        // Store as a packed bitstream
        vertex_positions.extend_from_bitslice(
            &position.x.view_bits::<Lsb0>()[..bits_per_vertex_position_channel_x as usize],
        );
        vertex_positions.extend_from_bitslice(
            &position.y.view_bits::<Lsb0>()[..bits_per_vertex_position_channel_y as usize],
        );
        vertex_positions.extend_from_bitslice(
            &position.z.view_bits::<Lsb0>()[..bits_per_vertex_position_channel_z as usize],
        );
    }

    meshlets.push(Meshlet {
        start_vertex_position_bit,
        start_vertex_attribute_id,
        start_index_id: meshlet.triangle_offset,
        vertex_count_minus_one: (meshlet.vertex_count - 1) as u8,
        triangle_count: meshlet.triangle_count as u8,
        padding: 0,
        bits_per_vertex_position_channel_x,
        bits_per_vertex_position_channel_y,
        bits_per_vertex_position_channel_z,
        vertex_position_quantization_factor,
        min_vertex_position_channel_x: min_quantized_position_channels.x as f32,
        min_vertex_position_channel_y: min_quantized_position_channels.y as f32,
        min_vertex_position_channel_z: min_quantized_position_channels.z as f32,
        min_vertex_uv,
        vertex_uv_extent,
    });
}

fn merge_spheres(a: BoundingSphere, b: BoundingSphere) -> BoundingSphere {
    let sr = a.radius().min(b.radius());
    let br = a.radius().max(b.radius());
    let len = a.center.distance(b.center);
    if len + sr <= br || sr == 0.0 || len == 0.0 {
        if a.radius() > b.radius() {
            a
        } else {
            b
        }
    } else {
        let radius = (sr + br + len) / 2.0;
        let center =
            (a.center + b.center + (a.radius() - b.radius()) * (a.center - b.center) / len) / 2.0;
        BoundingSphere::new(center, radius)
    }
}

#[derive(Copy, Clone)]
struct TempMeshletCullData {
    aabb: Aabb3d,
    lod_group_sphere: BoundingSphere,
    error: f32,
}

#[derive(Clone)]
struct TempMeshletGroup {
    aabb: Aabb3d,
    lod_bounds: BoundingSphere,
    parent_error: f32,
    meshlets: SmallVec<[u32; TARGET_MESHLETS_PER_GROUP]>,
}

impl Default for TempMeshletGroup {
    fn default() -> Self {
        Self {
            aabb: aabb_default(), // Default AABB to merge into
            lod_bounds: BoundingSphere::new(Vec3A::ZERO, 0.0),
            parent_error: f32::MAX,
            meshlets: SmallVec::new(),
        }
    }
}

// All the BVH build code was stolen from https://github.com/SparkyPotato/radiance/blob/4aa17a3a5be7a0466dc69713e249bbcee9f46057/crates/rad-renderer/src/assets/mesh/virtual_mesh.rs because it works and I'm lazy and don't want to reimplement it
struct TempBvhNode {
    group: u32,
    aabb: Aabb3d,
    children: SmallVec<[u32; 8]>,
}

#[derive(Default)]
struct BvhBuilder {
    nodes: Vec<TempBvhNode>,
    lods: Vec<Range<u32>>,
}

impl BvhBuilder {
    fn add_lod(&mut self, offset: u32, all_groups: &[TempMeshletGroup]) {
        let first = self.nodes.len() as u32;
        self.nodes.extend(
            all_groups
                .iter()
                .enumerate()
                .skip(offset as _)
                .map(|(i, group)| TempBvhNode {
                    group: i as u32,
                    aabb: group.aabb,
                    children: SmallVec::new(),
                }),
        );
        let end = self.nodes.len() as u32;
        if first != end {
            self.lods.push(first..end);
        }
    }

    fn surface_area(&self, nodes: &[u32]) -> f32 {
        nodes
            .iter()
            .map(|&x| self.nodes[x as usize].aabb)
            .reduce(|a, b| a.merge(&b))
            .expect("cannot find surface area of zero nodes")
            .visible_area()
    }

    fn sort_nodes_by_sah(&self, nodes: &mut [u32], splits: [usize; 8]) {
        // We use a BVH8, so just recursively binary split 3 times for near-optimal SAH
        for i in 0..3 {
            let parts = 1 << i; // 2^i
            let nodes_per_split = 8 >> i; // 8 / 2^i
            let half_count = nodes_per_split / 2;
            let mut offset = 0;
            for p in 0..parts {
                let first = p * nodes_per_split;
                let mut s0 = 0;
                let mut s1 = 0;
                for i in 0..half_count {
                    s0 += splits[first + i];
                    s1 += splits[first + half_count + i];
                }
                let c = s0 + s1;
                let nodes = &mut nodes[offset..(offset + c)];
                offset += c;

                let mut cost = f32::MAX;
                let mut axis = 0;
                let key = |x, ax| self.nodes[x as usize].aabb.center()[ax];
                for ax in 0..3 {
                    nodes.sort_unstable_by(|&x, &y| key(x, ax).partial_cmp(&key(y, ax)).unwrap());
                    let (left, right) = nodes.split_at(s0);
                    let c = self.surface_area(left) + self.surface_area(right);
                    if c < cost {
                        axis = ax;
                        cost = c;
                    }
                }
                if axis != 2 {
                    nodes.sort_unstable_by(|&x, &y| {
                        key(x, axis).partial_cmp(&key(y, axis)).unwrap()
                    });
                }
            }
        }
    }

    fn build_temp_inner(&mut self, nodes: &mut [u32], optimize: bool) -> u32 {
        let count = nodes.len();
        if count == 1 {
            nodes[0]
        } else if count <= 8 {
            let i = self.nodes.len();
            self.nodes.push(TempBvhNode {
                group: u32::MAX,
                aabb: aabb_default(),
                children: nodes.iter().copied().collect(),
            });
            i as _
        } else {
            // We need to split the nodes into 8 groups, with the smallest possible tree depth.
            // Additionally, no child should be more than one level deeper than the others.
            // At `l` levels, we can fit upto 8^l nodes.
            // The `max_child_size` is the largest power of 8 <= `count` (any larger and we'd have
            // unfilled nodes).
            // The `min_child_size` is thus 1 level (8 times) smaller.
            // After distributing `min_child_size` to all children, we have distributed
            // `min_child_size * 8` nodes (== `max_child_size`).
            // The remaining nodes are then distributed left to right.
            let max_child_size = 1 << ((count.ilog2() / 3) * 3);
            let min_child_size = max_child_size >> 3;
            let max_extra_per_node = max_child_size - min_child_size;
            let mut extra = count - max_child_size; // 8 * min_child_size
            let splits = core::array::from_fn(|_| {
                let size = extra.min(max_extra_per_node);
                extra -= size;
                min_child_size + size
            });

            if optimize {
                self.sort_nodes_by_sah(nodes, splits);
            }

            let mut offset = 0;
            let children = splits
                .into_iter()
                .map(|size| {
                    let i = self.build_temp_inner(&mut nodes[offset..(offset + size)], optimize);
                    offset += size;
                    i
                })
                .collect();

            let i = self.nodes.len();
            self.nodes.push(TempBvhNode {
                group: u32::MAX,
                aabb: aabb_default(),
                children,
            });
            i as _
        }
    }

    fn build_temp(&mut self) -> u32 {
        let mut lods = Vec::with_capacity(self.lods.len());
        for lod in core::mem::take(&mut self.lods) {
            let mut lod: Vec<_> = lod.collect();
            let root = self.build_temp_inner(&mut lod, true);
            let node = &self.nodes[root as usize];
            if node.group != u32::MAX || node.children.len() == 8 {
                lods.push(root);
            } else {
                lods.extend(node.children.iter().copied());
            }
        }
        self.build_temp_inner(&mut lods, false)
    }

    fn build_inner(
        &self,
        groups: &[TempMeshletGroup],
        out: &mut Vec<BvhNode>,
        max_depth: &mut u32,
        node: u32,
        depth: u32,
    ) -> u32 {
        *max_depth = depth.max(*max_depth);
        let node = &self.nodes[node as usize];
        let onode = out.len();
        out.push(BvhNode::default());

        for (i, &child_id) in node.children.iter().enumerate() {
            let child = &self.nodes[child_id as usize];
            if child.group != u32::MAX {
                let group = &groups[child.group as usize];
                let out = &mut out[onode];
                out.aabbs[i] = aabb_to_meshlet(group.aabb, group.parent_error, group.meshlets[0]);
                out.lod_bounds[i] = sphere_to_meshlet(group.lod_bounds);
                out.child_counts[i] = group.meshlets[1] as _;
            } else {
                let child_id = self.build_inner(groups, out, max_depth, child_id, depth + 1);
                let child = &out[child_id as usize];
                let mut aabb = aabb_default();
                let mut parent_error = 0.0f32;
                let mut lod_bounds = BoundingSphere::new(Vec3A::ZERO, 0.0);
                for i in 0..8 {
                    if child.child_counts[i] == 0 {
                        break;
                    }

                    aabb = aabb.merge(&Aabb3d::new(
                        child.aabbs[i].center,
                        child.aabbs[i].half_extent,
                    ));
                    lod_bounds = merge_spheres(
                        lod_bounds,
                        BoundingSphere::new(child.lod_bounds[i].center, child.lod_bounds[i].radius),
                    );
                    parent_error = parent_error.max(child.aabbs[i].error);
                }

                let out = &mut out[onode];
                out.aabbs[i] = aabb_to_meshlet(aabb, parent_error, child_id);
                out.lod_bounds[i] = sphere_to_meshlet(lod_bounds);
                out.child_counts[i] = u8::MAX;
            }
        }

        onode as _
    }

    fn build(
        self,
        meshlets: &mut Meshlets,
        mut groups: Vec<TempMeshletGroup>,
        cull_data: &mut Vec<TempMeshletCullData>,
    ) -> (Vec<BvhNode>, MeshletAabb, u32) {
        // The BVH requires group meshlets to be contiguous, so remap them first.
        let mut remap = Vec::with_capacity(meshlets.meshlets.len());
        let mut remapped_cull_data = Vec::with_capacity(cull_data.len());
        for group in groups.iter_mut() {
            let first = remap.len() as u32;
            let count = group.meshlets.len() as u32;
            remap.extend(
                group
                    .meshlets
                    .iter()
                    .map(|&m| meshlets.meshlets[m as usize]),
            );
            remapped_cull_data.extend(group.meshlets.iter().map(|&m| cull_data[m as usize]));
            group.meshlets.resize(2, 0);
            group.meshlets[0] = first;
            group.meshlets[1] = count;
        }
        meshlets.meshlets = remap;
        *cull_data = remapped_cull_data;

        let (out, aabb, max_depth) = self.build_nodes(&groups);
        verify_bvh_complete(&out, cull_data);
        (out, aabb, max_depth)
    }

    /// Emits the node array over groups whose `meshlets` already hold the `[first, count]` of a
    /// contiguous run in the meshlet array, so a caller that never had loose meshlets to remap
    /// (pruning a built mesh) shares the layout `from_mesh` produces.
    fn build_nodes(mut self, groups: &[TempMeshletGroup]) -> (Vec<BvhNode>, MeshletAabb, u32) {
        let mut out = vec![];
        let mut aabb = aabb_default();
        let mut max_depth = 0;

        if self.nodes.len() == 1 {
            let mut o = BvhNode::default();
            let group = &groups[0];
            o.aabbs[0] = aabb_to_meshlet(group.aabb, group.parent_error, group.meshlets[0]);
            o.lod_bounds[0] = sphere_to_meshlet(group.lod_bounds);
            o.child_counts[0] = group.meshlets[1] as _;
            out.push(o);
            aabb = group.aabb;
            max_depth = 1;
        } else {
            let root = self.build_temp();
            let root = self.build_inner(groups, &mut out, &mut max_depth, root, 1);
            assert_eq!(root, 0, "root must be 0");

            let root = &out[0];
            for i in 0..8 {
                if root.child_counts[i] == 0 {
                    break;
                }

                aabb = aabb.merge(&Aabb3d::new(
                    root.aabbs[i].center,
                    root.aabbs[i].half_extent,
                ));
            }
        }

        (
            out,
            MeshletAabb {
                center: aabb.center().into(),
                half_extent: aabb.half_size().into(),
            },
            max_depth,
        )
    }
}

/// Checks the LOD invariants the cull shaders rely on and that the tree reaches every meshlet.
fn verify_bvh_complete(out: &[BvhNode], cull_data: &[TempMeshletCullData]) {
    let mut reachable = vec![false; cull_data.len()];
    verify_bvh(out, cull_data, &mut reachable, 0);
    assert!(
        reachable.iter().all(|&x| x),
        "all meshlets must be reachable"
    );
}

fn verify_bvh(
    out: &[BvhNode],
    cull_data: &[TempMeshletCullData],
    reachable: &mut [bool],
    node: u32,
) {
    let node = &out[node as usize];
    for i in 0..8 {
        let sphere = node.lod_bounds[i];
        let error = node.aabbs[i].error;
        if node.child_counts[i] == u8::MAX {
            let child = &out[node.aabbs[i].child_offset as usize];
            for i in 0..8 {
                if child.child_counts[i] == 0 {
                    break;
                }
                assert!(
                    child.aabbs[i].error <= error,
                    "BVH errors are not monotonic"
                );
                let sphere_error = (sphere.center - child.lod_bounds[i].center).length()
                    - (sphere.radius - child.lod_bounds[i].radius);
                assert!(
                    sphere_error <= 0.0001,
                    "BVH lod spheres are not monotonic ({sphere_error})"
                );
            }
            verify_bvh(out, cull_data, reachable, node.aabbs[i].child_offset);
        } else {
            for m in 0..node.child_counts[i] as u32 {
                let mid = (m + node.aabbs[i].child_offset) as usize;
                let meshlet = &cull_data[mid];
                assert!(meshlet.error <= error, "meshlet errors are not monotonic");
                let sphere_error = (Vec3A::from(sphere.center) - meshlet.lod_group_sphere.center)
                    .length()
                    - (sphere.radius - meshlet.lod_group_sphere.radius());
                assert!(
                    sphere_error <= 0.0001,
                    "meshlet lod spheres are not monotonic: ({sphere_error})"
                );
                reachable[mid] = true;
            }
        }
    }
}

fn aabb_default() -> Aabb3d {
    Aabb3d {
        min: Vec3A::INFINITY,
        max: Vec3A::NEG_INFINITY,
    }
}

fn aabb_to_meshlet(aabb: Aabb3d, error: f32, child_offset: u32) -> MeshletAabbErrorOffset {
    MeshletAabbErrorOffset {
        center: aabb.center().into(),
        error,
        half_extent: aabb.half_size().into(),
        child_offset,
    }
}

fn sphere_to_meshlet(sphere: BoundingSphere) -> MeshletBoundingSphere {
    MeshletBoundingSphere {
        center: sphere.center.into(),
        radius: sphere.radius(),
    }
}

// TODO: Precise encode variant
fn octahedral_encode(v: Vec3) -> Vec2 {
    let n = v / (v.x.abs() + v.y.abs() + v.z.abs());
    let octahedral_wrap = (1.0 - n.yx().abs())
        * Vec2::new(
            if n.x >= 0.0 { 1.0 } else { -1.0 },
            if n.y >= 0.0 { 1.0 } else { -1.0 },
        );
    if n.z >= 0.0 {
        n.xy()
    } else {
        octahedral_wrap
    }
}

// https://www.w3.org/TR/WGSL/#pack2x16snorm-builtin
fn pack2x16snorm(v: Vec2) -> u32 {
    let v = v.clamp(Vec2::NEG_ONE, Vec2::ONE);
    let v = (v * 32767.0 + 0.5).floor().as_i16vec2();
    bytemuck::cast(v)
}

// https://www.w3.org/TR/WGSL/#pack2x16unorm-builtin
fn pack2x16unorm(v: Vec2) -> u32 {
    let v = (v.clamp(Vec2::ZERO, Vec2::ONE) * 65535.0 + 0.5)
        .floor()
        .as_u16vec2();
    bytemuck::cast(v)
}

/// An error produced by [`MeshletMesh::from_mesh`].
#[derive(Error, Debug)]
pub enum MeshToMeshletMeshConversionError {
    #[error("Mesh primitive topology is not TriangleList")]
    WrongMeshPrimitiveTopology,
    #[error("Mesh vertex attributes must be {required:?}, but got {provided:?}")]
    WrongMeshVertexAttributes {
        required: [MeshVertexAttribute; 3],
        provided: Vec<MeshVertexAttribute>,
    },
    #[error("Mesh has no indices")]
    MeshMissingIndices,
    #[error("Mesh has {0} indices, which is not a whole number of triangles")]
    WrongIndexCount(usize),
    #[error("Mesh index {index} is out of range for its {vertex_count} vertices")]
    IndexOutOfRange { index: u32, vertex_count: usize },
    #[error("Mesh has {required} vertices, but {provided} lock flags were given")]
    WrongLockCount { required: usize, provided: usize },
}

/// Which input vertices the simplifier must keep in place at every LOD, on
/// top of the group borders it always locks.
#[derive(Clone, Copy, Debug)]
enum InputLocks<'a> {
    None,
    /// Every vertex on an edge only one triangle uses.
    OpenBorders,
    /// The caller's choice, one flag per vertex.
    Explicit(&'a [bool]),
}

/// Per-vertex lock flags widened to every vertex sharing a locked position.
fn position_locks(locked: &[bool], position_only_vertex_remap: &[u32]) -> Vec<bool> {
    let mut locked_positions = vec![false; position_only_vertex_remap.len()];
    for (vertex_id, flag) in locked.iter().enumerate() {
        if *flag {
            locked_positions[position_only_vertex_remap[vertex_id] as usize] = true;
        }
    }
    spread_over_positions(&locked_positions, position_only_vertex_remap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meshlet::MeshletRaytracingGeometry;
    use bevy_asset::RenderAssetUsages;

    #[test]
    fn raytracing_geometry_selects_a_complete_meshlet_lod() {
        AsyncComputeTaskPool::get_or_init(TaskPool::default);
        const SIDE: usize = 24;
        let mut positions = Vec::with_capacity((SIDE + 1) * (SIDE + 1));
        let mut normals = Vec::with_capacity(positions.capacity());
        let mut uvs = Vec::with_capacity(positions.capacity());
        for y in 0..=SIDE {
            for x in 0..=SIDE {
                let fx = x as f32 / SIDE as f32;
                let fy = y as f32 / SIDE as f32;
                positions.push([fx, (fx * 9.0).sin() * (fy * 7.0).sin() * 0.02, fy]);
                normals.push([0.0, 1.0, 0.0]);
                uvs.push([fx, fy]);
            }
        }
        let mut indices = Vec::with_capacity(SIDE * SIDE * 6);
        for y in 0..SIDE {
            for x in 0..SIDE {
                let a = (y * (SIDE + 1) + x) as u32;
                let b = a + 1;
                let c = a + (SIDE + 1) as u32;
                let d = c + 1;
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        let source_triangles = indices.len() / 3;
        let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
            .with_inserted_indices(Indices::U32(indices));
        let meshlet = MeshletMesh::from_mesh(&mesh, 4).unwrap();

        let exact = meshlet.raytracing_geometry(0.0);
        assert_eq!(exact.indices.len() / 3, source_triangles);
        assert_eq!(exact.positions.len(), exact.normals.len());
        assert_eq!(exact.positions.len(), exact.uvs.len());
        assert!(exact
            .indices
            .iter()
            .all(|index| (*index as usize) < exact.positions.len()));

        let simplified = meshlet.raytracing_geometry(0.02);
        assert!(!simplified.indices.is_empty());
        assert!(simplified.indices.len() <= exact.indices.len());
    }

    #[test]
    fn raytracing_geometry_reports_the_error_it_achieved_not_the_one_requested() {
        AsyncComputeTaskPool::get_or_init(TaskPool::default);
        // Two triangles cannot be simplified, so this deviates from the reference by nothing no
        // matter how loose a bound it is asked for. Reporting the request instead lets one such
        // mesh set a scene-wide ray bias it has no error to justify.
        let quad = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_POSITION,
                vec![
                    [-0.5, 0.0, -0.5],
                    [0.5, 0.0, -0.5],
                    [-0.5, 0.0, 0.5],
                    [0.5, 0.0, 0.5],
                ],
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; 4])
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_UV_0,
                vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            )
            .with_inserted_indices(Indices::U32(vec![0, 2, 1, 1, 2, 3]));
        let meshlet = MeshletMesh::from_mesh(&quad, 4).unwrap();

        let geometry = meshlet.raytracing_geometry(0.02);
        assert_eq!(geometry.indices.len() / 3, 2);
        assert_eq!(geometry.achieved_error, 0.0);

        // And a cut that does simplify never overstates the bound it was given.
        assert!(meshlet.raytracing_geometry(0.0).achieved_error <= 0.0);
    }

    fn torus_meshlet_mesh() -> MeshletMesh {
        AsyncComputeTaskPool::get_or_init(TaskPool::default);
        MeshletMesh::from_mesh(&torus_mesh(), 4).unwrap()
    }

    /// A closed torus dense enough to simplify through several LODs.
    fn torus_mesh() -> Mesh {
        const MAJOR: u32 = 128;
        const MINOR: u32 = 64;
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        for i in 0..MAJOR {
            let u = i as f32 / MAJOR as f32;
            let (su, cu) = bevy_math::ops::sin_cos(u * core::f32::consts::TAU);
            for j in 0..MINOR {
                let v = j as f32 / MINOR as f32;
                let (sv, cv) = bevy_math::ops::sin_cos(v * core::f32::consts::TAU);
                let ring = 1.0 + 0.4 * cv;
                positions.push([ring * cu, 0.4 * sv, ring * su]);
                normals.push([cv * cu, sv, cv * su]);
                uvs.push([u, v]);
            }
        }
        let mut indices = Vec::new();
        for i in 0..MAJOR {
            for j in 0..MINOR {
                let a = i * MINOR + j;
                let b = i * MINOR + (j + 1) % MINOR;
                let c = ((i + 1) % MAJOR) * MINOR + j;
                let d = ((i + 1) % MAJOR) * MINOR + (j + 1) % MINOR;
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
            .with_inserted_indices(Indices::U32(indices))
    }

    /// Chunks of several groups (one thread), single groups (eight) and the serial build agree
    /// byte for byte.
    #[test]
    fn the_task_pool_does_not_change_the_build() {
        let (_, mesh) = open_grid(160);
        let encoded = |builder: MeshletMeshBuilder| {
            let mut file = Vec::new();
            builder.build(&mesh).unwrap().write(&mut file).unwrap();
            file
        };
        let serial = encoded(MeshletMeshBuilder::new(4).with_locked_borders());
        for threads in [1, 3, 8] {
            let task_pool = bevy_tasks::TaskPoolBuilder::new()
                .num_threads(threads)
                .build();
            let builder = MeshletMeshBuilder::new(4)
                .with_locked_borders()
                .with_task_pool(&task_pool);
            assert!(
                encoded(builder) == serial,
                "{threads} threads built a different mesh"
            );
        }
    }

    fn total_triangles(mesh: &MeshletMesh) -> usize {
        mesh.meshlets
            .iter()
            .map(|meshlet| meshlet.triangle_count as usize)
            .sum()
    }

    /// The bounded errors of every BVH leaf, ascending and deduplicated: the thresholds at which
    /// pruning changes what it keeps.
    fn leaf_errors(mesh: &MeshletMesh) -> Vec<f32> {
        let mut errors: Vec<f32> = mesh
            .bvh
            .iter()
            .flat_map(|node| (0..8).map(move |i| (node.child_counts[i], node.aabbs[i].error)))
            .filter(|&(count, error)| count != 0 && count != u8::MAX && error < f32::MAX)
            .map(|(_, error)| error)
            .collect();
        errors.sort_by(f32::total_cmp);
        errors.dedup();
        errors
    }

    /// A cut as a sorted bag of triangles, so cuts from differently ordered meshes compare.
    fn triangle_bag(geometry: &MeshletRaytracingGeometry) -> Vec<[[u32; 3]; 3]> {
        let mut triangles: Vec<_> = geometry
            .indices
            .chunks_exact(3)
            .map(|triangle| {
                let mut corners = <[u32; 3]>::try_from(triangle)
                    .unwrap()
                    .map(|index| geometry.positions[index as usize].map(f32::to_bits));
                corners.sort_unstable();
                corners
            })
            .collect();
        triangles.sort_unstable();
        triangles
    }

    /// The packed bytes a meshlet owns, independent of where they sit in the streams.
    fn meshlet_payload(
        mesh: &MeshletMesh,
        meshlet: &Meshlet,
    ) -> (BitVec<u32, Lsb0>, Vec<u32>, Vec<u32>, Vec<u8>) {
        let vertex_count = meshlet.vertex_count_minus_one as usize + 1;
        let bits_per_vertex = (meshlet.bits_per_vertex_position_channel_x
            + meshlet.bits_per_vertex_position_channel_y
            + meshlet.bits_per_vertex_position_channel_z) as usize;
        let position_start = meshlet.start_vertex_position_bit as usize;
        let attribute_start = meshlet.start_vertex_attribute_id as usize;
        let index_start = meshlet.start_index_id as usize;
        (
            mesh.vertex_positions.view_bits::<Lsb0>()
                [position_start..position_start + vertex_count * bits_per_vertex]
                .to_bitvec(),
            mesh.vertex_normals[attribute_start..attribute_start + vertex_count].to_vec(),
            mesh.vertex_uvs[attribute_start..attribute_start + vertex_count].to_vec(),
            mesh.indices[index_start..index_start + meshlet.triangle_count as usize * 3].to_vec(),
        )
    }

    /// Meshlets keyed by their tight AABB, which pruning copies verbatim and no two meshlets of
    /// one mesh share.
    fn meshlets_by_aabb(mesh: &MeshletMesh) -> HashMap<[u32; 6], usize> {
        let by_aabb: HashMap<_, _> = mesh
            .meshlet_cull_data
            .iter()
            .enumerate()
            .map(|(id, data)| {
                let mut key = [0; 6];
                key[..3].copy_from_slice(&data.aabb.center.to_array().map(f32::to_bits));
                key[3..].copy_from_slice(&data.aabb.half_extent.to_array().map(f32::to_bits));
                (key, id)
            })
            .collect();
        assert_eq!(by_aabb.len(), mesh.meshlets.len());
        by_aabb
    }

    #[test]
    fn read_decodes_what_write_encoded() {
        use crate::meshlet::asset::{MeshletMeshSaveOrLoadError, MESHLET_MESH_ASSET_VERSION};

        let mesh = torus_meshlet_mesh();
        let mut file = Vec::new();
        mesh.write(&mut file).unwrap();
        let read = MeshletMesh::read(&mut file.as_slice()).unwrap();
        assert_eq!(read.vertex_positions, mesh.vertex_positions);
        assert_eq!(read.vertex_normals, mesh.vertex_normals);
        assert_eq!(read.vertex_uvs, mesh.vertex_uvs);
        assert_eq!(read.indices, mesh.indices);
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&read.bvh),
            bytemuck::cast_slice::<_, u8>(&mesh.bvh)
        );
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&read.meshlets),
            bytemuck::cast_slice::<_, u8>(&mesh.meshlets)
        );
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&read.meshlet_cull_data),
            bytemuck::cast_slice::<_, u8>(&mesh.meshlet_cull_data)
        );
        assert_eq!(
            bytemuck::bytes_of(&read.aabb),
            bytemuck::bytes_of(&mesh.aabb)
        );
        assert_eq!(read.bvh_depth, mesh.bvh_depth);
        assert_eq!(read.triangle_count(), mesh.triangle_count());

        // The header is checked before anything is decompressed.
        assert!(matches!(
            MeshletMesh::read(&mut &b"not a meshlet mesh"[..]),
            Err(MeshletMeshSaveOrLoadError::WrongFileType)
        ));
        let mut stale = file.clone();
        stale[8..16].copy_from_slice(&(MESHLET_MESH_ASSET_VERSION + 1).to_le_bytes());
        assert!(matches!(
            MeshletMesh::read(&mut stale.as_slice()),
            Err(MeshletMeshSaveOrLoadError::WrongVersion { found }) if found == MESHLET_MESH_ASSET_VERSION + 1
        ));
        file.truncate(file.len() / 2);
        assert!(MeshletMesh::read(&mut file.as_slice()).is_err());
    }

    #[test]
    fn pruning_at_zero_keeps_every_meshlet() {
        let mesh = torus_meshlet_mesh();
        let pruned = mesh.pruned(0.0);
        assert_eq!(pruned.meshlets.len(), mesh.meshlets.len());
        assert_eq!(total_triangles(&pruned), total_triangles(&mesh));
        assert_eq!(pruned.packed_byte_len(), mesh.packed_byte_len());
        assert_eq!(pruned.vertex_positions.len(), mesh.vertex_positions.len());
        assert_eq!(pruned.vertex_normals.len(), mesh.vertex_normals.len());
        assert_eq!(pruned.bvh.len(), mesh.bvh.len());
        assert_eq!(pruned.meshlet_count(), mesh.meshlet_count());
        verify_bvh_complete(&pruned.bvh, &temp_cull_data(&pruned.meshlet_cull_data));
        assert_eq!(
            triangle_bag(&pruned.raytracing_geometry(0.0)),
            triangle_bag(&mesh.raytracing_geometry(0.0))
        );
        // Nothing lost its children, so nothing had its error rewritten.
        let mut original: Vec<u32> = mesh
            .meshlet_cull_data
            .iter()
            .map(|d| d.aabb.error.to_bits())
            .collect();
        let mut kept: Vec<u32> = pruned
            .meshlet_cull_data
            .iter()
            .map(|d| d.aabb.error.to_bits())
            .collect();
        original.sort_unstable();
        kept.sort_unstable();
        assert_eq!(original, kept);
    }

    #[test]
    fn pruning_trades_meshlets_for_an_error_bound() {
        let mesh = torus_meshlet_mesh();
        let errors = leaf_errors(&mesh);
        assert!(
            errors.len() >= 4,
            "torus must simplify through several LODs"
        );
        // Just past a leaf error, so each threshold drops at least one more group than the last;
        // the final one leaves only the unsimplifiable coarsest LOD.
        let thresholds = [
            errors[errors.len() / 4].next_up(),
            errors[errors.len() / 2].next_up(),
            errors[errors.len() * 3 / 4].next_up(),
            errors[errors.len() - 1].next_up(),
        ];
        let original_by_aabb = meshlets_by_aabb(&mesh);

        let mut previous_meshlets = mesh.meshlets.len();
        let mut previous_bytes = mesh.packed_byte_len();
        for min_error in thresholds {
            let pruned = mesh.pruned(min_error);
            assert!(pruned.meshlets.len() < previous_meshlets, "{min_error}");
            assert!(pruned.packed_byte_len() < previous_bytes, "{min_error}");
            assert_eq!(pruned.meshlets.len(), pruned.meshlet_cull_data.len());
            previous_meshlets = pruned.meshlets.len();
            previous_bytes = pruned.packed_byte_len();

            verify_bvh_complete(&pruned.bvh, &temp_cull_data(&pruned.meshlet_cull_data));
            assert!(pruned.bvh_depth >= 1);

            // The cut at the bound, and every looser one, is unchanged.
            for error in [min_error, min_error * 4.0, 1.0e6] {
                assert_eq!(
                    triangle_bag(&pruned.raytracing_geometry(error)),
                    triangle_bag(&mesh.raytracing_geometry(error)),
                    "cut at {error} after pruning at {min_error}"
                );
            }
            // A tighter request cannot reach detail that is gone.
            assert_eq!(
                triangle_bag(&pruned.raytracing_geometry(0.0)),
                triangle_bag(&mesh.raytracing_geometry(min_error))
            );

            // Survivors that lost their children read as full detail; every other error, sphere,
            // and byte of packed data is the original's.
            let mut finest_rewritten = 0;
            for (id, meshlet) in pruned.meshlets.iter().enumerate() {
                let data = &pruned.meshlet_cull_data[id];
                assert!(data.aabb.error == 0.0 || data.aabb.error >= min_error);

                let mut key = [0; 6];
                key[..3].copy_from_slice(&data.aabb.center.to_array().map(f32::to_bits));
                key[3..].copy_from_slice(&data.aabb.half_extent.to_array().map(f32::to_bits));
                let original_id = original_by_aabb[&key];
                let original = &mesh.meshlet_cull_data[original_id];
                assert_eq!(
                    data.lod_group_sphere.center.to_array().map(f32::to_bits),
                    original
                        .lod_group_sphere
                        .center
                        .to_array()
                        .map(f32::to_bits)
                );
                assert_eq!(
                    data.lod_group_sphere.radius,
                    original.lod_group_sphere.radius
                );
                if original.aabb.error < min_error {
                    assert_eq!(data.aabb.error, 0.0);
                    finest_rewritten += usize::from(original.aabb.error > 0.0);
                } else {
                    assert_eq!(data.aabb.error, original.aabb.error);
                }

                let original_meshlet = &mesh.meshlets[original_id];
                assert_eq!(
                    bytemuck::bytes_of(&Meshlet {
                        start_vertex_position_bit: 0,
                        start_vertex_attribute_id: 0,
                        start_index_id: 0,
                        ..*meshlet
                    }),
                    bytemuck::bytes_of(&Meshlet {
                        start_vertex_position_bit: 0,
                        start_vertex_attribute_id: 0,
                        start_index_id: 0,
                        ..*original_meshlet
                    })
                );
                assert_eq!(
                    meshlet_payload(&pruned, meshlet),
                    meshlet_payload(&mesh, original_meshlet)
                );
            }
            assert!(
                finest_rewritten > 0,
                "{min_error} left no simplified meshlet as the finest"
            );
        }
    }

    /// An `n` by `n` grid of quads on the XZ plane with open borders all
    /// round, as a partition of a larger surface has; returns its positions
    /// and the mesh.
    fn open_grid(n: u32) -> (Vec<[f32; 3]>, Mesh) {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                positions.push([x as f32, 0.0, y as f32]);
                normals.push([0.0, 1.0, 0.0]);
                uvs.push([x as f32 / n as f32, y as f32 / n as f32]);
            }
        }
        let mut indices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let i = y * (n + 1) + x;
                indices.extend_from_slice(&[i, i + n + 1, i + 1, i + 1, i + n + 1, i + n + 2]);
            }
        }
        let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions.clone())
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
            .with_inserted_indices(Indices::U32(indices));
        (positions, mesh)
    }

    fn contains_position(geometry: &MeshletRaytracingGeometry, p: &[f32; 3]) -> bool {
        geometry.positions.iter().any(|q| {
            (q[0] - p[0]).abs() < 1e-3 && (q[1] - p[1]).abs() < 1e-3 && (q[2] - p[2]).abs() < 1e-3
        })
    }

    /// The coarsest cut: the root of the LOD hierarchy carries an unbounded
    /// error, so any finite budget below it selects it.
    fn coarsest(mesh: &MeshletMesh) -> MeshletRaytracingGeometry {
        let cut = mesh.raytracing_geometry(1.0e6);
        assert!(!cut.indices.is_empty());
        cut
    }

    #[test]
    fn locked_borders_pin_a_partition_seam_through_every_lod() {
        AsyncComputeTaskPool::get_or_init(TaskPool::default);
        // Every interior vertex is free to move; the border must not.
        let n = 24u32;
        let (positions, mesh) = open_grid(n);
        let indices = match mesh.indices() {
            Some(Indices::U32(indices)) => indices.clone(),
            _ => unreachable!(),
        };
        let remap: Vec<u32> = (0..positions.len() as u32).collect();
        let locks = mesh_border_locks(&indices, &remap);
        let border = |i: usize| {
            let (x, y) = (i as u32 % (n + 1), i as u32 / (n + 1));
            x == 0 || y == 0 || x == n || y == n
        };
        assert!((0..positions.len()).all(|i| locks[i] == border(i)));

        // The coarsest cut of the locked build still contains every border
        // position of the source, where the free build is allowed to lose them.
        let locked = MeshletMesh::from_mesh_with_locked_borders(&mesh, 4).unwrap();
        let cut = coarsest(&locked);
        for (i, p) in positions.iter().enumerate() {
            if border(i) {
                assert!(
                    contains_position(&cut, p),
                    "border vertex {p:?} moved or vanished"
                );
            }
        }
    }

    /// An open grid with only one of its four borders locked: that edge must
    /// survive to the coarsest LOD, the opposite edge must lose vertices, and
    /// the whole must simplify far below what locking every open edge allows.
    #[test]
    fn explicit_locks_pin_only_the_named_seam() {
        AsyncComputeTaskPool::get_or_init(TaskPool::default);
        // Large enough that every locked border needs several meshlets at the coarsest LOD
        let n = 64u32;
        let (positions, mesh) = open_grid(n);
        let seam = |i: usize| (i as u32).is_multiple_of(n + 1);
        let locked: Vec<bool> = (0..positions.len()).map(seam).collect();

        assert!(matches!(
            MeshletMesh::from_mesh_with_locks(&mesh, 4, &locked[1..]),
            Err(MeshToMeshletMeshConversionError::WrongLockCount {
                required,
                provided
            }) if required == positions.len() && provided == positions.len() - 1
        ));

        let free = MeshletMesh::from_mesh(&mesh, 4).unwrap();
        let unlocked =
            MeshletMesh::from_mesh_with_locks(&mesh, 4, &vec![false; positions.len()]).unwrap();
        assert_eq!(unlocked.meshlets.len(), free.meshlets.len());
        assert_eq!(
            coarsest(&unlocked).indices.len(),
            coarsest(&free).indices.len()
        );

        let pinned = MeshletMesh::from_mesh_with_locks(&mesh, 4, &locked).unwrap();
        let bordered = MeshletMesh::from_mesh_with_locked_borders(&mesh, 4).unwrap();
        let cut = coarsest(&pinned);
        for (i, p) in positions.iter().enumerate() {
            if seam(i) {
                assert!(
                    contains_position(&cut, p),
                    "seam vertex {p:?} moved or vanished"
                );
            }
        }
        let far_edge_kept = positions
            .iter()
            .enumerate()
            .filter(|(i, _)| *i as u32 % (n + 1) == n)
            .filter(|(_, p)| contains_position(&cut, p))
            .count();
        assert!(
            far_edge_kept < n as usize + 1,
            "the unlocked border kept all {far_edge_kept} of its {} vertices",
            n + 1
        );
        let pinned_triangles = cut.indices.len() / 3;
        let bordered_triangles = coarsest(&bordered).indices.len() / 3;
        assert!(
            pinned_triangles * 2 < bordered_triangles,
            "one locked edge left {pinned_triangles} triangles, every open edge {bordered_triangles}"
        );
    }

    fn triangle_mesh(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> Mesh {
        let count = positions.len();
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; count])
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; count])
            .with_inserted_indices(Indices::U32(indices))
    }

    #[test]
    fn rejects_indices_that_are_not_whole_triangles_of_its_vertices() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(matches!(
            MeshletMesh::from_mesh(&triangle_mesh(positions.clone(), vec![]), 4),
            Err(MeshToMeshletMeshConversionError::WrongIndexCount(0))
        ));
        assert!(matches!(
            MeshletMesh::from_mesh(&triangle_mesh(positions.clone(), vec![0, 1, 2, 0]), 4),
            Err(MeshToMeshletMeshConversionError::WrongIndexCount(4))
        ));
        assert!(matches!(
            MeshletMesh::from_mesh(&triangle_mesh(positions, vec![0, 1, 3]), 4),
            Err(MeshToMeshletMeshConversionError::IndexOutOfRange {
                index: 3,
                vertex_count: 3
            })
        ));
    }

    #[test]
    fn a_single_meshlet_is_its_own_root() {
        AsyncComputeTaskPool::get_or_init(TaskPool::default);
        let mesh = triangle_mesh(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            vec![0, 2, 1],
        );
        let meshlet = MeshletMesh::from_mesh(&mesh, 4).unwrap();
        assert_eq!(meshlet.meshlets.len(), 1);
        assert_eq!(coarsest(&meshlet).indices.len(), 3);
        assert_eq!(meshlet.raytracing_geometry(0.0).achieved_error, 0.0);
    }

    #[test]
    fn meshlet_triangles_pack_without_slack() {
        let mesh = torus_meshlet_mesh();
        assert_eq!(mesh.indices.len(), total_triangles(&mesh) * 3);
    }

    /// Half of a grid is locked, so its groups get stuck at every level and their meshlets are
    /// carried into the next, alongside the free half's simplified meshlets.
    #[test]
    fn stuck_meshlets_carry_into_later_lods() {
        AsyncComputeTaskPool::get_or_init(TaskPool::default);
        let n = 64u32;
        let (positions, mesh) = open_grid(n);
        let locked_half = |i: usize| (i as u32 % (n + 1)) <= n / 2;
        let locked: Vec<bool> = (0..positions.len()).map(locked_half).collect();
        let meshlet = MeshletMesh::from_mesh_with_locks(&mesh, 4, &locked).unwrap();
        verify_bvh_complete(&meshlet.bvh, &temp_cull_data(&meshlet.meshlet_cull_data));

        let cut = coarsest(&meshlet);
        for (i, p) in positions.iter().enumerate() {
            if locked_half(i) {
                assert!(contains_position(&cut, p), "locked vertex {p:?} moved");
            }
        }
        let source_triangles = (n * n * 2) as usize;
        assert!(
            cut.indices.len() / 3 < source_triangles * 3 / 4,
            "the free half did not simplify"
        );

        for error in leaf_errors(&meshlet) {
            let pruned = meshlet.pruned(error);
            verify_bvh_complete(&pruned.bvh, &temp_cull_data(&pruned.meshlet_cull_data));
        }
    }

    /// Every quad has its own vertices with its own UVs, so every vertex sits on a UV seam and
    /// only permissive simplification can collapse across them.
    #[test]
    fn uv_seams_simplify_through_the_permissive_fallback() {
        AsyncComputeTaskPool::get_or_init(TaskPool::default);
        let n = 32u32;
        let mut positions = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let base = positions.len() as u32;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (px, py) = (x + dx, y + dy);
                    let height = bevy_math::ops::sin(px as f32 * 0.3)
                        * bevy_math::ops::sin(py as f32 * 0.2)
                        * 0.2;
                    positions.push([px as f32, height, py as f32]);
                    uvs.push([dx as f32, dy as f32]);
                }
                indices.extend_from_slice(&[
                    base,
                    base + 2,
                    base + 1,
                    base + 1,
                    base + 2,
                    base + 3,
                ]);
            }
        }
        let count = positions.len();
        let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; count])
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
            .with_inserted_indices(Indices::U32(indices));
        let meshlet = MeshletMesh::from_mesh(&mesh, 4).unwrap();
        let source_triangles = (n * n * 2) as usize;
        let coarsest_triangles = coarsest(&meshlet).indices.len() / 3;
        assert!(
            coarsest_triangles * 4 < source_triangles,
            "seams held the mesh at {coarsest_triangles} of {source_triangles} triangles"
        );
    }
}
