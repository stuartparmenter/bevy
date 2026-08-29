//! The load-time LOD budget: every cached part is pruned to `--raster-error`
//! and its BLAS cut is decoded from the same data as it loads.
//!
//! The cache holds every mesh at full detail, and full detail does not fit:
//! measured over 1,015 baked meshes, 676 M raster triangles took 16 GiB of
//! meshlet data (25.4 bytes per triangle), which projects to 78 GiB for the
//! 3.31 G-triangle scene against the meshlet manager's 8 GiB of pages, and
//! this fork streams neither meshlets nor BLASes. So the cut is made here, in
//! memory, as each part loads: `MeshletMesh::pruned` drops every LOD finer
//! than the bound and the runtime treats the survivors as full detail. Pruning
//! at load rather than in the bake keeps one cache serving every bound, and a
//! new bound costs a restart instead of a rebake.

use bevy::{
    asset::{io::Reader, Asset, AssetLoader, LoadContext, RenderAssetUsages},
    mesh::{Indices, Mesh},
    pbr::experimental::meshlet::{
        MeshletMesh, MeshletMeshSaveOrLoadError, MeshletRaytracingGeometry,
    },
    prelude::*,
    render::render_resource::PrimitiveTopology,
};
use thiserror::Error;

/// The meshlet manager's page geometry (`meshlet_mesh_manager.rs`:
/// `MESHLET_PAGE_SIZE` and `MESHLET_MAX_PAGES`), which it does not export. An
/// upload past the last free run fails with "the 128 meshlet data pages have
/// no free run of N bytes" and the part never renders.
pub const MESHLET_PAGE_SIZE: u64 = 64 * 1024 * 1024;
pub const MESHLET_MAX_PAGES: u64 = 128;
pub const MESHLET_PAGE_BUDGET: u64 = MESHLET_PAGE_SIZE * MESHLET_MAX_PAGES;

/// The error bounds a run loads its parts under.
#[derive(Clone, Copy, Debug)]
pub struct LodSettings {
    /// Metres of geometric error the finest resident meshlet LOD may carry.
    pub raster_error: f32,
    /// Metres of geometric error the BLAS cut may carry.
    pub raytracing_error: f32,
}

impl LodSettings {
    /// The bound the BLAS cut is selected at: never finer than the raster
    /// bound, since that detail is gone once the part is pruned.
    pub fn blas_error(&self) -> f32 {
        self.raytracing_error.max(self.raster_error)
    }
}

/// What one part came to once prepared.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PartStats {
    /// Triangles across every surviving LOD.
    pub raster_triangles: u64,
    pub meshlets: usize,
    /// Bytes the meshlet manager allocates for the pruned mesh.
    pub packed_bytes: u64,
    pub blas_triangles: u64,
    pub blas_vertices: u64,
    /// The largest error any BLAS meshlet carries.
    pub blas_achieved_error: f32,
}

/// One cache part as the runner consumes it: the pruned meshlet mesh and the
/// BLAS cut, both labeled sub-assets of the `.meshlet_mesh` file.
#[derive(Asset, TypePath)]
pub struct ZorahPart {
    #[dependency]
    pub meshlet: Handle<MeshletMesh>,
    #[dependency]
    pub blas: Handle<Mesh>,
    pub stats: PartStats,
}

#[derive(Debug, Error)]
pub enum PartLoadError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("meshlet mesh: {0}")]
    Meshlet(#[from] MeshletMeshSaveOrLoadError),
    #[error("the BLAS cut has no triangles")]
    EmptyBlas,
}

/// Loads a cache part as a [`ZorahPart`]. It shares the `.meshlet_mesh`
/// extension with bevy's own loader; the asset server picks this one by the
/// requested asset type.
#[derive(TypePath)]
pub struct ZorahPartLoader {
    pub settings: LodSettings,
}

impl AssetLoader for ZorahPartLoader {
    type Asset = ZorahPart;
    type Settings = ();
    type Error = PartLoadError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        load_context: &mut LoadContext<'_>,
    ) -> Result<ZorahPart, PartLoadError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let full = MeshletMesh::read(&mut bytes.as_slice())?;
        drop(bytes);
        let (meshlet, blas, stats) = prepare_part(&full, &self.settings)?;
        drop(full);
        Ok(ZorahPart {
            meshlet: load_context.add_labeled_asset("meshlet", meshlet),
            blas: load_context.add_labeled_asset("blas", blas),
            stats,
        })
    }

    fn extensions(&self) -> &[&str] {
        &["meshlet_mesh"]
    }
}

/// Prunes a full-detail part to the raster bound and decodes its BLAS cut.
///
/// The cut is selected on the full mesh rather than the pruned one: both pick
/// the same meshlets for any bound at or above the raster bound, but pruning
/// zeroes the error of the meshlets that become its finest level, and the
/// achieved error Solari biases its rays by has to be the real one.
pub fn prepare_part(
    full: &MeshletMesh,
    settings: &LodSettings,
) -> Result<(MeshletMesh, Mesh, PartStats), PartLoadError> {
    let geometry = full.raytracing_geometry(settings.blas_error());
    if geometry.indices.is_empty() {
        return Err(PartLoadError::EmptyBlas);
    }
    let meshlet = full.pruned(settings.raster_error);
    let stats = PartStats {
        raster_triangles: meshlet.triangle_count() as u64,
        meshlets: meshlet.meshlet_count(),
        packed_bytes: meshlet.packed_byte_len() as u64,
        blas_triangles: (geometry.indices.len() / 3) as u64,
        blas_vertices: geometry.positions.len() as u64,
        blas_achieved_error: geometry.achieved_error,
    };
    Ok((meshlet, blas_mesh(geometry), stats))
}

/// The mesh Solari builds a BLAS from.
///
/// Render-world only: Solari reads BLAS geometry from the render world's
/// `MeshAllocator` slices, and nothing in the main world inspects these
/// vertices, so a CPU copy of every part's cut would only cost memory. The
/// one thing lost is the automatic `Aabb` of a `Mesh3d` entity using it,
/// which leaves the few alpha-tested `Mesh3d` instances unculled.
pub fn blas_mesh(geometry: MeshletRaytracingGeometry) -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, geometry.positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, geometry.normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, geometry.uvs)
    .with_inserted_indices(Indices::U32(geometry.indices))
}

/// `bytes` as a `GiB` / `MiB` figure for logs.
pub fn human_bytes(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    let bytes = bytes as f64;
    if bytes >= MIB * 1024.0 {
        format!("{:.2} GiB", bytes / (MIB * 1024.0))
    } else {
        format!("{:.1} MiB", bytes / MIB)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use bevy::{
        math::ops,
        mesh::VertexAttributeValues,
        tasks::{AsyncComputeTaskPool, TaskPool},
    };

    /// A closed torus dense enough to simplify through several LODs.
    pub(crate) fn torus_meshlet_mesh() -> MeshletMesh {
        AsyncComputeTaskPool::get_or_init(TaskPool::default);
        const MAJOR: u32 = 128;
        const MINOR: u32 = 64;
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        for i in 0..MAJOR {
            let u = i as f32 / MAJOR as f32;
            let (su, cu) = ops::sin_cos(u * std::f32::consts::TAU);
            for j in 0..MINOR {
                let v = j as f32 / MINOR as f32;
                let (sv, cv) = ops::sin_cos(v * std::f32::consts::TAU);
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
        let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
            .with_inserted_indices(Indices::U32(indices));
        MeshletMesh::from_mesh(&mesh, 4).unwrap()
    }

    #[test]
    fn blas_error_never_undercuts_the_raster_bound() {
        let settings = LodSettings {
            raster_error: 0.05,
            raytracing_error: 0.02,
        };
        assert_eq!(settings.blas_error(), 0.05);
        let settings = LodSettings {
            raster_error: 0.004,
            raytracing_error: 0.05,
        };
        assert_eq!(settings.blas_error(), 0.05);
    }

    #[test]
    fn prepared_part_prunes_to_the_bound_and_cuts_the_blas() {
        let full = torus_meshlet_mesh();
        let identity = LodSettings {
            raster_error: 0.0,
            raytracing_error: 0.0,
        };
        let (meshlet, _, stats) = prepare_part(&full, &identity).unwrap();
        assert_eq!(stats.meshlets, full.meshlet_count());
        assert_eq!(stats.raster_triangles, full.triangle_count() as u64);
        assert_eq!(stats.packed_bytes, meshlet.packed_byte_len() as u64);
        assert_eq!(stats.blas_achieved_error, 0.0);
        assert_eq!(stats.blas_triangles, 128 * 64 * 2);

        let coarse = LodSettings {
            raster_error: 0.05,
            raytracing_error: 0.02,
        };
        let (pruned, blas, pruned_stats) = prepare_part(&full, &coarse).unwrap();
        assert!(pruned_stats.meshlets < stats.meshlets);
        assert!(pruned_stats.packed_bytes < stats.packed_bytes);
        assert!(pruned_stats.raster_triangles < stats.raster_triangles);
        assert!(pruned_stats.blas_triangles < stats.blas_triangles);
        assert!(pruned_stats.blas_achieved_error > 0.0);
        assert!(pruned_stats.blas_achieved_error <= coarse.blas_error());
        // The cut is the full mesh's cut at the effective bound, not the
        // pruned mesh's, so its error is reported unzeroed.
        let reference = full.raytracing_geometry(coarse.blas_error());
        assert_eq!(
            pruned_stats.blas_triangles as usize * 3,
            reference.indices.len()
        );
        assert_eq!(pruned_stats.blas_achieved_error, reference.achieved_error);
        assert_eq!(
            pruned
                .raytracing_geometry(coarse.blas_error())
                .indices
                .len(),
            reference.indices.len()
        );

        let Some(VertexAttributeValues::Float32x3(positions)) =
            blas.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions are not float3");
        };
        assert_eq!(positions.len() as u64, pruned_stats.blas_vertices);
        assert!(blas.attribute(Mesh::ATTRIBUTE_NORMAL).is_some());
        assert!(blas.attribute(Mesh::ATTRIBUTE_UV_0).is_some());
        assert_eq!(
            blas.indices().map(Indices::len),
            Some(pruned_stats.blas_triangles as usize * 3)
        );
    }

    #[test]
    fn budget_matches_the_manager() {
        assert_eq!(MESHLET_PAGE_BUDGET, 8 << 30);
        assert_eq!(human_bytes(MESHLET_PAGE_BUDGET), "8.00 GiB");
        assert_eq!(human_bytes(3 << 20), "3.0 MiB");
    }
}
