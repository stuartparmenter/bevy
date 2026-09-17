use bevy_asset::AssetId;
use bevy_ecs::prelude::*;
use bevy_mesh::{Indices, Mesh, PrimitiveTopology, VertexAttributeValues, VertexFormat};
use bevy_platform::collections::HashMap;
use bevy_render::{
    mesh::RenderMesh,
    render_asset::ExtractedAssets,
    render_resource::{Buffer, BufferInitDescriptor, BufferUsages},
    renderer::RenderDevice,
};
use tracing::warn;

/// Offsets and stride in words in the mesh allocator's interleaved vertex buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DeformationSourceLayout {
    pub stride: u32,
    pub position: u32,
    pub normal: u32,
    pub uv: u32,
    pub tangent: u32,
    pub joint_weights: u32,
    pub joint_indices: u32,
    pub joint_indices_u32: u32,
}

pub(super) struct DeformationSource {
    /// Shared by instances. Its ID also identifies the source asset revision.
    pub index_buffer: Buffer,
    pub vertex_count: u32,
    pub index_count: u32,
    pub layout: DeformationSourceLayout,
    pub morph_target_count: u32,
    pub max_joint_index: u32,
    pub skinned: bool,
}

#[derive(Resource, Default)]
pub(super) struct DeformationSources(HashMap<AssetId<Mesh>, DeformationSource>);

impl DeformationSources {
    pub fn get(&self, id: AssetId<Mesh>) -> Option<&DeformationSource> {
        self.0.get(&id)
    }
}

/// Reads extracted mesh data before `prepare_assets::<RenderMesh>` consumes it.
pub(super) fn prepare_deformation_sources(
    meshes: Res<ExtractedAssets<RenderMesh>>,
    render_device: Res<RenderDevice>,
    mut sources: ResMut<DeformationSources>,
) {
    for id in meshes.removed.iter().chain(&meshes.modified) {
        sources.0.remove(id);
    }

    for (id, mesh) in &meshes.extracted {
        sources.0.remove(id);
        if !mesh.enable_raytracing
            || !(mesh.contains_attribute(Mesh::ATTRIBUTE_JOINT_INDEX)
                || mesh.contains_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT)
                || mesh.has_morph_targets())
        {
            continue;
        }
        let data = match validate_source(mesh) {
            Ok(data) => data,
            Err(error) => {
                warn!("Cannot deform raytracing mesh {id:?}: {error}");
                continue;
            }
        };
        let limits = render_device.limits();
        let storage_limit =
            u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
        if data.indices.len() as u64 * 4 > storage_limit
            || data.vertex_count > limits.max_blas_primitive_count
            || data.indices.len() as u64 / 3 > u64::from(limits.max_blas_primitive_count)
        {
            warn!("Cannot deform raytracing mesh {id:?}: geometry exceeds device limits");
            continue;
        }
        let index_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("solari_deformation_indices"),
            contents: bytemuck::cast_slice(&data.indices),
            usage: BufferUsages::STORAGE | BufferUsages::BLAS_INPUT,
        });
        sources.0.insert(
            *id,
            DeformationSource {
                index_buffer,
                vertex_count: data.vertex_count,
                index_count: data.indices.len() as u32,
                layout: data.layout,
                morph_target_count: data.morph_target_count,
                max_joint_index: data.max_joint_index,
                skinned: data.skinned,
            },
        );
    }
}

#[derive(Debug)]
struct SourceData {
    indices: Vec<u32>,
    vertex_count: u32,
    layout: DeformationSourceLayout,
    morph_target_count: u32,
    max_joint_index: u32,
    skinned: bool,
}

fn validate_source(mesh: &Mesh) -> Result<SourceData, &'static str> {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return Err("expected TriangleList topology");
    }
    let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return Err("positions must be uncompressed Float32x3");
    };
    let vertex_count = u32::try_from(positions.len()).map_err(|_| "too many vertices")?;
    if vertex_count == 0 {
        return Err("mesh has no vertices");
    }
    let mut layout = DeformationSourceLayout {
        stride: 0,
        position: u32::MAX,
        normal: u32::MAX,
        uv: u32::MAX,
        tangent: u32::MAX,
        joint_weights: u32::MAX,
        joint_indices: u32::MAX,
        joint_indices_u32: 0,
    };
    let mut byte_offset = 0u64;
    let mut max_joint_index = 0;
    // Mesh packs attributes in this same order, including attributes we do not use.
    for (attribute, values) in mesh.attributes() {
        if values.len() != positions.len() {
            return Err("vertex attributes have different lengths");
        }
        let (slot, expected_format) = match attribute.id {
            id if id == Mesh::ATTRIBUTE_POSITION.id => {
                (&mut layout.position, VertexFormat::Float32x3)
            }
            id if id == Mesh::ATTRIBUTE_NORMAL.id => (&mut layout.normal, VertexFormat::Float32x3),
            id if id == Mesh::ATTRIBUTE_UV_0.id => (&mut layout.uv, VertexFormat::Float32x2),
            id if id == Mesh::ATTRIBUTE_TANGENT.id => {
                (&mut layout.tangent, VertexFormat::Float32x4)
            }
            id if id == Mesh::ATTRIBUTE_JOINT_WEIGHT.id => {
                (&mut layout.joint_weights, VertexFormat::Float32x4)
            }
            id if id == Mesh::ATTRIBUTE_JOINT_INDEX.id => {
                let format = match values {
                    VertexAttributeValues::Uint16x4(indices) => {
                        max_joint_index =
                            indices.iter().flatten().copied().max().unwrap_or(0).into();
                        VertexFormat::Uint16x4
                    }
                    VertexAttributeValues::Uint32x4(indices) => {
                        max_joint_index = indices.iter().flatten().copied().max().unwrap_or(0);
                        layout.joint_indices_u32 = 1;
                        VertexFormat::Uint32x4
                    }
                    _ => return Err("joint indices must be Uint16x4 or Uint32x4"),
                };
                (&mut layout.joint_indices, format)
            }
            _ => {
                byte_offset += attribute.format.size();
                continue;
            }
        };
        if attribute.format != expected_format {
            return Err("unsupported vertex attribute format; compressed deformation attributes are not supported");
        }
        if !byte_offset.is_multiple_of(4) {
            return Err("deformation attributes must be aligned to four bytes");
        }
        *slot = u32::try_from(byte_offset / 4).map_err(|_| "vertex stride is too large")?;
        byte_offset += attribute.format.size();
    }
    if layout.normal == u32::MAX {
        return Err("normals must be present as uncompressed Float32x3");
    }
    if !byte_offset.is_multiple_of(4) {
        return Err("vertex stride must be aligned to four bytes");
    }
    layout.stride = u32::try_from(byte_offset / 4).map_err(|_| "vertex stride is too large")?;
    let skinned = layout.joint_indices != u32::MAX;
    if skinned != (layout.joint_weights != u32::MAX) {
        return Err("skinning requires both joint indices and joint weights");
    }

    let indices = match mesh.indices() {
        Some(Indices::U16(indices)) => indices.iter().map(|&index| u32::from(index)).collect(),
        Some(Indices::U32(indices)) => indices.clone(),
        None => (0..vertex_count).collect(),
    };
    if indices.is_empty() || !indices.len().is_multiple_of(3) {
        return Err("index count must be a nonzero multiple of three");
    }
    if u32::try_from(indices.len()).is_err() {
        return Err("too many indices");
    }
    if indices.iter().any(|&index| index >= vertex_count) {
        return Err("index is outside the vertex buffer");
    }
    let morph_vertex_count = mesh.get_morph_targets().map_or(0, <[_]>::len);
    if !morph_vertex_count.is_multiple_of(positions.len()) {
        return Err("morph target data must contain one displacement per vertex per target");
    }
    let morph_target_count = u32::try_from(morph_vertex_count / positions.len())
        .map_err(|_| "too many morph targets")?;
    Ok(SourceData {
        indices,
        vertex_count,
        layout,
        morph_target_count,
        max_joint_index,
        skinned,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::RenderAssetUsages;
    use bevy_mesh::morph::MorphAttributes;

    fn triangle() -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0; 3]; 3])
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; 3])
    }

    #[test]
    fn optional_attributes_and_nonindexed_mesh() {
        let source = validate_source(&triangle()).unwrap();
        assert_eq!(source.indices, [0, 1, 2]);
        assert_eq!(source.layout.stride, 6);
        assert_eq!(source.layout.uv, u32::MAX);
        assert_eq!(source.layout.tangent, u32::MAX);
        assert!(!source.skinned);
    }

    #[test]
    fn extra_attributes_and_both_joint_index_formats() {
        for wide in [false, true] {
            let mut mesh = triangle()
                .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0; 2]; 3])
                .with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, vec![[0.0; 2]; 3])
                .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0; 4]; 3])
                .with_inserted_attribute(
                    Mesh::ATTRIBUTE_JOINT_WEIGHT,
                    vec![[1.0, 0.0, 0.0, 0.0]; 3],
                );
            let mut joint_attribute = Mesh::ATTRIBUTE_JOINT_INDEX;
            if wide {
                joint_attribute.format = VertexFormat::Uint32x4;
            }
            mesh.insert_attribute(
                joint_attribute,
                if wide {
                    VertexAttributeValues::Uint32x4(vec![[1, 2, 3, 9]; 3])
                } else {
                    VertexAttributeValues::Uint16x4(vec![[1, 2, 3, 9]; 3])
                },
            );
            let source = validate_source(&mesh).unwrap();
            let mut offset = 0;
            for (attribute, _) in mesh.attributes() {
                if attribute.id == Mesh::ATTRIBUTE_JOINT_INDEX.id {
                    assert_eq!(source.layout.joint_indices, offset / 4);
                }
                offset += attribute.format.size() as u32;
            }
            assert_eq!(source.layout.stride, offset / 4);
            assert_eq!(source.max_joint_index, 9);
            assert_eq!(source.layout.joint_indices_u32, u32::from(wide));
        }
    }

    #[test]
    fn widens_indices_and_rejects_invalid_triangles() {
        let mut mesh = triangle().with_inserted_indices(Indices::U16(vec![2, 1, 0]));
        assert_eq!(validate_source(&mesh).unwrap().indices, [2, 1, 0]);
        mesh.insert_indices(Indices::U32(vec![0, 1, 3]));
        assert!(validate_source(&mesh).is_err());
        mesh.insert_indices(Indices::U32(vec![0, 1]));
        assert!(validate_source(&mesh).is_err());
    }

    #[test]
    fn rejects_mismatched_attributes_and_incomplete_skinning() {
        let mut mesh = triangle();
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0; 3]; 2]);
        assert!(validate_source(&mesh).is_err());
        let mesh =
            triangle().with_inserted_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, vec![[1.0; 4]; 3]);
        assert!(validate_source(&mesh).is_err());
    }

    #[test]
    fn rejects_compressed_attributes_and_nontriangle_topology() {
        let mut mesh = triangle();
        mesh.compress_normals().unwrap();
        assert!(validate_source(&mesh).unwrap_err().contains("compressed"));
        let mesh = Mesh::new(
            PrimitiveTopology::TriangleStrip,
            RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0; 3]; 3])
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; 3]);
        assert!(validate_source(&mesh).unwrap_err().contains("TriangleList"));
    }

    #[test]
    fn validates_target_major_morph_data() {
        let mut mesh = triangle();
        mesh.set_morph_targets(vec![MorphAttributes::default(); 6]);
        assert_eq!(validate_source(&mesh).unwrap().morph_target_count, 2);
        mesh.set_morph_targets(vec![MorphAttributes::default(); 4]);
        assert!(validate_source(&mesh).is_err());
    }
}
