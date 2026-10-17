//! `EXT_mesh_gpu_instancing` instance transforms.
//!
//! The extension gives a mesh node per-instance `TRANSLATION`, `ROTATION` and
//! `SCALE` accessors. Each instance transform applies in the node's local
//! space, before the node's own transform.

use alloc::collections::BTreeMap;

use bevy_math::{Quat, Vec3, Vec4};
use bevy_transform::components::Transform;
use gltf::{
    accessor::{DataType, Dimensions, Item, Iter},
    animation::util::Rotations,
    Accessor, Document, Node,
};
use serde::Deserialize;

use crate::loader::{GltfError, EXT_MESH_GPU_INSTANCING};

/// The `EXT_mesh_gpu_instancing` block on a node.
#[derive(Deserialize)]
struct NodeExtension {
    /// Accessor indices keyed by attribute name.
    attributes: BTreeMap<String, usize>,
}

/// Reads the instance transforms of `node`.
///
/// Returns an empty `Vec` if `node` has no `EXT_mesh_gpu_instancing`
/// extension. Missing transform attributes default to the identity. Custom
/// attributes such as `_ID` count towards the instance count but are not
/// read; extension handlers can read them from the node.
///
/// Malformed extension data returns [`GltfError::InvalidMeshGpuInstancing`].
pub(crate) fn instance_transforms(
    document: &Document,
    node: &Node,
    buffer_data: &[Vec<u8>],
) -> Result<Vec<Transform>, GltfError> {
    let Some(value) = node.extension_value(EXT_MESH_GPU_INSTANCING) else {
        return Ok(Vec::new());
    };
    let invalid = |message: String| GltfError::InvalidMeshGpuInstancing {
        node: node.index(),
        message,
    };
    let extension = NodeExtension::deserialize(value).map_err(|err| invalid(err.to_string()))?;

    let mut count = None;
    let (mut translations, mut rotations, mut scales) = (None, None, None);
    for (name, &index) in &extension.attributes {
        let accessor = document
            .accessors()
            .nth(index)
            .ok_or_else(|| invalid(format!("{name} accessor {index} does not exist")))?;
        if let Some(count) = count
            && count != accessor.count()
        {
            return Err(invalid(format!(
                "{name} has {} elements, other attributes have {count}",
                accessor.count()
            )));
        }
        count = Some(accessor.count());
        match name.as_str() {
            "TRANSLATION" => translations = Some(accessor),
            "ROTATION" => rotations = Some(accessor),
            "SCALE" => scales = Some(accessor),
            _ => {}
        }
    }
    let Some(count) = count else {
        return Err(invalid("attributes is empty".into()));
    };

    let read_vec3 = |accessor: Accessor, name| match (accessor.dimensions(), accessor.data_type()) {
        (Dimensions::Vec3, DataType::F32) => {
            Ok(read::<[f32; 3]>(accessor, buffer_data, node, name)?.map(Vec3::from))
        }
        _ => Err(invalid(format!("{name} must be a float VEC3 accessor"))),
    };

    let mut transforms = vec![Transform::IDENTITY; count];
    if let Some(accessor) = translations {
        let translations = read_vec3(accessor, "TRANSLATION")?;
        for (transform, translation) in transforms.iter_mut().zip(translations) {
            transform.translation = translation;
        }
    }
    if let Some(accessor) = scales {
        for (transform, scale) in transforms.iter_mut().zip(read_vec3(accessor, "SCALE")?) {
            transform.scale = scale;
        }
    }
    if let Some(accessor) = rotations {
        let name = "ROTATION";
        let rotations = match (accessor.dimensions(), accessor.data_type()) {
            (Dimensions::Vec4, DataType::F32) => {
                Rotations::F32(read(accessor, buffer_data, node, name)?)
            }
            (Dimensions::Vec4, DataType::I8) if accessor.normalized() => {
                Rotations::I8(read(accessor, buffer_data, node, name)?)
            }
            (Dimensions::Vec4, DataType::I16) if accessor.normalized() => {
                Rotations::I16(read(accessor, buffer_data, node, name)?)
            }
            _ => {
                return Err(invalid(format!(
                    "{name} must be a float, normalized byte or normalized short VEC4 accessor"
                )));
            }
        };
        for (transform, rotation) in transforms.iter_mut().zip(rotations.into_f32()) {
            // Dequantized rotations are only approximately unit length.
            transform.rotation = Vec4::from(rotation)
                .try_normalize()
                .map_or(Quat::IDENTITY, Quat::from_vec4);
        }
    }

    Ok(transforms)
}

/// Iterates over the elements of `accessor`, whose type the caller checks.
fn read<'a, 's, T: Item>(
    accessor: Accessor<'a>,
    buffer_data: &'s [Vec<u8>],
    node: &Node,
    name: &str,
) -> Result<Iter<'s, T>, GltfError> {
    Iter::new(accessor, |buffer: gltf::Buffer| {
        buffer_data.get(buffer.index()).map(Vec::as_slice)
    })
    .ok_or_else(|| GltfError::InvalidMeshGpuInstancing {
        node: node.index(),
        message: format!("{name} has no readable data"),
    })
}
