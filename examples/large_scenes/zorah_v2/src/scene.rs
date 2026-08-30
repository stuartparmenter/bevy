//! The root glTF's node tree flattened into placed mesh instances.
//!
//! Both the bake (which meshes are referenced at all) and the runner (what to
//! spawn where) need the same walk, including the `EXT_mesh_gpu_instancing`
//! expansion the `gltf` crate does not perform itself.

use std::{collections::HashMap, fs, path::Path};

use bevy::{
    math::{Mat4, Quat, Vec3},
    transform::components::Transform,
};
use serde::Deserialize;
use thiserror::Error;

const EXT_MESH_GPU_INSTANCING: &str = "EXT_mesh_gpu_instancing";

/// One placed mesh: a node with a mesh, or one instance of an instanced node.
#[derive(Clone, Debug)]
pub struct SceneInstance {
    /// Root glTF mesh index.
    pub mesh: usize,
    /// Root glTF node index; the same node for every instance it expands to.
    pub node: usize,
    /// The node's name, empty when the file gives none.
    pub name: String,
    /// World transform in the export's own metres, Y-up space, flattened to
    /// TRS. Exact unless `chain` is non-empty.
    pub transform: Transform,
    /// The local TRS of every node from the scene root down to this
    /// placement, when their product carries shear (a rotated child under a
    /// non-uniformly scaled parent) that one `Transform` cannot hold; empty
    /// otherwise. The spawner mirrors such a chain as parent entities, whose
    /// affine `GlobalTransform` propagation reproduces the matrix exactly.
    pub chain: Vec<Transform>,
}

/// How far a world matrix may stray from its TRS flattening, relative to its
/// largest linear entry, before the placement keeps its node chain.
const SHEAR_TOLERANCE: f32 = 1e-3;

/// Whether `Transform::from_matrix(world)` reproduces `world`'s linear part.
fn is_trs(world: &Mat4, flattened: &Transform) -> bool {
    let rebuilt = flattened.to_matrix();
    let mut largest = 0.0f32;
    let mut error = 0.0f32;
    for column in 0..3 {
        for row in 0..3 {
            largest = largest.max(world.col(column)[row].abs());
            error = error.max((world.col(column)[row] - rebuilt.col(column)[row]).abs());
        }
    }
    error <= SHEAR_TOLERANCE * largest.max(f32::MIN_POSITIVE)
}

#[derive(Debug, Error)]
pub enum SceneError {
    #[error("glTF has no scene")]
    NoScene,
    #[error("node {node} ({name}): {message}")]
    Instancing {
        node: usize,
        name: String,
        message: String,
    },
    #[error("buffer {index} has no file URI")]
    BufferWithoutUri { index: usize },
    #[error("reading buffer {uri}: {source}")]
    BufferRead { uri: String, source: std::io::Error },
}

/// Flattens scene 0 of `document`, whose buffer URIs resolve against `root_dir`.
///
/// Only the buffers the instancing accessors live in are read (the export
/// keeps them in one `nodes.bin`); mesh geometry stays on disk.
pub fn walk_scene(
    document: &gltf::Document,
    root_dir: &Path,
) -> Result<Vec<SceneInstance>, SceneError> {
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .ok_or(SceneError::NoScene)?;

    let buffers = load_instancing_buffers(document, root_dir)?;
    let get_buffer = |buffer: gltf::Buffer| buffers.get(&buffer.index()).map(Vec::as_slice);

    let mut instances = Vec::new();
    // Each entry carries the parent's world matrix and the TRS chain that
    // produced it, so a sheared placement can hand the chain on.
    let mut stack = scene
        .nodes()
        .map(|node| (node, Mat4::IDENTITY, Vec::<Transform>::new()))
        .collect::<Vec<_>>();
    while let Some((node, parent, parent_chain)) = stack.pop() {
        let (translation, rotation, scale) = node.transform().decomposed();
        let local = Transform {
            translation: Vec3::from(translation),
            rotation: Quat::from_array(rotation),
            scale: Vec3::from(scale),
        };
        let world = parent * local.to_matrix();
        let mut chain = parent_chain;
        chain.push(local);
        if let Some(mesh) = node.mesh() {
            let name = node.name().unwrap_or_default().to_string();
            let locals = instance_transforms(document, &node, get_buffer)?;
            for instance_local in locals {
                let placed = world * instance_local.to_matrix();
                let transform = Transform::from_matrix(placed);
                let chain = if is_trs(&placed, &transform) {
                    Vec::new()
                } else {
                    let mut chain = chain.clone();
                    chain.push(instance_local);
                    chain
                };
                instances.push(SceneInstance {
                    mesh: mesh.index(),
                    node: node.index(),
                    name: name.clone(),
                    transform,
                    chain,
                });
            }
        }
        stack.extend(node.children().map(|child| (child, world, chain.clone())));
    }
    Ok(instances)
}

/// The local transforms an instanced node expands to; identity for a plain node.
fn instance_transforms<'a, 's, F>(
    document: &'a gltf::Document,
    node: &gltf::Node<'a>,
    get_buffer: F,
) -> Result<Vec<Transform>, SceneError>
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    let Some(extension) = node.extension_value(EXT_MESH_GPU_INSTANCING) else {
        return Ok(vec![Transform::IDENTITY]);
    };
    let error = |message: String| SceneError::Instancing {
        node: node.index(),
        name: node.name().unwrap_or_default().to_string(),
        message,
    };
    let attributes = extension
        .get("attributes")
        .and_then(|value| value.as_object())
        .ok_or_else(|| error("instancing extension has no attributes".into()))?;
    let accessor = |semantic: &str| -> Result<Option<gltf::Accessor<'a>>, SceneError> {
        let Some(index) = attributes.get(semantic) else {
            return Ok(None);
        };
        let index = index
            .as_u64()
            .ok_or_else(|| error(format!("{semantic} is not an accessor index")))?;
        document
            .accessors()
            .nth(index as usize)
            .map(Some)
            .ok_or_else(|| error(format!("{semantic} accessor {index} does not exist")))
    };
    let read = |semantic: &str, dimensions: gltf::accessor::Dimensions| {
        let Some(accessor) = accessor(semantic)? else {
            return Ok(None);
        };
        if accessor.dimensions() != dimensions
            || accessor.data_type() != gltf::accessor::DataType::F32
        {
            return Err(error(format!(
                "{semantic} accessor is not float {dimensions:?}"
            )));
        }
        Ok(Some(accessor))
    };
    let translations = read("TRANSLATION", gltf::accessor::Dimensions::Vec3)?
        .map(|accessor| {
            gltf::accessor::Iter::<[f32; 3]>::new(accessor, get_buffer.clone())
                .map(|iter| iter.map(Vec3::from).collect::<Vec<_>>())
                .ok_or_else(|| error("TRANSLATION buffer is unavailable".into()))
        })
        .transpose()?;
    let rotations = read("ROTATION", gltf::accessor::Dimensions::Vec4)?
        .map(|accessor| {
            gltf::accessor::Iter::<[f32; 4]>::new(accessor, get_buffer.clone())
                .map(|iter| iter.map(Quat::from_array).collect::<Vec<_>>())
                .ok_or_else(|| error("ROTATION buffer is unavailable".into()))
        })
        .transpose()?;
    let scales = read("SCALE", gltf::accessor::Dimensions::Vec3)?
        .map(|accessor| {
            gltf::accessor::Iter::<[f32; 3]>::new(accessor, get_buffer)
                .map(|iter| iter.map(Vec3::from).collect::<Vec<_>>())
                .ok_or_else(|| error("SCALE buffer is unavailable".into()))
        })
        .transpose()?;

    let counts = [
        translations.as_ref().map(Vec::len),
        rotations.as_ref().map(Vec::len),
        scales.as_ref().map(Vec::len),
    ];
    let mut present = counts.into_iter().flatten();
    let Some(count) = present.next() else {
        return Err(error("instancing extension has no TRS attribute".into()));
    };
    if present.any(|other| other != count) {
        return Err(error(format!(
            "instance attribute counts differ: {counts:?}"
        )));
    }
    Ok((0..count)
        .map(|i| Transform {
            translation: translations
                .as_ref()
                .map_or(Vec3::ZERO, |translations| translations[i]),
            rotation: rotations
                .as_ref()
                .map_or(Quat::IDENTITY, |rotations| rotations[i]),
            scale: scales.as_ref().map_or(Vec3::ONE, |scales| scales[i]),
        })
        .collect())
}

/// Reads every buffer an instancing accessor points into, keyed by buffer index.
fn load_instancing_buffers(
    document: &gltf::Document,
    root_dir: &Path,
) -> Result<HashMap<usize, Vec<u8>>, SceneError> {
    let mut buffers = HashMap::new();
    for node in document.nodes() {
        let Some(attributes) = node
            .extension_value(EXT_MESH_GPU_INSTANCING)
            .and_then(|value| value.get("attributes"))
            .and_then(|value| value.as_object())
        else {
            continue;
        };
        for index in attributes.values().filter_map(serde_json::Value::as_u64) {
            let Some(view) = document
                .accessors()
                .nth(index as usize)
                .and_then(|accessor| accessor.view())
            else {
                continue;
            };
            let buffer = view.buffer();
            if buffers.contains_key(&buffer.index()) {
                continue;
            }
            let gltf::buffer::Source::Uri(uri) = buffer.source() else {
                return Err(SceneError::BufferWithoutUri {
                    index: buffer.index(),
                });
            };
            let bytes = fs::read(root_dir.join(uri)).map_err(|source| SceneError::BufferRead {
                uri: uri.to_string(),
                source,
            })?;
            buffers.insert(buffer.index(), bytes);
        }
    }
    Ok(buffers)
}

/// The starting view from the export's sidecar `<name>.scene.json`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneView {
    pub position: Vec3,
    pub target: Vec3,
    /// Vertical field of view in degrees.
    pub fov_degrees: f32,
}

impl Default for SceneView {
    /// The sidecar's values, so a copy of the export without it still opens
    /// on the reference view.
    fn default() -> Self {
        Self {
            position: Vec3::new(43.4336, 9.56561, -0.0810344),
            target: Vec3::new(-95.954, 32.9765, -1.63703),
            fov_degrees: 35.0,
        }
    }
}

#[derive(Deserialize)]
struct SceneJson {
    view: Option<ViewJson>,
}

#[derive(Deserialize)]
struct ViewJson {
    position: Option<[f32; 3]>,
    lookat: Option<[f32; 3]>,
    fov: Option<f32>,
}

/// Parses the sidecar's `view`; fields it lacks keep the defaults.
pub fn parse_view(json: &[u8]) -> Result<SceneView, serde_json::Error> {
    let scene: SceneJson = serde_json::from_slice(json)?;
    let mut view = SceneView::default();
    if let Some(given) = scene.view {
        if let Some(position) = given.position {
            view.position = Vec3::from(position);
        }
        if let Some(target) = given.lookat {
            view.target = Vec3::from(target);
        }
        if let Some(fov) = given.fov.filter(|fov| *fov > 0.0 && *fov < 180.0) {
            view.fov_degrees = fov;
        }
    }
    Ok(view)
}

/// The sidecar beside `gltf_path`, or the defaults when it is missing or
/// unreadable. A malformed sidecar is reported as the returned warning, since
/// this runs before the app has a log subscriber.
pub fn read_view(gltf_path: &Path) -> (SceneView, Option<String>) {
    let sidecar = gltf_path.with_extension("scene.json");
    match fs::read(&sidecar) {
        Ok(bytes) => match parse_view(&bytes) {
            Ok(view) => (view, None),
            Err(error) => (
                SceneView::default(),
                Some(format!(
                    "{}: {error}; using the default view",
                    sidecar.display()
                )),
            ),
        },
        Err(_) => (SceneView::default(), None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_sidecar_view() {
        let view = parse_view(
            br#"{"view": {"position": [1, 2, 3], "lookat": [4, 5, 6], "up": [0, 1, 0], "fov": 40}}"#,
        )
        .unwrap();
        assert_eq!(view.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(view.target, Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(view.fov_degrees, 40.0);
        assert_eq!(parse_view(b"{}").unwrap(), SceneView::default());
    }

    /// Walks the real export when `ZORAH_ROOT` points at it: every chained
    /// placement's product must be the matrix its flattening only approximates.
    #[test]
    #[ignore = "needs the export at ZORAH_ROOT"]
    #[expect(clippy::print_stdout, reason = "a diagnostic for --nocapture")]
    fn real_export_chains_reproduce_their_matrices() {
        let root = std::path::PathBuf::from(std::env::var_os("ZORAH_ROOT").unwrap());
        let bytes = fs::read(root.join("zorah_textured_public.v1.gltf")).unwrap();
        let document = gltf::Gltf::from_slice_without_validation(&bytes)
            .unwrap()
            .document;
        let instances = walk_scene(&document, &root).unwrap();
        let chained = instances
            .iter()
            .filter(|instance| !instance.chain.is_empty())
            .collect::<Vec<_>>();
        let mut names = chained
            .iter()
            .map(|instance| {
                instance
                    .name
                    .split('_')
                    .take(3)
                    .collect::<Vec<_>>()
                    .join("_")
            })
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        assert!(
            !chained.is_empty(),
            "no placement kept a chain; expected the palms to"
        );
        // Surfaces in `--nocapture` output for anyone curious which props shear.
        println!(
            "{} of {} placements keep a chain: {names:?}",
            chained.len(),
            instances.len()
        );
        for instance in chained {
            let product = instance
                .chain
                .iter()
                .fold(Mat4::IDENTITY, |acc, local| acc * local.to_matrix());
            let flattened = instance.transform.to_matrix();
            assert!(!is_trs(&product, &instance.transform), "{}", instance.name);
            assert!(
                product.w_axis.abs_diff_eq(flattened.w_axis, 1e-3),
                "{}: translation drifted",
                instance.name
            );
        }
    }

    #[test]
    fn shear_is_detected_only_where_trs_cannot_hold_it() {
        let rigid = Transform::from_xyz(1.0, 2.0, 3.0)
            .with_rotation(Quat::from_rotation_y(0.7))
            .with_scale(Vec3::new(2.0, 3.0, -4.0));
        let world = rigid.to_matrix();
        assert!(is_trs(&world, &Transform::from_matrix(world)));

        // A rotated child under a non-uniform parent scale: the classic shear.
        let parent = Transform::from_scale(Vec3::new(1.0, 3.0, 1.0));
        let child = Transform::from_rotation(Quat::from_rotation_z(0.5));
        let sheared = parent.to_matrix() * child.to_matrix();
        assert!(!is_trs(&sheared, &Transform::from_matrix(sheared)));
    }
}
