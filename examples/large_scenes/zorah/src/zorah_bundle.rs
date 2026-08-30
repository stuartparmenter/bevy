//! Zorah-specific bundle assets produced by the offline converter.

use bevy::{
    asset::{
        io::Reader, AssetApp, AssetLoader, AsyncReadExt, LoadContext, RenderAssetUsages,
        UntypedHandle,
    },
    image::{
        CompressedImageFormatSupport, CompressedImageFormats, ImageAddressMode, ImageFilterMode,
        ImageSampler, ImageSamplerDescriptor, ImageType,
    },
    pbr::experimental::meshlet::MeshletMeshLoader,
    prelude::*,
    reflect::TypePath,
    render::{mesh::Indices, render_resource::PrimitiveTopology},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const ZORAH_BUNDLE_MAGIC: [u8; 8] = *b"ZORAHB01";
pub const ZORAH_BUNDLE_VERSION: u32 = 2;
const ZORAH_MESHLET_BLAS_MAGIC: [u8; 8] = *b"ZBLAS001";
/// The packer writes one JSON index per shard listing at most a few thousand
/// entries. Anything beyond this is a corrupt length prefix, not a real index.
const MAX_INDEX_BYTES: u64 = 64 << 20;
/// One entry is a single texture, meshlet, or BLAS payload; whole shards are
/// two orders of magnitude larger than the largest of them.
const MAX_ENTRY_BYTES: u64 = 1 << 30;
/// Length prefixes are read before the payload exists, so fill the buffer in
/// bounded steps: a truncated stream then fails on the first short read instead
/// of after the full allocation is committed.
const READ_CHUNK_BYTES: usize = 8 << 20;

#[derive(Asset, TypePath)]
pub struct ZorahBundle {
    // Strong handles keep every labeled asset alive while the bundle is loaded.
    #[allow(dead_code)]
    assets: Vec<UntypedHandle>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BundleIndex {
    pub format_version: u32,
    pub entries: Vec<BundleEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BundleEntry {
    pub label: String,
    pub byte_length: u64,
    #[serde(flatten)]
    pub kind: BundleEntryKind,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BundleEntryKind {
    Mesh,
    MeshletBlas,
    Meshlet,
    Image { srgb: bool },
}

pub struct ZorahBundlePlugin;

impl Plugin for ZorahBundlePlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<ZorahBundle>()
            .preregister_asset_loader::<ZorahBundleLoader>(&["zorah_bundle"]);
    }

    fn finish(&self, app: &mut App) {
        let compressed_formats = app
            .world()
            .get_resource::<CompressedImageFormatSupport>()
            .map_or(CompressedImageFormats::NONE, |support| support.0);
        app.register_asset_loader(ZorahBundleLoader { compressed_formats });
    }
}

#[derive(TypePath)]
pub struct ZorahBundleLoader {
    compressed_formats: CompressedImageFormats,
}

#[derive(Debug, Error)]
pub enum ZorahBundleError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("failed to decode bundled meshlets: {0}")]
    Meshlet(String),
    #[error("invalid Zorah bundle: {0}")]
    Invalid(String),
    #[error("failed to decode bundled image: {0}")]
    Image(String),
}

impl AssetLoader for ZorahBundleLoader {
    type Asset = ZorahBundle;
    type Settings = ();
    type Error = ZorahBundleError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        load_context: &mut LoadContext<'_>,
    ) -> Result<ZorahBundle, Self::Error> {
        let mut magic = [0; 8];
        reader.read_exact(&mut magic).await?;
        if magic != ZORAH_BUNDLE_MAGIC {
            return Err(ZorahBundleError::Invalid("wrong magic".into()));
        }
        let version = read_u32(reader).await?;
        if version != ZORAH_BUNDLE_VERSION {
            return Err(ZorahBundleError::Invalid(format!(
                "version {version}, expected {ZORAH_BUNDLE_VERSION}"
            )));
        }
        let index_length = read_u64(reader).await?;
        let index_bytes = read_payload(reader, index_length, MAX_INDEX_BYTES, "index").await?;
        let index: BundleIndex = serde_json::from_slice(&index_bytes)?;
        if index.format_version != ZORAH_BUNDLE_VERSION {
            return Err(ZorahBundleError::Invalid(format!(
                "index version {}, expected {ZORAH_BUNDLE_VERSION}",
                index.format_version
            )));
        }

        let mut assets = Vec::with_capacity(index.entries.len());
        for entry in index.entries {
            let bytes =
                read_payload(reader, entry.byte_length, MAX_ENTRY_BYTES, &entry.label).await?;
            let handle = match entry.kind {
                BundleEntryKind::Mesh => load_context
                    .add_labeled_asset(entry.label, mesh_from_converter_glb(&bytes, true)?)
                    .untyped(),
                BundleEntryKind::MeshletBlas => load_context
                    .add_labeled_asset(entry.label, meshlet_blas_from_bytes(&bytes)?)
                    .untyped(),
                BundleEntryKind::Meshlet => {
                    let mut meshlet_reader = bevy::asset::io::VecReader::new(bytes);
                    let meshlet = MeshletMeshLoader
                        .load(&mut meshlet_reader, &(), load_context)
                        .await
                        .map_err(|error| ZorahBundleError::Meshlet(error.to_string()))?;
                    load_context
                        .add_labeled_asset(entry.label, meshlet)
                        .untyped()
                }
                BundleEntryKind::Image { srgb } => {
                    // Render-world only: nothing reads bundle images from the
                    // main world, and the CPU copy of ~2 GiB of BC data per
                    // shard would otherwise stay resident for the whole run.
                    let image = Image::from_buffer(
                        &bytes,
                        ImageType::Extension("ktx2"),
                        self.compressed_formats,
                        srgb,
                        repeat_sampler(),
                        RenderAssetUsages::RENDER_WORLD,
                    )
                    .map_err(|error| ZorahBundleError::Image(error.to_string()))?;
                    load_context.add_labeled_asset(entry.label, image).untyped()
                }
            };
            assets.push(handle);
        }
        Ok(ZorahBundle { assets })
    }

    fn extensions(&self) -> &[&str] {
        &["zorah_bundle"]
    }
}

pub fn meshlet_blas_from_bytes(bytes: &[u8]) -> Result<Mesh, ZorahBundleError> {
    if bytes.len() < 20 || bytes[..8] != ZORAH_MESHLET_BLAS_MAGIC {
        return Err(ZorahBundleError::Invalid(
            "meshlet BLAS payload has the wrong magic".into(),
        ));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != 1 {
        return Err(ZorahBundleError::Invalid(format!(
            "meshlet BLAS payload version {version} is unsupported"
        )));
    }
    let vertex_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let index_count = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    let expected = 20usize
        .checked_add(vertex_count.checked_mul(32).ok_or_else(|| {
            ZorahBundleError::Invalid("meshlet BLAS vertex byte count overflow".into())
        })?)
        .and_then(|size| size.checked_add(index_count.checked_mul(4)?))
        .ok_or_else(|| ZorahBundleError::Invalid("meshlet BLAS byte count overflow".into()))?;
    if bytes.len() != expected {
        return Err(ZorahBundleError::Invalid(format!(
            "meshlet BLAS payload is {} bytes, expected {expected}",
            bytes.len()
        )));
    }
    let mut offset = 20;
    let mut positions = Vec::with_capacity(vertex_count);
    let mut normals = Vec::with_capacity(vertex_count);
    let mut uvs = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        let mut read_f32 = || {
            let value = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            offset += 4;
            value
        };
        positions.push([read_f32(), read_f32(), read_f32()]);
        normals.push([read_f32(), read_f32(), read_f32()]);
        uvs.push([read_f32(), read_f32()]);
    }
    let mut indices = Vec::with_capacity(index_count);
    for _ in 0..index_count {
        indices.push(u32::from_le_bytes(
            bytes[offset..offset + 4].try_into().unwrap(),
        ));
        offset += 4;
    }
    if indices.iter().any(|index| *index as usize >= vertex_count) {
        return Err(ZorahBundleError::Invalid(
            "meshlet BLAS index exceeds its vertex count".into(),
        ));
    }
    Ok(
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
            .with_inserted_indices(Indices::U32(indices)),
    )
}

async fn read_payload(
    reader: &mut dyn Reader,
    length: u64,
    limit: u64,
    what: &str,
) -> Result<Vec<u8>, ZorahBundleError> {
    if length > limit {
        return Err(ZorahBundleError::Invalid(format!(
            "{what} claims {length} bytes, over the {limit}-byte bound"
        )));
    }
    let length = usize::try_from(length)
        .map_err(|_| ZorahBundleError::Invalid(format!("{what} is too large for this target")))?;
    let mut bytes = Vec::new();
    while bytes.len() < length {
        let filled = bytes.len();
        bytes.resize(filled + (length - filled).min(READ_CHUNK_BYTES), 0);
        reader.read_exact(&mut bytes[filled..]).await?;
    }
    Ok(bytes)
}

async fn read_u32(reader: &mut dyn Reader) -> std::io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes).await?;
    Ok(u32::from_le_bytes(bytes))
}

async fn read_u64(reader: &mut dyn Reader) -> std::io::Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes).await?;
    Ok(u64::from_le_bytes(bytes))
}

fn repeat_sampler() -> ImageSampler {
    ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        address_mode_w: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: 16,
        ..default()
    })
}

pub fn mesh_from_converter_glb(
    bytes: &[u8],
    require_tangents: bool,
) -> Result<Mesh, ZorahBundleError> {
    let (document, payload) = read_glb(bytes)?;
    let primitive = document
        .get("meshes")
        .and_then(|meshes| meshes.get(0))
        .and_then(|mesh| mesh.get("primitives"))
        .and_then(|primitives| primitives.get(0))
        .ok_or_else(|| ZorahBundleError::Invalid("GLB has no mesh primitive".into()))?;
    let attributes = primitive
        .get("attributes")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| ZorahBundleError::Invalid("GLB primitive has no attributes".into()))?;
    let positions = read_vec3(&document, payload, attribute_index(attributes, "POSITION")?)?;
    let normals = read_vec3(&document, payload, attribute_index(attributes, "NORMAL")?)?;
    let uv0 = read_vec2(
        &document,
        payload,
        attribute_index(attributes, "TEXCOORD_0")?,
    )?;
    let tangents = attributes
        .get("TANGENT")
        .and_then(serde_json::Value::as_u64)
        .map(|index| read_vec4(&document, payload, index as usize))
        .transpose()?;
    if require_tangents && tangents.is_none() {
        return Err(ZorahBundleError::Invalid(
            "conventional GLB lacks tangents".into(),
        ));
    }
    if !require_tangents && tangents.is_some() {
        return Err(ZorahBundleError::Invalid(
            "meshlet-source GLB unexpectedly contains tangents".into(),
        ));
    }
    let index = primitive
        .get("indices")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| ZorahBundleError::Invalid("GLB primitive has no indices".into()))?;
    let indices = read_indices(&document, payload, index as usize, positions.len())?;

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uv0)
        .with_inserted_indices(Indices::U32(indices));
    if let Some(tangents) = tangents {
        mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, tangents);
    }
    Ok(mesh)
}

fn read_glb(bytes: &[u8]) -> Result<(serde_json::Value, &[u8]), ZorahBundleError> {
    if bytes.len() < 28 || &bytes[..4] != b"glTF" {
        return Err(ZorahBundleError::Invalid("truncated GLB".into()));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let total_length = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if version != 2 || total_length != bytes.len() {
        return Err(ZorahBundleError::Invalid("invalid GLB header".into()));
    }
    let json_length = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    if u32::from_le_bytes(bytes[16..20].try_into().unwrap()) != 0x4e4f534a {
        return Err(ZorahBundleError::Invalid("GLB JSON chunk is absent".into()));
    }
    let json_end = 20usize
        .checked_add(json_length)
        .ok_or_else(|| ZorahBundleError::Invalid("GLB JSON length overflow".into()))?;
    if json_end + 8 > bytes.len() {
        return Err(ZorahBundleError::Invalid("truncated GLB JSON chunk".into()));
    }
    let document: serde_json::Value = serde_json::from_slice(&bytes[20..json_end])?;
    let binary_length =
        u32::from_le_bytes(bytes[json_end..json_end + 4].try_into().unwrap()) as usize;
    if u32::from_le_bytes(bytes[json_end + 4..json_end + 8].try_into().unwrap()) != 0x004e4942 {
        return Err(ZorahBundleError::Invalid("GLB BIN chunk is absent".into()));
    }
    let binary_start = json_end + 8;
    let logical_length = document
        .get("buffers")
        .and_then(|buffers| buffers.get(0))
        .and_then(|buffer| buffer.get("byteLength"))
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| ZorahBundleError::Invalid("GLB buffer length is absent".into()))?
        as usize;
    if logical_length > binary_length || binary_start + binary_length != bytes.len() {
        return Err(ZorahBundleError::Invalid("invalid GLB BIN length".into()));
    }
    Ok((
        document,
        &bytes[binary_start..binary_start + logical_length],
    ))
}

fn attribute_index(
    attributes: &serde_json::Map<String, serde_json::Value>,
    semantic: &str,
) -> Result<usize, ZorahBundleError> {
    attributes
        .get(semantic)
        .and_then(serde_json::Value::as_u64)
        .map(|index| index as usize)
        .ok_or_else(|| ZorahBundleError::Invalid(format!("GLB lacks {semantic}")))
}

fn accessor_bytes<'a>(
    document: &serde_json::Value,
    payload: &'a [u8],
    accessor_index: usize,
    columns: usize,
    component_type: u64,
) -> Result<&'a [u8], ZorahBundleError> {
    let accessor = document
        .get("accessors")
        .and_then(|accessors| accessors.get(accessor_index))
        .ok_or_else(|| ZorahBundleError::Invalid("GLB accessor is absent".into()))?;
    if accessor
        .get("componentType")
        .and_then(serde_json::Value::as_u64)
        != Some(component_type)
    {
        return Err(ZorahBundleError::Invalid(
            "unexpected GLB component type".into(),
        ));
    }
    let view_index = accessor
        .get("bufferView")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| ZorahBundleError::Invalid("GLB accessor has no buffer view".into()))?
        as usize;
    let view = document
        .get("bufferViews")
        .and_then(|views| views.get(view_index))
        .ok_or_else(|| ZorahBundleError::Invalid("GLB buffer view is absent".into()))?;
    let offset = view
        .get("byteOffset")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
        .checked_add(
            accessor
                .get("byteOffset")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        )
        .ok_or_else(|| ZorahBundleError::Invalid("GLB accessor offset overflow".into()))?;
    let count = accessor
        .get("count")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| ZorahBundleError::Invalid("GLB accessor count is absent".into()))?;
    let component_size: u64 = match component_type {
        5125 | 5126 => 4,
        _ => {
            return Err(ZorahBundleError::Invalid(
                "unsupported GLB component type".into(),
            ))
        }
    };
    let length = count
        .checked_mul(columns as u64)
        .and_then(|value| value.checked_mul(component_size))
        .ok_or_else(|| ZorahBundleError::Invalid("GLB accessor length overflow".into()))?;
    let start = usize::try_from(offset)
        .map_err(|_| ZorahBundleError::Invalid("GLB accessor offset is too large".into()))?;
    let end = offset
        .checked_add(length)
        .and_then(|end| usize::try_from(end).ok())
        .ok_or_else(|| ZorahBundleError::Invalid("GLB accessor is too large".into()))?;
    payload
        .get(start..end)
        .ok_or_else(|| ZorahBundleError::Invalid("GLB accessor exceeds payload".into()))
}

fn read_vec2(
    document: &serde_json::Value,
    payload: &[u8],
    index: usize,
) -> Result<Vec<[f32; 2]>, ZorahBundleError> {
    read_float_vectors::<2>(accessor_bytes(document, payload, index, 2, 5126)?)
}

fn read_vec3(
    document: &serde_json::Value,
    payload: &[u8],
    index: usize,
) -> Result<Vec<[f32; 3]>, ZorahBundleError> {
    read_float_vectors::<3>(accessor_bytes(document, payload, index, 3, 5126)?)
}

fn read_vec4(
    document: &serde_json::Value,
    payload: &[u8],
    index: usize,
) -> Result<Vec<[f32; 4]>, ZorahBundleError> {
    read_float_vectors::<4>(accessor_bytes(document, payload, index, 4, 5126)?)
}

fn read_float_vectors<const N: usize>(bytes: &[u8]) -> Result<Vec<[f32; N]>, ZorahBundleError> {
    if !bytes.len().is_multiple_of(4 * N) {
        return Err(ZorahBundleError::Invalid(
            "misaligned float accessor".into(),
        ));
    }
    Ok(bytes
        .chunks_exact(4 * N)
        .map(|chunk| {
            core::array::from_fn(|column| {
                let start = column * 4;
                f32::from_le_bytes(chunk[start..start + 4].try_into().unwrap())
            })
        })
        .collect())
}

fn read_indices(
    document: &serde_json::Value,
    payload: &[u8],
    index: usize,
    vertex_count: usize,
) -> Result<Vec<u32>, ZorahBundleError> {
    let bytes = accessor_bytes(document, payload, index, 1, 5125)?;
    let indices: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if indices.iter().any(|index| *index as usize >= vertex_count) {
        return Err(ZorahBundleError::Invalid(
            "GLB index exceeds its vertex count".into(),
        ));
    }
    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{asset::io::VecReader, tasks::block_on};

    /// A triangle GLB in the layout `partition_mesh.py` writes: interleaved-free
    /// buffer views for POSITION, NORMAL, TEXCOORD_0 and u32 indices.
    fn triangle_glb(indices: [u32; 3]) -> Vec<u8> {
        let mut binary = Vec::new();
        for vertex in 0..3u32 {
            binary.extend_from_slice(&(vertex as f32).to_le_bytes());
            binary.extend_from_slice(&0.0f32.to_le_bytes());
            binary.extend_from_slice(&0.0f32.to_le_bytes());
        }
        for _ in 0..3 {
            binary.extend_from_slice(&0.0f32.to_le_bytes());
            binary.extend_from_slice(&1.0f32.to_le_bytes());
            binary.extend_from_slice(&0.0f32.to_le_bytes());
        }
        for _ in 0..3 {
            binary.extend_from_slice(&0.0f32.to_le_bytes());
            binary.extend_from_slice(&0.0f32.to_le_bytes());
        }
        for index in indices {
            binary.extend_from_slice(&index.to_le_bytes());
        }
        let document = serde_json::json!({
            "buffers": [{"byteLength": binary.len()}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 36},
                {"buffer": 0, "byteOffset": 36, "byteLength": 36},
                {"buffer": 0, "byteOffset": 72, "byteLength": 24},
                {"buffer": 0, "byteOffset": 96, "byteLength": 12},
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
                {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
                {"bufferView": 3, "componentType": 5125, "count": 3, "type": "SCALAR"},
            ],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                "indices": 3,
            }]}],
        });
        let mut json = serde_json::to_vec(&document).unwrap();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let total = 28 + json.len() + binary.len();
        let mut bytes = Vec::with_capacity(total);
        bytes.extend_from_slice(b"glTF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&(total as u32).to_le_bytes());
        bytes.extend_from_slice(&(json.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&0x4e4f_534au32.to_le_bytes());
        bytes.extend_from_slice(&json);
        bytes.extend_from_slice(&(binary.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&0x004e_4942u32.to_le_bytes());
        bytes.extend_from_slice(&binary);
        bytes
    }

    #[test]
    fn glb_indices_are_bounds_checked_against_the_vertex_count() {
        assert!(mesh_from_converter_glb(&triangle_glb([0, 1, 2]), false).is_ok());
        let error = mesh_from_converter_glb(&triangle_glb([0, 1, 9]), false).unwrap_err();
        assert!(matches!(error, ZorahBundleError::Invalid(_)), "{error}");
    }

    #[test]
    fn implausible_length_prefixes_fail_before_allocating() {
        let mut reader = VecReader::new(Vec::new());
        let error = block_on(read_payload(
            &mut reader,
            u64::MAX,
            MAX_ENTRY_BYTES,
            "entry",
        ))
        .expect_err("a length past the bound must not be allocated");
        assert!(matches!(error, ZorahBundleError::Invalid(_)), "{error}");
    }

    #[test]
    fn truncated_payloads_report_io_errors_without_committing_the_allocation() {
        let mut reader = VecReader::new(vec![0; 16]);
        let error = block_on(read_payload(
            &mut reader,
            MAX_ENTRY_BYTES,
            MAX_ENTRY_BYTES,
            "entry",
        ))
        .expect_err("a truncated stream must fail on the first short read");
        assert!(matches!(error, ZorahBundleError::Io(_)), "{error}");
    }
}
