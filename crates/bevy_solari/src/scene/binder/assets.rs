use super::{
    allocator::{RetainedBindingArray, SlotAllocator},
    instances::InstanceState,
    StandardMaterialAssets,
};
use bevy_asset::AssetId;
use bevy_color::{ColorToComponents, LinearRgba};
use bevy_image::Image;
use bevy_math::{Mat3, Vec3, Vec4};
use bevy_pbr::StandardMaterial;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    impl_atomic_pod,
    render_asset::{ExtractedAssets, RenderAssets},
    render_resource::{
        AtomicPod, AtomicSparseBufferVec, BufferUsages, Sampler, TextureFormat, TextureView,
    },
    texture::GpuImage,
};
use bevy_utils::once;
use bytemuck::{Pod, Zeroable};
use core::num::NonZeroU32;
use tracing::{info_span, warn};

pub const MAX_TEXTURE_COUNT: NonZeroU32 = NonZeroU32::new(5_000).unwrap();
const TEXTURE_MAP_NONE: u32 = u32::MAX;

/// The four textures a [`StandardMaterial`] can reference, in [`GpuMaterial`] field order.
type MaterialTextures = [Option<AssetId<Image>>; 4];

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuMaterial {
    normal_map_texture_id: u32,
    base_color_texture_id: u32,
    emissive_texture_id: u32,
    metallic_roughness_texture_id: u32,

    base_color: Vec3,
    perceptual_roughness: f32,
    emissive: Vec3,
    metallic: f32,
    flags: u32,
    _padding_a: u32,
    _padding_b: u32,
    reflectance: f32,
    /// A `mat3x3<f32>` on the shader side: three vec4-aligned columns.
    uv_transform: [Vec4; 3],
}

impl_atomic_pod!(GpuMaterial, GpuMaterialBlob);

/// Matches `MATERIAL_FLAGS_FLIP_NORMAL_MAP_Y` in `bindings.wesl`.
pub const MATERIAL_FLAGS_FLIP_NORMAL_MAP_Y: u32 = 1;
/// Matches `MATERIAL_FLAGS_METALLIC_ROUGHNESS_RG` in `bindings.wesl`.
pub const MATERIAL_FLAGS_METALLIC_ROUGHNESS_RG: u32 = 2;

/// Same decisions as `StandardMaterial::as_bind_group_shader_type`, so a surface shades the same
/// whether a raster pass or a ray resolves it. The caller skips any material whose textures are
/// not all `GpuImage`s yet, so a missing metallic-roughness texture here means there is none.
fn material_flags(material: &StandardMaterial, texture_assets: &RenderAssets<GpuImage>) -> u32 {
    let mut flags = 0;
    if material.flip_normal_map_y {
        flags |= MATERIAL_FLAGS_FLIP_NORMAL_MAP_Y;
    }
    let metallic_roughness_format = material
        .metallic_roughness_texture
        .as_ref()
        .and_then(|handle| texture_assets.get(handle.id()))
        .map(|texture| texture.texture_descriptor.format);
    if matches!(
        metallic_roughness_format,
        Some(
            TextureFormat::Rg8Unorm
                | TextureFormat::Rg16Unorm
                | TextureFormat::Bc5RgUnorm
                | TextureFormat::EacRg11Unorm
        )
    ) {
        flags |= MATERIAL_FLAGS_METALLIC_ROUGHNESS_RG;
    }
    flags
}

/// `uv_transform` as the three vec4-aligned columns of the shader's `mat3x3<f32>`.
fn uv_transform_columns(material: &StandardMaterial) -> [Vec4; 3] {
    let matrix = Mat3::from(material.uv_transform);
    [
        matrix.x_axis.extend(0.0),
        matrix.y_axis.extend(0.0),
        matrix.z_axis.extend(0.0),
    ]
}

/// Stable material and texture slots, plus the retry state for assets that aren't ready yet.
pub struct AssetState {
    pub textures: RetainedBindingArray<AssetId<Image>, (TextureView, Sampler)>,
    pub materials: AtomicSparseBufferVec<GpuMaterial>,
    pub material_slots: SlotAllocator<AssetId<StandardMaterial>>,
    material_textures: HashMap<AssetId<StandardMaterial>, MaterialTextures>,
    pub emissive_materials: HashSet<AssetId<StandardMaterial>>,
    /// Materials to retry because at least one required texture is not on the GPU yet.
    unresolved_materials: HashSet<AssetId<StandardMaterial>>,
    /// Bound images whose replacement GPU data has not landed yet.
    pending_texture_updates: HashSet<AssetId<Image>>,
}

impl AssetState {
    pub fn new() -> Self {
        Self {
            textures: RetainedBindingArray::new(),
            materials: AtomicSparseBufferVec::new(BufferUsages::STORAGE, "solari_materials".into()),
            material_slots: SlotAllocator::new(),
            material_textures: HashMap::default(),
            emissive_materials: HashSet::default(),
            unresolved_materials: HashSet::default(),
            pending_texture_updates: HashSet::default(),
        }
    }

    pub fn update_materials(
        &mut self,
        instances: &mut InstanceState,
        material_assets: &StandardMaterialAssets,
        texture_assets: &RenderAssets<GpuImage>,
    ) {
        let _span = info_span!("update_materials").entered();

        for material_id in &material_assets.removed {
            self.remove_material(*material_id, instances);
        }
        for material_id in &material_assets.changed {
            self.update_material(*material_id, instances, material_assets, texture_assets);
        }
    }

    /// Swaps in uploaded replacements and retries materials waiting on images.
    pub fn update_textures(
        &mut self,
        instances: &mut InstanceState,
        extracted_images: &ExtractedAssets<GpuImage>,
        texture_assets: &RenderAssets<GpuImage>,
        material_assets: &StandardMaterialAssets,
    ) {
        let _span = info_span!("update_textures").entered();

        let mut pending = core::mem::take(&mut self.pending_texture_updates);
        pending.extend(extracted_images.added.iter().copied());

        for image_id in pending {
            if !self.textures.contains(&image_id) {
                continue;
            }

            match texture_assets.get(image_id) {
                Some(image) => self.textures.replace(
                    &image_id,
                    (image.texture_view.clone(), image.sampler.clone()),
                ),
                None => {
                    self.pending_texture_updates.insert(image_id);
                }
            }
        }

        for material_id in core::mem::take(&mut self.unresolved_materials) {
            self.update_material(material_id, instances, material_assets, texture_assets);
        }
    }

    /// Resolves a material's textures and writes it into its stable slot.
    fn update_material(
        &mut self,
        material_id: AssetId<StandardMaterial>,
        instances: &mut InstanceState,
        material_assets: &StandardMaterialAssets,
        texture_assets: &RenderAssets<GpuImage>,
    ) {
        let Some(material) = material_assets.get(&material_id) else {
            self.remove_material(material_id, instances);
            return;
        };

        let was_resolved = self.material_slots.contains(&material_id);
        let handles = [
            &material.normal_map_texture,
            &material.base_color_texture,
            &material.emissive_texture,
            &material.metallic_roughness_texture,
        ];

        // Resolve first so a missing texture leaves no partially acquired slots
        let mut textures: MaterialTextures = [None; 4];
        for (slot, handle) in textures.iter_mut().zip(handles) {
            let Some(handle) = handle else { continue };
            let image_id = handle.id();
            if texture_assets.get(image_id).is_none() {
                self.defer_material(material_id, instances);
                return;
            }
            *slot = Some(image_id);
        }

        if self.new_texture_count(&textures) > self.textures.vacancies(MAX_TEXTURE_COUNT.get()) {
            // At the limit, release the old set once so a replacement can reuse those slots
            self.release_material_textures(material_id);
            self.material_textures.remove(&material_id);

            if self.new_texture_count(&textures) > self.textures.vacancies(MAX_TEXTURE_COUNT.get())
            {
                once!(warn!(
                    "Solari scene needs more than {} textures. Materials past that limit will not \
                     be rendered.",
                    MAX_TEXTURE_COUNT.get()
                ));
                self.defer_material(material_id, instances);
                return;
            }
        }

        let mut texture_ids = [TEXTURE_MAP_NONE; 4];
        for (texture_id, image_id) in texture_ids.iter_mut().zip(textures) {
            let Some(image_id) = image_id else { continue };
            let image = texture_assets.get(image_id).unwrap();
            if let Some(slot) = self
                .textures
                .acquire(image_id, MAX_TEXTURE_COUNT.get(), || {
                    (image.texture_view.clone(), image.sampler.clone())
                })
            {
                *texture_id = slot;
            }
        }

        self.release_material_textures(material_id);
        self.material_textures.insert(material_id, textures);
        self.unresolved_materials.remove(&material_id);

        let slot = self.material_slots.get_or_allocate(material_id);
        let emissive = material.emissive.to_vec3();
        let is_emissive = emissive != Vec3::ZERO;
        self.materials.grow_and_set(
            slot,
            GpuMaterial {
                normal_map_texture_id: texture_ids[0],
                base_color_texture_id: texture_ids[1],
                emissive_texture_id: texture_ids[2],
                metallic_roughness_texture_id: texture_ids[3],
                base_color: LinearRgba::from(material.base_color).to_vec3(),
                perceptual_roughness: material.perceptual_roughness,
                emissive,
                metallic: material.metallic,
                flags: material_flags(material, texture_assets),
                _padding_a: 0,
                _padding_b: 0,
                reflectance: material.reflectance,
                uv_transform: uv_transform_columns(material),
            },
        );

        let was_emissive = if is_emissive {
            !self.emissive_materials.insert(material_id)
        } else {
            self.emissive_materials.remove(&material_id)
        };
        if !was_resolved || was_emissive != is_emissive {
            instances.invalidate_material(material_id);
        }
    }

    fn new_texture_count(&self, textures: &MaterialTextures) -> u32 {
        let mut count = 0;
        for (index, image_id) in textures.iter().enumerate() {
            let Some(image_id) = image_id else { continue };
            let counted_already = textures[..index].contains(&Some(*image_id));
            if !counted_already && !self.textures.contains(image_id) {
                count += 1;
            }
        }
        count
    }

    fn defer_material(
        &mut self,
        material_id: AssetId<StandardMaterial>,
        instances: &mut InstanceState,
    ) {
        self.remove_material(material_id, instances);
        self.unresolved_materials.insert(material_id);
    }

    fn remove_material(
        &mut self,
        material_id: AssetId<StandardMaterial>,
        instances: &mut InstanceState,
    ) {
        self.unresolved_materials.remove(&material_id);
        if self.material_slots.remove(&material_id).is_none() {
            return;
        }

        self.release_material_textures(material_id);
        self.material_textures.remove(&material_id);
        self.emissive_materials.remove(&material_id);
        instances.invalidate_material(material_id);
    }

    fn release_material_textures(&mut self, material_id: AssetId<StandardMaterial>) {
        let Some(textures) = self.material_textures.get(&material_id).copied() else {
            return;
        };
        for image_id in textures.into_iter().flatten() {
            self.textures.release(&image_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MATERIAL_FLAGS_FLIP_NORMAL_MAP_Y, MATERIAL_FLAGS_METALLIC_ROUGHNESS_RG};

    #[test]
    fn material_flags_match_the_shader_constants() {
        let shader = include_str!("../bindings.wesl");
        assert!(shader.contains(&format!(
            "const MATERIAL_FLAGS_FLIP_NORMAL_MAP_Y = {MATERIAL_FLAGS_FLIP_NORMAL_MAP_Y}u;"
        )));
        assert!(shader.contains(&format!(
            "const MATERIAL_FLAGS_METALLIC_ROUGHNESS_RG = {MATERIAL_FLAGS_METALLIC_ROUGHNESS_RG}u;"
        )));
    }
}
