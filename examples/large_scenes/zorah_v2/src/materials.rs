//! The root glTF's materials as `StandardMaterial`s, built on first use.
//!
//! Every material is read into a plain [`MaterialSpec`] up front, but its
//! `StandardMaterial` and the textures behind it are only created when a
//! spawned part first asks for it: 4,418 KTX2 files at 4k each are far more
//! than one view, or a `--limit-meshes` run, ever binds.

use bevy::{
    asset::RenderAssetUsages,
    image::{
        ImageAddressMode, ImageFilterMode, ImageLoaderSettings, ImageSampler,
        ImageSamplerDescriptor,
    },
    math::{ops, Affine2},
    prelude::*,
    render::render_resource::Face,
};

/// glTF's cutoff for a MASK material that gives none (the export gives 0.3333).
const DEFAULT_ALPHA_CUTOFF: f32 = 0.5;

/// `--clay`: a rough 50% grey (0.5 linear reflectance), the conventional
/// lighting-check material.
const CLAY_BASE_COLOR: Color = Color::LinearRgba(LinearRgba::rgb(0.5, 0.5, 0.5));
const CLAY_ROUGHNESS: f32 = 0.9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AlphaSpec {
    Opaque,
    Mask(f32),
    Blend,
}

/// `KHR_texture_transform` as the file gives it; every slot of a material
/// carries the same one in this export.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UvTransform {
    pub offset: [f32; 2],
    pub rotation: f32,
    pub scale: [f32; 2],
}

/// One root glTF material, independent of the asset server.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialSpec {
    pub name: String,
    pub base_color: [f32; 4],
    pub base_color_texture: Option<String>,
    pub metallic: f32,
    pub roughness: f32,
    pub metallic_roughness_texture: Option<String>,
    pub normal_texture: Option<String>,
    pub emissive: [f32; 3],
    pub emissive_texture: Option<String>,
    pub alpha: AlphaSpec,
    pub double_sided: bool,
    pub unlit: bool,
    pub uv_transform: Option<UvTransform>,
    /// `KHR_materials_specular` factor and colour, when the extension is present.
    pub specular: Option<(f32, [f32; 3])>,
    /// `KHR_materials_transmission` factor; `0` when absent.
    pub transmission: f32,
    pub ior: Option<f32>,
}

impl MaterialSpec {
    pub fn is_transmissive(&self) -> bool {
        self.transmission > 0.0
    }

    pub fn is_emissive(&self) -> bool {
        self.emissive.iter().any(|channel| *channel > 0.0)
    }
}

/// Reads every material of `document`; texture paths are the image URIs,
/// relative to the root glTF's directory.
pub fn read_materials(document: &gltf::Document) -> Vec<MaterialSpec> {
    document
        .materials()
        .filter(|material| material.index().is_some())
        .map(|material| read_material(&material))
        .collect()
}

fn read_material(material: &gltf::Material) -> MaterialSpec {
    let pbr = material.pbr_metallic_roughness();
    let texture_uri = |info: &gltf::texture::Info| -> Option<String> {
        match info.texture().source().source() {
            gltf::image::Source::Uri { uri, .. } => Some(uri.to_string()),
            gltf::image::Source::View { .. } => None,
        }
    };
    let base_color_texture = pbr.base_color_texture();
    // `StandardMaterial` has one transform for every slot, so the first slot
    // carrying one wins (the gltf crate exposes none on the normal slot);
    // this export gives every slot the same one anyway.
    let uv_transform = base_color_texture
        .as_ref()
        .and_then(gltf::texture::Info::texture_transform)
        .or_else(|| {
            pbr.metallic_roughness_texture()
                .as_ref()
                .and_then(gltf::texture::Info::texture_transform)
        })
        .or_else(|| {
            material
                .emissive_texture()
                .as_ref()
                .and_then(gltf::texture::Info::texture_transform)
        })
        .map(|transform| UvTransform {
            offset: transform.offset(),
            rotation: transform.rotation(),
            scale: transform.scale(),
        });
    MaterialSpec {
        name: material.name().unwrap_or_default().to_string(),
        base_color: pbr.base_color_factor(),
        base_color_texture: base_color_texture.as_ref().and_then(texture_uri),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        metallic_roughness_texture: pbr
            .metallic_roughness_texture()
            .as_ref()
            .and_then(texture_uri),
        normal_texture: material.normal_texture().and_then(|normal| {
            match normal.texture().source().source() {
                gltf::image::Source::Uri { uri, .. } => Some(uri.to_string()),
                gltf::image::Source::View { .. } => None,
            }
        }),
        emissive: material.emissive_factor(),
        emissive_texture: material.emissive_texture().as_ref().and_then(texture_uri),
        alpha: match material.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaSpec::Opaque,
            gltf::material::AlphaMode::Mask => {
                AlphaSpec::Mask(material.alpha_cutoff().unwrap_or(DEFAULT_ALPHA_CUTOFF))
            }
            gltf::material::AlphaMode::Blend => AlphaSpec::Blend,
        },
        double_sided: material.double_sided(),
        unlit: material.unlit(),
        uv_transform,
        specular: material
            .specular()
            .map(|specular| (specular.specular_factor(), specular.specular_color_factor())),
        transmission: material
            .transmission()
            .map_or(0.0, |transmission| transmission.transmission_factor()),
        ior: material.ior(),
    }
}

/// The flags that shape every material.
#[derive(Clone, Debug)]
pub struct MaterialOptions {
    /// `--max-texture-size`; `None` keeps the full 4k mips.
    pub max_texture_size: Option<u32>,
    pub preserve_alpha: bool,
    pub double_sided_all: bool,
    pub gltf_specular: bool,
    pub emissive_boost: f32,
    pub clay: bool,
    /// `--solari-albedo`: the emission scale that maps base colour back to unit output.
    pub albedo_emission_scale: Option<f32>,
}

/// What the spawner needs to know about a material's handle.
#[derive(Clone)]
pub struct MaterialSlot {
    pub handle: Handle<StandardMaterial>,
    /// Alpha-tested or blended: needs `Mesh3d`, since meshlets shade opaque only.
    pub mesh_raster: bool,
    /// Stained glass: left out of the BLAS unless `--glass-in-blas`.
    pub transmissive: bool,
}

/// Materials by root glTF index, created on first request.
#[derive(Resource)]
pub struct MaterialCache {
    specs: Vec<MaterialSpec>,
    slots: Vec<Option<MaterialSlot>>,
    fallback: Option<MaterialSlot>,
    options: MaterialOptions,
    pub created: usize,
    pub textures_requested: usize,
}

impl MaterialCache {
    pub fn new(specs: Vec<MaterialSpec>, options: MaterialOptions) -> Self {
        let slots = vec![None; specs.len()];
        Self {
            specs,
            slots,
            fallback: None,
            options,
            created: 0,
            textures_requested: 0,
        }
    }

    /// The slot for a primitive's material index; a primitive without one, or
    /// with an index the file lacks, gets a neutral grey.
    pub fn get(
        &mut self,
        index: Option<usize>,
        asset_server: &AssetServer,
        materials: &mut Assets<StandardMaterial>,
    ) -> MaterialSlot {
        if let Some(index) = index.filter(|index| *index < self.specs.len()) {
            if let Some(slot) = &self.slots[index] {
                return slot.clone();
            }
            let spec = self.specs[index].clone();
            let material = self.build(&spec, asset_server);
            let slot = MaterialSlot {
                handle: materials.add(material),
                mesh_raster: self.options.preserve_alpha && spec.alpha != AlphaSpec::Opaque,
                transmissive: spec.is_transmissive(),
            };
            self.slots[index] = Some(slot.clone());
            self.created += 1;
            return slot;
        }
        if let Some(slot) = &self.fallback {
            return slot.clone();
        }
        let mut material = StandardMaterial {
            base_color: Color::srgb(0.55, 0.52, 0.48),
            perceptual_roughness: 0.65,
            ..default()
        };
        self.finish(&mut material);
        let slot = MaterialSlot {
            handle: materials.add(material),
            mesh_raster: false,
            transmissive: false,
        };
        self.fallback = Some(slot.clone());
        slot
    }

    fn build(&mut self, spec: &MaterialSpec, asset_server: &AssetServer) -> StandardMaterial {
        let options = self.options.clone();
        let mut load = |uri: &Option<String>, srgb: bool| -> Option<Handle<Image>> {
            let uri = uri.as_ref()?;
            self.textures_requested += 1;
            Some(load_texture(
                asset_server,
                uri,
                srgb,
                options.max_texture_size,
            ))
        };
        let textured = !options.clay;
        let mut material = StandardMaterial {
            base_color: if options.clay {
                CLAY_BASE_COLOR
            } else {
                Color::linear_rgba(
                    spec.base_color[0],
                    spec.base_color[1],
                    spec.base_color[2],
                    spec.base_color[3],
                )
            },
            base_color_texture: if textured {
                load(&spec.base_color_texture, true)
            } else {
                None
            },
            metallic: if options.clay { 0.0 } else { spec.metallic },
            perceptual_roughness: if options.clay {
                CLAY_ROUGHNESS
            } else {
                spec.roughness
            },
            // Two-channel BC5 (roughness R, metallic G) is detected from the
            // image format when the material binds.
            metallic_roughness_texture: if textured {
                load(&spec.metallic_roughness_texture, false)
            } else {
                None
            },
            normal_map_texture: load(&spec.normal_texture, false),
            emissive: LinearRgba::rgb(
                spec.emissive[0] * options.emissive_boost,
                spec.emissive[1] * options.emissive_boost,
                spec.emissive[2] * options.emissive_boost,
            ),
            emissive_texture: if spec.is_emissive() {
                load(&spec.emissive_texture, true)
            } else {
                None
            },
            alpha_mode: match spec.alpha {
                // Meshlets shade opaque only; without `--preserve-alpha` the
                // cutouts render as their full quads.
                AlphaSpec::Mask(cutoff) if options.preserve_alpha => AlphaMode::Mask(cutoff),
                AlphaSpec::Blend if options.preserve_alpha => AlphaMode::Blend,
                _ => AlphaMode::Opaque,
            },
            double_sided: spec.double_sided || options.double_sided_all,
            cull_mode: if spec.double_sided || options.double_sided_all {
                None
            } else {
                Some(Face::Back)
            },
            unlit: spec.unlit,
            uv_transform: spec.uv_transform.map_or(Affine2::IDENTITY, uv_affine),
            // Transmission is recorded for the BLAS decision only: a
            // transmissive `StandardMaterial` leaves the opaque pass, which
            // meshlets do not support, so the glass rasterizes opaque.
            ior: spec.ior.unwrap_or(1.5),
            ..default()
        };
        if options.gltf_specular
            && let Some((factor, color)) = spec.specular
        {
            // glTF's specularFactor scales F0 linearly; bevy's reflectance
            // maps 0.5 to the same 4%, as bevy_gltf does.
            material.reflectance = factor * 0.5;
            material.specular_tint = Color::linear_rgb(color[0], color[1], color[2]);
        }
        self.finish(&mut material);
        material
    }

    fn finish(&self, material: &mut StandardMaterial) {
        if let Some(scale) = self.options.albedo_emission_scale {
            emit_base_color(material, scale);
        }
    }
}

fn uv_affine(transform: UvTransform) -> Affine2 {
    // The same mapping bevy_gltf applies: glTF rotates UVs clockwise.
    Affine2::from_scale_angle_translation(
        transform.scale.into(),
        -transform.rotation,
        transform.offset.into(),
    )
}

/// Loads one KTX2 with the largest mips dropped at `max_dimension`, a
/// repeating anisotropic sampler, and no CPU copy kept.
fn load_texture(
    asset_server: &AssetServer,
    uri: &str,
    srgb: bool,
    max_dimension: Option<u32>,
) -> Handle<Image> {
    asset_server
        .load_builder()
        .with_settings::<ImageLoaderSettings>(move |settings| {
            settings.is_srgb = srgb;
            settings.max_dimension = max_dimension;
            settings.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                address_mode_u: ImageAddressMode::Repeat,
                address_mode_v: ImageAddressMode::Repeat,
                address_mode_w: ImageAddressMode::Repeat,
                mag_filter: ImageFilterMode::Linear,
                min_filter: ImageFilterMode::Linear,
                mipmap_filter: ImageFilterMode::Linear,
                anisotropy_clamp: 16,
                ..default()
            });
            settings.asset_usage = RenderAssetUsages::RENDER_WORLD;
        })
        .load::<Image>(uri.to_string())
}

/// `--solari-albedo`: the surface emits its base colour and reflects nothing,
/// so the traced image shows the textures themselves.
pub fn emit_base_color(material: &mut StandardMaterial, scale: f32) {
    let tint = material.base_color.to_linear();
    material.emissive = LinearRgba::rgb(tint.red * scale, tint.green * scale, tint.blue * scale);
    material.emissive_texture = material.base_color_texture.take();
    // Solari applies the camera exposure to emission unconditionally; a full
    // weight makes the raster preview agree with it.
    material.emissive_exposure_weight = 1.0;
    material.base_color = Color::BLACK;
    material.normal_map_texture = None;
    material.metallic_roughness_texture = None;
    material.occlusion_texture = None;
    material.metallic = 0.0;
    material.perceptual_roughness = 1.0;
    material.unlit = false;
}

/// The radiance an emitter needs so that the camera's exposure maps it back to
/// unit output: the inverse of Bevy's `Exposure::exposure`.
pub fn albedo_emission_scale(ev100: f32) -> f32 {
    1.2 * ops::exp2(ev100)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_extensions_from_a_minimal_document() {
        let json = r#"{
            "asset": {"version": "2.0"},
            "images": [{"uri": "textures/a_BaseColor.ktx2"}, {"uri": "textures/a_Normal.ktx2"}],
            "textures": [{"source": 0}, {"source": 1}],
            "materials": [{
                "name": "glass",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.5, 0.6, 0.7, 1.0],
                    "baseColorTexture": {"index": 0, "extensions": {"KHR_texture_transform": {"offset": [0.25, 0.0], "scale": [2.0, 2.0]}}},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.3
                },
                "normalTexture": {"index": 1},
                "emissiveFactor": [1.0, 1.0, 1.0],
                "alphaMode": "MASK",
                "alphaCutoff": 0.3333,
                "doubleSided": true,
                "extensions": {
                    "KHR_materials_transmission": {"transmissionFactor": 1.0},
                    "KHR_materials_ior": {"ior": 1.5},
                    "KHR_materials_specular": {"specularFactor": 0.498}
                }
            }]
        }"#;
        let document = gltf::Gltf::from_slice_without_validation(json.as_bytes())
            .unwrap()
            .document;
        let materials = read_materials(&document);
        assert_eq!(materials.len(), 1);
        let glass = &materials[0];
        assert_eq!(glass.name, "glass");
        assert_eq!(
            glass.base_color_texture.as_deref(),
            Some("textures/a_BaseColor.ktx2")
        );
        assert_eq!(
            glass.normal_texture.as_deref(),
            Some("textures/a_Normal.ktx2")
        );
        assert_eq!(glass.alpha, AlphaSpec::Mask(0.3333));
        assert!(glass.double_sided);
        assert!(glass.is_transmissive());
        assert!(glass.is_emissive());
        assert_eq!(glass.ior, Some(1.5));
        assert_eq!(glass.specular.map(|(factor, _)| factor), Some(0.498));
        let transform = glass.uv_transform.unwrap();
        assert_eq!(transform.offset, [0.25, 0.0]);
        assert_eq!(transform.scale, [2.0, 2.0]);
        let affine = uv_affine(transform);
        assert_eq!(
            affine.transform_point2(Vec2::new(0.5, 0.5)),
            Vec2::new(1.25, 1.0)
        );
    }
}
