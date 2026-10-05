use bevy_asset::{AssetPath, Handle};
use bevy_image::Image;

use gltf::Material;

#[cfg(feature = "pbr_specular_textures")]
use {crate::loader::gltf_ext::material::uv_channel, bevy_mesh::UvChannel};

/// Parsed data from the `KHR_materials_specular` extension.
///
/// The extension defines a dielectric F0 linear in its factors,
/// `F0 = ((ior - 1) / (ior + 1))^2 * specularColorFactor * specularFactor`,
/// while Bevy derives it quadratically,
/// `F0 = 0.16 * (reflectance * specular_tint)^2`. [`Self::reflectance`] and
/// [`Self::specular_tint`] take the square root of each term so the two agree.
/// The shader does the same for the specular and specular color textures.
///
/// See the specification:
/// <https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_specular/README.md>
pub(crate) struct SpecularExtension {
    pub(crate) specular_factor: f32,
    #[cfg(feature = "pbr_specular_textures")]
    pub(crate) specular_channel: UvChannel,
    #[cfg(feature = "pbr_specular_textures")]
    pub(crate) specular_texture: Option<Handle<Image>>,
    pub(crate) specular_color_factor: [f32; 3],
    #[cfg(feature = "pbr_specular_textures")]
    pub(crate) specular_color_channel: UvChannel,
    #[cfg(feature = "pbr_specular_textures")]
    pub(crate) specular_color_texture: Option<Handle<Image>>,
}

impl Default for SpecularExtension {
    fn default() -> Self {
        Self {
            specular_factor: 1.0,
            #[cfg(feature = "pbr_specular_textures")]
            specular_channel: UvChannel::default(),
            #[cfg(feature = "pbr_specular_textures")]
            specular_texture: None,
            specular_color_factor: [1.0, 1.0, 1.0],
            #[cfg(feature = "pbr_specular_textures")]
            specular_color_channel: UvChannel::default(),
            #[cfg(feature = "pbr_specular_textures")]
            specular_color_texture: None,
        }
    }
}

impl SpecularExtension {
    /// The [`StandardMaterial::reflectance`] whose F0 equals the F0 of `ior`
    /// scaled by `specularFactor`.
    ///
    /// [`StandardMaterial::reflectance`]: https://docs.rs/bevy/latest/bevy/pbr/struct.StandardMaterial.html#structfield.reflectance
    pub(crate) fn reflectance(&self, ior: f32) -> f32 {
        // glTF allows an IOR of 0 or at least 1; treat anything else as the
        // default instead of letting it reach F0.
        let ior = if ior == 0.0 || ior >= 1.0 { ior } else { 1.5 };
        // sqrt(((ior - 1) / (ior + 1))^2 / 0.16) = 2.5 * |ior - 1| / (ior + 1)
        2.5 * ((ior - 1.0) / (ior + 1.0)).abs() * self.specular_factor.max(0.0).sqrt()
    }

    /// The linear [`StandardMaterial::specular_tint`] whose square scales F0
    /// by `specularColorFactor`, per channel.
    ///
    /// [`StandardMaterial::specular_tint`]: https://docs.rs/bevy/latest/bevy/pbr/struct.StandardMaterial.html#structfield.specular_tint
    pub(crate) fn specular_tint(&self) -> [f32; 3] {
        self.specular_color_factor.map(|c| c.max(0.0).sqrt())
    }

    #[expect(
        clippy::allow_attributes,
        reason = "`unused_variables` is not always linted"
    )]
    #[allow(
        unused_variables,
        reason = "Depending on what features are used to compile this crate, certain parameters may end up unused."
    )]
    pub(crate) fn parse(
        material: &Material,
        textures: &[Handle<Image>],
        asset_path: AssetPath<'_>,
    ) -> Option<Self> {
        let specular = material.specular()?;

        #[cfg(feature = "pbr_specular_textures")]
        let _specular_channel = specular
            .specular_texture()
            .map(|info| uv_channel(material, "specular", info.tex_coord()))
            .unwrap_or_default();
        #[cfg(feature = "pbr_specular_textures")]
        let _specular_texture = specular.specular_texture().map(|info| {
            textures
                .get(info.texture().index())
                .cloned()
                .unwrap_or_default()
        });

        #[cfg(feature = "pbr_specular_textures")]
        let _specular_color_channel = specular
            .specular_color_texture()
            .map(|info| uv_channel(material, "specular color", info.tex_coord()))
            .unwrap_or_default();
        #[cfg(feature = "pbr_specular_textures")]
        let _specular_color_texture = specular.specular_color_texture().map(|info| {
            textures
                .get(info.texture().index())
                .cloned()
                .unwrap_or_default()
        });

        Some(SpecularExtension {
            specular_factor: specular.specular_factor(),
            #[cfg(feature = "pbr_specular_textures")]
            specular_channel: _specular_channel,
            #[cfg(feature = "pbr_specular_textures")]
            specular_texture: _specular_texture,
            specular_color_factor: specular.specular_color_factor(),
            #[cfg(feature = "pbr_specular_textures")]
            specular_color_channel: _specular_color_channel,
            #[cfg(feature = "pbr_specular_textures")]
            specular_color_texture: _specular_color_texture,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::SpecularExtension;

    #[test]
    fn f0_matches_the_extension() {
        let mut ext = SpecularExtension::default();
        for ior in [0.0_f32, 1.0, 1.33, 1.45, 1.5, 2.42] {
            let r = (ior - 1.0) / (ior + 1.0);
            let ior_f0 = r * r;
            for factor in [0.0, 0.1, 0.2, 0.5, 1.0] {
                for color in [0.0, 0.25, 1.0, 2.0] {
                    // The extension clamps this product to 1; the loader doesn't.
                    if ior_f0 * color > 1.0 {
                        continue;
                    }
                    ext.specular_factor = factor;
                    ext.specular_color_factor = [color; 3];
                    // Bevy's dielectric F0, from `calculate_F0_dielectric`.
                    let x = ext.reflectance(ior) * ext.specular_tint()[0];
                    let f0 = 0.16 * x * x;
                    let expected = ior_f0 * color * factor;
                    assert!(
                        (f0 - expected).abs() < 1e-5,
                        "ior {ior}, specularFactor {factor}, specularColorFactor {color}: F0 {f0}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn defaults_keep_bevy_default_reflectance() {
        assert_eq!(SpecularExtension::default().reflectance(1.5), 0.5);
    }

    #[test]
    fn invalid_ior_uses_the_default() {
        let ext = SpecularExtension::default();
        for ior in [-1.0, -0.5, 0.5, f32::NAN] {
            assert_eq!(ext.reflectance(ior), 0.5, "ior {ior}");
        }
    }
}
