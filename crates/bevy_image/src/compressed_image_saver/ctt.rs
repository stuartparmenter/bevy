use bevy_asset::{io::Writer, saver::SavedAsset, AssetPath, AsyncWriteExt};
use ctt::{
    convert, ColorSpace, Container, ConvertSettings, FormatDesc, ImageRef, MipmapFilter,
    PipelineOutput, Quality, SurfaceRef, TextureKind,
};

use super::{
    alpha_coverage::alpha_tested_mips,
    ctt_helpers::{bevy_to_ctt_alpha_mode, choose_ctt_compressed_format, ctt_format},
    CompressedImageSaverError, CompressedImageSaverSettings,
};
use crate::{Image, ImageFormat, ImageFormatSetting, ImageLoaderSettings};
use wgpu_types::TextureFormat;

#[derive(Default)]
pub struct CompressedImageSaverCtt;

impl CompressedImageSaverCtt {
    pub async fn save(
        &self,
        writer: &mut Writer,
        image: SavedAsset<'_, '_, Image>,
        settings: &CompressedImageSaverSettings,
        _asset_path: AssetPath<'_>,
    ) -> Result<ImageLoaderSettings, CompressedImageSaverError> {
        let Some(ref data) = image.data else {
            return Err(CompressedImageSaverError::UninitializedImage);
        };

        if image.texture_descriptor.mip_level_count != 1 {
            return Err(CompressedImageSaverError::CompressionFailed(
                "Expected texture_descriptor.mip_level_count to be 1".into(),
            ));
        }

        let is_srgb = image.texture_descriptor.format.is_srgb();
        let color_space = if is_srgb {
            ColorSpace::Srgb
        } else {
            ColorSpace::Linear
        };

        let input_format = ctt_format(image.texture_descriptor.format)?;
        let output_format = choose_ctt_compressed_format(
            image.texture_descriptor.format,
            color_space,
            settings.is_normal_map,
        )?;

        let is_cubemap = matches!(
            image.texture_view_descriptor,
            Some(wgpu_types::TextureViewDescriptor {
                dimension: Some(wgpu_types::TextureViewDimension::Cube),
                ..
            })
        );

        let bytes_per_pixel =
            crate::TextureFormatPixelInfo::pixel_size(&image.texture_descriptor.format).map_err(
                |_| CompressedImageSaverError::UnsupportedFormat(image.texture_descriptor.format),
            )? as u32;

        if settings.alpha_test_cutoff.is_some() {
            if !settings.generate_mipmaps {
                return Err(CompressedImageSaverError::InvalidSettings(
                    "alpha_test_cutoff needs generate_mipmaps",
                ));
            }
            if !matches!(
                image.texture_descriptor.format,
                TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
            ) {
                return Err(CompressedImageSaverError::InvalidSettings(
                    "alpha_test_cutoff needs an Rgba8Unorm or Rgba8UnormSrgb input",
                ));
            }
        }
        let layers = data.chunks_exact((image.width() * image.height() * bytes_per_pixel) as usize);
        // ctt borrows its surfaces, so alpha-tested chains are built up front.
        let alpha_tested_chains = settings.alpha_test_cutoff.map(|cutoff| {
            layers
                .clone()
                .map(|layer_data| {
                    alpha_tested_mips(layer_data, image.width(), image.height(), is_srgb, cutoff)
                })
                .collect::<Vec<_>>()
        });
        let surface = |data, width: u32, height: u32| SurfaceRef {
            data,
            width,
            height,
            depth: 1,
            stride: width * bytes_per_pixel,
            slice_stride: 0,
        };
        // Each layer's mip chain: the base alone for ctt to complete, or all of
        // an alpha-tested chain built here.
        let surfaces = match &alpha_tested_chains {
            Some(chains) => chains
                .iter()
                .map(|chain| {
                    chain
                        .iter()
                        .map(|level| surface(&level.texels, level.width, level.height))
                        .collect()
                })
                .collect(),
            None => layers
                .map(|layer_data| vec![surface(layer_data, image.width(), image.height())])
                .collect(),
        };
        let ctt_image = ImageRef {
            surfaces,
            kind: if is_cubemap {
                TextureKind::Cubemap
            } else {
                TextureKind::Texture2D
            },
            desc: FormatDesc {
                format: input_format,
                color_space,
                alpha: bevy_to_ctt_alpha_mode(settings.input_alpha_mode),
            },
        };

        let settings = ConvertSettings {
            format: Some(output_format),
            container: Container::ktx2_zstd(0),
            quality: Quality::default(),
            output_color_space: None,
            output_alpha: Some(bevy_to_ctt_alpha_mode(settings.output_alpha_mode)),
            allow_discarding_alpha: settings.is_normal_map,
            swizzle: None,
            // An alpha-tested chain arrives complete.
            mipmap: settings.generate_mipmaps && settings.alpha_test_cutoff.is_none(),
            mipmap_count: None,
            mipmap_filter: if settings.is_normal_map {
                MipmapFilter::Triangle
            } else {
                MipmapFilter::Lanczos3
            },
        };

        let output = convert(ctt_image, settings)
            .map_err(|e| CompressedImageSaverError::CompressionFailed(Box::new(e)))?;
        let PipelineOutput::Encoded(compressed_bytes) = &output else {
            return Err(CompressedImageSaverError::CompressionFailed(
                "Expected encoded output from ctt".into(),
            ));
        };

        writer.write_all(compressed_bytes).await?;

        Ok(ImageLoaderSettings {
            format: ImageFormatSetting::Format(ImageFormat::Ktx2),
            is_srgb,
            sampler: image.sampler.clone(),
            asset_usage: image.asset_usage,
            ..Default::default()
        })
    }
}
