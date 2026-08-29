#[cfg(any(feature = "flate2", feature = "zstd_rust"))]
use std::io::Read;

#[cfg(feature = "basis-universal")]
use basis_universal::{
    DecodeFlags, LowLevelUastcTranscoder, SliceParametersUastc, TranscoderBlockFormat,
};
use bevy_color::Srgba;
use bevy_utils::default;
#[cfg(any(feature = "flate2", feature = "zstd_rust", feature = "zstd_c"))]
use ktx2::SupercompressionScheme;
use ktx2::{
    dfd::{Basic, Block, ChannelTypeQualifiers, SampleInformation},
    ColorModel, Header,
};
use wgpu_types::{
    AstcBlock, AstcChannel, Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor,
    TextureViewDimension,
};

use super::{
    CompressedImageFormats, Image, SourceColorPrimaries, TextureChannelLayout, TextureError,
    TranscodeFormat,
};
use {bevy_utils::once, tracing::warn};

/// Converts KTX2 bytes to a bevy [`Image`] using the given compressed format support.
///
/// # Errors
///
/// Returns an error if the provided buffer contained invalid data, decompression fails, or transcoding
/// of unsupported data formats fails.
///
/// `max_dimension` drops leading mip levels as described on
/// [`ImageLoaderSettings::max_dimension`](crate::ImageLoaderSettings::max_dimension).
/// Dropped levels are never decompressed or transcoded, so a supercompressed file
/// costs only the levels that are kept.
#[cfg(feature = "ktx2")]
pub fn ktx2_buffer_to_image(
    buffer: &[u8],
    supported_compressed_formats: CompressedImageFormats,
    is_srgb: bool,
    max_dimension: Option<u32>,
) -> Result<Image, TextureError> {
    let ktx2 = ktx2::Reader::new(buffer)
        .map_err(|err| TextureError::InvalidData(format!("Failed to parse ktx2 file: {err:?}")))?;
    let Header {
        pixel_width: width,
        pixel_height: height,
        pixel_depth: depth,
        layer_count,
        face_count,
        level_count,
        supercompression_scheme,
        ..
    } = ktx2.header();
    let layer_count = layer_count.max(1);
    let face_count = face_count.max(1);
    let depth = depth.max(1);

    // Identify the format. Transcoding waits until the kept levels are in hand,
    // but the block size is needed now to place the cut.
    let texture_format = ktx2_get_texture_format(&ktx2, is_srgb);

    // Every dimension halves per level and each level holds all of its layers,
    // faces and slices, so cutting the chain is just starting at a later level with
    // the header's extent shifted to match. Everything below then sees the kept
    // chain as if the file had been authored at that size, except the texture's
    // kind: a 2D strip whose height collapses to 1 or a 3D texture whose depth does
    // must stay 2D or 3D, or the cap would change the binding type a shader sees.
    let skip = crate::image::mip_levels_to_skip(
        width,
        height,
        level_count,
        ktx2_block_dimensions(&texture_format),
        max_dimension,
    );
    let is_3d = depth > 1;
    let is_2d = height > 1;
    let width = (width >> skip).max(1);
    let height = (height >> skip).max(1);
    let depth = (depth >> skip).max(1);
    let level_count = level_count - skip;
    let kept_levels = || ktx2.levels().enumerate().skip(skip as usize);

    // Handle supercompression
    let mut levels: Vec<Vec<u8>>;
    if let Some(supercompression_scheme) = supercompression_scheme {
        match supercompression_scheme {
            #[cfg(feature = "flate2")]
            SupercompressionScheme::ZLIB => {
                levels = Vec::with_capacity(level_count as usize);
                for (level_index, level) in kept_levels() {
                    let mut decoder = flate2::bufread::ZlibDecoder::new(level.data);
                    let mut decompressed = Vec::new();
                    decoder.read_to_end(&mut decompressed).map_err(|err| {
                        TextureError::SuperDecompressionError(format!(
                            "Failed to decompress {supercompression_scheme:?} for mip {level_index}: {err:?}",
                        ))
                    })?;
                    levels.push(decompressed);
                }
            }
            #[cfg(all(feature = "zstd_rust", not(feature = "zstd_c")))]
            SupercompressionScheme::Zstandard => {
                levels = Vec::with_capacity(level_count as usize);
                for (level_index, level) in kept_levels() {
                    let mut cursor = std::io::Cursor::new(level.data);
                    let mut decoder = ruzstd::decoding::StreamingDecoder::new(&mut cursor)
                        .map_err(|err| TextureError::SuperDecompressionError(err.to_string()))?;
                    let mut decompressed = Vec::new();
                    decoder.read_to_end(&mut decompressed).map_err(|err| {
                        TextureError::SuperDecompressionError(format!(
                            "Failed to decompress {supercompression_scheme:?} for mip {level_index}: {err:?}",
                        ))
                    })?;
                    levels.push(decompressed);
                }
            }
            #[cfg(feature = "zstd_c")]
            SupercompressionScheme::Zstandard => {
                levels = Vec::with_capacity(level_count as usize);
                for (level_index, level) in kept_levels() {
                    levels.push(zstd::decode_all(level.data).map_err(|err| {
                        TextureError::SuperDecompressionError(format!(
                            "Failed to decompress {supercompression_scheme:?} for mip {level_index}: {err:?}",
                        ))
                    })?);
                }
            }
            _ => {
                return Err(TextureError::SuperDecompressionError(format!(
                    "Unsupported supercompression scheme: {supercompression_scheme:?}",
                )));
            }
        }
    } else {
        levels = kept_levels()
            .map(|(_, level)| level.data.to_vec())
            .collect();
    }

    // Tracks whether data assumed to be sRGB-encoded was decoded to linear on the CPU
    // during transcoding; in that case a linear output format does not contradict an
    // sRGB transfer function declared by the file, but it does override a linear one.
    let mut srgb_data_linearized_on_cpu = false;

    let texture_format = texture_format.or_else(|error| match error {
        // Transcode if needed and supported
        TextureError::FormatRequiresTranscodingError(transcode_format) => {
            let mut transcoded = vec![Vec::default(); levels.len()];
            let texture_format = match transcode_format {
                TranscodeFormat::R8UnormSrgb => {
                    srgb_data_linearized_on_cpu = true;
                    let (mut original_width, mut original_height) = (width, height);

                    for (level, level_data) in levels.iter().enumerate() {
                        transcoded[level] = level_data
                            .iter()
                            .copied()
                            .map(|v| (Srgba::gamma_function(v as f32 / 255.) * 255.).floor() as u8)
                            .collect::<Vec<u8>>();

                        // Next mip dimensions are half the current, minimum 1x1
                        original_width = (original_width / 2).max(1);
                        original_height = (original_height / 2).max(1);
                    }

                    TextureFormat::R8Unorm
                }
                TranscodeFormat::Rg8UnormSrgb => {
                    srgb_data_linearized_on_cpu = true;
                    let (mut original_width, mut original_height) = (width, height);

                    for (level, level_data) in levels.iter().enumerate() {
                        transcoded[level] = level_data
                            .iter()
                            .copied()
                            .map(|v| (Srgba::gamma_function(v as f32 / 255.) * 255.).floor() as u8)
                            .collect::<Vec<u8>>();

                        // Next mip dimensions are half the current, minimum 1x1
                        original_width = (original_width / 2).max(1);
                        original_height = (original_height / 2).max(1);
                    }

                    TextureFormat::Rg8Unorm
                }
                TranscodeFormat::Rgb8 => {
                    let mut rgba = vec![255u8; width as usize * height as usize * 4];
                    for (level, level_data) in levels.iter().enumerate() {
                        let n_pixels = (width as usize >> level).max(1) * (height as usize >> level).max(1);

                        let mut offset = 0;
                        for _layer in 0..layer_count {
                            for _face in 0..face_count {
                                for i in 0..n_pixels {
                                    rgba[i * 4] = level_data[offset];
                                    rgba[i * 4 + 1] = level_data[offset + 1];
                                    rgba[i * 4 + 2] = level_data[offset + 2];
                                    offset += 3;
                                }
                                transcoded[level].extend_from_slice(&rgba[0..n_pixels * 4]);
                            }
                        }
                    }

                    if is_srgb {
                        TextureFormat::Rgba8UnormSrgb
                    } else {
                        TextureFormat::Rgba8Unorm
                    }
                }
                #[cfg(feature = "basis-universal")]
                TranscodeFormat::Uastc(data_format) => {
                    let (transcode_block_format, texture_format) =
                        get_transcoded_formats(supported_compressed_formats, data_format, is_srgb);
                    let texture_format_info = texture_format;
                    let (block_width_pixels, block_height_pixels) = (
                        texture_format_info.block_dimensions().0,
                        texture_format_info.block_dimensions().1,
                    );
                    // Texture is not a depth or stencil format, it is possible to pass `None` and unwrap
                    let block_bytes = texture_format_info.block_copy_size(None).unwrap();

                    let transcoder = LowLevelUastcTranscoder::new();
                    for (level, level_data) in levels.iter().enumerate() {
                        let (level_width, level_height) = (
                            (width >> level as u32).max(1),
                            (height >> level as u32).max(1),
                        );
                        let (num_blocks_x, num_blocks_y) = (
                            level_width.div_ceil(block_width_pixels) .max(1),
                            level_height.div_ceil(block_height_pixels) .max(1),
                        );
                        let level_bytes = (num_blocks_x * num_blocks_y * block_bytes) as usize;

                        let mut offset = 0;
                        for _layer in 0..layer_count {
                            for _face in 0..face_count {
                                // NOTE: SliceParametersUastc does not implement Clone nor Copy so
                                // it has to be created per use
                                let slice_parameters = SliceParametersUastc {
                                    num_blocks_x,
                                    num_blocks_y,
                                    has_alpha: false,
                                    original_width: level_width,
                                    original_height: level_height,
                                };
                                transcoder
                                    .transcode_slice(
                                        &level_data[offset..(offset + level_bytes)],
                                        slice_parameters,
                                        DecodeFlags::HIGH_QUALITY,
                                        transcode_block_format,
                                    )
                                    .map(|mut transcoded_level| transcoded[level].append(&mut transcoded_level))
                                    .map_err(|error| {
                                        TextureError::SuperDecompressionError(format!(
                                            "Failed to transcode mip level {level} from UASTC to {transcode_block_format:?}: {error:?}",
                                        ))
                                    })?;
                                offset += level_bytes;
                            }
                        }
                    }
                    texture_format
                }
                // ETC1S is a subset of ETC1 which is a subset of ETC2
                // TODO: Implement transcoding
                TranscodeFormat::Etc1s => {
                    let texture_format = if is_srgb {
                        TextureFormat::Etc2Rgb8UnormSrgb
                    } else {
                        TextureFormat::Etc2Rgb8Unorm
                    };
                    if !supported_compressed_formats.supports(texture_format) {
                        return Err(error);
                    }
                    transcoded = levels.to_vec();
                    texture_format
                }
                #[cfg(not(feature = "basis-universal"))]
                _ => return Err(error),
            };
            levels = transcoded;
            Ok(texture_format)
        }
        _ => Err(error),
    })?;
    if !supported_compressed_formats.supports(texture_format) {
        return Err(TextureError::UnsupportedTextureFormat(format!(
            "Format not supported by this GPU: {texture_format:?}",
        )));
    }

    // Honor the color metadata in the file's data format descriptor. This is
    // metadata-only: the file's color primaries are stamped on the image (an explicit
    // loader setting still takes priority over the stamp, applied by the caller), and
    // transfer-function mismatches warn without changing the resolved texture format.
    let file_source_primaries = ktx2.color_primaries().and_then(|color_primaries| {
        let source_primaries = ktx2_color_primaries_to_source_primaries(color_primaries);
        if source_primaries.is_none() {
            once!(warn!(
                "KTX2 file declares color primaries {color_primaries:?}, which Bevy does not \
                support; assuming BT.709",
            ));
        }
        source_primaries
    });
    if let Some(transfer_function) = ktx2.transfer_function() {
        if transfer_function == ktx2::TransferFunction::SRGB
            && !texture_format.is_srgb()
            && !srgb_data_linearized_on_cpu
        {
            once!(warn!(
                "KTX2 file declares an sRGB transfer function but the resolved texture format \
                is linear; the loader settings win and the data is used as-is",
            ));
        } else if transfer_function == ktx2::TransferFunction::Linear && srgb_data_linearized_on_cpu
        {
            once!(warn!(
                "KTX2 file declares a linear transfer function but `is_srgb` is true; the \
                loader setting wins and the data was sRGB-decoded to linear during transcoding",
            ));
        } else if transfer_function == ktx2::TransferFunction::Linear && texture_format.is_srgb() {
            once!(warn!(
                "KTX2 file declares a linear transfer function but is being loaded as sRGB \
                (`is_srgb` is true); the loader setting wins and the data is used as-is",
            ));
        } else if transfer_function == ktx2::TransferFunction::PQEOTF
            || transfer_function == ktx2::TransferFunction::PQOETF
            || transfer_function == ktx2::TransferFunction::HLGOETF
            || transfer_function == ktx2::TransferFunction::HLGEOTF
        {
            once!(warn!(
                "KTX2 file declares an HDR transfer function ({transfer_function:?}); decoding \
                HDR transfer functions is not supported and the data is used as-is",
            ));
        }
    }

    // Collect all level data into a contiguous buffer
    let mut image_data = Vec::new();
    image_data.reserve_exact(levels.iter().map(Vec::len).sum());
    levels.iter().for_each(|level| image_data.extend(level));

    // Assign the data and fill in the rest of the metadata now the possible
    // error cases have been handled
    let mut image = Image::default();
    image.texture_descriptor.format = texture_format;
    image.data = Some(image_data);
    image.data_order = wgpu_types::TextureDataOrder::MipMajor;
    image.source_primaries = file_source_primaries.unwrap_or_default();
    // Note: we must give wgpu the logical texture dimensions, so it can correctly compute mip sizes.
    // However this currently causes wgpu to panic if the dimensions arent a multiple of blocksize.
    // See https://github.com/gfx-rs/wgpu/issues/7677 for more context.
    image.texture_descriptor.size = Extent3d {
        width,
        height,
        depth_or_array_layers: if layer_count > 1 || face_count > 1 {
            layer_count * face_count
        } else {
            depth
        }
        .max(1),
    };
    image.texture_descriptor.mip_level_count = level_count;
    image.texture_descriptor.dimension = if is_3d {
        TextureDimension::D3
    } else if image.is_compressed() || is_2d {
        TextureDimension::D2
    } else {
        TextureDimension::D1
    };
    let mut dimension = None;
    if face_count == 6 {
        dimension = Some(if layer_count > 1 {
            TextureViewDimension::CubeArray
        } else {
            TextureViewDimension::Cube
        });
    } else if layer_count > 1 {
        dimension = Some(TextureViewDimension::D2Array);
    } else if is_3d {
        dimension = Some(TextureViewDimension::D3);
    }
    if dimension.is_some() {
        image.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension,
            ..default()
        });
    }
    Ok(image)
}

/// Texel block size of the format the file resolves to, so a cut mip chain never
/// starts at a level smaller than one block. UASTC and ETC1S transcode to 4x4-block
/// targets or to uncompressed formats, and 4x4 alignment is safe for both.
#[cfg(feature = "ktx2")]
fn ktx2_block_dimensions(texture_format: &Result<TextureFormat, TextureError>) -> (u32, u32) {
    match texture_format {
        Ok(format) => format.block_dimensions(),
        Err(TextureError::FormatRequiresTranscodingError(
            TranscodeFormat::Uastc(_) | TranscodeFormat::Etc1s,
        )) => (4, 4),
        Err(_) => (1, 1),
    }
}

/// Determines an appropriate wgpu-compatible format based on compressed format support, and a
/// basis universal [`TextureChannelLayout`].
#[cfg(feature = "basis-universal")]
pub fn get_transcoded_formats(
    supported_compressed_formats: CompressedImageFormats,
    data_format: TextureChannelLayout,
    is_srgb: bool,
) -> (TranscoderBlockFormat, TextureFormat) {
    match data_format {
        TextureChannelLayout::Rrr => {
            if supported_compressed_formats.contains(CompressedImageFormats::BC) {
                (TranscoderBlockFormat::BC4, TextureFormat::Bc4RUnorm)
            } else if supported_compressed_formats.contains(CompressedImageFormats::ETC2) {
                (
                    TranscoderBlockFormat::ETC2_EAC_R11,
                    TextureFormat::EacR11Unorm,
                )
            } else {
                (TranscoderBlockFormat::RGBA32, TextureFormat::R8Unorm)
            }
        }
        TextureChannelLayout::Rrrg | TextureChannelLayout::Rg => {
            if supported_compressed_formats.contains(CompressedImageFormats::BC) {
                (TranscoderBlockFormat::BC5, TextureFormat::Bc5RgUnorm)
            } else if supported_compressed_formats.contains(CompressedImageFormats::ETC2) {
                (
                    TranscoderBlockFormat::ETC2_EAC_RG11,
                    TextureFormat::EacRg11Unorm,
                )
            } else {
                (TranscoderBlockFormat::RGBA32, TextureFormat::Rg8Unorm)
            }
        }
        // NOTE: Rgba16Float should be transcoded to BC6H/ASTC_HDR. Neither are supported by
        // basis-universal, nor is ASTC_HDR supported by wgpu
        TextureChannelLayout::Rgb | TextureChannelLayout::Rgba => {
            // NOTE: UASTC can be losslessly transcoded to ASTC4x4 and ASTC uses the same
            // space as BC7 (128-bits per 4x4 texel block) so prefer ASTC over BC for
            // transcoding speed and quality.
            if supported_compressed_formats.contains(CompressedImageFormats::ASTC_LDR) {
                (
                    TranscoderBlockFormat::ASTC_4x4,
                    TextureFormat::Astc {
                        block: AstcBlock::B4x4,
                        channel: if is_srgb {
                            AstcChannel::UnormSrgb
                        } else {
                            AstcChannel::Unorm
                        },
                    },
                )
            } else if supported_compressed_formats.contains(CompressedImageFormats::BC) {
                (
                    TranscoderBlockFormat::BC7,
                    if is_srgb {
                        TextureFormat::Bc7RgbaUnormSrgb
                    } else {
                        TextureFormat::Bc7RgbaUnorm
                    },
                )
            } else if supported_compressed_formats.contains(CompressedImageFormats::ETC2) {
                (
                    TranscoderBlockFormat::ETC2_RGBA,
                    if is_srgb {
                        TextureFormat::Etc2Rgba8UnormSrgb
                    } else {
                        TextureFormat::Etc2Rgba8Unorm
                    },
                )
            } else {
                (
                    TranscoderBlockFormat::RGBA32,
                    if is_srgb {
                        TextureFormat::Rgba8UnormSrgb
                    } else {
                        TextureFormat::Rgba8Unorm
                    },
                )
            }
        }
    }
}

/// Reads the [`TextureFormat`] from a [`ktx2::Reader`].
///
/// # Errors
///
/// Returns an error for invalid KTX2 data, or unsupported texture formats.
#[cfg(feature = "ktx2")]
pub fn ktx2_get_texture_format<Data: AsRef<[u8]>>(
    ktx2: &ktx2::Reader<Data>,
    is_srgb: bool,
) -> Result<TextureFormat, TextureError> {
    if let Some(format) = ktx2.header().format {
        return ktx2_format_to_texture_format(format, is_srgb);
    }

    for data_format_descriptor in ktx2.dfd_blocks() {
        if let Block::Basic(basic_data_format_descriptor) = data_format_descriptor {
            return ktx2_dfd_header_to_texture_format(basic_data_format_descriptor, is_srgb);
        }
    }

    Err(TextureError::UnsupportedTextureFormat(
        "Unknown".to_string(),
    ))
}

/// Maps KTX2 data-format-descriptor color primaries to [`SourceColorPrimaries`],
/// returning `None` for primary sets Bevy does not support.
fn ktx2_color_primaries_to_source_primaries(
    color_primaries: ktx2::ColorPrimaries,
) -> Option<SourceColorPrimaries> {
    if color_primaries == ktx2::ColorPrimaries::BT709 {
        Some(SourceColorPrimaries::Bt709)
    } else if color_primaries == ktx2::ColorPrimaries::BT2020 {
        Some(SourceColorPrimaries::Bt2020)
    } else if color_primaries == ktx2::ColorPrimaries::DISPLAYP3 {
        Some(SourceColorPrimaries::DisplayP3)
    } else {
        None
    }
}

enum DataType {
    Unorm,
    UnormSrgb,
    Snorm,
    Float,
    Uint,
    Sint,
}

// This can be obtained from core::mem::transmute::<f32, u32>(1.0f32). It is used for identifying
// normalized sample types as in Unorm or Snorm.
const F32_1_AS_U32: u32 = 1065353216;

fn sample_information_to_data_type(
    sample: &SampleInformation,
    is_srgb: bool,
) -> Result<DataType, TextureError> {
    // Exponent flag not supported
    if sample
        .channel_type_qualifiers
        .contains(ChannelTypeQualifiers::EXPONENT)
    {
        return Err(TextureError::UnsupportedTextureFormat(
            "Unsupported KTX2 channel type qualifier: exponent".to_string(),
        ));
    }
    Ok(
        if sample
            .channel_type_qualifiers
            .contains(ChannelTypeQualifiers::FLOAT)
        {
            // If lower bound of range is 0 then unorm, else if upper bound is 1.0f32 as u32
            if sample
                .channel_type_qualifiers
                .contains(ChannelTypeQualifiers::SIGNED)
            {
                if sample.upper == F32_1_AS_U32 {
                    DataType::Snorm
                } else {
                    DataType::Float
                }
            } else if is_srgb {
                DataType::UnormSrgb
            } else {
                DataType::Unorm
            }
        } else if sample
            .channel_type_qualifiers
            .contains(ChannelTypeQualifiers::SIGNED)
        {
            DataType::Sint
        } else {
            DataType::Uint
        },
    )
}

/// Reads the [`TextureFormat`] from a KTX2 data format descriptor header.
///
/// # Errors
///
/// Returns an error for invalid or unsupported texture formats.
#[cfg(feature = "ktx2")]
pub fn ktx2_dfd_header_to_texture_format(
    basic_data_format_descriptor: &Basic,
    is_srgb: bool,
) -> Result<TextureFormat, TextureError> {
    let sample_information = &basic_data_format_descriptor.sample_information;
    Ok(match basic_data_format_descriptor.color_model {
        Some(ColorModel::RGBSDA) => {
            match sample_information.len() {
                1 => {
                    // Only red channel allowed
                    if sample_information[0].channel_type != 0 {
                        return Err(TextureError::UnsupportedTextureFormat(
                            "Only red-component single-component KTX2 RGBSDA formats supported"
                                .to_string(),
                        ));
                    }

                    let sample = &sample_information[0];
                    let data_type = sample_information_to_data_type(sample, false)?;
                    match sample.bit_length.get() {
                        8 => match data_type {
                            DataType::Unorm => TextureFormat::R8Unorm,
                            DataType::UnormSrgb => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "UnormSrgb not supported for R8".to_string(),
                                ));
                            }
                            DataType::Snorm => TextureFormat::R8Snorm,
                            DataType::Float => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "Float not supported for R8".to_string(),
                                ));
                            }
                            DataType::Uint => TextureFormat::R8Uint,
                            DataType::Sint => TextureFormat::R8Sint,
                        },
                        16 => match data_type {
                            DataType::Unorm => TextureFormat::R16Unorm,
                            DataType::UnormSrgb => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "UnormSrgb not supported for R16".to_string(),
                                ));
                            }
                            DataType::Snorm => TextureFormat::R16Snorm,
                            DataType::Float => TextureFormat::R16Float,
                            DataType::Uint => TextureFormat::R16Uint,
                            DataType::Sint => TextureFormat::R16Sint,
                        },
                        32 => match data_type {
                            DataType::Unorm => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "Unorm not supported for R32".to_string(),
                                ));
                            }
                            DataType::UnormSrgb => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "UnormSrgb not supported for R32".to_string(),
                                ));
                            }
                            DataType::Snorm => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "Snorm not supported for R32".to_string(),
                                ));
                            }
                            DataType::Float => TextureFormat::R32Float,
                            DataType::Uint => TextureFormat::R32Uint,
                            DataType::Sint => TextureFormat::R32Sint,
                        },
                        v => {
                            return Err(TextureError::UnsupportedTextureFormat(format!(
                                "Unsupported sample bit length for RGBSDA 1-channel format: {v}",
                            )));
                        }
                    }
                }
                2 => {
                    // Only red and green channels allowed
                    if sample_information[0].channel_type != 0
                        || sample_information[1].channel_type != 1
                    {
                        return Err(TextureError::UnsupportedTextureFormat(
                            "Only red-green-component two-component KTX2 RGBSDA formats supported"
                                .to_string(),
                        ));
                    }
                    // Only same bit length for all channels
                    assert_eq!(
                        sample_information[0].bit_length,
                        sample_information[1].bit_length
                    );
                    // Only same channel type qualifiers for all channels
                    assert_eq!(
                        sample_information[0].channel_type_qualifiers,
                        sample_information[1].channel_type_qualifiers
                    );
                    // Only same sample range for all channels
                    assert_eq!(sample_information[0].lower, sample_information[1].lower);
                    assert_eq!(sample_information[0].upper, sample_information[1].upper);

                    let sample = &sample_information[0];
                    let data_type = sample_information_to_data_type(sample, false)?;
                    match sample.bit_length.get() {
                        8 => match data_type {
                            DataType::Unorm => TextureFormat::Rg8Unorm,
                            DataType::UnormSrgb => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "UnormSrgb not supported for Rg8".to_string(),
                                ));
                            }
                            DataType::Snorm => TextureFormat::Rg8Snorm,
                            DataType::Float => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "Float not supported for Rg8".to_string(),
                                ));
                            }
                            DataType::Uint => TextureFormat::Rg8Uint,
                            DataType::Sint => TextureFormat::Rg8Sint,
                        },
                        16 => match data_type {
                            DataType::Unorm => TextureFormat::Rg16Unorm,
                            DataType::UnormSrgb => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "UnormSrgb not supported for Rg16".to_string(),
                                ));
                            }
                            DataType::Snorm => TextureFormat::Rg16Snorm,
                            DataType::Float => TextureFormat::Rg16Float,
                            DataType::Uint => TextureFormat::Rg16Uint,
                            DataType::Sint => TextureFormat::Rg16Sint,
                        },
                        32 => match data_type {
                            DataType::Unorm => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "Unorm not supported for Rg32".to_string(),
                                ));
                            }
                            DataType::UnormSrgb => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "UnormSrgb not supported for Rg32".to_string(),
                                ));
                            }
                            DataType::Snorm => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "Snorm not supported for Rg32".to_string(),
                                ));
                            }
                            DataType::Float => TextureFormat::Rg32Float,
                            DataType::Uint => TextureFormat::Rg32Uint,
                            DataType::Sint => TextureFormat::Rg32Sint,
                        },
                        v => {
                            return Err(TextureError::UnsupportedTextureFormat(format!(
                                "Unsupported sample bit length for RGBSDA 2-channel format: {v}",
                            )));
                        }
                    }
                }
                3 => {
                    if sample_information[0].channel_type == 0
                        && sample_information[0].bit_length.get() == 11
                        && sample_information[1].channel_type == 1
                        && sample_information[1].bit_length.get() == 11
                        && sample_information[2].channel_type == 2
                        && sample_information[2].bit_length.get() == 10
                    {
                        TextureFormat::Rg11b10Ufloat
                    } else if sample_information[0].channel_type == 0
                        && sample_information[0].bit_length.get() == 9
                        && sample_information[1].channel_type == 1
                        && sample_information[1].bit_length.get() == 9
                        && sample_information[2].channel_type == 2
                        && sample_information[2].bit_length.get() == 9
                    {
                        TextureFormat::Rgb9e5Ufloat
                    } else if sample_information[0].channel_type == 0
                        && sample_information[0].bit_length.get() == 8
                        && sample_information[1].channel_type == 1
                        && sample_information[1].bit_length.get() == 8
                        && sample_information[2].channel_type == 2
                        && sample_information[2].bit_length.get() == 8
                    {
                        return Err(TextureError::FormatRequiresTranscodingError(
                            TranscodeFormat::Rgb8,
                        ));
                    } else {
                        return Err(TextureError::UnsupportedTextureFormat(
                            "3-component formats not supported".to_string(),
                        ));
                    }
                }
                4 => {
                    // Only RGBA or BGRA channels allowed
                    let is_rgba = sample_information[0].channel_type == 0;
                    assert!(
                        sample_information[0].channel_type == 0
                            || sample_information[0].channel_type == 2
                    );
                    assert_eq!(sample_information[1].channel_type, 1);
                    assert_eq!(
                        sample_information[2].channel_type,
                        if is_rgba { 2 } else { 0 }
                    );
                    assert_eq!(sample_information[3].channel_type, 15);

                    // Handle one special packed format
                    if sample_information[0].bit_length.get() == 10
                        && sample_information[1].bit_length.get() == 10
                        && sample_information[2].bit_length.get() == 10
                        && sample_information[3].bit_length.get() == 2
                    {
                        return Ok(TextureFormat::Rgb10a2Unorm);
                    }

                    // Only same bit length for all channels
                    assert!(
                        sample_information[0].bit_length == sample_information[1].bit_length
                            && sample_information[0].bit_length == sample_information[2].bit_length
                            && sample_information[0].bit_length == sample_information[3].bit_length
                    );
                    assert!(
                        sample_information[0].lower == sample_information[1].lower
                            && sample_information[0].lower == sample_information[2].lower
                            && sample_information[0].lower == sample_information[3].lower
                    );
                    assert!(
                        sample_information[0].upper == sample_information[1].upper
                            && sample_information[0].upper == sample_information[2].upper
                            && sample_information[0].upper == sample_information[3].upper
                    );

                    let sample = &sample_information[0];
                    let data_type = sample_information_to_data_type(sample, is_srgb)?;
                    match sample.bit_length.get() {
                        8 => match data_type {
                            DataType::Unorm => {
                                if is_rgba {
                                    TextureFormat::Rgba8Unorm
                                } else {
                                    TextureFormat::Bgra8Unorm
                                }
                            }
                            DataType::UnormSrgb => {
                                if is_rgba {
                                    TextureFormat::Rgba8UnormSrgb
                                } else {
                                    TextureFormat::Bgra8UnormSrgb
                                }
                            }
                            DataType::Snorm => {
                                if is_rgba {
                                    TextureFormat::Rgba8Snorm
                                } else {
                                    return Err(TextureError::UnsupportedTextureFormat(
                                        "Bgra8 not supported for Snorm".to_string(),
                                    ));
                                }
                            }
                            DataType::Float => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "Float not supported for Rgba8/Bgra8".to_string(),
                                ));
                            }
                            DataType::Uint => {
                                if is_rgba {
                                    // NOTE: This is more about how you want to use the data so
                                    // TextureFormat::Rgba8Uint is incorrect here
                                    if is_srgb {
                                        TextureFormat::Rgba8UnormSrgb
                                    } else {
                                        TextureFormat::Rgba8Unorm
                                    }
                                } else {
                                    return Err(TextureError::UnsupportedTextureFormat(
                                        "Bgra8 not supported for Uint".to_string(),
                                    ));
                                }
                            }
                            DataType::Sint => {
                                if is_rgba {
                                    // NOTE: This is more about how you want to use the data so
                                    // TextureFormat::Rgba8Sint is incorrect here
                                    TextureFormat::Rgba8Snorm
                                } else {
                                    return Err(TextureError::UnsupportedTextureFormat(
                                        "Bgra8 not supported for Sint".to_string(),
                                    ));
                                }
                            }
                        },
                        16 => match data_type {
                            DataType::Unorm => {
                                if is_rgba {
                                    TextureFormat::Rgba16Unorm
                                } else {
                                    return Err(TextureError::UnsupportedTextureFormat(
                                        "Bgra16 not supported for Unorm".to_string(),
                                    ));
                                }
                            }
                            DataType::UnormSrgb => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "UnormSrgb not supported for Rgba16/Bgra16".to_string(),
                                ));
                            }
                            DataType::Snorm => {
                                if is_rgba {
                                    TextureFormat::Rgba16Snorm
                                } else {
                                    return Err(TextureError::UnsupportedTextureFormat(
                                        "Bgra16 not supported for Snorm".to_string(),
                                    ));
                                }
                            }
                            DataType::Float => {
                                if is_rgba {
                                    TextureFormat::Rgba16Float
                                } else {
                                    return Err(TextureError::UnsupportedTextureFormat(
                                        "Bgra16 not supported for Float".to_string(),
                                    ));
                                }
                            }
                            DataType::Uint => {
                                if is_rgba {
                                    TextureFormat::Rgba16Uint
                                } else {
                                    return Err(TextureError::UnsupportedTextureFormat(
                                        "Bgra16 not supported for Uint".to_string(),
                                    ));
                                }
                            }
                            DataType::Sint => {
                                if is_rgba {
                                    TextureFormat::Rgba16Sint
                                } else {
                                    return Err(TextureError::UnsupportedTextureFormat(
                                        "Bgra16 not supported for Sint".to_string(),
                                    ));
                                }
                            }
                        },
                        32 => match data_type {
                            DataType::Unorm => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "Unorm not supported for Rgba32/Bgra32".to_string(),
                                ));
                            }
                            DataType::UnormSrgb => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "UnormSrgb not supported for Rgba32/Bgra32".to_string(),
                                ));
                            }
                            DataType::Snorm => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "Snorm not supported for Rgba32/Bgra32".to_string(),
                                ));
                            }
                            DataType::Float => {
                                if is_rgba {
                                    TextureFormat::Rgba32Float
                                } else {
                                    return Err(TextureError::UnsupportedTextureFormat(
                                        "Bgra32 not supported for Float".to_string(),
                                    ));
                                }
                            }
                            DataType::Uint => {
                                if is_rgba {
                                    TextureFormat::Rgba32Uint
                                } else {
                                    return Err(TextureError::UnsupportedTextureFormat(
                                        "Bgra32 not supported for Uint".to_string(),
                                    ));
                                }
                            }
                            DataType::Sint => {
                                if is_rgba {
                                    TextureFormat::Rgba32Sint
                                } else {
                                    return Err(TextureError::UnsupportedTextureFormat(
                                        "Bgra32 not supported for Sint".to_string(),
                                    ));
                                }
                            }
                        },
                        v => {
                            return Err(TextureError::UnsupportedTextureFormat(format!(
                                "Unsupported sample bit length for RGBSDA 4-channel format: {v}",
                            )));
                        }
                    }
                }
                v => {
                    return Err(TextureError::UnsupportedTextureFormat(format!(
                        "Unsupported channel count for RGBSDA format: {v}",
                    )));
                }
            }
        }
        Some(ColorModel::YUVSDA)
        | Some(ColorModel::YIQSDA)
        | Some(ColorModel::LabSDA)
        | Some(ColorModel::CMYKA)
        | Some(ColorModel::HSVAAng)
        | Some(ColorModel::HSLAAng)
        | Some(ColorModel::HSVAHex)
        | Some(ColorModel::HSLAHex)
        | Some(ColorModel::YCgCoA)
        | Some(ColorModel::YcCbcCrc)
        | Some(ColorModel::ICtCp)
        | Some(ColorModel::CIEXYZ)
        | Some(ColorModel::CIEXYY) => {
            return Err(TextureError::UnsupportedTextureFormat(format!(
                "{:?}",
                basic_data_format_descriptor.color_model
            )));
        }
        Some(ColorModel::XYZW) => {
            // Same number of channels in both texel block dimensions and sample info descriptions
            assert_eq!(
                basic_data_format_descriptor.texel_block_dimensions[0].get() as usize,
                sample_information.len()
            );
            match sample_information.len() {
                4 => {
                    // Only RGBA or BGRA channels allowed
                    assert_eq!(sample_information[0].channel_type, 0);
                    assert_eq!(sample_information[1].channel_type, 1);
                    assert_eq!(sample_information[2].channel_type, 2);
                    assert_eq!(sample_information[3].channel_type, 3);
                    // Only same bit length for all channels
                    assert!(
                        sample_information[0].bit_length == sample_information[1].bit_length
                            && sample_information[0].bit_length == sample_information[2].bit_length
                            && sample_information[0].bit_length == sample_information[3].bit_length
                    );
                    // Only same channel type qualifiers for all channels
                    assert!(
                        sample_information[0].channel_type_qualifiers
                            == sample_information[1].channel_type_qualifiers
                            && sample_information[0].channel_type_qualifiers
                                == sample_information[2].channel_type_qualifiers
                            && sample_information[0].channel_type_qualifiers
                                == sample_information[3].channel_type_qualifiers
                    );
                    // Only same sample range for all channels
                    assert!(
                        sample_information[0].lower == sample_information[1].lower
                            && sample_information[0].lower == sample_information[2].lower
                            && sample_information[0].lower == sample_information[3].lower
                    );
                    assert!(
                        sample_information[0].upper == sample_information[1].upper
                            && sample_information[0].upper == sample_information[2].upper
                            && sample_information[0].upper == sample_information[3].upper
                    );

                    let sample = &sample_information[0];
                    let data_type = sample_information_to_data_type(sample, false)?;
                    match sample.bit_length.get() {
                        8 => match data_type {
                            DataType::Unorm => TextureFormat::Rgba8Unorm,
                            DataType::UnormSrgb => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "UnormSrgb not supported for XYZW".to_string(),
                                ));
                            }
                            DataType::Snorm => TextureFormat::Rgba8Snorm,
                            DataType::Float => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "Float not supported for Rgba8/Bgra8".to_string(),
                                ));
                            }
                            DataType::Uint => TextureFormat::Rgba8Uint,
                            DataType::Sint => TextureFormat::Rgba8Sint,
                        },
                        16 => match data_type {
                            DataType::Unorm => TextureFormat::Rgba16Unorm,
                            DataType::UnormSrgb => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "UnormSrgb not supported for Rgba16/Bgra16".to_string(),
                                ));
                            }
                            DataType::Snorm => TextureFormat::Rgba16Snorm,
                            DataType::Float => TextureFormat::Rgba16Float,
                            DataType::Uint => TextureFormat::Rgba16Uint,
                            DataType::Sint => TextureFormat::Rgba16Sint,
                        },
                        32 => match data_type {
                            DataType::Unorm => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "Unorm not supported for Rgba32/Bgra32".to_string(),
                                ));
                            }
                            DataType::UnormSrgb => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "UnormSrgb not supported for Rgba32/Bgra32".to_string(),
                                ));
                            }
                            DataType::Snorm => {
                                return Err(TextureError::UnsupportedTextureFormat(
                                    "Snorm not supported for Rgba32/Bgra32".to_string(),
                                ));
                            }
                            DataType::Float => TextureFormat::Rgba32Float,
                            DataType::Uint => TextureFormat::Rgba32Uint,
                            DataType::Sint => TextureFormat::Rgba32Sint,
                        },
                        v => {
                            return Err(TextureError::UnsupportedTextureFormat(format!(
                                "Unsupported sample bit length for XYZW 4-channel format: {v}",
                            )));
                        }
                    }
                }
                v => {
                    return Err(TextureError::UnsupportedTextureFormat(format!(
                        "Unsupported channel count for XYZW format: {v}",
                    )));
                }
            }
        }
        Some(ColorModel::BC1A) => {
            if is_srgb {
                TextureFormat::Bc1RgbaUnormSrgb
            } else {
                TextureFormat::Bc1RgbaUnorm
            }
        }
        Some(ColorModel::BC2) => {
            if is_srgb {
                TextureFormat::Bc2RgbaUnormSrgb
            } else {
                TextureFormat::Bc2RgbaUnorm
            }
        }
        Some(ColorModel::BC3) => {
            if is_srgb {
                TextureFormat::Bc3RgbaUnormSrgb
            } else {
                TextureFormat::Bc3RgbaUnorm
            }
        }
        Some(ColorModel::BC4) => {
            if sample_information[0].lower == 0 {
                TextureFormat::Bc4RUnorm
            } else {
                TextureFormat::Bc4RSnorm
            }
        }
        // FIXME: Red and green channels can be swapped for ATI2n/3Dc
        Some(ColorModel::BC5) => {
            if sample_information[0].lower == 0 {
                TextureFormat::Bc5RgUnorm
            } else {
                TextureFormat::Bc5RgSnorm
            }
        }
        Some(ColorModel::BC6H) => {
            if sample_information[0].lower == 0 {
                TextureFormat::Bc6hRgbUfloat
            } else {
                TextureFormat::Bc6hRgbFloat
            }
        }
        Some(ColorModel::BC7) => {
            if is_srgb {
                TextureFormat::Bc7RgbaUnormSrgb
            } else {
                TextureFormat::Bc7RgbaUnorm
            }
        }
        // ETC1 a subset of ETC2 only supporting Rgb8
        Some(ColorModel::ETC1) => {
            if is_srgb {
                TextureFormat::Etc2Rgb8UnormSrgb
            } else {
                TextureFormat::Etc2Rgb8Unorm
            }
        }
        Some(ColorModel::ETC2) => match sample_information.len() {
            1 => {
                let sample = &sample_information[0];
                match sample.channel_type {
                    0 => {
                        if sample_information[0]
                            .channel_type_qualifiers
                            .contains(ChannelTypeQualifiers::SIGNED)
                        {
                            TextureFormat::EacR11Snorm
                        } else {
                            TextureFormat::EacR11Unorm
                        }
                    }
                    2 => {
                        if is_srgb {
                            TextureFormat::Etc2Rgb8UnormSrgb
                        } else {
                            TextureFormat::Etc2Rgb8Unorm
                        }
                    }
                    _ => {
                        return Err(TextureError::UnsupportedTextureFormat(format!(
                            "Invalid ETC2 sample channel type: {}",
                            sample.channel_type
                        )))
                    }
                }
            }
            2 => {
                let sample0 = &sample_information[0];
                let sample1 = &sample_information[1];
                if sample0.channel_type == 0 && sample1.channel_type == 1 {
                    if sample0
                        .channel_type_qualifiers
                        .contains(ChannelTypeQualifiers::SIGNED)
                    {
                        TextureFormat::EacRg11Snorm
                    } else {
                        TextureFormat::EacRg11Unorm
                    }
                } else if sample0.channel_type == 2 && sample1.channel_type == 15 {
                    if is_srgb {
                        TextureFormat::Etc2Rgb8A1UnormSrgb
                    } else {
                        TextureFormat::Etc2Rgb8A1Unorm
                    }
                } else if sample0.channel_type == 15 && sample1.channel_type == 2 {
                    if is_srgb {
                        TextureFormat::Etc2Rgba8UnormSrgb
                    } else {
                        TextureFormat::Etc2Rgba8Unorm
                    }
                } else {
                    return Err(TextureError::UnsupportedTextureFormat(format!(
                        "Invalid ETC2 2-sample channel types: {} {}",
                        sample0.channel_type, sample1.channel_type
                    )));
                }
            }
            v => {
                return Err(TextureError::UnsupportedTextureFormat(format!(
                    "Unsupported channel count for ETC2 format: {v}",
                )));
            }
        },
        Some(ColorModel::ASTC) => TextureFormat::Astc {
            block: match (
                basic_data_format_descriptor.texel_block_dimensions[0].get(),
                basic_data_format_descriptor.texel_block_dimensions[1].get(),
            ) {
                (4, 4) => AstcBlock::B4x4,
                (5, 4) => AstcBlock::B5x4,
                (5, 5) => AstcBlock::B5x5,
                (6, 5) => AstcBlock::B6x5,
                (8, 5) => AstcBlock::B8x5,
                (8, 8) => AstcBlock::B8x8,
                (10, 5) => AstcBlock::B10x5,
                (10, 6) => AstcBlock::B10x6,
                (10, 8) => AstcBlock::B10x8,
                (10, 10) => AstcBlock::B10x10,
                (12, 10) => AstcBlock::B12x10,
                (12, 12) => AstcBlock::B12x12,
                d => {
                    return Err(TextureError::UnsupportedTextureFormat(format!(
                        "Invalid ASTC dimension: {} x {}",
                        d.0, d.1
                    )))
                }
            },
            channel: if is_srgb {
                AstcChannel::UnormSrgb
            } else {
                AstcChannel::Unorm
            },
        },
        Some(ColorModel::ETC1S) => {
            return Err(TextureError::FormatRequiresTranscodingError(
                TranscodeFormat::Etc1s,
            ));
        }
        Some(ColorModel::PVRTC) => {
            return Err(TextureError::UnsupportedTextureFormat(
                "PVRTC is not supported".to_string(),
            ));
        }
        Some(ColorModel::PVRTC2) => {
            return Err(TextureError::UnsupportedTextureFormat(
                "PVRTC2 is not supported".to_string(),
            ));
        }
        Some(ColorModel::UASTC) => {
            return Err(TextureError::FormatRequiresTranscodingError(
                TranscodeFormat::Uastc(match sample_information[0].channel_type {
                    0 => TextureChannelLayout::Rgb,
                    3 => TextureChannelLayout::Rgba,
                    4 => TextureChannelLayout::Rrr,
                    5 => TextureChannelLayout::Rrrg,
                    6 => TextureChannelLayout::Rg,
                    channel_type => {
                        return Err(TextureError::UnsupportedTextureFormat(format!(
                            "Invalid KTX2 UASTC channel type: {channel_type}",
                        )))
                    }
                }),
            ));
        }
        None => {
            return Err(TextureError::UnsupportedTextureFormat(
                "Unspecified KTX2 color model".to_string(),
            ));
        }
        _ => {
            return Err(TextureError::UnsupportedTextureFormat(format!(
                "Unknown KTX2 color model: {:?}",
                basic_data_format_descriptor.color_model
            )));
        }
    })
}

/// Converts a KTX2 texture format identifier to a [`TextureFormat`].
///
/// # Errors
///
/// Returns an error for unsupported texture formats.
#[cfg(feature = "ktx2")]
pub fn ktx2_format_to_texture_format(
    ktx2_format: ktx2::Format,
    is_srgb: bool,
) -> Result<TextureFormat, TextureError> {
    Ok(match ktx2_format {
        ktx2::Format::R8_UNORM | ktx2::Format::R8_SRGB => {
            if is_srgb {
                return Err(TextureError::FormatRequiresTranscodingError(
                    TranscodeFormat::R8UnormSrgb,
                ));
            }
            TextureFormat::R8Unorm
        }
        ktx2::Format::R8_SNORM => TextureFormat::R8Snorm,
        ktx2::Format::R8_UINT => TextureFormat::R8Uint,
        ktx2::Format::R8_SINT => TextureFormat::R8Sint,
        ktx2::Format::R8G8_UNORM | ktx2::Format::R8G8_SRGB => {
            if is_srgb {
                return Err(TextureError::FormatRequiresTranscodingError(
                    TranscodeFormat::Rg8UnormSrgb,
                ));
            }
            TextureFormat::Rg8Unorm
        }
        ktx2::Format::R8G8_SNORM => TextureFormat::Rg8Snorm,
        ktx2::Format::R8G8_UINT => TextureFormat::Rg8Uint,
        ktx2::Format::R8G8_SINT => TextureFormat::Rg8Sint,
        ktx2::Format::R8G8B8_UNORM | ktx2::Format::R8G8B8_SRGB => {
            return Err(TextureError::FormatRequiresTranscodingError(
                TranscodeFormat::Rgb8,
            ));
        }
        ktx2::Format::R8G8B8A8_UNORM | ktx2::Format::R8G8B8A8_SRGB => {
            if is_srgb {
                TextureFormat::Rgba8UnormSrgb
            } else {
                TextureFormat::Rgba8Unorm
            }
        }
        ktx2::Format::R8G8B8A8_SNORM => TextureFormat::Rgba8Snorm,
        ktx2::Format::R8G8B8A8_UINT => TextureFormat::Rgba8Uint,
        ktx2::Format::R8G8B8A8_SINT => TextureFormat::Rgba8Sint,
        ktx2::Format::B8G8R8A8_UNORM | ktx2::Format::B8G8R8A8_SRGB => {
            if is_srgb {
                TextureFormat::Bgra8UnormSrgb
            } else {
                TextureFormat::Bgra8Unorm
            }
        }
        ktx2::Format::A2R10G10B10_UNORM_PACK32 => TextureFormat::Rgb10a2Unorm,

        ktx2::Format::R16_UNORM => TextureFormat::R16Unorm,
        ktx2::Format::R16_SNORM => TextureFormat::R16Snorm,
        ktx2::Format::R16_UINT => TextureFormat::R16Uint,
        ktx2::Format::R16_SINT => TextureFormat::R16Sint,
        ktx2::Format::R16_SFLOAT => TextureFormat::R16Float,
        ktx2::Format::R16G16_UNORM => TextureFormat::Rg16Unorm,
        ktx2::Format::R16G16_SNORM => TextureFormat::Rg16Snorm,
        ktx2::Format::R16G16_UINT => TextureFormat::Rg16Uint,
        ktx2::Format::R16G16_SINT => TextureFormat::Rg16Sint,
        ktx2::Format::R16G16_SFLOAT => TextureFormat::Rg16Float,

        ktx2::Format::R16G16B16A16_UNORM => TextureFormat::Rgba16Unorm,
        ktx2::Format::R16G16B16A16_SNORM => TextureFormat::Rgba16Snorm,
        ktx2::Format::R16G16B16A16_UINT => TextureFormat::Rgba16Uint,
        ktx2::Format::R16G16B16A16_SINT => TextureFormat::Rgba16Sint,
        ktx2::Format::R16G16B16A16_SFLOAT => TextureFormat::Rgba16Float,
        ktx2::Format::R32_UINT => TextureFormat::R32Uint,
        ktx2::Format::R32_SINT => TextureFormat::R32Sint,
        ktx2::Format::R32_SFLOAT => TextureFormat::R32Float,
        ktx2::Format::R32G32_UINT => TextureFormat::Rg32Uint,
        ktx2::Format::R32G32_SINT => TextureFormat::Rg32Sint,
        ktx2::Format::R32G32_SFLOAT => TextureFormat::Rg32Float,

        ktx2::Format::R32G32B32A32_UINT => TextureFormat::Rgba32Uint,
        ktx2::Format::R32G32B32A32_SINT => TextureFormat::Rgba32Sint,
        ktx2::Format::R32G32B32A32_SFLOAT => TextureFormat::Rgba32Float,

        ktx2::Format::B10G11R11_UFLOAT_PACK32 => TextureFormat::Rg11b10Ufloat,
        ktx2::Format::E5B9G9R9_UFLOAT_PACK32 => TextureFormat::Rgb9e5Ufloat,

        ktx2::Format::X8_D24_UNORM_PACK32 => TextureFormat::Depth24Plus,
        ktx2::Format::D32_SFLOAT => TextureFormat::Depth32Float,

        ktx2::Format::D24_UNORM_S8_UINT => TextureFormat::Depth24PlusStencil8,

        ktx2::Format::BC1_RGB_UNORM_BLOCK
        | ktx2::Format::BC1_RGB_SRGB_BLOCK
        | ktx2::Format::BC1_RGBA_UNORM_BLOCK
        | ktx2::Format::BC1_RGBA_SRGB_BLOCK => {
            if is_srgb {
                TextureFormat::Bc1RgbaUnormSrgb
            } else {
                TextureFormat::Bc1RgbaUnorm
            }
        }
        ktx2::Format::BC2_UNORM_BLOCK | ktx2::Format::BC2_SRGB_BLOCK => {
            if is_srgb {
                TextureFormat::Bc2RgbaUnormSrgb
            } else {
                TextureFormat::Bc2RgbaUnorm
            }
        }
        ktx2::Format::BC3_UNORM_BLOCK | ktx2::Format::BC3_SRGB_BLOCK => {
            if is_srgb {
                TextureFormat::Bc3RgbaUnormSrgb
            } else {
                TextureFormat::Bc3RgbaUnorm
            }
        }
        ktx2::Format::BC4_UNORM_BLOCK => TextureFormat::Bc4RUnorm,
        ktx2::Format::BC4_SNORM_BLOCK => TextureFormat::Bc4RSnorm,
        ktx2::Format::BC5_UNORM_BLOCK => TextureFormat::Bc5RgUnorm,
        ktx2::Format::BC5_SNORM_BLOCK => TextureFormat::Bc5RgSnorm,
        ktx2::Format::BC6H_UFLOAT_BLOCK => TextureFormat::Bc6hRgbUfloat,
        ktx2::Format::BC6H_SFLOAT_BLOCK => TextureFormat::Bc6hRgbFloat,
        ktx2::Format::BC7_UNORM_BLOCK | ktx2::Format::BC7_SRGB_BLOCK => {
            if is_srgb {
                TextureFormat::Bc7RgbaUnormSrgb
            } else {
                TextureFormat::Bc7RgbaUnorm
            }
        }
        ktx2::Format::ETC2_R8G8B8_UNORM_BLOCK | ktx2::Format::ETC2_R8G8B8_SRGB_BLOCK => {
            if is_srgb {
                TextureFormat::Etc2Rgb8UnormSrgb
            } else {
                TextureFormat::Etc2Rgb8Unorm
            }
        }
        ktx2::Format::ETC2_R8G8B8A1_UNORM_BLOCK | ktx2::Format::ETC2_R8G8B8A1_SRGB_BLOCK => {
            if is_srgb {
                TextureFormat::Etc2Rgb8A1UnormSrgb
            } else {
                TextureFormat::Etc2Rgb8A1Unorm
            }
        }
        ktx2::Format::ETC2_R8G8B8A8_UNORM_BLOCK | ktx2::Format::ETC2_R8G8B8A8_SRGB_BLOCK => {
            if is_srgb {
                TextureFormat::Etc2Rgba8UnormSrgb
            } else {
                TextureFormat::Etc2Rgba8Unorm
            }
        }
        ktx2::Format::EAC_R11_UNORM_BLOCK => TextureFormat::EacR11Unorm,
        ktx2::Format::EAC_R11_SNORM_BLOCK => TextureFormat::EacR11Snorm,
        ktx2::Format::EAC_R11G11_UNORM_BLOCK => TextureFormat::EacRg11Unorm,
        ktx2::Format::EAC_R11G11_SNORM_BLOCK => TextureFormat::EacRg11Snorm,
        ktx2::Format::ASTC_4x4_UNORM_BLOCK | ktx2::Format::ASTC_4x4_SRGB_BLOCK => {
            TextureFormat::Astc {
                block: AstcBlock::B4x4,
                channel: if is_srgb {
                    AstcChannel::UnormSrgb
                } else {
                    AstcChannel::Unorm
                },
            }
        }
        ktx2::Format::ASTC_5x4_UNORM_BLOCK | ktx2::Format::ASTC_5x4_SRGB_BLOCK => {
            TextureFormat::Astc {
                block: AstcBlock::B5x4,
                channel: if is_srgb {
                    AstcChannel::UnormSrgb
                } else {
                    AstcChannel::Unorm
                },
            }
        }
        ktx2::Format::ASTC_5x5_UNORM_BLOCK | ktx2::Format::ASTC_5x5_SRGB_BLOCK => {
            TextureFormat::Astc {
                block: AstcBlock::B5x5,
                channel: if is_srgb {
                    AstcChannel::UnormSrgb
                } else {
                    AstcChannel::Unorm
                },
            }
        }
        ktx2::Format::ASTC_6x5_UNORM_BLOCK | ktx2::Format::ASTC_6x5_SRGB_BLOCK => {
            TextureFormat::Astc {
                block: AstcBlock::B6x5,
                channel: if is_srgb {
                    AstcChannel::UnormSrgb
                } else {
                    AstcChannel::Unorm
                },
            }
        }
        ktx2::Format::ASTC_6x6_UNORM_BLOCK | ktx2::Format::ASTC_6x6_SRGB_BLOCK => {
            TextureFormat::Astc {
                block: AstcBlock::B6x6,
                channel: if is_srgb {
                    AstcChannel::UnormSrgb
                } else {
                    AstcChannel::Unorm
                },
            }
        }
        ktx2::Format::ASTC_8x5_UNORM_BLOCK | ktx2::Format::ASTC_8x5_SRGB_BLOCK => {
            TextureFormat::Astc {
                block: AstcBlock::B8x5,
                channel: if is_srgb {
                    AstcChannel::UnormSrgb
                } else {
                    AstcChannel::Unorm
                },
            }
        }
        ktx2::Format::ASTC_8x6_UNORM_BLOCK | ktx2::Format::ASTC_8x6_SRGB_BLOCK => {
            TextureFormat::Astc {
                block: AstcBlock::B8x6,
                channel: if is_srgb {
                    AstcChannel::UnormSrgb
                } else {
                    AstcChannel::Unorm
                },
            }
        }
        ktx2::Format::ASTC_8x8_UNORM_BLOCK | ktx2::Format::ASTC_8x8_SRGB_BLOCK => {
            TextureFormat::Astc {
                block: AstcBlock::B8x8,
                channel: if is_srgb {
                    AstcChannel::UnormSrgb
                } else {
                    AstcChannel::Unorm
                },
            }
        }
        ktx2::Format::ASTC_10x5_UNORM_BLOCK | ktx2::Format::ASTC_10x5_SRGB_BLOCK => {
            TextureFormat::Astc {
                block: AstcBlock::B10x5,
                channel: if is_srgb {
                    AstcChannel::UnormSrgb
                } else {
                    AstcChannel::Unorm
                },
            }
        }
        ktx2::Format::ASTC_10x6_UNORM_BLOCK | ktx2::Format::ASTC_10x6_SRGB_BLOCK => {
            TextureFormat::Astc {
                block: AstcBlock::B10x6,
                channel: if is_srgb {
                    AstcChannel::UnormSrgb
                } else {
                    AstcChannel::Unorm
                },
            }
        }
        ktx2::Format::ASTC_10x8_UNORM_BLOCK | ktx2::Format::ASTC_10x8_SRGB_BLOCK => {
            TextureFormat::Astc {
                block: AstcBlock::B10x8,
                channel: if is_srgb {
                    AstcChannel::UnormSrgb
                } else {
                    AstcChannel::Unorm
                },
            }
        }
        ktx2::Format::ASTC_10x10_UNORM_BLOCK | ktx2::Format::ASTC_10x10_SRGB_BLOCK => {
            TextureFormat::Astc {
                block: AstcBlock::B10x10,
                channel: if is_srgb {
                    AstcChannel::UnormSrgb
                } else {
                    AstcChannel::Unorm
                },
            }
        }
        ktx2::Format::ASTC_12x10_UNORM_BLOCK | ktx2::Format::ASTC_12x10_SRGB_BLOCK => {
            TextureFormat::Astc {
                block: AstcBlock::B12x10,
                channel: if is_srgb {
                    AstcChannel::UnormSrgb
                } else {
                    AstcChannel::Unorm
                },
            }
        }
        ktx2::Format::ASTC_12x12_UNORM_BLOCK | ktx2::Format::ASTC_12x12_SRGB_BLOCK => {
            TextureFormat::Astc {
                block: AstcBlock::B12x12,
                channel: if is_srgb {
                    AstcChannel::UnormSrgb
                } else {
                    AstcChannel::Unorm
                },
            }
        }
        ktx2::Format::ASTC_4x4_SFLOAT_BLOCK => TextureFormat::Astc {
            block: AstcBlock::B4x4,
            channel: AstcChannel::Hdr,
        },
        ktx2::Format::ASTC_5x4_SFLOAT_BLOCK => TextureFormat::Astc {
            block: AstcBlock::B5x4,
            channel: AstcChannel::Hdr,
        },
        ktx2::Format::ASTC_5x5_SFLOAT_BLOCK => TextureFormat::Astc {
            block: AstcBlock::B5x5,
            channel: AstcChannel::Hdr,
        },
        ktx2::Format::ASTC_6x5_SFLOAT_BLOCK => TextureFormat::Astc {
            block: AstcBlock::B6x5,
            channel: AstcChannel::Hdr,
        },
        ktx2::Format::ASTC_6x6_SFLOAT_BLOCK => TextureFormat::Astc {
            block: AstcBlock::B6x6,
            channel: AstcChannel::Hdr,
        },
        ktx2::Format::ASTC_8x5_SFLOAT_BLOCK => TextureFormat::Astc {
            block: AstcBlock::B8x5,
            channel: AstcChannel::Hdr,
        },
        ktx2::Format::ASTC_8x6_SFLOAT_BLOCK => TextureFormat::Astc {
            block: AstcBlock::B8x6,
            channel: AstcChannel::Hdr,
        },
        ktx2::Format::ASTC_8x8_SFLOAT_BLOCK => TextureFormat::Astc {
            block: AstcBlock::B8x8,
            channel: AstcChannel::Hdr,
        },
        ktx2::Format::ASTC_10x5_SFLOAT_BLOCK => TextureFormat::Astc {
            block: AstcBlock::B10x5,
            channel: AstcChannel::Hdr,
        },
        ktx2::Format::ASTC_10x6_SFLOAT_BLOCK => TextureFormat::Astc {
            block: AstcBlock::B10x6,
            channel: AstcChannel::Hdr,
        },
        ktx2::Format::ASTC_10x8_SFLOAT_BLOCK => TextureFormat::Astc {
            block: AstcBlock::B10x8,
            channel: AstcChannel::Hdr,
        },
        ktx2::Format::ASTC_10x10_SFLOAT_BLOCK => TextureFormat::Astc {
            block: AstcBlock::B10x10,
            channel: AstcChannel::Hdr,
        },
        ktx2::Format::ASTC_12x10_SFLOAT_BLOCK => TextureFormat::Astc {
            block: AstcBlock::B12x10,
            channel: AstcChannel::Hdr,
        },
        ktx2::Format::ASTC_12x12_SFLOAT_BLOCK => TextureFormat::Astc {
            block: AstcBlock::B12x12,
            channel: AstcChannel::Hdr,
        },
        _ => {
            return Err(TextureError::UnsupportedTextureFormat(format!(
                "{ktx2_format:?}"
            )))
        }
    })
}

#[cfg(test)]
mod tests {
    use crate::{CompressedImageFormats, SourceColorPrimaries};

    use super::ktx2_buffer_to_image;

    #[test]
    fn test_ktx_levels() {
        // R8UnormSrgb texture with 4x4 pixels data and 3 levels of mipmaps
        let buffer = vec![
            0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 10, 0x1a, 10, 0x0f, 0, 0, 0, 1,
            0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0,
            0, 0, 0x98, 0, 0, 0, 0x2c, 0, 0, 0, 0xc4, 0, 0, 0, 0x5c, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0x28, 1, 0, 0, 0, 0, 0, 0, 0x10, 0, 0, 0, 0, 0, 0, 0, 0x10,
            0, 0, 0, 0, 0, 0, 0, 0x24, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,
            0, 0, 0, 0x20, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0x2c, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0x28, 0, 1, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0, 0, 0, 0x12, 0, 0, 0, 0x4b, 0x54, 0x58,
            0x6f, 0x72, 0x69, 0x65, 0x6e, 0x74, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0, 0x72, 0x64, 0, 0,
            0, 0x10, 0, 0, 0, 0x4b, 0x54, 0x58, 0x73, 0x77, 0x69, 0x7a, 0x7a, 0x6c, 0x65, 0, 0x72,
            0x72, 0x72, 0x31, 0, 0x2c, 0, 0, 0, 0x4b, 0x54, 0x58, 0x77, 0x72, 0x69, 0x74, 0x65,
            0x72, 0, 0x74, 0x6f, 0x6b, 0x74, 0x78, 0x20, 0x76, 0x34, 0x2e, 0x33, 0x2e, 0x30, 0x7e,
            0x32, 0x38, 0x20, 0x2f, 0x20, 0x6c, 0x69, 0x62, 0x6b, 0x74, 0x78, 0x20, 0x76, 0x34,
            0x2e, 0x33, 0x2e, 0x30, 0x7e, 0x31, 0, 0x4a, 0, 0, 0, 0x4a, 0x4a, 0x4a, 0x4a, 0x4a,
            0x4a, 0x4a, 0x4a, 0x4a, 0x4a, 0x4a, 0x4a, 0x4a, 0x4a, 0x4a, 0x4a, 0x4a, 0x4a, 0x4a,
            0x4a,
        ];
        let supported_compressed_formats = CompressedImageFormats::empty();
        let result = ktx2_buffer_to_image(&buffer, supported_compressed_formats, true, None);
        assert!(result.is_ok());
    }

    /// Builds a minimal valid KTX2 file: a 1x1 texture in the given 8-bit-per-channel
    /// `vkFormat` (with `pixel` carrying one texel) with a single level and a basic
    /// data format descriptor block carrying the given raw `colorPrimaries` and
    /// `transferFunction` bytes (`0` = unspecified).
    fn minimal_ktx2(
        vk_format: u32,
        color_primaries: u8,
        transfer_function: u8,
        pixel: &[u8],
    ) -> Vec<u8> {
        let mut buffer = Vec::new();
        // Identifier
        buffer.extend_from_slice(&[
            0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
        ]);
        buffer.extend_from_slice(&vk_format.to_le_bytes());
        buffer.extend_from_slice(&1u32.to_le_bytes()); // typeSize
        buffer.extend_from_slice(&1u32.to_le_bytes()); // pixelWidth
        buffer.extend_from_slice(&1u32.to_le_bytes()); // pixelHeight
        buffer.extend_from_slice(&0u32.to_le_bytes()); // pixelDepth
        buffer.extend_from_slice(&0u32.to_le_bytes()); // layerCount
        buffer.extend_from_slice(&1u32.to_le_bytes()); // faceCount
        buffer.extend_from_slice(&1u32.to_le_bytes()); // levelCount
        buffer.extend_from_slice(&0u32.to_le_bytes()); // supercompressionScheme
                                                       // Section index: a 28-byte DFD at offset 104 (4-byte total size + a 24-byte
                                                       // basic block with no sample information), no key/value data, no
                                                       // supercompression global data.
        buffer.extend_from_slice(&104u32.to_le_bytes()); // dfdByteOffset
        buffer.extend_from_slice(&28u32.to_le_bytes()); // dfdByteLength
        buffer.extend_from_slice(&0u32.to_le_bytes()); // kvdByteOffset
        buffer.extend_from_slice(&0u32.to_le_bytes()); // kvdByteLength
        buffer.extend_from_slice(&0u64.to_le_bytes()); // sgdByteOffset
        buffer.extend_from_slice(&0u64.to_le_bytes()); // sgdByteLength
                                                       // Level index (1 entry): the single one-texel level directly after the DFD.
        buffer.extend_from_slice(&132u64.to_le_bytes()); // byteOffset
        buffer.extend_from_slice(&(pixel.len() as u64).to_le_bytes()); // byteLength
        buffer.extend_from_slice(&(pixel.len() as u64).to_le_bytes()); // uncompressedByteLength
        assert_eq!(buffer.len(), 104);
        // Data format descriptor
        buffer.extend_from_slice(&28u32.to_le_bytes()); // dfdTotalSize
        buffer.extend_from_slice(&0u32.to_le_bytes()); // vendorId | descriptorType (basic)
        buffer.extend_from_slice(&2u16.to_le_bytes()); // versionNumber
        buffer.extend_from_slice(&24u16.to_le_bytes()); // descriptorBlockSize (no samples)
        buffer.push(1); // colorModel = RGBSDA
        buffer.push(color_primaries);
        buffer.push(transfer_function);
        buffer.push(0); // flags
        buffer.extend_from_slice(&[0, 0, 0, 0]); // texelBlockDimension (stored as n - 1)
        buffer.extend_from_slice(&[pixel.len() as u8, 0, 0, 0, 0, 0, 0, 0]); // bytesPlanes
        assert_eq!(buffer.len(), 132);
        // Level data: one texel
        buffer.extend_from_slice(pixel);
        buffer
    }

    /// Builds a minimal valid KTX2 file: a 1x1 white RGBA8 texture (`VkFormat` 37 =
    /// `R8G8B8A8_UNORM`) with the given raw `colorPrimaries` and `transferFunction`
    /// bytes.
    fn minimal_rgba8_ktx2(color_primaries: u8, transfer_function: u8) -> Vec<u8> {
        minimal_ktx2(
            37,
            color_primaries,
            transfer_function,
            &[255, 255, 255, 255],
        )
    }

    #[test]
    fn dfd_color_primaries_are_stamped_on_the_image() {
        for (color_primaries, expected) in [
            // Unspecified primaries fall back to the BT.709 default.
            (0, SourceColorPrimaries::Bt709),
            (1, SourceColorPrimaries::Bt709),
            (4, SourceColorPrimaries::Bt2020),
            (10, SourceColorPrimaries::DisplayP3),
            // BT.601 (EBU): carried by the file but unsupported; falls back to BT.709.
            (2, SourceColorPrimaries::Bt709),
        ] {
            let buffer = minimal_rgba8_ktx2(color_primaries, /* Linear */ 1);
            let image = ktx2_buffer_to_image(&buffer, CompressedImageFormats::empty(), false, None)
                .unwrap();
            assert_eq!(
                image.source_primaries, expected,
                "DFD colorPrimaries byte {color_primaries} should stamp {expected:?}",
            );
            assert_eq!(
                image.texture_descriptor.format,
                wgpu_types::TextureFormat::Rgba8Unorm,
                "stamping the primaries must not change the resolved texture format",
            );
        }
    }

    #[test]
    fn caller_is_srgb_still_wins_over_dfd_transfer_function() {
        // The file declares a linear transfer function, but the caller requests sRGB:
        // the caller wins (with a warning), byte-identical to the previous behavior.
        let buffer = minimal_rgba8_ktx2(/* BT709 */ 1, /* Linear */ 1);
        let image =
            ktx2_buffer_to_image(&buffer, CompressedImageFormats::empty(), true, None).unwrap();
        assert_eq!(
            image.texture_descriptor.format,
            wgpu_types::TextureFormat::Rgba8UnormSrgb,
        );
        assert_eq!(image.source_primaries, SourceColorPrimaries::Bt709);
    }

    #[test]
    fn linear_declared_r8_with_caller_is_srgb_is_decoded_on_the_cpu() {
        // The file declares a linear transfer function, but the caller requests sRGB.
        // R8 has no sRGB texture format, so the loader sRGB-decodes the data on the
        // CPU during transcoding (with a warning) and resolves to the non-sRGB format.
        let buffer = minimal_ktx2(
            /* R8_UNORM */ 9,
            /* BT709 */ 1,
            /* Linear */ 1,
            &[128],
        );
        let image =
            ktx2_buffer_to_image(&buffer, CompressedImageFormats::empty(), true, None).unwrap();
        assert_eq!(
            image.texture_descriptor.format,
            wgpu_types::TextureFormat::R8Unorm,
        );
        // sRGB-encoded byte 128 decodes to linear 55: the file's declared linear
        // transfer really is overridden by the caller's `is_srgb`.
        assert_eq!(image.data.as_deref(), Some(&[55u8][..]));
    }

    /// Builds a valid single-layer 2D KTX2 file with a full mip chain: `levels[i]` is
    /// the stored bytes of level `i` (already supercompressed when `supercompression`
    /// is not 0) paired with its uncompressed length. Levels are laid out in the file
    /// smallest-first as the spec requires, so the index is what a reader must trust.
    fn mip_chain_ktx2(
        vk_format: u32,
        width: u32,
        height: u32,
        bytes_per_texel: u8,
        supercompression: u32,
        levels: &[(Vec<u8>, u64)],
    ) -> Vec<u8> {
        let level_count = levels.len() as u32;
        let dfd_offset = 80 + 24 * level_count;
        let data_start = dfd_offset + 28;
        let mut offsets = vec![0u64; levels.len()];
        let mut cursor = data_start as u64;
        for (index, (stored, _)) in levels.iter().enumerate().rev() {
            offsets[index] = cursor;
            cursor += stored.len().div_ceil(4) as u64 * 4;
        }

        let mut buffer = Vec::new();
        buffer.extend_from_slice(&[
            0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
        ]);
        buffer.extend_from_slice(&vk_format.to_le_bytes());
        buffer.extend_from_slice(&1u32.to_le_bytes()); // typeSize
        buffer.extend_from_slice(&width.to_le_bytes());
        buffer.extend_from_slice(&height.to_le_bytes());
        buffer.extend_from_slice(&0u32.to_le_bytes()); // pixelDepth
        buffer.extend_from_slice(&0u32.to_le_bytes()); // layerCount
        buffer.extend_from_slice(&1u32.to_le_bytes()); // faceCount
        buffer.extend_from_slice(&level_count.to_le_bytes());
        buffer.extend_from_slice(&supercompression.to_le_bytes());
        buffer.extend_from_slice(&dfd_offset.to_le_bytes());
        buffer.extend_from_slice(&28u32.to_le_bytes()); // dfdByteLength
        buffer.extend_from_slice(&0u32.to_le_bytes()); // kvdByteOffset
        buffer.extend_from_slice(&0u32.to_le_bytes()); // kvdByteLength
        buffer.extend_from_slice(&0u64.to_le_bytes()); // sgdByteOffset
        buffer.extend_from_slice(&0u64.to_le_bytes()); // sgdByteLength
        for (index, (stored, uncompressed_length)) in levels.iter().enumerate() {
            buffer.extend_from_slice(&offsets[index].to_le_bytes());
            buffer.extend_from_slice(&(stored.len() as u64).to_le_bytes());
            buffer.extend_from_slice(&uncompressed_length.to_le_bytes());
        }
        assert_eq!(buffer.len() as u32, dfd_offset);
        buffer.extend_from_slice(&28u32.to_le_bytes()); // dfdTotalSize
        buffer.extend_from_slice(&0u32.to_le_bytes()); // vendorId | descriptorType (basic)
        buffer.extend_from_slice(&2u16.to_le_bytes()); // versionNumber
        buffer.extend_from_slice(&24u16.to_le_bytes()); // descriptorBlockSize (no samples)
        buffer.push(1); // colorModel = RGBSDA
        buffer.push(1); // colorPrimaries = BT709
        buffer.push(1); // transferFunction = Linear
        buffer.push(0); // flags
        buffer.extend_from_slice(&[0, 0, 0, 0]); // texelBlockDimension (stored as n - 1)
        buffer.extend_from_slice(&[bytes_per_texel, 0, 0, 0, 0, 0, 0, 0]); // bytesPlanes
        assert_eq!(buffer.len() as u32, data_start);
        for (index, (stored, _)) in levels.iter().enumerate().rev() {
            assert_eq!(buffer.len() as u64, offsets[index]);
            buffer.extend_from_slice(stored);
            buffer.resize(buffer.len().div_ceil(4) * 4, 0);
        }
        buffer
    }

    /// An 8x8 R8 chain of four levels, each level filled with its own index.
    fn r8_8x8_levels() -> Vec<(Vec<u8>, u64)> {
        (0..4u32)
            .map(|level| {
                let texels = ((8 >> level) * (8 >> level)) as usize;
                (vec![level as u8; texels], texels as u64)
            })
            .collect()
    }

    #[test]
    fn max_dimension_drops_leading_levels() {
        let buffer = mip_chain_ktx2(/* R8_UNORM */ 9, 8, 8, 1, 0, &r8_8x8_levels());
        let image =
            ktx2_buffer_to_image(&buffer, CompressedImageFormats::empty(), false, Some(2)).unwrap();
        let size = image.texture_descriptor.size;
        assert_eq!(
            (size.width, size.height, size.depth_or_array_layers),
            (2, 2, 1)
        );
        assert_eq!(image.texture_descriptor.mip_level_count, 2);
        // Level 2 (2x2) followed by level 3 (1x1), tightly packed.
        assert_eq!(image.data.as_deref(), Some(&[2, 2, 2, 2, 3][..]));
        assert_eq!(
            image.texture_descriptor.format,
            wgpu_types::TextureFormat::R8Unorm
        );

        // The transcoding path (R8 requested as sRGB) sees the same cut chain.
        let image =
            ktx2_buffer_to_image(&buffer, CompressedImageFormats::empty(), true, Some(2)).unwrap();
        assert_eq!(image.texture_descriptor.size.width, 2);
        assert_eq!(image.texture_descriptor.mip_level_count, 2);
        assert_eq!(image.data.as_ref().map(Vec::len), Some(5));
    }

    #[test]
    fn max_dimension_never_drops_the_smallest_level() {
        let buffer = mip_chain_ktx2(/* R8_UNORM */ 9, 8, 8, 1, 0, &r8_8x8_levels());
        for max_dimension in [0, 1] {
            let image = ktx2_buffer_to_image(
                &buffer,
                CompressedImageFormats::empty(),
                false,
                Some(max_dimension),
            )
            .unwrap();
            assert_eq!(image.texture_descriptor.size.width, 1);
            assert_eq!(image.texture_descriptor.mip_level_count, 1);
            assert_eq!(image.data.as_deref(), Some(&[3][..]));
        }

        // A bound the image already meets, or no bound at all, changes nothing.
        for max_dimension in [Some(8), Some(4096), None] {
            let image = ktx2_buffer_to_image(
                &buffer,
                CompressedImageFormats::empty(),
                false,
                max_dimension,
            )
            .unwrap();
            assert_eq!(image.texture_descriptor.size.width, 8);
            assert_eq!(image.texture_descriptor.mip_level_count, 4);
            assert_eq!(image.data.as_ref().map(Vec::len), Some(64 + 16 + 4 + 1));
        }

        // A single-level image is untouched by any bound.
        let single = mip_chain_ktx2(9, 8, 8, 1, 0, &r8_8x8_levels()[..1]);
        let image =
            ktx2_buffer_to_image(&single, CompressedImageFormats::empty(), false, Some(1)).unwrap();
        assert_eq!(image.texture_descriptor.size.width, 8);
        assert_eq!(image.texture_descriptor.mip_level_count, 1);
    }

    #[test]
    fn max_dimension_keeps_block_compressed_base_block_aligned() {
        // BC7 sRGB, 16x16 with a full chain down to 1x1; the two smallest levels are
        // padded to one 4x4 block each. A bound of 1 cannot start the chain at 1x1
        // because wgpu requires a block-aligned base, so it lands on 4x4.
        let block = [0u8; 16];
        let levels: Vec<(Vec<u8>, u64)> = [16u32, 8, 4, 2, 1]
            .into_iter()
            .map(|extent| {
                let blocks = extent.div_ceil(4).pow(2) as usize;
                (block.repeat(blocks), (blocks * 16) as u64)
            })
            .collect();
        let buffer = mip_chain_ktx2(/* BC7_SRGB_BLOCK */ 146, 16, 16, 16, 0, &levels);
        let image =
            ktx2_buffer_to_image(&buffer, CompressedImageFormats::BC, true, Some(1)).unwrap();
        assert_eq!(
            image.texture_descriptor.format,
            wgpu_types::TextureFormat::Bc7RgbaUnormSrgb
        );
        assert_eq!(image.texture_descriptor.size.width, 4);
        assert_eq!(image.texture_descriptor.size.height, 4);
        assert_eq!(image.texture_descriptor.mip_level_count, 3);
        assert_eq!(image.data.as_ref().map(Vec::len), Some(3 * 16));
    }

    #[test]
    fn max_dimension_keeps_texture_kind_when_an_axis_collapses() {
        // R8 8x2 strip: the chain is 8x2, 4x1, 2x1, 1x1. A bound of 2 starts at 2x1,
        // which must still bind as a 2D texture like the uncut file does.
        let levels: Vec<(Vec<u8>, u64)> = [(8u32, 2u32), (4, 1), (2, 1), (1, 1)]
            .into_iter()
            .map(|(w, h)| (vec![7u8; (w * h) as usize], u64::from(w * h)))
            .collect();
        let buffer = mip_chain_ktx2(9, 8, 2, 1, 0, &levels);
        let image =
            ktx2_buffer_to_image(&buffer, CompressedImageFormats::empty(), false, Some(2)).unwrap();
        assert_eq!(image.texture_descriptor.size.width, 2);
        assert_eq!(image.texture_descriptor.size.height, 1);
        assert_eq!(image.texture_descriptor.mip_level_count, 2);
        assert_eq!(
            image.texture_descriptor.dimension,
            wgpu_types::TextureDimension::D2
        );
        assert!(image.texture_view_descriptor.is_none());
        assert_eq!(image.data.as_ref().map(Vec::len), Some(2 + 1));
    }

    /// A zstd frame carrying `payload` as one raw (stored) block: a single-segment
    /// frame header with a one-byte content size, then a last-block header of type
    /// `Raw`. Any zstd decoder accepts it, and it needs no encoder to build.
    #[cfg(feature = "zstd")]
    fn zstd_stored_frame(payload: &[u8]) -> Vec<u8> {
        assert!(payload.len() < 256);
        let mut frame = vec![0x28, 0xb5, 0x2f, 0xfd, 0x20, payload.len() as u8];
        let block_header = 1 | ((payload.len() as u32) << 3);
        frame.extend_from_slice(&block_header.to_le_bytes()[..3]);
        frame.extend_from_slice(payload);
        frame
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn max_dimension_skips_decompressing_dropped_zstd_levels() {
        // The two dropped levels hold bytes that are not a zstd frame at all; the load
        // can only succeed if they are never handed to the decoder.
        let levels: Vec<(Vec<u8>, u64)> = r8_8x8_levels()
            .into_iter()
            .enumerate()
            .map(|(index, (texels, length))| {
                if index < 2 {
                    (vec![0xff; 7], length)
                } else {
                    (zstd_stored_frame(&texels), length)
                }
            })
            .collect();
        let buffer = mip_chain_ktx2(
            /* R8_UNORM */ 9, 8, 8, 1, /* Zstandard */ 2, &levels,
        );

        assert!(
            ktx2_buffer_to_image(&buffer, CompressedImageFormats::empty(), false, None).is_err(),
            "a full load must hit the corrupt levels",
        );
        let image =
            ktx2_buffer_to_image(&buffer, CompressedImageFormats::empty(), false, Some(2)).unwrap();
        assert_eq!(image.texture_descriptor.size.width, 2);
        assert_eq!(image.texture_descriptor.size.height, 2);
        assert_eq!(image.texture_descriptor.mip_level_count, 2);
        assert_eq!(image.data.as_deref(), Some(&[2, 2, 2, 2, 3][..]));
    }

    /// Loads a real 4096x4096, 13-level, BC7 sRGB, zstd-supercompressed KTX2 from
    /// NVIDIA's Zorah texture set at 1024. Run with the file's path in
    /// `BEVY_KTX2_MAX_DIMENSION_FIXTURE`; ignored because the fixture is not in tree.
    #[cfg(feature = "zstd")]
    #[test]
    #[ignore = "needs a 4096x4096 BC7 zstd KTX2 fixture named by BEVY_KTX2_MAX_DIMENSION_FIXTURE"]
    fn zorah_base_color_loads_at_1024() {
        let path = std::env::var("BEVY_KTX2_MAX_DIMENSION_FIXTURE")
            .expect("BEVY_KTX2_MAX_DIMENSION_FIXTURE must name the fixture");
        let buffer = std::fs::read(&path).unwrap();
        let image =
            ktx2_buffer_to_image(&buffer, CompressedImageFormats::BC, true, Some(1024)).unwrap();
        assert_eq!(
            image.texture_descriptor.format,
            wgpu_types::TextureFormat::Bc7RgbaUnormSrgb
        );
        let size = image.texture_descriptor.size;
        assert_eq!(
            (size.width, size.height, size.depth_or_array_layers),
            (1024, 1024, 1)
        );
        assert_eq!(image.texture_descriptor.mip_level_count, 11);
        let expected_len: usize = (0..11u32)
            .map(|level| (1024u32 >> level).max(1).div_ceil(4).pow(2) as usize * 16)
            .sum();
        assert_eq!(image.data.as_ref().map(Vec::len), Some(expected_len));
    }
}
