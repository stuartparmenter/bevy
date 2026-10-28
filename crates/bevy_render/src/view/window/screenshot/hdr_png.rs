use alloc::borrow::Cow;
use bevy_color::{Chromaticity, RgbPrimaries};
use bevy_image::Image;
use bevy_math::Vec3;

use crate::transfer_functions::{gamut_matrix, pq_inverse_eotf_from_nits};

// libjxl's Rec2100PQ ICC profile, CC0.
const REC2100PQ_ICC: &[u8] = include_bytes!("rec2100pq.icc");

/// Encodes a linear BT.709 `Rgba32Float` image, where `1.0` is `paper_white_nits`, as a 16-bit
/// PNG with BT.2020 primaries and the PQ transfer function.
pub(super) fn encode(image: &Image, paper_white_nits: f32) -> Result<Vec<u8>, png::EncodingError> {
    let size = image.texture_descriptor.size;
    let data = image.data.as_deref().unwrap_or_default();
    let to_bt2020 = gamut_matrix(RgbPrimaries::BT709, RgbPrimaries::BT2020);

    let texels = data.len() / 16;
    let mut samples = Vec::with_capacity(texels * 6);
    let mut luminances = Vec::with_capacity(texels);
    for texel in data.chunks_exact(16) {
        let [r, g, b, _] = bytemuck::pod_read_unaligned::<[f32; 4]>(texel);
        let rgb = to_bt2020 * Vec3::new(r, g, b);
        let nits = rgb
            .to_array()
            .map(|value| value.max(0.0) * paper_white_nits);
        luminances.push(0.2627 * nits[0] + 0.6780 * nits[1] + 0.0593 * nits[2]);
        for nits in nits {
            // 10 bits match a 10-bit PQ swapchain, and the six zero low bits compress much
            // better.
            let q10 = (pq_inverse_eotf_from_nits(nits).clamp(0.0, 1.0) * 1023.0).round() as u16;
            samples.extend_from_slice(&(q10 << 6).to_be_bytes());
        }
    }

    write_png(
        size.width,
        size.height,
        &samples,
        content_light_level(&mut luminances),
    )
}

/// `(MaxCLL, MaxFALL)` for the `cLLi` chunk, in units of 0.0001 nits.
///
/// `MaxCLL` is the 99.99th percentile luminance, not the maximum. A few bright texels would
/// otherwise raise it, and viewers that tonemap to `MaxCLL` would dim the whole image.
fn content_light_level(luminances: &mut [f32]) -> (u32, u32) {
    if luminances.is_empty() {
        return (0, 0);
    }
    let index = ((luminances.len() - 1) as f64 * 0.9999) as usize;
    let (_, max_cll, _) = luminances.select_nth_unstable_by(index, f32::total_cmp);
    let max_cll = *max_cll;
    let average = luminances.iter().sum::<f32>() / luminances.len() as f32;
    let to_units = |nits: f32| (nits.max(0.0) as f64 * 10000.0).min(u32::MAX as f64) as u32;
    (to_units(max_cll), to_units(average))
}

/// Writes big-endian 16-bit RGB PQ samples as a PNG with these chunks:
///
/// - `cICP` `[9, 16, 0, 1]`: BT.2020 primaries, PQ, RGB, full range.
/// - `cLLi`: without it, viewers assume 10000 nit content and tonemap the image down.
/// - `iCCP` and `cHRM`: for decoders and re-encoders that ignore `cICP`.
/// - `sBIT`: the samples have 10 significant bits.
///
/// [`png::Info`] has no fields for `cICP`, `cLLi` and `sBIT`, so they are written as raw chunks.
fn write_png(
    width: u32,
    height: u32,
    samples: &[u8],
    (max_cll, max_fall): (u32, u32),
) -> Result<Vec<u8>, png::EncodingError> {
    let mut info = png::Info::with_size(width, height);
    info.bit_depth = png::BitDepth::Sixteen;
    info.color_type = png::ColorType::Rgb;
    info.icc_profile = Some(Cow::Borrowed(REC2100PQ_ICC));
    let xy = |c: Chromaticity| (c.x, c.y);
    let bt2020 = RgbPrimaries::BT2020;
    info.source_chromaticities = Some(png::SourceChromaticities::new(
        xy(bt2020.white),
        xy(bt2020.red),
        xy(bt2020.green),
        xy(bt2020.blue),
    ));

    let mut clli = [0u8; 8];
    clli[..4].copy_from_slice(&max_cll.to_be_bytes());
    clli[4..].copy_from_slice(&max_fall.to_be_bytes());

    let mut out = Vec::new();
    let mut encoder = png::Encoder::with_info(&mut out, info)?;
    encoder.set_compression(png::Compression::High);
    encoder.set_filter(png::Filter::Adaptive);
    let mut writer = encoder.write_header()?;
    writer.write_chunk(png::chunk::cICP, &[9, 16, 0, 1])?;
    writer.write_chunk(png::chunk::cLLI, &clli)?;
    writer.write_chunk(png::chunk::sBIT, &[10, 10, 10])?;

    writer.write_image_data(samples)?;
    writer.finish()?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_carries_hdr_chunks_and_samples() {
        let samples: Vec<u8> = (0..12u16)
            .flat_map(|i| ((i * 80) << 6).to_be_bytes())
            .collect();
        let out = write_png(2, 2, &samples, (10_000_000, 2_000_000)).unwrap();

        let mut reader = png::Decoder::new(std::io::Cursor::new(&out))
            .read_info()
            .unwrap();
        let info = reader.info();

        let cicp = info.coding_independent_code_points.unwrap();
        assert_eq!(
            [
                cicp.color_primaries,
                cicp.transfer_function,
                cicp.matrix_coefficients
            ],
            [9, 16, 0]
        );
        assert!(cicp.is_video_full_range_image);
        let clli = info.content_light_level.unwrap();
        assert_eq!(clli.max_content_light_level, 10_000_000);
        assert_eq!(clli.max_frame_average_light_level, 2_000_000);
        assert_eq!(info.icc_profile.as_deref(), Some(REC2100PQ_ICC));
        assert_eq!(info.sbit.as_deref(), Some([10u8, 10, 10].as_slice()));
        assert_eq!(
            info.chromaticities(),
            Some(png::SourceChromaticities::new(
                (0.3127, 0.3290),
                (0.708, 0.292),
                (0.170, 0.797),
                (0.131, 0.046),
            ))
        );

        let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
        let frame = reader.next_frame(&mut buf).unwrap();
        assert_eq!(&buf[..frame.buffer_size()], samples);
    }
}
