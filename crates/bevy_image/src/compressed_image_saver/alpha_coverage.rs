//! Mip chains that keep an alpha-tested texture's coverage.

use bevy_color::Srgba;

/// One level of a mip chain: straight-alpha RGBA8 texels.
pub struct MipLevel {
    pub texels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// The full mip chain of straight-alpha RGBA8 `texels` (`width` x `height`,
/// color sRGB-encoded when `srgb`), with each level's alpha scaled so the
/// share of it passing `cutoff` is as near the base level's as its alpha
/// values allow (Castaño, "Computing Alpha Mipmaps", 2010).
///
/// Coverage is measured as the GPU samples the texture, bilinearly between
/// texels: a level's isolated texels just above the cutoff pass only near
/// their centers, so counting texels would overstate it.
///
/// Levels are 2x2 box filtered from the unscaled level above, color in
/// linear light weighted by alpha, so transparent texels do not bleed into
/// opaque ones.
pub fn alpha_tested_mips(
    texels: &[u8],
    mut width: u32,
    mut height: u32,
    srgb: bool,
    cutoff: f32,
) -> Vec<MipLevel> {
    let linear = |c: u8| {
        let c = f32::from(c) / 255.0;
        if srgb {
            Srgba::gamma_function(c)
        } else {
            c
        }
    };
    let encoded = |c: f32| {
        to_u8(if srgb {
            Srgba::gamma_function_inverse(c)
        } else {
            c
        })
    };
    let mut level: Vec<[f32; 4]> = texels
        .chunks_exact(4)
        .map(|t| {
            [
                linear(t[0]),
                linear(t[1]),
                linear(t[2]),
                f32::from(t[3]) / 255.0,
            ]
        })
        .collect();
    let alpha = |level: &[[f32; 4]]| level.iter().map(|t| t[3]).collect::<Vec<_>>();
    let target = passing_share(&alpha(&level), width, height, cutoff, 1.0);
    let mut chain = vec![MipLevel {
        texels: texels.to_vec(),
        width,
        height,
    }];
    while width > 1 || height > 1 {
        (level, width, height) = downsample(&level, width, height);
        let scale = coverage_scale(&alpha(&level), width, height, cutoff, target);
        let texels = level
            .iter()
            .flat_map(|&[r, g, b, a]| [encoded(r), encoded(g), encoded(b), to_u8(a * scale)])
            .collect();
        chain.push(MipLevel {
            texels,
            width,
            height,
        });
    }
    chain
}

/// Subsamples per texel along each axis when measuring coverage.
const SUBSAMPLES: u32 = 4;

/// The share of `alpha` (`width` x `height`), times `scale` and clamped to
/// 1, that reaches `cutoff` when sampled bilinearly with clamped edges, from
/// `SUBSAMPLES` x `SUBSAMPLES` samples spread over each texel.
fn passing_share(alpha: &[f32], width: u32, height: u32, cutoff: f32, scale: f32) -> f32 {
    let scaled: Vec<f32> = alpha.iter().map(|a| (a * scale).min(1.0)).collect();
    // Each subsample's two neighboring texel indices and the weight of the
    // second, along one axis.
    let taps = |size: u32| {
        (0..size * SUBSAMPLES)
            .map(|s| {
                let at = (s as f32 + 0.5) / SUBSAMPLES as f32 - 0.5;
                let first = at.floor();
                let clamp = |i: f32| (i.max(0.0) as u32).min(size - 1) as usize;
                (clamp(first), clamp(first + 1.0), at - first)
            })
            .collect::<Vec<_>>()
    };
    let (columns, rows) = (taps(width), taps(height));
    let row = |y: usize| &scaled[y * width as usize..][..width as usize];
    let mut passing = 0usize;
    for &(y0, y1, ty) in &rows {
        let (top, bottom) = (row(y0), row(y1));
        for &(x0, x1, tx) in &columns {
            let upper = top[x0] + (top[x1] - top[x0]) * tx;
            let lower = bottom[x0] + (bottom[x1] - bottom[x0]) * tx;
            if upper + (lower - upper) * ty >= cutoff {
                passing += 1;
            }
        }
    }
    passing as f32 / (columns.len() * rows.len()) as f32
}

/// The alpha scale that brings the share of `alpha` (`width` x `height`)
/// passing `cutoff` nearest `target`. Coverage only changes in steps, so the
/// bisection's bounds end either side of the step that crosses `target` and
/// the nearer one wins; 12 steps resolve the scale finer than 8-bit alpha.
fn coverage_scale(alpha: &[f32], width: u32, height: u32, cutoff: f32, target: f32) -> f32 {
    let share = |scale| passing_share(alpha, width, height, cutoff, scale);
    let (mut low, mut high) = (0.0, 4.0 / cutoff);
    for _ in 0..12 {
        let mid = (low + high) * 0.5;
        if share(mid) < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    let error = |scale| (share(scale) - target).abs();
    if error(low) < error(high) {
        low
    } else {
        high
    }
}

/// `level` 2x2 box filtered to half size; an odd last row or column is
/// dropped, as in any halving mip chain.
fn downsample(level: &[[f32; 4]], width: u32, height: u32) -> (Vec<[f32; 4]>, u32, u32) {
    let (half_width, half_height) = ((width / 2).max(1), (height / 2).max(1));
    let mut half = Vec::with_capacity((half_width * half_height) as usize);
    for y in 0..half_height {
        for x in 0..half_width {
            let mut weighted = [0.0f32; 3];
            let mut plain = [0.0f32; 3];
            let (mut alpha, mut count) = (0.0, 0.0);
            for sy in (2 * y)..(2 * y + 2).min(height) {
                for sx in (2 * x)..(2 * x + 2).min(width) {
                    let texel = level[(sy * width + sx) as usize];
                    for c in 0..3 {
                        weighted[c] += texel[c] * texel[3];
                        plain[c] += texel[c];
                    }
                    alpha += texel[3];
                    count += 1.0;
                }
            }
            let [r, g, b] = if alpha > 0.0 {
                weighted.map(|c| c / alpha)
            } else {
                plain.map(|c| c / count)
            };
            half.push([r, g, b, alpha / count]);
        }
    }
    (half, half_width, half_height)
}

fn to_u8(c: f32) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 64x64 texture with an eighth of its texels opaque, scattered like
    /// leaves: plain box mips average them towards 0.125 alpha, under a 0.5
    /// cutoff.
    fn sparse_leaves() -> Vec<u8> {
        (0u32..64 * 64)
            .flat_map(|i| {
                // murmur3's finaliser: neighbouring texels land independently.
                let mut hash = i.wrapping_mul(0x9E37_79B9);
                hash = (hash ^ (hash >> 16)).wrapping_mul(0x85EB_CA6B);
                hash = (hash ^ (hash >> 13)).wrapping_mul(0xC2B2_AE35);
                hash ^= hash >> 16;
                [40, 120, 30, if hash % 8 == 0 { 255 } else { 0 }]
            })
            .collect()
    }

    fn coverage(level: &MipLevel) -> f32 {
        let alpha: Vec<f32> = level
            .texels
            .chunks_exact(4)
            .map(|t| f32::from(t[3]) / 255.0)
            .collect();
        passing_share(&alpha, level.width, level.height, 0.5, 1.0)
    }

    #[test]
    fn levels_keep_the_base_coverage() {
        let texels = sparse_leaves();
        let chain = alpha_tested_mips(&texels, 64, 64, true, 0.5);
        assert_eq!(chain.len(), 7);
        assert_eq!(chain[0].texels, texels);
        assert_eq!((chain[6].width, chain[6].height), (1, 1));
        let base = coverage(&chain[0]);
        // From 16x16 down each texel averages 16 or more, fine enough steps
        // of alpha to land near the base coverage (a 2x2 average only offers
        // quarters).
        for level in &chain[2..4] {
            let got = coverage(level);
            assert!(
                (got - base).abs() < 0.05,
                "{}x{}: {got} vs {base}",
                level.width,
                level.height
            );
        }
    }

    #[test]
    fn color_ignores_transparent_texels() {
        let texels: Vec<u8> = (0..16)
            .flat_map(|i| {
                if i == 0 {
                    [40, 120, 30, 255]
                } else {
                    [255, 0, 255, 0]
                }
            })
            .collect();
        let chain = alpha_tested_mips(&texels, 4, 4, true, 0.5);
        assert_eq!(&chain[2].texels[..3], &[40, 120, 30]);
    }

    #[test]
    fn odd_sizes_halve_down_to_one_texel() {
        let sizes: Vec<_> = alpha_tested_mips(&[255; 5 * 3 * 4], 5, 3, false, 0.5)
            .iter()
            .map(|level| (level.width, level.height))
            .collect();
        assert_eq!(sizes, [(5, 3), (2, 1), (1, 1)]);
    }
}
