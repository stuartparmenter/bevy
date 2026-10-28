//! Blend weights for [`BloomScatterModel::Gt7Glare`](super::BloomScatterModel::Gt7Glare).
//!
//! The model follows "Physically Based Tone Mapping and Glare in Gran Turismo 7" (Polyphony
//! Digital, SIGGRAPH 2025), which builds lens glare from a weighted sum of blur levels. These
//! weights come from the diffraction pattern of a circular aperture (the Airy pattern), whose
//! energy within radius `r` is `E(v) = 1 - J0(v)^2 - J1(v)^2` with `v = pi * r / (lambda * N)`.
//!
//! Level `k` gets `E` between radii `2^k` and `2^(k+1)` level-0 texels, and level 0 also gets the
//! core. One level-0 texel is 2 microns. Each level is averaged over 400 to 700 nm, weighted by
//! the Gaussian fit `1.019 * exp(-285.4 * (lambda - 0.559)^2)` of the photopic luminosity
//! function (`lambda` in microns), and the weights for each f-number are normalized to sum to 1.

use bevy_math::{ops, FloatExt};

/// The number of pyramid levels the weights cover, which is the level count for the default
/// [`Bloom::max_mip_dimension`](super::Bloom::max_mip_dimension).
const LEVELS: usize = 8;

/// The f-numbers the table holds, in full stops.
const F_NUMBERS: [f32; 10] = [1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0];

pub(super) const DEFAULT_F_NUMBER: f32 = 5.6;

/// The fraction of glare energy each pyramid level receives, per entry of [`F_NUMBERS`].
#[rustfmt::skip]
const WEIGHTS: [[f32; LEVELS]; F_NUMBERS.len()] = [
    [0.9726225, 0.013796542, 0.006898271, 0.0034491355, 0.0017245677, 0.00086228386, 0.00043114193, 0.00021557097],
    [0.9613017, 0.019683303, 0.009658412, 0.004829206, 0.002414603, 0.0012073015, 0.00060365075, 0.00030182538],
    [0.9430178, 0.029814422, 0.0137995165, 0.0068997582, 0.0034498791, 0.0017249396, 0.0008624698, 0.0004312349],
    [0.91862756, 0.0429644, 0.019689245, 0.009661328, 0.004830664, 0.002415332, 0.001207666, 0.000603833],
    [0.88773185, 0.05569274, 0.029827286, 0.01380547, 0.006902735, 0.0034513676, 0.0017256838, 0.0008628419],
    [0.8402882, 0.07889436, 0.042990357, 0.019701142, 0.009667166, 0.004833583, 0.0024167914, 0.0012083957],
    [0.79718775, 0.09131077, 0.055740833, 0.029853044, 0.013817392, 0.006908696, 0.003454348, 0.001727174],
    [0.63371426, 0.20822804, 0.08005063, 0.042136833, 0.019235399, 0.009505614, 0.004752807, 0.0023764034],
    [0.39158967, 0.40697736, 0.09146875, 0.055837274, 0.029904693, 0.013841298, 0.006920649, 0.0034603246],
    [0.2332217, 0.40200213, 0.20872405, 0.08024132, 0.042237204, 0.019281218, 0.009528257, 0.0047641285],
];

fn level_weights(f_number: f32) -> [f32; LEVELS] {
    let n = if f_number.is_nan() {
        DEFAULT_F_NUMBER
    } else {
        f_number.clamp(F_NUMBERS[0], F_NUMBERS[F_NUMBERS.len() - 1])
    };
    let upper = F_NUMBERS
        .partition_point(|&entry| entry < n)
        .clamp(1, F_NUMBERS.len() - 1);
    let lower = upper - 1;
    // The f-numbers are spaced geometrically, so interpolate in log space.
    let t = ops::log2(n / F_NUMBERS[lower]) / ops::log2(F_NUMBERS[upper] / F_NUMBERS[lower]);
    core::array::from_fn(|k| WEIGHTS[lower][k].lerp(WEIGHTS[upper][k], t))
}

/// The upsampling blend constant for pyramid level `mip`, where 0 is the final composite.
///
/// Each pass lerps level `mip` into level `mip - 1`, so level `k` ends up scaled by
/// `b_0 * ... * b_k * (1 - b_(k+1))`. Ratios of tail sums make that equal
/// `intensity * weights[k]`. A shorter pyramid gives the remaining tail to its last level.
pub(super) fn blend_factor(f_number: f32, intensity: f32, mip: usize) -> f32 {
    if mip == 0 {
        return intensity.clamp(0.0, 1.0);
    }
    if mip >= LEVELS {
        return 0.0;
    }
    let weights = level_weights(f_number);
    let tail: f32 = weights[mip..].iter().sum();
    tail / (tail + weights[mip - 1])
}
