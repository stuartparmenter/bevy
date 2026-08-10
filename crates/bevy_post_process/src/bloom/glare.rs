//! Physically derived veiling-glare weights for the bloom pyramid
//! ([`BloomScatterModel::Gt7Glare`](super::BloomScatterModel::Gt7Glare)).
//!
//! Gran Turismo 7 replaces bloom with a glare that approximates the camera's
//! far-field (Fraunhofer) diffraction point-spread function as a weighted sum
//! of progressively blurred buffers, with per-level weights that depend on the
//! aperture F-number (SIGGRAPH 2025 PBS course, "Physically Based Tone Mapping
//! and Glare in Gran Turismo 7", Polyphony Digital, slides 177-187). Polyphony's
//! 240 hand-calibrated weights are not published, so this module derives its own
//! from the same physical model.
//!
//! The Fraunhofer pattern of an ideal circular aperture is the Airy pattern. Its
//! encircled energy, the fraction of the PSF energy within radius `r` of the
//! center, is (Rayleigh; Born & Wolf, *Principles of Optics*, section 8.5.2)
//!
//! ```text
//! E(v) = 1 - J0(v)^2 - J1(v)^2,     v = pi*r / (lambda*N)
//! ```
//!
//! where `J0`/`J1` are Bessel functions of the first kind, `lambda` is the
//! wavelength and `N` the F-number. Pyramid level `k` reproduces scattering
//! radii of `[t*2^k, t*2^(k+1))` around a bright source, where `t` is the size
//! of a level-0 texel, so its energy is the encircled-energy difference over
//! that annulus. Level 0 also absorbs the core, `r < t`. Each band is integrated
//! over the visible range, weighting each wavelength by a single-Gaussian
//! approximation of the CIE 1924 photopic luminosity function
//! `V(lambda) ~= 1.019*exp(-285.4*(lambda - 0.559)^2)` (lambda in microns). That
//! also smooths the monochromatic Airy rings, as a polychromatic PSF does.
//!
//! Asymptotically `J0(v)^2 + J1(v)^2 -> 2/(pi*v)` (Abramowitz & Stegun 9.2.1),
//! so far from the core each octave-spaced level receives half the energy of the
//! previous one.
//!
//! The bands cover the full PSF and the table is normalized per F-number, so
//! [`Bloom::intensity`](super::Bloom::intensity) stays the fraction of energy
//! scattered out of the sharp image, and the F-number controls only how that
//! energy spreads across the pyramid levels.
//!
//! Mapping image-plane microns to pyramid texels needs a virtual sensor scale:
//! one level-0 texel is 2 microns across. That is the one tuning constant of the
//! derivation, chosen so the f/1 to f/22 ladder sweeps the Airy core from well
//! below one texel to a few texels at the pyramid's reference resolution of 512
//! rows.
//!
//! The weights are achromatic, one per level rather than one per channel.
//! GT7-style chromatic dispersion would triple the upsample cost.
//!
//! [`GLARE_WEIGHT_TABLE`] holds the finished weights as literals: the Bessel
//! quadrature behind them costs a few milliseconds, too much to spend at startup
//! for a model most apps never enable. The derivation lives in this module's
//! tests, and `tests::table_matches_derivation` fails until changed literals are
//! regenerated.

use bevy_math::ops;
use bevy_utils::once;
use tracing::warn;

/// Pyramid levels (octave-spaced annular bands) the weight table covers. Matches
/// the default chain depth (`max_mip_dimension = 512`, 8 mips); [`blend_factor`]
/// handles other depths.
pub(crate) const GLARE_BANDS: usize = 8;

/// The standard full-stop aperture ladder the weight table is precomputed for.
pub(crate) const F_NUMBER_LADDER: [f32; 10] = [1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0];

/// The F-number substituted for non-finite or non-positive
/// [`Gt7Glare::f_number`](super::BloomScatterModel::Gt7Glare) values.
pub(crate) const DEFAULT_F_NUMBER: f32 = 5.6;

/// For each entry of [`F_NUMBER_LADDER`], the normalized energy fraction each
/// pyramid level receives. Baked from the derivation in this module's tests.
static GLARE_WEIGHT_TABLE: [[f32; GLARE_BANDS]; F_NUMBER_LADDER.len()] = [
    // f/1.0
    [
        0.9726225,
        0.013796542,
        0.006898271,
        0.0034491355,
        0.0017245677,
        0.00086228386,
        0.00043114193,
        0.00021557097,
    ],
    // f/1.4
    [
        0.9613017,
        0.019683303,
        0.009658412,
        0.004829206,
        0.002414603,
        0.0012073015,
        0.00060365075,
        0.00030182538,
    ],
    // f/2.0
    [
        0.9430178,
        0.029814422,
        0.0137995165,
        0.0068997582,
        0.0034498791,
        0.0017249396,
        0.0008624698,
        0.0004312349,
    ],
    // f/2.8
    [
        0.91862756,
        0.0429644,
        0.019689245,
        0.009661328,
        0.004830664,
        0.002415332,
        0.001207666,
        0.000603833,
    ],
    // f/4.0
    [
        0.88773185,
        0.05569274,
        0.029827286,
        0.01380547,
        0.006902735,
        0.0034513676,
        0.0017256838,
        0.0008628419,
    ],
    // f/5.6
    [
        0.8402882,
        0.07889436,
        0.042990357,
        0.019701142,
        0.009667166,
        0.004833583,
        0.0024167914,
        0.0012083957,
    ],
    // f/8.0
    [
        0.79718775,
        0.09131077,
        0.055740833,
        0.029853044,
        0.013817392,
        0.006908696,
        0.003454348,
        0.001727174,
    ],
    // f/11
    [
        0.63371426,
        0.20822804,
        0.08005063,
        0.042136833,
        0.019235399,
        0.009505614,
        0.004752807,
        0.0023764034,
    ],
    // f/16
    [
        0.39158967,
        0.40697736,
        0.09146875,
        0.055837274,
        0.029904693,
        0.013841298,
        0.006920649,
        0.0034603246,
    ],
    // f/22
    [
        0.2332217,
        0.40200213,
        0.20872405,
        0.08024132,
        0.042237204,
        0.019281218,
        0.009528257,
        0.0047641285,
    ],
];

fn sanitize_f_number(f_number: f32) -> f32 {
    if f_number.is_finite() && f_number > 0.0 {
        f_number
    } else {
        once!(warn!(
            "BloomScatterModel::Gt7Glare f_number must be finite and positive (got {f_number}); \
            using f/{DEFAULT_F_NUMBER}"
        ));
        DEFAULT_F_NUMBER
    }
}

/// Normalized per-level glare weights for an arbitrary F-number. Interpolates
/// [`GLARE_WEIGHT_TABLE`] linearly in `log2(N)` to match the ladder's geometric
/// spacing, and clamps to the ladder ends. Linear interpolation of normalized
/// weight vectors stays normalized.
pub(crate) fn mip_weights(f_number: f32) -> [f32; GLARE_BANDS] {
    let table = &GLARE_WEIGHT_TABLE;
    let n = sanitize_f_number(f_number).clamp(
        F_NUMBER_LADDER[0],
        F_NUMBER_LADDER[F_NUMBER_LADDER.len() - 1],
    );
    let upper = F_NUMBER_LADDER.iter().position(|&entry| n <= entry);
    let Some(upper) = upper else {
        // Unreachable after the clamp; be safe anyway.
        return table[F_NUMBER_LADDER.len() - 1];
    };
    if upper == 0 {
        return table[0];
    }
    let lower = upper - 1;
    let t = (ops::log2(n) - ops::log2(F_NUMBER_LADDER[lower]))
        / (ops::log2(F_NUMBER_LADDER[upper]) - ops::log2(F_NUMBER_LADDER[lower]));
    let mut weights = [0.0f32; GLARE_BANDS];
    for (k, weight) in weights.iter_mut().enumerate() {
        *weight = table[lower][k] * (1.0 - t) + table[upper][k] * t;
    }
    weights
}

/// The upsample blend constant for the glare model, used in
/// `compute_blend_factor` in place of the
/// [`Aesthetic`](super::BloomScatterModel::Aesthetic) curve.
///
/// The bloom node composites the pyramid bottom-up through chained
/// energy-conserving lerps (`out = lerp(dst, src, blend)`), so the final image
/// is
///
/// ```text
/// (1 - b0)*image + sum_k b0*...*bk*(1 - b[k+1])*level_k
/// ```
///
/// Solving for the constants that realize the target per-level weights
/// `intensity*w_k`, and `1 - intensity` for the sharp image, gives tail-sum
/// ratios: with `T_j = sum of w_k over k >= j`,
///
/// ```text
/// b0  = intensity*T_0 = intensity   (the final pass)
/// b_j = T_j / T_(j-1)               (level j blended into level j-1)
/// ```
///
/// Chains shallower than [`GLARE_BANDS`] fold the wide-band tail into their
/// deepest level; deeper chains blend the extra levels with weight zero. `mip`
/// follows `compute_blend_factor`'s convention: 0 is the final composite onto
/// the view target.
pub(crate) fn blend_factor(f_number: f32, intensity: f32, mip: u32) -> f32 {
    let intensity = if intensity.is_finite() {
        intensity.clamp(0.0, 1.0)
    } else {
        0.0
    };
    if mip == 0 {
        return intensity;
    }
    let j = mip as usize;
    if j >= GLARE_BANDS {
        return 0.0;
    }
    let weights = mip_weights(f_number);
    let tail_prev: f32 = weights[j - 1..].iter().sum();
    let tail: f32 = weights[j..].iter().sum();
    if tail_prev <= f32::MIN_POSITIVE {
        0.0
    } else {
        (tail / tail_prev).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_platform::sync::LazyLock;
    use core::f64::consts::PI;

    /// Virtual sensor pitch of one pyramid level-0 texel, in micrometers.
    /// See the module docs.
    const TEXEL_PITCH_MICRONS: f64 = 2.0;

    /// Wavelength integration range (microns) and sample count for the
    /// polychromatic PSF. Photopic weight is negligible outside 400-700 nm.
    const LAMBDA_MIN_MICRONS: f64 = 0.40;
    const LAMBDA_MAX_MICRONS: f64 = 0.70;
    const WAVELENGTH_SAMPLES: usize = 16;

    /// Single-Gaussian approximation of the CIE 1924 photopic luminosity
    /// function `V(lambda)`, lambda in micrometers (peak 559 nm, sigma ~42 nm).
    fn photopic_weight(lambda_microns: f64) -> f64 {
        let d = lambda_microns - 0.559;
        1.019 * (-285.4 * d * d).exp()
    }

    /// Bessel function of the first kind `J_n(x)` via Simpson integration of
    /// Bessel's integral `J_n(x) = (1/pi) * integral_0^pi cos(n*t - x*sin t) dt`.
    /// The integrand extends to a smooth periodic function, so the rule
    /// converges geometrically once the sample count exceeds `x`.
    fn bessel_j(n: u32, x: f64) -> f64 {
        // 128 (even) keeps ~1e-16 absolute error up to the seam at x = 16 and
        // gives a bit-identical f32 table to 512 intervals. Below ~64 it drifts.
        const INTERVALS: usize = 128;
        let h = PI / INTERVALS as f64;
        let f = |tau: f64| (f64::from(n) * tau - x * tau.sin()).cos();
        let mut sum = f(0.0) + f(PI);
        for i in 1..INTERVALS {
            let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
            sum += weight * f(i as f64 * h);
        }
        sum * h / (3.0 * PI)
    }

    /// The radius beyond which [`airy_encircled_energy`] switches from the
    /// exact Bessel form to the `1/v` tail.
    const AIRY_ASYMPTOTIC_SEAM: f64 = 16.0;

    /// Fraction of the Airy pattern's total energy within the dimensionless
    /// radius `v = pi*r/(lambda*N)`: `E(v) = 1 - J0(v)^2 - J1(v)^2`.
    ///
    /// For `v >=` [`AIRY_ASYMPTOTIC_SEAM`] this uses the ring-averaged
    /// asymptotic `J0^2 + J1^2 -> 2/(pi*v)` (relative error `O(v^-2)`), with its
    /// constant calibrated so the two branches meet at the seam. That keeps `E`
    /// continuous and monotonic, which the band-energy differences rely on.
    fn airy_encircled_energy(v: f64) -> f64 {
        // seam * (J0(seam)^2 + J1(seam)^2), ~= 2/pi up to the ring residual.
        static TAIL_CONSTANT: LazyLock<f64> = LazyLock::new(|| {
            let j0 = bessel_j(0, AIRY_ASYMPTOTIC_SEAM);
            let j1 = bessel_j(1, AIRY_ASYMPTOTIC_SEAM);
            AIRY_ASYMPTOTIC_SEAM * (j0 * j0 + j1 * j1)
        });
        if v <= 0.0 {
            return 0.0;
        }
        if v < AIRY_ASYMPTOTIC_SEAM {
            let j0 = bessel_j(0, v);
            let j1 = bessel_j(1, v);
            1.0 - j0 * j0 - j1 * j1
        } else {
            1.0 - *TAIL_CONSTANT / v
        }
    }

    /// Raw (un-normalized) photopically weighted energy fractions of the
    /// polychromatic Airy PSF over each pyramid level's annulus
    /// `r` in `[t*2^k, t*2^(k+1))`, `t =` [`TEXEL_PITCH_MICRONS`].
    ///
    /// Level 0's band extends down to the center (`[0, 2t)`), so the core is
    /// part of the finest level. Excluding a sub-texel core instead makes the
    /// normalized shape non-monotonic in N as the Airy bulk crosses the cutoff.
    fn raw_band_energies(f_number: f64) -> [f64; GLARE_BANDS] {
        let mut bands = [0.0; GLARE_BANDS];
        let mut total_weight = 0.0;
        for i in 0..WAVELENGTH_SAMPLES {
            let lambda = LAMBDA_MIN_MICRONS
                + (LAMBDA_MAX_MICRONS - LAMBDA_MIN_MICRONS) * i as f64
                    / (WAVELENGTH_SAMPLES - 1) as f64;
            let weight = photopic_weight(lambda);
            total_weight += weight;
            let v_per_micron = PI / (lambda * f_number);
            for (k, band) in bands.iter_mut().enumerate() {
                let r_inner = if k == 0 {
                    0.0
                } else {
                    TEXEL_PITCH_MICRONS * f64::powi(2.0, k as i32)
                };
                let r_outer = TEXEL_PITCH_MICRONS * f64::powi(2.0, k as i32 + 1);
                *band += weight
                    * (airy_encircled_energy(v_per_micron * r_outer)
                        - airy_encircled_energy(v_per_micron * r_inner));
            }
        }
        for band in &mut bands {
            *band /= total_weight;
        }
        bands
    }

    /// [`raw_band_energies`] normalized to sum to 1.
    fn normalized_band_weights(f_number: f64) -> [f32; GLARE_BANDS] {
        let raw = raw_band_energies(f_number);
        let sum: f64 = raw.iter().sum();
        let mut weights = [0.0f32; GLARE_BANDS];
        for (weight, raw) in weights.iter_mut().zip(raw) {
            *weight = (raw / sum) as f32;
        }
        weights
    }

    /// `J0`/`J1` against reference values (Abramowitz & Stegun, table 9.1).
    #[test]
    fn bessel_reference_values() {
        let cases = [
            (0, 0.0, 1.0),
            (0, 1.0, 0.765_197_686_6),
            (0, 2.0, 0.223_890_779_1),
            (0, 5.0, -0.177_596_771_3),
            (1, 0.0, 0.0),
            (1, 1.0, 0.440_050_585_7),
            (1, 2.0, 0.576_724_807_8),
            (1, 5.0, -0.327_579_137_6),
        ];
        for (n, x, expected) in cases {
            assert!(
                (bessel_j(n, x) - expected).abs() < 1e-9,
                "J{n}({x}) = {} != {expected}",
                bessel_j(n, x)
            );
        }
        // First zeros: J0(2.404826), J1(3.831706).
        assert!(bessel_j(0, 2.404_825_557_7).abs() < 1e-9);
        assert!(bessel_j(1, 3.831_705_970_2).abs() < 1e-9);
    }

    #[test]
    fn encircled_energy_is_a_cdf() {
        assert_eq!(airy_encircled_energy(0.0), 0.0);
        let mut previous = 0.0;
        // Step across the exact/asymptotic seam at v = 16 too.
        for i in 1..=4000 {
            let v = i as f64 * 0.01;
            let e = airy_encircled_energy(v);
            assert!(
                e >= previous - 1e-9,
                "encircled energy not monotonic at v = {v}"
            );
            assert!(e < 1.0);
            previous = e;
        }
        // Born & Wolf: E(first zero of J1, v = 3.8317) ~= 0.8378.
        assert!((airy_encircled_energy(3.831_705_970_2) - 0.8378).abs() < 1e-3);
    }

    #[test]
    fn raw_energies_conserve_and_cover_the_psf() {
        let mut previous_total = f64::INFINITY;
        let mut previous_core_share = f64::INFINITY;
        for n in F_NUMBER_LADDER {
            let raw = raw_band_energies(f64::from(n));
            let total: f64 = raw.iter().sum();
            assert!(raw.iter().all(|&w| w >= 0.0), "negative weight at f/{n}");
            assert!(total <= 1.0, "f/{n} scatters more than total energy");
            assert!(total > 0.99, "f/{n} bands cover only {total} of the PSF");
            assert!(
                total < previous_total,
                "residual beyond the table not growing at f/{n}"
            );
            assert!(
                raw[0] < previous_core_share,
                "level-0 energy not draining outward at f/{n}"
            );
            previous_total = total;
            previous_core_share = raw[0];
        }
    }

    /// Regenerate the baked literals if this fails.
    #[test]
    fn table_matches_derivation() {
        for (i, n) in F_NUMBER_LADDER.into_iter().enumerate() {
            let derived = normalized_band_weights(f64::from(n));
            assert_eq!(
                GLARE_WEIGHT_TABLE[i], derived,
                "f/{n}: baked table entry does not match the derivation"
            );
        }
    }

    /// At f/16 and up the core's bulk ring crosses into level 1 and the peak
    /// moves off the finest level, so falloff is not asserted there.
    #[test]
    fn table_normalized_and_monotonic_falloff_through_f11() {
        for (i, n) in F_NUMBER_LADDER.into_iter().enumerate() {
            let weights = GLARE_WEIGHT_TABLE[i];
            let sum: f32 = weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "f/{n} weights sum to {sum}");
            if n <= 11.0 {
                for k in 0..GLARE_BANDS - 1 {
                    assert!(
                        weights[k] > weights[k + 1],
                        "f/{n}: weight[{k}] = {} <= weight[{}] = {}",
                        weights[k],
                        k + 1,
                        weights[k + 1]
                    );
                }
            }
        }
    }

    #[test]
    fn spread_increases_with_f_number() {
        let mean_level = |weights: &[f32; GLARE_BANDS]| -> f32 {
            weights.iter().enumerate().map(|(k, w)| k as f32 * w).sum()
        };
        let mut previous_mean = mean_level(&GLARE_WEIGHT_TABLE[0]);
        for (i, n) in F_NUMBER_LADDER.into_iter().enumerate().skip(1) {
            let mean = mean_level(&GLARE_WEIGHT_TABLE[i]);
            assert!(
                mean > previous_mean,
                "mean level not increasing at f/{n}: {mean} <= {previous_mean}"
            );
            previous_mean = mean;
        }
        let first = mean_level(&GLARE_WEIGHT_TABLE[0]);
        let last = mean_level(&GLARE_WEIGHT_TABLE[F_NUMBER_LADDER.len() - 1]);
        assert!(last - first > 1.0, "f/1 {first} -> f/22 {last}");
    }

    #[test]
    fn f_number_interpolation() {
        for (i, n) in F_NUMBER_LADDER.into_iter().enumerate() {
            assert_eq!(mip_weights(n), GLARE_WEIGHT_TABLE[i]);
        }
        assert_eq!(mip_weights(0.25), GLARE_WEIGHT_TABLE[0]);
        assert_eq!(
            mip_weights(1000.0),
            GLARE_WEIGHT_TABLE[F_NUMBER_LADDER.len() - 1]
        );
        // Between f/4 and f/5.6 every band stays within the bracket values.
        let mid = mip_weights(4.75);
        let (lo, hi) = (GLARE_WEIGHT_TABLE[4], GLARE_WEIGHT_TABLE[5]);
        for k in 0..GLARE_BANDS {
            let (min, max) = (lo[k].min(hi[k]), lo[k].max(hi[k]));
            assert!(mid[k] >= min - 1e-7 && mid[k] <= max + 1e-7);
        }
        assert!((mid.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        // Continuity at a ladder entry.
        let just_below = mip_weights(5.6 - 1e-4);
        for k in 0..GLARE_BANDS {
            assert!((just_below[k] - GLARE_WEIGHT_TABLE[5][k]).abs() < 1e-3);
        }
    }

    #[test]
    fn invalid_f_number_degrades_to_default() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -2.8] {
            assert_eq!(mip_weights(bad), mip_weights(DEFAULT_F_NUMBER));
        }
    }

    /// Reconstructs the per-level contributions from the chained lerp blend
    /// constants and checks they reproduce `intensity*w_k`. The mip counts cover
    /// the full 8-level chain, a shallower chain, and a deeper one.
    #[test]
    fn blend_constants_reproduce_weights() {
        let f_number = 4.0;
        let intensity = 0.3;
        let weights = mip_weights(f_number);

        for mip_count in [4usize, 8, 10] {
            let levels = mip_count; // blend factors used: mip = 0..mip_count
            let blends: Vec<f32> = (0..levels)
                .map(|mip| blend_factor(f_number, intensity, mip as u32))
                .collect();

            let mut product = 1.0f32;
            let mut contributions = Vec::new();
            for k in 0..levels {
                product *= blends[k];
                let next = if k + 1 < levels { blends[k + 1] } else { 0.0 };
                contributions.push(product * (1.0 - next));
            }
            // The deepest level is never blended into, so it keeps the product.
            let last = contributions.len() - 1;
            contributions[last] = product;

            let total: f32 = contributions.iter().sum();
            assert!(
                (total - intensity).abs() < 1e-6,
                "mip_count {mip_count}: scattered total {total} != intensity"
            );
            for (k, contribution) in contributions.iter().enumerate() {
                let expected = if k + 1 < mip_count.min(GLARE_BANDS) {
                    intensity * weights[k]
                } else if k == mip_count.min(GLARE_BANDS) - 1 {
                    // Deepest represented level: the folded tail.
                    intensity * weights[k..].iter().sum::<f32>()
                } else {
                    0.0
                };
                assert!(
                    (contribution - expected).abs() < 1e-6,
                    "mip_count {mip_count}, level {k}: {contribution} != {expected}"
                );
            }
        }
    }

    #[test]
    fn blend_factor_degenerate_inputs() {
        assert_eq!(blend_factor(5.6, f32::NAN, 0), 0.0);
        assert_eq!(blend_factor(5.6, 2.0, 0), 1.0);
        assert_eq!(blend_factor(5.6, -1.0, 0), 0.0);
        assert_eq!(blend_factor(5.6, 0.5, GLARE_BANDS as u32 + 5), 0.0);
        for n in F_NUMBER_LADDER {
            for mip in 0..12 {
                let b = blend_factor(n, 0.7, mip);
                assert!((0.0..=1.0).contains(&b), "f/{n} mip {mip}: {b}");
            }
        }
    }
}
