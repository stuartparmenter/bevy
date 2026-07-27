//! Physically derived veiling-glare weights for the bloom pyramid
//! ([`BloomScatterModel::Gt7Glare`](super::BloomScatterModel::Gt7Glare)).
//!
//! # The model
//!
//! Gran Turismo 7 ships no separate bloom: its "glare" *is* the bloom, an
//! approximation of the camera's far-field (Fraunhofer) diffraction
//! point-spread function by a weighted sum of progressively blurred buffers
//! — the same downsample/upsample pyramid Bevy's bloom already builds — with
//! per-level composite weights that depend on the aperture F-number
//! (SIGGRAPH 2025 PBS course, "Physically Based Tone Mapping and Glare in
//! Gran Turismo 7", Polyphony Digital, slides 177–187). Polyphony's 240
//! hand-calibrated weights (per-level × per-channel × 10 F-numbers) are not
//! published, so this module does **not** clone them. Its weights come from
//! the same physical model the talk describes:
//!
//! The Fraunhofer diffraction pattern of an ideal circular aperture is the
//! Airy pattern. Its *encircled energy* — the fraction of the PSF's total
//! energy within radius `r` of the center — has the closed form (Rayleigh;
//! see Born & Wolf, *Principles of Optics*, §8.5.2)
//!
//! ```text
//! E(v) = 1 − J0(v)² − J1(v)²,     v = π·r / (λ·N)
//! ```
//!
//! where `J0`/`J1` are Bessel functions of the first kind, `λ` is the
//! wavelength and `N` the F-number. Each bloom pyramid level `k` reproduces
//! scattering radii of roughly `[t·2ᵏ, t·2ᵏ⁺¹)` around a bright source,
//! where `t` is the size of a level-0 texel, so the physically correct
//! energy for level `k` is the encircled-energy difference over that
//! annulus (level 0 also absorbs the central core, `r < t`, so the whole
//! PSF is accounted for). The weights are integrated over the visible band,
//! weighting each wavelength by a single-Gaussian approximation of the CIE
//! 1924 photopic luminosity function
//! `V(λ) ≈ 1.019·exp(−285.4·(λ − 0.559)²)` (λ in µm), which also smooths
//! out the monochromatic Airy rings the way a real polychromatic PSF does.
//!
//! Two consequences of the physics worth knowing:
//!
//! - **Stopping down spreads the glare.** The Airy pattern scales linearly
//!   with `λ·N`, so a *large* F-number (small aperture, e.g. f/22) pushes
//!   energy into wide pyramid levels (the photographic "starburst"/diffraction
//!   softening regime), while a small F-number (f/1.0) keeps the PSF
//!   sub-texel — nearly all energy lands in the finest blur level and the
//!   glare is, correctly, almost invisible.
//! - **The Airy tail loses half its energy per octave.** Asymptotically
//!   `J0(v)² + J1(v)² → 2/(π·v)` (Abramowitz & Stegun 9.2.1), so far from the
//!   core each successive (octave-spaced) pyramid level receives half the
//!   energy of the previous one — the heavy tail that distinguishes
//!   physically based glare from a Gaussian blur.
//!
//! # Normalization
//!
//! The bands cover the full PSF (the residual beyond the widest band is
//! < 0.5% even at f/22), and the table is normalized per F-number, so the
//! [`Bloom::intensity`](super::Bloom::intensity) dial keeps its meaning as
//! the total fraction of energy scattered out of the sharp image (the chain
//! stays energy-conserving) and the F-number controls only how that energy
//! is distributed across the pyramid levels. The distribution responds
//! monotonically: the energy-weighted mean blur radius strictly grows as
//! the aperture stops down.
//!
//! # Calibration
//!
//! Mapping image-plane microns to pyramid texels requires a virtual sensor
//! scale: one pyramid level-0 texel is 2 µm across. It is the one perceptual
//! tuning constant of the derivation, chosen so that the standard f/1–f/22
//! ladder sweeps the Airy core from well below one texel to a few texels at
//! the pyramid's reference resolution (512 rows), which is the range over
//! which the level-weight *shape* responds to the F-number. The physical
//! model fixes the shape; this constant fixes the overall angular scale.
//!
//! The weights are achromatic (one weight per level, not per channel):
//! chromatic dispersion à la GT7 would triple the upsample cost for
//! per-channel blur radii and is left as a follow-up.
//!
//! # Where the numbers live
//!
//! [`GLARE_WEIGHT_TABLE`] holds the finished weights as literals. The Bessel
//! quadrature behind them costs a few milliseconds, too much to spend at every
//! app's startup for a scatter model most apps never enable, so the derivation
//! — the Airy, photopic and band-integration code, and the sensor pitch and
//! wavelength range it is calibrated with — lives in this module's tests.
//! `tests::table_matches_derivation` compares the two bit for bit, so changing
//! the model fails until the literals are regenerated.

use bevy_math::ops;
use bevy_utils::once;
use tracing::warn;

/// The number of pyramid levels (octave-spaced annular bands) the weight
/// table covers. Matches the default bloom chain depth
/// (`max_mip_dimension = 512` → 8 mips). Deeper chains receive zero weight
/// beyond this band range (the residual tail energy out there is < 0.5% even
/// at f/22); shallower chains fold the remaining tail into their last level
/// automatically through the tail-sum blend construction in
/// [`blend_factor`].
pub(crate) const GLARE_BANDS: usize = 8;

/// The standard full-stop aperture ladder the weight table is precomputed
/// for. Queries between entries interpolate linearly in `log2(N)` (i.e. by
/// F-stop); queries outside clamp.
pub(crate) const F_NUMBER_LADDER: [f32; 10] = [1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0];

/// The F-number substituted for non-finite or non-positive
/// [`Gt7Glare::f_number`](super::BloomScatterModel::Gt7Glare) values
/// (mid-ladder, a common photographic walk-around aperture).
pub(crate) const DEFAULT_F_NUMBER: f32 = 5.6;

/// The per-F-stop weight table: for each entry of [`F_NUMBER_LADDER`], the
/// normalized energy fraction each pyramid level receives.
///
/// Baked from the derivation in this module's tests; see the module docs.
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

/// Replaces a non-finite or non-positive F-number with
/// [`DEFAULT_F_NUMBER`], warning once.
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

/// Normalized per-level glare weights for an arbitrary F-number:
/// interpolates [`GLARE_WEIGHT_TABLE`] linearly in `log2(N)` (per F-stop,
/// matching the ladder's geometric spacing), clamping to the ladder ends.
/// Linear interpolation of normalized weight vectors stays normalized.
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

/// The upsample blend constant for the glare model, replacing the
/// hand-tuned parametric curve of
/// [`BloomScatterModel::Aesthetic`](super::BloomScatterModel::Aesthetic) in
/// `compute_blend_factor`.
///
/// The bloom node composites the pyramid bottom-up through chained
/// energy-conserving lerps (`out = lerp(dst, src, blend)` via blend
/// constants), so the final image is
///
/// ```text
/// (1 − b₀)·image + Σₖ b₀·…·bₖ·(1 − bₖ₊₁)·levelₖ
/// ```
///
/// Solving for the per-pass constants that realize the target per-level
/// weights `intensity·ŵₖ` (and `1 − intensity` for the sharp image) gives
/// tail-sum ratios: with `Tⱼ = Σₖ₌ⱼ ŵₖ`,
///
/// ```text
/// b₀ = intensity·T₀ = intensity        (the final pass)
/// bⱼ = Tⱼ / Tⱼ₋₁                       (level j blended into level j−1)
/// ```
///
/// This construction makes chains shallower than [`GLARE_BANDS`] fold the
/// un-representable wide-band tail into their deepest level, and chains
/// deeper than it blend the extra levels with weight zero. `mip` follows
/// `compute_blend_factor`'s convention (0 = the final composite onto the
/// view target).
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

    // The derivation `GLARE_WEIGHT_TABLE` is baked from, kept out of the
    // shipped build; `table_matches_derivation` keeps the two in sync.

    /// Virtual sensor pitch of one pyramid level-0 texel, in micrometers.
    ///
    /// See the module docs ("Calibration"): this constant maps pyramid texels
    /// to image-plane distances inside the Airy formula and therefore sets how
    /// large the diffraction pattern appears on screen. At 2 µm the Airy core
    /// radius (`1.22·λ·N`) spans ~0.3 texels at f/1.0 and ~7 texels at f/22
    /// for λ = 555 nm.
    const TEXEL_PITCH_MICRONS: f64 = 2.0;

    /// Wavelength integration range (µm) and sample count for the
    /// polychromatic PSF. 400–700 nm covers where the photopic weight is
    /// non-negligible.
    const LAMBDA_MIN_MICRONS: f64 = 0.40;
    const LAMBDA_MAX_MICRONS: f64 = 0.70;
    const WAVELENGTH_SAMPLES: usize = 16;

    /// Single-Gaussian approximation of the CIE 1924 photopic luminosity
    /// function `V(λ)`, with `λ` in micrometers (peak at 555 nm, σ ≈ 42 nm).
    fn photopic_weight(lambda_microns: f64) -> f64 {
        let d = lambda_microns - 0.559;
        1.019 * (-285.4 * d * d).exp()
    }

    /// Bessel function of the first kind `J_n(x)` via Simpson integration of
    /// Bessel's integral `J_n(x) = (1/π)·∫₀^π cos(n·τ − x·sin τ) dτ`.
    ///
    /// Accurate to machine precision for the `x ≤ 16` range it is used in (the
    /// integrand extends to a smooth periodic function, so the composite rule
    /// converges geometrically once the sample count exceeds `x`).
    fn bessel_j(n: u32, x: f64) -> f64 {
        // 128 (even) keeps ~1e-16 absolute error up to the asymptotic seam at
        // x = 16 and produces a bit-identical f32 weight table to 512
        // intervals (verified; degradation only begins below ~64 intervals).
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
    /// radius `v = π·r/(λ·N)`: `E(v) = 1 − J0(v)² − J1(v)²` (Born & Wolf
    /// §8.5.2).
    ///
    /// For `v ≥` [`AIRY_ASYMPTOTIC_SEAM`] the ring-averaged asymptotic
    /// `J0² + J1² → 2/(π·v)` (relative error `O(v⁻²)`) is used, with its
    /// constant calibrated so the two branches meet exactly at the seam — this
    /// keeps `E` continuous and monotonic, which the band-energy differences
    /// rely on.
    fn airy_encircled_energy(v: f64) -> f64 {
        // seam · (J0(seam)² + J1(seam)²), ≈ 2/π up to the O(v⁻²) ring
        // residual.
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
    /// `r ∈ [t·2ᵏ, t·2ᵏ⁺¹)`, `t =` [`TEXEL_PITCH_MICRONS`], for the given
    /// F-number.
    ///
    /// Level 0's band extends down to the center (`[0, 2t)`): the diffraction
    /// core belongs to the finest blur level, so the distribution describes
    /// the *whole* PSF and the sharp/scattered split stays entirely with
    /// `Bloom::intensity`. (Excluding a sub-texel "core" region instead makes
    /// the normalized shape non-monotonic in N as the Airy bulk crosses the
    /// arbitrary cutoff — tested and rejected.) The only energy not covered is
    /// the residual beyond the last band, < 0.5% even at f/22.
    fn raw_band_energies(f_number: f64) -> [f64; GLARE_BANDS] {
        let mut bands = [0.0; GLARE_BANDS];
        let mut total_weight = 0.0;
        for i in 0..WAVELENGTH_SAMPLES {
            let lambda = LAMBDA_MIN_MICRONS
                + (LAMBDA_MAX_MICRONS - LAMBDA_MIN_MICRONS) * i as f64
                    / (WAVELENGTH_SAMPLES - 1) as f64;
            let weight = photopic_weight(lambda);
            total_weight += weight;
            // v = π·r/(λ·N)
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

    /// [`raw_band_energies`] normalized to sum to 1 (the distribution shape;
    /// the total scattered amount is `Bloom::intensity`'s job, see module
    /// docs).
    fn normalized_band_weights(f_number: f64) -> [f32; GLARE_BANDS] {
        let raw = raw_band_energies(f_number);
        let sum: f64 = raw.iter().sum();
        let mut weights = [0.0f32; GLARE_BANDS];
        for (weight, raw) in weights.iter_mut().zip(raw) {
            *weight = (raw / sum) as f32;
        }
        weights
    }

    /// `J0`/`J1` against standard reference values (Abramowitz & Stegun,
    /// table 9.1) and the first zeros.
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

    /// The encircled energy is a CDF: 0 at the center, monotonically
    /// increasing, approaching 1; and the first dark ring encloses the
    /// textbook 83.8% of the energy.
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
        // Born & Wolf: E(first zero of J1, v = 3.8317) ≈ 0.8378.
        assert!((airy_encircled_energy(3.831_705_970_2) - 0.8378).abs() < 1e-3);
    }

    /// Raw band energies are physical fractions of the PSF's total energy:
    /// non-negative, summing to at most 1, covering ≥ 99% of the PSF at
    /// every ladder entry (the bands span the whole pattern, core included),
    /// with the residual beyond the widest band growing as the pattern
    /// widens — and the finest level's share strictly draining outward as
    /// the aperture stops down.
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

    /// Every table entry reproduces, bit for bit, the normalized band weights
    /// the derivation produces for its F-number. Regenerate the literals if
    /// this fails.
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

    /// Each table entry is normalized, and through f/11 the level weights
    /// decay strictly monotonically — the diffraction core plus the heavy
    /// half-per-octave Airy tail. (At f/16+ the core's bulk ring crosses
    /// into level 1 and the peak physically moves off the finest level, so
    /// monotonicity from level 0 is *not* asserted there.)
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

    /// Stopping down spreads the glare: the energy-weighted mean level index
    /// strictly increases along the entire F-stop ladder (the Airy pattern
    /// scales with λ·N), by more than a full pyramid level end to end.
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
        // End to end, stopping down from f/1 to f/22 must spread the glare
        // by more than a full pyramid level.
        let first = mean_level(&GLARE_WEIGHT_TABLE[0]);
        let last = mean_level(&GLARE_WEIGHT_TABLE[F_NUMBER_LADDER.len() - 1]);
        assert!(last - first > 1.0, "f/1 {first} -> f/22 {last}");
    }

    /// F-number interpolation: exact at ladder entries, clamped outside,
    /// continuous and normalized in between.
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
        // Between f/4 and f/5.6 every band lies between (or at) the bracket
        // values and the vector stays normalized.
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

    /// Invalid F-numbers degrade to the default instead of poisoning the
    /// weights.
    #[test]
    fn invalid_f_number_degrades_to_default() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -2.8] {
            assert_eq!(mip_weights(bad), mip_weights(DEFAULT_F_NUMBER));
        }
    }

    /// Reconstructs the effective per-level contributions from the chained
    /// lerp blend constants and checks they reproduce `intensity·ŵₖ`
    /// exactly, with `1 − intensity` left for the sharp image — for the full
    /// 8-level chain, a shallower chain (tail folded into the deepest
    /// level), and a deeper chain (zero weight past the table).
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

            // contribution(level k) = b0·…·bk·(1 − b_{k+1}), last level
            // keeps the full product.
            let mut product = 1.0f32;
            let mut contributions = Vec::new();
            for k in 0..levels {
                product *= blends[k];
                let next = if k + 1 < levels { blends[k + 1] } else { 0.0 };
                contributions.push(product * (1.0 - next));
            }
            // The deepest pyramid level is never blended *into*; its
            // contribution is the bare product.
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
                    // Levels past the table get nothing.
                    0.0
                };
                assert!(
                    (contribution - expected).abs() < 1e-6,
                    "mip_count {mip_count}, level {k}: {contribution} != {expected}"
                );
            }
        }
    }

    /// Degenerate inputs to the blend factor are safe.
    #[test]
    fn blend_factor_degenerate_inputs() {
        assert_eq!(blend_factor(5.6, f32::NAN, 0), 0.0);
        assert_eq!(blend_factor(5.6, 2.0, 0), 1.0);
        assert_eq!(blend_factor(5.6, -1.0, 0), 0.0);
        assert_eq!(blend_factor(5.6, 0.5, GLARE_BANDS as u32 + 5), 0.0);
        // All pass constants are valid lerp factors.
        for n in F_NUMBER_LADDER {
            for mip in 0..12 {
                let b = blend_factor(n, 0.7, mip);
                assert!((0.0..=1.0).contains(&b), "f/{n} mip {mip}: {b}");
            }
        }
    }
}
