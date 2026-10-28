//! Vertical cloud profiles: Nubis's height x type lookup with physical contents.
//!
//! One Rgba16Float texture, [`CLOUD_PROFILE_HEIGHT_RES`] texels per row against metres above the
//! row's cloud base (texel i at i / (N - 1) x [`CLOUD_PROFILE_RANGE_M`]), one row per cloud
//! type ([`CLOUD_PROFILE_TYPE_RES`]: 0 stratocumulus .. 1 deep cumulus, see
//! [`CloudProfiles::Physical`]):
//! - R, G: cloud fraction (peak 1) as a bottom and a top factor (Nubis's two lookups); times
//!   coverage it is the dimensional profile that the erosion turns into clouds;
//! - B: in-cloud liquid water (kg/m^3); divided by the layer's
//!   [`VolumetricCloudLayer::liquid_water`](super::VolumetricCloudLayer::liquid_water) it scales
//!   the eroded density;
//! - A: the cloud base above the surface (km, the same along a row): each air mass has its own
//!   lifting condensation level (marine cumulus near 0.55 km, continental near 1.7 km). The
//!   profile is sampled at the height above it.
//!
//! The next rows, filled on the GPU, hold the shape- and billow-noise distributions per noise mip
//! and the cloud scheme's threshold table, which make the profile the cloud fraction exactly
//! (`clouds.wesl` `cloud_thermal_excess`). The last [`CLOUD_PROFILE_TYPE_RES`] rows hold the
//! clouds' lean per type (R: downwind drift of the cloudy updraft per [`CLOUD_LEAN_SHEAR`] of
//! shear, against the same height above the base; `clouds.wesl` `cloud_sample_pos`). SI
//! throughout: m, kg/m^3.

use bevy_asset::Handle;
use bevy_ecs::component::Component;
use bevy_image::Image;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};

/// Texels per row: height above the cloud base (texel i at i / (N - 1) x range).
pub const CLOUD_PROFILE_HEIGHT_RES: usize = 256;
/// Rows of cloud types (texel j at type j / (N - 1)).
pub const CLOUD_PROFILE_TYPE_RES: usize = 64;
/// Height range of the rows (m above the cloud base; must match `clouds.wesl`).
pub const CLOUD_PROFILE_RANGE_M: f32 = 4000.0;

/// The vertical structure of the clouds of a [`VolumetricCloudLayer`](super::VolumetricCloudLayer) (module docs).
#[derive(Clone, Component, Reflect, Default)]
#[reflect(Clone, Default)]
pub enum CloudProfiles {
    /// Baked offline from a surface-layer parcel ensemble (Neggers, Köhler & Beljaars 2009,
    /// J. Atmos. Sci. 66): the updrafts are the top 10 % of the surface-layer distribution with
    /// spreads from the surface fluxes, lifted with entrainment 1 / (tau w); cloud fraction is the
    /// share of saturated rising parcels plus the saturated mean state, liquid water their mean.
    /// Cloud fraction is the share of saturated rising parcels plus the saturated mean state,
    /// liquid water the parcels' mean, against height above the base. Each air mass has its own
    /// cloud base (its lifting condensation level), stored with the profiles, and the cloud layer
    /// spans them ([`physical_profile_bounds`]). The type axis orders published LES cases by
    /// cloud-top height, like Nubis's stratus .. cumulus axis, linear between:
    /// - 0: stratocumulus, DYCOMS-II RF01 (Stevens et al. 2005): an overcast deck ~270 m deep,
    ///   base 575 m;
    /// - 1/3: shallow trade cumulus, BOMEX (Siebesma et al. 2003): the fraction peaks at the base
    ///   (555 m) and falls off to ~1.4 km above it (within ~20 % of the LES above 1 km);
    /// - 2/3: deeper trade cumulus, RICO's equilibrium state (van Zanten et al. 2011, Fig. 4):
    ///   0.05 at the base (570 m), ~0.015 through the cloud layer, tops near 3 km (LES: 0.06,
    ///   ~0.02, ~2.7 km);
    /// - 1: continental cumulus, the CASS composite of the ARM Southern Great Plains site at
    ///   1330 LST (Zhang et al. 2017, J. Atmos. Sci. 74): base 1675 m, the fraction falls off
    ///   through ~2.1 km above it as the cloudy updrafts detrain (Neggers et al. 2009, sec. 5c),
    ///   within the bulk and bin microphysics LES, and tapers to 0 over the ~400 m where the
    ///   parcels stop (top ~4.4 km; LES 4.2 km, ARSCL above 4.3 km); liquid water ~1.5x the bulk
    ///   LES aloft.
    /// The baker records its inputs in the KTX2 key/value data.
    #[default]
    Physical,
    /// The analytic envelope `x^bottom (1 - x)^top`, normalized to a peak of 1, of the height x
    /// in a column whose top rises with coverage from
    /// [`VolumetricCloudLayer::profile_top_min`] to the layer top, with uniform liquid water and
    /// Nubis's `1 - profile` erosion threshold (evaluated in the shader).
    ///
    /// [`VolumetricCloudLayer::profile_top_min`]: super::VolumetricCloudLayer::profile_top_min
    Envelope {
        /// Exponent of the rise above the base.
        bottom: f32,
        /// Exponent of the fall toward the top.
        top: f32,
    },
    /// A lookup image of the format above (Rgba16Float, 64 x 32), e.g. baked for another air
    /// mass or painted. Until it loads, or if its format differs (logged), [`Self::Physical`].
    Image(Handle<Image>),
    /// Nubis's vertical profile (Schneider, "Nubis, Evolved", SIGGRAPH 2022 Advances, slides 22,
    /// 26 and 28): the top-type and bottom-type gradients, height in the cloud layer by cloud
    /// type, multiplied (one weather type drives both), with Nubis's `1 - profile` erosion
    /// threshold and uniform liquid water. The 64 x 64 lookups were recovered from the deck and
    /// rebuilt as smooth functions (cloud-top boundary per type, gradients between base and top),
    /// within ~1.5 grey levels RMS of the original, then baked at this lookup's resolution.
    Nubis,
}

impl CloudProfiles {
    /// Adiabatic liquid water growing linearly above the base times an adiabatic fraction falling
    /// linearly toward the top (Warner 1955): `x (1 - x)`.
    pub const ADIABATIC: Self = Self::Envelope {
        bottom: 1.0,
        top: 1.0,
    };
    /// Nubis Evolved (SIGGRAPH 2022) envelope-model gradients: `x^2 (1 - x)^1.5`.
    pub const NUBIS_ENVELOPE: Self = Self::Envelope {
        bottom: 2.0,
        top: 1.5,
    };
}

/// The baked [`CloudProfiles::Physical`] set: an uncompressed Rgba16Float KTX2.
const PHYSICAL_KTX2: &[u8] = include_bytes!("cloud_profiles.ktx2");
/// The baked [`CloudProfiles::Nubis`] set (R bottom-type, G top-type gradient).
const NUBIS_KTX2: &[u8] = include_bytes!("cloud_profiles_nubis.ktx2");
/// The lean of the [`CloudProfiles::Physical`] clouds, one row per type, baked with them from the
/// same parcel ensemble: R the downwind drift (m) of the cloudy updraft at the row's height above
/// its base per [`CLOUD_LEAN_SHEAR`] of shear, x(z) = S int_0^z z' / w(z') dz' with w the mean
/// updraft speed of the cloudy rising parcels (G, m/s); held above the last rising level. The
/// updraft accelerates above the base and decelerates near its top, unlike a constant speed:
/// marine trade cumulus rise at 1-2 m/s near the base and 3-6 m/s aloft, continental cumulus at
/// 6-9 m/s, so marine clouds lean more near their bases and continental clouds stand more upright
/// (the constant 3 m/s leaned every cloud alike).
const LEAN_KTX2: &[u8] = include_bytes!("cloud_lean.ktx2");

/// Shear the lean rows are tabulated for (1/s): the lean is the layer's shear / this x the row.
pub const CLOUD_LEAN_SHEAR: f32 = 1.0e-3;

/// The single level of an uncompressed KTX2 (vkFormat 97 = R16G16B16A16_SFLOAT,
/// [`CLOUD_PROFILE_HEIGHT_RES`] x [`CLOUD_PROFILE_TYPE_RES`], checked by a test).
fn ktx2_texels(ktx2: &'static [u8]) -> &'static [u8] {
    let u64_at = |o: usize| u64::from_le_bytes(ktx2[o..o + 8].try_into().unwrap()) as usize;
    let (offset, len) = (u64_at(80), u64_at(88));
    &ktx2[offset..offset + len]
}

/// Texels of [`CloudProfiles::Physical`].
pub(crate) fn physical_profile_texels() -> &'static [u8] {
    ktx2_texels(PHYSICAL_KTX2)
}

/// Texels of [`CloudProfiles::Nubis`].
pub(crate) fn nubis_profile_texels() -> &'static [u8] {
    ktx2_texels(NUBIS_KTX2)
}

/// Lean rows of [`CloudProfiles::Physical`] (see `LEAN_KTX2`).
pub(crate) fn physical_lean_texels() -> &'static [u8] {
    ktx2_texels(LEAN_KTX2)
}

/// Lean rows for an updraft rising at a constant `updraft_speed` (m/s), every type alike:
/// S z^2 / (2 w) per [`CLOUD_LEAN_SHEAR`], for profile images that carry no updraft.
pub(crate) fn constant_lean_texels(updraft_speed: f32) -> Vec<u8> {
    let w = updraft_speed.max(0.1);
    (0..CLOUD_PROFILE_TYPE_RES * CLOUD_PROFILE_HEIGHT_RES)
        .flat_map(|i| {
            let z = (i % CLOUD_PROFILE_HEIGHT_RES) as f32 / (CLOUD_PROFILE_HEIGHT_RES - 1) as f32
                * CLOUD_PROFILE_RANGE_M;
            [CLOUD_LEAN_SHEAR * z * z / (2.0 * w), w, 0.0, 0.0].map(f32_to_f16)
        })
        .flat_map(u16::to_le_bytes)
        .collect()
}

/// Height range of the clouds of [`CloudProfiles::Physical`] above the surface (m): the lowest
/// cloud base and the highest top (cloud fraction above 1e-3 of its peak) over all types. With
/// the physical profiles this is the cloud layer, whatever the layer's own bounds.
pub fn physical_profile_bounds() -> (f32, f32) {
    static BOUNDS: std::sync::OnceLock<(f32, f32)> = std::sync::OnceLock::new();
    *BOUNDS.get_or_init(|| {
        let texels = physical_profile_texels();
        let texel = |row: usize, i: usize, c: usize| {
            let o = ((row * CLOUD_PROFILE_HEIGHT_RES + i) * 4 + c) * 2;
            f16_to_f32(u16::from_le_bytes([texels[o], texels[o + 1]]))
        };
        let mut lowest = f32::MAX;
        let mut highest = 0.0f32;
        for row in 0..CLOUD_PROFILE_TYPE_RES {
            let base = texel(row, 0, 3) * 1000.0;
            lowest = lowest.min(base);
            if let Some(i) = (0..CLOUD_PROFILE_HEIGHT_RES)
                .rev()
                .find(|&i| texel(row, i, 0) * texel(row, i, 1) > 1e-3)
            {
                let z = i as f32 / (CLOUD_PROFILE_HEIGHT_RES - 1) as f32 * CLOUD_PROFILE_RANGE_M;
                highest = highest.max(base + z);
            }
        }
        (lowest, highest)
    })
}

/// f32 from IEEE half (finite values).
fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as f32;
    sign * if exp == 0 {
        mant * 2f32.powi(-24)
    } else {
        (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
    }
}

/// IEEE half from f32 (finite values below 65504; subnormals flush to 0).
pub(crate) fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7bff;
    }
    // Round to nearest on the 13 dropped mantissa bits (a carry into the exponent is correct).
    let mant = bits & 0x7f_ffff;
    let h = ((exp as u32) << 10) | (mant >> 13);
    sign | (h + ((mant >> 12) & 1)).min(0x7bff) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baked_sets_are_rgba16f_lookups() {
        for (ktx2, texels) in [
            (PHYSICAL_KTX2, physical_profile_texels()),
            (NUBIS_KTX2, nubis_profile_texels()),
            (LEAN_KTX2, physical_lean_texels()),
        ] {
            let u32_at = |o: usize| u32::from_le_bytes(ktx2[o..o + 4].try_into().unwrap());
            assert_eq!(&ktx2[..12], b"\xabKTX 20\xbb\r\n\x1a\n");
            assert_eq!(u32_at(12), 97);
            assert_eq!(u32_at(20) as usize, CLOUD_PROFILE_HEIGHT_RES);
            assert_eq!(u32_at(24) as usize, CLOUD_PROFILE_TYPE_RES);
            assert_eq!(texels.len(), 8 * CLOUD_PROFILE_HEIGHT_RES * CLOUD_PROFILE_TYPE_RES);
        }
    }

    #[test]
    fn physical_bounds_span_the_air_masses() {
        // Marine bases near 0.55 km, the continental cumulus tops near 4.5 km.
        let (base, top) = physical_profile_bounds();
        assert!((500.0..600.0).contains(&base), "{base}");
        assert!((4000.0..5000.0).contains(&top), "{top}");
    }

    #[test]
    fn lean_rows() {
        let texel = |t: &[u8], row: usize, i: usize, c: usize| {
            let o = ((row * CLOUD_PROFILE_HEIGHT_RES + i) * 4 + c) * 2;
            f16_to_f32(u16::from_le_bytes([t[o], t[o + 1]]))
        };
        // Constant updraft: S z^2 / (2 w) at the top of the range, 4 km at 3 m/s.
        let constant = constant_lean_texels(3.0);
        let top = texel(&constant, 10, CLOUD_PROFILE_HEIGHT_RES - 1, 0);
        assert!((top - 1e-3 * 4000.0 * 4000.0 / 6.0).abs() < 2.0, "{top}");
        // Physical: no drift at the base, never decreasing upwards; continental cumulus (6-9
        // m/s) lean less than deep trade cumulus (1-6 m/s) at the same height.
        let lean = physical_lean_texels();
        for row in 0..CLOUD_PROFILE_TYPE_RES {
            assert_eq!(texel(lean, row, 0, 0), 0.0);
            for i in 1..CLOUD_PROFILE_HEIGHT_RES {
                assert!(texel(lean, row, i, 0) >= texel(lean, row, i - 1, 0));
            }
        }
        let at_1500 = 1500 * (CLOUD_PROFILE_HEIGHT_RES - 1) / 4000;
        assert!(texel(lean, 63, at_1500, 0) < 0.5 * texel(lean, 42, at_1500, 0));
    }

    #[test]
    fn half_conversion() {
        assert_eq!(f16_to_f32(f32_to_f16(0.555)), 0.55517578);
        assert_eq!(f32_to_f16(0.0), 0);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f32_to_f16(4.0), 0x4400);
    }
}
