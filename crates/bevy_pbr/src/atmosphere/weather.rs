//! World-anchored weather influences composed into each view's weather window
//! (`weather_window.wesl`; `.clouds-handoff/notes/weather_layers_design.md`, A2).

use bevy_ecs::prelude::*;
use bevy_math::{Mat4, Vec2, Vec3};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::Extract;
use bevy_transform::components::GlobalTransform;

/// How an influence combines with the weather below it (the names image editors use). Every mode
/// mixes with the base by `opacity` times the mask: `mix(base, mode(base, value), opacity * mask)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub enum CloudBlend {
    /// The influence's value (a full-opacity Normal influence replaces the weather).
    #[default]
    Normal,
    /// base + value: more cloud over a region.
    Add,
    /// base - value: clear a hole.
    Subtract,
    /// base x value: thin a region smoothly.
    Multiply,
    /// max(base, value): at least this much.
    Max,
    /// min(base, value): at most this much.
    Min,
}

/// Shape of an influence's mask in its local frame (x and z of its `Transform`, metres).
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub enum CloudInfluenceShape {
    /// Rectangle of the given half extents.
    Rectangle(Vec2),
    /// Ellipse of the given semi-axes.
    Ellipse(Vec2),
}

impl Default for CloudInfluenceShape {
    fn default() -> Self {
        Self::Ellipse(Vec2::splat(4_000.0))
    }
}

/// Channel values an influence writes; `None` leaves a channel alone. Art channels are read by
/// the Nubis and envelope profiles (and coverage by the physical ones as their presence
/// multiplier); physical channels by the physical profiles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[reflect(Default)]
pub struct CloudInfluenceValues {
    /// Coverage (art A.r), in [0, 1].
    pub coverage: Option<f32>,
    /// Cloud type (art A.g), in [0, 1].
    pub cloud_type: Option<f32>,
    /// Nubis density scale (art A.b), in [0, 1].
    pub density_scale: Option<f32>,
    /// Cloud-fraction scale (physical B.r).
    pub cloud_fraction: Option<f32>,
    /// Air mass (physical B.g): the profile row, 0 stratocumulus .. 1 deep cumulus. Each air mass
    /// has its own cloud base (up to ~1.1 km apart), and the shear lean folds the clouds where the
    /// base slopes more than ~0.6 (`fbm_noise_3d.wesl` weather_type): feather an air-mass change
    /// over at least ~3 km (the smoothstep's slope is 1.5x the mean).
    pub air_mass: Option<f32>,
    /// Cloud droplet number concentration (physical B.b), cm^-3: maritime ~50-150, continental
    /// several hundred.
    pub droplet_number: Option<f32>,
    /// Cloud-base offset (physical B.a), m.
    pub base_offset: Option<f32>,
}

/// A world-anchored edit of the weather: spawn with a `Transform` (its position and yaw place
/// the mask; the procedural weather drifts with the wind underneath it). Composed into the
/// weather window of every view in [`CloudInfluence::order`], within 32 km of the camera.
///
/// ```ignore
/// // Clear a hole 6 km across, 10 km north of the origin.
/// commands.spawn((
///     Transform::from_xyz(0.0, 0.0, -10_000.0),
///     CloudInfluence {
///         shape: CloudInfluenceShape::Ellipse(Vec2::splat(3_000.0)),
///         blend: CloudBlend::Subtract,
///         values: CloudInfluenceValues { coverage: Some(1.0), ..default() },
///         ..default()
///     },
/// ));
/// ```
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component, Default)]
#[require(bevy_transform::components::Transform)]
pub struct CloudInfluence {
    pub shape: CloudInfluenceShape,
    /// Width of the mask's soft edge, inside the shape (m).
    pub feather: f32,
    pub blend: CloudBlend,
    /// In [0, 1].
    pub opacity: f32,
    /// Application order, lowest first (like a UI z-index).
    pub order: i32,
    pub values: CloudInfluenceValues,
}

impl Default for CloudInfluence {
    fn default() -> Self {
        Self {
            shape: CloudInfluenceShape::default(),
            feather: 1_000.0,
            blend: CloudBlend::Normal,
            opacity: 1.0,
            order: 0,
            values: CloudInfluenceValues::default(),
        }
    }
}

/// Influences per view (`weather_window.wesl` MAX_INFLUENCES).
pub const MAX_CLOUD_INFLUENCES: usize = 16;

/// The extracted influences, in world space, sorted by order.
#[derive(Resource, Default, Clone)]
pub struct ExtractedCloudInfluences(pub Vec<(Vec3, Vec3, CloudInfluence)>);

pub fn extract_cloud_influences(
    mut extracted: ResMut<ExtractedCloudInfluences>,
    influences: Extract<Query<(&GlobalTransform, &CloudInfluence)>>,
) {
    extracted.0.clear();
    let mut all: Vec<_> = influences
        .iter()
        .map(|(gt, i)| (gt.translation(), gt.right().into(), i.clone()))
        .collect();
    all.sort_by_key(|(_, _, i)| i.order);
    if all.len() > MAX_CLOUD_INFLUENCES {
        bevy_log::warn_once!(
            "{} CloudInfluences; the first {MAX_CLOUD_INFLUENCES} by order are composed",
            all.len()
        );
        all.truncate(MAX_CLOUD_INFLUENCES);
    }
    extracted.0 = all;
}

/// `weather_window.wesl` WeatherWindowParams for one view, as raw f32 words (u32 fields bit
/// cast): origin, droplet number, influence count; the map frame (advection offset, scale); then
/// per influence 5 x vec4: centre xz and x axis xz (atmosphere space); half extents, feather,
/// shape; blend, opacity, channel mask; art values; physical values.
pub fn weather_window_params(
    origin: Vec2,
    droplet_number: f32,
    map_scale: f32,
    map_offset: Vec3,
    world_to_atmosphere: &Mat4,
    influences: &ExtractedCloudInfluences,
) -> Vec<f32> {
    let n = influences.0.len().min(MAX_CLOUD_INFLUENCES);
    let mut words = vec![
        origin.x,
        origin.y,
        droplet_number,
        f32::from_bits(n as u32),
        map_offset.x,
        map_offset.z,
        map_scale,
        0.0,
    ];
    for (translation, right, influence) in influences.0.iter().take(n) {
        let c = world_to_atmosphere.transform_point3(*translation);
        let a = world_to_atmosphere.transform_vector3(*right);
        let axis = Vec2::new(a.x, a.z).normalize_or(Vec2::X);
        let (half, shape) = match influence.shape {
            CloudInfluenceShape::Rectangle(h) => (h, 0u32),
            CloudInfluenceShape::Ellipse(h) => (h, 1u32),
        };
        let v = influence.values;
        let mut mask = 0u32;
        let mut val = |bit: u32, x: Option<f32>| {
            if x.is_some() {
                mask |= 1 << bit;
            }
            x.unwrap_or(0.0)
        };
        let art = [
            val(0, v.coverage),
            val(1, v.cloud_type),
            val(2, v.density_scale),
            0.0,
        ];
        let phys = [
            val(4, v.cloud_fraction),
            val(5, v.air_mass),
            val(6, v.droplet_number),
            val(7, v.base_offset),
        ];
        let blend = influence.blend as u32;
        words.extend_from_slice(&[c.x, c.z, axis.x, axis.y]);
        words.extend_from_slice(&[half.x.max(1.0), half.y.max(1.0), influence.feather.max(0.0), f32::from_bits(shape)]);
        words.extend_from_slice(&[f32::from_bits(blend), influence.opacity.clamp(0.0, 1.0), f32::from_bits(mask), 0.0]);
        words.extend_from_slice(&art);
        words.extend_from_slice(&phys);
    }
    // Fixed size: the uniform declares MAX_CLOUD_INFLUENCES entries.
    words.resize(8 + 20 * MAX_CLOUD_INFLUENCES, 0.0);
    words
}
