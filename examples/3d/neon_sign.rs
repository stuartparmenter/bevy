//! Cyberpunk street at civil twilight: a composed still showcasing the HDR pipeline.
//!
//! A three-story "BEVY" blade sign of real 3D neon tubes juts from a brick tenement
//! over a rain-slicked street canyon. The scene exercises, all at once:
//!
//! * HDR emissive values far above 1.0 (tube cores pinned at the display's peak nits),
//! * `Tonemapping::GranTurismo7` with live `GranTurismo7Params` tweaking,
//! * `Bloom` with the physically derived `BloomScatterModel::Gt7Glare`,
//! * physically based `Atmosphere` scattering with the sun ~2 degrees BELOW the
//!   horizon (the orange-to-blue twilight band at the vanishing point),
//! * `AtmosphereEnvironmentMapLight` as the only ambient light source,
//! * volumetric fog (street haze, sodium lamp cones, headlights, a steam plume),
//! * deferred rendering + screen-space reflections: the sign smears across the wet
//!   asphalt and snaps to a mirror in the gutter puddles,
//! * an scRGB-linear HDR swapchain via `DisplayTarget`, with the negotiated result
//!   reported from `WindowResolvedTransfer`.
//!
//! ## Controls
//!
//! | Key             | Action                                                          |
//! |:----------------|:----------------------------------------------------------------|
//! | `1` / `2` / `3` | Neon color scheme (Magenta/Cyan, Cyan/Amber, Acid/Violet)       |
//! | `R`             | Toggle screen-space reflections (the wet street ablation)       |
//! | `H`             | Toggle HDR (scRGB) vs SDR display output                        |
//! | `B`             | Cycle bloom (GT7 glare f/2.0 -> f/2.8 -> f/5.6 -> f/11 -> Aesthetic -> off) |
//! | `G`             | Cycle GT7 tonemapper blend ratio (0.3 / 0.6 / 1.0)              |
//! | `C`             | Toggle the parked car's headlights                              |
//! | `F`             | 0.7 s neon flicker burst                                        |
//! | `[` / `]`       | HDR paper white -/+ 40 nits (match your panel's SDR slider)    |
//! | `Up` / `Down`   | Exposure (ev100); the emissive neon is exposure-immune          |

use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, PI, TAU};

use bevy::{
    anti_alias::taa::TemporalAntiAliasing,
    camera::{Exposure, Hdr},
    core_pipeline::tonemapping::{GranTurismo7Params, Tonemapping},
    gltf::GltfMaterialName,
    image::{
        ImageAddressMode, ImageFilterMode, ImageLoaderSettings, ImageSampler,
        ImageSamplerDescriptor,
    },
    light::{
        atmosphere::ScatteringMedium, light_consts::lux, Atmosphere, AtmosphereEnvironmentMapLight,
        FogVolume, SunDisk, VolumetricFog, VolumetricLight,
    },
    math::{ops, Affine2},
    pbr::{
        AtmosphereSettings, ContactShadows, DefaultOpaqueRendererMethod,
        ScreenSpaceAmbientOcclusion, ScreenSpaceReflections,
    },
    post_process::bloom::{Bloom, BloomScatterModel},
    prelude::*,
    render::{working_color_space::WorkingColorSpace, RenderPlugin},
    window::{DisplayTarget, DisplayTransfer, PrimaryWindow, WindowResolvedTransfer},
    world_serialization::WorldInstanceReady,
};

// --- Layout ------------------------------------------------------------------
//
// The street runs along Z and the camera looks down -Z. Building A (the hero
// tenement with the blade sign) is on the left at facade plane x = -7.5; building
// B is on the right at x = +7.5. y = 0 is the road surface.

/// Camera position: a raised pedestrian view (~3 m), standing in the road. The
/// extra meter over eye height pulls the sign's mirror streak up into frame.
const CAMERA_POS: Vec3 = Vec3::new(0.5, 3.0, 12.0);
/// Aim point. The look-at height and the sign's down-street position were chosen
/// with mirror geometry in mind: a ground reflection of an emissive at height `h`
/// and horizontal distance `D` lands `D * h_cam / (h_cam + h)` meters from the
/// camera, and that landing zone must clear the frame-bottom ray or the SSR streak
/// is invisible. With the camera at y 3.0, this slightly-downward aim, and the
/// sign at z = -12, the frame-bottom ground cut sits ~5.8 m out while the B/E/V/Y
/// streaks land ~5.2-8.8 m out: an E/V/Y streak across the lower third. The aim
/// is panned a meter further left (x -2.0) so that streak column lands to the
/// RIGHT of the HUD text block instead of underneath it.
const CAMERA_LOOK: Vec3 = Vec3::new(-2.0, 1.9, -14.0);

/// Blade sign center (x, z). It is perpendicular to building A's facade so the
/// tube letters face the camera (and therefore exist in the G-buffer SSR sees).
const SIGN_X: f32 = -6.55;
const SIGN_Z: f32 = -12.0;
/// Tube geometry sits proud of the sign panel's front face.
const TUBE_Z: f32 = SIGN_Z + 0.30;
/// Hero tube radius. Fat enough that the SSR ray thickness (0.12) does not
/// swallow the reflected tubes at 20+ meters.
const TUBE_R: f32 = 0.055;
/// Letter cell height for B/E/V/Y.
const LETTER_H: f32 = 1.5;
/// Letter centers, top (B) to bottom (Y).
const LETTER_YS: [f32; 4] = [10.9, 9.0, 7.1, 5.2];
/// Bird-in-ring emblem center height, above the letters. 12.3 (not 12.55): at
/// 12.55 the ring kissed the frame top with zero headroom and the hero sign
/// read as cropped; paired with the smaller 0.48 ring radius so the ring
/// bottom (~11.78) still clears the B's top stroke (~11.71).
const EMBLEM_Y: f32 = 12.3;

// --- Photometric ladder --------------------------------------------------------
//
// The scene is graded against a 200-nit paper-white / 1000-nit-peak HDR target.
// With GT7's frame-buffer unit = 100 nits and Bevy scene-linear 1.0 -> 2.5 fb,
// post-exposure scene-linear >= ~5.9 reaches the 1000-nit display peak.
// The intended brightness hierarchy, bottom to top:
//
//   ~0.5 nit   deep shadow under the parked car
//   2-5 nits   wet asphalt lit by spill + twilight IBL
//   10-50 nits twilight sky band at the vanishing point
//   100-250    lit apartment windows (emissive ~0.5-4.5 scene-linear)
//   1000 nits  neon tube cores and lamp heads (emissive 28-110: heavily
//              overshot so the cores still clip at peak AFTER haze and aerial
//              perspective attenuate them along the view path)
//
// Emissive materials bypass `Exposure` (default `emissive_exposure_weight: 0.0`,
// which the deferred G-buffer enforces anyway), so the Up/Down exposure keys swing
// the street and sky while the neon stays locked at display peak.

/// 7.5: bright enough that the midground road (sodium pools, headlight wash)
/// reads above ~8/255 in the tonemapped frame -- at 7.75 it crushed to ~4/255 --
/// while the twilight band core still sits BELOW the emissive windows and neon
/// on the ladder above (8.25 left ~85% of the frame under 20/255).
const EV100_DEFAULT: f32 = 7.5;
const EV100_MIN: f32 = 6.0;
const EV100_MAX: f32 = 11.0;

/// 2.2M lm (not 1.2M): the magenta wash on the brick facade regressed when the
/// ambient env-map was pulled back per the contrast-is-king directive, and the
/// LOCAL spill light is the sanctioned way to buy it back -- it re-anchors the
/// sign to its wall without lifting the global dark field. The hum/flicker
/// animation scales from this constant, so the wash stays slaved to the tubes.
const SPILL_LUMENS: f32 = 2_200_000.0;
const ENTRANCE_LUMENS: f32 = 800_000.0;
/// 55M lm (not 36M): pushed through the widened 0.85 rad cone and aimed at the
/// street CENTER (not the gutter), each lamp must drop a readable sodium circle
/// on the roadway AND a visible shaft in the haze. The near lamp moved from
/// z -8 to z -14 to clear the Porsche occlusion; 36M dropped readable pools but
/// the heads themselves never resolved as light SOURCES, so the emissive head
/// face grew (see `spawn_street_furniture`) and the lumens rose in step. The
/// sodium pools are LOCAL: they do not lift the dark field between the lamps.
const LAMP_LUMENS: f32 = 55_000_000.0;
/// 18M lm: the beams point AWAY from the camera, so only weak back-scatter
/// comes back through the haze; even 9M read as a nose hotspot with no
/// visible shafts. 18M plus a dedicated beam-path fog slab (see `spawn_fog`)
/// is what makes the named wow beat (the volumetric cones + warm pool at
/// z -5..-14) read at final exposure.
const HEADLIGHT_LUMENS: f32 = 18_000_000.0;
/// The broken BAR sign's red point light is slaved to the sign's emissive duty
/// cycle at this many lumens per emissive unit, so the wall and the fog stutter
/// in exact sync with the tubes.
const BAR_LUMENS_PER_EMISSIVE: f32 = 360.0;

/// White-hot emissive for the alpha-masked Bevy bird. Deliberately a uniform
/// emissive with NO `emissive_texture`: the bird texels are near-black, and
/// multiplying them into a white-hot emissive would dim it to almost nothing.
/// 18 (not 35): still far over the display peak, but low enough that the GT7
/// glare halo does not swallow the bird silhouette into a shapeless scribble.
const BIRD_EMISSIVE: LinearRgba = LinearRgba {
    red: 18.0,
    green: 18.0,
    blue: 18.0,
    alpha: 1.0,
};

/// Broken "BAR" sign red (duty-cycled in `animate_neon`).
const BAR_RED: LinearRgba = LinearRgba {
    red: 22.0,
    green: 0.5,
    blue: 0.12,
    alpha: 1.0,
};

/// Animated cool-TV window emissive base.
const TV_EMISSIVE: LinearRgba = LinearRgba {
    red: 0.5,
    green: 1.05,
    blue: 1.9,
    alpha: 1.0,
};

// --- Color schemes -------------------------------------------------------------
//
// One dominant channel per color: GT7's per-channel clamp + blend_ratio keep the
// hue stable as the tubes blow past the display peak, and under
// `WorkingColorSpace::Rec2020` the wide-gamut primaries survive to the encoder.
// Colors are authored as ordinary Rec.709 values; the engine converts once at the
// seams -- never pre-convert.
//
// The emissive magnitudes deliberately overshoot the display peak by ~4x over
// the "barely clips" values: street haze and aerial-perspective extinction along
// the ~24 m view path eat tube luminance BEFORE tonemapping, and the letter
// cores must still clip to a white-hot center after that attenuation.

struct Scheme {
    name: &'static str,
    /// Letter tubes + the right building's torus ring.
    primary: LinearRgba,
    /// Emblem ring + storefront border tubes.
    secondary: LinearRgba,
    /// Spill point light color (sRGB components).
    primary_glow: [f32; 3],
    /// Storefront entrance spot color (sRGB components).
    secondary_glow: [f32; 3],
}

const SCHEMES: [Scheme; 3] = [
    Scheme {
        name: "Magenta/Cyan",
        // 141 (not 110): scaled ~1.28x in lockstep across channels (hue-stable)
        // to buy back the extinction from the street haze slab, so the V/Y
        // letters inside the slab still clip white-hot at the display peak
        // (the cores carry ~4x clip headroom, so there is room to spend).
        primary: LinearRgba {
            red: 141.0,
            green: 5.2,
            blue: 46.0,
            alpha: 1.0,
        },
        secondary: LinearRgba {
            red: 2.4,
            green: 64.0,
            blue: 88.0,
            alpha: 1.0,
        },
        primary_glow: [1.0, 0.12, 0.55],
        secondary_glow: [0.1, 0.85, 1.0],
    },
    Scheme {
        name: "Cyan/Amber",
        primary: LinearRgba {
            red: 2.4,
            green: 64.0,
            blue: 80.0,
            alpha: 1.0,
        },
        secondary: LinearRgba {
            red: 104.0,
            green: 36.0,
            blue: 2.8,
            alpha: 1.0,
        },
        primary_glow: [0.1, 0.85, 1.0],
        secondary_glow: [1.0, 0.62, 0.12],
    },
    Scheme {
        name: "Acid/Violet",
        primary: LinearRgba {
            red: 12.0,
            green: 104.0,
            blue: 4.8,
            alpha: 1.0,
        },
        secondary: LinearRgba {
            red: 40.0,
            green: 4.8,
            blue: 104.0,
            alpha: 1.0,
        },
        primary_glow: [0.35, 1.0, 0.2],
        secondary_glow: [0.55, 0.25, 1.0],
    },
];

// --- Bloom modes -----------------------------------------------------------------
//
// NEVER set `low_frequency_boost` / curvature / prefilter alongside `Gt7Glare`:
// they are dead fields under the physical glare model.

/// Bloom mix fraction. The `EnergyConserving` composite is `lerp(src, blurred,
/// intensity)` and the blurred term is Karis-clamped to ~4 fb units, so the mix
/// SHAVES the tube cores: at 0.35 a core lands near 0.65x its commanded value.
/// 0.18 keeps >= 82% of source energy at the cores -- the 28-110 emissives still
/// pin the 1000-nit display peak after haze extinction -- and the halo loss is
/// bought back with a wider default aperture (f/2.0) instead of a bigger mix.
const BLOOM_INTENSITY: f32 = 0.18;

const BLOOM_F_NUMBERS: [f32; 4] = [2.0, 2.8, 5.6, 11.0];
const BLOOM_MODE_NAMES: [&str; 6] = [
    "GT7 glare f/2.0",
    "GT7 glare f/2.8",
    "GT7 glare f/5.6",
    "GT7 glare f/11",
    "Aesthetic (Bloom::NATURAL)",
    "off",
];

const BLEND_RATIOS: [f32; 3] = [0.3, 0.6, 1.0];

// --- Letter stroke data ------------------------------------------------------------
//
// Letters are data-driven: straight strokes become `Capsule3d` tubes (the round
// caps self-blend at the joints, doubling as joint caps), and curved strokes are
// REAL bent-glass torus arcs (`TorusMeshBuilder::angle_range`). Coordinates are in
// a normalized cell: half-height 0.5, scaled by the letter height at spawn.
// Arcs are right-opening half circles: (center, major_radius).

struct LetterSpec {
    strokes: &'static [[Vec2; 2]],
    arcs: &'static [(Vec2, f32)],
}

const LETTER_B: LetterSpec = LetterSpec {
    strokes: &[
        [Vec2::new(-0.30, -0.5), Vec2::new(-0.30, 0.5)],
        [Vec2::new(-0.30, 0.5), Vec2::new(-0.04, 0.5)],
        [Vec2::new(-0.30, 0.0), Vec2::new(-0.04, 0.0)],
        [Vec2::new(-0.30, -0.5), Vec2::new(-0.04, -0.5)],
    ],
    arcs: &[
        (Vec2::new(-0.04, 0.25), 0.25),
        (Vec2::new(-0.04, -0.25), 0.25),
    ],
};

const LETTER_E: LetterSpec = LetterSpec {
    strokes: &[
        [Vec2::new(-0.30, -0.5), Vec2::new(-0.30, 0.5)],
        [Vec2::new(-0.30, 0.5), Vec2::new(0.26, 0.5)],
        [Vec2::new(-0.30, 0.0), Vec2::new(0.18, 0.0)],
        [Vec2::new(-0.30, -0.5), Vec2::new(0.26, -0.5)],
    ],
    arcs: &[],
};

const LETTER_V: LetterSpec = LetterSpec {
    strokes: &[
        [Vec2::new(-0.30, 0.5), Vec2::new(0.0, -0.5)],
        [Vec2::new(0.30, 0.5), Vec2::new(0.0, -0.5)],
    ],
    arcs: &[],
};

const LETTER_Y: LetterSpec = LetterSpec {
    strokes: &[
        [Vec2::new(-0.30, 0.5), Vec2::new(0.0, 0.03)],
        [Vec2::new(0.30, 0.5), Vec2::new(0.0, 0.03)],
        [Vec2::new(0.0, 0.03), Vec2::new(0.0, -0.5)],
    ],
    arcs: &[],
};

const LETTER_A: LetterSpec = LetterSpec {
    strokes: &[
        [Vec2::new(-0.30, -0.5), Vec2::new(0.0, 0.5)],
        [Vec2::new(0.30, -0.5), Vec2::new(0.0, 0.5)],
        [Vec2::new(-0.17, -0.07), Vec2::new(0.17, -0.07)],
    ],
    arcs: &[],
};

const LETTER_R: LetterSpec = LetterSpec {
    strokes: &[
        [Vec2::new(-0.30, -0.5), Vec2::new(-0.30, 0.5)],
        [Vec2::new(-0.30, 0.5), Vec2::new(-0.04, 0.5)],
        [Vec2::new(-0.30, 0.0), Vec2::new(-0.04, 0.0)],
        [Vec2::new(-0.04, 0.0), Vec2::new(0.30, -0.5)],
    ],
    arcs: &[(Vec2::new(-0.04, 0.25), 0.25)],
};

// --- App -------------------------------------------------------------------------

fn main() {
    App::new()
        // Deferred G-buffer rendering: required so SSR can reflect the opaque
        // emissive tube geometry (forward meshes never appear in reflections).
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(ClearColor(Color::BLACK))
        // All ambient light comes from the atmosphere's generated environment map.
        .insert_resource(GlobalAmbientLight::NONE)
        .init_resource::<NeonScene>()
        .add_plugins(DefaultPlugins.set(RenderPlugin {
            // GT7's native working space; pairs with the scRGB HDR output so the
            // wide-gamut neon primaries survive to the encoder.
            working_color_space: WorkingColorSpace::Rec2020,
            ..default()
        }))
        .add_systems(Startup, (setup_hdr_display, setup, print_controls))
        .add_systems(
            Update,
            (handle_input, animate_neon, animate_world, update_hud).chain(),
        )
        .add_observer(boost_taillights)
        .run();
}

fn print_controls() {
    println!("Neon sign controls:");
    println!("    1 / 2 / 3 - neon color scheme (Magenta/Cyan, Cyan/Amber, Acid/Violet)");
    println!("    R         - toggle screen space reflections");
    println!("    H         - toggle HDR (scRGB) vs SDR display output");
    println!("    B         - cycle bloom (GT7 glare f/2.0, f/2.8, f/5.6, f/11, Aesthetic, off)");
    println!("    G         - cycle GT7 blend ratio (0.3 / 0.6 / 1.0)");
    println!("    C         - toggle headlights");
    println!("    F         - neon flicker burst");
    println!("    [ / ]     - HDR paper white -/+ 40 nits (120..480; peak stays 1000)");
    println!("    Up / Down - exposure (ev100); neon is exposure-immune");
}

// --- State -------------------------------------------------------------------------

#[derive(Resource)]
struct NeonScene {
    scheme: usize,
    /// Remaining seconds of the F-key flicker burst.
    flicker_secs: f32,
    bloom_mode: usize,
    blend_idx: usize,
    ssr_on: bool,
    hdr_requested: bool,
    headlights_on: bool,
    /// HDR paper-white level in nits ([ / ] keys); peak stays fixed at 1000.
    paper_white_nits: f32,
    // Animated material handles (shared; quantized so the whole scene uses a
    // handful of material instances).
    primary_mat: Handle<StandardMaterial>,
    dying_mat: Handle<StandardMaterial>,
    secondary_mat: Handle<StandardMaterial>,
    bird_mat: Handle<StandardMaterial>,
    bar_mat: Handle<StandardMaterial>,
    tv_mat: Handle<StandardMaterial>,
}

impl Default for NeonScene {
    fn default() -> Self {
        Self {
            scheme: 0,
            flicker_secs: 0.0,
            bloom_mode: 0,
            blend_idx: 1,
            ssr_on: true,
            hdr_requested: false,
            headlights_on: true,
            paper_white_nits: 200.0,
            primary_mat: Handle::default(),
            dying_mat: Handle::default(),
            secondary_mat: Handle::default(),
            bird_mat: Handle::default(),
            bar_mat: Handle::default(),
            tv_mat: Handle::default(),
        }
    }
}

#[derive(Component)]
struct HudText;
#[derive(Component)]
struct SignSpillLight;
#[derive(Component)]
struct EntranceLight;
#[derive(Component)]
struct BarStutterLight;
#[derive(Component)]
struct Headlight;
#[derive(Component)]
struct SteamPlume;

/// Deterministic LCG so window layouts and TV flicker reproduce exactly under the
/// CI fixed-timestep harness. NO `rand` crate.
struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / 16_777_216.0
    }
}

// --- HDR display -----------------------------------------------------------------

/// Paper white is user-adjustable ([ / ] keys) so the HDR output can be matched
/// against the desktop's "SDR content brightness" slider; the 1000-nit peak is
/// fixed, so lowering paper white WIDENS the neon-over-paper-white headroom.
fn hdr_display_target(paper_white_nits: f32) -> DisplayTarget {
    DisplayTarget::SDR_SRGB
        .with_paper_white(paper_white_nits)
        .with_peak(1000.0)
        .with_transfer(DisplayTransfer::ScRgbLinear)
}

/// Requests an scRGB-linear HDR swapchain.
///
/// GATED on CI: the `ci_testing` screenshotter reads back the raw swapchain, and an
/// fp16 scRGB readback saved to PNG clips at 80 nits (everything blows out to
/// white). CI and iteration screenshots must capture the tonemapped SDR output,
/// so the override is skipped whenever `CI_TESTING_CONFIG` is set.
fn setup_hdr_display(
    mut display_target: Single<&mut DisplayTarget, With<PrimaryWindow>>,
    mut state: ResMut<NeonScene>,
) {
    if std::env::var("CI_TESTING_CONFIG").is_ok() {
        state.hdr_requested = false;
        return;
    }
    state.hdr_requested = true;
    **display_target = hdr_display_target(state.paper_white_nits);
}

// --- Setup -----------------------------------------------------------------------

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
    mut state: ResMut<NeonScene>,
    asset_server: Res<AssetServer>,
) {
    let mut cache = MeshCache::default();
    // Fixed seed: the window light distribution is part of the composition.
    let mut lcg = Lcg(0x00B3_55ED);

    spawn_camera(&mut commands);
    spawn_sky(&mut commands, &mut scattering_mediums);
    spawn_fog(&mut commands);

    let mats = build_materials(&mut materials, &asset_server, &mut state);

    spawn_street(&mut commands, &mut meshes, &mats);
    spawn_building_a(&mut commands, &mut meshes, &mut cache, &mats, &mut lcg);
    spawn_blade_sign(&mut commands, &mut meshes, &mut cache, &mats);
    spawn_building_b(&mut commands, &mut meshes, &mut cache, &mats, &mut lcg);
    spawn_skyline(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut cache,
        &mats,
        &mut lcg,
    );
    spawn_street_furniture(&mut commands, &mut meshes, &mats);
    spawn_porsche(&mut commands, &asset_server);
    spawn_lights(&mut commands);

    // HUD: rides the single GT7-tonemapped 3D camera. Do NOT add a second camera
    // (a 2D camera with `Tonemapping::None` desaturates under Rec2020).
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(14.0),
            ..default()
        },
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.72)),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
        HudText,
    ));
}

/// SSR march tuned for 5.5 cm neon tubes reflected at 20+ m. ONE shared
/// constructor for the camera spawn AND the R-key re-insert, so the ablation
/// toggle restores exactly the same reflections it removed.
///
/// * `min_perceptual_roughness` 0.0..0.0: the default 0.08..0.12 excludes
///   mirror surfaces, which would exclude the near-mirror gutter puddles.
/// * Defaults (thickness 0.25, 10 linear steps) can swallow the thin tubes;
///   thinner rays + a denser march + bisection/secant refinement keep the
///   letter streak gap-free -- at `linear_steps` 16 the ray march undersampled
///   the tubes in the mirror puddles and the BEVY reflection smeared into a
///   single noisy magenta blob.
fn ssr_settings() -> ScreenSpaceReflections {
    ScreenSpaceReflections {
        min_perceptual_roughness: 0.0..0.0,
        thickness: 0.12,
        linear_steps: 32,
        bisection_steps: 8,
        use_secant: true,
        ..default()
    }
}

/// The full deferred + screen-space camera stack (mirrors `atmosphere.rs`).
fn spawn_camera(commands: &mut Commands) {
    commands
        .spawn((
            Camera3d::default(),
            Projection::Perspective(PerspectiveProjection {
                fov: 50.0_f32.to_radians(),
                ..default()
            }),
            Transform::from_translation(CAMERA_POS).looking_at(CAMERA_LOOK, Vec3::Y),
            Hdr,
            // Deferred + TAA both require MSAA off.
            Msaa::Off,
            TemporalAntiAliasing::default(),
            Tonemapping::GranTurismo7,
            // Plumbs the display target's peak luminance into the GT7 operator.
            GranTurismo7Params::default(),
            Exposure {
                ev100: EV100_DEFAULT,
            },
            // See BLOOM_INTENSITY: the low mix keeps the tube cores pinned at
            // the display peak; f/2.0 supplies the halo instead.
            Bloom {
                intensity: BLOOM_INTENSITY,
                scatter: BloomScatterModel::Gt7Glare {
                    f_number: BLOOM_F_NUMBERS[0],
                },
                ..Bloom::NATURAL
            },
            AtmosphereSettings::default(),
            // The twilight sky drives IBL and is the env-map fallback that fills sky
            // into SSR misses on the wet street. 512^2: a 256 cubemap undersamples
            // the specular term and reads as white firefly speckle on smooth panes
            // and the car body under TAA. intensity 1.5, NOT higher: a 2.5 boost
            // lifted the shadow floor but mirrored a milky env-sky sheen across
            // the wet road (brighter than the sky band it reflected) and fed the
            // SSR-miss speckle. Per the owner's directive contrast is king --
            // crushed blacks are an accepted aesthetic, so the floor stays low
            // and the env map only rims the facades.
            AtmosphereEnvironmentMapLight {
                intensity: 1.5,
                size: UVec2::splat(512),
                ..default()
            },
            // jitter 0.4 / 96 steps (not 1.0 / 64): full-amplitude jitter left
            // single-frame stochastic speckle across the midground that TAA
            // never fully resolved in stills; a denser march needs less jitter
            // to hide banding.
            VolumetricFog {
                ambient_intensity: 0.0,
                jitter: 0.4,
                step_count: 96,
                ..default()
            },
        ))
        // Split across `insert` to stay under the bundle tuple-arity limit.
        .insert((
            ContactShadows::default(),
            ssr_settings(),
            ScreenSpaceAmbientOcclusion::default(),
        ));
}

fn spawn_sky(commands: &mut Commands, scattering_mediums: &mut Assets<ScatteringMedium>) {
    // A slightly hazier-than-stock atmosphere for a smoggy city dusk.
    let medium =
        scattering_mediums.add(ScatteringMedium::earth(256, 256).with_density_multiplier(1.6));
    commands.spawn(Atmosphere::earth(medium));

    // The sun travels toward (-0.25, 0.038, 0.97): it sits ~2.2 degrees BELOW the
    // horizon, down-street (-Z) and slightly right of the axis. (A smaller +y on
    // the light's TRAVEL direction raises the sun toward the horizon -- at 0.055
    // / ~3.2 degrees the twilight band at the vanishing point underdelivered.)
    // Direct surface light is exactly zero (transmittance x visible-sun-ratio),
    // but the sky-view LUT paints the warm twilight band at the vanishing point
    // and the deep blue zenith above. No `VolumetricLight` here: below the
    // horizon its fog in-scatter is multiplied to zero while still costing a
    // full raymarch.
    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadow_maps_enabled: true,
            contact_shadows_enabled: true,
            ..default()
        },
        Transform::default().looking_to(Vec3::new(-0.25, 0.038, 0.97).normalize(), Vec3::Y),
        // The disk would render by default even below the horizon haze.
        SunDisk::OFF,
    ));
}

fn spawn_fog(commands: &mut Commands) {
    // City haze slab over the street; the volumetric lights paint into it. The
    // slab top (7.5 m) must submerge the streetlamp heads at y ~6 or the bright
    // near-source sections of their cones march through empty space.
    // scattering_asymmetry 0.0 (pure isotropic) matters just as much: the
    // headlight beams point away from the camera, and any forward lobe scatters
    // their light away from the viewer -- even 0.2 left the cones invisible;
    // isotropic phase maximizes back-scatter without touching density.
    // density_factor 0.11 (not 0.15): 0.15 read as readable lamp cones but the
    // accumulated extinction over the 10-90 m view path extinguished everything
    // the street depends on for depth -- the wet road sheen, the twilight band
    // at the vanishing point, and the taillight energy all grayed out. 0.11
    // keeps the cones and the warm down-street in-scatter glow while letting
    // the distant payoffs through; the local beam-path and plume slabs (below)
    // carry the dense-fog beats instead of this global slab. The WARM fog
    // color (not the old blue-gray) lets the haze pick up the twilight band's
    // warmth instead of graying it out.
    commands.spawn((
        FogVolume {
            fog_color: Color::srgb(0.70, 0.62, 0.58),
            density_factor: 0.11,
            absorption: 0.12,
            scattering: 0.9,
            scattering_asymmetry: 0.0,
            ..default()
        },
        Transform::from_scale(Vec3::new(70.0, 7.5, 90.0))
            .with_translation(Vec3::new(0.0, 3.75, -25.0)),
    ));

    // Dedicated beam-path slab hugging the headlight cones ahead of the
    // Porsche's nose: the beams point AWAY from the camera, so in the thin
    // ambient haze only weak back-scatter returns and the spec'd "cones
    // punching through ground mist" beat never read. A denser (0.30) isotropic
    // pocket along z -9..0 gives the cones a medium to paint into without
    // raising the whole street's haze (which would dim the V/Y letters).
    commands.spawn((
        FogVolume {
            density_factor: 0.30,
            absorption: 0.08,
            scattering: 1.0,
            scattering_asymmetry: 0.0,
            ..default()
        },
        Transform::from_scale(Vec3::new(3.0, 1.5, 9.0)).with_translation(Vec3::new(2.9, 0.8, -4.5)),
    ));

    // Foreground-right steam plume over the grate; density animated in
    // `animate_world`. Dense (0.75 base): this is the designated FOREGROUND
    // depth layer and at 0.32 and even 0.50 it never read at all -- the wider
    // 2.0 m footprint buys a longer view path through the column so whatever
    // street light reaches it has a chance to register. Placed at (3.5, 5.0),
    // just right of the parked car's tail and INSIDE the right frame edge --
    // at the old (4.3, 6.5) it projected ~40 deg off-axis against the 39.7 deg
    // half-fov and read only as a corner smudge. Here it rises against the
    // dark wall behind the car and catches the taillight red, while its left
    // edge still clears the hero body (centered over the car it de-glossed
    // the paint as a gray wash). The taller 2.4 scale carries the column up
    // past the car's roofline. Isotropic phase (0.0, not 0.4) for the same
    // reason as the haze slab: the plume is viewed side-on and a forward lobe
    // throws any in-scatter down-street away from the camera.
    commands.spawn((
        FogVolume {
            density_factor: 0.75,
            absorption: 0.08,
            scattering: 1.0,
            scattering_asymmetry: 0.0,
            ..default()
        },
        Transform::from_scale(Vec3::new(2.0, 2.4, 2.0)).with_translation(Vec3::new(3.5, 1.3, 5.0)),
        SteamPlume,
    ));
}

// --- Materials ---------------------------------------------------------------------

struct SharedMats {
    asphalt: Handle<StandardMaterial>,
    puddle: Handle<StandardMaterial>,
    sidewalk: Handle<StandardMaterial>,
    lane_paint: Handle<StandardMaterial>,
    manhole: Handle<StandardMaterial>,
    grate: Handle<StandardMaterial>,
    concrete: Handle<StandardMaterial>,
    concrete_b: Handle<StandardMaterial>,
    stucco: Handle<StandardMaterial>,
    brick: Handle<StandardMaterial>,
    metal: Handle<StandardMaterial>,
    sign_panel: Handle<StandardMaterial>,
    rust: Handle<StandardMaterial>,
    cable: Handle<StandardMaterial>,
    tower: Handle<StandardMaterial>,
    lamp_glow: Handle<StandardMaterial>,
    door_glow: Handle<StandardMaterial>,
    win_dark: Handle<StandardMaterial>,
    win_dim_a: Handle<StandardMaterial>,
    win_dim_b: Handle<StandardMaterial>,
    win_warm_a: Handle<StandardMaterial>,
    win_warm_b: Handle<StandardMaterial>,
    win_tv: Handle<StandardMaterial>,
    primary_neon: Handle<StandardMaterial>,
    dying_neon: Handle<StandardMaterial>,
    secondary_neon: Handle<StandardMaterial>,
    bird: Handle<StandardMaterial>,
    bar_neon: Handle<StandardMaterial>,
}

fn neon_material(emissive: LinearRgba) -> StandardMaterial {
    StandardMaterial {
        // Dead-glass tube color: a small always-lit floor so flickered-off tubes
        // (the mostly-off BAR sign especially) still silhouette as cold glass
        // instead of vanishing into the facade -- 0.09 (not 0.06) so the off
        // BAR letters still rim against Building B's near-black wall.
        base_color: Color::srgb(0.09, 0.09, 0.105),
        perceptual_roughness: 0.4,
        emissive,
        ..default()
    }
}

fn lit_window_material(emissive: LinearRgba) -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgb(0.01, 0.012, 0.015),
        perceptual_roughness: 0.1,
        emissive,
        ..default()
    }
}

fn build_materials(
    materials: &mut Assets<StandardMaterial>,
    asset_server: &AssetServer,
    state: &mut NeonScene,
) -> SharedMats {
    // Normal maps must be loaded linear (is_srgb: false) and tile with a Repeat
    // sampler. anisotropy_clamp 16: the brick band is seen at a hard grazing
    // angle and without anisotropic filtering its courses mip-smear into
    // diagonal streaks along the view direction (anisotropy requires ALL
    // filter modes linear, including mipmap_filter).
    let repeat_descriptor = || {
        ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::Repeat,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            mipmap_filter: ImageFilterMode::Linear,
            anisotropy_clamp: 16,
            ..default()
        })
    };
    let brick_color: Handle<Image> = asset_server
        .load_builder()
        .with_settings(move |settings: &mut ImageLoaderSettings| {
            settings.sampler = repeat_descriptor();
        })
        .load("textures/parallax_example/cube_color.png");
    let brick_normal: Handle<Image> = asset_server
        .load_builder()
        .with_settings(move |settings: &mut ImageLoaderSettings| {
            settings.is_srgb = false;
            settings.sampler = repeat_descriptor();
        })
        .load("textures/parallax_example/cube_normal.png");
    let brick_depth: Handle<Image> = asset_server
        .load_builder()
        .with_settings(move |settings: &mut ImageLoaderSettings| {
            settings.is_srgb = false;
            settings.sampler = repeat_descriptor();
        })
        .load("textures/parallax_example/cube_depth.png");

    // Wet asphalt: the SSR hero surface. Deliberately NO normal map: three
    // tiling/roughness iterations proved the per-texel normal slopes were the
    // root cause of the road noise (0.08 read as crumpled foil, 0.22 and 0.26
    // both sparkled with cyan env-map fireflies under TAA) and they scattered
    // the sign streak into glitter. A flat plane at roughness 0.30 lets SSR's
    // own roughness-driven blur form the coherent vertical neon smear, and
    // SSR's per-frame jitter keeps the result alive under TAA without any uv
    // drift. Reflectance 0.42 + the near-black 0.028 base, NOT 0.6/0.045:
    // higher values mirrored the env-map sky into a milky gray sheen that
    // lifted the whole road ~5x over the <= 10-14/255 floor the grade demands
    // (the road read BRIGHTER than the sky band it reflected, and brighter
    // than the dry sidewalk -- inverted wet/dry contrast). Wetness here means
    // bright streak cores 3-5x over a DARK field, not a brighter field.
    let asphalt = materials.add(StandardMaterial {
        base_color: Color::srgb(0.028, 0.028, 0.034),
        perceptual_roughness: 0.30,
        metallic: 0.0,
        reflectance: 0.42,
        ..default()
    });

    let scheme = &SCHEMES[0];
    let primary_neon = materials.add(neon_material(scheme.primary));
    // The dying "E" gets its own handle so it can sputter independently;
    // everything else shares one handle per scheme color.
    let dying_neon = materials.add(neon_material(scheme.primary));
    let secondary_neon = materials.add(neon_material(scheme.secondary));
    let bar_neon = materials.add(neon_material(LinearRgba::BLACK));

    // The Bevy bird: alpha-MASKED, not blended. `AlphaMode::Mask` stays on the
    // deferred path, so the white-hot bird appears in the SSR street reflection
    // (Blend would vanish from it).
    let bird = materials.add(StandardMaterial {
        base_color: Color::BLACK,
        base_color_texture: Some(asset_server.load("branding/bevy_bird_dark.png")),
        emissive: BIRD_EMISSIVE,
        alpha_mode: AlphaMode::Mask(0.5),
        ..default()
    });

    let win_tv = materials.add(lit_window_material(TV_EMISSIVE));

    state.primary_mat = primary_neon.clone();
    state.dying_mat = dying_neon.clone();
    state.secondary_mat = secondary_neon.clone();
    state.bird_mat = bird.clone();
    state.bar_mat = bar_neon.clone();
    state.tv_mat = win_tv.clone();

    SharedMats {
        asphalt,
        // Near-mirror standing water; only works because the camera's SSR
        // min_perceptual_roughness range is 0.0..0.0. Reflectance 0.45 (not
        // 0.6): high reflectance both blew the reflected tube cores past the
        // bloom knee (illegible white-cored blob) and mirrored the env-map sky
        // fallback into bright naked discs wherever SSR rays miss geometry.
        puddle: materials.add(StandardMaterial {
            base_color: Color::srgb(0.012, 0.014, 0.018),
            perceptual_roughness: 0.04,
            reflectance: 0.45,
            ..default()
        }),
        // DRY sidewalk (roughness 0.85) against the wet road: the dry/wet
        // juxtaposition is what reads as rain. NO normal map: the reused water
        // normals glinted as wavy flowing cyan water under the entrance light
        // (brighter than the hero reflection); flat shading at roughness 0.85
        // carries the dry concrete on its own.
        sidewalk: materials.add(StandardMaterial {
            base_color: Color::srgb(0.16, 0.155, 0.145),
            perceptual_roughness: 0.85,
            reflectance: 0.3,
            ..default()
        }),
        // Worn paint, deliberately dull: at base 0.30 / roughness 0.55 the
        // nearest dash caught the env map and floated as a bright pink stick.
        lane_paint: materials.add(StandardMaterial {
            base_color: Color::srgb(0.20, 0.20, 0.19),
            perceptual_roughness: 0.75,
            ..default()
        }),
        // Dark dull iron: at metallic 0.9 / roughness 0.35 the disc mirrored
        // the env-map sky at grazing angle and read as a bright naked-primitive
        // ellipse on the road.
        manhole: materials.add(StandardMaterial {
            base_color: Color::srgb(0.08, 0.08, 0.09),
            metallic: 0.2,
            perceptual_roughness: 0.6,
            ..default()
        }),
        // Steam grate iron, darker and rougher than the manhole: the grate
        // sits under the foreground steam plume and must read as a dark
        // fixture sunk into the asphalt. At the manhole's 0.08 base, any
        // light placed near the plume floodlit the disc into a bright
        // "dinner plate" ellipse on the road; the near-black base keeps it
        // dark even under incidental spill.
        grate: materials.add(StandardMaterial {
            base_color: Color::srgb(0.03, 0.03, 0.033),
            perceptual_roughness: 0.8,
            ..default()
        }),
        concrete: materials.add(StandardMaterial {
            base_color: Color::srgb(0.07, 0.07, 0.075),
            perceptual_roughness: 0.9,
            reflectance: 0.25,
            ..default()
        }),
        concrete_b: materials.add(StandardMaterial {
            base_color: Color::srgb(0.10, 0.10, 0.115),
            perceptual_roughness: 0.85,
            ..default()
        }),
        stucco: materials.add(StandardMaterial {
            base_color: Color::srgb(0.07, 0.07, 0.075),
            perceptual_roughness: 0.9,
            ..default()
        }),
        // Parallax-mapped brick band; the grazing camera angle is exactly where
        // parallax pops -- and also exactly where it smears, so keep the depth
        // scale modest and the layer count high or the bricks streak diagonally.
        // The colored-squares source texture needs help to read as brick: the
        // deep red-brown tint pulls its green/purple squares toward mortar-and-
        // brick, and the tighter tiling (~0.35 m squares across the 44 x 16 m
        // band) hits brick-coursing scale instead of stretched planks.
        brick: materials.add(StandardMaterial {
            base_color: Color::srgb(0.42, 0.26, 0.20),
            base_color_texture: Some(brick_color),
            normal_map_texture: Some(brick_normal),
            depth_map: Some(brick_depth),
            parallax_depth_scale: 0.015,
            max_parallax_layer_count: 48.0,
            perceptual_roughness: 0.9,
            // 16 vertical tiles (not 11): paired with the anisotropic sampler,
            // the tighter vertical repeat keeps horizontal brick courses
            // legible under the foreshortening of the grazing camera angle.
            uv_transform: Affine2::from_scale(Vec2::new(30.0, 16.0)),
            ..default()
        }),
        // Half-rough, half-metal: shiny enough for a cyan neon rim on the roller
        // shutters, dull enough that the twilight env map does not sparkle.
        // Base 0.30 (not 0.22) so the shutter corrugation catches a visible
        // cyan rim inside the otherwise-black storefront recess.
        metal: materials.add(StandardMaterial {
            base_color: Color::srgb(0.30, 0.30, 0.30),
            metallic: 0.5,
            perceptual_roughness: 0.55,
            ..default()
        }),
        sign_panel: materials.add(StandardMaterial {
            base_color: Color::srgb(0.09, 0.09, 0.10),
            metallic: 0.85,
            perceptual_roughness: 0.6,
            ..default()
        }),
        rust: materials.add(StandardMaterial {
            base_color: Color::srgb(0.07, 0.05, 0.04),
            perceptual_roughness: 0.9,
            ..default()
        }),
        cable: materials.add(StandardMaterial {
            base_color: Color::srgb(0.01, 0.01, 0.01),
            perceptual_roughness: 0.7,
            ..default()
        }),
        // True silhouette: zero reflectance so the background towers stay flat
        // black shapes against the twilight band instead of sheening brown.
        tower: materials.add(StandardMaterial {
            base_color: Color::srgb(0.015, 0.015, 0.02),
            perceptual_roughness: 1.0,
            reflectance: 0.0,
            ..default()
        }),
        // Bright enough that both sodium heads read as light SOURCES (small
        // bloom kernels) at 20+ m, not just grey boxes over their pools (40,
        // not 28: the near lamp retreated to z -14 and the heads dimmed below
        // the point-source threshold).
        lamp_glow: materials.add(lit_window_material(LinearRgba {
            red: 40.0,
            green: 26.0,
            blue: 13.0,
            alpha: 1.0,
        })),
        door_glow: materials.add(lit_window_material(LinearRgba {
            red: 3.0,
            green: 1.6,
            blue: 0.6,
            alpha: 1.0,
        })),
        // Opaque "black glass": stays deferred, catches twilight env-map + SSR
        // glints at grazing angles. Roughness 0.30 / reflectance 0.45, NOT
        // smoother or hotter: at 0.06 every pane sparkled with undersampled
        // env-map fireflies, and even at 0.22/0.7 the SSR jitter alternating
        // between dark building hits and the bright sky fallback left panes
        // reading as solid fields of white speckle under TAA.
        win_dark: materials.add(StandardMaterial {
            base_color: Color::srgb(0.02, 0.024, 0.03),
            perceptual_roughness: 0.30,
            reflectance: 0.45,
            ..default()
        }),
        // Lit windows, quantized to a handful of shared instances. The four
        // tiers run a ~1.7-1.9x ladder (0.5 / 0.95 / 2.6 / 4.5 in the red
        // channel): the old dim tier measured ~9/255, indistinguishable from
        // unlit glass, so "dim" is doubled to read as curtained-but-occupied;
        // the old warm tier (6.0) tonemapped to clipped beige cards, but the
        // halved 3.0 undershot the 100-250-nit ladder rung (brightest pane
        // ~139/255, smoldering rather than occupied), so warm splits the
        // difference at 4.5, deepened toward tungsten orange. All four stay
        // decades under the neon, protecting the sign's hierarchy.
        win_dim_a: materials.add(lit_window_material(LinearRgba {
            red: 0.95,
            green: 0.52,
            blue: 0.20,
            alpha: 1.0,
        })),
        win_dim_b: materials.add(lit_window_material(LinearRgba {
            red: 0.50,
            green: 0.27,
            blue: 0.10,
            alpha: 1.0,
        })),
        win_warm_a: materials.add(lit_window_material(LinearRgba {
            red: 4.5,
            green: 2.1,
            blue: 0.55,
            alpha: 1.0,
        })),
        win_warm_b: materials.add(lit_window_material(LinearRgba {
            red: 2.6,
            green: 1.2,
            blue: 0.35,
            alpha: 1.0,
        })),
        win_tv,
        primary_neon,
        dying_neon,
        secondary_neon,
        bird,
        bar_neon,
    }
}

/// Lighting-TD window distribution: ~62% dark / 22% dim-warm / 12% warm / 4%
/// animated TV. A dim-majority facade reads as real dusk.
fn pick_window_material<'m>(lcg: &mut Lcg, mats: &'m SharedMats) -> &'m Handle<StandardMaterial> {
    let roll = lcg.unit();
    if roll < 0.62 {
        &mats.win_dark
    } else if roll < 0.73 {
        &mats.win_dim_a
    } else if roll < 0.84 {
        &mats.win_dim_b
    } else if roll < 0.90 {
        &mats.win_warm_a
    } else if roll < 0.96 {
        &mats.win_warm_b
    } else {
        &mats.win_tv
    }
}

// --- Tube helpers ---------------------------------------------------------------

/// Caches capsule and torus-arc meshes so the ~150 tube segments in the scene
/// share a handful of mesh handles (capsule lengths quantized to 5 cm; the round
/// caps absorb the slack and double as joint caps).
#[derive(Default)]
struct MeshCache {
    tubes: HashMap<(i32, i32), Handle<Mesh>>,
    arcs: HashMap<(i32, i32), Handle<Mesh>>,
}

impl MeshCache {
    fn tube(&mut self, meshes: &mut Assets<Mesh>, radius: f32, length: f32) -> Handle<Mesh> {
        let key = (
            (radius * 1000.0).round() as i32,
            ((length * 20.0).round() as i32).max(1),
        );
        self.tubes
            .entry(key)
            .or_insert_with(|| {
                meshes.add(Capsule3d {
                    radius,
                    half_length: key.1 as f32 / 40.0,
                })
            })
            .clone()
    }

    /// Right-opening half-circle torus arc, generated in the XZ plane; callers
    /// rotate it by `Quat::from_rotation_x(FRAC_PI_2)` to stand it in a sign's
    /// XY plane.
    fn half_arc(&mut self, meshes: &mut Assets<Mesh>, major: f32, tube: f32) -> Handle<Mesh> {
        let key = (
            (major * 1000.0).round() as i32,
            (tube * 1000.0).round() as i32,
        );
        self.arcs
            .entry(key)
            .or_insert_with(|| {
                meshes.add(
                    Torus {
                        minor_radius: tube,
                        major_radius: major,
                    }
                    .mesh()
                    .angle_range(-FRAC_PI_2..=FRAC_PI_2),
                )
            })
            .clone()
    }
}

/// Spawns one straight neon tube between two points in `frame`'s local XY plane.
/// `Capsule3d` is Y-axis-aligned natively; `Quat::from_rotation_arc` aims it.
fn spawn_tube_segment(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    cache: &mut MeshCache,
    material: &Handle<StandardMaterial>,
    frame: Transform,
    seg: [Vec2; 2],
    radius: f32,
) {
    let a = frame.transform_point(seg[0].extend(0.0));
    let b = frame.transform_point(seg[1].extend(0.0));
    let delta = b - a;
    let length = delta.length();
    let mesh = cache.tube(meshes, radius, length);
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material.clone()),
        Transform::from_translation(a.midpoint(b))
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, delta / length)),
    ));
}

/// Spawns a polyline of neon tube segments (used for the storefront border).
fn spawn_neon_polyline(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    cache: &mut MeshCache,
    material: &Handle<StandardMaterial>,
    frame: Transform,
    points: &[Vec2],
    radius: f32,
) {
    for pair in points.windows(2) {
        spawn_tube_segment(
            commands,
            meshes,
            cache,
            material,
            frame,
            [pair[0], pair[1]],
            radius,
        );
    }
}

/// Spawns one letter from its stroke/arc data. `size` is (cell height,
/// tube radius). The bent-glass bowls of B and R are REAL torus arcs.
fn spawn_letter(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    cache: &mut MeshCache,
    material: &Handle<StandardMaterial>,
    frame: Transform,
    spec: &LetterSpec,
    size: (f32, f32),
) {
    let (height, radius) = size;
    for stroke in spec.strokes {
        spawn_tube_segment(
            commands,
            meshes,
            cache,
            material,
            frame,
            [stroke[0] * height, stroke[1] * height],
            radius,
        );
    }
    for (center, major) in spec.arcs {
        let mesh = cache.half_arc(meshes, major * height, radius);
        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(frame.transform_point((*center * height).extend(0.0)))
                // TorusMeshBuilder generates in the XZ plane; stand it up into
                // the letter's XY plane.
                .with_rotation(frame.rotation * Quat::from_rotation_x(FRAC_PI_2)),
        ));
    }
}

// --- Street --------------------------------------------------------------------

fn spawn_street(commands: &mut Commands, meshes: &mut Assets<Mesh>, mats: &SharedMats) {
    // Roadway: 9 m wide, 80 m long, a flat plane -- the wet look is all
    // SSR + roughness, no normal map (see the asphalt material).
    let road_mesh = meshes.add(Plane3d::default().mesh().size(9.0, 80.0));
    commands.spawn((
        Mesh3d(road_mesh),
        MeshMaterial3d(mats.asphalt.clone()),
        Transform::from_xyz(0.0, 0.0, -25.0),
    ));

    // Raised, DRY sidewalks: leading lines + the wet/dry rain contrast.
    let sidewalk_mesh = meshes.add(Cuboid::new(3.0, 0.18, 80.0));
    for x in [-6.0, 6.0] {
        commands.spawn((
            Mesh3d(sidewalk_mesh.clone()),
            MeshMaterial3d(mats.sidewalk.clone()),
            Transform::from_xyz(x, 0.09, -25.0),
        ));
    }

    // Lane dashes leading the eye to the vanishing point.
    let dash_mesh = meshes.add(Cuboid::new(0.12, 0.006, 1.5));
    for i in 0..18 {
        commands.spawn((
            Mesh3d(dash_mesh.clone()),
            MeshMaterial3d(mats.lane_paint.clone()),
            Transform::from_xyz(0.0, 0.003, 10.0 - 3.0 * i as f32),
        ));
    }

    // Gutter puddle strips: guaranteed near-mirrors along both curbs.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.7, 0.008, 50.0))),
        MeshMaterial3d(mats.puddle.clone()),
        Transform::from_xyz(-4.0, 0.006, -10.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.7, 0.008, 36.0))),
        MeshMaterial3d(mats.puddle.clone()),
        Transform::from_xyz(4.1, 0.006, -16.0),
    ));

    // Hero puddles, placed at the COMPUTED letter landing points of the
    // sign-to-camera mirror corridor (landing = lerp(camera, letter, t) with
    // t = h_cam / (h_cam + h_letter)): from (0.5, 3.0, 12.0) the Y mirrors to
    // ground (-2.08, 3.3), the V to (-1.59, 5.0), the E to (-1.26, 6.0); the
    // B's landing (z ~6.9) falls below the frame-bottom ground cut (~z 6.2),
    // so no puddle is spent on it. Two long ellipses STRETCHED along the
    // corridor (sz ~2x sx) so the letters read as a streak, not a blob.
    let puddle_mesh = meshes.add(Circle::new(1.0));
    let flat = Quat::from_rotation_x(-FRAC_PI_2);
    // The first puddle's corridor is stretched (sz 2.4, center z 3.6) so the
    // hot magenta landing blob of the Y's reflection pulls up off the
    // frame-bottom cut instead of being cropped by it.
    // The last puddle sits in the Porsche taillight's red-reflection corridor:
    // the light bar (h ~0.9 at the z ~2.6 tail) mirrors to ground z ~4.8 from
    // this camera, the bumper to z ~3.7. Wide enough (1.1 x 0.9) to catch the
    // whole red bar's corridor, still small so the red SSR dominates over the
    // env-sky fallback.
    // The two sign-corridor puddles run long (sz 3.0 / 2.6) and the second one
    // wide (sx 1.5): the stretched mirrors catch more of the V/Y letterforms
    // so the lower-third streak reads as LETTERS, not isolated magenta dabs.
    for (x, z, sx, sz) in [
        (-2.0, 3.6, 1.0, 3.0),
        (-1.55, 6.4, 1.5, 2.6),
        (2.6, 4.5, 1.1, 0.9),
    ] {
        commands.spawn((
            Mesh3d(puddle_mesh.clone()),
            MeshMaterial3d(mats.puddle.clone()),
            Transform::from_xyz(x, 0.012, z)
                .with_rotation(flat)
                .with_scale(Vec3::new(sx, sz, 1.0)),
        ));
    }

    // Manhole disc.
    commands.spawn((
        Mesh3d(puddle_mesh),
        MeshMaterial3d(mats.manhole.clone()),
        Transform::from_xyz(0.8, 0.014, 7.0)
            .with_rotation(flat)
            .with_scale(Vec3::new(0.4, 0.4, 1.0)),
    ));
}

// --- Building A (hero tenement, left) ----------------------------------------------

fn spawn_building_a(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    cache: &mut MeshCache,
    mats: &SharedMats,
    lcg: &mut Lcg,
) {
    // Body: facade plane at x = -7.5, z in [-34, 10].
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(6.0, 20.0, 44.0))),
        MeshMaterial3d(mats.concrete.clone()),
        Transform::from_xyz(-10.5, 10.0, -12.0),
    ));

    // Parallax brick band over the upper facade (y 4..20).
    let brick_mesh = meshes.add(
        Mesh::from(Cuboid::new(0.3, 16.0, 44.0))
            .with_generated_tangents()
            .unwrap(),
    );
    commands.spawn((
        Mesh3d(brick_mesh),
        MeshMaterial3d(mats.brick.clone()),
        Transform::from_xyz(-7.35, 12.0, -12.0),
    ));

    // Ground-floor stucco band, split around the storefront opening (z -1.2..3.2).
    for (z, len) in [(-17.6, 32.8), (6.6, 6.8)] {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(0.35, 4.0, len))),
            MeshMaterial3d(mats.stucco.clone()),
            Transform::from_xyz(-7.325, 2.0, z),
        ));
    }
    // Header over the storefront opening.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.35, 0.6, 4.4))),
        MeshMaterial3d(mats.stucco.clone()),
        Transform::from_xyz(-7.325, 3.7, 1.0),
    ));

    // Pilasters articulating the brick band.
    let pilaster_mesh = meshes.add(Cuboid::new(0.45, 16.0, 0.45));
    for i in 0..10 {
        commands.spawn((
            Mesh3d(pilaster_mesh.clone()),
            MeshMaterial3d(mats.concrete.clone()),
            Transform::from_xyz(-7.225, 12.0, 8.8 - 4.4 * i as f32),
        ));
    }

    // Ledge at the stucco/brick boundary, cornice and parapet on top.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.5, 0.25, 44.0))),
        MeshMaterial3d(mats.concrete.clone()),
        Transform::from_xyz(-7.3, 4.0, -12.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.7, 0.5, 44.0))),
        MeshMaterial3d(mats.concrete.clone()),
        Transform::from_xyz(-7.35, 20.25, -12.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.4, 0.8, 44.0))),
        MeshMaterial3d(mats.concrete.clone()),
        Transform::from_xyz(-7.4, 20.9, -12.0),
    ));

    // Roof water tank: a classic tenement silhouette against the twilight sky.
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(1.3, 2.6))),
        MeshMaterial3d(mats.rust.clone()),
        Transform::from_xyz(-10.0, 22.9, -38.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cone::new(1.35, 0.7))),
        MeshMaterial3d(mats.rust.clone()),
        Transform::from_xyz(-10.0, 24.55, -38.0),
    ));
    let leg_mesh = meshes.add(Cylinder::new(0.08, 1.6));
    for (dx, dz) in [(-0.9, -0.9), (-0.9, 0.9), (0.9, -0.9), (0.9, 0.9)] {
        commands.spawn((
            Mesh3d(leg_mesh.clone()),
            MeshMaterial3d(mats.metal.clone()),
            Transform::from_xyz(-10.0 + dx, 20.8, -38.0 + dz),
        ));
    }

    // Downpipe + conduit pipes hugging the facade near the sign.
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.07, 16.0))),
        MeshMaterial3d(mats.metal.clone()),
        Transform::from_xyz(-7.13, 12.0, -18.6),
    ));
    let conduit_mesh = meshes.add(Cylinder::new(0.05, 16.0));
    for z in [-9.9, -10.15, -10.4] {
        commands.spawn((
            Mesh3d(conduit_mesh.clone()),
            MeshMaterial3d(mats.metal.clone()),
            Transform::from_xyz(-7.14, 12.0, z),
        ));
    }

    // Windows: 5 floors x 18 columns (2 per pilaster bay). The lowest two floors
    // get full modules (reveal frame + pane + sill + lintel); upper floors get
    // simple framed panes. SSAO + contact shadows shade the reveals.
    let frame_mesh = meshes.add(Cuboid::new(0.28, 1.9, 1.3));
    let pane_mesh = meshes.add(Cuboid::new(0.04, 1.6, 1.05));
    let sill_mesh = meshes.add(Cuboid::new(0.18, 0.08, 1.4));
    let ac_mesh = meshes.add(Cuboid::new(0.45, 0.55, 0.75));
    for (floor, y) in [5.4_f32, 8.6, 11.8, 15.0, 18.2].into_iter().enumerate() {
        for bay in 0..9 {
            let bay_z = 8.8 - 4.4 * bay as f32;
            for dz in [-1.5, -2.9] {
                let z = bay_z + dz;
                // Dark reveal frame, slightly proud of the brick face.
                commands.spawn((
                    Mesh3d(frame_mesh.clone()),
                    MeshMaterial3d(mats.win_dark.clone()),
                    Transform::from_xyz(-7.32, y, z),
                ));
                // Glass pane, proud of the frame's STREET face (-7.18): the
                // frame is a solid cuboid, so a pane placed inside its volume
                // is swallowed whole and no window ever reads as lit.
                commands.spawn((
                    Mesh3d(pane_mesh.clone()),
                    MeshMaterial3d(pick_window_material(lcg, mats).clone()),
                    Transform::from_xyz(-7.15, y, z),
                ));
                if floor < 2 {
                    // Sill + lintel on the most-scrutinized lower floors.
                    for sy in [y - 1.0, y + 1.0] {
                        commands.spawn((
                            Mesh3d(sill_mesh.clone()),
                            MeshMaterial3d(mats.concrete.clone()),
                            Transform::from_xyz(-7.22, sy, z),
                        ));
                    }
                    // A few AC units hung under lower windows.
                    if lcg.unit() < 0.33 {
                        commands.spawn((
                            Mesh3d(ac_mesh.clone()),
                            MeshMaterial3d(mats.metal.clone()),
                            Transform::from_xyz(-7.275, y - 1.35, z),
                        ));
                    }
                }
            }
        }
    }

    // Recessed storefront (z -1.2..3.2, to a depth of x = -9.0).
    let return_mesh = meshes.add(Cuboid::new(1.5, 3.4, 0.12));
    for z in [-1.26, 3.26] {
        commands.spawn((
            Mesh3d(return_mesh.clone()),
            MeshMaterial3d(mats.stucco.clone()),
            Transform::from_xyz(-8.25, 1.7, z),
        ));
    }
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.12, 3.4, 4.4))),
        MeshMaterial3d(mats.stucco.clone()),
        Transform::from_xyz(-9.06, 1.7, 1.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.5, 0.12, 4.4))),
        MeshMaterial3d(mats.stucco.clone()),
        Transform::from_xyz(-8.25, 3.46, 1.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.5, 0.1, 4.4))),
        MeshMaterial3d(mats.sidewalk.clone()),
        Transform::from_xyz(-8.25, 0.05, 1.0),
    ));
    // Warm glass door at the back of the recess.
    commands.spawn((
        Mesh3d(meshes.add(Rectangle::new(1.1, 2.3))),
        MeshMaterial3d(mats.door_glow.clone()),
        Transform::from_xyz(-8.99, 1.25, 1.6).with_rotation(Quat::from_rotation_y(FRAC_PI_2)),
    ));

    // Cyan tube border outlining the storefront opening (secondary scheme color).
    let border_frame =
        Transform::from_xyz(-7.12, 1.7, 1.0).with_rotation(Quat::from_rotation_y(FRAC_PI_2));
    spawn_neon_polyline(
        commands,
        meshes,
        cache,
        &mats.secondary_neon,
        border_frame,
        &[
            Vec2::new(-2.3, -1.72),
            Vec2::new(-2.3, 1.78),
            Vec2::new(2.3, 1.78),
            Vec2::new(2.3, -1.72),
            Vec2::new(-2.3, -1.72),
        ],
        0.045,
    );

    // Roller shutters with ridge strips along the dark ground floor.
    let shutter_mesh = meshes.add(Cuboid::new(0.12, 3.0, 2.8));
    let ridge_mesh = meshes.add(Cuboid::new(0.04, 0.05, 2.7));
    for z in [-7.5, -11.5, -15.5] {
        commands.spawn((
            Mesh3d(shutter_mesh.clone()),
            MeshMaterial3d(mats.metal.clone()),
            Transform::from_xyz(-7.08, 1.5, z),
        ));
        for i in 0..6 {
            commands.spawn((
                Mesh3d(ridge_mesh.clone()),
                MeshMaterial3d(mats.metal.clone()),
                Transform::from_xyz(-7.0, 0.4 + 0.5 * i as f32, z),
            ));
        }
    }
}

// --- The blade sign -----------------------------------------------------------------

fn spawn_blade_sign(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    cache: &mut MeshCache,
    mats: &SharedMats,
) {
    // Backing panel, inner edge flush with the facade.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.9, 9.2, 0.22))),
        MeshMaterial3d(mats.sign_panel.clone()),
        Transform::from_xyz(SIGN_X, 9.0, SIGN_Z),
    ));

    // Diagonal brace struts from the facade to the panel.
    let brace_frame = Transform::from_xyz(0.0, 0.0, SIGN_Z);
    for (a, b) in [
        (Vec2::new(-7.5, 13.9), Vec2::new(-5.75, 13.45)),
        (Vec2::new(-7.5, 4.1), Vec2::new(-5.75, 4.55)),
    ] {
        spawn_tube_segment(
            commands,
            meshes,
            cache,
            &mats.metal,
            brace_frame,
            [a, b],
            0.04,
        );
    }

    // Letters B / E / V / Y, stacked top to bottom. One shared material for
    // B / V / Y; the "E" is the dying letter with its own handle.
    let letters: [(&LetterSpec, &Handle<StandardMaterial>); 4] = [
        (&LETTER_B, &mats.primary_neon),
        (&LETTER_E, &mats.dying_neon),
        (&LETTER_V, &mats.primary_neon),
        (&LETTER_Y, &mats.primary_neon),
    ];
    for ((spec, material), y) in letters.into_iter().zip(LETTER_YS) {
        spawn_letter(
            commands,
            meshes,
            cache,
            material,
            Transform::from_xyz(SIGN_X, y, TUBE_Z),
            spec,
            (LETTER_H, TUBE_R),
        );
    }

    // Bird-in-ring emblem above the letters: a full torus ring in the secondary
    // color holding the white-hot alpha-masked Bevy bird.
    commands.spawn((
        Mesh3d(meshes.add(Torus {
            minor_radius: 0.045,
            // 0.48 (not 0.55): pairs with the lowered EMBLEM_Y so the ring
            // gains frame-top headroom without its bottom clipping the B.
            major_radius: 0.48,
        })),
        MeshMaterial3d(mats.secondary_neon.clone()),
        Transform::from_xyz(SIGN_X, EMBLEM_Y, TUBE_Z)
            .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Rectangle::new(0.85, 0.85))),
        MeshMaterial3d(mats.bird.clone()),
        Transform::from_xyz(SIGN_X, EMBLEM_Y, TUBE_Z - 0.06),
    ));
}

// --- Building B (right) ----------------------------------------------------------------

fn spawn_building_b(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    cache: &mut MeshCache,
    mats: &SharedMats,
    lcg: &mut Lcg,
) {
    // Body: facade plane at x = +7.5 facing the street.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(6.0, 16.0, 45.0))),
        MeshMaterial3d(mats.concrete_b.clone()),
        Transform::from_xyz(10.5, 8.0, -27.5),
    ));

    // Ledges between floors.
    let ledge_mesh = meshes.add(Cuboid::new(0.3, 0.18, 45.0));
    for y in [5.2, 8.0, 10.8] {
        commands.spawn((
            Mesh3d(ledge_mesh.clone()),
            MeshMaterial3d(mats.concrete_b.clone()),
            Transform::from_xyz(7.4, y, -27.5),
        ));
    }

    // Sparser windows, but with the same dark-reveal treatment as Building A
    // (frame + proud pane + sill) so the facade reads as punched openings
    // rather than emissive stickers on a black box.
    let frame_mesh = meshes.add(Cuboid::new(0.28, 1.7, 1.3));
    let pane_mesh = meshes.add(Rectangle::new(1.1, 1.4));
    let sill_mesh = meshes.add(Cuboid::new(0.18, 0.08, 1.3));
    let pane_rot = Quat::from_rotation_y(-FRAC_PI_2);
    for y in [3.8, 6.6, 9.4, 12.2] {
        for col in 0..13 {
            let z = -7.0 - 3.2 * col as f32;
            // Leave a clear patch for the BAR sign.
            if (5.5..7.5).contains(&y) && (-31.0..-27.5).contains(&z) {
                continue;
            }
            commands.spawn((
                Mesh3d(frame_mesh.clone()),
                MeshMaterial3d(mats.win_dark.clone()),
                Transform::from_xyz(7.5, y, z),
            ));
            // Bias a few dim-warm panes onto the upper near-camera floors: with
            // the pure LCG roll this face went almost all-dark and Building B
            // read as a solid black mass filling the frame's upper right. The
            // roll still runs for every pane so the LCG stream (and the rest of
            // the window layout) is undisturbed.
            let picked = pick_window_material(lcg, mats);
            let forced = if y > 12.0 && col == 1 {
                Some(&mats.win_warm_b)
            } else if (y > 12.0 && col == 3) || ((9.0..10.0).contains(&y) && col == 2) {
                Some(&mats.win_dim_a)
            } else {
                None
            };
            // Pane proud of the frame's street face (7.36) -- the solid frame
            // cuboid would otherwise swallow it.
            commands.spawn((
                Mesh3d(pane_mesh.clone()),
                MeshMaterial3d(forced.unwrap_or(picked).clone()),
                Transform::from_xyz(7.34, y, z).with_rotation(pane_rot),
            ));
            commands.spawn((
                Mesh3d(sill_mesh.clone()),
                MeshMaterial3d(mats.concrete_b.clone()),
                Transform::from_xyz(7.3, y - 0.78, z),
            ));
        }
    }

    // The +Z end cap (z = -5) faces the camera and fills the frame's upper
    // right; with no openings it read as a featureless black void from the
    // frame edge to the first lit window. Three framed panes break it up --
    // dim/dark tiers ONLY (never warm) so nothing at the frame edge competes
    // with the sign, and FIXED material handles rather than
    // `pick_window_material`: the LCG roll order is part of the composition
    // and must not shift. Rectangle faces +Z natively, so the panes need no
    // rotation; the frame cuboid turns 90 degrees to lie in the cap plane.
    let cap_rot = Quat::from_rotation_y(FRAC_PI_2);
    for (x, y, material) in [
        (9.2, 9.4, &mats.win_dim_b),
        (11.0, 6.6, &mats.win_dark),
        (10.2, 12.2, &mats.win_dim_a),
    ] {
        commands.spawn((
            Mesh3d(frame_mesh.clone()),
            MeshMaterial3d(mats.win_dark.clone()),
            Transform::from_xyz(x, y, -5.0).with_rotation(cap_rot),
        ));
        commands.spawn((
            Mesh3d(pane_mesh.clone()),
            MeshMaterial3d((*material).clone()),
            Transform::from_xyz(x, y, -4.84),
        ));
    }

    // The broken "BAR" blade: red stick letters on a cabinet, duty-cycled in
    // `animate_neon` together with its slaved red point light.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.16, 1.4, 2.8))),
        MeshMaterial3d(mats.sign_panel.clone()),
        Transform::from_xyz(7.52, 6.4, -29.5),
    ));
    let bar_base =
        Transform::from_xyz(7.38, 6.4, -29.5).with_rotation(Quat::from_rotation_y(-FRAC_PI_2));
    for (spec, dx) in [(&LETTER_B, -0.8), (&LETTER_A, 0.0), (&LETTER_R, 0.8)] {
        spawn_letter(
            commands,
            meshes,
            cache,
            &mats.bar_neon,
            bar_base * Transform::from_xyz(dx, 0.0, 0.0),
            spec,
            (0.9, 0.045),
        );
    }

    // Magenta torus ring accent further up the facade (primary scheme color, so
    // it hums and retints with the hero letters).
    commands.spawn((
        Mesh3d(meshes.add(Torus {
            minor_radius: 0.05,
            major_radius: 0.55,
        })),
        MeshMaterial3d(mats.primary_neon.clone()),
        Transform::from_xyz(7.4, 9.0, -22.0).with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
    ));
}

// --- Skyline: street terminator, towers, billboards, cables ------------------------------

fn spawn_skyline(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    cache: &mut MeshCache,
    mats: &SharedMats,
    lcg: &mut Lcg,
) {
    // Distant emissives must overshoot: at 60-90 m the aerial perspective, the
    // street haze slab, and a ~1-2 px on-screen footprint under TAA swallow the
    // near-building window values entirely, so these two run far hotter than
    // their kin (doubled again after only 2-3 background windows survived to
    // the frame and the horizon read as a void).
    let distant_warm = materials.add(lit_window_material(LinearRgba {
        red: 72.0,
        green: 38.0,
        blue: 13.0,
        alpha: 1.0,
    }));
    let distant_dim = materials.add(lit_window_material(LinearRgba {
        red: 6.0,
        green: 3.2,
        blue: 1.2,
        alpha: 1.0,
    }));

    // Building C terminates the street but leaves the sky gap above for the
    // twilight band at the vanishing point. NOT the zero-reflectance `tower`
    // silhouette material: this face fills prime center frame under a 10-50 nit
    // twilight sky, and with no sky response it rendered as a dead black
    // rectangle -- a touch of reflectance lets the sky rim it.
    let building_c = materials.add(StandardMaterial {
        base_color: Color::srgb(0.04, 0.038, 0.045),
        perceptual_roughness: 0.9,
        reflectance: 0.2,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(24.0, 10.0, 6.0))),
        MeshMaterial3d(building_c),
        Transform::from_xyz(4.0, 5.0, -63.0),
    ));
    // 14 panes, four of them FORCED lit (boosted distant materials): relying on
    // the 62%-dark LCG roll left every pane dark and nothing broke the void.
    let c_pane_mesh = meshes.add(Rectangle::new(0.8, 1.1));
    for i in 0..14 {
        let x = -6.0 + 19.0 * lcg.unit();
        let y = 1.5 + 6.5 * lcg.unit();
        let material = match i {
            0 | 7 => &distant_warm,
            3 | 10 => &distant_dim,
            _ => pick_window_material(lcg, mats),
        };
        commands.spawn((
            Mesh3d(c_pane_mesh.clone()),
            MeshMaterial3d(material.clone()),
            Transform::from_xyz(x, y, -59.95),
        ));
    }

    // Background tower silhouettes flanking the twilight band. The leftmost
    // tower is pushed out to x -18.5 so its right edge stops intruding into the
    // sky gap and occluding the warm band core at the vanishing point.
    let towers: [(Vec3, Vec3); 4] = [
        (Vec3::new(-18.5, 13.0, -78.0), Vec3::new(14.0, 26.0, 10.0)),
        (Vec3::new(15.0, 16.0, -84.0), Vec3::new(12.0, 32.0, 9.0)),
        (Vec3::new(-3.0, 10.0, -92.0), Vec3::new(18.0, 20.0, 12.0)),
        (Vec3::new(22.0, 11.0, -72.0), Vec3::new(10.0, 22.0, 8.0)),
    ];
    let tower_window_mesh = meshes.add(Rectangle::new(0.6, 0.9));
    for (center, size) in towers {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::from_size(size))),
            MeshMaterial3d(mats.tower.clone()),
            Transform::from_translation(center),
        ));
        // Sparse lit windows on the camera-facing (+Z) face.
        let front_z = center.z + size.z / 2.0 + 0.05;
        let cols = (size.x / 1.8) as i32;
        let rows = (size.y / 2.2) as i32;
        for col in 0..cols {
            for row in 0..rows {
                if lcg.unit() >= 0.12 {
                    continue;
                }
                let x = center.x - size.x / 2.0 + 1.2 + 1.8 * col as f32;
                let y = center.y - size.y / 2.0 + 1.4 + 2.2 * row as f32;
                let material = if lcg.unit() < 0.5 {
                    &distant_warm
                } else {
                    &distant_dim
                };
                commands.spawn((
                    Mesh3d(tower_window_mesh.clone()),
                    MeshMaterial3d(material.clone()),
                    Transform::from_xyz(x, y, front_z),
                ));
            }
        }
    }

    // Two distant emissive billboards flanking the vanishing point, snapped
    // flush INSIDE their tower faces so they read as mounted, not floating.
    // Emissives ~8x the lit-window ladder, doubled twice: at 73-80 m the
    // original values vanished entirely at ev100 7.75, and even the 4x pass
    // left the twilight band's flanks reading as faint smudges instead of
    // signage punctuating the horizon.
    commands.spawn((
        Mesh3d(meshes.add(Rectangle::new(5.0, 2.8))),
        MeshMaterial3d(materials.add(lit_window_material(LinearRgba {
            red: 110.0,
            green: 30.0,
            blue: 2.4,
            alpha: 1.0,
        }))),
        Transform::from_xyz(-12.0, 15.0, -72.9),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Rectangle::new(4.5, 2.5))),
        MeshMaterial3d(materials.add(lit_window_material(LinearRgba {
            red: 3.2,
            green: 80.0,
            blue: 128.0,
            alpha: 1.0,
        }))),
        Transform::from_xyz(11.5, 13.0, -79.4),
    ));

    // Power cables spanning the canyon: chained capsule segments approximating
    // a catenary with ~0.4 m of mid-span droop (a perfectly straight cylinder
    // reads as a wire-frame artifact). The nearest span is routed at z = 2.0,
    // which projects just above the frame top, so no cable slices through the
    // blade sign's letters or emblem; the asymmetric end heights stand in for
    // the old roll tilt.
    for (y_left, y_right, z) in [(8.6, 8.3, 2.0), (9.3, 9.0, -16.0), (8.2, 8.5, -26.0)] {
        let droop = 0.4;
        let points: Vec<Vec2> = (0..=4)
            .map(|i| {
                let t = i as f32 / 4.0;
                let y = y_left + (y_right - y_left) * t - droop * 4.0 * t * (1.0 - t);
                Vec2::new(-7.5 + 15.0 * t, y)
            })
            .collect();
        spawn_neon_polyline(
            commands,
            meshes,
            cache,
            &mats.cable,
            Transform::from_xyz(0.0, 0.0, z),
            &points,
            0.018,
        );
    }
}

// --- Street furniture -------------------------------------------------------------------

fn spawn_street_furniture(commands: &mut Commands, meshes: &mut Assets<Mesh>, mats: &SharedMats) {
    // Two sodium streetlamps on the right sidewalk.
    let pole_mesh = meshes.add(Cylinder::new(0.08, 6.2));
    let arm_mesh = meshes.add(Cuboid::new(1.6, 0.1, 0.1));
    let head_mesh = meshes.add(Cuboid::new(0.55, 0.14, 0.26));
    // A chunky emissive lens (0.45 x 0.10 x 0.22) hung BELOW the head housing:
    // the old 2 cm sliver tucked under the housing presented almost no emissive
    // area to a camera looking up from street level, so the lamps lit pools
    // without ever reading as light sources themselves.
    let glow_mesh = meshes.add(Cuboid::new(0.45, 0.10, 0.22));
    // The near lamp stands at z -14: at z -8 the parked Porsche occludes its
    // pool from this camera, leaving the whole midground unlit.
    for (i, z) in [-14.0_f32, -28.0].into_iter().enumerate() {
        commands.spawn((
            Mesh3d(pole_mesh.clone()),
            MeshMaterial3d(mats.metal.clone()),
            Transform::from_xyz(4.9, 3.28, z),
        ));
        commands.spawn((
            Mesh3d(arm_mesh.clone()),
            MeshMaterial3d(mats.metal.clone()),
            Transform::from_xyz(4.15, 6.32, z),
        ));
        commands.spawn((
            Mesh3d(head_mesh.clone()),
            MeshMaterial3d(mats.metal.clone()),
            Transform::from_xyz(3.45, 6.3, z),
        ));
        commands.spawn((
            Mesh3d(glow_mesh.clone()),
            MeshMaterial3d(mats.lamp_glow.clone()),
            Transform::from_xyz(3.45, 6.14, z),
        ));

        // Pool math: see LAMP_LUMENS. The aim leans hard toward the street
        // CENTER (-0.55 in x) -- the old near-vertical aim from x 3.45 dropped
        // the pool on the sidewalk and gutter, leaving the roadway black -- and
        // the 0.85 cone widens it into a sodium circle spanning the lane. Only
        // the NEAR lamp pays for shadow maps + contact shadows; both get
        // volumetric cones.
        let near = i == 0;
        commands.spawn((
            SpotLight {
                color: Color::srgb(1.0, 0.78, 0.5),
                intensity: LAMP_LUMENS,
                range: 22.0,
                radius: 0.15,
                outer_angle: 0.85,
                inner_angle: 0.35,
                shadow_maps_enabled: near,
                contact_shadows_enabled: near,
                ..default()
            },
            Transform::from_xyz(3.45, 5.95, z)
                .looking_to(Vec3::new(-0.55, -1.0, -0.12).normalize(), Vec3::Z),
            VolumetricLight,
        ));
    }

    // Steam grate, flush with the road under the animated plume (at (3.5, 5.0)
    // with the plume; see `spawn_fog` -- its x 3.05 near edge just clears the
    // taillight puddle's red corridor, which lands at x <= ~3.04). The
    // dedicated near-black `grate` material, NOT the shared manhole iron: lit
    // from above, the manhole's 0.08 base floodlit into a bright "dinner
    // plate" ellipse on the road while the plume itself stayed invisible --
    // exactly inverted. The grate must stay a dark fixture.
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.45, 0.05))),
        MeshMaterial3d(mats.grate.clone()),
        Transform::from_xyz(3.5, 0.025, 5.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.36, 0.054))),
        MeshMaterial3d(mats.grate.clone()),
        Transform::from_xyz(3.5, 0.025, 5.0),
    ));

    // Deliberately NO "injector" light inside the steam plume: every attempt
    // to light the column from within (25k-180k lm, y 0.9-2.2, ranges 3-5 m)
    // floodlit the grate disc and sidewalk into a naked bright ellipse long
    // before the steam itself picked up visible in-scatter -- at this exposure
    // the surfaces always win. A dark grate beats a lit ellipse, so the plume
    // leans on the street's own lights (the sign spill is within range) for a
    // faint volumetric presence instead.
}

// --- The Porsche -------------------------------------------------------------------------

fn spawn_porsche(commands: &mut Commands, asset_server: &AssetServer) {
    // This branch spawns glTF scenes via `WorldAssetRoot` (NOT `SceneRoot`).
    // The asset's nose faces +Z, so a half-turn parks it nose-down-street with
    // the tail light bar toward the camera.
    commands.spawn((
        WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("porsche.glb"))),
        Transform::from_xyz(2.9, 0.0, 0.5).with_rotation(Quat::from_rotation_y(PI)),
    ));

    // Headlights: volumetric cones punching down-street through the ground mist
    // toward the twilight band (toggle with C). Shadows off; the fog does the
    // work.
    for x in [2.2, 3.6] {
        commands.spawn((
            SpotLight {
                color: Color::srgb(1.0, 0.94, 0.82),
                intensity: HEADLIGHT_LUMENS,
                range: 45.0,
                radius: 0.05,
                outer_angle: 0.75,
                inner_angle: 0.25,
                shadow_maps_enabled: false,
                ..default()
            },
            // -0.30 slope, not near-parallel: at grazing incidence the beams
            // leave no visible pool on the asphalt at all; this steeper drop
            // lands a nearer, larger warm splash on the road around z -4..-10.
            // The -0.22 yaw walks both beams toward the street CENTER so the
            // splash and cones peek past the car's left flank from this camera
            // (dead ahead they were fully occluded by the body). The 0.75 cone
            // widens the splash into a proper headlight wash; the denser
            // beam-path fog slab (see `spawn_fog`) keeps the volumetric cones
            // reading despite the spread-out luminance.
            Transform::from_xyz(x, 0.68, -1.75)
                .looking_to(Vec3::new(-0.22, -0.30, -1.0).normalize(), Vec3::Y),
            VolumetricLight,
            Headlight,
        ));
    }
}

/// Boosts the Porsche's tail-light bar into HDR red once the glTF instance is
/// ready, and clamps the body paint's roughness. The asset's `tex_shiny`
/// material has an LDR emissive factor that would never bloom; cloning it and
/// raising `emissive` (keeping the emissive texture as a mask) makes the bar
/// glow. The `paint` material ships glossier than this scene can afford: under
/// the small undersampled env map + SSR jitter the body sparkled with white
/// specular fireflies, so its roughness is floored at 0.35. Matching uses
/// `GltfMaterialName` -- the `Name` component on mesh entities holds mesh names
/// like `boot.003_0`, not material names.
fn boost_taillights(
    ready: On<WorldInstanceReady>,
    children: Query<&Children>,
    mesh_materials: Query<(&MeshMaterial3d<StandardMaterial>, &GltfMaterialName)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
) {
    let mut boosted: Option<Handle<StandardMaterial>> = None;
    let mut matte_paint: Option<Handle<StandardMaterial>> = None;
    for descendant in children.iter_descendants(ready.entity) {
        let Ok((handle, material_name)) = mesh_materials.get(descendant) else {
            continue;
        };
        match material_name.0.as_str() {
            "tex_shiny" => {
                let hot_handle = match &boosted {
                    Some(hot_handle) => hot_handle.clone(),
                    None => {
                        // Skip (rather than insert a placeholder handle) if the
                        // source material is not loaded yet: that leaves
                        // `boosted` as `None` so the asset-drift warn below
                        // still fires.
                        let Some(material) = materials.get(handle.id()) else {
                            continue;
                        };
                        let mut hot = material.clone();
                        // Far hotter than the target brightness: the retained
                        // emissive_texture mask texels multiply this factor
                        // DOWN hard, so the factor must overshoot harder (900;
                        // 280 left a dim strip that never crossed the GT7
                        // glare threshold) for the masked result to clip
                        // toward peak red, pick up a halo, and leave a red SSR
                        // streak in the puddle under the tail.
                        hot.emissive = LinearRgba {
                            red: 900.0,
                            green: 22.0,
                            blue: 12.0,
                            alpha: 1.0,
                        };
                        let hot_handle = materials.add(hot);
                        boosted = Some(hot_handle.clone());
                        hot_handle
                    }
                };
                commands
                    .entity(descendant)
                    .insert(MeshMaterial3d(hot_handle));
            }
            "paint" => {
                let matte_handle = match &matte_paint {
                    Some(matte_handle) => matte_handle.clone(),
                    None => {
                        let Some(material) = materials.get(handle.id()) else {
                            continue;
                        };
                        let mut matte = material.clone();
                        matte.perceptual_roughness = matte.perceptual_roughness.max(0.35);
                        let matte_handle = materials.add(matte);
                        matte_paint = Some(matte_handle.clone());
                        matte_handle
                    }
                };
                commands
                    .entity(descendant)
                    .insert(MeshMaterial3d(matte_handle));
            }
            _ => {}
        }
    }
    if boosted.is_none() {
        warn!(
            "porsche.glb: 'tex_shiny' material missing or not loaded; tail-light boost skipped (asset drift?)"
        );
    }
    if matte_paint.is_none() {
        warn!(
            "porsche.glb: 'paint' material missing or not loaded; body roughness clamp skipped (asset drift?)"
        );
    }
}

// --- Scene lights --------------------------------------------------------------------------

fn spawn_lights(commands: &mut Commands) {
    let scheme = &SCHEMES[0];

    // Emissive surfaces do not illuminate; this point light fakes the sign tubes'
    // photometric output -- magenta wash on the brick, sidewalk, car roof, and a
    // colored bloom in the haze. It hums and flickers with the tubes.
    commands.spawn((
        PointLight {
            color: Color::srgb(
                scheme.primary_glow[0],
                scheme.primary_glow[1],
                scheme.primary_glow[2],
            ),
            intensity: SPILL_LUMENS,
            // 32 m: the magenta diffuse term must reach the z 4..9 corridor
            // where the SSR letter streak lands, anchoring the sign to the road.
            range: 32.0,
            // A fat 2.0 m source: at radius 0.5 the analytic specular term
            // painted a hard bare-bulb disc on the brick between the windows,
            // and at 1.2 the mirror puddles still imaged it as a structureless
            // blown magenta orb at the frame bottom; 2.0 melts that specular
            // image into a broad soft glow.
            radius: 2.0,
            shadow_maps_enabled: false,
            ..default()
        },
        // Tucked against the facade just DOWN-street of the sign panel: the
        // mirror-direction spot its specular highlight lands on (window glass
        // and brick) is then hidden behind the panel from this camera (it
        // otherwise blooms into a floating pink disk on the panes). Height 11.0
        // is specular-LOAD-BEARING: a light at height h mirrors on the ground
        // at D*h_cam/(h_cam+h) from the camera, and h >= ~10.5 pushes that
        // hotspot inside the frame-bottom ground cut (~5.8 m) -- at h 9.0 the
        // analytic specular landed IN the hero puddle and blew out into a
        // giant magenta ball that swallowed the letter reflections.
        Transform::from_xyz(-6.3, 11.0, -12.3),
        VolumetricLight,
        SignSpillLight,
    ));

    // Cyan pool spilling from the storefront onto the sidewalk and car flank.
    commands.spawn((
        SpotLight {
            color: Color::srgb(
                scheme.secondary_glow[0],
                scheme.secondary_glow[1],
                scheme.secondary_glow[2],
            ),
            intensity: ENTRANCE_LUMENS,
            range: 14.0,
            radius: 0.3,
            outer_angle: 1.1,
            inner_angle: 0.6,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(-7.0, 3.4, 1.0)
            .looking_to(Vec3::new(0.5, -1.0, 0.0).normalize(), Vec3::Z),
        EntranceLight,
    ));

    // A second, small cyan spot aimed INTO the storefront recess: the sidewalk
    // spot above never lights the recess interior, which left the tube border
    // outlining a pure black void (100k, not 50k: at 50k the recess interior
    // still measured ~7/255 and the shutter door never read). Shares
    // `EntranceLight` so it retints with the scheme.
    commands.spawn((
        SpotLight {
            color: Color::srgb(
                scheme.secondary_glow[0],
                scheme.secondary_glow[1],
                scheme.secondary_glow[2],
            ),
            intensity: 100_000.0,
            range: 5.0,
            radius: 0.2,
            outer_angle: 1.0,
            inner_angle: 0.5,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(-6.7, 3.0, 1.0)
            .looking_to(Vec3::new(-0.6, -0.5, 0.0).normalize(), Vec3::Z),
        EntranceLight,
    ));

    // Red stutter light slaved EXACTLY to the broken BAR sign's emissive duty
    // cycle -- the wall and the fog must flash with the tubes or the fake shows.
    commands.spawn((
        PointLight {
            color: Color::srgb(1.0, 0.08, 0.05),
            intensity: 0.0,
            range: 12.0,
            radius: 0.3,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(7.0, 6.4, -29.0),
        VolumetricLight,
        BarStutterLight,
    ));
}

// --- Animation -----------------------------------------------------------------------------
//
// Everything is driven by VIRTUAL elapsed time (no `rand`, no wall clock), so the
// CI harness's fixed_frame_time reproduces identical frames.

/// Duty cycle of the dying "E": a 7.3 s loop opening with a 0.45 s ~24 Hz
/// sputter, plus one brown-out double-blink (dip to 12%) mid-cycle.
///
/// CI-LOAD-BEARING: with `fixed_frame_time` 0.0166 the harness screenshots at
/// frame 300 (t = 4.98 s, phase 4.98) and frame 800 (t = 13.28 s, phase 5.98).
/// Both instants must land in the steady ON region -- keep every OFF/dip window
/// clear of them.
fn dying_e_factor(t: f32) -> f32 {
    let phase = t % 7.3;
    if phase < 0.45 {
        if ((phase * 24.0) as u32).is_multiple_of(2) {
            1.0
        } else {
            0.04
        }
    } else if (3.20..3.32).contains(&phase) || (3.38..3.50).contains(&phase) {
        0.12
    } else {
        1.0
    }
}

/// Duty cycle of the broken BAR sign: a 9.1 s mostly-off loop with short red
/// stutter bursts. "Off" is a faint 0.05 ember, not 0.0, so the fixture still
/// reads as a present-but-dying sign (a dull red trace over Building B's black
/// wall) in any capture instant that lands outside a burst.
///
/// CI-LOAD-BEARING: the harness captures at virtual t = 4.98 s (phase 4.98) and
/// t = 13.28 s (phase 4.18). Both land inside the steady double-burst windows
/// below, so the BAR red (and its slaved point light rimming Building B's wall)
/// is ON in every CI frame -- without it the building was a solid black mass
/// filling the frame's upper right at the captured instants.
fn bar_sign_factor(t: f32) -> f32 {
    const EMBER: f32 = 0.05;
    let phase = t % 9.1;
    if phase < 0.4 {
        if ((phase * 24.0) as u32).is_multiple_of(3) {
            EMBER
        } else {
            1.0
        }
    } else if phase < 2.6 {
        EMBER
    } else if phase < 2.95 {
        if ((phase * 30.0) as u32).is_multiple_of(2) {
            1.0
        } else {
            0.15
        }
    } else if (4.10..4.45).contains(&phase) || (4.85..5.15).contains(&phase) {
        // The dying double-flash; steady (not chattering) so the CI capture
        // phases 4.18 and 4.98 are deterministically lit.
        0.85
    } else if phase < 6.8 {
        EMBER
    } else if phase < 7.1 {
        0.85
    } else {
        EMBER
    }
}

/// TV glow: random-steps in [0.6, 1.4] every ~0.18 s (deterministic hash of the
/// step index; no `rand`).
fn tv_factor(t: f32) -> f32 {
    let step = (t / 0.18) as u32;
    let mut lcg = Lcg(step.wrapping_mul(0x9E37_79B9).wrapping_add(0x0BEE_F00D));
    lcg.next();
    0.6 + 0.8 * lcg.unit()
}

fn scaled(color: LinearRgba, factor: f32) -> LinearRgba {
    LinearRgba {
        red: color.red * factor,
        green: color.green * factor,
        blue: color.blue * factor,
        alpha: 1.0,
    }
}

fn animate_neon(
    time: Res<Time>,
    mut state: ResMut<NeonScene>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut spill_lights: Query<&mut PointLight, (With<SignSpillLight>, Without<BarStutterLight>)>,
    mut bar_lights: Query<&mut PointLight, With<BarStutterLight>>,
    mut entrance_lights: Query<&mut SpotLight, (With<EntranceLight>, Without<Headlight>)>,
) {
    let t = time.elapsed_secs();
    state.flicker_secs = (state.flicker_secs - time.delta_secs()).max(0.0);

    // Mains-hum shimmer on the hero tubes and their spill light.
    let hum = 1.0 + 0.02 * ops::sin(t * 8.3) + 0.01 * ops::sin(t * 21.7);
    // F-key burst: hard 22 Hz chatter for 0.7 s on tubes AND spill together.
    let burst = if state.flicker_secs > 0.0 {
        if ((state.flicker_secs * 22.0) as u32).is_multiple_of(3) {
            0.04
        } else {
            1.0
        }
    } else {
        1.0
    };
    let tubes = hum * burst;

    let scheme = &SCHEMES[state.scheme];
    if let Some(mut material) = materials.get_mut(&state.primary_mat) {
        material.emissive = scaled(scheme.primary, tubes);
    }
    if let Some(mut material) = materials.get_mut(&state.dying_mat) {
        material.emissive = scaled(scheme.primary, tubes * dying_e_factor(t));
    }
    if let Some(mut material) = materials.get_mut(&state.secondary_mat) {
        material.emissive = scaled(scheme.secondary, tubes);
    }
    if let Some(mut material) = materials.get_mut(&state.bird_mat) {
        material.emissive = scaled(BIRD_EMISSIVE, burst);
    }

    let bar = bar_sign_factor(t);
    if let Some(mut material) = materials.get_mut(&state.bar_mat) {
        material.emissive = scaled(BAR_RED, bar);
    }
    if let Some(mut material) = materials.get_mut(&state.tv_mat) {
        material.emissive = scaled(TV_EMISSIVE, tv_factor(t));
    }

    for mut light in &mut spill_lights {
        light.color = Color::srgb(
            scheme.primary_glow[0],
            scheme.primary_glow[1],
            scheme.primary_glow[2],
        );
        light.intensity = SPILL_LUMENS * tubes;
    }
    for mut light in &mut entrance_lights {
        light.color = Color::srgb(
            scheme.secondary_glow[0],
            scheme.secondary_glow[1],
            scheme.secondary_glow[2],
        );
    }
    // Slaved exactly to the BAR emissive duty cycle.
    for mut light in &mut bar_lights {
        light.intensity = bar * BAR_RED.red * BAR_LUMENS_PER_EMISSIVE;
    }
}

fn animate_world(
    time: Res<Time>,
    mut headlights: Query<&mut SpotLight, With<Headlight>>,
    mut plumes: Query<&mut FogVolume, With<SteamPlume>>,
) {
    let t = time.elapsed_secs();

    // (The asphalt used to drift a normal map here to keep SSR fresh under TAA;
    // the normal map is gone -- SSR's own per-frame jitter does that job now.)

    // +/-2% filament tremor on the headlights at 1.3 Hz.
    for mut light in &mut headlights {
        light.intensity = HEADLIGHT_LUMENS * (1.0 + 0.02 * ops::sin(t * 1.3 * TAU));
    }

    // Breathing steam plume (the +/-30% swing tracks the 0.75 base density).
    for mut plume in &mut plumes {
        plume.density_factor = 0.75 + 0.225 * ops::sin(t * 0.4);
    }
}

// --- Input -----------------------------------------------------------------------------------

fn handle_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut commands: Commands,
    mut state: ResMut<NeonScene>,
    camera: Single<(Entity, &mut Exposure, &mut GranTurismo7Params), With<Camera3d>>,
    mut headlights: Query<&mut Visibility, With<Headlight>>,
    mut display_target: Single<&mut DisplayTarget, With<PrimaryWindow>>,
) {
    let (camera_entity, mut exposure, mut gt7_params) = camera.into_inner();

    for (key, idx) in [
        (KeyCode::Digit1, 0),
        (KeyCode::Digit2, 1),
        (KeyCode::Digit3, 2),
    ] {
        if keyboard.just_pressed(key) {
            state.scheme = idx;
        }
    }

    // The killer ablation: the street snaps from neon mirror to env-map-flat.
    if keyboard.just_pressed(KeyCode::KeyR) {
        state.ssr_on = !state.ssr_on;
        if state.ssr_on {
            commands.entity(camera_entity).insert(ssr_settings());
        } else {
            commands
                .entity(camera_entity)
                .remove::<ScreenSpaceReflections>();
        }
    }

    // Live HDR <-> SDR output switch; GT7 re-keys its curve from the RESOLVED
    // transfer (reported on the HUD with a 1-frame lag).
    if keyboard.just_pressed(KeyCode::KeyH) {
        state.hdr_requested = !state.hdr_requested;
        **display_target = if state.hdr_requested {
            hdr_display_target(state.paper_white_nits)
        } else {
            DisplayTarget::SDR_SRGB
        };
    }

    // Paper-white trim in 40-nit steps (peak stays 1000): lets the user match
    // the desktop's "SDR content brightness" slider so the H toggle compares
    // like for like. The DisplayTarget is only re-written in HDR mode -- the
    // SDR_SRGB target ignores paper white, and the value is remembered for the
    // next H press either way.
    let mut paper_white_changed = false;
    if keyboard.just_pressed(KeyCode::BracketLeft) {
        state.paper_white_nits = (state.paper_white_nits - 40.0).clamp(120.0, 480.0);
        paper_white_changed = true;
    }
    if keyboard.just_pressed(KeyCode::BracketRight) {
        state.paper_white_nits = (state.paper_white_nits + 40.0).clamp(120.0, 480.0);
        paper_white_changed = true;
    }
    if paper_white_changed && state.hdr_requested {
        **display_target = hdr_display_target(state.paper_white_nits);
    }

    if keyboard.just_pressed(KeyCode::KeyB) {
        state.bloom_mode = (state.bloom_mode + 1) % BLOOM_MODE_NAMES.len();
        match state.bloom_mode {
            mode @ 0..=3 => {
                commands.entity(camera_entity).insert(Bloom {
                    intensity: BLOOM_INTENSITY,
                    scatter: BloomScatterModel::Gt7Glare {
                        f_number: BLOOM_F_NUMBERS[mode],
                    },
                    ..Bloom::NATURAL
                });
            }
            4 => {
                // Legacy parametric curve for comparison.
                commands.entity(camera_entity).insert(Bloom::NATURAL);
            }
            _ => {
                commands.entity(camera_entity).remove::<Bloom>();
            }
        }
    }

    // Camera-skew bleach (0.3) vs hue-stable ICtCp (1.0) on the over-peak tubes.
    if keyboard.just_pressed(KeyCode::KeyG) {
        state.blend_idx = (state.blend_idx + 1) % BLEND_RATIOS.len();
        gt7_params.blend_ratio = BLEND_RATIOS[state.blend_idx];
    }

    if keyboard.just_pressed(KeyCode::KeyC) {
        state.headlights_on = !state.headlights_on;
        for mut visibility in &mut headlights {
            *visibility = if state.headlights_on {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
        }
    }

    if keyboard.just_pressed(KeyCode::KeyF) {
        state.flicker_secs = 0.7;
    }

    // Continuous exposure swing: the street, sky, and lamp pools brighten and
    // darken while the neon stays locked -- emissive bypasses `Exposure` at the
    // default `emissive_exposure_weight: 0.0`.
    if keyboard.pressed(KeyCode::ArrowUp) {
        exposure.ev100 = (exposure.ev100 - 2.0 * time.delta_secs()).clamp(EV100_MIN, EV100_MAX);
    }
    if keyboard.pressed(KeyCode::ArrowDown) {
        exposure.ev100 = (exposure.ev100 + 2.0 * time.delta_secs()).clamp(EV100_MIN, EV100_MAX);
    }
}

// --- HUD -------------------------------------------------------------------------------------

fn update_hud(
    state: Res<NeonScene>,
    camera: Single<&Exposure, With<Camera3d>>,
    resolved: Single<Option<&WindowResolvedTransfer>, With<PrimaryWindow>>,
    mut hud: Single<&mut Text, With<HudText>>,
) {
    // `WindowResolvedTransfer` is absent until the first surface configuration
    // and lags requests by one frame; treat absence as pending rather than lying
    // about the output mode.
    let output = match resolved.map(|resolved| resolved.0) {
        None => "Output: pending",
        Some(DisplayTransfer::ScRgbLinear) => "Output: scRGB HDR (peak 1000 nits)",
        Some(DisplayTransfer::ExtendedSrgb) => "Output: extended sRGB HDR (encoded)",
        Some(DisplayTransfer::Pq) | Some(DisplayTransfer::Hlg) => "Output: PQ/HLG HDR",
        Some(DisplayTransfer::Srgb) => "Output: SDR sRGB",
    };

    hud.0 = format!(
        "1/2/3 - neon scheme [{}]\n\
         R - reflections (SSR) [{}]\n\
         H - HDR output [{}]\n\
         B - bloom [{}]\n\
         G - GT7 blend ratio [{:.1}]\n\
         C - headlights [{}]\n\
         F - flicker burst\n\
         [/] - paper white: {} nits\n\
         Up/Down - exposure ev100 [{:.2}]\n\
         {}",
        SCHEMES[state.scheme].name,
        if state.ssr_on { "on" } else { "off" },
        if state.hdr_requested {
            "scRGB requested"
        } else {
            "SDR"
        },
        BLOOM_MODE_NAMES[state.bloom_mode],
        BLEND_RATIOS[state.blend_idx],
        if state.headlights_on { "on" } else { "off" },
        state.paper_white_nits as i32,
        camera.ev100,
        output,
    );
}
