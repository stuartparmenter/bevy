//! Calibrates the luminance of an HDR display.
//!
//! The window requests HDR output with [`DisplayTarget::hdr`]. Each step shows a pattern
//! and sets one luminance field of the window's [`DisplayTarget`]. A field left `None`
//! uses what the display reports, or a default for the color space. The text shows what
//! the display reports, the color space the surface uses, and the values the renderer
//! resolves from all of them.
//!
//! The peak luminance step works best with the display in its game or HGIG mode, where
//! it does no tonemapping of its own.

use bevy::{
    camera::ScalingMode,
    core_pipeline::tonemapping::{DebandDither, Tonemapping},
    prelude::*,
    render::view::WindowDisplayHdrInfo,
    window::{
        DisplayTarget, PrimaryWindow, ResolvedDisplayTarget, WindowSurfaceColorSpaces,
        MAX_LUMINANCE_NITS,
    },
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_display_target: Some(DisplayTarget {
                hdr: true,
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::BLACK))
        .init_resource::<Step>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (change_step, adjust_value, update_pattern, update_text).chain(),
        )
        .run();
}

#[derive(Resource, Default, Clone, Copy, PartialEq, Eq)]
enum Step {
    #[default]
    PeakLuminance,
    PaperWhite,
    BlackLevel,
}

impl Step {
    const ALL: [Step; 3] = [Step::PeakLuminance, Step::PaperWhite, Step::BlackLevel];

    fn name(self) -> &'static str {
        match self {
            Step::PeakLuminance => "Peak luminance",
            Step::PaperWhite => "Paper white",
            Step::BlackLevel => "Black level",
        }
    }

    fn instruction(self) -> &'static str {
        match self {
            Step::PeakLuminance => {
                "Raise the peak luminance until the small square disappears into the card."
            }
            Step::PaperWhite => "Set paper white so the card looks like a white page.",
            Step::BlackLevel => {
                "Lower the black level until the square disappears, then raise it until the \
                square is barely visible."
            }
        }
    }

    fn get(self, target: &DisplayTarget) -> Option<f32> {
        match self {
            Step::PeakLuminance => target.peak_luminance_nits,
            Step::PaperWhite => target.paper_white_nits,
            Step::BlackLevel => target.min_luminance_nits,
        }
    }

    fn field(self, target: &mut DisplayTarget) -> &mut Option<f32> {
        match self {
            Step::PeakLuminance => &mut target.peak_luminance_nits,
            Step::PaperWhite => &mut target.paper_white_nits,
            Step::BlackLevel => &mut target.min_luminance_nits,
        }
    }

    fn resolved(self, target: &ResolvedDisplayTarget) -> f32 {
        match self {
            Step::PeakLuminance => target.peak_luminance_nits,
            Step::PaperWhite => target.paper_white_nits,
            Step::BlackLevel => target.min_luminance_nits,
        }
    }

    /// The range the step can set, in nits.
    fn range(self) -> (f32, f32) {
        match self {
            Step::PeakLuminance => (100.0, MAX_LUMINANCE_NITS),
            Step::PaperWhite => (80.0, 500.0),
            Step::BlackLevel => (0.001, 5.0),
        }
    }

    /// The luminance of the background, the card and the square, relative to paper white.
    fn levels(self, target: &ResolvedDisplayTarget) -> [f32; 3] {
        let paper_white = target.paper_white_nits;
        match self {
            // The square is at the maximum luminance. At the display's real peak, the card
            // and the square clip to the same luminance.
            Step::PeakLuminance => [
                0.0,
                target.peak_luminance_nits / paper_white,
                MAX_LUMINANCE_NITS / paper_white,
            ],
            Step::PaperWhite => [0.15, 1.0, 0.35],
            Step::BlackLevel => [0.0, 0.0, target.min_luminance_nits / paper_white],
        }
    }
}

/// The background, the card and the square, in that order.
#[derive(Component)]
struct PatternQuad(usize);

#[derive(Component)]
struct Pattern;

const VIEW_HEIGHT: f32 = 8.0;

/// HDR displays reach a higher peak over a small area than over the whole screen, so the
/// card is about a tenth of the view.
const QUADS: [(Vec2, f32); 3] = [
    (Vec2::new(16.0, 8.0), 0.0),
    (Vec2::new(4.6, 1.8), 0.1),
    (Vec2::splat(1.2), 0.2),
];

/// How fast the value changes while an arrow key is held. It is multiplied or divided by
/// `e^ADJUST_RATE` per second, because equal ratios of luminance look like equal steps.
const ADJUST_RATE: f32 = 1.0;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        // The pattern values go to the display unchanged.
        Tonemapping::Linear,
        DebandDither::Disabled,
        Projection::from(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: VIEW_HEIGHT,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands
        .spawn((Pattern, Transform::default(), Visibility::Hidden))
        .with_children(|parent| {
            for (index, (size, z)) in QUADS.into_iter().enumerate() {
                parent.spawn((
                    PatternQuad(index),
                    Mesh3d(meshes.add(Rectangle::from_size(size))),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: Color::BLACK,
                        unlit: true,
                        ..default()
                    })),
                    Transform::from_xyz(0.0, 0.0, z),
                ));
            }
        });

    commands.spawn((
        Text::default(),
        Node {
            position_type: PositionType::Absolute,
            top: px(12),
            left: px(12),
            ..default()
        },
    ));
}

fn change_step(keys: Res<ButtonInput<KeyCode>>, mut step: ResMut<Step>) {
    if keys.just_pressed(KeyCode::ArrowDown) {
        *step = match *step {
            Step::PeakLuminance => Step::PaperWhite,
            _ => Step::BlackLevel,
        };
    }
    if keys.just_pressed(KeyCode::ArrowUp) {
        *step = match *step {
            Step::BlackLevel => Step::PaperWhite,
            _ => Step::PeakLuminance,
        };
    }
}

/// Left and Right change the current step's value, starting from the resolved one. R sets
/// it back to `None`.
fn adjust_value(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    step: Res<Step>,
    window: Single<(&mut DisplayTarget, Option<&ResolvedDisplayTarget>), With<PrimaryWindow>>,
) {
    let (mut target, resolved) = window.into_inner();
    let direction =
        f32::from(keys.pressed(KeyCode::ArrowRight)) - f32::from(keys.pressed(KeyCode::ArrowLeft));
    let value = if keys.just_pressed(KeyCode::KeyR) {
        None
    } else if let Some(resolved) = resolved
        && direction != 0.0
    {
        let (min, max) = step.range();
        let value = step
            .get(&target)
            .unwrap_or_else(|| step.resolved(resolved))
            .max(min);
        let value = value * ops::exp(direction * ADJUST_RATE * time.delta_secs());
        Some(value.clamp(min, max))
    } else {
        return;
    };
    // Writing an unchanged value would still mark the `DisplayTarget` as changed.
    if step.get(&target) != value {
        *step.field(&mut target) = value;
    }
}

fn update_pattern(
    step: Res<Step>,
    resolved: Single<Option<&ResolvedDisplayTarget>, With<PrimaryWindow>>,
    mut pattern: Single<&mut Visibility, With<Pattern>>,
    quads: Query<(&PatternQuad, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let resolved = *resolved;
    // Without an HDR surface, the pattern would clip at SDR white.
    let is_hdr = resolved.is_some_and(|resolved| resolved.color_space.is_hdr());
    pattern.set_if_neq(if is_hdr {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    });

    let levels = step.levels(resolved.unwrap_or(&ResolvedDisplayTarget::default()));
    for (quad, material) in &quads {
        let level = levels[quad.0];
        let color = Color::linear_rgb(level, level, level);
        // Writing the material every frame would upload it every frame.
        if materials
            .get(&material.0)
            .is_some_and(|material| material.base_color != color)
        {
            materials.get_mut(&material.0).unwrap().base_color = color;
        }
    }
}

fn update_text(
    step: Res<Step>,
    window: Single<
        (
            &DisplayTarget,
            Option<&ResolvedDisplayTarget>,
            Option<&WindowSurfaceColorSpaces>,
            Option<&WindowDisplayHdrInfo>,
        ),
        With<PrimaryWindow>,
    >,
    mut text: Single<&mut Text>,
) {
    let (target, resolved, surface, info) = *window;
    let nits = |value: Option<f32>| value.map_or_else(|| "not reported".to_string(), format_nits);

    let mut out = String::from("(Up/Down) Step  (Left/Right) Adjust  (R) Reset\n\n");
    match surface {
        Some(surface) if surface.resolved.is_hdr() => {
            out.push_str(step.instruction());
            out.push_str("\n\n");
        }
        _ => out.push_str(
            "The output is SDR. Turn on HDR in the system display settings to calibrate.\n\n",
        ),
    }

    out.push_str("Calibrated (resolved):\n");
    for s in Step::ALL {
        let calibrated = s.get(target).map_or("None".to_string(), format_nits);
        let resolved = resolved.map(|resolved| s.resolved(resolved));
        out.push_str(&format!(
            "{} {}: {calibrated} ({})\n",
            if s == *step { ">" } else { " " },
            s.name(),
            nits(resolved),
        ));
    }

    out.push_str("\nDisplay reports:\n");
    match info {
        Some(info) => {
            let luminance = info.luminance.unwrap_or_default();
            out.push_str(&format!(
                "  Peak: {}\n  Black: {}\n  SDR white: {}\n  Headroom: {}\n",
                nits(luminance.max_nits),
                nits(luminance.min_nits),
                nits(luminance.sdr_white_nits),
                info.tone_map_headroom().map_or_else(
                    || "not reported".to_string(),
                    |headroom| format!("{headroom:.2}x")
                ),
            ));
        }
        None => out.push_str("  Nothing\n"),
    }

    if let Some(surface) = surface {
        out.push_str(&format!(
            "\nSurface color space: {:?}\nSupported: {:?}\n",
            surface.resolved, surface.supported
        ));
    }

    text.set_if_neq(Text(out));
}

fn format_nits(nits: f32) -> String {
    if nits < 10.0 {
        format!("{nits:.3} nits")
    } else {
        format!("{nits:.0} nits")
    }
}
