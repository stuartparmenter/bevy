//! CPU model of `resolve_triangle_anchor` in `triangle_anchor.wesl`. The fixtures use
//! different current and previous vertices, transforms and instance slots, because
//! identity mappings over static geometry cannot expose history misuse.

use bevy_math::Vec3;

const INSTANCE_NOT_PRESENT_THIS_FRAME: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Anchor {
    instance: u32,
    triangle: u32,
    barycentrics: [f32; 2],
}

struct Instance {
    current: Vec<[Vec3; 3]>,
    previous: Vec<[Vec3; 3]>,
    current_translation: Vec3,
    previous_translation: Vec3,
}

fn resolve(
    anchor: Anchor,
    previous_frame: bool,
    translations: &[u32],
    instances: &[Instance],
) -> Option<Vec3> {
    let id = if previous_frame {
        *translations.get(anchor.instance as usize)?
    } else {
        anchor.instance
    };
    if previous_frame && id == INSTANCE_NOT_PRESENT_THIS_FRAME {
        return None;
    }
    let instance = instances.get(id as usize)?;
    let (vertices, translation) = if previous_frame {
        (&instance.previous, instance.previous_translation)
    } else {
        (&instance.current, instance.current_translation)
    };
    let triangle = vertices.get(anchor.triangle as usize)?;
    let [u, v] = anchor.barycentrics;
    if !(u >= -1e-5 && v >= -1e-5 && u <= 1.00001 && v <= 1.00001 && u + v <= 1.00001) {
        return None;
    }
    Some(
        triangle[0]
            + (triangle[1] - triangle[0]) * u
            + (triangle[2] - triangle[0]) * v
            + translation,
    )
}

fn moving_triangle() -> Instance {
    Instance {
        previous: vec![[Vec3::ZERO, Vec3::X, Vec3::Y]],
        current: vec![[Vec3::Z, Vec3::X + Vec3::Z, Vec3::Y + Vec3::Z]],
        previous_translation: Vec3::new(10.0, 20.0, 30.0),
        current_translation: Vec3::new(100.0, 200.0, 300.0),
    }
}

fn anchor(instance: u32) -> Anchor {
    Anchor {
        instance,
        triangle: 0,
        barycentrics: [0.25, 0.5],
    }
}

#[test]
fn previous_anchor_follows_remapped_instance_previous_vertices_and_transform() {
    let instances = [moving_triangle()];
    // Previous slot 2 is current slot 0. Indexing slot 2 directly, or using the
    // current vertices or transform, gives a different answer.
    assert_eq!(
        resolve(
            anchor(2),
            true,
            &[
                INSTANCE_NOT_PRESENT_THIS_FRAME,
                INSTANCE_NOT_PRESENT_THIS_FRAME,
                0
            ],
            &instances
        ),
        Some(Vec3::new(10.25, 20.5, 30.0))
    );
    assert_eq!(
        resolve(anchor(0), false, &[], &instances),
        Some(Vec3::new(100.25, 200.5, 301.0))
    );
}

#[test]
fn previous_anchor_rejects_destroyed_reused_and_out_of_range_slots() {
    let instances = [moving_triangle()];
    // Current slot 0 is occupied by another owner: the translation takes
    // precedence over current geometry that looks valid.
    assert_eq!(
        resolve(
            anchor(0),
            true,
            &[INSTANCE_NOT_PRESENT_THIS_FRAME],
            &instances
        ),
        None
    );
    assert!(resolve(anchor(0), false, &[], &instances).is_some());
    assert_eq!(resolve(anchor(2), true, &[0], &instances), None);
    assert_eq!(resolve(anchor(0), true, &[9], &instances), None);
    assert_eq!(resolve(anchor(u32::MAX), false, &[], &instances), None);
}

#[test]
fn anchor_rejects_a_missing_triangle() {
    let instances = [moving_triangle()];
    let stale = Anchor {
        triangle: 1,
        ..anchor(0)
    };
    assert_eq!(resolve(stale, true, &[0], &instances), None);
    assert_eq!(resolve(stale, false, &[], &instances), None);
}

#[test]
fn previous_anchor_does_not_follow_a_neighbor_triangle_or_current_vertices() {
    let mut instance = moving_triangle();
    instance
        .previous
        .push([Vec3::splat(80.0), Vec3::splat(81.0), Vec3::splat(82.0)]);
    let expected = Vec3::new(10.25, 20.5, 30.0);
    let mut instances = [instance];
    assert_eq!(resolve(anchor(0), true, &[0], &instances), Some(expected));
    instances[0].current[0] = [Vec3::splat(-500.0); 3];
    assert_eq!(resolve(anchor(0), true, &[0], &instances), Some(expected));
    assert_ne!(
        resolve(
            Anchor {
                triangle: 1,
                ..anchor(0)
            },
            true,
            &[0],
            &instances
        ),
        Some(expected)
    );
}

#[test]
fn anchor_rejects_invalid_barycentrics_but_keeps_ray_edge_roundoff() {
    let instances = [moving_triangle()];
    for barycentrics in [
        [f32::NAN, 0.0],
        [0.0, f32::NAN],
        [f32::INFINITY, 0.0],
        [-0.1, 0.0],
        [0.8, 0.8],
    ] {
        assert_eq!(
            resolve(
                Anchor {
                    barycentrics,
                    ..anchor(0)
                },
                true,
                &[0],
                &instances
            ),
            None
        );
    }
    assert!(resolve(
        Anchor {
            barycentrics: [-0.000001, 0.5],
            ..anchor(0)
        },
        true,
        &[0],
        &instances
    )
    .is_some());
}

#[test]
fn current_anchor_does_not_require_previous_frame_data() {
    let mut instance = moving_triangle();
    instance.previous.clear();
    let instances = [instance];
    assert_eq!(
        resolve(anchor(0), false, &[], &instances),
        Some(Vec3::new(100.25, 200.5, 301.0))
    );
    assert_eq!(resolve(anchor(0), true, &[0], &instances), None);
}

/// The vertices are f32, so the reference is those exact inputs evaluated in f64.
#[test]
fn edge_form_avoids_large_world_weighted_sum_error() {
    // One `[a, b, c]` triple of vertex coordinates per axis.
    let axes = [
        [1422.013_f32, 1422.031, 1421.998],
        [31.037, 31.183, 31.216],
        [1442.017, 1442.063, 1442.091],
    ];
    let mut improvements = 0;
    for i in 0..=64 {
        for j in 0..=64 - i {
            let u = i as f32 / 64.0;
            let v = j as f32 / 64.0;
            for (axis, [a, b, c]) in axes.into_iter().enumerate() {
                let expected = f64::from(a)
                    + (f64::from(b) - f64::from(a)) * f64::from(u)
                    + (f64::from(c) - f64::from(a)) * f64::from(v);
                let edge = a + (b - a) * u + (c - a) * v;
                let weighted = a * (1.0 - u - v) + b * u + c * v;
                let ulp = f32::from_bits(a.to_bits() + 1) - a;
                let error = (f64::from(edge) - expected).abs();
                assert!(
                    error <= f64::from(ulp),
                    "axis {axis}: error {error}, ulp {ulp}"
                );
                if error < (f64::from(weighted) - expected).abs() {
                    improvements += 1;
                }
            }
        }
    }
    assert!(
        improvements > 100,
        "fixture must exercise cancellation loss"
    );
}
