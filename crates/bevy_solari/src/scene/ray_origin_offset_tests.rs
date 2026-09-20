//! Numerical scenarios for `offset_surface_ray_origin` in `bindings.wesl`, mirrored in f32
//! and intersected against the true triangle plane.

use bevy_math::Vec3;

/// `EXACT_NORMAL_TANGENT_GUARD` in `bindings.wesl`.
const EXACT_NORMAL_TANGENT_GUARD: f32 = 0.000001;

/// A tangent guard for a face normal that is only known to within a few degrees.
const WIDE_GUARD: f32 = 0.05;

/// A sine inside [`WIDE_GUARD`], used to perturb normals the way an imprecise source would.
const NORMAL_ERROR: f32 = 0.01;

/// A world position far enough from the origin for one f32 ULP to be about 0.1 mm.
const FAR_POSITION: Vec3 = Vec3::new(300.0, 45.0, 1442.0);

/// `offset_surface_ray_origin`. A NaN cosine fails the comparisons and leaves `p` unchanged.
fn offset_surface(p: Vec3, n: Vec3, guard: f32, d: Vec3) -> Vec3 {
    let cosine = n.dot(d);
    if cosine.abs() > guard && cosine.abs() > EXACT_NORMAL_TANGENT_GUARD {
        let n = if cosine > 0.0 { n } else { -n };
        let distance = (4.0 * f32::EPSILON * n.abs().dot(p.abs())).max(0.00001);
        p + n * distance
    } else {
        p
    }
}

/// Ray parameter at which `p + d * t` crosses the plane through `plane` with normal `normal`.
fn plane_hit(p: Vec3, d: Vec3, plane: Vec3, normal: Vec3) -> f32 {
    normal.dot(plane - p) / normal.dot(d)
}

/// `n` tilted by [`NORMAL_ERROR`] toward each tangent direction, plus `n` itself.
fn imprecise_normals(n: Vec3) -> [Vec3; 5] {
    let axis = if n.x.abs() < 0.8 { Vec3::X } else { Vec3::Y };
    let tangent = n.cross(axis).normalize();
    let bitangent = n.cross(tangent);
    let cosine = (1.0 - NORMAL_ERROR * NORMAL_ERROR).sqrt();
    [
        n,
        n * cosine + tangent * NORMAL_ERROR,
        n * cosine - tangent * NORMAL_ERROR,
        n * cosine + bitangent * NORMAL_ERROR,
        n * cosine - bitangent * NORMAL_ERROR,
    ]
}

/// `z` moved one ULP against `side`, so a ray leaving toward `side` starts behind its own plane.
fn one_ulp_behind(z: f32, side: f32) -> f32 {
    f32::from_bits(z.to_bits().wrapping_add_signed(-(side as i32)))
}

#[test]
fn one_ulp_position_error_does_not_intersect_its_own_surface() {
    let plane = FAR_POSITION;
    let n = Vec3::Z;
    for side in [-1.0, 1.0] {
        let d = Vec3::new(0.99, 0.0, 0.1 * side).normalize();
        let p = Vec3::new(plane.x, plane.y, one_ulp_behind(plane.z, side));
        assert!(
            plane_hit(p, d, plane, n) > 0.001,
            "unshifted origin self-intersects"
        );
        for normal in imprecise_normals(n) {
            let shifted = offset_surface(p, normal, WIDE_GUARD, d);
            assert!(
                plane_hit(shifted, d, plane, n) < 0.0,
                "own plane is behind the ray"
            );
            let neighbor = plane + n * (0.005 * side);
            assert!(
                plane_hit(shifted, d, neighbor, n) > 0.001,
                "an occluder 5 mm away stays visible"
            );
        }
    }
}

#[test]
fn zero_normal_is_a_no_op_and_offsets_stay_small() {
    for coordinate in [0.0, 100.0, 1000.0, 1442.0, 10000.0] {
        let p = Vec3::splat(coordinate);
        for guard in [0.0, EXACT_NORMAL_TANGENT_GUARD, WIDE_GUARD] {
            for d in [Vec3::Z, -Vec3::Z, Vec3::X, Vec3::splat(f32::NAN)] {
                assert_eq!(offset_surface(p, Vec3::ZERO, guard, d), p);
            }
        }

        for normal in imprecise_normals(Vec3::Z) {
            let shifted = offset_surface(p, normal, WIDE_GUARD, Vec3::Z);
            let displacement = (shifted - p).length();
            assert!(displacement > 0.0 && displacement < 0.005);
            // The offset scales with position: at 10 km it approaches 5 mm, so geometry
            // closer than that to the surface is not preserved there.
            let separation = if coordinate < 10000.0 { 0.005 } else { 0.01 };
            assert!(plane_hit(shifted, Vec3::Z, p + Vec3::Z * separation, Vec3::Z) > 0.001);
        }
    }
}

#[test]
fn either_winding_offsets_toward_the_outgoing_side() {
    let p = FAR_POSITION;
    for n in [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(0.5, -0.8, 0.3).normalize(),
    ] {
        for d in [n, -n] {
            for winding in [n, -n] {
                for normal in imprecise_normals(winding) {
                    let shifted = offset_surface(p, normal, WIDE_GUARD, d);
                    assert!((shifted - p).dot(d) > 0.0);
                    assert!(plane_hit(shifted, d, p, n) < 0.0);
                }
            }
        }
    }
}

#[test]
fn exact_normal_offsets_inside_a_wider_guard_band() {
    let plane = FAR_POSITION;
    for cosine in [-0.01_f32, -0.001, -0.00001, 0.00001, 0.001, 0.01] {
        let side = cosine.signum();
        let d = Vec3::new((1.0 - cosine * cosine).sqrt(), 0.0, cosine);
        let p = Vec3::new(plane.x, plane.y, one_ulp_behind(plane.z, side));
        assert!(plane_hit(p, d, plane, Vec3::Z) > 0.001);
        assert_eq!(
            offset_surface(p, Vec3::Z, WIDE_GUARD, d),
            p,
            "the wide guard leaves a grazing ray on its self-intersecting origin"
        );
        for n in [Vec3::Z, -Vec3::Z] {
            let shifted = offset_surface(p, n, EXACT_NORMAL_TANGENT_GUARD, d);
            assert!(plane_hit(shifted, d, plane, Vec3::Z) < 0.0);
            assert!(plane_hit(shifted, d, plane + Vec3::Z * (0.005 * side), Vec3::Z) > 0.001);
        }
    }
}

#[test]
fn exact_normal_keeps_its_tangent_guard() {
    let p = FAR_POSITION;
    for cosine in [-0.0000001, 0.0, 0.0000001] {
        let d = Vec3::new(1.0, 0.0, cosine).normalize();
        assert_eq!(offset_surface(p, Vec3::Z, EXACT_NORMAL_TANGENT_GUARD, d), p);
        // A caller cannot select a guard below the exact one.
        assert_eq!(offset_surface(p, Vec3::Z, 0.0, d), p);
    }
    // The guard only gates the offset; it does not scale it.
    assert_eq!(
        offset_surface(p, Vec3::Z, EXACT_NORMAL_TANGENT_GUARD, Vec3::Z),
        offset_surface(p, Vec3::Z, WIDE_GUARD, Vec3::Z)
    );
}

#[test]
fn grazing_ray_does_not_hit_its_own_finite_triangle() {
    // A tall, narrow, slightly twisted triangle at a large world coordinate.
    let root = FAR_POSITION;
    let a = root + Vec3::new(-0.01, 0.0, 0.0);
    let b = root + Vec3::new(0.01, 0.0, 0.0);
    let c = root + Vec3::new(0.0096, 0.06, 0.00504);
    let e1 = b - a;
    let e2 = c - a;
    let n = e1.cross(e2).normalize();
    let point = a + e1 * 0.35 + e2 * 0.25;
    let hit = |origin: Vec3, direction: Vec3| {
        let h = direction.cross(e2);
        let inverse = 1.0 / e1.dot(h);
        let s = origin - a;
        let u = inverse * s.dot(h);
        let q = s.cross(e1);
        let v = inverse * direction.dot(q);
        let t = inverse * e2.dot(q);
        u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t >= 0.001
    };
    for side in [-1.0_f32, 1.0] {
        let direction = (e2.normalize() + n * (0.01 * side)).normalize();
        let mut origin = point;
        origin.z = one_ulp_behind(origin.z, side);
        assert!(
            hit(origin, direction),
            "unshifted ray hits within its own triangle"
        );
        assert_eq!(offset_surface(origin, n, WIDE_GUARD, direction), origin);
        for winding in [n, -n] {
            let shifted = offset_surface(origin, winding, EXACT_NORMAL_TANGENT_GUARD, direction);
            assert!(!hit(shifted, direction));
        }
    }
}
