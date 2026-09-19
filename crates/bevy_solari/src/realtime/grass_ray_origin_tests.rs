//! Numerical regression scenarios for the shader's grass-only origin policy.
//! Uses f32 arithmetic, oct8x2 round trips, and intersections against the true
//! triangle plane (not the quantized normal). Shader compilation is validated separately.
use bevy_math::{Vec2, Vec3};

fn oct16(n: Vec3) -> Vec3 {
    let v = n / n.abs().element_sum();
    let sign = |x: f32| if x >= 0.0 { 1.0 } else { -1.0 };
    let e = if v.z >= 0.0 {
        Vec2::new(v.x, v.y)
    } else {
        Vec2::new((1.0 - v.y.abs()) * sign(v.x), (1.0 - v.x.abs()) * sign(v.y))
    };
    let e = ((e * 0.5 + Vec2::splat(0.5)) * 255.0).round() / 255.0;
    let f = e * 2.0 - Vec2::ONE;
    let mut n = Vec3::new(f.x, f.y, 1.0 - f.x.abs() - f.y.abs());
    let t = (-n.z).clamp(0.0, 1.0);
    n.x -= sign(n.x) * t;
    n.y -= sign(n.y) * t;
    n.normalize()
}

fn offset(p: Vec3, n: Vec3, d: Vec3) -> Vec3 {
    let cosine = n.dot(d);
    if cosine.abs() <= 0.02 {
        return p;
    }
    triangle_offset(p, n, d)
}

fn triangle_offset(p: Vec3, n: Vec3, d: Vec3) -> Vec3 {
    let cosine = n.dot(d);
    if !(cosine.abs() > 0.000001) {
        return p;
    }
    let n = if cosine > 0.0 { n } else { -n };
    let distance = (4.0 * f32::EPSILON * n.abs().dot(p.abs())).max(0.00001);
    p + n * distance
}

fn plane_hit(p: Vec3, d: Vec3, plane: Vec3, normal: Vec3) -> f32 {
    normal.dot(plane - p) / normal.dot(d)
}

#[test]
fn one_ulp_depth_error_no_longer_intersects_its_blade() {
    let plane = Vec3::new(300.0, 45.0, 1442.0);
    let n = Vec3::Z;
    let encoded = oct16(n);
    for side in [-1.0, 1.0] {
        let d = Vec3::new(0.99, 0.0, 0.1 * side).normalize();
        let z = f32::from_bits(plane.z.to_bits().wrapping_add_signed(-(side as i32)));
        let p = Vec3::new(plane.x, plane.y, z);
        assert!(
            plane_hit(p, d, plane, n) > 0.001,
            "legacy origin produces a self hit"
        );
        let shifted = offset(p, encoded, d);
        assert!(
            plane_hit(shifted, d, plane, n) < 0.0,
            "own plane is behind ray"
        );
        let neighbor = plane + n * (0.005 * side);
        assert!(
            plane_hit(shifted, d, neighbor, n) > 0.001,
            "nearby real caster remains visible"
        );
    }
}

#[test]
fn quantization_sign_reversal_does_not_introduce_a_grazing_self_hit() {
    let n = Vec3::new(0.52699474, 0.84768432, 0.06089206).normalize();
    let d = Vec3::new(-0.83326383, 0.49988333, 0.23619491).normalize();
    let q = oct16(n);
    assert!(n.dot(d) < 0.0 && q.dot(d) > 0.0);
    let p = Vec3::ZERO;
    // Without the guard even the 10um minimum invents a ~1cm self hit.
    assert!(plane_hit(p + q * 0.00001, d, p, n) > 0.001);
    assert_eq!(offset(p, q, d), p);
    assert_eq!(offset(p, q, -d), p);
}

#[test]
fn round_trip_normals_never_choose_wrong_side_outside_guard() {
    for x in -16..=16 {
        for y in -16..=16 {
            for z in [-16, -3, 3, 16] {
                let n = Vec3::new(x as f32, y as f32, z as f32).normalize();
                let q = oct16(n);
                let axis = if n.x.abs() < 0.8 { Vec3::X } else { Vec3::Y };
                let tangent = n.cross(axis).normalize();
                for cosine in [-0.1, -0.03, -0.001, 0.001, 0.03, 0.1] {
                    let d = (tangent + n * cosine).normalize();
                    if q.dot(d).abs() > 0.02 {
                        assert!(q.dot(d) * n.dot(d) > 0.0);
                    } else {
                        assert_eq!(offset(Vec3::splat(1442.0), q, d), Vec3::splat(1442.0));
                    }
                }
            }
        }
    }
}

#[test]
fn ordinary_origins_unchanged_and_scaled_offsets_remain_small() {
    for coordinate in [0.0, 100.0, 1000.0, 1442.0, 10000.0] {
        let p = Vec3::splat(coordinate);
        assert_eq!(offset(p, Vec3::ZERO, Vec3::Z), p);
        let shifted = offset(p, oct16(Vec3::Z), Vec3::Z);
        let displacement = (shifted - p).length();
        assert!(displacement > 0.0 && displacement < 0.005);
        let separation = if coordinate < 10000.0 { 0.005 } else { 0.01 };
        assert!(plane_hit(shifted, Vec3::Z, p + Vec3::Z * separation, Vec3::Z) > 0.001);
    }
    // This policy is a few-ULP heuristic, not a universal bound: at 10km its
    // offset approaches 5mm, so preserving submillimetre geometry is not promised.
}

#[test]
fn either_triangle_winding_offsets_toward_outgoing_hemisphere() {
    let p = Vec3::new(300.0, 45.0, 1442.0);
    for n in [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(0.5, -0.8, 0.3).normalize(),
    ] {
        for d in [n, -n] {
            for winding in [n, -n] {
                let shifted = offset(p, oct16(winding), d);
                assert!((shifted - p).dot(d) > 0.0);
                assert!(plane_hit(shifted, d, p, n) < 0.0);
            }
        }
    }
}

#[test]
fn rt_normal_offsets_inside_packed_normal_deadzone() {
    let plane = Vec3::new(300.0, 45.0, 1442.0);
    for cosine in [-0.01_f32, -0.001, -0.00001, 0.00001, 0.001, 0.01] {
        let side = cosine.signum();
        let d = Vec3::new((1.0 - cosine * cosine).sqrt(), 0.0, cosine);
        let p = Vec3::new(
            plane.x,
            plane.y,
            f32::from_bits(plane.z.to_bits().wrapping_add_signed(-(side as i32))),
        );
        assert!(plane_hit(p, d, plane, Vec3::Z) > 0.001);
        assert_eq!(
            offset(p, Vec3::Z, d),
            p,
            "old reference kept the bad origin"
        );
        for n in [Vec3::Z, -Vec3::Z] {
            let shifted = triangle_offset(p, n, d);
            assert!(plane_hit(shifted, d, plane, Vec3::Z) < 0.0);
            assert!(plane_hit(shifted, d, plane + Vec3::Z * (0.005 * side), Vec3::Z) > 0.001);
        }
    }
}

#[test]
fn rt_normal_keeps_arithmetic_guard_and_legacy_paths() {
    let p = Vec3::new(300.0, 45.0, 1442.0);
    for cosine in [-0.0000001, 0.0, 0.0000001] {
        let d = Vec3::new(1.0, 0.0, cosine).normalize();
        assert_eq!(triangle_offset(p, Vec3::Z, d), p);
    }
    assert_eq!(triangle_offset(p, Vec3::ZERO, Vec3::Z), p);
    assert_eq!(
        triangle_offset(p, Vec3::Z, Vec3::Z),
        offset(p, Vec3::Z, Vec3::Z)
    );
    let shader = include_str!("../scene/bindings.wesl");
    assert!(shader.contains("if !(abs(n_dot_d) > geometric_normal.w) { return position; }"));
    assert!(include_str!("gbuffer_utils.wesl").contains("vec4(receiver_geometric_normal, 0.02)"));
    assert!(include_str!("grass_receiver.wesl").contains("vec4(receiver.normal, 0.000001)"));
    assert!(shader.contains("if !(abs(n_dot_d) > 0.000001) { return position; }"));
    let diagnostic = include_str!("shadow_diagnostics.wesl");
    assert!(diagnostic
        .contains("origin = offset_triangle_ray_origin(origin, receiver.normal, direction)"));
    assert!(diagnostic.contains("trace_visibility(origin, vec4(direction, 0.0))"));
}

#[test]
fn grazing_ray_no_longer_hits_the_same_finite_blade_triangle() {
    // First triangle of the shared 30cm-high, 2cm-wide curved blade template.
    let root = Vec3::new(300.0, 45.0, 1442.0);
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
        origin.z = f32::from_bits(origin.z.to_bits().wrapping_add_signed(-(side as i32)));
        assert!(
            hit(origin, direction),
            "legacy ray hits within the actual blade triangle"
        );
        assert_eq!(offset(origin, n, direction), origin);
        for winding in [n, -n] {
            assert!(!hit(triangle_offset(origin, winding, direction), direction));
        }
    }
}
