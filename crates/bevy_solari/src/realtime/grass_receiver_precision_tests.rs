//! Numerical regression for the world-space barycentric reconstruction used
//! by cached receiver anchors. The RT vertices themselves are f32; compare
//! arithmetic against those exact inputs evaluated in f64, not ideal geometry.

#[test]
fn edge_form_anchor_avoids_large_world_weighted_sum_error() {
    let shader = include_str!("grass_receiver_geometry.wesl");
    assert!(shader.contains("points[0] + (points[1] - points[0]) * bary.x + (points[2] - points[0]) * bary.y"));
    let vertices = [
        [1422.013_f32, 31.037, 1442.017],
        [1422.031, 31.183, 1442.063],
        [1421.998, 31.216, 1442.091],
    ];
    let mut improvements = 0;
    for i in 0..=64 {
        for j in 0..=64 - i {
            let u = i as f32 / 64.0;
            let v = j as f32 / 64.0;
            for axis in 0..3 {
                let a = vertices[0][axis];
                let b = vertices[1][axis];
                let c = vertices[2][axis];
                let expected = f64::from(a) + (f64::from(b) - f64::from(a)) * f64::from(u)
                    + (f64::from(c) - f64::from(a)) * f64::from(v);
                let edge = a + (b - a) * u + (c - a) * v;
                let weighted = a * (1.0 - u - v) + b * u + c * v;
                let ulp = f32::from_bits(a.to_bits() + 1) - a;
                let error = (f64::from(edge) - expected).abs();
                assert!(error <= f64::from(ulp), "axis {axis}: error {error}, ulp {ulp}");
                if error < (f64::from(weighted) - expected).abs() {
                    improvements += 1;
                }
            }
        }
    }
    assert!(improvements > 100, "fixture must exercise cancellation loss");
}
