//! CPU reference scenarios for interpreting the GPU blocker diagnostic. These
//! test fixtures distinguish grouping errors from valid curled-blade occlusion;
//! they complement shader compilation, not a hardware ray-query integration test.
#[derive(Clone, Copy)]
struct Hit {
    instance: u32,
    primitive: u32,
    mask: u8,
    group: u32,
    distance: f32,
}
#[derive(Debug, PartialEq)]
enum Class {
    Unmatched,
    Clear,
    SameTriangle,
    SameBlade,
    OtherGrass,
    Solid,
}
fn classify(receiver: Option<Hit>, blocker: Option<Hit>) -> Class {
    let Some(a) = receiver.filter(|h| h.mask == 2 && h.group == 9) else {
        return Class::Unmatched;
    };
    let Some(b) = blocker else {
        return Class::Clear;
    };
    if a.instance == b.instance {
        if a.primitive == b.primitive {
            return Class::SameTriangle;
        }
        if a.primitive / a.group == b.primitive / a.group {
            return Class::SameBlade;
        }
    }
    if b.mask == 2 && b.group > 0 {
        Class::OtherGrass
    } else {
        Class::Solid
    }
}
fn grass(instance: u32, primitive: u32, distance: f32) -> Hit {
    Hit {
        instance,
        primitive,
        mask: 2,
        group: 9,
        distance,
    }
}
#[test]
fn triangle_blade_and_instance_boundaries_are_distinct() {
    let r = grass(7, 8, 0.0);
    assert_eq!(classify(Some(r), Some(r)), Class::SameTriangle);
    assert_eq!(classify(Some(r), Some(grass(7, 0, 0.01))), Class::SameBlade);
    assert_eq!(
        classify(Some(r), Some(grass(7, 9, 0.01))),
        Class::OtherGrass
    );
    assert_eq!(
        classify(Some(r), Some(grass(8, 8, 0.01))),
        Class::OtherGrass
    );
    assert_eq!(
        classify(
            Some(r),
            Some(Hit {
                mask: 255,
                group: 0,
                ..grass(8, 8, 0.01)
            })
        ),
        Class::Solid
    );
    assert_eq!(classify(Some(r), None), Class::Clear);
}
#[test]
fn missing_or_proxy_receiver_is_not_reported_as_clear_or_self() {
    assert_eq!(classify(None, None), Class::Unmatched);
    let far = Hit {
        group: 1,
        ..grass(7, 8, 0.0)
    };
    assert_eq!(classify(Some(far), Some(far)), Class::Unmatched);
    let solid = Hit {
        mask: 255,
        ..grass(7, 8, 0.0)
    };
    assert_eq!(classify(Some(solid), Some(solid)), Class::Unmatched);
}
#[test]
fn closest_blocker_not_first_traversed_controls_classification() {
    let r = grass(7, 8, 0.0);
    // Hardware traversal order is not distance order: the other blade may be
    // accepted first although a receiver-triangle self hit is closer.
    let hits = [grass(8, 0, 0.5), grass(7, 8, 0.003), grass(7, 7, 0.02)];
    let closest = hits
        .into_iter()
        .min_by(|a, b| a.distance.total_cmp(&b.distance));
    assert_eq!(classify(Some(r), closest), Class::SameTriangle);
    assert_eq!(classify(Some(r), Some(hits[0])), Class::OtherGrass);
    // Ensure the actual diagnostic requests closest-hit traversal and never
    // substitutes the early-termination flag used by normal binary shadows.
    let shader = include_str!("shadow_diagnostics.wesl");
    let probe = shader
        .split("fn probe_surface(")
        .nth(1)
        .unwrap()
        .split("fn blocker_color(")
        .next()
        .unwrap();
    assert!(probe.contains("RAY_FLAG_NONE"));
    assert!(!probe.contains("RAY_FLAG_TERMINATE_ON_FIRST_HIT"));
}
