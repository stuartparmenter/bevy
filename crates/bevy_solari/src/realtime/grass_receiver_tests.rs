//! CPU reference scenarios for cached grass receiver anchors. Deliberately use
//! different current/previous vertex positions and instance slots: a fixture
//! with identity mappings/static geometry cannot expose history misuse.
use bevy_math::Vec3;

const INVALID: u32 = u32::MAX;
#[derive(Clone, Copy)]
struct CachedAnchor {
    instance: u32,
    triangle: u32,
    bary: [f32; 2],
}
struct Instance {
    current: Vec<[Vec3; 3]>,
    previous: Vec<[Vec3; 3]>,
    current_translation: Vec3,
    previous_translation: Vec3,
    exact_grass: bool,
}

// CPU oracle for the intended shader contract. Shader-source assertions below
// bind the history branch to its actual translation/vertex/transform buffers.
fn resolve(
    anchor: CachedAnchor,
    previous_frame: bool,
    translations: &[u32],
    instances: &[Instance],
) -> Option<Vec3> {
    if anchor.instance == INVALID {
        return None;
    }
    let id = if previous_frame {
        *translations.get(anchor.instance as usize)?
    } else {
        anchor.instance
    };
    if id == INVALID {
        return None;
    }
    let instance = instances.get(id as usize)?;
    if !instance.exact_grass {
        return None;
    }
    let (vertices, translation) = if previous_frame {
        (&instance.previous, instance.previous_translation)
    } else {
        (&instance.current, instance.current_translation)
    };
    let triangle = vertices.get(anchor.triangle as usize)?;
    let [u, v] = anchor.bary;
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
fn moving_blade() -> Instance {
    Instance {
        previous: vec![[Vec3::ZERO, Vec3::X, Vec3::Y]],
        current: vec![[Vec3::Z, Vec3::X + Vec3::Z, Vec3::Y + Vec3::Z]],
        previous_translation: Vec3::new(10.0, 20.0, 30.0),
        current_translation: Vec3::new(100.0, 200.0, 300.0),
        exact_grass: true,
    }
}
fn anchor(instance: u32) -> CachedAnchor {
    CachedAnchor {
        instance,
        triangle: 0,
        bary: [0.25, 0.5],
    }
}

#[test]
fn receiver_history_follows_remapped_instance_previous_vertices_and_transform() {
    let instances = [moving_blade()];
    // Old TLAS instance 2 is current instance 0. Looking up slot 2 directly
    // or using current vertices/transform must not silently pass this case.
    assert_eq!(
        resolve(anchor(2), true, &[INVALID, INVALID, 0], &instances),
        Some(Vec3::new(10.25, 20.5, 30.0))
    );
    assert_eq!(
        resolve(anchor(0), false, &[], &instances),
        Some(Vec3::new(100.25, 200.5, 301.0))
    );
}

#[test]
fn receiver_history_rejects_destroyed_reused_and_reassigned_slots() {
    let instances = [moving_blade()];
    // Current slot 0 is occupied, but belongs to a new owner: previous-frame
    // translation MUST take precedence over seemingly valid current geometry.
    assert_eq!(resolve(anchor(0), true, &[INVALID], &instances), None);
    // The binder also emits this sentinel for topology-generation changes.
    assert!(resolve(anchor(0), false, &[], &instances).is_some());
    assert_eq!(resolve(anchor(2), true, &[0], &instances), None);
    assert_eq!(resolve(anchor(0), true, &[9], &instances), None);
    assert_eq!(resolve(anchor(INVALID), false, &[], &instances), None);
}

#[test]
fn receiver_history_rejects_missing_triangles_and_proxy_substitution() {
    let mut instances = [moving_blade()];
    let stale = CachedAnchor {
        triangle: 1,
        ..anchor(0)
    };
    assert_eq!(resolve(stale, true, &[0], &instances), None);
    instances[0].exact_grass = false;
    assert_eq!(resolve(anchor(0), true, &[0], &instances), None);
}

#[test]
fn receiver_anchor_barycentrics_do_not_follow_neighbor_triangle_or_current_wind() {
    let mut instance = moving_blade();
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
            CachedAnchor {
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
fn receiver_cache_rejects_invalid_barycentrics_but_keeps_ray_edge_roundoff() {
    let instances = [moving_blade()];
    for bary in [
        [f32::NAN, 0.0],
        [f32::INFINITY, 0.0],
        [-0.1, 0.0],
        [0.8, 0.8],
    ] {
        assert_eq!(
            resolve(CachedAnchor { bary, ..anchor(0) }, true, &[0], &instances),
            None
        );
    }
    assert!(resolve(
        CachedAnchor {
            bary: [-0.000001, 0.5],
            ..anchor(0)
        },
        true,
        &[0],
        &instances
    )
    .is_some());
}

#[test]
fn receiver_shader_validates_history_before_fetching_current_slot_geometry() {
    let shader = include_str!("grass_receiver_geometry.wesl");
    let resolve = shader
        .split("fn resolve_receiver_anchor(")
        .nth(1)
        .unwrap()
        .split("fn match_receiver(")
        .next()
        .unwrap();
    let remap = resolve
        .find("instance = previous_frame_instance_id_translations[instance]")
        .unwrap();
    let reject = resolve
        .find("if instance == INSTANCE_NOT_PRESENT_THIS_FRAME { return receiver; }")
        .unwrap();
    let fetch = resolve
        .find("var geometry = geometry_ids[instance]")
        .unwrap();
    assert!(remap < reject && reject < fetch,
        "cached previous slot must not index recycled current geometry before translation/rejection");
    assert!(resolve.contains("instance >= arrayLength(&previous_frame_instance_id_translations)"));
    assert!(resolve.contains("anchor.y >= geometry.triangle_count"));
    assert!(resolve.contains("geometry.diagnostic_primitives_per_group != 9u"));
    assert!(resolve.contains("geometry.vertex_buffer_id = geometry.previous_vertex_buffer_id"));
    assert!(resolve.contains("transform = previous_frame_transforms[instance]"));
    assert!(resolve.contains("load_vertices(geometry, anchor.y)"));
}

#[test]
fn current_receiver_does_not_require_previous_scene_buffers() {
    let mut instance = moving_blade();
    instance.previous.clear();
    let instances = [instance];
    assert_eq!(
        resolve(anchor(0), false, &[], &instances),
        Some(Vec3::new(100.25, 200.5, 301.0))
    );
    assert_eq!(resolve(anchor(0), true, &[0], &instances), None);
    let shader = include_str!("grass_receiver_geometry.wesl");
    assert!(
        shader.contains("if previous && instance >= arrayLength(&previous_frame_transforms)"),
        "current anchors must work with the non-ReSTIR dummy previous-transform buffer"
    );
}
