//! Ordinary Bevy animation components also deform Solari's raytracing geometry.

use super::*;
use bevy::{
    asset::RenderAssetUsages,
    camera::primitives::Aabb,
    mesh::{
        morph::{MeshMorphWeights, MorphAttributes},
        skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
        PrimitiveTopology,
    },
};

#[derive(Component)]
pub(super) struct AnimatedJoint;

pub(super) fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut inverse_bindposes: ResMut<Assets<SkinnedMeshInverseBindposes>>,
    args: Res<Args>,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] dlss_rr_supported: Option<
        Res<DlssRayReconstructionSupported>,
    >,
) {
    let inverse_bindposes =
        inverse_bindposes.add(vec![Mat4::IDENTITY, Mat4::from_translation(-Vec3::Y)]);

    for (column, color) in [
        Color::srgb(0.8, 0.15, 0.1),
        Color::srgb(0.1, 0.7, 0.2),
        Color::srgb(0.1, 0.3, 0.9),
    ]
    .into_iter()
    .enumerate()
    {
        let skinned = column != 1;
        let morphed = column != 0;
        let mesh = meshes.add(ribbon(skinned, morphed));
        let translation = Vec3::new((column as f32 - 1.0) * 2.5, 0.1, 0.0);
        let skin = skinned.then(|| {
            let root = commands
                .spawn(Transform::from_translation(translation))
                .id();
            let joint = commands
                .spawn((AnimatedJoint, Transform::from_xyz(0.0, 1.0, 0.0)))
                .id();
            commands.entity(root).add_child(joint);
            SkinnedMesh {
                inverse_bindposes: inverse_bindposes.clone(),
                joints: vec![root, joint],
            }
        });

        let mut entity = commands.spawn((
            RaytracingMesh3d(mesh.clone()),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                double_sided: true,
                cull_mode: None,
                ..default()
            })),
            Transform::from_translation(translation),
            // Conservative raster bounds covering the animation's full range.
            Aabb::from_min_max(Vec3::new(-2.0, 0.0, -0.1), Vec3::new(2.0, 3.0, 0.1)),
        ));
        entity.insert_if(Mesh3d(mesh), || args.pathtracer != Some(true));
        if let Some(skin) = skin {
            entity.insert(skin);
        }
        if morphed {
            entity.insert(MeshMorphWeights::Value { weights: vec![0.0] });
        }
    }

    let floor = meshes.add(
        Plane3d::default()
            .mesh()
            .size(20.0, 20.0)
            .build()
            .with_generated_tangents()
            .unwrap(),
    );
    commands
        .spawn((
            RaytracingMesh3d(floor.clone()),
            MeshMaterial3d(materials.add(Color::srgb(0.7, 0.7, 0.7))),
        ))
        .insert_if(Mesh3d(floor), || args.pathtracer != Some(true));

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(3.0, 5.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let mut camera = commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        FreeCamera {
            walk_speed: 3.0,
            run_speed: 10.0,
            ..default()
        },
        Transform::from_xyz(0.0, 3.5, 9.0).looking_at(Vec3::Y, Vec3::Y),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
    ));
    if args.pathtracer == Some(true) {
        camera.insert(Pathtracer::default());
    } else {
        camera.insert(SolariLighting::default());
    }
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    if dlss_rr_supported.is_some() {
        camera.insert(Dlss::<DlssRayReconstructionFeature>::default());
    }
}

fn ribbon(skinned: bool, morphed: bool) -> Mesh {
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut weights = Vec::new();
    let mut targets = Vec::new();
    for row in 0..=8 {
        let t = row as f32 / 8.0;
        for side in [-1.0, 1.0] {
            positions.push([side * 0.4, t * 2.0, 0.0]);
            uvs.push([(side + 1.0) * 0.5, t]);
            weights.push([1.0 - t, t, 0.0, 0.0]);
            targets.push(MorphAttributes::new(
                Vec3::new(side * 0.5 * (PI * t).sin(), 0.0, 0.0),
                Vec3::ZERO,
                Vec3::ZERO,
            ));
        }
    }
    let indices = (0..8)
        .flat_map(|row| {
            let i = row * 2;
            [i, i + 1, i + 3, i, i + 3, i + 2]
        })
        .collect::<Vec<u16>>();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 18])
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[1.0, 0.0, 0.0, 1.0]; 18])
    .with_inserted_indices(Indices::U16(indices));
    if skinned {
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_INDEX,
            VertexAttributeValues::Uint16x4(vec![[0, 1, 0, 0]; 18]),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, weights);
    }
    if morphed {
        mesh.set_morph_targets(targets);
    }
    mesh
}

pub(super) fn animate(
    time: Res<Time>,
    mut joints: Query<&mut Transform, With<AnimatedJoint>>,
    mut morphs: Query<&mut MeshMorphWeights>,
    mut pathtracers: Query<&mut Pathtracer>,
) {
    // Do not accumulate samples from different poses. Camera motion also resets
    // accumulation during pathtracer extraction.
    for mut pathtracer in &mut pathtracers {
        pathtracer.reset = time.delta_secs() != 0.0;
    }
    for mut joint in &mut joints {
        joint.rotation = Quat::from_rotation_z(time.elapsed_secs().sin() * 0.7);
    }
    for mut morph in &mut morphs {
        if let MeshMorphWeights::Value { weights } = &mut *morph {
            weights[0] = 0.5 + 0.5 * (time.elapsed_secs() * 1.3).sin();
        }
    }
}

pub(super) fn update_pathtracer_text(
    time: Res<Time<Virtual>>,
    mut text: Single<&mut Text, With<ControlText>>,
) {
    text.0 = format!(
        "Left: skinning | Middle: morph targets | Right: both\n(Space): {}",
        if time.is_paused() {
            "Resume animation"
        } else {
            "Pause animation to accumulate pathtracing samples"
        },
    );
}
