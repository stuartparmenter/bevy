use super::{
    deformation_source::DeformationSources, RaytracingGeometry, RaytracingGeometryBuffers,
    RaytracingGeometryPreviousVertices, RaytracingGeometryUpdateMode, RaytracingMesh3d,
    RaytracingProducerEncoder,
};
use bevy_asset::{load_embedded_asset, AssetServer, Assets};
use bevy_ecs::{
    component::Component,
    entity::{Entity, EntityHashMap, EntityHashSet},
    lifecycle::RemovedComponents,
    query::{Has, With},
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_math::{Mat4, UVec4};
use bevy_mesh::{
    morph::MeshMorphWeights,
    skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
};
use bevy_pbr::{MeshDeformationRequests, MorphIndices, MorphUniforms, SkinUniforms};
use bevy_render::{
    mesh::{allocator::MeshAllocator, RenderMesh},
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    sync_world::{MainEntity, RenderEntity},
    Extract,
};
use bevy_transform::components::GlobalTransform;

/// Marks geometry owned by the built-in mesh deformation producer.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub(super) struct RaytracingMeshDeformation {
    skinned: bool,
    morphed: bool,
    joint_count: u32,
    skin_ready: bool,
}

pub(super) fn extract_mesh_deformations(
    mut commands: Commands,
    meshes: Extract<
        Query<(
            Entity,
            RenderEntity,
            &RaytracingMesh3d,
            Option<&SkinnedMesh>,
            Has<MeshMorphWeights>,
            Has<RaytracingGeometry>,
        )>,
    >,
    explicit_geometry: Extract<Query<(), With<RaytracingGeometry>>>,
    inverse_bindposes: Extract<Res<Assets<SkinnedMeshInverseBindposes>>>,
    joints: Extract<Query<&GlobalTransform>>,
    existing: Query<(Entity, &MainEntity, &RaytracingMeshDeformation)>,
    mut removed_geometry: Extract<RemovedComponents<RaytracingGeometry>>,
    mut requests: ResMut<MeshDeformationRequests>,
) {
    // Structural extraction removes native geometry too when a custom marker was removed.
    // Its deferred commands have not been applied yet in ExtractSchedule.
    let removed_geometry: EntityHashSet = removed_geometry.read().collect();
    let mut live = EntityHashSet::default();
    for (main_entity, render_entity, _, skin, morphed, explicit) in &meshes {
        if explicit || (skin.is_none() && !morphed) {
            continue;
        }
        live.insert(render_entity);
        requests.insert(main_entity.into());
        let deformation = RaytracingMeshDeformation {
            skinned: skin.is_some(),
            morphed,
            joint_count: skin.map_or(0, |skin| skin.joints.len() as u32),
            skin_ready: skin.is_none_or(|skin| {
                inverse_bindposes
                    .get(&skin.inverse_bindposes)
                    .is_some_and(|poses| poses.len() >= skin.joints.len())
                    && skin.joints.iter().all(|joint| joints.contains(*joint))
            }),
        };
        if !existing
            .get(render_entity)
            .is_ok_and(|(_, _, previous)| *previous == deformation)
            || removed_geometry.contains(&main_entity)
        {
            commands
                .entity(render_entity)
                .insert((deformation, RaytracingGeometry));
        }
    }
    for (entity, main_entity, _) in &existing {
        if live.contains(&entity) {
            continue;
        }
        let mut entity = commands.entity(entity);
        entity.remove::<(
            RaytracingMeshDeformation,
            RaytracingGeometryBuffers,
            RaytracingGeometryPreviousVertices,
        )>();
        if !explicit_geometry.contains(main_entity.id()) {
            entity.remove::<RaytracingGeometry>();
        }
    }
}

#[derive(Clone, ShaderType)]
struct DeformationParams {
    entity_from_world: Mat4,
    world_from_entity: Mat4,
    // Vertex count, source word offset, word stride, and skin matrix offset.
    vertex: UVec4,
    // Position, normal, UV, and tangent word offsets within a vertex.
    attributes: UVec4,
    // Joint weight/index word offsets, U32 index flag, and history-valid flag.
    skin: UVec4,
    // Morph target offset, weight offset, and weight count.
    morph: UVec4,
}

struct DeformedMesh {
    source: BufferId,
    vertices: Buffer,
    previous_vertices: Buffer,
    bind_groups: Vec<([BufferId; 5], BindGroup)>,
    history_valid: bool,
}

#[derive(Resource)]
pub(super) struct MeshDeformationPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
    dummy: Buffer,
}

impl FromWorld for MeshDeformationPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let layout = BindGroupLayoutDescriptor::new(
            "solari_mesh_deformation",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer::<DeformationParams>(true),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_sized(false, None),
                    storage_buffer_sized(false, None),
                ),
            ),
        );
        let shader = load_embedded_asset!(world.resource::<AssetServer>(), "deformation.wesl");
        let pipeline =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("solari_mesh_deformation".into()),
                    layout: vec![layout.clone()],
                    shader,
                    entry_point: Some("deform".into()),
                    ..Default::default()
                });
        let dummy = device.create_buffer(&BufferDescriptor {
            label: Some("solari_mesh_deformation_dummy"),
            size: 64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Self {
            layout,
            pipeline,
            dummy,
        }
    }
}

#[derive(Resource)]
pub(super) struct MeshDeformations {
    meshes: EntityHashMap<DeformedMesh>,
    params: DynamicUniformBuffer<DeformationParams>,
}

impl FromWorld for MeshDeformations {
    fn from_world(world: &mut World) -> Self {
        let alignment = world
            .resource::<RenderDevice>()
            .limits()
            .min_uniform_buffer_offset_alignment;
        Self {
            meshes: EntityHashMap::default(),
            params: DynamicUniformBuffer::new_with_alignment(u64::from(alignment)),
        }
    }
}

/// Fills native geometry before publishing its buffers to the BLAS and scene preparation systems.
pub(super) fn prepare_mesh_deformations(
    mut commands: Commands,
    meshes: Query<(
        Entity,
        &MainEntity,
        &RaytracingMesh3d,
        &RaytracingMeshDeformation,
        &GlobalTransform,
        Option<&RaytracingGeometryBuffers>,
        Has<RaytracingGeometry>,
    )>,
    sources: Res<DeformationSources>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    allocator: Res<MeshAllocator>,
    skins: Res<SkinUniforms>,
    morph_indices: Res<MorphIndices>,
    morphs: Res<MorphUniforms>,
    pipeline: Res<MeshDeformationPipeline>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut producer: ResMut<RaytracingProducerEncoder>,
    mut deformations: ResMut<MeshDeformations>,
) {
    let MeshDeformations {
        meshes: outputs,
        params,
    } = &mut *deformations;
    outputs.retain(|entity, _| meshes.contains(*entity));
    params.clear();
    let compute = pipeline_cache.get_compute_pipeline(pipeline.pipeline);
    let mut dispatches = Vec::new();

    for (entity, main_entity, mesh, deformation, transform, existing, has_geometry) in &meshes {
        // A separate raytracing mesh need not have the raster mesh's morph targets.
        if !deformation.skinned
            && render_meshes
                .get(mesh.id())
                .is_some_and(|mesh| !mesh.has_morph_targets())
        {
            outputs.remove(&entity);
            if has_geometry {
                commands.entity(entity).remove::<(
                    RaytracingGeometry,
                    RaytracingGeometryBuffers,
                    RaytracingGeometryPreviousVertices,
                )>();
            }
            continue;
        }
        if !has_geometry {
            commands.entity(entity).insert(RaytracingGeometry);
        }
        let inputs = (|| {
            compute?;
            if !deformation.skin_ready {
                return None;
            }
            let source = sources.get(mesh.id())?;
            let render_mesh = render_meshes.get(mesh.id())?;
            if render_mesh.vertex_count != source.vertex_count {
                return None;
            }
            let vertices = allocator.mesh_vertex_slice(&mesh.id())?;
            let world_from_entity = Mat4::from(transform.affine());
            let determinant = world_from_entity.determinant();
            if determinant == 0.0 || !determinant.is_finite() {
                return None;
            }
            let entity_from_world = world_from_entity.inverse();
            if !entity_from_world.is_finite() {
                return None;
            }
            let skin_index = if deformation.skinned {
                if !source.skinned || source.max_joint_index >= deformation.joint_count {
                    return None;
                }
                skins.skin_index(*main_entity)?
            } else {
                u32::MAX
            };
            let (target_buffer, weights_buffer, targets_offset, weights_offset, weight_count) =
                if deformation.morphed && source.morph_target_count > 0 {
                    let (offset, count) = morph_indices.current_weights_info(*main_entity)?;
                    let count = count.min(source.morph_target_count);
                    if count == 0 {
                        (&pipeline.dummy, &pipeline.dummy, 0, 0, 0)
                    } else {
                        let targets = allocator.mesh_morph_target_slice(&mesh.id())?;
                        (
                            targets.buffer,
                            morphs.current_buffer.buffer()?,
                            targets.range.start,
                            offset,
                            count,
                        )
                    }
                } else {
                    (&pipeline.dummy, &pipeline.dummy, 0, 0, 0)
                };
            let joints = if deformation.skinned {
                &skins.current_buffer
            } else {
                &pipeline.dummy
            };
            let layout = &source.layout;
            let base = vertices.range.start.checked_mul(layout.stride)?;
            let max_vertices = device.limits().max_compute_workgroups_per_dimension * 64;
            let storage_limit = u64::from(device.limits().max_storage_buffer_binding_size)
                .min(device.limits().max_buffer_size);
            if source.vertex_count > max_vertices
                || u64::from(source.vertex_count) * RaytracingGeometryBuffers::VERTEX_STRIDE
                    > storage_limit
                || [vertices.buffer, joints, weights_buffer, target_buffer]
                    .iter()
                    .any(|buffer| buffer.size() > storage_limit)
            {
                return None;
            }
            Some((
                source,
                vertices.buffer,
                joints,
                weights_buffer,
                target_buffer,
                DeformationParams {
                    entity_from_world,
                    world_from_entity,
                    vertex: UVec4::new(source.vertex_count, base, layout.stride, skin_index),
                    attributes: UVec4::new(
                        layout.position,
                        layout.normal,
                        layout.uv,
                        layout.tangent,
                    ),
                    skin: UVec4::new(
                        layout.joint_weights,
                        layout.joint_indices,
                        layout.joint_indices_u32,
                        0,
                    ),
                    morph: UVec4::new(targets_offset, weights_offset, weight_count, 0),
                },
            ))
        })();
        let Some((source, vertices, joints, weights, targets, mut uniform)) = inputs else {
            outputs.remove(&entity);
            if existing.is_some() {
                commands.entity(entity).remove::<(
                    RaytracingGeometryBuffers,
                    RaytracingGeometryPreviousVertices,
                )>();
            }
            continue;
        };
        if outputs
            .get(&entity)
            .is_none_or(|entry| entry.source != source.index_buffer.id())
        {
            let size = u64::from(source.vertex_count) * RaytracingGeometryBuffers::VERTEX_STRIDE;
            let vertices = device.create_buffer(&BufferDescriptor {
                label: Some("solari_deformed_vertices"),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::BLAS_INPUT,
                mapped_at_creation: false,
            });
            let previous_vertices = device.create_buffer(&BufferDescriptor {
                label: Some("solari_previous_deformed_vertices"),
                size,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            outputs.insert(
                entity,
                DeformedMesh {
                    source: source.index_buffer.id(),
                    vertices,
                    previous_vertices,
                    bind_groups: Vec::new(),
                    history_valid: false,
                },
            );
        }
        let output = outputs.get_mut(&entity).unwrap();
        uniform.skin.w = u32::from(output.history_valid);
        let offset = params.push(&uniform);
        dispatches.push((
            entity,
            offset,
            source.vertex_count,
            vertices,
            joints,
            weights,
            targets,
        ));
        if existing.is_none_or(|buffers| buffers.vertex_buffer.id() != output.vertices.id()) {
            commands.entity(entity).insert((
                RaytracingGeometryBuffers {
                    ray_mask: 0xFF,
                    diagnostic_primitives_per_group: 0,
                    vertex_buffer: output.vertices.clone(),
                    index_buffer: source.index_buffer.clone(),
                    vertex_count: source.vertex_count,
                    index_count: source.index_count,
                    update_mode: RaytracingGeometryUpdateMode::RebuildEveryFrame,
                },
                RaytracingGeometryPreviousVertices(output.previous_vertices.clone()),
            ));
        }
    }
    if dispatches.is_empty() {
        return;
    }
    params.write_buffer(&device, &queue);
    let params_buffer = params.buffer().unwrap();
    let encoder = producer.encoder(&device);
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("solari_mesh_deformation"),
        timestamp_writes: None,
    });
    pass.set_pipeline(compute.unwrap());
    for (entity, offset, count, vertices, joints, weights, targets) in dispatches {
        let output = outputs.get_mut(&entity).unwrap();
        let key = [
            params_buffer.id(),
            vertices.id(),
            joints.id(),
            weights.id(),
            targets.id(),
        ];
        let group = if let Some(index) = output.bind_groups.iter().position(|(ids, _)| *ids == key)
        {
            &output.bind_groups[index].1
        } else {
            // Joint and weight buffers alternate each frame. Bound the cache across reallocations.
            if output.bind_groups.len() >= 2 {
                output.bind_groups.remove(0);
            }
            let group = device.create_bind_group(
                "solari_mesh_deformation",
                &pipeline_cache.get_bind_group_layout(&pipeline.layout),
                &BindGroupEntries::sequential((
                    params.binding().unwrap(),
                    vertices.as_entire_binding(),
                    joints.as_entire_binding(),
                    weights.as_entire_binding(),
                    targets.as_entire_binding(),
                    output.vertices.as_entire_binding(),
                    output.previous_vertices.as_entire_binding(),
                )),
            );
            output.bind_groups.push((key, group));
            &output.bind_groups.last().unwrap().1
        };
        pass.set_bind_group(0, group, &[offset]);
        pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        output.history_valid = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::extract::extract_raytracing_scene_structural;
    use bevy_ecs::schedule::{IntoScheduleConfigs, Schedule, ScheduleBuildSettings};
    use bevy_render::MainWorld;

    struct ExtractionTest {
        world: World,
        schedule: Schedule,
        main_entity: Entity,
        render_entity: Entity,
    }

    impl ExtractionTest {
        fn new() -> Self {
            let mut world = World::new();
            let mut main = MainWorld::default();
            main.init_resource::<Assets<SkinnedMeshInverseBindposes>>();
            let render_entity = world.spawn_empty().id();
            let main_entity = main
                .spawn((
                    RaytracingMesh3d::default(),
                    RenderEntity::from(render_entity),
                    MeshMorphWeights::Value { weights: vec![1.0] },
                ))
                .id();
            world
                .entity_mut(render_entity)
                .insert(MainEntity::from(main_entity));
            world.insert_resource(main);
            world.init_resource::<MeshDeformationRequests>();
            let mut schedule = Schedule::default();
            // Match ExtractSchedule: ordering does not apply structural commands early.
            schedule.set_build_settings(ScheduleBuildSettings {
                auto_insert_apply_deferred: false,
                ..Default::default()
            });
            schedule.set_apply_final_deferred(false);
            schedule.add_systems((
                extract_raytracing_scene_structural,
                extract_mesh_deformations.after(extract_raytracing_scene_structural),
            ));
            Self {
                world,
                schedule,
                main_entity,
                render_entity,
            }
        }

        fn extract(&mut self) {
            self.world
                .insert_resource(MeshDeformationRequests::default());
            self.schedule.run(&mut self.world);
            self.schedule.apply_deferred(&mut self.world);
            self.world.resource_mut::<MainWorld>().clear_trackers();
            self.world.clear_trackers();
        }

        fn native(&self) -> bool {
            self.world
                .get::<RaytracingMeshDeformation>(self.render_entity)
                .is_some()
        }

        fn geometry(&self) -> bool {
            self.world
                .get::<RaytracingGeometry>(self.render_entity)
                .is_some()
        }

        fn requested(&self) -> bool {
            self.world
                .resource::<MeshDeformationRequests>()
                .contains(self.main_entity.into())
        }
    }

    #[test]
    fn native_deformation_follows_component_lifetime() {
        let mut test = ExtractionTest::new();
        test.extract();
        assert!(test.native() && test.geometry() && test.requested());

        test.world
            .resource_mut::<MainWorld>()
            .entity_mut(test.main_entity)
            .remove::<MeshMorphWeights>();
        test.extract();
        assert!(!test.native() && !test.geometry() && !test.requested());
        assert!(test
            .world
            .get::<RaytracingMesh3d>(test.render_entity)
            .is_some());

        test.world
            .resource_mut::<MainWorld>()
            .entity_mut(test.main_entity)
            .insert(MeshMorphWeights::Value { weights: vec![0.5] });
        test.extract();
        assert!(test.native() && test.geometry() && test.requested());

        test.world
            .resource_mut::<MainWorld>()
            .entity_mut(test.main_entity)
            .remove::<RaytracingMesh3d>();
        test.extract();
        assert!(!test.native() && !test.geometry() && !test.requested());

        test.world
            .resource_mut::<MainWorld>()
            .entity_mut(test.main_entity)
            .insert(RaytracingMesh3d::default());
        test.extract();
        assert!(test.native() && test.geometry() && test.requested());
    }

    #[test]
    fn custom_geometry_overrides_native_deformation() {
        let mut test = ExtractionTest::new();
        test.extract();
        test.world
            .resource_mut::<MainWorld>()
            .entity_mut(test.main_entity)
            .insert(RaytracingGeometry);
        test.extract();
        assert!(!test.native() && test.geometry() && !test.requested());

        test.world
            .resource_mut::<MainWorld>()
            .entity_mut(test.main_entity)
            .remove::<RaytracingGeometry>();
        test.extract();
        assert!(test.native() && test.geometry() && test.requested());
    }

    #[test]
    fn same_frame_marker_changes_preserve_native_geometry() {
        let mut test = ExtractionTest::new();
        test.extract();
        // A transient custom marker must not erase the render-only native marker.
        test.world
            .resource_mut::<MainWorld>()
            .entity_mut(test.main_entity)
            .insert(RaytracingGeometry)
            .remove::<RaytracingGeometry>();
        test.extract();
        assert!(test.native() && test.geometry() && test.requested());

        test.world
            .resource_mut::<MainWorld>()
            .entity_mut(test.main_entity)
            .remove::<RaytracingMesh3d>()
            .insert(RaytracingMesh3d::default());
        test.extract();
        assert!(test.native() && test.geometry() && test.requested());
        assert!(test
            .world
            .get::<RaytracingMesh3d>(test.render_entity)
            .is_some());
    }

    #[test]
    fn skin_requires_loaded_bindposes_and_joint_transforms() {
        let mut test = ExtractionTest::new();
        let joint;
        let inverse_bindposes;
        {
            let mut main = test.world.resource_mut::<MainWorld>();
            joint = main.spawn_empty().id();
            inverse_bindposes = main
                .resource_mut::<Assets<SkinnedMeshInverseBindposes>>()
                .add(SkinnedMeshInverseBindposes::from(vec![Mat4::IDENTITY]));
            main.entity_mut(test.main_entity).insert(SkinnedMesh {
                inverse_bindposes: inverse_bindposes.clone(),
                joints: vec![joint],
            });
        }
        test.extract();
        assert!(
            !test
                .world
                .get::<RaytracingMeshDeformation>(test.render_entity)
                .unwrap()
                .skin_ready
        );
        assert!(test.geometry() && test.requested());

        test.world
            .resource_mut::<MainWorld>()
            .entity_mut(joint)
            .insert(GlobalTransform::IDENTITY);
        test.extract();
        assert!(
            test.world
                .get::<RaytracingMeshDeformation>(test.render_entity)
                .unwrap()
                .skin_ready
        );

        test.world
            .resource_mut::<MainWorld>()
            .resource_mut::<Assets<SkinnedMeshInverseBindposes>>()
            .remove(inverse_bindposes.id());
        test.extract();
        assert!(
            !test
                .world
                .get::<RaytracingMeshDeformation>(test.render_entity)
                .unwrap()
                .skin_ready
        );

        test.world
            .resource_mut::<MainWorld>()
            .resource_mut::<Assets<SkinnedMeshInverseBindposes>>()
            .insert(
                inverse_bindposes.id(),
                SkinnedMeshInverseBindposes::from(vec![Mat4::IDENTITY]),
            )
            .unwrap();
        test.extract();
        assert!(
            test.world
                .get::<RaytracingMeshDeformation>(test.render_entity)
                .unwrap()
                .skin_ready
        );
    }
}
