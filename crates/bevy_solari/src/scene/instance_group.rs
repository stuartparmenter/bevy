//! [`RaytracingInstanceGroup`]: instances whose transforms the application writes on the GPU.
//!
//! Each instance of a group is a render-world entity carrying [`RaytracingMesh3d`], so the binder
//! allocates its slot and resolves its geometry, material and BLAS as for any other instance.
//! They also carry [`GpuWrittenTransform`], so the binder never writes their slots' transforms.
//! Instead, every frame after the sparse buffer uploads, a
//! compute pass copies each instance's transform from the group's buffer into its slot of the
//! binder's transform buffer, and the transform it copied the frame before into the previous-frame
//! transform buffer.

use super::{
    binder::{GpuTransform, GpuWrittenTransform, RaytracingSceneBindings},
    RaytracingInstanceGroup, RaytracingInstanceGroupMember, RaytracingInstanceMask,
    RaytracingMesh3d, RaytracingProducerEncoder,
};
use bevy_asset::{load_embedded_asset, AssetId};
use bevy_ecs::{
    entity::{Entity, EntityHashMap},
    lifecycle::RemovedComponents,
    query::Changed,
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_pbr::{MeshMaterial3d, PreviousGlobalTransform};
use bevy_render::{
    diagnostic::RecordDiagnostics,
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
        BufferDescriptor, BufferId, BufferInitDescriptor, BufferUsages, CachedComputePipelineId,
        ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderStages,
    },
    renderer::{RenderContext, RenderDevice},
    storage::{GpuShaderBuffer, ShaderBuffer},
    Extract,
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::{default, once};
use core::ops::Range;
use tracing::warn;

const TRANSFORM_SIZE: u64 = size_of::<GpuTransform>() as u64;
const WORKGROUP_SIZE: u32 = 64;

/// A group's render-world state.
struct Group {
    members: Vec<RaytracingInstanceGroupMember>,
    mask: RaytracingInstanceMask,
    transforms: AssetId<ShaderBuffer>,
    /// The render entity of each instance, in instance order.
    instances: Vec<Entity>,
    /// Each instance's slot, once every instance holds one.
    slots: Option<Buffer>,
    /// Each instance's transform read last frame, once the transforms are on the GPU.
    previous: Option<Buffer>,
    bind_group: Option<(BindGroup, [BufferId; 5])>,
}

impl Group {
    fn new(group: &RaytracingInstanceGroup) -> Self {
        Self {
            members: group.members.clone(),
            mask: group.mask,
            transforms: group.transforms.id(),
            instances: Vec::with_capacity(group.instance_count() as usize),
            slots: None,
            previous: None,
            bind_group: None,
        }
    }

    /// Whether `group` traces the same instances as this one, up to its count.
    fn matches(&self, group: &RaytracingInstanceGroup) -> bool {
        self.members == group.members
            && self.mask == group.mask
            && self.transforms == group.transforms.id()
    }

    fn despawn(self, commands: &mut Commands) {
        for entity in self.instances {
            commands.entity(entity).despawn();
        }
    }
}

/// Every group, by its main-world entity.
#[derive(Resource, Default)]
pub struct RaytracingInstanceGroups(EntityHashMap<Group>);

/// Creates and removes the render entities of each group's instances.
pub fn extract_raytracing_instance_groups(
    changed: Extract<Query<(Entity, &RaytracingInstanceGroup), Changed<RaytracingInstanceGroup>>>,
    mut removed: Extract<RemovedComponents<RaytracingInstanceGroup>>,
    mut groups: ResMut<RaytracingInstanceGroups>,
    mut commands: Commands,
) {
    for main_entity in removed.read() {
        if let Some(group) = groups.0.remove(&main_entity) {
            group.despawn(&mut commands);
        }
    }

    for (main_entity, group) in &changed {
        let count = group.instance_count();
        match groups.0.get_mut(&main_entity) {
            // Instances are copy-major, so a new count only adds or removes the last ones.
            Some(existing) if existing.matches(group) => {
                let current = existing.instances.len() as u32;
                if count < current {
                    for entity in existing.instances.drain(count as usize..) {
                        commands.entity(entity).despawn();
                    }
                    // The bound slots include the despawned instances'.
                    existing.bind_group = None;
                }
                existing.slots = None;
                spawn_instances(&mut commands, main_entity, group, current..count);
            }
            _ => {
                if let Some(previous) = groups.0.insert(main_entity, Group::new(group)) {
                    previous.despawn(&mut commands);
                }
                spawn_instances(&mut commands, main_entity, group, 0..count);
            }
        }
    }
}

/// Spawns the render entities of `group`'s instances `range` and appends them to its instances.
fn spawn_instances(
    commands: &mut Commands,
    main_entity: Entity,
    group: &RaytracingInstanceGroup,
    range: Range<u32>,
) {
    if range.is_empty() {
        return;
    }
    let members = group.members.clone();
    let mask = group.mask;
    commands.queue(move |world: &mut World| {
        let member_count = members.len() as u32;
        let bundles = range.map(|instance| {
            let member = &members[(instance % member_count) as usize];
            (
                RaytracingMesh3d(member.mesh.clone()),
                MeshMaterial3d(member.material.clone()),
                GpuWrittenTransform,
                GlobalTransform::default(),
                PreviousGlobalTransform::default(),
                mask,
            )
        });
        let spawned: Vec<Entity> = world.spawn_batch(bundles).collect();
        match world
            .resource_mut::<RaytracingInstanceGroups>()
            .0
            .get_mut(&main_entity)
        {
            Some(group) => group.instances.extend(spawned),
            None => {
                for entity in spawned {
                    world.despawn(entity);
                }
            }
        }
    });
}

/// Compute pipeline that copies group transforms into instance slots.
#[derive(Resource)]
pub struct RaytracingInstanceGroupPipeline {
    layout: BindGroupLayoutDescriptor,
    id: CachedComputePipelineId,
}

impl FromWorld for RaytracingInstanceGroupPipeline {
    fn from_world(world: &mut World) -> Self {
        let layout = BindGroupLayoutDescriptor::new(
            "raytracing_instance_group_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_sized(false, None),
                    storage_buffer_sized(false, None),
                ),
            ),
        );
        let shader = load_embedded_asset!(world, "instance_group_transforms.wesl");
        let id =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("raytracing_instance_group_pipeline".into()),
                    layout: vec![layout.clone()],
                    shader,
                    entry_point: Some("write_instance_group_transforms".into()),
                    ..default()
                });
        Self { layout, id }
    }
}

/// Resolves each group's instance slots and binds its buffers, once the binder has allocated
/// every instance a slot and the group's transforms are on the GPU.
pub fn prepare_raytracing_instance_groups(
    mut groups: ResMut<RaytracingInstanceGroups>,
    bindings: Res<RaytracingSceneBindings>,
    shader_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    pipeline: Res<RaytracingInstanceGroupPipeline>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut producer: ResMut<RaytracingProducerEncoder>,
) {
    if groups.0.is_empty() {
        return;
    }
    if !bindings.uses_raw_tlas_build() {
        once!(warn!(
            "RaytracingInstanceGroup transforms are written on the GPU, which needs the raw TLAS \
             build path; group instances are not traced."
        ));
        return;
    }
    let Some((instance_transforms, previous_transforms)) = bindings.instance_transform_buffers()
    else {
        return;
    };

    for group in groups.0.values_mut() {
        let Some(transforms) = shader_buffers.get(group.transforms) else {
            continue;
        };
        let needed = group.instances.len() as u64 * TRANSFORM_SIZE;
        if transforms.buffer.size() < needed {
            once!(warn!(
                "A RaytracingInstanceGroup's transform buffer holds {} bytes but its instances \
                 need {needed}.",
                transforms.buffer.size()
            ));
            continue;
        }
        let slots = match group.slots.as_ref() {
            Some(slots) => slots,
            None => {
                let Some(slots) = group
                    .instances
                    .iter()
                    .map(|entity| bindings.instance_slot(*entity))
                    .collect::<Option<Vec<u32>>>()
                    .filter(|slots| !slots.is_empty())
                else {
                    continue;
                };
                group.slots.insert(
                    render_device.create_buffer_with_data(&BufferInitDescriptor {
                        label: Some("raytracing_instance_group_slots"),
                        contents: bytemuck::cast_slice(&slots),
                        usage: BufferUsages::STORAGE,
                    }),
                )
            }
        };
        let previous = match group.previous.take() {
            Some(previous) if previous.size() == needed => group.previous.insert(previous),
            kept => {
                let previous = render_device.create_buffer(&BufferDescriptor {
                    label: Some("raytracing_instance_group_previous_transforms"),
                    size: needed,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let encoder = producer.encoder(&render_device);
                // Instances keep the transforms they had last frame. The rest have none: they take
                // this frame's, copied after the application's producer passes have written them.
                let kept = kept.map_or(0, |kept| {
                    let size = kept.size().min(needed);
                    encoder.copy_buffer_to_buffer(&kept, 0, &previous, 0, size);
                    size
                });
                if kept < needed {
                    encoder.copy_buffer_to_buffer(
                        &transforms.buffer,
                        kept,
                        &previous,
                        kept,
                        needed - kept,
                    );
                }
                group.previous.insert(previous)
            }
        };
        let ids = [
            transforms.buffer.id(),
            previous.id(),
            slots.id(),
            instance_transforms.id(),
            previous_transforms.id(),
        ];
        if group
            .bind_group
            .as_ref()
            .is_some_and(|(_, bound)| *bound == ids)
        {
            continue;
        }
        let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
        let bind_group = render_device.create_bind_group(
            "raytracing_instance_group_bind_group",
            &layout,
            &BindGroupEntries::sequential((
                transforms.buffer.as_entire_binding(),
                previous.as_entire_binding(),
                slots.as_entire_binding(),
                instance_transforms.as_entire_binding(),
                previous_transforms.as_entire_binding(),
            )),
        );
        group.bind_group = Some((bind_group, ids));
    }
}

/// Copies every group's transforms into its instance slots, after the sparse buffer uploads and
/// before the TLAS build.
pub fn write_raytracing_instance_group_transforms(
    groups: Res<RaytracingInstanceGroups>,
    pipeline: Res<RaytracingInstanceGroupPipeline>,
    pipeline_cache: Res<PipelineCache>,
    mut render_context: RenderContext,
) {
    let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.id) else {
        return;
    };
    if groups.0.values().all(|group| group.bind_group.is_none()) {
        return;
    }
    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let command_encoder = render_context.command_encoder();
    let time_span = diagnostics.time_span(command_encoder, "raytracing_instance_groups");
    {
        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("raytracing_instance_groups"),
            timestamp_writes: None,
        });
        pass.set_pipeline(compute_pipeline);
        for group in groups.0.values() {
            let Some((bind_group, _)) = group.bind_group.as_ref() else {
                continue;
            };
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(
                (group.instances.len() as u32).div_ceil(WORKGROUP_SIZE),
                1,
                1,
            );
        }
    }
    time_span.end(command_encoder);
}
