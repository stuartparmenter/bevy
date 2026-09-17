use core::mem::{self, size_of};

use bevy_asset::{prelude::AssetChanged, Assets};
use bevy_camera::visibility::ViewVisibility;
use bevy_ecs::prelude::*;
use bevy_math::Mat4;
use bevy_mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes};
use bevy_render::render_resource::{Buffer, BufferDescriptor};
use bevy_render::settings::WgpuLimits;
use bevy_render::sync_world::{MainEntity, MainEntityHashMap};
use bevy_render::{
    batching::NoAutomaticBatching,
    render_resource::BufferUsages,
    renderer::{RenderDevice, RenderQueue},
    Extract,
};
use bevy_transform::prelude::GlobalTransform;
use offset_allocator::{Allocation, Allocator};
use tracing::error;

use crate::MeshDeformationRequests;

/// Maximum number of joints supported for skinned meshes.
///
/// It is used to allocate buffers.
/// The correctness of the value depends on the GPU/platform.
/// The current value is chosen because it is guaranteed to work everywhere.
/// To allow for bigger values, a check must be made for the limits
/// of the GPU at runtime, which would mean not using consts anymore.
pub const MAX_JOINTS: usize = 256;

/// The total number of joints we support.
///
/// This is 256 GiB worth of joint matrices, which we will never hit under any
/// reasonable circumstances.
const MAX_TOTAL_JOINTS: u32 = 1024 * 1024 * 1024;

/// The number of joints that we allocate at a time.
///
/// Some hardware requires that uniforms be allocated on 256-byte boundaries, so
/// we need to allocate 4 64-byte matrices at a time to satisfy alignment
/// requirements.
const JOINTS_PER_ALLOCATION_UNIT: u32 = (256 / size_of::<Mat4>()) as u32;

/// The location of the first joint matrix in the skin uniform buffer.
#[derive(Clone, Copy)]
pub struct SkinByteOffset {
    /// The byte offset of the first joint matrix.
    pub byte_offset: u32,
}

impl SkinByteOffset {
    /// Index to be in address space based on the size of a skin uniform.
    const fn from_index(index: usize) -> Self {
        SkinByteOffset {
            byte_offset: (index * size_of::<Mat4>()) as u32,
        }
    }

    /// Returns this skin index in elements (not bytes).
    ///
    /// Each element is a 4x4 matrix.
    pub fn index(&self) -> u32 {
        self.byte_offset / size_of::<Mat4>() as u32
    }
}

/// The GPU buffers containing joint matrices for all skinned meshes.
///
/// This is double-buffered: we store the joint matrices of each mesh for the
/// previous frame in addition to those of each mesh for the current frame. This
/// is for motion vector calculation. Every frame, we swap buffers and overwrite
/// the joint matrix buffer from two frames ago with the data for the current
/// frame.
///
/// Notes on implementation: see comment on top of the `extract_skins` system.
#[derive(Resource)]
pub struct SkinUniforms {
    /// The CPU-side buffer that stores the joint matrices for skinned meshes in
    /// the current frame.
    pub current_staging_buffer: Vec<Mat4>,
    /// The GPU-side buffer that stores the joint matrices for skinned meshes in
    /// the current frame.
    pub current_buffer: Buffer,
    /// The GPU-side buffer that stores the joint matrices for skinned meshes in
    /// the previous frame.
    pub prev_buffer: Buffer,
    /// The offset allocator that manages the placement of the joints within the
    /// [`Self::current_buffer`].
    allocator: Allocator,
    /// Allocation information that we keep about each skin.
    skin_uniform_info: MainEntityHashMap<SkinUniformInfo>,
    /// The total number of joints in the scene.
    ///
    /// We use this as part of our heuristic to decide whether to use
    /// fine-grained change detection.
    total_joints: usize,
}

pub fn skin_uniforms_from_world(device: Res<RenderDevice>, mut commands: Commands) {
    let buffer_usages = (if skins_use_uniform_buffers(&device.limits()) {
        BufferUsages::UNIFORM
    } else {
        BufferUsages::STORAGE
    }) | BufferUsages::COPY_DST;

    // Create the current and previous buffer with the minimum sizes.
    //
    // These will be swapped every frame.
    let current_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("skin uniform buffer"),
        size: MAX_JOINTS as u64 * size_of::<Mat4>() as u64,
        usage: buffer_usages,
        mapped_at_creation: false,
    });
    let prev_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("skin uniform buffer"),
        size: MAX_JOINTS as u64 * size_of::<Mat4>() as u64,
        usage: buffer_usages,
        mapped_at_creation: false,
    });

    let res = SkinUniforms {
        current_staging_buffer: vec![],
        current_buffer,
        prev_buffer,
        allocator: Allocator::new(MAX_TOTAL_JOINTS),
        skin_uniform_info: MainEntityHashMap::default(),
        total_joints: 0,
    };

    commands.insert_resource(res);
}

impl SkinUniforms {
    /// Returns the current offset in joints of the skin in the buffer.
    pub fn skin_index(&self, skin: MainEntity) -> Option<u32> {
        self.skin_uniform_info
            .get(&skin)
            .map(SkinUniformInfo::offset)
    }

    /// Returns the current offset in bytes of the skin in the buffer.
    pub fn skin_byte_offset(&self, skin: MainEntity) -> Option<SkinByteOffset> {
        self.skin_uniform_info.get(&skin).map(|skin_uniform_info| {
            SkinByteOffset::from_index(skin_uniform_info.offset() as usize)
        })
    }

    /// Returns an iterator over all skins in the scene.
    pub fn all_skins(&self) -> impl Iterator<Item = &MainEntity> {
        self.skin_uniform_info.keys()
    }
}

/// Allocation information about each skin.
struct SkinUniformInfo {
    /// The allocation of the joints within the [`SkinUniforms::current_buffer`].
    allocation: Allocation,
    /// The entities that comprise the joints.
    joints: Vec<MainEntity>,
}

impl SkinUniformInfo {
    /// The offset in joints within the [`SkinUniforms::current_staging_buffer`].
    fn offset(&self) -> u32 {
        self.allocation.offset * JOINTS_PER_ALLOCATION_UNIT
    }
}

/// Returns true if skinning must use uniforms (and dynamic offsets) because
/// storage buffers aren't supported on the current platform.
pub fn skins_use_uniform_buffers(limits: &WgpuLimits) -> bool {
    bevy_render::storage_buffers_are_unsupported(limits)
}

/// Uploads the buffers containing the joints to the GPU.
pub fn prepare_skins(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    uniform: ResMut<SkinUniforms>,
) {
    let uniform = uniform.into_inner();

    if uniform.current_staging_buffer.is_empty() {
        return;
    }

    // Swap current and previous buffers.
    mem::swap(&mut uniform.current_buffer, &mut uniform.prev_buffer);

    // Resize the buffers if necessary. Include extra space equal to `MAX_JOINTS`
    // because we need to be able to bind a full uniform buffer's worth of data
    // if skins use uniform buffers on this platform.
    let needed_size = (uniform.current_staging_buffer.len() as u64 + MAX_JOINTS as u64)
        * size_of::<Mat4>() as u64;
    if uniform.current_buffer.size() < needed_size {
        let mut new_size = uniform.current_buffer.size();
        while new_size < needed_size {
            // 1.5× growth factor.
            new_size = (new_size + new_size / 2).next_multiple_of(4);
        }

        // Create the new buffers.
        let buffer_usages = if skins_use_uniform_buffers(&render_device.limits()) {
            BufferUsages::UNIFORM
        } else {
            BufferUsages::STORAGE
        } | BufferUsages::COPY_DST;
        uniform.current_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("skin uniform buffer"),
            usage: buffer_usages,
            size: new_size,
            mapped_at_creation: false,
        });
        uniform.prev_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("skin uniform buffer"),
            usage: buffer_usages,
            size: new_size,
            mapped_at_creation: false,
        });

        // We've created a new `prev_buffer` but we don't have the previous joint
        // data needed to fill it out correctly. Use the current joint data
        // instead.
        //
        // TODO: This is a bug - will cause motion blur to ignore joint movement
        // for one frame.
        render_queue.write_buffer(
            &uniform.prev_buffer,
            0,
            bytemuck::must_cast_slice(&uniform.current_staging_buffer[..]),
        );
    }

    // Write the data from `uniform.current_staging_buffer` into
    // `uniform.current_buffer`.
    render_queue.write_buffer(
        &uniform.current_buffer,
        0,
        bytemuck::must_cast_slice(&uniform.current_staging_buffer[..]),
    );

    // We don't need to write `uniform.prev_buffer` because we already wrote it
    // last frame, and the data should still be on the GPU.
}

// Notes on implementation:
// We define the uniform binding as an array<mat4x4<f32>, N> in the shader,
// where N is the maximum number of Mat4s we can fit in the uniform binding,
// which may be as little as 16kB or 64kB. But, we may not need all N.
// We may only need, for example, 10.
//
// If we used uniform buffers ‘normally’ then we would have to write a full
// binding of data for each dynamic offset binding, which is wasteful, makes
// the buffer much larger than it needs to be, and uses more memory bandwidth
// to transfer the data, which then costs frame time So @superdump came up
// with this design: just bind data at the specified offset and interpret
// the data at that offset as an array<T, N> regardless of what is there.
//
// So instead of writing N Mat4s when you only need 10, you write 10, and
// then pad up to the next dynamic offset alignment. Then write the next.
// And for the last dynamic offset binding, make sure there is a full binding
// of data after it so that the buffer is of size
// `last dynamic offset` + `array<mat4x4<f32>>`.
//
// Then when binding the first dynamic offset, the first 10 entries in the array
// are what you expect, but if you read the 11th you’re reading ‘invalid’ data
// which could be padding or could be from the next binding.
//
// In this way, we can pack ‘variable sized arrays’ into uniform buffer bindings
// which normally only support fixed size arrays. You just have to make sure
// in the shader that you only read the values that are valid for that binding.
pub fn extract_skins(
    skin_uniforms: ResMut<SkinUniforms>,
    deformation_requests: Res<MeshDeformationRequests>,
    skinned_meshes: Extract<Query<(Entity, Option<&ViewVisibility>, &SkinnedMesh)>>,
    changed_skinned_meshes: Extract<
        Query<
            (Entity, Option<&ViewVisibility>, &SkinnedMesh),
            Or<(Changed<SkinnedMesh>, AssetChanged<SkinnedMesh>)>,
        >,
    >,
    skinned_mesh_inverse_bindposes: Extract<Res<Assets<SkinnedMeshInverseBindposes>>>,
    changed_transforms: Extract<Query<(Entity, &GlobalTransform), Changed<GlobalTransform>>>,
    joints: Extract<Query<&GlobalTransform>>,
    mut removed_skinned_meshes_query: Extract<RemovedComponents<SkinnedMesh>>,
) {
    let skin_uniforms = skin_uniforms.into_inner();

    // Reconcile visibility and deformation requests, then update joint transforms.
    for (entity, visibility, skin) in &skinned_meshes {
        let skin_entity = MainEntity::from(entity);
        if !visibility.is_some_and(|visibility| visibility.get())
            && !deformation_requests.contains(skin_entity)
        {
            remove_skin(skin_uniforms, skin_entity);
            continue;
        }

        if changed_skinned_meshes.contains(entity)
            || !skin_uniforms.skin_uniform_info.contains_key(&skin_entity)
        {
            remove_skin(skin_uniforms, skin_entity);
            add_skin(
                skin_entity,
                skin,
                skin_uniforms,
                &skinned_mesh_inverse_bindposes,
                &joints,
            );
        } else {
            extract_joints_for_skin(
                skin_entity,
                skin,
                skin_uniforms,
                &skinned_mesh_inverse_bindposes,
                &changed_transforms,
            );
        }
    }

    for skinned_mesh_entity in removed_skinned_meshes_query.read() {
        // A skin removed and re-added in the same frame was already initialized above.
        if !skinned_meshes.contains(skinned_mesh_entity) {
            remove_skin(skin_uniforms, skinned_mesh_entity.into());
        }
    }
}

/// Extracts all joints for a single skin and writes their transforms into the
/// CPU staging buffer.
fn extract_joints_for_skin(
    skin_entity: MainEntity,
    skin: &SkinnedMesh,
    skin_uniforms: &mut SkinUniforms,
    skinned_mesh_inverse_bindposes: &Assets<SkinnedMeshInverseBindposes>,
    changed_transforms: &Query<(Entity, &GlobalTransform), Changed<GlobalTransform>>,
) {
    // Fetch information about the skin.
    let Some(skin_uniform_info) = skin_uniforms.skin_uniform_info.get(&skin_entity) else {
        return;
    };
    let Some(skinned_mesh_inverse_bindposes) =
        skinned_mesh_inverse_bindposes.get(&skin.inverse_bindposes)
    else {
        return;
    };

    // Calculate and write in the new joint matrices, if they changed this frame.
    for (joint_index, (&joint, skinned_mesh_inverse_bindpose)) in skin
        .joints
        .iter()
        .zip(skinned_mesh_inverse_bindposes.iter())
        .enumerate()
    {
        // Skip if the global transform for this joint didn't change.
        let Ok((_, joint_transform)) = changed_transforms.get(joint) else {
            continue;
        };

        let joint_matrix = joint_transform.affine() * *skinned_mesh_inverse_bindpose;
        skin_uniforms.current_staging_buffer[skin_uniform_info.offset() as usize + joint_index] =
            joint_matrix;
    }
}

/// Allocates space for a new skin in the buffers, and populates its joints.
fn add_skin(
    skinned_mesh_entity: MainEntity,
    skinned_mesh: &SkinnedMesh,
    skin_uniforms: &mut SkinUniforms,
    skinned_mesh_inverse_bindposes: &Assets<SkinnedMeshInverseBindposes>,
    joints: &Query<&GlobalTransform>,
) {
    // Allocate space for the joints.
    let Some(allocation) = skin_uniforms.allocator.allocate(
        skinned_mesh
            .joints
            .len()
            .div_ceil(JOINTS_PER_ALLOCATION_UNIT as usize) as u32,
    ) else {
        error!(
            "Out of space for skin: {:?}. Tried to allocate space for {:?} joints.",
            skinned_mesh_entity,
            skinned_mesh.joints.len()
        );
        return;
    };

    // Store that allocation.
    let skin_uniform_info = SkinUniformInfo {
        allocation,
        joints: skinned_mesh
            .joints
            .iter()
            .map(|entity| MainEntity::from(*entity))
            .collect(),
    };

    let skinned_mesh_inverse_bindposes =
        skinned_mesh_inverse_bindposes.get(&skinned_mesh.inverse_bindposes);

    for (joint_index, &joint) in skinned_mesh.joints.iter().enumerate() {
        // Calculate the initial joint matrix.
        let skinned_mesh_inverse_bindpose =
            skinned_mesh_inverse_bindposes.and_then(|skinned_mesh_inverse_bindposes| {
                skinned_mesh_inverse_bindposes.get(joint_index)
            });
        let joint_matrix = match (skinned_mesh_inverse_bindpose, joints.get(joint)) {
            (Some(skinned_mesh_inverse_bindpose), Ok(transform)) => {
                transform.affine() * *skinned_mesh_inverse_bindpose
            }
            _ => Mat4::IDENTITY,
        };

        // Write in the new joint matrix, growing the staging buffer if
        // necessary.
        let buffer_index = skin_uniform_info.offset() as usize + joint_index;
        if skin_uniforms.current_staging_buffer.len() < buffer_index + 1 {
            skin_uniforms
                .current_staging_buffer
                .resize(buffer_index + 1, Mat4::IDENTITY);
        }
        skin_uniforms.current_staging_buffer[buffer_index] = joint_matrix;
    }

    // Record the number of joints.
    skin_uniforms.total_joints += skinned_mesh.joints.len();

    skin_uniforms
        .skin_uniform_info
        .insert(skinned_mesh_entity, skin_uniform_info);
}

/// Deallocates a skin and removes it from the [`SkinUniforms`].
fn remove_skin(skin_uniforms: &mut SkinUniforms, skinned_mesh_entity: MainEntity) {
    let Some(old_skin_uniform_info) = skin_uniforms.skin_uniform_info.remove(&skinned_mesh_entity)
    else {
        return;
    };

    // Free the allocation.
    skin_uniforms
        .allocator
        .free(old_skin_uniform_info.allocation);

    // Update the total number of joints.
    skin_uniforms.total_joints -= old_skin_uniform_info.joints.len();
}

// NOTE: The skinned joints uniform buffer has to be bound at a dynamic offset per
// entity and so cannot currently be batched on WebGL 2.
pub fn no_automatic_skin_batching(
    mut commands: Commands,
    query: Query<Entity, (With<SkinnedMesh>, Without<NoAutomaticBatching>)>,
    render_device: Res<RenderDevice>,
) {
    if !skins_use_uniform_buffers(&render_device.limits()) {
        return;
    }

    for entity in &query {
        commands.entity(entity).try_insert(NoAutomaticBatching);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::system::RunSystemOnce;
    use bevy_math::Vec3;
    use bevy_platform::future::block_on;
    use bevy_render::MainWorld;

    fn dummy_device() -> RenderDevice {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::NOOP,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            display: None,
            backend_options: wgpu::BackendOptions {
                noop: wgpu::NoopBackendOptions {
                    enable: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        });
        let adapter =
            block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
        let (device, _) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        RenderDevice::from(device)
    }

    #[test]
    fn requested_skins_follow_joint_updates_and_request_lifetime() {
        let mut main_world = MainWorld::default();
        let mut bindposes = Assets::<SkinnedMeshInverseBindposes>::default();
        let bindposes_handle = bindposes.add(vec![Mat4::IDENTITY]);
        main_world.insert_resource(bindposes);
        let joint = main_world.spawn_empty().id();
        let skin = SkinnedMesh {
            inverse_bindposes: bindposes_handle,
            joints: vec![joint],
        };
        let requested = main_world.spawn(skin.clone()).id();
        let visible = main_world
            .spawn((skin.clone(), ViewVisibility::VISIBLE))
            .id();
        let offscreen = main_world.spawn((skin, ViewVisibility::HIDDEN)).id();

        let mut render_world = World::new();
        render_world.insert_resource(main_world);
        render_world.insert_resource(dummy_device());
        render_world.init_resource::<MeshDeformationRequests>();
        render_world
            .resource_mut::<MeshDeformationRequests>()
            .insert(requested.into());
        render_world
            .run_system_once(skin_uniforms_from_world)
            .unwrap();
        let mut extract = Schedule::default();
        extract.add_systems(extract_skins);
        extract.run(&mut render_world);

        let uniforms = render_world.resource::<SkinUniforms>();
        let offset = uniforms.skin_index(requested.into()).unwrap();
        assert!(uniforms.skin_index(visible.into()).is_some());
        assert!(uniforms.skin_index(offscreen.into()).is_none());
        assert_eq!(
            uniforms.current_staging_buffer[offset as usize],
            Mat4::IDENTITY
        );

        // Joint transforms can arrive after the mesh instance is extracted.
        let transform = GlobalTransform::from_translation(Vec3::new(1.0, 2.0, 3.0));
        {
            let mut main_world = render_world.resource_mut::<MainWorld>();
            main_world.clear_trackers();
            main_world.entity_mut(joint).insert(transform);
            main_world
                .entity_mut(requested)
                .insert(ViewVisibility::VISIBLE);
        }
        extract.run(&mut render_world);
        let uniforms = render_world.resource::<SkinUniforms>();
        assert_eq!(uniforms.skin_index(requested.into()), Some(offset));
        assert_eq!(
            uniforms.current_staging_buffer[offset as usize],
            transform.to_matrix()
        );

        // Raster visibility does not end a requested skin's allocation.
        {
            let mut main_world = render_world.resource_mut::<MainWorld>();
            main_world.clear_trackers();
            main_world
                .entity_mut(requested)
                .insert(ViewVisibility::HIDDEN);
        }
        extract.run(&mut render_world);
        assert_eq!(
            render_world
                .resource::<SkinUniforms>()
                .skin_index(requested.into()),
            Some(offset)
        );

        render_world.insert_resource(MeshDeformationRequests::default());
        render_world.resource_mut::<MainWorld>().clear_trackers();
        extract.run(&mut render_world);
        assert!(render_world
            .resource::<SkinUniforms>()
            .skin_index(requested.into())
            .is_none());
        assert!(render_world
            .resource::<SkinUniforms>()
            .skin_index(visible.into())
            .is_some());

        render_world
            .resource_mut::<MeshDeformationRequests>()
            .insert(requested.into());
        render_world.resource_mut::<MainWorld>().clear_trackers();
        extract.run(&mut render_world);
        assert!(render_world
            .resource::<SkinUniforms>()
            .skin_index(requested.into())
            .is_some());
        {
            let mut main_world = render_world.resource_mut::<MainWorld>();
            main_world.clear_trackers();
            main_world.entity_mut(requested).remove::<SkinnedMesh>();
        }
        extract.run(&mut render_world);
        assert!(render_world
            .resource::<SkinUniforms>()
            .skin_index(requested.into())
            .is_none());
    }
}
