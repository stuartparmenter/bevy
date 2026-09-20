use super::SolariLighting;
use crate::scene::RaytracingSceneNeedsPreviousFrameData;
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy_anti_alias::dlss::{
    Dlss, DlssRayReconstructionFeature, ViewDlssRayReconstructionTextures,
};
use bevy_camera::MainPassResolutionOverride;
use bevy_diagnostic::FrameCount;
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy_ecs::query::Has;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    system::{Commands, Query, Res},
};
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages, TextureDescriptor,
        TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::CachedTexture,
};
use bytemuck::{Pod, Zeroable};
use core::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// Size of the `LightSample` shader struct in bytes.
const LIGHT_SAMPLE_STRUCT_SIZE: u64 = 8;

/// Size of the `ResolvedLightSamplePacked` shader struct in bytes.
const RESOLVED_LIGHT_SAMPLE_STRUCT_SIZE: u64 = 24;

/// Size of the `Reservoir` shader struct in bytes.
const RESERVOIR_STRUCT_SIZE: u64 = 64;

pub const LIGHT_TILE_BLOCKS: u64 = 128;
pub const LIGHT_TILE_SAMPLES_PER_BLOCK: u64 = 1024;

/// Amount of entries in the world cache (must be a power of 2, and >= 2^10)
pub const WORLD_CACHE_SIZE: u64 = 2u64.pow(20);
/// Sum of per-cell field sizes in `WorldCache`. Keep in sync with `bindings.wesl`.
const WORLD_CACHE_ENTRY_SIZE: u64 = 84;
/// Size of the fixed `b` array (`array<u32, WORLD_CACHE_SIZE / 1024>`).
const WORLD_CACHE_B_SIZE: u64 = (WORLD_CACHE_SIZE / 1024) * size_of::<u32>() as u64;
/// Offset of `active_cells_count`.
pub const WORLD_CACHE_ACTIVE_CELLS_COUNT_OFFSET: u64 =
    WORLD_CACHE_SIZE * WORLD_CACHE_ENTRY_SIZE + WORLD_CACHE_B_SIZE;
/// Must stay under wgpu's default `max_storage_buffer_binding_size` (128 MiB or 2^27 bytes).
pub const WORLD_CACHE_BUFFER_SIZE: u64 =
    (WORLD_CACHE_ACTIVE_CELLS_COUNT_OFFSET + size_of::<u32>() as u64).next_multiple_of(16);

/// GPU representation of the user-configurable [`SolariLighting`] settings, plus
/// per-frame state.
///
/// Field order and types must match the `SolariLightingSettings` struct in
/// `bindings.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SolariLightingUniforms {
    confidence_weight_cap: f32,
    primary_di_samples: u32,
    secondary_di_samples: u32,
    max_bounces: u32,
    world_cache_max_temporal_samples: f32,
    world_cache_direct_light_sample_count: u32,
    world_cache_max_gi_ray_distance: f32,
    world_cache_cell_updates_soft_target: u32,
    world_cache_position_base_cell_size: f32,
    world_cache_position_lod_scale: f32,
    frame_rng: u32,
    reset: u32,
    history_reset: u32,
    receiver_overrides: u32,
}

impl SolariLightingUniforms {
    /// `resources_created`: the view's lighting resources hold no history at all.
    /// `history_invalid`: the per-pixel history does not describe the previous frame.
    fn new(
        settings: &SolariLighting,
        frame_count: u32,
        resources_created: bool,
        history_invalid: bool,
    ) -> Self {
        let reset = settings.reset || resources_created;
        Self {
            confidence_weight_cap: settings.confidence_weight_cap,
            primary_di_samples: settings.primary_di_samples,
            secondary_di_samples: settings.secondary_di_samples,
            max_bounces: settings.max_bounces,
            world_cache_max_temporal_samples: settings.world_cache_max_temporal_samples,
            world_cache_direct_light_sample_count: settings.world_cache_direct_light_sample_count,
            world_cache_max_gi_ray_distance: settings.world_cache_max_gi_ray_distance,
            world_cache_cell_updates_soft_target: settings.world_cache_cell_updates_soft_target,
            world_cache_position_base_cell_size: settings.world_cache_position_base_cell_size,
            world_cache_position_lod_scale: settings.world_cache_position_lod_scale,
            frame_rng: frame_count.wrapping_mul(5782582),
            reset: reset as u32,
            history_reset: (reset || history_invalid) as u32,
            receiver_overrides: settings.receiver_overrides as u32,
        }
    }
}

/// Declares to the raytracing scene whether any view needs last frame's TLAS and light ids.
pub fn setup_raytracing_scene_needs_previous_frame_data(
    views: Query<&SolariLighting>,
    needs_previous_frame_data: Option<Res<RaytracingSceneNeedsPreviousFrameData>>,
    mut commands: Commands,
) {
    let restir_used = views.iter().any(|solari_lighting| solari_lighting.restir);
    match (restir_used, needs_previous_frame_data.is_some()) {
        (true, false) => commands.insert_resource(RaytracingSceneNeedsPreviousFrameData),
        (false, true) => commands.remove_resource::<RaytracingSceneNeedsPreviousFrameData>(),
        _ => {}
    }
}

/// Internal rendering resources used for Solari lighting.
#[derive(Component)]
pub struct SolariLightingResources {
    pub constants: Buffer,
    pub light_tile_samples: Buffer,
    pub light_tile_resolved_samples: Buffer,
    pub reservoirs: Option<SolariReservoirBuffers>,
    pub history: LightingHistory,
    pub world_cache: Buffer,
    pub world_cache_active_cells_dispatch: Buffer,
    pub view_size: UVec2,
}

/// Tracks whether a view's per-pixel history describes the previous frame, and which of
/// its ping-pong textures this frame writes. The write index advances only once a full
/// lighting pass is recorded, and a frame without one invalidates the history.
pub struct LightingHistory {
    current_frame: AtomicU32,
    last_rendered_frame: AtomicU64,
    write_index: AtomicUsize,
}

impl LightingHistory {
    fn new(frame: u32) -> Self {
        Self {
            current_frame: AtomicU32::new(frame),
            last_rendered_frame: AtomicU64::new(u64::MAX),
            write_index: AtomicUsize::new(0),
        }
    }

    /// True when the per-pixel history does not describe the previous frame.
    fn begin_frame(&self, frame: u32) -> bool {
        self.current_frame.store(frame, Ordering::Relaxed);
        self.last_rendered_frame.load(Ordering::Relaxed) != u64::from(frame.wrapping_sub(1))
    }

    pub fn write_index(&self) -> usize {
        self.write_index.load(Ordering::Relaxed)
    }

    pub fn record_rendered(&self) {
        self.write_index.fetch_xor(1, Ordering::Relaxed);
        self.last_rendered_frame.store(
            u64::from(self.current_frame.load(Ordering::Relaxed)),
            Ordering::Relaxed,
        );
    }
}

pub struct SolariReservoirBuffers {
    pub a: Buffer,
    pub b: Buffer,
}

/// A view's receiver override textures, present while [`SolariLighting::receiver_overrides`] is set.
///
/// A texel replaces the ray origin policy of its pixel's primary surface. `receiver_override.wesl`
/// defines the texel encoding and the functions that write it. Producers fill
/// [`write_view`](Self::write_view) in a system in
/// [`SolariLightingSystems::WriteReceiverOverrides`](super::SolariLightingSystems::WriteReceiverOverrides).
#[derive(Component)]
pub struct SolariReceiverOverrides {
    /// Both slots hold the same texture when the view keeps no previous frame.
    textures: [CachedTexture; 2],
    /// This frame's [`LightingHistory::write_index`], fixed for every system that renders the view.
    write_index: AtomicUsize,
    size: UVec2,
}

impl SolariReceiverOverrides {
    /// Format of the override textures.
    pub const FORMAT: TextureFormat = TextureFormat::Rgba32Uint;

    fn new(size: UVec2, keep_previous: bool, render_device: &RenderDevice) -> Self {
        let create = |label| {
            let texture = render_device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: Self::FORMAT,
                usage: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING
                    | TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let default_view = texture.create_view(&TextureViewDescriptor::default());
            CachedTexture {
                texture,
                default_view,
            }
        };
        let a = create("solari_lighting_receiver_overrides_a");
        let b = if keep_previous {
            create("solari_lighting_receiver_overrides_b")
        } else {
            a.clone()
        };
        Self {
            textures: [a, b],
            write_index: AtomicUsize::new(0),
            size,
        }
    }

    /// The texture producers write this frame. Solari clears it to zero first.
    pub fn write_view(&self) -> &TextureView {
        &self.textures[self.write_index.load(Ordering::Relaxed)].default_view
    }

    /// The texture written on the previous rendered frame.
    pub(super) fn previous_view(&self) -> &TextureView {
        &self.textures[self.write_index.load(Ordering::Relaxed) ^ 1].default_view
    }

    /// Size of the override textures in texels, equal to the view's lighting resolution.
    pub fn size(&self) -> UVec2 {
        self.size
    }
}

pub fn prepare_solari_lighting_resources(
    #[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))] query: Query<(
        Entity,
        &ExtractedCamera,
        &SolariLighting,
        Option<&SolariLightingResources>,
        Option<&SolariReceiverOverrides>,
        Option<&MainPassResolutionOverride>,
    )>,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] query: Query<(
        Entity,
        &ExtractedCamera,
        &SolariLighting,
        Option<&SolariLightingResources>,
        Option<&SolariReceiverOverrides>,
        Option<&MainPassResolutionOverride>,
        Has<Dlss<DlssRayReconstructionFeature>>,
        Option<&ViewDlssRayReconstructionTextures>,
    )>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    frame_count: Res<FrameCount>,
    mut commands: Commands,
) {
    for query_item in &query {
        #[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))]
        let (
            entity,
            camera,
            solari_lighting,
            solari_lighting_resources,
            receiver_overrides,
            resolution_override,
        ) = query_item;
        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        let (
            entity,
            camera,
            solari_lighting,
            solari_lighting_resources,
            receiver_overrides,
            resolution_override,
            has_dlss_rr,
            dlss_rr_textures,
        ) = query_item;

        let Some(mut view_size) = camera.physical_viewport_size else {
            continue;
        };
        if let Some(MainPassResolutionOverride(resolution_override)) = resolution_override {
            view_size = *resolution_override;
        }

        // Manage RR guide textures separately so toggling RR at the same resolution
        // does not reset lighting history.
        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        if has_dlss_rr {
            let stale = dlss_rr_textures
                .is_none_or(|t| t.diffuse_albedo.texture.size() != view_size.to_extents());
            if stale {
                commands
                    .entity(entity)
                    .insert(create_dlss_rr_textures(view_size, &render_device));
            }
        } else if dlss_rr_textures.is_some() {
            commands
                .entity(entity)
                .remove::<ViewDlssRayReconstructionTextures>();
        }

        let reusable = solari_lighting_resources.filter(|r| {
            r.view_size == view_size
                && r.reservoirs.is_some() == solari_lighting.restir
                && receiver_overrides.is_some() == solari_lighting.receiver_overrides
        });
        let history_invalid =
            reusable.is_none_or(|resources| resources.history.begin_frame(frame_count.0));
        let uniforms = SolariLightingUniforms::new(
            solari_lighting,
            frame_count.0,
            reusable.is_none(),
            history_invalid,
        );

        if let Some(solari_lighting_resources) = reusable {
            // The constants uniform can change every frame, so always upload it.
            render_queue.write_buffer(
                &solari_lighting_resources.constants,
                0,
                bytemuck::bytes_of(&uniforms),
            );
            // The index only advances once lighting is recorded, after every system that
            // writes or reads the override textures this frame.
            if let Some(receiver_overrides) = receiver_overrides {
                receiver_overrides.write_index.store(
                    solari_lighting_resources.history.write_index(),
                    Ordering::Relaxed,
                );
            }
            continue;
        }

        let constants = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("solari_lighting_constants"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let light_tile_samples = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_lighting_light_tile_samples"),
            size: LIGHT_TILE_BLOCKS * LIGHT_TILE_SAMPLES_PER_BLOCK * LIGHT_SAMPLE_STRUCT_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let light_tile_resolved_samples = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_lighting_light_tile_resolved_samples"),
            size: LIGHT_TILE_BLOCKS
                * LIGHT_TILE_SAMPLES_PER_BLOCK
                * RESOLVED_LIGHT_SAMPLE_STRUCT_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let reservoirs = solari_lighting.restir.then(|| {
            let reservoirs_buffer = |name| {
                render_device.create_buffer(&BufferDescriptor {
                    label: Some(name),
                    size: (view_size.x * view_size.y) as u64 * RESERVOIR_STRUCT_SIZE,
                    usage: BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            };
            SolariReservoirBuffers {
                a: reservoirs_buffer("solari_lighting_reservoirs_a"),
                b: reservoirs_buffer("solari_lighting_reservoirs_b"),
            }
        });

        let world_cache = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_lighting_world_cache"),
            size: WORLD_CACHE_BUFFER_SIZE,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let world_cache_active_cells_dispatch = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_lighting_world_cache_active_cells_dispatch"),
            size: size_of::<[u32; 3]>() as u64,
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let mut entity_commands = commands.entity(entity);
        entity_commands.insert(SolariLightingResources {
            constants,
            light_tile_samples,
            light_tile_resolved_samples,
            reservoirs,
            history: LightingHistory::new(frame_count.0),
            world_cache,
            world_cache_active_cells_dispatch,
            view_size,
        });
        if solari_lighting.receiver_overrides {
            entity_commands.insert(SolariReceiverOverrides::new(
                view_size,
                solari_lighting.restir,
                &render_device,
            ));
        } else {
            entity_commands.remove::<SolariReceiverOverrides>();
        }
    }
}

/// The DLSS RR guide textures, sized to the view. Managed independently of
/// [`SolariLightingResources`] — see the creation site above.
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
fn create_dlss_rr_textures(
    view_size: UVec2,
    render_device: &RenderDevice,
) -> ViewDlssRayReconstructionTextures {
    let create = |label: &'static str, format: TextureFormat| {
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: view_size.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let default_view = texture.create_view(&TextureViewDescriptor::default());
        CachedTexture {
            texture,
            default_view,
        }
    };
    ViewDlssRayReconstructionTextures {
        diffuse_albedo: create("solari_lighting_diffuse_albedo", TextureFormat::Rgba8Unorm),
        specular_albedo: create("solari_lighting_specular_albedo", TextureFormat::Rgba8Unorm),
        normal_roughness: create(
            "solari_lighting_normal_roughness",
            TextureFormat::Rgba16Float,
        ),
        depth: create("solari_lighting_dlss_rr_depth", TextureFormat::R32Float),
        specular_motion_vectors: create(
            "solari_lighting_specular_motion_vectors",
            TextureFormat::Rg16Float,
        ),
    }
}

#[cfg(test)]
mod lighting_history_tests {
    use super::{LightingHistory, SolariLightingUniforms};

    #[test]
    fn history_advances_only_on_rendered_frames() {
        let history = LightingHistory::new(10);
        assert!(history.begin_frame(10));
        assert_eq!(history.write_index(), 0);
        history.record_rendered();
        assert_eq!(history.write_index(), 1);
        assert!(!history.begin_frame(11));
        // Frame 11 is prepared but a pipeline is unavailable, so nothing is recorded.
        assert!(history.begin_frame(12));
        assert_eq!(history.write_index(), 1);
        history.record_rendered();
        assert_eq!(history.write_index(), 0);
        assert!(!history.begin_frame(13));
    }

    #[test]
    fn recreated_resources_and_wrapping_frame_count_reset_correctly() {
        let history = LightingHistory::new(u32::MAX);
        assert!(history.begin_frame(u32::MAX));
        history.record_rendered();
        assert!(!history.begin_frame(0));
        history.record_rendered();
        assert!(!history.begin_frame(1));
        // Newly created resources hold no history, whatever the frame count.
        assert!(LightingHistory::new(1).begin_frame(1));
    }

    #[test]
    fn uniforms_match_shader_struct_size() {
        assert_eq!(size_of::<SolariLightingUniforms>(), 56);
    }
}
