//! Importance-sampled environment lighting from the Solari camera's `EnvironmentMapLight`.
//!
//! The cubemap's mip 0 is the radiance; `intensity` and `rotation` apply exactly as in `bevy_pbr`'s
//! raster path and the skybox. Sampling it uses an importance pyramid: the six cube faces on a
//! 4x2 face grid of one `R32Float` atlas (faces 0..3 on row 0, 4..5 on row 1, two empty slots of
//! weight 0), each mip-0 texel holding `luminance * texel solid angle`, mipped down by summation to
//! a 1x1 root that equals the total weight. A sample descends the pyramid from the root, picking a
//! quadrant per level with two uniforms that are rescaled after every choice and finally jitter the
//! direction inside the leaf texel (the hierarchical warp RTXDI and Falcor use); its pdf is
//! `leaf / root` times the leaf's Jacobian. The whole walk is a pure function of the light sample
//! seed, which `ReSTIR` relies on: reservoirs store only `(light_id, seed)` and re-resolve the
//! direction every frame, so the same seed must always yield the same direction.
//!
//! The pyramid's face size follows the bound cubemap: the source face size rounded down to a power
//! of two and capped at [`ENVIRONMENT_IMPORTANCE_MAX_FACE_SIZE`], so up to the cap every leaf is
//! one source texel. A coarser pyramid would sample a few-texel sun disc at its neighbourhood's
//! average density: the leaf jitter is uniform, so most samples aimed at the disc's leaf miss it and
//! the hits carry the leaf's too-low pdf. The pyramid is reallocated (texture, mip views, downsample
//! bind groups) whenever the required face size changes; the initial one is small and only ever
//! bound while there is no environment. At the cap the atlas is 4096x2048 `R32Float`, 32 MiB for
//! mip 0 and about 43 MiB with the mips.
//!
//! The pyramid is rebuilt when the bound cubemap changes (a different asset or texture, or the
//! bound asset re-prepared this frame) and every frame while the map is a
//! `GeneratedEnvironmentMapLight` placeholder, whose contents `bevy_pbr`'s generation nodes rewrite
//! each frame. A map modified while unbound is rebuilt on its next binding; one modified while
//! another map is bound is not noticed if it comes back with the same texture. The build runs in
//! the `Render` schedule ahead of the render graph, so a regenerated map is sampled through the
//! pyramid of its previous frame's contents: one frame of lag in the sampling density only, never
//! in the radiance, so it cannot bias the result. A per-frame rebuild costs one cubemap tap per
//! atlas texel plus the downsample chain: 2M taps and about 2.8M texel writes for a typical 512/face
//! generated atmosphere, four times that at the cap.
//!
//! Without an `EnvironmentMapLight` there is no environment light source at all: no entry in the
//! light list, so its selection probability is 0, and a zero tint, so a missed ray collects 0. The
//! cube binding is then a 1x1 black placeholder that exists only because the bind group layout is
//! static; it is never sampled as light.

use bevy_asset::{load_embedded_asset, AssetId, AssetServer};
use bevy_ecs::{
    component::Component,
    resource::Resource,
    system::{Commands, Res, ResMut},
};
use bevy_image::Image;
use bevy_math::{Quat, UVec2};
use bevy_render::{
    diagnostic::RecordDiagnostics as _,
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{self, texture_2d, texture_cube, texture_storage_2d, uniform_buffer},
        AddressMode, BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor,
        BindGroupLayoutEntries, CachedComputePipelineId, ComputePassDescriptor,
        ComputePipelineDescriptor, Extent3d, FilterMode, MipmapFilterMode, PipelineCache, Sampler,
        SamplerBindingType, SamplerDescriptor, ShaderStages, ShaderType, StorageTextureAccess,
        Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureId, TextureSampleType,
        TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension, UniformBuffer,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    texture::GpuImage,
};
use bevy_utils::default;

/// Largest face size, in texels, of the importance pyramid: a source with larger faces is sampled
/// through a pyramid of this size, its leaves averaging `(source / cap)^2` texels each.
pub const ENVIRONMENT_IMPORTANCE_MAX_FACE_SIZE: u32 = 1024;
/// Face size of the pyramid allocated at startup, bound only while there is no environment.
const ENVIRONMENT_IMPORTANCE_PLACEHOLDER_FACE_SIZE: u32 = 16;
const ENVIRONMENT_IMPORTANCE_LABEL: &str = "solari_environment_importance_map";
/// Faces per row of the importance atlas.
pub(crate) const ENVIRONMENT_ATLAS_FACES_X: u32 = 4;
/// Face rows of the importance atlas.
pub(crate) const ENVIRONMENT_ATLAS_FACES_Y: u32 = 2;
/// Lower bound on a texel's luminance, so every real texel keeps a nonzero probability. Only the
/// build shader applies it; this mirror pins the literal there.
#[cfg(test)]
const ENVIRONMENT_IMPORTANCE_FLOOR: f32 = 1e-6;

/// Mip levels of a pyramid with faces of `face_size` texels: the atlas is `4N x 2N`, so
/// `log2(N) + 2` halvings reach `2x1` and one more reaches the `1x1` root.
pub fn importance_mip_count(face_size: u32) -> u32 {
    face_size.ilog2() + 3
}

/// Pyramid face size for a source cubemap with faces of `source_face_size` texels: the source size
/// rounded down to a power of two, at most [`ENVIRONMENT_IMPORTANCE_MAX_FACE_SIZE`].
pub fn importance_face_size(source_face_size: u32) -> u32 {
    (1u32 << source_face_size.max(1).ilog2()).min(ENVIRONMENT_IMPORTANCE_MAX_FACE_SIZE)
}

/// The `EnvironmentMapLight` of a Solari camera, mirrored into the render world.
#[derive(Component, Clone, Debug)]
pub(crate) struct ExtractedSolariEnvironmentMap {
    pub specular_map: AssetId<Image>,
    pub intensity: f32,
    /// `EnvironmentMapLight::rotation`, un-inverted. The binder inverts it when uploading.
    pub rotation: Quat,
    /// The map is a `GeneratedEnvironmentMapLight` placeholder whose GPU contents are rewritten
    /// every frame, so the pyramid is rebuilt every frame too.
    pub contents_change_every_frame: bool,
}

/// A mip pyramid of luminance-times-solid-angle weights over the cube faces.
pub struct ImportancePyramid {
    /// `(asset, texture)` the contents were built from; `None` until first targeted.
    pub source: Option<(AssetId<Image>, TextureId)>,
    /// `R32Float`, `4N x 2N`, `log2(N) + 3` mips.
    pub texture: Texture,
    /// All mips; bound at scene binding 19.
    pub view: TextureView,
    /// Mip 0 alone, the build pass's output.
    mip0_view: TextureView,
    /// `[level - 1] -> [level]` for every level from 1, the downsample passes' bindings.
    downsample_bind_groups: Vec<BindGroup>,
}

impl ImportancePyramid {
    fn new(
        render_device: &RenderDevice,
        downsample_layout: &BindGroupLayout,
        label: &'static str,
        face_size: u32,
    ) -> Self {
        let mip_count = importance_mip_count(face_size);
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: ENVIRONMENT_ATLAS_FACES_X * face_size,
                height: ENVIRONMENT_ATLAS_FACES_Y * face_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING
                | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor {
            label: Some(label),
            ..default()
        });
        let mut mip_views: Vec<_> = (0..mip_count)
            .map(|level| {
                texture.create_view(&TextureViewDescriptor {
                    label: Some(label),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..default()
                })
            })
            .collect();
        let downsample_bind_groups = mip_views
            .windows(2)
            .map(|pair| {
                render_device.create_bind_group(
                    "environment_importance_map_downsample_bind_group",
                    downsample_layout,
                    &BindGroupEntries::sequential((&pair[0], &pair[1])),
                )
            })
            .collect();
        Self {
            source: None,
            texture,
            view,
            mip0_view: mip_views.swap_remove(0),
            downsample_bind_groups,
        }
    }

    pub fn face_size(&self) -> u32 {
        self.texture.size().width / ENVIRONMENT_ATLAS_FACES_X
    }

    pub fn mip_count(&self) -> u32 {
        self.texture.mip_level_count()
    }

    /// Size of mip `level`.
    pub fn level_size(&self, level: u32) -> UVec2 {
        let size = self.texture.size();
        UVec2::new(size.width >> level, (size.height >> level).max(1))
    }
}

/// Most taps per atlas texel axis the build pass takes when the source cubemap lacks the mip that
/// would match the atlas resolution; a mip-less 4096-face map is fully covered, larger ones are
/// strided.
pub const ENVIRONMENT_IMPORTANCE_MAX_OVERSAMPLE: u32 = 16;

#[derive(ShaderType, Clone, Copy, Default)]
pub(crate) struct EnvironmentImportanceBuildConstants {
    pub face_size: u32,
    pub source_mip: f32,
    /// Taps per axis inside each atlas texel, averaged; 1 when `source_mip` matches the atlas.
    pub oversample: u32,
}

/// Source mip and per-axis tap count for a source cubemap of `source_face_size` texels with
/// `source_mip_count` mips, sampled into a pyramid with `face_size` faces. The mip whose texels
/// match the atlas texels is ideal; when the source has fewer mips, the finest available one is
/// read at `2^(missing levels)` taps per axis (capped) so a small bright feature between the
/// texel centres still contributes.
pub(crate) fn importance_source_sampling(
    source_face_size: u32,
    source_mip_count: u32,
    face_size: u32,
) -> (u32, u32) {
    let ideal_mip = source_face_size
        .max(1)
        .ilog2()
        .saturating_sub(face_size.ilog2());
    let source_mip = ideal_mip.min(source_mip_count.saturating_sub(1));
    let oversample =
        (1u32 << (ideal_mip - source_mip).min(31)).min(ENVIRONMENT_IMPORTANCE_MAX_OVERSAMPLE);
    (source_mip, oversample)
}

/// The importance pyramid Solari samples the environment through, and the bindings that stand in
/// for the cubemap when there is none.
#[derive(Resource)]
pub struct EnvironmentImportanceMaps {
    /// Linear/clamp sampler used for every cubemap read (build pass and scene bind group).
    pub sampler: Sampler,
    /// The pyramid for the view's cubemap, re-targeted per asset and reallocated when the source's
    /// face size asks for a different [`importance_face_size`]; `source` is `None` while unused.
    pub pyramid: ImportancePyramid,
    /// Layout of the downsample passes, needed to build a reallocated pyramid's bind groups.
    downsample_layout: BindGroupLayout,
    /// A 1x1 black cube bound at scene binding 17 while no `EnvironmentMapLight` is bound. Only a
    /// placeholder for the static layout: with no environment light entry and a zero tint it is
    /// never sampled as light.
    pub placeholder_cube: TextureView,
    /// A [`request`](Self::request) asked for a rebuild and the build pass has not run since.
    pub(crate) needs_build: bool,
    pub(crate) build_constants: UniformBuffer<EnvironmentImportanceBuildConstants>,
}

impl EnvironmentImportanceMaps {
    /// Targets the pyramid at `map`'s cubemap, scheduling a rebuild when the asset or its GPU
    /// texture differs from what the pyramid was last built from, when `modified` says the asset
    /// was re-prepared this frame (an in-place update keeps the texture), or on every call while
    /// the map's contents change every frame. A retarget reallocates the pyramid first when the
    /// cubemap's face size asks for a different pyramid size.
    pub(crate) fn request(
        &mut self,
        map: &ExtractedSolariEnvironmentMap,
        gpu: &GpuImage,
        modified: bool,
        device: &RenderDevice,
        queue: &RenderQueue,
    ) {
        let source = (map.specular_map, gpu.texture.id());
        let retargeted = self.pyramid.source != Some(source);
        if retargeted {
            let source_face_size = gpu.texture_descriptor.size.width;
            let face_size = importance_face_size(source_face_size);
            if self.pyramid.face_size() != face_size {
                bevy_log::info!(
                    source_face_size,
                    face_size,
                    "environment importance pyramid reallocated"
                );
                self.pyramid = ImportancePyramid::new(
                    device,
                    &self.downsample_layout,
                    ENVIRONMENT_IMPORTANCE_LABEL,
                    face_size,
                );
            }
            self.pyramid.source = Some(source);
            let (source_mip, oversample) = importance_source_sampling(
                source_face_size,
                gpu.texture_descriptor.mip_level_count,
                face_size,
            );
            self.build_constants
                .set(EnvironmentImportanceBuildConstants {
                    face_size,
                    source_mip: source_mip as f32,
                    oversample,
                });
            self.build_constants.write_buffer(device, queue);
        }
        self.needs_build |= retargeted || modified || map.contents_change_every_frame;
    }

    /// Forgets the pyramid's source, so the next [`request`](Self::request) rebuilds whatever it
    /// binds: a map modified while unbound would otherwise come back with a stale pyramid.
    pub(crate) fn release(&mut self) {
        self.pyramid.source = None;
        self.needs_build = false;
    }
}

/// Layouts and pipelines of the importance pyramid build.
#[derive(Resource)]
pub struct EnvironmentImportanceMapPipelines {
    pub build_layout: BindGroupLayoutDescriptor,
    pub build: CachedComputePipelineId,
    pub downsample: CachedComputePipelineId,
}

pub fn init_environment_importance_maps(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let sampler = render_device.create_sampler(&SamplerDescriptor {
        label: Some("solari_environment_map_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: MipmapFilterMode::Linear,
        ..default()
    });

    let build_layout = BindGroupLayoutDescriptor::new(
        "environment_importance_map_build_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_cube(TextureSampleType::Float { filterable: true }),
                binding_types::sampler(SamplerBindingType::Filtering),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<EnvironmentImportanceBuildConstants>(false),
            ),
        ),
    );
    let downsample_layout = BindGroupLayoutDescriptor::new(
        "environment_importance_map_downsample_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );

    let build = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("environment_importance_map_build_pipeline".into()),
        layout: vec![build_layout.clone()],
        shader: load_embedded_asset!(
            asset_server.as_ref(),
            "environment_importance_map_build.wesl"
        ),
        entry_point: Some("build_environment_importance_map".into()),
        ..default()
    });
    let downsample = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("environment_importance_map_downsample_pipeline".into()),
        layout: vec![downsample_layout.clone()],
        shader: load_embedded_asset!(
            asset_server.as_ref(),
            "environment_importance_map_downsample.wesl"
        ),
        entry_point: Some("downsample_environment_importance_map".into()),
        ..default()
    });

    let downsample_layout = pipeline_cache.get_bind_group_layout(&downsample_layout);
    let pyramid = ImportancePyramid::new(
        &render_device,
        &downsample_layout,
        ENVIRONMENT_IMPORTANCE_LABEL,
        ENVIRONMENT_IMPORTANCE_PLACEHOLDER_FACE_SIZE,
    );

    // wgpu zero-initialises it, which is the black the placeholder wants.
    let placeholder_cube = render_device
        .create_texture(&TextureDescriptor {
            label: Some("solari_environment_placeholder_cube"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&TextureViewDescriptor {
            label: Some("solari_environment_placeholder_cube"),
            dimension: Some(TextureViewDimension::Cube),
            ..default()
        });

    commands.insert_resource(EnvironmentImportanceMaps {
        sampler,
        pyramid,
        downsample_layout,
        placeholder_cube,
        needs_build: false,
        build_constants: UniformBuffer::default(),
    });
    commands.insert_resource(EnvironmentImportanceMapPipelines {
        build_layout,
        build,
        downsample,
    });
}

/// Rebuilds the pyramid when [`EnvironmentImportanceMaps::request`] asked for it.
///
/// Runs after the scene bind group was prepared (which already bound the pyramid's view) and before
/// `Core3d`, so Solari's passes read the finished pyramid. While a pipeline is still compiling the
/// pyramid stays all zero: every environment pdf is 0, NEE environment candidates get weight 0 and
/// BRDF misses keep MIS weight 1, which is unbiased, just not importance sampled. The same holds
/// on a generated map's first frame, whose placeholder is still black.
pub fn build_environment_importance_maps(
    pipelines: Res<EnvironmentImportanceMapPipelines>,
    mut maps: ResMut<EnvironmentImportanceMaps>,
    images: Res<RenderAssets<GpuImage>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    if !maps.needs_build {
        return;
    }
    let (Some(build_pipeline), Some(downsample_pipeline)) = (
        pipeline_cache.get_compute_pipeline(pipelines.build),
        pipeline_cache.get_compute_pipeline(pipelines.downsample),
    ) else {
        return;
    };
    let maps = &mut *maps;
    let pyramid = &maps.pyramid;
    let Some(source) = pyramid.source.and_then(|(asset, _)| images.get(asset)) else {
        return;
    };
    let Some(build_constants) = maps.build_constants.binding() else {
        return;
    };

    let build_bind_group = render_device.create_bind_group(
        "environment_importance_map_build_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipelines.build_layout),
        &BindGroupEntries::sequential((
            &source.texture_view,
            &maps.sampler,
            &pyramid.mip0_view,
            build_constants,
        )),
    );

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    let time_span = diagnostics.time_span(encoder, "environment_importance_map");

    {
        let size = pyramid.level_size(0);
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("environment_importance_map_build"),
            timestamp_writes: None,
        });
        pass.set_pipeline(build_pipeline);
        pass.set_bind_group(0, &build_bind_group, &[]);
        pass.dispatch_workgroups(size.x.div_ceil(8), size.y.div_ceil(8), 1);
    }
    for (level, bind_group) in (1..).zip(&pyramid.downsample_bind_groups) {
        let size = pyramid.level_size(level);
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("environment_importance_map_downsample"),
            timestamp_writes: None,
        });
        pass.set_pipeline(downsample_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(size.x.div_ceil(8), size.y.div_ceil(8), 1);
    }

    time_span.end(encoder);
    maps.needs_build = false;
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::{Vec2, Vec3};
    use core::f32::consts::PI;

    // Rust mirror of `environment_map.wesl` and the `bevy_pbr` cube helpers it calls, so the atlas
    // math and the pyramid's pdf can be checked without a GPU. Same formulas; face coordinates are
    // the 0..1 `uv` of `dir_to_cube_uv`.

    /// `bevy_pbr::render::utils::dir_to_cube_uv`.
    fn cube_direction_to_face(c: Vec3) -> (u32, Vec2) {
        let a = c.abs();
        let (face, uv) = if a.x >= a.y && a.x >= a.z {
            if c.x > 0.0 {
                (0, Vec2::new(-c.z, -c.y) / c.x)
            } else {
                (1, Vec2::new(c.z, -c.y) / a.x)
            }
        } else if a.y >= a.x && a.y >= a.z {
            if c.y > 0.0 {
                (2, Vec2::new(c.x, c.z) / c.y)
            } else {
                (3, Vec2::new(c.x, -c.z) / a.y)
            }
        } else if c.z > 0.0 {
            (4, Vec2::new(c.x, -c.y) / c.z)
        } else {
            (5, Vec2::new(-c.x, -c.y) / a.z)
        };
        (face, uv * 0.5 + 0.5)
    }

    /// `bevy_pbr::render::utils::sample_cube_dir`.
    fn cube_face_to_direction(face: u32, uv: Vec2) -> Vec3 {
        let st = uv * 2.0 - 1.0;
        let (s, t) = (st.x, st.y);
        match face {
            0 => Vec3::new(1.0, -t, -s),
            1 => Vec3::new(-1.0, -t, s),
            2 => Vec3::new(s, 1.0, t),
            3 => Vec3::new(s, -1.0, -t),
            4 => Vec3::new(s, -t, 1.0),
            5 => Vec3::new(-s, -t, -1.0),
            _ => Vec3::ZERO,
        }
        .normalize()
    }

    fn face_to_atlas_texel((face, uv): (u32, Vec2), face_size: u32) -> UVec2 {
        let uv = uv.clamp(Vec2::ZERO, Vec2::ONE);
        let local = (uv * face_size as f32)
            .as_uvec2()
            .min(UVec2::splat(face_size - 1));
        UVec2::new(
            face % ENVIRONMENT_ATLAS_FACES_X,
            face / ENVIRONMENT_ATLAS_FACES_X,
        ) * face_size
            + local
    }

    /// Face and `uv` of the point `jitter` inside mip-0 atlas `texel`, or `None` in the two empty
    /// slots.
    fn atlas_position_to_face(texel: UVec2, jitter: Vec2, face_size: u32) -> Option<(u32, Vec2)> {
        let face_xy = texel / face_size;
        let face = face_xy.y * ENVIRONMENT_ATLAS_FACES_X + face_xy.x;
        if face >= 6 {
            return None;
        }
        let local = (texel - face_xy * face_size).as_vec2() + jitter;
        Some((face, local / face_size as f32))
    }

    /// `1 / (solid angle of a face texel)` at `uv` for a face of `face_size` texels.
    fn cube_face_jacobian(uv: Vec2, face_size: u32) -> f32 {
        let st = uv * 2.0 - 1.0;
        let n = face_size as f32;
        let x = st.dot(st) + 1.0;
        0.25 * n * n * x * x.sqrt()
    }

    fn cube_face_texel_solid_angle(uv: Vec2, face_size: u32) -> f32 {
        1.0 / cube_face_jacobian(uv, face_size)
    }

    /// Sum-of-children mip step, the rule `environment_importance_map_downsample.wesl` applies.
    fn downsample_level(source: &[f32], source_size: UVec2) -> (Vec<f32>, UVec2) {
        let size = UVec2::new((source_size.x / 2).max(1), (source_size.y / 2).max(1));
        let mut out = vec![0.0; (size.x * size.y) as usize];
        for y in 0..size.y {
            for x in 0..size.x {
                let mut sum = 0.0;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (sx, sy) = (x * 2 + dx, y * 2 + dy);
                        if sx < source_size.x && sy < source_size.y {
                            sum += source[(sy * source_size.x + sx) as usize];
                        }
                    }
                }
                out[(y * size.x + x) as usize] = sum;
            }
        }
        (out, size)
    }

    /// The pyramid the build and downsample passes produce for a constant-luminance cubemap with
    /// faces of `face_size` texels: mip 0 holds each texel's solid angle, then sums.
    fn constant_pyramid_levels(face_size: u32) -> Vec<(Vec<f32>, UVec2)> {
        let size = UVec2::new(4, 2) * face_size;
        let mut level0 = vec![0.0; (size.x * size.y) as usize];
        for y in 0..size.y {
            for x in 0..size.x {
                if let Some((_, uv)) =
                    atlas_position_to_face(UVec2::new(x, y), Vec2::splat(0.5), face_size)
                {
                    level0[(y * size.x + x) as usize] = cube_face_texel_solid_angle(uv, face_size);
                }
            }
        }
        let mut levels = vec![(level0, size)];
        for _ in 1..importance_mip_count(face_size) {
            let (source, source_size) = levels.last().unwrap();
            levels.push(downsample_level(source, *source_size));
        }
        levels
    }

    /// Density per steradian a pyramid of `levels` samples `direction` with, the shader's
    /// `environment_direction_pdf`.
    fn pyramid_pdf(levels: &[(Vec<f32>, UVec2)], face_size: u32, direction: Vec3) -> f32 {
        let (face, uv) = cube_direction_to_face(direction);
        let texel = face_to_atlas_texel((face, uv), face_size);
        let (level0, size0) = &levels[0];
        let leaf = level0[(texel.y * size0.x + texel.x) as usize];
        let root = levels.last().unwrap().0[0];
        (leaf / root) * cube_face_jacobian(uv, face_size)
    }

    fn faces_and_axes() -> [(u32, Vec3); 6] {
        [
            (0, Vec3::X),
            (1, Vec3::NEG_X),
            (2, Vec3::Y),
            (3, Vec3::NEG_Y),
            (4, Vec3::Z),
            (5, Vec3::NEG_Z),
        ]
    }

    #[test]
    fn cube_face_mapping_matches_wgpu_face_selection() {
        for (face, axis) in faces_and_axes() {
            let (f, uv) = cube_direction_to_face(axis);
            assert_eq!(f, face);
            assert!(uv.abs_diff_eq(Vec2::splat(0.5), 1e-6), "{axis}: {uv}");
        }
        let cases = [
            (Vec3::new(0.5, 0.2, 1.0), 4, Vec2::new(0.75, 0.4)),
            (Vec3::new(1.0, 0.3, 0.4), 0, Vec2::new(0.3, 0.35)),
            (Vec3::new(-0.2, -1.0, 0.6), 3, Vec2::new(0.4, 0.2)),
        ];
        for (direction, face, uv) in cases {
            let (f, got) = cube_direction_to_face(direction);
            assert_eq!(f, face, "{direction}");
            assert!(got.abs_diff_eq(uv, 1e-6), "{direction}: {got} want {uv}");
        }
    }

    #[test]
    fn cube_face_roundtrip() {
        for face in 0..6 {
            for i in 0..17 {
                for j in 0..17 {
                    let uv = Vec2::new(
                        0.0005 + 0.999 * i as f32 / 16.0,
                        0.0005 + 0.999 * j as f32 / 16.0,
                    );
                    let (f, got) = cube_direction_to_face(cube_face_to_direction(face, uv));
                    assert_eq!(f, face, "{face} {uv}");
                    assert!(got.abs_diff_eq(uv, 1e-5), "{face} {uv}: {got}");
                }
            }
        }
        let golden = PI * (3.0 - 5f32.sqrt());
        for i in 0..4096 {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / 4096.0;
            let r = (1.0 - y * y).sqrt();
            let phi = golden * i as f32;
            let (sin_phi, cos_phi) = bevy_math::ops::sin_cos(phi);
            let d = Vec3::new(r * cos_phi, y, r * sin_phi);
            let (face, uv) = cube_direction_to_face(d);
            let back = cube_face_to_direction(face, uv);
            assert!(back.abs_diff_eq(d, 1e-5), "{d}: {back}");
        }
    }

    #[test]
    fn atlas_texel_roundtrip() {
        let centre = Vec2::splat(0.5);
        for n in [1u32, 4, 256] {
            for face in 0..6u32 {
                let origin = UVec2::new(face % 4, face / 4) * n;
                for local in [
                    UVec2::new(0, 0),
                    UVec2::new(n - 1, 0),
                    UVec2::new(0, n - 1),
                    UVec2::new(n - 1, n - 1),
                    UVec2::new(n / 2, n / 2),
                ] {
                    let texel = origin + local;
                    let coords = atlas_position_to_face(texel, centre, n).expect("real face");
                    assert_eq!(coords.0, face);
                    assert_eq!(face_to_atlas_texel(coords, n), texel, "N={n} face={face}");
                }
            }
            for column in [2, 3] {
                assert_eq!(
                    atlas_position_to_face(UVec2::new(column * n, n), centre, n),
                    None
                );
                assert_eq!(
                    atlas_position_to_face(UVec2::new(column * n + n - 1, 2 * n - 1), centre, n),
                    None
                );
            }
        }
        let (face, uv) = atlas_position_to_face(UVec2::ZERO, centre, 256).unwrap();
        assert_eq!(face, 0);
        assert!(uv.abs_diff_eq(Vec2::splat(0.5 / 256.0), 1e-6), "{uv}");
    }

    #[test]
    fn texel_solid_angles_sum_to_the_sphere() {
        let n = 64;
        let mut total = 0.0f64;
        for y in 0..n {
            for x in 0..n {
                let uv = (Vec2::new(x as f32, y as f32) + 0.5) / n as f32;
                total += 6.0 * cube_face_texel_solid_angle(uv, n) as f64;
            }
        }
        let sphere = 4.0 * core::f64::consts::PI;
        assert!(
            ((total - sphere) / sphere).abs() < 2e-3,
            "{total} vs {sphere}"
        );
    }

    #[test]
    fn pyramid_levels_sum_to_the_root() {
        // Face size 1: one texel per face subtends 4 sr, the two empty slots 0, the root 24.
        let levels = constant_pyramid_levels(1);
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0].0, [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 0.0]);
        assert_eq!(levels[1].0, [16.0, 8.0]);
        assert_eq!(levels[2].0, [24.0]);

        // Any face size: the root is the whole sphere, 4 pi, up to the texel-centre quadrature.
        let levels = constant_pyramid_levels(16);
        assert_eq!(levels.len() as u32, importance_mip_count(16));
        assert_eq!(levels.last().unwrap().1, UVec2::ONE);
        let root = levels.last().unwrap().0[0] as f64;
        let sphere = 4.0 * core::f64::consts::PI;
        assert!(
            ((root - sphere) / sphere).abs() < 2e-2,
            "{root} vs {sphere}"
        );
    }

    #[test]
    fn pyramid_pdf_integrates_to_one() {
        for face_size in [1u32, 8] {
            let levels = constant_pyramid_levels(face_size);
            let n = 128;
            let mut integral = 0.0f64;
            for face in 0..6 {
                for y in 0..n {
                    for x in 0..n {
                        let uv = (Vec2::new(x as f32, y as f32) + 0.5) / n as f32;
                        let direction = cube_face_to_direction(face, uv);
                        let pdf = pyramid_pdf(&levels, face_size, direction);
                        integral += pdf as f64 / cube_face_jacobian(uv, n) as f64;
                    }
                }
            }
            assert!((integral - 1.0).abs() < 1e-3, "N={face_size}: {integral}");
        }
        // Face size 1: leaf 4 of root 24, times the Jacobian at the sample's own uv.
        let levels = constant_pyramid_levels(1);
        let uv = Vec2::new(0.65, 0.4);
        let expected = (4.0 / 24.0) * cube_face_jacobian(uv, 1);
        let got = pyramid_pdf(&levels, 1, cube_face_to_direction(2, uv));
        assert!((got - expected).abs() < 1e-5, "{got} vs {expected}");
    }

    #[test]
    fn importance_mip_counts() {
        assert_eq!(importance_mip_count(1024), 13);
        assert_eq!(importance_mip_count(256), 11);
        assert_eq!(importance_mip_count(16), 7);
    }

    #[test]
    fn pyramid_face_size_follows_the_source_up_to_the_cap() {
        // At or below the cap every leaf is one source texel, whatever the source's mip chain.
        let face_size = importance_face_size(1024);
        assert_eq!(face_size, 1024);
        assert_eq!(importance_source_sampling(1024, 1, face_size), (0, 1));
        assert_eq!(importance_source_sampling(1024, 11, face_size), (0, 1));
        assert_eq!(importance_face_size(128), 128);
        assert_eq!(importance_source_sampling(128, 1, 128), (0, 1));
        // Above the cap the leaves average source texels: a full chain through its mip, a mip-less
        // source through taps.
        let face_size = importance_face_size(4096);
        assert_eq!(face_size, ENVIRONMENT_IMPORTANCE_MAX_FACE_SIZE);
        assert_eq!(importance_source_sampling(4096, 1, face_size), (0, 4));
        assert_eq!(importance_source_sampling(4096, 13, face_size), (2, 1));
        // A non-power-of-two source rounds down, so no leaf straddles a texel boundary.
        assert_eq!(importance_face_size(96), 64);
        assert_eq!(importance_face_size(1), 1);
        assert_eq!(importance_face_size(0), 1);
        assert!(ENVIRONMENT_IMPORTANCE_MAX_FACE_SIZE.is_power_of_two());
        assert!(ENVIRONMENT_IMPORTANCE_PLACEHOLDER_FACE_SIZE.is_power_of_two());
    }

    #[test]
    fn source_sampling_covers_mipless_cubemaps() {
        // Full mip chain: read the mip that matches the atlas, one tap.
        assert_eq!(importance_source_sampling(1024, 11, 256), (2, 1));
        assert_eq!(importance_source_sampling(256, 9, 256), (0, 1));
        // Smaller than the atlas: mip 0, one (bilinear) tap.
        assert_eq!(importance_source_sampling(128, 1, 256), (0, 1));
        assert_eq!(importance_source_sampling(1, 1, 256), (0, 1));
        // A GeneratedEnvironmentMapLight placeholder: full chain at the source resolution.
        assert_eq!(importance_source_sampling(512, 10, 256), (1, 1));
        assert_eq!(importance_source_sampling(128, 8, 256), (0, 1));
        // Mip-less converter output: mip 0 with one tap per source texel.
        assert_eq!(importance_source_sampling(1024, 1, 256), (0, 4));
        assert_eq!(importance_source_sampling(4096, 1, 256), (0, 16));
        // Partial chain: the finest available mip, oversampled for the missing levels.
        assert_eq!(importance_source_sampling(2048, 2, 256), (1, 4));
        // Beyond the cap the taps stride the source.
        assert_eq!(
            importance_source_sampling(8192, 1, 256),
            (0, ENVIRONMENT_IMPORTANCE_MAX_OVERSAMPLE)
        );
    }

    #[test]
    fn shader_constants_match_rust() {
        let library = include_str!("environment_map.wesl");
        assert!(library.contains("const ENVIRONMENT_ATLAS_FACES_X = 4u;"));
        assert!(library.contains("const ENVIRONMENT_ATLAS_FACES_Y = 2u;"));
        assert!(library.contains(&format!(
            "const ENVIRONMENT_IMPORTANCE_FLOOR = {ENVIRONMENT_IMPORTANCE_FLOOR:?}f;"
        )));
        assert_eq!(ENVIRONMENT_ATLAS_FACES_X, 4);
        assert_eq!(ENVIRONMENT_ATLAS_FACES_Y, 2);

        let bindings = include_str!("bindings.wesl");
        assert!(bindings.contains("@group(0) @binding(17) var environment_map: texture_cube<f32>;"));
        assert!(bindings.contains("@group(0) @binding(18) var environment_map_sampler: sampler;"));
        assert!(bindings
            .contains("@group(0) @binding(19) var environment_importance_map: texture_2d<f32>;"));

        let sampling = include_str!("sampling.wesl");
        assert!(sampling.contains("sample_environment_direction(light_sample.seed)"));
        assert!(sampling.contains("1.0 / sample.pdf"));
    }
}
