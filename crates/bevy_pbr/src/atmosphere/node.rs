use bevy_camera::{MainPassResolutionOverride, Viewport};
use bevy_ecs::system::Res;
use bevy_image::ToExtents;
use bevy_math::{UVec2, UVec3, Vec3Swizzles};
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    extract_component::DynamicUniformIndex,
    render_resource::{ComputePass, ComputePassDescriptor, PipelineCache, RenderPassDescriptor},
    renderer::{RenderContext, ViewQuery},
    view::{ExtractedView, RetainedViewEntity, ViewTarget, ViewUniformOffset},
};

use crate::{resources::GpuAtmosphere, ViewLightsUniformOffset};

use super::{
    resources::{
        AtmosphereBindGroups, AtmosphereLutPipelines, AtmosphereTextures,
        AtmosphereTransformsOffset, RenderSkyHazePipelineId, RenderSkyPipelineId,
    },
    GpuAtmosphereSettings, GpuCloudLayers,
};

pub fn atmosphere_luts(
    view: ViewQuery<(
        &GpuAtmosphereSettings,
        &AtmosphereBindGroups,
        &AtmosphereTextures,
        &DynamicUniformIndex<GpuAtmosphere>,
        &DynamicUniformIndex<GpuAtmosphereSettings>,
        &AtmosphereTransformsOffset,
        &ViewUniformOffset,
        &ViewLightsUniformOffset,
        Option<&DynamicUniformIndex<GpuCloudLayers>>,
        &super::ExtractedAtmosphere,
        &ExtractedView,
        Option<&GpuCloudLayers>,
        Option<&super::resources::CloudLightTurn>,
    )>,
    pipelines: Res<AtmosphereLutPipelines>,
    layouts: Res<super::resources::AtmosphereBindGroupLayouts>,
    weather: Option<Res<super::fbm_noise::FbmNoiseTexture>>,
    influences: Option<Res<super::weather::ExtractedCloudInfluences>>,
    gpu_media: Res<bevy_render::render_asset::RenderAssets<crate::GpuScatteringMedium>>,
    mut per_view: bevy_ecs::system::ResMut<PerViewLutState>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (
        settings,
        bind_groups,
        textures,
        atmosphere_uniforms_offset,
        settings_uniforms_offset,
        atmosphere_transforms_offset,
        view_uniforms_offset,
        lights_uniforms_offset,
        cloud_layer_uniforms_offset,
        extracted_atmosphere,
        extracted_view,
        gpu_cloud_layers,
        light_turn,
    ) = view.into_inner();
    // This system runs once per view: amortised state (the LUT cache, the cloud-field rebuild
    // stage) is per view, or two views would split one counter between them (audit 9).
    let state = per_view.get(extracted_view.retained_view_entity);

    let (
        Some(transmittance_lut_pipeline),
        Some(multiscattering_lut_pipeline),
        Some(sky_view_lut_pipeline),
        Some(aerial_view_lut_pipeline),
    ) = (
        pipeline_cache.get_compute_pipeline(pipelines.transmittance_lut),
        pipeline_cache.get_compute_pipeline(pipelines.multiscattering_lut),
        pipeline_cache.get_compute_pipeline(pipelines.sky_view_lut),
        pipeline_cache.get_compute_pipeline(pipelines.aerial_view_lut),
    )
    else {
        return;
    };

    let (cloud_shadow_map_pipeline, cloud_shadow_filter_pipeline, cloud_shadow_temporal_pipeline) = (
        pipeline_cache.get_compute_pipeline(pipelines.cloud_shadow_map),
        pipeline_cache.get_compute_pipeline(pipelines.cloud_shadow_filter),
        pipeline_cache.get_compute_pipeline(pipelines.cloud_shadow_temporal),
    );

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    fn dispatch_2d(compute_pass: &mut ComputePass, size: UVec2) {
        const WORKGROUP_SIZE: u32 = 16;
        let workgroups_x = size.x.div_ceil(WORKGROUP_SIZE);
        let workgroups_y = size.y.div_ceil(WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    fn dispatch_2d_temporal(compute_pass: &mut ComputePass, size: UVec2) {
        const WORKGROUP_SIZE: u32 = 8;
        let workgroups_x = size.x.div_ceil(WORKGROUP_SIZE);
        let workgroups_y = size.y.div_ceil(WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    let timing_device = ctx.render_device().clone();
    // Weather window around this view's camera (`weather_window.wesl`), before every pass that
    // samples the weather (the cloud-shadowed LUTs, shadow map, field, clouds, probes).
    let window_bind_group = match (gpu_cloud_layers, weather.as_deref()) {
        (Some(layers), Some(weather)) => {
            let words = super::weather::weather_window_params(
                layers.weather_window_origin,
                layers.droplet_number,
                layers.noise_scale.max(1.0),
                layers.noise_offset,
                &extracted_atmosphere.world_to_atmosphere,
                influences.as_deref().unwrap_or(&Default::default()),
            );
            let params = timing_device.create_buffer_with_data(
                &bevy_render::render_resource::BufferInitDescriptor {
                    label: Some("weather_window_params"),
                    contents: bytemuck::cast_slice(&words),
                    usage: bevy_render::render_resource::BufferUsages::UNIFORM,
                },
            );
            Some(timing_device.create_bind_group(
                "weather_window_bind_group",
                &pipeline_cache.get_bind_group_layout(&layouts.weather_window),
                &bevy_render::render_resource::BindGroupEntries::sequential((
                    &weather.texture.default_view,
                    &textures.weather_window_art.default_view,
                    &textures.weather_window_physical.default_view,
                    params.as_entire_binding(),
                )),
            ))
        }
        _ => None,
    };
    let command_encoder = ctx.command_encoder();
    if let (Some(bind_group), Some(pipeline)) = (
        window_bind_group.as_ref(),
        pipeline_cache.get_compute_pipeline(pipelines.weather_window),
    ) {
        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("weather_window"),
            timestamp_writes: super::gpu_timing::compute(&timing_device, "weather_window"),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        let n = super::resources::WEATHER_WINDOW_SIZE.div_ceil(8);
        pass.dispatch_workgroups(n, n, 1);
    }

    // A compute pass whose encoder label, GPU timing entry (`gpu_timing.rs`) and name in traces
    // are all `$label`.
    macro_rules! labelled_compute_pass {
        ($label:expr) => {
            command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some($label),
                timestamp_writes: super::gpu_timing::compute(&timing_device, $label),
            })
        };
    }

    // Pass 1: the atmosphere LUTs, one compute pass each, so each has its own encoder label,
    // GPU timing and diagnostics span in traces (Metal System Trace times encoders, not dispatches).
    {
        // Transmittance + multiscattering LUTs depend only on the atmosphere/medium and LUT
        // settings, not on the view or sun: rebuild them only when those inputs (or the LUT
        // textures) change. The medium enters by its prepared GPU textures, not its handle: an
        // in-place edit of the asset keeps the handle but re-prepares it into new textures, so
        // the LUTs follow it on the next frame (audit 17; a periodic rebuild every 240 frames
        // used to bound that staleness).
        let lut_key = {
            use core::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            let a = extracted_atmosphere;
            for f in [
                a.inner_radius,
                a.outer_radius,
                a.ground_albedo.x,
                a.ground_albedo.y,
                a.ground_albedo.z,
            ] {
                f.to_bits().hash(&mut h);
            }
            a.medium.hash(&mut h);
            if let Some(medium) = gpu_media.get(a.medium) {
                medium.density_lut.id().hash(&mut h);
                medium.scattering_lut.id().hash(&mut h);
            }
            (
                settings.transmittance_lut_size,
                settings.multiscattering_lut_size,
            )
                .hash(&mut h);
            (
                settings.transmittance_lut_samples,
                settings.multiscattering_lut_dirs,
                settings.multiscattering_lut_samples,
            )
                .hash(&mut h);
            textures.transmittance_lut.texture.id().hash(&mut h);
            textures.multiscattering_lut.texture.id().hash(&mut h);
            h.finish()
        };
        let lut_static_dirty =
            state.lut_key != Some(lut_key) || std::env::var_os("ATMOSPHERE_LUT_ALWAYS").is_some();
        state.lut_key = Some(lut_key);

        if lut_static_dirty {
            let mut luts_pass = labelled_compute_pass!("atmosphere_transmittance_lut");
            let pass_span = diagnostics.pass_span(&mut luts_pass, "atmosphere_transmittance_lut");
            luts_pass.set_pipeline(transmittance_lut_pipeline);
            luts_pass.set_bind_group(
                0,
                &bind_groups.transmittance_lut,
                &[
                    atmosphere_uniforms_offset.index(),
                    settings_uniforms_offset.index(),
                ],
            );
            dispatch_2d(&mut luts_pass, settings.transmittance_lut_size);
            pass_span.end(&mut luts_pass);
            drop(luts_pass);

            let mut luts_pass = labelled_compute_pass!("atmosphere_multiscattering_lut");
            let pass_span = diagnostics.pass_span(&mut luts_pass, "atmosphere_multiscattering_lut");
            luts_pass.set_pipeline(multiscattering_lut_pipeline);
            luts_pass.set_bind_group(
                0,
                &bind_groups.multiscattering_lut,
                &[
                    atmosphere_uniforms_offset.index(),
                    settings_uniforms_offset.index(),
                ],
            );
            luts_pass.dispatch_workgroups(
                settings.multiscattering_lut_size.x,
                settings.multiscattering_lut_size.y,
                1,
            );
            pass_span.end(&mut luts_pass);
        }

        // Sky View / Aerial View LUTs. With a VolumetricCloudLayer, the cloud-shadowed variants (the cloud
        // shadow map darkens their sun in-scattering; it is last frame's map, traced below).
        let cloud_luts = match (
            cloud_layer_uniforms_offset.as_ref(),
            bind_groups.sky_view_lut_clouds.as_ref(),
            bind_groups.aerial_view_lut_clouds.as_ref(),
            pipeline_cache.get_compute_pipeline(pipelines.sky_view_lut_clouds),
            pipeline_cache.get_compute_pipeline(pipelines.aerial_view_lut_clouds),
        ) {
            (Some(o), Some(sbg), Some(abg), Some(sp), Some(ap)) => {
                Some((o.index(), sbg, abg, sp, ap))
            }
            _ => None,
        };
        let mut luts_pass = labelled_compute_pass!("atmosphere_sky_view_lut");
        let pass_span = diagnostics.pass_span(&mut luts_pass, "atmosphere_sky_view_lut");
        if let Some((cloud_off, sbg, _, sp, _)) = cloud_luts {
            luts_pass.set_pipeline(sp);
            luts_pass.set_bind_group(
                0,
                sbg,
                &[
                    atmosphere_uniforms_offset.index(),
                    settings_uniforms_offset.index(),
                    atmosphere_transforms_offset.index(),
                    view_uniforms_offset.offset,
                    lights_uniforms_offset.offset,
                    cloud_off,
                ],
            );
        } else {
            luts_pass.set_pipeline(sky_view_lut_pipeline);
            luts_pass.set_bind_group(
                0,
                &bind_groups.sky_view_lut,
                &[
                    atmosphere_uniforms_offset.index(),
                    settings_uniforms_offset.index(),
                    atmosphere_transforms_offset.index(),
                    view_uniforms_offset.offset,
                    lights_uniforms_offset.offset,
                ],
            );
        }
        dispatch_2d(&mut luts_pass, settings.sky_view_lut_size);
        pass_span.end(&mut luts_pass);
        drop(luts_pass);

        // The aerial-perspective LUT is only read by render_sky's lookup-texture path; in
        // raymarched mode every pixel's haze is marched (render_sky / its haze prepass, and the
        // cloud pass in front of clouds), so building the 32^3 cloud-shadowed LUT was wasted.
        let need_ap_lut = settings.rendering_method != 1
            || std::env::var_os("ATMOSPHERE_DBG")
                .is_some_and(|v| v.to_string_lossy().contains("ATM_DBG_HAZE_AP_LUT"));
        if need_ap_lut {
            let mut luts_pass = labelled_compute_pass!("atmosphere_aerial_view_lut");
            let pass_span = diagnostics.pass_span(&mut luts_pass, "atmosphere_aerial_view_lut");
            if let Some((cloud_off, _, abg, _, ap)) = cloud_luts {
                luts_pass.set_pipeline(ap);
                luts_pass.set_bind_group(
                    0,
                    abg,
                    &[
                        atmosphere_uniforms_offset.index(),
                        settings_uniforms_offset.index(),
                        view_uniforms_offset.offset,
                        lights_uniforms_offset.offset,
                        cloud_off,
                    ],
                );
            } else {
                luts_pass.set_pipeline(aerial_view_lut_pipeline);
                luts_pass.set_bind_group(
                    0,
                    &bind_groups.aerial_view_lut,
                    &[
                        atmosphere_uniforms_offset.index(),
                        settings_uniforms_offset.index(),
                        view_uniforms_offset.offset,
                        lights_uniforms_offset.offset,
                    ],
                );
            }
            dispatch_2d(&mut luts_pass, settings.aerial_view_lut_size.xy());
            pass_span.end(&mut luts_pass);
        }
    }

    // Pass 2: cloud shadow map tracing (Unreal-style front depth + extinction stats).
    // Needed whenever clouds exist (both atmosphere modes). Kept in its own compute pass (and diagnostics span)
    // so its GPU cost shows up separately from the atmosphere LUTs in profilers.
    if cloud_layer_uniforms_offset.is_some() {
        let mut cloud_shadow_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloud_shadow_map"),
            timestamp_writes: super::gpu_timing::compute(&timing_device, "cloud_shadow_map"),
        });
        let pass_span = diagnostics.pass_span(&mut cloud_shadow_pass, "cloud_shadow_map");
        {
            if let (Some(cloud_shadow_map_pipeline), Some(_cloud_shadow_filter_pipeline)) =
                (cloud_shadow_map_pipeline, cloud_shadow_filter_pipeline)
            {
                if let (Some(cloud_layer_uniforms_offset), Some(cloud_shadow_map_bg)) = (
                    cloud_layer_uniforms_offset.as_ref(),
                    bind_groups.cloud_shadow_map.as_ref(),
                ) {
                    cloud_shadow_pass.set_pipeline(cloud_shadow_map_pipeline);
                    cloud_shadow_pass.set_bind_group(
                        0,
                        cloud_shadow_map_bg,
                        &[
                            atmosphere_uniforms_offset.index(),
                            settings_uniforms_offset.index(),
                            view_uniforms_offset.offset,
                            lights_uniforms_offset.offset,
                            cloud_layer_uniforms_offset.index(),
                        ],
                    );
                    // With time-slicing, one thread traces one texel per N x N tile.
                    let interleave = settings.cloud_shadow_map_update_interleave.max(1);
                    dispatch_2d(
                        &mut cloud_shadow_pass,
                        UVec2::new(
                            settings.cloud_shadow_map_size.x.div_ceil(interleave),
                            settings.cloud_shadow_map_size.y.div_ceil(interleave),
                        ),
                    );
                }
            }
        }
        pass_span.end(&mut cloud_shadow_pass);
    }

    // Cloud SDF (Nubis Cubed voxel cloud distance): seeds from the coarse density, then a
    // separable Euclidean distance transform. One pass per step for the storage->sampled barriers.
    if let (Some(cloud_layer_uniforms_offset), Some(shadow_bg), Some(edt_bgs)) = (
        cloud_layer_uniforms_offset.as_ref(),
        bind_groups.cloud_shadow_map.as_ref(),
        bind_groups.cloud_field_edt.as_ref(),
    ) {
        let ids = [
            pipelines.cloud_field_init,
            pipelines.cloud_field_edt_x,
            pipelines.cloud_field_edt_y,
            pipelines.cloud_field_edt_z,
            pipelines.cloud_field_escape,
        ];
        let resolved: Option<Vec<_>> = ids
            .iter()
            .map(|id| pipeline_cache.get_compute_pipeline(*id))
            .collect();
        if let Some(resolved) = resolved {
            let groups = UVec3::new(
                super::resources::CLOUD_FIELD_RES_XY.div_ceil(8),
                super::resources::CLOUD_FIELD_RES_XY.div_ceil(8),
                super::resources::CLOUD_FIELD_RES_Z,
            );
            // Amortised rebuild: one stage per frame (init, x, y, z + escape), then 4 idle frames,
            // so the field follows weather/wind within 8 frames at ~1/8 of the full cost per frame.
            // A field this view has not built yet (a new view, or new textures) is built in full
            // first, so nothing reads an incomplete field (audit 9, 20).
            let field_id = textures.cloud_field_sdf.texture.id();
            let full_build = state.field_texture != Some(field_id);
            state.field_texture = Some(field_id);
            let stage = (state.field_frame % 8) as usize;
            state.field_frame = if full_build {
                0
            } else {
                state.field_frame.wrapping_add(1)
            };
            for (i, pipeline) in resolved.into_iter().enumerate() {
                // escape (4) runs right after edt_z (3) in the same frame: both switch to the new
                // build cell together (clouds_raymarch reads the escape volume with the SDF's cell).
                if !full_build && i != stage && !(i == 4 && stage == 3) {
                    continue;
                }
                let mut pass = labelled_compute_pass!(
                    [
                        "cloud_field_init",
                        "cloud_field_edt_x",
                        "cloud_field_edt_y",
                        "cloud_field_edt_z",
                        "cloud_field_escape",
                    ][i]
                );
                pass.set_pipeline(pipeline);
                if i == 0 {
                    pass.set_bind_group(
                        0,
                        shadow_bg,
                        &[
                            atmosphere_uniforms_offset.index(),
                            settings_uniforms_offset.index(),
                            view_uniforms_offset.offset,
                            lights_uniforms_offset.offset,
                            cloud_layer_uniforms_offset.index(),
                        ],
                    );
                } else {
                    pass.set_bind_group(0, &edt_bgs[i - 1], &[]);
                }
                // init: 8x8x1 groups; edt_x/y/z: one workgroup per grid line (cloud_field.wesl).
                let (n_xy, n_z) = (groups.x * 8, groups.z);
                match i {
                    0 => pass.dispatch_workgroups(groups.x, groups.y, groups.z),
                    1 => pass.dispatch_workgroups(1, n_xy, n_z),
                    2 => pass.dispatch_workgroups(n_xy, 1, n_z),
                    3 => pass.dispatch_workgroups(n_xy, n_xy, 1),
                    // escape: half horizontal resolution, 4x4x4 groups.
                    _ => pass.dispatch_workgroups(n_xy / 8, n_xy / 8, n_z.div_ceil(4)),
                }
            }
            // Diffuse sunlight (`cloud_field.wesl` diffusion solve). Not precomputable (the field
            // and the sun move), but both change slowly: per frame the source of 1/8 of the cells
            // (every cell over the field's 8-frame rebuild), one red-black SOR iteration
            // warm-started from the previous frame's solution, and the publish; on a full build
            // the whole source and 32 iterations from zero.
            let diffusion = pipelines
                .cloud_field_diffusion
                .map(|id| pipeline_cache.get_compute_pipeline(id));
            if let (
                Some(bg),
                [Some(source), Some(red), Some(black), Some(publish), Some(reset), Some(source_slice), Some(shift), Some(shift_commit)],
            ) = (bind_groups.cloud_field_diffusion.as_ref(), diffusion)
            {
                let g = UVec3::new(
                    super::resources::CLOUD_FIELD_RES_XY.div_ceil(4),
                    super::resources::CLOUD_FIELD_RES_XY.div_ceil(4),
                    super::resources::CLOUD_FIELD_RES_Z.div_ceil(4),
                );
                let offsets = [
                    atmosphere_uniforms_offset.index(),
                    settings_uniforms_offset.index(),
                    view_uniforms_offset.offset,
                    lights_uniforms_offset.offset,
                    cloud_layer_uniforms_offset.index(),
                ];
                // The steps run in order (wgpu: storage read-write between dispatches). One pass
                // by default, each step a debug group (named in GPU captures); with
                // ATMOSPHERE_PROFILE_PASSES one labelled pass per step, so Metal System Trace
                // shows each. Same frame rate either way (2560x1440, Apple M5), but the split
                // steps' encoder durations include overlap with other work (2.0-2.5 ms in all,
                // against 0.32 ms for the one pass): read them as an upper bound.
                type Dispatch<'a> = (&'a bevy_render::render_resource::ComputePipeline, UVec3);
                let mut steps: Vec<(&'static str, Vec<Dispatch>)> = Vec::new();
                // SOR colours index half the cells in x, the source slice an eighth of the cells.
                let half_x = UVec3::new(g.x.div_ceil(2), g.y, g.z);
                // A turning light moves every cell's source (its sun transmittance) at once:
                // past 1/8 of a voxel over the last 16 frames (`CloudLightTurn`; the shadow map
                // it reads converges within those frames), the whole source is refreshed each
                // frame and the solve takes 8 iterations (red-black SOR on this grid: 8 cut the
                // error of a 30-degree change to 0.25 of its start, against 0.65 for one).
                // Otherwise one slice of the source and one iteration per frame.
                let turning = !full_build
                    && light_turn.is_some_and(|t| t.recent > t.field_tolerance);
                if full_build {
                    steps.push(("cloud_diffusion_reset", vec![(reset, g)]));
                    steps.push(("cloud_diffusion_source", vec![(source, g)]));
                } else {
                    // The extinction was rebuilt (stage 0), maybe at a new grid cell: move the
                    // solution and source with it before this frame's iteration.
                    if stage == 0 {
                        steps.push(("cloud_diffusion_shift", vec![(shift, g), (shift_commit, g)]));
                    }
                    if turning {
                        steps.push(("cloud_diffusion_source", vec![(source, g)]));
                    } else {
                        steps.push((
                            "cloud_diffusion_source_slice",
                            vec![(source_slice, (g + UVec3::ONE) / 2)],
                        ));
                    }
                }
                let sor = [(red, half_x), (black, half_x)];
                let iterations = if full_build {
                    32
                } else if turning {
                    8
                } else {
                    1
                };
                steps.push(("cloud_diffusion_sor", sor.repeat(iterations)));
                steps.push(("cloud_diffusion_publish", vec![(publish, g)]));
                let run = |pass: &mut ComputePass, dispatches: &[Dispatch]| {
                    pass.set_bind_group(0, bg, &offsets);
                    for (p, n) in dispatches {
                        pass.set_pipeline(p);
                        pass.dispatch_workgroups(n.x, n.y, n.z);
                    }
                };
                if profile_passes() {
                    for (label, dispatches) in &steps {
                        run(&mut labelled_compute_pass!(label), dispatches);
                    }
                } else {
                    let mut pass = labelled_compute_pass!("cloud_diffusion");
                    for (label, dispatches) in &steps {
                        pass.push_debug_group(label);
                        run(&mut pass, dispatches);
                        pass.pop_debug_group();
                    }
                }
            }
        }
    }

    // IMPORTANT:
    // We run the cloud shadow *filter* in a separate compute pass so the backend can insert the
    // required resource state transitions (storage-write -> sampled-read) between tracing and filtering.
    // Without this, the filter can appear to do nothing on some backends.
    if settings.cloud_shadow_map_spatial_filter_iterations > 0
        && settings.cloud_shadow_map_update_interleave <= 1
        && cloud_layer_uniforms_offset.is_some()
        && bind_groups.cloud_shadow_filter_a_to_b.is_some()
        && bind_groups.cloud_shadow_filter_b_to_a.is_some()
    {
        let Some(cloud_shadow_filter_pipeline) = cloud_shadow_filter_pipeline else {
            return;
        };
        let iters = settings.cloud_shadow_map_spatial_filter_iterations;
        let iters_even = (iters + 1) & !1;
        if iters_even > 0 {
            let mut filter_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cloud_shadow_filter"),
                timestamp_writes: None,
            });

            filter_pass.set_pipeline(cloud_shadow_filter_pipeline);
            for i in 0..iters_even {
                let bg = if (i & 1) == 0 {
                    bind_groups.cloud_shadow_filter_a_to_b.as_ref().unwrap()
                } else {
                    bind_groups.cloud_shadow_filter_b_to_a.as_ref().unwrap()
                };
                filter_pass.set_bind_group(0, bg, &[settings_uniforms_offset.index()]);
                dispatch_2d(&mut filter_pass, settings.cloud_shadow_map_size);
            }
        }
    }

    // Temporal filter: blend current (tmp) with history, write to cloud_shadow_map.
    // Then copy cloud_shadow_map -> history for next frame.
    if settings.cloud_shadow_uses_temporal_pass() && cloud_layer_uniforms_offset.is_some() {
        if let (Some(temporal_pipeline), Some(temporal_bg)) = (
            cloud_shadow_temporal_pipeline,
            bind_groups.cloud_shadow_temporal.as_ref(),
        ) {
            let mut temporal_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cloud_shadow_temporal"),
                timestamp_writes: super::gpu_timing::compute(
                    &timing_device,
                    "cloud_shadow_temporal",
                ),
            });
            temporal_pass.set_pipeline(temporal_pipeline);
            temporal_pass.set_bind_group(0, temporal_bg, &[]);
            dispatch_2d_temporal(&mut temporal_pass, settings.cloud_shadow_map_size);
        }

        // Copy cloud_shadow_map to history for next frame. Copies run in an unlabelled blit
        // encoder ("Blit Command N" in Metal traces); the debug group names it in GPU captures.
        let copy_size = settings.cloud_shadow_map_size.to_extents();
        command_encoder.push_debug_group("cloud_shadow_map_history_copy");
        command_encoder.copy_texture_to_texture(
            textures.cloud_shadow_map.texture.as_image_copy(),
            textures.cloud_shadow_map_history.texture.as_image_copy(),
            copy_size,
        );
        command_encoder.pop_debug_group();
    }

    // The map's mip chain (`cloud_shadow_mips.wesl`): lookups read it at their pixel footprint.
    if cloud_layer_uniforms_offset.is_some()
        && let Some(mips_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.cloud_shadow_mips)
    {
        let texture = &textures.cloud_shadow_map.texture;
        let layout = pipeline_cache.get_bind_group_layout(&layouts.cloud_shadow_mips);
        let level = |mip: u32| {
            texture.create_view(&bevy_render::render_resource::TextureViewDescriptor {
                label: Some("cloud_shadow_map_mip"),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let mut pass = labelled_compute_pass!("cloud_shadow_mips");
        pass.set_pipeline(mips_pipeline);
        for mip in 1..texture.mip_level_count() {
            let bind_group = timing_device.create_bind_group(
                "cloud_shadow_mips_bind_group",
                &layout,
                &bevy_render::render_resource::BindGroupEntries::sequential((
                    &level(mip - 1),
                    &level(mip),
                )),
            );
            pass.set_bind_group(0, &bind_group, &[]);
            let n = (texture.width() >> mip).max(1).div_ceil(8);
            pass.dispatch_workgroups(n, ((texture.height() >> mip).max(1)).div_ceil(8), 1);
        }
    }
}

pub fn render_sky(
    view: ViewQuery<(
        &ExtractedCamera,
        &AtmosphereBindGroups,
        &ViewTarget,
        &DynamicUniformIndex<GpuAtmosphere>,
        &DynamicUniformIndex<GpuAtmosphereSettings>,
        &AtmosphereTransformsOffset,
        &ViewUniformOffset,
        &ViewLightsUniformOffset,
        Option<&DynamicUniformIndex<GpuCloudLayers>>,
        &RenderSkyPipelineId,
        Option<&MainPassResolutionOverride>,
        &AtmosphereTextures,
        Option<&RenderSkyHazePipelineId>,
        Option<&crate::light_probe::generate::RenderEnvironmentMap>,
        &ExtractedView,
    )>,
    pipeline_cache: Res<PipelineCache>,
    pipelines: Res<AtmosphereLutPipelines>,
    lut_state: Res<PerViewLutState>,
    mut history_writes: bevy_ecs::system::ResMut<super::resources::CloudHistoryWrites>,
    frame_count: Res<bevy_diagnostic::FrameCount>,
    noise_dump: (
        Option<Res<super::perlin_worley_noise::PerlinWorleyNoiseTexture>>,
        Option<Res<super::detail_noise::DetailNoiseTexture>>,
        Option<Res<super::fbm_noise::FbmNoiseTexture>>,
    ),
    mut ctx: RenderContext,
) {
    let (
        camera,
        atmosphere_bind_groups,
        view_target,
        atmosphere_uniforms_offset,
        settings_uniforms_offset,
        atmosphere_transforms_offset,
        view_uniforms_offset,
        lights_uniforms_offset,
        cloud_layer_uniforms_offset,
        render_sky_pipeline_id,
        resolution_override,
        textures,
        haze_pipeline_id,
        env_map,
        extracted_view,
    ) = view.into_inner();

    let Some(render_sky_pipeline) = pipeline_cache.get_render_pipeline(render_sky_pipeline_id.0)
    else {
        return;
    }; //TODO: warning
       // Like a pipeline that is not ready: no sky until this view's cloud field is built, rather
       // than clouds lit from an unbuilt field (zero escape probability, f_ms/(1 - f_ms) ~ 3e6).
    if cloud_layer_uniforms_offset.is_some()
        && !lut_state.field_built(
            extracted_view.retained_view_entity,
            textures.cloud_field_sdf.texture.id(),
        )
    {
        return;
    }

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let dump_device = ctx.render_device().clone();
    let timing_device = ctx.render_device().clone();
    let command_encoder = ctx.command_encoder();

    // Debug readback (CLOUD_DUMP_DIR): textures as they are when this frame's sky starts, i.e.
    // this frame's shadow map / noise and the *previous* frame's cloud trace + reconstruct.
    let dump_frame = super::debug_dump::begin_frame(frame_count.0);
    if let Some(frame) = dump_frame {
        use bevy_math::UVec3;
        let d = |enc: &mut _,
                 name: &str,
                 tex: &bevy_render::render_resource::Texture,
                 mip: u32,
                 size: UVec3,
                 bpt: u32,
                 ext: &str| {
            // Views dump on the same frame: tag the files with the camera order.
            let name = format!("{name}_c{}", camera.order);
            super::debug_dump::dump(&dump_device, enc, frame, &name, tex, mip, size, bpt, ext);
        };
        let s2 = |t: &bevy_render::render_resource::Texture| UVec3::new(t.width(), t.height(), 1);
        for (name, t) in [
            ("shadow_map", &textures.cloud_shadow_map.texture),
            ("shadow_raw", &textures.cloud_shadow_map_tmp.texture),
            ("clouds_trace_prev", &textures.clouds_trace.texture),
            ("clouds_recon_prev", &textures.clouds.texture),
            ("ms_lut", &textures.multiscattering_lut.texture),
            ("sky_view_lut", &textures.sky_view_lut.texture),
            ("weather_window_art", &textures.weather_window_art.texture),
        ] {
            d(command_encoder, name, t, 0, s2(t), 8, "rgba16f");
        }
        // The view's raw atmosphere probe cube (6 layers), before the environment-map filter.
        if let Some(env) = env_map {
            let t = &env.environment_map.texture;
            d(
                command_encoder,
                "env_raw",
                t,
                0,
                UVec3::new(t.width(), t.height(), 6),
                8,
                "rgba16f",
            );
        }
        // The cloud field and its diffusion solve, as this frame left them (all 3D).
        let s3 = |t: &bevy_render::render_resource::Texture| {
            UVec3::new(t.width(), t.height(), t.depth_or_array_layers())
        };
        for (name, t, bpt, ext) in [
            ("field_ext", &textures.cloud_field_ext.texture, 8, "rgba16f"),
            ("field_esc", &textures.cloud_field_esc.texture, 8, "rgba16f"),
            ("field_diff", &textures.cloud_field_diff.texture, 8, "rgba16f"),
            ("field_diff_u", &textures.cloud_field_diff_u.texture, 4, "r32f"),
            ("field_diff_src", &textures.cloud_field_diff_src.texture, 4, "r32f"),
        ] {
            d(command_encoder, name, t, 0, s3(t), bpt, ext);
        }
        let (pw, det, wx) = &noise_dump;
        if let Some(pw) = pw {
            d(
                command_encoder,
                "shape_noise_mip0",
                &pw.texture.texture,
                0,
                pw.size,
                8,
                "rgba16f",
            );
        }
        if let Some(det) = det {
            d(
                command_encoder,
                "detail_noise_mip0",
                &det.texture.texture,
                0,
                det.size,
                8,
                "rgba16f",
            );
        }
        if let Some(wx) = wx {
            let t = &wx.scheme_noise.texture;
            d(
                command_encoder,
                "scheme_noise_mip0",
                t,
                0,
                UVec3::new(t.width(), t.height(), t.depth_or_array_layers()),
                8,
                "rgba16f",
            );
            let t = &wx.profile.texture;
            d(command_encoder, "profile_lut", t, 0, s2(t), 8, "rgba16f");
            d(
                command_encoder,
                "weather",
                &wx.texture.texture,
                0,
                UVec3::new(wx.size.x, wx.size.y, 1),
                8,
                "rgba16f",
            );
        }
    }

    // What this frame writes of the cloud and haze histories (`CloudHistoryWrites`).
    let mut clouds_written = None;
    let mut haze_written = haze_pipeline_id
        .is_none()
        .then(|| textures.haze_inscattering.texture.id());

    // Dedicated low-res cloud raymarch (design.md §1.2). Its output is composited by render_sky.
    if let (Some(cloud_layer_offset), Some(bind_group), Some(pipeline)) = (
        cloud_layer_uniforms_offset.as_ref(),
        atmosphere_bind_groups.clouds_raymarch.as_ref(),
        pipeline_cache.get_compute_pipeline(
            if atmosphere_bind_groups.clouds_raymarch_multisampled {
                pipelines.clouds_raymarch_msaa
            } else {
                pipelines.clouds_raymarch
            },
        ),
    ) {
        let mut clouds_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("clouds_raymarch"),
            timestamp_writes: super::gpu_timing::compute(&timing_device, "clouds_raymarch"),
        });
        let pass_span = diagnostics.pass_span(&mut clouds_pass, "clouds_raymarch");
        clouds_pass.set_pipeline(pipeline);
        clouds_pass.set_bind_group(
            0,
            bind_group,
            &[
                atmosphere_uniforms_offset.index(),
                settings_uniforms_offset.index(),
                atmosphere_transforms_offset.index(),
                view_uniforms_offset.offset,
                lights_uniforms_offset.offset,
                cloud_layer_offset.index(),
            ],
        );
        let size = textures.clouds_trace.texture.size();
        clouds_pass.dispatch_workgroups(size.width.div_ceil(8), size.height.div_ceil(8), 1);
        pass_span.end(&mut clouds_pass);
        drop(clouds_pass);

        // Checkerboard reconstruction into this frame's half-res buffer (ping-pong with history).
        if let (Some(reconstruct_bind_group), Some(reconstruct_pipeline)) = (
            atmosphere_bind_groups.clouds_reconstruct.as_ref(),
            pipeline_cache.get_compute_pipeline(pipelines.clouds_reconstruct),
        ) {
            let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("clouds_reconstruct"),
                timestamp_writes: super::gpu_timing::compute(&timing_device, "clouds_reconstruct"),
            });
            let pass_span = diagnostics.pass_span(&mut pass, "clouds_reconstruct");
            pass.set_pipeline(reconstruct_pipeline);
            pass.set_bind_group(
                0,
                reconstruct_bind_group,
                &[
                    atmosphere_uniforms_offset.index(),
                    view_uniforms_offset.offset,
                ],
            );
            let size = textures.clouds.texture.size();
            pass.dispatch_workgroups(size.width.div_ceil(8), size.height.div_ceil(8), 1);
            pass_span.end(&mut pass);
            clouds_written = Some(textures.clouds.texture.id());
        }
    }

    // Half-resolution haze prepass (render_sky.wesl `haze`): the per-pixel atmosphere march at
    // half resolution, upsampled depth-aware by render_sky. Same bind group layout as render_sky
    // (cloud and geometry shadows), with placeholders in the haze slots it writes.
    if let (Some(haze_id), Some(haze_bg), Some(cloud_offset)) = (
        haze_pipeline_id,
        atmosphere_bind_groups.render_sky_haze.as_ref(),
        cloud_layer_uniforms_offset.as_ref(),
    ) && let Some(haze_pipeline) = pipeline_cache.get_render_pipeline(haze_id.0)
    {
        fn attachment(
            view: &bevy_render::render_resource::TextureView,
        ) -> Option<bevy_render::render_resource::RenderPassColorAttachment<'_>> {
            Some(bevy_render::render_resource::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: bevy_render::render_resource::Operations {
                    load: bevy_render::render_resource::LoadOp::Clear(Default::default()),
                    store: bevy_render::render_resource::StoreOp::Store,
                },
            })
        }
        let mut haze_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("render_sky_haze"),
            color_attachments: &[
                attachment(&textures.haze_inscattering.default_view),
                attachment(&textures.haze_transmittance.default_view),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: super::gpu_timing::render(&timing_device, "render_sky_haze"),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        haze_pass.set_pipeline(haze_pipeline);
        haze_pass.set_bind_group(
            0,
            haze_bg,
            &[
                atmosphere_uniforms_offset.index(),
                settings_uniforms_offset.index(),
                atmosphere_transforms_offset.index(),
                view_uniforms_offset.offset,
                lights_uniforms_offset.offset,
                cloud_offset.index(),
            ],
        );
        haze_pass.draw(0..3, 0..1);
        drop(haze_pass);
        haze_written = Some(textures.haze_inscattering.texture.id());
    }
    history_writes.record(
        extracted_view.retained_view_entity,
        frame_count.0,
        clouds_written,
        haze_written,
    );

    let mut render_sky_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("render_sky"),
        color_attachments: &[Some(view_target.get_color_attachment())],
        depth_stencil_attachment: None,
        timestamp_writes: super::gpu_timing::render(&timing_device, "render_sky"),
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let pass_span = diagnostics.pass_span(&mut render_sky_pass, "render_sky");

    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
    {
        render_sky_pass.set_viewport(
            viewport.physical_position.x as f32,
            viewport.physical_position.y as f32,
            viewport.physical_size.x as f32,
            viewport.physical_size.y as f32,
            viewport.depth.start,
            viewport.depth.end,
        );
    }

    render_sky_pass.set_pipeline(render_sky_pipeline);

    // Select correct bind group + dynamic offsets based on whether the view has VolumetricCloudLayer.
    // No-cloud variant omits the VolumetricCloudLayer binding entirely.
    //
    // If cloud bind group isn't ready yet, skip this pass (pipeline cache will catch up next frame).
    let (bind_group, cloud_layer_offset) = match (
        cloud_layer_uniforms_offset.as_ref(),
        atmosphere_bind_groups.render_sky_clouds.as_ref(),
    ) {
        (Some(offset), Some(bg)) => (bg, Some(offset)),
        _ => (&atmosphere_bind_groups.render_sky_no_clouds, None),
    };

    let offsets_no_clouds = [
        atmosphere_uniforms_offset.index(),
        settings_uniforms_offset.index(),
        atmosphere_transforms_offset.index(),
        view_uniforms_offset.offset,
        lights_uniforms_offset.offset,
    ];

    if let Some(cloud_layer_offset) = cloud_layer_offset {
        let offsets_clouds = [
            offsets_no_clouds[0],
            offsets_no_clouds[1],
            offsets_no_clouds[2],
            offsets_no_clouds[3],
            offsets_no_clouds[4],
            cloud_layer_offset.index(),
        ];
        render_sky_pass.set_bind_group(0, bind_group, &offsets_clouds);
    } else {
        render_sky_pass.set_bind_group(0, bind_group, &offsets_no_clouds);
    }
    render_sky_pass.draw(0..3, 0..1);
    pass_span.end(&mut render_sky_pass);
    drop(render_sky_pass);
    // Debug readback of the final linear (pre-TAA, pre-tonemap) scene color.
    if let Some(frame) = dump_frame {
        let t = view_target.main_texture();
        super::debug_dump::dump(
            &dump_device,
            command_encoder,
            frame,
            &format!("scene_hdr_c{}", camera.order),
            t,
            0,
            UVec3::new(t.width(), t.height(), 1),
            8,
            "rgba16f",
        );
    }
    super::gpu_timing::end_frame(&timing_device, command_encoder);
}

/// Amortised per-view state of [`atmosphere_luts`]. A resource, not a `Local`, so
/// [`render_sky`] can tell whether a view's cloud field has been built.
#[derive(Default, bevy_ecs::resource::Resource)]
pub struct PerViewLutState {
    views: bevy_platform::collections::HashMap<RetainedViewEntity, (u64, ViewLutState)>,
    tick: u64,
}

#[derive(Default)]
pub struct ViewLutState {
    /// Inputs of the cached transmittance / multiscattering LUTs.
    lut_key: Option<u64>,
    /// Cloud-field rebuild stage counter and the SDF texture it was built into.
    field_frame: u32,
    field_texture: Option<bevy_render::render_resource::TextureId>,
}

impl PerViewLutState {
    /// Whether `view` has completed a full build of the cloud field into `field`. A field that
    /// was never built (its pipelines or bind groups were not ready yet) holds zeroed metadata
    /// and escape probabilities, which the cloud trace would read as a valid field (audit 20).
    pub fn field_built(
        &self,
        view: RetainedViewEntity,
        field: bevy_render::render_resource::TextureId,
    ) -> bool {
        self.views
            .get(&view)
            .is_some_and(|(_, state)| state.field_texture == Some(field))
    }

    /// The state of a view; views not rendered for 600 invocations are forgotten.
    fn get(&mut self, view: RetainedViewEntity) -> &mut ViewLutState {
        self.tick += 1;
        let tick = self.tick;
        if tick.is_multiple_of(600) {
            self.views.retain(|_, (last, _)| tick - *last < 600);
        }
        let entry = self.views.entry(view).or_default();
        entry.0 = tick;
        &mut entry.1
    }
}

/// `ATMOSPHERE_PROFILE_PASSES=1`: split multi-step compute work into one labelled pass per step,
/// so Metal System Trace (which times encoders) shows each step. Slower than the default.
fn profile_passes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATMOSPHERE_PROFILE_PASSES").is_some())
}
