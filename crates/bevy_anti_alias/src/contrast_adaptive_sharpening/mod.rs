use crate::{fxaa::fxaa, smaa::smaa};
use bevy_app::prelude::*;
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer};
use bevy_camera::Camera;
use bevy_core_pipeline::{
    camera_stack::ViewStackContract,
    schedule::{Core2d, Core2dSystems, Core3d, Core3dSystems},
    FullscreenShader,
};
use bevy_ecs::{prelude::*, query::QueryItem};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    camera::ExtractedCamera,
    extract_component::{ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin},
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    renderer::RenderDevice,
    sync_component::SyncComponent,
    view::{ExtractedView, ViewTarget},
    Render, RenderApp, RenderStartup, RenderSystems,
};

mod node;

pub use node::cas;

/// Applies a contrast adaptive sharpening (CAS) filter to the camera.
///
/// CAS is usually used in combination with shader based anti-aliasing methods
/// such as FXAA or TAA to regain some of the lost detail from the blurring that they introduce.
///
/// CAS is designed to adjust the amount of sharpening applied to different areas of an image
/// based on the local contrast. This can help avoid over-sharpening areas with high contrast
/// and under-sharpening areas with low contrast.
///
/// To use this, add the [`ContrastAdaptiveSharpening`] component to a 2D or 3D camera.
#[derive(Component, Reflect, Clone)]
#[reflect(Component, Default, Clone)]
pub struct ContrastAdaptiveSharpening {
    /// Enable or disable sharpening.
    pub enabled: bool,
    /// Adjusts sharpening strength. Higher values increase the amount of sharpening.
    ///
    /// Clamped between 0.0 and 1.0.
    ///
    /// The default value is 0.6.
    pub sharpening_strength: f32,
    /// Whether to try and avoid sharpening areas that are already noisy.
    ///
    /// You probably shouldn't use this, and just leave it set to false.
    /// You should generally apply any sort of film grain or similar effects after CAS
    /// and upscaling to avoid artifacts.
    pub denoise: bool,
}

impl Default for ContrastAdaptiveSharpening {
    fn default() -> Self {
        ContrastAdaptiveSharpening {
            enabled: true,
            sharpening_strength: 0.6,
            denoise: false,
        }
    }
}

#[derive(Component, Default, Reflect, Clone)]
#[reflect(Component, Default, Clone)]
pub struct DenoiseCas(bool);

/// The uniform struct extracted from [`ContrastAdaptiveSharpening`] attached to a [`Camera`].
/// Will be available for use in the CAS shader.
#[doc(hidden)]
#[derive(Component, ShaderType, Clone)]
pub struct CasUniform {
    sharpness: f32,
}

impl SyncComponent<RenderApp> for ContrastAdaptiveSharpening {
    type Target = (DenoiseCas, CasUniform);
}

impl ExtractComponent<RenderApp> for ContrastAdaptiveSharpening {
    type QueryData = &'static Self;
    type QueryFilter = With<Camera>;
    type Out = (DenoiseCas, CasUniform);

    fn extract_component(item: QueryItem<Self::QueryData>) -> Option<Self::Out> {
        if !item.enabled || item.sharpening_strength == 0.0 {
            return None;
        }
        Some((
            DenoiseCas(item.denoise),
            CasUniform {
                // above 1.0 causes extreme artifacts and fireflies
                sharpness: item.sharpening_strength.clamp(0.0, 1.0),
            },
        ))
    }
}

/// Adds Support for Contrast Adaptive Sharpening (CAS).
#[derive(Default)]
pub struct CasPlugin;

impl Plugin for CasPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "robust_contrast_adaptive_sharpening.wesl");

        app.add_plugins((
            ExtractComponentPlugin::<ContrastAdaptiveSharpening>::default(),
            UniformComponentPlugin::<CasUniform>::default(),
        ));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_cas_pipeline)
            .add_systems(Render, prepare_cas_pipelines.in_set(RenderSystems::Prepare))
            .add_systems(
                Core3d,
                cas.after(fxaa)
                    .after(smaa)
                    .in_set(Core3dSystems::PostProcess),
            )
            .add_systems(
                Core2d,
                cas.after(fxaa)
                    .after(smaa)
                    .in_set(Core2dSystems::PostProcess),
            );
    }
}

#[derive(Resource)]
pub struct CasPipeline {
    layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
    variants: Variants<RenderPipeline, CasPipelineSpecializer>,
}

pub fn init_cas_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "sharpening_texture_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                // CAS Settings
                uniform_buffer::<CasUniform>(true),
            ),
        ),
    );

    let sampler = render_device.create_sampler(&SamplerDescriptor::default());

    let fragment_shader = load_embedded_asset!(
        asset_server.as_ref(),
        "robust_contrast_adaptive_sharpening.wesl"
    );

    let variants = Variants::new(
        CasPipelineSpecializer,
        RenderPipelineDescriptor {
            label: Some("contrast_adaptive_sharpening".into()),
            layout: vec![layout.clone()],
            vertex: fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: fragment_shader,
                ..Default::default()
            }),
            ..Default::default()
        },
    );

    commands.insert_resource(CasPipeline {
        layout,
        sampler,
        variants,
    });
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, SpecializerKey)]
pub struct CasPipelineKey {
    target_format: TextureFormat,
    denoise: bool,
    /// [`ViewStackContract::is_hdr_encode`]; selects the `HDR_DISPLAY_TARGET`
    /// shader path.
    hdr_encode: bool,
}

pub struct CasPipelineSpecializer;

impl Specializer<RenderPipeline> for CasPipelineSpecializer {
    type Key = CasPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        descriptor: &mut <RenderPipeline as Specializable>::Descriptor,
    ) -> Result<Canonical<Self::Key>, BevyError> {
        let fragment = descriptor.fragment_mut()?;

        if key.denoise {
            fragment.shader_defs.push("RCAS_DENOISE".into());
        }

        if key.hdr_encode {
            fragment.shader_defs.push("HDR_DISPLAY_TARGET".into());
        }

        fragment.set_target(
            0,
            ColorTargetState {
                format: key.target_format,
                blend: None,
                write_mask: ColorWrites::ALL,
            },
        );

        Ok(key)
    }
}

fn prepare_cas_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut sharpening_pipeline: ResMut<CasPipeline>,
    cameras: Query<
        (Entity, &ExtractedView, &DenoiseCas, &ViewStackContract),
        (
            With<ExtractedCamera>,
            With<ViewTarget>,
            Or<(Added<CasUniform>, Changed<DenoiseCas>)>,
        ),
    >,
    mut removals: RemovedComponents<CasUniform>,
) -> Result<(), BevyError> {
    for entity in removals.read() {
        commands.entity(entity).remove::<ViewCasPipeline>();
    }

    for (entity, view, denoise_cas, contract) in &cameras {
        let pipeline_id = sharpening_pipeline.variants.specialize(
            &pipeline_cache,
            CasPipelineKey {
                denoise: denoise_cas.0,
                target_format: view.target_format,
                hdr_encode: contract.is_hdr_encode(),
            },
        )?;

        commands.entity(entity).insert(ViewCasPipeline(pipeline_id));
    }

    Ok(())
}

#[derive(Component)]
pub struct ViewCasPipeline(CachedRenderPipelineId);

#[cfg(test)]
mod tests {
    //! Single-channel CPU mirrors of the RCAS math in
    //! `robust_contrast_adaptive_sharpening.wesl`, exact for grayscale
    //! neighborhoods, where the per-channel WGSL vector math collapses to the
    //! same scalars.

    const FSR_RCAS_LIMIT: f32 = 0.1875;
    const PEAK_C: (f32, f32) = (10.0, -40.0);
    /// The largest value an `Rgba16Float` target can store.
    const F16_MAX: f32 = 65504.0;

    fn compress(c: f32) -> f32 {
        let v = c.max(0.0);
        v / (1.0 + v)
    }

    fn decompress(c: f32) -> f32 {
        let v = c.clamp(0.0, 1.0);
        v / (1.0 - v).max(1.0 / F16_MAX)
    }

    /// The "Limiters" block of the fragment shader, for a grayscale cross
    /// neighborhood.
    fn rcas_lobe(b: f32, d: f32, f: f32, h: f32, sharpness: f32) -> f32 {
        let mn4 = b.min(d).min(f.min(h));
        let mx4 = b.max(d).max(f.max(h));
        let hit_min = mn4 / (4.0 * mx4);
        let hit_max = (PEAK_C.0 - mx4) / (PEAK_C.1 + 4.0 * mn4);
        let lobe_rgb = (-hit_min).max(hit_max);
        (-FSR_RCAS_LIMIT).max(lobe_rgb.min(0.0)) * sharpness
    }

    /// The full grayscale RCAS filter (def-less / SDR shape).
    fn rcas(b: f32, d: f32, e: f32, f: f32, h: f32, sharpness: f32) -> f32 {
        let lobe = rcas_lobe(b, d, f, h, sharpness);
        (lobe * b + lobe * d + lobe * f + lobe * h + e) / (4.0 * lobe + 1.0)
    }

    /// The `HDR_DISPLAY_TARGET` path: compress, RCAS, decompress, then bound
    /// overshoot by `max(local_max, 1.0)`.
    fn rcas_hdr(b: f32, d: f32, e: f32, f: f32, h: f32, sharpness: f32) -> f32 {
        let sharpened = decompress(rcas(
            compress(b),
            compress(d),
            compress(e),
            compress(f),
            compress(h),
            sharpness,
        ));
        let local_max = b.max(d).max(f).max(h).max(e);
        sharpened.min(local_max.max(1.0))
    }

    #[test]
    fn range_compression_round_trips() {
        for x in [0.0_f32, 0.001, 0.18, 0.5, 1.0, 2.5, 10.0, 100.0, 10000.0] {
            let round_tripped = decompress(compress(x));
            // scale tolerance quadratically with x because of precision limits
            let tolerance = (x * x * f32::EPSILON * 4.0).max(1e-7);
            assert!(
                (round_tripped - x).abs() <= tolerance,
                "round trip of {x} gave {round_tripped}"
            );
        }
    }

    #[test]
    fn range_compression_edge_cases() {
        // Negative inputs clamp to zero (monotonic, invertible domain).
        assert_eq!(compress(-5.0), 0.0);
        assert!(compress(F16_MAX) < 1.0);
        // Decompression saturates at the f16 maximum instead of infinity,
        // even for an out-of-contract input of exactly 1.0.
        assert_eq!(decompress(1.0), F16_MAX);
        assert!(decompress(2.0).is_finite());
    }

    #[test]
    fn hdr_path_is_bounded_and_finite_on_hdr_input() {
        // Neighborhoods that break the SDR math: flat and edged regions up to
        // 100x paper white.
        for peak in [2.5_f32, 4.0, 10.0, 16.0, 100.0] {
            for (b, d, e, f, h) in [
                (peak, peak, peak, peak, peak),
                (peak, 0.0, peak * 0.5, 0.0, peak),
                (0.0, peak, peak, peak, 0.0),
            ] {
                let lobe = rcas_lobe(compress(b), compress(d), compress(f), compress(h), 1.0);
                assert!(
                    lobe.is_finite() && (-FSR_RCAS_LIMIT..=0.0).contains(&lobe),
                    "compressed-space lobe out of contract: {lobe}"
                );
                let out = rcas_hdr(b, d, e, f, h, 1.0);
                assert!(out.is_finite(), "HDR RCAS produced a non-finite value");
                // RCAS limits its output to the neighborhood range. The
                // compression is monotonic, so that bound survives
                // decompression, modulo f32 rounding.
                let neighborhood_max = b.max(d).max(e).max(f).max(h);
                assert!(
                    (0.0..=neighborhood_max * (1.0 + 1e-4) + 1e-6).contains(&out),
                    "HDR RCAS output {out} escaped the neighborhood range [0, {neighborhood_max}]"
                );
            }
        }
    }

    #[test]
    fn hdr_path_sharpens_sdr_range_edges() {
        // On an in-range edge the HDR path still sharpens: it darkens a dark
        // center pixel surrounded by bright ones, like the SDR path does.
        let sdr = rcas(1.0, 1.0, 0.2, 1.0, 1.0, 1.0);
        let hdr = rcas_hdr(1.0, 1.0, 0.2, 1.0, 1.0, 1.0);
        assert!(sdr < 0.2);
        assert!(hdr < 0.2);
    }
}
