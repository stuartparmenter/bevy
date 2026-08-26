use bevy_app::prelude::*;
use bevy_asset::{
    embedded_asset, load_embedded_asset, AssetServer, Assets, Handle, RenderAssetUsages,
};
use bevy_camera::CompositingSpace;
use bevy_ecs::prelude::*;
use bevy_image::{CompressedImageFormats, Image, ImageSampler, ImageType};
#[cfg(not(feature = "tonemapping_luts"))]
use bevy_log::error;
use bevy_log::warn_once;
use bevy_render::{
    camera::TonemapInShader,
    extract_component::{ExtractComponentPlugin, UniformComponentPlugin},
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{sampler, texture_2d, texture_3d, uniform_buffer},
        *,
    },
    renderer::RenderDevice,
    texture::{FallbackImage, GpuImage},
    view::{ColorGrading, ExtractedView, ViewDisplayTarget, ViewTarget, ViewUniform},
    working_color_space::WorkingColorSpace,
    GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::{load_shader_library, Shader, ShaderDefVal};
use bevy_window::DisplayGamut;
use bitflags::bitflags;

mod gt7;
mod node;

pub use bevy_render::view::{DebandDither, Tonemapping};
use bevy_utils::default;
pub use gt7::{
    queue_gt7_params_uniforms, GranTurismo7Params, Gt7ParamsUniform, GRAN_TURISMO_SDR_PAPER_WHITE,
    REFERENCE_LUMINANCE,
};
pub use node::tonemapping;

use crate::{
    camera_stack::{StackRole, ViewStackContract},
    FullscreenShader,
};

/// 3D LUT (look up table) textures used for tonemapping
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct TonemappingLuts {
    pub blender_filmic: Handle<Image>,
    pub agx: Handle<Image>,
    pub tony_mc_mapface: Handle<Image>,
}

pub struct TonemappingPlugin;

impl Plugin for TonemappingPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "lut_bindings.wesl");
        load_shader_library!(app, "gt7.wesl");

        embedded_asset!(app, "tonemapping_frag.wesl");

        if !app.world().is_resource_added::<TonemappingLuts>() {
            let mut images = app.world_mut().resource_mut::<Assets<Image>>();

            #[cfg(feature = "tonemapping_luts")]
            let tonemapping_luts = {
                TonemappingLuts {
                    blender_filmic: images.add(setup_tonemapping_lut_image(
                        include_bytes!("luts/Blender_-11_12.ktx2"),
                        ImageType::Extension("ktx2"),
                    )),
                    agx: images.add(setup_tonemapping_lut_image(
                        include_bytes!("luts/AgX-default_contrast.ktx2"),
                        ImageType::Extension("ktx2"),
                    )),
                    tony_mc_mapface: images.add(setup_tonemapping_lut_image(
                        include_bytes!("luts/tony_mc_mapface.ktx2"),
                        ImageType::Extension("ktx2"),
                    )),
                }
            };

            #[cfg(not(feature = "tonemapping_luts"))]
            let tonemapping_luts = {
                let placeholder = images.add(lut_placeholder());
                TonemappingLuts {
                    blender_filmic: placeholder.clone(),
                    agx: placeholder.clone(),
                    tony_mc_mapface: placeholder,
                }
            };

            app.insert_resource(tonemapping_luts);
        }

        app.add_plugins(ExtractResourcePlugin::<TonemappingLuts>::default());

        app.add_plugins((
            ExtractComponentPlugin::<Tonemapping>::default(),
            ExtractComponentPlugin::<DebandDither>::default(),
            ExtractComponentPlugin::<GranTurismo7Params>::default(),
            // Views without a `Gt7ParamsUniform`, every view in a default SDR project,
            // leave the buffer unallocated.
            UniformComponentPlugin::<Gt7ParamsUniform>::default(),
        ));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_gpu_resource::<SpecializedRenderPipelines<TonemappingPipeline>>()
            .add_systems(RenderStartup, init_tonemapping_pipeline)
            .add_systems(
                Render,
                (
                    // `block_on_render_pipeline` mutates `PipelineCache`. Ignore
                    // ambiguities against other pipeline-cache users, like the upscaling
                    // system does (https://github.com/bevyengine/bevy/issues/14770).
                    prepare_view_tonemapping_pipelines
                        .in_set(RenderSystems::Prepare)
                        .ambiguous_with_all(),
                    // `prepare_view_tonemapping_pipelines` consumes this and is
                    // `ambiguous_with_all`, so an ambiguity report cannot catch a missing
                    // edge here.
                    queue_gt7_params_uniforms.in_set(RenderSystems::Queue),
                ),
            );
    }
}

#[derive(Resource)]
pub struct TonemappingPipeline {
    /// View uniform, HDR source texture and sampler, and the tonemapping LUT
    /// (bindings 0-4). Used by pipelines that do not bind the GT7 params uniform.
    texture_bind_group: BindGroupLayoutDescriptor,
    /// [`Self::texture_bind_group`] plus the per-view [`Gt7ParamsUniform`] at binding 5,
    /// for pipelines with [`TonemappingPipelineKeyFlags::GT7_PARAMS_UNIFORM`].
    gt7_params_bind_group: BindGroupLayoutDescriptor,
    sampler: Sampler,
    fullscreen_shader: FullscreenShader,
    fragment_shader: Handle<Shader>,
}

/// Render-world marker: the view's white-balance matrix
/// ([`ColorGradingUniform::balance`](bevy_render::view::ColorGradingUniform)) gets an extra
/// correction on the GPU, outside the static [`ColorGrading`] temperature and tint values.
///
/// The pass enables its `WHITE_BALANCE` shader def only when the static temperature or tint
/// is non-zero. A GPU-side producer, such as `AutoWhiteBalance` in `bevy_post_process`, must
/// insert this marker on the render-world view entity to keep that shader path compiled in
/// when both values are zero.
#[derive(Component, Default, Clone, Copy, Debug)]
pub struct ExternalWhiteBalance;

bitflags! {
    /// Various flags describing what tonemapping needs to do.
    ///
    /// This allows the shader to skip unneeded steps.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct TonemappingPipelineKeyFlags: u8 {
        /// The hue needs to be changed.
        const HUE_ROTATE                = 0x01;
        /// The white balance needs to be adjusted.
        const WHITE_BALANCE             = 0x02;
        /// Saturation/contrast/gamma/gain/lift for one or more sections
        /// (shadows, midtones, highlights) need to be adjusted.
        const SECTIONAL_COLOR_GRADING   = 0x04;
        /// The view composites in gamma-encoded sRGB space.
        const SRGB_COMPOSITING          = 0x08;
        /// The view composites in Oklab space.
        const OKLAB_COMPOSITING         = 0x10;
        /// Binds the per-view [`Gt7ParamsUniform`] at binding 5 and pushes the
        /// `GT7_PARAMS_UNIFORM` shader def, replacing the GT7 method's baked SDR
        /// defaults with prepared per-camera values.
        const GT7_PARAMS_UNIFORM        = 0x20;
        /// Pushes the `TONEMAP_OUTPUT_REC2020` shader def. Set exactly when the view's
        /// [`ResolvedTonemapping::output_gamut`] is [`DisplayGamut::Rec2020`].
        const TONEMAP_OUTPUT_REC2020    = 0x80;
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TonemappingPipelineKey {
    target_format: TextureFormat,
    deband_dither: DebandDither,
    tonemapping: Tonemapping,
    flags: TonemappingPipelineKeyFlags,
}

impl SpecializedRenderPipeline for TonemappingPipeline {
    type Key = TonemappingPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();

        shader_defs.push(ShaderDefVal::UInt(
            "TONEMAPPING_LUT_TEXTURE_BINDING_INDEX".into(),
            3,
        ));
        shader_defs.push(ShaderDefVal::UInt(
            "TONEMAPPING_LUT_SAMPLER_BINDING_INDEX".into(),
            4,
        ));

        if let DebandDither::Enabled = key.deband_dither {
            shader_defs.push("DEBAND_DITHER".into());
        }

        // Define shader flags depending on the color grading options in use.
        if key.flags.contains(TonemappingPipelineKeyFlags::HUE_ROTATE) {
            shader_defs.push("HUE_ROTATE".into());
        }
        if key
            .flags
            .contains(TonemappingPipelineKeyFlags::WHITE_BALANCE)
        {
            shader_defs.push("WHITE_BALANCE".into());
        }
        if key
            .flags
            .contains(TonemappingPipelineKeyFlags::SECTIONAL_COLOR_GRADING)
        {
            shader_defs.push("SECTIONAL_COLOR_GRADING".into());
        }

        if key
            .flags
            .contains(TonemappingPipelineKeyFlags::SRGB_COMPOSITING)
        {
            shader_defs.push("COMPOSITING_SPACE_SRGB".into());
        }
        if key
            .flags
            .contains(TonemappingPipelineKeyFlags::OKLAB_COMPOSITING)
        {
            shader_defs.push("COMPOSITING_SPACE_OKLAB".into());
        }

        if key
            .flags
            .contains(TonemappingPipelineKeyFlags::GT7_PARAMS_UNIFORM)
        {
            shader_defs.push("GT7_PARAMS_UNIFORM".into());
            shader_defs.push(ShaderDefVal::UInt(
                "GT7_PARAMS_BINDING_INDEX".into(),
                GT7_PARAMS_BINDING_INDEX,
            ));
        }
        if key
            .flags
            .contains(TonemappingPipelineKeyFlags::TONEMAP_OUTPUT_REC2020)
        {
            shader_defs.push("TONEMAP_OUTPUT_REC2020".into());
        }

        match key.tonemapping {
            Tonemapping::None | Tonemapping::Linear => {
                shader_defs.push("TONEMAP_METHOD_LINEAR".into());
            }
            Tonemapping::Reinhard => shader_defs.push("TONEMAP_METHOD_REINHARD".into()),
            Tonemapping::ReinhardLuminance => {
                shader_defs.push("TONEMAP_METHOD_REINHARD_LUMINANCE".into());
            }
            Tonemapping::AcesFitted => shader_defs.push("TONEMAP_METHOD_ACES_FITTED".into()),
            Tonemapping::AgX => {
                #[cfg(not(feature = "tonemapping_luts"))]
                error!(
                    "AgX tonemapping requires the `tonemapping_luts` feature.
                    Either enable the `tonemapping_luts` feature for bevy in `Cargo.toml` (recommended),
                    or use a different `Tonemapping` method for your `Camera2d`/`Camera3d`."
                );
                shader_defs.push("TONEMAP_METHOD_AGX".into());
            }
            Tonemapping::SomewhatBoringDisplayTransform => {
                shader_defs.push("TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM".into());
            }
            Tonemapping::TonyMcMapface => {
                #[cfg(not(feature = "tonemapping_luts"))]
                error!(
                    "TonyMcMapFace tonemapping requires the `tonemapping_luts` feature.
                    Either enable the `tonemapping_luts` feature for bevy in `Cargo.toml` (recommended),
                    or use a different `Tonemapping` method for your `Camera2d`/`Camera3d`."
                );
                shader_defs.push("TONEMAP_METHOD_TONY_MC_MAPFACE".into());
            }
            Tonemapping::BlenderFilmic => {
                #[cfg(not(feature = "tonemapping_luts"))]
                error!(
                    "BlenderFilmic tonemapping requires the `tonemapping_luts` feature.
                    Either enable the `tonemapping_luts` feature for bevy in `Cargo.toml` (recommended),
                    or use a different `Tonemapping` method for your `Camera2d`/`Camera3d`."
                );
                shader_defs.push("TONEMAP_METHOD_BLENDER_FILMIC".into());
            }
            Tonemapping::KhronosPbrNeutral => shader_defs.push("TONEMAP_METHOD_PBR_NEUTRAL".into()),
            Tonemapping::GranTurismo7 => {
                shader_defs.push("TONEMAP_METHOD_GRAN_TURISMO_7".into());
            }
        }
        let bind_group_layout = gt7_layout(
            self,
            key.flags
                .contains(TonemappingPipelineKeyFlags::GT7_PARAMS_UNIFORM),
        )
        .clone();

        RenderPipelineDescriptor {
            label: Some("tonemapping pipeline".into()),
            layout: vec![bind_group_layout],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

/// Binding index of the per-view [`Gt7ParamsUniform`] in the tonemapping bind group. Pushed
/// into `gt7.wesl` as a shader def so another bind group can use a different index.
pub const GT7_PARAMS_BINDING_INDEX: u32 = 5;

pub fn init_tonemapping_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
) {
    let mut entries = DynamicBindGroupLayoutEntries::new_with_indices(
        ShaderStages::FRAGMENT,
        (
            (0, uniform_buffer::<ViewUniform>(true)),
            (
                1,
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
            (2, sampler(SamplerBindingType::NonFiltering)),
        ),
    );
    let lut_layout_entries = get_lut_bind_group_layout_entries();
    entries = entries.extend_with_indices(((3, lut_layout_entries[0]), (4, lut_layout_entries[1])));

    let tonemap_texture_bind_group =
        BindGroupLayoutDescriptor::new("tonemapping_hdr_texture_bind_group_layout", &entries);

    // Kept separate so views without the params uniform use the base layout unchanged.
    let gt7_params_entries = entries.extend_with_indices(((
        GT7_PARAMS_BINDING_INDEX,
        uniform_buffer::<Gt7ParamsUniform>(true),
    ),));
    let tonemap_gt7_params_bind_group = BindGroupLayoutDescriptor::new(
        "tonemapping_gt7_params_bind_group_layout",
        &gt7_params_entries,
    );

    let sampler = render_device.create_sampler(&SamplerDescriptor::default());

    commands.insert_resource(TonemappingPipeline {
        texture_bind_group: tonemap_texture_bind_group,
        gt7_params_bind_group: tonemap_gt7_params_bind_group,
        sampler,
        fullscreen_shader: fullscreen_shader.clone(),
        fragment_shader: load_embedded_asset!(asset_server.as_ref(), "tonemapping_frag.wesl"),
    });
}

/// A view's specialized tonemapping pipeline and the method it runs.
///
/// Only views that run the tonemapping pass carry this component, so the pass's
/// `ViewQuery` skips the views that opt out.
#[derive(Component)]
pub struct ViewTonemappingPipeline {
    pipeline_id: CachedRenderPipelineId,
    /// The tonemapping method the pipeline runs. Selects the LUT and keys the
    /// pass's bind-group cache.
    method: Tonemapping,
    binds_gt7_params: bool,
}

/// Pipeline specialization and the tonemapping pass both select the layout here.
fn gt7_layout(
    pipeline: &TonemappingPipeline,
    binds_gt7_params: bool,
) -> &BindGroupLayoutDescriptor {
    if binds_gt7_params {
        &pipeline.gt7_params_bind_group
    } else {
        &pipeline.texture_bind_group
    }
}

/// A view's resolved tonemapping decisions (see [`resolve_tonemapping`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ResolvedTonemapping {
    /// The method the view's pipeline runs: the authored method, or
    /// [`Tonemapping::GranTurismo7`] after the SDR-only substitution.
    pub method: Tonemapping,
    /// The color primaries of the tonemapping pass output for the view.
    ///
    /// [`DisplayGamut::Rec2020`] exactly when [`Self::method`] is
    /// [`Tonemapping::GranTurismo7`] and the view's resolved display target requests an HDR
    /// transfer. The pipeline is then specialized with
    /// [`TonemappingPipelineKeyFlags::TONEMAP_OUTPUT_REC2020`] and GT7 emits its native
    /// linear Rec.2020 display-referred output (see `gt7.wesl`). Every other configuration
    /// emits Rec.709 display-linear.
    pub output_gamut: DisplayGamut,
}

/// Resolves the tonemapping decisions a view's pipeline runs, from its authored method
/// and its resolved display target.
///
/// An SDR-only method ([`Tonemapping::is_sdr_only`]) on a view whose display target
/// requests an HDR transfer ([`ViewDisplayTarget::is_hdr_transfer`]) is substituted with
/// [`Tonemapping::GranTurismo7`], which is fully algorithmic and so always available
/// without the `tonemapping_luts` feature. Every other configuration keeps the authored
/// method. A missing method resolves as [`Tonemapping::None`].
///
/// The stack resolver derives the display encoder's source gamut
/// ([`ViewStackContract::source_gamut`](crate::camera_stack::ViewStackContract)) from
/// [`ResolvedTonemapping::output_gamut`], so the tonemapping and display-encoding pipelines
/// cannot disagree about the post-tonemap buffer's primaries. Pass the view's authored
/// method, not a substituted one. The camera's [`Tonemapping`] component is never
/// mutated.
pub fn resolve_tonemapping(
    tonemapping: Option<&Tonemapping>,
    view_display_target: &ViewDisplayTarget,
) -> ResolvedTonemapping {
    let authored = *tonemapping.unwrap_or(&Tonemapping::None);
    let is_hdr_transfer = view_display_target.is_hdr_transfer();
    let method = if authored.is_sdr_only() && is_hdr_transfer {
        Tonemapping::GranTurismo7
    } else {
        authored
    };
    ResolvedTonemapping {
        method,
        output_gamut: if method == Tonemapping::GranTurismo7 && is_hdr_transfer {
            DisplayGamut::Rec2020
        } else {
            DisplayGamut::Rec709
        },
    }
}

/// Whether a view's tonemapping pipeline binds the per-view [`Gt7ParamsUniform`]
/// ([`TonemappingPipelineKeyFlags::GT7_PARAMS_UNIFORM`]).
///
/// GT7's HDR mode is selected inside the prepared uniform, with the peak taken from the
/// display target (see [`Gt7ParamsUniform::new`]), so every GT7 view on an HDR-transfer
/// target needs one, even without a [`GranTurismo7Params`] component.
///
/// [`queue_gt7_params_uniforms`] is the only caller. It records the answer as the presence
/// of the view's [`Gt7ParamsUniform`] component, which is what
/// [`prepare_view_tonemapping_pipelines`] keys the shader def and layout selection on, so
/// the pipeline layout and the bound buffer cannot disagree.
pub fn gt7_params_uniform_active(
    resolved: Tonemapping,
    has_params: bool,
    is_hdr_transfer: bool,
) -> bool {
    resolved == Tonemapping::GranTurismo7 && (has_params || is_hdr_transfer)
}

/// Picks the pipeline flags for a view's color grading and compositing space. The sprite
/// and 2D mesh pipelines derive their encode shader defs from the same
/// `CompositingSpace`, so the pass decodes and re-encodes in the space the main texture
/// actually holds.
///
/// `compositing_space` must be the resolved space from the view's `ViewStackContract`.
/// Stack members share one main texture, so the decode and re-encode flags have to match
/// the one space the whole stack composites in.
///
/// `output_gamut` must be the view's [`ResolvedTonemapping::output_gamut`].
///
/// `has_gt7_params_uniform` must be the presence of the view's [`Gt7ParamsUniform`]
/// component, never a re-derivation of [`gt7_params_uniform_active`].
fn tonemapping_key_flags(
    color_grading: &ColorGrading,
    external_white_balance: bool,
    compositing_space: Option<CompositingSpace>,
    output_gamut: DisplayGamut,
    has_gt7_params_uniform: bool,
) -> TonemappingPipelineKeyFlags {
    // As an optimization, we omit parts of the shader that are unneeded.
    let mut flags = TonemappingPipelineKeyFlags::empty();
    flags.set(
        TonemappingPipelineKeyFlags::HUE_ROTATE,
        color_grading.global.hue != 0.0,
    );
    flags.set(
        TonemappingPipelineKeyFlags::WHITE_BALANCE,
        color_grading.global.temperature != 0.0
            || color_grading.global.tint != 0.0
            || external_white_balance,
    );
    flags.set(
        TonemappingPipelineKeyFlags::SECTIONAL_COLOR_GRADING,
        color_grading
            .all_sections()
            .any(|section| *section != default()),
    );

    // `CompositingSpace::Linear` and no component both set neither flag, so
    // scene-linear views share one key.
    flags.set(
        TonemappingPipelineKeyFlags::SRGB_COMPOSITING,
        compositing_space == Some(CompositingSpace::Srgb),
    );
    flags.set(
        TonemappingPipelineKeyFlags::OKLAB_COMPOSITING,
        compositing_space == Some(CompositingSpace::Oklab),
    );

    flags.set(
        TonemappingPipelineKeyFlags::GT7_PARAMS_UNIFORM,
        has_gt7_params_uniform,
    );
    flags.set(
        TonemappingPipelineKeyFlags::TONEMAP_OUTPUT_REC2020,
        output_gamut == DisplayGamut::Rec2020,
    );
    flags
}

pub fn prepare_view_tonemapping_pipelines(
    mut commands: Commands,
    mut pipeline_cache: ResMut<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<TonemappingPipeline>>,
    upscaling_pipeline: Res<TonemappingPipeline>,
    view_targets: Query<
        (
            Entity,
            &ExtractedView,
            &ViewStackContract,
            Option<&Tonemapping>,
            Option<&DebandDither>,
            &ViewDisplayTarget,
            Has<Gt7ParamsUniform>,
            Has<ExternalWhiteBalance>,
            Option<&ViewTonemappingPipeline>,
            Has<TonemapInShader>,
        ),
        // `ViewStackContract` is overwritten in place and never removed, so a view whose
        // `ViewTarget` was dropped keeps a stale contract. This filter must stay even
        // though no `ViewTarget` field is read here.
        With<ViewTarget>,
    >,
    working_color_space: Res<WorkingColorSpace>,
) {
    for (
        entity,
        view,
        contract,
        tonemapping,
        dither,
        view_display_target,
        has_gt7_params_uniform,
        external_white_balance,
        existing_pipeline,
        tonemap_in_shader,
    ) in view_targets.iter()
    {
        // A `HandledBy` view doesn't run the pass. The finalizer tonemaps
        // the shared buffer once. Render world entities are retained, so
        // remove a stale component.
        if matches!(contract.tonemap, StackRole::HandledBy(_)) {
            if existing_pipeline.is_some() {
                commands.entity(entity).remove::<ViewTonemappingPipeline>();
            }
            continue;
        }
        let requested_tonemapping = *tonemapping.unwrap_or(&Tonemapping::None);
        let resolved = resolve_tonemapping(tonemapping, view_display_target);
        if resolved.method != requested_tonemapping {
            warn_once!(
                "A camera uses `Tonemapping::{requested_tonemapping:?}`, an SDR-only method \
                whose output is capped at paper white, but renders to an HDR display target; \
                substituting `Tonemapping::GranTurismo7` for this view (using the camera's \
                `GranTurismo7Params` if present, otherwise the defaults). Set \
                `Tonemapping::GranTurismo7` on the camera explicitly to adopt the substitute \
                and silence this warning, or use an SDR display target to keep \
                `Tonemapping::{requested_tonemapping:?}`."
            );
        }

        if working_color_space.is_rec2020() && resolved.method == Tonemapping::None {
            warn_once!(
                "A camera uses `Tonemapping::None` under `WorkingColorSpace::Rec2020`, so \
                nothing converts its Rec.2020 working colors back to the display gamut and \
                saturated colors come out desaturated (grayscale is unaffected). Use \
                `Tonemapping::Linear` to convert with no tone curve, or a method like \
                `Tonemapping::GranTurismo7`."
            );
        }

        // `Tonemapping::None` views opt out of the pass and `TonemapInShader` views
        // tonemap in their material shaders, so neither runs the pass and neither
        // needs a pipeline. Render world entities are retained, so remove a stale
        // component.
        if resolved.method == Tonemapping::None || tonemap_in_shader {
            if existing_pipeline.is_some() {
                commands.entity(entity).remove::<ViewTonemappingPipeline>();
            }
            continue;
        }

        let flags = tonemapping_key_flags(
            &view.color_grading,
            external_white_balance,
            contract.compositing_space,
            resolved.output_gamut,
            has_gt7_params_uniform,
        );

        let key = TonemappingPipelineKey {
            target_format: view.target_format,
            deband_dither: *dither.unwrap_or(&DebandDither::Disabled),
            tonemapping: resolved.method,
            flags,
        };
        let pipeline = pipelines.specialize(&pipeline_cache, &upscaling_pipeline, key);

        // The upscaling blit blocks on its own pipeline and writes whatever is in the
        // main texture to the out texture, so an unready tonemapping pipeline would
        // show raw scene-linear frames. Block here too. This is O(1) once the pipeline
        // is compiled.
        pipeline_cache.block_on_render_pipeline(pipeline);

        // The key determines the pipeline id, so an unchanged id means an
        // unchanged component.
        if existing_pipeline.is_none_or(|existing| existing.pipeline_id != pipeline) {
            commands.entity(entity).insert(ViewTonemappingPipeline {
                pipeline_id: pipeline,
                method: resolved.method,
                binds_gt7_params: has_gt7_params_uniform,
            });
        }
    }
}

pub fn get_lut_bindings<'a>(
    images: &'a RenderAssets<GpuImage>,
    tonemapping_luts: &'a TonemappingLuts,
    tonemapping: &Tonemapping,
    fallback_image: &'a FallbackImage,
) -> (&'a TextureView, &'a Sampler) {
    let image = match tonemapping {
        // AgX lut texture used when tonemapping doesn't need a texture since it's very small (32x32x32)
        Tonemapping::None
        | Tonemapping::Linear
        | Tonemapping::Reinhard
        | Tonemapping::ReinhardLuminance
        | Tonemapping::AcesFitted
        | Tonemapping::AgX
        | Tonemapping::KhronosPbrNeutral
        | Tonemapping::SomewhatBoringDisplayTransform
        | Tonemapping::GranTurismo7 => &tonemapping_luts.agx,
        Tonemapping::TonyMcMapface => &tonemapping_luts.tony_mc_mapface,
        Tonemapping::BlenderFilmic => &tonemapping_luts.blender_filmic,
    };
    let lut_image = images.get(image).unwrap_or(&fallback_image.d3);
    (&lut_image.texture_view, &lut_image.sampler)
}

pub fn get_lut_bind_group_layout_entries() -> [BindGroupLayoutEntryBuilder; 2] {
    [
        texture_3d(TextureSampleType::Float { filterable: true }),
        sampler(SamplerBindingType::Filtering),
    ]
}

#[expect(clippy::allow_attributes, reason = "`dead_code` is not always linted.")]
#[allow(
    dead_code,
    reason = "There is unused code when the `tonemapping_luts` feature is disabled."
)]
fn setup_tonemapping_lut_image(bytes: &[u8], image_type: ImageType) -> Image {
    let image_sampler = ImageSampler::Descriptor(bevy_image::ImageSamplerDescriptor {
        label: Some("Tonemapping LUT sampler".to_string()),
        address_mode_u: bevy_image::ImageAddressMode::ClampToEdge,
        address_mode_v: bevy_image::ImageAddressMode::ClampToEdge,
        address_mode_w: bevy_image::ImageAddressMode::ClampToEdge,
        mag_filter: bevy_image::ImageFilterMode::Linear,
        min_filter: bevy_image::ImageFilterMode::Linear,
        mipmap_filter: bevy_image::ImageFilterMode::Linear,
        ..default()
    });
    Image::from_buffer(
        bytes,
        image_type,
        CompressedImageFormats::NONE,
        false,
        image_sampler,
        // LUT must be kept in main world for render recovery reasons
        RenderAssetUsages::default(),
        None,
    )
    .unwrap()
}

pub fn lut_placeholder() -> Image {
    let format = TextureFormat::Rgba8Unorm;
    let data = vec![255, 0, 255, 255];
    Image {
        data: Some(data),
        data_order: TextureDataOrder::default(),
        texture_descriptor: TextureDescriptor {
            size: Extent3d::default(),
            format,
            dimension: TextureDimension::D3,
            label: None,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        sampler: ImageSampler::Default,
        texture_view_descriptor: None,
        asset_usage: RenderAssetUsages::RENDER_WORLD,
        copy_on_resize: false,
        source_color_primaries: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_window::{DisplayTarget, DisplayTransfer};

    fn hdr_view_display_target() -> ViewDisplayTarget {
        ViewDisplayTarget(DisplayTarget {
            paper_white_nits: 200.0,
            peak_luminance_nits: 1000.0,
            transfer: DisplayTransfer::ScRgbLinear,
            ..DisplayTarget::SDR_SRGB
        })
    }

    fn sdr_view_display_target() -> ViewDisplayTarget {
        ViewDisplayTarget(DisplayTarget::SDR_SRGB)
    }

    #[test]
    fn sdr_only_excludes_exactly_none_linear_and_gran_turismo_7() {
        for method in Tonemapping::ALL {
            let expected = !matches!(
                method,
                Tonemapping::None | Tonemapping::Linear | Tonemapping::GranTurismo7
            );
            assert_eq!(
                method.is_sdr_only(),
                expected,
                "is_sdr_only mismatch for {method:?}"
            );
        }
    }

    #[test]
    fn sdr_only_method_substitution_fires_exactly_for_sdr_only_methods_on_hdr_targets() {
        let hdr = hdr_view_display_target();
        for method in Tonemapping::ALL {
            let resolved = resolve_tonemapping(Some(&method), &hdr).method;
            if method.is_sdr_only() {
                assert_eq!(
                    resolved,
                    Tonemapping::GranTurismo7,
                    "expected substitution for {method:?} on an HDR target"
                );
            } else {
                assert_eq!(resolved, method);
            }
        }
    }

    #[test]
    fn sdr_only_method_substitution_never_fires_off_hdr_targets() {
        for method in Tonemapping::ALL {
            assert_eq!(
                resolve_tonemapping(Some(&method), &sdr_view_display_target()).method,
                method
            );
        }
        assert_eq!(
            resolve_tonemapping(None, &hdr_view_display_target()).method,
            Tonemapping::None
        );
    }

    #[test]
    fn output_gamut_matches_the_resolved_method() {
        let hdr = hdr_view_display_target();
        for method in Tonemapping::ALL {
            let expected = if matches!(method, Tonemapping::None | Tonemapping::Linear) {
                DisplayGamut::Rec709
            } else {
                DisplayGamut::Rec2020
            };
            assert_eq!(
                resolve_tonemapping(Some(&method), &hdr).output_gamut,
                expected,
                "output gamut mismatch for {method:?} on an HDR target"
            );
            assert_eq!(
                resolve_tonemapping(Some(&method), &sdr_view_display_target()).output_gamut,
                DisplayGamut::Rec709
            );
        }
    }

    #[test]
    fn gt7_params_uniform_active_table() {
        let hdr = hdr_view_display_target();
        let gt7 = Tonemapping::GranTurismo7;

        // On an SDR target, GT7 binds the uniform only if the camera opted in with params.
        assert!(gt7_params_uniform_active(gt7, true, false));
        assert!(!gt7_params_uniform_active(gt7, false, false));

        // On an HDR-transfer target, GT7 always binds it.
        assert!(gt7_params_uniform_active(gt7, false, true));
        assert!(gt7_params_uniform_active(gt7, true, true));

        // Substituted views resolve to GT7, so the arm above covers them.
        for method in Tonemapping::ALL {
            if !method.is_sdr_only() {
                continue;
            }
            let resolved = resolve_tonemapping(Some(&method), &hdr).method;
            assert!(gt7_params_uniform_active(resolved, false, true));
            assert!(gt7_params_uniform_active(resolved, true, true));
        }

        // Non-GT7 resolved methods never bind it, params or not.
        for method in Tonemapping::ALL {
            if method == gt7 {
                continue;
            }
            assert!(!gt7_params_uniform_active(method, false, false));
            assert!(!gt7_params_uniform_active(method, true, true));
        }
    }

    #[test]
    fn solo_sdr_default_keys_empty_flags() {
        for method in Tonemapping::ALL {
            let resolved = resolve_tonemapping(Some(&method), &sdr_view_display_target());
            let flags = tonemapping_key_flags(
                &ColorGrading::default(),
                false,
                None,
                resolved.output_gamut,
                false,
            );
            assert_eq!(
                flags,
                TonemappingPipelineKeyFlags::empty(),
                "flags must be empty for {method:?} on a plain SDR target"
            );
        }
    }

    #[test]
    fn resolved_compositing_space_sets_exactly_its_flag() {
        let flags_for = |space: Option<CompositingSpace>| {
            tonemapping_key_flags(
                &ColorGrading::default(),
                false,
                space,
                DisplayGamut::Rec709,
                false,
            )
        };
        assert_eq!(
            flags_for(Some(CompositingSpace::Oklab)),
            TonemappingPipelineKeyFlags::OKLAB_COMPOSITING
        );
        assert_eq!(
            flags_for(Some(CompositingSpace::Srgb)),
            TonemappingPipelineKeyFlags::SRGB_COMPOSITING
        );
        assert_eq!(
            flags_for(Some(CompositingSpace::Linear)),
            TonemappingPipelineKeyFlags::empty()
        );
        assert_eq!(flags_for(None), TonemappingPipelineKeyFlags::empty());
    }
}
