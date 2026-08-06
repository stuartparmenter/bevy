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

/// These live in `bevy_render` so camera extraction can pick the main-texture format from
/// the operator. The pass that uses them is here.
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
            // Packs the per-view `Gt7ParamsUniform`s that `queue_gt7_params_uniforms`
            // inserts. Views without one, every view in a default SDR project, leave the
            // buffer unallocated.
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
                    // Mutates `PipelineCache` through `block_on_render_pipeline`.
                    // Ambiguities against other pipeline-cache users are ignored, the
                    // same way the upscaling system ignores them
                    // (https://github.com/bevyengine/bevy/issues/14770).
                    prepare_view_tonemapping_pipelines
                        .in_set(RenderSystems::Prepare)
                        .ambiguous_with_all(),
                    // `prepare_view_tonemapping_pipelines`, one of the `Prepare` consumers
                    // of what this writes, is `ambiguous_with_all`, so an ambiguity report
                    // would not catch a missing edge here.
                    queue_gt7_params_uniforms.in_set(RenderSystems::Queue),
                ),
            );
    }
}

#[derive(Resource)]
pub struct TonemappingPipeline {
    /// The base tonemapping layout: view uniform, HDR source texture and sampler, and the
    /// tonemapping LUT (bindings 0-4). Used by every view that does not run the GT7
    /// operator.
    texture_bind_group: BindGroupLayoutDescriptor,
    /// [`Self::texture_bind_group`] plus the per-view [`Gt7ParamsUniform`] at binding 5.
    /// Used by pipelines specialized with
    /// [`TonemappingPipelineKeyFlags::GT7_PARAMS_UNIFORM`].
    gt7_params_bind_group: BindGroupLayoutDescriptor,
    sampler: Sampler,
    fullscreen_shader: FullscreenShader,
    fragment_shader: Handle<Shader>,
}

/// Render-world marker: the view's white-balance matrix
/// ([`ColorGradingUniform::balance`](bevy_render::view::ColorGradingUniform)) is composed
/// with an extra correction on the GPU, outside the static [`ColorGrading`] temperature
/// and tint values.
///
/// The tonemapping pass normally enables its `WHITE_BALANCE` shader def only when the
/// static temperature or tint is non-zero (see [`prepare_view_tonemapping_pipelines`]). A
/// GPU-side producer, such as `AutoWhiteBalance` in `bevy_post_process`, must insert this
/// marker on the render-world view entity (for example through its
/// [`ExtractComponent::Out`](bevy_render::extract_component::ExtractComponent) bundle) so
/// the shader path that reads the matrix stays compiled in when those values are zero.
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
        /// The per-view [`Gt7ParamsUniform`] is bound at binding 5 and the
        /// `GT7_PARAMS_UNIFORM` shader def is pushed, replacing the GT7 operator's
        /// baked SDR defaults with prepared per-camera values.
        ///
        /// Set from the presence of the view's [`Gt7ParamsUniform`] component, which
        /// [`queue_gt7_params_uniforms`] inserts on non-deferred views
        /// [`gt7_params_uniform_active`] holds for.
        const GT7_PARAMS_UNIFORM        = 0x10;
        /// Pushes the `TONEMAP_OUTPUT_REC2020` shader def. Set exactly when the view's
        /// [`ResolvedTonemapping::output_gamut`] is [`DisplayGamut::Rec2020`].
        const TONEMAP_OUTPUT_REC2020    = 0x80;
        /// The view composites in gamma-encoded sRGB space
        /// ([`CompositingSpace::Srgb`](bevy_camera::CompositingSpace::Srgb)). Main pass
        /// shaders write sRGB-encoded values, so this pass decodes to scene-linear
        /// before tone mapping and re-encodes the result, keeping the convention the
        /// upscaling blit expects (`SRGB_TO_LINEAR`). Pushes the
        /// `COMPOSITING_SPACE_SRGB` shader def.
        const SRGB_COMPOSITING          = 0x20;
        /// The view composites in Oklab space
        /// ([`CompositingSpace::Oklab`](bevy_camera::CompositingSpace::Oklab)). Like
        /// [`Self::SRGB_COMPOSITING`], but with the Oklab transforms (the blit's
        /// `OKLAB_TO_LINEAR` counterpart). Pushes the `COMPOSITING_SPACE_OKLAB` shader
        /// def.
        const OKLAB_COMPOSITING         = 0x40;
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
                shader_defs.push("TONEMAP_METHOD_NONE".into());
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

/// Binding index of the per-view [`Gt7ParamsUniform`] in the tonemapping bind group. Only
/// part of the layout when [`TonemappingPipelineKeyFlags::GT7_PARAMS_UNIFORM`] is set.
/// Pushed into `gt7.wesl` as a shader def so another bind group can use a different index.
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

    // The GT7 operator also binds its per-camera params uniform at binding 5. Kept
    // separate so every non-GT7 view uses the base layout with no extra bindings.
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

/// The specialized tonemapping pipeline of a view, plus the resolved values the tonemapping
/// node needs to run it.
///
/// Only views that run the tonemapping pass carry this. `Tonemapping::None` opt-outs, views
/// that fold tone mapping into their material shaders, and stack members whose finalizer
/// tone-maps the composed buffer do not, so the `tonemapping` node's `ViewQuery` skips them
/// and the pass records no GPU work.
#[derive(Component)]
pub struct ViewTonemappingPipeline {
    pipeline_id: CachedRenderPipelineId,
    /// The resolved operator ([`resolve_tonemapping`]) the pipeline runs. Selects the LUT
    /// and keys the node's bind-group cache.
    operator: Tonemapping,
    /// Whether the pipeline's layout binds the per-view [`Gt7ParamsUniform`]
    /// ([`TonemappingPipelineKeyFlags::GT7_PARAMS_UNIFORM`]).
    binds_gt7_params: bool,
}

/// Selects between the two tonemapping bind group layouts: the GT7 variant with the
/// per-view params uniform at binding 5, or the base layout. Pipeline specialization and
/// the tonemapping node both go through here.
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

/// A view's resolved tone-mapping decisions (see [`resolve_tonemapping`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ResolvedTonemapping {
    /// The operator the view's pipeline runs: the authored operator, or
    /// [`Tonemapping::GranTurismo7`] after the SDR-only substitution.
    pub operator: Tonemapping,
    /// The color primaries of the tonemapping pass output for the view.
    ///
    /// [`DisplayGamut::Rec2020`] exactly when [`Self::operator`] is
    /// [`Tonemapping::GranTurismo7`] and the view's resolved display target requests an
    /// HDR transfer. The pipeline is then specialized with
    /// [`TonemappingPipelineKeyFlags::TONEMAP_OUTPUT_REC2020`] and GT7 emits its native
    /// linear Rec.2020 display-referred output with no Rec.709 back-conversion (see
    /// `gt7.wesl`). Every other configuration emits Rec.709 display-linear. Under the
    /// Rec.2020 working space the Rec.709-fit operators get a Rec.2020 to Rec.709
    /// conversion at the pass entry (see `tonemapping.wesl`).
    pub output_gamut: DisplayGamut,
}

/// Resolves the tone-mapping decisions a view's pipeline runs, from its authored operator
/// and its resolved display target.
///
/// An SDR-only operator ([`Tonemapping::is_sdr_only`]) on a view whose display target
/// requests an HDR transfer ([`ViewDisplayTarget::is_hdr_transfer`]) is substituted with
/// [`Tonemapping::GranTurismo7`], which is fully algorithmic and so always available
/// without the `tonemapping_luts` feature.
///
/// Every other configuration keeps the authored operator: SDR views (including HDR requests
/// downgraded at surface negotiation), HDR views already using
/// [`Tonemapping::GranTurismo7`], [`Tonemapping::Linear`] (unbounded output), and
/// [`Tonemapping::None`] (a pass-through, not an SDR-only operator; the display encoder
/// warns about it separately). A missing operator resolves as [`Tonemapping::None`].
///
/// The output gamut is derived from the substituted operator in the same evaluation, and
/// the stack resolver derives the display encoder's source gamut
/// ([`ViewStackContract::source_gamut`](crate::camera_stack::ViewStackContract)) from
/// [`ResolvedTonemapping::output_gamut`], so the tonemapping and display-encoding pipelines
/// cannot disagree about the post-tonemap buffer's primaries. Pass the view's authored
/// operator, not a substituted one. The camera's [`Tonemapping`] component is never
/// mutated.
pub fn resolve_tonemapping(
    tonemapping: Option<&Tonemapping>,
    view_display_target: &ViewDisplayTarget,
) -> ResolvedTonemapping {
    let authored = *tonemapping.unwrap_or(&Tonemapping::None);
    let is_hdr_transfer = view_display_target.is_hdr_transfer();
    let operator = if authored.is_sdr_only() && is_hdr_transfer {
        Tonemapping::GranTurismo7
    } else {
        authored
    };
    ResolvedTonemapping {
        operator,
        output_gamut: if operator == Tonemapping::GranTurismo7 && is_hdr_transfer {
            DisplayGamut::Rec2020
        } else {
            DisplayGamut::Rec709
        },
    }
}

/// Whether a view's tonemapping pipeline binds the per-view [`Gt7ParamsUniform`]
/// ([`TonemappingPipelineKeyFlags::GT7_PARAMS_UNIFORM`]).
///
/// True when the view's resolved operator ([`resolve_tonemapping`]) is
/// [`Tonemapping::GranTurismo7`] and either:
///
/// * the camera opted in with a [`GranTurismo7Params`] component, or
/// * the view's resolved display target requests an HDR transfer
///   ([`ViewDisplayTarget::is_hdr_transfer`]). GT7's HDR mode is selected inside the
///   prepared uniform, with the peak taken from the display target (see
///   [`Gt7ParamsUniform::new`]), so every GT7 view on an HDR target needs one.
///
/// [`queue_gt7_params_uniforms`] is the only caller. It evaluates this once per view and
/// records the answer as the presence of the view's [`Gt7ParamsUniform`] component, which is
/// what [`prepare_view_tonemapping_pipelines`] keys the shader def and layout selection on,
/// so the pipeline layout and the bound buffer cannot disagree.
pub fn gt7_params_uniform_active(
    resolved: Tonemapping,
    has_params: bool,
    is_hdr_transfer: bool,
) -> bool {
    resolved == Tonemapping::GranTurismo7 && (has_params || is_hdr_transfer)
}

/// Derives a view's [`TonemappingPipelineKeyFlags`] from its color grading, resolved
/// compositing space, and resolved output gamut.
///
/// `compositing_space` must be the resolved space from the view's `ViewStackContract`, never
/// the camera's raw request: stack members share one main texture, so the decode and
/// re-encode flags have to match the one space the whole stack composites in.
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
    // Also kept compiled in when a GPU-side producer composes into the view's balance
    // matrix. See `ExternalWhiteBalance`.
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

    // `CompositingSpace::Linear` and no component both set neither flag, so scene-linear
    // views share one key. See the flag docs for what the encoded spaces need.
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
            Has<ViewTonemappingPipeline>,
            Has<TonemapInShader>,
        ),
        // `ViewStackContract` is overwritten in place and never removed, so a view whose
        // `ViewTarget` was dropped keeps a stale contract. This filter gates on liveness
        // and must stay even though no `ViewTarget` field is read here.
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
        has_view_tonemapping_pipeline,
        tonemap_in_shader,
    ) in view_targets.iter()
    {
        // Cameras stacked on a shared main texture tone-map once, on the stack's
        // finalizer, so earlier cameras' pixels are not tone-mapped again when the
        // finalizer's fullscreen pass runs over the composed buffer. Render-world entities
        // are retained, so the component has to be removed when a view joins a stack.
        if matches!(contract.tonemap, StackRole::Deferred(_)) {
            if has_view_tonemapping_pipeline {
                commands.entity(entity).remove::<ViewTonemappingPipeline>();
            }
            continue;
        }
        let requested_tonemapping = *tonemapping.unwrap_or(&Tonemapping::None);
        let resolved = resolve_tonemapping(tonemapping, view_display_target);
        if resolved.operator != requested_tonemapping {
            warn_once!(
                "A camera uses `Tonemapping::{requested_tonemapping:?}`, an SDR-only operator \
                whose output is capped at paper white, but renders to an HDR display target; \
                substituting `Tonemapping::GranTurismo7` for this view (using the camera's \
                `GranTurismo7Params` if present, otherwise the defaults). Set \
                `Tonemapping::GranTurismo7` on the camera explicitly to adopt the substitute \
                and silence this warning, or use an SDR display target to keep \
                `Tonemapping::{requested_tonemapping:?}`."
            );
        }

        if working_color_space.is_rec2020() && resolved.operator == Tonemapping::None {
            warn_once!(
                "A camera uses `Tonemapping::None` under `WorkingColorSpace::Rec2020`, so \
                nothing converts its Rec.2020 working colors back to the display gamut and \
                saturated colors come out desaturated (grayscale is unaffected). Use \
                `Tonemapping::Linear` to convert with no tone curve, or an operator like \
                `Tonemapping::GranTurismo7`."
            );
        }

        // `Tonemapping::None` views opt out of the pass and `TonemapInShader` views fold
        // tone mapping into their material shaders, so the node runs for neither. Don't
        // specialize and block on compiling a pipeline it never binds. Render-world
        // entities are retained, so a stale component has to be removed, but only if
        // present: default SDR views then issue no command.
        let no_pass = resolved.operator == Tonemapping::None || tonemap_in_shader;
        if no_pass {
            if has_view_tonemapping_pipeline {
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
            tonemapping: resolved.operator,
            flags,
        };
        let pipeline = pipelines.specialize(&pipeline_cache, &upscaling_pipeline, key);

        // The upscaling blit blocks on its own pipeline and presents whatever is in the
        // main texture, so an unready tonemapping pipeline would present raw scene-linear
        // frames at startup or after a key change. Block here too. This is O(1) once the
        // pipeline is compiled.
        pipeline_cache.block_on_render_pipeline(pipeline);

        commands.entity(entity).insert(ViewTonemappingPipeline {
            pipeline_id: pipeline,
            operator: resolved.operator,
            binds_gt7_params: has_gt7_params_uniform,
        });
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
        source_primaries: Default::default(),
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
        for operator in Tonemapping::ALL {
            let expected = !matches!(
                operator,
                Tonemapping::None | Tonemapping::Linear | Tonemapping::GranTurismo7
            );
            assert_eq!(
                operator.is_sdr_only(),
                expected,
                "is_sdr_only mismatch for {operator:?}"
            );
        }
    }

    #[test]
    fn sdr_only_operator_substitution_fires_exactly_for_sdr_only_operators_on_hdr_targets() {
        let hdr = hdr_view_display_target();
        for operator in Tonemapping::ALL {
            let resolved = resolve_tonemapping(Some(&operator), &hdr).operator;
            if operator.is_sdr_only() {
                // An SDR-only operator on an HDR transfer resolves to the GT7 substitute.
                assert_eq!(
                    resolved,
                    Tonemapping::GranTurismo7,
                    "expected substitution for {operator:?} on an HDR target"
                );
            } else {
                // `None`, `Linear`, and GT7 itself pass through unchanged.
                assert_eq!(resolved, operator);
            }
        }
    }

    #[test]
    fn sdr_only_operator_substitution_never_fires_off_hdr_targets() {
        for operator in Tonemapping::ALL {
            // Plain SDR target, including an HDR request downgraded at surface
            // negotiation.
            assert_eq!(
                resolve_tonemapping(Some(&operator), &sdr_view_display_target()).operator,
                operator
            );
        }
        // A missing operator is `None`, which is never substituted.
        assert_eq!(
            resolve_tonemapping(None, &hdr_view_display_target()).operator,
            Tonemapping::None
        );
    }

    #[test]
    fn output_gamut_matches_the_resolved_operator() {
        let hdr = hdr_view_display_target();
        for operator in Tonemapping::ALL {
            // On HDR targets the resolved operator is GT7 for everything except `None`
            // and `Linear`, so those two are the only Rec.709 outputs.
            let expected = if matches!(operator, Tonemapping::None | Tonemapping::Linear) {
                DisplayGamut::Rec709
            } else {
                DisplayGamut::Rec2020
            };
            assert_eq!(
                resolve_tonemapping(Some(&operator), &hdr).output_gamut,
                expected,
                "output gamut mismatch for {operator:?} on an HDR target"
            );
            // SDR targets, downgraded requests included, are always Rec.709.
            assert_eq!(
                resolve_tonemapping(Some(&operator), &sdr_view_display_target()).output_gamut,
                DisplayGamut::Rec709
            );
        }
    }

    #[test]
    fn gt7_params_uniform_active_table() {
        let hdr = hdr_view_display_target();
        let gt7 = Tonemapping::GranTurismo7;

        // On an SDR target, GT7 binds the uniform only if the camera opted in with
        // params. Cameras without the component keep the baked SDR defaults.
        assert!(gt7_params_uniform_active(gt7, true, false));
        assert!(!gt7_params_uniform_active(gt7, false, false));

        // On an HDR-transfer target, GT7 always binds it, with the camera's params if
        // present and the defaults otherwise.
        assert!(gt7_params_uniform_active(gt7, false, true));
        assert!(gt7_params_uniform_active(gt7, true, true));

        // Substituted views resolve to GT7 on an HDR target, so the arm above covers
        // them.
        for operator in Tonemapping::ALL {
            if !operator.is_sdr_only() {
                continue;
            }
            let resolved = resolve_tonemapping(Some(&operator), &hdr).operator;
            assert!(gt7_params_uniform_active(resolved, false, true));
            assert!(gt7_params_uniform_active(resolved, true, true));
        }

        // Non-GT7 resolved operators never bind it, params or not.
        for operator in Tonemapping::ALL {
            if operator == gt7 {
                continue;
            }
            assert!(!gt7_params_uniform_active(operator, false, false));
            assert!(!gt7_params_uniform_active(operator, true, true));
        }
    }

    /// A solo default camera (default grading, no resolved compositing space, plain SDR
    /// target) keys an empty flag set, whatever the authored operator.
    #[test]
    fn solo_sdr_default_keys_empty_flags() {
        for operator in Tonemapping::ALL {
            let resolved = resolve_tonemapping(Some(&operator), &sdr_view_display_target());
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
                "flags must be empty for {operator:?} on a plain SDR target"
            );
        }
    }

    /// The resolved compositing space sets exactly the matching decode and re-encode
    /// flag. `Some(Linear)` keys like no space.
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
