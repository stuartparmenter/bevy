//! Display encoding: the gamut-transform + transfer-encoding stage of the
//! separated display pipeline (tone map → gamut transform → transfer
//! encoding).
//!
//! The tone-mapping pass outputs *display-linear* color, scaled so `1.0` =
//! paper white, in per-view source primaries (Rec.709 for every effective
//! operator except `Tonemapping::GranTurismo7` on HDR targets — authored or
//! substituted for an SDR-only operator (`effective_tonemapping`) — which
//! emits its native Rec.2020; see `tonemap_output_gamut`); UI then composites in that
//! same space. This pass — scheduled after the UI pass and before the upscaling
//! blit — converts that buffer into the display's signal: a 3×3 gamut
//! transform from the source primaries to the display primaries, an
//! out-of-gamut handling step (ACES-RGC-style chroma compression when the
//! gamut transform can produce out-of-gamut colors — see
//! [`DisplayGamutCompression`] — plus an always-on `max(0)` safety clip), and
//! the display transfer function (OETF).
//!
//! **Plain SDR targets never run this pass.** For the default
//! [`DisplayTarget::SDR_SRGB`](bevy_window::DisplayTarget) (and any other
//! target whose transfer is [`DisplayTransfer::Srgb`]), the exact sRGB OETF is
//! applied for free by the hardware on the upscaling blit's `*UnormSrgb`
//! writeback; such views never receive a
//! [`ViewDisplayEncodingPipeline`], so the node's `ViewQuery` does not match
//! and the pass is skipped without touching the GPU. The shader-side transfer
//! functions (scRGB / PQ / extended-sRGB) activate only for HDR transfers.
//!
//! Surface negotiation (`create_surfaces` in `bevy_render::view::window`)
//! configures the swapchain this pass's output is presented through, using
//! wgpu's surface color-space API: an
//! `Rgba16Float` extended-sRGB-linear swapchain for
//! [`DisplayTransfer::ScRgbLinear`] (macOS/iOS Metal, Windows Vulkan/DX12,
//! Wayland Vulkan); an encoded extended-range sRGB swapchain (`ExtendedSrgb` /
//! `ExtendedDisplayP3`) for [`DisplayTransfer::ExtendedSrgb`] (Metal, Vulkan,
//! and browser WebGPU — the web HDR path); or an HDR10 swapchain (typically
//! `Rgb10a2Unorm`) for [`DisplayTransfer::Pq`] where the backend and OS
//! advertise it. In all cases the encoded output of this pass is blitted to
//! the surface unchanged (no hardware sRGB encode — these formats have no sRGB
//! view). When the backend cannot fulfil the requested transfer, the view's
//! **resolved** display target degrades (to plain SDR, with warnings), so the
//! predicate below always reflects what the surface can actually show.

use crate::FullscreenShader;
use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer, Handle};
use bevy_camera::CompositingSpace;
use bevy_color::LinearRgba;
use bevy_ecs::prelude::*;
use bevy_math::Vec3;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    renderer::RenderDevice,
    transfer_functions::{pq_inverse_eotf_from_nits, scrgb_encode, srgb_oetf_extended},
    view::{DisplayTargetUniform, ExtractedView, ViewTarget},
    working_color_space::{REC709_TO_DISPLAYP3, REC709_TO_REC2020},
    GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::Shader;
use bevy_utils::default;
use bevy_window::{DisplayGamut, DisplayTransfer};

use crate::camera_stack::{coerce_display_encode, ResolvedEncoding, StackRole, ViewStackContract};

#[cfg(test)]
mod gamut_compression;
mod node;

pub use node::display_encoding;

/// Adds the display-encoding pass (gamut transform + transfer encoding) used
/// by views whose resolved [`DisplayTarget`](bevy_window::DisplayTarget)
/// requests an HDR transfer function.
///
/// The `display_encoding` node itself is registered in the `Core2d` / `Core3d`
/// schedules by their plugins (after the UI pass, before upscaling).
pub struct DisplayEncodingPlugin;

impl Plugin for DisplayEncodingPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "display_encoding.wgsl");

        app.register_type::<DisplayGamutCompression>()
            .init_resource::<DisplayGamutCompression>()
            .add_plugins(ExtractResourcePlugin::<DisplayGamutCompression>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_gpu_resource::<SpecializedRenderPipelines<DisplayEncodingPipeline>>()
            .add_systems(RenderStartup, init_display_encoding_pipeline)
            .add_systems(
                Render,
                // Mutates `PipelineCache` (`block_on_render_pipeline`);
                // ordering ambiguities against other pipeline-cache users
                // are ignored, like the upscaling system
                // (see https://github.com/bevyengine/bevy/issues/14770).
                prepare_view_display_encoding_pipelines
                    .in_set(RenderSystems::Prepare)
                    .ambiguous_with_all(),
            );
    }
}

/// Controls how the display-encoding pass handles colors that fall outside
/// the display gamut after its gamut-transform stage.
///
/// The primary handling is a perceptual,
/// hue-approximate chroma compression toward the achromatic axis in the
/// style of the ACES 1.3 Reference Gamut Compression (Academy S-2020-001,
/// "RGC"; reference implementation `lib/RGC_common.ctl` in `aces-dev`) —
/// `gamut_compress` in `display_encoding.wgsl` documents the algorithm and
/// constants, and `gamut_compression.rs` next to it holds the CPU mirror and
/// its tests. The debug fallback is the
/// plain hue-shifting per-channel clip (`max(c, 0.0)`), which always runs
/// after the compression as a final safety (PQ encoding requires
/// non-negative input) and *is* the entire handling when compression is off.
///
/// # Why ACES RGC and not an exact hue-preserving `ICtCp` compression?
///
/// The ideal out-of-gamut strategy is hue-preserving compression in `ICtCp`.
/// An *exact* constant-hue mapping needs the distance
/// to the RGB gamut boundary along the chroma direction in `ICtCp`, which has
/// no closed form — production implementations (e.g. ACES 2.0's output
/// transform) iterate a chroma bisection per pixel through three matrix pairs
/// and six PQ evaluations per step. The ACES RGC is the published,
/// battle-tested cheap alternative: closed-form, monotonic, NaN-free,
/// exactly identity below the threshold, and *approximately* hue-preserving
/// (measured `ICtCp` hue drift in this implementation: ≈ 1–4.5° for moderately
/// out-of-gamut colors, ≈ 5–6° for the extreme Rec.2020 green/red primaries,
/// and ≈ 16° worst case for the Rec.2020 blue corner — see the fixture
/// tests in `gamut_compression.rs`; the per-channel clip it replaces drifts
/// substantially more and, unlike the compression, collapses distinct
/// out-of-gamut colors onto one another). A true `ICtCp` boundary search
/// would replace the `DISPLAY_GAMUT_COMPRESSION` shader path; the ACES RGC
/// is the cheap default.
///
/// # Reachable contractions
///
/// Out-of-gamut colors can only come out of the gamut stage when it
/// *contracts* — when the pass's input primaries are wider than the resolved
/// display primaries. The pass's input gamut is per-view (see
/// [`ViewStackContract::source_gamut`]): Rec.2020 when the buffer was
/// produced by `Tonemapping::GranTurismo7` on an HDR-transfer target —
/// authored, or substituted for an SDR-only operator
/// (`effective_tonemapping`) — as the operator emits its native Rec.2020
/// display-referred output; Rec.709 otherwise. Under
/// [`DisplayGamutCompression::Auto`] the compression is therefore active for
/// the two reachable contractions: GT7 HDR-native Rec.2020 input onto a
/// Rec.709-coordinate scRGB / extended-sRGB signal, and onto a Display-P3
/// extended-sRGB signal (Display-P3 ⊂ Rec.2020). Identity transforms
/// (Rec.709 → scRGB / extended-sRGB, GT7's Rec.2020 → PQ/Rec.2020) and the
/// Rec.709 → Rec.2020 PQ and Rec.709 → Display-P3 expansions keep the plain
/// clip, which is a no-op for their in-gamut-by-construction inputs.
/// [`DisplayGamutCompression::Always`] forces compression on for every
/// HDR-transfer view (e.g. to exercise or demonstrate the path).
#[derive(Resource, ExtractResource, Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
#[reflect(Resource, Debug, Default, Clone, PartialEq, Hash)]
#[extract_app(RenderApp)]
pub enum DisplayGamutCompression {
    /// Compress exactly when the gamut stage can produce out-of-gamut colors
    /// (a gamut contraction). Identity and expanding transforms keep the
    /// plain clip, which is a no-op for their in-gamut-by-construction
    /// inputs. This is the default.
    #[default]
    Auto,
    /// Always compress on views the display-encoding pass runs for.
    ///
    /// Compression is not free for in-gamut colors: channels whose
    /// distance from the achromatic axis exceeds the ACES RGC threshold
    /// (≈ 0.8) are pulled slightly inward to make room for the compressed
    /// out-of-gamut range, so forcing this on a path with no possible
    /// out-of-gamut input desaturates highly saturated colors for no
    /// benefit.
    Always,
    /// Debug fallback: replace the compression with the
    /// hue-shifting per-channel clip, for A/B comparison. Maps to
    /// [`OutOfGamutHandling::Clip`] — the same pipeline a non-contracting
    /// view gets under [`Self::Auto`], with the always-on `max(0)` safety
    /// clip as the entire handling.
    Clip,
}

/// The resolved per-pipeline out-of-gamut handling of the gamut stage, after
/// [`DisplayGamutCompression`] and the gamut-contraction rule have been
/// applied by [`prepare_view_display_encoding_pipelines`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum OutOfGamutHandling {
    /// Only the always-on final `max(c, 0.0)` safety clip; no shader def.
    /// Used when the gamut stage cannot produce out-of-gamut colors, and
    /// when [`DisplayGamutCompression::Clip`] selects the per-channel clip
    /// as the debug fallback.
    Clip,
    /// ACES-RGC-style chroma compression (`DISPLAY_GAMUT_COMPRESSION`),
    /// followed by the safety clip.
    Compress,
}

/// Whether transforming from `source` primaries to `display` primaries can
/// produce out-of-gamut (negative-component) colors, i.e. whether the source
/// gamut is strictly wider than the display gamut
/// (Rec.2020 ⊃ Display P3 ⊃ Rec.709).
const fn is_gamut_contraction(source: DisplayGamut, display: DisplayGamut) -> bool {
    const fn coverage_rank(gamut: DisplayGamut) -> u8 {
        match gamut {
            DisplayGamut::Rec709 => 0,
            DisplayGamut::DisplayP3 => 1,
            DisplayGamut::Rec2020 => 2,
        }
    }
    coverage_rank(source) > coverage_rank(display)
}

/// Encodes a Rec.709 display-linear, paper-white-relative clear color
/// (1.0 = paper white) into the signal values an HDR out texture stores:
/// gamut conversion to the resolved display primaries, paper-white luminance
/// scaling, and the resolved transfer encoding. This is the CPU mirror of
/// what the display-encoding pass ([`display_encoding.wgsl`]) does to rendered
/// pixels, applied to the [`LoadOp::Clear`]
/// value of the out texture so a finalizer's clear matches the encoded pixels
/// it composites over (a viewport/letterboxed region no blit covers would
/// otherwise present raw display-linear values as HDR signal).
///
/// The gamut stage is selected by [`ResolvedEncoding::gamut`] alone: Rec.709
/// is the identity (the authored clear is already Rec.709), Rec.2020 applies
/// [`REC709_TO_REC2020`], and Display-P3 applies [`REC709_TO_DISPLAYP3`]. The
/// transfer stage then encodes each channel: [`DisplayTransfer::ScRgbLinear`]
/// is [`scrgb_encode`] (sign-preserving, no clamp); [`DisplayTransfer::Pq`]
/// clamps non-negative (a negative base under the non-integer PQ exponent
/// would be `NaN`), scales to absolute nits, and runs
/// [`pq_inverse_eotf_from_nits`]; [`DisplayTransfer::ExtendedSrgb`]
/// scRGB-normalizes and runs the sign-preserving [`srgb_oetf_extended`] (no
/// `max(0)` clip). Alpha passes through unchanged for the alpha-blended
/// upscale path.
///
/// `paper_white_nits` must be
/// [`DisplayTarget::sanitized_paper_white_nits`](bevy_window::DisplayTarget::sanitized_paper_white_nits),
/// never the raw authored field: the GPU side folds the sanitized value, so
/// the encoded clear must use it too for the two to match on degenerate
/// authored paper whites.
///
/// [`WorkingColorSpace`](bevy_render::working_color_space::WorkingColorSpace)
/// is deliberately NOT consulted: the authored clear color is a display-referred
/// Rec.709 intent, not a scene-referred working-space buffer value, so it is
/// not subject to the working-space 709 -> 2020 conversion that scene colors
/// receive.
///
/// Only the reachable HDR transfers (scRGB, PQ, extended-sRGB) are handled;
/// [`DisplayTransfer::Srgb`] (hardware-encoded on the blit, no pass) is
/// `unreachable!`, the same contract as the encoder's
/// [`specialize`](DisplayEncodingPipeline::specialize).
///
/// [`display_encoding.wgsl`]: crate::display_encoding
pub fn encode_out_texture_clear_color(
    color: LinearRgba,
    encoding: &ResolvedEncoding,
    paper_white_nits: f32,
) -> LinearRgba {
    // A hand-built encoding that skipped the resolver's coercion chain could
    // pair a transfer with a gamut the encoder never produces (e.g. PQ with
    // Rec.709); fail loudly in debug rather than encoding a signal no surface
    // presents.
    debug_assert_eq!(
        coerce_display_encode(encoding.transfer, encoding.gamut),
        (encoding.transfer, encoding.gamut),
        "ResolvedEncoding must be a fixed point of the coercion chain"
    );

    // Value match, never an unconditional matrix multiply: the Rec.709 arm
    // must pass channels through bit-identically (an identity-`Mat3` multiply
    // flips `-0.0` to `+0.0` and turns non-finite channels into `NaN`), and
    // the encoded clear must match the encoded pixels bit-for-bit on the
    // identity-gamut paths.
    let authored = Vec3::new(color.red, color.green, color.blue);
    let linear = match encoding.gamut {
        DisplayGamut::Rec709 => authored,
        DisplayGamut::Rec2020 => REC709_TO_REC2020 * authored,
        DisplayGamut::DisplayP3 => REC709_TO_DISPLAYP3 * authored,
    };
    let rgb = match encoding.transfer {
        DisplayTransfer::ScRgbLinear => Vec3::new(
            scrgb_encode(linear.x, paper_white_nits),
            scrgb_encode(linear.y, paper_white_nits),
            scrgb_encode(linear.z, paper_white_nits),
        ),
        DisplayTransfer::Pq => Vec3::new(
            pq_inverse_eotf_from_nits(linear.x.max(0.0) * paper_white_nits),
            pq_inverse_eotf_from_nits(linear.y.max(0.0) * paper_white_nits),
            pq_inverse_eotf_from_nits(linear.z.max(0.0) * paper_white_nits),
        ),
        // The OETF is sign-preserving, so — like the shader — no `max(0)`
        // clip is applied.
        DisplayTransfer::ExtendedSrgb => Vec3::new(
            srgb_oetf_extended(scrgb_encode(linear.x, paper_white_nits)),
            srgb_oetf_extended(scrgb_encode(linear.y, paper_white_nits)),
            srgb_oetf_extended(scrgb_encode(linear.z, paper_white_nits)),
        ),
        // sRGB never reaches this helper (hardware encode on the blit). Same
        // contract as the encoder's specialize transfer arm.
        DisplayTransfer::Srgb => {
            unreachable!(
                "only HDR transfers (scRGB / PQ / extended-sRGB) encode an out-texture clear color"
            )
        }
    };
    LinearRgba {
        red: rgb.x,
        green: rgb.y,
        blue: rgb.z,
        alpha: color.alpha,
    }
}

/// Render-world resource holding the display-encoding pass's bind group
/// layout, sampler, and shaders.
#[derive(Resource)]
pub struct DisplayEncodingPipeline {
    layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
    fullscreen_shader: FullscreenShader,
    fragment_shader: Handle<Shader>,
}

/// Initializes [`DisplayEncodingPipeline`] at [`RenderStartup`].
pub fn init_display_encoding_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "display_encoding_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                sampler(SamplerBindingType::NonFiltering),
                // The per-view display-target calibration (paper white),
                // produced by `prepare_view_display_targets` (bevy_render);
                // this pass is its only binder.
                uniform_buffer::<DisplayTargetUniform>(true),
            ),
        ),
    );

    let sampler = render_device.create_sampler(&SamplerDescriptor::default());

    commands.insert_resource(DisplayEncodingPipeline {
        layout,
        sampler,
        fullscreen_shader: fullscreen_shader.clone(),
        fragment_shader: load_embedded_asset!(asset_server.as_ref(), "display_encoding.wgsl"),
    });
}

/// Specialization key for the display-encoding pipeline.
///
/// Every field except `target_format` comes from the view's
/// [`ViewStackContract`]: `gamut` and `transfer` are the **resolved** values
/// after the prepare-time coercions in the phase-2 stack resolver
/// ([`resolve_camera_stack_contracts`](crate::camera_stack::resolve_camera_stack_contracts);
/// PQ forces Rec.2020, scRGB forces Rec.709 — scRGB signals are by
/// definition expressed in extended Rec.709/sRGB coordinates — and extended-sRGB
/// keeps Rec.709 or Display P3 while coercing Rec.2020 to Rec.709), so the
/// pipeline hashes on what is actually encoded, not on what was requested. With
/// the per-view [`source_gamut`](Self::source_gamut), the reachable
/// source × display × transfer combinations are:
///
/// | source (tonemap output) | display gamut | transfer | gamut stage |
/// |---|---|---|---|
/// | Rec.709 | Rec.709 | scRGB / extended-sRGB | identity |
/// | Rec.709 | Rec.2020 | PQ | expansion (`DISPLAY_GAMUT_REC2020`) |
/// | Rec.709 | Display P3 | extended-sRGB | expansion (`DISPLAY_GAMUT_DISPLAYP3`) |
/// | Rec.2020 (GT7 HDR) | Rec.709 | scRGB / extended-sRGB | contraction (`GAMUT_REC2020_TO_REC709`, compression active under `Auto`) |
/// | Rec.2020 (GT7 HDR) | Rec.2020 | PQ | identity |
/// | Rec.2020 (GT7 HDR) | Display P3 | extended-sRGB | contraction (`GAMUT_REC2020_TO_DISPLAYP3`, compression active under `Auto`) |
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct DisplayEncodingPipelineKey {
    /// Format of the main texture the pass writes to (the
    /// [`post_process_write`](bevy_render::view::ViewTarget::post_process_write)
    /// destination).
    pub target_format: TextureFormat,
    /// The view's RESOLVED [`CompositingSpace`]
    /// ([`ViewStackContract::compositing_space`]), if any. `Some(Srgb)` /
    /// `Some(Oklab)` main textures hold encoded values that must be decoded
    /// back to linear before gamut/transfer encoding (the same decode the
    /// upscaling blit performs for non-encoded views, which this pass takes
    /// over).
    pub source_space: Option<CompositingSpace>,
    /// The color primaries of the pass's input
    /// ([`ViewStackContract::source_gamut`]): the tonemap output gamut of the
    /// buffer this view's encode reads — the stack's last tonemap-enabled
    /// member's for deferred encodes, this view's own for solo encodes (see
    /// [`tonemap_output_gamut`](crate::tonemapping::tonemap_output_gamut)).
    /// Rec.2020 when that operator is GT7 (authored or substituted) on an
    /// HDR-transfer target, Rec.709 otherwise.
    ///
    /// Post-tonemap UI converts its Rec.709-authored colors to `source_gamut`
    /// per view (the `OUTPUT_GAMUT_REC2020` writer-encode, keyed off
    /// [`ViewStackContract::source_gamut_is_rec2020`]), so saturated UI colors no
    /// longer oversaturate on a Rec.2020 (GT7) HDR view. Pre-tonemap writers (PBR
    /// meshes, 3D gizmos) instead convert off the global `WorkingColorSpace`, so
    /// they match `source_gamut` only when an operator marks the buffer Rec.2020
    /// (e.g. GT7 on an HDR target); a `Tonemapping::None` view leaves
    /// `source_gamut` Rec.709 while the buffer holds Rec.2020, which a tonemapping
    /// pass would otherwise reconcile. Emissive UI above paper white remains a
    /// follow-up (see `plans/ui-hdr-rfc.md`).
    pub source_gamut: DisplayGamut,
    /// The resolved display gamut the source color is transformed to.
    pub gamut: DisplayGamut,
    /// The resolved transfer function. Only HDR transfers occur here
    /// ([`DisplayTransfer::ScRgbLinear`], [`DisplayTransfer::Pq`], or
    /// [`DisplayTransfer::ExtendedSrgb`]); sRGB targets never get this pass.
    pub transfer: DisplayTransfer,
    /// The resolved out-of-gamut handling of the gamut stage (see
    /// [`DisplayGamutCompression`]). Under the default
    /// [`DisplayGamutCompression::Auto`] this is
    /// [`OutOfGamutHandling::Compress`] exactly when the gamut stage is a
    /// contraction (a Rec.2020 source onto the narrower Rec.709 scRGB /
    /// extended-sRGB or Display-P3 extended-sRGB signal) and
    /// [`OutOfGamutHandling::Clip`] otherwise (identity and expanding stages
    /// cannot produce out-of-gamut colors).
    pub out_of_gamut: OutOfGamutHandling,
}

impl SpecializedRenderPipeline for DisplayEncodingPipeline {
    type Key = DisplayEncodingPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();

        // Same def names (and semantics) as the upscaling blit, which skips
        // its own decode when this pass ran.
        match key.source_space {
            Some(CompositingSpace::Srgb) => shader_defs.push("SRGB_TO_LINEAR".into()),
            Some(CompositingSpace::Oklab) => shader_defs.push("OKLAB_TO_LINEAR".into()),
            Some(CompositingSpace::Linear) | None => {}
        }

        match (key.source_gamut, key.gamut) {
            // Identity transforms: a Rec.709 tonemap output onto a
            // Rec.709-coordinate scRGB / extended-sRGB signal, and GT7's
            // HDR-native Rec.2020 output onto a PQ/Rec.2020 signal.
            (DisplayGamut::Rec709, DisplayGamut::Rec709)
            | (DisplayGamut::Rec2020, DisplayGamut::Rec2020) => {}
            // Expansion (in-gamut by construction): Rec.709 tonemap output
            // onto a PQ/Rec.2020 signal.
            (DisplayGamut::Rec709, DisplayGamut::Rec2020) => {
                shader_defs.push("DISPLAY_GAMUT_REC2020".into());
            }
            // Contraction: GT7's HDR-native Rec.2020 output onto the
            // Rec.709-coordinate scRGB / extended-sRGB signal — for which
            // `Auto` keys in the compression below.
            (DisplayGamut::Rec2020, DisplayGamut::Rec709) => {
                shader_defs.push("GAMUT_REC2020_TO_REC709".into());
            }
            // Expansion (in-gamut by construction): Rec.709 tonemap output
            // onto an extended-Display-P3 signal.
            (DisplayGamut::Rec709, DisplayGamut::DisplayP3) => {
                shader_defs.push("DISPLAY_GAMUT_DISPLAYP3".into());
            }
            // Contraction: GT7's HDR-native Rec.2020 output onto an
            // extended-Display-P3 signal (Display-P3 ⊂ Rec.2020), for which
            // `Auto` keys in the compression below.
            (DisplayGamut::Rec2020, DisplayGamut::DisplayP3) => {
                shader_defs.push("GAMUT_REC2020_TO_DISPLAYP3".into());
            }
            // The tonemapping pass never emits a Display-P3 *source* gamut
            // (`tonemap_output_gamut` yields only Rec.709 or Rec.2020); the
            // display side is now a real gamut (handled above).
            (DisplayGamut::DisplayP3, _) => unreachable!(
                "the tonemapping pass never emits a DisplayP3 source gamut \
                 (tonemap_output_gamut yields only Rec709 or Rec2020)"
            ),
        }

        match key.transfer {
            DisplayTransfer::ScRgbLinear => shader_defs.push("DISPLAY_TRANSFER_SCRGB".into()),
            DisplayTransfer::Pq => shader_defs.push("DISPLAY_TRANSFER_PQ".into()),
            DisplayTransfer::ExtendedSrgb => {
                shader_defs.push("DISPLAY_TRANSFER_EXTENDED_SRGB".into());
            }
            // sRGB never reaches this pass (hardware encode on the blit).
            DisplayTransfer::Srgb => unreachable!(
                "only HDR transfers (scRGB / PQ / extended-sRGB) are encoded by the \
                 display-encoding pass"
            ),
        }

        match key.out_of_gamut {
            // The always-on max(0) safety clip is the entire handling.
            OutOfGamutHandling::Clip => {}
            OutOfGamutHandling::Compress => {
                shader_defs.push("DISPLAY_GAMUT_COMPRESSION".into());
            }
        }

        RenderPipelineDescriptor {
            label: Some("display_encoding pipeline".into()),
            layout: vec![self.layout.clone()],
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

/// The specialized display-encoding pipeline of a view.
///
/// Present **only** on views whose resolved display target requests an HDR
/// transfer function; for everything else (every view on a default SDR sRGB
/// target) the component is absent, so the `display_encoding` node's
/// `ViewQuery` does not match and the pass is skipped without recording any
/// GPU work.
#[derive(Component)]
pub struct ViewDisplayEncodingPipeline {
    pipeline_id: CachedRenderPipelineId,
}

/// Derives a view's [`DisplayEncodingPipelineKey`] from its
/// [`ViewStackContract`] and the global [`DisplayGamutCompression`] policy,
/// or `None` when the view runs no encode pass this frame: its stack's
/// resolved display target requests no HDR transfer (`encoding` is `None`),
/// or the pass is deferred to the stack's finalizer, which encodes the
/// composed buffer once.
fn display_encoding_key(
    target_format: TextureFormat,
    contract: &ViewStackContract,
    gamut_compression: DisplayGamutCompression,
) -> Option<DisplayEncodingPipelineKey> {
    let encoding = contract.encoding?;
    if matches!(contract.encode, StackRole::Deferred(_)) {
        return None;
    }
    let out_of_gamut = match gamut_compression {
        DisplayGamutCompression::Auto => {
            if is_gamut_contraction(contract.source_gamut, encoding.gamut) {
                OutOfGamutHandling::Compress
            } else {
                OutOfGamutHandling::Clip
            }
        }
        DisplayGamutCompression::Always => OutOfGamutHandling::Compress,
        DisplayGamutCompression::Clip => OutOfGamutHandling::Clip,
    };
    Some(DisplayEncodingPipelineKey {
        target_format,
        source_space: contract.compositing_space,
        source_gamut: contract.source_gamut,
        gamut: encoding.gamut,
        transfer: encoding.transfer,
        out_of_gamut,
    })
}

/// Specializes the display-encoding pipeline for views that need it and keeps
/// the [`ViewDisplayEncodingPipeline`] marker in sync (inserted for views
/// whose [`ViewStackContract`] carries resolved encode parameters and a
/// non-deferred encode role, removed otherwise).
///
/// Every key input but one — the resolved transfer and gamut (after the
/// coercion chain), and the pass's input gamut and compositing space — comes
/// from the contract resolved by
/// [`resolve_camera_stack_contracts`](crate::camera_stack::resolve_camera_stack_contracts),
/// which also owns the coercion and display-target diagnostics. The
/// out-of-gamut handling is resolved here from the [`DisplayGamutCompression`]
/// policy and the contract's source/display gamut pair, so a policy change
/// re-keys (and re-specializes) the pipeline on the next prepare.
pub fn prepare_view_display_encoding_pipelines(
    mut commands: Commands,
    mut pipeline_cache: ResMut<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<DisplayEncodingPipeline>>,
    encoding_pipeline: Res<DisplayEncodingPipeline>,
    gamut_compression: Res<DisplayGamutCompression>,
    views: Query<
        (
            Entity,
            &ExtractedView,
            &ViewStackContract,
            Has<ViewDisplayEncodingPipeline>,
        ),
        // `ViewStackContract` is overwritten in place and never removed, so a
        // view whose `ViewTarget` was dropped keeps a stale contract. This
        // filter is the liveness gate that makes stale contracts unreachable;
        // it must stay even though no `ViewTarget` field is read here.
        With<ViewTarget>,
    >,
) {
    for (entity, view, contract, has_pipeline) in &views {
        let Some(key) = display_encoding_key(view.target_format, contract, *gamut_compression)
        else {
            // Either an sRGB transfer (hardware encode on the upscaling
            // blit, no pass) or an encode deferred to the stack's finalizer,
            // which encodes the composed buffer; this view must not run the
            // pass. (Render-world entities are retained, so a stale
            // component must be actively removed — but only if present, so
            // plain SDR views issue no command.)
            if has_pipeline {
                commands
                    .entity(entity)
                    .remove::<ViewDisplayEncodingPipeline>();
            }
            continue;
        };

        let pipeline_id = pipelines.specialize(&pipeline_cache, &encoding_pipeline, key);

        // The pass-through upscaling blit for HDR transfers blocks on its
        // own pipeline and presents the main texture as-is, so an unready
        // encoder pipeline would present raw display-linear values — on a
        // PQ swapchain those read as severely distorted signal. Block until
        // the encoder is compiled; this is O(1) once it is, and only ever
        // runs for HDR-transfer views.
        pipeline_cache.block_on_render_pipeline(pipeline_id);

        commands
            .entity(entity)
            .insert(ViewDisplayEncodingPipeline { pipeline_id });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_stack::{resolve_contracts, ContractInput, ContractOutput, ResolvedEncoding};
    use crate::tonemapping::Tonemapping;
    use bevy_camera::CompositingSpace;

    fn entity(raw: u32) -> Entity {
        Entity::from_raw_u32(raw).unwrap()
    }

    /// A GT7 member on a PQ target that clears its target.
    fn gt7_clearing(raw: u32, index: usize) -> ContractInput<u32> {
        ContractInput {
            entity: entity(raw),
            texture: 0,
            sorted_index: index,
            composites_fullscreen: false,
            encode_enabled: true,
            output_writes: true,
            explicit_blend: false,
            tonemap_output_gamut: DisplayGamut::Rec2020,
            compositing_space: None,
            loads_previous: false,
            operator: Tonemapping::GranTurismo7,
        }
    }

    /// A `Tonemapping::None` fullscreen `ClearColorConfig::None` overlay on
    /// the same PQ target.
    fn passthrough_overlay(raw: u32, index: usize) -> ContractInput<u32> {
        ContractInput {
            entity: entity(raw),
            texture: 0,
            sorted_index: index,
            composites_fullscreen: true,
            encode_enabled: true,
            output_writes: true,
            explicit_blend: false,
            tonemap_output_gamut: DisplayGamut::Rec709,
            compositing_space: None,
            loads_previous: true,
            operator: Tonemapping::None,
        }
    }

    /// Builds the [`ViewStackContract`] the resolver's ECS layer inserts for
    /// one resolved view.
    fn contract(
        output: &ContractOutput,
        encoding: Option<(DisplayTransfer, DisplayGamut)>,
    ) -> ViewStackContract {
        ViewStackContract {
            tonemap: output.tonemap,
            encode: output.encode,
            blit: output.blit,
            compositing_space: output.compositing_space,
            source_gamut: output.source_gamut,
            encoding: encoding.map(|(transfer, gamut)| ResolvedEncoding { transfer, gamut }),
        }
    }

    /// Derives the pipeline key for one contract on the canonical
    /// `Rgba16Float` target under the default `DisplayGamutCompression::Auto`;
    /// the policy tests call [`display_encoding_key`] directly.
    fn auto_key(contract: &ViewStackContract) -> Option<DisplayEncodingPipelineKey> {
        display_encoding_key(
            TextureFormat::Rgba16Float,
            contract,
            DisplayGamutCompression::Auto,
        )
    }

    const PQ: Option<(DisplayTransfer, DisplayGamut)> =
        Some((DisplayTransfer::Pq, DisplayGamut::Rec2020));

    /// Canonical S1 repro: GT7 base + `Tonemapping::None` overlay on a PQ
    /// window. The overlay finalizes the encode for the composed buffer the
    /// BASE tone-mapped, so the key's source gamut is Rec.2020 (no 709->2020
    /// double expansion) and its source space is the resolved linear.
    #[test]
    fn s1_deferred_encode_keys_the_buffer_not_the_finalizer() {
        let outputs = resolve_contracts(vec![gt7_clearing(1, 0), passthrough_overlay(2, 1)]);

        // The deferring base runs no encode pass and derives no key.
        let base = contract(&outputs[&entity(1)], PQ);
        assert_eq!(auto_key(&base), None);

        let finalizer = contract(&outputs[&entity(2)], PQ);
        let key = auto_key(&finalizer).expect("the encode finalizer derives a key");
        assert_eq!(
            key,
            DisplayEncodingPipelineKey {
                target_format: TextureFormat::Rgba16Float,
                source_space: None,
                source_gamut: DisplayGamut::Rec2020,
                gamut: DisplayGamut::Rec2020,
                transfer: DisplayTransfer::Pq,
                out_of_gamut: OutOfGamutHandling::Clip,
            }
        );
    }

    /// Full S1 variant: the overlay carries an authored
    /// `CompositingSpace::Oklab` request, but phase 1 resolves the group to
    /// linear (the GT7 base is not a `Camera2d`), so the contract carries
    /// `None` and the key must not select the `OKLAB_TO_LINEAR` decode.
    #[test]
    fn s1_oklab_request_resolved_away_does_not_key_the_decode() {
        // `compositing_space` is the phase-1 RESOLVED value; the authored
        // Oklab request never reaches the contract.
        let outputs = resolve_contracts(vec![gt7_clearing(1, 0), passthrough_overlay(2, 1)]);
        let finalizer = contract(&outputs[&entity(2)], PQ);
        let key = auto_key(&finalizer).unwrap();
        assert_eq!(key.source_space, None);
    }

    /// A resolved compositing space passes through to the key verbatim.
    #[test]
    fn resolved_compositing_space_keys_the_decode() {
        let mut base = gt7_clearing(1, 0);
        base.compositing_space = Some(CompositingSpace::Srgb);
        let mut overlay = passthrough_overlay(2, 1);
        overlay.compositing_space = Some(CompositingSpace::Srgb);
        let outputs = resolve_contracts(vec![base, overlay]);
        let finalizer = contract(&outputs[&entity(2)], PQ);
        let key = auto_key(&finalizer).unwrap();
        assert_eq!(key.source_space, Some(CompositingSpace::Srgb));
    }

    /// Negative control: a solo GT7 camera on PQ keys exactly as before the
    /// contract port (its own operator IS the buffer's producer).
    #[test]
    fn solo_gt7_on_pq_keys_its_own_gamut() {
        let outputs = resolve_contracts(vec![gt7_clearing(1, 0)]);
        let solo = contract(&outputs[&entity(1)], PQ);
        let key = auto_key(&solo).unwrap();
        assert_eq!(
            key,
            DisplayEncodingPipelineKey {
                target_format: TextureFormat::Rgba16Float,
                source_space: None,
                source_gamut: DisplayGamut::Rec2020,
                gamut: DisplayGamut::Rec2020,
                transfer: DisplayTransfer::Pq,
                out_of_gamut: OutOfGamutHandling::Clip,
            }
        );
    }

    /// Negative control: a solo `Tonemapping::None` camera on PQ keys the
    /// Rec.709 source gamut (the 709->2020 expansion is correct there).
    #[test]
    fn solo_passthrough_on_pq_keys_rec709_source() {
        let outputs = resolve_contracts(vec![passthrough_overlay(1, 0)]);
        let solo = contract(&outputs[&entity(1)], PQ);
        let key = auto_key(&solo).unwrap();
        assert_eq!(key.source_gamut, DisplayGamut::Rec709);
        assert_eq!(key.gamut, DisplayGamut::Rec2020);
        assert_eq!(key.transfer, DisplayTransfer::Pq);
        assert_eq!(key.out_of_gamut, OutOfGamutHandling::Clip);
    }

    /// A contraction (GT7's Rec.2020 output onto an scRGB Rec.709 signal)
    /// keys the compression under the default `Auto` handling.
    #[test]
    fn gt7_onto_scrgb_keys_the_contraction_compression() {
        let outputs = resolve_contracts(vec![gt7_clearing(1, 0)]);
        let solo = contract(
            &outputs[&entity(1)],
            Some((DisplayTransfer::ScRgbLinear, DisplayGamut::Rec709)),
        );
        let key = auto_key(&solo).unwrap();
        assert_eq!(key.source_gamut, DisplayGamut::Rec2020);
        assert_eq!(key.gamut, DisplayGamut::Rec709);
        assert_eq!(key.out_of_gamut, OutOfGamutHandling::Compress);
    }

    /// A Rec.709 source onto an extended-Display-P3 signal is an expansion
    /// (Display-P3 ⊃ Rec.709): the gamut stage cannot produce out-of-gamut
    /// colors, so `Auto` keeps the plain clip.
    #[test]
    fn rec709_onto_extended_displayp3_keys_the_expansion() {
        let outputs = resolve_contracts(vec![passthrough_overlay(1, 0)]);
        let solo = contract(
            &outputs[&entity(1)],
            Some((DisplayTransfer::ExtendedSrgb, DisplayGamut::DisplayP3)),
        );
        let key = auto_key(&solo).unwrap();
        assert_eq!(key.source_gamut, DisplayGamut::Rec709);
        assert_eq!(key.gamut, DisplayGamut::DisplayP3);
        assert_eq!(key.transfer, DisplayTransfer::ExtendedSrgb);
        assert_eq!(key.out_of_gamut, OutOfGamutHandling::Clip);
    }

    /// GT7's Rec.2020 output onto an extended-Display-P3 signal is a
    /// contraction (Display-P3 ⊂ Rec.2020): `Auto` keys the compression.
    #[test]
    fn gt7_onto_extended_displayp3_keys_the_contraction_compression() {
        let outputs = resolve_contracts(vec![gt7_clearing(1, 0)]);
        let solo = contract(
            &outputs[&entity(1)],
            Some((DisplayTransfer::ExtendedSrgb, DisplayGamut::DisplayP3)),
        );
        let key = auto_key(&solo).unwrap();
        assert_eq!(key.source_gamut, DisplayGamut::Rec2020);
        assert_eq!(key.gamut, DisplayGamut::DisplayP3);
        assert_eq!(key.transfer, DisplayTransfer::ExtendedSrgb);
        assert_eq!(key.out_of_gamut, OutOfGamutHandling::Compress);
    }

    /// `DisplayGamutCompression::Always` keys the compression even on a
    /// non-contracting path (identity Rec.709 -> Rec.709 scRGB), where `Auto`
    /// keeps the plain clip.
    #[test]
    fn always_policy_keys_compression_on_a_non_contracting_path() {
        let outputs = resolve_contracts(vec![passthrough_overlay(1, 0)]);
        let solo = contract(
            &outputs[&entity(1)],
            Some((DisplayTransfer::ScRgbLinear, DisplayGamut::Rec709)),
        );
        let key = display_encoding_key(
            TextureFormat::Rgba16Float,
            &solo,
            DisplayGamutCompression::Always,
        )
        .unwrap();
        assert_eq!(key.source_gamut, DisplayGamut::Rec709);
        assert_eq!(key.gamut, DisplayGamut::Rec709);
        assert_eq!(key.out_of_gamut, OutOfGamutHandling::Compress);
    }

    /// `DisplayGamutCompression::Clip` keys the plain clip even on a
    /// contracting path (GT7's Rec.2020 onto an scRGB Rec.709 signal), where
    /// `Auto` keys the compression.
    #[test]
    fn clip_policy_keys_the_plain_clip_on_a_contracting_path() {
        let outputs = resolve_contracts(vec![gt7_clearing(1, 0)]);
        let solo = contract(
            &outputs[&entity(1)],
            Some((DisplayTransfer::ScRgbLinear, DisplayGamut::Rec709)),
        );
        let key = display_encoding_key(
            TextureFormat::Rgba16Float,
            &solo,
            DisplayGamutCompression::Clip,
        )
        .unwrap();
        assert_eq!(key.source_gamut, DisplayGamut::Rec2020);
        assert_eq!(key.gamut, DisplayGamut::Rec709);
        assert_eq!(key.out_of_gamut, OutOfGamutHandling::Clip);
    }

    /// SDR groups carry no encode parameters and derive no key.
    #[test]
    fn sdr_contract_derives_no_key() {
        let mut solo_input = gt7_clearing(1, 0);
        solo_input.encode_enabled = false;
        let outputs = resolve_contracts(vec![solo_input]);
        let solo = contract(&outputs[&entity(1)], None);
        assert_eq!(auto_key(&solo), None);
    }

    use bevy_render::transfer_functions::{
        pq_inverse_eotf_from_nits, scrgb_encode, srgb_oetf_extended,
    };
    use bevy_render::working_color_space::{REC709_TO_DISPLAYP3, REC709_TO_REC2020};
    use bevy_window::DisplayTarget;

    /// Builds a [`ResolvedEncoding`] with the given transfer at its canonical
    /// gamut (Rec.709 for scRGB and extended-sRGB, Rec.2020 for PQ) — see
    /// [`encoding_with_gamut`] for the extended-sRGB Display-P3 case.
    fn encoding(transfer: DisplayTransfer) -> ResolvedEncoding {
        ResolvedEncoding {
            transfer,
            gamut: match transfer {
                DisplayTransfer::ScRgbLinear | DisplayTransfer::ExtendedSrgb => {
                    DisplayGamut::Rec709
                }
                _ => DisplayGamut::Rec2020,
            },
        }
    }

    /// Builds a [`ResolvedEncoding`] with an explicit gamut.
    fn encoding_with_gamut(transfer: DisplayTransfer, gamut: DisplayGamut) -> ResolvedEncoding {
        ResolvedEncoding { transfer, gamut }
    }

    /// PQ white at paper-white 100 encodes each channel as
    /// `pq_inverse_eotf_from_nits(100.0)` (~0.5081).
    #[test]
    fn pq_white_at_paper_white_encodes_each_channel() {
        let out = encode_out_texture_clear_color(
            LinearRgba::WHITE,
            &encoding(DisplayTransfer::Pq),
            100.0,
        );
        // Rec.709 white maps to Rec.2020 white (the matrix rows sum to 1), so
        // every channel is 100 nits through PQ.
        let expected = pq_inverse_eotf_from_nits(100.0);
        assert!((expected - 0.5081).abs() < 1e-3, "{expected}");
        assert_eq!(out.red.to_bits(), expected.to_bits());
        assert_eq!(out.green.to_bits(), expected.to_bits());
        assert_eq!(out.blue.to_bits(), expected.to_bits());
    }

    /// PQ red gamut-converts through `REC709_TO_REC2020` before the per-channel
    /// transfer encode.
    #[test]
    fn pq_red_gamut_converts_before_encoding() {
        let out =
            encode_out_texture_clear_color(LinearRgba::RED, &encoding(DisplayTransfer::Pq), 100.0);
        let rec2020 = REC709_TO_REC2020 * Vec3::new(1.0, 0.0, 0.0);
        assert_eq!(
            out.red.to_bits(),
            pq_inverse_eotf_from_nits(rec2020.x.max(0.0) * 100.0).to_bits()
        );
        assert_eq!(
            out.green.to_bits(),
            pq_inverse_eotf_from_nits(rec2020.y.max(0.0) * 100.0).to_bits()
        );
        assert_eq!(
            out.blue.to_bits(),
            pq_inverse_eotf_from_nits(rec2020.z.max(0.0) * 100.0).to_bits()
        );
    }

    /// scRGB scales each channel by `paper_white / 80` (identity gamut).
    #[test]
    fn scrgb_scales_by_paper_white_over_80() {
        let color = LinearRgba::new(0.5, 0.25, 1.0, 1.0);
        let out =
            encode_out_texture_clear_color(color, &encoding(DisplayTransfer::ScRgbLinear), 100.0);
        assert_eq!(out.red.to_bits(), scrgb_encode(0.5, 100.0).to_bits());
        assert_eq!(out.green.to_bits(), scrgb_encode(0.25, 100.0).to_bits());
        assert_eq!(out.blue.to_bits(), scrgb_encode(1.0, 100.0).to_bits());
        // 100 / 80 = 1.25.
        assert_eq!(out.red.to_bits(), 0.625f32.to_bits());
    }

    /// Alpha passes through unchanged for both transfers.
    #[test]
    fn alpha_passes_through() {
        let color = LinearRgba::new(0.3, 0.6, 0.9, 0.42);
        assert_eq!(
            encode_out_texture_clear_color(color, &encoding(DisplayTransfer::Pq), 100.0).alpha,
            0.42
        );
        assert_eq!(
            encode_out_texture_clear_color(color, &encoding(DisplayTransfer::ScRgbLinear), 100.0)
                .alpha,
            0.42
        );
    }

    /// Negative channels clamp to zero before the PQ transfer (a negative base
    /// under the non-integer PQ exponent would be `NaN`); scRGB leaves them
    /// signed (the signal is unbounded).
    #[test]
    fn negative_channels_clamp_before_pq() {
        let color = LinearRgba::new(-0.5, 0.0, 1.0, 1.0);
        let pq = encode_out_texture_clear_color(color, &encoding(DisplayTransfer::Pq), 100.0);
        // After the 709 -> 2020 mix the red channel is still negative; it must
        // clamp to the encode of zero nits, never `NaN`.
        let rec2020 = REC709_TO_REC2020 * Vec3::new(-0.5, 0.0, 1.0);
        assert!(rec2020.x < 0.0);
        assert!(pq.red.is_finite());
        assert_eq!(pq.red.to_bits(), pq_inverse_eotf_from_nits(0.0).to_bits());

        // scRGB carries the negative through unclamped.
        let scrgb =
            encode_out_texture_clear_color(color, &encoding(DisplayTransfer::ScRgbLinear), 100.0);
        assert_eq!(scrgb.red.to_bits(), scrgb_encode(-0.5, 100.0).to_bits());
        assert!(scrgb.red < 0.0);
    }

    /// Extended-sRGB over Rec.709 (the `ExtendedSrgb` color space): identity
    /// gamut, then `srgb_oetf_extended(scrgb_encode(ch, pw))` per channel.
    #[test]
    fn extended_srgb_over_rec709_encodes_each_channel() {
        let color = LinearRgba::new(0.5, 0.25, 1.0, 1.0);
        let out =
            encode_out_texture_clear_color(color, &encoding(DisplayTransfer::ExtendedSrgb), 100.0);
        for (channel, value) in [(out.red, 0.5), (out.green, 0.25), (out.blue, 1.0)] {
            assert_eq!(
                channel.to_bits(),
                srgb_oetf_extended(scrgb_encode(value, 100.0)).to_bits()
            );
        }
        // 18% gray at an 80-nit paper white round-trips SDR: scrgb_encode is
        // identity there, so this is the plain sRGB encode (~0.4613).
        let gray = encode_out_texture_clear_color(
            LinearRgba::new(0.18, 0.18, 0.18, 1.0),
            &encoding(DisplayTransfer::ExtendedSrgb),
            80.0,
        );
        assert!((gray.red - 0.461_356).abs() < 1e-4);
    }

    /// Extended-sRGB over Display-P3 (the `ExtendedDisplayP3` color space):
    /// the authored Rec.709 clear gamut-converts through `REC709_TO_DISPLAYP3`
    /// before the per-channel OETF (mirrors `pq_red_gamut_converts_before_encoding`).
    #[test]
    fn extended_srgb_over_displayp3_gamut_converts_before_encoding() {
        let out = encode_out_texture_clear_color(
            LinearRgba::RED,
            &encoding_with_gamut(DisplayTransfer::ExtendedSrgb, DisplayGamut::DisplayP3),
            100.0,
        );
        let p3 = REC709_TO_DISPLAYP3 * Vec3::new(1.0, 0.0, 0.0);
        assert_eq!(
            out.red.to_bits(),
            srgb_oetf_extended(scrgb_encode(p3.x, 100.0)).to_bits()
        );
        assert_eq!(
            out.green.to_bits(),
            srgb_oetf_extended(scrgb_encode(p3.y, 100.0)).to_bits()
        );
        assert_eq!(
            out.blue.to_bits(),
            srgb_oetf_extended(scrgb_encode(p3.z, 100.0)).to_bits()
        );
        // The conversion is non-trivial (not identity): Rec.709 red spreads
        // into all three P3 channels (709 red sits inside the wider P3 gamut,
        // so this expansion stays positive — unlike a contraction it produces
        // no negatives), so the encoded green/blue are non-zero.
        assert!(p3.y > 0.0 && p3.z > 0.0);
        assert!(out.green > 0.0 && out.blue > 0.0);
    }

    /// Extended-sRGB leaves negatives signed (the OETF is sign-preserving),
    /// distinct from PQ's clamp.
    #[test]
    fn extended_srgb_is_sign_preserving() {
        let color = LinearRgba::new(-0.5, 0.0, 1.0, 1.0);
        let out =
            encode_out_texture_clear_color(color, &encoding(DisplayTransfer::ExtendedSrgb), 100.0);
        assert_eq!(
            out.red.to_bits(),
            srgb_oetf_extended(scrgb_encode(-0.5, 100.0)).to_bits()
        );
        assert!(out.red < 0.0);
        assert!(out.red.is_finite());
    }

    /// The Rec.709 gamut arm passes channels through bit-identically: a
    /// `-0.0` clear channel keeps its sign bit through the pure-multiply
    /// scRGB encode (an identity-matrix multiply would flip it to `+0.0`,
    /// `1.0 * -0.0 + 0.0 * g + 0.0 * b == +0.0`, and break bit-identity with
    /// the encoded pixels, which never see a matrix on the identity path).
    #[test]
    fn identity_gamut_preserves_the_sign_bit_of_negative_zero() {
        let color = LinearRgba::new(-0.0, 0.5, 0.25, 1.0);
        let scrgb =
            encode_out_texture_clear_color(color, &encoding(DisplayTransfer::ScRgbLinear), 80.0);
        assert_eq!(scrgb.red.to_bits(), (-0.0f32).to_bits());
    }

    /// Alpha passes through unchanged for the extended-sRGB transfer too.
    #[test]
    fn extended_srgb_alpha_passes_through() {
        let color = LinearRgba::new(0.3, 0.6, 0.9, 0.42);
        assert_eq!(
            encode_out_texture_clear_color(color, &encoding(DisplayTransfer::ExtendedSrgb), 100.0)
                .alpha,
            0.42
        );
    }

    /// An authored `paper_white_nits` of `0.0` sanitizes to 100 nits, so the
    /// caller (which passes `sanitized_paper_white_nits()`) encodes white as
    /// 100 nits rather than blacking out the clear.
    #[test]
    fn degenerate_paper_white_encodes_as_100_nits() {
        let sanitized = DisplayTarget {
            paper_white_nits: 0.0,
            ..DisplayTarget::SDR_SRGB
        }
        .sanitized_paper_white_nits();
        assert_eq!(sanitized, 100.0);

        let out = encode_out_texture_clear_color(
            LinearRgba::WHITE,
            &encoding(DisplayTransfer::Pq),
            sanitized,
        );
        assert_eq!(
            out.red.to_bits(),
            pq_inverse_eotf_from_nits(100.0).to_bits()
        );
    }
}
