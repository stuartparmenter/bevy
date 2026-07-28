// Per-view display-target calibration uniform.
//
// Mirrors `DisplayTargetUniform` in
// `bevy_render::view::display_target_uniform` (Rust); the two must stay
// field-for-field in sync. Only the display-encoding pass binds and reads the
// uniform, and only on HDR-transfer targets; SDR pipelines never reference
// this module.

#define_import_path bevy_render::display_target

// The resolved calibration of the display a view is presented on.
//
// Luminance fields are in nits (cd/m²). `gamut` and `transfer` hold the
// `u32` indices documented on the Rust struct. Gamut conversion matrices are
// not part of this uniform; the gamut-transform pass derives them per
// pipeline.
struct DisplayTargetUniform {
    // Luminance of "paper white" (1.0 at the tone-map operator output), nits.
    paper_white_nits: f32,
    // Maximum luminance of the display, nits.
    peak_luminance_nits: f32,
    // Black level of the display, nits.
    min_luminance_nits: f32,
    // Display gamut as a u32 index.
    gamut: u32,
    // Resolved transfer function as a u32 index.
    transfer: u32,
}
