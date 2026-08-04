// Per-view display-target calibration uniform.
//
// Mirrors `DisplayTargetUniform` in
// `bevy_render::view::display_target_uniform` on the Rust side. The two must
// stay field-for-field in sync. Only the display-encoding pass binds and reads
// the uniform, and only on HDR-transfer targets.

#define_import_path bevy_render::display_target

// The resolved calibration of the display a view is presented on.
//
// Paper white is the only calibration value read at runtime. The display gamut
// and transfer select compile-time shader defs in the display-encoding
// pipeline instead.
struct DisplayTargetUniform {
    // Luminance of "paper white" (1.0 at the tone-map operator output), nits.
    paper_white_nits: f32,
}
