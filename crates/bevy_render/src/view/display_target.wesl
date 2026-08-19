// Mirrors `DisplayTargetUniform` in `bevy_render::view::display_target_uniform`.
// The two must stay field-for-field in sync. Only the display-encoding pass
// binds this uniform, and only on HDR-transfer targets.
struct DisplayTargetUniform {
    // Luminance of "paper white" (1.0 at the tone-map operator output), nits.
    paper_white_nits: f32,
}
