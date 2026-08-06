// Working-color-space conversion helpers.
//
// The scene-referred working color space is linear Rec.709 by default, and
// linear Rec.2020 when `WorkingColorSpace::Rec2020` is set on `RenderPlugin`.
// See `bevy_render::working_color_space` on the Rust side. Under Rec.2020 the
// pipeline cache gives every shader the `WORKING_COLOR_SPACE_REC2020` def.
// Call sites should gate these helpers behind that def, so default Rec.709
// projects compose byte-identically to shaders that predate this module.
//
// The matrices themselves are not gated. The display-encoding pass imports all
// four for its gamut stage, which is keyed by the display target rather than
// the working space.

// Linear Rec.709 -> Rec.2020 matrix (D65, per ITU-R BT.2087), derived in f64
// by the Lindbloom primaries-to-XYZ method. The f64 literals round to f32
// values bit-identical to `bevy_render::working_color_space::REC709_TO_REC2020`
// on the Rust side and to `GT7_REC_709_TO_REC_2020` in gt7.wgsl. Keep them in
// sync. Re-deriving from chromaticities (`bevy_color::rgb_to_rgb_matrix`) lands
// a few ULP away and is not a substitute.
const REC709_TO_REC2020 = mat3x3<f32>(
    0.627403895934699, 0.06909728935823199, 0.016391438875150228,   // column 0
    0.32928303837788375, 0.919540395075459, 0.08801330787722578,    // column 1
    0.043313065687417246, 0.011362315566309154, 0.895595253247624,  // column 2
);

// Inverse of the above (linear Rec.2020 -> Rec.709). Bit-identical to
// `bevy_render::working_color_space::REC2020_TO_REC709` on the Rust side.
const REC2020_TO_REC709 = mat3x3<f32>(
    1.6604910021084347, -0.12455047452159052, -0.01815076335490522, // column 0
    -0.5876411387885496, 1.1328998971259598, -0.10057889800800739,  // column 1
    -0.07284986331988484, -0.008349422604369487, 1.1187296613629125, // column 2
);

// Linear Rec.709 -> Display-P3 matrix (both D65, DCI-P3 primaries), derived by
// the same method. Bit-identical to
// `bevy_render::working_color_space::REC709_TO_DISPLAYP3` on the Rust side.
// The shared blue primary makes the third column's off-diagonals exactly zero.
const REC709_TO_DISPLAYP3 = mat3x3<f32>(
    0.822461968714363, 0.0331941988509616, 0.01708263072112,     // column 0
    0.177538031285638, 0.966805801149038, 0.0723974406639634,    // column 1
    0.0, 0.0, 0.910519928614916,                                 // column 2
);

// Linear Rec.2020 -> Display-P3 matrix (both D65). Display-P3 sits inside
// Rec.2020, so this contracts the gamut and can produce negative components.
// Bit-identical to `bevy_render::working_color_space::REC2020_TO_DISPLAYP3` on
// the Rust side.
const REC2020_TO_DISPLAYP3 = mat3x3<f32>(
    1.34357825258433, -0.0652974527891195, 0.00282178726170105,  // column 0
    -0.282179670526136, 1.07578791584857, -0.0195984945244942,   // column 1
    -0.0613985820581962, -0.010490463059455, 1.01677670726279,   // column 2
);

// Converts a linear Rec.709 color into the linear Rec.2020 working space.
// Negative components convert like any other value. Nothing is clamped.
fn rec709_to_rec2020(color: vec3<f32>) -> vec3<f32> {
    return REC709_TO_REC2020 * color;
}

// Converts a linear Rec.2020 working-space color to linear Rec.709.
// Colors outside the Rec.709 gamut come out with negative components. Callers
// feeding Rec.709-fit consumers (tone mapping operators, LUTs) must clamp.
fn rec2020_to_rec709(color: vec3<f32>) -> vec3<f32> {
    return REC2020_TO_REC709 * color;
}
