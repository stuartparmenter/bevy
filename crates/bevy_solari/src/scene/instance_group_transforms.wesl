// Copies each instance of a raytracing instance group from the group's transforms into its slot,
// and the transform it copied last frame into the slot's previous-frame transform.

@group(0) @binding(0) var<storage, read> group_transforms: array<array<vec4<f32>, 3>>;
@group(0) @binding(1) var<storage, read_write> group_previous: array<array<vec4<f32>, 3>>;
@group(0) @binding(2) var<storage, read> slots: array<u32>;
@group(0) @binding(3) var<storage, read_write> transforms: array<array<vec4<f32>, 3>>;
@group(0) @binding(4) var<storage, read_write> previous_transforms: array<array<vec4<f32>, 3>>;

@compute @workgroup_size(64, 1, 1)
fn write_instance_group_transforms(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let instance = global_id.x;
    if instance >= arrayLength(&slots) {
        return;
    }
    let slot = slots[instance];
    if slot >= arrayLength(&transforms) || slot >= arrayLength(&previous_transforms) {
        return;
    }
    let transform = group_transforms[instance];
    previous_transforms[slot] = group_previous[instance];
    transforms[slot] = transform;
    group_previous[instance] = transform;
}
