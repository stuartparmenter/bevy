import package::realtime::grass_receiver_geometry::{Receiver, match_receiver};

//! Deliberately unfiltered diagnostics. Receiver identity is inferred from a
//! closest camera ray agreeing with raster position and triangle normal; this
//! cannot distinguish coincident blades without a raster primitive-ID attachment.
import package::realtime::gbuffer_utils::{ResolvedGPixel, gpixel_resolve};
import package::realtime::bindings::{constants, view, depth_buffer, gbuffer};
import package::scene::bindings::{light_sources, directional_lights, LIGHT_SOURCE_KIND_DIRECTIONAL, geometry_ids, trace_ray, offset_surface_ray_origin, offset_triangle_ray_origin, RAY_T_MIN, RAY_T_MAX};
import package::scene::sampling::{trace_visibility, trace_visibility_from_surface, resolve_light_sample, LightSample};

enable wgpu_ray_query;

const UNMATCHED = vec3(1.0, 0.0, 1.0);
const SAME_TRIANGLE = vec3(1.0, 0.0, 0.0);
const SAME_BLADE = vec3(1.0, 0.35, 0.0);
const OTHER_GRASS = vec3(0.0, 0.3, 1.0);
const SOLID_BLOCKER = vec3(0.08);


struct Probe {
    receiver: Receiver,
    blocked: bool,
    blocker_instance: u32,
    blocker_primitive: u32,
    blocker_position: vec3<f32>,
}

fn dominant_directional_light() -> u32 {
    var selected = 0xFFFFFFFFu;
    var brightness = 0.0;
    for (var i = 0u; i < arrayLength(&light_sources); i++) {
        let light = light_sources[i];
        if light.kind == LIGHT_SOURCE_KIND_DIRECTIONAL {
            let candidate = dot(directional_lights[light.id].luminance, vec3(0.2126, 0.7152, 0.0722));
            if candidate > brightness {
                selected = i;
                brightness = candidate;
            }
        }
    }
    return selected;
}

fn is_grass_instance(instance_id: u32) -> bool {
    let geometry = geometry_ids[instance_id];
    // Ordinary masks are 0xFF, so checking only bit 1 would mislabel solids.
    return geometry.ray_mask == 0x02u && geometry.diagnostic_primitives_per_group > 0u;
}

fn same_group(a_instance: u32, a_primitive: u32, b_instance: u32, b_primitive: u32) -> bool {
    if a_instance != b_instance { return false; }
    let size = geometry_ids[a_instance].diagnostic_primitives_per_group;
    return size > 0u && a_primitive / max(size, 1u) == b_primitive / max(size, 1u);
}


fn probe_surface(surface: ResolvedGPixel, pixel: vec2<u32>, direction: vec3<f32>) -> Probe {
    var probe: Probe;
    probe.receiver = match_receiver(surface, pixel);
    if !probe.receiver.matched { return probe; }
    let origin = offset_surface_ray_origin(surface.ray_origin, surface.geometric_normal, direction);
    // Closest blocker, NOT traversal's first accepted blocker. Never reject the
    // receiver triangle: seeing it is precisely what this diagnostic tests.
    let blocker = trace_ray(origin, direction, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);
    probe.blocked = blocker.kind != RAY_QUERY_INTERSECTION_NONE;
    if probe.blocked {
        probe.blocker_instance = blocker.instance_custom_data;
        probe.blocker_primitive = blocker.primitive_index;
        probe.blocker_position = origin + direction * blocker.t;
    }
    return probe;
}

fn blocker_color(probe: Probe) -> vec3<f32> {
    if !probe.receiver.matched { return UNMATCHED; }
    if !probe.blocked { return vec3(1.0); }
    if probe.receiver.instance_id == probe.blocker_instance && probe.receiver.primitive_id == probe.blocker_primitive { return SAME_TRIANGLE; }
    if same_group(probe.receiver.instance_id, probe.receiver.primitive_id, probe.blocker_instance, probe.blocker_primitive) { return SAME_BLADE; }
    if is_grass_instance(probe.blocker_instance) { return OTHER_GRASS; }
    return SOLID_BLOCKER;
}

fn center_probe_highlight(surface: ResolvedGPixel, pixel: vec2<u32>, direction: vec3<f32>) -> vec3<f32> {
    let size = view.main_pass_viewport.zw;
    let center = vec2<u32>(size * 0.5);
    let center_depth = textureLoad(depth_buffer, center, 0);
    var probe: Probe;
    if center_depth > 0.0 {
        let center_surface = gpixel_resolve(textureLoad(gbuffer, center, 0), center_depth, center, size, view.world_from_clip);
        probe = probe_surface(center_surface, center, direction);
    }
    var color = vec3(0.08);
    if surface.is_grass {
        let receiver = match_receiver(surface, pixel);
        color = vec3(0.25);
        if !receiver.matched { color = UNMATCHED * 0.35; }
        if probe.receiver.matched && receiver.matched {
            if same_group(receiver.instance_id, receiver.primitive_id, probe.receiver.instance_id, probe.receiver.primitive_id) { color = vec3(0.0, 1.0, 1.0); }
            if probe.blocked {
                if same_group(receiver.instance_id, receiver.primitive_id, probe.blocker_instance, probe.blocker_primitive) { color = vec3(0.7, 0.45, 0.0); }
                if receiver.instance_id == probe.blocker_instance && receiver.primitive_id == probe.blocker_primitive {
                    color = vec3(1.0, 1.0, 0.0);
                    if probe.receiver.instance_id == probe.blocker_instance && probe.receiver.primitive_id == probe.blocker_primitive { color = SAME_TRIANGLE; }
                }
            }
        }
    }
    let p = vec2<f32>(pixel) + 0.5;
    let c = vec2<f32>(center) + 0.5;
    if probe.receiver.matched && probe.blocked {
        let clip = view.clip_from_world * vec4(probe.blocker_position, 1.0);
        if clip.w > 0.0 {
            let end = (clip.xy / clip.w * vec2(0.5, -0.5) + 0.5) * size;
            let segment = end - c;
            let t = clamp(dot(p - c, segment) / max(dot(segment, segment), 1e-6), 0.0, 1.0);
            let line_distance = length(p - (c + t * segment));
            let marker_distance = length(p - end);
            if line_distance < 0.75 || abs(marker_distance - 6.0) < 1.0 { color = vec3(1.0, 1.0, 0.0); }
        }
    }
    let dc = abs(p - c);
    if (dc.x < 0.75 && dc.y < 5.0) || (dc.y < 0.75 && dc.x < 5.0) {
        color = UNMATCHED;
        if probe.receiver.matched {
            color = select(vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 1.0), probe.blocked);
        }
    }
    return color;
}

fn raw_directional_visibility(surface: ResolvedGPixel, pixel: vec2<u32>, seed: u32) -> vec3<f32> {
    let selected = dominant_directional_light();
    if selected == 0xFFFFFFFFu { return UNMATCHED; }
    let light = light_sources[selected];
    let direction = normalize(directional_lights[light.id].direction_to_light);
    if constants.raw_directional_visibility == 5u {
        return center_probe_highlight(surface, pixel, direction);
    }
    if surface.is_grass && constants.raw_directional_visibility == 3u {
        return blocker_color(probe_surface(surface, pixel, direction));
    }
    if surface.is_grass && constants.raw_directional_visibility == 4u {
        let receiver = match_receiver(surface, pixel);
        if !receiver.matched { return UNMATCHED; }
        // Use the RT triangle's normal without the packed-normal uncertainty
        // band. The ordinary path/classification still exercise that band.
        var origin = receiver.position;
        if bool(constants.grass_geometric_ray_origin) {
            origin = offset_triangle_ray_origin(origin, receiver.normal, direction);
        } else if !bool(constants.grass_unshifted_ray_origin) {
            origin += surface.world_normal * RAY_T_MIN;
        }
        return vec3(trace_visibility(origin, vec4(direction, 0.0)));
    }
    var light_position = vec4(direction, 0.0);
    if constants.raw_directional_visibility == 2u {
        light_position = resolve_light_sample(LightSample(selected << 16u, seed), light).world_position;
    }
    return vec3(trace_visibility_from_surface(surface.ray_origin, surface.geometric_normal, light_position));
}
