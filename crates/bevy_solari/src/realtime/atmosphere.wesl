import bevy_pbr::atmosphere::functions::directional_light_attenuation;
import package::scene::sampling::{calculate_resolved_light_contribution, LightContribution, ResolvedLightSample};
import package::realtime::bindings::{atmosphere, atmosphere_transmittance_lut, atmosphere_transmittance_sampler};

enable wgpu_ray_query;

/// `calculate_resolved_light_contribution`, with directional lights attenuated by the view's
/// atmosphere at `ray_origin`, as in the raster light loop (`pbr_lighting::directional_light`).
/// The reference pathtracer has no per-view bindings and shades without the atmosphere.
///
/// Directional light samples are points on the light's disk, so testing each one against the
/// horizon (an angular size of 0) averages to the visible fraction of the disk. Without soft
/// shadows every sample is the disk's centre, and the light cuts off at the horizon.
fn calculate_light_contribution(resolved_light_sample: ResolvedLightSample, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>) -> LightContribution {
    var light_contribution = calculate_resolved_light_contribution(resolved_light_sample, ray_origin, origin_world_normal);
    // Views without an atmosphere bind a zeroed fallback (`outer_radius` 0). Lights behind the
    // surface contribute nothing, so they skip the lookup.
    if atmosphere.outer_radius > 0.0 && resolved_light_sample.world_position.w == 0.0 && dot(light_contribution.wi, origin_world_normal) > 0.0 {
        light_contribution.radiance *= directional_light_attenuation(
            atmosphere,
            atmosphere_transmittance_lut,
            atmosphere_transmittance_sampler,
            ray_origin,
            light_contribution.wi,
            0.0,
        );
    }
    return light_contribution;
}
