use super::allocator::SlotAllocator;
use bevy_color::ColorToComponents;
use bevy_ecs::{entity::Entity, system::Query};
use bevy_math::{
    ops::{cos, sin, FloatPow},
    Vec3,
};
use bevy_pbr::ExtractedDirectionalLight;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::render_resource::{AtomicSparseBufferVec, BufferUsages};
use bevy_render::{impl_atomic_pod, render_resource::AtomicPod};
use bytemuck::{Pod, Zeroable};
use core::sync::atomic::{AtomicBool, Ordering};
use core::{f32::consts::TAU, hash::Hash};
use tracing::{error, info_span};

const LIGHT_NOT_PRESENT_THIS_FRAME: u32 = u32::MAX;
pub const MAX_EMISSIVE_TRIANGLES_PER_LIGHT: u32 = u16::MAX as u32;
/// Light ids are packed into 16 bits alongside a 16-bit triangle id, and index 65535 with
/// triangle 65535 would alias `NULL_LIGHT_ID`.
pub const MAX_LIGHT_SOURCES: usize = u16::MAX as usize;

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuLightSource {
    // The low bit is the kind. For emissive meshes, the upper 31 bits are the
    // first triangle in this logical light's at-most-65535-triangle chunk.
    kind: u32,
    id: u32,
}

/// Stable identity for one source in the light array.
///
/// An entity can contribute several kinds at once, and an emissive mesh larger than one triangle
/// chunk contributes one source per chunk, so the entity alone is not enough to identify a source.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LightSourceId {
    EmissiveMesh { entity: Entity, first_triangle: u32 },
    Directional(Entity),
    /// The single environment slot, the Solari camera's `EnvironmentMapLight`.
    Environment,
}

impl LightSourceId {
    fn is_emissive(&self) -> bool {
        matches!(self, LightSourceId::EmissiveMesh { .. })
    }
}

#[derive(Default)]
pub struct LightIndex {
    indices: HashMap<LightSourceId, u32>,
    ids: Vec<LightSourceId>,
    changed: HashSet<LightSourceId>,
}

impl LightIndex {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    fn get(&self, id: &LightSourceId) -> Option<u32> {
        self.indices.get(id).copied()
    }

    fn insert(&mut self, id: LightSourceId) -> u32 {
        if let Some(&index) = self.indices.get(&id) {
            return index;
        }

        let index = self.ids.len() as u32;
        self.ids.push(id);
        self.indices.insert(id, index);
        self.changed.insert(id);
        index
    }

    /// Removes `id` and reports both its old index and the old final index.
    ///
    /// Only the ids tracked here are swapped down. When the two indices differ, the caller has to
    /// mirror that swap in `sources`, copying the element at the old final index into the hole.
    fn remove(&mut self, id: LightSourceId) -> Option<(u32, u32)> {
        let index = self.indices.remove(&id)?;
        self.changed.insert(id);

        let last = self.ids.len() as u32 - 1;
        self.ids.swap_remove(index as usize);

        if index != last {
            let moved = self.ids[index as usize];
            self.indices.insert(moved, index);
            self.changed.insert(moved);
        }

        Some((index, last))
    }
}

impl GpuLightSource {
    pub fn new_emissive_mesh_light(instance_id: u32, first_triangle: u32) -> GpuLightSource {
        assert!(
            first_triangle <= u32::MAX >> 1,
            "emissive light triangle offset exceeds its 31-bit encoding"
        );
        Self {
            kind: first_triangle << 1,
            id: instance_id,
        }
    }

    fn new_directional_light(directional_light_id: u32) -> GpuLightSource {
        Self {
            kind: 1,
            id: directional_light_id,
        }
    }

    /// The one environment entry: resolved through the importance pyramid, and flagged so the
    /// shaders MIS-weight it against the environment radiance a missed BRDF ray picks up.
    fn new_environment_light() -> GpuLightSource {
        Self { kind: 3, id: 0 }
    }
}

/// How many emissive-mesh sources fit once the directional and environment lights have reserved
/// theirs. [`MAX_LIGHT_SOURCES`] bounds the whole list, and only the emissive sources have
/// anything to yield: a truncated sun disk is NEE-only with no environment radiance to fall back
/// on, while a truncated emissive chunk is only unsampled emission.
pub fn emissive_light_source_budget(
    directional_light_count: usize,
    environment_light_count: usize,
) -> usize {
    MAX_LIGHT_SOURCES.saturating_sub(directional_light_count + environment_light_count)
}

pub fn emissive_triangle_chunks(triangle_count: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..triangle_count)
        .step_by(MAX_EMISSIVE_TRIANGLES_PER_LIGHT as usize)
        .map(move |first_triangle| {
            (
                first_triangle,
                (triangle_count - first_triangle).min(MAX_EMISSIVE_TRIANGLES_PER_LIGHT),
            )
        })
}

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuDirectionalLight {
    direction_to_light: Vec3,
    cos_theta_max: f32,
    luminance: Vec3,
    inverse_pdf: f32,
}

impl_atomic_pod!(GpuLightSource, GpuLightSourceBlob);
impl_atomic_pod!(GpuDirectionalLight, GpuDirectionalLightBlob);

/// Floor for a sun disk's angular size, in radians. 0.028 degrees is far narrower than any sun a
/// scene asks for and samples as the light direction either way, but it keeps the cone's solid angle
/// off zero so `illuminance / solid_angle` stays finite. `SunDisk::OFF` is 0.
const MIN_SUN_DISK_ANGULAR_SIZE: f32 = 1.0 / 2048.0;

impl GpuDirectionalLight {
    fn new(directional_light: &ExtractedDirectionalLight) -> Self {
        let (cos_theta_max, solid_angle) =
            Self::sun_disk_cone(directional_light.sun_disk_angular_size);
        let luminance =
            (directional_light.color.to_vec3() * directional_light.illuminance) / solid_angle;

        Self {
            direction_to_light: directional_light.transform.back().into(),
            cos_theta_max,
            luminance,
            inverse_pdf: solid_angle,
        }
    }

    /// `cos_theta_max` and the solid angle, in steradians, of the cone a sun disk `angular_size`
    /// radians across subtends.
    ///
    /// `TAU * (1.0 - cos(angular_size / 2.0))` cancels to exactly 0.0 in f32 below 2^-11 radians, so
    /// the solid angle goes through `1 - cos(x) = 2 sin^2(x / 2)` instead. That leaves
    /// [`MIN_SUN_DISK_ANGULAR_SIZE`] answering only for a disk that is genuinely zero, negative or
    /// NaN, rather than for wherever f32 happens to lose the subtraction.
    fn sun_disk_cone(angular_size: f32) -> (f32, f32) {
        let angular_size = angular_size.max(MIN_SUN_DISK_ANGULAR_SIZE);
        (
            cos(angular_size / 2.0),
            TAU * 2.0 * sin(angular_size / 4.0).squared(),
        )
    }
}

/// Light slots and the incremental previous-frame id translation state.
pub struct LightState {
    /// Kept gap-free because shaders derive the light count with `arrayLength`.
    pub sources: AtomicSparseBufferVec<GpuLightSource>,
    pub directional_lights: AtomicSparseBufferVec<GpuDirectionalLight>,
    pub previous_frame_id_translations: AtomicSparseBufferVec<u32>,
    pub index: LightIndex,
    /// Light ids as of the last frame whose translation table the lighting shader actually read.
    previous_index: HashMap<LightSourceId, u32>,
    nonidentity_translations: Vec<u32>,
    directional_slots: SlotAllocator<LightSourceId>,
    /// Set by the lighting node once it has recorded work reading the translation table.
    translations_consumed: AtomicBool,
    /// Emissive chunk sources refused at the cap, retried whenever slots free up. The instances
    /// still render, the extra emitters just stop being explicitly sampled.
    dropped_emissives: HashMap<LightSourceId, GpuLightSource>,
    reported_light_source_overflow: bool,
    directional_count: usize,
    environment_count: usize,
}

impl LightState {
    pub fn new() -> Self {
        Self {
            sources: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                "solari_light_sources".into(),
            ),
            directional_lights: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                "solari_directional_lights".into(),
            ),
            previous_frame_id_translations: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                "solari_previous_frame_light_id_translations".into(),
            ),
            index: LightIndex::default(),
            previous_index: HashMap::default(),
            nonidentity_translations: Vec::new(),
            directional_slots: SlotAllocator::new(),
            translations_consumed: AtomicBool::new(false),
            dropped_emissives: HashMap::default(),
            reported_light_source_overflow: false,
            directional_count: 0,
            environment_count: 0,
        }
    }

    pub fn directional_light_count(&self) -> usize {
        self.directional_count
    }

    pub fn environment_light_count(&self) -> usize {
        self.environment_count
    }

    pub fn emissive_light_count(&self) -> usize {
        self.index
            .len()
            .saturating_sub(self.directional_count + self.environment_count)
    }

    fn emissive_budget(&self) -> usize {
        emissive_light_source_budget(self.directional_count, self.environment_count)
    }

    pub fn update(
        &mut self,
        directional_lights: &Query<(Entity, &ExtractedDirectionalLight)>,
        environment_present: bool,
    ) {
        // There are few enough directional lights to just walk them every frame
        let _span = info_span!("update_lights").entered();

        let mut live_directional_lights = HashSet::<LightSourceId>::default();
        let mut directional_count = 0;
        for (entity, directional_light) in directional_lights {
            let id = LightSourceId::Directional(entity);
            live_directional_lights.insert(id);
            directional_count += 1;

            let slot = self.directional_slots.get_or_allocate(id);
            self.directional_lights
                .grow_and_set(slot, GpuDirectionalLight::new(directional_light));
            self.add_light(id, GpuLightSource::new_directional_light(slot));
        }

        let stale: Vec<LightSourceId> = self
            .directional_slots
            .keys()
            .copied()
            .filter(|id| !live_directional_lights.contains(id))
            .collect();
        for id in stale {
            self.directional_slots.remove(&id);
            self.remove_light(id);
        }

        self.directional_count = directional_count;
        self.environment_count = usize::from(environment_present);

        if environment_present {
            self.add_light(
                LightSourceId::Environment,
                GpuLightSource::new_environment_light(),
            );
        } else {
            self.remove_light(LightSourceId::Environment);
        }

        self.rebalance_emissives();
        self.pin_environment_last();
        self.write_light_id_translations();
    }

    /// Moves the environment source to the end of the list. The shaders derive the environment's
    /// selection probability from the last entry's kind, so it has to stay there.
    fn pin_environment_last(&mut self) {
        let Some(index) = self.index.get(&LightSourceId::Environment) else {
            return;
        };
        if index == self.index.len() as u32 - 1 {
            return;
        }
        self.remove_light(LightSourceId::Environment);
        let index = self.index.insert(LightSourceId::Environment);
        self.sources
            .grow_and_set(index, GpuLightSource::new_environment_light());
    }

    pub fn add_light(&mut self, id: LightSourceId, source: GpuLightSource) {
        if id.is_emissive()
            && self.index.get(&id).is_none()
            && self.emissive_light_count() >= self.emissive_budget()
        {
            self.dropped_emissives.insert(id, source);
            return;
        }

        self.dropped_emissives.remove(&id);
        let index = self.index.insert(id);
        self.sources.grow_and_set(index, source);
    }

    /// Removes a light, moving the last one down into the hole so the array stays gap-free.
    pub fn remove_light(&mut self, id: LightSourceId) {
        self.dropped_emissives.remove(&id);
        let Some((index, last)) = self.index.remove(id) else {
            return;
        };

        if index != last {
            let source = self.sources.get(last);
            self.sources.grow_and_set(index, source);
        }
    }

    /// Keeps the emissive sources within [`emissive_light_source_budget`], demoting the newest
    /// past the cap and reviving dropped ones whenever slots free up.
    fn rebalance_emissives(&mut self) {
        while self.emissive_light_count() > self.emissive_budget() {
            let Some(id) = self
                .index
                .ids
                .iter()
                .rev()
                .copied()
                .find(LightSourceId::is_emissive)
            else {
                break;
            };
            let source = self.sources.get(self.index.get(&id).unwrap());
            self.remove_light(id);
            self.dropped_emissives.insert(id, source);
        }

        while self.emissive_light_count() < self.emissive_budget() {
            let Some(&id) = self.dropped_emissives.keys().next() else {
                break;
            };
            let source = self.dropped_emissives.remove(&id).unwrap();
            let index = self.index.insert(id);
            self.sources.grow_and_set(index, source);
        }

        if !self.dropped_emissives.is_empty() && !self.reported_light_source_overflow {
            error!(
                dropped = self.dropped_emissives.len(),
                maximum = MAX_LIGHT_SOURCES,
                "too many light sources in the scene; the excess will not be sampled"
            );
            self.reported_light_source_overflow = true;
        }
    }

    /// Rolls the translation table over for a new frame.
    ///
    /// `previous_index` and `changed` only advance once the shader has read the table. The
    /// lighting node bails out while its pipelines compile, and the reservoirs keep the older ids
    /// across such a gap, so the next table has to translate from those instead. `has_consumers`
    /// is false when no view runs Solari lighting, where deferring forever would grow both
    /// without bound.
    pub fn begin_frame(&mut self, has_consumers: bool) {
        for index in core::mem::take(&mut self.nonidentity_translations) {
            self.previous_frame_id_translations
                .grow_and_set(index, index);
        }

        if !has_consumers || self.translations_consumed.swap(false, Ordering::Relaxed) {
            for id in core::mem::take(&mut self.index.changed) {
                match self.index.get(&id) {
                    Some(index) => self.previous_index.insert(id, index),
                    None => self.previous_index.remove(&id),
                };
            }
        }
    }

    /// Records that the lighting shader read this frame's translation table.
    pub fn note_translations_consumed(&self) {
        self.translations_consumed.store(true, Ordering::Relaxed);
    }

    /// Records where each light that moved or disappeared this frame ended up, so that reservoirs
    /// still carrying last frame's light ids can be remapped.
    fn write_light_id_translations(&mut self) {
        for id in &self.index.changed {
            // Lights that first appeared since the last read table have no previous id
            let Some(&previous) = self.previous_index.get(id) else {
                continue;
            };
            let current = self.index.get(id).unwrap_or(LIGHT_NOT_PRESENT_THIS_FRAME);

            if current != previous {
                self.previous_frame_id_translations
                    .grow_and_set(previous, current);
                self.nonidentity_translations.push(previous);
            }
        }

        // Every index the shader might read has to be backed by a real element
        let light_count = self.index.len() as u32;
        let translations = &mut self.previous_frame_id_translations;
        if translations.len() < light_count {
            let start = translations.len();
            translations.grow(light_count);
            for index in start..light_count {
                translations.set(index, index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        emissive_light_source_budget, emissive_triangle_chunks, GpuDirectionalLight,
        GpuLightSource, LightIndex, LightSourceId, MAX_LIGHT_SOURCES, MIN_SUN_DISK_ANGULAR_SIZE,
    };
    use bevy_ecs::entity::Entity;

    #[test]
    fn light_index_keeps_sources_on_the_same_entity_independent() {
        let entity = Entity::PLACEHOLDER;
        let emissive = LightSourceId::EmissiveMesh {
            entity,
            first_triangle: 0,
        };
        let directional = LightSourceId::Directional(entity);
        let mut lights = LightIndex::default();

        assert_eq!(lights.insert(emissive), 0);
        assert_eq!(lights.insert(directional), 1);
        assert_eq!(lights.insert(emissive), 0);
        assert_eq!(lights.len(), 2);

        assert_eq!(lights.remove(emissive), Some((0, 1)));
        assert_eq!(lights.get(&emissive), None);
        assert_eq!(lights.get(&directional), Some(0));
        assert_eq!(lights.len(), 1);

        assert_eq!(lights.remove(directional), Some((0, 0)));
        assert!(lights.is_empty());
    }

    #[test]
    fn the_emissive_budget_leaves_room_for_every_directional_and_environment_light() {
        // The whole list has to fit, so a truncation can never reach the sun, which is NEE-only.
        assert_eq!(
            emissive_light_source_budget(3, 1) + 3 + 1,
            MAX_LIGHT_SOURCES
        );
        // Only the tail truncation can help a scene whose own lights exceed the cap between them.
        assert_eq!(emissive_light_source_budget(MAX_LIGHT_SOURCES, 1), 0);
    }

    #[test]
    fn a_sun_disk_narrower_than_f32_can_resolve_keeps_its_solid_angle() {
        // TAU * (1.0 - cos(angular_size / 2.0)) is exactly 0.0 in f32 at and below
        // MIN_SUN_DISK_ANGULAR_SIZE, and dividing an illuminance by it emits nothing at all.
        for angular_size in [
            0.0,
            0.0001,
            MIN_SUN_DISK_ANGULAR_SIZE,
            0.0010472,
            0.00615,
            0.00930842,
            core::f32::consts::PI,
        ] {
            let (_, solid_angle) = GpuDirectionalLight::sun_disk_cone(angular_size);
            let clamped = angular_size.max(MIN_SUN_DISK_ANGULAR_SIZE) as f64;
            let expected = core::f64::consts::TAU * (1.0 - (clamped / 2.0).cos());

            assert!(solid_angle > 0.0, "{angular_size} rad has no solid angle");
            assert!(
                ((solid_angle as f64 - expected) / expected).abs() < 1e-5,
                "{angular_size} rad: {solid_angle} sr, want {expected} sr"
            );
        }
    }

    #[test]
    fn emissive_triangle_chunks_preserve_every_triangle() {
        let cases = [
            (0, vec![]),
            (1, vec![(0, 1)]),
            (65_535, vec![(0, 65_535)]),
            (65_536, vec![(0, 65_535), (65_535, 1)]),
            (109_512, vec![(0, 65_535), (65_535, 43_977)]),
            (131_070, vec![(0, 65_535), (65_535, 65_535)]),
        ];

        for (triangle_count, expected) in cases {
            assert_eq!(
                emissive_triangle_chunks(triangle_count).collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn emissive_light_encodes_chunk_offset_without_changing_instance() {
        let light = GpuLightSource::new_emissive_mesh_light(42, 65_535);
        assert_eq!(light.kind, 65_535 << 1);
        assert_eq!(light.id, 42);
    }

    #[test]
    fn light_source_kinds_match_the_shader_constants() {
        let shader = include_str!("../bindings.wesl");

        let environment = GpuLightSource::new_environment_light();
        assert_eq!(environment.id, 0);
        assert!(shader.contains(&format!(
            "const LIGHT_SOURCE_KIND_ENVIRONMENT = {}u;",
            environment.kind
        )));
        // The environment kind must still read as non-emissive-mesh in the shader's low-bit test.
        assert_eq!(environment.kind & 1, 1);
        // The shader derives the environment's selection probability from the last entry's kind,
        // so the light set must keep pinning it last.
        assert!(shader
            .contains("light_sources[light_count - 1u].kind != LIGHT_SOURCE_KIND_ENVIRONMENT"));
        assert_eq!(GpuLightSource::new_directional_light(0).kind & 1, 1);
        assert_eq!(GpuLightSource::new_emissive_mesh_light(0, 0).kind & 1, 0);
    }
}
