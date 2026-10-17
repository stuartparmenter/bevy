use super::allocator::SlotAllocator;
use crate::scene::extract::{
    ExtractedRaytracingDirectionalLight, ExtractedRaytracingPointLight,
    ExtractedRaytracingRectLight, ExtractedRaytracingSpotLight,
};
use bevy_ecs::{entity::Entity, query::Changed, system::Query};
use bevy_math::{
    ops::{cos, sin, FloatPow},
    Vec3,
};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::render_resource::{AtomicSparseBufferVec, BufferUsages};
use bevy_render::{impl_atomic_pod, render_resource::AtomicPod};
use bytemuck::{Pod, Zeroable};
use core::sync::atomic::{AtomicBool, Ordering};
use core::{f32::consts::TAU, hash::Hash};
use bevy_utils::once;
use tracing::error;

const LIGHT_NOT_PRESENT_THIS_FRAME: u32 = u32::MAX;
pub const MAX_EMISSIVE_TRIANGLES_PER_LIGHT: u32 = u16::MAX as u32;
/// Light ids are packed into 16 bits alongside a 16-bit triangle id, and index 65535 with
/// triangle 65535 would alias `NULL_LIGHT_ID`.
pub const MAX_LIGHT_SOURCES: usize = u16::MAX as usize;

const LIGHT_SOURCE_KIND_BITS: u32 = 3;

const LIGHT_SOURCE_KIND_EMISSIVE_MESH: u32 = 0;
const LIGHT_SOURCE_KIND_DIRECTIONAL: u32 = 1;
const LIGHT_SOURCE_KIND_POINT: u32 = 2;
const LIGHT_SOURCE_KIND_RECT: u32 = 3;
const LIGHT_SOURCE_KIND_ENVIRONMENT: u32 = 4;

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuLightSource {
    // The low LIGHT_SOURCE_KIND_BITS bits are the kind. For emissive meshes, the rest are the first
    // triangle in this logical light's at-most-65535-triangle chunk.
    kind: u32,
    id: u32,
}

/// Stable identity for one light in the light array.
///
/// An entity can contribute several kinds at once, and an emissive mesh larger than one triangle
/// chunk contributes one source per chunk, so the entity alone is not enough to identify a source.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LightSourceId {
    EmissiveMesh { entity: Entity, first_triangle: u32 },
    DirectionalLight(Entity),
    PointLight(Entity),
    SpotLight(Entity),
    RectLight(Entity),
    /// The single environment slot, the scene's environment map light.
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
            first_triangle <= u32::MAX >> LIGHT_SOURCE_KIND_BITS,
            "emissive light triangle offset exceeds its encoding"
        );
        Self {
            kind: (first_triangle << LIGHT_SOURCE_KIND_BITS) | LIGHT_SOURCE_KIND_EMISSIVE_MESH,
            id: instance_id,
        }
    }

    fn new_directional_light(directional_light_id: u32) -> GpuLightSource {
        Self {
            kind: LIGHT_SOURCE_KIND_DIRECTIONAL,
            id: directional_light_id,
        }
    }

    fn new_point_light(point_light_id: u32) -> GpuLightSource {
        Self {
            kind: LIGHT_SOURCE_KIND_POINT,
            id: point_light_id,
        }
    }

    fn new_rect_light(rect_light_id: u32) -> GpuLightSource {
        Self {
            kind: LIGHT_SOURCE_KIND_RECT,
            id: rect_light_id,
        }
    }

    /// The one environment entry: resolved through the importance pyramid, and flagged so the
    /// shaders MIS-weight it against the environment radiance a missed BRDF ray picks up.
    fn new_environment_light() -> GpuLightSource {
        Self {
            kind: LIGHT_SOURCE_KIND_ENVIRONMENT,
            id: 0,
        }
    }
}

/// How many emissive-mesh sources fit once the analytic lights (directional, point, spot and rect)
/// and the environment have reserved theirs. [`MAX_LIGHT_SOURCES`] bounds the whole list, and only
/// the emissive sources have anything to yield: an analytic light is NEE-only, while a truncated
/// emissive chunk is only unsampled emission.
pub fn emissive_light_source_budget(non_emissive_light_count: usize) -> usize {
    MAX_LIGHT_SOURCES.saturating_sub(non_emissive_light_count)
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

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuPointLight {
    position: Vec3,
    radius: f32,
    intensity: Vec3,
    cos_inner: f32,
    axis: Vec3,
    cos_outer: f32,
}

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuRectLight {
    center: Vec3,
    area: f32,
    edge_x: Vec3,
    _padding0: f32,
    edge_y: Vec3,
    _padding1: f32,
    radiance: Vec3,
    _padding2: f32,
}

impl_atomic_pod!(GpuLightSource, GpuLightSourceBlob);
impl_atomic_pod!(GpuDirectionalLight, GpuDirectionalLightBlob);
impl_atomic_pod!(GpuPointLight, GpuPointLightBlob);
impl_atomic_pod!(GpuRectLight, GpuRectLightBlob);

/// Floor for a sun disk's angular size, in radians. 0.028 degrees is far narrower than any sun a
/// scene asks for and samples as the light direction either way, but it keeps the cone's solid angle
/// far enough from zero that `illuminance / solid_angle` stays finite.
const MIN_SUN_DISK_ANGULAR_SIZE: f32 = 1.0 / 2048.0;

impl GpuDirectionalLight {
    fn new(directional_light: &ExtractedRaytracingDirectionalLight) -> Self {
        let (cos_theta_max, luminance, inverse_pdf) =
            match Self::sun_disk_cone(directional_light.sun_disk_angular_size) {
                Some((cos_theta_max, solid_angle)) => (
                    cos_theta_max,
                    directional_light.illuminance / solid_angle,
                    solid_angle,
                ),
                // A sun disk of no size (`SunDisk::OFF`) is a delta light
                None => (1.0, directional_light.illuminance, 1.0),
            };

        Self {
            direction_to_light: directional_light.direction_to_light,
            cos_theta_max,
            luminance,
            inverse_pdf,
        }
    }

    /// `cos_theta_max` and the solid angle, in steradians, of the cone a sun disk `angular_size`
    /// radians across subtends, or `None` for a disk of no size (zero, negative or NaN).
    ///
    /// `TAU * (1.0 - cos(angular_size / 2.0))` cancels to exactly 0.0 in f32 below 2^-11 radians, so
    /// the solid angle goes through `1 - cos(x) = 2 sin^2(x / 2)` instead, and
    /// [`MIN_SUN_DISK_ANGULAR_SIZE`] only bounds how small a positive size can get.
    fn sun_disk_cone(angular_size: f32) -> Option<(f32, f32)> {
        if !(angular_size > 0.0) {
            return None;
        }
        let angular_size = angular_size.max(MIN_SUN_DISK_ANGULAR_SIZE);
        Some((
            cos(angular_size / 2.0),
            TAU * 2.0 * sin(angular_size / 4.0).squared(),
        ))
    }
}

impl GpuPointLight {
    fn new_point(point_light: &ExtractedRaytracingPointLight) -> Self {
        // Point lights get a cone that never cuts anything off
        Self {
            position: point_light.position,
            radius: point_light.radius,
            intensity: point_light.intensity,
            cos_inner: -1.0,
            axis: Vec3::NEG_Z,
            cos_outer: -1.0,
        }
    }

    fn new_spot(spot_light: &ExtractedRaytracingSpotLight) -> Self {
        Self {
            position: spot_light.position,
            radius: spot_light.radius,
            intensity: spot_light.intensity,
            cos_inner: spot_light.cos_inner,
            axis: spot_light.axis,
            cos_outer: spot_light.cos_outer,
        }
    }
}

impl GpuRectLight {
    fn new(rect_light: &ExtractedRaytracingRectLight) -> Self {
        Self {
            center: rect_light.center,
            area: rect_light.area,
            edge_x: rect_light.edge_x,
            _padding0: 0.0,
            edge_y: rect_light.edge_y,
            _padding1: 0.0,
            radiance: rect_light.radiance,
            _padding2: 0.0,
        }
    }
}

/// Light slots and the incremental previous-frame id translation state.
pub struct LightState {
    /// Kept gap-free because shaders derive the light count with `arrayLength`.
    pub sources: AtomicSparseBufferVec<GpuLightSource>,
    pub directional_lights: AtomicSparseBufferVec<GpuDirectionalLight>,
    pub point_lights: AtomicSparseBufferVec<GpuPointLight>,
    pub rect_lights: AtomicSparseBufferVec<GpuRectLight>,
    pub previous_frame_id_translations: AtomicSparseBufferVec<u32>,
    pub index: LightIndex,
    /// Light ids as of the last frame whose translation table the lighting shader actually read.
    previous_index: HashMap<LightSourceId, u32>,
    nonidentity_translations: Vec<u32>,
    directional_light_slots: SlotAllocator<Entity>,
    /// Spot lights share the point light buffer, and so cannot be keyed by just Entity.
    point_light_slots: SlotAllocator<LightSourceId>,
    rect_light_slots: SlotAllocator<Entity>,
    /// Set by the lighting node once it has recorded work reading the translation table.
    translations_consumed: AtomicBool,
    /// Emissive chunk sources refused at the cap, retried whenever slots free up. The instances
    /// still render, the extra emitters just stop being explicitly sampled.
    dropped_emissives: HashMap<LightSourceId, GpuLightSource>,
    reported_light_source_overflow: bool,
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
            point_lights: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                "solari_point_lights".into(),
            ),
            rect_lights: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                "solari_rect_lights".into(),
            ),
            previous_frame_id_translations: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                "solari_previous_frame_light_id_translations".into(),
            ),
            index: LightIndex::default(),
            previous_index: HashMap::default(),
            nonidentity_translations: Vec::new(),
            directional_light_slots: SlotAllocator::new(),
            point_light_slots: SlotAllocator::new(),
            rect_light_slots: SlotAllocator::new(),
            translations_consumed: AtomicBool::new(false),
            dropped_emissives: HashMap::default(),
            reported_light_source_overflow: false,
        }
    }

    pub fn directional_light_count(&self) -> usize {
        self.directional_light_slots.len()
    }

    /// Point and spot lights, which share a buffer.
    pub fn point_light_count(&self) -> usize {
        self.point_light_slots.len()
    }

    pub fn rect_light_count(&self) -> usize {
        self.rect_light_slots.len()
    }

    pub fn environment_light_count(&self) -> usize {
        usize::from(self.index.get(&LightSourceId::Environment).is_some())
    }

    fn non_emissive_light_count(&self) -> usize {
        self.directional_light_count()
            + self.point_light_count()
            + self.rect_light_count()
            + self.environment_light_count()
    }

    pub fn emissive_light_count(&self) -> usize {
        self.index
            .len()
            .saturating_sub(self.non_emissive_light_count())
    }

    fn emissive_budget(&self) -> usize {
        emissive_light_source_budget(self.non_emissive_light_count())
    }

    /// Applies this frame's light changes.
    ///
    /// Lights are retained, so only the ones that changed are processed. Removals go first so that
    /// a light removed and re-added in the same frame ends up present.
    pub fn update_directional_lights(
        &mut self,
        changed: &Query<
            (Entity, &ExtractedRaytracingDirectionalLight),
            Changed<ExtractedRaytracingDirectionalLight>,
        >,
        removed: impl IntoIterator<Item = Entity>,
    ) {
        // Removals go first so that a light removed and re-added in the same frame ends up present
        for entity in removed {
            if self.directional_light_slots.remove(&entity).is_some() {
                self.remove_light(LightSourceId::DirectionalLight(entity));
            }
        }
        for (entity, directional_light) in changed {
            let slot = self.directional_light_slots.get_or_allocate(entity);
            self.directional_lights
                .grow_and_set(slot, GpuDirectionalLight::new(directional_light));
            self.add_light(
                LightSourceId::DirectionalLight(entity),
                GpuLightSource::new_directional_light(slot),
            );
        }
    }

    pub fn update_point_lights(
        &mut self,
        changed: &Query<
            (Entity, &ExtractedRaytracingPointLight),
            Changed<ExtractedRaytracingPointLight>,
        >,
        removed: impl IntoIterator<Item = Entity>,
    ) {
        for entity in removed {
            self.remove_point_or_spot_light(LightSourceId::PointLight(entity));
        }
        for (entity, point_light) in changed {
            self.set_point_or_spot_light(
                LightSourceId::PointLight(entity),
                GpuPointLight::new_point(point_light),
            );
        }
    }

    pub fn update_spot_lights(
        &mut self,
        changed: &Query<
            (Entity, &ExtractedRaytracingSpotLight),
            Changed<ExtractedRaytracingSpotLight>,
        >,
        removed: impl IntoIterator<Item = Entity>,
    ) {
        for entity in removed {
            self.remove_point_or_spot_light(LightSourceId::SpotLight(entity));
        }
        for (entity, spot_light) in changed {
            self.set_point_or_spot_light(
                LightSourceId::SpotLight(entity),
                GpuPointLight::new_spot(spot_light),
            );
        }
    }

    pub fn update_rect_lights(
        &mut self,
        changed: &Query<
            (Entity, &ExtractedRaytracingRectLight),
            Changed<ExtractedRaytracingRectLight>,
        >,
        removed: impl IntoIterator<Item = Entity>,
    ) {
        for entity in removed {
            if self.rect_light_slots.remove(&entity).is_some() {
                self.remove_light(LightSourceId::RectLight(entity));
            }
        }
        for (entity, rect_light) in changed {
            let slot = self.rect_light_slots.get_or_allocate(entity);
            self.rect_lights
                .grow_and_set(slot, GpuRectLight::new(rect_light));
            self.add_light(
                LightSourceId::RectLight(entity),
                GpuLightSource::new_rect_light(slot),
            );
        }
    }

    pub fn finish_update(&mut self, environment_present: bool) {
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

    fn set_point_or_spot_light(&mut self, id: LightSourceId, light: GpuPointLight) {
        let slot = self.point_light_slots.get_or_allocate(id);
        self.point_lights.grow_and_set(slot, light);
        self.add_light(id, GpuLightSource::new_point_light(slot));
    }

    fn remove_point_or_spot_light(&mut self, id: LightSourceId) {
        if self.point_light_slots.remove(&id).is_some() {
            self.remove_light(id);
        }
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

        // Only emissive sources yield to the cap, so analytic lights alone can still exceed it,
        // and their ids would then alias in the shaders' 16-bit light id
        if self.index.len() > MAX_LIGHT_SOURCES {
            once!(error!(
                lights = self.index.len(),
                maximum = MAX_LIGHT_SOURCES,
                "too many analytic lights in the scene; lighting will be wrong"
            ));
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
        GpuLightSource, LightIndex, LightSourceId, LIGHT_SOURCE_KIND_BITS, MAX_LIGHT_SOURCES,
        MIN_SUN_DISK_ANGULAR_SIZE,
    };
    use bevy_ecs::entity::Entity;

    const ENTITY: Entity = Entity::PLACEHOLDER;
    const EMISSIVE: LightSourceId = LightSourceId::EmissiveMesh {
        entity: ENTITY,
        first_triangle: 0,
    };

    #[test]
    fn light_index_keeps_sources_on_the_same_entity_independent() {
        let mut lights = LightIndex::default();

        assert_eq!(lights.insert(EMISSIVE), 0);
        assert_eq!(lights.insert(LightSourceId::DirectionalLight(ENTITY)), 1);
        assert_eq!(lights.insert(LightSourceId::PointLight(ENTITY)), 2);
        assert_eq!(lights.insert(LightSourceId::SpotLight(ENTITY)), 3);
        assert_eq!(lights.insert(LightSourceId::RectLight(ENTITY)), 4);
        assert_eq!(lights.len(), 5);
    }

    #[test]
    fn light_index_insert_is_idempotent() {
        let point = LightSourceId::PointLight(ENTITY);
        let mut lights = LightIndex::default();

        assert_eq!(lights.insert(point), 0);
        assert_eq!(lights.insert(point), 0);
        assert_eq!(lights.len(), 1);
    }

    #[test]
    fn light_index_remove_moves_the_last_source_into_the_hole() {
        let emissive = EMISSIVE;
        let point = LightSourceId::PointLight(ENTITY);
        let rect = LightSourceId::RectLight(ENTITY);
        let mut lights = LightIndex::default();
        lights.insert(emissive);
        lights.insert(point);
        lights.insert(rect);

        assert_eq!(lights.remove(emissive), Some((0, 2)));
        assert_eq!(lights.get(&emissive), None);
        assert_eq!(lights.get(&rect), Some(0));
        assert_eq!(lights.get(&point), Some(1));
        assert_eq!(lights.len(), 2);
    }

    #[test]
    fn light_index_remove_of_the_last_source_moves_nothing() {
        let directional = LightSourceId::DirectionalLight(ENTITY);
        let rect = LightSourceId::RectLight(ENTITY);
        let mut lights = LightIndex::default();
        lights.insert(directional);
        lights.insert(rect);

        assert_eq!(lights.remove(rect), Some((1, 1)));
        assert_eq!(lights.get(&directional), Some(0));
        assert_eq!(lights.len(), 1);
    }

    #[test]
    fn light_index_remove_of_a_missing_source_returns_none() {
        let mut lights = LightIndex::default();
        lights.insert(LightSourceId::PointLight(ENTITY));

        assert_eq!(lights.remove(LightSourceId::RectLight(ENTITY)), None);
        assert_eq!(lights.len(), 1);
    }

    #[test]
    fn light_index_remove_of_every_source_leaves_it_empty() {
        let emissive = EMISSIVE;
        let directional = LightSourceId::DirectionalLight(ENTITY);
        let mut lights = LightIndex::default();
        lights.insert(emissive);
        lights.insert(directional);

        assert_eq!(lights.remove(emissive), Some((0, 1)));
        assert_eq!(lights.remove(directional), Some((0, 0)));
        assert!(lights.is_empty());
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
        // The environment kind has to fit the kind bits so light_source_kind() reads it back.
        assert_eq!(environment.kind >> LIGHT_SOURCE_KIND_BITS, 0);
        // The shader derives the environment's selection probability from the last entry's kind,
        // so the light set must keep pinning it last.
        assert!(shader.contains(
            "light_source_kind(light_sources[light_count - 1u]) != LIGHT_SOURCE_KIND_ENVIRONMENT"
        ));
    }

    #[test]
    fn the_emissive_budget_leaves_room_for_every_other_light() {
        // The whole list has to fit, so a truncation can never reach an analytic light, which is
        // NEE-only.
        assert_eq!(emissive_light_source_budget(3 + 1) + 3 + 1, MAX_LIGHT_SOURCES);
        // Only the tail truncation can help a scene whose own lights exceed the cap between them.
        assert_eq!(emissive_light_source_budget(MAX_LIGHT_SOURCES + 1), 0);
    }

    #[test]
    fn a_sun_disk_of_no_size_is_a_delta_light() {
        for angular_size in [0.0, -1.0, f32::NAN] {
            assert_eq!(GpuDirectionalLight::sun_disk_cone(angular_size), None);
        }
    }

    #[test]
    fn a_sun_disk_narrower_than_f32_can_resolve_keeps_its_solid_angle() {
        // TAU * (1.0 - cos(angular_size / 2.0)) is exactly 0.0 in f32 at and below
        // MIN_SUN_DISK_ANGULAR_SIZE, and dividing an illuminance by it emits nothing at all.
        for angular_size in [
            0.0001,
            MIN_SUN_DISK_ANGULAR_SIZE,
            0.0010472,
            0.00615,
            0.00930842,
            core::f32::consts::PI,
        ] {
            let (_, solid_angle) = GpuDirectionalLight::sun_disk_cone(angular_size).unwrap();
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
        assert_eq!(light.kind, 65_535 << LIGHT_SOURCE_KIND_BITS);
        assert_eq!(light.id, 42);
    }
}
