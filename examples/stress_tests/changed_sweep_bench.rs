//! Times a single `Changed`-filter sweep over millions of mesh entities in a
//! bare [`World`] (no `App`, no plugins, no assets).
//!
//! Measures the exact `Or<Changed<...>>` filter set used by
//! `extract_meshes_for_gpu_building` against slimmer variants, at 0%, 1%, and
//! 100% changed, via both the sequential fold path and `par_iter`.

use core::any::TypeId;
use std::hint::black_box;
use std::time::{Duration, Instant};

use bevy::{
    asset::Handle,
    camera::{
        primitives::Aabb,
        visibility::{
            InheritedVisibility, NoCpuCulling, NoFrustumCulling, ViewVisibility, Visibility,
            VisibilityClass, VisibilityRange,
        },
    },
    ecs::change_detection::DetectChangesMut,
    ecs::prelude::*,
    ecs::query::QueryFilter,
    light::{NotShadowCaster, NotShadowReceiver, TransmittedShadowReceiver},
    math::Vec3,
    mesh::{skinning::SkinnedMesh, Mesh, Mesh3d, MeshTag},
    pbr::{Lightmap, MeshMaterial3d, PreviousGlobalTransform, StandardMaterial},
    render::batching::NoAutomaticBatching,
    tasks::{ComputeTaskPool, TaskPool},
    transform::components::{GlobalTransform, Transform},
};

/// The exact filter from `extract_meshes_for_gpu_building`
/// (crates/bevy_pbr/src/render/mesh.rs).
type FullFilter = Or<(
    Changed<ViewVisibility>,
    Changed<GlobalTransform>,
    Changed<PreviousGlobalTransform>,
    Changed<Lightmap>,
    Changed<Aabb>,
    Changed<Mesh3d>,
    Changed<MeshTag>,
    Or<(
        Changed<NoFrustumCulling>,
        Changed<NotShadowReceiver>,
        Changed<TransmittedShadowReceiver>,
        Changed<NotShadowCaster>,
        Changed<NoAutomaticBatching>,
        Changed<NoCpuCulling>,
    )>,
    Changed<VisibilityRange>,
    Changed<SkinnedMesh>,
)>;

/// Only the filtered components actually present on a typical mesh archetype.
type PresentFilter = Or<(
    Changed<ViewVisibility>,
    Changed<GlobalTransform>,
    Changed<PreviousGlobalTransform>,
    Changed<Aabb>,
    Changed<Mesh3d>,
)>;

/// A hypothetical slimmed-down set: only the components a mesh must react to.
type SlimFilter = Or<(
    Changed<ViewVisibility>,
    Changed<GlobalTransform>,
    Changed<Mesh3d>,
)>;

type SingleFilter = Changed<GlobalTransform>;

const WARMUP: usize = 3;
const REPS: usize = 15;

fn main() {
    ComputeTaskPool::get_or_init(TaskPool::default);
    let counts: Vec<usize> = {
        let args: Vec<usize> = std::env::args()
            .skip(1)
            .filter_map(|a| a.parse().ok())
            .collect();
        if args.is_empty() {
            vec![3_000_000, 3_500_000, 4_000_000, 4_500_000, 5_000_000]
        } else {
            args
        }
    };
    println!(
        "threads={} | seq = iter().fold, par = par_iter().for_each | times in ms (min/median)",
        ComputeTaskPool::get().thread_num()
    );
    println!(
        "{:>9} {:>8} {:<10} {:>12} {:>15} {:>15}",
        "entities", "scenario", "filter", "matched", "seq min/med", "par min/med"
    );
    for n in counts {
        run_size(n);
    }
}

fn run_size(n: usize) {
    let mut world = World::new();

    let mesh_handle: Handle<Mesh> = Handle::default();
    let material_handle: Handle<StandardMaterial> = Handle::default();
    // Mirror what `VisibilityPlugin`'s `add_visibility_class::<Mesh3d>` hook
    // would have produced in a full `App`.
    let mut visibility_class = VisibilityClass::default();
    visibility_class.push(TypeId::of::<Mesh3d>());

    let spawn_start = Instant::now();
    let entities: Vec<Entity> = world
        .spawn_batch((0..n).map(|i| {
            let translation = Vec3::new(
                (i % 1000) as f32,
                ((i / 1000) % 1000) as f32,
                (i / 1_000_000) as f32,
            );
            let transform = Transform::from_translation(translation);
            let global = GlobalTransform::from(transform);
            (
                transform,
                global,
                Visibility::default(),
                InheritedVisibility::VISIBLE,
                ViewVisibility::default(),
                Mesh3d(mesh_handle.clone()),
                MeshMaterial3d::<StandardMaterial>(material_handle.clone()),
                Aabb::from_min_max(Vec3::splat(-0.5), Vec3::splat(0.5)),
                PreviousGlobalTransform(global.affine()),
                visibility_class.clone(),
            )
        }))
        .collect();
    eprintln!(
        "-- spawned {n} entities in {:.2?} ({} archetypes)",
        spawn_start.elapsed(),
        world.archetypes().len()
    );

    // Freshly spawned: everything is Added/Changed.
    bench_all(&mut world, n, "100%");

    // Steady state: nothing changed since the last sweep.
    world.clear_trackers();
    bench_all(&mut world, n, "0%");

    // Touch 1% of transforms, spread evenly across the table.
    for e in entities.iter().step_by(100) {
        world
            .get_mut::<GlobalTransform>(*e)
            .unwrap()
            .set_changed();
    }
    bench_all(&mut world, n, "1%");
}

fn bench_all(world: &mut World, n: usize, scenario: &str) {
    bench::<SingleFilter>(world, n, scenario, "single1");
    bench::<SlimFilter>(world, n, scenario, "slim3");
    bench::<PresentFilter>(world, n, scenario, "present5");
    bench::<FullFilter>(world, n, scenario, "full15");
}

fn bench<F: QueryFilter + 'static>(world: &mut World, n: usize, scenario: &str, label: &str) {
    // Sequential: the fold override is the batched mask path.
    let mut state = world.query_filtered::<(), F>();
    for _ in 0..WARMUP {
        black_box(state.iter(world).fold(0u64, |acc, _| acc + 1));
    }
    let mut seq_times = Vec::with_capacity(REPS);
    let mut matched = 0u64;
    for _ in 0..REPS {
        let t = Instant::now();
        matched = state.iter(world).fold(0u64, |acc, _| acc + 1);
        seq_times.push(t.elapsed());
        black_box(matched);
    }

    // Parallel: same shape as extract_meshes_for_gpu_building's
    // par_iter().for_each_init. Fetch Entity + black_box so the sweep cannot
    // be optimized out.
    let mut par_state = world.query_filtered::<Entity, F>();
    for _ in 0..WARMUP {
        par_state.par_iter(world).for_each(|e| {
            black_box(e);
        });
    }
    let mut par_times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        par_state.par_iter(world).for_each(|e| {
            black_box(e);
        });
        par_times.push(t.elapsed());
    }

    println!(
        "{:>9} {:>8} {:<10} {:>12} {:>15} {:>15}",
        n,
        scenario,
        label,
        matched,
        fmt_stats(&mut seq_times),
        fmt_stats(&mut par_times),
    );
}

fn fmt_stats(times: &mut [Duration]) -> String {
    times.sort_unstable();
    let min = times[0];
    let med = times[times.len() / 2];
    format!("{:.3}/{:.3}", min.as_secs_f64() * 1e3, med.as_secs_f64() * 1e3)
}
