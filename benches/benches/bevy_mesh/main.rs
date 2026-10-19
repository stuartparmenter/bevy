use bevy_asset::RenderAssetUsages;
use bevy_math::ops::{cos, sin};
use bevy_mesh::{Indices, Mesh, MeshVertexAttribute, PrimitiveTopology, VertexFormat};
use core::hint::black_box;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

fn grid(size: usize) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    for y in 0..=size {
        for x in 0..=size {
            let x = x as f32 / size as f32;
            let y = y as f32 / size as f32;
            positions.push([x, y, 0.15 * sin(x * 10.) * cos(y * 10.)]);
            let nx = -1.5 * cos(x * 10.) * cos(y * 10.);
            let ny = 1.5 * sin(x * 10.) * sin(y * 10.);
            let length = (nx * nx + ny * ny + 1.).sqrt();
            normals.push([nx / length, ny / length, 1. / length]);
            uvs.push([x, y]);
        }
    }
    let indices = (0..size)
        .flat_map(|y| {
            (0..size).flat_map(move |x| {
                let i = (y * (size + 1) + x) as u32;
                let next = i + (size + 1) as u32;
                [i, i + 1, next, i + 1, next + 1, next]
            })
        })
        .collect();
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(Indices::U32(indices))
}

fn mesh_operations(c: &mut Criterion) {
    let mut dedup = c.benchmark_group("mesh/merge_duplicate_vertices");
    dedup.sample_size(30);
    for size in [32, 128, 256] {
        let indexed = grid(size);
        let duplicated = indexed.clone().with_duplicated_vertices();
        let mut unique = indexed;
        unique.remove_indices();
        for (layout, mesh) in [("shared_grid", duplicated), ("all_unique", unique)] {
            dedup.bench_with_input(BenchmarkId::new(layout, size), &mesh, |b, mesh| {
                b.iter_batched_ref(
                    || mesh.clone(),
                    |mesh| {
                        mesh.merge_duplicate_vertices().unwrap();
                        black_box(mesh.indices());
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    let mut custom = grid(128).with_duplicated_vertices();
    for id in 100..114 {
        custom.insert_attribute(
            MeshVertexAttribute::new("custom", id, VertexFormat::Uint32),
            vec![id as u32; custom.count_vertices()],
        );
    }
    dedup.bench_function("17_stream_fallback", |b| {
        b.iter_batched_ref(
            || custom.clone(),
            |mesh| {
                mesh.merge_duplicate_vertices().unwrap();
                black_box(mesh.indices());
            },
            BatchSize::LargeInput,
        );
    });
    dedup.finish();
}

criterion_group!(benches, mesh_operations);
criterion_main!(benches);
