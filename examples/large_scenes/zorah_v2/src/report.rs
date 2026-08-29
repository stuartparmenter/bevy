//! `--report-lod-budget`: what the cache would cost to keep resident at each
//! error bound, measured without an `App` or a GPU.
//!
//! Every part of every complete manifest is decoded once, pruned at each
//! step of the raster ladder and cut at each step of the BLAS ladder, and
//! only the totals are kept, so the walk holds `--bake-workers` parts in
//! memory at a time however large the cache is. A cache still being written
//! is fine: a directory without a manifest, a manifest that does not parse,
//! or a part that fails to open is counted and skipped.

use std::{
    collections::{BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

use bevy::{pbr::experimental::meshlet::MeshletMesh, prelude::*};

use crate::{
    bake::{MeshJob, MeshManifest, MANIFEST_FILE},
    lod::{human_bytes, MESHLET_MAX_PAGES, MESHLET_PAGE_BUDGET, MESHLET_PAGE_SIZE},
};

/// Raster bounds in metres; 0 is the full-detail cache as loaded.
pub const RASTER_LADDER: [f32; 7] = [0.0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032];
/// BLAS bounds in metres.
pub const BLAS_LADDER: [f32; 4] = [0.02, 0.05, 0.1, 0.2];
/// Solari reads a BLAS from a 32-byte vertex (position, normal, UV as
/// `f32`s) and `u32` indices: the buffers `MeshAllocator` holds. The
/// acceleration structure the driver builds from them is not counted; after
/// compaction it is typically of the same order.
pub const BLAS_VERTEX_BYTES: u64 = 32;
pub const BLAS_INDEX_BYTES: u64 = 4;

/// One part measured at every rung of both ladders.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PartSample {
    /// Per `RASTER_LADDER` entry: packed bytes and triangles across LODs.
    pub raster: Vec<(u64, u64)>,
    /// Per `BLAS_LADDER` entry: triangles and vertices of the cut.
    pub blas: Vec<(u64, u64)>,
}

/// The ladders summed over every part measured.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Totals {
    pub parts: usize,
    pub raster_bytes: Vec<u64>,
    pub raster_triangles: Vec<u64>,
    pub blas_triangles: Vec<u64>,
    pub blas_vertices: Vec<u64>,
}

impl Totals {
    pub fn new(raster_rungs: usize, blas_rungs: usize) -> Self {
        Self {
            parts: 0,
            raster_bytes: vec![0; raster_rungs],
            raster_triangles: vec![0; raster_rungs],
            blas_triangles: vec![0; blas_rungs],
            blas_vertices: vec![0; blas_rungs],
        }
    }

    pub fn add(&mut self, sample: &PartSample) {
        assert_eq!(sample.raster.len(), self.raster_bytes.len());
        assert_eq!(sample.blas.len(), self.blas_triangles.len());
        self.parts += 1;
        for (rung, &(bytes, triangles)) in sample.raster.iter().enumerate() {
            self.raster_bytes[rung] += bytes;
            self.raster_triangles[rung] += triangles;
        }
        for (rung, &(triangles, vertices)) in sample.blas.iter().enumerate() {
            self.blas_triangles[rung] += triangles;
            self.blas_vertices[rung] += vertices;
        }
    }

    /// The estimated bytes of BLAS input geometry at a rung.
    pub fn blas_bytes(&self, rung: usize) -> u64 {
        self.blas_vertices[rung] * BLAS_VERTEX_BYTES
            + self.blas_triangles[rung] * 3 * BLAS_INDEX_BYTES
    }
}

/// Prunes and cuts one decoded part at every rung.
pub fn measure_part(full: &MeshletMesh, raster: &[f32], blas: &[f32]) -> PartSample {
    PartSample {
        raster: raster
            .iter()
            .map(|&error| {
                let pruned = full.pruned(error);
                (
                    pruned.packed_byte_len() as u64,
                    pruned.triangle_count() as u64,
                )
            })
            .collect(),
        blas: blas
            .iter()
            .map(|&error| {
                let geometry = full.raytracing_geometry(error);
                (
                    (geometry.indices.len() / 3) as u64,
                    geometry.positions.len() as u64,
                )
            })
            .collect(),
    }
}

/// What the walk found besides the totals.
#[derive(Debug, Default)]
struct WalkCounts {
    manifests: usize,
    /// Directories without a parseable manifest: still baking, or stale.
    incomplete: usize,
    parts_failed: usize,
    /// Triangles the complete manifests' sources held.
    source_triangles: u64,
    stems: BTreeSet<String>,
}

struct Shared {
    /// Part files still to measure.
    queue: Mutex<Vec<PathBuf>>,
    totals: Mutex<Totals>,
    parts_failed: Mutex<Vec<String>>,
}

/// Walks `cache_dir` on `workers` threads and logs the tables. `jobs` is the
/// bake plan of the root glTF, for the coverage line.
pub fn run(cache_dir: &Path, jobs: &[MeshJob], workers: usize) {
    let started = Instant::now();
    let (counts, work) = collect_work(cache_dir);
    info!(
        "measuring {} parts of {} complete manifests in {} ({} directories skipped) on {} threads",
        work.len(),
        counts.manifests,
        cache_dir.display(),
        counts.incomplete,
        workers
    );
    let shared = Arc::new(Shared {
        queue: Mutex::new(work),
        totals: Mutex::new(Totals::new(RASTER_LADDER.len(), BLAS_LADDER.len())),
        parts_failed: Mutex::new(Vec::new()),
    });
    let threads = (0..workers.max(1))
        .map(|index| {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name(format!("zorah-report-{index}"))
                .spawn(move || worker(&shared))
                .expect("spawning a report worker thread")
        })
        .collect::<Vec<_>>();
    for thread in threads {
        let _ = thread.join();
    }
    let totals = shared
        .totals
        .lock()
        .map(|totals| totals.clone())
        .unwrap_or_default();
    let failed = shared
        .parts_failed
        .lock()
        .map(|failed| failed.clone())
        .unwrap_or_default();
    for part in &failed {
        warn!("skipped {part}");
    }

    let job_triangles = jobs
        .iter()
        .map(|job| (job.stem.as_str(), job.triangles()))
        .collect::<HashMap<_, _>>();
    let covered_jobs = jobs
        .iter()
        .filter(|job| counts.stems.contains(&job.stem))
        .count();
    let covered_meshes = jobs
        .iter()
        .filter(|job| counts.stems.contains(&job.stem))
        .map(|job| job.mesh_indices.len())
        .sum::<usize>();
    let total_meshes = jobs.iter().map(|job| job.mesh_indices.len()).sum::<usize>();
    let covered_triangles = counts
        .stems
        .iter()
        .filter_map(|stem| job_triangles.get(stem.as_str()))
        .sum::<u64>();
    let total_triangles = job_triangles.values().sum::<u64>();
    info!(
        "cache coverage: {covered_jobs}/{} geometry files ({covered_meshes}/{total_meshes} root meshes), \
         {covered_triangles}/{total_triangles} source triangles ({:.1}%); {} parts measured, {} skipped, in {:.1?}",
        jobs.len(),
        if total_triangles == 0 {
            0.0
        } else {
            covered_triangles as f64 * 100.0 / total_triangles as f64
        },
        totals.parts,
        failed.len() + counts.parts_failed,
        started.elapsed()
    );
    info!(
        "meshlet page budget: {} ({MESHLET_MAX_PAGES} pages of {})",
        human_bytes(MESHLET_PAGE_BUDGET),
        human_bytes(MESHLET_PAGE_SIZE)
    );
    for line in raster_table(&totals, counts.source_triangles) {
        info!("{line}");
    }
    for line in blas_table(&totals) {
        info!("{line}");
    }
}

/// Every part file of every complete manifest, plus the counts of what was
/// skipped. A directory is only entered through its manifest, so a mesh still
/// baking (no manifest yet) is never half-read.
fn collect_work(cache_dir: &Path) -> (WalkCounts, Vec<PathBuf>) {
    let mut counts = WalkCounts::default();
    let mut work = Vec::new();
    let Ok(entries) = fs::read_dir(cache_dir) else {
        warn!("cannot read {}", cache_dir.display());
        return (counts, work);
    };
    let mut dirs = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    dirs.sort();
    for dir in dirs {
        let manifest = fs::read(dir.join(MANIFEST_FILE))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<MeshManifest>(&bytes).ok());
        let Some(manifest) = manifest else {
            counts.incomplete += 1;
            continue;
        };
        counts.manifests += 1;
        counts.source_triangles += manifest.source_triangles;
        counts.stems.insert(manifest.stem.clone());
        for part in &manifest.parts {
            let path = dir.join(&part.meshlet_file);
            if path.is_file() {
                work.push(path);
            } else {
                counts.parts_failed += 1;
                warn!("{} is missing", path.display());
            }
        }
    }
    (counts, work)
}

fn worker(shared: &Shared) {
    loop {
        let path = match shared.queue.lock() {
            Ok(mut queue) => queue.pop(),
            Err(_) => return,
        };
        let Some(path) = path else {
            return;
        };
        let sample = fs::read(&path)
            .map_err(|error| error.to_string())
            .and_then(|bytes| {
                MeshletMesh::read(&mut bytes.as_slice()).map_err(|error| error.to_string())
            })
            .map(|full| measure_part(&full, &RASTER_LADDER, &BLAS_LADDER));
        match sample {
            Ok(sample) => {
                if let Ok(mut totals) = shared.totals.lock() {
                    totals.add(&sample);
                }
            }
            Err(error) => {
                if let Ok(mut failed) = shared.parts_failed.lock() {
                    failed.push(format!("{}: {error}", path.display()));
                }
            }
        }
    }
}

/// One line per raster rung: bytes against the page budget.
pub fn raster_table(totals: &Totals, source_triangles: u64) -> Vec<String> {
    let mut lines = vec![format!(
        "raster ladder over {} parts ({source_triangles} source triangles): error -> resident meshlet bytes, triangles across LODs, bytes/triangle, share of the page budget",
        totals.parts
    )];
    for (rung, error) in RASTER_LADDER.iter().enumerate() {
        let bytes = totals.raster_bytes[rung];
        let triangles = totals.raster_triangles[rung];
        lines.push(format!(
            "  --raster-error {error:<6} {:>12} {triangles:>14} tri {:>6.1} B/tri {:>7.1}% of budget{}",
            human_bytes(bytes),
            if triangles == 0 {
                0.0
            } else {
                bytes as f64 / triangles as f64
            },
            bytes as f64 * 100.0 / MESHLET_PAGE_BUDGET as f64,
            if bytes > MESHLET_PAGE_BUDGET {
                " (over)"
            } else {
                ""
            }
        ));
    }
    lines
}

/// One line per BLAS rung, with the input-geometry estimate.
pub fn blas_table(totals: &Totals) -> Vec<String> {
    let mut lines = vec![format!(
        "BLAS ladder: error -> cut triangles, vertices, input geometry bytes ({BLAS_VERTEX_BYTES} B/vertex + {BLAS_INDEX_BYTES} B/index; the acceleration structure itself is extra)"
    )];
    for (rung, error) in BLAS_LADDER.iter().enumerate() {
        lines.push(format!(
            "  --raytracing-error {error:<5} {:>14} tri {:>14} vtx {:>12}",
            totals.blas_triangles[rung],
            totals.blas_vertices[rung],
            human_bytes(totals.blas_bytes(rung))
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lod::tests::torus_meshlet_mesh;

    #[test]
    fn totals_sum_every_rung() {
        let mut totals = Totals::new(2, 2);
        totals.add(&PartSample {
            raster: vec![(100, 10), (60, 6)],
            blas: vec![(4, 6), (2, 4)],
        });
        totals.add(&PartSample {
            raster: vec![(50, 5), (40, 4)],
            blas: vec![(1, 3), (1, 3)],
        });
        assert_eq!(totals.parts, 2);
        assert_eq!(totals.raster_bytes, vec![150, 100]);
        assert_eq!(totals.raster_triangles, vec![15, 10]);
        assert_eq!(totals.blas_triangles, vec![5, 3]);
        assert_eq!(totals.blas_vertices, vec![9, 7]);
        assert_eq!(totals.blas_bytes(0), 9 * 32 + 5 * 3 * 4);
        assert_eq!(totals.blas_bytes(1), 7 * 32 + 3 * 3 * 4);
    }

    #[test]
    fn ladders_are_monotonic_on_a_real_mesh() {
        let full = torus_meshlet_mesh();
        let sample = measure_part(&full, &RASTER_LADDER, &BLAS_LADDER);
        assert_eq!(sample.raster.len(), RASTER_LADDER.len());
        assert_eq!(sample.blas.len(), BLAS_LADDER.len());
        // Rung zero is the whole mesh (compaction aside).
        assert_eq!(sample.raster[0].1, full.triangle_count() as u64);
        assert!(sample.raster[0].0 <= full.packed_byte_len() as u64);
        for pair in sample.raster.windows(2) {
            assert!(pair[1].0 <= pair[0].0, "bytes grew: {:?}", sample.raster);
            assert!(
                pair[1].1 <= pair[0].1,
                "triangles grew: {:?}",
                sample.raster
            );
        }
        for pair in sample.blas.windows(2) {
            assert!(pair[1].0 <= pair[0].0, "cut grew: {:?}", sample.blas);
        }
        // A torus this dense simplifies well below its source at 3.2 cm.
        assert!(sample.raster.last().unwrap().1 < sample.raster[0].1);

        let mut totals = Totals::new(RASTER_LADDER.len(), BLAS_LADDER.len());
        totals.add(&sample);
        totals.add(&sample);
        assert_eq!(totals.raster_bytes[0], sample.raster[0].0 * 2);
        let raster = raster_table(&totals, 2 * 128 * 64 * 2);
        assert_eq!(raster.len(), RASTER_LADDER.len() + 1);
        assert!(raster[1].contains("--raster-error 0 "));
        let blas = blas_table(&totals);
        assert_eq!(blas.len(), BLAS_LADDER.len() + 1);
        assert!(blas[1].contains("--raytracing-error 0.02"));
    }
}
