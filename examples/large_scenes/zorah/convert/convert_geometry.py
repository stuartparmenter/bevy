#!/usr/bin/env python3
"""Batch-convert the meshes referenced by Zorah scene manifests.

This is intentionally Zorah-specific.  It resolves `/Game` object paths into
the known source-project layout, streams each uncooked FMeshDescription, and
partitions it into bounded self-contained tangent-free GLB assets.  Material
interfaces are assigned later from the exact UStaticMesh.StaticMaterials data;
mesh-description slot names are never used to guess a material package.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from mesh_description import extract
from partition_mesh import partition, write_glb


ENGINE_PRIMITIVES = {
    "/Engine/BasicShapes/Cube.Cube": "cube",
    "/Engine/EngineMeshes/Cube.Cube": "cube",
    "/Engine/BasicShapes/Plane.Plane": "plane",
}


def builtin_arrays(kind: str) -> tuple[object, object, object, object, object]:
    """Generate UE's metre-wide BasicShapes in converted Bevy coordinates."""
    if kind == "plane":
        positions = np.asarray([
            [-0.5, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.0, -0.5],
            [-0.5, 0.0, -0.5],
        ], dtype="<f4")
        normals = np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype="<f4"), (4, 1))
        tangents = np.tile(np.asarray([[1.0, 0.0, 0.0, -1.0]], dtype="<f4"), (4, 1))
        uv0 = np.asarray([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype="<f4")
        indices = np.asarray([0, 1, 2, 0, 2, 3], dtype="<u4")
        return positions, normals, tangents, uv0, indices
    if kind != "cube":
        raise ValueError(f"unknown engine primitive {kind}")

    faces = [
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        ((0, -1, 0), (0, 0, -1), (1, 0, 0)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
    ]
    positions: list[object] = []
    normals: list[object] = []
    tangents: list[object] = []
    uv0: list[tuple[float, float]] = []
    indices: list[int] = []
    for normal_values, tangent_values, bitangent_values in faces:
        normal = np.asarray(normal_values, dtype="<f4")
        tangent = np.asarray(tangent_values, dtype="<f4")
        bitangent = np.asarray(bitangent_values, dtype="<f4")
        center = normal * 0.5
        first = len(positions)
        positions.extend([
            center - tangent * 0.5 - bitangent * 0.5,
            center + tangent * 0.5 - bitangent * 0.5,
            center + tangent * 0.5 + bitangent * 0.5,
            center - tangent * 0.5 + bitangent * 0.5,
        ])
        normals.extend([normal] * 4)
        tangents.extend([np.append(tangent, 1.0)] * 4)
        uv0.extend([(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)])
        indices.extend([first, first + 1, first + 2, first, first + 2, first + 3])
    return (
        np.asarray(positions, dtype="<f4"),
        np.asarray(normals, dtype="<f4"),
        np.asarray(tangents, dtype="<f4"),
        np.asarray(uv0, dtype="<f4"),
        np.asarray(indices, dtype="<u4"),
    )


def write_builtin_mesh(destination: Path, object_path: str) -> dict[str, object]:
    kind = ENGINE_PRIMITIVES[object_path]
    identifier = hashlib.sha256(object_path.encode()).hexdigest()[:16]
    parts = destination / "geometry" / identifier / "parts"
    parts.mkdir(parents=True)
    positions, normals, _tangents, uv0, indices = builtin_arrays(kind)
    meshlet_name = "part-000000.glb"
    meshlet_sha256 = write_glb(
        parts / meshlet_name,
        f"zorah-engine-{kind}",
        0,
        positions,
        normals,
        None,
        uv0,
        indices,
    )
    partition_record = {
        "geometry": meshlet_name,
        "mesh": f"{meshlet_name}#Mesh0/Primitive0",
        "meshlet": meshlet_name,
        "material_slot": 0,
        "triangles": len(indices) // 3,
        "vertices": len(positions),
        "aabb_min": positions.min(axis=0).tolist(),
        "aabb_max": positions.max(axis=0).tolist(),
        "uv_min": uv0.min(axis=0).tolist(),
        "uv_max": uv0.max(axis=0).tolist(),
        "meshlet_sha256": meshlet_sha256,
        "blas_triangles": len(indices) // 3,
        "blas_vertices": len(positions),
        "blas_achieved_error": 0.0,
    }
    parts_manifest = {
        "format": "zorah-partitioned-mesh-v4",
        "source_manifest": {"format": "zorah-engine-primitive-v1", "object": object_path},
        "target_triangles": len(indices) // 3,
        "spatial_grid": [1, 1, 1],
        "source_bounds_ue_cm": None,
        "source_triangles": len(indices) // 3,
        "output_triangles": len(indices) // 3,
        "partitions": [partition_record],
    }
    (parts / "manifest.json").write_text(json.dumps(parts_manifest, indent=2) + "\n")
    return {
        "object": object_path,
        "source": "generated Zorah engine-content substitute",
        "asset_id": identifier,
        "parts_manifest": f"geometry/{identifier}/parts/manifest.json",
        "material_slots": [{
            "mesh": object_path,
            "slot": "0",
            "name": "DefaultMaterial",
            "material": None,
            "resolution": "authored-component-override-required",
        }],
    }


def package_from_object_path(content_root: Path, object_path: str) -> Path:
    if not object_path.startswith("/Game/"):
        raise ValueError(f"unsupported non-project mesh reference {object_path}")
    package = object_path[len("/Game/") :].split(".", 1)[0]
    return content_root / f"{package}.uasset"


def convert_mesh(
    content_root: Path,
    temporary: Path,
    object_path: str,
    triangles: int,
    scan_chunk: int,
    keep_intermediate: bool,
    capture: bool,
) -> dict[str, object]:
    """Extract and partition one mesh into its own geometry/<id> directory.

    This runs in a worker process, so it returns records instead of appending
    to the batch state, and buffers the phase progress the parent replays once
    the mesh is done rather than interleaving it with every other worker.
    """
    identifier = hashlib.sha256(object_path.encode()).hexdigest()[:16]
    mesh_root = temporary / "geometry" / identifier
    progress = io.StringIO()
    sink = contextlib.redirect_stdout(progress) if capture else contextlib.nullcontext()
    result: dict[str, object] = {"object": object_path}
    with sink:
        try:
            source = package_from_object_path(content_root, object_path)
            if not source.is_file():
                raise FileNotFoundError(f"source package is absent: {source}")
            intermediate = mesh_root / "source"
            source_manifest = extract(source, intermediate)
            partition(
                intermediate,
                mesh_root / "parts",
                target=triangles,
                scan_chunk=scan_chunk,
            )
            slots = list(source_manifest["material_slots"])
            if not keep_intermediate:
                shutil.rmtree(intermediate)
            result["record"] = {
                "object": object_path,
                "source": str(source),
                "asset_id": identifier,
                "parts_manifest": f"geometry/{identifier}/parts/manifest.json",
                "material_slots": [
                    {
                        "mesh": object_path,
                        "slot": str(slot_index),
                        "name": slot_name,
                        "material": None,
                        "resolution": "pending-exact-unreal-static-material",
                    }
                    for slot_index, slot_name in enumerate(slots)
                ],
            }
        except (EOFError, OSError, RuntimeError, ValueError) as error:
            # A mesh that failed contributes no partitions, and its extracted
            # arrays can reach tens of GiB; nothing should publish them.
            if not keep_intermediate:
                shutil.rmtree(mesh_root, ignore_errors=True)
            result["failure"] = {
                "mesh": object_path,
                "error_type": type(error).__name__,
                "message": str(error),
            }
    result["progress"] = progress.getvalue()
    return result


def report_mesh(result: dict[str, object], done: int, total: int) -> None:
    sys.stdout.write(str(result["progress"]))
    sys.stdout.flush()
    object_path = result["object"]
    failure = result.get("failure")
    if failure is not None:
        print(
            f"ZORAH_GEOMETRY_ERROR mesh={object_path} error={failure['message']}",
            file=sys.stderr,
            flush=True,
        )
        return
    record = result["record"]
    print(
        f"ZORAH_GEOMETRY_DONE {done}/{total} "
        f"mesh={object_path} slots={len(record['material_slots'])}",
        flush=True,
    )


def convert_meshes(
    content_root: Path,
    temporary: Path,
    selected_meshes: list[str],
    triangles: int,
    scan_chunk: int,
    keep_intermediate: bool,
    jobs: int,
) -> list[dict[str, object]]:
    """Convert every selected mesh, in parallel when more than one job is asked.

    Meshes are independent - each writes only its own geometry/<id> directory -
    so the parent just collects their records; callers sort them to keep the
    manifest independent of completion order.
    """
    total = len(selected_meshes)
    results: list[dict[str, object]] = []
    if jobs <= 1:
        for object_path in selected_meshes:
            results.append(
                convert_mesh(
                    content_root,
                    temporary,
                    object_path,
                    triangles,
                    scan_chunk,
                    keep_intermediate,
                    False,
                )
            )
            report_mesh(results[-1], len(results), total)
        return results

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(
                convert_mesh,
                content_root,
                temporary,
                object_path,
                triangles,
                scan_chunk,
                keep_intermediate,
                True,
            )
            for object_path in selected_meshes
        ]
        try:
            for future in as_completed(futures):
                results.append(future.result())
                report_mesh(results[-1], len(results), total)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return results


def load_scene_manifests(paths: list[Path]) -> tuple[list[dict], list[str], list[str]]:
    scenes = []
    meshes: set[str] = set()
    material_overrides: set[str] = set()
    for path in paths:
        scene = json.loads(path.read_text(encoding="utf-8"))
        if scene.get("format") != "zorah-scene-manifest-v5":
            raise ValueError(f"not a Zorah scene manifest: {path}")
        scenes.append({
            "level": scene["level"],
            "source": str(path.resolve()),
            "path": f"scenes/{scene['level']}.json",
            "actor_count": len(scene["actors"]),
        })
        meshes.update(scene["referenced_meshes"])
        for actor in scene["actors"]:
            for component in actor["components"]:
                material_overrides.update(component.get("override_materials", []))
    return scenes, sorted(meshes), sorted(material_overrides)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("scene_manifests", type=Path, nargs="+")
    parser.add_argument("--triangles", type=int, default=250_000)
    parser.add_argument("--scan-chunk", type=int, default=1_000_000)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="validation-only maximum mesh count; zero converts every reference",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="retain decompressed mesh-description arrays after partitioning",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help=(
            "mesh worker processes; each holds one partition scan, so peak "
            "memory is roughly --jobs times the --scan-chunk/--triangles cost"
        ),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    content_root = project_root / "Content"
    destination = args.destination.resolve()
    if destination.exists():
        parser.error(f"destination already exists: {destination}")
    if not content_root.is_dir():
        parser.error(f"project has no Content directory: {content_root}")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.jobs < 1:
        parser.error("--jobs must be at least one")

    scenes, meshes, override_materials = load_scene_manifests(args.scene_manifests)
    external_meshes = [mesh for mesh in meshes if not mesh.startswith("/Game/")]
    project_meshes = [mesh for mesh in meshes if mesh.startswith("/Game/")]
    selected_meshes = project_meshes[: args.limit or None]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    converted: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    material_resolutions: list[dict[str, str | None]] = []
    material_objects: set[str] = {
        material for material in override_materials if material.startswith("/Game/")
    }
    try:
        (temporary / "scenes").mkdir()
        for scene in scenes:
            shutil.copy2(scene["source"], temporary / str(scene["path"]))
        results = convert_meshes(
            content_root,
            temporary,
            selected_meshes,
            args.triangles,
            args.scan_chunk,
            args.keep_intermediate,
            args.jobs,
        )
        # Workers finish out of order; the manifest must not depend on that.
        for result in sorted(results, key=lambda result: str(result["object"])):
            record = result.get("record")
            if record is None:
                failures.append(result["failure"])
                continue
            converted.append(record)
            material_resolutions.extend(record["material_slots"])

        for object_path in external_meshes:
            if object_path not in ENGINE_PRIMITIVES:
                failures.append({
                    "mesh": object_path,
                    "error_type": "UnsupportedEngineContent",
                    "message": "no Zorah-local substitute is defined",
                })
                continue
            converted.append(write_builtin_mesh(temporary, object_path))
            print(f"ZORAH_GEOMETRY_BUILTIN mesh={object_path}", flush=True)

        manifest = {
            "format": "zorah-geometry-manifest-v3",
            "project_root": str(project_root),
            "scenes": scenes,
            "referenced_mesh_count": len(meshes),
            "project_mesh_count": len(project_meshes),
            "external_meshes": external_meshes,
            "generated_engine_meshes": sorted(
                mesh for mesh in external_meshes if mesh in ENGINE_PRIMITIVES
            ),
            "selected_mesh_count": len(selected_meshes),
            "triangle_partition_limit": args.triangles,
            "meshes": sorted(converted, key=lambda record: str(record["object"])),
            "material_resolutions": sorted(
                material_resolutions,
                key=lambda record: (str(record["mesh"]), int(str(record["slot"]))),
            ),
            "failures": failures,
        }
        (temporary / "geometry.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "material-input.json").write_text(
            json.dumps(sorted(material_objects), indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        f"ZORAH_GEOMETRY_BATCH_DONE referenced={len(meshes)} "
        f"converted={len(converted)} failures={len(failures)} output={destination}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
