#!/usr/bin/env python3
"""Partition a Zorah mesh-description intermediate into bounded runtime meshes.

The input arrays are memory mapped.  A first pass assigns triangles to a
material/spatial cell and appends only their u32 IDs to small bucket files.  A
second pass turns each bucket into chunks no larger than ``--triangles`` and
deduplicates seam vertices within that chunk.  Peak memory is therefore tied
to the requested partition size rather than to Zorah's largest 32M-triangle
source mesh.

Each partition is emitted as one tangent-free, self-contained standard GLB.
The offline Rust packer builds the raster meshlet hierarchy and selects a
fixed-error LOD from it for Solari's hardware BLAS input. These loose files
exist only in the resumable conversion work tree; the final runtime output
consists of bundle assets.

Positions/normals are converted from UE (X forward, Y right, Z up, cm) to
Bevy (X right, Y up, -Z forward, metres). UE MeshDescription triangles are
clockwise in their left-handed coordinate system, so the basis reflection
itself produces the counter-clockwise winding expected by glTF and Bevy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

def require_numpy() -> None:
    if np is None:
        raise RuntimeError(
            "numpy is required; run with `sfw uv run --with numpy python ...`"
        )


def count(manifest: dict, domain: str) -> int:
    return int(manifest["elements"][domain]["element_count"])


def source_arrays(
    source: Path,
) -> tuple[dict, object, object, object, object, object, object]:
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("format") != "zorah-mesh-description-v2":
        raise ValueError("input is not a zorah-mesh-description-v2 directory")
    positions = np.memmap(
        source / "positions.f32x3", dtype="<f4", mode="r", shape=(count(manifest, "Vertices"), 3)
    )
    vi_count = count(manifest, "VertexInstances")
    triangles_count = count(manifest, "Triangles")
    vi_vertices = np.memmap(
        source / "vertex_instance_vertices.i32", dtype="<i4", mode="r", shape=(vi_count,)
    )
    vi_normals = np.memmap(
        source / "vertex_instance_normals.f32x3", dtype="<f4", mode="r", shape=(vi_count, 3)
    )
    vi_uv0 = np.memmap(
        source / "vertex_instance_uv0.f32x2", dtype="<f4", mode="r", shape=(vi_count, 2)
    )
    triangles = np.memmap(
        source / "triangle_vertex_instances.i32x3",
        dtype="<i4",
        mode="r",
        shape=(triangles_count, 3),
    )
    materials = np.memmap(
        source / "triangle_materials.i32", dtype="<i4", mode="r", shape=(triangles_count,)
    )
    return (
        manifest,
        positions,
        vi_vertices,
        vi_normals,
        vi_uv0,
        triangles,
        materials,
    )


class BucketFiles:
    def __init__(self, directory: Path, limit: int = 128):
        self.directory = directory
        self.limit = limit
        self.handles: OrderedDict[int, object] = OrderedDict()
        self.keys: set[int] = set()

    def append(self, key: int, ids: object) -> None:
        handle = self.handles.pop(key, None)
        if handle is None:
            if len(self.handles) >= self.limit:
                _, old = self.handles.popitem(last=False)
                old.close()
            handle = (self.directory / f"{key:016x}.ids").open("ab")
        self.handles[key] = handle
        self.keys.add(key)
        ids.astype("<u4", copy=False).tofile(handle)

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def map_ue_vectors(values: object, scale: float) -> object:
    result = np.empty_like(values, dtype="<f4")
    result[:, 0] = values[:, 1] * scale
    result[:, 1] = values[:, 2] * scale
    result[:, 2] = -values[:, 0] * scale
    return result


def gltf_document(
    name: str,
    material: int,
    out_positions: object,
    out_normals: object,
    out_tangents: object | None,
    out_uv0: object,
    out_indices: object,
) -> dict:
    positions_offset = 0
    normals_offset = out_positions.nbytes
    tangents_offset = normals_offset + out_normals.nbytes
    uv_offset = tangents_offset + (0 if out_tangents is None else out_tangents.nbytes)
    indices_offset = uv_offset + out_uv0.nbytes
    aabb_min = out_positions.min(axis=0)
    aabb_max = out_positions.max(axis=0)
    binary_length = indices_offset + out_indices.nbytes
    buffer_views = [
        {"buffer": 0, "byteOffset": positions_offset, "byteLength": out_positions.nbytes, "target": 34962},
        {"buffer": 0, "byteOffset": normals_offset, "byteLength": out_normals.nbytes, "target": 34962},
    ]
    accessors = [
        {
            "bufferView": 0,
            "componentType": 5126,
            "count": len(out_positions),
            "type": "VEC3",
            "min": aabb_min.tolist(),
            "max": aabb_max.tolist(),
        },
        {"bufferView": 1, "componentType": 5126, "count": len(out_normals), "type": "VEC3"},
    ]
    attributes = {"POSITION": 0, "NORMAL": 1}
    if out_tangents is not None:
        attributes["TANGENT"] = len(accessors)
        buffer_views.append(
            {"buffer": 0, "byteOffset": tangents_offset, "byteLength": out_tangents.nbytes, "target": 34962}
        )
        accessors.append(
            {"bufferView": len(buffer_views) - 1, "componentType": 5126, "count": len(out_tangents), "type": "VEC4"}
        )
    attributes["TEXCOORD_0"] = len(accessors)
    buffer_views.append(
        {"buffer": 0, "byteOffset": uv_offset, "byteLength": out_uv0.nbytes, "target": 34962}
    )
    accessors.append(
        {"bufferView": len(buffer_views) - 1, "componentType": 5126, "count": len(out_uv0), "type": "VEC2"}
    )
    index_accessor = len(accessors)
    buffer_views.append(
        {"buffer": 0, "byteOffset": indices_offset, "byteLength": out_indices.nbytes, "target": 34963}
    )
    accessors.append(
        {"bufferView": len(buffer_views) - 1, "componentType": 5125, "count": len(out_indices), "type": "SCALAR"}
    )
    return {
        "asset": {"version": "2.0", "generator": "Zorah partition_mesh.py"},
        "buffers": [{"byteLength": binary_length}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "meshes": [{
            "name": name,
            "extras": {"zorah_material_slot": material},
            "primitives": [{
                "attributes": attributes,
                "indices": index_accessor,
                "mode": 4,
            }],
        }],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }


def write_glb(
    path: Path,
    name: str,
    material: int,
    out_positions: object,
    out_normals: object,
    out_tangents: object | None,
    out_uv0: object,
    out_indices: object,
) -> str:
    """Write a self-contained glTF 2.0 binary without buffering its payload.

    Returns the SHA-256 of the written file, digested from the same bytes as
    they are emitted so the manifest never re-reads the multi-MiB result.
    """
    document = gltf_document(
        name, material, out_positions, out_normals, out_tangents, out_uv0, out_indices
    )
    json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)
    binary_length = int(document["buffers"][0]["byteLength"])
    binary_padding = -binary_length % 4
    total_length = 12 + 8 + len(json_bytes) + 8 + binary_length + binary_padding
    digest = hashlib.sha256()
    arrays = [out_positions, out_normals, out_uv0, out_indices]
    if out_tangents is not None:
        arrays.insert(2, out_tangents)
    with path.open("xb") as stream:
        for header in (
            struct.pack("<4sII", b"glTF", 2, total_length),
            struct.pack("<II", len(json_bytes), 0x4E4F534A),
            json_bytes,
            struct.pack("<II", binary_length + binary_padding, 0x004E4942),
        ):
            stream.write(header)
            digest.update(header)
        for array in arrays:
            contiguous = np.ascontiguousarray(array)
            contiguous.tofile(stream)
            digest.update(contiguous)
        stream.write(b"\0" * binary_padding)
        digest.update(b"\0" * binary_padding)
    return digest.hexdigest()


def read_glb(path: Path) -> tuple[dict, bytes]:
    """Read the JSON and logical BIN payload from a converter-produced GLB."""
    encoded = path.read_bytes()
    if len(encoded) < 20:
        raise ValueError(f"truncated GLB: {path}")
    magic, version, total_length = struct.unpack_from("<4sII", encoded)
    if magic != b"glTF" or version != 2 or total_length != len(encoded):
        raise ValueError(f"invalid GLB header: {path}")
    json_length, json_kind = struct.unpack_from("<II", encoded, 12)
    if json_kind != 0x4E4F534A:
        raise ValueError(f"GLB has no leading JSON chunk: {path}")
    json_start = 20
    json_end = json_start + json_length
    document = json.loads(encoded[json_start:json_end].decode("utf-8"))
    binary_length, binary_kind = struct.unpack_from("<II", encoded, json_end)
    if binary_kind != 0x004E4942:
        raise ValueError(f"GLB has no BIN chunk: {path}")
    binary_start = json_end + 8
    logical_length = int(document["buffers"][0]["byteLength"])
    if logical_length > binary_length or binary_start + binary_length != len(encoded):
        raise ValueError(f"invalid GLB BIN chunk: {path}")
    return document, encoded[binary_start : binary_start + logical_length]


def repair_zero_normals(
    out_normals: object, valid: object, out_positions: object, faces: object
) -> None:
    """Replace zero-length source normals with the area-weighted face normal.

    glTF requires unit normals and MeshletMesh::from_mesh octahedral-encodes
    them, where a zero vector decodes to an arbitrary fixed direction.
    """
    affected = faces[~valid[faces].all(axis=1)]
    corners = np.asarray(out_positions[affected], dtype=np.float64)
    face_normals = np.cross(
        corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    )
    accumulated = np.zeros(out_normals.shape, dtype=np.float64)
    for corner in range(3):
        np.add.at(accumulated, affected[:, corner], face_normals)
    lengths = np.linalg.norm(accumulated, axis=1)
    usable = lengths > 0.0
    # Zero area everywhere around the vertex leaves no direction to recover.
    accumulated[~usable] = (0.0, 1.0, 0.0)
    accumulated[usable] /= lengths[usable, None]
    out_normals[~valid] = accumulated[~valid]


def write_partition(
    destination: Path,
    stem: str,
    material: int,
    triangle_ids: object,
    positions: object,
    vi_vertices: object,
    vi_normals: object,
    vi_uv0: object,
    triangles: object,
) -> dict:
    corner_vi = np.asarray(triangles[triangle_ids], dtype="<i4").reshape(-1)
    corner_position = np.asarray(vi_vertices[corner_vi], dtype="<i4")

    # Exact seam key: source position identity plus bit-exact normal and UV.
    keys = np.empty(
        corner_vi.size,
        dtype=np.dtype([
            ("p", "<i4"),
            ("n", "<u4", (3,)),
            ("uv", "<u4", (2,)),
        ]),
    )
    keys["p"] = corner_position
    keys["n"] = np.asarray(vi_normals[corner_vi]).view("<u4")
    keys["uv"] = np.asarray(vi_uv0[corner_vi]).view("<u4")
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)

    out_positions = map_ue_vectors(np.asarray(positions[corner_position[first]]), 0.01)
    out_normals = map_ue_vectors(np.asarray(vi_normals[corner_vi[first]]), 1.0)
    out_uv0 = np.asarray(vi_uv0[corner_vi[first]], dtype="<f4")
    faces = np.ascontiguousarray(inverse.astype("<u4", copy=False).reshape(-1, 3))

    lengths = np.linalg.norm(out_normals, axis=1)
    nonzero = lengths > 0.0
    out_normals[nonzero] /= lengths[nonzero, None]
    if not nonzero.all():
        repair_zero_normals(out_normals, nonzero, out_positions, faces)
    out_indices = faces.reshape(-1)

    meshlet_name = f"{stem}.glb"
    meshlet_sha256 = write_glb(
        destination / meshlet_name,
        stem,
        material,
        out_positions,
        out_normals,
        None,
        out_uv0,
        out_indices,
    )
    return {
        "geometry": meshlet_name,
        "mesh": f"{meshlet_name}#Mesh0/Primitive0",
        "meshlet": meshlet_name,
        "material_slot": material,
        "triangles": len(out_indices) // 3,
        "vertices": len(out_positions),
        "aabb_min": out_positions.min(axis=0).tolist(),
        "aabb_max": out_positions.max(axis=0).tolist(),
        "uv_min": out_uv0.min(axis=0).tolist(),
        "uv_max": out_uv0.max(axis=0).tolist(),
        "meshlet_sha256": meshlet_sha256,
        "blas_triangles": len(out_indices) // 3,
        "blas_vertices": len(out_positions),
        "blas_achieved_error": 0.0,
    }


def partition(
    source: Path,
    destination: Path,
    target: int,
    scan_chunk: int,
) -> None:
    require_numpy()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    if target < 128:
        raise ValueError("--triangles must be at least 128")
    (
        manifest,
        positions,
        vi_vertices,
        vi_normals,
        vi_uv0,
        triangles,
        materials,
    ) = source_arrays(source)
    triangle_count = len(triangles)
    # Solari stores an emissive mesh light's triangle id in 16 bits. These are
    # the Zorah source slots whose material instances enable emission; cap just
    # those partitions instead of multiplying the entire scene's entity count.
    emissive_slots = {
        index
        for index, name in enumerate(manifest.get("material_slots", []))
        if name.casefold() in {
            "mi_firewood_a",
            "mi_courtyard_lampsphericalmetal_a_glass",
            "mi_courtyard_lampsphericalmetal_a_glass_1",
        }
    }

    position_min = np.asarray(positions).min(axis=0).astype(np.float64)
    position_max = np.asarray(positions).max(axis=0).astype(np.float64)
    extent = np.maximum(position_max - position_min, 1e-9)
    desired_cells = max(1, math.ceil(triangle_count / target))
    grid = max(1, math.ceil(desired_cells ** (1.0 / 3.0)))

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.buckets.", dir=parent))
    buckets = BucketFiles(temp)
    try:
        for start in range(0, triangle_count, scan_chunk):
            end = min(start + scan_chunk, triangle_count)
            tri = np.asarray(triangles[start:end])
            vertex_ids = np.asarray(vi_vertices[tri])
            centers = np.asarray(positions[vertex_ids], dtype=np.float64).mean(axis=1)
            cells = np.floor((centers - position_min) / extent * grid).astype(np.int64)
            np.clip(cells, 0, grid - 1, out=cells)
            spatial = cells[:, 0] + grid * (cells[:, 1] + grid * cells[:, 2])
            mats = np.asarray(materials[start:end], dtype=np.int64)
            if np.any(mats < 0):
                raise ValueError(f"negative material slot in triangles {start}..{end}")
            keys = (mats.astype(np.uint64) << np.uint64(32)) | spatial.astype(np.uint64)
            ids = np.arange(start, end, dtype=np.uint32)
            # A stable sort keeps each bucket in ascending triangle order, and
            # np.unique yields the keys ascending, so the bucket files match a
            # per-key mask scan byte for byte at a fraction of the cost.
            order = np.argsort(keys, kind="stable")
            sorted_ids = ids[order]
            unique_keys, starts = np.unique(keys[order], return_index=True)
            stops = np.append(starts[1:], len(sorted_ids))
            for key, first, last in zip(unique_keys, starts, stops):
                buckets.append(int(key), sorted_ids[first:last])
            print(f"ZORAH_PARTITION_SCAN triangles={end}/{triangle_count}", flush=True)
        buckets.close()

        destination.mkdir()
        outputs = []
        partition_id = 0
        for key in sorted(buckets.keys):
            material = key >> 32
            ids_path = temp / f"{key:016x}.ids"
            ids = np.memmap(ids_path, dtype="<u4", mode="r")
            partition_target = min(target, 60_000) if material in emissive_slots else target
            for offset in range(0, len(ids), partition_target):
                triangle_ids = np.asarray(
                    ids[offset : offset + partition_target], dtype=np.int64
                )
                stem = f"part-{partition_id:06d}"
                outputs.append(
                    write_partition(
                        destination,
                        stem,
                        material,
                        triangle_ids,
                        positions,
                        vi_vertices,
                        vi_normals,
                        vi_uv0,
                        triangles,
                    )
                )
                partition_id += 1
                print(
                    f"ZORAH_PARTITION_WRITE partitions={partition_id} "
                    f"triangles={outputs[-1]['triangles']} vertices={outputs[-1]['vertices']}",
                    flush=True,
                )
            # Windows refuses to delete a file that still has a live mapping, and
            # the last bucket's would otherwise outlive the temporary directory.
            del ids

        output_triangles = sum(part["triangles"] for part in outputs)
        if output_triangles != triangle_count:
            raise ValueError(
                f"partitioning emitted {output_triangles} triangles from a "
                f"{triangle_count}-triangle source mesh"
            )

        output_manifest = {
            "format": "zorah-partitioned-mesh-v4",
            "source_manifest": manifest,
            "target_triangles": target,
            "spatial_grid": [grid, grid, grid],
            "source_bounds_ue_cm": {
                "min": position_min.tolist(),
                "max": position_max.tolist(),
            },
            "source_triangles": triangle_count,
            "output_triangles": output_triangles,
            "partitions": outputs,
        }
        (destination / "manifest.json").write_text(
            json.dumps(output_manifest, indent=2) + "\n"
        )
        print(
            f"ZORAH_PARTITION_DONE partitions={len(outputs)} triangles={triangle_count} "
            f"output={destination}",
            flush=True,
        )
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination)
        raise
    finally:
        buckets.close()
        shutil.rmtree(temp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--triangles", type=int, default=250_000)
    parser.add_argument("--scan-chunk", type=int, default=1_000_000)
    args = parser.parse_args()
    partition(
        args.source.resolve(),
        args.destination.resolve(),
        args.triangles,
        args.scan_chunk,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ZORAH_PARTITION_ERROR {error}", file=sys.stderr)
        raise SystemExit(1)
