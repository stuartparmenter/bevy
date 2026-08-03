#!/usr/bin/env python3
"""Verify the compact Zorah runtime asset tree and every bundle reference."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


MAGIC = b"ZORAHB01"
VERSION = 2
MAX_RUNTIME_FILES = 50
EXPECTED_LIGHT_MINIMUMS = {
    "GreenHouse_Level": {"directional": 1, "sky": 1},
    "Restir_Level": {"directional": 1, "spot": 1, "sky": 1},
    "ThroneRoom_Level": {"directional": 1, "point": 1, "spot": 26, "sky": 1},
}
EXPECTED_POST_PROCESS_EXPOSURE = {
    "GreenHouse_Level": (4.0, 5.0, 0.99),
    "Restir_Level": (None, None, -2.5),
    "ThroneRoom_Level": (8.0, 8.0, None),
}
EXPECTED_UNRESOLVED_COMPONENTS = {
    "GreenHouse_Level": 27,
    "Restir_Level": 0,
    "ThroneRoom_Level": 0,
}
EXPECTED_ATMOSPHERE_OVERRIDES = {
    "GreenHouse_Level": set(),
    "Restir_Level": {"rayleigh_scattering_per_km", "sky_luminance_factor"},
    "ThroneRoom_Level": {
        "rayleigh_scattering_per_km",
        "rayleigh_scattering_scale",
        "mie_scattering_per_km",
        "mie_exponential_distribution_km",
    },
}
EXPECTED_HEIGHT_FOG_OVERRIDES = {
    "GreenHouse_Level": {"enable_volumetric_fog", "volumetric_fog_distance_cm"},
    "Restir_Level": {
        "fog_density",
        "fog_height_falloff",
        "enable_volumetric_fog",
        "volumetric_fog_albedo",
        "volumetric_fog_distance_cm",
    },
    "ThroneRoom_Level": {"enable_volumetric_fog", "volumetric_fog_distance_cm"},
}
def load(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing runtime manifest {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def bundle_entries(path: Path) -> dict[str, str]:
    with path.open("rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError(f"wrong bundle magic in {path}")
        version, index_length = struct.unpack("<IQ", stream.read(12))
        if version != VERSION:
            raise ValueError(f"bundle {path} has version {version}, expected {VERSION}")
        index = json.loads(stream.read(index_length))
        if index.get("format_version") != VERSION:
            raise ValueError(f"bundle {path} index has the wrong version")
        entries: dict[str, str] = {}
        payload_length = 0
        for entry in index.get("entries", []):
            label = entry["label"]
            if label in entries:
                raise ValueError(f"duplicate label {label} in {path}")
            entries[label] = entry["kind"]
            payload_length += int(entry["byte_length"])
        payload_start = stream.tell()
        stream.seek(0, 2)
        if stream.tell() - payload_start != payload_length:
            raise ValueError(f"bundle payload length mismatch in {path}")
        return entries


def split_reference(reference: str) -> tuple[Path, str]:
    path, separator, label = reference.partition("#")
    if not separator or not path.endswith(".zorah_bundle") or not label:
        raise ValueError(f"invalid bundled asset reference {reference}")
    return Path(path), label


def require_post_process_exposure(scene: dict) -> None:
    records = [
        actor["post_process"]
        for actor in scene.get("actors", [])
        if actor.get("post_process") is not None
        and actor["post_process"].get("enabled", True)
        and actor["post_process"].get("unbound", False)
        and float(actor["post_process"].get("blend_weight", 1.0)) > 0.0
    ]
    if len(records) != 1:
        raise ValueError(
            f"scene {scene['level']} must have one active unbound post-process volume, "
            f"found {len(records)}"
        )
    record = records[0]
    if record.get("auto_exposure_method") != "AEM_Histogram":
        raise ValueError(f"scene {scene['level']} lost histogram auto exposure")
    expected = EXPECTED_POST_PROCESS_EXPOSURE[scene["level"]]
    for field, expected_value in zip(
        (
            "auto_exposure_min_ev100",
            "auto_exposure_max_ev100",
            "auto_exposure_bias",
        ),
        expected,
    ):
        value = record.get(field)
        if expected_value is None:
            if value is not None:
                raise ValueError(
                    f"scene {scene['level']} unexpectedly overrides {field}: {value}"
                )
            continue
        if value is None or not math.isfinite(float(value)) or not math.isclose(
            float(value), expected_value, abs_tol=1e-5
        ):
            raise ValueError(
                f"scene {scene['level']} has {field}={value}, expected {expected_value}"
            )


def require_atmosphere_and_height_fog(scene: dict) -> None:
    atmosphere_records = [
        actor["atmosphere"]
        for actor in scene.get("actors", [])
        if actor.get("type") == "SkyAtmosphere" and actor.get("atmosphere") is not None
    ]
    fog_records = [
        actor["height_fog"]
        for actor in scene.get("actors", [])
        if actor.get("type") == "ExponentialHeightFog"
        and actor.get("height_fog") is not None
    ]
    if len(atmosphere_records) != 1 or len(fog_records) != 1:
        raise ValueError(
            f"scene {scene['level']} must contain one atmosphere and one height-fog record"
        )
    for field in EXPECTED_ATMOSPHERE_OVERRIDES[scene["level"]]:
        if atmosphere_records[0].get(field) is None:
            raise ValueError(f"scene {scene['level']} lost atmosphere override {field}")
    for field in EXPECTED_HEIGHT_FOG_OVERRIDES[scene["level"]]:
        if fog_records[0].get(field) is None:
            raise ValueError(f"scene {scene['level']} lost height-fog override {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_root", type=Path)
    args = parser.parse_args()
    root = args.asset_root.resolve()

    geometry = load(root / "geometry.json")
    textures = load(root / "textures.exported.json")
    materials = load(root / "materials.json")
    scenes = [
        load(root / "scenes" / f"{level}.json")
        for level in ("GreenHouse_Level", "Restir_Level", "ThroneRoom_Level")
    ]
    for scene in scenes:
        if scene.get("format") != "zorah-scene-manifest-v3":
            raise ValueError(f"scene {scene.get('level')} predates exported UE lights")
        require_post_process_exposure(scene)
        require_atmosphere_and_height_fog(scene)
        unresolved = [
            component
            for actor in scene.get("actors", [])
            for component in actor.get("components", [])
            if component.get("mesh") is None
        ]
        if len(unresolved) != EXPECTED_UNRESOLVED_COMPONENTS[scene["level"]]:
            raise ValueError(
                f"scene {scene['level']} has {len(unresolved)} unresolved static-mesh "
                f"components, expected {EXPECTED_UNRESOLVED_COMPONENTS[scene['level']]}"
            )
        if any(
            component.get("missing_reason") != "source-package-not-in-sample"
            for component in unresolved
        ):
            raise ValueError(
                f"scene {scene['level']} has an unexplained unresolved mesh component"
            )
        for actor in scene.get("actors", []):
            if "lights" not in actor:
                raise ValueError(f"actor lacks light array: {actor.get('name')}")
        light_counts: dict[str, int] = {}
        for actor in scene.get("actors", []):
            for light in actor["lights"]:
                kind = light.get("type")
                light_counts[kind] = light_counts.get(kind, 0) + 1
        for kind, minimum in EXPECTED_LIGHT_MINIMUMS[scene["level"]].items():
            if light_counts.get(kind, 0) < minimum:
                raise ValueError(
                    f"scene {scene['level']} exported too few {kind} lights: "
                    f"{light_counts.get(kind, 0)} < {minimum}"
                )
    bundles = sorted((root / "bundles").glob("*.zorah_bundle"))
    if not bundles:
        raise ValueError("runtime tree contains no Zorah bundles")
    indices = {path.relative_to(root): bundle_entries(path) for path in bundles}

    referenced: list[tuple[str, str]] = []
    partitions = 0
    meshes_by_object = {mesh["object"]: mesh for mesh in geometry.get("meshes", [])}
    for mesh in geometry.get("meshes", []):
        if "parts_manifest" in mesh:
            raise ValueError("runtime geometry still references loose partition manifests")
        for partition in mesh.get("partitions", []):
            partitions += 1
            referenced.append((partition["geometry"], "meshlet_blas"))
            referenced.append((partition["meshlet"], "meshlet"))
    for texture in textures.get("exported", []):
        referenced.append((texture["output"], "image"))

    for scene in scenes:
        scene_meshes = {
            component.get("mesh")
            for actor in scene.get("actors", [])
            for component in actor.get("components", [])
            if component.get("mesh") in meshes_by_object
        }
        scene_geometry_bundles = {
            split_reference(partition[key])[0]
            for mesh_object in scene_meshes
            for partition in meshes_by_object[mesh_object].get("partitions", [])
            for key in ("geometry", "meshlet")
        }
        for mesh_object, mesh in meshes_by_object.items():
            if mesh_object in scene_meshes:
                continue
            for partition in mesh.get("partitions", []):
                for key in ("geometry", "meshlet"):
                    bundle, _ = split_reference(partition[key])
                    if bundle in scene_geometry_bundles:
                        raise ValueError(
                            f"scene {scene['level']} geometry shard also contains "
                            f"unused mesh {mesh_object}: {bundle}"
                        )
    for reference, expected_kind in referenced:
        path, label = split_reference(reference)
        entries = indices.get(path)
        if entries is None:
            raise ValueError(f"reference points at missing bundle {path}")
        kind = entries.get(label)
        if kind != expected_kind:
            raise ValueError(
                f"reference {reference} has kind {kind!r}, expected {expected_kind!r}"
            )

    files = [path for path in root.rglob("*") if path.is_file()]
    if len(files) > MAX_RUNTIME_FILES:
        raise ValueError(
            f"runtime tree has {len(files)} physical files, expected at most {MAX_RUNTIME_FILES}"
        )
    forbidden = [
        path for path in files
        if path.suffix.lower() in {".glb", ".png", ".meta", ".meshlet_mesh"}
    ]
    if forbidden:
        raise ValueError(f"runtime tree retains loose assets: {forbidden[:4]}")

    print(
        "ZORAH_BUNDLE_VERIFY_OK "
        f"files={len(files)} bundles={len(bundles)} entries={sum(map(len, indices.values()))} "
        f"partitions={partitions} textures={len(textures.get('exported', []))} "
        f"materials={len(materials.get('materials', []))} actors={sum(len(scene.get('actors', [])) for scene in scenes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
