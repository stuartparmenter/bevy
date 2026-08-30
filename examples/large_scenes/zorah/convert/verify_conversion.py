#!/usr/bin/env python3
"""Verify a complete Zorah asset tree before launching Bevy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from partition_mesh import read_glb
from texture_source import ATLAS_LAYOUT_VERSION, atlas_layout_matters


EXPECTED_LEVELS = {"GreenHouse_Level", "Restir_Level", "ThroneRoom_Level"}
EXPLICIT_RUNTIME_ENGINE_MATERIALS = {
    "/Engine/EngineDebugMaterials/BlackUnlitMaterial.BlackUnlitMaterial",
    "/Engine/EngineMaterials/WorldGridMaterial.WorldGridMaterial",
}
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
    "GreenHouse_Level": {
        "enable_volumetric_fog",
        "volumetric_fog_distance_cm",
        "volumetric_fog_extinction_scale",
        "volumetric_fog_near_fade_in_distance_cm",
        "volumetric_fog_scattering_distribution",
    },
    "Restir_Level": {
        "fog_density",
        "fog_height_falloff",
        "enable_volumetric_fog",
        "volumetric_fog_albedo",
        "volumetric_fog_distance_cm",
        "volumetric_fog_near_fade_in_distance_cm",
        "volumetric_fog_scattering_distribution",
    },
    "ThroneRoom_Level": {
        "enable_volumetric_fog",
        "volumetric_fog_distance_cm",
        "volumetric_fog_extinction_scale",
        "volumetric_fog_scattering_distribution",
    },
}
EXPECTED_RECOVERED_OODLE_BLOCKS = {
    "/Game/Assets/Environment/ThroneRoom_Cornice_C/Textures/"
    "T_ThroneRoom_Cornice_C1_Normal.T_ThroneRoom_Cornice_C1_Normal": [82],
}
MATERIAL_TEXTURE_ROLES = {
    "base_color": (
        "basecolor",
        "albedo",
        "diffuse",
        "diffusetexture",
        "basecolortexture",
        "marblebasecolor",
        "goldbasecolor",
    ),
    "normal": (
        "normal",
        "normalmap",
        "normaltexture",
        "marblenormal",
        "marblechippingnormal",
        "goldbasenormal",
    ),
    "orm": (
        "orm",
        "ors",
        "occlusionroughnessmetallic",
        "packedorm",
        "marbleorm",
        "goldbaseorm",
    ),
}


def load(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing manifest {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_files(root: Path, paths: list[Path]) -> None:
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"missing {len(missing)} converted files: {preview}")


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
    if len(atmosphere_records) != 1:
        raise ValueError(
            f"scene {scene['level']} must export one SkyAtmosphere component, "
            f"found {len(atmosphere_records)}"
        )
    if len(fog_records) != 1:
        raise ValueError(
            f"scene {scene['level']} must export one ExponentialHeightFog component, "
            f"found {len(fog_records)}"
        )
    atmosphere = atmosphere_records[0]
    fog = fog_records[0]
    if not atmosphere.get("visible", True) or atmosphere.get("hidden_in_game", False):
        raise ValueError(f"scene {scene['level']} exported a disabled atmosphere")
    if not fog.get("visible", True) or fog.get("hidden_in_game", False):
        raise ValueError(f"scene {scene['level']} exported disabled height fog")
    for field in EXPECTED_ATMOSPHERE_OVERRIDES[scene["level"]]:
        if atmosphere.get(field) is None:
            raise ValueError(f"scene {scene['level']} lost atmosphere override {field}")
    for field in EXPECTED_HEIGHT_FOG_OVERRIDES[scene["level"]]:
        if fog.get(field) is None:
            raise ValueError(f"scene {scene['level']} lost height-fog override {field}")


def require_runtime_vertex_format(document: dict, mesh_name: str) -> None:
    primitive = document["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]
    expected = {
        "POSITION": "VEC3",
        "NORMAL": "VEC3",
        "TANGENT": "VEC4",
        "TEXCOORD_0": "VEC2",
    }
    counts = set()
    for semantic, accessor_type in expected.items():
        if semantic not in attributes:
            raise ValueError(f"{mesh_name} lacks required {semantic} vertex data")
        accessor = document["accessors"][attributes[semantic]]
        if accessor["componentType"] != 5126 or accessor["type"] != accessor_type:
            raise ValueError(f"{mesh_name} has invalid {semantic} accessor")
        counts.add(int(accessor["count"]))
    if len(counts) != 1:
        raise ValueError(f"{mesh_name} vertex attribute counts disagree")


def require_meshlet_source_vertex_format(document: dict, mesh_name: str) -> None:
    primitive = document["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]
    if set(attributes) != {"POSITION", "NORMAL", "TEXCOORD_0"}:
        raise ValueError(
            f"{mesh_name} must have Bevy's exact meshlet-source vertex attributes"
        )


def normalized_parameter_name(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum())


def parameter_key(parameter: dict) -> tuple[str, str, int]:
    return (
        str(parameter.get("name", "")),
        str(parameter.get("association", "")),
        int(parameter.get("index", -1)),
    )


def effective_texture_parameters(
    object_name: str,
    records: dict[str, dict],
    cache: dict[str, list[dict]],
    stack: set[str],
) -> list[dict]:
    if object_name in cache:
        return cache[object_name]
    if object_name in stack:
        raise ValueError(f"material inheritance cycle at {object_name}")
    record = records.get(object_name)
    if record is None:
        return []
    stack.add(object_name)
    parameters = list(
        effective_texture_parameters(
            str(record.get("parent", "")), records, cache, stack
        )
    )
    by_key = {parameter_key(parameter): index for index, parameter in enumerate(parameters)}
    for parameter in record.get("textures", []):
        key = parameter_key(parameter)
        if key in by_key:
            parameters[by_key[key]] = parameter
        else:
            by_key[key] = len(parameters)
            parameters.append(parameter)
    stack.remove(object_name)
    cache[object_name] = parameters
    return parameters


def select_texture(parameters: list[dict], desired_names: tuple[str, ...]) -> str | None:
    candidates: list[tuple[tuple[int, int], str]] = []
    for parameter in parameters:
        normalized = normalized_parameter_name(str(parameter.get("name", "")))
        if normalized not in desired_names or parameter.get("value") is None:
            continue
        index = int(parameter.get("index", -1))
        scope_rank = 0 if index == -1 else 1 if index == 0 else 2 + index
        candidates.append(
            ((desired_names.index(normalized), scope_rank), str(parameter["value"]))
        )
    return min(candidates)[1] if candidates else None


def texture_grid(record: dict) -> tuple[int, int]:
    blocks = record.get("blocks", [])
    return (
        max(int(block["block_x"]) for block in blocks) + 1,
        max(int(block["block_y"]) for block in blocks) + 1,
    )


def material_texture_grid_mismatches(
    materials: dict,
    texture_records: dict[str, dict],
) -> list[str]:
    records = {record["object"]: record for record in materials["materials"]}
    cache: dict[str, list[dict]] = {}
    mismatches = []
    for object_name in sorted(records):
        parameters = effective_texture_parameters(object_name, records, cache, set())
        selected = {
            role: select_texture(parameters, desired_names)
            for role, desired_names in MATERIAL_TEXTURE_ROLES.items()
        }
        grids = {
            role: texture_grid(texture_records[reference])
            for role, reference in selected.items()
            if reference in texture_records
        }
        if len(set(grids.values())) > 1:
            mismatches.append(f"{object_name}: {grids}")
    return mismatches


def texture_blocks(record: dict) -> set[tuple[int, int]]:
    return {
        (int(block["block_x"]), int(block["block_y"]))
        for block in record.get("blocks", [])
    }


def selected_material_atlases(
    materials: dict,
    texture_records: dict[str, dict],
) -> dict[str, list[tuple[str, tuple[int, int], set[tuple[int, int]]]]]:
    """List the UDIM atlas of every texture each material samples."""
    records = {record["object"]: record for record in materials["materials"]}
    cache: dict[str, list[dict]] = {}
    result = {}
    for object_name in sorted(records):
        parameters = effective_texture_parameters(object_name, records, cache, set())
        references = sorted(
            {
                reference
                for desired_names in MATERIAL_TEXTURE_ROLES.values()
                if (reference := select_texture(parameters, desired_names))
                in texture_records
            }
        )
        atlases = [
            (
                reference,
                texture_grid(texture_records[reference]),
                texture_blocks(texture_records[reference]),
            )
            for reference in references
        ]
        if atlases:
            result[object_name] = atlases
    return result


def addressed_indices(low: float, high: float, count: int) -> list[int]:
    """Atlas cells a UV span reaches, wrapped the way a Repeat sampler wraps.

    The tolerance keeps geometry that only grazes the next tile's edge from
    claiming a cell it never actually shades.
    """
    first = math.floor(low + 0.01)
    last = max(first, math.floor(high - 0.01))
    if last - first + 1 >= count:
        return list(range(count))
    return sorted({index % count for index in range(first, last + 1)})


def effective_material(
    overrides: list[str],
    slots: list[str | None],
    material_slot: int,
    material_index: int,
) -> str | None:
    """Resolve one partition's material the way UE does.

    A component's OverrideMaterials array is indexed by StaticMaterials slot,
    while a partition's material_slot is its LOD0 render-section index; older
    manifests without material_index are section-indexed throughout.
    """
    if material_index < len(overrides) and overrides[material_index]:
        return overrides[material_index]
    return slots[material_slot] if material_slot < len(slots) else None


def extend_uv_bounds(
    bounds: dict[str, tuple[list[float], list[float]]],
    material: str,
    uv_min: list[float],
    uv_max: list[float],
) -> None:
    if material in bounds:
        existing_min, existing_max = bounds[material]
        bounds[material] = (
            [min(left, right) for left, right in zip(existing_min, uv_min)],
            [max(left, right) for left, right in zip(existing_max, uv_max)],
        )
    else:
        bounds[material] = (list(uv_min), list(uv_max))


def material_atlas_reports(
    root: Path,
    geometry: dict,
    scenes: list[dict],
    material_atlases: dict[str, list[tuple[str, tuple[int, int], set[tuple[int, int]]]]],
) -> tuple[list[str], list[str]]:
    """Report how visible geometry lands on the UDIM atlases it samples.

    Two lists: materials whose UVs leave the atlas's own UDIM domain, and
    materials addressing cells no block covers. Both still render - the sampler
    wraps, and unauthored cells hold the exporter's neutral fill - so these are
    content warnings rather than conversion failures.
    """
    source_bounds: dict[str, tuple[list[float], list[float]]] = {}
    partitions_by_mesh: dict[str, list[tuple[int, int, list[float], list[float]]]] = {}
    slots_by_mesh: dict[str, list[str | None]] = {}
    for mesh in geometry["meshes"]:
        slots = [slot.get("material") for slot in mesh.get("material_slots", [])]
        slots_by_mesh[mesh["object"]] = slots
        manifest = load(root / mesh["parts_manifest"])
        partitions = []
        for partition in manifest["partitions"]:
            slot = int(partition["material_slot"])
            material_index = int(partition.get("material_index", slot))
            uv_min = [float(value) for value in partition["uv_min"]]
            uv_max = [float(value) for value in partition["uv_max"]]
            partitions.append((slot, material_index, uv_min, uv_max))
            if slot < len(slots) and slots[slot]:
                extend_uv_bounds(source_bounds, str(slots[slot]), uv_min, uv_max)
        partitions_by_mesh[mesh["object"]] = partitions

    wrapped: dict[tuple[str, str], str] = {}
    reported: dict[tuple[str, str], str] = {}
    for scene in scenes:
        bounds = {
            material: (list(uv_min), list(uv_max))
            for material, (uv_min, uv_max) in source_bounds.items()
        }
        used = set()
        for actor in scene["actors"]:
            if actor.get("hidden", False):
                continue
            for component in actor["components"]:
                if not component.get("visible", True) or component.get(
                    "hidden_in_game", False
                ):
                    continue
                mesh_name = component.get("mesh")
                if mesh_name not in partitions_by_mesh:
                    continue
                overrides = component.get("override_materials", [])
                slots = slots_by_mesh[mesh_name]
                for slot, material_index, uv_min, uv_max in partitions_by_mesh[mesh_name]:
                    material = effective_material(overrides, slots, slot, material_index)
                    if material:
                        used.add(material)
                        extend_uv_bounds(bounds, material, uv_min, uv_max)
        if not used:
            # Every slot resolving to null still passes the reference checks
            # below, because there is then nothing left to require.
            raise ValueError(
                f"scene {scene['level']} resolves no material for any visible "
                "component; every static-material slot and override is null"
            )
        for material in sorted(used):
            if material not in bounds:
                continue
            uv_min, uv_max = bounds[material]
            for texture, (columns, rows), blocks in material_atlases.get(material, []):
                if (columns, rows) == (1, 1):
                    continue
                # UV ranges outside the atlas's own UDIM domain are valid: the
                # sampler repeats the atlas as a whole, just as UE repeats its
                # virtual-texture block pattern. Name them anyway - a material
                # that leaves its domain is sampling a tile it was not authored
                # against.
                if (
                    uv_min[0] < -0.02
                    or uv_max[0] > columns + 0.02
                    or uv_min[1] < -(rows - 1) - 0.02
                    or uv_max[1] > 1.02
                ):
                    wrapped.setdefault(
                        (material, texture),
                        f"{scene['level']} {material} {texture}: "
                        f"grid={columns}x{rows} uv={uv_min}..{uv_max}",
                    )
                # UDIM row 0 is mesh v in [0, 1] and sits at the bottom of the
                # image, so image V spans v + rows - 1.
                addressed_columns = addressed_indices(uv_min[0], uv_max[0], columns)
                addressed_rows = [
                    rows - 1 - image_row
                    for image_row in addressed_indices(
                        uv_min[1] + rows - 1, uv_max[1] + rows - 1, rows
                    )
                ]
                missing = sorted(
                    (column, row)
                    for column in addressed_columns
                    for row in addressed_rows
                    if (column, row) not in blocks
                )
                if missing:
                    reported.setdefault(
                        (material, texture),
                        f"{scene['level']} {material} {texture}: "
                        f"grid={columns}x{rows} uv={uv_min}..{uv_max} "
                        f"unauthored={missing}",
                    )
    return (
        [wrapped[key] for key in sorted(wrapped)],
        [reported[key] for key in sorted(reported)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_root", type=Path)
    parser.add_argument(
        "--skip-geometry-payloads",
        action="store_true",
        help="validate manifests/materials/scenes without reopening unchanged GLB payloads",
    )
    parser.add_argument(
        "--payload-meshes",
        type=Path,
        help="JSON array of mesh object paths whose payloads are verified anyway",
    )
    args = parser.parse_args()
    root = args.asset_root.resolve()
    payload_meshes = (
        set(json.loads(args.payload_meshes.read_text(encoding="utf-8")))
        if args.payload_meshes is not None
        else set()
    )

    geometry = load(root / "geometry.json")
    source_material_path = root / "materials.source.json"
    materials = load(
        source_material_path
        if source_material_path.is_file()
        else root / "materials.json"
    )
    texture_sources = load(root / "textures.json")
    texture_exports = load(root / "textures.exported.json")
    runtime_material_path = root / "materials.runtime.json"
    runtime_texture_path = root / "textures.runtime.json"
    runtime_materials = (
        load(runtime_material_path) if runtime_material_path.is_file() else materials
    )
    runtime_texture_exports = (
        load(runtime_texture_path) if runtime_texture_path.is_file() else texture_exports
    )
    if geometry.get("failures"):
        raise ValueError(f"geometry has {len(geometry['failures'])} failures")
    if materials.get("failures"):
        raise ValueError(f"materials have {len(materials['failures'])} failures")
    if texture_sources.get("failures"):
        raise ValueError(f"texture inventory has {len(texture_sources['failures'])} failures")
    if texture_exports.get("failures"):
        raise ValueError(f"texture export has {len(texture_exports['failures'])} failures")
    if runtime_materials.get("failures"):
        raise ValueError(
            f"runtime materials have {len(runtime_materials['failures'])} failures"
        )
    if runtime_texture_exports.get("failures"):
        raise ValueError(
            f"runtime textures have {len(runtime_texture_exports['failures'])} failures"
        )

    converted = {record["object"]: record for record in geometry["meshes"]}
    material_objects = {record["object"] for record in materials["materials"]}
    source_texture_records = {
        record["object"]: record for record in texture_sources["textures"]
    }
    exported_texture_records = {
        record["object"]: record for record in texture_exports["exported"]
    }
    source_textures = set(source_texture_records)
    exported_textures = set(exported_texture_records)
    runtime_material_objects = {
        record["object"] for record in runtime_materials["materials"]
    }
    runtime_texture_records = {
        record["object"]: record for record in runtime_texture_exports["exported"]
    }
    if len(runtime_texture_records) != len(runtime_texture_exports["exported"]):
        raise ValueError("runtime texture export contains duplicate object records")
    runtime_material_textures = {
        parameter["value"]
        for record in runtime_materials["materials"]
        for parameter in record.get("textures", [])
        if parameter.get("value") is not None
    }
    if runtime_material_textures != set(runtime_texture_records):
        raise ValueError(
            "runtime texture manifest mismatch: "
            f"missing={len(runtime_material_textures - set(runtime_texture_records))} "
            f"extra={len(set(runtime_texture_records) - runtime_material_textures)}"
        )
    if len(source_texture_records) != len(texture_sources["textures"]):
        raise ValueError("texture inventory contains duplicate object records")
    if len(exported_texture_records) != len(texture_exports["exported"]):
        raise ValueError("texture export contains duplicate object records")
    material_textures = {
        parameter["value"]
        for record in materials["materials"]
        for parameter in record.get("textures", [])
        if parameter.get("value") is not None
    }
    if material_textures != source_textures:
        raise ValueError(
            "texture inventory mismatch: "
            f"missing={len(material_textures - source_textures)} "
            f"extra={len(source_textures - material_textures)}"
        )
    if source_textures != exported_textures:
        raise ValueError(
            "texture export mismatch: "
            f"missing={len(source_textures - exported_textures)} "
            f"extra={len(exported_textures - source_textures)}"
        )
    holed_atlases: list[str] = []
    for record in texture_sources["textures"]:
        blocks = record.get("blocks", [])
        if not blocks:
            raise ValueError(f"texture has no source blocks: {record['object']}")
        ranges = sorted(
            (
                int(block["payload_offset"]),
                int(block["payload_offset"]) + int(block["payload_size"]),
            )
            for block in blocks
        )
        if ranges[0][0] != 0 or ranges[-1][1] != int(record["payload_size"]):
            raise ValueError(f"texture blocks do not span payload: {record['object']}")
        if any(left[1] != right[0] for left, right in zip(ranges, ranges[1:])):
            raise ValueError(f"texture blocks overlap or have gaps: {record['object']}")
        tile_width = int(blocks[0]["width"])
        tile_height = int(blocks[0]["height"])
        if any(
            int(block["width"]) != tile_width
            or int(block["height"]) != tile_height
            for block in blocks
        ):
            raise ValueError(f"texture source has nonuniform block sizes: {record['object']}")
        cells = texture_blocks(record)
        if len(cells) != len(blocks):
            raise ValueError(f"texture source has duplicate UDIM blocks: {record['object']}")
        if min(min(cell) for cell in cells) < 0:
            raise ValueError(f"texture source has a negative UDIM block: {record['object']}")
        expected_columns = max(cell[0] for cell in cells) + 1
        expected_rows = max(cell[1] for cell in cells) + 1
        if (
            expected_columns * tile_width != int(record["width"])
            or expected_rows * tile_height != int(record["height"])
        ):
            raise ValueError(f"texture source grid does not fill its atlas: {record['object']}")
        # The exporter pastes block (bx, by) at (bx * tile_width,
        # (rows - 1 - by) * tile_height), so block row 0 is the bottom row and
        # every block has to sit inside the atlas under that row order.
        for block_x, block_y in cells:
            right = block_x * tile_width + tile_width
            bottom = (expected_rows - 1 - block_y) * tile_height + tile_height
            if right > int(record["width"]) or bottom > int(record["height"]):
                raise ValueError(f"texture block exceeds atlas: {record['object']}")
        # A grid that does not start at block (0, 0), or that skips a cell,
        # leaves unauthored cells holding the exporter's neutral fill. That is
        # legal content, so it is reported rather than rejected.
        if len(cells) < expected_columns * expected_rows:
            holed_atlases.append(str(record["object"]))
        exported = exported_texture_records[record["object"]]
        if (
            int(exported["source_grid_columns"]) != expected_columns
            or int(exported["source_grid_rows"]) != expected_rows
            or int(exported["source_block_count"]) != len(blocks)
        ):
            raise ValueError(f"texture source grid changed during export: {record['object']}")
        if atlas_layout_matters(
            expected_columns, expected_rows, len(blocks)
        ) and exported.get("atlas_layout_version") != ATLAS_LAYOUT_VERSION:
            raise ValueError(
                f"atlas was assembled by layout version "
                f"{exported.get('atlas_layout_version')}, not {ATLAS_LAYOUT_VERSION}: "
                f"{record['object']}"
            )
        if [int(record["width"]), int(record["height"])] != [
            int(value) for value in exported["source_size"]
        ]:
            raise ValueError(f"texture source size changed during export: {record['object']}")
        output_size = [int(value) for value in exported["output_size"]]
        if min(output_size) <= 0:
            raise ValueError(f"texture export has invalid dimensions: {record['object']}")
        if int(exported.get("output_bit_depth", 8)) <= 8 and any(
            value % 4 for value in output_size
        ):
            raise ValueError(
                f"block-compressed texture has unaligned dimensions: {record['object']} "
                f"{output_size}"
            )
        max_size = int(texture_exports.get("max_size", 0))
        if max_size > 0 and max(output_size) > max_size:
            raise ValueError(f"texture export exceeds size cap: {record['object']}")
        recovered = [int(value) for value in exported.get("recovered_oodle_blocks", [])]
        if recovered != EXPECTED_RECOVERED_OODLE_BLOCKS.get(record["object"], []):
            raise ValueError(f"unexpected Oodle recovery record: {record['object']}")
    grid_mismatches = material_texture_grid_mismatches(
        materials, source_texture_records
    )
    if grid_mismatches:
        raise ValueError(
            f"{len(grid_mismatches)} materials select incompatible texture grids: "
            + "; ".join(grid_mismatches[:4])
        )
    material_atlases = selected_material_atlases(materials, source_texture_records)

    levels: set[str] = set()
    scene_meshes: set[str] = set()
    component_meshes: set[str] = set()
    scene_materials: set[str] = set()
    actor_count = 0
    scenes = []
    for scene_path in sorted((root / "scenes").glob("*.json")):
        scene = load(scene_path)
        if scene.get("format") != "zorah-scene-manifest-v6":
            raise ValueError(f"stale scene manifest without decal data: {scene_path}")
        scenes.append(scene)
        levels.add(scene["level"])
        require_post_process_exposure(scene)
        require_atmosphere_and_height_fog(scene)
        unresolved = [
            component
            for actor in scene["actors"]
            for component in actor["components"]
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
        actor_count += len(scene["actors"])
        scene_meshes.update(scene["referenced_meshes"])
        for actor in scene["actors"]:
            if "lights" not in actor:
                raise ValueError(f"actor lacks exported light array: {actor.get('name')}")
            for light in actor["lights"]:
                if light.get("type") not in {"point", "spot", "directional", "sky"}:
                    raise ValueError(f"unsupported exported light type: {light.get('type')}")
                if float(light.get("intensity", -1.0)) < 0.0:
                    raise ValueError(f"invalid light intensity: {light}")
                if "transform" not in light or "color" not in light:
                    raise ValueError(f"incomplete exported light record: {light}")
            # A DecalActor projects a material that reaches no mesh slot, so it
            # is required here or nothing requires it at all.
            scene_materials.update(
                decal["material"]
                for decal in actor.get("decals", [])
                if str(decal.get("material", "")).startswith("/Game/")
            )
            for component in actor["components"]:
                if component.get("mesh") is not None:
                    component_meshes.add(component["mesh"])
                overrides = component.get("override_materials", [])
                if not isinstance(overrides, list) or any(
                    not isinstance(material, str) for material in overrides
                ):
                    raise ValueError(
                        "component override_materials must preserve slot-shaped "
                        f"string entries: {component.get('name')}"
                    )
                # UE can retain stale OverrideMaterials entries after a static mesh's
                # material-slot count shrinks. Preserve the slot-shaped array exactly,
                # but only entries addressed by an actual partition affect rendering.
                # The effective-material checks below intentionally index by each
                # partition's material slot and therefore ignore such trailing entries.
                scene_materials.update(
                    material
                    for material in overrides
                    if material.startswith("/Game/")
                )
        light_counts: dict[str, int] = {}
        for actor in scene["actors"]:
            for light in actor["lights"]:
                light_counts[light["type"]] = light_counts.get(light["type"], 0) + 1
        for kind, minimum in EXPECTED_LIGHT_MINIMUMS[scene["level"]].items():
            if light_counts.get(kind, 0) < minimum:
                raise ValueError(
                    f"scene {scene['level']} exported {light_counts.get(kind, 0)} {kind} "
                    f"lights, expected at least {minimum}"
                )
    if levels != EXPECTED_LEVELS:
        raise ValueError(f"expected {sorted(EXPECTED_LEVELS)}, got {sorted(levels)}")
    if scene_meshes != set(converted):
        raise ValueError(
            "scene geometry mismatch: "
            f"missing={len(scene_meshes - set(converted))} "
            f"extra={len(set(converted) - scene_meshes)}"
        )
    # referenced_meshes and the component list are written by separate loops in
    # the extractor, so a component can name a mesh nothing ever converted.
    dangling_meshes = component_meshes - set(converted)
    if dangling_meshes:
        raise ValueError(
            f"{len(dangling_meshes)} components reference unconverted meshes: "
            + ", ".join(sorted(dangling_meshes)[:4])
        )

    partition_count = 0
    triangle_count = 0
    conventional_triangle_count = 0
    saw_legacy_conventional_geometry = False
    referenced_files: list[Path] = []
    slot_materials: set[str] = set()
    for mesh in converted.values():
        mesh_slots = mesh.get("material_slots", [])
        slot_materials.update(
            slot["material"]
            for slot in mesh.get("material_slots", [])
            if slot.get("material") is not None
        )
        manifest_path = root / mesh["parts_manifest"]
        part_manifest = load(manifest_path)
        part_format = part_manifest.get("format")
        if part_format not in {
            "zorah-partitioned-mesh-v3",
            "zorah-partitioned-mesh-v4",
        }:
            raise ValueError(f"unsupported partition format in {mesh['object']}")
        legacy_conventional_geometry = part_format == "zorah-partitioned-mesh-v3"
        saw_legacy_conventional_geometry |= legacy_conventional_geometry
        if part_manifest["source_triangles"] != part_manifest["output_triangles"]:
            raise ValueError(f"triangle loss in {mesh['object']}")
        parent = manifest_path.parent
        for part in part_manifest["partitions"]:
            partition_count += 1
            triangle_count += int(part["triangles"])
            if legacy_conventional_geometry:
                conventional_triangle_count += int(part["blas_triangles"])
            if "uv_min" not in part or "uv_max" not in part:
                raise ValueError(f"partition lacks UV bounds in {mesh['object']}")
            uv_min = [float(value) for value in part["uv_min"]]
            uv_max = [float(value) for value in part["uv_max"]]
            if (
                len(uv_min) != 2
                or len(uv_max) != 2
                or not all(math.isfinite(value) for value in uv_min + uv_max)
                or any(low > high for low, high in zip(uv_min, uv_max))
            ):
                raise ValueError(f"partition has invalid UV bounds in {mesh['object']}")
            if args.skip_geometry_payloads and mesh["object"] not in payload_meshes:
                continue
            meshlet_path = parent / part["meshlet"]
            meshlet, meshlet_payload = read_glb(meshlet_path)
            require_meshlet_source_vertex_format(meshlet, str(meshlet_path))
            if "uri" in meshlet["buffers"][0]:
                raise ValueError(f"partition is not self-contained GLB in {mesh['object']}")
            if hashlib.sha256(meshlet_path.read_bytes()).hexdigest() != part.get(
                "meshlet_sha256"
            ):
                raise ValueError(f"meshlet GLB hash mismatch in {mesh['object']}")
            referenced_files.append(meshlet_path)
            if legacy_conventional_geometry:
                conventional_path = parent / part["geometry"]
                conventional, conventional_payload = read_glb(conventional_path)
                require_runtime_vertex_format(conventional, str(conventional_path))
                if "uri" in conventional["buffers"][0]:
                    raise ValueError(
                        f"partition is not self-contained GLB in {mesh['object']}"
                    )
                # The legacy conventional twin carries a 16-byte tangent per
                # vertex while the meshlet source intentionally does not.
                full_conventional_payload = (
                    len(meshlet_payload) + int(part["vertices"]) * 16
                )
                if (
                    float(part["blas_achieved_ratio"]) < 1.0
                    and len(conventional_payload) >= full_conventional_payload
                ):
                    raise ValueError(
                        f"conventional GLB is not compact in {mesh['object']}"
                    )
                if hashlib.sha256(conventional_path.read_bytes()).hexdigest() != part.get(
                    "geometry_sha256"
                ):
                    raise ValueError(
                        f"conventional GLB hash mismatch in {mesh['object']}"
                    )
                referenced_files.append(conventional_path)
    require_files(root, referenced_files)
    wrapped_atlas_uvs, atlas_holes = material_atlas_reports(
        root, geometry, scenes, material_atlases
    )
    for outside in wrapped_atlas_uvs:
        print(f"ZORAH_VERIFY_ATLAS_WRAP {outside}", flush=True)
    for hole in atlas_holes:
        print(f"ZORAH_VERIFY_ATLAS_HOLE {hole}", flush=True)
    if saw_legacy_conventional_geometry and conventional_triangle_count >= triangle_count:
        raise ValueError("conventional Solari geometry was not compacted")
    required_exported_materials = (
        scene_materials | slot_materials
    ) - EXPLICIT_RUNTIME_ENGINE_MATERIALS
    missing_materials = required_exported_materials - material_objects
    if missing_materials:
        raise ValueError(f"missing {len(missing_materials)} source material records")
    missing_runtime_materials = required_exported_materials - runtime_material_objects
    if missing_runtime_materials:
        raise ValueError(
            f"missing {len(missing_runtime_materials)} runtime material records"
        )

    texture_files: list[Path] = []
    for record in texture_exports["exported"]:
        image = root / record["output"]
        texture_files.extend([image, image.with_name(f"{image.name}.meta")])
    require_files(root, texture_files)
    runtime_texture_files = [
        root / record["output"] for record in runtime_texture_exports["exported"]
    ]
    require_files(root, runtime_texture_files)

    print(
        "ZORAH_VERIFY_OK "
        f"levels={len(levels)} actors={actor_count} meshes={len(converted)} "
        f"partitions={partition_count} triangles={triangle_count} "
        f"legacy_blas_triangles={conventional_triangle_count} materials={len(material_objects)} "
        f"textures={len(exported_textures)} runtime_materials={len(runtime_material_objects)} "
        f"runtime_textures={len(runtime_texture_records)} "
        f"holed_atlases={len(holed_atlases)} atlas_hole_materials={len(atlas_holes)} "
        f"wrapped_atlas_uvs={len(wrapped_atlas_uvs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
