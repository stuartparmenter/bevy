#!/usr/bin/env python3
"""Run the complete Zorah-source to Bevy-assets conversion pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


LEVELS = ("GreenHouse_Level", "Restir_Level", "ThroneRoom_Level")
EXPECTED_POST_PROCESS_EXPOSURE = {
    "GreenHouse_Level": (4.0, 5.0, 0.99),
    "Restir_Level": (None, None, -2.5),
    "ThroneRoom_Level": (8.0, 8.0, None),
}
# (BloomMethod, BloomIntensity) per level. Only ThroneRoom ticks the bloom
# override bits; the other two volumes serialize the same numbers with the bits
# clear, so UE never applies them and the export drops them.
EXPECTED_POST_PROCESS_BLOOM = {
    "GreenHouse_Level": (None, None),
    "Restir_Level": (None, None),
    "ThroneRoom_Level": ("BM_FFT", 0.003),
}
# One actor per external-actor package: a UChildActorComponent's child is saved
# beside its parent in the same package, and the export re-parents it onto the
# parent instead of listing it as an actor of its own.
#
# referenced_meshes counts Niagara mesh-renderer meshes as well as component
# ones: the butterfly mesh in GreenHouse, and the two nebula spheres in
# ThroneRoom. Restir's one Niagara system renders no mesh.
EXPECTED_SCENE_INVENTORY = {
    "GreenHouse_Level": {
        "actors": 1788,
        "actor_packages": 1788,
        "unresolved_mesh_components": 27,
        "referenced_meshes": 300,
        "decal_components": 232,
        "niagara_components": 2,
    },
    "Restir_Level": {
        "actors": 1924,
        "actor_packages": 1924,
        "unresolved_mesh_components": 0,
        "referenced_meshes": 271,
        "decal_components": 323,
        "niagara_components": 1,
    },
    "ThroneRoom_Level": {
        "actors": 6773,
        "actor_packages": 6773,
        "unresolved_mesh_components": 0,
        "referenced_meshes": 115,
        "decal_components": 17,
        "niagara_components": 14,
    },
}
CONVERT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_OUTPUT = CONVERT_DIRECTORY.parent / "assets"
PROJECT = CONVERT_DIRECTORY / "ZorahConvert.csproj"
ASSEMBLY = CONVERT_DIRECTORY / "bin" / "Release" / "net10.0" / "ZorahConvert.dll"
REPOSITORY = CONVERT_DIRECTORY.parents[3]
PACK_STATE_FORMAT = "zorah-pack-state-v1"
PACK_PIPELINE_VERSION = 1
# Formats zorah_pack stamps into pack.json. Bump alongside ZORAH_BUNDLE_VERSION
# (src/zorah_bundle.rs) and MESHLET_MESH_ASSET_VERSION
# (crates/bevy_pbr/src/meshlet/asset.rs); a stale stamp invalidates every packed
# artifact because the runtime rejects the payloads it hard-links.
EXPECTED_BUNDLE_FORMAT_VERSION = 2
EXPECTED_MESHLET_ASSET_VERSION = 4
# Bump when mesh_description.py, partition_mesh.py, or convert_geometry.py change
# how a mesh is extracted or partitioned. It invalidates the whole loose geometry
# cache, which only --rebuild-geometry can re-partition.
# 2: Tangent/BinormalSign streams no longer extracted; zero-length normals
#    repaired at partition time.
GEOMETRY_PIPELINE_VERSION = 2
# Per-mesh rebuild keys live beside the geometry manifest instead of inside it:
# geometry.json is hashed into the pack fingerprint and copied into the runtime
# tree, and source timestamps belong in neither.
GEOMETRY_INPUTS_FORMAT = "zorah-geometry-inputs-v1"
GEOMETRY_INPUTS_NAME = "geometry-inputs.json"
# Scene manifests are pure ZorahConvert output, and scene_manifest_is_current
# only inspects what the manifest reports about the level. Blueprint-template
# resolution changed the extractor without moving a single one of those counts,
# so the cached manifests survived it; the extractor's own content hash is what
# invalidates them.
CONVERTER_INPUTS_FORMAT = "zorah-converter-inputs-v1"
CONVERTER_INPUTS_NAME = "converter-inputs.json"
# UPointLightComponent's class defaults. A blueprint-embedded light that reaches
# a manifest with all three at once was read from the class archetype instead of
# the blueprint's component template.
ARCHETYPE_LIGHT_INTENSITY = 3.1415927410125732
ARCHETYPE_LIGHT_ATTENUATION_RADIUS = 1000.0
ENGINE_PRIMITIVES = {
    "/Engine/BasicShapes/Cube.Cube",
    "/Engine/EngineMeshes/Cube.Cube",
    "/Engine/BasicShapes/Plane.Plane",
}
WORLD_GRID_MATERIAL = "/Engine/EngineMaterials/WorldGridMaterial.WorldGridMaterial"
ENGINE_MATERIALS = {
    "/Engine/EngineDebugMaterials/BlackUnlitMaterial.BlackUnlitMaterial",
    WORLD_GRID_MATERIAL,
}


def run(command: list[str]) -> None:
    print("ZORAH_CONVERT_RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_if_changed(path: Path, value: object) -> None:
    encoded = json.dumps(value, indent=2) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == encoded:
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def stable_hash(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def native_path(path: Path, executable: str) -> str:
    """Translate paths when WSL launches a native Windows executable."""
    resolved = path.resolve()
    if os.name == "nt" or not executable.casefold().endswith(".exe"):
        return str(resolved)
    parts = resolved.parts
    if (
        len(parts) >= 3
        and parts[0] == "/"
        and parts[1].casefold() == "mnt"
        and len(parts[2]) == 1
        and parts[2].isalpha()
    ):
        drive = parts[2].upper()
        tail = "\\".join(parts[3:])
        return f"{drive}:\\{tail}" if tail else f"{drive}:\\"
    distro = os.environ.get("WSL_DISTRO_NAME")
    if not distro:
        return str(resolved)
    windows_tail = str(resolved).replace("/", "\\")
    return f"\\\\wsl.localhost\\{distro}{windows_tail}"


def find_dotnet(requested: str | None) -> str:
    if requested:
        return requested
    discovered = shutil.which("dotnet")
    if discovered:
        return discovered
    wsl_dotnet = Path("/mnt/c/Program Files/dotnet/dotnet.exe")
    if wsl_dotnet.is_file():
        return str(wsl_dotnet)
    raise RuntimeError("dotnet was not found; install the .NET 10 SDK or pass --dotnet")


def dotnet_path(path: Path, dotnet: str) -> str:
    """Translate WSL paths when orchestrating the native Windows dotnet host."""
    return native_path(path, dotnet)


def find_cargo(requested: str | None, output: Path) -> str:
    if requested:
        return requested
    if os.name != "nt" and str(output).startswith("/mnt/"):
        windows_cargo = Path("/mnt/c/Users/stuart/.cargo/bin/cargo.exe")
        if windows_cargo.is_file():
            return str(windows_cargo)
    return shutil.which("cargo") or "cargo"


def cue4parse(dotnet: str, project_root: Path, *arguments: Path | str) -> None:
    run(
        [
            dotnet,
            dotnet_path(ASSEMBLY, dotnet),
            dotnet_path(project_root, dotnet),
            *(dotnet_path(value, dotnet) if isinstance(value, Path) else value for value in arguments),
        ]
    )


def cargo_command(cargo: str, binary: str, *arguments: str | Path) -> list[str]:
    # Packer builds get their own target dir with machine-native Rust codegen.
    # A separate dir keeps interactive `cargo run` artifacts untouched, and
    # --config (rather than env vars) survives the WSL-to-Windows-cargo
    # interop boundary, which drops the caller's environment. No CFLAGS or
    # CXXFLAGS: the ctt compressor backends compile scalar/SSE/AVX variants of
    # the same sources with their own per-variant flags and runtime dispatch,
    # and a blanket /arch flag breaks the scalar variants.
    target_dir = native_path(REPOSITORY / "target" / "packer", cargo).replace("\\", "/")
    return [
        cargo,
        "+1.96.1",
        "--config",
        f'build.target-dir="{target_dir}"',
        "--config",
        'build.rustflags=["-C","target-cpu=native"]',
        "run",
        "--release",
        "--manifest-path",
        native_path(REPOSITORY / "Cargo.toml", cargo),
        "-p",
        "zorah",
        "--no-default-features",
        "--features",
        "packer",
        "--bin",
        binary,
        "--",
        *(native_path(value, cargo) if isinstance(value, Path) else value for value in arguments),
    ]


def verify_runtime(asset_root: Path, cargo: str, *, audit_capacity: bool) -> None:
    run([sys.executable, str(CONVERT_DIRECTORY / "verify_bundles.py"), str(asset_root)])
    if audit_capacity:
        run(cargo_command(cargo, "zorah_capacity", asset_root))


def require_successful_geometry(output: Path) -> None:
    manifest_path = output / "geometry.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"{output} exists without geometry.json; remove it or choose another output"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") not in {
        "zorah-geometry-manifest-v2",
        "zorah-geometry-manifest-v3",
    }:
        raise RuntimeError(
            "geometry conversion format is unsupported; rerun into a clean output directory "
            "instead of resuming it"
        )
    if manifest.get("failures"):
        raise RuntimeError(
            f"geometry conversion contains {len(manifest['failures'])} failures; "
            "inspect geometry.json before resuming"
        )


def texture_inventory_matches(materials: Path, textures: Path) -> bool:
    if not textures.is_file():
        return False
    try:
        material_document = json.loads(materials.read_text(encoding="utf-8"))
        texture_document = json.loads(textures.read_text(encoding="utf-8"))
        return set(material_document.get("texture_references", [])) == {
            record["object"] for record in texture_document.get("textures", [])
        }
    except (OSError, KeyError, TypeError, ValueError):
        return False


def discard_texture_exports(loose: Path) -> None:
    """Drop every exported texture payload so the next export pass rebuilds it."""
    discarded = 0
    for manifest_name, key in (
        ("textures.exported.json", "exported"),
        ("textures.json", "textures"),
    ):
        manifest_path = loose / manifest_name
        if not manifest_path.is_file():
            continue
        for record in load_json(manifest_path).get(key, []):
            output = loose / str(record["output"])
            for path in (output, output.with_name(f"{output.name}.meta")):
                if path.is_file():
                    path.unlink()
                    discarded += 1
    (loose / "textures.exported.json").unlink(missing_ok=True)
    print(f"ZORAH_CONVERT_TEXTURE_REFRESH discarded={discarded}", flush=True)


def refresh_texture_inventory(
    dotnet: str,
    project_root: Path,
    materials: Path,
    loose: Path,
    staging: Path,
    *,
    refresh: bool = False,
) -> None:
    if staging.exists():
        staged_manifest = staging / "textures.json"
        if refresh or not texture_inventory_matches(materials, staged_manifest):
            shutil.rmtree(staging)
    if not staging.exists():
        cue4parse(dotnet, project_root, "texture-export", materials, staging)
    manifest = staging / "textures.json"
    if not manifest.is_file():
        raise RuntimeError("texture inventory export did not produce textures.json")
    for source in sorted(staging.rglob("*.png")):
        destination = loose / source.relative_to(staging)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not refresh:
            source.unlink()
        else:
            os.replace(source, destination)
    os.replace(manifest, loose / "textures.json")
    shutil.rmtree(staging)


def scene_manifest_is_current(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        records = [
            actor["post_process"]
            for actor in document.get("actors", [])
            if actor.get("post_process") is not None
            and actor["post_process"].get("enabled", True)
            and actor["post_process"].get("unbound", False)
            and float(actor["post_process"].get("blend_weight", 1.0)) > 0.0
        ]
        if document.get("format") != "zorah-scene-manifest-v4" or len(records) != 1:
            return False
        level = document.get("level")
        inventory = EXPECTED_SCENE_INVENTORY.get(level)
        if inventory is None or document.get("failures"):
            return False
        if (
            len(document.get("actors", [])) != inventory["actors"]
            or int(document.get("actor_package_count", -1))
            != inventory["actor_packages"]
            or int(document.get("unresolved_static_mesh_components", -1))
            != inventory["unresolved_mesh_components"]
            or len(document.get("referenced_meshes", []))
            != inventory["referenced_meshes"]
            or int(document.get("decal_components", -1))
            != inventory["decal_components"]
            or int(document.get("niagara_components", -1))
            != inventory["niagara_components"]
        ):
            return False
        record = records[0]
        if record.get("auto_exposure_method") != "AEM_Histogram":
            return False
        exposure = EXPECTED_POST_PROCESS_EXPOSURE.get(level)
        bloom = EXPECTED_POST_PROCESS_BLOOM.get(level)
        if exposure is None or bloom is None:
            return False
        bloom_method, bloom_intensity = bloom
        if record.get("bloom_method") != bloom_method:
            return False
        for field, expected_value in zip(
            (
                "bloom_intensity",
                "auto_exposure_min_ev100",
                "auto_exposure_max_ev100",
                "auto_exposure_bias",
            ),
            (bloom_intensity, *exposure),
        ):
            value = record.get(field)
            if expected_value is None:
                if value is not None:
                    return False
            elif value is None or not math.isfinite(float(value)) or not math.isclose(
                float(value), expected_value, abs_tol=1e-5
            ):
                return False
        return True
    except (OSError, TypeError, ValueError):
        return False


def converter_fingerprint() -> str:
    """Content hash of the ZorahConvert sources that produce scene manifests.

    The sources rather than the built assembly: this is the same tree the
    --skip-dotnet-build path assumes is already compiled, and it stays stable
    across rebuilds that change nothing.
    """
    sources = sorted(
        [*CONVERT_DIRECTORY.glob("*.cs"), PROJECT], key=lambda path: path.name
    )
    return stable_hash(
        [(path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources]
    )


def scene_cache_is_current(scene_directory: Path, scene_paths: list[Path]) -> bool:
    """Accept cached scene manifests only if today's extractor wrote them."""
    try:
        stamp = load_json(scene_directory / CONVERTER_INPUTS_NAME)
    except (OSError, TypeError, ValueError):
        return False
    return (
        stamp.get("format") == CONVERTER_INPUTS_FORMAT
        and stamp.get("converter") == converter_fingerprint()
        and all(scene_manifest_is_current(path) for path in scene_paths)
    )


def stamp_scene_cache(scene_directory: Path) -> None:
    write_json_if_changed(
        scene_directory / CONVERTER_INPUTS_NAME,
        {"format": CONVERTER_INPUTS_FORMAT, "converter": converter_fingerprint()},
    )


def is_archetype_light(light: dict) -> bool:
    """True for a light record still at UPointLightComponent's class defaults.

    Intensity PI, a 1000 uu attenuation radius and an identity relative
    transform together are the class archetype, never an authored lamp.
    """
    transform = light.get("transform", {})
    translation = transform.get("translation", {})
    rotation = transform.get("rotation", {})
    scale = transform.get("scale", {})
    return (
        float(light.get("intensity", 0.0)) == ARCHETYPE_LIGHT_INTENSITY
        and float(light.get("attenuation_radius", 0.0))
        == ARCHETYPE_LIGHT_ATTENUATION_RADIUS
        and all(float(translation.get(axis, 0.0)) == 0.0 for axis in "xyz")
        and all(float(rotation.get(axis, 0.0)) == 0.0 for axis in "xyz")
        and float(rotation.get("w", 1.0)) == 1.0
        and all(float(scale.get(axis, 1.0)) == 1.0 for axis in "xyz")
    )


def validate_blueprint_light_archetypes(scene_paths: list[Path]) -> None:
    """Reject lights whose blueprint component template was never merged in.

    Catches a manifest exported before LoadBlueprintComponentTemplates existed
    as well as a regression in the archetype chain itself.
    """
    for scene_path in scene_paths:
        for actor in load_json(scene_path).get("actors", []):
            for light in actor.get("lights", []):
                if is_archetype_light(light):
                    raise RuntimeError(
                        f"{scene_path.name} actor "
                        f"{actor.get('label') or actor.get('name')} light "
                        f"{light.get('name')} carries un-merged blueprint archetype "
                        "defaults; re-export the scene manifests"
                    )


def is_child_actor_template(actor: dict) -> bool:
    """True for the actor a UChildActorComponent spawned from its template.

    UE names that export <component>_<child class>_CAT, so the actor's own type
    appears inside its name; nothing else in the levels is named that way.
    """
    return f"_{actor.get('type', '')}_CAT" in (actor.get("name") or "")


def validate_child_actor_placement(scene_paths: list[Path]) -> None:
    """Reject a child actor still standing as an actor of its own.

    The level saves it with an identity root transform, because its placement
    lives on the parent's UChildActorComponent, so one that survives the export
    puts its meshes at the map origin instead of on its parent.
    """
    for scene_path in scene_paths:
        for actor in load_json(scene_path).get("actors", []):
            if is_child_actor_template(actor):
                raise RuntimeError(
                    f"{scene_path.name} actor "
                    f"{actor.get('label') or actor.get('name')} is an un-parented "
                    "child actor; re-export the scene manifests"
                )


def install_scene_manifests(
    next_paths: list[Path], work: Path, loose: Path, output: Path
) -> None:
    """Publish freshly exported scene manifests to every tree that caches them.

    The work cache is seeded first: a crash in between costs one repack, while
    updating the runtime tree alone lets the next repack embed stale scenes.
    """
    (work / "scenes").mkdir(parents=True, exist_ok=True)
    for path in next_paths:
        shutil.copy2(path, work / "scenes" / path.name)
        if loose.is_dir():
            (loose / "scenes").mkdir(exist_ok=True)
            shutil.copy2(path, loose / "scenes" / path.name)
    stamp_scene_cache(work / "scenes")
    (output / "scenes").mkdir(exist_ok=True)
    for path in next_paths:
        os.replace(path, output / "scenes" / path.name)


def scene_mesh_usage(scene_paths: list[Path]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for level_index, path in enumerate(scene_paths):
        for actor in load_json(path).get("actors", []):
            for component in actor.get("components", []):
                mesh = component.get("mesh")
                if mesh:
                    usage[mesh] = usage.get(mesh, 0) | (1 << level_index)
    return usage


def source_package(project_root: Path, object_name: str) -> Path:
    """Resolve a /Game object path exactly as convert_geometry.py does."""
    package = object_name[len("/Game/") :].split(".", 1)[0]
    return project_root / "Content" / f"{package}.uasset"


def geometry_mesh_inputs(project_root: Path, object_name: str) -> dict:
    """Per-mesh rebuild key.

    The partition limit and the pipeline version invalidate the whole cache at
    once, so they are compared separately rather than repeated per mesh.
    """
    if not object_name.startswith("/Game/"):
        # Engine primitives are generated from convert_geometry.py's own tables.
        return {"generated": True}
    try:
        stat = source_package(project_root, object_name).stat()
    except OSError:
        return {"source_size": None, "source_mtime_ns": None}
    return {"source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns}


def install_geometry_build(
    project_root: Path,
    loose: Path,
    work: Path,
    scene_paths: list[Path],
    triangles: int,
) -> None:
    """Convert every referenced mesh into staging, then adopt it as the cache."""
    staging = work / "geometry.full"
    if staging.exists():
        # convert_geometry.py renames its temporary tree into place only after a
        # complete run, so an existing staging tree is finished geometry.
        print(f"ZORAH_CONVERT_GEOMETRY_ADOPT staging={staging}", flush=True)
    else:
        run(
            [
                sys.executable,
                str(CONVERT_DIRECTORY / "convert_geometry.py"),
                str(project_root),
                str(staging),
                *(str(path) for path in scene_paths),
                "--triangles",
                str(triangles),
            ]
        )
    require_successful_geometry(staging)
    loose.mkdir(parents=True, exist_ok=True)
    for name in ("geometry", "scenes"):
        if (staging / name).is_dir():
            if (loose / name).exists():
                shutil.rmtree(loose / name)
            os.replace(staging / name, loose / name)
    for name in ("geometry.json", "material-input.json"):
        if (staging / name).is_file():
            os.replace(staging / name, loose / name)
    # Stamp provenance immediately; the next reconcile fills in the per-mesh
    # records. Leaving the file absent instead would make a tree built here
    # indistinguishable from one built by an older pipeline.
    write_json_if_changed(
        loose / GEOMETRY_INPUTS_NAME,
        {"format": GEOMETRY_INPUTS_FORMAT, "pipeline_version": GEOMETRY_PIPELINE_VERSION},
    )
    shutil.rmtree(staging)


def reconcile_geometry_cache(
    project_root: Path,
    loose: Path,
    work: Path,
    scene_paths: list[Path],
    triangles: int,
) -> list[str]:
    """Rebuild the meshes whose scene references or source packages changed."""
    geometry_path = loose / "geometry.json"
    geometry = load_json(geometry_path)
    cached_triangles = geometry.get("triangle_partition_limit")
    if cached_triangles is not None and int(cached_triangles) != triangles:
        raise RuntimeError(
            f"loose geometry is partitioned at --triangles {cached_triangles}; rerun "
            f"with --triangles {cached_triangles} or pass --rebuild-geometry to "
            "re-partition the cache"
        )
    inputs_path = loose / GEOMETRY_INPUTS_NAME
    cached_inputs = load_json(inputs_path) if inputs_path.is_file() else {}
    if cached_inputs.get("format") != GEOMETRY_INPUTS_FORMAT:
        cached_inputs = {}
    # A missing or foreign-format inputs file means unknown provenance (legacy
    # tree, or a crash between install_geometry_build's unlink and the
    # re-stamp), so it must not default to the current version.
    cached_pipeline = cached_inputs.get("pipeline_version", 0)
    if int(cached_pipeline) != GEOMETRY_PIPELINE_VERSION:
        raise RuntimeError(
            f"loose geometry was built by pipeline version {cached_pipeline}; pass "
            "--rebuild-geometry to re-extract it"
        )
    referenced = {
        mesh
        for scene_path in scene_paths
        for mesh in load_json(scene_path).get("referenced_meshes", [])
    }
    existing = {mesh["object"]: mesh for mesh in geometry.get("meshes", [])}
    expected_inputs = {
        object_name: geometry_mesh_inputs(project_root, object_name)
        for object_name in referenced
    }
    recorded_inputs = cached_inputs.get("meshes", {})
    stale = {
        object_name
        for object_name, mesh in existing.items()
        if object_name in referenced
        # A cache predating the per-mesh keys adopts the current source state.
        and (
            recorded_inputs.get(object_name, expected_inputs[object_name])
            != expected_inputs[object_name]
            or not (loose / mesh["parts_manifest"]).is_file()
        )
    }
    missing = referenced - existing.keys()
    obsolete = existing.keys() - referenced
    if stale == existing.keys() & referenced and len(stale) > 1:
        # The delta stage stages a whole second copy of what it rebuilds, which
        # only fits while the rebuild is a fraction of the cache.
        raise RuntimeError(
            f"every cached mesh ({len(stale)}) has a changed source package; pass "
            "--rebuild-geometry rather than rebuilding them through the delta cache"
        )
    rebuild = sorted(missing | stale)

    added_meshes: list[dict] = []
    added_resolutions: list[dict] = []
    if rebuild:
        print(
            f"ZORAH_CONVERT_GEOMETRY_REBUILD added={len(missing)} stale={len(stale)}",
            flush=True,
        )
        delta_scene = work / "geometry-delta-scene.json"
        delta_output = work / "geometry.delta"
        if delta_output.exists():
            raise RuntimeError(f"interrupted geometry delta exists: {delta_output}")
        write_json_if_changed(
            delta_scene,
            {
                "format": "zorah-scene-manifest-v4",
                "level": "GeometryDelta",
                "actors": [],
                "referenced_meshes": rebuild,
            },
        )
        run(
            [
                sys.executable,
                str(CONVERT_DIRECTORY / "convert_geometry.py"),
                str(project_root),
                str(delta_output),
                str(delta_scene),
                "--triangles",
                str(triangles),
            ]
        )
        require_successful_geometry(delta_output)
        delta = load_json(delta_output / "geometry.json")
        if {mesh["object"] for mesh in delta.get("meshes", [])} != set(rebuild):
            raise RuntimeError("geometry delta did not produce the exact rebuilt mesh set")
        for mesh in delta.get("meshes", []):
            asset_id = mesh["asset_id"]
            source = delta_output / "geometry" / asset_id
            destination = loose / "geometry" / asset_id
            if destination.exists():
                # Either a mesh being rebuilt, or an asset directory that outlived
                # an earlier drop from the manifest; the delta output replaces both.
                shutil.rmtree(destination)
            os.replace(source, destination)
            added_meshes.append(mesh)
        added_resolutions = delta.get("material_resolutions", [])
        shutil.rmtree(delta_output)

    rebuilt = {mesh["object"] for mesh in added_meshes}
    retained_meshes = [
        mesh
        for mesh in geometry.get("meshes", [])
        if mesh["object"] in referenced and mesh["object"] not in rebuilt
    ]
    geometry["meshes"] = sorted(
        retained_meshes + added_meshes,
        key=lambda record: record["object"],
    )
    geometry["triangle_partition_limit"] = triangles
    retained_objects = {mesh["object"] for mesh in retained_meshes}
    geometry["material_resolutions"] = sorted(
        [
            record
            for record in geometry.get("material_resolutions", [])
            if record["mesh"] in retained_objects
        ]
        + added_resolutions,
        key=lambda record: (record["mesh"], int(record["slot"])),
    )
    geometry["referenced_mesh_count"] = len(referenced)
    geometry["project_mesh_count"] = sum(
        mesh.startswith("/Game/") for mesh in referenced
    )
    geometry["external_meshes"] = sorted(
        mesh for mesh in referenced if not mesh.startswith("/Game/")
    )
    geometry["generated_engine_meshes"] = sorted(
        mesh for mesh in geometry["external_meshes"] if mesh in ENGINE_PRIMITIVES
    )
    geometry["selected_mesh_count"] = geometry["project_mesh_count"]
    geometry["scenes"] = [
        {
            "level": document["level"],
            "source": str(path.resolve()),
            "path": f"scenes/{document['level']}.json",
            "actor_count": len(document.get("actors", [])),
        }
        for path in scene_paths
        for document in [load_json(path)]
    ]
    write_json_if_changed(geometry_path, geometry)
    write_json_if_changed(
        inputs_path,
        {
            "format": GEOMETRY_INPUTS_FORMAT,
            "pipeline_version": GEOMETRY_PIPELINE_VERSION,
            "triangles": triangles,
            "meshes": {
                mesh["object"]: expected_inputs[mesh["object"]]
                for mesh in geometry["meshes"]
            },
        },
    )
    for mesh in existing.values():
        # Dropped asset directories must go, or re-adding the mesh later collides
        # with its own deterministic asset id.
        if mesh["object"] in obsolete:
            shutil.rmtree(loose / "geometry" / mesh["asset_id"], ignore_errors=True)
    if rebuild or obsolete:
        print(
            f"ZORAH_CONVERT_GEOMETRY_DELTA rebuilt={len(rebuild)} removed={len(obsolete)}",
            flush=True,
        )
    return rebuild


def mesh_material_manifest_is_current(path: Path, geometry: dict) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = load_json(path)
        if manifest.get("format") != "zorah-mesh-material-manifest-v2":
            return False
        if manifest.get("failures"):
            return False
        expected = {
            mesh["object"]
            for mesh in geometry.get("meshes", [])
            if mesh["object"].startswith("/Game/")
        }
        actual = {mesh["object"] for mesh in manifest.get("meshes", [])}
        return actual == expected
    except (KeyError, OSError, TypeError, ValueError):
        return False


def stamp_partition_material_indices(
    loose: Path, mesh: dict, section_materials: dict[int, int]
) -> None:
    """Record the UE material-slot index each partition renders with.

    A partition's material_slot is its LOD0 render-section index, but a
    component's OverrideMaterials array is indexed by StaticMaterials slot, so
    the runtime needs both numbers to reproduce UE's assignment.
    """
    parts_manifest = mesh.get("parts_manifest")
    if not parts_manifest:
        return
    manifest_path = loose / parts_manifest
    manifest = load_json(manifest_path)
    for partition in manifest.get("partitions", []):
        section_index = int(partition["material_slot"])
        partition["material_index"] = section_materials.get(section_index, section_index)
    write_json_if_changed(manifest_path, manifest)


def apply_exact_mesh_materials(
    loose: Path,
    mesh_material_manifest: Path,
    scene_paths: list[Path],
) -> None:
    geometry_path = loose / "geometry.json"
    geometry = load_json(geometry_path)
    exact = {
        mesh["object"]: mesh
        for mesh in load_json(mesh_material_manifest).get("meshes", [])
    }
    resolutions = []
    material_objects: set[str] = set()
    for mesh in geometry.get("meshes", []):
        object_name = mesh["object"]
        if not object_name.startswith("/Game/"):
            # The only supported non-project meshes are the explicitly generated
            # ENGINE_PRIMITIVES in convert_geometry.py. Their scene components
            # carry authored per-slot overrides; there is no source UStaticMesh
            # package in the downloadable project from which to read slots.
            resolutions.extend(mesh.get("material_slots", []))
            stamp_partition_material_indices(loose, mesh, {})
            continue
        if object_name not in exact:
            raise RuntimeError(
                f"exact Unreal static-material record is missing for {object_name}"
            )
        source_slots = mesh.get("material_slots", [])
        exact_slots = exact[object_name].get("slots", [])
        section_materials = {
            int(section["section"]): int(section["material_index"])
            for section in exact[object_name].get("sections", [])
            if int(section["lod"]) == 0
        }
        resolved_slots = []
        for section_index, source_slot in enumerate(source_slots):
            # This is FMeshSectionInfoMap::Get's exact UE default: when a map
            # has no explicit LOD/section entry, the section uses the material
            # at the same index.
            material_index = section_materials.get(section_index, section_index)
            if material_index < 0 or material_index >= len(exact_slots):
                raise RuntimeError(
                    f"Unreal LOD0 section {section_index} of {object_name} maps to "
                    f"invalid material index {material_index}"
                )
            exact_slot = exact_slots[material_index]
            if int(exact_slot["index"]) != material_index:
                raise RuntimeError(f"non-contiguous Unreal material slots in {object_name}")
            material = exact_slot.get("material")
            if (
                material is not None
                and not material.startswith("/Game/")
                and material not in ENGINE_MATERIALS
            ):
                raise RuntimeError(
                    f"unsupported Engine material {material} in {object_name} "
                    f"slot {material_index}"
                )
            if material is not None and material.startswith("/Game/"):
                material_objects.add(material)
            entry = {
                "mesh": object_name,
                "slot": str(section_index),
                "material_index": material_index,
                "name": exact_slot.get("imported_slot_name")
                or exact_slot.get("slot_name")
                or source_slot.get("name")
                or str(section_index),
                "material": material,
                "resolution": "exact-unreal-section-material"
                if material is not None
                else "explicit-null-unreal-static-material",
            }
            resolutions.append(entry)
            resolved_slots.append(entry)
        mesh["material_slots"] = resolved_slots
        stamp_partition_material_indices(loose, mesh, section_materials)

    for scene_path in scene_paths:
        for actor in load_json(scene_path).get("actors", []):
            # A DecalActor's material is never reachable through a mesh slot, so
            # it only enters the manifest from the decal component itself.
            material_objects.update(
                decal["material"]
                for decal in actor.get("decals", [])
                if isinstance(decal.get("material"), str)
                and decal["material"].startswith("/Game/")
            )
            for component in actor.get("components", []):
                material_objects.update(
                    material
                    for material in component.get("override_materials", [])
                    if isinstance(material, str) and material.startswith("/Game/")
                )
                for material in component.get("override_materials", []):
                    if (
                        isinstance(material, str)
                        and material.startswith("/Engine/")
                        and material not in ENGINE_MATERIALS
                    ):
                        raise RuntimeError(
                            f"unsupported authored Engine material {material} in "
                            f"{scene_path.name} actor {actor.get('name')} "
                            f"component {component.get('name')}"
                        )
    geometry["material_resolutions"] = sorted(
        resolutions,
        key=lambda record: (record["mesh"], int(record["slot"])),
    )
    write_json_if_changed(geometry_path, geometry)
    write_json_if_changed(loose / "material-input.json", sorted(material_objects))


def report_diagnostic_materials(loose: Path, material_manifest: Path) -> None:
    """Name every mesh slot that renders as the magenta diagnostic material.

    The material manifest only records the object path a lookup failed on, so
    join it back to geometry.json's per-slot resolutions to say which mesh and
    which slot asked for it. Nothing here is fixable from the download: see
    KnownMissingProjectMaterials in Program.cs for what was ruled out.
    """
    reasons = {WORLD_GRID_MATERIAL: "unreal-unassigned-slot-fallback"}
    reasons.update(
        (record["object"], "project-material-absent-from-download")
        for record in load_json(material_manifest).get("materials", [])
        if record.get("type") == "MissingSourceMaterial"
    )
    for record in load_json(loose / "geometry.json").get("material_resolutions", []):
        reason = reasons.get(record.get("material"))
        if reason is None:
            continue
        print(
            f"ZORAH_CONVERT_DIAGNOSTIC_MATERIAL mesh={record['mesh']} "
            f"slot={record['slot']} name={record['name']} "
            f"object={record['material']} reason={reason}",
            flush=True,
        )


def validate_engine_primitive_overrides(scene_paths: list[Path]) -> None:
    """Require an authored material for every generated Engine primitive."""
    for scene_path in scene_paths:
        for actor in load_json(scene_path).get("actors", []):
            for component in actor.get("components", []):
                mesh = component.get("mesh")
                if not isinstance(mesh, str) or not mesh.startswith("/Engine/"):
                    continue
                if mesh not in ENGINE_PRIMITIVES:
                    raise RuntimeError(
                        f"unsupported Engine mesh {mesh} in {scene_path.name} "
                        f"actor {actor.get('name')} component {component.get('name')}"
                    )
                overrides = component.get("override_materials", [])
                if not overrides or not isinstance(overrides[0], str) or not overrides[0]:
                    raise RuntimeError(
                        f"Engine primitive {mesh} has no authored slot-0 material in "
                        f"{scene_path.name} actor {actor.get('name')} "
                        f"component {component.get('name')}"
                    )


def material_manifest_matches_input(materials: Path, material_input: Path) -> bool:
    if not materials.is_file() or not material_input.is_file():
        return False
    try:
        requested = load_json(material_input)
        manifest = load_json(materials)
        return not manifest.get("failures") and manifest.get("requested") == requested
    except (OSError, TypeError, ValueError):
        return False


# Shard sizing is a property of the packed tree, not of the loose inputs: the
# packer hard-links reused bundles at whatever size they already have. Both
# fingerprints therefore ignore it, and shard_setting_conflicts compares the
# requested sizes against the ones pack.json recorded.
def geometry_input_fingerprint(
    loose: Path,
    scene_paths: list[Path],
    *,
    triangles: int,
    raytracing_error: float,
) -> str:
    geometry = load_json(loose / "geometry.json")
    parts = []
    for mesh in sorted(geometry.get("meshes", []), key=lambda record: record["object"]):
        manifest = load_json(loose / mesh["parts_manifest"])
        parts.append((mesh["object"], manifest))
    return stable_hash(
        {"pipeline_version": PACK_PIPELINE_VERSION, "geometry": geometry},
        parts,
        scene_mesh_usage(scene_paths),
        {
            "triangles": triangles,
            "raytracing_error": raytracing_error,
        },
    )


def runtime_input_fingerprint(
    loose: Path,
    scene_paths: list[Path],
    *,
    max_texture_size: int,
) -> str:
    texture_manifest = load_json(loose / "textures.runtime.json")
    texture_state = []
    for record in sorted(texture_manifest.get("exported", []), key=lambda value: value["object"]):
        path = loose / record["output"]
        stat = path.stat()
        texture_state.append((record["object"], stat.st_size, stat.st_mtime_ns))
    return stable_hash(
        {"pipeline_version": PACK_PIPELINE_VERSION},
        load_json(loose / "materials.runtime.json"),
        texture_manifest,
        texture_state,
        [(path.name, load_json(path)) for path in scene_paths],
        # A raised or lowered cap re-exports the affected textures in place, so
        # it must invalidate a packed tree even when no manifest byte moved.
        {"max_texture_size": max_texture_size},
    )


def bundle_reference_exists(root: Path, reference: object) -> bool:
    if not isinstance(reference, str):
        return False
    path, separator, label = reference.partition("#")
    return bool(separator and label and path.startswith("bundles/") and (root / path).is_file())


def static_partition(record: dict) -> dict:
    # Rust's JSON round-trip may change the last decimal digit of bounds/UVs.
    # Content hashes plus topology/material counts are the authoritative packed
    # geometry identity and avoid a false full rebuild for harmless formatting.
    keys = (
        "material_slot",
        "triangles",
        "vertices",
        "meshlet_sha256",
        "geometry_sha256",
    )
    return {key: record.get(key) for key in keys}


def geometry_reuse_compatible(loose: Path, runtime: Path, raytracing_error: float) -> bool:
    """Validate an old packed geometry tree without rereading multi-GB GLBs."""
    try:
        source_geometry = load_json(loose / "geometry.json")
        runtime_geometry = load_json(runtime / "geometry.json")
        source_meshes = {mesh["object"]: mesh for mesh in source_geometry["meshes"]}
        runtime_meshes = {mesh["object"]: mesh for mesh in runtime_geometry["meshes"]}
        if source_meshes.keys() != runtime_meshes.keys():
            return False
        referenced_bundles: set[str] = set()
        for object_name, source_mesh in source_meshes.items():
            packed_mesh = runtime_meshes[object_name]
            # Material assignments live only in geometry.json; changing exact
            # Unreal material references does not change any packed mesh bytes.
            for key in ("asset_id",):
                if source_mesh.get(key) != packed_mesh.get(key):
                    return False
            source_parts = load_json(loose / source_mesh["parts_manifest"])["partitions"]
            packed_parts = packed_mesh.get("partitions", [])
            if len(source_parts) != len(packed_parts):
                return False
            for source_part, packed_part in zip(source_parts, packed_parts):
                if static_partition(source_part) != static_partition(packed_part):
                    return False
                if not math.isclose(
                    float(packed_part.get("blas_achieved_error", math.nan)),
                    raytracing_error,
                    rel_tol=1e-5,
                    abs_tol=1e-6,
                ):
                    return False
                for key in ("geometry", "meshlet"):
                    reference = packed_part.get(key)
                    if not isinstance(reference, str):
                        return False
                    bundle, separator, label = reference.partition("#")
                    if not separator or not label or not bundle.startswith("bundles/"):
                        return False
                    referenced_bundles.add(bundle)
        if not all((runtime / bundle).is_file() for bundle in referenced_bundles):
            return False
        return True
    except (KeyError, OSError, TypeError, ValueError):
        return False


def recover_interrupted_install(output: Path, next_output: Path, old_output: Path) -> None:
    """Finish a tree swap that stopped between its two renames.

    The packed tree is staged in .next and the previous tree parked in .old, so
    an interrupted swap can leave no output directory at all. Both stagers write
    pack.json last, which makes it the completion marker for .next.
    """
    if not output.exists():
        if (next_output / "pack.json").is_file():
            os.replace(next_output, output)
            print(f"ZORAH_CONVERT_INSTALL_RECOVERED source={next_output.name}", flush=True)
        elif old_output.is_dir():
            os.replace(old_output, output)
            print(f"ZORAH_CONVERT_INSTALL_RESTORED source={old_output.name}", flush=True)
    if output.exists() and old_output.exists():
        # The swap itself completed; only the backup removal was interrupted.
        shutil.rmtree(old_output)


def install_packed_output(pack_output: Path, output: Path, old_output: Path) -> None:
    """Replace the runtime tree with a staged pack, restoring it if the swap fails.

    Packing is staged only when a previous tree has to survive it, so
    pack_output == output means the pack landed in place and needs no swap.
    output exists in both cases once packing is done, so it cannot stand in for
    that decision here.
    """
    if pack_output == output:
        return
    if old_output.exists():
        raise RuntimeError(f"replacement backup appeared mid-run: {old_output}")
    os.replace(output, old_output)
    try:
        os.replace(pack_output, output)
    except BaseException:
        os.replace(old_output, output)
        raise
    shutil.rmtree(old_output)


def pack_versions_match(runtime: Path) -> bool:
    """Refuse packed artifacts whose formats predate the current engine.

    Nothing in the loose tree changes when a bundle or meshlet format is bumped,
    so without these stamps the pipeline keeps or hard-links payloads the runtime
    rejects as WrongVersion. Only bundle_format_version is required: the packer
    has always written it, while trees packed before it stamped the meshlet and
    pipeline versions carry no evidence either way.
    """
    try:
        state = load_json(runtime / "pack.json")
    except (OSError, TypeError, ValueError):
        return False
    if state.get("bundle_format_version") != EXPECTED_BUNDLE_FORMAT_VERSION:
        return False
    return all(
        state.get(key, expected) == expected
        for key, expected in (
            ("meshlet_asset_version", EXPECTED_MESHLET_ASSET_VERSION),
            ("pack_pipeline_version", PACK_PIPELINE_VERSION),
        )
    )


def shard_setting_conflicts(
    runtime: Path, geometry_shard_bytes: int, texture_shard_bytes: int
) -> list[str]:
    """Report requested shard sizes a packed tree cannot adopt through reuse."""
    try:
        state = load_json(runtime / "pack.json")
    except (OSError, TypeError, ValueError):
        return []
    conflicts = []
    for flag, key, requested in (
        ("--geometry-shard-gib", "geometry_shard_bytes", geometry_shard_bytes),
        ("--texture-shard-gib", "texture_shard_bytes", texture_shard_bytes),
    ):
        recorded = state.get(key)
        if isinstance(recorded, int) and recorded != requested:
            conflicts.append(f"{flag} ({recorded} packed, {requested} requested)")
    return conflicts


def scenes_changed(runtime: Path, scene_paths: list[Path]) -> bool:
    """The Solari capacity audit's per-level working set is scene-derived."""
    try:
        return any(
            load_json(runtime / "scenes" / path.name) != load_json(path)
            for path in scene_paths
        )
    except (OSError, TypeError, ValueError):
        return True


def pack_state_matches(
    runtime: Path, geometry_fingerprint: str, runtime_fingerprint: str
) -> bool:
    try:
        state = load_json(runtime / "pack.json")
        return (
            state.get("format") == PACK_STATE_FORMAT
            and state.get("geometry_fingerprint") == geometry_fingerprint
            and state.get("runtime_fingerprint") == runtime_fingerprint
            and pack_versions_match(runtime)
        )
    except (OSError, TypeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path, help="extracted Zorah UE project")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"converted asset directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--dotnet", help="dotnet executable; auto-detected by default")
    parser.add_argument("--skip-dotnet-build", action="store_true")
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--refresh-source",
        action="store_true",
        help="refresh cached scene/material metadata from the fixed UE source tree",
    )
    parser.add_argument(
        "--rebuild-geometry",
        action="store_true",
        help="discard and rebuild the expensive loose geometry cache",
    )
    parser.add_argument(
        "--scenes-only",
        action="store_true",
        help="refresh the three light/actor manifests in an existing runtime tree",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--loose-only",
        action="store_true",
        help="refresh and verify the resumable loose conversion without packing bundles",
    )
    parser.add_argument(
        "--discard-work",
        action="store_true",
        help="delete the resumable loose conversion cache after success",
    )
    parser.add_argument("--triangles", type=int, default=250_000)
    parser.add_argument(
        "--raytracing-error",
        type=float,
        default=0.02,
        help="absolute meshlet LOD error in metres for Solari BLAS geometry",
    )
    parser.add_argument("--max-texture-size", type=int, default=8192)
    parser.add_argument(
        "--material-bake-size",
        type=int,
        default=8192,
        help="maximum dimension for flattened Zorah material textures",
    )
    # The ctt compressors are single-threaded per texture, so jobs are the only
    # texture-phase parallelism; sizes are skewed, so more lanes than cores/4
    # mostly just raises peak memory.
    parser.add_argument("--texture-jobs", type=int, default=min(16, os.cpu_count() or 8))
    parser.add_argument(
        "--geometry-jobs",
        type=int,
        default=0,
        help="concurrent partition conversions in zorah_pack; 0 sizes from the packer host's cores",
    )
    parser.add_argument("--verbose", action="store_true", help="log every converted texture/material")
    parser.add_argument("--cargo", help="cargo executable; Windows cargo is preferred for /mnt/c outputs")
    parser.add_argument("--geometry-shard-gib", type=float, default=0.5)
    parser.add_argument("--texture-shard-gib", type=float, default=2.0)
    parser.add_argument(
        "--reshard-existing",
        action="store_true",
        help="quickly rewrite an existing runtime tree with current shard sizing",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output = args.output.resolve()
    if not (project_root / "Content").is_dir():
        parser.error(f"project has no Content directory: {project_root}")
    if args.triangles < 128:
        parser.error("--triangles must be at least 128")
    if not math.isfinite(args.raytracing_error) or args.raytracing_error < 0.0:
        parser.error("--raytracing-error must be finite and non-negative")
    if args.max_texture_size <= 0 or args.max_texture_size % 4:
        parser.error("--max-texture-size must be a positive multiple of four")
    if args.material_bake_size <= 0 or args.material_bake_size % 4:
        parser.error("--material-bake-size must be a positive multiple of four")
    if args.texture_jobs < 1:
        parser.error("--texture-jobs must be at least one")
    if args.geometry_jobs < 0:
        parser.error("--geometry-jobs must be zero (auto) or positive")
    if args.geometry_shard_gib < 0.0625:
        parser.error("--geometry-shard-gib must be at least 0.0625")
    if args.texture_shard_gib < 0.0625:
        parser.error("--texture-shard-gib must be at least 0.0625")
    for next_suffix, old_suffix in ((".next", ".old"), (".reshard.next", ".reshard.old")):
        recover_interrupted_install(
            output,
            output.parent / f".{output.name}{next_suffix}",
            output.parent / f".{output.name}{old_suffix}",
        )
    if args.scenes_only and not output.is_dir():
        parser.error(f"--scenes-only needs an existing runtime asset directory: {output}")
    if args.reshard_existing and not output.is_dir():
        parser.error(f"--reshard-existing needs an existing runtime asset directory: {output}")
    if args.scenes_only and args.reshard_existing:
        parser.error("--scenes-only and --reshard-existing cannot be combined")
    if args.loose_only and (args.scenes_only or args.reshard_existing):
        parser.error("--loose-only cannot be combined with scene-only or reshard modes")
    cargo = find_cargo(args.cargo, output)
    work = output.parent / f".{output.name}.convert"
    loose = work / "loose"

    if args.reshard_existing:
        next_output = output.parent / f".{output.name}.reshard.next"
        old_output = output.parent / f".{output.name}.reshard.old"
        if next_output.exists() or old_output.exists():
            parser.error("an interrupted reshard staging directory already exists")
        geometry_shard_bytes = int(args.geometry_shard_gib * 1024 * 1024 * 1024)
        run([
            sys.executable,
            str(CONVERT_DIRECTORY / "reshard_bundles.py"),
            str(output),
            str(next_output),
            "--geometry-shard-bytes",
            str(geometry_shard_bytes),
        ])
        verify_runtime(next_output, cargo, audit_capacity=True)
        os.replace(output, old_output)
        try:
            os.replace(next_output, output)
        except BaseException:
            os.replace(old_output, output)
            raise
        shutil.rmtree(old_output)
        print(f"ZORAH_CONVERT_RESHARD_DONE output={output}")
        return 0

    dotnet = find_dotnet(args.dotnet)
    if not args.skip_dotnet_build:
        run([dotnet, "build", dotnet_path(PROJECT, dotnet), "--configuration", "Release"])
    if not ASSEMBLY.is_file():
        raise RuntimeError(f"ZorahConvert assembly was not built: {ASSEMBLY}")

    if args.scenes_only:
        next_scene_directory = output.parent / f".{output.name}.scenes.next"
        if next_scene_directory.exists():
            parser.error(f"scene staging directory already exists: {next_scene_directory}")
        cue4parse(dotnet, project_root, "scene-manifests", next_scene_directory)
        next_paths = [next_scene_directory / f"{level}.json" for level in LEVELS]
        if not all(scene_manifest_is_current(path) for path in next_paths):
            raise RuntimeError("scene-only export did not produce current light manifests")
        validate_blueprint_light_archetypes(next_paths)
        validate_child_actor_placement(next_paths)
        install_scene_manifests(next_paths, work, loose, output)
        next_scene_directory.rmdir()
        verify_runtime(output, cargo, audit_capacity=True)
        print(f"ZORAH_CONVERT_SCENES_DONE output={output}")
        return 0

    scene_directory = work / "scenes"
    scene_paths = [scene_directory / f"{level}.json" for level in LEVELS]
    scene_directory.mkdir(parents=True, exist_ok=True)
    if args.refresh_source or not scene_cache_is_current(scene_directory, scene_paths):
        next_scene_directory = work / "scenes.next"
        next_scene_paths = [
            next_scene_directory / f"{level}.json" for level in LEVELS
        ]
        if not next_scene_directory.exists():
            cue4parse(dotnet, project_root, "scene-manifests", next_scene_directory)
        if not all(scene_manifest_is_current(path) for path in next_scene_paths):
            raise RuntimeError(
                "strict scene export differs from the fixed Zorah 1.1.0 inventory"
            )
        for level, path in zip(LEVELS, scene_paths):
            os.replace(next_scene_directory / f"{level}.json", path)
        next_scene_directory.rmdir()
        stamp_scene_cache(scene_directory)

    if args.rebuild_geometry:
        # Scoped to geometry: exported textures and material bakes in the same
        # cache have no bearing on extraction or partitioning.
        shutil.rmtree(loose / "geometry", ignore_errors=True)
        (loose / "geometry.json").unlink(missing_ok=True)
        (loose / GEOMETRY_INPUTS_NAME).unlink(missing_ok=True)
    geometry_built = not (loose / "geometry.json").is_file()
    if geometry_built:
        install_geometry_build(project_root, loose, work, scene_paths, args.triangles)
    require_successful_geometry(loose)
    rebuilt_meshes = reconcile_geometry_cache(
        project_root,
        loose,
        work,
        scene_paths,
        args.triangles,
    )
    (loose / "scenes").mkdir(exist_ok=True)
    for level, path in zip(LEVELS, scene_paths):
        shutil.copy2(path, loose / "scenes" / f"{level}.json")
    validate_engine_primitive_overrides(scene_paths)
    validate_blueprint_light_archetypes(scene_paths)
    validate_child_actor_placement(scene_paths)

    geometry_document = load_json(loose / "geometry.json")
    mesh_material_input = work / "mesh-material-input.json"
    write_json_if_changed(
        mesh_material_input,
        sorted(
            mesh["object"]
            for mesh in geometry_document.get("meshes", [])
            if mesh["object"].startswith("/Game/")
        ),
    )
    mesh_materials = work / "mesh-materials.source.json"
    if args.refresh_source or not mesh_material_manifest_is_current(
        mesh_materials, geometry_document
    ):
        next_mesh_materials = work / "mesh-materials.source.next.json"
        if next_mesh_materials.exists():
            parser.error(
                f"interrupted mesh-material update exists: {next_mesh_materials}"
            )
        cue4parse(
            dotnet,
            project_root,
            "mesh-material-manifest",
            mesh_material_input,
            next_mesh_materials,
        )
        os.replace(next_mesh_materials, mesh_materials)
    apply_exact_mesh_materials(loose, mesh_materials, scene_paths)

    materials = loose / "materials.source.json"
    if args.refresh_source or not material_manifest_matches_input(
        materials, loose / "material-input.json"
    ):
        next_materials = work / "materials.source.next.json"
        if next_materials.exists() and not material_manifest_matches_input(
            next_materials, loose / "material-input.json"
        ):
            next_materials.unlink()
        if not next_materials.exists():
            cue4parse(
                dotnet,
                project_root,
                "material-manifest",
                loose / "material-input.json",
                next_materials,
            )
        os.replace(next_materials, materials)
    report_diagnostic_materials(loose, materials)

    textures = loose / "textures.json"
    if args.refresh_source or not texture_inventory_matches(materials, textures):
        if args.refresh_source:
            # Nothing in the texture reuse key covers a repainted source texture,
            # so a source refresh has to drop the payloads it exported from it.
            discard_texture_exports(loose)
        refresh_texture_inventory(
            dotnet,
            project_root,
            materials,
            loose,
            work / "textures.inventory.next",
            refresh=args.refresh_source,
        )

    texture_command = [
        sys.executable,
        str(CONVERT_DIRECTORY / "texture_source.py"),
        str(textures),
        str(project_root),
        str(loose),
        "--max-size",
        str(args.max_texture_size),
        "--jobs",
        str(args.texture_jobs),
    ]
    # An interrupted first texture pass can leave valid PNGs before the final
    # manifest is committed. The texture exporter can validate and reuse them.
    if (loose / "textures.exported.json").is_file() or next(
        loose.rglob("*.png"), None
    ) is not None:
        texture_command.append("--resume")
    if args.verbose:
        texture_command.append("--verbose")
    run(texture_command)
    material_bake_command = [
        sys.executable,
        str(CONVERT_DIRECTORY / "material_bake.py"),
        str(materials),
        str(loose / "textures.exported.json"),
        str(loose),
        "--max-size",
        str(min(args.material_bake_size, args.max_texture_size)),
        "--jobs",
        str(args.texture_jobs),
    ]
    if (loose / "material_bakes.json").is_file():
        material_bake_command.append("--resume")
    if args.verbose:
        material_bake_command.append("--verbose")
    run(material_bake_command)
    verify_command = [sys.executable, str(CONVERT_DIRECTORY / "verify_conversion.py"), str(loose)]
    if not geometry_built:
        verify_command.append("--skip-geometry-payloads")
        if rebuilt_meshes:
            # Meshes the delta just converted are as fresh as a full build's, and
            # their GLB payloads are otherwise verified on no run at all.
            payload_meshes = work / "geometry-payload-meshes.json"
            write_json_if_changed(payload_meshes, rebuilt_meshes)
            verify_command.extend(("--payload-meshes", str(payload_meshes)))
    run(verify_command)
    if args.loose_only:
        print(f"ZORAH_CONVERT_LOOSE_DONE output={loose}")
        return 0
    geometry_shard_bytes = int(args.geometry_shard_gib * 1024 * 1024 * 1024)
    texture_shard_bytes = int(args.texture_shard_gib * 1024 * 1024 * 1024)
    packed_scene_paths = [loose / "scenes" / f"{level}.json" for level in LEVELS]
    geometry_fingerprint = geometry_input_fingerprint(
        loose,
        packed_scene_paths,
        triangles=args.triangles,
        raytracing_error=args.raytracing_error,
    )
    runtime_fingerprint = runtime_input_fingerprint(
        loose,
        packed_scene_paths,
        max_texture_size=args.max_texture_size,
    )
    next_output = output.parent / f".{output.name}.next"
    if not args.rebuild_geometry:
        for candidate in (output, next_output):
            conflicts = (
                shard_setting_conflicts(candidate, geometry_shard_bytes, texture_shard_bytes)
                if candidate.is_dir()
                else []
            )
            if conflicts:
                parser.error(
                    f"{candidate.name} was packed with different shard sizing: "
                    + ", ".join(conflicts)
                    + "; apply --geometry-shard-gib with --reshard-existing, or pass "
                    "--rebuild-geometry to repack from scratch"
                )
    audit_capacity = scenes_changed(output, packed_scene_paths)
    if output.is_dir() and pack_state_matches(output, geometry_fingerprint, runtime_fingerprint):
        verify_runtime(output, cargo, audit_capacity=audit_capacity)
        if args.discard_work:
            shutil.rmtree(work, ignore_errors=True)
        print(f"ZORAH_CONVERT_DONE output={output} reused=all")
        return 0

    pack_output = next_output if output.exists() else output
    resume_packed_output = pack_output.exists()
    if resume_packed_output and not pack_state_matches(
        pack_output, geometry_fingerprint, runtime_fingerprint
    ):
        parser.error(
            f"interrupted pack staging directory does not match current inputs: {pack_output}"
        )
    # A complete pack for stale inputs is still a valid hard-link source.
    reuse_root = None
    if not args.rebuild_geometry:
        reuse_root = next(
            (
                candidate
                for candidate in (output, next_output)
                if candidate.is_dir()
                and candidate != pack_output
                and pack_versions_match(candidate)
            ),
            None,
        )
    reuse_geometry = reuse_root is not None and geometry_reuse_compatible(
        loose, reuse_root, args.raytracing_error
    )
    audit_capacity = audit_capacity or not reuse_geometry
    pack_arguments: list[str | Path] = [
        loose,
        pack_output,
        "--geometry-shard-bytes",
        str(geometry_shard_bytes),
        "--texture-shard-bytes",
        str(texture_shard_bytes),
        "--raytracing-error",
        str(args.raytracing_error),
        "--geometry-fingerprint",
        geometry_fingerprint,
        "--runtime-fingerprint",
        runtime_fingerprint,
        "--texture-jobs",
        str(args.texture_jobs),
    ]
    if args.geometry_jobs:
        pack_arguments.extend(("--geometry-jobs", str(args.geometry_jobs)))
    if not resume_packed_output:
        if reuse_root is not None:
            pack_arguments.extend(("--reuse-from", reuse_root))
        run(cargo_command(cargo, "zorah_pack", *pack_arguments))
    else:
        print(f"ZORAH_CONVERT_PACK_RESUME output={pack_output}")
    verify_runtime(pack_output, cargo, audit_capacity=audit_capacity)
    install_packed_output(pack_output, output, output.parent / f".{output.name}.old")
    if next_output.exists() and next_output != pack_output:
        # A stale staging tree that only served as a hard-link source.
        shutil.rmtree(next_output)
    if args.discard_work:
        shutil.rmtree(work, ignore_errors=True)
    print(f"ZORAH_CONVERT_DONE output={output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(
            f"ZORAH_CONVERT_ERROR command exited with code {error.returncode}",
            file=sys.stderr,
        )
        raise SystemExit(error.returncode)
