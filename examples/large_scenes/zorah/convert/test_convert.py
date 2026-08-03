import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import convert
import verify_conversion


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_pack_state(runtime: Path, **overrides: object) -> None:
    state = {
        "format": convert.PACK_STATE_FORMAT,
        "bundle_format_version": convert.EXPECTED_BUNDLE_FORMAT_VERSION,
        "meshlet_asset_version": convert.EXPECTED_MESHLET_ASSET_VERSION,
        "pack_pipeline_version": convert.PACK_PIPELINE_VERSION,
        "geometry_fingerprint": "geometry",
        "runtime_fingerprint": "runtime",
        "geometry_shard_bytes": 512 * 1024 * 1024,
        "texture_shard_bytes": 2 * 1024 * 1024 * 1024,
    }
    state.update(overrides)
    write_json(runtime / "pack.json", state)


def asset_id_for(mesh: str) -> str:
    return hashlib.sha256(mesh.encode()).hexdigest()[:16]


class GeometryCache:
    """Minimal loose geometry cache plus the source packages it was built from."""

    def __init__(self, root: Path, meshes: list[str], triangles: int = 250_000) -> None:
        self.project_root = root / "project"
        self.loose = root / "loose"
        self.work = root / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.scene = root / "scene.json"
        records = []
        for mesh in meshes:
            asset_id = asset_id_for(mesh)
            self.write_source(mesh, b"source")
            write_json(
                self.loose / "geometry" / asset_id / "parts" / "manifest.json",
                {"partitions": [{"material_slot": 0, "meshlet": "part-000000.glb"}]},
            )
            records.append(
                {
                    "object": mesh,
                    "asset_id": asset_id,
                    "parts_manifest": f"geometry/{asset_id}/parts/manifest.json",
                    "material_slots": [],
                }
            )
        self.write_scene(meshes)
        # Written the way convert_geometry.py writes it, so byte comparisons in
        # the tests see the same manifest shape and formatting.
        convert.write_json_if_changed(
            self.loose / "geometry.json",
            {
                "format": "zorah-geometry-manifest-v3",
                "project_root": str(self.project_root),
                "scenes": [
                    {
                        "level": "GreenHouse_Level",
                        "source": str(self.scene.resolve()),
                        "path": "scenes/GreenHouse_Level.json",
                        "actor_count": 0,
                    }
                ],
                "referenced_mesh_count": len(meshes),
                "project_mesh_count": len(meshes),
                "external_meshes": [],
                "generated_engine_meshes": [],
                "selected_mesh_count": len(meshes),
                "triangle_partition_limit": triangles,
                "meshes": records,
                "material_resolutions": [],
                "failures": [],
            },
        )
        # install_geometry_build stamps provenance before reconcile fills in
        # the per-mesh records; a stampless tree is refused as legacy.
        write_json(
            self.loose / convert.GEOMETRY_INPUTS_NAME,
            {
                "format": convert.GEOMETRY_INPUTS_FORMAT,
                "pipeline_version": convert.GEOMETRY_PIPELINE_VERSION,
            },
        )

    def write_source(self, mesh: str, payload: bytes) -> None:
        package = convert.source_package(self.project_root, mesh)
        package.parent.mkdir(parents=True, exist_ok=True)
        package.write_bytes(payload)

    def write_scene(self, meshes: list[str]) -> None:
        write_json(
            self.scene,
            {
                "format": "zorah-scene-manifest-v3",
                "level": "GreenHouse_Level",
                "actors": [],
                "referenced_meshes": meshes,
            },
        )

    def manifest(self) -> dict:
        return json.loads((self.loose / "geometry.json").read_text())

    def reconcile(self, triangles: int = 250_000) -> list[str]:
        return convert.reconcile_geometry_cache(
            self.project_root, self.loose, self.work, [self.scene], triangles
        )

    def fake_delta(self, delta_output: Path, scene_path: Path) -> None:
        """Stand in for convert_geometry.py's output for the requested meshes."""
        meshes = json.loads(scene_path.read_text())["referenced_meshes"]
        records = []
        for mesh in meshes:
            asset_id = asset_id_for(mesh)
            write_json(
                delta_output / "geometry" / asset_id / "parts" / "manifest.json",
                {"partitions": [{"material_slot": 0, "meshlet": "rebuilt.glb"}]},
            )
            records.append(
                {
                    "object": mesh,
                    "asset_id": asset_id,
                    "parts_manifest": f"geometry/{asset_id}/parts/manifest.json",
                    "material_slots": [],
                }
            )
        write_json(
            delta_output / "geometry.json",
            {
                "format": "zorah-geometry-manifest-v3",
                "meshes": records,
                "material_resolutions": [],
                "failures": [],
            },
        )


@contextmanager
def stub_manifest_inventory():
    """Accept any scene manifest, leaving only the converter stamp under test."""
    original = convert.scene_manifest_is_current
    convert.scene_manifest_is_current = lambda path: True
    try:
        yield
    finally:
        convert.scene_manifest_is_current = original


@contextmanager
def stub_geometry_delta(cache: GeometryCache):
    """Run the delta stage without invoking convert_geometry.py."""
    calls: list[list[str]] = []
    original = convert.run

    def fake_run(command: list[str]) -> None:
        calls.append(command)
        cache.fake_delta(Path(command[3]), Path(command[4]))

    convert.run = fake_run
    try:
        yield calls
    finally:
        convert.run = original


class IncrementalConversionTests(unittest.TestCase):
    def test_static_partition_uses_content_identity_not_float_formatting(self):
        source = {
            "material_slot": 2,
            "triangles": 10,
            "vertices": 12,
            "meshlet_sha256": "meshlet",
            "geometry_sha256": "geometry",
            "uv_max": [0.1],
            "meshlet": "part.glb",
        }
        packed = {
            **source,
            "uv_max": [0.10000000000000001],
            "meshlet": "bundles/zorah-000.zorah_bundle#g/0/meshlet",
            "geometry": "bundles/zorah-000.zorah_bundle#g/0/meshlet_blas",
            "blas_triangles": 5,
        }
        self.assertEqual(convert.static_partition(source), convert.static_partition(packed))

    def test_geometry_reuse_accepts_matching_partition_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loose = root / "loose"
            runtime = root / "runtime"
            mesh = {
                "object": "mesh",
                "asset_id": "asset",
                "material_slots": [],
                "parts_manifest": "geometry/parts.json",
            }
            partition = {
                "material_slot": 0,
                "triangles": 10,
                "vertices": 12,
                "meshlet_sha256": "meshlet",
                "geometry_sha256": "geometry",
            }
            write_json(loose / "geometry.json", {"meshes": [mesh]})
            write_json(loose / "geometry/parts.json", {"partitions": [partition]})
            bundle = runtime / "bundles/zorah-000.zorah_bundle"
            bundle.parent.mkdir(parents=True)
            bundle.touch()
            packed = {
                key: value for key, value in mesh.items() if key != "parts_manifest"
            }
            packed["partitions"] = [
                {
                    **partition,
                    "geometry": "bundles/zorah-000.zorah_bundle#g/0/meshlet_blas",
                    "meshlet": "bundles/zorah-000.zorah_bundle#g/0/meshlet",
                    "blas_achieved_error": 0.02,
                }
            ]
            write_json(runtime / "geometry.json", {"meshes": [packed]})

            self.assertTrue(convert.geometry_reuse_compatible(loose, runtime, 0.02))
            mesh["material_slots"] = [{"material": "/Game/New.New"}]
            write_json(loose / "geometry.json", {"meshes": [mesh]})
            self.assertTrue(convert.geometry_reuse_compatible(loose, runtime, 0.02))
            packed["partitions"][0]["meshlet_sha256"] = "changed"
            write_json(runtime / "geometry.json", {"meshes": [packed]})
            self.assertFalse(convert.geometry_reuse_compatible(loose, runtime, 0.02))

    def test_exact_unreal_material_slots_replace_geometry_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loose = root / "loose"
            mesh = "/Game/Meshes/SM_Test.SM_Test"
            write_json(
                loose / "geometry.json",
                {
                    "meshes": [
                        {
                            "object": mesh,
                            "parts_manifest": "geometry/asset/parts/manifest.json",
                            "material_slots": [
                                {
                                    "slot": "0",
                                    "name": "mesh-description-name",
                                    "material": None,
                                    "resolution": "pending-exact-unreal-static-material",
                                }
                            ],
                        }
                    ]
                },
            )
            write_json(
                loose / "geometry" / "asset" / "parts" / "manifest.json",
                {"partitions": [{"material_slot": 0, "meshlet": "part-000000.glb"}]},
            )
            exact = root / "mesh-materials.json"
            material = "/Game/Materials/MI_Exact.MI_Exact"
            unused_material = "/Game/Materials/MI_Unused.MI_Unused"
            write_json(
                exact,
                {
                    "format": "zorah-mesh-material-manifest-v2",
                    "meshes": [
                        {
                            "object": mesh,
                            "slots": [
                                {
                                    "index": 0,
                                    "material": unused_material,
                                    "slot_name": "UnusedSlot",
                                    "imported_slot_name": "UnusedImportedSlot",
                                },
                                {
                                    "index": 1,
                                    "material": material,
                                    "slot_name": "SlotName",
                                    "imported_slot_name": "ImportedSlotName",
                                }
                            ],
                            "sections": [
                                {"lod": 0, "section": 0, "material_index": 1}
                            ],
                        }
                    ],
                    "failures": [],
                },
            )
            scene = root / "scene.json"
            override = "/Game/Materials/MI_Override.MI_Override"
            write_json(
                scene,
                {
                    "actors": [
                        {"components": [{"override_materials": [override]}]}
                    ]
                },
            )

            convert.apply_exact_mesh_materials(loose, exact, [scene])

            geometry = json.loads((loose / "geometry.json").read_text())
            slot = geometry["meshes"][0]["material_slots"][0]
            self.assertEqual(slot["material"], material)
            self.assertEqual(slot["name"], "ImportedSlotName")
            self.assertEqual(slot["resolution"], "exact-unreal-section-material")
            self.assertEqual(slot["material_index"], 1)
            partitions = json.loads(
                (loose / "geometry" / "asset" / "parts" / "manifest.json").read_text()
            )["partitions"]
            self.assertEqual(partitions[0]["material_index"], 1)
            self.assertEqual(
                json.loads((loose / "material-input.json").read_text()),
                [material, override],
            )

    def test_slot_name_falls_back_to_the_section_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loose = root / "loose"
            mesh = "/Game/Meshes/SM_Test.SM_Test"
            write_json(
                loose / "geometry.json",
                {"meshes": [{"object": mesh, "material_slots": [{"name": ""}]}]},
            )
            exact = root / "mesh-materials.json"
            write_json(
                exact,
                {
                    "format": "zorah-mesh-material-manifest-v2",
                    "meshes": [
                        {
                            "object": mesh,
                            "slots": [{"index": 0, "material": None}],
                            "sections": [],
                        }
                    ],
                    "failures": [],
                },
            )

            convert.apply_exact_mesh_materials(loose, exact, [])

            geometry = json.loads((loose / "geometry.json").read_text())
            self.assertEqual(geometry["meshes"][0]["material_slots"][0]["name"], "0")

    def test_override_materials_index_by_material_slot(self):
        overrides = ["", "/Game/Materials/MI_Override.MI_Override"]
        slots = ["/Game/Materials/MI_Base.MI_Base"]
        self.assertEqual(
            verify_conversion.effective_material(overrides, slots, 0, 1),
            "/Game/Materials/MI_Override.MI_Override",
        )
        # Manifests without material_index stay section-indexed.
        self.assertEqual(
            verify_conversion.effective_material(overrides, slots, 0, 0),
            "/Game/Materials/MI_Base.MI_Base",
        )

    def test_diagnostic_materials_name_the_requesting_mesh_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loose = root / "loose"
            missing = "/Game/Materials/MI_Absent.MI_Absent"
            write_json(
                loose / "geometry.json",
                {
                    "material_resolutions": [
                        {
                            "mesh": "/Game/Meshes/SM_Bush.SM_Bush",
                            "slot": "1",
                            "name": "MI_Absent",
                            "material": missing,
                        },
                        {
                            "mesh": "/Game/Meshes/SM_Cap.SM_Cap",
                            "slot": "0",
                            "name": "M_Stone",
                            "material": convert.WORLD_GRID_MATERIAL,
                        },
                        {
                            "mesh": "/Game/Meshes/SM_Pot.SM_Pot",
                            "slot": "0",
                            "name": "MI_Pot",
                            "material": "/Game/Materials/MI_Pot.MI_Pot",
                        },
                    ]
                },
            )
            manifest = root / "materials.source.json"
            write_json(
                manifest,
                {
                    "materials": [
                        {"object": missing, "type": "MissingSourceMaterial"},
                        {
                            "object": "/Game/Materials/MI_Pot.MI_Pot",
                            "type": "MaterialInstanceConstant",
                        },
                    ]
                },
            )

            output = io.StringIO()
            with redirect_stdout(output):
                convert.report_diagnostic_materials(loose, manifest)

            reported = [
                line
                for line in output.getvalue().splitlines()
                if line.startswith("ZORAH_CONVERT_DIAGNOSTIC_MATERIAL")
            ]
            self.assertEqual(
                reported,
                [
                    "ZORAH_CONVERT_DIAGNOSTIC_MATERIAL "
                    "mesh=/Game/Meshes/SM_Bush.SM_Bush slot=1 name=MI_Absent "
                    f"object={missing} reason=project-material-absent-from-download",
                    "ZORAH_CONVERT_DIAGNOSTIC_MATERIAL "
                    "mesh=/Game/Meshes/SM_Cap.SM_Cap slot=0 name=M_Stone "
                    f"object={convert.WORLD_GRID_MATERIAL} "
                    "reason=unreal-unassigned-slot-fallback",
                ],
            )

    def test_engine_primitive_requires_authored_override(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "scene.json"
            write_json(
                scene,
                {
                    "actors": [
                        {
                            "name": "Actor",
                            "components": [
                                {
                                    "name": "Component",
                                    "mesh": "/Engine/BasicShapes/Cube.Cube",
                                    "override_materials": [],
                                }
                            ],
                        }
                    ]
                },
            )
            with self.assertRaisesRegex(RuntimeError, "no authored slot-0 material"):
                convert.validate_engine_primitive_overrides([scene])

    def test_pack_state_requires_both_domains(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            write_pack_state(runtime)
            self.assertTrue(convert.pack_state_matches(runtime, "geometry", "runtime"))
            self.assertFalse(convert.pack_state_matches(runtime, "changed", "runtime"))
            self.assertFalse(convert.pack_state_matches(runtime, "geometry", "changed"))

    def test_pack_state_rejects_superseded_format_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            write_pack_state(
                runtime, meshlet_asset_version=convert.EXPECTED_MESHLET_ASSET_VERSION + 1
            )
            self.assertFalse(convert.pack_state_matches(runtime, "geometry", "runtime"))
            write_pack_state(runtime, bundle_format_version=1)
            self.assertFalse(convert.pack_state_matches(runtime, "geometry", "runtime"))
            write_pack_state(runtime, pack_pipeline_version="unstamped")
            self.assertFalse(convert.pack_state_matches(runtime, "geometry", "runtime"))
            state = json.loads((runtime / "pack.json").read_text())
            del state["meshlet_asset_version"]
            del state["pack_pipeline_version"]
            write_json(runtime / "pack.json", state)
            # Trees packed before the stamps existed carry no evidence either way.
            self.assertTrue(convert.pack_state_matches(runtime, "geometry", "runtime"))

    def test_shard_sizing_conflicts_are_reported_per_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            gibibyte = 1024 * 1024 * 1024
            write_pack_state(runtime)
            self.assertEqual(
                convert.shard_setting_conflicts(runtime, gibibyte // 2, 2 * gibibyte),
                [],
            )
            conflicts = convert.shard_setting_conflicts(runtime, gibibyte, 2 * gibibyte)
            self.assertEqual(len(conflicts), 1)
            self.assertIn("--geometry-shard-gib", conflicts[0])
            # An unrecorded size cannot be judged and must not block reuse.
            write_json(runtime / "pack.json", {"format": convert.PACK_STATE_FORMAT})
            self.assertEqual(
                convert.shard_setting_conflicts(runtime, gibibyte, gibibyte), []
            )

    def test_texture_size_cap_invalidates_a_packed_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            loose = Path(directory) / "loose"
            write_json(loose / "materials.runtime.json", {"materials": []})
            texture = loose / "textures" / "texture.png"
            texture.parent.mkdir(parents=True, exist_ok=True)
            texture.write_bytes(b"png")
            write_json(
                loose / "textures.runtime.json",
                {"exported": [{"object": "T", "output": "textures/texture.png"}]},
            )
            scene = loose / "scenes" / "GreenHouse_Level.json"
            write_json(scene, {"level": "GreenHouse_Level", "actors": []})

            clamped = convert.runtime_input_fingerprint(
                loose, [scene], max_texture_size=2048
            )
            raised = convert.runtime_input_fingerprint(
                loose, [scene], max_texture_size=8192
            )
            self.assertNotEqual(clamped, raised)

    def test_geometry_cache_stamps_per_mesh_inputs_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as directory:
            mesh = "/Game/Meshes/SM_A.SM_A"
            cache = GeometryCache(Path(directory), [mesh])
            manifest = (cache.loose / "geometry.json").read_bytes()

            self.assertEqual(cache.reconcile(), [])

            inputs = json.loads((cache.loose / convert.GEOMETRY_INPUTS_NAME).read_text())
            self.assertEqual(inputs["meshes"][mesh]["source_size"], len(b"source"))
            # Source timestamps must stay out of the manifest the packer hashes.
            self.assertEqual((cache.loose / "geometry.json").read_bytes(), manifest)
            self.assertEqual(cache.reconcile(), [])

    def test_changed_source_package_rebuilds_only_that_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            first = "/Game/Meshes/SM_A.SM_A"
            second = "/Game/Meshes/SM_B.SM_B"
            cache = GeometryCache(Path(directory), [first, second])
            cache.reconcile()
            cache.write_source(second, b"source-changed")

            with stub_geometry_delta(cache) as calls:
                self.assertEqual(cache.reconcile(), [second])

            self.assertEqual(
                json.loads(Path(calls[0][4]).read_text())["referenced_meshes"], [second]
            )
            manifests = {
                record["object"]: json.loads(
                    (cache.loose / record["parts_manifest"]).read_text()
                )
                for record in cache.manifest()["meshes"]
            }
            self.assertEqual(manifests[first]["partitions"][0]["meshlet"], "part-000000.glb")
            self.assertEqual(manifests[second]["partitions"][0]["meshlet"], "rebuilt.glb")
            self.assertEqual(cache.reconcile(), [])

    def test_wholesale_source_churn_refuses_the_delta_path(self):
        with tempfile.TemporaryDirectory() as directory:
            meshes = ["/Game/Meshes/SM_A.SM_A", "/Game/Meshes/SM_B.SM_B"]
            cache = GeometryCache(Path(directory), meshes)
            cache.reconcile()
            for mesh in meshes:
                cache.write_source(mesh, b"source-changed")
            with self.assertRaisesRegex(RuntimeError, "--rebuild-geometry"):
                cache.reconcile()

    def test_triangle_limit_change_refuses_the_cached_partitioning(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = GeometryCache(Path(directory), ["/Game/Meshes/SM_A.SM_A"])
            cache.reconcile()
            with self.assertRaisesRegex(RuntimeError, "--rebuild-geometry"):
                cache.reconcile(triangles=100_000)

    def test_dropped_mesh_directory_never_blocks_a_later_re_add(self):
        with tempfile.TemporaryDirectory() as directory:
            first = "/Game/Meshes/SM_A.SM_A"
            second = "/Game/Meshes/SM_B.SM_B"
            cache = GeometryCache(Path(directory), [first, second])
            cache.reconcile()
            dropped = cache.loose / "geometry" / asset_id_for(second)

            cache.write_scene([first])
            self.assertEqual(cache.reconcile(), [])
            self.assertFalse(dropped.exists())

            # A directory that outlived an older drop must not wedge the re-add.
            write_json(dropped / "parts" / "manifest.json", {"partitions": []})
            cache.write_scene([first, second])
            with stub_geometry_delta(cache):
                self.assertEqual(cache.reconcile(), [second])
            self.assertEqual(
                json.loads((dropped / "parts" / "manifest.json").read_text())["partitions"][0][
                    "meshlet"
                ],
                "rebuilt.glb",
            )

    def test_scene_only_refresh_updates_every_cached_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            loose = work / "loose"
            output = root / "assets"
            for tree in (loose, output):
                write_json(tree / "scenes" / "Restir_Level.json", {"level": "stale"})
            staged = root / "staging" / "Restir_Level.json"
            write_json(staged, {"level": "fresh"})

            convert.install_scene_manifests([staged], work, loose, output)

            for tree in (work, loose, output):
                scene = json.loads((tree / "scenes" / "Restir_Level.json").read_text())
                self.assertEqual(scene["level"], "fresh")
            self.assertFalse(staged.exists())
            with stub_manifest_inventory():
                self.assertTrue(
                    convert.scene_cache_is_current(
                        work / "scenes", [work / "scenes" / "Restir_Level.json"]
                    )
                )

    def test_changed_extractor_invalidates_cached_scene_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            scenes = Path(directory) / "scenes"
            manifest = scenes / "Restir_Level.json"
            write_json(manifest, {"level": "Restir_Level"})
            stamp = scenes / convert.CONVERTER_INPUTS_NAME

            with stub_manifest_inventory():
                # An unstamped cache predates the fingerprint and is unknown work.
                self.assertFalse(convert.scene_cache_is_current(scenes, [manifest]))
                convert.stamp_scene_cache(scenes)
                self.assertTrue(convert.scene_cache_is_current(scenes, [manifest]))
                write_json(
                    stamp,
                    {
                        "format": convert.CONVERTER_INPUTS_FORMAT,
                        "converter": "extractor-changed",
                    },
                )
                self.assertFalse(convert.scene_cache_is_current(scenes, [manifest]))

    def test_blueprint_archetype_lights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "Restir_Level.json"
            light = {
                "name": "PointLight",
                "intensity": convert.ARCHETYPE_LIGHT_INTENSITY,
                "attenuation_radius": convert.ARCHETYPE_LIGHT_ATTENUATION_RADIUS,
                "transform": {
                    "translation": {"x": 0, "y": 0, "z": 0},
                    "rotation": {"x": 0, "y": 0, "z": 0, "w": 1},
                    "scale": {"x": 1, "y": 1, "z": 1},
                },
            }
            write_json(scene, {"actors": [{"label": "Lamp", "lights": [light]}]})
            with self.assertRaisesRegex(RuntimeError, "un-merged blueprint archetype"):
                convert.validate_blueprint_light_archetypes([scene])

            # Any one of the three authored values means a merged template.
            for key, value in (
                ("intensity", 180.0),
                ("attenuation_radius", 800.0),
            ):
                write_json(
                    scene, {"actors": [{"lights": [{**light, key: value}]}]}
                )
                convert.validate_blueprint_light_archetypes([scene])
            placed = {
                **light,
                "transform": {
                    **light["transform"],
                    "translation": {"x": 0, "y": 0, "z": 112.5},
                },
            }
            write_json(scene, {"actors": [{"lights": [placed]}]})
            convert.validate_blueprint_light_archetypes([scene])

    def test_interrupted_install_promotes_the_packed_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "assets"
            next_output = root / ".assets.next"
            old_output = root / ".assets.old"

            write_pack_state(next_output)
            write_json(old_output / "geometry.json", {"meshes": []})
            convert.recover_interrupted_install(output, next_output, old_output)
            self.assertTrue((output / "pack.json").is_file())
            self.assertFalse(next_output.exists())
            self.assertFalse(old_output.exists())

    def test_interrupted_install_restores_the_previous_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "assets"
            next_output = root / ".assets.next"
            old_output = root / ".assets.old"

            # An incomplete staging tree has no pack.json to promote.
            write_json(next_output / "geometry.json", {"meshes": []})
            write_json(old_output / "marker.json", {"previous": True})
            convert.recover_interrupted_install(output, next_output, old_output)
            self.assertTrue((output / "marker.json").is_file())
            self.assertFalse(old_output.exists())

            # A backup left behind by an interrupted cleanup is just removed.
            write_json(old_output / "marker.json", {"previous": True})
            convert.recover_interrupted_install(output, next_output, old_output)
            self.assertTrue((output / "marker.json").is_file())
            self.assertFalse(old_output.exists())

    def test_pack_written_in_place_is_installed_without_a_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "assets"
            old_output = root / ".assets.old"

            # No previous tree existed, so the pack landed straight in output.
            write_pack_state(output)
            convert.install_packed_output(output, output, old_output)
            self.assertTrue((output / "pack.json").is_file())
            self.assertFalse(old_output.exists())

    def test_staged_pack_replaces_the_previous_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "assets"
            next_output = root / ".assets.next"
            old_output = root / ".assets.old"

            write_json(output / "marker.json", {"previous": True})
            write_pack_state(next_output)
            convert.install_packed_output(next_output, output, old_output)
            self.assertTrue((output / "pack.json").is_file())
            self.assertFalse((output / "marker.json").exists())
            self.assertFalse(next_output.exists())
            self.assertFalse(old_output.exists())

    def test_install_refuses_a_backup_that_appeared_mid_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "assets"
            next_output = root / ".assets.next"
            old_output = root / ".assets.old"

            write_json(output / "marker.json", {"previous": True})
            write_pack_state(next_output)
            write_json(old_output / "marker.json", {"unexpected": True})
            with self.assertRaisesRegex(RuntimeError, "backup appeared mid-run"):
                convert.install_packed_output(next_output, output, old_output)
            self.assertTrue((output / "marker.json").is_file())
            self.assertTrue((next_output / "pack.json").is_file())


if __name__ == "__main__":
    unittest.main()
