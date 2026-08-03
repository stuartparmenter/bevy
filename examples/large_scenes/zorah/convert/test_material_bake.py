from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import material_bake


def parameter(name, value, association="GlobalParameter", index=-1):
    return {
        "name": name,
        "association": association,
        "index": index,
        "value": value,
    }


def guid_parameter(name, value, expression_guid, association="GlobalParameter", index=-1):
    return parameter(name, value, association, index) | {
        "expression_guid": expression_guid
    }


class MaterialBakeTests(unittest.TestCase):
    def effective_with(self, *, object_name="/Game/Test.Test", scalars=(), vectors=(), textures=(), switches=()):
        material = material_bake.EffectiveMaterial(
            object=object_name,
            package="Test.uasset",
            type="MaterialInstanceConstant",
        )
        for field, values in (
            (material.scalars, scalars),
            (material.vectors, vectors),
            (material.textures, textures),
            (material.switches, switches),
        ):
            for value in values:
                field[material_bake.parameter_key(value)] = value
        return material

    def test_dormant_generic_emissive_default_is_not_exported(self):
        material = self.effective_with(
            scalars=[parameter("Emissive Intensity", 1.0)],
            switches=[parameter("Enable Emissive", False)],
        )
        runtime = material_bake.runtime_material_record(material, {})
        self.assertFalse(runtime["emissive"])
        self.assertFalse(runtime["scalars"])

    def test_custom_global_emission_family_is_exported(self):
        material = self.effective_with(
            scalars=[parameter("Global Emission Intensity", 70.0)],
            vectors=[parameter("Emission Tint", "FFFFAB (FLinearColor)")],
        )
        runtime = material_bake.runtime_material_record(material, {})
        self.assertTrue(runtime["emissive"])
        self.assertEqual(runtime["scalars"][0]["value"], 70.0)
        emissive_color = next(
            parameter
            for parameter in runtime["vectors"]
            if parameter["name"] == "Emissive Color"
        )
        self.assertEqual(emissive_color["value"], "FFFFABFF (FLinearColor)")

    def test_layer_scoped_enabled_emission_keeps_its_mask(self):
        scope = ("LayerParameter", 0)
        material = self.effective_with(
            scalars=[parameter("Emissive Intensity", 1.0, *scope)],
            vectors=[parameter("Emissive Color", "FF7B00 (FLinearColor)", *scope)],
            textures=[parameter("Extra", "/Game/FireMask.FireMask", *scope)],
            switches=[parameter("Enable Emissive", True, *scope)],
        )
        runtime = material_bake.runtime_material_record(material, {})
        self.assertTrue(runtime["emissive"])
        self.assertEqual(runtime["textures"][0]["value"], "/Game/FireMask.FireMask")

    def test_layer_function_defaults_are_scoped_before_instance_overrides(self):
        manifest = {
            "materials": [
                {
                    "package": "Layer.uasset",
                    "object": "/Game/Layer.Layer",
                    "type": "MaterialFunctionMaterialLayer",
                    "scalars": [parameter("UV Scale U", 2.0)],
                    "vectors": [],
                    "textures": [parameter("Base Color", "/Game/Default.Default")],
                    "static_switches": [],
                    "layers": [],
                    "blends": [],
                    "base_overrides": {},
                },
                {
                    "package": "Instance.uasset",
                    "object": "/Game/Instance.Instance",
                    "type": "MaterialInstanceConstant",
                    "parent": None,
                    "scalars": [parameter("UV Scale U", 3.0, "LayerParameter", 0)],
                    "vectors": [],
                    "textures": [],
                    "static_switches": [],
                    "layers": ["/Game/Layer.Layer"],
                    "blends": [],
                    "base_overrides": {},
                },
            ]
        }
        effective = material_bake.Resolver(manifest).resolve("/Game/Instance.Instance")
        self.assertEqual(effective.scalar(("UV Scale U",), "LayerParameter", 0), 3.0)
        self.assertEqual(
            material_bake.texture_reference(
                effective, material_bake.BASE_NAMES, "LayerParameter", 0
            )[0],
            "/Game/Default.Default",
        )

    def test_tiled_sampler_repeats_instead_of_clamping(self):
        image = Image.fromarray(
            np.asarray([[[255, 0, 0, 255], [0, 255, 0, 255]]], dtype=np.uint8),
            "RGBA",
        )
        pixels = material_bake.sample_tiled(image, (4, 1), (2.0, 1.0, 0.0, 0.0, 0.0))
        np.testing.assert_allclose(pixels[0, 0], pixels[0, 2])
        np.testing.assert_allclose(pixels[0, 1], pixels[0, 3])

    def test_generated_bake_preserves_two_by_one_source_tiles(self):
        image = Image.fromarray(
            np.asarray(
                [[[255, 0, 0, 255], [255, 0, 0, 255], [0, 255, 0, 255], [0, 255, 0, 255]]],
                dtype=np.uint8,
            ),
            "RGBA",
        )
        pixels = material_bake.sample_tiled(
            image,
            (4, 1),
            (1.0, 1.0, 0.0, 0.0, 0.0),
            source_grid=(2, 1),
            target_grid=(2, 1),
        )
        np.testing.assert_array_equal(
            np.rint(pixels * 255.0).astype(np.uint8), np.asarray(image)
        )

        record = material_bake.image_record(
            "/Generated/Test",
            "test.png",
            (4, 1),
            srgb=True,
            normal_map=False,
            grid=(2, 1),
        )
        self.assertEqual(record["source_block_count"], 2)
        self.assertEqual(record["source_grid_columns"], 2)
        self.assertEqual(record["source_grid_rows"], 1)

    def test_one_tile_source_repeats_across_two_tile_output(self):
        image = Image.fromarray(
            np.asarray([[[255, 0, 0, 255], [0, 255, 0, 255]]], dtype=np.uint8),
            "RGBA",
        )
        pixels = material_bake.sample_tiled(
            image,
            (4, 1),
            (1.0, 1.0, 0.0, 0.0, 0.0),
            source_grid=(1, 1),
            target_grid=(2, 1),
        )
        np.testing.assert_allclose(pixels[0, 0], pixels[0, 2])
        np.testing.assert_allclose(pixels[0, 1], pixels[0, 3])

    def test_multi_row_bake_preserves_the_exporter_row_order(self):
        # UDIM row 0 (mesh v in [0, 1]) is the bottom image row, so the source
        # here is red on UDIM row 0 and blue on UDIM row 1.
        image = Image.fromarray(
            np.asarray([[[0, 0, 255, 255]], [[255, 0, 0, 255]]], dtype=np.uint8),
            "RGBA",
        )
        pixels = material_bake.sample_tiled(
            image,
            (1, 2),
            (1.0, 1.0, 0.0, 0.0, 0.0),
            source_grid=(1, 2),
            target_grid=(1, 2),
        )
        np.testing.assert_array_equal(
            np.rint(pixels * 255.0).astype(np.uint8), np.asarray(image)
        )

    def test_taller_target_grid_wraps_on_udim_rows_not_image_rows(self):
        image = Image.fromarray(
            np.asarray([[[0, 0, 255, 255]], [[255, 0, 0, 255]]], dtype=np.uint8),
            "RGBA",
        )
        pixels = material_bake.sample_tiled(
            image,
            (1, 3),
            (1.0, 1.0, 0.0, 0.0, 0.0),
            source_grid=(1, 2),
            target_grid=(1, 3),
        )
        # Target image rows 0..2 hold UDIM rows 2, 1, 0, which wrap onto source
        # UDIM rows 0, 1, 0 - the source's bottom, top, bottom image rows.
        np.testing.assert_array_equal(
            np.rint(pixels[:, 0] * 255.0).astype(np.uint8),
            np.asarray(
                [[255, 0, 0, 255], [0, 0, 255, 255], [255, 0, 0, 255]],
                dtype=np.uint8,
            ),
        )

    def test_flattened_layer_zero_material_keeps_textures_and_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for object_name, filename, color, srgb, normal in (
                ("/Game/Base", "base.png", (255, 0, 0, 255), True, False),
                ("/Game/Normal", "normal.png", (128, 128, 255, 255), False, True),
                ("/Game/Orm", "orm.png", (255, 128, 0, 255), False, False),
            ):
                Image.new("RGBA", (4, 4), color).save(root / filename)
                records.append(
                    {
                        "object": object_name,
                        "output": filename,
                        "output_size": [4, 4],
                        "srgb": srgb,
                        "normal_map": normal,
                    }
                )
            texture_set = material_bake.TextureSet(root, {"exported": records})
            scope = ("LayerParameter", 0)
            effective = self.effective_with(
                textures=[
                    parameter("Base Color", "/Game/Base", *scope),
                    parameter("Normal", "/Game/Normal", *scope),
                    parameter("ORM", "/Game/Orm", *scope),
                ],
                scalars=[parameter("UV Scale U", 2.0, *scope)],
            )

            runtime, generated, approximations = material_bake.bake_material(
                effective, texture_set, root, 4
            )

            self.assertEqual(approximations, set())
            self.assertEqual(
                {item["name"] for item in runtime["textures"]},
                {"Base Color", "Normal", "ORM"},
            )
            self.assertEqual(len(generated), 3)
            self.assertTrue(all(record["material_bake"] for record in generated))

    def test_hue_adjustment_is_vectorized_and_wraps(self):
        red = np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        green = material_bake.adjust_hue_saturation(red, 1.0 / 3.0, 1.0)
        np.testing.assert_allclose(green, [[[0.0, 1.0, 0.0]]], atol=1.0e-5)

    def test_two_layers_generate_standard_bevy_texture_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for object_name, filename, color, srgb, normal in (
                ("/Game/Base0", "base0.png", (255, 0, 0, 255), True, False),
                ("/Game/Base1", "base1.png", (0, 255, 0, 255), True, False),
                ("/Game/Normal", "normal.png", (128, 128, 255, 255), False, True),
                ("/Game/Orm", "orm.png", (255, 128, 0, 255), False, False),
                ("/Game/Mask", "mask.png", (255, 255, 255, 255), False, False),
            ):
                Image.new("RGBA", (4, 4), color).save(root / filename)
                records.append(
                    {
                        "object": object_name,
                        "output": filename,
                        "output_size": [4, 4],
                        "srgb": srgb,
                        "normal_map": normal,
                    }
                )
            texture_set = material_bake.TextureSet(root, {"exported": records})
            effective = material_bake.EffectiveMaterial(
                object="/Game/Test.Test",
                package="Test.uasset",
                type="MaterialInstanceConstant",
                layers=["Layer", "Layer"],
                blends=["Blend"],
            )
            for index, base in enumerate(("/Game/Base0", "/Game/Base1")):
                for name, value in (
                    ("Base Color", base),
                    ("Normal", "/Game/Normal"),
                    ("ORM", "/Game/Orm"),
                ):
                    item = parameter(name, value, "LayerParameter", index)
                    effective.textures[material_bake.parameter_key(item)] = item
            mask = parameter("Texture Mask", "/Game/Mask", "BlendParameter", 0)
            effective.textures[material_bake.parameter_key(mask)] = mask

            runtime, generated, approximations = material_bake.bake_material(
                effective, texture_set, root, 4
            )
            self.assertEqual(approximations, set())
            self.assertEqual({item["name"] for item in runtime["textures"]}, {"Base Color", "Normal", "ORM"})
            self.assertEqual(len(generated), 3)
            base_record = next(record for record in generated if record["srgb"])
            baked = np.asarray(Image.open(root / base_record["output"]))
            self.assertTrue(np.all(baked[..., 1] > baked[..., 0]))

    def test_passthrough_layer_preserves_the_authored_layer_below(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGBA", (4, 4), (32, 96, 160, 255)).save(root / "base.png")
            Image.new("RGBA", (4, 4), (255, 255, 255, 255)).save(root / "mask.png")
            texture_set = material_bake.TextureSet(
                root,
                {
                    "exported": [
                        {
                            "object": "/Game/Base",
                            "output": "base.png",
                            "output_size": [4, 4],
                            "srgb": True,
                            "normal_map": False,
                        },
                        {
                            "object": "/Game/Mask",
                            "output": "mask.png",
                            "output_size": [4, 4],
                            "srgb": False,
                            "normal_map": False,
                        },
                    ]
                },
            )
            effective = material_bake.EffectiveMaterial(
                object="/Game/Test.Test",
                package="Test.uasset",
                type="MaterialInstanceConstant",
                layers=["/Game/Layer.Simple", material_bake.PASSTHROUGH_LAYER],
                blends=["/Game/Blend.Height"],
            )
            base = parameter("Base Color", "/Game/Base", "LayerParameter", 0)
            mask = parameter("Texture Mask", "/Game/Mask", "BlendParameter", 0)
            effective.textures[material_bake.parameter_key(base)] = base
            effective.textures[material_bake.parameter_key(mask)] = mask

            runtime, generated, approximations = material_bake.bake_material(
                effective, texture_set, root, 4
            )

            self.assertEqual(
                approximations,
                {
                    "layer 0 has no normal texture; baked a neutral fill",
                    "layer 0 has no surface texture; baked a neutral fill",
                },
            )
            self.assertEqual(
                {item["name"] for item in runtime["textures"]},
                {"Base Color", "Normal", "ORM"},
            )
            base_record = next(record for record in generated if record["srgb"])
            baked = np.asarray(Image.open(root / base_record["output"]))
            np.testing.assert_array_equal(baked, np.full((4, 4, 4), (32, 96, 160, 255)))

    def test_hex_colors_round_trip_through_the_fcolor_convention(self):
        for text in ("FFFFFF", "000000", "676056", "AAAAAA", "FF0025"):
            color = material_bake.parse_color(
                f"{text} (FLinearColor)", (0.0, 0.0, 0.0, 1.0)
            )
            self.assertEqual(material_bake.hex_color(color), f"{text}FF")

    def test_hex_colors_decode_as_srgb_with_linear_alpha(self):
        color = material_bake.parse_color("808080 (FLinearColor)", (1.0, 1.0, 1.0, 1.0))
        self.assertAlmostEqual(float(color[0]), 0.2158605, places=5)
        self.assertEqual(float(color[3]), 1.0)

    def test_textureless_tint_is_written_as_srgb_hex(self):
        material = self.effective_with(
            vectors=[parameter("Tint", "808080 (FLinearColor)")]
        )
        runtime = material_bake.runtime_material_record(material, {})
        tint = next(
            item for item in runtime["vectors"] if item["name"] == "Tint"
        )
        self.assertEqual(tint["value"], "808080FF (FLinearColor)")

    def test_hdr_emissive_color_folds_its_peak_into_the_intensity(self):
        material = self.effective_with(
            scalars=[parameter("Emissive Intensity", 2.0)],
            vectors=[parameter("Emissive Color", [4.0, 2.0, 0.0, 1.0])],
            switches=[parameter("Enable Emissive", True)],
        )
        approximations: set[str] = set()
        runtime = material_bake.runtime_material_record(material, {}, approximations)
        intensity = next(
            item for item in runtime["scalars"] if item["name"] == "Emissive Intensity"
        )
        color = next(
            item for item in runtime["vectors"] if item["name"] == "Emissive Color"
        )
        self.assertEqual(intensity["value"], 8.0)
        self.assertEqual(color["value"], "FFBC00FF (FLinearColor)")
        self.assertEqual(approximations, set())

    def test_negative_emissive_components_are_clamped_and_recorded(self):
        material = self.effective_with(
            scalars=[parameter("Emissive Intensity", 1.0)],
            vectors=[parameter("Emissive Color", [1.0, -0.5, 0.25, 1.0])],
            switches=[parameter("Enable Emissive", True)],
        )
        approximations: set[str] = set()
        runtime = material_bake.runtime_material_record(material, {}, approximations)
        color = next(
            item for item in runtime["vectors"] if item["name"] == "Emissive Color"
        )
        self.assertTrue(color["value"].startswith("FF00"))
        self.assertIn("emissive color clamped to the 0..1 hex range", approximations)

    def test_runtime_name_families_cover_the_runtime_lists(self):
        main_rs = Path(__file__).resolve().parent.parent / "src" / "main.rs"
        if not main_rs.is_file():
            self.skipTest("main.rs is not available next to the converter")
        source = main_rs.read_text(encoding="utf-8")

        def rust_names(constant: str) -> list[str]:
            match = re.search(
                rf"const {constant}: &\[&str\] = &\[(.*?)\];", source, re.DOTALL
            )
            self.assertIsNotNone(match, f"{constant} is missing from main.rs")
            return re.findall(r'"([^"]+)"', match.group(1))

        for constant, names in (
            ("BASE_COLOR_TEXTURE_NAMES", material_bake.BASE_NAMES),
            ("NORMAL_TEXTURE_NAMES", material_bake.NORMAL_NAMES),
            ("ORM_TEXTURE_NAMES", material_bake.SURFACE_NAMES),
            ("EMISSIVE_TEXTURE_NAMES", material_bake.EMISSIVE_TEXTURE_NAMES),
            ("EMISSIVE_INTENSITY_NAMES", material_bake.EMISSIVE_INTENSITY_NAMES),
        ):
            expected = rust_names(constant)
            selectable: list[str] = []
            for name in names:
                normalized = material_bake.normalized(name)
                if normalized not in selectable:
                    selectable.append(normalized)
            # A name the runtime binds but the bake never selects is dropped
            # from the material record entirely.
            self.assertLessEqual(set(expected), set(selectable), constant)
            # Shared names must rank identically or the two sides disagree about
            # which texture a material uses.
            self.assertEqual(
                expected,
                [name for name in selectable if name in set(expected)],
                constant,
            )

    def test_material_graph_slots_are_selectable_by_the_bake_and_runtime(self):
        program_cs = Path(__file__).resolve().parent / "Program.cs"
        main_rs = Path(__file__).resolve().parent.parent / "src" / "main.rs"
        if not main_rs.is_file():
            self.skipTest("main.rs is not available next to the converter")
        extractor = program_cs.read_text(encoding="utf-8")
        runtime = main_rs.read_text(encoding="utf-8")

        graph_slots = re.search(
            r"MaterialGraphTextureInputs =\s*\[(.*?)\];", extractor, re.DOTALL
        )
        self.assertIsNotNone(graph_slots, "MaterialGraphTextureInputs is missing")
        emitted = dict(re.findall(r'\("([^"]+)", "([^"]+)"\)', graph_slots.group(1)))
        orm = re.search(
            r'MaterialGraphOrmParameter = "([^"]+)"', extractor
        )
        self.assertIsNotNone(orm, "MaterialGraphOrmParameter is missing")
        emitted["Roughness"] = orm.group(1)

        # Every name the graph walk emits has to land in a selection family on
        # both sides, or a connection-derived texture is written into the source
        # manifest and then silently dropped.
        for input_name, families in (
            ("BaseColor", (material_bake.BASE_NAMES, "BASE_COLOR_TEXTURE_NAMES")),
            ("Normal", (material_bake.NORMAL_NAMES, "NORMAL_TEXTURE_NAMES")),
            ("EmissiveColor", (material_bake.EMISSIVE_NAMES, "EMISSIVE_TEXTURE_NAMES")),
            ("Roughness", (material_bake.SURFACE_NAMES, "ORM_TEXTURE_NAMES")),
        ):
            parameter_name = material_bake.normalized(emitted[input_name])
            bake_family, rust_constant = families
            self.assertIn(
                parameter_name,
                [material_bake.normalized(name) for name in bake_family],
                input_name,
            )
            self.assertIn(f'"{parameter_name}"', runtime, rust_constant)
        # The folded roughness lerp rides on the scalar the runtime multiplies
        # into the packed map, and only a name texture_carries_metallic accepts
        # keeps the metalness channel.
        self.assertIn('"roughness"', runtime)
        carries_metallic = re.search(
            r"fn texture_carries_metallic.*?\n    \}\n", runtime, re.DOTALL
        )
        self.assertIsNotNone(carries_metallic)
        self.assertIn(
            f'"{material_bake.normalized(emitted["Roughness"])}"',
            carries_metallic.group(0),
        )

    def test_graph_derived_mirror_slots_bake_without_regenerating_maps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGBA", (4, 4), (200, 200, 200, 255)).save(root / "albedo.png")
            Image.new("RGBA", (4, 4), (190, 70, 176, 255)).save(root / "orm.png")
            texture_set = material_bake.TextureSet(
                root,
                {
                    "exported": [
                        {
                            "object": "/Game/Mirror_Albedo",
                            "output": "albedo.png",
                            "output_size": [4, 4],
                            "srgb": True,
                            "normal_map": False,
                        },
                        {
                            "object": "/Game/Mirror_ORM",
                            "output": "orm.png",
                            "output_size": [4, 4],
                            "srgb": False,
                            "normal_map": False,
                        },
                    ]
                },
            )
            # The shape the extractor's graph walk writes for a base UMaterial
            # that samples its maps through unnamed nodes: canonical slot names
            # plus the roughness lerp folded into the scalar.
            material = self.effective_with(
                scalars=[parameter("Roughness", 0.1)],
                textures=[
                    parameter("Base Color", "/Game/Mirror_Albedo"),
                    parameter("ORM", "/Game/Mirror_ORM"),
                ],
            )

            runtime, generated, _ = material_bake.bake_material(
                material, texture_set, root, 4
            )

            self.assertEqual(generated, [])
            self.assertEqual(
                [(item["name"], item["value"]) for item in runtime["textures"]],
                [("Base Color", "/Game/Mirror_Albedo"), ("ORM", "/Game/Mirror_ORM")],
            )
            self.assertEqual(
                [(item["name"], item["value"]) for item in runtime["scalars"]],
                [("Roughness", 0.1)],
            )
            self.assertEqual(runtime["vectors"], [])

    def test_pass_through_textures_use_the_runtime_parameter_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGBA", (4, 4), (255, 128, 0, 255)).save(root / "rot.png")
            texture_set = material_bake.TextureSet(
                root,
                {
                    "exported": [
                        {
                            "object": "/Game/Rot",
                            "output": "rot.png",
                            "output_size": [4, 4],
                            "srgb": False,
                            "normal_map": False,
                        }
                    ]
                },
            )
            material = self.effective_with(textures=[parameter("ROT", "/Game/Rot")])

            runtime, generated, _ = material_bake.bake_material(
                material, texture_set, root, 4
            )

            self.assertEqual(generated, [])
            self.assertEqual(
                [(item["name"], item["value"]) for item in runtime["textures"]],
                [("ORM", "/Game/Rot")],
            )

    def test_single_layer_water_exports_its_own_parameter_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGBA", (4, 4), (128, 128, 255, 255)).save(root / "waves.png")
            texture_set = material_bake.TextureSet(
                root,
                {
                    "exported": [
                        {
                            "object": "/Game/Waves",
                            "output": "waves.png",
                            "output_size": [4, 4],
                            "srgb": False,
                            "normal_map": True,
                        }
                    ]
                },
            )
            # MI_Water_EOS_ReflectionPond's authored values. None of the names
            # the generic path selects on appear anywhere in this family.
            material = self.effective_with(
                scalars=[
                    parameter("Water Roughness", 0.02),
                    parameter("Water Specular", 1.0),
                    parameter("Wave A Speed", 0.0082),
                    parameter("Wave A UV X", 3.700028),
                    parameter("Wave A UV Y", 5.5),
                ],
                vectors=[parameter("Water Base Color", "000000 (FLinearColor)")],
                textures=[
                    parameter("Wave A Normal", "/Game/Waves"),
                    parameter("Wave B Normal", "/Game/Waves"),
                ],
            )
            material.base_overrides = {"ShadingModel": "MSM_SingleLayerWater"}

            runtime, generated, approximations = material_bake.bake_material(
                material, texture_set, root, 4
            )

            self.assertEqual(generated, [])
            self.assertEqual(
                [(item["name"], item["value"]) for item in runtime["textures"]],
                [("Normal", "/Game/Waves")],
            )
            self.assertEqual(
                [(item["name"], item["value"]) for item in runtime["scalars"]],
                [
                    ("Water Roughness", 0.02),
                    ("Water Specular", 1.0),
                    ("Wave A Speed", 0.0082),
                    ("Wave A UV X", 3.700028),
                    ("Wave A UV Y", 5.5),
                ],
            )
            self.assertEqual(
                [(item["name"], item["value"]) for item in runtime["vectors"]],
                [("Water Base Color", "000000FF (FLinearColor)")],
            )
            self.assertIn(
                "single-layer water renders without its absorbing volume",
                approximations,
            )

    def test_sixteen_bit_sources_are_rescaled_not_clipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mask.png"
            Image.fromarray(
                np.asarray([[0, 32768, 65535, 65535]], dtype=np.uint16)
            ).save(path)
            pixels = np.asarray(material_bake.open_rgba(path))
            np.testing.assert_array_equal(pixels[0, :, 0], [0, 128, 255, 255])

    def test_base_color_filtering_happens_in_linear_light(self):
        image = Image.fromarray(
            np.asarray([[[0, 0, 0, 255], [255, 255, 255, 255]]], dtype=np.uint8),
            "RGBA",
        )
        controls = (1.0, 1.0, 0.0, 0.0, 0.0)
        linear = material_bake.sample_tiled(image, (4, 1), controls, is_srgb=True)
        gamma = material_bake.sample_tiled(image, (4, 1), controls)
        # A quarter of the way from black to white is 0.25 of the encoded value
        # in gamma space but 0.537 once the blend happens in linear light.
        self.assertAlmostEqual(float(gamma[0, 1, 0]), 0.25, places=5)
        self.assertAlmostEqual(float(linear[0, 1, 0]), 0.5372, places=3)

    def test_heavy_minification_is_recorded_as_an_approximation(self):
        image = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
        approximations: set[str] = set()
        material_bake.sample_tiled(
            image,
            (4, 4),
            (1.0, 1.0, 0.0, 0.0, 0.0),
            approximations=approximations,
        )
        self.assertEqual(
            approximations, {"bilinear sampling minifies without a prefilter"}
        )

    def foliage_manifest(self, *, switch_name, switch_value):
        """A master plus one instance that renamed both of its overrides."""
        return {
            "materials": [
                {
                    "package": "Master.uasset",
                    "object": "/Game/Master.Master",
                    "type": "Material",
                    "scalars": [],
                    "vectors": [],
                    "textures": [
                        guid_parameter("Base Color Texture", "/Game/Default", "BCEB"),
                        guid_parameter("Opacity Mask Texture", "/Game/White", "FA32"),
                    ],
                    "static_switches": [
                        guid_parameter(
                            material_bake.OPACITY_MASK_SWITCH, False, "69A5"
                        )
                    ],
                    "layers": [],
                    "blends": [],
                    "base_overrides": {},
                },
                {
                    "package": "Instance.uasset",
                    "object": "/Game/Instance.Instance",
                    "type": "MaterialInstanceConstant",
                    "parent": "/Game/Master.Master",
                    "scalars": [],
                    "vectors": [],
                    "textures": [
                        guid_parameter("Diffuse Texture", "/Game/Base", "BCEB"),
                        guid_parameter("Opacity Mask Texture", "/Game/Mask", "FA32"),
                    ],
                    "static_switches": [
                        guid_parameter(switch_name, switch_value, "69A5")
                    ],
                    "layers": [],
                    "blends": [],
                    "base_overrides": {},
                },
            ]
        }

    def test_renamed_master_parameters_reconcile_through_their_guid(self):
        manifest = self.foliage_manifest(switch_name="Use_OpacityMask", switch_value=True)
        effective = material_bake.Resolver(manifest).resolve("/Game/Instance.Instance")
        reference, name = material_bake.texture_reference(
            effective, material_bake.BASE_NAMES, "GlobalParameter", -1
        )
        # The stale "Diffuse Texture" would otherwise leave the master's default
        # in the higher-priority "Base Color Texture" slot.
        self.assertEqual(reference, "/Game/Base")
        self.assertEqual(name, "Base Color Texture")
        self.assertTrue(effective.switch((material_bake.OPACITY_MASK_SWITCH,)))

    def opacity_texture_set(self, root):
        Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(root / "base.png")
        Image.fromarray(
            np.tile(np.asarray([0, 0, 255, 255], dtype=np.uint8), (4, 1)), "L"
        ).save(root / "mask.png")
        return material_bake.TextureSet(
            root,
            {
                "exported": [
                    {
                        "object": name,
                        "output": f"{filename}.png",
                        "output_size": [4, 4],
                        "srgb": srgb,
                        "normal_map": False,
                    }
                    for name, filename, srgb in (
                        ("/Game/Base", "base", True),
                        ("/Game/Mask", "mask", False),
                    )
                ]
            },
        )

    def test_separate_opacity_mask_composites_into_base_color_alpha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            texture_set = self.opacity_texture_set(root)
            manifest = self.foliage_manifest(
                switch_name="Use_OpacityMask", switch_value=True
            )
            effective = material_bake.Resolver(manifest).resolve("/Game/Instance.Instance")

            runtime, generated, _ = material_bake.bake_material(
                effective, texture_set, root, 4
            )

            self.assertEqual(
                [item["value"] for item in runtime["textures"]],
                [generated[0]["object"]],
            )
            baked = np.asarray(Image.open(root / generated[0]["output"]))
            np.testing.assert_array_equal(baked[0, :, 3], [0, 0, 255, 255])
            np.testing.assert_array_equal(baked[..., 0], 255)

    def test_base_color_alpha_side_of_the_switch_keeps_the_source_texture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            texture_set = self.opacity_texture_set(root)
            manifest = self.foliage_manifest(
                switch_name="Use_OpacityMask", switch_value=False
            )
            effective = material_bake.Resolver(manifest).resolve("/Game/Instance.Instance")

            runtime, generated, _ = material_bake.bake_material(
                effective, texture_set, root, 4
            )

            # Nothing to composite, so the albedo passes through unbaked and
            # keeps whatever alpha it already carries.
            self.assertEqual(generated, [])
            self.assertEqual(
                [item["value"] for item in runtime["textures"]], ["/Game/Base"]
            )


if __name__ == "__main__":
    unittest.main()
