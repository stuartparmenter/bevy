from __future__ import annotations

import struct
import tempfile
import unittest
import zlib
from collections.abc import Collection
from pathlib import Path

from PIL import Image

import texture_source


def png_with_iend_bytes_in_idat() -> bytes:
    """Build a valid PNG whose stored IDAT data contains the bytes "IEND"."""
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 4, 1, 8, 0, 0, 0, 0)
    rows = bytes([0, 73, 69, 78, 68])
    return (
        texture_source.PNG_SIGNATURE
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows, 0))
        + chunk(b"IEND", b"")
    )


def texture_record(output: str, size: tuple[int, int]) -> dict[str, object]:
    return {
        "object": "/Game/Test.Test",
        "package": "Test.uasset",
        "output": output,
        "width": size[0],
        "height": size[1],
        "pixel_format": "TSF_BGRA8",
        "source_compression": "TSCF_PNG",
        "srgb": True,
        "normal_map": False,
        "payload_size": 64,
        "blocks": [
            {"block_x": 0, "block_y": 0, "width": size[0], "height": size[1]}
        ],
    }


def block_atlas(
    columns: int,
    rows: int,
    *,
    skip: Collection[tuple[int, int]] = (),
    srgb: bool = True,
    normal_map: bool = False,
    pixel_format: str = "TSF_BGRA8",
    tile: int = 2,
) -> tuple[dict[str, object], bytes]:
    """Build a uniform UDIM block set whose colors identify their block."""
    blocks: list[dict[str, int]] = []
    payload = bytearray()
    for block_y in range(rows):
        for block_x in range(columns):
            if (block_x, block_y) in skip:
                continue
            red, green, blue = 16 + 32 * block_x, 16 + 32 * block_y, 200
            texel = (
                bytes([blue, green, red, 255])
                if pixel_format == "TSF_BGRA8"
                else bytes([green])
            )
            blocks.append(
                {
                    "block_x": block_x,
                    "block_y": block_y,
                    "width": tile,
                    "height": tile,
                    "payload_offset": len(payload),
                    "payload_size": len(texel) * tile * tile,
                }
            )
            payload.extend(texel * (tile * tile))
    record = {
        "object": "/Game/Test.Test",
        "package": "Test.uasset",
        "output": "atlas.png",
        "width": columns * tile,
        "height": rows * tile,
        "pixel_format": pixel_format,
        "source_compression": "TSCF_None",
        "srgb": srgb,
        "is_normal_map": normal_map,
        "payload_size": len(payload),
        "blocks": blocks,
    }
    return record, bytes(payload)


class AtlasLayoutTests(unittest.TestCase):
    def test_udim_row_zero_is_the_bottom_row_of_the_atlas(self):
        record, payload = block_atlas(2, 3)
        image, *_ = texture_source.decode_texture_blocks(payload, record)
        self.assertEqual(image.size, (4, 6))
        for block in record["blocks"]:
            block_x, block_y = block["block_x"], block["block_y"]
            left = block_x * 2
            top = (3 - 1 - block_y) * 2
            expected = (16 + 32 * block_x, 16 + 32 * block_y, 200, 255)
            self.assertEqual(image.getpixel((left, top)), expected)
            self.assertEqual(image.getpixel((left + 1, top + 1)), expected)

    def test_unauthored_cells_take_neutral_fill_per_texture_class(self):
        for srgb, normal_map, expected in (
            (True, False, texture_source.ATLAS_FILL_COLOR),
            (False, False, texture_source.ATLAS_FILL_SURFACE),
            (False, True, texture_source.ATLAS_FILL_NORMAL),
        ):
            with self.subTest(srgb=srgb, normal_map=normal_map):
                record, payload = block_atlas(
                    2, 2, skip={(1, 1)}, srgb=srgb, normal_map=normal_map
                )
                image, *_ = texture_source.decode_texture_blocks(payload, record)
                # Block (1, 1) is the top-right cell under this row order.
                self.assertEqual(image.getpixel((2, 0)), expected)
                self.assertEqual(image.getpixel((3, 1)), expected)
                self.assertNotEqual(image.getpixel((2, 2)), expected)

    def test_unauthored_single_channel_cells_are_not_black(self):
        record, payload = block_atlas(2, 2, skip={(1, 1)}, pixel_format="TSF_G8")
        image, *_ = texture_source.decode_texture_blocks(payload, record)
        self.assertEqual(image.mode, "L")
        self.assertEqual(
            image.getpixel((2, 0)), texture_source.ATLAS_FILL_SINGLE_CHANNEL
        )
        self.assertNotEqual(texture_source.ATLAS_FILL_SINGLE_CHANNEL, 0)

    def test_layout_version_only_matters_to_multi_row_or_holed_atlases(self):
        self.assertFalse(texture_source.atlas_layout_matters(1, 1, 1))
        self.assertFalse(texture_source.atlas_layout_matters(5, 1, 5))
        self.assertTrue(texture_source.atlas_layout_matters(5, 1, 4))
        self.assertTrue(texture_source.atlas_layout_matters(10, 3, 30))


class TextureResumeTests(unittest.TestCase):
    def test_normal_map_correction_is_exact_not_name_based(self):
        corrected = next(iter(texture_source.NORMAL_MAP_SOURCE_CORRECTIONS))
        self.assertTrue(
            texture_source.is_normal_map(
                {"object": corrected, "is_normal_map": False}
            )
        )
        self.assertFalse(
            texture_source.is_normal_map(
                {
                    "object": "/Game/Other/T_LooksLike_Normal.T_LooksLike_Normal",
                    "is_normal_map": False,
                }
            )
        )

    def test_unchanged_export_reuses_manifest_without_decoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "texture.png"
            Image.new("RGBA", (4, 4), (1, 2, 3, 255)).save(destination)
            source = {
                "object": "/Game/Test.Test",
                "package": "Test.uasset",
                "output": "texture.png",
                "width": 4,
                "height": 4,
                "pixel_format": "TSF_BGRA8",
                "source_compression": "TSCF_PNG",
                "srgb": True,
                "normal_map": False,
                "payload_size": 64,
                "blocks": [
                    {
                        "block_x": 0,
                        "block_y": 0,
                        "width": 4,
                        "height": 4,
                    }
                ],
            }
            destination.with_name("texture.png.meta").write_text(
                texture_source.image_meta(True, False), encoding="utf-8"
            )
            exported = {
                "object": source["object"],
                "source": source["package"],
                "source_size": [4, 4],
                "output": source["output"],
                "output_size": [4, 4],
                "source_format": source["pixel_format"],
                "source_compression": source["source_compression"],
                "srgb": True,
                "normal_map": False,
                "source_block_count": 1,
                "source_grid_columns": 1,
                "source_grid_rows": 1,
                "source_payload_size": 64,
                "output_bit_depth": 8,
                "output_file_size": destination.stat().st_size,
            }
            reused = texture_source.reusable_texture(root, source, exported, 8192)
            self.assertIsNotNone(reused)
            self.assertTrue(reused["resumed"])

            exported["output_file_size"] += 1
            self.assertIsNone(
                texture_source.reusable_texture(root, source, exported, 8192)
            )

    def prepared_export(self, root: Path, size: tuple[int, int]):
        destination = root / "texture.png"
        Image.new("RGBA", size, (1, 2, 3, 255)).save(destination)
        destination.with_name("texture.png.meta").write_text(
            texture_source.image_meta(True, False), encoding="utf-8"
        )
        source = texture_record("texture.png", size)
        exported = {
            "object": source["object"],
            "source": source["package"],
            "source_size": list(size),
            "output": source["output"],
            "output_size": list(size),
            "source_format": source["pixel_format"],
            "source_compression": source["source_compression"],
            "srgb": True,
            "normal_map": False,
            "source_block_count": 1,
            "source_grid_columns": 1,
            "source_grid_rows": 1,
            "source_payload_size": 64,
            "output_bit_depth": 8,
            "output_file_size": destination.stat().st_size,
        }
        return source, exported

    def test_resume_re_exports_when_the_size_cap_changed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, exported = self.prepared_export(root, (8, 8))
            self.assertIsNotNone(
                texture_source.reusable_texture(root, source, exported, 8)
            )
            # A lowered cap must shrink the export instead of keeping it.
            self.assertIsNone(
                texture_source.reusable_texture(root, source, exported, 4)
            )
            # A raised cap must re-export at the resolution it now allows.
            exported["output_size"] = [4, 4]
            self.assertIsNone(
                texture_source.reusable_texture(root, source, exported, 8)
            )

    def multi_row_export(self, root: Path):
        source, exported = self.prepared_export(root, (8, 8))
        source["blocks"] = [
            {"block_x": block_x, "block_y": block_y, "width": 4, "height": 4}
            for block_y in range(2)
            for block_x in range(2)
        ]
        exported["source_block_count"] = 4
        exported["source_grid_columns"] = 2
        exported["source_grid_rows"] = 2
        exported["atlas_layout_version"] = texture_source.ATLAS_LAYOUT_VERSION
        return source, exported

    def test_resume_re_exports_atlases_built_by_an_older_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, exported = self.multi_row_export(root)
            self.assertIsNotNone(
                texture_source.reusable_texture(root, source, exported, 8192)
            )
            exported["atlas_layout_version"] = texture_source.ATLAS_LAYOUT_VERSION - 1
            self.assertIsNone(
                texture_source.reusable_texture(root, source, exported, 8192)
            )
            del exported["atlas_layout_version"]
            self.assertIsNone(
                texture_source.reusable_texture(root, source, exported, 8192)
            )
            # An orphaned atlas carries no record of the layout that wrote it.
            self.assertIsNone(texture_source.existing_texture(root, source, 8192))

    def test_resume_keeps_unaffected_exports_across_layout_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, exported = self.prepared_export(root, (8, 8))
            source["blocks"] = [
                {"block_x": block_x, "block_y": 0, "width": 4, "height": 8}
                for block_x in range(2)
            ]
            exported["source_block_count"] = 2
            exported["source_grid_columns"] = 2
            self.assertIsNotNone(
                texture_source.reusable_texture(root, source, exported, 8192)
            )
            self.assertIsNotNone(texture_source.existing_texture(root, source, 8192))

    def test_orphan_export_is_adopted_only_at_the_current_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.prepared_export(root, (8, 8))
            adopted = texture_source.existing_texture(root, source, 8192)
            self.assertIsNotNone(adopted)
            self.assertEqual(adopted["output_size"], [8, 8])
            self.assertIsNone(texture_source.existing_texture(root, source, 4))

    def test_expected_output_size_matches_the_export_clamp(self):
        self.assertEqual(
            texture_source.expected_output_size((8192, 4096), 4096, 8), [4096, 2048]
        )
        self.assertEqual(texture_source.expected_output_size((6, 6), 0, 8), [8, 8])
        # High-precision exports pass through uncompressed, so they are not
        # rounded up to a block boundary.
        self.assertEqual(texture_source.expected_output_size((6, 6), 0, 16), [6, 6])

    def test_bit_depth_comes_from_the_mode_not_a_substring(self):
        self.assertEqual(texture_source.image_bit_depth(Image.new("I", (4, 4))), 16)
        self.assertEqual(texture_source.image_bit_depth(Image.new("I;16", (4, 4))), 16)
        self.assertEqual(texture_source.image_bit_depth(Image.new("L", (4, 4))), 8)

    def test_embedded_png_ignores_iend_bytes_inside_idat(self):
        raw = png_with_iend_bytes_in_idat()
        self.assertLess(raw.find(b"IEND"), len(raw) - 8)
        encoded, offset, bit_depth = texture_source.embedded_png(raw, 4, 1)
        self.assertEqual(encoded, raw)
        self.assertEqual((offset, bit_depth), (0, 8))

    def test_unexplained_payload_surplus_is_an_error(self):
        raw = bytes(4 * 4 * 4)
        image, consumed, prefix, bit_depth = texture_source.decode_image(
            raw, 4, 4, "TSF_BGRA8", "TSCF_None"
        )
        self.assertEqual((image.size, consumed, prefix, bit_depth), ((4, 4), 64, 0, 8))
        with self.assertRaises(ValueError):
            texture_source.decode_image(
                raw + bytes(16), 4, 4, "TSF_BGRA8", "TSCF_None"
            )


if __name__ == "__main__":
    unittest.main()
