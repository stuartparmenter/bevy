#!/usr/bin/env python3
"""Export Zorah's uncooked UE 5.4 editor texture payloads without Unreal.

CUE4Parse supplies the texture metadata, but its regular texture decoder only
handles cooked PlatformData.  Zorah stores source pixels as an
FEditorBulkData/FCompressedBuffer in the package trailer.  This script reuses
the bounded Oodle block reader from mesh_description.py and handles the exact
source formats found by the Zorah converter.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

from mesh_description import OodleBlockReader


KNOWN_CORRUPT_OODLE_BLOCKS = {
    "/Game/Assets/Environment/ThroneRoom_Cornice_C/Textures/"
    "T_ThroneRoom_Cornice_C1_Normal.T_ThroneRoom_Cornice_C1_Normal": {82},
}

BLOCK_COMPRESSION_ALIGNMENT = 4
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Assembled-atlas contract. UDIM block (bx, by) is pasted at
# (bx * tile_width, (rows - 1 - by) * tile_height), so block row 0 - the
# UDIM 1001 row, mesh v in [0, 1] - is the BOTTOM row of the image and the
# runtime addresses the whole atlas with the single affine
# u' = u / columns, v' = (v + rows - 1) / rows.
# Cells no block covers hold neutral content, never black.
# Bump this whenever either rule changes; resume re-exports the textures the
# change can reach.
ATLAS_LAYOUT_VERSION = 2
ATLAS_FILL_COLOR = (128, 128, 128, 255)
ATLAS_FILL_NORMAL = (128, 128, 255, 255)
# Occlusion 1, roughness 0.5, metallic 0. A zeroed ORM cell instead reads as an
# unoccluded-but-unlit mirror, and whole-atlas mip generation averages it into
# every neighboring tile.
ATLAS_FILL_SURFACE = (255, 128, 0, 255)
# No single-channel source has an unauthored cell today. 255 keeps a future one
# loudly visible - a blown-out mask rather than the silent black this fill
# exists to eliminate - but every single-channel role here (emissive, height)
# reads zero as its inert value, so a real case wants the role, not a constant.
ATLAS_FILL_SINGLE_CHANNEL = 255
# Pillow reopens 16-bit grayscale PNGs as one of these, and "I" hides its depth
# from a substring test on the mode name.
HIGH_PRECISION_MODES = {"I", "I;16", "I;16L", "I;16B", "I;16N"}


def block_aligned_size(size: tuple[int, int]) -> tuple[int, int]:
    """Round an image extent up to a complete BC/ETC 4x4 block."""
    return tuple(
        max(BLOCK_COMPRESSION_ALIGNMENT, (dimension + 3) // 4 * 4)
        for dimension in size
    )


def image_bit_depth(image) -> int:
    """Report the depth PNG will store this image mode at."""
    return 16 if image.mode in HIGH_PRECISION_MODES else 8


def expected_output_size(
    source_size: tuple[int, int], max_size: int, output_bit_depth: int
) -> list[int]:
    """Reproduce export_texture's clamping so resume can validate old exports."""
    width, height = source_size
    if max_size > 0 and max(width, height) > max_size:
        scale = max_size / max(width, height)
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
    if output_bit_depth <= 8:
        width, height = block_aligned_size((width, height))
    return [width, height]


def align_for_block_compression(image, output_bit_depth: int):
    """Resample 8-bit compressor inputs to a GPU-valid block extent."""
    if output_bit_depth > 8:
        # The saver passes Zorah's high-precision formats through uncompressed.
        return image
    aligned = block_aligned_size(image.size)
    if aligned == image.size:
        return image
    return image.resize(aligned, Image.Resampling.LANCZOS)


def image_meta(is_srgb: bool, is_normal_map: bool) -> str:
    alpha_mode = "Opaque" if is_normal_map else "Straight"
    return f'''(
    meta_format_version: "1.0",
    asset: Process(
        processor: "LoadTransformAndSave<ImageLoader, IdentityAssetTransformer<Image>, CompressedImageSaver>",
        settings: (
            loader_settings: (
                format: FromExtension,
                texture_format: None,
                is_srgb: {str(is_srgb).lower()},
                sampler: Descriptor (ImageSamplerDescriptor(
                    address_mode_u: Repeat,
                    address_mode_v: Repeat,
                    address_mode_w: Repeat,
                    mag_filter: Linear,
                    min_filter: Linear,
                    mipmap_filter: Linear,
                    lod_min_clamp: 0,
                    lod_max_clamp: 32.0,
                    compare: None,
                    anisotropy_clamp: 16,
                    border_color: None,
                    label: None,
                )),
                asset_usage: ("MAIN_WORLD | RENDER_WORLD"),
                array_layout: None,
                source_primaries: None,
            ),
            transformer_settings: (),
            saver_settings: (
                is_normal_map: {str(is_normal_map).lower()},
                input_alpha_mode: {alpha_mode},
                output_alpha_mode: {alpha_mode},
                generate_mipmaps: true,
                quality: VeryFast,
            ),
        ),
    ),
)\n'''


NORMAL_MAP_SOURCE_CORRECTIONS = {
    # This package is the sole Zorah 1.1.0 texture whose serialized UE
    # IsNormalMap classification is false despite containing tangent-space
    # normal data. Keep the correction exact; never infer texture semantics
    # from a filename.
    "/Game/Assets/Environment/FruitBowl_A/Textures/"
    "T_Grapes_A02_Branch_Normal.T_Grapes_A02_Branch_Normal",
}


def is_normal_map(record: dict[str, object]) -> bool:
    return bool(record.get("is_normal_map", False)) or str(
        record.get("object", "")
    ) in NORMAL_MAP_SOURCE_CORRECTIONS


def is_srgb(record: dict[str, object]) -> bool:
    # Normal vectors are data, never colors. Force a linear texture even when
    # UE/CUE4Parse's source flag incorrectly marks the package as sRGB.
    return bool(record.get("srgb")) and not is_normal_map(record)


def source_grid(record: dict[str, object]) -> tuple[int, int, int]:
    blocks = list(record.get("blocks") or [])
    if not blocks:
        return 1, 1, 1
    return (
        max(int(block["block_x"]) for block in blocks) + 1,
        max(int(block["block_y"]) for block in blocks) + 1,
        len(blocks),
    )


def atlas_layout_matters(
    grid_columns: int, grid_rows: int, block_count: int
) -> bool:
    """Report whether ATLAS_LAYOUT_VERSION can change this texture's pixels.

    A single-row atlas with every cell populated assembles identically under
    every layout version, so resume must keep those exports instead of redoing
    the several hundred textures the layout rules cannot reach.
    """
    return grid_rows > 1 or block_count < grid_columns * grid_rows


def atlas_fill(record: dict[str, object], mode: str):
    """Pick the neutral value for atlas cells no UDIM block covers."""
    if mode not in {"RGBA", "RGB"}:
        return ATLAS_FILL_SINGLE_CHANNEL
    if is_normal_map(record):
        fill = ATLAS_FILL_NORMAL
    else:
        # sRGB is the only per-texture signal separating color from packed
        # data; every linear non-normal source Zorah exports is an ORM map.
        fill = ATLAS_FILL_COLOR if is_srgb(record) else ATLAS_FILL_SURFACE
    return fill if mode == "RGBA" else fill[:3]


def png_chunks(raw: bytes, offset: int):
    """Walk the chunk structure of the PNG whose signature starts at offset."""
    position = offset + len(PNG_SIGNATURE)
    while position + 12 <= len(raw):
        length = int.from_bytes(raw[position : position + 4], "big")
        kind = raw[position + 4 : position + 8]
        end = position + 12 + length
        if end > len(raw):
            raise ValueError(f"embedded PNG {kind!r} chunk runs past the payload")
        yield kind, position, end
        if kind == b"IEND":
            return
        position = end
    raise ValueError("TSCF_PNG payload has no IEND chunk")


def embedded_png(raw: bytes, width: int, height: int):
    if Image is None:
        raise RuntimeError(
            "Pillow is required; run with `sfw uv run --with pyooz --with pillow`"
        )
    offset = raw.find(PNG_SIGNATURE)
    if offset < 0:
        raise ValueError("TSCF_PNG payload has no PNG signature")
    # The literal bytes "IEND" can occur inside compressed IDAT data, so the end
    # of the stream has to come from the chunk lengths.
    bit_depth = 8
    encoded_size = 0
    for index, (kind, start, end) in enumerate(png_chunks(raw, offset)):
        if index == 0:
            if kind != b"IHDR":
                raise ValueError("TSCF_PNG payload does not begin with IHDR")
            bit_depth = raw[start + 16]
        encoded_size = end - offset
    encoded = raw[offset : offset + encoded_size]
    with Image.open(io.BytesIO(encoded)) as source:
        if source.size != (width, height):
            raise ValueError(f"embedded PNG is {source.size}; expected {width}x{height}")
    return encoded, offset, bit_depth


def decode_image(
    raw: bytes,
    width: int,
    height: int,
    source_format: str,
    source_compression: str,
):
    if Image is None:
        raise RuntimeError(
            "Pillow is required; run with `sfw uv run --with pyooz --with pillow`"
    )
    if source_compression == "TSCF_PNG":
        encoded, offset, bit_depth = embedded_png(raw, width, height)
        with Image.open(io.BytesIO(encoded)) as source:
            source.load()
            image = source.copy()
        return image, len(encoded), offset, bit_depth
    if source_compression not in {"TSCF_None", "None", ""}:
        raise ValueError(f"unsupported Zorah source compression {source_compression}")

    formats = {
        "TSF_BGRA8": (4, "RGBA", "BGRA"),
        "TSF_RGBA8": (4, "RGBA", "RGBA"),
        "TSF_G8": (1, "L", "L"),
        "TSF_G16": (2, "I;16", "I;16L"),
    }
    if source_format == "TSF_RGBA16":
        expected = width * height * 8
        if len(raw) < expected:
            raise ValueError(
                f"{source_format} payload has {len(raw)} bytes; expected at least {expected}"
            )
        # StandardMaterial images are presently normalized 8-bit textures.
        # Select the high byte of each little-endian source component rather
        # than interpreting the 16-bit plane as half floats.
        prefix = len(raw) - expected
        rgba8 = bytes(memoryview(raw)[prefix:][1::2])
        return Image.frombytes("RGBA", (width, height), rgba8), expected, prefix, 8
    if source_format == "TSF_RGBA16F":
        expected = width * height * 8
        if len(raw) < expected:
            raise ValueError(
                f"{source_format} payload has {len(raw)} bytes; expected at least {expected}"
            )
        prefix = len(raw) - expected
        # Zorah only uses this format for its two 2048x1 curve atlases. Convert
        # the editor half floats to normalized RGBA8 before BC7 processing.
        rgba8 = bytearray(width * height * 4)
        for pixel, components in enumerate(
            struct.iter_unpack("<4e", memoryview(raw)[prefix:])
        ):
            start = pixel * 4
            rgba8[start : start + 4] = bytes(
                max(0, min(255, round(float(component) * 255.0)))
                for component in components
            )
        return Image.frombytes("RGBA", (width, height), bytes(rgba8)), expected, prefix, 8
    try:
        bytes_per_pixel, mode, raw_mode = formats[source_format]
    except KeyError as error:
        raise ValueError(f"unsupported Zorah texture source format {source_format}") from error
    expected = width * height * bytes_per_pixel
    # Only TSF_RGBA16 was ever observed with a payload prefix, and it is handled
    # above. Any other surplus here is unexplained, and guessing which end it
    # sits at silently shifts every pixel.
    if len(raw) != expected:
        raise ValueError(
            f"{source_format} payload has {len(raw)} bytes; expected exactly "
            f"{expected} for {width}x{height}"
        )
    return (
        Image.frombytes(mode, (width, height), raw, "raw", raw_mode),
        expected,
        0,
        8 if source_format not in {"TSF_G16"} else 16,
    )


def decode_texture_blocks(raw: bytes, record: dict[str, object]):
    blocks = list(record.get("blocks") or [])
    if not blocks:
        blocks = [{
            "block_x": 0,
            "block_y": 0,
            "width": int(record["width"]),
            "height": int(record["height"]),
            "payload_offset": 0,
            "payload_size": len(raw),
        }]
    decoded = []
    consumed_size = 0
    prefix_size = 0
    tail_size = 0
    output_bit_depth = 64
    for block in blocks:
        start = int(block["payload_offset"])
        end = start + int(block["payload_size"])
        if start < 0 or end > len(raw) or end <= start:
            raise ValueError(f"invalid texture block byte range {start}..{end}")
        image, consumed, prefix, bit_depth = decode_image(
            raw[start:end],
            int(block["width"]),
            int(block["height"]),
            str(record["pixel_format"]),
            str(record.get("source_compression") or "TSCF_None"),
        )
        decoded.append((block, image))
        consumed_size += consumed
        prefix_size += prefix
        tail_size += int(block["payload_size"]) - consumed - prefix
        output_bit_depth = min(output_bit_depth, bit_depth)
    if len(decoded) == 1:
        image = decoded[0][1]
        if image.size != (int(record["width"]), int(record["height"])):
            raise ValueError(
                f"single source block is {image.size}; assembled texture is "
                f"{record['width']}x{record['height']}"
            )
        return image, consumed_size, prefix_size, tail_size, output_bit_depth

    first_block = decoded[0][0]
    tile_width = int(first_block["width"])
    tile_height = int(first_block["height"])
    first_image = decoded[0][1]
    _, grid_rows, _ = source_grid(record)
    assembled = Image.new(
        first_image.mode,
        (int(record["width"]), int(record["height"])),
        atlas_fill(record, first_image.mode),
    )
    for block, image in decoded:
        # UDIM block row 0 is the bottom row of the image; see
        # ATLAS_LAYOUT_VERSION.
        position = (
            int(block["block_x"]) * tile_width,
            (grid_rows - 1 - int(block["block_y"])) * tile_height,
        )
        if (
            min(position) < 0
            or position[0] + image.width > assembled.width
            or position[1] + image.height > assembled.height
        ):
            raise ValueError(
                f"texture block at {position} with size {image.size} exceeds {assembled.size}"
            )
        assembled.paste(image, position)
    return assembled, consumed_size, prefix_size, tail_size, output_bit_depth


def export_texture(
    project_root: Path,
    output_root: Path,
    record: dict[str, object],
    max_size: int,
    replace_existing: bool = False,
) -> dict[str, object]:
    source = project_root / "Content" / str(record["package"])
    destination = output_root / str(record["output"])
    if destination.exists() and not replace_existing:
        raise FileExistsError(f"refusing to overwrite {destination}")
    with OodleBlockReader(
        source,
        recover_bad_blocks=KNOWN_CORRUPT_OODLE_BLOCKS.get(str(record["object"])),
    ) as reader:
        expected_payload = int(record["payload_size"])
        if reader.info.raw_size != expected_payload:
            raise ValueError(
                f"trailer raw size {reader.info.raw_size} does not match "
                f"CUE4Parse payload size {expected_payload}"
            )
        raw = reader.read_exact(reader.info.raw_size)
        if reader.raw_position != reader.info.raw_size:
            raise ValueError("texture payload was not consumed exactly")
        recovered_oodle_blocks = list(reader.recovered_blocks)

    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    source_compression = str(record.get("source_compression") or "TSCF_None")
    width = int(record["width"])
    height = int(record["height"])
    source_size = (width, height)
    try:
        blocks = list(record.get("blocks") or [])
        if (
            source_compression == "TSCF_PNG"
            and max_size == 0
            and len(blocks) <= 1
            and width % BLOCK_COMPRESSION_ALIGNMENT == 0
            and height % BLOCK_COMPRESSION_ALIGNMENT == 0
        ):
            encoded, prefix_size, output_bit_depth = embedded_png(raw, width, height)
            consumed_size = len(encoded)
            tail_size = len(raw) - consumed_size - prefix_size
            temporary.write_bytes(encoded)
            output_size = source_size
        else:
            (
                image,
                consumed_size,
                prefix_size,
                tail_size,
                output_bit_depth,
            ) = decode_texture_blocks(
                raw, record
            )
            if max_size > 0 and max(image.size) > max_size:
                scale = max_size / max(image.size)
                image = image.resize(
                    (
                        max(1, round(image.width * scale)),
                        max(1, round(image.height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
                # Resampling keeps the source mode, so a 16-bit texture is still
                # written - and has to be recorded - at 16 bits.
                output_bit_depth = image_bit_depth(image)
            image = align_for_block_compression(image, output_bit_depth)
            image.save(temporary, format="PNG", compress_level=1)
            output_size = image.size
        temporary.replace(destination)
        destination.with_name(f"{destination.name}.meta").write_text(
            image_meta(is_srgb(record), is_normal_map(record)), encoding="utf-8"
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    grid_columns, grid_rows, block_count = source_grid(record)
    return {
        "object": record["object"],
        "source": str(source),
        "source_size": list(source_size),
        "output": str(record["output"]),
        "output_size": list(output_size),
        "source_format": record["pixel_format"],
        "source_compression": record.get("source_compression"),
        "srgb": is_srgb(record),
        "normal_map": is_normal_map(record),
        "source_payload_size": len(raw),
        "source_consumed_bytes": consumed_size,
        "source_payload_prefix_bytes": prefix_size,
        "source_payload_tail_bytes": tail_size,
        "recovered_oodle_blocks": recovered_oodle_blocks,
        "source_block_count": block_count,
        "source_grid_columns": grid_columns,
        "source_grid_rows": grid_rows,
        "atlas_layout_version": ATLAS_LAYOUT_VERSION,
        "output_bit_depth": output_bit_depth,
        "output_file_size": destination.stat().st_size,
    }


def existing_texture(
    output_root: Path,
    record: dict[str, object],
    max_size: int,
) -> dict[str, object] | None:
    """Adopt an image/meta pair an interrupted batch left with no manifest record."""
    if Image is None:
        raise RuntimeError("Pillow is required to validate resumed texture exports")
    destination = output_root / str(record["output"])
    meta = destination.with_name(f"{destination.name}.meta")
    if not destination.is_file() or not meta.is_file():
        return None
    grid_columns, grid_rows, block_count = source_grid(record)
    if atlas_layout_matters(grid_columns, grid_rows, block_count):
        # An orphaned image carries no record of the layout that assembled it,
        # and the pixels alone cannot prove which row order was used.
        return None
    with Image.open(destination) as source:
        source.load()
        output_bit_depth = image_bit_depth(source)
        original_size = source.size
        image = align_for_block_compression(source, output_bit_depth)
        output_size = list(image.size)
        if output_size != expected_output_size(
            (int(record["width"]), int(record["height"])), max_size, output_bit_depth
        ):
            return None
        expected_meta = image_meta(is_srgb(record), is_normal_map(record))
        if meta.read_text(encoding="utf-8") != expected_meta:
            meta.write_text(expected_meta, encoding="utf-8")
        if image.size != original_size:
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", dir=destination.parent
            )
            os.close(file_descriptor)
            temporary = Path(temporary_name)
            try:
                image.save(temporary, format="PNG", compress_level=1)
                temporary.replace(destination)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
    return {
        "object": record["object"],
        "source": str(record["package"]),
        "source_size": [int(record["width"]), int(record["height"])],
        "output": str(record["output"]),
        "output_size": output_size,
        "source_format": record["pixel_format"],
        "source_compression": record.get("source_compression"),
        "srgb": is_srgb(record),
        "normal_map": is_normal_map(record),
        "source_block_count": block_count,
        "source_grid_columns": grid_columns,
        "source_grid_rows": grid_rows,
        "atlas_layout_version": ATLAS_LAYOUT_VERSION,
        "source_payload_size": int(record["payload_size"]),
        "source_consumed_bytes": None,
        "source_payload_prefix_bytes": None,
        "source_payload_tail_bytes": None,
        "recovered_oodle_blocks": sorted(
            KNOWN_CORRUPT_OODLE_BLOCKS.get(str(record["object"]), set())
        ),
        "output_bit_depth": output_bit_depth,
        "output_file_size": destination.stat().st_size,
        "resumed": True,
    }


def reusable_texture(
    output_root: Path,
    source: dict[str, object],
    exported: dict[str, object] | None,
    max_size: int,
) -> dict[str, object] | None:
    """Cheaply validate an unchanged prior export without decoding its PNG."""
    if exported is None:
        return None
    destination = output_root / str(source["output"])
    meta = destination.with_name(f"{destination.name}.meta")
    if not destination.is_file() or not meta.is_file():
        return None
    if meta.read_text(encoding="utf-8") != image_meta(is_srgb(source), is_normal_map(source)):
        return None
    grid_columns, grid_rows, block_count = source_grid(source)
    if atlas_layout_matters(grid_columns, grid_rows, block_count) and (
        exported.get("atlas_layout_version") != ATLAS_LAYOUT_VERSION
    ):
        return None
    expected = {
        "object": source["object"],
        "source": str(source["package"]),
        "source_size": [int(source["width"]), int(source["height"])],
        "output": str(source["output"]),
        "source_format": source["pixel_format"],
        "source_compression": source.get("source_compression"),
        "srgb": is_srgb(source),
        "normal_map": is_normal_map(source),
        "source_block_count": block_count,
        "source_grid_columns": grid_columns,
        "source_grid_rows": grid_rows,
        "source_payload_size": int(source["payload_size"]),
    }
    if any(exported.get(key) != value for key, value in expected.items()):
        return None
    try:
        output_size = [int(value) for value in exported["output_size"]]
        output_bit_depth = int(exported.get("output_bit_depth", 8))
        output_file_size = int(exported["output_file_size"])
    except (KeyError, TypeError, ValueError):
        return None
    # Comparing against the size the CURRENT --max-size would produce, rather
    # than only rejecting oversized output, makes both a raised and a lowered
    # cap re-export exactly the textures it affects.
    if (
        len(output_size) != 2
        or output_size
        != expected_output_size(
            (int(source["width"]), int(source["height"])), max_size, output_bit_depth
        )
        or destination.stat().st_size != output_file_size
    ):
        return None
    result = dict(exported)
    result["resumed"] = True
    return result


def convert_texture(job: dict[str, object]):
    """Export one texture in a worker process; every value here is picklable."""
    index = int(job["index"])
    record = job["record"]
    project_root = job["project_root"]
    output_root = job["output_root"]
    max_size = int(job["max_size"])
    resume = bool(job["resume"])
    try:
        result = None
        if resume:
            destination = output_root / str(record["output"])
            rebuild_atlas = (
                bool(job["replace_multiblock"])
                and destination.exists()
                and len(record.get("blocks") or []) > 1
            )
            if not rebuild_atlas:
                result = reusable_texture(
                    output_root, record, job["previous"], max_size
                )
                if result is None and job["previous"] is None:
                    # A batch interrupted before its manifest was written leaves
                    # valid pairs with no record. A record that exists but no
                    # longer matches means the export itself is stale, so it has
                    # to be redone rather than relabelled.
                    result = existing_texture(output_root, record, max_size)
        if result is None:
            result = export_texture(
                project_root,
                output_root,
                record,
                max_size,
                replace_existing=resume,
            )
        return index, result, None
    except (EOFError, OSError, RuntimeError, ValueError) as error:
        return index, None, {
            "object": str(record.get("object")),
            "error_type": type(error).__name__,
            "message": str(error),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--max-size",
        type=int,
        default=0,
        help="optional longest output edge; zero preserves full source resolution",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="validation-only maximum texture count; zero exports every record",
    )
    parser.add_argument(
        "--object",
        dest="object_filter",
        help="validation-only substring filter for a texture object path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="validate and retain completed image/meta pairs from an interrupted batch",
    )
    parser.add_argument(
        "--replace-multiblock",
        action="store_true",
        help="with --resume, rebuild existing multi-block atlases instead of retaining them",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel texture workers; two is safe for Zorah's largest payloads on 32+ GiB hosts",
    )
    parser.add_argument("--verbose", action="store_true", help="log every texture record")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    if args.max_size < 0:
        parser.error("--max-size cannot be negative")
    if args.max_size and args.max_size % BLOCK_COMPRESSION_ALIGNMENT:
        parser.error("--max-size must be a multiple of four for block compression")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.jobs < 1:
        parser.error("--jobs must be at least one")
    if args.replace_multiblock and not args.resume:
        parser.error("--replace-multiblock requires --resume")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        record
        for record in manifest["textures"]
        if bool(record.get("editor_source"))
        and (
            args.object_filter is None
            or args.object_filter in str(record.get("object", ""))
        )
    ][: args.limit or None]
    result_path = output_root / "textures.exported.json"
    previous_by_object: dict[str, dict[str, object]] = {}
    if args.resume and result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        previous_by_object = {
            str(record["object"]): record for record in previous.get("exported", [])
        }
    exported: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    jobs = [
        {
            "index": index,
            "record": record,
            "project_root": project_root,
            "output_root": output_root,
            "max_size": args.max_size,
            "resume": args.resume,
            "replace_multiblock": args.replace_multiblock,
            "previous": previous_by_object.get(str(record["object"])),
        }
        for index, record in enumerate(records, start=1)
    ]

    # Processes rather than threads: the Oodle block reader assembles payloads
    # in a Python-level copy loop and the TSF unpacking is pure Python, so
    # workers spend most of their time holding the GIL.
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for index, result, failure in executor.map(convert_texture, jobs):
            if failure is None:
                assert result is not None
                exported.append(result)
                if args.verbose:
                    print(
                        f"ZORAH_TEXTURE_SOURCE_DONE {index}/{len(records)} "
                        f"size={result['output_size']} output={result['output']}",
                        flush=True,
                    )
                continue
            failures.append(failure)
            print(
                f"ZORAH_TEXTURE_SOURCE_ERROR object={failure['object']} "
                f"error={failure['message']}",
                file=sys.stderr,
                flush=True,
            )

    if result_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {result_path}")
    result = {
        "format": "zorah-texture-export-v1",
        "source_manifest": str(manifest_path),
        "max_size": args.max_size,
        "exported": sorted(exported, key=lambda record: str(record["object"])),
        "failures": failures,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"ZORAH_TEXTURE_SOURCE_BATCH_DONE exported={len(exported)} "
        f"failures={len(failures)} manifest={result_path}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
