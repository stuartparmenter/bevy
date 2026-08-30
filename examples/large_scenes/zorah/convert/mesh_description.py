#!/usr/bin/env python3
"""Stream Zorah's UE 5.4 source meshes out of uncooked .uasset files.

This deliberately implements only the FMeshDescription layout used by the
Zorah sample.  It reads the package-trailer FCompressedBuffer one 256 KiB
block at a time and copies the useful arrays to disk without constructing a
Python object per vertex.  That is important for Zorah: the largest 2 GiB
package expands to more than 8 GiB of mesh-description data.

The output is an intermediate format, not a runtime Bevy asset.  A later stage
partitions these arrays by material/spatial cell, simplifies them, and feeds
the resulting bounded meshes to Bevy's MeshletMesh::from_mesh.

Requires the external ``pyooz`` package for Oodle decompression.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

try:
    import ooz
except ImportError:
    ooz = None


PACKAGE_TAG = 0x9E2A83C1
COMPRESSED_BUFFER_MAGIC = 0xB7756362
COMPRESSED_BUFFER_HEADER_SIZE = 64
NONE_METHOD = 0
OODLE_METHOD = 3

TYPE_SIZES = {
    0: 16,  # FVector4f
    1: 12,  # FVector3f
    2: 8,   # FVector2f
    3: 4,   # float
    4: 4,   # int32
    5: 4,   # bool (serialized as int32)
}

# TArray<bool>'s bulk payload uses one byte per value even though a standalone
# serialized bool/default occupies four bytes.
BULK_TYPE_SIZES = {**TYPE_SIZES, 5: 1}

TYPE_NAMES = {
    0: "FVector4f",
    1: "FVector3f",
    2: "FVector2f",
    3: "float",
    4: "int32",
    5: "bool",
    6: "FName",
}

# These are the only source arrays required to reconstruct Zorah render
# vertices. Triangle indices address vertex instances, which carry normal and
# UV seams and map back to the position-only vertex array. The pipeline is
# tangent-free, so the Tangent and BinormalSign streams are skipped rather
# than written out.
CAPTURES = {
    ("Vertices", "Position"): "positions.f32x3",
    ("VertexInstances", "VertexIndex"): "vertex_instance_vertices.i32",
    ("VertexInstances", "TextureCoordinate"): "vertex_instance_uv0.f32x2",
    ("VertexInstances", "Normal"): "vertex_instance_normals.f32x3",
    ("Triangles", "VertexInstanceIndex"): "triangle_vertex_instances.i32x3",
    ("Triangles", "PolygonGroupIndex"): "triangle_materials.i32",
}


@dataclass
class CompressedBufferInfo:
    method: int
    block_size: int
    block_count: int
    raw_size: int
    compressed_size: int
    payload_offset: int


class OodleBlockReader:
    """A forward-only raw-data reader over a UE FCompressedBuffer."""

    def __init__(
        self,
        package_path: Path,
        recover_bad_blocks: set[int] | None = None,
    ):
        self.path = package_path
        self.file = package_path.open("rb")
        self.info, self.block_sizes = self._read_header()
        self.recover_bad_blocks = recover_bad_blocks or set()
        self.recovered_blocks: list[int] = []
        self.block_index = 0
        self.raw_position = 0
        self.block = b""
        self.block_position = 0

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "OodleBlockReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _read_header(self) -> tuple[CompressedBufferInfo, list[int]]:
        file_size = self.file.seek(0, os.SEEK_END)
        if file_size < 20:
            raise ValueError("file is too short to contain a UE package trailer")

        self.file.seek(0)
        if struct.unpack("<I", self.file.read(4))[0] != PACKAGE_TAG:
            raise ValueError("not an Unreal package")

        # Footer is: tag:u64, trailer_length:u64, package_tag:u32.
        self.file.seek(file_size - 20)
        _footer_tag, trailer_length, package_tag = struct.unpack(
            "<QQI", self.file.read(20)
        )
        if package_tag != PACKAGE_TAG:
            raise ValueError("package trailer has the wrong closing tag")
        if trailer_length <= 0 or trailer_length > file_size:
            raise ValueError(f"invalid package trailer length {trailer_length}")

        trailer_offset = file_size - trailer_length
        self.file.seek(trailer_offset)
        header = self.file.read(28)
        if len(header) != 28:
            raise ValueError("truncated package trailer header")
        _header_tag, _version, header_length, payloads_length, payload_count = (
            struct.unpack("<QiIQi", header)
        )
        if payload_count != 1:
            raise ValueError(
                f"Zorah mesh package has {payload_count} trailer payloads; expected 1"
            )

        payload_offset = trailer_offset + header_length
        if payload_offset + payloads_length > file_size:
            raise ValueError("package trailer payload extends beyond end of file")
        self.file.seek(payload_offset)
        compressed_header = self.file.read(COMPRESSED_BUFFER_HEADER_SIZE)
        if len(compressed_header) != COMPRESSED_BUFFER_HEADER_SIZE:
            raise ValueError("truncated FCompressedBuffer header")

        magic = struct.unpack_from(">I", compressed_header, 0)[0]
        method = compressed_header[8]
        block_size_exponent = compressed_header[11]
        block_count = struct.unpack_from(">I", compressed_header, 12)[0]
        raw_size = struct.unpack_from(">Q", compressed_header, 16)[0]
        compressed_size = struct.unpack_from(">Q", compressed_header, 24)[0]
        if magic != COMPRESSED_BUFFER_MAGIC:
            raise ValueError(f"bad FCompressedBuffer magic 0x{magic:08x}")
        if method not in (NONE_METHOD, OODLE_METHOD):
            raise ValueError(f"unsupported compression method {method}")
        if compressed_size > payloads_length:
            raise ValueError("compressed buffer extends beyond trailer payload")

        if method == NONE_METHOD:
            # An uncompressed payload stores no block-size table: the raw bytes
            # follow the header directly, so the whole payload is one block.
            expected_size = COMPRESSED_BUFFER_HEADER_SIZE + raw_size
            if expected_size != compressed_size:
                raise ValueError(
                    f"uncompressed size mismatch: header={compressed_size}, "
                    f"raw={expected_size}"
                )
            return (
                CompressedBufferInfo(
                    method=method,
                    block_size=raw_size,
                    block_count=1,
                    raw_size=raw_size,
                    compressed_size=compressed_size,
                    payload_offset=payload_offset,
                ),
                [raw_size],
            )

        if block_size_exponent <= 0 or block_size_exponent >= 32:
            raise ValueError(f"invalid block-size exponent {block_size_exponent}")

        sizes_data = self.file.read(block_count * 4)
        if len(sizes_data) != block_count * 4:
            raise ValueError("truncated compressed-block size table")
        block_sizes = list(struct.unpack(f">{block_count}I", sizes_data))
        block_size = 1 << block_size_exponent
        expected_size = COMPRESSED_BUFFER_HEADER_SIZE + len(sizes_data) + sum(block_sizes)
        if expected_size != compressed_size:
            raise ValueError(
                f"compressed size mismatch: header={compressed_size}, blocks={expected_size}"
            )

        return (
            CompressedBufferInfo(
                method=method,
                block_size=block_size,
                block_count=block_count,
                raw_size=raw_size,
                compressed_size=compressed_size,
                payload_offset=payload_offset,
            ),
            block_sizes,
        )

    def _raw_block_size(self, index: int) -> int:
        if index + 1 < self.info.block_count:
            return self.info.block_size
        return self.info.raw_size - self.info.block_size * (self.info.block_count - 1)

    def _load_block(self) -> None:
        if self.block_index >= self.info.block_count:
            raise EOFError("read past end of decompressed mesh payload")
        compressed_size = self.block_sizes[self.block_index]
        raw_size = self._raw_block_size(self.block_index)
        compressed = self.file.read(compressed_size)
        if len(compressed) != compressed_size:
            raise EOFError(f"truncated compressed block {self.block_index}")
        if compressed_size >= raw_size:
            self.block = compressed[:raw_size]
        else:
            if ooz is None:
                raise RuntimeError(
                    "pyooz is required; install it into an isolated environment "
                    "with `sfw uv pip install pyooz`"
                )
            try:
                self.block = ooz.decompress(compressed, raw_size)
            except RuntimeError as error:
                if (
                    self.block_index in self.recover_bad_blocks
                    and len(self.block) == raw_size
                ):
                    # A single source block in Zorah 1.1.0 is damaged in the
                    # original archive. Repeating the preceding equal-sized
                    # block preserves all other source data and produces a
                    # narrow 16-scanline normal-map repair.
                    self.block = bytes(self.block)
                    self.recovered_blocks.append(self.block_index)
                else:
                    raise RuntimeError(
                        f"Oodle block {self.block_index} failed: "
                        f"compressed={compressed_size} raw={raw_size}: {error}"
                    ) from error
            if len(self.block) != raw_size:
                raise ValueError(
                    f"Oodle block {self.block_index} decoded to {len(self.block)} "
                    f"bytes; expected {raw_size}"
                )
        self.block_index += 1
        self.block_position = 0

    def read_exact(self, size: int) -> bytes:
        if size < 0:
            raise ValueError(f"negative read size {size}")
        result = bytearray(size)
        view = memoryview(result)
        written = 0
        while written < size:
            if self.block_position == len(self.block):
                self._load_block()
            count = min(size - written, len(self.block) - self.block_position)
            view[written : written + count] = self.block[
                self.block_position : self.block_position + count
            ]
            written += count
            self.block_position += count
            self.raw_position += count
        return bytes(result)

    def skip(self, size: int) -> None:
        if size < 0:
            raise ValueError(f"negative skip size {size}")
        while size:
            remaining = len(self.block) - self.block_position
            if remaining:
                count = min(size, remaining)
                self.block_position += count
                self.raw_position += count
                size -= count
                continue

            if self.block_index >= self.info.block_count:
                raise EOFError("skip past end of decompressed mesh payload")

            raw_block_size = self._raw_block_size(self.block_index)
            if size >= raw_block_size:
                self.file.seek(self.block_sizes[self.block_index], os.SEEK_CUR)
                self.block_index += 1
                self.raw_position += raw_block_size
                size -= raw_block_size
            else:
                self._load_block()

    def copy_exact(self, output: BinaryIO, size: int) -> None:
        while size:
            if self.block_position == len(self.block):
                self._load_block()
            count = min(size, len(self.block) - self.block_position)
            output.write(self.block[self.block_position : self.block_position + count])
            self.block_position += count
            self.raw_position += count
            size -= count

    def i32(self) -> int:
        return struct.unpack("<i", self.read_exact(4))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.read_exact(4))[0]

    def fstring(self) -> str:
        length = self.i32()
        if length > 0:
            data = self.read_exact(length)
            return data[:-1].decode("latin-1", errors="replace")
        if length < 0:
            data = self.read_exact(-length * 2)
            return data[:-2].decode("utf-16-le", errors="replace")
        return ""


class MeshDescriptionExtractor:
    def __init__(self, source: Path, output: Path | None):
        self.source = source
        self.output = output
        self.arrays: dict[str, dict[str, int | str]] = {}
        self.elements: dict[str, dict[str, object]] = {}
        self.material_slots: list[str] = []

    def run(self) -> dict[str, object]:
        with OodleBlockReader(self.source) as reader:
            entry_count = reader.i32()
            if entry_count <= 0 or entry_count > 32:
                raise ValueError(f"implausible mesh element map size {entry_count}")
            for _ in range(entry_count):
                element_name = reader.fstring()
                channel_count = reader.i32()
                if channel_count <= 0 or channel_count > 16:
                    raise ValueError(
                        f"{element_name} has implausible channel count {channel_count}"
                    )
                channels = [
                    self._element_container(reader, element_name, channel_index)
                    for channel_index in range(channel_count)
                ]
                self.elements[element_name] = (
                    channels[0] if len(channels) == 1 else {"channels": channels}
                )
            if reader.raw_position != reader.info.raw_size:
                raise ValueError(
                    f"mesh parser stopped at {reader.raw_position} of "
                    f"{reader.info.raw_size} decompressed bytes"
                )

            return {
                "format": "zorah-mesh-description-v2",
                "source": str(self.source),
                "source_size": self.source.stat().st_size,
                "compressed_buffer": asdict(reader.info),
                "elements": self.elements,
                "arrays": self.arrays,
                "material_slots": self.material_slots,
            }

    def _element_container(
        self, reader: OodleBlockReader, element_name: str, element_channel: int
    ) -> dict[str, object]:
        bit_count = reader.i32()
        if bit_count < 0:
            raise ValueError(f"negative bit-array size in {element_name}")
        word_count = (bit_count + 31) // 32
        valid_count = 0
        for _ in range(word_count):
            valid_count += reader.u32().bit_count()
        hole_count = reader.i32()
        element_count = reader.i32()
        attribute_count = reader.i32()
        if min(hole_count, element_count, attribute_count) < 0:
            raise ValueError(f"negative container count in {element_name}")
        if hole_count:
            raise ValueError(
                f"{element_name} has {hole_count} sparse holes; no Zorah mesh seen so far does"
            )

        attributes: dict[str, dict[str, object]] = {}
        for _ in range(attribute_count):
            name = reader.fstring().strip()
            attributes[name] = self._attribute(
                reader, element_name, element_channel, name
            )
        return {
            "bit_count": bit_count,
            "valid_count": valid_count,
            "hole_count": hole_count,
            "element_count": element_count,
            "attributes": attributes,
        }

    def _attribute(
        self,
        reader: OodleBlockReader,
        element_name: str,
        element_channel: int,
        attribute_name: str,
    ) -> dict[str, object]:
        attribute_type = reader.u32()
        extent = reader.u32()
        element_count = reader.i32()
        channel_count = reader.i32()
        if attribute_type not in TYPE_NAMES:
            raise ValueError(
                f"unknown attribute type {attribute_type} for {element_name}.{attribute_name}"
            )
        if min(element_count, channel_count) < 0:
            raise ValueError(f"negative attribute count in {element_name}.{attribute_name}")
        if extent == 0:
            raise ValueError(
                f"unbounded attribute {element_name}.{attribute_name} is not used by Zorah"
            )

        channels: list[dict[str, object]] = []
        for channel_index in range(channel_count):
            array_extent = reader.u32()
            if attribute_type in TYPE_SIZES:
                element_size = reader.i32()
                value_count = reader.i32()
                if element_size != BULK_TYPE_SIZES[attribute_type]:
                    raise ValueError(
                        f"unexpected element size {element_size} for "
                        f"{element_name}.{attribute_name} ({TYPE_NAMES[attribute_type]})"
                    )
                if value_count < 0:
                    raise ValueError(
                        f"negative value count in {element_name}.{attribute_name}"
                    )
                byte_count = element_size * value_count
                capture_name = (
                    CAPTURES.get((element_name, attribute_name))
                    if element_channel == 0
                    else None
                )
                if capture_name is not None and channel_index == 0 and self.output:
                    destination = self.output / capture_name
                    with destination.open("wb") as output_file:
                        reader.copy_exact(output_file, byte_count)
                    self.arrays[capture_name] = {
                        "type": TYPE_NAMES[attribute_type],
                        "extent": extent,
                        "element_size": element_size,
                        "value_count": value_count,
                        "byte_count": byte_count,
                    }
                else:
                    reader.skip(byte_count)
                channels.append(
                    {
                        "array_extent": array_extent,
                        "value_count": value_count,
                        "byte_count": byte_count,
                        "captured": capture_name
                        if capture_name is not None and channel_index == 0
                        else None,
                    }
                )
            elif attribute_type == 6:
                value_count = reader.i32()
                values = [reader.fstring() for _ in range(value_count)]
                if (
                    (element_name, attribute_name)
                    == ("PolygonGroups", "ImportedMaterialSlotName")
                    and element_channel == 0
                    and channel_index == 0
                ):
                    self.material_slots = values
                channels.append(
                    {
                        "array_extent": array_extent,
                        "value_count": value_count,
                        "values": values,
                    }
                )

        self._skip_default(reader, attribute_type)
        flags = reader.u32()
        return {
            "type": TYPE_NAMES[attribute_type],
            "extent": extent,
            "element_count": element_count,
            "channel_count": channel_count,
            "channels": channels,
            "flags": flags,
        }

    @staticmethod
    def _skip_default(reader: OodleBlockReader, attribute_type: int) -> None:
        if attribute_type in TYPE_SIZES:
            reader.skip(TYPE_SIZES[attribute_type])
        elif attribute_type == 6:
            reader.fstring()


def extract(source: Path, destination: Path) -> dict[str, object]:
    if destination.exists():
        raise FileExistsError(
            f"destination already exists: {destination} (remove it explicitly to replace it)"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        manifest = MeshDescriptionExtractor(source, temporary).run()
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subcommands.add_parser(
        "inspect", help="stream through a mesh and print its schema without writing arrays"
    )
    inspect_parser.add_argument("source", type=Path)
    extract_parser = subcommands.add_parser(
        "extract", help="write the compact Zorah mesh-description intermediate"
    )
    extract_parser.add_argument("source", type=Path)
    extract_parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_file() or source.suffix.lower() != ".uasset":
        parser.error(f"source is not a .uasset file: {source}")

    if args.command == "inspect":
        manifest = MeshDescriptionExtractor(source, None).run()
    else:
        manifest = extract(source, args.destination.resolve())
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EOFError, OSError, RuntimeError, ValueError) as error:
        print(f"ZORAH_CONVERT_ERROR {error}", file=sys.stderr)
        raise SystemExit(1)
