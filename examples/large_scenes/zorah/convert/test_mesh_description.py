"""Package-trailer decoding tests for `mesh_description.OodleBlockReader`."""

from __future__ import annotations

import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mesh_description import (
    COMPRESSED_BUFFER_HEADER_SIZE,
    COMPRESSED_BUFFER_MAGIC,
    NONE_METHOD,
    PACKAGE_TAG,
    OodleBlockReader,
)

TRAILER_HEADER_LENGTH = 28


def write_uncompressed_package(path: Path, payload: bytes, *, method: int) -> None:
    """Write the smallest package whose trailer holds `payload` uncompressed.

    Mirrors what UE emits for `ECompressedBufferCompressor::None`: a 64-byte
    FCompressedBuffer header followed directly by the raw bytes, with no
    block-size table.
    """
    buffer_header = bytearray(COMPRESSED_BUFFER_HEADER_SIZE)
    struct.pack_into(">I", buffer_header, 0, COMPRESSED_BUFFER_MAGIC)
    buffer_header[8] = method
    buffer_header[11] = 0  # block-size exponent is unused without a size table
    struct.pack_into(">I", buffer_header, 12, 1)
    struct.pack_into(">Q", buffer_header, 16, len(payload))
    struct.pack_into(">Q", buffer_header, 24, COMPRESSED_BUFFER_HEADER_SIZE + len(payload))

    payloads = bytes(buffer_header) + payload
    trailer = (
        struct.pack(
            "<QiIQi",
            0,
            0,
            TRAILER_HEADER_LENGTH,
            len(payloads),
            1,
        )
        + payloads
    )
    trailer_length = len(trailer) + 20
    footer = struct.pack("<QQI", 0, trailer_length, PACKAGE_TAG)
    path.write_bytes(struct.pack("<I", PACKAGE_TAG) + trailer + footer)


class UncompressedPayloadTest(unittest.TestCase):
    def test_uncompressed_payload_reads_back_verbatim(self):
        payload = bytes(range(256)) * 40
        with TemporaryDirectory() as directory:
            package = Path(directory) / "uncompressed.uasset"
            write_uncompressed_package(package, payload, method=NONE_METHOD)
            with OodleBlockReader(package) as reader:
                self.assertEqual(reader.info.method, NONE_METHOD)
                self.assertEqual(reader.info.raw_size, len(payload))
                self.assertEqual(reader.info.block_count, 1)
                self.assertEqual(reader.read_exact(len(payload)), payload)

    def test_uncompressed_payload_survives_a_seek(self):
        payload = bytes(range(256)) * 40
        with TemporaryDirectory() as directory:
            package = Path(directory) / "uncompressed.uasset"
            write_uncompressed_package(package, payload, method=NONE_METHOD)
            with OodleBlockReader(package) as reader:
                reader.skip(300)
                self.assertEqual(reader.read_exact(16), payload[300:316])

    def test_unknown_compression_method_is_rejected(self):
        with TemporaryDirectory() as directory:
            package = Path(directory) / "mystery.uasset"
            write_uncompressed_package(package, b"payload", method=7)
            with self.assertRaisesRegex(ValueError, "unsupported compression method 7"):
                OodleBlockReader(package)


if __name__ == "__main__":
    unittest.main()
