#!/usr/bin/env python3
"""Re-shard an existing Zorah runtime tree without decoding its assets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


MAGIC = b"ZORAHB01"
VERSION = 2
DEFAULT_GEOMETRY_SHARD_BYTES = 512 * 1024 * 1024
COPY_BUFFER_BYTES = 8 * 1024 * 1024


@dataclass
class SourceEntry:
    record: dict
    offset: int


def read_index(path: Path) -> tuple[list[SourceEntry], int]:
    with path.open("rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError(f"wrong bundle magic in {path}")
        version, index_length = struct.unpack("<IQ", stream.read(12))
        if version != VERSION:
            raise ValueError(f"unsupported bundle version {version} in {path}")
        index = json.loads(stream.read(index_length))
        if index.get("format_version") != VERSION:
            raise ValueError(f"wrong index version in {path}")
        offset = stream.tell()
        entries = []
        for record in index.get("entries", []):
            entries.append(SourceEntry(record, offset))
            offset += int(record["byte_length"])
        stream.seek(0, 2)
        if stream.tell() != offset:
            raise ValueError(f"payload length mismatch in {path}")
        return entries, offset


def copy_exact(source: BinaryIO, destination: BinaryIO, length: int) -> None:
    remaining = length
    while remaining:
        block = source.read(min(remaining, COPY_BUFFER_BYTES))
        if not block:
            raise EOFError("bundle payload ended early")
        destination.write(block)
        remaining -= len(block)


class GeometryShardWriter:
    def __init__(self, output: Path, target_bytes: int) -> None:
        self.output = output
        self.target_bytes = target_bytes
        self.next_id = 0
        self.payload = None
        self.payload_path = None
        self.entries: list[dict] = []
        self.payload_bytes = 0
        self.file_name = ""
        self.references: dict[str, str] = {}

    def ensure_room(self, byte_length: int) -> None:
        if self.payload_bytes and self.payload_bytes + byte_length > self.target_bytes:
            self.finish()

    def add_group(
        self,
        source_path: Path,
        source_root: str,
        entries: list[SourceEntry],
    ) -> None:
        group_bytes = sum(int(entry.record["byte_length"]) for entry in entries)
        self.ensure_room(group_bytes)
        if self.payload is None:
            self.file_name = f"zorah-g-{self.next_id:03}.zorah_bundle"
            self.next_id += 1
            self.payload_path = self.output / "bundles" / f".{self.file_name}.payload"
            self.payload = self.payload_path.open("wb")
        with source_path.open("rb") as source:
            for entry in entries:
                record = entry.record
                source.seek(entry.offset)
                copy_exact(source, self.payload, int(record["byte_length"]))
                self.entries.append(record)
                label = record["label"]
                self.references[f"{source_root}#{label}"] = (
                    f"bundles/{self.file_name}#{label}"
                )
                self.payload_bytes += int(record["byte_length"])

    def finish(self) -> None:
        if self.payload is None:
            return
        self.payload.flush()
        self.payload.close()
        index = json.dumps(
            {"format_version": VERSION, "entries": self.entries},
            separators=(",", ":"),
        ).encode("utf-8")
        destination = self.output / "bundles" / self.file_name
        with destination.open("wb") as bundle:
            bundle.write(MAGIC)
            bundle.write(struct.pack("<IQ", VERSION, len(index)))
            bundle.write(index)
            with self.payload_path.open("rb") as payload:
                shutil.copyfileobj(payload, bundle, COPY_BUFFER_BYTES)
        self.payload_path.unlink()
        print(
            "ZORAH_RESHARD_BUNDLE "
            f"bundle={self.file_name} entries={len(self.entries)} "
            f"payload_bytes={self.payload_bytes}",
            flush=True,
        )
        self.payload = None
        self.payload_path = None
        self.entries = []
        self.payload_bytes = 0
        self.file_name = ""


def grouped_geometry_entries(entries: list[SourceEntry]) -> list[list[SourceEntry]]:
    groups: list[list[SourceEntry]] = []
    for entry in entries:
        label = entry.record["label"]
        prefix, separator, _ = label.rpartition("/")
        if not separator:
            raise ValueError(f"geometry label has no partition prefix: {label}")
        if groups and groups[-1][0].record["label"].rpartition("/")[0] == prefix:
            groups[-1].append(entry)
        else:
            groups.append([entry])
    for group in groups:
        kinds = {entry.record["kind"] for entry in group}
        if kinds != {"meshlet", "meshlet_blas"}:
            raise ValueError(
                f"incomplete geometry pair {group[0].record['label']}: {sorted(kinds)}"
            )
    return groups


def copy_or_link(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def rewrite_references(value, references: dict[str, str]):
    if isinstance(value, dict):
        return {key: rewrite_references(child, references) for key, child in value.items()}
    if isinstance(value, list):
        return [rewrite_references(child, references) for child in value]
    if isinstance(value, str):
        return references.get(value, value)
    return value


def reshard(source: Path, output: Path, geometry_shard_bytes: int) -> None:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise ValueError(f"runtime asset tree does not exist: {source}")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if geometry_shard_bytes < 64 * 1024 * 1024:
        raise ValueError("geometry shards must be at least 64 MiB")
    pack_state = None
    pack_state_path = source / "pack.json"
    if pack_state_path.is_file():
        pack_state = json.loads(pack_state_path.read_text(encoding="utf-8"))
        packed_bytes = pack_state.get("geometry_shard_bytes")
        # Shard boundaries are only ever split here: each source bundle is
        # finished separately to preserve the packer's level-usage grouping.
        if isinstance(packed_bytes, int) and geometry_shard_bytes > packed_bytes:
            raise ValueError(
                f"resharding cannot grow shards from {packed_bytes} to "
                f"{geometry_shard_bytes} bytes; repack into a clean output instead"
            )
    (output / "bundles").mkdir(parents=True)
    writer = GeometryShardWriter(output, geometry_shard_bytes)
    texture_bundles = 0
    for bundle in sorted((source / "bundles").glob("*.zorah_bundle")):
        relative = bundle.relative_to(source).as_posix()
        entries, _ = read_index(bundle)
        kinds = {entry.record["kind"] for entry in entries}
        if kinds <= {"meshlet", "meshlet_blas"}:
            for group in grouped_geometry_entries(entries):
                writer.add_group(bundle, relative, group)
            # Preserve packer's level-usage boundary even when the old shard
            # ended substantially below its target size.
            writer.finish()
        elif kinds == {"image"}:
            copy_or_link(bundle, output / "bundles" / bundle.name)
            texture_bundles += 1
        else:
            raise ValueError(f"mixed or unsupported bundle kinds in {bundle}: {sorted(kinds)}")
    writer.finish()

    geometry = json.loads((source / "geometry.json").read_text(encoding="utf-8"))
    geometry = rewrite_references(geometry, writer.references)
    (output / "geometry.json").write_text(
        json.dumps(geometry, indent=2) + "\n",
        encoding="utf-8",
    )
    for name in ("materials.json", "textures.exported.json"):
        shutil.copy2(source / name, output / name)
    shutil.copytree(source / "scenes", output / "scenes", copy_function=shutil.copy2)
    if pack_state is not None:
        # Written last: convert.py treats pack.json as the completion marker of a
        # staged tree, and without it the next run repacks the whole scene.
        pack_state["geometry_shard_bytes"] = geometry_shard_bytes
        (output / "pack.json").write_text(
            json.dumps(pack_state, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        "ZORAH_RESHARD_DONE "
        f"geometry_bundles={writer.next_id} texture_bundles={texture_bundles} "
        f"rewritten_references={len(writer.references)} output={output}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--geometry-shard-bytes",
        type=int,
        default=DEFAULT_GEOMETRY_SHARD_BYTES,
    )
    args = parser.parse_args()
    reshard(args.source, args.output, args.geometry_shard_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
