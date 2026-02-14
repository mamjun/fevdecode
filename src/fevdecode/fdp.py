from __future__ import annotations

import hashlib
import io
import os
import struct
from dataclasses import dataclass
from typing import Iterable, List

MAGIC = b"FDP0"
VERSION = 1

_HEADER_STRUCT = struct.Struct("<4sHH")
_ENTRY_HEADER_STRUCT = struct.Struct("<H")
_ENTRY_TAIL_STRUCT = struct.Struct("<I32sQ")


@dataclass(frozen=True)
class FdpEntry:
    name: str
    size: int
    sha256: bytes
    offset: int


class FdpFormatError(ValueError):
    pass


def _hash_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def build_fdp(entries: Iterable[tuple[str, bytes]], output_path: str) -> List[FdpEntry]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    payloads = []
    for name, data in entries:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Entry {name} is not bytes")
        payloads.append((name, bytes(data)))

    table = []
    offset = 0
    for name, data in payloads:
        sha = _hash_bytes(data)
        table.append((name, len(data), sha, offset))
        offset += len(data)

    with open(output_path, "wb") as fp:
        fp.write(_HEADER_STRUCT.pack(MAGIC, VERSION, len(table)))
        for name, size, sha, off in table:
            name_bytes = name.encode("utf-8")
            if len(name_bytes) > 65535:
                raise ValueError(f"Entry name too long: {name}")
            fp.write(_ENTRY_HEADER_STRUCT.pack(len(name_bytes)))
            fp.write(name_bytes)
            fp.write(_ENTRY_TAIL_STRUCT.pack(size, sha, off))
        for _, data in payloads:
            fp.write(data)

    return [FdpEntry(name, size, sha, off) for name, size, sha, off in table]


def parse_fdp(input_path: str) -> List[FdpEntry]:
    with open(input_path, "rb") as fp:
        header = fp.read(_HEADER_STRUCT.size)
        if len(header) != _HEADER_STRUCT.size:
            raise FdpFormatError("File too small")
        magic, version, count = _HEADER_STRUCT.unpack(header)
        if magic != MAGIC:
            raise FdpFormatError("Bad magic")
        if version != VERSION:
            raise FdpFormatError(f"Unsupported version: {version}")

        entries: List[FdpEntry] = []
        for _ in range(count):
            name_len_bytes = fp.read(_ENTRY_HEADER_STRUCT.size)
            if len(name_len_bytes) != _ENTRY_HEADER_STRUCT.size:
                raise FdpFormatError("Truncated entry header")
            (name_len,) = _ENTRY_HEADER_STRUCT.unpack(name_len_bytes)
            name_bytes = fp.read(name_len)
            if len(name_bytes) != name_len:
                raise FdpFormatError("Truncated entry name")
            tail = fp.read(_ENTRY_TAIL_STRUCT.size)
            if len(tail) != _ENTRY_TAIL_STRUCT.size:
                raise FdpFormatError("Truncated entry tail")
            size, sha, offset = _ENTRY_TAIL_STRUCT.unpack(tail)
            entries.append(FdpEntry(name_bytes.decode("utf-8"), size, sha, offset))

        data_start = fp.tell()
        for entry in entries:
            fp.seek(data_start + entry.offset)
            data = fp.read(entry.size)
            if len(data) != entry.size:
                raise FdpFormatError(f"Truncated payload for {entry.name}")
            if _hash_bytes(data) != entry.sha256:
                raise FdpFormatError(f"Checksum mismatch for {entry.name}")

        return entries


def extract_fdp(input_path: str, output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    with open(input_path, "rb") as fp:
        header = fp.read(_HEADER_STRUCT.size)
        if len(header) != _HEADER_STRUCT.size:
            raise FdpFormatError("File too small")
        magic, version, count = _HEADER_STRUCT.unpack(header)
        if magic != MAGIC:
            raise FdpFormatError("Bad magic")
        if version != VERSION:
            raise FdpFormatError(f"Unsupported version: {version}")

        entries: List[FdpEntry] = []
        for _ in range(count):
            name_len_bytes = fp.read(_ENTRY_HEADER_STRUCT.size)
            (name_len,) = _ENTRY_HEADER_STRUCT.unpack(name_len_bytes)
            name_bytes = fp.read(name_len)
            tail = fp.read(_ENTRY_TAIL_STRUCT.size)
            size, sha, offset = _ENTRY_TAIL_STRUCT.unpack(tail)
            entries.append(FdpEntry(name_bytes.decode("utf-8"), size, sha, offset))

        data_start = fp.tell()
        written = []
        for entry in entries:
            fp.seek(data_start + entry.offset)
            data = fp.read(entry.size)
            if _hash_bytes(data) != entry.sha256:
                raise FdpFormatError(f"Checksum mismatch for {entry.name}")
            out_path = os.path.join(output_dir, entry.name)
            os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
            with open(out_path, "wb") as out:
                out.write(data)
            written.append(out_path)
        return written
