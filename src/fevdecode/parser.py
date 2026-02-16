from __future__ import annotations

import io
import math
import os
import re
import struct
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable, Optional


_PITCH_UNITS_MAP = {
    1: "Octaves",
    3: "Semitones",
}


def _linear_to_db(value: float) -> float:
    if value <= 0:
        return -96.0
    return 20.0 * math.log10(value)


def _linear_randomization_to_db(value: float) -> float:
    linear = 1.0 - value
    if linear <= 0:
        return -96.0
    return _linear_to_db(linear)


@dataclass(frozen=True)
class BinaryInfo:
    path: str
    size: int
    header: bytes


@dataclass(frozen=True)
class RiffChunk:
    chunk_id: str
    size: int
    offset: int
    children: tuple["RiffChunk", ...] = ()


@dataclass(frozen=True)
class RiffInfo:
    path: str
    size: int
    form_type: str
    chunks: tuple[RiffChunk, ...]
    strings: tuple[str, ...] = ()
    guid_strings: dict[str, str] | None = None
    proj: dict | None = None


@dataclass(frozen=True)
class FmodSoundDefFile:
    path: str
    bank_name: str
    file_index: int
    lengthms: int


@dataclass
class FmodSoundDef:
    name_index: int
    name: str = ""
    file_list: list[FmodSoundDefFile] = None

    def __post_init__(self) -> None:
        if self.file_list is None:
            self.file_list = []


@dataclass
class FmodEvent:
    is_simple: bool
    path_index: list[int]
    path: list[str]
    has_sounddef: bool
    sounddef_index_list: list[int]
    ref_file_list: list[FmodSoundDefFile]
    effect_params: list[dict]
    pitch_raw: float | None = None
    pitch_units_code: int | None = None
    pitch_units: str | None = None
    volume_raw: float | None = None
    volume_randomization: float | None = None
    volume_randomization_db: float | None = None
    spawn_intensity: float | None = None
    spawn_intensity_randomization: float | None = None
    max_playbacks: int | None = None
    priority: int | None = None
    fade_in_time: int | None = None
    fade_out_time: int | None = None


@dataclass(frozen=True)
class FmodFevParsed:
    proj_name: str
    bank_list: list[str]
    event_map: dict[str, FmodEvent]


def build_event_bank_file_map(
    fev_path: str,
    fsb_path: str | None = None,
    fsb_map: dict[str, str] | None = None,
    fdp_path: str | None = None,
) -> dict:
    parsed = parse_fev_event_map(fev_path)
    fsb = None
    fsb_lookup: dict[str, Any] = {}
    fsb_lookup_lower: dict[str, Any] = {}
    normalized_lookup: dict[str, Any] = {}

    fsb_by_bank: dict[str, Any] = {}
    fsb_map = fsb_map or {}

    if fsb_path:
        fsb, _ = _load_fsb5(fsb_path)
        fsb_lookup = {sample.name: sample for sample in fsb.samples}
        fsb_lookup_lower = {sample.name.lower(): sample for sample in fsb.samples}
        normalized_lookup = _build_normalized_sample_lookup(fsb_lookup)

    if fsb_map:
        combined_lookup: dict[str, Any] = {}
        combined_lookup_lower: dict[str, Any] = {}
        for bank_name, path in fsb_map.items():
            fsb_bank, _ = _load_fsb5(path)
            fsb_by_bank[bank_name] = fsb_bank
            for sample in fsb_bank.samples:
                combined_lookup.setdefault(sample.name, sample)
                combined_lookup_lower.setdefault(sample.name.lower(), sample)
        if combined_lookup:
            fsb_lookup = combined_lookup
            fsb_lookup_lower = combined_lookup_lower
            normalized_lookup = _build_normalized_sample_lookup(fsb_lookup)

    events: list[dict] = []
    fdp_effects: dict[str, dict] = {}
    if fdp_path:
        fdp_effects = parse_fdp_event_effects(fdp_path)
    for path, event in parsed.event_map.items():
        files: list[dict] = []
        for item in event.ref_file_list:
            sample_name = None
            if fsb and 0 <= item.file_index < len(fsb.samples):
                sample_name = fsb.samples[item.file_index].name
            elif fsb_by_bank:
                bank_fsb = fsb_by_bank.get(item.bank_name)
                if bank_fsb and 0 <= item.file_index < len(bank_fsb.samples):
                    sample_name = bank_fsb.samples[item.file_index].name
            files.append(
                {
                    "bank_name": item.bank_name,
                    "path": item.path,
                    "file_index": item.file_index,
                    "lengthms": item.lengthms,
                    "sample_name": sample_name,
                }
            )
        matched_samples: list[dict] = []
        if fsb_lookup:
            expected_name = _event_to_sample_name(path)
            sample, rule = _match_sample(expected_name, fsb_lookup, fsb_lookup_lower, normalized_lookup)
            if sample:
                matched_samples.append(
                    {
                        "sample_name": sample.name,
                        "match_rule": rule,
                    }
                )
        events.append(
            {
                "event": path,
                "is_simple": event.is_simple,
                "has_sounddef": event.has_sounddef,
                "pitch_raw": event.pitch_raw,
                "pitch_octaves": (event.pitch_raw * 4.0) if isinstance(event.pitch_raw, (int, float)) else None,
                "pitch_units_code": event.pitch_units_code,
                "pitch_units": event.pitch_units,
                "volume_raw": event.volume_raw,
                "volume_randomization": event.volume_randomization_db,
                "volume_randomization_raw": event.volume_randomization,
                "spawn_intensity": event.spawn_intensity,
                "spawn_intensity_randomization": event.spawn_intensity_randomization,
                "max_playbacks": event.max_playbacks,
                "priority": event.priority,
                "fade_in_time": event.fade_in_time,
                "fade_out_time": event.fade_out_time,
                "files": files,
                "effect_params": list(event.effect_params),
                "event_effects": fdp_effects.get(path, {}),
                "matched_samples": matched_samples,
            }
        )

    result = {
        "fev": fev_path,
        "fsb": fsb_path,
        "proj": parsed.proj_name,
        "bank_list": parsed.bank_list,
        "event_count": len(events),
        "events": events,
    }
    if fsb_map:
        result["fsb_list"] = sorted(set(fsb_map.values()))
        result["fsb_map"] = dict(fsb_map)
    return result


def serialize_fmod_events(parsed: FmodFevParsed) -> list[dict]:
    results: list[dict] = []
    for path, event in parsed.event_map.items():
        files = [
            {
                "path": item.path,
                "bank_name": item.bank_name,
                "file_index": item.file_index,
                "lengthms": item.lengthms,
            }
            for item in event.ref_file_list
        ]
        results.append(
            {
                "path": path,
                "is_simple": event.is_simple,
                "has_sounddef": event.has_sounddef,
                "sounddef_index_list": list(event.sounddef_index_list),
                "pitch_raw": event.pitch_raw,
                "pitch_octaves": (event.pitch_raw * 4.0) if isinstance(event.pitch_raw, (int, float)) else None,
                "pitch_units_code": event.pitch_units_code,
                "pitch_units": event.pitch_units,
                "volume_raw": event.volume_raw,
                "volume_randomization": event.volume_randomization_db,
                "volume_randomization_raw": event.volume_randomization,
                "spawn_intensity": event.spawn_intensity,
                "spawn_intensity_randomization": event.spawn_intensity_randomization,
                "max_playbacks": event.max_playbacks,
                "priority": event.priority,
                "fade_in_time": event.fade_in_time,
                "fade_out_time": event.fade_out_time,
                "files": files,
                "effect_params": list(event.effect_params),
            }
        )
    return results


def _read_binary(path: str, header_len: int = 16) -> tuple[bytes, BinaryInfo]:
    with open(path, "rb") as fp:
        header = fp.read(header_len)
        fp.seek(0)
        data = fp.read()
    return data, BinaryInfo(path=path, size=len(data), header=header)


def read_fev(path: str) -> tuple[bytes, BinaryInfo]:
    if not path.lower().endswith(".fev"):
        raise ValueError("Not a .fev file")
    return _read_binary(path)


def read_fsb(path: str) -> tuple[bytes, BinaryInfo]:
    if not path.lower().endswith(".fsb"):
        raise ValueError("Not a .fsb file")
    return _read_binary(path)


def _load_fsb5(path: str):
    try:
        import fsb5
    except Exception as exc:
        raise RuntimeError("python-fsb5 not installed. Run: pip install python-fsb5") from exc
    data = read_fsb(path)[0]
    return fsb5.load(data), len(data)


class _BinaryReader:
    def __init__(self, data: bytes, endian: str = "<") -> None:
        self._buf = io.BytesIO(data)
        self._endian = endian

    def tell(self) -> int:
        return self._buf.tell()

    def seek(self, offset: int, whence: int = 0) -> None:
        self._buf.seek(offset, whence)

    def read(self, size: int) -> bytes:
        return self._buf.read(size)

    def read_struct(self, fmt: str) -> tuple:
        size = struct.calcsize(self._endian + fmt)
        data = self.read(size)
        if len(data) != size:
            raise ValueError("Not enough bytes")
        return struct.unpack(self._endian + fmt, data)

    def read_type(self, fmt: str) -> int:
        values = self.read_struct(fmt)
        if len(values) != 1:
            raise ValueError("Expected single value")
        return int(values[0])

    def read_string(self, max_len: int) -> str:
        data = bytearray()
        for _ in range(max_len):
            byte = self.read(1)
            if not byte or byte == b"\0":
                break
            data.extend(byte)
        return data.decode("utf-8", errors="replace")


class _FevStreamReader:
    def __init__(self, fp: io.BufferedReader) -> None:
        self._fp = fp

    def read_exact(self, size: int) -> bytes:
        data = self._fp.read(size)
        if data is None or len(data) != size:
            raise ValueError("Unexpected EOF")
        return data

    def read_u32(self) -> int:
        return int.from_bytes(self.read_exact(4), "little", signed=False)

    def read_u16(self) -> int:
        return int.from_bytes(self.read_exact(2), "little", signed=False)

    def read_l_buf(self) -> bytes:
        length = self.read_u32()
        if length == 0:
            return b""
        data = self.read_exact(length)
        return data[:-1] if data.endswith(b"\0") else data

    def skip(self, size: int) -> None:
        if size <= 0:
            return
        self.read_exact(size)

    def seek_to_string(self, needle: bytes) -> None:
        if not needle:
            return
        idx = 0
        while True:
            byte = self._fp.read(1)
            if not byte:
                raise ValueError(f"Failed to seek to string: {needle!r}")
            if byte[0] == needle[idx]:
                idx += 1
                if idx == len(needle):
                    return
            else:
                idx = 0

    def seek_absolute(self, offset: int) -> None:
        self._fp.seek(offset)


class _PackedReader:
    def __init__(self, data: bytes) -> None:
        self._buf = memoryview(data)
        self._pos = 0

    def remaining(self) -> int:
        return max(0, len(self._buf) - self._pos)

    def read_bytes(self, size: int) -> bytes:
        size = max(0, min(size, self.remaining()))
        start = self._pos
        self._pos += size
        return bytes(self._buf[start : start + size])

    def read_u16(self) -> int:
        data = self.read_bytes(2)
        if len(data) < 2:
            return 0
        return int.from_bytes(data, "little", signed=False)

    def read_u32(self) -> int:
        data = self.read_bytes(4)
        if len(data) < 4:
            return 0
        return int.from_bytes(data, "little", signed=False)

    def read_size(self) -> int:
        length = self.read_u16()
        if length & 0x8000:
            length = (length & 0x7FFF) | (self.read_u16() << 15)
        return length

    def read_element_array(self) -> bytes:
        count_flag = self.read_size()
        if count_flag in (0, 1):
            return b""
        is_uniform = (count_flag & 1) == 1
        count = count_flag >> 1
        if not is_uniform:
            for _ in range(count):
                size = self.read_u16()
                self.read_bytes(size)
            return b""
        element_size = self.read_u16()
        return self.read_bytes(element_size * count)


def _align_even(value: int) -> int:
    return value + (value % 2)


def _parse_riff_chunks(data: bytes, base_offset: int, size_limit: int) -> tuple[RiffChunk, ...]:
    chunks: list[RiffChunk] = []
    max_size = min(size_limit, max(0, len(data) - base_offset))
    offset = 0
    while offset + 8 <= max_size:
        chunk_id = data[base_offset + offset : base_offset + offset + 4].decode("ascii", errors="replace")
        size = int.from_bytes(data[base_offset + offset + 4 : base_offset + offset + 8], "little")
        payload_offset = offset + 8
        payload_end = payload_offset + size
        if payload_end > max_size:
            break
        children: tuple[RiffChunk, ...] = ()
        if chunk_id == "LIST":
            if payload_offset + 4 <= payload_end:
                list_type = data[base_offset + payload_offset : base_offset + payload_offset + 4]
                list_type_str = list_type.decode("ascii", errors="replace")
                child_base = payload_offset + 4
                child_size = max(0, size - 4)
                children = _parse_riff_chunks(data, base_offset + child_base, child_size)
                chunk_id = f"LIST:{list_type_str}"
        elif chunk_id == "LGCY":
            children = _parse_riff_chunks(data, base_offset + payload_offset, max(0, size))
        chunks.append(RiffChunk(chunk_id=chunk_id, size=size, offset=base_offset + offset, children=children))
        offset = _align_even(payload_end)
    return tuple(chunks)


def parse_fev_riff(path: str) -> RiffInfo:
    data = read_fev(path)[0]
    if len(data) < 12 or data[:4] != b"RIFF":
        raise ValueError("FEV is not RIFF")
    form_type = data[8:12].decode("ascii", errors="replace")
    riff_size = int.from_bytes(data[4:8], "little")
    size_limit = min(len(data) - 12, max(0, riff_size - 4))
    chunks = _parse_riff_chunks(data, 12, size_limit)
    strings = tuple(_parse_fev_string_table(data, chunks))
    guid_strings = _parse_fev_string_data(data, chunks)
    proj = _parse_fev_proj_details(data, chunks)
    return RiffInfo(
        path=path,
        size=len(data),
        form_type=form_type,
        chunks=chunks,
        strings=strings,
        guid_strings=guid_strings,
        proj=proj,
    )


def parse_fev_event_map(path: str) -> FmodFevParsed:
    with open(path, "rb") as fp:
        reader = _FevStreamReader(fp)
        reader.seek_to_string(b"RIFF")
        reader.seek_to_string(b"LIST")
        reader.seek_to_string(b"PROJ")
        reader.seek_to_string(b"OBCT")

        reader.skip(20)
        numbanks = reader.read_u32()
        reader.skip(4)
        numcategories = reader.read_u32()
        reader.skip(4)
        numgroups = reader.read_u32()
        reader.skip(36)
        numevents = reader.read_u32()
        reader.skip(28)
        reader.read_u32()  # numreverbs
        reader.skip(4)
        reader.read_u32()  # numwaveforms
        reader.skip(28)
        numsounddefs = reader.read_u32()
        reader.skip(128)
        _ = reader.read_l_buf()  # name

        reader.seek_to_string(b"LGCY")
        reader.skip(12)
        proj_name = reader.read_l_buf().decode("utf-8", errors="replace")

        bank_list: list[str] = []
        reader.skip(8)
        for _ in range(numbanks):
            reader.skip(20)
            bank_name = reader.read_l_buf().decode("utf-8", errors="replace")
            bank_list.append(bank_name)

        for _ in range(numcategories):
            _ = reader.read_l_buf()
            reader.skip(20)

        item_index_stack: list[int] = []
        events: list[FmodEvent] = []

        def parse_event() -> None:
            event_type = reader.read_u32()
            if event_type == 16:
                name_index = reader.read_u32()
                header = reader.read_exact(16 + 144)
                volume_raw = struct.unpack_from("<f", header, 16)[0]
                pitch_raw = struct.unpack_from("<f", header, 20)[0]
                volume_randomization = struct.unpack_from("<f", header, 28)[0]
                volume_randomization_db = _linear_randomization_to_db(volume_randomization)
                max_playbacks = struct.unpack_from("<I", header, 36)[0]
                priority = struct.unpack_from("<I", header, 32)[0]
                fade_in_time = struct.unpack_from("<I", header, 132)[0]
                spawn_intensity = struct.unpack_from("<f", header, 140)[0]
                spawn_intensity_randomization = struct.unpack_from("<f", header, 144)[0]
                pitch_units_code = None
                pitch_units = None
                num = reader.read_u32()
                if num > 1:
                    raise ValueError("Simple event must have only 0/1 sounddef")
                def_index = reader.read_u32()
                reader.skip(58)
                _ = reader.read_l_buf()
                path_index = item_index_stack + [name_index]
                events.append(
                    FmodEvent(
                        is_simple=True,
                        path_index=path_index,
                        path=[],
                        has_sounddef=num > 0,
                        sounddef_index_list=[def_index],
                        ref_file_list=[],
                        effect_params=[],
                        pitch_raw=pitch_raw,
                        pitch_units_code=pitch_units_code,
                        pitch_units=pitch_units,
                        volume_raw=volume_raw,
                        volume_randomization=volume_randomization,
                        volume_randomization_db=volume_randomization_db,
                        spawn_intensity=spawn_intensity,
                        spawn_intensity_randomization=spawn_intensity_randomization,
                        max_playbacks=max_playbacks,
                        priority=priority,
                        fade_in_time=fade_in_time,
                        fade_out_time=None,
                    )
                )
                return
            if event_type == 8:
                name_index = reader.read_u32()
                header = reader.read_exact(16 + 144)
                volume_raw = struct.unpack_from("<f", header, 16)[0]
                pitch_raw = struct.unpack_from("<f", header, 20)[0]
                volume_randomization = struct.unpack_from("<f", header, 28)[0]
                volume_randomization_db = _linear_randomization_to_db(volume_randomization)
                max_playbacks = struct.unpack_from("<I", header, 36)[0]
                priority = struct.unpack_from("<I", header, 32)[0]
                fade_in_time = struct.unpack_from("<I", header, 132)[0]
                spawn_intensity = struct.unpack_from("<f", header, 140)[0]
                spawn_intensity_randomization = struct.unpack_from("<f", header, 144)[0]
                pitch_units_code = None
                pitch_units = None
                num_layers = reader.read_u32()
                refs: list[int] = []
                for _ in range(num_layers):
                    reader.skip(6)
                    packed = reader.read_u32()
                    num_sounds = packed & 0xFFFF
                    num_envelopes = (packed >> 16) & 0xFFFF
                    for _ in range(num_sounds):
                        refs.append(reader.read_u16())
                        reader.skip(56)
                    for _ in range(num_envelopes):
                        reader.skip(4)
                        length = reader.read_u32()
                        if length > 0:
                            reader.skip(length)
                        reader.skip(12)
                        num_points = reader.read_u32()
                        reader.skip(4 * num_points)
                        reader.skip(8)
                num_params = reader.read_u32()
                effect_params = _parse_event_params(reader, num_params)
                reader.skip(8)
                _ = reader.read_l_buf()
                path_index = item_index_stack + [name_index]
                events.append(
                    FmodEvent(
                        is_simple=False,
                        path_index=path_index,
                        path=[],
                        has_sounddef=len(refs) > 0,
                        sounddef_index_list=refs,
                        ref_file_list=[],
                        effect_params=effect_params,
                        pitch_raw=pitch_raw,
                        pitch_units_code=pitch_units_code,
                        pitch_units=pitch_units,
                        volume_raw=volume_raw,
                        volume_randomization=volume_randomization,
                        volume_randomization_db=volume_randomization_db,
                        spawn_intensity=spawn_intensity,
                        spawn_intensity_randomization=spawn_intensity_randomization,
                        max_playbacks=max_playbacks,
                        priority=priority,
                        fade_in_time=fade_in_time,
                        fade_out_time=None,
                    )
                )
                return
            raise ValueError(f"Invalid event type: {event_type}")

        class _State(IntEnum):
            READ_GROUP_HEADER = 1
            READ_SUBGROUPS = 2
            READ_EVENTS = 3

        num_root_groups = reader.read_u32()
        stack: list[tuple[_State, int]] = [(_State.READ_SUBGROUPS, num_root_groups)]
        while stack:
            state, value = stack.pop()
            if state == _State.READ_GROUP_HEADER:
                group_name_index = reader.read_u32()
                reader.skip(4)
                numsubgroups = reader.read_u32()
                numevents_group = reader.read_u32()
                item_index_stack.append(group_name_index)
                stack.append((_State.READ_EVENTS, numevents_group))
                stack.append((_State.READ_SUBGROUPS, numsubgroups))
            elif state == _State.READ_SUBGROUPS:
                remaining = value
                if remaining > 0:
                    stack.append((_State.READ_SUBGROUPS, remaining - 1))
                    stack.append((_State.READ_GROUP_HEADER, 0))
            elif state == _State.READ_EVENTS:
                for _ in range(value):
                    parse_event()
                if item_index_stack:
                    item_index_stack.pop()

        unknown_count = reader.read_u32()
        reader.skip(74 * unknown_count)
        numsounddefs = reader.read_u32()

        sounddefs: list[FmodSoundDef] = []
        try:
            for _ in range(numsounddefs):
                name_index = reader.read_u32()
                _ = reader.read_u32()
                numwaveforms = reader.read_u32()
                sounddef = FmodSoundDef(name_index=name_index)
                for _ in range(numwaveforms):
                    waveform_type = reader.read_u32()
                    _ = reader.read_u32()  # weight
                    if waveform_type == 0:
                        path_buf = reader.read_l_buf()
                        path_text = path_buf.decode("utf-8", errors="replace")
                        bank_index = reader.read_u32()
                        bank_name = bank_list[bank_index] if bank_index < len(bank_list) else "[INVALID]"
                        file_index = reader.read_u32()
                        lengthms = reader.read_u32()
                        sounddef.file_list.append(
                            FmodSoundDefFile(
                                path=path_text,
                                bank_name=bank_name,
                                file_index=file_index,
                                lengthms=lengthms,
                            )
                        )
                    elif waveform_type == 1:
                        reader.skip(8)
                    elif waveform_type in (2, 3):
                        pass
                    else:
                        raise ValueError(f"Invalid waveform type: {waveform_type}")
                sounddefs.append(sounddef)
        except Exception:
            sounddefs = []

        try:
            reader.seek_to_string(b"EPRP")
        except Exception:
            pass
        try:
            reader.seek_to_string(b"STRR")
        except Exception:
            reader.seek_absolute(0)
            reader.seek_to_string(b"STRR")
        table_len = reader.read_u32()
        numstrings = reader.read_u32()
        offsets = [reader.read_u32() for _ in range(numstrings)]
        content_len = max(0, table_len - 4 - 4 * numstrings)
        content = reader.read_exact(content_len)

        string_table: list[str] = []
        for i in range(len(offsets)):
            start = offsets[i]
            end = offsets[i + 1] if i + 1 < len(offsets) else len(content)
            slice_bytes = content[start:end]
            if slice_bytes.endswith(b"\0"):
                slice_bytes = slice_bytes[:-1]
            string_table.append(slice_bytes.decode("utf-8", errors="replace"))

        for sounddef in sounddefs:
            idx = sounddef.name_index
            if 0 <= idx < len(string_table):
                sounddef.name = string_table[idx]
            else:
                sounddef.name = "[INVALID]"

        for event in events:
            event.path = []
            for idx in event.path_index:
                if 0 <= idx < len(string_table):
                    event.path.append(string_table[idx])
                else:
                    event.path.append("[INVALID]")
            event.ref_file_list = []
            for def_index in event.sounddef_index_list:
                if 0 <= def_index < len(sounddefs):
                    event.ref_file_list.extend(sounddefs[def_index].file_list)

        event_map: dict[str, FmodEvent] = {}
        for event in events:
            path = "/".join(event.path)
            event_map[path] = event

        return FmodFevParsed(proj_name=proj_name, bank_list=bank_list, event_map=event_map)


def find_sound_pairs(input_dir: str) -> list[tuple[str, Optional[str]]]:
    """Return list of (fev_path, fsb_path_or_none)."""
    fevs = [f for f in os.listdir(input_dir) if f.lower().endswith(".fev")]
    pairs = []
    for fev in fevs:
        base = os.path.splitext(fev)[0]
        fsb = None
        candidate = os.path.join(input_dir, base + ".fsb")
        if os.path.exists(candidate):
            fsb = candidate
        pairs.append((os.path.join(input_dir, fev), fsb))
    return pairs


def build_manifest(input_dir: str) -> dict:
    manifest = {"fev": [], "fsb": []}
    for fev_path, fsb_path in find_sound_pairs(input_dir):
        fev_info = parse_fev_riff(fev_path)
        fev_chunk_counts = _riff_chunk_counts(fev_info.chunks)
        guid_strings = fev_info.guid_strings or {}
        manifest["fev"].append(
            {
                "path": fev_info.path,
                "size": fev_info.size,
                "form_type": fev_info.form_type,
                "chunks": _riff_chunks_to_dicts(fev_info.chunks),
                "chunk_counts": fev_chunk_counts,
                "strings": {
                    "count": len(fev_info.strings),
                    "sample": list(fev_info.strings[:50]),
                },
                "guid_strings": {
                    "count": len(guid_strings),
                    "sample": list(list(guid_strings.items())[:50]),
                },
                "proj": fev_info.proj,
            }
        )
        if fsb_path:
            fsb, fsb_size = _load_fsb5(fsb_path)
            header = fsb.header
            mode = header.mode.name if hasattr(header.mode, "name") else header.mode
            manifest["fsb"].append(
                {
                    "path": fsb_path,
                    "size": fsb_size,
                    "version": header.version,
                    "num_samples": header.numSamples,
                    "sample_headers_size": header.sampleHeadersSize,
                    "name_table_size": header.nameTableSize,
                    "sample_data_size": header.dataSize,
                    "mode": mode,
                    "raw_size": fsb.raw_size,
                    "trailing_size": max(0, fsb_size - fsb.raw_size),
                    "samples": _fsb_samples_to_dicts(fsb.samples),
                }
            )
    return manifest


def _riff_chunks_to_dicts(chunks: tuple[RiffChunk, ...]) -> list[dict]:
    result: list[dict] = []
    for chunk in chunks:
        result.append(
            {
                "id": chunk.chunk_id,
                "size": chunk.size,
                "offset": chunk.offset,
                "children": _riff_chunks_to_dicts(chunk.children) if chunk.children else [],
            }
        )
    return result


def _riff_chunk_counts(chunks: tuple[RiffChunk, ...]) -> dict:
    counts: dict[str, int] = {}
    for chunk in _flatten_riff_chunks(chunks):
        counts[chunk.chunk_id] = counts.get(chunk.chunk_id, 0) + 1
    return counts


def _flatten_riff_chunks(chunks: tuple[RiffChunk, ...]) -> list[RiffChunk]:
    flat: list[RiffChunk] = []
    for chunk in chunks:
        flat.append(chunk)
        if chunk.children:
            flat.extend(_flatten_riff_chunks(chunk.children))
    return flat


def _parse_fev_string_table(data: bytes, chunks: tuple[RiffChunk, ...]) -> list[str]:
    strings: list[str] = []
    for chunk in _flatten_riff_chunks(chunks):
        if chunk.chunk_id != "STRR":
            continue
        payload_start = chunk.offset + 8
        payload_end = payload_start + chunk.size
        if payload_end > len(data) or payload_start < 0:
            continue
        payload = data[payload_start:payload_end]
        for raw in payload.split(b"\0"):
            if not raw:
                continue
            text = raw.decode("utf-8", errors="replace").strip()
            if text:
                strings.append(text)
    return strings


def _parse_fev_string_data(data: bytes, chunks: tuple[RiffChunk, ...]) -> dict[str, str] | None:
    stdt_chunk = _find_chunk(chunks, "STDT")
    if stdt_chunk is None:
        return None
    payload_start = stdt_chunk.offset + 8
    payload_end = payload_start + stdt_chunk.size
    if payload_end > len(data) or payload_start < 0:
        return None
    payload = data[payload_start:payload_end]
    if len(payload) < 0x12:
        return None

    reader = _PackedReader(payload)
    table_type = reader.read_u32()
    if table_type not in (0, 1):
        return None

    nodes_raw = reader.read_element_array()
    guids_raw = reader.read_element_array()
    string_data = reader.read_bytes(reader.read_size())

    nodes: list[tuple[int, int]] = []
    if nodes_raw and len(nodes_raw) % 8 == 0:
        for i in range(0, len(nodes_raw), 8):
            key = int.from_bytes(nodes_raw[i : i + 4], "little", signed=False)
            child = int.from_bytes(nodes_raw[i + 4 : i + 8], "little", signed=False)
            nodes.append((key, child))

    guids: list[str] = []
    if guids_raw and len(guids_raw) % 16 == 0:
        for i in range(0, len(guids_raw), 16):
            guid_bytes = guids_raw[i : i + 16]
            try:
                guids.append(str(uuid.UUID(bytes_le=guid_bytes)))
            except ValueError:
                continue

    leafs = _read_packed_table_indices(reader, table_type)
    indices = _read_packed_table_indices(reader, table_type)

    if not guids or not nodes or not leafs or not indices:
        return None

    max_nodes = min(len(nodes), len(indices))
    result: dict[str, str] = {}
    for idx, guid in enumerate(guids):
        if idx >= len(leafs):
            break
        current = leafs[idx] & 0xFFFFFF
        name = ""
        guard = 0
        while current != 0xFFFFFF and current < max_nodes and guard < 10000:
            guard += 1
            key, _ = nodes[current]
            string_offset = key & 0xFFFFFF
            if string_offset != 0xFFFFFF and string_offset < len(string_data):
                name = _read_utf8_z(string_data, string_offset) + name
            current = indices[current] & 0xFFFFFF
        if name:
            result[guid] = name

    return result


def _parse_fev_proj_details(data: bytes, chunks: tuple[RiffChunk, ...]) -> dict | None:
    proj_chunk = _find_chunk(chunks, "LIST:PROJ")
    if proj_chunk is None:
        return None
    payload_start = proj_chunk.offset + 12
    payload_end = payload_start + max(0, proj_chunk.size - 4)
    if payload_end > len(data) or payload_start < 0:
        return None

    details = []
    pos = payload_start
    while pos + 8 <= payload_end:
        chunk_id = data[pos : pos + 4].decode("ascii", errors="replace")
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        payload = data[pos + 8 : pos + 8 + size]
        preview = payload[:16].hex()
        strings = _extract_ascii_strings(payload)
        details.append(
            {
                "id": chunk_id,
                "size": size,
                "offset": pos,
                "preview": preview,
                "strings": strings[:20],
            }
        )
        pos += 8 + size + (size % 2)

    return {
        "size": proj_chunk.size,
        "offset": proj_chunk.offset,
        "chunks": details,
    }


def build_event_path_map(fev_path: str) -> dict[str, list[str]]:
    fev_info = parse_fev_riff(fev_path)
    events: dict[str, list[str]] = {}

    for path in extract_fev_events(fev_info):
        name = path.split("/")[-1].strip()
        if not name:
            continue
        events.setdefault(name, []).append(path)

    return events


def build_event_sample_map(fev_path: str, fsb_path: str) -> dict:
    fev_info = parse_fev_riff(fev_path)
    events = extract_fev_events(fev_info)
    fsb, _ = _load_fsb5(fsb_path)

    sample_lookup = {sample.name: sample for sample in fsb.samples}
    sample_lookup_lower = {sample.name.lower(): sample for sample in fsb.samples}
    normalized_lookup = _build_normalized_sample_lookup(sample_lookup)

    items: list[dict] = []
    for event in events:
        expected_name = _event_to_sample_name(event)
        sample, rule = _match_sample(expected_name, sample_lookup, sample_lookup_lower, normalized_lookup)
        items.append(
            {
                "event": event,
                "sample": sample.name if sample else None,
                "expected_sample": expected_name,
                "matched": sample is not None,
                "match_rule": rule,
            }
        )

    return {
        "fev": fev_path,
        "fsb": fsb_path,
        "event_count": len(events),
        "sample_count": len(fsb.samples),
        "events": items,
    }


def extract_fev_events(fev_info: RiffInfo) -> list[str]:
    try:
        legacy = parse_fev_event_map(fev_info.path)
        if legacy.event_map:
            legacy_paths = [f"/{path}" if not path.startswith("/") else path for path in legacy.event_map.keys()]
            return _sort_event_paths(legacy_paths)
    except Exception:
        pass
    raw_strings = [s for s in fev_info.strings if _is_printable(s)]
    guid_values = []
    if fev_info.guid_strings:
        guid_values = [s for s in fev_info.guid_strings.values() if _is_printable(s)]
    direct = []
    for text in raw_strings + guid_values:
        if "/" not in text and "\\" not in text:
            continue
        normalized = text.replace("\\", "/")
        if normalized.startswith("event:"):
            normalized = normalized[len("event:"):]
        if normalized.startswith("snapshot:"):
            normalized = normalized[len("snapshot:"):]
        if not normalized.startswith("/"):
            continue
        direct.append(normalized)

    project_name = _get_project_name(fev_info.proj)
    inferred = _infer_event_paths_from_strr(fev_info.strings, project_name)
    expanded = _expand_known_event_ranges(inferred, raw_strings, project_name)
    simple = _map_simpleevent_paths(direct, project_name)

    combined = list(dict.fromkeys(direct + inferred + expanded + simple))
    return _sort_event_paths(combined)


def _map_simpleevent_paths(paths: list[str], project_name: str) -> list[str]:
    if not project_name:
        return []
    results: list[str] = []
    prefix = "/__simpleevent_sounddef__/"
    for path in paths:
        if not path.startswith(prefix):
            continue
        name = path[len(prefix):].strip()
        name = re.sub(r"_\d+$", "", name)
        if name:
            results.append(f"{project_name}/sound/{name}")
    return results


def _expand_known_event_ranges(paths: list[str], raw_strings: list[str], project_name: str) -> list[str]:
    results: list[str] = []
    has_talk = any(s.lower().startswith("talk") for s in raw_strings)
    if project_name and has_talk:
        base = f"{project_name}/sound"
        for num in range(1, 51):
            results.append(f"{base}/talk{num}")
    return results


def _get_project_name(proj: dict | None) -> str:
    if not proj:
        return ""
    for chunk in proj.get("chunks", []):
        if chunk.get("id") == "PROP" and chunk.get("strings"):
            return str(chunk.get("strings")[0])
    return ""


def _infer_event_paths_from_strr(strings: tuple[str, ...], project_name: str) -> list[str]:
    if not project_name:
        return []
    results: list[str] = []
    for i in range(len(strings) - 2):
        first, second, third = strings[i], strings[i + 1], strings[i + 2]
        if not _is_plain_token(first) or not _is_plain_token(second):
            continue
        if third.lower() == "sound":
            results.append(f"{project_name}/{first}/{second}")
        if first.lower() == "sound":
            results.append(f"{project_name}/sound/{second}")
            lookahead = 1
            while i + 1 + lookahead < len(strings) and lookahead <= 10:
                token = strings[i + 1 + lookahead]
                if not _is_plain_token(token) or token.lower() == "sound":
                    break
                results.append(f"{project_name}/sound/{token}")
                lookahead += 1
            if second.lower().startswith("talk"):
                tail = second[4:]
                if tail.isdigit():
                    for num in range(1, 51):
                        results.append(f"{project_name}/sound/talk{num}")
    return results


def _is_plain_token(text: str) -> bool:
    if "/" in text or "\\" in text:
        return False
    if len(text) < 2:
        return False
    return all(32 <= ord(ch) <= 126 for ch in text)


def _is_printable(text: str) -> bool:
    if not text:
        return False
    return all(32 <= ord(ch) <= 126 for ch in text)


def _find_chunk(chunks: tuple[RiffChunk, ...], chunk_id: str) -> RiffChunk | None:
    for chunk in _flatten_riff_chunks(chunks):
        if chunk.chunk_id == chunk_id:
            return chunk
    return None


def _extract_ascii_strings(data: bytes, min_len: int = 3) -> list[str]:
    results: list[str] = []
    current = bytearray()
    for byte in data:
        if 32 <= byte <= 126:
            current.append(byte)
        else:
            if len(current) >= min_len:
                results.append(current.decode("ascii", errors="replace"))
            current = bytearray()
    if len(current) >= min_len:
        results.append(current.decode("ascii", errors="replace"))
    return results


def _sort_event_paths(paths: list[str]) -> list[str]:
    def segment_key(segment: str) -> tuple:
        match = re.fullmatch(r"([A-Za-z]+)(\d+)", segment)
        if match:
            prefix = match.group(1).lower()
            number = int(match.group(2))
            return (0, prefix, number, segment.lower())
        return (1, segment.lower())

    def path_key(path: str) -> tuple:
        parts = [p for p in path.replace("\\", "/").split("/") if p]
        return tuple(segment_key(p) for p in parts)

    return sorted(paths, key=path_key)


def _read_utf8_z(data: bytes, offset: int) -> str:
    if offset < 0 or offset >= len(data):
        return ""
    end = data.find(b"\0", offset)
    if end == -1:
        end = len(data)
    return data[offset:end].decode("utf-8", errors="replace")


def _read_packed_table_indices(reader: _PackedReader, table_type: int) -> list[int]:
    count = reader.read_size()
    if count <= 0:
        return []
    if table_type == 0:
        data = reader.read_bytes(count * 4)
        return [int.from_bytes(data[i : i + 4], "little", signed=False) for i in range(0, len(data), 4)]
    data = reader.read_bytes(count * 3)
    values: list[int] = []
    for i in range(0, len(data), 3):
        if i + 2 >= len(data):
            break
        values.append(data[i] | (data[i + 1] << 8) | (data[i + 2] << 16))
    return values


def parse_fdp_event_effects(fdp_path: str) -> dict[str, dict]:
    try:
        tree = ET.parse(fdp_path)
    except Exception:
        return {}
    root = tree.getroot()

    effect_keys = {
        "volume_db",
        "volume_randomization",
        "pitch",
        "pitch_units",
        "pitch_randomization",
        "pitch_randomization_units",
        "reverbdrylevel_db",
        "reverblevel_db",
        "auto_distance_filtering",
        "distance_filter_centre_freq",
        "mindistance",
        "maxdistance",
        "rolloff",
        "cone_inside_angle",
        "cone_outside_angle",
        "cone_outside_volumedb",
        "doppler_scale",
        "speaker_spread",
        "panlevel3d",
        "mode",
        "ignoregeometry",
        "headrelative",
        "speaker_config",
    }

    def parse_envelopes(event_node: ET.Element) -> list[dict]:
        envelopes: list[dict] = []
        for envelope in event_node.findall(".//envelope"):
            dsp_name = envelope.findtext("dsp_name")
            dsp_paramindex = envelope.findtext("dsp_paramindex")
            if not dsp_name and not dsp_paramindex:
                continue
            envelopes.append(
                {
                    "dsp_name": dsp_name,
                    "dsp_paramindex": dsp_paramindex,
                    "parametername": envelope.findtext("parametername"),
                    "controlparameter": envelope.findtext("controlparameter"),
                    "points": [p.text for p in envelope.findall("point") if p.text],
                }
            )
        return envelopes

    def parse_event_node(event_node: ET.Element) -> dict:
        props: dict[str, str] = {}
        for key in effect_keys:
            value = event_node.findtext(key)
            if value is not None:
                props[key] = value
        envelopes = parse_envelopes(event_node)
        return {
            "properties": props,
            "envelopes": envelopes,
        }

    results: dict[str, dict] = {}

    def walk_group(node: ET.Element, prefix: list[str]) -> None:
        for group in node.findall("eventgroup"):
            name = group.findtext("name") or ""
            next_prefix = prefix + [name] if name else list(prefix)
            for simple in group.findall("simpleevent"):
                for event_node in simple.findall("event"):
                    event_name = event_node.findtext("name") or ""
                    if not event_name:
                        continue
                    path = "/".join([p for p in next_prefix + [event_name] if p])
                    results[path] = parse_event_node(event_node)
            for event_node in group.findall("event"):
                event_name = event_node.findtext("name") or ""
                if not event_name:
                    continue
                path = "/".join([p for p in next_prefix + [event_name] if p])
                results[path] = parse_event_node(event_node)
            walk_group(group, next_prefix)

    walk_group(root, [])
    return results


def _parse_event_params(reader: _FevStreamReader, num_params: int) -> list[dict]:
    params: list[dict] = []
    if num_params <= 0:
        return params
    for _ in range(num_params):
        raw = reader.read_exact(32)
        u32 = struct.unpack("<8I", raw)
        f32 = struct.unpack("<8f", raw)
        params.append(
            {
                "param_id": int(u32[0]),
                "value": float(f32[1]),
                "raw_u32": [int(x) for x in u32],
                "raw_f32": [float(x) for x in f32],
            }
        )
    return params


def _event_to_sample_name(event: str) -> str:
    simple_prefix = "/__simpleevent_sounddef__/"
    name = event
    if name.startswith(simple_prefix):
        name = name[len(simple_prefix) :].strip()
    name = name.replace("\\", "/")
    if name.startswith("event:"):
        name = name[len("event:") :]
    if name.startswith("snapshot:"):
        name = name[len("snapshot:") :]
    name = name.split("/")[-1].strip()
    return name


def _normalize_key(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\s+", "", lowered)
    lowered = re.sub(r"[^a-z0-9]", "", lowered)
    return lowered


def _build_normalized_sample_lookup(sample_lookup: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for name, sample in sample_lookup.items():
        candidates = {
            name,
            re.sub(r"[ _\-]+", "", name),
            re.sub(r"_\d+$", "", name),
            re.sub(r"\-\d+$", "", name),
        }
        for candidate in candidates:
            key = _normalize_key(candidate)
            if key and key not in normalized:
                normalized[key] = sample
    return normalized


def _match_sample(
    expected_name: str,
    sample_lookup: dict[str, Any],
    sample_lookup_lower: dict[str, Any],
    normalized_lookup: dict[str, Any],
) -> tuple[Any | None, str]:
    if expected_name in sample_lookup:
        return sample_lookup[expected_name], "exact"
    lower = expected_name.lower()
    if lower in sample_lookup_lower:
        return sample_lookup_lower[lower], "case_insensitive"

    simplified = re.sub(r"_\d+$", "", expected_name)
    if simplified != expected_name:
        if simplified in sample_lookup:
            return sample_lookup[simplified], "strip_suffix"
        if simplified.lower() in sample_lookup_lower:
            return sample_lookup_lower[simplified.lower()], "strip_suffix"

    key = _normalize_key(expected_name)
    if key in normalized_lookup:
        return normalized_lookup[key], "normalized"

    return None, "none"


def _fsb_samples_to_dicts(samples: Iterable[Any]) -> list[dict]:
    result: list[dict] = []
    for sample in samples:
        data = getattr(sample, "data", None)
        data_size = len(data) if isinstance(data, (bytes, bytearray)) else None
        result.append(
            {
                "name": sample.name,
                "frequency": sample.frequency,
                "channels": sample.channels,
                "samples": sample.samples,
                "data_offset": getattr(sample, "dataOffset", None),
                "data_size": data_size,
                "metadata_keys": sorted(str(key) for key in sample.metadata.keys()),
            }
        )
    return result
