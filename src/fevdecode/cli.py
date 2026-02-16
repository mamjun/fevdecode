from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime
from typing import Iterable

from .fdp import extract_fdp, parse_fdp
from .fdp_project import build_fdp_project_from_fev
from .mpeg import clean_mpeg_padding
from .parser import (
    build_event_bank_file_map,
    build_event_path_map,
    build_event_sample_map,
    extract_fev_events,
    find_sound_pairs,
    parse_fev_event_map,
    parse_fev_riff,
    read_fev,
    read_fsb,
)


def cmd_extract(args: argparse.Namespace) -> int:
    extract_fdp(args.input, args.output)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    entries = parse_fdp(args.input)
    for entry in entries:
        print(f"{entry.name}\t{entry.size}")
    return 0


def _write_sample(output_dir: str, name: str, ext: str, data: bytes) -> str:
    os.makedirs(output_dir, exist_ok=True)
    safe_name = name.replace("/", "_").replace("\\", "_")
    path = os.path.join(output_dir, f"{safe_name}.{ext}")
    with open(path, "wb") as fp:
        fp.write(data)
    return path


def _sanitize_path_segment(segment: str) -> str:
    segment = segment.strip().replace("/", "_").replace("\\", "_")
    segment = re.sub(r"[<>:\"/\\|?*]", "_", segment)
    return segment or "unnamed"


def _event_to_dir(event_path: str) -> str:
    normalized = event_path.replace("\\", "/").strip()
    if normalized.startswith("event:"):
        normalized = normalized[len("event:"):]
    if normalized.startswith("snapshot:"):
        normalized = normalized[len("snapshot:"):]
    normalized = normalized.strip("/")
    if not normalized:
        return "root"
    parts = [p for p in normalized.split("/") if p]
    return os.path.join(*[_sanitize_path_segment(p) for p in parts])


def _normalize_event_filter(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    if normalized.startswith("event:"):
        normalized = normalized[len("event:"):]
    if normalized.startswith("snapshot:"):
        normalized = normalized[len("snapshot:"):]
    normalized = normalized.strip("/")
    return normalized


def _event_matches_filter(event: dict, filter_prefix: str) -> bool:
    if not filter_prefix:
        return True
    normalized_filter = _normalize_event_filter(filter_prefix)
    if not normalized_filter:
        return True
    prefixes = {normalized_filter}
    if normalized_filter.startswith("sfx/"):
        prefixes.add(normalized_filter[len("sfx/"):])
    else:
        prefixes.add(f"sfx/{normalized_filter}")

    event_path = _normalize_event_filter(event.get("event", ""))
    if event_path and any(event_path.startswith(prefix) for prefix in prefixes):
        return True
    for entry in event.get("files", []):
        file_path = _normalize_event_filter(entry.get("path", ""))
        if file_path and any(file_path.startswith(prefix) for prefix in prefixes):
            return True
    return False


def _ensure_unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    index = 1
    while True:
        candidate = f"{base}_{index}{ext}"
        if not os.path.exists(candidate):
            return candidate
        index += 1


def _get_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    repo_root = _get_repo_root()
    log_dir = os.path.join(repo_root, "log")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")
    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    logger.info("Log started: %s", log_path)
    return logger


def _ensure_console_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            return
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _find_local_ffmpeg() -> str | None:
    repo_root = _get_repo_root()
    candidates = [
        os.path.join(repo_root, "libs", "ffmpeg.exe"),
        os.path.join(repo_root, "libs", "ffmpeg", "ffmpeg.exe"),
        os.path.join(repo_root, "libs", "ffmpeg", "bin", "ffmpeg.exe"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _find_local_ffprobe() -> str | None:
    repo_root = _get_repo_root()
    candidates = [
        os.path.join(repo_root, "libs", "ffprobe.exe"),
        os.path.join(repo_root, "libs", "ffmpeg", "ffprobe.exe"),
        os.path.join(repo_root, "libs", "ffmpeg", "bin", "ffprobe.exe"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _resolve_libs_path(base_libs: str, required_dlls: tuple[str, ...] = ()) -> str:
    arch = "x64" if struct.calcsize("P") * 8 == 64 else "x86"
    arch_libs = os.path.join(base_libs, arch)
    candidates = []
    if os.path.isdir(arch_libs):
        candidates.append(arch_libs)
    if os.path.isdir(base_libs):
        candidates.append(base_libs)
    for candidate in candidates:
        if all(os.path.isfile(os.path.join(candidate, dll)) for dll in required_dlls):
            return candidate
    return candidates[0] if candidates else base_libs


def _load_fsb5_library(path: str):
    try:
        import fsb5
    except Exception as exc:
        raise RuntimeError("python-fsb5 not installed. Run: pip install python-fsb5") from exc
    with open(path, "rb") as fp:
        data = fp.read()
    return fsb5.load(data)


def _fsb5_extension(fsb) -> str:
    try:
        return fsb.get_sample_extension()
    except Exception:
        mode = getattr(getattr(fsb, "header", None), "mode", None)
        return mode.file_extension if mode is not None else "bin"


def _is_vorbis_fsb(fsb) -> bool:
    mode = getattr(getattr(fsb, "header", None), "mode", None)
    name = getattr(mode, "name", None)
    if name:
        return name.upper() == "VORBIS"
    return str(mode).endswith("VORBIS")


def _decode_to_wav(data: bytes, ext: str) -> bytes:
    if ext.lower() == "wav":
        return data
    ext_lower = ext.lower()
    local_ffmpeg = _find_local_ffmpeg()
    ffmpeg_exec = local_ffmpeg or shutil.which("ffmpeg")
    if ffmpeg_exec:
        src_path = None
        dst_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{ext_lower}", delete=False) as src:
                src.write(data)
                src_path = src.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as dst:
                dst_path = dst.name
            command = [
                ffmpeg_exec,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                src_path,
                "-f",
                "wav",
                dst_path,
            ]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0 or not dst_path or not os.path.exists(dst_path):
                message = result.stderr.decode("utf-8", errors="replace").strip()
                if not message:
                    if result.returncode == 3221225781:
                        message = "ffmpeg failed to start (missing DLLs). Ensure full ffmpeg package in libs/ffmpeg"
                    else:
                        message = f"ffmpeg exited with code {result.returncode}"
                raise RuntimeError(f"ffmpeg decode failed: {message}")
            with open(dst_path, "rb") as fp:
                wav_data = fp.read()
            if not wav_data:
                raise RuntimeError("ffmpeg decode failed: empty output")
            return wav_data
        finally:
            if src_path and os.path.exists(src_path):
                os.remove(src_path)
            if dst_path and os.path.exists(dst_path):
                os.remove(dst_path)
    if ext_lower == "mp3":
        raise RuntimeError("ffmpeg not found for mp3 decode. Install ffmpeg or place it under libs/ffmpeg")
    local_ffprobe = _find_local_ffprobe()
    if local_ffmpeg:
        ffmpeg_dir = os.path.dirname(local_ffmpeg)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    try:
        from pydub import AudioSegment
    except Exception as exc:
        raise RuntimeError("pydub not installed. Run: pip install pydub") from exc
    if local_ffmpeg:
        AudioSegment.converter = local_ffmpeg
    if local_ffprobe:
        AudioSegment.ffprobe = local_ffprobe
    audio = AudioSegment.from_file(io.BytesIO(data), format=ext_lower)
    out = io.BytesIO()
    audio.export(out, format="wav")
    return out.getvalue()


def _trim_wav_to_samples(wav_data: bytes, target_frames: int | None) -> bytes:
    if not isinstance(target_frames, int) or target_frames <= 0:
        return wav_data
    try:
        with wave.open(io.BytesIO(wav_data), "rb") as reader:
            params = reader.getparams()
            total_frames = reader.getnframes()
            if target_frames >= total_frames:
                return wav_data
            frames = reader.readframes(target_frames)
        out = io.BytesIO()
        with wave.open(out, "wb") as writer:
            writer.setparams(params)
            writer.writeframes(frames)
        return out.getvalue()
    except Exception:
        return wav_data


def cmd_extract_fsb(args: argparse.Namespace) -> int:
    fsb = _load_fsb5_library(args.input)
    ext = _fsb5_extension(fsb)
    written = 0
    for sample in fsb.samples:
        data = fsb.rebuild_sample(sample)
        _write_sample(args.output, sample.name, ext, data)
        written += 1
    print(f"Extracted {written} samples to {args.output}")
    return 0


def cmd_extract_fsb_audio(args: argparse.Namespace) -> int:
    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    base_libs = os.path.abspath(os.path.join(os.getcwd(), "libs"))
    libs_path = _resolve_libs_path(base_libs, ("libvorbis.dll", "libogg.dll"))
    original_cwd = os.getcwd()
    if os.path.isdir(libs_path):
        os.environ["PATH"] = libs_path + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(libs_path)
        except AttributeError:
            pass
        os.chdir(libs_path)
    fsb = _load_fsb5_library(input_path)
    is_vorbis = _is_vorbis_fsb(fsb)
    ext = "ogg" if is_vorbis else _fsb5_extension(fsb)
    written = 0
    failed = 0
    missing_crc: list[int] = []
    try:
        for sample in fsb.samples:
            try:
                data = fsb.rebuild_sample(sample)
                if ext == "mp3" and not args.no_fix_mp3:
                    data = clean_mpeg_padding(data)
                if not is_vorbis:
                    data = _decode_to_wav(data, ext)
                    output_ext = "wav"
                else:
                    output_ext = ext
            except Exception as exc:
                failed += 1
                match = re.search(r"crc32=(\d+)", str(exc))
                if match:
                    missing_crc.append(int(match.group(1)))
                continue
            _write_sample(output_dir, sample.name, output_ext, data)
            written += 1
    finally:
        os.chdir(original_cwd)
    if missing_crc:
        missing_path = os.path.join(output_dir, "missing_vorbis_headers.json")
        with open(missing_path, "w", encoding="utf-8") as fp:
            json.dump(sorted(set(missing_crc)), fp, indent=2)
        print(f"Missing Vorbis headers: {len(set(missing_crc))} (saved to {missing_path})")
    if failed:
        print(f"Extracted {written} samples to {args.output} (skipped {failed})")
    else:
        print(f"Extracted {written} samples to {args.output}")
    return 0


def _resolve_fsb_for_fev(fev_path: str) -> str:
    directory = os.path.dirname(fev_path)
    base = os.path.splitext(os.path.basename(fev_path))[0]
    candidate = os.path.join(directory, base + ".fsb")
    if os.path.exists(candidate):
        return candidate
    fsb_files = [f for f in os.listdir(directory) if f.lower().endswith(".fsb")]
    if len(fsb_files) == 1:
        return os.path.join(directory, fsb_files[0])
    raise FileNotFoundError("Could not resolve associated .fsb file")


def _resolve_fsb_map_for_fev(fev_path: str) -> dict[str, str]:
    parsed = parse_fev_event_map(fev_path)
    directory = os.path.dirname(fev_path)
    fsb_map: dict[str, str] = {}
    for bank_name in parsed.bank_list:
        if not bank_name:
            continue
        candidates: list[str] = []
        if bank_name.lower().endswith(".fsb"):
            candidates.append(bank_name)
        else:
            candidates.append(f"{bank_name}.fsb")
        for candidate_name in candidates:
            candidate = os.path.join(directory, candidate_name)
            if os.path.isfile(candidate):
                fsb_map[bank_name] = candidate
                break
    return fsb_map


def _resolve_fdp_for_fev(fev_path: str) -> str | None:
    directory = os.path.dirname(fev_path)
    base = os.path.splitext(os.path.basename(fev_path))[0]
    candidates = [
        os.path.join(directory, base + ".fdp"),
        os.path.join(_get_repo_root(), "fdp", base + ".fdp"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def cmd_extract_events(args: argparse.Namespace) -> int:
    fev_path = os.path.abspath(args.fev)
    fsb_path = os.path.abspath(args.fsb) if args.fsb else _resolve_fsb_for_fev(fev_path)
    output_root = os.path.abspath(args.output)

    libs_base = os.path.abspath(os.path.join(os.getcwd(), "libs"))
    libs_path = _resolve_libs_path(libs_base, ("libvorbis.dll", "libogg.dll"))
    original_cwd = os.getcwd()
    if os.path.isdir(libs_path):
        os.environ["PATH"] = libs_path + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(libs_path)
        except AttributeError:
            pass
        os.chdir(libs_path)
    fsb = _load_fsb5_library(fsb_path)
    ext = _fsb5_extension(fsb)
    event_map = build_event_path_map(fev_path)
    fev_info = parse_fev_riff(fev_path)
    events = extract_fev_events(fev_info)
    sample_lookup = {sample.name: sample for sample in fsb.samples}
    sample_lookup_lower = {sample.name.lower(): sample for sample in fsb.samples}
    sample_index = {sample.name: idx for idx, sample in enumerate(fsb.samples)}
    sample_index_lower = {sample.name.lower(): idx for idx, sample in enumerate(fsb.samples)}

    events_out: list[dict] = []
    def event_to_sample_name(event: str) -> str:
        simple_prefix = "/__simpleevent_sounddef__/"
        if event.startswith(simple_prefix):
            name = event[len(simple_prefix):].strip()
            return re.sub(r"_\d+$", "", name)
        return event.split("/")[-1].strip()

    for path in events:
        sample_name = event_to_sample_name(path)
        sample = sample_lookup.get(sample_name) or sample_lookup_lower.get(sample_name.lower())
        idx = sample_index.get(sample_name)
        if idx is None:
            idx = sample_index_lower.get(sample_name.lower())
        if sample is None or idx is None:
            events_out.append({
                "event": path,
                "sample": sample_name,
                "status": "missing_sample",
            })
            continue
        try:
            data = fsb.rebuild_sample(sample)
            if ext == "mp3" and not args.no_fix_mp3:
                data = clean_mpeg_padding(data)
        except Exception as exc:
            events_out.append({
                "event": path,
                "sample": sample.name,
                "status": "error",
                "error": str(exc),
            })
            continue
        relative = path.replace("\\", "/").strip("/")
        output_path = os.path.join(output_root, os.path.splitext(os.path.basename(fev_path))[0], relative + f".{ext}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as out:
            out.write(data)
        events_out.append({
            "event": path,
            "sample": sample.name,
            "status": "written",
            "output": output_path,
        })

    os.chdir(original_cwd)
    report = {
        "fev": fev_path,
        "fsb": fsb_path,
        "extension": ext,
        "events": events_out,
        "event_count": len(events),
    }
    report_path = os.path.join(output_root, os.path.splitext(os.path.basename(fev_path))[0], "event_map.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print(f"Wrote {len(events_out)} entries to {report_path}")
    return 0


def cmd_event_map(args: argparse.Namespace) -> int:
    fev_path = os.path.abspath(args.fev)
    fsb_path = os.path.abspath(args.fsb) if args.fsb else _resolve_fsb_for_fev(fev_path)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mapping = build_event_sample_map(fev_path, fsb_path)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(mapping, fp, ensure_ascii=False, indent=2)
    print(f"Wrote {len(mapping.get('events', []))} entries to {output_path}")
    return 0


def cmd_effect_param_report(args: argparse.Namespace) -> int:
    fev_path = os.path.abspath(args.fev)
    output_path = os.path.abspath(args.output)
    parsed = parse_fev_event_map(fev_path)

    stats: dict[int, dict] = {}
    for path, event in parsed.event_map.items():
        for param in event.effect_params:
            param_id = int(param.get("param_id", -1))
            value = param.get("value")
            if param_id < 0 or not isinstance(value, (int, float)):
                continue
            entry = stats.setdefault(
                param_id,
                {
                    "param_id": param_id,
                    "count": 0,
                    "min": value,
                    "max": value,
                    "sum": 0.0,
                    "sample_values": [],
                    "sample_events": [],
                },
            )
            entry["count"] += 1
            entry["sum"] += float(value)
            entry["min"] = min(entry["min"], value)
            entry["max"] = max(entry["max"], value)
            if len(entry["sample_values"]) < 5:
                entry["sample_values"].append(value)
            if len(entry["sample_events"]) < 5:
                entry["sample_events"].append(path)

    params = []
    for item in stats.values():
        avg = item["sum"] / item["count"] if item["count"] else 0.0
        params.append(
            {
                "param_id": item["param_id"],
                "count": item["count"],
                "min": item["min"],
                "max": item["max"],
                "avg": avg,
                "sample_values": item["sample_values"],
                "sample_events": item["sample_events"],
            }
        )

    params.sort(key=lambda x: (-x["count"], x["param_id"]))

    report = {
        "fev": fev_path,
        "event_count": len(parsed.event_map),
        "param_count": len(params),
        "params": params,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print(f"Wrote {len(params)} params to {output_path}")
    return 0


def _load_event_bank_audio_map(json_path: str) -> dict[str, list[str]]:
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"event_bank_files.json not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    sample_map: dict[str, list[str]] = {}
    for event in data.get("events", []) or []:
        for entry in event.get("files", []) or []:
            sample_name = entry.get("sample_name")
            audio_output = entry.get("audio_output")
            if not sample_name or not audio_output:
                continue
            sample_map.setdefault(sample_name, []).append(audio_output)
    return sample_map


def _read_wav_info(path: str) -> dict[str, int | float] | None:
    if not os.path.isfile(path):
        return None
    try:
        with wave.open(path, "rb") as reader:
            rate = int(reader.getframerate())
            frames = int(reader.getnframes())
            channels = int(reader.getnchannels())
        duration = frames / rate if rate else 0.0
        return {
            "rate": rate,
            "frames": frames,
            "channels": channels,
            "duration": duration,
        }
    except Exception:
        return None


def _resolve_sample_compare_outputs(output_arg: str | None, event_bank_json: str, fsb_path: str) -> tuple[str, str]:
    base_name = os.path.splitext(os.path.basename(fsb_path))[0]
    default_dir = os.path.dirname(event_bank_json)
    default_csv = os.path.join(default_dir, f"{base_name}_sample_compare.csv")
    if not output_arg:
        csv_path = default_csv
    else:
        output_arg = os.path.abspath(output_arg)
        if output_arg.lower().endswith(".csv"):
            csv_path = output_arg
        else:
            csv_path = os.path.join(output_arg, f"{base_name}_sample_compare.csv")
    json_path = os.path.splitext(csv_path)[0] + ".summary.json"
    return csv_path, json_path


def cmd_sample_compare(args: argparse.Namespace) -> int:
    fsb_path = os.path.abspath(args.fsb)
    event_bank_json = os.path.abspath(args.event_bank_json)
    rate_tol = float(args.rate_tolerance)
    duration_tol = float(args.duration_tolerance)

    fsb = _load_fsb5_library(fsb_path)
    sample_audio_map = _load_event_bank_audio_map(event_bank_json)
    csv_path, json_path = _resolve_sample_compare_outputs(args.output, event_bank_json, fsb_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None

    total = 0
    missing_audio = 0
    missing_wav = 0
    rate_mismatch = 0
    duration_mismatch = 0
    rows: list[dict[str, object]] = []

    for sample in fsb.samples:
        total += 1
        name = getattr(sample, "name", "")
        freq = getattr(sample, "frequency", None)
        samples = getattr(sample, "samples", None)
        channels = getattr(sample, "channels", None)
        expected_duration = None
        if isinstance(freq, (int, float)) and freq:
            if isinstance(samples, (int, float)):
                expected_duration = float(samples) / float(freq)

        candidates = sample_audio_map.get(name) or []
        wav_info = None
        wav_path = None
        for candidate in candidates:
            info = _read_wav_info(candidate)
            if info:
                wav_info = info
                wav_path = candidate
                break

        if not candidates:
            missing_audio += 1
        if candidates and not wav_info:
            missing_wav += 1

        wav_rate = wav_info["rate"] if wav_info else None
        wav_frames = wav_info["frames"] if wav_info else None
        wav_duration = wav_info["duration"] if wav_info else None
        wav_channels = wav_info["channels"] if wav_info else None

        rate_ratio = None
        duration_ratio = None
        notes = []
        if isinstance(freq, (int, float)) and freq and isinstance(wav_rate, int) and wav_rate:
            rate_ratio = wav_rate / float(freq)
            if abs(rate_ratio - 1.0) > rate_tol:
                rate_mismatch += 1
                notes.append("rate_mismatch")
        if expected_duration and isinstance(wav_duration, float) and wav_duration:
            duration_ratio = wav_duration / expected_duration
            if abs(duration_ratio - 1.0) > duration_tol:
                duration_mismatch += 1
                notes.append("duration_mismatch")

        rows.append(
            {
                "sample_name": name,
                "fsb_frequency": freq,
                "fsb_samples": samples,
                "fsb_channels": channels,
                "expected_duration_sec": expected_duration,
                "wav_path": wav_path,
                "wav_rate": wav_rate,
                "wav_frames": wav_frames,
                "wav_channels": wav_channels,
                "wav_duration_sec": wav_duration,
                "rate_ratio": rate_ratio,
                "duration_ratio": duration_ratio,
                "notes": ";".join(notes),
            }
        )

    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "sample_name",
                "fsb_frequency",
                "fsb_samples",
                "fsb_channels",
                "expected_duration_sec",
                "wav_path",
                "wav_rate",
                "wav_frames",
                "wav_channels",
                "wav_duration_sec",
                "rate_ratio",
                "duration_ratio",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "fsb": fsb_path,
        "event_bank_json": event_bank_json,
        "total_samples": total,
        "missing_audio_output": missing_audio,
        "missing_wav_files": missing_wav,
        "rate_mismatch": rate_mismatch,
        "duration_mismatch": duration_mismatch,
        "rate_tolerance": rate_tol,
        "duration_tolerance": duration_tol,
        "csv": csv_path,
    }
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(f"Wrote sample compare CSV: {csv_path}")
    print(f"Wrote summary JSON: {json_path}")
    print(
        "Summary: total=%s missing_audio=%s missing_wav=%s rate_mismatch=%s duration_mismatch=%s"
        % (total, missing_audio, missing_wav, rate_mismatch, duration_mismatch)
    )
    return 0


def cmd_event_bank_files(args: argparse.Namespace) -> int:
    logger = _get_logger("event_bank_files")
    started = time.time()
    fev_path = os.path.abspath(args.fev)
    fdp_path = os.path.abspath(args.fdp) if getattr(args, "fdp", None) else _resolve_fdp_for_fev(fev_path)
    fsb_map = getattr(args, "fsb_map", None)
    if fsb_map:
        fsb_map = {bank: os.path.abspath(path) for bank, path in fsb_map.items()}
        fsb_path = None
    else:
        fsb_path = os.path.abspath(args.fsb) if args.fsb else _resolve_fsb_for_fev(fev_path)
        fsb_map = None
    fev_name = os.path.splitext(os.path.basename(fev_path))[0]
    output_base = args.output_dir or os.path.join(os.getcwd(), "build")
    output_root = os.path.abspath(os.path.join(output_base, fev_name))
    logger.info("Start event_bank_files: fev=%s fsb=%s output_root=%s", fev_path, fsb_path or fsb_map, output_root)
    if args.output:
        output_name = os.path.basename(args.output)
        if not output_name.lower().endswith(".json"):
            output_name += ".json"
    else:
        output_name = f"{fev_name}_event_bank_files.json"
    output_path = os.path.join(output_root, output_name)
    if args.audio_root:
        audio_name = os.path.basename(args.audio_root.rstrip("/\\"))
        if not audio_name:
            audio_name = "audio"
    else:
        audio_name = "audio"
    audio_root = os.path.join(output_root, audio_name)

    mapping = build_event_bank_file_map(fev_path, fsb_path, fsb_map, fdp_path)
    total_events = len(mapping.get("events", []))
    logger.info("Parsed event map: %s events", total_events)
    filter_prefix = getattr(args, "event_path", None)
    if filter_prefix:
        filtered_events = [event for event in mapping.get("events", []) if _event_matches_filter(event, filter_prefix)]
        mapping["events"] = filtered_events
        mapping["event_count"] = len(filtered_events)
        mapping["event_filter"] = filter_prefix
        logger.info("Filtered events: %s -> %s by prefix=%s", total_events, len(filtered_events), filter_prefix)
    logger.info("Audio root: %s", audio_root)

    libs_base = os.path.abspath(os.path.join(os.getcwd(), "libs"))
    libs_path = _resolve_libs_path(libs_base, ("libvorbis.dll", "libogg.dll"))
    original_cwd = os.getcwd()
    if os.path.isdir(libs_path):
        os.environ["PATH"] = libs_path + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(libs_path)
        except AttributeError:
            pass
        os.chdir(libs_path)
    written = 0
    failed = 0
    missing_crc: list[int] = []
    missing_samples = 0
    missing_fsb = 0

    jobs = max(1, int(getattr(args, "jobs", 1) or 1))
    if fsb_map:
        available_banks = set(fsb_map.keys())
        for event in mapping.get("events", []):
            for entry in event.get("files", []):
                bank_name = entry.get("bank_name")
                if bank_name and bank_name not in available_banks:
                    entry["audio_status"] = "missing_fsb"
                    entry["audio_error"] = f"missing_fsb:{bank_name}"
                    missing_fsb += 1

    fsb_targets: dict[str, str]
    if fsb_map:
        fsb_targets = fsb_map
    else:
        fsb_targets = {"*": fsb_path}

    for bank_name, bank_fsb_path in fsb_targets.items():
        if not bank_fsb_path or not os.path.isfile(bank_fsb_path):
            continue
        fsb = _load_fsb5_library(bank_fsb_path)
        logger.info("Parsed FSB samples: %s (%s)", len(fsb.samples), bank_fsb_path)
        is_vorbis = _is_vorbis_fsb(fsb)
        ext = "ogg" if is_vorbis else _fsb5_extension(fsb)
        logger.info("Decode mode: ffmpeg (jobs=%s)", jobs)
        decoder_name = "ffmpeg" if ext != "wav" else "passthrough"

        def _decode_entry(file_index: int) -> bytes:
            sample = fsb.samples[file_index]
            data = fsb.rebuild_sample(sample)
            if ext == "mp3" and not args.no_fix_mp3:
                data = clean_mpeg_padding(data)
            wav_data = _decode_to_wav(data, ext)
            target_frames = getattr(sample, "samples", None)
            return _trim_wav_to_samples(wav_data, int(target_frames) if target_frames else None)
        tasks: list[tuple[dict, str, int]] = []
        for event in mapping.get("events", []):
            event_path = event.get("event", "")
            event_dir = os.path.join(audio_root, _event_to_dir(event_path))
            logger.info("Event: %s -> %s", event_path, event_dir)
            files = event.get("files", [])
            for entry in files:
                if bank_name != "*" and entry.get("bank_name") != bank_name:
                    continue
                file_index = entry.get("file_index")
                if not isinstance(file_index, int) or not (0 <= file_index < len(fsb.samples)):
                    entry["audio_status"] = "missing_sample"
                    missing_samples += 1
                    continue
                tasks.append((entry, event_dir, file_index))

        def _write_audio(entry: dict, event_dir: str, file_index: int, wav_data: bytes) -> None:
            base_name = entry.get("sample_name") or os.path.splitext(os.path.basename(entry.get("path", "")))[0]
            if not base_name:
                base_name = f"file_{file_index}"
            safe_name = _sanitize_path_segment(base_name)
            output_ext = "wav"
            output_file = os.path.join(event_dir, f"{safe_name}.{output_ext}")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "wb") as out:
                out.write(wav_data)
            entry["audio_status"] = "written"
            entry["audio_output"] = output_file
            entry["audio_decoder"] = decoder_name
            logger.info("Wrote audio: %s", output_file)
            logger.info("Audio decode: %s -> %s", entry["audio_decoder"], output_file)

        if jobs > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
                future_map = {
                    executor.submit(_decode_entry, file_index): (entry, event_dir, file_index)
                    for entry, event_dir, file_index in tasks
                }
                for future in concurrent.futures.as_completed(future_map):
                    entry, event_dir, file_index = future_map[future]
                    try:
                        wav_data = future.result()
                    except Exception as exc:
                        failed += 1
                        match = re.search(r"crc32=(\d+)", str(exc))
                        if match:
                            missing_crc.append(int(match.group(1)))
                        entry["audio_status"] = "error"
                        entry["audio_error"] = str(exc)
                        logger.warning("Decode failed: index=%s error=%s", file_index, exc)
                        continue
                    _write_audio(entry, event_dir, file_index, wav_data)
                    written += 1
        else:
            for entry, event_dir, file_index in tasks:
                try:
                    wav_data = _decode_entry(file_index)
                except Exception as exc:
                    failed += 1
                    match = re.search(r"crc32=(\d+)", str(exc))
                    if match:
                        missing_crc.append(int(match.group(1)))
                    entry["audio_status"] = "error"
                    entry["audio_error"] = str(exc)
                    logger.warning("Decode failed: index=%s error=%s", file_index, exc)
                    continue
                _write_audio(entry, event_dir, file_index, wav_data)
                written += 1

    os.chdir(original_cwd)

    if missing_crc:
        missing_path = os.path.join(audio_root, "missing_vorbis_headers.json")
        os.makedirs(audio_root, exist_ok=True)
        with open(missing_path, "w", encoding="utf-8") as fp:
            json.dump(sorted(set(missing_crc)), fp, indent=2)
        mapping["missing_vorbis_headers"] = missing_path

    mapping["audio_root"] = audio_root
    mapping["audio_written"] = written
    mapping["audio_failed"] = failed
    mapping["audio_missing_samples"] = missing_samples
    if missing_fsb:
        mapping["audio_missing_fsb"] = missing_fsb

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(mapping, fp, ensure_ascii=False, indent=2)
    print(f"Wrote {len(mapping.get('events', []))} entries to {output_path}")
    print(f"Exported {written} files to {audio_root}")
    if failed or missing_samples:
        print(f"Skipped {failed} errors, {missing_samples} missing samples")
    logger.info("Done event_bank_files in %.2fs", time.time() - started)
    return 0


def cmd_gen_fdp(args: argparse.Namespace) -> int:
    logger = _get_logger("gen_fdp")
    started = time.time()
    logger.info("Start gen_fdp: fev=%s template=%s output_dir=%s", args.fev, args.template, args.output_dir)
    output_path = build_fdp_project_from_fev(args.fev, args.template, args.output_dir)
    print(f"Wrote {output_path}")
    logger.info("Generated fdp: %s", output_path)
    logger.info("Done gen_fdp in %.2fs", time.time() - started)
    return 0


def cmd_gen_all(args: argparse.Namespace) -> int:
    logger = _get_logger("gen_all")
    _ensure_console_logger(logger)
    started = time.time()
    if not args.name and not args.fev:
        raise ValueError("Provide --name or --fev")
    if args.name and args.fev:
        raise ValueError("Use either --name or --fev, not both")
    if args.name:
        if args.name.lower().endswith(".fev"):
            args.name = os.path.splitext(args.name)[0]
        sound_dir = os.path.abspath(args.sound_dir or os.path.join(os.getcwd(), "sound"))
        fev_path = os.path.join(sound_dir, f"{args.name}.fev")
        args.fev = fev_path
    if args.fsb:
        fsb_path = os.path.abspath(args.fsb)
        fsb_map = None
    else:
        fsb_path = None
        fsb_map = _resolve_fsb_map_for_fev(os.path.abspath(args.fev))
        if not fsb_map:
            try:
                fsb_path = _resolve_fsb_for_fev(os.path.abspath(args.fev))
            except Exception:
                fsb_path = None
    output_base = args.output_dir or os.path.join(os.getcwd(), "build")
    fev_name = args.name or os.path.splitext(os.path.basename(args.fev))[0]
    output_root = os.path.abspath(os.path.join(output_base, fev_name))
    if os.path.isdir(output_root):
        shutil.rmtree(output_root)
        logger.info("Removed output directory: %s", output_root)
    logger.info("Start gen_all: fev=%s fsb=%s output_dir=%s", args.fev, fsb_path or fsb_map, args.output_dir)
    if getattr(args, "jobs", None) is None:
        args.jobs = 16
    event_args = argparse.Namespace(
        fev=args.fev,
        fsb=fsb_path,
        fsb_map=fsb_map,
        output=args.output,
        audio_root=args.audio_root,
        no_fix_mp3=args.no_fix_mp3,
        jobs=args.jobs,
        output_dir=args.output_dir,
        fdp=getattr(args, "fdp", None),
        event_path=getattr(args, "event_path", None),
    )
    try:
        has_fsb = bool(fsb_map) or (fsb_path and os.path.isfile(fsb_path))
        if not has_fsb:
            raise FileNotFoundError("FSB not found")
        cmd_event_bank_files(event_args)
        logger.info("event_bank_files step complete")
    except Exception as exc:
        logger.warning("event_bank_files skipped: %s", exc)

    pitch_units_fdp = getattr(args, "fdp", None) or _resolve_fdp_for_fev(args.fev)
    output_path = build_fdp_project_from_fev(args.fev, args.template, args.output_dir, pitch_units_fdp)
    print(f"Wrote {output_path}")
    logger.info("Generated fdp: %s", output_path)
    logger.info("Done gen_all in %.2fs", time.time() - started)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fevdecode")
    sub = parser.add_subparsers(dest="command", required=True)

    extract_cmd = sub.add_parser("extract", help="extract fdp to directory")
    extract_cmd.add_argument("--input", required=True, help="input .fdp file")
    extract_cmd.add_argument("--output", required=True, help="output directory")
    extract_cmd.set_defaults(func=cmd_extract)

    list_cmd = sub.add_parser("list", help="list entries in fdp")
    list_cmd.add_argument("--input", required=True, help="input .fdp file")
    list_cmd.set_defaults(func=cmd_list)

    extract_fsb_cmd = sub.add_parser("extract-fsb", help="extract raw samples from .fsb")
    extract_fsb_cmd.add_argument("--input", required=True, help="input .fsb file")
    extract_fsb_cmd.add_argument("--output", required=True, help="output directory")
    extract_fsb_cmd.set_defaults(func=cmd_extract_fsb)

    extract_audio_cmd = sub.add_parser("extract-fsb-audio", help="extract decoded audio samples from .fsb")
    extract_audio_cmd.add_argument("--input", required=True, help="input .fsb file")
    extract_audio_cmd.add_argument("--output", required=True, help="output directory")
    extract_audio_cmd.add_argument("--no-fix-mp3", action="store_true", help="do not remove mp3 inter-frame padding")
    extract_audio_cmd.set_defaults(func=cmd_extract_fsb_audio)

    events_cmd = sub.add_parser("extract-events", help="extract audio by FEV event paths")
    events_cmd.add_argument("--fev", required=True, help="input .fev file")
    events_cmd.add_argument("--fsb", required=False, help="input .fsb file (optional)")
    events_cmd.add_argument("--output", required=True, help="output root directory")
    events_cmd.add_argument("--no-fix-mp3", action="store_true", help="do not remove mp3 inter-frame padding")
    events_cmd.set_defaults(func=cmd_extract_events)

    event_map_cmd = sub.add_parser("event-map", help="write event-to-sample mapping json")
    event_map_cmd.add_argument("--fev", required=True, help="input .fev file")
    event_map_cmd.add_argument("--fsb", required=False, help="input .fsb file (optional)")
    event_map_cmd.add_argument("--output", required=True, help="output json path")
    event_map_cmd.set_defaults(func=cmd_event_map)

    effect_param_cmd = sub.add_parser("effect-param-report", help="summarize event effect params")
    effect_param_cmd.add_argument("--fev", required=True, help="input .fev file")
    effect_param_cmd.add_argument("--output", required=True, help="output json path")
    effect_param_cmd.set_defaults(func=cmd_effect_param_report)

    sample_compare_cmd = sub.add_parser("sample-compare", help="compare FSB sample metadata with exported WAVs")
    sample_compare_cmd.add_argument("--fsb", required=True, help="input .fsb file")
    sample_compare_cmd.add_argument(
        "--event-bank-json",
        required=True,
        help="event_bank_files.json generated by event-bank-files",
    )
    sample_compare_cmd.add_argument(
        "--output",
        required=False,
        help="output CSV path or directory (default: alongside event_bank_files.json)",
    )
    sample_compare_cmd.add_argument(
        "--rate-tolerance",
        type=float,
        default=0.01,
        help="allowed ratio drift between wav_rate and fsb_frequency (default: 0.01)",
    )
    sample_compare_cmd.add_argument(
        "--duration-tolerance",
        type=float,
        default=0.01,
        help="allowed ratio drift between wav_duration and expected duration (default: 0.01)",
    )
    sample_compare_cmd.set_defaults(func=cmd_sample_compare)

    bank_files_cmd = sub.add_parser("event-bank-files", help="write event-bank files json and export wav audio")
    bank_files_cmd.add_argument("--fev", required=True, help="input .fev file")
    bank_files_cmd.add_argument("--fsb", required=False, help="input .fsb file (optional)")
    bank_files_cmd.add_argument("--output", required=False, help="output json file name (always placed under build/{fev_name})")
    bank_files_cmd.add_argument("--audio-root", required=False, help="audio subdirectory name (always placed under build/{fev_name})")
    bank_files_cmd.add_argument("--output-dir", required=False, help="output base directory (default: build; output is {base}/{fev_name})")
    bank_files_cmd.add_argument("--fdp", required=False, help="input .fdp file (optional, for effect params)")
    bank_files_cmd.add_argument(
        "--event-path",
        required=False,
        help="only export events under this path prefix (e.g. sfx/creatures/boss)",
    )
    bank_files_cmd.add_argument("--no-fix-mp3", action="store_true", help="do not remove mp3 inter-frame padding")
    bank_files_cmd.add_argument("--jobs", type=int, default=16, help="number of worker threads for ffmpeg decoding")
    bank_files_cmd.set_defaults(func=cmd_event_bank_files)

    gen_fdp_cmd = sub.add_parser("gen-fdp", help="generate FMOD .fdp project from .fev")
    gen_fdp_cmd.add_argument("--fev", required=True, help="input .fev file")
    gen_fdp_cmd.add_argument(
        "--template",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "temp", "basic.fdp"),
        help="template .fdp file (default: temp/basic.fdp)",
    )
    gen_fdp_cmd.add_argument(
        "--output-dir",
        required=False,
        help="output base directory (default: build; output is {base}/{fev_name})",
    )
    gen_fdp_cmd.set_defaults(func=cmd_gen_fdp)

    gen_all_cmd = sub.add_parser("gen-all", help="generate event_bank_files and .fdp in one step")
    gen_all_cmd.add_argument("--fev", required=False, help="input .fev file")
    gen_all_cmd.add_argument("--name", required=False, help="project name (looks for sound/{name}.fev/.fsb)")
    gen_all_cmd.add_argument("--fsb", required=False, help="input .fsb file (optional)")
    gen_all_cmd.add_argument("--sound-dir", required=False, help="sound directory (default: ./sound)")
    gen_all_cmd.add_argument("--output", required=False, help="event_bank_files.json output name (placed under build/{fev_name})")
    gen_all_cmd.add_argument("--audio-root", required=False, help="audio subdirectory name (placed under build/{fev_name})")
    gen_all_cmd.add_argument(
        "--event-path",
        required=False,
        help="only export events under this path prefix (e.g. sfx/creatures/boss)",
    )
    gen_all_cmd.add_argument("--no-fix-mp3", action="store_true", help="do not remove mp3 inter-frame padding")
    gen_all_cmd.add_argument("--jobs", type=int, default=16, help="number of worker threads for ffmpeg decoding")
    gen_all_cmd.add_argument("--fdp", required=False, help="input .fdp file (optional, for effect params)")
    gen_all_cmd.add_argument(
        "--template",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "temp", "basic.fdp"),
        help="template .fdp file (default: temp/basic.fdp)",
    )
    gen_all_cmd.add_argument(
        "--output-dir",
        required=False,
        help="output base directory (default: build; output is {base}/{fev_name})",
    )
    gen_all_cmd.set_defaults(func=cmd_gen_all)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
