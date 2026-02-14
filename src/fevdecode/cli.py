from __future__ import annotations

import argparse
import concurrent.futures
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


def cmd_event_bank_files(args: argparse.Namespace) -> int:
    logger = _get_logger("event_bank_files")
    started = time.time()
    fev_path = os.path.abspath(args.fev)
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

    mapping = build_event_bank_file_map(fev_path, fsb_path, fsb_map)
    logger.info("Parsed event map: %s events", len(mapping.get("events", [])))
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
        if is_vorbis:
            logger.info("Decode mode: python-fsb5 (ogg)")
        else:
            logger.info("Decode mode: ffmpeg (jobs=%s)", jobs)

        def _decode_entry(file_index: int) -> bytes:
            sample = fsb.samples[file_index]
            data = fsb.rebuild_sample(sample)
            if ext == "mp3" and not args.no_fix_mp3:
                data = clean_mpeg_padding(data)
            if is_vorbis:
                return data
            return _decode_to_wav(data, ext)
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
            output_ext = "ogg" if is_vorbis else "wav"
            output_file = os.path.join(event_dir, f"{safe_name}.{output_ext}")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "wb") as out:
                out.write(wav_data)
            entry["audio_status"] = "written"
            entry["audio_output"] = output_file
            entry["audio_decoder"] = "python-fsb5" if is_vorbis else "ffmpeg"
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
    started = time.time()
    if not args.name and not args.fev:
        raise ValueError("Provide --name or --fev")
    if args.name and args.fev:
        raise ValueError("Use either --name or --fev, not both")
    if args.name:
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
    )
    try:
        has_fsb = bool(fsb_map) or (fsb_path and os.path.isfile(fsb_path))
        if not has_fsb:
            raise FileNotFoundError("FSB not found")
        cmd_event_bank_files(event_args)
        logger.info("event_bank_files step complete")
    except Exception as exc:
        logger.warning("event_bank_files skipped: %s", exc)

    output_path = build_fdp_project_from_fev(args.fev, args.template, args.output_dir)
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

    bank_files_cmd = sub.add_parser("event-bank-files", help="write event-bank files json and export wav audio")
    bank_files_cmd.add_argument("--fev", required=True, help="input .fev file")
    bank_files_cmd.add_argument("--fsb", required=False, help="input .fsb file (optional)")
    bank_files_cmd.add_argument("--output", required=False, help="output json file name (always placed under build/{fev_name})")
    bank_files_cmd.add_argument("--audio-root", required=False, help="audio subdirectory name (always placed under build/{fev_name})")
    bank_files_cmd.add_argument("--output-dir", required=False, help="output base directory (default: build; output is {base}/{fev_name})")
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
    gen_all_cmd.add_argument("--no-fix-mp3", action="store_true", help="do not remove mp3 inter-frame padding")
    gen_all_cmd.add_argument("--jobs", type=int, default=16, help="number of worker threads for ffmpeg decoding")
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
