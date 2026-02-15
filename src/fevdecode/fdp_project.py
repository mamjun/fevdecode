from __future__ import annotations

import json
import os
import shutil
import uuid

try:
    import lxml.etree as ET
except Exception:  # pragma: no cover - fallback when lxml not installed
    import xml.etree.ElementTree as ET

from .parser import parse_fdp_event_effects, parse_fev_event_map


def _indent(elem: ET.Element, level: int = 0) -> None:
    indent_text = "\n" + "    " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent_text + "    "
        for child in elem:
            _indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent_text
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent_text


def _find_child_with_name(parent: ET.Element, tag: str, name_value: str) -> ET.Element | None:
    for child in parent.findall(tag):
        name = child.findtext("name")
        if name == name_value:
            return child
    return None


def _new_guid() -> str:
    return "{" + str(uuid.uuid4()) + "}"


def _bank_for_length(lengthms: int | None, sfx_bank: str, bgm_bank: str) -> str:
    if lengthms is not None and lengthms > 30000:
        return bgm_bank
    return sfx_bank


def _normalize_waveform_path(raw_path: str, output_root: str) -> str:
    if not raw_path:
        return raw_path
    normalized = raw_path.replace("\\", "/")
    if os.path.isabs(raw_path):
        rel = os.path.relpath(raw_path, output_root).replace("\\", "/")
        return f"./{rel.lstrip('./')}"
    if normalized.startswith("./") or normalized.startswith("../"):
        rel = os.path.normpath(normalized).replace("\\", "/")
        return f"./{rel.lstrip('./')}"
    if normalized.startswith("audio/"):
        rel = os.path.normpath(normalized).replace("\\", "/")
        return f"./{rel.lstrip('./')}"
    full_path = os.path.join(output_root, "audio", normalized)
    rel = os.path.relpath(full_path, output_root).replace("\\", "/")
    return f"./{rel.lstrip('./')}"


def _load_audio_output_map(json_path: str, output_root: str) -> dict[tuple[str, int], str]:
    if not os.path.isfile(json_path):
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception:
        return {}
    mapping: dict[tuple[str, int], str] = {}
    for event in data.get("events", []):
        event_path = event.get("event")
        if not event_path:
            continue
        for entry in event.get("files", []):
            file_index = entry.get("file_index")
            audio_output = entry.get("audio_output")
            if not isinstance(file_index, int) or not audio_output:
                continue
            rel = os.path.relpath(audio_output, output_root).replace("\\", "/")
            mapping[(event_path, file_index)] = f"./{rel.lstrip('./')}"
    return mapping


def _resolve_fsb_map_from_bank_list(bank_list: list[str], fev_dir: str) -> dict[str, str]:
    fsb_map: dict[str, str] = {}
    for bank_name in bank_list:
        if not bank_name:
            continue
        candidate_name = bank_name if bank_name.lower().endswith(".fsb") else f"{bank_name}.fsb"
        candidate = os.path.join(fev_dir, candidate_name)
        if os.path.isfile(candidate):
            fsb_map[bank_name] = candidate
    return fsb_map


def _build_missing_fsb_report(
    json_path: str,
    fev_path: str,
) -> tuple[dict | None, set[str]]:
    if not os.path.isfile(json_path):
        return None, set()
    try:
        with open(json_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception:
        return None, set()
    bank_list = list(data.get("bank_list", []) or [])
    fsb_map = data.get("fsb_map") or {}
    if not fsb_map:
        fsb_map = _resolve_fsb_map_from_bank_list(bank_list, os.path.dirname(fev_path))
    missing_banks = [bank for bank in bank_list if bank and bank not in fsb_map]
    if not missing_banks:
        return None, set()
    missing_events: list[dict] = []
    skip_events: set[str] = set()
    for event in data.get("events", []) or []:
        event_path = event.get("event")
        if not event_path:
            continue
        missing_for_event = sorted(
            {
                entry.get("bank_name")
                for entry in event.get("files", []) or []
                if entry.get("bank_name") in missing_banks
            }
        )
        if missing_for_event:
            missing_events.append({"event": event_path, "missing_banks": missing_for_event})
            skip_events.add(event_path)
    report = {
        "fev": fev_path,
        "missing_fsb": missing_banks,
        "events": missing_events,
    }
    return report, skip_events


def _build_soundbank_waveform(filename: str) -> ET.Element:
    waveform = ET.Element("waveform")
    _append_text(waveform, "filename", filename)
    _append_text(waveform, "guid", _new_guid())
    _append_text(waveform, "mindistance", "1")
    _append_text(waveform, "maxdistance", "10000")
    _append_text(waveform, "deffreq", "44100")
    _append_text(waveform, "defvol", "1")
    _append_text(waveform, "defpan", "0")
    _append_text(waveform, "defpri", "128")
    _append_text(waveform, "xmafiltering", "0")
    _append_text(waveform, "channelmode", "0")
    _append_text(waveform, "quality_crossplatform", "0")
    _append_text(waveform, "quality", "-1")
    for platform in (
        "PC",
        "XBOX360",
        "PSP",
        "PS3",
        "WII",
        "WiiU",
        "3DS",
        "NGP",
        "ANDROID",
        "IOS",
    ):
        _append_text(waveform, f"_{platform}_resamplemode", "0")
        _append_text(waveform, f"_{platform}_optimisedratereduction", "100")
        _append_text(waveform, f"_{platform}_fixedsamplerate", "48000")
    _append_text(waveform, "notes", "")
    return waveform


def _build_sounddef(
    name: str,
    file_list: list[dict],
    sfx_bank: str,
    bgm_bank: str,
    output_root: str,
) -> ET.Element:
    sounddef = ET.Element("sounddef")
    _append_text(sounddef, "name", name)
    _append_text(sounddef, "guid", _new_guid())
    _append_text(sounddef, "type", "randomnorepeat")
    _append_text(sounddef, "playlistmode", "0")
    _append_text(sounddef, "randomrepeatsounds", "0")
    _append_text(sounddef, "randomrepeatsilences", "0")
    _append_text(sounddef, "shuffleglobal", "0")
    _append_text(sounddef, "sequentialrememberposition", "0")
    _append_text(sounddef, "sequentialglobal", "0")
    _append_text(sounddef, "spawntime_min", "0")
    _append_text(sounddef, "spawntime_max", "0")
    _append_text(sounddef, "spawn_max", "1")
    _append_text(sounddef, "mode", "0")
    _append_text(sounddef, "pitch", "0")
    _append_text(sounddef, "pitch_randmethod", "1")
    _append_text(sounddef, "pitch_random_min", "0")
    _append_text(sounddef, "pitch_random_max", "0")
    _append_text(sounddef, "pitch_randomization", "0")
    _append_text(sounddef, "pitch_recalculate", "0")
    _append_text(sounddef, "volume_db", "0")
    _append_text(sounddef, "volume_randmethod", "1")
    _append_text(sounddef, "volume_random_min", "0")
    _append_text(sounddef, "volume_random_max", "0")
    _append_text(sounddef, "volume_randomization", "0")
    _append_text(sounddef, "position_randomization_min", "0")
    _append_text(sounddef, "position_randomization", "0")
    _append_text(sounddef, "trigger_delay_min", "0")
    _append_text(sounddef, "trigger_delay_max", "0")
    _append_text(sounddef, "spawncount", "0")
    _append_text(sounddef, "notes", "")
    _append_text(sounddef, "entrylistmode", "1")
    for item in file_list:
        waveform = ET.SubElement(sounddef, "waveform")
        filename = item.get("audio_output") or _normalize_waveform_path(item["path"], output_root)
        _append_text(waveform, "filename", filename)
        bank_name = _bank_for_length(item.get("lengthms"), sfx_bank, bgm_bank)
        _append_text(waveform, "soundbankname", bank_name)
        _append_text(waveform, "weight", "100")
        _append_text(waveform, "percentagelocked", "0")
    return sounddef


def _append_text(parent: ET.Element, tag: str, text: str) -> ET.Element:
    child = ET.SubElement(parent, tag)
    child.text = text
    return child


def _apply_event_properties(event: ET.Element, properties: dict[str, str] | None) -> None:
    if not properties:
        return
    for key, value in properties.items():
        if value is None:
            continue
        node = event.find(key)
        if node is None:
            node = ET.SubElement(event, key)
        node.text = value


def _build_simpleevent(
    event_name: str,
    sounddef_name: str,
    bank_name: str,
    category: str,
    template_name: str,
    loop_enabled: bool,
    pitch_octaves: float | None = None,
    pitch_units: str | None = None,
    event_properties: dict[str, str] | None = None,
) -> ET.Element:
    simpleevent = ET.Element("simpleevent")
    event = ET.SubElement(simpleevent, "event")
    _append_text(event, "name", event_name)
    _append_text(event, "guid", _new_guid())
    _append_text(event, "parameter_nextid", "0")
    _append_text(event, "layer_nextid", "1")
    layer = ET.SubElement(event, "layer")
    _append_text(layer, "name", "layer00")
    _append_text(layer, "height", "100")
    _append_text(layer, "envelope_nextid", "0")
    _append_text(layer, "mute", "0")
    _append_text(layer, "solo", "0")
    _append_text(layer, "soundlock", "0")
    _append_text(layer, "envlock", "0")
    _append_text(layer, "priority", "-1")
    sound = ET.SubElement(layer, "sound")
    _append_text(sound, "name", sounddef_name)
    _append_text(sound, "x", "0")
    _append_text(sound, "width", "1")
    _append_text(sound, "startmode", "0")
    _append_text(sound, "loopmode", "0" if loop_enabled else "1")
    _append_text(sound, "loopcount2", "-1" if loop_enabled else "0")
    _append_text(sound, "autopitchenabled", "0")
    _append_text(sound, "autopitchparameter", "0")
    _append_text(sound, "autopitchreference", "0")
    _append_text(sound, "autopitchatzero", "0")
    _append_text(sound, "finetune", "0")
    _append_text(sound, "volume", "1")
    _append_text(sound, "fadeintype", "0")
    _append_text(sound, "fadeouttype", "0")
    for platform in (
        "PC",
        "XBOX360",
        "PSP",
        "PS3",
        "WII",
        "WiiU",
        "3DS",
        "NGP",
        "ANDROID",
        "IOS",
    ):
        _append_text(layer, f"_{platform}_enable", "1")
    _append_text(event, "car_rpm", "0")
    _append_text(event, "car_rpmsmooth", "0.075")
    _append_text(event, "car_loadsmooth", "0.05")
    _append_text(event, "car_loadscale", "6")
    _append_text(event, "volume_db", "0")
    if isinstance(pitch_octaves, (int, float)):
        _append_text(event, "pitch", f"{pitch_octaves}")
    else:
        _append_text(event, "pitch", "0")
    _append_text(event, "pitch_units", pitch_units or "Octaves")
    _append_text(event, "pitch_randomization", "0")
    _append_text(event, "pitch_randomization_units", "Octaves")
    _append_text(event, "volume_randomization", "0")
    _append_text(event, "priority", "128")
    _append_text(event, "maxplaybacks", "1")
    _append_text(event, "maxplaybacks_behavior", "Steal_quietest")
    _append_text(event, "stealpriority", "10000")
    _append_text(event, "mode", "x_3d")
    _append_text(event, "ignoregeometry", "Yes")
    _append_text(event, "rolloff", "Linear")
    _append_text(event, "mindistance", "3")
    _append_text(event, "maxdistance", "30")
    _append_text(event, "auto_distance_filtering", "Off")
    _append_text(event, "distance_filter_centre_freq", "-5.48613e+303")
    _append_text(event, "headrelative", "World_relative")
    _append_text(event, "oneshot", "No" if loop_enabled else "Yes")
    _append_text(event, "istemplate", "No")
    _append_text(event, "usetemplate", template_name)
    _append_text(event, "notes", "")
    _append_text(event, "category", category)
    _append_text(event, "position_randomization_min", "0")
    _append_text(event, "position_randomization", "0")
    _append_text(event, "speaker_l", "1")
    _append_text(event, "speaker_c", "0")
    _append_text(event, "speaker_r", "1")
    _append_text(event, "speaker_ls", "0")
    _append_text(event, "speaker_rs", "0")
    _append_text(event, "speaker_lb", "0")
    _append_text(event, "speaker_rb", "0")
    _append_text(event, "speaker_lfe", "0")
    _append_text(event, "speaker_config", "0")
    _append_text(event, "speaker_pan_r", "1")
    _append_text(event, "speaker_pan_theta", "0")
    _append_text(event, "cone_inside_angle", "360")
    _append_text(event, "cone_outside_angle", "360")
    _append_text(event, "cone_outside_volumedb", "0")
    _append_text(event, "doppler_scale", "1")
    _append_text(event, "reverbdrylevel_db", "0")
    _append_text(event, "reverblevel_db", "0")
    _append_text(event, "speaker_spread", "0")
    _append_text(event, "panlevel3d", "1")
    _append_text(event, "fadein_time", "0")
    _append_text(event, "fadeout_time", "0")
    _append_text(event, "spawn_intensity", "1")
    _append_text(event, "spawn_intensity_randomization", "0")
    _apply_event_properties(event, event_properties)
    for key in (
        "LAYERS",
        "KEEP_EFFECTS_PARAMS",
        "VOLUME",
        "PITCH",
        "PITCH_RANDOMIZATION",
        "VOLUME_RANDOMIZATION",
        "PRIORITY",
        "MAX_PLAYBACKS",
        "MAX_PLAYBACKS_BEHAVIOR",
        "STEAL_PRIORITY",
        "MODE",
        "IGNORE_GEOMETRY",
        "X_3D_ROLLOFF",
        "X_3D_MIN_DISTANCE",
        "X_3D_MAX_DISTANCE",
        "X_3D_POSITION",
        "X_3D_MIN_POSITION_RANDOMIZATION",
        "X_3D_POSITION_RANDOMIZATION",
        "X_3D_CONE_INSIDE_ANGLE",
        "X_3D_CONE_OUTSIDE_ANGLE",
        "X_3D_CONE_OUTSIDE_VOLUME",
        "X_3D_DOPPLER_FACTOR",
        "REVERB_WET_LEVEL",
        "REVERB_DRY_LEVEL",
        "X_3D_SPEAKER_SPREAD",
        "X_3D_PAN_LEVEL",
        "X_2D_SPEAKER_L",
        "X_2D_SPEAKER_C",
        "X_2D_SPEAKER_R",
        "X_2D_SPEAKER_LS",
        "X_2D_SPEAKER_RS",
        "X_2D_SPEAKER_LR",
        "X_2D_SPEAKER_RR",
        "X_SPEAKER_LFE",
        "ONESHOT",
        "FADEIN_TIME",
        "FADEOUT_TIME",
        "NOTES",
        "USER_PROPERTIES",
        "CATEGORY",
    ):
        _append_text(event, f"TEMPLATE_PROP_{key}", "1")
    for platform in (
        "PC",
        "XBOX360",
        "PSP",
        "PS3",
        "WII",
        "WiiU",
        "3DS",
        "NGP",
        "ANDROID",
        "IOS",
    ):
        _append_text(event, f"_{platform}_enabled", "1")
    _append_text(simpleevent, "playcount", "0")
    _append_text(simpleevent, "loopcount", "0")
    _append_text(simpleevent, "queuedsounds", "1")
    _append_text(simpleevent, "polyphony", "1")
    _append_text(simpleevent, "totalgrains", "0")
    _append_text(simpleevent, "mingraininterval", "1")
    _append_text(simpleevent, "maxgraininterval", "1")
    _append_text(simpleevent, "randomrepeatsounds", "0")
    _append_text(simpleevent, "randomrepeatsilences", "0")
    _append_text(simpleevent, "shuffleglobal", "0")
    _append_text(simpleevent, "sequentialrememberposition", "0")
    _append_text(simpleevent, "sequentialglobal", "0")
    _append_text(simpleevent, "bankname", bank_name)
    for platform in (
        "PC",
        "XBOX360",
        "PSP",
        "PS3",
        "WII",
        "WiiU",
        "3DS",
        "NGP",
        "ANDROID",
        "IOS",
    ):
        _append_text(simpleevent, f"_{platform}_resamplemode", "Off")
        _append_text(simpleevent, f"_{platform}_automaticsampleratepercentage", "100")
        _append_text(simpleevent, f"_{platform}_fixedsamplerate", "48000")
    return simpleevent


def build_fdp_project_from_fev(
    fev_path: str,
    template_path: str,
    output_dir: str | None = None,
    pitch_units_fdp: str | None = None,
) -> str:
    if not os.path.isfile(fev_path):
        raise FileNotFoundError(f"FEV not found: {fev_path}")
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    parsed = parse_fev_event_map(fev_path)
    pitch_units_map: dict[str, str] = {}
    fdp_event_props: dict[str, dict] = {}
    if pitch_units_fdp and os.path.isfile(pitch_units_fdp):
        try:
            effects = parse_fdp_event_effects(pitch_units_fdp)
            fdp_event_props = effects
            for path, item in effects.items():
                units = item.get("properties", {}).get("pitch_units")
                if units:
                    pitch_units_map[path] = units
        except Exception:
            pitch_units_map = {}
            fdp_event_props = {}
    fev_name = os.path.splitext(os.path.basename(fev_path))[0]
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_base = output_dir or os.path.join(repo_root, "build")
    output_root = os.path.join(output_base, fev_name)
    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(output_root, f"{fev_name}.fdp")

    with open(template_path, "r", encoding="utf-8", errors="replace") as fp:
        template_text = fp.read()
    root = ET.fromstring(template_text)
    tree = ET.ElementTree(root)

    project_name = root.find("name")
    if project_name is not None:
        project_name.text = fev_name

    sfx_bank = f"{fev_name}_sfx"
    bgm_bank = f"{fev_name}_bgm"

    current_bank = root.find("currentbank")
    if current_bank is not None:
        current_bank.text = bgm_bank

    bank_nodes: dict[str, ET.Element] = {}
    for bank in root.findall("soundbank"):
        name_node = bank.find("name")
        if name_node is None or not name_node.text:
            continue
        if name_node.text == "basic_sfx":
            name_node.text = sfx_bank
        elif name_node.text == "basic_bgm":
            name_node.text = bgm_bank
        bank_nodes[name_node.text] = bank

    sounddef_nextid = root.find("sounddef_nextid")
    next_id = int(sounddef_nextid.text) if sounddef_nextid is not None and sounddef_nextid.text else 0

    master_folder = _find_child_with_name(root, "sounddeffolder", "master")
    if master_folder is None:
        raise ValueError("Template missing master sounddeffolder")
    simple_folder = _find_child_with_name(master_folder, "sounddeffolder", "__simpleevent_sounddef__")
    if simple_folder is None:
        raise ValueError("Template missing __simpleevent_sounddef__ folder")

    for child in list(simple_folder):
        if child.tag == "sounddef":
            simple_folder.remove(child)

    event_tree: dict[str, dict] = {}
    sounddef_map: dict[str, str] = {}

    event_bank_json = os.path.join(output_root, f"{fev_name}_event_bank_files.json")
    audio_output_map = _load_audio_output_map(event_bank_json, output_root)
    missing_report, skip_events = _build_missing_fsb_report(event_bank_json, fev_path)
    if missing_report:
        miss_path = os.path.join(output_root, "missfsb.json")
        with open(miss_path, "w", encoding="utf-8") as fp:
            json.dump(missing_report, fp, ensure_ascii=False, indent=2)
    else:
        skip_events = set()

    bank_waveforms: dict[str, set[str]] = {sfx_bank: set(), bgm_bank: set()}

    for event_path, event in sorted(parsed.event_map.items()):
        if event_path in skip_events:
            continue
        parts = [p for p in event_path.split("/") if p]
        if not parts:
            continue
        event_name = parts[-1]
        sounddef_name = f"/__simpleevent_sounddef__/{event_name}_{next_id}"
        next_id += 1
        sounddef_map[event_path] = sounddef_name
        node = event_tree
        for segment in parts[:-1]:
            node = node.setdefault(segment, {"__events__": []})
        node.setdefault("__events__", []).append((event_name, sounddef_name, event_path))

        file_items = []
        for item in event.ref_file_list:
            if not item.path:
                continue
            audio_output = audio_output_map.get((event_path, item.file_index))
            resolved_path = audio_output or _normalize_waveform_path(item.path, output_root)
            target_bank = _bank_for_length(item.lengthms, sfx_bank, bgm_bank)
            bank_waveforms.setdefault(target_bank, set()).add(resolved_path)
            file_items.append(
                {
                    "path": item.path,
                    "lengthms": item.lengthms,
                    "audio_output": audio_output,
                }
            )
        simple_folder.append(_build_sounddef(sounddef_name, file_items, sfx_bank, bgm_bank, output_root))

    if sounddef_nextid is not None:
        sounddef_nextid.text = str(next_id)

    for bank_name, bank_node in bank_nodes.items():
        for child in list(bank_node):
            if child.tag == "waveform":
                bank_node.remove(child)
        for filename in sorted(bank_waveforms.get(bank_name, set())):
            bank_node.append(_build_soundbank_waveform(filename))

    for child in list(root):
        if child.tag == "eventgroup":
            root.remove(child)

    insert_before = root.find("default_soundbank_props")
    insert_index = list(root).index(insert_before) if insert_before is not None else len(root)

    def _append_eventgroup(parent: ET.Element, group_name: str, node: dict) -> None:
        eventgroup = ET.SubElement(parent, "eventgroup")
        _append_text(eventgroup, "name", group_name)
        _append_text(eventgroup, "guid", _new_guid())
        _append_text(eventgroup, "eventgroup_nextid", "0")
        _append_text(eventgroup, "event_nextid", "0")
        _append_text(eventgroup, "open", "1")
        _append_text(eventgroup, "notes", "")

        for event_name, sounddef_name, event_path in node.get("__events__", []):
            event = parsed.event_map[event_path]
            use_loop_template = any(
                item.lengthms and item.lengthms > 30000 for item in event.ref_file_list
            )
            template_name = "basic3dbgploop" if use_loop_template else "basic3doneshot"
            bank_name = sfx_bank
            if any(item.lengthms and item.lengthms > 30000 for item in event.ref_file_list):
                bank_name = bgm_bank
            pitch_octaves = event.pitch_raw * 4.0 if isinstance(event.pitch_raw, (int, float)) else None
            pitch_units = event.pitch_units or pitch_units_map.get(event_path)
            event_properties = fdp_event_props.get(event_path, {}).get("properties") if fdp_event_props else None
            eventgroup.append(
                _build_simpleevent(
                    event_name,
                    sounddef_name,
                    bank_name,
                    "set_sfx/sfx",
                    template_name,
                    use_loop_template,
                    pitch_octaves,
                    pitch_units,
                    event_properties,
                )
            )

        for child_name, child_node in sorted(
            ((k, v) for k, v in node.items() if k != "__events__"),
            key=lambda item: item[0],
        ):
            _append_eventgroup(eventgroup, child_name, child_node)

    for group_name, node in sorted(event_tree.items(), key=lambda item: item[0]):
        eventgroup = ET.Element("eventgroup")
        _append_text(eventgroup, "name", group_name)
        _append_text(eventgroup, "guid", _new_guid())
        _append_text(eventgroup, "eventgroup_nextid", "0")
        _append_text(eventgroup, "event_nextid", "0")
        _append_text(eventgroup, "open", "1")
        _append_text(eventgroup, "notes", "")

        for event_name, sounddef_name, event_path in node.get("__events__", []):
            event = parsed.event_map[event_path]
            use_loop_template = any(
                item.lengthms and item.lengthms > 30000 for item in event.ref_file_list
            )
            template_name = "basic3dbgploop" if use_loop_template else "basic3doneshot"
            bank_name = sfx_bank
            if any(item.lengthms and item.lengthms > 30000 for item in event.ref_file_list):
                bank_name = bgm_bank
            pitch_octaves = event.pitch_raw * 4.0 if isinstance(event.pitch_raw, (int, float)) else None
            pitch_units = event.pitch_units or pitch_units_map.get(event_path)
            event_properties = fdp_event_props.get(event_path, {}).get("properties") if fdp_event_props else None
            eventgroup.append(
                _build_simpleevent(
                    event_name,
                    sounddef_name,
                    bank_name,
                    "set_sfx/sfx",
                    template_name,
                    use_loop_template,
                    pitch_octaves,
                    pitch_units,
                    event_properties,
                )
            )

        for child_name, child_node in sorted(
            ((k, v) for k, v in node.items() if k != "__events__"),
            key=lambda item: item[0],
        ):
            _append_eventgroup(eventgroup, child_name, child_node)

        root.insert(insert_index, eventgroup)
        insert_index += 1

    _indent(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=False)

    template_dir = os.path.dirname(template_path)
    fdt_path = os.path.join(template_dir, "basic3dsound.fdt")
    if os.path.isfile(fdt_path):
        shutil.copy2(fdt_path, os.path.join(output_root, os.path.basename(fdt_path)))
    return output_path
