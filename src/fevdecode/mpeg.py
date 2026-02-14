from __future__ import annotations

from typing import Optional


_BITRATES_MPEG1_L3 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
_BITRATES_MPEG2_L3 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]
_SAMPLE_RATES = {
    3: [44100, 48000, 32000, 0],
    2: [22050, 24000, 16000, 0],
    0: [11025, 12000, 8000, 0],
}


def _find_sync(data: bytes, start: int) -> Optional[int]:
    for idx in range(start, len(data) - 1):
        if data[idx] == 0xFF and (data[idx + 1] & 0xE0) == 0xE0:
            return idx
    return None


def _frame_length(b2: int, b3: int) -> Optional[int]:
    version_id = (b2 >> 3) & 0x3
    layer = (b2 >> 1) & 0x3
    bitrate_idx = (b3 >> 4) & 0xF
    sample_rate_idx = (b3 >> 2) & 0x3
    padding = (b3 >> 1) & 0x1

    if layer != 1:
        return None
    if version_id not in _SAMPLE_RATES:
        return None

    sample_rate = _SAMPLE_RATES[version_id][sample_rate_idx]
    if sample_rate == 0:
        return None

    if version_id == 3:
        bitrate = _BITRATES_MPEG1_L3[bitrate_idx] * 1000
        if bitrate == 0:
            return None
        return int((144 * bitrate) // sample_rate + padding)

    bitrate = _BITRATES_MPEG2_L3[bitrate_idx] * 1000
    if bitrate == 0:
        return None
    return int((72 * bitrate) // sample_rate + padding)


def clean_mpeg_padding(data: bytes) -> bytes:
    """Remove inter-frame padding bytes from MPEG Layer III streams."""
    out = bytearray()
    pos = _find_sync(data, 0)
    if pos is None:
        return data

    while pos is not None and pos + 4 <= len(data):
        b2 = data[pos + 1]
        b3 = data[pos + 2]
        frame_len = _frame_length(b2, b3)
        if frame_len is None or frame_len <= 0:
            break
        end = pos + frame_len
        if end > len(data):
            break
        out.extend(data[pos:end])
        next_pos = _find_sync(data, end)
        pos = next_pos

    return bytes(out) if out else data