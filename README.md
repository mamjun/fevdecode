# fevdecode

一个轻量级的 Python 项目骨架，用于从 `sound/` 目录读取 `.fev` 与 `.fsb` 文件，并打包为自定义的 `.fdp` 文件。当前实现提供稳定的读写容器格式（`FDP0`），方便后续替换为真实的 FEV/FSP/FDP 解析逻辑。

## 快速开始

```powershell
python -m fevdecode build --input sound --output build\sora.fdp
```

查看解析摘要：

```powershell
python -m fevdecode inspect --input sound
```

`inspect` 现在会在 FEV 中解析 `STRR` 字符串表，并在清单里给出字符串数量与样例，同时解析 `PROJ` 子块的预览与字符串。

从 FSB 提取样本：

```powershell
python -m fevdecode extract-fsb --input sound\sora.fsb --output build\fsb_samples
```

从 FSB 提取可播放音频（内置 FSB5 解析；Vorbis 需要头部表）：

```powershell
pip install -r requirements.txt
python -m fevdecode extract-fsb-audio --input sound\sora.fsb --output build\fsb_audio

如需使用 `fmodex-lw.exe` 直接解码为 WAV（建议放置于 `libs/fmodex-lw/`）：

```powershell
python -m fevdecode extract-fsb-audio --use-fmodex --input sound\sora.fsb --output build\fsb_audio_wav
```

可以指定并行进程数（仅对 fmodex 解码生效，默认 16）：

```powershell
python -m fevdecode extract-fsb-audio --use-fmodex --jobs 4 --input sound\sora.fsb --output build\fsb_audio_wav
```
```

按事件层级生成 `event_bank_files.json` 并导出 WAV（依赖 `pydub` + `ffmpeg`）：

```powershell
pip install -r requirements.txt
python -m fevdecode event-bank-files --fev sound\sora.fev --fsb sound\sora.fsb

使用 `fmodex-lw.exe` 解码 WAV：

```powershell
python -m fevdecode event-bank-files --use-fmodex --fev sound\sora.fev --fsb sound\sora.fsb
```

并行解码示例（默认优先 fmodex）：

```powershell
python -m fevdecode event-bank-files --jobs 32 --fev sound\sora.fev --fsb sound\sora.fsb
```
```

禁用 fmodex，使用 ffmpeg：

```powershell
python -m fevdecode event-bank-files --no-fmodex --jobs 32 --fev sound\sora.fev --fsb sound\sora.fsb
```
```

默认输出到 `build/{fev_name}/`，例如 `build/sora/`。`--output` 与 `--audio-root` 只接受文件名/子目录名，仍会放在该目录下。

若 Vorbis 样本缺少头部信息，会在 `audio` 目录下生成 `missing_vorbis_headers.json`。

默认输出目录为 `build/{fev_name}/`，与 FEV 解析结果目录保持一致。
可自定义模板与输出目录（`--output-dir` 为输出根目录，实际输出到 `{output-dir}/{fev_name}/`）：

> 说明：Vorbis 样本会通过内置 Ogg 重建后使用 `ffmpeg` 直接解码（无需 `libvorbis`），并会优先读取项目内的 `libs/ffmpeg.exe`、`libs/ffmpeg/ffmpeg.exe` 或 `libs/ffmpeg/bin/ffmpeg.exe`；若不存在则使用系统 PATH。`pydub` 作为兜底路径时可能需要 `ffprobe`。

如果导出的 MP3 无法播放，可尝试默认的去填充逻辑；也可以禁用：

```powershell
python -m fevdecode extract-fsb-audio --no-fix-mp3 --input sound\sora.fsb --output build\fsb_audio_raw
```

按 FEV 事件路径解包音频并生成映射：

```powershell
python -m fevdecode extract-events --fev sound\sora.fev --output build
```

根据 FEV 解析结果生成 FMOD 工程 `.fdp`（以 `temp/basic.fdp` 为模板）：

```powershell
python -m fevdecode gen-fdp --fev sound\sora.fev
```

一键生成 `event_bank_files.json` + 音频 + `.fdp`（先尝试音频导出，再生成 `.fdp`；先解析 `.fev` 的 `bank_list`，自动检测对应 `.fsb` 并参与音频解析，缺失则跳过；也可手动指定 `--fsb`）：

```powershell
python -m fevdecode gen-all --fev sound\sora.fev

手动指定 FSB：

```powershell
python -m fevdecode gen-all --fev sound\sora.fev --fsb sound\sora.fsb
```
```

按项目名生成（自动读取 `sound/{name}.fev` / `sound/{name}.fsb`）：

```powershell
python -m fevdecode gen-all --name dontstarve_DLC003
```

可自定义模板与输出目录：

```powershell
python -m fevdecode gen-fdp --fev sound\sora.fev --template temp\basic.fdp --output-dir build
```

默认输出目录为 `build/{fev_name}/`，与 FEV 解析结果目录保持一致。

生成时会自动将模板目录下的 `basic3dsound.fdt` 一并拷贝到输出目录。

生成的 `.fdp` 会将 `soundbank` 重命名为 `{fev_name}_sfx` 与 `{fev_name}_bgm`，并按 $30s$ 阈值分配音频。若 `bank_list` 中某些 `.fsb` 缺失，会跳过相关事件，并在 `build/{fev_name}/missfsb.json` 记录缺失的 FSB 与事件列表。

`waveform/filename` 使用 `./` 相对路径；若存在 `build/{fev_name}/{fev_name}_event_bank_files.json`，会优先采用其中的 `audio_output` 结果。

`extract-events` 会自动从 FEV 的字符串表中提取所有事件（含 `__simpleevent_sounddef__` 与推断路径），生成完整映射。

如果是 Vorbis 样本且缺少头部信息，会跳过并生成 `missing_vorbis_headers.json`，用于后续补全表。

解包：

```powershell
python -m fevdecode extract --input build\sora.fdp --output build\extract
```

## 说明

- 已接入基础真实格式解析（`FEV` 为 RIFF，`FSB` 为 FSB5 头部结构）。
- Vorbis 需要提供头部表：放置 `src/fevdecode/vorbis_headers.py`（定义 `lookup: dict[int, bytes]`）或 `vorbis_headers.py` 于仓库根目录。
- 构建时会在 `fdp` 内加入 `manifest.json` 解析摘要。
- `fdp` 采用自定义容器格式，便于逐步替换为真实格式。

## 日志

关键/耗时步骤会写入 `log/` 目录的运行日志（例如 `event_bank_files_YYYYMMDD_HHMMSS.log`）。
日志包含事件级与音频输出级别的详细记录，便于排查生成过程中的单个条目问题。
