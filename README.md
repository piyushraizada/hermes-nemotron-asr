# Nemotron ASR for Hermes — CPU-only streaming dictation

[![ci](https://github.com/piyushraizada/hermes-nemotron-asr/actions/workflows/ci.yml/badge.svg)](https://github.com/piyushraizada/hermes-nemotron-asr/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![model: NVIDIA Open Model License](https://img.shields.io/badge/model-NVIDIA%20Open%20Model%20License-76b900.svg)](NOTICE)

Local, streaming speech-to-text for [Hermes Agent](https://hermes-agent.nousresearch.com) that runs
on **four CPU threads**. No GPU, no API key, no audio leaving the machine.

Partial text appears in the dictation pill **while you speak**, and the final transcript lands the
moment you stop instead of seconds later.

```
PARTIAL @0.56s: The
PARTIAL @1.12s: The repo server is
PARTIAL @1.68s: The repo server is running on
PARTIAL @2.24s: The repo server is running on port eight
...
FINAL: The repo server is running on port eight zero two one. This is a streaming transcription smoke test
```

Measured on an i5-13400 (4 threads, 560 ms chunks): **RTF ~0.15**, ~0.9 GB RAM, a 7 s utterance
finalizes in **~1.3 s**. The batch path over the same server takes 2–5 s for the same audio — the
streaming endpoints are what remove that tail.

## What's here

| Path | What it is |
|---|---|
| `server/nemotron_server.py` | Resident ASR server. Loads the model once, serves batch + streaming HTTP. |
| `plugin/nemotron-stream/` | Hermes transcription provider (`nemotron`). ~150 lines of glue. |
| `installer/install.ps1` | Windows installer: venv, model, autostart, plugin, config. |
| `installer/install.sh` | macOS / Linux equivalent (no autostart). |
| `tests/` | Smoke test for the server and a plugin check that needs no gateway. |

The server is useful on its own — anything that can POST WAV bytes can use it.

## Requirements

- Python 3.10+
- `ffmpeg` on PATH (only for the file-transcription fallback)
- ~700 MB disk (633 MB model + venv)
- Hermes Agent 0.22.5+ (only for the plugin half)

## Install

### Windows

```powershell
git clone https://github.com/piyushraizada/hermes-nemotron-asr
cd hermes-nemotron-asr
powershell -ExecutionPolicy Bypass -File installer\install.ps1
```

That creates `venv/`, downloads the int8 model (~442 MB, one time), registers a hidden logon
launcher, starts the server on port 8020, copies the plugin into `%LOCALAPPDATA%\hermes\plugins\`,
and sets:

```
stt.provider = nemotron
stt.streaming = true
```

Options: `-ModelDir`, `-Port`, `-Threads`, `-NoAutostart`, `-NoHermes`.

### macOS / Linux

```bash
git clone https://github.com/piyushraizada/hermes-nemotron-asr
cd hermes-nemotron-asr
./installer/install.sh
```

It prints the systemd / launchd snippet for starting the server at login — autostart is left to you
because it differs per distro.

### Manual

```bash
python -m venv venv && . venv/bin/activate      # Windows: venv\Scripts\activate
pip install sherpa-onnx numpy
# download + extract the model into ./model (see installer/install.sh for the URL)
python server/nemotron_server.py --model-dir ./model
```

Verify with the smoke test (needs any speech recording):

```bash
python tests/smoke_test.py --wav speech.wav
```

## How it works

```
mic ──► Hermes ──► plugin ──► POST /stream/push ──► server ──► sherpa-onnx
                       ▲                              │
                       └──── {"text": partial} ◄──────┘
```

- **One persistent decoder per utterance.** `/stream/open` creates an `OnlineStream`; every push
  keeps its state, so partials accumulate. Text is committed at a decoder endpoint and the stream is
  reset, which is why a pause mid-sentence doesn't wipe what came before.
- **560 ms pushes.** The plugin buffers mic audio to 560 ms (17920 bytes of 16 kHz mono s16le) before
  each round trip — small enough to feel live, large enough that the model runs efficiently.
- **Fallback.** If a live session can't open or dies mid-take, Hermes falls back to the provider's
  file path: ffmpeg → 16 kHz mono WAV → `POST /infer`. A take is never lost.

### Endpoints

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/health` | — | `ok` |
| POST | `/infer` | 16 kHz mono PCM16 WAV | transcript text |
| POST | `/stream/open` | — | `{"session_id": "..."}` |
| POST | `/stream/push?sid=` | raw 16 kHz mono PCM16 | `{"text": partial}` |
| POST | `/stream/final?sid=` | — | `{"text": final}` |
| POST | `/stream/cancel?sid=` | — | `{"ok": true}` |

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `NEMOTRON_MODEL_DIR` | `<repo>/model` | model directory |
| `NEMOTRON_ASR_URL` | `http://127.0.0.1:8020` | server base URL used by the plugin |
| `FFMPEG_BIN` | `ffmpeg` on PATH | converter for the file path |

Server flags: `--model-dir`, `--port` (8020), `--threads` (4), `--chunk-ms` (560).

**Threads:** 4 is the sweet spot on a 4 P-core CPU. More threads than physical performance cores
does not help — the decoder is memory-bandwidth bound. On hybrid Intel CPUs the server pins itself to
the P-cores at startup; Windows otherwise parks boot-launched threads on E-cores, which measured ~8×
slower.

## Uninstall

```powershell
powershell -ExecutionPolicy Bypass -File installer\uninstall.ps1
```

Removes the autostart entry, stops the server, removes the plugin, and restores `stt.provider` to
`local`. Add `-Purge` to also delete `venv/`, `model/`, and the logs.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| No partials, text appears only at the end | `stt.streaming` is off, or the provider isn't streaming-capable | `hermes config set stt.streaming true` |
| `no live STT for the configured provider` in gateway log | plugin not loaded | `hermes plugins list` → `hermes plugins enable nemotron-stream` |
| Dictation fails with a connection error | server not running | `curl 127.0.0.1:8020/health` → restart `run-server.cmd` |
| Transcripts cut off mid-sentence | chunk size too small / endpoint resets | keep `--chunk-ms 560` |
| First take after login is slow | model still loading (~3 s) | expected; it stays resident |

Logs: `server.log` in the repo root (or the file you redirected to), and
`%LOCALAPPDATA%\hermes\logs\gateway.log` for the plugin side.

## Model and license

The model is [NVIDIA Nemotron Speech Streaming 0.6B](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b),
served through the int8 ONNX export from the
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project. It is **not** bundled here — the
installer downloads it.

- License: **NVIDIA Open Model License Agreement** — commercial use allowed; redistribution requires
  a copy of the agreement and the attribution line in [`NOTICE`](NOTICE). Not gated, so no
  click-through at install.
- This repository's own code is MIT (see [`LICENSE`](LICENSE)).

## Credits

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — the runtime that makes CPU streaming practical
- [NVIDIA](https://huggingface.co/nvidia) — the model
- [Nous Research](https://nousresearch.com) — Hermes Agent and its pluggable transcription provider API
