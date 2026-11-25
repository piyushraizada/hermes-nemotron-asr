# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-11-25

First public release.

### Added

- **Resident ASR server** (`server/nemotron_server.py`): loads Nemotron 0.6B int8 once
  (~0.9 GB RAM) and serves HTTP on 127.0.0.1:8020.
  - `POST /infer` — stateless transcription of a 16 kHz mono PCM16 WAV.
  - `POST /stream/open|push|final|cancel` — stateful live sessions. One persistent
    `OnlineStream` per utterance; text is committed at decoder endpoints, so partials
    accumulate across pauses instead of restarting.
  - `GET /health`.
  - Pins itself to the P-core group on Windows hybrid CPUs (Windows parks boot-launched
    threads on E-cores, which measured ~8× slower).
- **Hermes provider** (`plugin/nemotron-stream/`): registers `nemotron` as an STT provider
  with `streaming_capable`, so `stt.streaming: true` renders partial text in the dictation
  pill while the user speaks. File-path fallback (ffmpeg → `/infer`) covers failed live
  sessions.
- **Installers**: `installer/install.ps1` (Windows — venv, model download, hidden logon
  launcher, plugin install, config) and `installer/install.sh` (macOS/Linux, prints
  systemd/launchd guidance). `uninstall.ps1` reverses everything, with `-Purge` to delete
  the venv and model.
- **Tests**: `tests/smoke_test.py` compares batch and streaming output against a live
  server; `tests/plugin_check.py` exercises the provider without a gateway.
- **Docs**: README with measured numbers, endpoint table, configuration and a
  troubleshooting matrix; `NOTICE` carrying the NVIDIA model attribution.

### Measured

- i5-13400, 4 threads, 560 ms chunks: **RTF ~0.15**; a 7 s utterance finalizes in
  **~1.3 s** streaming vs **2–5 s** batch over the same model.

### Notes

- Model weights are **not** distributed in this repository. The installer downloads the
  int8 ONNX export from the sherpa-onnx releases; the NVIDIA Open Model License governs
  the weights (see `NOTICE`).
- Project code is MIT.

[1.0.0]: https://github.com/piyushraizada/hermes-nemotron-asr/releases/tag/v1.0.0
