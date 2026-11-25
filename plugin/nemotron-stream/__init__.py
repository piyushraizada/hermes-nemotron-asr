"""Nemotron streaming STT provider for Hermes Agent.

Registers ``nemotron`` as an STT provider backed by the resident sherpa-onnx
Nemotron 0.6B server (see server/nemotron_server.py, default 127.0.0.1:8020).

* ``open_stream_session`` -> live partial transcripts while the user speaks
  (``stt.streaming: true``); one stateful decoder per utterance.
* ``transcribe``          -> file path: ffmpeg to 16 kHz mono WAV, then ``POST /infer``.

Audio never leaves the machine. If the server is down the live session fails to
open, Hermes falls back to the file path, and the failure is reported rather
than hidden.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from pathlib import Path

from agent.transcription_provider import (
    TranscriptionProvider,
    TranscriptionStreamSession,
)

logger = logging.getLogger(__name__)

PROVIDER = "nemotron"
CHUNK_BYTES = 17920  # 560 ms of 16 kHz mono s16le


def server_url() -> str:
    """Base URL of the ASR server ($NEMOTRON_ASR_URL, default 127.0.0.1:8020)."""
    return (os.environ.get("NEMOTRON_ASR_URL") or "http://127.0.0.1:8020").rstrip("/")


def _ffmpeg() -> str:
    return os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"


def _post(path: str, data: bytes = b"", timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(server_url() + path, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


class NemotronStreamSession(TranscriptionStreamSession):
    """One utterance: PCM in, partials out, final envelope at end of speech."""

    def __init__(self, language: str | None = None, prompt: str | None = None) -> None:
        self._sid = json.loads(_post("/stream/open", timeout=10))["session_id"]
        self._buf = bytearray()
        self._text = ""
        self._error: str | None = None
        self._lock = threading.Lock()
        logger.debug("nemotron stream opened %s", self._sid[:8])

    def push_audio(self, chunk: bytes) -> None:
        if self._error or not chunk:
            return
        with self._lock:
            self._buf.extend(chunk)
            if len(self._buf) < CHUNK_BYTES:
                return
            data = bytes(self._buf)
            self._buf.clear()
        try:
            out = json.loads(_post(f"/stream/push?sid={self._sid}", data))
            text = out.get("text") or ""
            if text:
                self._text = text
        except Exception as exc:  # never raise into the feeder thread
            self._error = f"{type(exc).__name__}: {exc}"
            logger.warning("nemotron stream push failed: %s", self._error)

    def partial_transcript(self) -> str:
        return self._text

    def finalize(self) -> dict:
        if self._error:
            return {"success": False, "transcript": "", "provider": PROVIDER,
                    "error": self._error}
        try:
            with self._lock:
                rest = bytes(self._buf)
                self._buf.clear()
            if rest:
                out = json.loads(_post(f"/stream/push?sid={self._sid}", rest))
                if out.get("text"):
                    self._text = out["text"]
            text = json.loads(_post(f"/stream/final?sid={self._sid}", timeout=60)).get("text", "")
            return {"success": True, "transcript": (text or self._text).strip(),
                    "provider": PROVIDER}
        except Exception as exc:
            return {"success": False, "transcript": "", "provider": PROVIDER,
                    "error": f"{type(exc).__name__}: {exc}"}


class NemotronProvider(TranscriptionProvider):
    """STT provider row named ``nemotron`` (matches ``stt.provider``)."""

    @property
    def name(self) -> str:
        return PROVIDER

    @property
    def display_name(self) -> str:
        return "Nemotron 0.6B (local, streaming)"

    def list_models(self) -> list[dict]:
        return [{"id": "nemotron-0.6b-560ms", "display": "Nemotron 0.6B int8, 560 ms"}]

    def default_model(self) -> str | None:
        return "nemotron-0.6b-560ms"

    @property
    def streaming_capable(self) -> bool:
        return True

    def open_stream_session(self, *, language: str | None = None,
                            prompt: str | None = None) -> NemotronStreamSession:
        return NemotronStreamSession(language=language, prompt=prompt)

    def transcribe(self, file_path: str, *, model: str | None = None,
                   language: str | None = None, **extra) -> dict:
        """File path: convert to 16 kHz mono WAV, then one stateless /infer call."""
        try:
            with tempfile.TemporaryDirectory() as tmp:
                wav = Path(tmp) / "take.wav"
                proc = subprocess.run(
                    [_ffmpeg(), "-y", "-loglevel", "error", "-i", file_path,
                     "-ar", "16000", "-ac", "1", "-f", "wav", str(wav)],
                    capture_output=True, text=True, timeout=120,
                )
                if proc.returncode != 0 or not wav.is_file():
                    detail = (proc.stderr or "").strip().splitlines()
                    return {"success": False, "transcript": "", "provider": PROVIDER,
                            "error": detail[-1] if detail else "ffmpeg conversion failed"}
                text = _post("/infer", wav.read_bytes(), timeout=180)
                return {"success": True,
                        "transcript": text.decode("utf-8", "replace").strip(),
                        "provider": PROVIDER}
        except Exception as exc:
            logger.warning("nemotron file transcription failed: %s", exc)
            return {"success": False, "transcript": "", "provider": PROVIDER,
                    "error": f"{type(exc).__name__}: {exc}"}


def register(ctx) -> None:
    ctx.register_transcription_provider(NemotronProvider())
    logger.info("nemotron-stream registered the `nemotron` STT provider")
