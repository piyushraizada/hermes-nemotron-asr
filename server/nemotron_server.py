"""Resident Nemotron 0.6B streaming ASR server (sherpa-onnx, CPU int8).

One model, one port. Loads once (~2-3 s, ~0.9 GB RAM) and serves:

  GET  /health                    -> "ok"
  POST /infer                     -> body = 16 kHz mono PCM16 WAV bytes; response = transcript text
  POST /stream/open               -> {"session_id": "..."}
  POST /stream/push?sid=<id>      -> body = raw 16 kHz mono PCM16; response = {"text": partial}
  POST /stream/final?sid=<id>     -> {"text": final}
  POST /stream/cancel?sid=<id>    -> {"ok": true}

`/infer` is stateless: one request, one decoder. The `/stream/*` family keeps a
persistent decoder per utterance so partial transcripts accumulate instead of
restarting on every chunk — that is what makes live dictation feel instant.

Measured on an i5-13400 with 4 threads and 560 ms chunks: RTF ~0.15, a 7 s
utterance finalizes in ~1.3 s. Silence returns an empty transcript.

Usage:
    python nemotron_server.py --model-dir /path/to/model [--port 8020] [--threads 4]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import threading
import time
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import sherpa_onnx

DEFAULT_PORT = 8020
TAIL_PAD_S = 0.66
SESSION_TTL_S = 180.0

RECOGNIZER = None
SESSIONS: dict[str, "LiveSession"] = {}
SESSIONS_LOCK = threading.Lock()
DECODE_LOCK = threading.Lock()


def default_model_dir() -> Path:
    """Model directory: $NEMOTRON_MODEL_DIR, else <repo>/model."""
    env = os.environ.get("NEMOTRON_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent / "model"


def model_files(model_dir: Path) -> dict[str, str]:
    """Resolve the four files a transducer needs, with a readable error."""
    names = {
        "tokens": "tokens.txt",
        "encoder": "encoder.int8.onnx",
        "decoder": "decoder.int8.onnx",
        "joiner": "joiner.int8.onnx",
    }
    found: dict[str, str] = {}
    for key, name in names.items():
        path = model_dir / name
        if not path.is_file():
            # Tolerate an unquantized export sitting next to the int8 one.
            alt = model_dir / name.replace(".int8", "")
            if alt.is_file():
                path = alt
            else:
                raise SystemExit(
                    f"missing {name} in {model_dir}\n"
                    "Download the model first (see installer/README) or pass --model-dir."
                )
        found[key] = str(path)
    return found


def pin_to_performance_cores() -> str:
    """Keep decode threads on the P-core group (Windows hybrid CPUs only).

    Windows 11's scheduler parks threads of boot-launched processes on E-cores;
    int8 decode there measured ~8x slower (9.6 s clip: 1.6 s -> 12.7 s). Query
    each core's EfficiencyClass and pin the process to the group with the
    highest value. No-op on non-hybrid CPUs and non-Windows hosts.
    """
    if not sys.platform.startswith("win"):
        return "non-Windows host"
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetLogicalProcessorInformationEx.argtypes = [
            ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetLogicalProcessorInformationEx.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
        size = wintypes.DWORD(0)
        kernel32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        if not kernel32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
            return f"query failed (err {kernel32.GetLastError()})"
        # RelationProcessorCore entries (48 bytes on Win11 hybrid):
        # +0 rel, +4 len, +9 EfficiencyClass, +32 affinity mask (u64).
        cores, off = [], 0
        while off < size.value:
            rel, ln = struct.unpack_from("<II", buf, off)
            if rel == 0 and ln >= 40:
                cores.append((buf[off + 9], struct.unpack_from("<Q", buf, off + 32)[0]))
            off += ln
        if not cores:
            return "no core info"
        best_class = max(c[0] for c in cores)
        masks = [m for eff, m in cores if eff == best_class]
        if not masks or any(m == 0 for m in masks) or len(set(masks)) != len(masks):
            return f"implausible core masks, skipping ({[hex(m) for m in masks]})"
        if all(eff == best_class for eff, _ in cores):
            return "not a hybrid CPU, leaving affinity alone"
        mask = 0
        for m in masks:
            mask |= m
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
            return f"SetProcessAffinityMask(0x{mask:X}) failed"
        return f"pinned to {mask.bit_count()} P-core logicals (mask 0x{mask:X})"
    except Exception as exc:  # never let the pin attempt break STT
        return f"skipped ({type(exc).__name__}: {exc})"


def build_recognizer(threads: int, files: dict[str, str]):
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=files["tokens"],
        encoder=files["encoder"],
        decoder=files["decoder"],
        joiner=files["joiner"],
        num_threads=threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        model_type="nemotron",
    )


def transcribe(wav_bytes: bytes, chunk_ms: int) -> str:
    """Stateless one-shot transcription of a complete WAV."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        if wf.getframerate() != 16000 or wf.getnchannels() != 1:
            raise ValueError(
                f"expect 16 kHz mono, got {wf.getframerate()} Hz x{wf.getnchannels()}")
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    pcm = pcm.astype(np.float32) / 32768.0

    stream = RECOGNIZER.create_stream()
    step = int(16000 * chunk_ms / 1000)
    with DECODE_LOCK:
        for i in range(0, len(pcm), step):
            stream.accept_waveform(16000, pcm[i:i + step])
            while RECOGNIZER.is_ready(stream):
                RECOGNIZER.decode_stream(stream)
        stream.accept_waveform(
            16000, np.zeros(int(16000 * TAIL_PAD_S), dtype=np.float32))
        stream.input_finished()
        while RECOGNIZER.is_ready(stream):
            RECOGNIZER.decode_stream(stream)
        return RECOGNIZER.get_result(stream).strip()


class LiveSession:
    """One utterance: a persistent sherpa-onnx stream that keeps its decoder state."""

    def __init__(self) -> None:
        self.stream = RECOGNIZER.create_stream()
        self.committed: list[str] = []
        self.lock = threading.Lock()
        self.touched = time.time()
        self.closed = False

    def _decode(self) -> None:
        with DECODE_LOCK:
            while RECOGNIZER.is_ready(self.stream):
                RECOGNIZER.decode_stream(self.stream)

    def push(self, pcm_bytes: bytes) -> str:
        usable = len(pcm_bytes) - len(pcm_bytes) % 2
        if usable <= 0:
            return self.partial()
        pcm = np.frombuffer(pcm_bytes[:usable], dtype=np.int16).astype(np.float32) / 32768.0
        with self.lock:
            self.stream.accept_waveform(16000, pcm)
            self._decode()
            if RECOGNIZER.is_endpoint(self.stream):
                text = RECOGNIZER.get_result(self.stream).strip()
                if text:
                    self.committed.append(text)
                RECOGNIZER.reset(self.stream)
        self.touched = time.time()
        return self.partial()

    def partial(self) -> str:
        with self.lock:
            current = RECOGNIZER.get_result(self.stream).strip()
        return " ".join(self.committed + ([current] if current else [])).strip()

    def final(self) -> str:
        with self.lock:
            self.stream.accept_waveform(
                16000, np.zeros(int(16000 * TAIL_PAD_S), dtype=np.float32))
            self.stream.input_finished()
            self._decode()
            current = RECOGNIZER.get_result(self.stream).strip()
            if current:
                self.committed.append(current)
            self.closed = True
        self.touched = time.time()
        return " ".join(self.committed).strip()


def _sweep_sessions() -> None:
    cutoff = time.time() - SESSION_TTL_S
    with SESSIONS_LOCK:
        for sid in [k for k, s in SESSIONS.items() if s.closed or s.touched < cutoff]:
            SESSIONS.pop(sid, None)


class Handler(BaseHTTPRequestHandler):
    chunk_ms = 560

    def log_message(self, fmt, *args):  # keep stderr for our own lines
        pass

    def _send(self, code: int, body: str):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code: int, payload: dict):
        self._send(code, json.dumps(payload))

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            self._send(200, "ok")
        else:
            self._send(404, "not found")

    def do_POST(self):
        parts = urlsplit(self.path)
        route = parts.path
        sid = (parse_qs(parts.query).get("sid") or [""])[0]

        if route == "/stream/open":
            try:
                _sweep_sessions()
                new_sid = uuid.uuid4().hex
                with SESSIONS_LOCK:
                    SESSIONS[new_sid] = LiveSession()
                print(f"[nemotron] stream open {new_sid[:8]}", flush=True)
                self._json(200, {"session_id": new_sid})
            except Exception as exc:
                print(f"[nemotron] ERROR {type(exc).__name__}: {exc}", flush=True)
                self._json(500, {"error": str(exc)})
            return

        if route in ("/stream/push", "/stream/final", "/stream/cancel"):
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(n) if n > 0 else b""
                with SESSIONS_LOCK:
                    session = SESSIONS.get(sid)
                if session is None:
                    self._json(404, {"error": "unknown session"})
                    return
                if route == "/stream/cancel":
                    session.closed = True
                    with SESSIONS_LOCK:
                        SESSIONS.pop(sid, None)
                    self._json(200, {"ok": True})
                    return
                if route == "/stream/push":
                    self._json(200, {"text": session.push(body)})
                    return
                t0 = time.time()
                text = session.final()
                with SESSIONS_LOCK:
                    SESSIONS.pop(sid, None)
                print(f"[nemotron] stream final {sid[:8]} "
                      f"{int((time.time()-t0)*1000)} ms, {len(text)} chars", flush=True)
                self._json(200, {"text": text})
            except Exception as exc:
                print(f"[nemotron] ERROR {type(exc).__name__}: {exc}", flush=True)
                self._json(500, {"error": str(exc)})
            return

        if route != "/infer":
            self._send(404, "not found")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n)
            t0 = time.time()
            text = transcribe(body, self.chunk_ms)
            print(f"[nemotron] infer {int((time.time()-t0)*1000)} ms, "
                  f"{len(text)} chars", flush=True)
            self._send(200, text)
        except Exception as exc:
            print(f"[nemotron] ERROR {type(exc).__name__}: {exc}", flush=True)
            self._send(500, f"error: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Resident Nemotron 0.6B streaming ASR server")
    ap.add_argument("--model-dir", default=str(default_model_dir()),
                    help="directory holding tokens.txt + encoder/decoder/joiner .onnx "
                         "(default: $NEMOTRON_MODEL_DIR or <repo>/model)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk-ms", type=int, default=560)
    args = ap.parse_args()

    global RECOGNIZER
    files = model_files(Path(args.model_dir).expanduser())
    print(f"[nemotron] model dir: {Path(args.model_dir).expanduser()}", flush=True)
    print(f"[nemotron] cores: {pin_to_performance_cores()}", flush=True)
    t0 = time.time()
    print("[nemotron] loading model...", flush=True)
    RECOGNIZER = build_recognizer(args.threads, files)
    print(f"[nemotron] model loaded in {time.time()-t0:.2f}s", flush=True)

    Handler.chunk_ms = args.chunk_ms
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[nemotron] listening on http://127.0.0.1:{args.port}/", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
