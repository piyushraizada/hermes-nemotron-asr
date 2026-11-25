"""Exercise the plugin provider without a running Hermes gateway.

Stubs the two base classes the provider subclasses, then drives both paths
against a live server:

    python tests/plugin_check.py --wav speech.wav [--url http://127.0.0.1:8020]

Useful for contributors and for checking a server change before installing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

PLUGIN = Path(__file__).resolve().parent.parent / "plugin" / "nemotron-stream" / "__init__.py"
CHUNK_BYTES = 17920


def load_plugin():
    pkg = types.ModuleType("agent")
    pkg.__path__ = []
    base = types.ModuleType("agent.transcription_provider")

    class TranscriptionProvider:
        pass

    class TranscriptionStreamSession:
        pass

    base.TranscriptionProvider = TranscriptionProvider
    base.TranscriptionStreamSession = TranscriptionStreamSession
    sys.modules["agent"] = pkg
    sys.modules["agent.transcription_provider"] = base

    spec = importlib.util.spec_from_file_location("nemotron_stream_check", PLUGIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--url", default=os.environ.get("NEMOTRON_ASR_URL", "http://127.0.0.1:8020"))
    args = ap.parse_args()
    os.environ["NEMOTRON_ASR_URL"] = args.url.rstrip("/")

    mod = load_plugin()
    provider = mod.NemotronProvider()
    print("provider:", provider.name, "| streaming:", provider.streaming_capable,
          "| server:", mod.server_url())

    t0 = time.time()
    file_result = provider.transcribe(args.wav)
    print(f"FILE PATH ({time.time()-t0:.2f}s): {json.dumps(file_result)[:240]}")

    ffmpeg = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"
    tmp = Path(tempfile.mkdtemp(prefix="nemotron-plugin-"))
    raw = tmp / "take.raw"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", args.wav,
                    "-ar", "16000", "-ac", "1", "-f", "s16le", str(raw)], check=True)

    t0 = time.time()
    session = provider.open_stream_session()
    pcm = raw.read_bytes()
    last = ""
    for i in range(0, len(pcm), CHUNK_BYTES):
        session.push_audio(pcm[i:i + CHUNK_BYTES])
        partial = session.partial_transcript()
        if partial != last:
            last = partial
            print(f"  partial @{i/32000:5.2f}s: {partial}")
    final = session.finalize()
    print(f"STREAM ({time.time()-t0:.2f}s): {json.dumps(final)[:240]}")
    shutil.rmtree(tmp, ignore_errors=True)

    ok = (file_result.get("success") and final.get("success")
          and file_result["transcript"][:20].lower() == final["transcript"][:20].lower())
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
