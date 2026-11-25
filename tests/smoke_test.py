"""Smoke test for the Nemotron ASR server: batch /infer and live /stream/*.

Usage:
    python tests/smoke_test.py --wav speech.wav [--url http://127.0.0.1:8020]

Exit code 0 means both paths produced text and they agree. Requires ffmpeg on
PATH (or FFMPEG_BIN). The wav may be any sample rate / channel count.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

CHUNK_BYTES = 17920  # 560 ms of 16 kHz mono s16le


def post(url: str, path: str, data: bytes = b"", timeout: float = 180.0) -> bytes:
    req = urllib.request.Request(url + path, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="speech recording to transcribe")
    ap.add_argument("--url", default=os.environ.get("NEMOTRON_ASR_URL", "http://127.0.0.1:8020"))
    ap.add_argument("--keep", action="store_true", help="keep the converted files")
    args = ap.parse_args()

    url = args.url.rstrip("/")
    ffmpeg = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"
    tmp = Path(tempfile.mkdtemp(prefix="nemotron-smoke-"))
    wav16, raw = tmp / "take16.wav", tmp / "take16.raw"
    for out, fmt in ((wav16, "wav"), (raw, "s16le")):
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", args.wav,
                        "-ar", "16000", "-ac", "1", "-f", fmt, str(out)], check=True)

    batch = post(url, "/infer", wav16.read_bytes()).decode("utf-8", "replace").strip()
    print("BATCH:", batch or "(empty)")

    sid = json.loads(post(url, "/stream/open"))["session_id"]
    pcm = raw.read_bytes()
    partials = []
    for i in range(0, len(pcm), CHUNK_BYTES):
        text = json.loads(post(url, f"/stream/push?sid={sid}", pcm[i:i + CHUNK_BYTES])).get("text", "")
        if text and (not partials or partials[-1] != text):
            partials.append(text)
            print(f"PARTIAL @{i/32000:5.2f}s: {text}")
    final = json.loads(post(url, f"/stream/final?sid={sid}")).get("text", "").strip()
    print("FINAL:", final or "(empty)")

    ok = bool(batch) and bool(final) and batch[:20].lower() == final[:20].lower()
    print("RESULT:", "PASS" if ok else "FAIL")
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
