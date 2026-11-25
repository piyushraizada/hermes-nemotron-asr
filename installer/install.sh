#!/usr/bin/env bash
# Install the Nemotron ASR server + Hermes plugin on macOS / Linux.
#
#   ./installer/install.sh [--port 8020] [--threads 4] [--no-hermes] [--no-start]
#
# Autostart is NOT registered here: it differs per distro. The script prints a
# systemd unit (Linux) or launchd hint (macOS) you can copy.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${NEMOTRON_MODEL_DIR:-$REPO/model}"
PORT=8020
THREADS=4
DO_HERMES=1
DO_START=1
MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --no-hermes) DO_HERMES=0; shift ;;
    --no-start) DO_START=0; shift ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

say() { printf '\033[36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!!  %s\033[0m\n' "$*"; }
die() { printf '\033[31mXX  %s\033[0m\n' "$*" >&2; exit 1; }

say "Checking prerequisites"
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)'; then PY="$c"; break; fi
  fi
done
[[ -n "$PY" ]] || die "Python 3.10+ not found"
say "Python: $($PY --version)"
command -v ffmpeg >/dev/null 2>&1 || warn "ffmpeg not found - the file-transcription fallback needs it"

# venv + deps
if [[ ! -x "$REPO/venv/bin/python" ]]; then
  say "Creating venv"
  "$PY" -m venv "$REPO/venv"
fi
VENV_PY="$REPO/venv/bin/python"
say "Installing sherpa-onnx and numpy"
"$VENV_PY" -m pip install --quiet --upgrade pip
"$VENV_PY" -m pip install --quiet sherpa-onnx numpy

# model
need_model=0
for f in tokens.txt encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx; do
  [[ -f "$MODEL_DIR/$f" ]] || need_model=1
done
if [[ $need_model -eq 1 ]]; then
  say "Downloading model (~442 MB, one time)"
  mkdir -p "$MODEL_DIR"
  tmp="$(mktemp -d)"
  if command -v curl >/dev/null 2>&1; then curl -fL --progress-bar "$MODEL_URL" -o "$tmp/model.tar.bz2"
  else wget -q --show-progress "$MODEL_URL" -O "$tmp/model.tar.bz2"; fi
  say "Extracting"
  tar -xjf "$tmp/model.tar.bz2" -C "$MODEL_DIR"
  # flatten the sherpa-onnx-<name>/ directory
  for d in "$MODEL_DIR"/*/; do
    if [[ -f "$d/tokens.txt" ]]; then
      mv "$d"/* "$MODEL_DIR"/ && rmdir "$d"
    fi
  done
  rm -rf "$tmp"
else
  say "Model already present in $MODEL_DIR"
fi

# start
if [[ $DO_START -eq 1 ]]; then
  say "Starting server on port $PORT"
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    say "already running"
  else
    nohup "$VENV_PY" "$REPO/server/nemotron_server.py" \
      --model-dir "$MODEL_DIR" --port "$PORT" --threads "$THREADS" \
      >> "$REPO/server.log" 2>&1 &
    for _ in $(seq 1 40); do
      sleep 0.5
      curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    done
  fi
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
    && say "Server healthy: http://127.0.0.1:$PORT/health" \
    || die "server did not come up - see $REPO/server.log"
fi

# hermes plugin
if [[ $DO_HERMES -eq 1 ]]; then
  HERMES_HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
  DEST="$HERMES_HOME_DIR/plugins/nemotron-stream"
  say "Installing plugin into $DEST"
  mkdir -p "$DEST"
  cp -r "$REPO/plugin/nemotron-stream/." "$DEST/"
  if command -v hermes >/dev/null 2>&1; then
    hermes plugins enable nemotron-stream || true
    hermes config set stt.provider nemotron
    hermes config set stt.streaming true
    say "Hermes configured: stt.provider=nemotron, stt.streaming=true"
  else
    warn "hermes not on PATH. After installing Hermes run:"
    warn "  hermes plugins enable nemotron-stream"
    warn "  hermes config set stt.provider nemotron"
    warn "  hermes config set stt.streaming true"
  fi
fi

cat <<EOF

Done. Start at login (optional):

  Linux (systemd user unit ~/.config/systemd/user/nemotron-asr.service):

    [Unit]
    Description=Nemotron ASR server
    After=network.target

    [Service]
    ExecStart=$VENV_PY $REPO/server/nemotron_server.py --model-dir $MODEL_DIR --port $PORT --threads $THREADS
    Restart=on-failure

    [Install]
    WantedBy=default.target

    then: systemctl --user daemon-reload && systemctl --user enable --now nemotron-asr

  macOS: create ~/Library/LaunchAgents/com.local.nemotron-asr.plist with the same
  program arguments, then: launchctl load ~/Library/LaunchAgents/com.local.nemotron-asr.plist

Smoke test (needs a speech recording):
  $VENV_PY tests/smoke_test.py --wav speech.wav
EOF
