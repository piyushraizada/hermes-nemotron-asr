<#
.SYNOPSIS
    Install local CPU-only streaming speech-to-text (Nemotron 0.6B + Hermes plugin).

.DESCRIPTION
    Creates a venv, installs sherpa-onnx + numpy, downloads the int8 model,
    registers a hidden autostart launcher, starts the server, installs the
    `nemotron-stream` plugin into Hermes and points stt.provider at it.

.PARAMETER ModelDir
    Where the model lives. Default: <repo>\model

.PARAMETER Port
    Server port. Default 8020.

.PARAMETER Threads
    Decoder threads. Default 4 (best on 4 P-cores; RTF ~0.15).

.PARAMETER NoAutostart
    Do not register the logon launcher (start the server manually instead).

.PARAMETER NoHermes
    Only install the server; skip the plugin and stt config.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File installer\install.ps1
#>
[CmdletBinding()]
param(
    [string]$ModelDir,
    [int]$Port = 8020,
    [int]$Threads = 4,
    [switch]$NoAutostart,
    [switch]$NoHermes
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ModelDir = if ($ModelDir) { $ModelDir } else { Join-Path $RepoRoot "model" }
$VenvDir  = Join-Path $RepoRoot "venv"
$LogFile  = Join-Path $RepoRoot "server.log"
$ModelUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2"

function Say($msg)  { Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "!!  $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "XX  $msg" -ForegroundColor Red; exit 1 }

# ── prerequisites ────────────────────────────────────────────────────────────
Say "Checking prerequisites"
$Python = $null
foreach ($cand in @("python", "py")) {
    $cmd = Get-Command $cand -ErrorAction SilentlyContinue
    if ($cmd) {
        $ver = & $cmd.Source -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and [version]$ver -ge [version]"3.10") { $Python = $cmd.Source; break }
    }
}
if (-not $Python) { Fail "Python 3.10+ not found on PATH. Install it from python.org (tick 'Add to PATH')." }
Say "Python: $Python"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Warn "ffmpeg not found on PATH - the file-transcription fallback needs it."
    Warn "Install with: winget install Gyan.FFmpeg"
}

# ── venv + dependencies ──────────────────────────────────────────────────────
if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    Say "Creating venv at $VenvDir"
    & $Python -m venv $VenvDir
}
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"
Say "Installing sherpa-onnx and numpy (this pulls ~50 MB)"
& $VenvPy -m pip install --quiet --upgrade pip
& $VenvPy -m pip install --quiet sherpa-onnx numpy
if ($LASTEXITCODE -ne 0) { Fail "pip install failed" }

# ── model ────────────────────────────────────────────────────────────────────
$needed = @("tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx")
$haveModel = $true
foreach ($f in $needed) { if (-not (Test-Path (Join-Path $ModelDir $f))) { $haveModel = $false } }

if (-not $haveModel) {
    Say "Downloading model (~442 MB, one time)"
    New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
    $tar = Join-Path $env:TEMP "nemotron-0.6b-560ms-int8.tar.bz2"
    if (-not (Test-Path $tar)) {
        Invoke-WebRequest -Uri $ModelUrl -OutFile $tar -UseBasicParsing
    }
    Say "Extracting"
    & tar -xjf $tar -C $ModelDir
    if ($LASTEXITCODE -ne 0) { Fail "tar extraction failed" }
    # Flatten sherpa-onnx-<name>\ up one level
    Get-ChildItem -Path $ModelDir -Directory | ForEach-Object {
        if (Test-Path (Join-Path $_.FullName "tokens.txt")) {
            Get-ChildItem $_.FullName | Move-Item -Destination $ModelDir -Force
            Remove-Item $_.FullName -Recurse -Force
        }
    }
    Remove-Item $tar -Force -ErrorAction SilentlyContinue
    Say "Model ready in $ModelDir"
} else {
    Say "Model already present in $ModelDir"
}

# ── server launcher ─────────────────────────────────────────────────────────
$ServerPy = Join-Path $RepoRoot "server\nemotron_server.py"
$RunCmd   = Join-Path $RepoRoot "run-server.cmd"

Say "Writing launcher $RunCmd"
$cmdContent = @"
@echo off
"$VenvPy" "$ServerPy" --model-dir "$ModelDir" --port $Port --threads $Threads >> "$LogFile" 2>&1
"@
Set-Content -Path $RunCmd -Value $cmdContent -Encoding ASCII

if (-not $NoAutostart) {
    $startup = [Environment]::GetFolderPath("Startup")
    $vbs = Join-Path $startup "Nemotron ASR Server.vbs"
    # VBScript doubles its own quotes; build the line by concatenation to keep it readable.
    $runLine = 'sh.Run ""' + $RunCmd + '""", 0, False'
    $content = "' Starts the resident Nemotron ASR server hidden (port $Port)." + "`r`n" +
               'Set sh = CreateObject("WScript.Shell")' + "`r`n" + $runLine + "`r`n"
    Set-Content -Path $vbs -Value $content -Encoding ASCII
    Say "Autostart registered: $vbs"
}

# ── start it now ────────────────────────────────────────────────────────────
Say "Starting server on port $Port"
$health = "http://127.0.0.1:$Port/health"
$already = $false
try { $already = (Invoke-WebRequest -Uri $health -TimeoutSec 2 -UseBasicParsing).Content -match "ok" } catch { }

if (-not $already) {
    Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", "`"$RunCmd`"") -WindowStyle Hidden
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 500
        try { if ((Invoke-WebRequest -Uri $health -TimeoutSec 2 -UseBasicParsing).Content -match "ok") { break } } catch { }
    }
}
try {
    $ok = (Invoke-WebRequest -Uri $health -TimeoutSec 3 -UseBasicParsing).Content -match "ok"
} catch { $ok = $false }
if ($ok) { Say "Server healthy: $health" } else { Fail "Server did not come up - see $LogFile" }

# ── Hermes plugin ───────────────────────────────────────────────────────────
if (-not $NoHermes) {
    $hermesHome = if ($env:HERMES_HOME) { $env:HERMES_HOME } else { Join-Path $env:LOCALAPPDATA "hermes" }
    $pluginDst = Join-Path $hermesHome "plugins\nemotron-stream"
    Say "Installing plugin into $pluginDst"
    New-Item -ItemType Directory -Force -Path $pluginDst | Out-Null
    Copy-Item (Join-Path $RepoRoot "plugin\nemotron-stream\*") $pluginDst -Recurse -Force

    if (Get-Command hermes -ErrorAction SilentlyContinue) {
        & hermes plugins enable nemotron-stream
        & hermes config set stt.provider nemotron
        & hermes config set stt.streaming true
        Say "Hermes configured: stt.provider=nemotron, stt.streaming=true"
    } else {
        Warn "hermes CLI not on PATH. After installing Hermes, run:"
        Warn "  hermes plugins enable nemotron-stream"
        Warn "  hermes config set stt.provider nemotron"
        Warn "  hermes config set stt.streaming true"
    }
}

Say "Done. Press the mic in Hermes and talk - text should appear as you speak."
Say "Server log: $LogFile"
