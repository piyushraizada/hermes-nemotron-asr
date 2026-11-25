<#
.SYNOPSIS
    Remove the Nemotron ASR server, its autostart entry and the Hermes plugin.

.PARAMETER Purge
    Also delete venv\, model\, logs and the generated launcher.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File installer\uninstall.ps1 -Purge
#>
[CmdletBinding()]
param([switch]$Purge)

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

# 1. stop the server
Say "Stopping server processes"
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*nemotron_server.py*" } |
    ForEach-Object {
        try { Stop-Process -Id $_.ProcessId -Force; Say "stopped PID $($_.ProcessId)" } catch { }
    }

# 2. autostart
$vbs = Join-Path ([Environment]::GetFolderPath("Startup")) "Nemotron ASR Server.vbs"
if (Test-Path $vbs) { Remove-Item $vbs -Force; Say "removed autostart $vbs" }

# 3. plugin + config
$hermesHome = if ($env:HERMES_HOME) { $env:HERMES_HOME } else { Join-Path $env:LOCALAPPDATA "hermes" }
$plugin = Join-Path $hermesHome "plugins\nemotron-stream"
if (Test-Path $plugin) { Remove-Item $plugin -Recurse -Force; Say "removed plugin $plugin" }

if (Get-Command hermes -ErrorAction SilentlyContinue) {
    & hermes config set stt.provider local   | Out-Null
    & hermes config set stt.streaming false  | Out-Null
    Say "stt.provider restored to local"
} else {
    Write-Host "!!  hermes not on PATH - set stt.provider back to 'local' manually"
}

# 4. optional purge
if ($Purge) {
    foreach ($item in @("venv", "model", "server.log", "server.log.err", "run-server.cmd")) {
        $path = Join-Path $RepoRoot $item
        if (Test-Path $path) { Remove-Item $path -Recurse -Force; Say "deleted $item" }
    }
}

Say "Done."
