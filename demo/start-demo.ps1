<#
    Starts the whole demo with one command: the detector service, plus a
    Cloudflare tunnel that gives it a public HTTPS URL, and prints that URL.

    Exists because the manual route produced three separate failures in a
    row, all of them avoidable:
      - the example checkpoint path got pasted verbatim (it was a placeholder)
      - `cd` and the python command were run as two shells, so the relative
        `.\.venv\...` path did not resolve
      - port 8000 was already in use, and the error did not say by what

    So this script locates the checkpoint itself, uses absolute paths, and
    checks the port before binding.

    Run it:
        powershell -ExecutionPolicy Bypass -File demo\start-demo.ps1

    Stop everything with Ctrl+C.
#>

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo "training\.venv\Scripts\python.exe"
$trainingDir = Join-Path $repo "training"
$port = 8000

Write-Host ""
Write-Host "=== bookshelf detector demo ===" -ForegroundColor Cyan
Write-Host ""

# --- the interpreter -------------------------------------------------------
if (-not (Test-Path $python)) {
    Write-Host "Cannot find the project's Python at:" -ForegroundColor Red
    Write-Host "  $python"
    Write-Host "Expected training\.venv to exist. Nothing else will work without it."
    exit 1
}

# --- the checkpoint --------------------------------------------------------
# Explicit env var wins; otherwise take the newest checkpoint_epoch_*.pt from
# Downloads, which is where a file pulled out of Drive lands.
$checkpoint = $env:CHECKPOINT
if (-not $checkpoint) {
    $candidates = @()
    foreach ($dir in @("$HOME\Downloads", $repo)) {
        if (Test-Path $dir) {
            $candidates += Get-ChildItem -Path $dir -Filter "checkpoint_epoch_*.pt" -Recurse -ErrorAction SilentlyContinue
        }
    }
    if ($candidates.Count -gt 0) {
        $checkpoint = ($candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
        Write-Host "checkpoint (auto-detected): $checkpoint" -ForegroundColor Green
    }
} else {
    Write-Host "checkpoint (from CHECKPOINT): $checkpoint" -ForegroundColor Green
}

if (-not $checkpoint -or -not (Test-Path $checkpoint)) {
    Write-Host "No checkpoint found." -ForegroundColor Red
    Write-Host "Download one from Drive (checkpoints_ctrl14\checkpoint_epoch_004.pt)"
    Write-Host "into your Downloads folder, then run this again. Or set it by hand:"
    Write-Host '  $env:CHECKPOINT="C:\full\path\to\checkpoint.pt"'
    exit 1
}

$sizeMb = [math]::Round((Get-Item $checkpoint).Length / 1MB)
Write-Host "  ($sizeMb MB)"

# --- the port --------------------------------------------------------------
# Binding a busy port fails with WinError 10048, whose message does not say
# which process holds it. Say so, and offer to take it over.
$busy = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    $owner = (Get-Process -Id $busy.OwningProcess -ErrorAction SilentlyContinue).ProcessName
    Write-Host ""
    Write-Host "Port $port is already in use by $owner (PID $($busy.OwningProcess))." -ForegroundColor Yellow
    Write-Host "That is probably an older copy of this same server."
    $answer = Read-Host "Stop it and take over the port? [y/N]"
    if ($answer -eq "y") {
        Stop-Process -Id $busy.OwningProcess -Force
        Start-Sleep -Seconds 2
        Write-Host "stopped." -ForegroundColor Green
    } else {
        Write-Host "Leaving it alone. If it is already serving the right checkpoint,"
        Write-Host "you do not need this script at all -- just open the tunnel URL."
        exit 0
    }
}

# --- start the service -----------------------------------------------------
Write-Host ""
Write-Host "starting detector (first load takes ~40s, it is a 336MB model)..." -ForegroundColor Cyan

$env:CHECKPOINT = $checkpoint
$server = Start-Process -FilePath $python `
    -ArgumentList @("-m", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "$port") `
    -WorkingDirectory $trainingDir -PassThru -NoNewWindow

# Poll /health rather than sleeping a fixed amount: model load time varies
# with disk cache, and a fixed sleep is either too short or wastes the demo's
# opening seconds.
$ready = $false
foreach ($attempt in 1..60) {
    Start-Sleep -Seconds 2
    if ($server.HasExited) {
        Write-Host "The server exited during startup -- see its output above." -ForegroundColor Red
        exit 1
    }
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3
        $ready = $true
        break
    } catch { }
}

if (-not $ready) {
    Write-Host "Server did not answer /health in two minutes. Stopping." -ForegroundColor Red
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "detector ready on $($health.device)." -ForegroundColor Green

# --- start the tunnel ------------------------------------------------------
$cloudflared = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cloudflared) {
    foreach ($guess in @("${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe",
                         "$env:ProgramFiles\cloudflared\cloudflared.exe")) {
        if (Test-Path $guess) { $cloudflared = $guess; break }
    }
}

if (-not $cloudflared) {
    Write-Host ""
    Write-Host "cloudflared not found, so no public URL." -ForegroundColor Yellow
    Write-Host "Install it with:  winget install --id Cloudflare.cloudflared"
    Write-Host "The demo still works locally at http://localhost:$port"
    Write-Host ""
    Write-Host "Ctrl+C to stop." -ForegroundColor Cyan
    Wait-Process -Id $server.Id
    exit 0
}

Write-Host "opening public tunnel..." -ForegroundColor Cyan
$tunnelLog = Join-Path $env:TEMP "bookshelf-tunnel.log"
if (Test-Path $tunnelLog) { Remove-Item $tunnelLog -Force }

$tunnel = Start-Process -FilePath $cloudflared `
    -ArgumentList @("tunnel", "--url", "http://localhost:$port") `
    -PassThru -NoNewWindow -RedirectStandardError $tunnelLog

$publicUrl = $null
foreach ($attempt in 1..30) {
    Start-Sleep -Seconds 2
    if (Test-Path $tunnelLog) {
        $match = Select-String -Path $tunnelLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" `
                 -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($match) { $publicUrl = $match.Matches[0].Value; break }
    }
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
if ($publicUrl) {
    Write-Host "  DEMO IS LIVE:" -ForegroundColor Green
    Write-Host "  $publicUrl" -ForegroundColor White
    Write-Host ""
    Write-Host "  Open it on a phone and photograph a shelf."
    Write-Host "  This URL changes every time you restart the tunnel."
} else {
    Write-Host "  Tunnel started but its URL did not appear in the log." -ForegroundColor Yellow
    Write-Host "  Check $tunnelLog"
    Write-Host "  Local demo works regardless: http://localhost:$port"
}
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Ctrl+C to stop both." -ForegroundColor Cyan

try {
    Wait-Process -Id $server.Id
} finally {
    foreach ($id in @($tunnel.Id, $server.Id)) {
        Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "stopped." -ForegroundColor Cyan
}
