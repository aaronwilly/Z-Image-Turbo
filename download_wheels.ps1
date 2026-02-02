# Run this ONCE while online to populate ./wheels for offline install.
# Requires: Python with pip (venv recommended).
# Downloads PyTorch 2.10.0 with CUDA 12.6 (2.10.0 wheels are on cu126 index).

$ErrorActionPreference = "Stop"
$wheelsDir = "wheels"
$torchVersion = "2.10.0"
$torchIndex = "https://download.pytorch.org/whl/cu126"

if (-not (Test-Path "requirements-offline.txt")) {
    Write-Error "requirements-offline.txt not found. Run from project root."
}

New-Item -ItemType Directory -Force -Path $wheelsDir | Out-Null

# Remove any existing torch wheels so we get exactly 2.10.0+cu126
Get-ChildItem -Path $wheelsDir -Filter "torch-*.whl" -ErrorAction SilentlyContinue | Remove-Item -Force

Write-Host "Downloading PyTorch $torchVersion (CUDA 12.6) to $wheelsDir..." -ForegroundColor Cyan
pip download "torch==$torchVersion" --index-url $torchIndex -d $wheelsDir

Write-Host "Downloading other deps to $wheelsDir..." -ForegroundColor Cyan
pip download -r requirements-offline.txt -d $wheelsDir

# Remove CPU torch if it was pulled as a dependency (keep only cu* torch)
Get-ChildItem -Path $wheelsDir -Filter "torch-*.whl" -ErrorAction SilentlyContinue | Where-Object { $_.Name -notmatch "cu\d+" } | Remove-Item -Force

Write-Host "Done. Copy the '$wheelsDir' folder and requirements-offline.txt to the offline machine, then run install_offline.ps1" -ForegroundColor Green
