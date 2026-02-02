# Offline install from local wheels. No network used.
# Requires: ./wheels folder and requirements-offline.txt (from download_wheels.ps1).
# Installs PyTorch (CUDA) from wheels first, then the rest.

$ErrorActionPreference = "Stop"

if (-not (Test-Path "wheels")) {
    Write-Error "Folder 'wheels' not found. Run download_wheels.ps1 on a connected machine first, then copy wheels + requirements-offline.txt here."
}
if (-not (Test-Path "requirements-offline.txt")) {
    Write-Error "requirements-offline.txt not found."
}

# Install the CUDA torch 2.10.0 wheel explicitly (wheels may contain both CPU and CUDA)
$cudaTorch = Get-ChildItem -Path "wheels" -Filter "torch*cu*.whl" -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "2\.10\.0" } | Select-Object -First 1
if (-not $cudaTorch) {
    $cudaTorch = Get-ChildItem -Path "wheels" -Filter "torch*cu*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
}
if (-not $cudaTorch) {
    Write-Error "No PyTorch CUDA wheel (torch*cu*.whl) found in wheels. Re-run download_wheels.ps1 while online to get torch 2.10.0+cu126."
}
Write-Host "Installing PyTorch (CUDA) from $($cudaTorch.Name)..." -ForegroundColor Cyan
pip install --no-index "$($cudaTorch.FullName)"

Write-Host "Installing other deps from wheels..." -ForegroundColor Cyan
pip install --no-index --find-links=wheels -r requirements-offline.txt

Write-Host "Offline install finished." -ForegroundColor Green
