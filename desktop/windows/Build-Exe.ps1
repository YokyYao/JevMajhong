param([string]$PythonPath = 'python')
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (!(Test-Path (Join-Path $ProjectRoot 'desktop\jev_sidecar'))) { $ProjectRoot = $PSScriptRoot }
Set-Location $ProjectRoot
& $PythonPath -m PyInstaller --noconfirm --clean desktop/windows/JevMahjong.spec
if ($LASTEXITCODE -ne 0) { throw 'Windows executable build failed.' }
& $PythonPath tools/package_runtime.py --dist dist/JevMahjong
if ($LASTEXITCODE -ne 0) { throw 'Runtime packaging failed.' }
