$ErrorActionPreference = 'Stop'
if (Test-Path (Join-Path $PSScriptRoot 'desktop\requirements-ui.txt')) { $ProjectRoot = $PSScriptRoot }
else { $ProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent }
Set-Location $ProjectRoot
$PythonPath = Join-Path $ProjectRoot '.desktop-venv\Scripts\python.exe'
& $PythonPath -m pip install 'pyinstaller>=6,<7'
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller installation failed.' }
& $PythonPath -m PyInstaller --noconfirm --clean --onedir --windowed --name JevMahjong --collect-all onnxruntime --collect-submodules mahjong --collect-submodules jevany --paths . desktop/launcher.py
if ($LASTEXITCODE -ne 0) { throw 'Windows executable build failed.' }
Copy-Item -Recurse -Force assets dist/JevMahjong/assets
Copy-Item -Recurse -Force models dist/JevMahjong/models
Copy-Item README-Windows.md dist/JevMahjong/README-Windows.md
Copy-Item -Recurse -Force desktop dist/JevMahjong/desktop-source
Write-Host 'Windows build: dist/JevMahjong/JevMahjong.exe'
