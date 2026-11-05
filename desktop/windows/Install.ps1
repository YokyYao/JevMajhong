$ErrorActionPreference = 'Stop'
if (Test-Path (Join-Path $PSScriptRoot 'desktop\requirements-ui.txt')) {
    $ProjectRoot = $PSScriptRoot
} else {
    $ProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
}
Set-Location $ProjectRoot
$EnvironmentRoot = Join-Path $ProjectRoot '.desktop-venv'
$PythonPath = Join-Path $EnvironmentRoot 'Scripts\python.exe'
if (!(Test-Path $PythonPath)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv $EnvironmentRoot
    } else {
        Write-Host 'Python 3.12 is required. Install Python 3.12 (64-bit), including the Python Launcher, then run Install.cmd again.'
        Start-Process 'https://www.python.org/downloads/windows/'
        exit 1
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python 3.12 environment.' }
}
& $PythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip setup failed.' }
& $PythonPath -m pip install -r desktop/requirements-ui.txt
if ($LASTEXITCODE -ne 0) { throw 'Desktop dependencies failed to install.' }
Write-Host 'Installing NVIDIA CUDA PyTorch. This download may be several GB.'
& $PythonPath -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
& $PythonPath -m pip install -r desktop/requirements-model.txt
if ($LASTEXITCODE -ne 0) { throw 'Jev dependencies failed to install.' }
& $PythonPath desktop/fetch_assets.py
if ($LASTEXITCODE -ne 0) { throw 'Tile detector download or verification failed.' }
& $PythonPath -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Driver or CUDA PyTorch check required')"
Write-Host 'Installation finished. Double-click Launch.cmd.'
