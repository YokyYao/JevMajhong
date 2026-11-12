param([string]$PythonPath = 'python')
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (!(Test-Path (Join-Path $ProjectRoot 'desktop\jev_sidecar'))) { $ProjectRoot = $PSScriptRoot }
Set-Location $ProjectRoot
$ResolvedPython = (Get-Command $PythonPath -ErrorAction Stop).Source
$BuildPath = $env:PATH
try {
    # Qt uses the Windows ICU API. Unrelated Poppler/libheif DLLs on PATH
    # can otherwise be bundled under the same names with incompatible exports.
    $env:PATH = "$(Split-Path $ResolvedPython);$env:SystemRoot\System32;$env:SystemRoot"
    & $ResolvedPython -m PyInstaller --noconfirm --clean desktop/windows/JevMahjong.spec
    if ($LASTEXITCODE -ne 0) { throw 'Windows executable build failed.' }
} finally {
    $env:PATH = $BuildPath
}
& $PythonPath tools/package_runtime.py --dist dist/JevMahjong
if ($LASTEXITCODE -ne 0) { throw 'Runtime packaging failed.' }
