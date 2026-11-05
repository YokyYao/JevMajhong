@echo off
cd /d "%~dp0"
if exist "..\..\desktop\jev_sidecar\app.py" cd /d "%~dp0..\.."
if not exist ".desktop-venv\Scripts\pythonw.exe" (
  echo Please run Install.cmd first.
  pause
  exit /b 1
)
start "Jev Mahjong" ".desktop-venv\Scripts\pythonw.exe" -m desktop.launcher
