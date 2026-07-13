@echo off
setlocal ENABLEDELAYEDEXPANSION

set "ROOT_DIR=%~dp0"
set "VENV_DIR=%ROOT_DIR%.venv"

where python >nul 2>&1
if errorlevel 1 (
  echo [X] Python is not installed. Install Python 3.11+ from https://www.python.org/
  exit /b 1
)

if not exist "%VENV_DIR%" (
  echo ^> Creating venv...
  python -m venv "%VENV_DIR%"
)

set "PY=%VENV_DIR%\Scripts\python.exe"
set "PIP=%VENV_DIR%\Scripts\pip.exe"

if not exist "%VENV_DIR%\.deps-installed" (
  echo ^> Installing deps ^(first run may take a couple minutes for torch^)...
  "%PIP%" install --upgrade pip >nul
  "%PIP%" install -r "%ROOT_DIR%requirements.txt"
  echo done > "%VENV_DIR%\.deps-installed"
)

if not exist "%ROOT_DIR%.env" (
  copy "%ROOT_DIR%.env.example" "%ROOT_DIR%.env" >nul
  echo ^> Created .env — edit DROIDCAM_URL to point at your phone, then re-run.
  echo   File: %ROOT_DIR%.env
  exit /b 0
)

echo ^> Starting server on http://localhost:8001 ...
start "" http://localhost:8001/
cd /d "%ROOT_DIR%" && "%VENV_DIR%\Scripts\uvicorn.exe" app.main:app --host 0.0.0.0 --port 8001
endlocal
