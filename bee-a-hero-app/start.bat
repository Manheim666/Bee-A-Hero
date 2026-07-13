@echo off
REM One-command launcher for Windows.
REM Sets up backend venv, installs deps, seeds the DB, starts backend + frontend,
REM and opens the browser. Re-run any time — it skips work that's already done.

setlocal ENABLEDELAYEDEXPANSION

set "ROOT_DIR=%~dp0"
set "BACKEND_DIR=%ROOT_DIR%backend"
set "FRONTEND_DIR=%ROOT_DIR%frontend"
set "VENV_DIR=%BACKEND_DIR%\.venv"
set "LOG_DIR=%ROOT_DIR%.logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM --- prerequisites ----------------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
  echo [X] Python is not installed. Install Python 3.11+ from https://www.python.org/ and re-run.
  exit /b 1
)
where node >nul 2>&1
if errorlevel 1 (
  echo [X] Node.js is not installed. Install Node 20+ from https://nodejs.org/ and re-run.
  exit /b 1
)

REM --- backend ---------------------------------------------------------------
if not exist "%VENV_DIR%" (
  echo ^> Creating Python venv...
  python -m venv "%VENV_DIR%"
)

set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PIP=%VENV_DIR%\Scripts\pip.exe"

if not exist "%VENV_DIR%\.deps-installed" (
  echo ^> Installing backend deps...
  "%PIP%" install --upgrade pip >nul
  "%PIP%" install -r "%BACKEND_DIR%\requirements.txt"
  echo done > "%VENV_DIR%\.deps-installed"
)

if not exist "%BACKEND_DIR%\.env" (
  copy "%BACKEND_DIR%\.env.example" "%BACKEND_DIR%\.env" >nul
  echo ^> Created backend\.env from example ^(set ANTHROPIC_API_KEY there for real AI^).
)

if not exist "%BACKEND_DIR%\bee.db" (
  echo ^> Seeding demo user + sample video...
  pushd "%BACKEND_DIR%"
  "%PYTHON%" -m seed
  popd
)

echo ^> Starting backend on http://localhost:8000 ...
start "Bee-A-Hero backend" /min cmd /c "cd /d %BACKEND_DIR% && %VENV_DIR%\Scripts\uvicorn.exe app.main:app --port 8000 > %LOG_DIR%\backend.log 2>&1"

REM --- frontend --------------------------------------------------------------
if not exist "%FRONTEND_DIR%\node_modules" (
  echo ^> Installing frontend deps ^(first run only^)...
  pushd "%FRONTEND_DIR%"
  call npm install
  popd
)

echo ^> Starting frontend on http://localhost:5173 ...
start "Bee-A-Hero frontend" /min cmd /c "cd /d %FRONTEND_DIR% && npm run dev > %LOG_DIR%\frontend.log 2>&1"

REM Wait a few seconds, then open browser.
timeout /t 5 /nobreak >nul
start "" http://localhost:5173/

echo.
echo ---------------------------------------------
echo   Bee-A-Hero is running
echo     App:  http://localhost:5173
echo     API:  http://localhost:8000/docs
echo     Login: demo@bee.dev / beehero123
echo   Logs: %LOG_DIR%\backend.log, frontend.log
echo   Close the two "Bee-A-Hero" windows to stop.
echo ---------------------------------------------
endlocal
