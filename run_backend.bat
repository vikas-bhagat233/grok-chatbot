@echo off
setlocal
cd /d %~dp0

echo Starting backend on http://127.0.0.1:8000
echo Open the UI at: http://127.0.0.1:8000/ui
echo Keep this window open while using the chatbot
echo.

set "PYEXE=python"
if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"

REM Start uvicorn in its own window so it won't get terminated by the parent shell.
start "Grok Backend" cmd /k "%PYEXE% -m uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info"

REM Try to open the UI automatically.
start "" "http://127.0.0.1:8000/ui"

echo.
echo Backend launched in a separate window.
echo If the UI says "Failed to reach backend", ensure the "Grok Backend" window is still running.
echo Press any key to close this launcher...
pause >nul
