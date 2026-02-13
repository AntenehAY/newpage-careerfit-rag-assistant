@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found. Run: python -m venv .venv
    exit /b 1
)

echo Starting server from: %CD%
echo Open http://127.0.0.1:8000/ in your browser
echo.

.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
