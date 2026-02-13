# Career Intelligence Assistant - Server startup script
# Run this from the project root. Ensures correct working directory.

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "ERROR: Virtual environment not found. Run: python -m venv .venv" -ForegroundColor Red
    exit 1
}

Write-Host "Starting server from: $ProjectRoot" -ForegroundColor Green
Write-Host "Open http://127.0.0.1:8000/ in your browser" -ForegroundColor Cyan
Write-Host ""

& $PythonExe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
