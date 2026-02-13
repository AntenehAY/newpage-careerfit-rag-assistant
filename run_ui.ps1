# Career Intelligence Assistant - Streamlit UI
# Run this from the project root. Start the API first with run_server.ps1.

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "ERROR: Virtual environment not found. Run: python -m venv .venv" -ForegroundColor Red
    exit 1
}

Write-Host "Starting Streamlit UI from: $ProjectRoot" -ForegroundColor Green
Write-Host "Make sure the API is running (run_server.ps1)" -ForegroundColor Cyan
Write-Host ""

& $PythonExe -m streamlit run ui/streamlit_app.py
