# Career Intelligence Assistant - Docker run script
# Starts all services in detached mode

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".env")) {
    Write-Host "WARNING: .env not found. Copy .env.docker.example to .env and add your API keys." -ForegroundColor Yellow
}

Write-Host "Starting containers..." -ForegroundColor Cyan
docker-compose up -d

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Services started:" -ForegroundColor Green
    Write-Host "  API:  http://localhost:8000" -ForegroundColor White
    Write-Host "  UI:   http://localhost:8501" -ForegroundColor White
    Write-Host ""
    Write-Host "Check status: docker-compose ps" -ForegroundColor Gray
} else {
    exit 1
}
