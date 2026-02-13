# Career Intelligence Assistant - Docker stop script
# Stops and removes containers

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Stopping containers..." -ForegroundColor Cyan
docker-compose down

if ($LASTEXITCODE -eq 0) {
    Write-Host "Containers stopped." -ForegroundColor Green
} else {
    exit 1
}
