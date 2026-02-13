# Career Intelligence Assistant - Docker build script
# Builds images with no cache for clean production deployment

param(
    [switch]$WithCache
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($WithCache) {
    Write-Host "Building Docker images (with cache)..." -ForegroundColor Cyan
    docker-compose build
} else {
    Write-Host "Building Docker images (no cache)..." -ForegroundColor Cyan
    docker-compose build --no-cache
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build complete." -ForegroundColor Green
} else {
    Write-Host "Build failed." -ForegroundColor Red
    exit 1
}
