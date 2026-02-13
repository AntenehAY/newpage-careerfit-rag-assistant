# Career Intelligence Assistant - Docker logs script
# Follow logs for API (or specify service: .\logs.ps1 ui)

param(
    [string]$Service = "api"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Following logs for: $Service" -ForegroundColor Cyan
docker-compose logs -f $Service
