# Kill any process using port 8000
# Run this if you get "Error 10048: only one usage of each socket address is normally permitted"

$port = 8000
$connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" }

if ($connections) {
    $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($pid in $pids) {
        $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
        Write-Host "Stopping process $pid ($($proc.ProcessName)) using port $port..."
        Stop-Process -Id $pid -Force
    }
    Write-Host "Port $port is now free. You can start the server."
} else {
    Write-Host "No process is using port $port."
}
