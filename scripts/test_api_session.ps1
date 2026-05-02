. .\.venv\Scripts\Activate.ps1

$base = "http://127.0.0.1:8000"

Invoke-RestMethod -Uri "$base/session/stop" -Method Post | Out-Null
Invoke-RestMethod -Uri "$base/session/start" -Method Post -ContentType "application/json" -Body '{"source_type":"synthetic","source_uri":"demo","detector_name":"yolox_onnx","tracker_name":"bytetrack"}'
Start-Sleep -Milliseconds 500
Invoke-RestMethod -Uri "$base/session/status" -Method Get
Invoke-RestMethod -Uri "$base/session/snapshot" -Method Get
Write-Host "Annotated frame URL: $base/session/frame"
