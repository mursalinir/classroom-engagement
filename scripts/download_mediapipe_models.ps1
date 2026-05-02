$ProgressPreference = "SilentlyContinue"

$targetDir = "models\mediapipe"
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

$downloads = @(
  @{
    Url = "https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2.task"
    Output = Join-Path $targetDir "face_landmarker_v2.task"
  },
  @{
    Url = "https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task"
    Output = Join-Path $targetDir "pose_landmarker.task"
  }
)

foreach ($item in $downloads) {
  curl.exe -L $item.Url -o $item.Output
}

Write-Host "MediaPipe task models downloaded to $targetDir"
