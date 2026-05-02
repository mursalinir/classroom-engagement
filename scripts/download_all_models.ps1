$ProgressPreference = "SilentlyContinue"

$downloads = @(
  @{
    Url = "https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2.task"
    Output = "models\mediapipe\face_landmarker_v2.task"
  },
  @{
    Url = "https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task"
    Output = "models\mediapipe\pose_landmarker.task"
  },
  @{
    Url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx"
    Output = "models\onnx\yolox_tiny.onnx"
  },
  @{
    Url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
    Output = "models\onnx\yolox_s.onnx"
  },
  @{
    Url = "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx?download=true"
    Output = "models\onnx\facial_expression_recognition_mobilefacenet_2022july.onnx"
  },
  @{
    Url = "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx?download=true"
    Output = "models\onnx\face_detection_yunet_2023mar.onnx"
  }
)

foreach ($item in $downloads) {
  $targetDir = Split-Path -Parent $item.Output
  New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

  if (Test-Path $item.Output) {
    Write-Host "Exists:" $item.Output
    continue
  }

  Write-Host "Downloading:" $item.Output
  curl.exe -L $item.Url -o $item.Output
}

Write-Host "All local model assets are prepared."
