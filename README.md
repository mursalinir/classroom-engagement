# Classroom Engagement Analytics

Real-time classroom analytics demo for student counting, exit monitoring, face-based engagement cues, posture heuristics, and class-state estimation.

This repository is designed as both:

- a working local computer vision demo
- a teaching example of how to structure a professional ML product

Related docs:

- [design.md](./design.md)
- [document.md](./document.md)
- [technical_document.md](./technical_document.md)

## What This Project Does

The system reads a live or recorded video source and estimates:

- student count
- visible face count
- tracked person IDs
- scene departures
- exit-zone events
- attention, posture, interaction, confusion, and face-visibility signals
- overall class state such as `neutral`, `engaged`, `confused`, `energized`, or `disengaged`

Outputs are available in:

- a Streamlit dashboard
- a FastAPI backend

## Current Technical Approach

The project uses a modular pipeline:

`video source -> detector -> tracker -> signal analyzers -> class snapshot -> class state -> UI/API`

Current implementation:

- detector: `hybrid_person`
- person detector options:
  - `yolox_onnx`
  - `hog_person`
  - `motion_blob`
- tracker: `bytetrack` or `centroid`
- face detector: `YuNet`
- face expression model: ONNX MobileFaceNet FER
- pose: lightweight heuristic over the person crop
- class state: rule-based engine

Important note:

- this is a real-time engineering demo, not a clinically or academically validated sentiment system
- class-state inference is built from multiple weak signals, not one perfect model
- the user-facing app now supports `webcam` and `video_file` inputs

## Repository Layout

Main folders:

```text
apps/
  api/
  streamlit_app/

configs/
  app.yaml
  models.yaml
  rules.yaml

src/classroom_engagement/
  api/
  core/
  pipeline/
  ui/
  video/

models/
  onnx/
  mediapipe/
  openvino/

scripts/
tests/
data/
```

Important files:

- [apps/streamlit_app/app.py](./apps/streamlit_app/app.py)
- [apps/api/main.py](./apps/api/main.py)
- [src/classroom_engagement/pipeline/engine.py](./src/classroom_engagement/pipeline/engine.py)
- [src/classroom_engagement/pipeline/components.py](./src/classroom_engagement/pipeline/components.py)
- [src/classroom_engagement/pipeline/signals.py](./src/classroom_engagement/pipeline/signals.py)
- [configs/models.yaml](./configs/models.yaml)
- [configs/app.yaml](./configs/app.yaml)

## Local Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements\base.txt
pip install -r requirements\dev.txt
pip install -r requirements\ml.txt
```

### 3. Download model assets

```powershell
.\scripts\download_all_models.ps1
```

### 4. Validate runtime

```powershell
.\scripts\validate_local_runtime.ps1
```

## Run The Streamlit Demo

```powershell
streamlit run apps\streamlit_app\app.py
```

Recommended demo modes:

### Webcam demo

- `Input Type = webcam`
- `Detector = hybrid_person` or `yolox_onnx`
- `Tracker = bytetrack`

### Video file demo

- `Input Type = video_file`
- provide the full file path
- `Detector = hybrid_person` or `yolox_onnx`
- `Tracker = bytetrack`

### Runtime smoke test

- use a short classroom video file if a webcam is not available

## Run The API

```powershell
uvicorn apps.api.main:app --reload
```

Useful endpoints:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/version`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/session/status`
- `http://127.0.0.1:8000/session/snapshot`
- `http://127.0.0.1:8000/session/events`
- `http://127.0.0.1:8000/session/frame`

Quick API session test:

```powershell
.\scripts\test_api_session.ps1
```

## Docker Guide

This repository includes:

- [Dockerfile](./Dockerfile)
- [docker-compose.yml](./docker-compose.yml)
- [.dockerignore](./.dockerignore)

### Start Docker services

```powershell
docker compose up --build
```

Services:

- Streamlit UI: `http://localhost:8501`
- FastAPI: `http://localhost:8000`
- FastAPI docs: `http://localhost:8000/docs`

### Stop Docker services

```powershell
docker compose down
```

### Docker defaults

The containers start with safe defaults:

- `SOURCE_TYPE=webcam`
- `SOURCE_URI=0`
- `DETECTOR_NAME=hybrid_person`
- `TRACKER_NAME=bytetrack`

Why:

- these match the main local demo configuration
- you can still override them in `docker-compose.yml` if your environment differs

### Important Docker limitation

The current local Windows GPU path uses `DirectML`.

That path does **not** automatically carry over into a standard Linux Docker container.

So in Docker:

- ONNX Runtime will usually fall back to CPU
- Docker is best used here for packaging and service orchestration
- the strongest local GPU demo is still the native Windows run

If you want Linux/NVIDIA GPU support later, that should be added as a separate CUDA container path.

## Current Model And Runtime Configuration

As defined in [configs/models.yaml](./configs/models.yaml):

### Detector

- default: `hybrid_person`
- primary weights: `models/onnx/yolox_s.onnx`
- input size: `640 x 640`
- confidence threshold: `0.2`
- providers:
  - `DmlExecutionProvider`
  - `CPUExecutionProvider`

### Tracker

- default: `bytetrack`

### Expression model

- name: `onnx_mobilefacenet_fer`
- weights: `models/onnx/facial_expression_recognition_mobilefacenet_2022july.onnx`
- face detector: `models/onnx/face_detection_yunet_2023mar.onnx`

### Pose

- current implementation: `bbox_pose_heuristic`

### Performance throttling

- full-frame visible face count every `3` frames
- expression refresh every `3` frames
- visible face count max width `960`
- expression refresh limited to largest `6` tracks

## How Student Count Works

Student count is not produced by one single model.

Current logic in [src/classroom_engagement/pipeline/engine.py](./src/classroom_engagement/pipeline/engine.py):

```python
tracked_count = len(tracks)
student_count = max(tracked_count, visible_face_count)
```

So the final count is built from:

1. tracked people from detector + tracker
2. full-frame visible face count from YuNet

Why:

- tracking may undercount crowded scenes
- face counting may miss hidden faces
- taking the maximum is a practical heuristic for this demo

## How Exit Counting Works

Exit count is based on:

- tracked person center point
- a manually configured exit rectangle
- disappearance after staying in the zone

A person is counted as an exit when:

1. the track enters the exit zone
2. it stays there long enough
3. the track disappears

Config lives in:

- [configs/app.yaml](./configs/app.yaml)

UI controls also allow runtime adjustment of the exit zone.

## What The Dashboard Signals Mean

The right-side bars are not all direct model outputs.

### Attention

Derived from:

- face expression probabilities
- posture heuristic

### Posture

Derived from:

- bounding-box shape and crop heuristic

### Face Visibility

Derived from:

- whether visible faces are being found reliably
- ratio logic between visible faces and estimated student count

### Interaction

Derived from:

- hand-raise style heuristic from upper-body crop energy

### Confusion

Derived from:

- facial expression probabilities such as `fearful`, `surprised`, `sad`, and `disgust`

Important:

- these are **engineered product signals**
- they are not yet classroom-specific learned metrics

## Testing And Validation

Run unit tests:

```powershell
pytest tests
```

Compile-check the code:

```powershell
python -m compileall src apps tests
```

Validate local runtime and model files:

```powershell
.\scripts\validate_local_runtime.ps1
```

## Troubleshooting

### 1. `ModuleNotFoundError: classroom_engagement`

Run from the project root:

```powershell
streamlit run apps\streamlit_app\app.py
```

The app also injects `src/` into `sys.path`.

### 2. Docker build fails with Linux engine / pipe errors

Start Docker Desktop first, then run:

```powershell
docker compose up --build
```

### 3. GPU seems underused

That is normal in this pipeline because:

- some stages are GPU accelerated
- many stages are still CPU-side
- UI rendering and tracking are CPU work

### 4. Student count looks wrong in crowded classroom video

That is a known limitation of the current demo approach.

Reasons:

- generic person detector
- wide-angle classroom scenes
- small faces
- occlusion
- rule-based aggregation

The architecture is correct, but dense classroom counting still needs a stronger dedicated model path.

### 5. Right-side panel looked blurry during updates

This was reduced by:

- solid panel backgrounds
- slower independent UI refresh for status sections
- separate fast feed updates from slower status updates

## Scripts

Useful scripts:

- [scripts/setup_env.ps1](./scripts/setup_env.ps1)
- [scripts/download_all_models.ps1](./scripts/download_all_models.ps1)
- [scripts/download_mediapipe_models.ps1](./scripts/download_mediapipe_models.ps1)
- [scripts/validate_local_runtime.ps1](./scripts/validate_local_runtime.ps1)
- [scripts/test_api_session.ps1](./scripts/test_api_session.ps1)
- [scripts/run_demo.ps1](./scripts/run_demo.ps1)
- [scripts/run_api.ps1](./scripts/run_api.ps1)

## Project Status

Implemented:

- local Streamlit demo
- FastAPI backend
- webcam and video-file sources
- YOLOX ONNX detector support
- ByteTrack support
- hybrid detector fallback path
- YuNet face detection and alignment
- ONNX facial expression inference
- heuristic pose scoring
- class-state rule engine
- exit-zone counting
- live event timeline
- Docker packaging

Not yet production-grade:

- dense classroom counting
- classroom-specific engagement model
- dataset and training pipeline
- calibrated evaluation benchmark
- Linux GPU container runtime

## Recommended Reading Order

If you are new to the codebase, read in this order:

1. [README.md](./README.md)
2. [design.md](./design.md)
3. [technical_document.md](./technical_document.md)
4. [apps/streamlit_app/app.py](./apps/streamlit_app/app.py)
5. [src/classroom_engagement/video/source.py](./src/classroom_engagement/video/source.py)
6. [src/classroom_engagement/core/schemas.py](./src/classroom_engagement/core/schemas.py)
7. [src/classroom_engagement/pipeline/engine.py](./src/classroom_engagement/pipeline/engine.py)
8. [src/classroom_engagement/pipeline/components.py](./src/classroom_engagement/pipeline/components.py)
9. [src/classroom_engagement/pipeline/signals.py](./src/classroom_engagement/pipeline/signals.py)
10. [apps/api/main.py](./apps/api/main.py)

## Summary

This repository is a modular ML product demo for classroom analytics.

It is most useful for:

- local CV experimentation
- UI and API prototyping
- ML product teaching
- demonstrating how multiple weak signals are combined into one product output
