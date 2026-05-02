# Classroom Engagement Analytics - System Design

## 1. Overview

This project will build a real-time classroom engagement analytics product. A camera connected to a computer observes students in a classroom and the system estimates class-level engagement from observable signals such as attendance, exits, facial cues, posture, orientation, and interaction events.

The system does not claim to read inner emotions directly. Instead, it fuses measurable signals into teacher-facing states such as `engaged`, `neutral`, `confused`, `disengaged`, and `energized`.

## 2. Product Definition

### Primary Goal

Provide a live classroom dashboard that helps a teacher or school operator understand how the class is responding over time.

### Core Outputs

- current student count
- students entering or leaving
- face visibility ratio
- attention direction proxy
- posture and interaction cues
- rolling class engagement score
- class state over the last 30 to 120 seconds
- event log for changes such as `engagement_drop`, `raised_hands_increase`, or `students_left_room`

### Non-Goals

- biometric identification of named students
- grading students
- high-stakes disciplinary decisions
- claiming direct access to inner mental state
- fully automated teacher evaluation

## 3. Users

- teacher running a live class
- school operator reviewing session summaries
- ML engineer improving models and thresholds
- annotators preparing training data

## 4. High-Level Requirements

### Functional Requirements

- ingest webcam, USB camera, video file, or RTSP stream
- detect students in frame
- maintain per-student track IDs across frames
- estimate student-level cues from face and pose
- detect entry and exit events
- aggregate student-level cues into class-level metrics
- expose live metrics through a demo UI
- persist session-level analytics and event logs

### Non-Functional Requirements

- real-time target: 10 to 20 FPS on the MVP path
- end-to-end latency target: under 500 ms for dashboard updates
- graceful degradation when faces are too small or occluded
- modular pipeline with independently swappable models
- explainable outputs with confidence and evidence
- privacy-aware handling of video and derived features

## 5. Success Metrics

### Product Metrics

- teacher-rated usefulness
- false alert rate
- clarity of dashboard signals
- stability of class-state transitions

### ML Metrics

- person detection precision and recall
- MOT metrics for tracking consistency
- exit event precision and recall
- student-level cue accuracy
- class-state macro F1
- calibration quality for confidence scores

### System Metrics

- FPS
- frame drop rate
- inference latency by module
- CPU and GPU utilization
- memory usage

## 6. Assumptions and Constraints

- MVP uses a single front-facing classroom camera
- classroom size is small to medium for face-based cues to remain useful
- larger rooms may require 4K capture or multi-camera setups
- class-level state is derived from multiple low-level signals, not a single direct classifier
- legal consent, privacy controls, and retention policy must be defined before real-world deployment

## 7. System Architecture

```text
Camera / Video Source
    -> Frame Ingestion
    -> Detection and Tracking
    -> Face and Pose Processing
    -> Student-Level Feature Extraction
    -> Temporal Aggregation
    -> Class-State Engine
    -> API / UI / Storage / Monitoring
```

## 8. Detailed Pipeline

### 8.1 Frame Ingestion

Responsibilities:

- read frames from webcam, file, or RTSP
- normalize frame size and timestamps
- support frame skipping for stable throughput
- buffer frames for asynchronous inference if needed

Outputs:

- frame
- frame_id
- timestamp
- source metadata

### 8.2 Student Detection and Tracking

Responsibilities:

- detect persons in frame
- filter weak detections
- assign stable track IDs
- estimate student count

Recommended stack:

- detector: YOLOX
- tracker: ByteTrack

Outputs per frame:

- `track_id`
- `bbox`
- `confidence`
- `track_age`
- `lost_frames`

### 8.3 Face Processing

Responsibilities:

- detect visible faces within person regions or full frame
- compute landmarks and blendshape features
- estimate coarse head orientation proxy
- pass crops to facial expression models

Recommended stack:

- MediaPipe Face Landmarker
- EmotiEffLib for emotion and engagement baseline

Outputs per face:

- face bounding box
- landmarks
- blendshape vector
- face visibility flag
- expression probabilities
- engagement probability

### 8.4 Pose Processing

Responsibilities:

- estimate posture and body orientation
- derive cues such as head-down, leaning, hand raise, restless motion

Recommended stack:

- MVP: MediaPipe Pose Landmarker
- Production: MMPose RTMPose for stronger multi-person pose

Outputs per student:

- body keypoints
- pose confidence
- derived posture flags

### 8.5 Event Detection

Responsibilities:

- detect entering and leaving students
- detect hand raises
- detect prolonged inattentive posture
- detect laughter or amusement proxies through face signals

Methods:

- rule-based over tracked trajectories and low-level cues for MVP
- optional clip-level action recognition later through MMAction2

### 8.6 Student-Level Feature Builder

Aggregate frame-wise signals into short temporal windows.

Candidate features:

- face visible ratio over last N seconds
- average head-up ratio
- average engagement probability
- smile / laugh probability trend
- confusion-like expression trend
- hand raise count
- motion intensity
- time seated vs absent

### 8.7 Temporal Aggregation

Window sizes:

- short: 5 to 10 seconds
- medium: 30 seconds
- long: 60 to 120 seconds

Methods:

- exponential moving averages
- robust rolling summaries
- per-student temporal smoothing
- class-level aggregation with outlier handling

### 8.8 Class-State Engine

This module maps aggregated features to teacher-facing states.

#### MVP

Use interpretable rules and weighted scoring.

Example:

- low student count change + high face-up ratio + high hand-raise count -> `engaged`
- stable attendance + elevated confusion expressions + low interaction -> `confused`
- increased exits + low attention proxy + low visible engagement -> `disengaged`
- elevated smiles + active interaction + stable attention -> `energized`

#### Later Version

Train a temporal structured model on aggregated features:

- XGBoost or LightGBM baseline
- LSTM or Temporal Transformer for sequence learning

Outputs:

- class state
- class engagement score
- uncertainty score
- reason codes

## 9. Model Strategy

### Recommended Open-Source Stack

- person detection: YOLOX
- tracking: ByteTrack
- face landmarks: MediaPipe Face Landmarker
- facial expression baseline: EmotiEffLib
- pose baseline: MediaPipe Pose Landmarker
- stronger pose: MMPose RTMPose
- temporal actions: MMAction2 with skeleton-based models such as PoseC3D or ST-GCN
- deployment optimization: OpenVINO on CPU-first environments

### Model Selection Principles

- permissive licensing for deployability
- explainability over novelty
- low-latency inference
- modular upgrade path
- independent offline evaluation of each module

## 10. Data Strategy

### Data Sources

- staged classroom recordings
- real classroom sessions with consent
- synthetic or semi-controlled pilot sessions for edge cases

### Required Coverage

- varying lighting
- multiple classroom layouts
- occlusion and partial visibility
- different ethnicities and appearances
- students wearing masks or glasses when relevant
- different teaching styles

### Dataset Splits

- train
- validation
- test
- holdout room or holdout teacher split for robustness

## 11. Annotation Plan

### Frame-Level Labels

- person box
- face visible
- head up / down
- looking forward / away
- hand raised
- phone use
- smiling / laughing proxy
- asleep / drowsy proxy

### Track-Level Labels

- entered room
- exited room
- absent from seat
- sustained inattentive behavior

### Clip-Level Labels

- engaged
- neutral
- confused
- disengaged
- energized

### Session-Level Labels

- engagement trend
- difficulty spikes
- interaction quality

## 12. Evaluation Plan

### Module Evaluation

- detection: precision, recall, mAP
- tracking: IDF1, HOTA, ID switches
- exits: event precision and recall
- expression and engagement: accuracy, F1, calibration
- pose-derived cues: per-label F1

### End-to-End Evaluation

- class-state macro F1
- session summary agreement with human raters
- alert usefulness and false positive rate
- latency and throughput under target hardware

### Robustness Evaluation

- low light
- motion blur
- partial occlusion
- distant faces
- crowded scenes

## 13. Privacy, Ethics, and Governance

- require informed consent before recording real classrooms
- separate analytics from identity whenever possible
- avoid face recognition by default
- retain derived analytics longer than raw video when possible
- support configurable retention windows
- log model versions, thresholds, and operator actions
- expose uncertainty instead of overconfident labels

## 14. Deployment Design

### MVP

- Python application
- local Streamlit demo
- webcam or video file input
- local storage for session outputs

### Production Direction

- FastAPI backend
- Streamlit or React dashboard
- WebSocket live updates
- worker-based inference pipeline
- PostgreSQL for session metadata
- object storage for video when enabled
- MLflow for experiment tracking
- DVC for dataset versioning

### Optimization

- ONNX export where practical
- OpenVINO for CPU deployment
- optional TensorRT path for NVIDIA deployments later

## 15. Repository Layout

```text
ML Pipeline/
├── design.md
├── README.md
├── .gitignore
├── .env.example
├── pyproject.toml
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── ml.txt
├── configs/
│   ├── app.yaml
│   ├── models.yaml
│   └── rules.yaml
├── apps/
│   ├── api/
│   │   └── main.py
│   └── streamlit_app/
│       └── app.py
├── src/
│   └── classroom_engagement/
│       ├── __init__.py
│       ├── settings.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── schemas.py
│       │   └── class_state.py
│       └── pipeline/
│           ├── __init__.py
│           └── engine.py
├── scripts/
│   └── run_demo.ps1
├── tests/
│   └── test_imports.py
├── docs/
│   └── architecture.md
├── data/
│   ├── raw/.gitkeep
│   ├── interim/.gitkeep
│   ├── processed/.gitkeep
│   ├── annotations/.gitkeep
│   └── samples/.gitkeep
└── models/
    ├── checkpoints/.gitkeep
    ├── onnx/.gitkeep
    └── openvino/.gitkeep
```

## 16. Implementation Phases

### Phase 1: Scaffold and Baseline

- initialize repo and package structure
- add configs and local demo app
- implement video ingestion, detector, tracker stubs
- produce class count and session events

### Phase 2: Multi-Signal MVP

- add face and pose processing
- add student-level feature extraction
- implement rule-based class-state engine
- expose dashboard and event timeline

### Phase 3: Data and Training

- define annotation schema and tools
- collect pilot dataset
- train clip-level and class-level models
- benchmark and calibrate thresholds

### Phase 4: Productionization

- optimize inference
- add backend APIs and persistence
- add experiment tracking and monitoring
- prepare deployment scripts and model registry

## 17. First Engineering Deliverables

- project scaffold
- detailed configuration schema
- frame ingestion interface
- detector and tracker adapters
- event and state schemas
- Streamlit MVP dashboard
- FastAPI health and session endpoints

## 18. Immediate Next Steps

1. implement the project scaffold and config system
2. wire webcam ingestion into the Streamlit app
3. add person detection and tracking abstraction
4. add session event logging and rolling metrics
5. integrate face and pose baselines
6. implement the first rule-based class-state engine
