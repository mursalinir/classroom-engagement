# Classroom Engagement Analytics

## Complete Technical Guide

This document explains how this project works from the ground up.

It is written for someone who may have:

- no machine learning background
- no computer vision background
- no Python project experience
- no familiarity with the codebase

The goal is to make the system understandable both as:

- a working software product
- an educational example of how a professional ML engineer builds a real-time ML application

## 1. What This Project Actually Does

This project is a **local real-time classroom analytics system**.

It takes a live video source such as:

- a webcam
- a classroom video file
- a synthetic demo stream

Then it processes each frame and tries to estimate:

- how many students are visible
- whether students are entering or leaving
- whether a student passed through a configured exit area
- approximate attention and posture signals
- approximate face-based emotion signals
- an overall class state such as `neutral`, `engaged`, `confused`, `energized`, or `disengaged`

It then shows those outputs in:

- a **Streamlit UI dashboard**
- a **FastAPI backend API**

This project runs **fully locally**. It does not require a paid API.

## 2. Important Reality Check

This is a **demo system**, not a medically or academically validated sentiment engine.

The current implementation is best understood as:

- a professional project scaffold
- a modular real-time computer vision pipeline
- a teachable ML product example
- a baseline for future classroom-specific model improvement

It is useful for demonstrating:

- how video is ingested
- how detectors and trackers are composed
- how signals are aggregated
- how an ML result is turned into a product UI and API

It is **not** a guarantee of perfect student counting or perfect sentiment recognition in a crowded classroom.

## 3. Core Idea Behind the Product

Instead of trying to directly “read minds,” the system estimates classroom state from **observable visual signals**.

Example observable signals:

- visible faces
- tracked persons
- face expression probabilities
- posture heuristics
- hand-raise heuristics
- scene departures
- exit-zone crossings

These signals are combined into a **class snapshot**.

Then a rule engine turns that snapshot into a simplified class state.

So the core philosophy is:

`raw video -> low-level visual signals -> structured classroom metrics -> class-level interpretation`

That is a much better engineering design than a single black-box “emotion model.”

## 4. High-Level Architecture

The system is organized into layers:

1. **Input Layer**
   Reads video from webcam, file, or synthetic source.

2. **Detection Layer**
   Finds candidate people in the frame.

3. **Tracking Layer**
   Assigns stable IDs to detections across frames.

4. **Signal Layer**
   Estimates face expression, posture, and visibility signals.

5. **Aggregation Layer**
   Converts per-track signals into class-level metrics.

6. **State Inference Layer**
   Converts class-level metrics into a readable classroom state.

7. **Presentation Layer**
   Displays the results in the UI and the API.

## 5. Repository Structure

The most important files are:

```text
apps/
  api/main.py
  streamlit_app/app.py

configs/
  app.yaml
  models.yaml

src/classroom_engagement/
  api/
    schemas.py
    service.py
  core/
    class_state.py
    schemas.py
  pipeline/
    components.py
    engine.py
    signals.py
  ui/
    formatting.py
  video/
    source.py
  settings.py

scripts/
  download_all_models.ps1
  run_api.ps1
  run_demo.ps1
  setup_env.ps1
  validate_local_runtime.ps1
```

How to think about this structure:

- `apps/` contains entrypoints the user runs
- `src/` contains the actual reusable application logic
- `configs/` controls runtime behavior
- `scripts/` helps set up, validate, and run the project

## 6. Main User Entry Points

There are two main ways to use the system.

### 6.1 Streamlit UI

File:

- `apps/streamlit_app/app.py`

Purpose:

- visual dashboard
- control source and runtime options
- show live annotated video
- show classroom metrics and events

Run:

```powershell
streamlit run apps\streamlit_app\app.py
```

### 6.2 FastAPI API

File:

- `apps/api/main.py`

Purpose:

- start and stop sessions programmatically
- read current status
- fetch latest snapshot
- fetch event history
- fetch latest annotated frame

Run:

```powershell
uvicorn apps.api.main:app --reload
```

## 7. End-to-End Frame Flow

This is the most important section in the whole document.

Every frame goes through this approximate path:

```text
VideoSource.read()
  -> PipelineEngine.process_frame()
    -> detector.detect(frame)
    -> tracker.update(detections)
    -> exit-zone logic
    -> full-frame face counting
    -> per-track feature analysis
    -> class snapshot creation
    -> class state inference
    -> event creation
  -> PipelineEngine.annotate_frame()
  -> UI/API display
```

If you understand this path, you understand the project.

## 8. The Data Models

File:

- `src/classroom_engagement/core/schemas.py`

This file defines the system’s shared data structures.

These are the “contracts” used across modules.

### 8.1 `Detection`

Represents a detector output.

Fields:

- `bbox`: bounding box `(x1, y1, x2, y2)`
- `confidence`: confidence score from `0.0` to `1.0`
- `label`: usually `"person"` or `"motion_blob"`

Meaning:

- “I found something that might be a student at this location.”

### 8.2 `TrackedStudent`

Represents a tracked person after tracking.

Fields include:

- `track_id`
- `bbox`
- `confidence`
- `lost_frames`
- `in_exit_zone`
- `attention_score`
- `posture_score`
- `hand_raised`
- `face_visible`

Meaning:

- “This is not just a detection. This is the same detected student over time.”

### 8.3 `StudentTrackFeature`

Represents inferred behavioral features for a track.

Example fields:

- `attention_score`
- `engagement_score`
- `confusion_score`
- `smile_score`
- `posture_score`
- `hand_raised`
- `face_visible`

Meaning:

- “These are the soft signals derived from a tracked student crop.”

### 8.4 `ClassSnapshot`

This is the most important output object.

It contains class-level metrics such as:

- `student_count`
- `visible_student_count`
- `exit_events`
- `scene_departures`
- `engagement_score`
- `attention_ratio`
- `confusion_ratio`
- `smile_ratio`
- `interaction_ratio`
- `posture_ratio`
- `face_visible_ratio`
- `state`

Meaning:

- “This is the current classroom summary.”

### 8.5 `PipelineResult`

This is the final result produced per frame.

It contains:

- frame index
- FPS
- source name
- class snapshot
- list of tracks
- list of student features
- events
- exit-zone config
- session metrics

Meaning:

- “This is the complete output package for one processed frame.”

## 9. Video Input Layer

File:

- `src/classroom_engagement/video/source.py`

Main classes:

- `VideoSourceConfig`
- `VideoSource`

### 9.1 `VideoSourceConfig`

Defines how input should be read:

- `source_type`
- `source_uri`
- `max_frame_width`
- synthetic mode dimensions

### 9.2 `VideoSource`

This class abstracts the origin of frames.

It supports:

- `webcam`
- `video_file`
- `synthetic`

Important methods:

- `open()`
- `read()`
- `release()`
- `describe()`

### 9.3 Why This Abstraction Matters

If the project directly used OpenCV everywhere in the UI and API code, it would become messy.

Instead, all consumers interact with a consistent interface:

- read one frame
- get timestamp
- handle end-of-stream cleanly

That is a professional design choice.

### 9.4 Synthetic Mode

Synthetic mode generates fake moving rectangles.

Why this exists:

- useful for smoke tests
- useful when no camera is available
- useful to verify pipeline and UI behavior without real human data

This is a product engineering feature, not an ML feature.

## 10. Detection Layer

File:

- `src/classroom_engagement/pipeline/components.py`

The project defines an abstract base:

- `Detector`

This is important because it allows multiple detector backends with one shared interface.

Each detector must implement:

```python
detect(frame) -> list[Detection]
```

### 10.1 `MotionBlobDetector`

This is the simplest detector.

How it works:

- background subtraction
- morphological cleanup
- contour extraction
- contour area filtering
- contour to bounding box conversion

Why it exists:

- fast fallback
- useful for synthetic mode
- useful when real model inference is unavailable

This is **not a real person detector**.

### 10.2 `HOGPersonDetector`

Uses OpenCV’s built-in HOG people detector.

How it works:

- slides a classical detector over the image
- returns human-like bounding boxes

Pros:

- simple
- no external ONNX model needed

Cons:

- slower than modern optimized models in some settings
- less accurate than modern detectors

### 10.3 `YOLOXOnnxDetector`

This is the real ML detector in the current pipeline.

Purpose:

- detect persons using a YOLOX model exported to ONNX

Key steps:

1. load ONNX Runtime session
2. choose execution providers
3. preprocess the frame with letterboxing
4. run inference
5. interpret output tensor
6. keep only person-class detections
7. apply Non-Maximum Suppression
8. return standardized `Detection` objects

#### Important idea: Letterbox preprocessing

The input frame may not match the detector’s fixed expected shape.

So the code:

- resizes while preserving aspect ratio
- pads the frame
- records scale and padding

After inference, the boxes are mapped back to original image coordinates.

That is why the detector has `_preprocess()` and `_postprocess()` methods.

### 10.4 `HybridPersonDetector`

This is the default practical demo detector.

It tries:

1. `YOLOX`
2. `HOG`
3. `MotionBlob`

Why this is useful:

- if YOLOX fails, the demo does not completely break
- if the ONNX model is missing, the app still shows something
- if the scene is synthetic, motion fallback can still produce visible results

This is a product resilience decision.

## 11. Tracking Layer

Also in:

- `src/classroom_engagement/pipeline/components.py`

Tracking answers a different question from detection.

Detection asks:

- “Where are people in this frame?”

Tracking asks:

- “Which person in this frame is the same person as before?”

The shared tracker interface is:

```python
update(detections) -> (tracks, entered_ids, departed_ids)
```

### 11.1 `CentroidTracker`

This is a simple tracker.

How it works:

- compute center point of each detection
- compare to previous tracked centers
- assign the nearest match if distance is small enough
- create new IDs for unmatched detections
- mark tracks as departed after too many lost frames

Pros:

- easy to understand
- good for teaching

Cons:

- can break easily in crowded scenes
- identity switches happen more often

### 11.2 `ByteTrackTracker`

This is the stronger tracker.

It wraps ByteTrack through the `supervision` package.

Why ByteTrack is better:

- more stable in real scenes
- handles object re-association better than naive centroid matching
- more suitable for live demos

Internally, the code:

- converts `Detection` objects into `supervision.Detections`
- runs ByteTrack
- converts results back into local `TrackedStudent` objects
- emits `entered` and `departed` track IDs

Important note:

ByteTrack is still only as good as the detections it receives.

Bad detections produce bad tracking.

## 12. Signal Extraction Layer

File:

- `src/classroom_engagement/pipeline/signals.py`

This module extracts richer signals from track crops.

It contains:

- face alignment
- face detection
- facial expression inference
- pose heuristic inference

### 12.1 Why Signals Are Separate From Detection

Detection says:

- “There is a person here.”

Signal extraction says:

- “What does that person’s crop suggest about attention, posture, or expression?”

That is a separate problem and should remain a separate module.

### 12.2 `FaceAlignment`

This class aligns a detected face to a standard 112x112 face template.

Why alignment matters:

- face models work better when the eyes, nose, and mouth are in consistent positions
- tilted or shifted faces reduce model quality

The class uses five landmark points and computes an affine transform.

### 12.3 `YuNetFaceDetector`

Purpose:

- detect faces
- detect landmarks
- align the best face
- count visible faces in a frame

This class is used in two ways:

1. **Per-track face detection**
   From a tracked person crop, it tries to find a face.

2. **Full-frame face counting**
   On the whole frame, it estimates how many faces are visible.

Why full-frame face counting is important:

- in a classroom, people may be small and densely packed
- person tracking may undercount
- face count can be a better approximation for visible students

### 12.4 `OnnxExpressionAnalyzer`

This is the local face expression inference module.

Purpose:

- take a student crop
- detect and align the face using YuNet
- run ONNX facial expression inference
- convert raw class probabilities into higher-level scores

Current label set:

- angry
- disgust
- fearful
- happy
- neutral
- sad
- surprised

#### How raw emotions become useful product scores

The model does not directly predict:

- attention
- engagement
- confusion

So the project maps emotion probabilities into those concepts.

Examples from the code:

- `smile_score` is influenced by `happy` and `surprised`
- `confusion_score` is influenced by `fearful`, `surprised`, `sad`, and `disgust`
- `attention_score` is influenced by `neutral`, `happy`, `surprised`, and slightly penalized by `sad` and `angry`

This is an **engineering mapping**, not a scientific truth.

That distinction is important.

### 12.5 `HeuristicPoseAnalyzer`

This is not a learned pose model.

It is a lightweight heuristic over the track bounding box.

It estimates:

- `posture_score`
- `hand_raised`
- a small attention contribution

How:

- uses box aspect ratio as a weak posture proxy
- compares image variation in upper and lower crop bands

Why use a heuristic here:

- cheap
- local
- easy to run on Windows
- avoids an extra heavy pose model

This is useful for a demo, but it is less reliable than a real pose estimator.

### 12.6 `build_signal_analyzers()`

This factory reads config and creates:

- a face analyzer
- a pose analyzer

If a model fails to load, it records warnings instead of crashing the whole system immediately.

That is another product-stability choice.

## 13. Pipeline Engine

File:

- `src/classroom_engagement/pipeline/engine.py`

This file is the heart of the system.

The single most important class is:

- `PipelineEngine`

If this project were a factory, `PipelineEngine` would be the assembly line manager.

### 13.1 What `PipelineEngine` Owns

During initialization, it creates or stores:

- class state engine
- detector
- tracker
- face analyzer
- pose analyzer
- frame-level face counter
- performance-related settings
- exit-zone configuration
- tracking memory
- feature cache
- cumulative session counters

This design lets one engine instance keep state over time.

That is crucial because:

- tracking depends on previous frames
- FPS depends on previous timestamps
- exit events depend on temporal history
- caches reduce repeated heavy computation

### 13.2 The Main Method: `process_frame()`

This is the single most important execution function.

It does these steps:

1. increment frame index
2. compute FPS
3. detect objects
4. update tracker
5. apply exit-zone logic to active tracks
6. estimate visible face count over the frame
7. build per-track features
8. update global track set
9. classify departures into scene departures or room exits
10. build class snapshot
11. inject cumulative counts into snapshot
12. infer classroom state
13. build events
14. return `PipelineResult`

This is the project’s primary real-time processing loop.

### 13.3 Why `student_count = max(tracked_count, visible_face_count)`

Inside `_build_snapshot()`, the project uses:

```python
student_count = max(tracked_count, visible_face_count)
```

Reason:

- tracking may miss people in crowded scenes
- face counting may miss people with hidden faces
- taking the maximum is a simple bias toward undercount reduction

This is a practical heuristic.

It is not guaranteed to be perfect.

### 13.4 Aggregation Logic

If signal features exist, the engine computes class-level averages:

- mean attention
- mean confusion
- mean smile
- mean interaction
- mean posture
- mean face visibility

Then engagement is computed as a weighted combination:

```text
0.45 * attention
+ 0.20 * posture
+ 0.15 * interaction
+ 0.10 * smile
+ 0.10 * face_visibility
- 0.15 * confusion
```

This produces a value from `0.0` to `1.0`.

This is one of the most important design choices in the project:

- the system does not rely on a single model output
- it fuses multiple weak signals into one stronger summary

### 13.5 Fallback Snapshot Logic

If no meaningful signal features are available, the engine still produces a snapshot.

It uses fallback defaults such as:

- moderate attention
- low confusion
- moderate posture
- low interaction

Why:

- avoid empty dashboards
- allow the product to remain usable even if a submodule is weak

This is good for demos, but it also means some values are heuristic rather than model-derived.

### 13.6 Feature Caching

The engine keeps a cache:

- `feature_cache`

Why this exists:

- expression inference is expensive
- face detection is expensive
- doing full analysis on every track on every frame is too slow

So the system:

- refreshes only every `expression_interval` frames
- refreshes only the largest `max_expression_tracks` tracks
- reuses cached features otherwise

This is a real-time engineering optimization.

### 13.7 Full-Frame Visible Face Counting

The engine also keeps:

- `frame_face_counter`
- `recent_visible_counts`
- `cached_visible_face_count`

How it works:

- full-frame YuNet face counting does not run on every frame
- when it does run, the frame may be downscaled
- recent counts are smoothed using a median window

Why:

- improve stability
- reduce CPU cost
- reduce jitter in visible student count

### 13.8 Exit-Zone Logic

This is one of the clearest product features in the code.

Process:

1. define a rectangular exit zone in normalized coordinates
2. convert it to pixel coordinates for each frame
3. compute center point of each track
4. record how many frames the track remains in the zone
5. when a track disappears, classify the departure

If the track was in the zone long enough:

- count as `student_exited_room`

Otherwise:

- count as generic `track_departed`

This is a nice example of turning raw tracking into a product-level semantic event.

### 13.9 Frame Annotation

Method:

- `annotate_frame()`

Purpose:

- draw track boxes
- draw exit zone
- show per-track overlay text

The UI displays this annotated frame.

This method is purely for presentation and debugging.

It does not affect the inference results.

## 14. Class State Inference

File:

- `src/classroom_engagement/core/class_state.py`

Class:

- `ClassStateEngine`

This is currently a **rule-based engine**.

Rules:

- if exits exist and attention is low -> `disengaged`
- if confusion is high and interaction is low -> `confused`
- if smile is high and interaction is high -> `energized`
- if attention and engagement are high -> `engaged`
- otherwise -> `neutral`

Why rules are used:

- easy to explain
- easy to debug
- easy to modify in class
- no additional training pipeline needed

A future version could replace this with:

- logistic regression
- gradient boosting
- temporal transformer
- LSTM

But rules are the right starting point for a teaching demo.

## 15. Streamlit UI Architecture

File:

- `apps/streamlit_app/app.py`

This file is the user-facing demo app.

### 15.1 What It Does

The UI:

- loads settings
- loads configs
- initializes Streamlit session state
- lets the user choose source and runtime options
- creates a live processing fragment
- renders metrics
- renders live feed
- renders class status board
- renders signal overview
- renders event history

### 15.2 Session State

The app stores runtime objects in `st.session_state`:

- `video_source`
- `pipeline_engine`
- `event_rows`
- `running`
- `last_result`
- `last_frame_rgb`
- warning text
- exit-zone settings

Why this matters:

Streamlit reruns the script often.

Without session state, the pipeline would reset constantly.

### 15.3 `_ensure_runtime()`

This function is very important.

It:

- builds a signature from current runtime settings
- resets runtime if the signature changed
- creates the `PipelineEngine`
- creates the `VideoSource`

This is how the UI keeps the pipeline synchronized with sidebar configuration.

### 15.4 `_run_single_iteration()`

This function performs one processing step:

1. ensure runtime exists
2. read frame
3. process frame through engine
4. annotate frame
5. store result for UI rendering
6. append formatted events

This is effectively the UI-side frame loop.

### 15.5 `live_loop()`

This function is wrapped with:

```python
@st.fragment(run_every="...")
```

That means Streamlit reruns this fragment on a schedule.

This is how the dashboard behaves like a live application instead of a static page.

### 15.6 Display Logic

The UI organizes outputs into:

- top metric cards
- live classroom feed
- class status board
- signal overview
- recent classroom events

This is the layer where technical outputs are translated into teacher-friendly presentation.

### 15.7 Teaching-Friendly Text

Functions like:

- `_status_style()`
- `_summary_text()`
- `_teacher_tip()`

convert metrics into readable language.

This is a very important product lesson:

ML products are not only about inference.
They are also about interpretation and communication.

## 16. API Architecture

Files:

- `apps/api/main.py`
- `src/classroom_engagement/api/service.py`
- `src/classroom_engagement/api/schemas.py`

### 16.1 `apps/api/main.py`

This file defines HTTP routes.

Key endpoints:

- `GET /`
- `GET /health`
- `GET /version`
- `GET /session/status`
- `POST /session/start`
- `POST /session/stop`
- `GET /session/snapshot`
- `GET /session/events`
- `GET /session/frame`

### 16.2 `SessionService`

This is the real engine behind the API.

It manages:

- current session state
- background processing thread
- latest pipeline result
- latest JPEG frame
- event history

### 16.3 Why a Background Thread Is Used

The API should not block every request waiting for a long real-time loop.

So the code:

- starts a worker thread
- keeps reading frames in the background
- updates shared state
- lets API endpoints read the latest state at any time

This is a common real-time service design.

### 16.4 Session Start Flow

When `/session/start` is called:

1. previous runner is stopped
2. request settings are stored
3. `PipelineEngine` is created
4. `VideoSource` is created and opened
5. one initial frame is processed
6. annotated frame is encoded as JPEG
7. session state becomes `running`
8. background thread starts

That initial frame processing is important because it validates the session before fully committing to the background loop.

## 17. Configuration System

Files:

- `src/classroom_engagement/settings.py`
- `configs/models.yaml`
- `configs/app.yaml`

### 17.1 `Settings`

The `Settings` dataclass:

- reads environment variables
- resolves config paths
- loads YAML files
- resolves default detector and tracker names

This is the bridge between environment configuration and application behavior.

### 17.2 `models.yaml`

This file defines ML runtime choices.

Important sections:

- detector
- tracker
- expression
- pose
- performance

Examples:

- which YOLOX model file to use
- confidence threshold
- which providers ONNX Runtime should try
- whether expression analysis is enabled
- how often face counting should run
- how many tracks get expression inference

### 17.3 `app.yaml`

This file defines more product-oriented runtime settings:

- app metadata
- source defaults
- target FPS
- max frame width
- exit-zone geometry

This separation is useful:

- `models.yaml` describes model behavior
- `app.yaml` describes app runtime behavior

## 18. Model Runtime and Hardware Use

This project is designed to run locally on Windows with GPU support where possible.

### 18.1 Current GPU-Accelerated Parts

Using ONNX Runtime DirectML:

- YOLOX detector
- ONNX facial expression model

Configured providers typically include:

- `DmlExecutionProvider`
- `CPUExecutionProvider`

Meaning:

- try GPU through DirectML first
- fall back to CPU if needed

### 18.2 Current CPU-Heavy Parts

Still mostly CPU:

- YuNet face detector through OpenCV
- heuristic pose analysis
- tracking bookkeeping
- drawing overlays
- Streamlit rendering

### 18.3 Why GPU Usage May Still Look Low

Because the full application is not only model inference.

The total pipeline includes:

- frame I/O
- CPU preprocessing
- face counting
- tracking
- UI rendering

So the GPU may not appear fully saturated even if the detector is using it.

## 19. Performance Optimizations Already in the Code

The code includes several practical optimizations.

### 19.1 Face counting is throttled

Instead of counting faces every frame:

- count every `visible_face_count_interval` frames

### 19.2 Face counting can downscale the frame

Before full-frame face counting:

- the frame may be resized to `visible_face_count_max_width`

### 19.3 Expression inference is throttled

It is not run for every track on every frame.

Instead:

- refresh every `expression_interval` frames
- refresh only the largest `max_expression_tracks` tracks

### 19.4 Cached features are reused

That avoids unnecessary repeated computation.

These are examples of real-time system design decisions.

## 20. Why Accuracy Is Still Imperfect

This matters for honest teaching.

The system can be impressive as an engineering demo, while still having real ML limitations.

Main reasons:

### 20.1 Classroom scenes are hard

- many students
- small faces
- occlusion
- blur
- lighting variation
- overlapping bodies

### 20.2 Generic models are not classroom-specific

YOLOX is trained for general object detection, not classroom counting.

Face emotion models are usually trained on curated face datasets, not wide classroom footage.

### 20.3 Current pose path is heuristic

That is cheaper and simpler, but less accurate than a learned pose estimator.

### 20.4 Class state is rule-based

That is interpretable, but not optimized from labeled classroom data.

So if performance is weak in a crowded classroom video, that is expected.

The project is still technically valuable because the architecture is correct and extensible.

## 21. How the Current Student Count Works

This is a common point of confusion.

The system does **not** rely on one single counting method.

It combines:

- tracked persons
- visible full-frame faces

Then uses:

```text
student_count = max(number_of_tracks, visible_face_count)
```

Why this is done:

- person tracking often undercounts crowded seated scenes
- face counting often undercounts hidden faces
- taking the maximum is a simple compromise

This is a heuristic, not a guaranteed ground-truth count.

## 22. How the Exit Count Works

The UI shows exited students based on cumulative exit events.

A student is counted as exited only when:

1. the track entered the exit zone
2. it stayed there long enough
3. the track then disappeared

This is much better than simply counting anyone near a door.

Still, it depends on tracking quality.

## 23. How Class State Is Produced

The system does not directly output class state from one model.

Instead:

1. build a `ClassSnapshot`
2. compute ratios and engagement score
3. pass it to `ClassStateEngine`
4. apply rule logic

Examples:

- high attention + high engagement -> `engaged`
- exits + low attention -> `disengaged`
- high confusion + low interaction -> `confused`
- high smile + high interaction -> `energized`

This is a classic example of a **hybrid ML + rule system**.

## 24. How to Run the Project

### 24.1 Setup

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements\base.txt
pip install -r requirements\dev.txt
pip install -r requirements\ml.txt
.\scripts\download_all_models.ps1
```

### 24.2 Validate local runtime

```powershell
.\scripts\validate_local_runtime.ps1
```

### 24.3 Run Streamlit

```powershell
streamlit run apps\streamlit_app\app.py
```

### 24.4 Run API

```powershell
uvicorn apps.api.main:app --reload
```

## 25. Recommended Learning Order For Beginners

If someone is new, do not start by reading every file at once.

Read in this order:

1. `README.md`
2. `apps/streamlit_app/app.py`
3. `src/classroom_engagement/video/source.py`
4. `src/classroom_engagement/core/schemas.py`
5. `src/classroom_engagement/pipeline/engine.py`
6. `src/classroom_engagement/pipeline/components.py`
7. `src/classroom_engagement/pipeline/signals.py`
8. `src/classroom_engagement/core/class_state.py`
9. `src/classroom_engagement/api/service.py`
10. `configs/models.yaml`
11. `configs/app.yaml`

That order follows the product’s execution path.

## 26. Suggested Classroom Demonstration Flow

If you are teaching students how a professional ML engineer builds a product, use this order:

1. Explain the business problem
2. Show the repository structure
3. Explain the shared schemas
4. Show how video input is abstracted
5. Explain detection vs tracking
6. Explain face and posture signals
7. Explain aggregation into class metrics
8. Explain rule-based class-state inference
9. Show how UI and API consume the same pipeline result
10. Explain performance tradeoffs and model limitations

This sequence teaches both software design and ML thinking.

## 27. Common Engineering Lessons From This Project

This codebase demonstrates several professional lessons:

### 27.1 Build modular systems

Detectors, trackers, analyzers, and UI are separable modules.

### 27.2 Use shared schemas

Strong data contracts reduce chaos across modules.

### 27.3 Keep fallbacks

The hybrid detector keeps the demo alive when a primary model fails.

### 27.4 Prefer explainable baselines first

Rule-based class-state logic is easier to debug than a fully learned opaque model.

### 27.5 Optimize incrementally

The project throttles expensive steps instead of blindly increasing complexity.

### 27.6 ML products are not just models

An ML product includes:

- runtime control
- caching
- visualization
- API design
- failure handling
- hardware compatibility

## 28. What Would Improve This Project Next

If this were taken toward a stronger production system, the next upgrades would likely be:

1. classroom-specific head or crowd counting model
2. stronger pose model instead of heuristic posture
3. temporal smoothing for class state over time windows
4. classroom-specific labeled dataset
5. learned class-state aggregation model
6. better multi-face sampling strategy for crowded rooms
7. offline evaluation benchmarks with annotated videos

These are natural next steps for a serious ML roadmap.

## 29. Key Terms Explained Simply

### Detection

Finding where an object is in an image.

### Tracking

Following the same detected object across frames.

### Inference

Running a trained model on new data.

### ONNX

A portable model format that many runtimes can execute.

### DirectML

A Windows acceleration layer used here to run ONNX models on local GPU.

### Bounding Box

A rectangle around a detected object.

### Confidence Score

A number showing how sure the model is.

### Heuristic

A rule-of-thumb method, not a trained model.

### Aggregation

Combining multiple low-level signals into a higher-level summary.

## 30. Final Summary

This project is a **real-time classroom engagement analytics demo** built as a modular ML product.

Its most important technical idea is:

`video -> detection -> tracking -> signal extraction -> aggregation -> class-state inference -> UI/API`

Its most important engineering idea is:

**do not treat an ML product as just one model.**

Instead, treat it as a complete system with:

- input handling
- model orchestration
- stateful processing
- caching
- configuration
- presentation
- monitoring endpoints
- clear failure modes

That is how a professional ML engineer thinks about building a real product.
