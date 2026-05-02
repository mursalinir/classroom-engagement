# MediaPipe Task Models

Place MediaPipe task model bundles in this directory if you want to enable the
task-based face and pose analyzers.

Expected default paths:

- `models/mediapipe/face_landmarker_v2.task`
- `models/mediapipe/pose_landmarker.task`

The code supports two MediaPipe runtime shapes:

- legacy `mediapipe.solutions` if available
- newer `mediapipe.tasks` when these model asset files are present
