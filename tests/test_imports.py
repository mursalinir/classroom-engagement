import numpy as np
from pathlib import Path
import pytest

from classroom_engagement.core.class_state import ClassStateEngine
from classroom_engagement.core.schemas import ClassSnapshot, StudentTrackFeature, TrackedStudent
from classroom_engagement.pipeline.components import build_detector, build_tracker
from classroom_engagement.pipeline.engine import PipelineEngine
from classroom_engagement.pipeline.signals import OnnxExpressionAnalyzer


def test_pipeline_imports() -> None:
    engine = PipelineEngine()
    snapshot = ClassSnapshot()
    state = engine.infer_class_state(snapshot).state

    assert isinstance(ClassStateEngine(), ClassStateEngine)
    assert state in {"unknown", "neutral", "engaged", "confused", "disengaged", "energized"}


def test_pipeline_process_frame() -> None:
    engine = PipelineEngine()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    result = engine.process_frame(frame=frame, timestamp=1.0, source_name="test")
    annotated = engine.annotate_frame(frame, result)

    assert result.source_name == "test"
    assert annotated.shape == frame.shape


def test_build_yolox_detector_without_model() -> None:
    detector = build_detector(
        "yolox_onnx",
        config={
            "weights": "models/onnx/does_not_exist.onnx",
        },
    )

    with pytest.raises(FileNotFoundError):
        detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))


def test_build_yolox_detector_with_downloaded_model() -> None:
    weights_path = Path("models/onnx/yolox_tiny.onnx")
    if not weights_path.exists():
        pytest.skip("Downloaded YOLOX model is not present.")

    detector = build_detector(
        "yolox_onnx",
        config={
            "weights": str(weights_path),
            "input_width": 416,
            "input_height": 416,
            "providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
        },
    )

    detections = detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    assert isinstance(detections, list)


def test_build_bytetrack_tracker() -> None:
    tracker = build_tracker("bytetrack", config={"frame_rate": 15, "lost_track_buffer": 2})
    tracks, entered, departed = tracker.update([])

    assert tracks == []
    assert entered == []
    assert departed == []


def test_build_hybrid_detector() -> None:
    detector = build_detector(
        "hybrid_person",
        config={
            "weights": "models/onnx/yolox_tiny.onnx",
            "input_width": 416,
            "input_height": 416,
            "providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
        },
    )
    detections = detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    assert isinstance(detections, list)


def test_build_expression_analyzer() -> None:
    weights_path = Path("models/onnx/facial_expression_recognition_mobilefacenet_2022july.onnx")
    face_detector_path = Path("models/onnx/face_detection_yunet_2023mar.onnx")
    if not weights_path.exists() or not face_detector_path.exists():
        pytest.skip("Downloaded expression or face detector model is not present.")

    analyzer = OnnxExpressionAnalyzer(
        {
            "weights": str(weights_path),
            "providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
            "face_detector": {
                "enabled": True,
                "model_path": str(face_detector_path),
            },
        }
    )
    feature = analyzer.analyze(np.full((160, 120, 3), 127, dtype=np.uint8), track_id=1)
    assert feature is None or feature.face_visible is True


def test_exit_zone_departure_classification() -> None:
    engine = PipelineEngine(
        models_config={
            "app_runtime": {
                "exit_zone": {
                    "enabled": True,
                    "x1_ratio": 0.5,
                    "y1_ratio": 0.0,
                    "x2_ratio": 1.0,
                    "y2_ratio": 1.0,
                    "min_frames_in_zone": 1,
                }
            }
        }
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tracks = [
        TrackedStudent(track_id=7, bbox=(60, 10, 90, 90), confidence=0.9),
    ]

    updated = engine._apply_exit_zone(frame, tracks)
    exit_events, scene_departures = engine._classify_departures([7], timestamp=1.0)

    assert updated[0].in_exit_zone is True
    assert exit_events == [7]
    assert scene_departures == []


def test_scene_departure_without_exit_zone_hit() -> None:
    engine = PipelineEngine(
        models_config={
            "app_runtime": {
                "exit_zone": {
                    "enabled": True,
                    "x1_ratio": 0.8,
                    "y1_ratio": 0.0,
                    "x2_ratio": 1.0,
                    "y2_ratio": 1.0,
                    "min_frames_in_zone": 1,
                }
            }
        }
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tracks = [
        TrackedStudent(track_id=3, bbox=(10, 10, 30, 40), confidence=0.9),
    ]

    updated = engine._apply_exit_zone(frame, tracks)
    exit_events, scene_departures = engine._classify_departures([3], timestamp=1.0)

    assert updated[0].in_exit_zone is False
    assert exit_events == []
    assert scene_departures == [3]


def test_snapshot_from_student_features() -> None:
    engine = PipelineEngine(models_config={"face_landmarks": {"enabled": False}, "pose": {"enabled": False}})
    tracks = [TrackedStudent(track_id=1, bbox=(0, 0, 10, 10))]
    features = [
        StudentTrackFeature(
            track_id=1,
            attention_score=0.8,
            confusion_score=0.1,
            smile_score=0.2,
            posture_score=0.7,
            hand_raised=True,
            face_visible=True,
            visible=True,
        )
    ]

    snapshot = engine._build_snapshot(tracks, features, visible_face_count=3)

    assert snapshot.student_count == 3
    assert snapshot.visible_student_count == 3
    assert snapshot.attention_ratio > 0.7
    assert snapshot.posture_ratio > 0.6
    assert snapshot.face_visible_ratio == 1.0
