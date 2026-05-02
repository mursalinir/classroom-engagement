from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from pathlib import Path
import warnings
from typing import Any

import cv2
import numpy as np

from classroom_engagement.core.schemas import Detection, TrackedStudent


class Detector(ABC):
    name = "detector"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError


class Tracker(ABC):
    name = "tracker"

    @abstractmethod
    def update(
        self, detections: list[Detection]
    ) -> tuple[list[TrackedStudent], list[int], list[int]]:
        raise NotImplementedError


class MotionBlobDetector(Detector):
    """Lightweight local detector for the first demo slice.

    This is a placeholder for YOLOX. It detects moving foreground blobs and
    turns them into student-like detections so the local demo is immediately usable.
    """

    name = "motion_blob"

    def __init__(
        self,
        min_area: int = 2500,
        learning_rate: float = 0.002,
        max_detections: int = 12,
    ) -> None:
        self.min_area = min_area
        self.learning_rate = learning_rate
        self.max_detections = max_detections
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=32,
            detectShadows=False,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        mask = self.background.apply(frame, learningRate=self.learning_rate)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        frame_area = frame.shape[0] * frame.shape[1]

        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            confidence = min(0.99, max(0.1, area / max(frame_area * 0.08, 1)))
            detections.append(
                Detection(
                    bbox=(x, y, x + w, y + h),
                    confidence=confidence,
                    label="motion_blob",
                )
            )

            if len(detections) >= self.max_detections:
                break

        return detections


class HOGPersonDetector(Detector):
    """CPU-friendly person detector using OpenCV's built-in HOG people detector."""

    name = "hog_person"

    def __init__(self, hit_threshold: float = 0.0) -> None:
        self.hit_threshold = hit_threshold
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> list[Detection]:
        rects, weights = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
            hitThreshold=self.hit_threshold,
        )

        detections: list[Detection] = []
        for (x, y, w, h), weight in zip(rects, weights):
            detections.append(
                Detection(
                    bbox=(int(x), int(y), int(x + w), int(y + h)),
                    confidence=float(min(0.99, max(0.1, weight))),
                    label="person",
                )
            )
        return detections


class YOLOXOnnxDetector(Detector):
    """YOLOX ONNX detector with letterbox preprocessing and person-class filtering."""

    name = "yolox_onnx"

    def __init__(
        self,
        weights_path: str | Path,
        input_width: int = 640,
        input_height: int = 640,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        person_class_id: int = 0,
        providers: list[str] | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.input_width = input_width
        self.input_height = input_height
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.person_class_id = person_class_id
        self.providers = providers or ["CPUExecutionProvider"]
        self._session: Any | None = None
        self._input_name: str | None = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        session = self._ensure_session()
        blob, ratio, pad = self._preprocess(frame)
        outputs = session.run(None, {self._input_name: blob})
        return self._postprocess(
            raw_output=outputs[0],
            frame_shape=frame.shape[:2],
            ratio=ratio,
            pad=pad,
        )

    def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"YOLOX ONNX model not found: {self.weights_path}. "
                "Place the exported model at this path or update configs/models.yaml."
            )

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Install requirements/base.txt or add onnxruntime-gpu."
            ) from exc

        available = set(ort.get_available_providers())
        selected_providers = [provider for provider in self.providers if provider in available]
        if not selected_providers:
            selected_providers = ["CPUExecutionProvider"]

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(self.weights_path),
            sess_options=session_options,
            providers=selected_providers,
        )
        self._input_name = self._session.get_inputs()[0].name
        return self._session

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        image = frame.astype(np.float32)
        padded = np.full((self.input_height, self.input_width, 3), 114.0, dtype=np.float32)

        ratio = min(self.input_height / image.shape[0], self.input_width / image.shape[1])
        resized_width = max(1, int(image.shape[1] * ratio))
        resized_height = max(1, int(image.shape[0] * ratio))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        padded[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

        blob = padded.transpose(2, 0, 1)[None, :, :, :]
        return blob, ratio, (pad_x, pad_y)

    def _postprocess(
        self,
        raw_output: np.ndarray,
        frame_shape: tuple[int, int],
        ratio: float,
        pad: tuple[int, int],
    ) -> list[Detection]:
        predictions = np.squeeze(raw_output, axis=0) if raw_output.ndim == 3 else raw_output
        if predictions.ndim != 2 or predictions.shape[0] == 0:
            return []

        if predictions.shape[1] == 6:
            return self._postprocess_xyxy(predictions, frame_shape)

        if predictions.shape[1] < 6:
            raise RuntimeError(
                f"Unsupported YOLOX output shape: {predictions.shape}. "
                "Expected Nx6 or Nx(5+num_classes)."
            )

        boxes_xywh = predictions[:, :4]
        objectness = predictions[:, 4]
        class_scores = predictions[:, 5:]
        if class_scores.shape[1] <= self.person_class_id:
            raise RuntimeError(
                f"YOLOX output does not contain class index {self.person_class_id}. "
                f"Output shape: {predictions.shape}"
            )

        person_scores = objectness * class_scores[:, self.person_class_id]
        keep = person_scores >= self.confidence_threshold
        if not np.any(keep):
            return []

        boxes_xywh = boxes_xywh[keep]
        person_scores = person_scores[keep]
        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        boxes_xyxy[:, [0, 2]] -= pad[0]
        boxes_xyxy[:, [1, 3]] -= pad[1]
        boxes_xyxy /= max(ratio, 1e-6)
        boxes_xyxy = self._clip_boxes(boxes_xyxy, frame_shape)
        return self._nms_to_detections(boxes_xyxy, person_scores, label="person")

    def _postprocess_xyxy(
        self, predictions: np.ndarray, frame_shape: tuple[int, int]
    ) -> list[Detection]:
        boxes_xyxy = self._clip_boxes(predictions[:, :4].copy(), frame_shape)
        scores = predictions[:, 4]
        class_ids = predictions[:, 5].astype(int)

        keep = (scores >= self.confidence_threshold) & (class_ids == self.person_class_id)
        if not np.any(keep):
            return []

        return self._nms_to_detections(boxes_xyxy[keep], scores[keep], label="person")

    def _nms_to_detections(
        self, boxes_xyxy: np.ndarray, scores: np.ndarray, label: str
    ) -> list[Detection]:
        if boxes_xyxy.size == 0:
            return []

        boxes_for_nms = [
            [
                int(box[0]),
                int(box[1]),
                max(1, int(box[2] - box[0])),
                max(1, int(box[3] - box[1])),
            ]
            for box in boxes_xyxy
        ]
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_for_nms,
            scores=scores.tolist(),
            score_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
        )
        if len(indices) == 0:
            return []

        selected = np.array(indices).reshape(-1)
        detections: list[Detection] = []
        for idx in selected:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(min(0.99, max(0.0, scores[idx]))),
                    label=label,
                )
            )
        return detections

    @staticmethod
    def _xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
        boxes_xyxy = boxes_xywh.copy()
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
        return boxes_xyxy

    @staticmethod
    def _clip_boxes(boxes_xyxy: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
        height, width = frame_shape
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, width - 1)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, height - 1)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, width - 1)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, height - 1)
        return boxes_xyxy


class HybridPersonDetector(Detector):
    """Detector that prefers YOLOX but falls back to HOG and motion for demos."""

    name = "hybrid_person"

    def __init__(
        self,
        weights_path: str | Path = "models/onnx/yolox_tiny.onnx",
        input_width: int = 416,
        input_height: int = 416,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.45,
        person_class_id: int = 0,
        providers: list[str] | None = None,
    ) -> None:
        self.yolox = YOLOXOnnxDetector(
            weights_path=weights_path,
            input_width=input_width,
            input_height=input_height,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            person_class_id=person_class_id,
            providers=providers,
        )
        self.hog = HOGPersonDetector()
        self.motion = MotionBlobDetector(min_area=1800, learning_rate=0.003, max_detections=8)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        try:
            yolox_detections = self.yolox.detect(frame)
        except Exception:
            yolox_detections = []
        if yolox_detections:
            return yolox_detections

        hog_detections = self.hog.detect(frame)
        if hog_detections:
            return hog_detections

        return self.motion.detect(frame)


@dataclass(slots=True)
class _TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    lost_frames: int = 0
    centroid: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))


class CentroidTracker(Tracker):
    name = "centroid"

    def __init__(self, distance_threshold: float = 120.0, max_lost_frames: int = 12) -> None:
        self.distance_threshold = distance_threshold
        self.max_lost_frames = max_lost_frames
        self.next_track_id = 1
        self.tracks: dict[int, _TrackState] = {}

    def update(
        self, detections: list[Detection]
    ) -> tuple[list[TrackedStudent], list[int], list[int]]:
        entered: list[int] = []
        departed: list[int] = []
        unmatched_track_ids = set(self.tracks.keys())

        detection_centroids = [self._centroid(det.bbox) for det in detections]
        assignments: dict[int, int] = {}

        for det_idx, centroid in enumerate(detection_centroids):
            best_track_id: int | None = None
            best_distance = float("inf")

            for track_id, track in self.tracks.items():
                if track_id not in unmatched_track_ids:
                    continue
                distance = self._distance(track.centroid, centroid)
                if distance < best_distance and distance <= self.distance_threshold:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is not None:
                assignments[det_idx] = best_track_id
                unmatched_track_ids.remove(best_track_id)

        for det_idx, track_id in assignments.items():
            detection = detections[det_idx]
            self.tracks[track_id] = _TrackState(
                track_id=track_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                lost_frames=0,
                centroid=detection_centroids[det_idx],
            )

        for det_idx, detection in enumerate(detections):
            if det_idx in assignments:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = _TrackState(
                track_id=track_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                lost_frames=0,
                centroid=detection_centroids[det_idx],
            )
            entered.append(track_id)

        for track_id in list(unmatched_track_ids):
            track = self.tracks[track_id]
            track.lost_frames += 1
            if track.lost_frames > self.max_lost_frames:
                departed.append(track_id)
                del self.tracks[track_id]

        active_tracks = [
            TrackedStudent(
                track_id=track.track_id,
                bbox=track.bbox,
                confidence=track.confidence,
                lost_frames=track.lost_frames,
            )
            for track in self.tracks.values()
            if track.lost_frames == 0
        ]
        active_tracks.sort(key=lambda item: item.track_id)
        return active_tracks, entered, departed

    @staticmethod
    def _centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.dist(a, b)


class ByteTrackTracker(Tracker):
    name = "bytetrack"

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: float = 15.0,
        minimum_consecutive_frames: int = 1,
    ) -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                from supervision import Detections
                from supervision.tracker.byte_tracker.core import ByteTrack
        except ImportError as exc:
            raise RuntimeError(
                "supervision is not installed. Install requirements/base.txt to use ByteTrack."
            ) from exc

        self._detections_cls = Detections
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                frame_rate=frame_rate,
                minimum_consecutive_frames=minimum_consecutive_frames,
            )
        self.max_lost_frames = max(1, int(lost_track_buffer))
        self._missing_counts: dict[int, int] = {}
        self._active_ids: set[int] = set()

    def update(
        self, detections: list[Detection]
    ) -> tuple[list[TrackedStudent], list[int], list[int]]:
        supervised_detections = self._to_supervision_detections(detections)
        tracked = self._tracker.update_with_detections(supervised_detections)

        current_ids: set[int] = set()
        active_tracks: list[TrackedStudent] = []
        confidences = (
            tracked.confidence if tracked.confidence is not None else np.zeros(len(tracked.xyxy))
        )

        if len(tracked.xyxy) > 0 and tracked.tracker_id is not None:
            for bbox, confidence, track_id in zip(tracked.xyxy, confidences, tracked.tracker_id):
                track_id_int = int(track_id)
                current_ids.add(track_id_int)
                self._missing_counts[track_id_int] = 0
                active_tracks.append(
                    TrackedStudent(
                        track_id=track_id_int,
                        bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                        confidence=float(min(0.99, max(0.0, confidence))),
                        lost_frames=0,
                    )
                )

        entered = sorted(current_ids - self._active_ids)
        departed = self._update_missing_counts(current_ids)
        self._active_ids = set(self._missing_counts.keys())
        active_tracks.sort(key=lambda item: item.track_id)
        return active_tracks, entered, departed

    def _to_supervision_detections(self, detections: list[Detection]) -> Any:
        if not detections:
            return self._detections_cls.empty()

        xyxy = np.array([det.bbox for det in detections], dtype=np.float32)
        confidence = np.array([det.confidence for det in detections], dtype=np.float32)
        class_id = np.zeros(len(detections), dtype=int)
        return self._detections_cls(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

    def _update_missing_counts(self, current_ids: set[int]) -> list[int]:
        departed: list[int] = []
        tracked_ids = set(self._missing_counts.keys()) | current_ids | self._active_ids
        for track_id in sorted(tracked_ids):
            if track_id in current_ids:
                self._missing_counts[track_id] = 0
                continue

            previous_missing = self._missing_counts.get(track_id, 0)
            next_missing = previous_missing + 1
            self._missing_counts[track_id] = next_missing
            if next_missing > self.max_lost_frames:
                departed.append(track_id)
                self._missing_counts.pop(track_id, None)
        return departed


def build_detector(name: str, config: dict[str, Any] | None = None) -> Detector:
    normalized = name.strip().lower()
    config = config or {}
    if normalized == MotionBlobDetector.name:
        return MotionBlobDetector()
    if normalized == HOGPersonDetector.name:
        return HOGPersonDetector()
    if normalized == HybridPersonDetector.name:
        return HybridPersonDetector(
            weights_path=config.get("weights", "models/onnx/yolox_tiny.onnx"),
            input_width=int(config.get("input_width", 416)),
            input_height=int(config.get("input_height", 416)),
            confidence_threshold=float(config.get("confidence_threshold", 0.3)),
            nms_threshold=float(config.get("nms_threshold", 0.45)),
            person_class_id=int(config.get("person_class_id", 0)),
            providers=list(config.get("providers", ["DmlExecutionProvider", "CPUExecutionProvider"])),
        )
    if normalized == YOLOXOnnxDetector.name:
        return YOLOXOnnxDetector(
            weights_path=config.get("weights", "models/onnx/yolox_s.onnx"),
            input_width=int(config.get("input_width", 640)),
            input_height=int(config.get("input_height", 640)),
            confidence_threshold=float(config.get("confidence_threshold", 0.35)),
            nms_threshold=float(config.get("nms_threshold", 0.45)),
            person_class_id=int(config.get("person_class_id", 0)),
            providers=list(config.get("providers", ["CPUExecutionProvider"])),
        )
    raise ValueError(f"Unsupported detector: {name}")


def build_tracker(name: str, config: dict[str, Any] | None = None) -> Tracker:
    normalized = name.strip().lower()
    config = config or {}
    if normalized == CentroidTracker.name:
        return CentroidTracker()
    if normalized == ByteTrackTracker.name:
        return ByteTrackTracker(
            track_activation_threshold=float(config.get("track_activation_threshold", 0.25)),
            lost_track_buffer=int(config.get("lost_track_buffer", 30)),
            minimum_matching_threshold=float(config.get("minimum_matching_threshold", 0.8)),
            frame_rate=float(config.get("frame_rate", 15.0)),
            minimum_consecutive_frames=int(config.get("minimum_consecutive_frames", 1)),
        )
    raise ValueError(f"Unsupported tracker: {name}")
