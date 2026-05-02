from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from classroom_engagement.core.schemas import StudentTrackFeature


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float32)
    scores -= np.max(scores)
    exp = np.exp(scores)
    return exp / max(float(np.sum(exp)), 1e-6)


class SignalAnalyzer(Protocol):
    def analyze(self, crop_bgr: np.ndarray, track_id: int) -> StudentTrackFeature | None:
        ...


class FaceAlignment:
    def __init__(self) -> None:
        self._std_points = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )

    def align(self, image_bgr: np.ndarray, lm5_points: np.ndarray) -> np.ndarray:
        source = lm5_points.astype(np.float32).reshape(5, 2)
        transform, _ = cv2.estimateAffinePartial2D(source, self._std_points, method=cv2.LMEDS)
        if transform is None:
            return cv2.resize(image_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
        return cv2.warpAffine(image_bgr, transform, (112, 112))


@dataclass
class YuNetFaceDetector:
    config: dict[str, Any]

    def __post_init__(self) -> None:
        self._model_path = Path(self.config.get("model_path", ""))
        if not self._model_path.exists():
            raise RuntimeError(
                "YuNet model asset is missing. "
                f"Expected: {self._model_path}"
            )

        self._score_threshold = float(self.config.get("score_threshold", 0.75))
        self._nms_threshold = float(self.config.get("nms_threshold", 0.3))
        self._top_k = int(self.config.get("top_k", 5000))
        self._backend_id = int(self.config.get("backend_id", int(cv2.dnn.DNN_BACKEND_OPENCV)))
        self._target_id = int(self.config.get("target_id", int(cv2.dnn.DNN_TARGET_CPU)))
        self._detector = cv2.FaceDetectorYN.create(
            model=str(self._model_path),
            config="",
            input_size=(320, 320),
            score_threshold=self._score_threshold,
            nms_threshold=self._nms_threshold,
            top_k=self._top_k,
            backend_id=self._backend_id,
            target_id=self._target_id,
        )
        self._aligner = FaceAlignment()

    def detect_faces(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.size == 0:
            return np.empty((0, 15), dtype=np.float32)

        height, width = image_bgr.shape[:2]
        if height < 24 or width < 24:
            return np.empty((0, 15), dtype=np.float32)

        self._detector.setInputSize((width, height))
        result = self._detector.detect(image_bgr)
        faces = result[1]
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)
        return faces

    def detect_and_align(self, crop_bgr: np.ndarray) -> tuple[np.ndarray | None, float]:
        faces = self.detect_faces(crop_bgr)
        if len(faces) == 0:
            return None, 0.0

        best_face = max(faces, key=lambda face: float(face[-1]))
        landmarks = best_face[4:14].reshape(5, 2)
        aligned = self._aligner.align(crop_bgr, landmarks)
        return aligned, float(best_face[-1])

    def count_faces(self, image_bgr: np.ndarray, min_face_size: int = 14) -> int:
        faces = self.detect_faces(image_bgr)
        if len(faces) == 0:
            return 0

        count = 0
        for face in faces:
            width = float(face[2])
            height = float(face[3])
            if width >= min_face_size and height >= min_face_size:
                count += 1
        return count


@dataclass
class OnnxExpressionAnalyzer:
    config: dict[str, Any]

    def __post_init__(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime-directml is not installed. Install requirements/ml.txt to enable GPU expression inference."
            ) from exc

        self._ort = ort
        self._weights_path = Path(self.config.get("weights", ""))
        if not self._weights_path.exists():
            raise RuntimeError(
                "Expression model asset is missing. "
                f"Expected: {self._weights_path}"
            )

        requested = list(
            self.config.get("providers", ["DmlExecutionProvider", "CPUExecutionProvider"])
        )
        available = set(ort.get_available_providers())
        providers = [provider for provider in requested if provider in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(self._weights_path),
            sess_options=session_options,
            providers=providers,
        )
        self._input_name = self._session.get_inputs()[0].name
        self._labels = list(
            self.config.get(
                "labels",
                ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"],
            )
        )
        face_detector_config = dict(self.config.get("face_detector", {}))
        self._fallback_to_proxy = bool(self.config.get("fallback_to_proxy", False))
        self._face_detector: YuNetFaceDetector | None = None
        if face_detector_config.get("enabled", True):
            self._face_detector = YuNetFaceDetector(face_detector_config)

    def analyze(self, crop_bgr: np.ndarray, track_id: int) -> StudentTrackFeature | None:
        if crop_bgr.size == 0:
            return None

        face_crop: np.ndarray | None = None
        face_confidence = 0.0
        if self._face_detector is not None:
            face_crop, face_confidence = self._face_detector.detect_and_align(crop_bgr)
        if face_crop is None and self._fallback_to_proxy:
            face_crop = self._extract_face_proxy(crop_bgr)
            face_confidence = 0.0
        if face_crop is None or face_crop.size == 0:
            return None

        input_blob = self._preprocess(face_crop)
        raw_output = self._session.run(None, {self._input_name: input_blob})[0]
        probabilities = _softmax(np.squeeze(raw_output))
        emotion_scores = dict(zip(self._labels, probabilities.tolist(), strict=False))

        happy = emotion_scores.get("happy", 0.0)
        neutral = emotion_scores.get("neutral", 0.0)
        surprised = emotion_scores.get("surprised", 0.0)
        sad = emotion_scores.get("sad", 0.0)
        fearful = emotion_scores.get("fearful", 0.0)
        angry = emotion_scores.get("angry", 0.0)
        disgust = emotion_scores.get("disgust", 0.0)

        smile_score = _clip01(happy + 0.35 * surprised)
        confusion_score = _clip01(0.55 * fearful + 0.25 * surprised + 0.15 * sad + 0.10 * disgust)
        attention_score = _clip01(
            0.55 * neutral
            + 0.30 * happy
            + 0.15 * surprised
            - 0.15 * sad
            - 0.10 * angry
        )
        confidence_boost = 0.15 if face_confidence >= 0.85 else 0.0
        attention_score = _clip01(attention_score + confidence_boost)

        return StudentTrackFeature(
            track_id=track_id,
            attention_score=attention_score,
            confusion_score=confusion_score,
            smile_score=smile_score,
            engagement_score=_clip01(
                0.50 * attention_score + 0.30 * smile_score - 0.15 * confusion_score
            ),
            face_visible=True,
            visible=True,
            present=True,
            hand_raised=False,
        )

    @staticmethod
    def _extract_face_proxy(crop_bgr: np.ndarray) -> np.ndarray:
        height, width = crop_bgr.shape[:2]
        if height < 32 or width < 24:
            return np.empty((0, 0, 3), dtype=np.uint8)

        top = 0
        bottom = max(int(height * 0.62), 1)
        left = max(int(width * 0.16), 0)
        right = min(int(width * 0.84), width)
        face_crop = crop_bgr[top:bottom, left:right]
        if face_crop.size == 0:
            return np.empty((0, 0, 3), dtype=np.uint8)

        square_size = max(face_crop.shape[0], face_crop.shape[1])
        padded = np.full((square_size, square_size, 3), 127, dtype=np.uint8)
        y_offset = (square_size - face_crop.shape[0]) // 2
        x_offset = (square_size - face_crop.shape[1]) // 2
        padded[
            y_offset : y_offset + face_crop.shape[0],
            x_offset : x_offset + face_crop.shape[1],
        ] = face_crop
        return padded

    @staticmethod
    def _preprocess(face_crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = rgb.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        return image.transpose(2, 0, 1)[None, :, :, :]


@dataclass
class HeuristicPoseAnalyzer:
    config: dict[str, Any]

    def analyze(self, crop_bgr: np.ndarray, track_id: int) -> StudentTrackFeature | None:
        if crop_bgr.size == 0:
            return None

        height, width = crop_bgr.shape[:2]
        if height < 20 or width < 12:
            return None

        aspect_ratio = height / max(width, 1)
        posture_score = _clip01((aspect_ratio - 1.0) / 1.4)

        upper_band = crop_bgr[: max(int(height * 0.28), 1), :, :]
        lower_band = crop_bgr[int(height * 0.45) :, :, :]
        upper_energy = float(np.std(upper_band)) / 128.0 if upper_band.size else 0.0
        lower_energy = float(np.std(lower_band)) / 128.0 if lower_band.size else 0.0
        hand_raised = bool(upper_energy > lower_energy * 1.18 and upper_energy > 0.22)

        attention_score = _clip01(0.30 * posture_score + (0.18 if hand_raised else 0.0))
        return StudentTrackFeature(
            track_id=track_id,
            attention_score=attention_score,
            posture_score=posture_score,
            hand_raised=hand_raised,
            visible=True,
            present=True,
        )


def build_signal_analyzers(
    models_config: dict[str, Any] | None,
) -> tuple[SignalAnalyzer | None, SignalAnalyzer | None, list[str]]:
    config = models_config or {}
    warnings: list[str] = []
    face_analyzer: SignalAnalyzer | None = None
    pose_analyzer: SignalAnalyzer | None = None

    expression_config = config.get("expression", {})
    if expression_config.get("enabled", False):
        try:
            face_analyzer = OnnxExpressionAnalyzer(expression_config)
        except Exception as exc:
            warnings.append(f"Expression analyzer unavailable: {exc}")

    pose_config = config.get("pose", {})
    if pose_config.get("enabled", False):
        try:
            pose_name = pose_config.get("name", "bbox_pose_heuristic")
            if pose_name == "bbox_pose_heuristic":
                pose_analyzer = HeuristicPoseAnalyzer(pose_config)
            else:
                warnings.append(f"Unsupported pose analyzer: {pose_name}")
        except Exception as exc:
            warnings.append(f"Pose analyzer unavailable: {exc}")

    return face_analyzer, pose_analyzer, warnings
