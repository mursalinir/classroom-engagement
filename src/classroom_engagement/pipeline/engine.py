from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import cv2
import numpy as np
from typing import Any

from classroom_engagement.core.class_state import ClassStateEngine
from classroom_engagement.core.schemas import (
    ClassSnapshot,
    ExitZone,
    PipelineEvent,
    PipelineResult,
    SessionMetrics,
    StudentTrackFeature,
    TrackedStudent,
)
from classroom_engagement.pipeline.components import build_detector, build_tracker
from classroom_engagement.pipeline.signals import YuNetFaceDetector, build_signal_analyzers


@dataclass(slots=True)
class _TrackMemory:
    last_centroid: tuple[float, float]
    last_bbox: tuple[int, int, int, int]
    frames_in_exit_zone: int = 0
    was_in_exit_zone: bool = False


@dataclass(slots=True)
class _FeatureCacheEntry:
    feature: StudentTrackFeature
    frame_index: int


class PipelineEngine:
    """Local demo pipeline with pluggable detector and tracker interfaces."""

    def __init__(
        self,
        detector_name: str = "motion_blob",
        tracker_name: str = "centroid",
        models_config: dict[str, Any] | None = None,
    ) -> None:
        self.class_state_engine = ClassStateEngine()
        detector_config = (models_config or {}).get("detector", {})
        tracker_config = (models_config or {}).get("tracker", {})
        app_runtime_config = (models_config or {}).get("app_runtime", {})
        self.detector = build_detector(detector_name, config=detector_config)
        self.tracker = build_tracker(tracker_name, config=tracker_config)
        self.detector_name = detector_name
        self.tracker_name = tracker_name
        self.face_analyzer, self.pose_analyzer, self.signal_warnings = build_signal_analyzers(
            models_config or {}
        )
        self.frame_face_counter = self._build_frame_face_counter(models_config or {})
        performance_config = (models_config or {}).get("performance", {})
        self.visible_face_count_interval = max(
            1, int(performance_config.get("visible_face_count_interval", 3))
        )
        self.visible_face_count_max_width = max(
            320, int(performance_config.get("visible_face_count_max_width", 960))
        )
        self.visible_face_min_size = max(
            8, int(performance_config.get("visible_face_min_size", 12))
        )
        self.expression_interval = max(1, int(performance_config.get("expression_interval", 3)))
        self.max_expression_tracks = max(1, int(performance_config.get("max_expression_tracks", 6)))
        self.frame_index = 0
        self._last_frame_ts: float | None = None
        self.exit_zone = self._build_exit_zone(app_runtime_config)
        self.track_memory: dict[int, _TrackMemory] = {}
        self.feature_cache: dict[int, _FeatureCacheEntry] = {}
        self.total_scene_departures = 0
        self.total_exit_events = 0
        self.total_tracks_seen: set[int] = set()
        self.total_event_count = 0
        self.fps_sum = 0.0
        self.fps_samples = 0
        self.recent_visible_counts: deque[int] = deque(maxlen=5)
        self.cached_visible_face_count = 0
        self._startup_warnings_emitted = False

    def infer_class_state(self, snapshot: ClassSnapshot) -> ClassSnapshot:
        snapshot.state = self.class_state_engine.infer_state(snapshot)
        return snapshot

    def process_frame(self, frame: np.ndarray, timestamp: float, source_name: str) -> PipelineResult:
        self.frame_index += 1
        fps = self._compute_fps(timestamp)
        if fps > 0:
            self.fps_sum += fps
            self.fps_samples += 1

        detections = self.detector.detect(frame)
        tracks, entered, departed = self.tracker.update(detections)
        tracks = self._apply_exit_zone(frame=frame, tracks=tracks)
        visible_face_count = self._estimate_visible_face_count(frame)
        student_features, tracks = self._build_student_features(frame=frame, tracks=tracks)
        self.total_tracks_seen.update(track.track_id for track in tracks)

        exit_events, scene_departures = self._classify_departures(
            departed=departed,
            timestamp=timestamp,
        )
        self.total_scene_departures += len(scene_departures)
        self.total_exit_events += len(exit_events)

        snapshot = self._build_snapshot(tracks, student_features, visible_face_count)
        snapshot.exit_events = self.total_exit_events
        snapshot.scene_departures = self.total_scene_departures
        snapshot = self.infer_class_state(snapshot)

        events = self._build_events(
            entered=entered,
            scene_departures=scene_departures,
            exit_events=exit_events,
            timestamp=timestamp,
        )
        self.total_event_count += len(events)
        return PipelineResult(
            frame_index=self.frame_index,
            fps=fps,
            source_name=source_name,
            class_snapshot=snapshot,
            tracks=tracks,
            student_features=student_features,
            events=events,
            exit_zone=self.exit_zone,
            session_metrics=self._build_session_metrics(active_tracks=len(tracks)),
        )

    def annotate_frame(self, frame: np.ndarray, result: PipelineResult) -> np.ndarray:
        annotated = frame.copy()
        for track in result.tracks:
            x1, y1, x2, y2 = track.bbox
            color = (0, 140, 255) if track.in_exit_zone else (0, 200, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"ID {track.track_id} A:{track.attention_score:.2f}{' H' if track.hand_raised else ''}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )

        if result.exit_zone.enabled:
            zone = self._exit_zone_pixels(frame.shape[1], frame.shape[0], result.exit_zone)
            cv2.rectangle(annotated, (zone[0], zone[1]), (zone[2], zone[3]), (0, 140, 255), 2)
            cv2.putText(
                annotated,
                "EXIT ZONE",
                (zone[0], max(zone[1] - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 140, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _build_snapshot(
        self,
        tracks: list[TrackedStudent],
        student_features: list[StudentTrackFeature],
        visible_face_count: int = 0,
    ) -> ClassSnapshot:
        tracked_count = len(tracks)
        student_count = max(tracked_count, visible_face_count)
        if student_features and self._has_signal_features(student_features):
            attention_ratio = float(
                np.mean([feature.attention_score for feature in student_features])
            )
            confusion_ratio = float(
                np.mean([feature.confusion_score for feature in student_features])
            )
            smile_ratio = float(np.mean([feature.smile_score for feature in student_features]))
            interaction_ratio = float(
                np.mean([1.0 if feature.hand_raised else 0.0 for feature in student_features])
            )
            posture_ratio = float(np.mean([feature.posture_score for feature in student_features]))
            face_visible_ratio = float(
                np.mean([1.0 if feature.face_visible else 0.0 for feature in student_features])
            )
            if student_count > 0 and visible_face_count > 0:
                face_visible_ratio = max(face_visible_ratio, min(1.0, visible_face_count / student_count))
            engagement_score = float(
                np.clip(
                    0.45 * attention_ratio
                    + 0.20 * posture_ratio
                    + 0.15 * interaction_ratio
                    + 0.10 * smile_ratio
                    + 0.10 * face_visible_ratio
                    - 0.15 * confusion_ratio,
                    0.0,
                    1.0,
                )
            )
        else:
            attention_ratio = 0.65 if student_count > 0 else 0.0
            confusion_ratio = 0.10 if student_count > 0 else 0.0
            smile_ratio = 0.15 if student_count > 0 else 0.0
            interaction_ratio = min(1.0, student_count * 0.05) if student_count > 0 else 0.0
            posture_ratio = 0.55 if student_count > 0 else 0.0
            face_visible_ratio = min(1.0, visible_face_count / max(student_count, 1)) if student_count > 0 else 0.0
            engagement_score = min(1.0, 0.35 + student_count * 0.08) if student_count > 0 else 0.0
        return ClassSnapshot(
            student_count=student_count,
            visible_student_count=max(
                visible_face_count,
                sum(1 for feature in student_features if feature.visible) if student_features else tracked_count,
            ),
            engagement_score=engagement_score,
            attention_ratio=attention_ratio,
            confusion_ratio=confusion_ratio,
            smile_ratio=smile_ratio,
            interaction_ratio=interaction_ratio,
            posture_ratio=posture_ratio,
            face_visible_ratio=face_visible_ratio,
        )

    def _build_frame_face_counter(self, models_config: dict[str, Any]) -> YuNetFaceDetector | None:
        expression_config = models_config.get("expression", {})
        face_detector_config = expression_config.get("face_detector", {})
        if not face_detector_config.get("enabled", False):
            return None
        try:
            return YuNetFaceDetector(face_detector_config)
        except Exception as exc:
            self.signal_warnings = [*getattr(self, "signal_warnings", []), f"Face counter unavailable: {exc}"]
            return None

    def _estimate_visible_face_count(self, frame: np.ndarray) -> int:
        if self.frame_face_counter is None:
            return 0
        if self.frame_index > 1 and (self.frame_index % self.visible_face_count_interval) != 0:
            return self.cached_visible_face_count

        counting_frame = frame
        height, width = frame.shape[:2]
        if width > self.visible_face_count_max_width:
            scale = self.visible_face_count_max_width / width
            resized_size = (int(width * scale), int(height * scale))
            counting_frame = cv2.resize(frame, resized_size, interpolation=cv2.INTER_AREA)
        try:
            count = self.frame_face_counter.count_faces(
                counting_frame,
                min_face_size=self.visible_face_min_size,
            )
        except Exception:
            return self.cached_visible_face_count
        self.recent_visible_counts.append(count)
        if not self.recent_visible_counts:
            self.cached_visible_face_count = count
            return count
        self.cached_visible_face_count = int(
            np.median(np.array(self.recent_visible_counts, dtype=np.float32))
        )
        return self.cached_visible_face_count

    def _build_session_metrics(self, active_tracks: int) -> SessionMetrics:
        average_fps = self.fps_sum / self.fps_samples if self.fps_samples > 0 else 0.0
        return SessionMetrics(
            total_tracks_seen=len(self.total_tracks_seen),
            active_tracks=active_tracks,
            scene_departures=self.total_scene_departures,
            room_exit_events=self.total_exit_events,
            event_count=self.total_event_count,
            average_fps=average_fps,
        )

    def _build_events(
        self,
        entered: list[int],
        scene_departures: list[int],
        exit_events: list[int],
        timestamp: float,
        ) -> list[PipelineEvent]:
        events: list[PipelineEvent] = []
        if not self._startup_warnings_emitted:
            for warning_message in self.signal_warnings:
                events.append(
                    PipelineEvent(
                        event_type="warning",
                        message=warning_message,
                        timestamp=timestamp,
                    )
                )
            self._startup_warnings_emitted = True
        for track_id in entered:
            events.append(
                PipelineEvent(
                    event_type="track_entered",
                    message=f"Track {track_id} entered the scene",
                    track_id=track_id,
                    timestamp=timestamp,
                )
            )
        for track_id in exit_events:
            events.append(
                PipelineEvent(
                    event_type="student_exited_room",
                    message=f"Track {track_id} exited through the configured door zone",
                    track_id=track_id,
                    timestamp=timestamp,
                )
            )
        for track_id in scene_departures:
            events.append(
                PipelineEvent(
                    event_type="track_departed",
                    message=f"Track {track_id} departed the scene",
                    track_id=track_id,
                    timestamp=timestamp,
                )
        )
        return events

    def _build_student_features(
        self,
        frame: np.ndarray,
        tracks: list[TrackedStudent],
    ) -> tuple[list[StudentTrackFeature], list[TrackedStudent]]:
        if self.face_analyzer is None and self.pose_analyzer is None:
            return [], tracks

        sorted_tracks = sorted(
            tracks,
            key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]),
            reverse=True,
        )
        refresh_track_ids = {
            track.track_id for track in sorted_tracks[: self.max_expression_tracks]
        }
        should_refresh = (self.frame_index % self.expression_interval) == 0
        features: list[StudentTrackFeature] = []
        updated_tracks: list[TrackedStudent] = []
        for track in tracks:
            cache_entry = self.feature_cache.get(track.track_id)
            feature = cache_entry.feature.model_copy() if cache_entry is not None else StudentTrackFeature(track_id=track.track_id)

            if should_refresh and track.track_id in refresh_track_ids:
                crop = self._crop_frame(frame, track.bbox)
                refreshed = StudentTrackFeature(track_id=track.track_id)
                if self.face_analyzer is not None:
                    face_feature = self.face_analyzer.analyze(crop, track.track_id)
                    if face_feature is not None:
                        refreshed = self._merge_features(refreshed, face_feature)
                if self.pose_analyzer is not None:
                    pose_feature = self.pose_analyzer.analyze(crop, track.track_id)
                    if pose_feature is not None:
                        refreshed = self._merge_features(refreshed, pose_feature)
                feature = refreshed

            feature.engagement_score = float(
                np.clip(
                    0.5 * feature.attention_score
                    + 0.2 * feature.posture_score
                    + 0.15 * (1.0 if feature.hand_raised else 0.0)
                    + 0.15 * feature.smile_score
                    - 0.1 * feature.confusion_score,
                    0.0,
                    1.0,
                )
            )
            self.feature_cache[track.track_id] = _FeatureCacheEntry(
                feature=feature.model_copy(),
                frame_index=self.frame_index,
            )
            features.append(feature)
            updated_tracks.append(
                track.model_copy(
                    update={
                        "attention_score": feature.attention_score,
                        "posture_score": feature.posture_score,
                        "hand_raised": feature.hand_raised,
                        "face_visible": feature.face_visible,
                    }
                )
            )
        return features, updated_tracks

    @staticmethod
    def _has_signal_features(student_features: list[StudentTrackFeature]) -> bool:
        return any(
            feature.face_visible
            or feature.posture_score > 0.0
            or feature.attention_score > 0.0
            or feature.hand_raised
            or feature.smile_score > 0.0
            or feature.confusion_score > 0.0
            for feature in student_features
        )

    @staticmethod
    def _merge_features(
        base: StudentTrackFeature,
        incoming: StudentTrackFeature,
    ) -> StudentTrackFeature:
        return base.model_copy(
            update={
                "attention_score": max(base.attention_score, incoming.attention_score),
                "engagement_score": max(base.engagement_score, incoming.engagement_score),
                "confusion_score": max(base.confusion_score, incoming.confusion_score),
                "smile_score": max(base.smile_score, incoming.smile_score),
                "posture_score": max(base.posture_score, incoming.posture_score),
                "hand_raised": base.hand_raised or incoming.hand_raised,
                "visible": base.visible or incoming.visible,
                "present": base.present or incoming.present,
                "face_visible": base.face_visible or incoming.face_visible,
            }
        )

    @staticmethod
    def _crop_frame(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return frame[y1:y2, x1:x2]

    def _compute_fps(self, timestamp: float) -> float:
        if self._last_frame_ts is None:
            self._last_frame_ts = timestamp
            return 0.0
        delta = max(timestamp - self._last_frame_ts, 1e-6)
        self._last_frame_ts = timestamp
        return 1.0 / delta

    def _build_exit_zone(self, app_runtime_config: dict[str, Any]) -> ExitZone:
        exit_zone_config = app_runtime_config.get("exit_zone", {})
        return ExitZone(
            enabled=bool(exit_zone_config.get("enabled", False)),
            x1_ratio=float(exit_zone_config.get("x1_ratio", 0.0)),
            y1_ratio=float(exit_zone_config.get("y1_ratio", 0.0)),
            x2_ratio=float(exit_zone_config.get("x2_ratio", 1.0)),
            y2_ratio=float(exit_zone_config.get("y2_ratio", 1.0)),
            min_frames_in_zone=int(exit_zone_config.get("min_frames_in_zone", 1)),
        )

    def _apply_exit_zone(self, frame: np.ndarray, tracks: list[TrackedStudent]) -> list[TrackedStudent]:
        if not self.exit_zone.enabled:
            for track in tracks:
                self.track_memory[track.track_id] = _TrackMemory(
                    last_centroid=self._bbox_center(track.bbox),
                    last_bbox=track.bbox,
                )
            return tracks

        width = frame.shape[1]
        height = frame.shape[0]
        zone = self._exit_zone_pixels(width, height, self.exit_zone)
        updated_tracks: list[TrackedStudent] = []
        for track in tracks:
            centroid = self._bbox_center(track.bbox)
            in_exit_zone = self._point_in_zone(centroid, zone)
            memory = self.track_memory.get(track.track_id)
            frames_in_zone = 1 if in_exit_zone else 0
            if memory is not None:
                frames_in_zone = memory.frames_in_exit_zone + 1 if in_exit_zone else 0
            self.track_memory[track.track_id] = _TrackMemory(
                last_centroid=centroid,
                last_bbox=track.bbox,
                frames_in_exit_zone=frames_in_zone,
                was_in_exit_zone=in_exit_zone or (memory.was_in_exit_zone if memory else False),
            )
            updated_tracks.append(track.model_copy(update={"in_exit_zone": in_exit_zone}))
        return updated_tracks

    def _classify_departures(
        self, departed: list[int], timestamp: float
    ) -> tuple[list[int], list[int]]:
        del timestamp
        exit_events: list[int] = []
        scene_departures: list[int] = []
        for track_id in departed:
            self.feature_cache.pop(track_id, None)
            memory = self.track_memory.pop(track_id, None)
            if memory is None:
                scene_departures.append(track_id)
                continue

            if self.exit_zone.enabled and memory.frames_in_exit_zone >= self.exit_zone.min_frames_in_zone:
                exit_events.append(track_id)
            else:
                scene_departures.append(track_id)
        return exit_events, scene_departures

    @staticmethod
    def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _point_in_zone(point: tuple[float, float], zone: tuple[int, int, int, int]) -> bool:
        x, y = point
        return zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]

    @staticmethod
    def _exit_zone_pixels(
        width: int,
        height: int,
        exit_zone: ExitZone,
    ) -> tuple[int, int, int, int]:
        x1 = int(width * exit_zone.x1_ratio)
        y1 = int(height * exit_zone.y1_ratio)
        x2 = int(width * exit_zone.x2_ratio)
        y2 = int(height * exit_zone.y2_ratio)
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
