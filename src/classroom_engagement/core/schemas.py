from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Detection(BaseModel):
    bbox: tuple[int, int, int, int]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    label: str = "student"


class TrackedStudent(BaseModel):
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    lost_frames: int = 0
    in_exit_zone: bool = False
    attention_score: float = Field(default=0.0, ge=0.0, le=1.0)
    posture_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hand_raised: bool = False
    face_visible: bool = False


class StudentTrackFeature(BaseModel):
    track_id: int
    attention_score: float = Field(default=0.0, ge=0.0, le=1.0)
    engagement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confusion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    smile_score: float = Field(default=0.0, ge=0.0, le=1.0)
    posture_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hand_raised: bool = False
    visible: bool = True
    present: bool = True
    face_visible: bool = False


class ClassSnapshot(BaseModel):
    student_count: int = 0
    visible_student_count: int = 0
    exit_events: int = 0
    scene_departures: int = 0
    engagement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    attention_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    confusion_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    smile_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    interaction_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    posture_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    face_visible_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    state: str = "unknown"


class PipelineEvent(BaseModel):
    event_type: Literal[
        "system",
        "track_entered",
        "track_departed",
        "student_exited_room",
        "warning",
    ]
    message: str
    track_id: int | None = None
    timestamp: float = 0.0


class ExitZone(BaseModel):
    enabled: bool = False
    x1_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    y1_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    x2_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    y2_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    min_frames_in_zone: int = Field(default=1, ge=1)


class SessionMetrics(BaseModel):
    total_tracks_seen: int = 0
    active_tracks: int = 0
    scene_departures: int = 0
    room_exit_events: int = 0
    event_count: int = 0
    average_fps: float = Field(default=0.0, ge=0.0)


class PipelineResult(BaseModel):
    frame_index: int = 0
    fps: float = Field(default=0.0, ge=0.0)
    source_name: str = "unknown"
    class_snapshot: ClassSnapshot = Field(default_factory=ClassSnapshot)
    tracks: list[TrackedStudent] = Field(default_factory=list)
    student_features: list[StudentTrackFeature] = Field(default_factory=list)
    events: list[PipelineEvent] = Field(default_factory=list)
    exit_zone: ExitZone = Field(default_factory=ExitZone)
    session_metrics: SessionMetrics = Field(default_factory=SessionMetrics)
