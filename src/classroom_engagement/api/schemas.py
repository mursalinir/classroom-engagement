from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from classroom_engagement.core.schemas import PipelineEvent, PipelineResult


class SessionStatusResponse(BaseModel):
    session_id: str
    state: Literal["idle", "running", "stopped"]
    source_type: str
    source_uri: str
    detector_name: str
    tracker_name: str
    event_count: int = 0
    has_snapshot: bool = False
    last_error: str | None = None
    frame_index: int = 0


class SessionControlRequest(BaseModel):
    source_type: str = "webcam"
    source_uri: str = "0"
    detector_name: str = "motion_blob"
    tracker_name: str = "centroid"


class SessionControlResponse(BaseModel):
    ok: bool = True
    message: str
    status: SessionStatusResponse


class SnapshotResponse(BaseModel):
    status: SessionStatusResponse
    snapshot: PipelineResult | None = None


class EventsResponse(BaseModel):
    status: SessionStatusResponse
    events: list[PipelineEvent] = Field(default_factory=list)


class RootResponse(BaseModel):
    name: str
    version: str
    docs_url: str
    health_url: str
    session_status_url: str
    session_snapshot_url: str
    session_frame_url: str
