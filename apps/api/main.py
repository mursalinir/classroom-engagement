from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

from classroom_engagement.api.schemas import (
    EventsResponse,
    RootResponse,
    SessionControlRequest,
    SessionControlResponse,
    SessionStatusResponse,
    SnapshotResponse,
)
from classroom_engagement.api.service import get_session_service

app = FastAPI(title="Classroom Engagement Analytics API", version="0.1.0")
session_service = get_session_service()


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return RootResponse(**session_service.root_payload())


@app.get("/health")
def health() -> dict[str, str]:
    return session_service.health_payload()


@app.get("/version")
def version() -> dict[str, str]:
    return session_service.version_payload()


@app.get("/session/status", response_model=SessionStatusResponse)
def session_status() -> SessionStatusResponse:
    return session_service.get_status()


@app.post("/session/start", response_model=SessionControlResponse)
def session_start(request: SessionControlRequest) -> SessionControlResponse:
    return session_service.start_session(request)


@app.post("/session/stop", response_model=SessionControlResponse)
def session_stop() -> SessionControlResponse:
    return session_service.stop_session()


@app.get("/session/snapshot", response_model=SnapshotResponse)
def session_snapshot() -> SnapshotResponse:
    return session_service.get_snapshot()


@app.get("/session/events", response_model=EventsResponse)
def session_events(limit: int = Query(default=50, ge=1, le=200)) -> EventsResponse:
    return session_service.get_events(limit=limit)


@app.get("/session/frame")
def session_frame() -> Response:
    frame_jpeg = session_service.get_latest_frame_jpeg()
    if frame_jpeg is None:
        raise HTTPException(status_code=404, detail="No annotated frame is available yet.")
    return Response(content=frame_jpeg, media_type="image/jpeg")
