from __future__ import annotations

import copy
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
import time
import cv2

from classroom_engagement.api.schemas import (
    EventsResponse,
    SessionControlRequest,
    SessionControlResponse,
    SessionStatusResponse,
    SnapshotResponse,
)
from classroom_engagement.core.schemas import PipelineEvent, PipelineResult
from classroom_engagement.pipeline.engine import PipelineEngine
from classroom_engagement.settings import Settings, get_settings
from classroom_engagement.video.source import VideoSource, VideoSourceConfig


@dataclass(slots=True)
class _SessionState:
    session_id: str = "default"
    state: str = "idle"
    source_type: str = "webcam"
    source_uri: str = "0"
    detector_name: str = "motion_blob"
    tracker_name: str = "centroid"
    events: list[PipelineEvent] = field(default_factory=list)
    last_snapshot: PipelineResult | None = None
    last_error: str | None = None
    last_frame_jpeg: bytes | None = None


@dataclass(slots=True)
class _RunnerState:
    thread: Thread | None = None
    stop_event: Event | None = None
    source: VideoSource | None = None
    engine: PipelineEngine | None = None


class SessionService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._lock = Lock()
        self._state = _SessionState(
            source_type=self.settings.source_type,
            source_uri=self.settings.source_uri,
            detector_name=self.settings.resolved_detector_name(),
            tracker_name=self.settings.resolved_tracker_name(),
        )
        self._runner = _RunnerState()
        self._models_config = self.settings.load_models_config().get("models", {})
        self._app_config = self.settings.load_app_config()

    def root_payload(self) -> dict[str, str]:
        return {
            "name": "Classroom Engagement Analytics API",
            "version": "0.1.0",
            "docs_url": "/docs",
            "health_url": "/health",
            "session_status_url": "/session/status",
            "session_snapshot_url": "/session/snapshot",
            "session_frame_url": "/session/frame",
        }

    def health_payload(self) -> dict[str, str]:
        return {"status": "ok"}

    def version_payload(self) -> dict[str, str]:
        return {"version": "0.1.0"}

    def get_status(self) -> SessionStatusResponse:
        with self._lock:
            frame_index = self._state.last_snapshot.frame_index if self._state.last_snapshot else 0
            return SessionStatusResponse(
                session_id=self._state.session_id,
                state=self._state.state,
                source_type=self._state.source_type,
                source_uri=self._state.source_uri,
                detector_name=self._state.detector_name,
                tracker_name=self._state.tracker_name,
                event_count=len(self._state.events),
                has_snapshot=self._state.last_snapshot is not None,
                last_error=self._state.last_error,
                frame_index=frame_index,
            )

    def start_session(self, request: SessionControlRequest) -> SessionControlResponse:
        self._stop_runner(emit_event=False)

        with self._lock:
            self._state.state = "stopped"
            self._state.source_type = request.source_type
            self._state.source_uri = request.source_uri
            self._state.detector_name = request.detector_name
            self._state.tracker_name = request.tracker_name
            self._state.events = []
            self._state.last_snapshot = None
            self._state.last_error = None
            self._state.last_frame_jpeg = None

        runtime_models_config = self._build_runtime_models_config()
        engine = PipelineEngine(
            detector_name=request.detector_name,
            tracker_name=request.tracker_name,
            models_config=runtime_models_config,
        )
        source = VideoSource(self._build_source_config(request))

        try:
            source.open()
            frame, timestamp = source.read()
            if frame is None:
                raise RuntimeError("Video source did not return an initial frame.")
            result = engine.process_frame(frame=frame, timestamp=timestamp, source_name=source.describe())
            frame_jpeg = self._encode_frame(engine.annotate_frame(frame, result))
        except Exception as exc:
            source.release()
            with self._lock:
                self._state.state = "stopped"
                self._state.last_error = str(exc)
            self._append_event(
                PipelineEvent(
                    event_type="warning",
                    message=f"Session failed to start: {exc}",
                    timestamp=time.time(),
                )
            )
            return SessionControlResponse(
                ok=False,
                message=f"Session failed to start: {exc}",
                status=self.get_status(),
            )

        with self._lock:
            self._state.state = "running"
            self._state.last_snapshot = result
            self._state.last_frame_jpeg = frame_jpeg
            self._runner = _RunnerState(
                thread=None,
                stop_event=Event(),
                source=source,
                engine=engine,
            )

        self._append_event(
            PipelineEvent(
                event_type="system",
                message=(
                    f"Session started with source={request.source_type}:{request.source_uri}, "
                    f"detector={request.detector_name}, tracker={request.tracker_name}"
                ),
                timestamp=time.time(),
            )
        )
        self._merge_result_events(result)

        thread = Thread(target=self._run_loop, name="session-runner", daemon=True)
        self._runner.thread = thread
        thread.start()
        return SessionControlResponse(message="Session started", status=self.get_status())

    def stop_session(self) -> SessionControlResponse:
        was_running = self._stop_runner(emit_event=True)
        with self._lock:
            self._state.state = "stopped"
        return SessionControlResponse(
            message="Session stopped" if was_running else "Session already stopped",
            status=self.get_status(),
        )

    def get_snapshot(self) -> SnapshotResponse:
        with self._lock:
            snapshot = copy.deepcopy(self._state.last_snapshot)
        return SnapshotResponse(status=self.get_status(), snapshot=snapshot)

    def get_events(self, limit: int = 50) -> EventsResponse:
        with self._lock:
            events = list(self._state.events[-limit:])
        return EventsResponse(status=self.get_status(), events=events)

    def get_latest_frame_jpeg(self) -> bytes | None:
        with self._lock:
            return self._state.last_frame_jpeg

    def _append_event(self, event: PipelineEvent) -> None:
        with self._lock:
            self._state.events.append(event)
            self._state.events = self._state.events[-200:]

    def _merge_result_events(self, result: PipelineResult) -> None:
        if not result.events:
            return
        with self._lock:
            self._state.events.extend(result.events)
            self._state.events = self._state.events[-200:]

    def _run_loop(self) -> None:
        assert self._runner.stop_event is not None
        assert self._runner.source is not None
        assert self._runner.engine is not None

        target_fps = int(self._app_config.get("runtime", {}).get("target_fps", 15))
        sleep_seconds = 1.0 / max(target_fps, 1)

        try:
            while not self._runner.stop_event.is_set():
                frame, timestamp = self._runner.source.read()
                if frame is None:
                    self._append_event(
                        PipelineEvent(
                            event_type="warning",
                            message="Video stream ended or frame could not be read.",
                            timestamp=time.time(),
                        )
                    )
                    with self._lock:
                        self._state.state = "stopped"
                    break

                result = self._runner.engine.process_frame(
                    frame=frame,
                    timestamp=timestamp,
                    source_name=self._runner.source.describe(),
                )
                annotated = self._runner.engine.annotate_frame(frame, result)
                encoded_frame = self._encode_frame(annotated)
                with self._lock:
                    self._state.last_snapshot = result
                    self._state.last_error = None
                    self._state.last_frame_jpeg = encoded_frame
                self._merge_result_events(result)

                if self._state.source_type == "synthetic":
                    time.sleep(sleep_seconds)
        except Exception as exc:
            self._append_event(
                PipelineEvent(
                    event_type="warning",
                    message=f"Session runtime error: {exc}",
                    timestamp=time.time(),
                )
            )
            with self._lock:
                self._state.state = "stopped"
                self._state.last_error = str(exc)
        finally:
            if self._runner.source is not None:
                self._runner.source.release()

    def _stop_runner(self, emit_event: bool) -> bool:
        thread = self._runner.thread
        stop_event = self._runner.stop_event
        was_running = thread is not None or stop_event is not None

        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        if self._runner.source is not None:
            self._runner.source.release()

        self._runner = _RunnerState()
        if emit_event:
            self._append_event(
                PipelineEvent(
                    event_type="system",
                    message="Session stopped",
                    timestamp=time.time(),
                )
            )
        return was_running

    def _build_source_config(self, request: SessionControlRequest) -> VideoSourceConfig:
        runtime_config = self._app_config.get("runtime", {})
        return VideoSourceConfig(
            source_type=request.source_type,
            source_uri=request.source_uri,
            max_frame_width=int(runtime_config.get("max_frame_width", 1280)),
        )

    def _build_runtime_models_config(self) -> dict:
        runtime_models_config = copy.deepcopy(self._models_config)
        runtime_models_config["app_runtime"] = {
            "exit_zone": self._app_config.get("exit_zone", {})
        }
        return runtime_models_config

    @staticmethod
    def _encode_frame(frame) -> bytes | None:
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return encoded.tobytes()


_SESSION_SERVICE: SessionService | None = None


def get_session_service() -> SessionService:
    global _SESSION_SERVICE
    if _SESSION_SERVICE is None:
        _SESSION_SERVICE = SessionService()
    return _SESSION_SERVICE
