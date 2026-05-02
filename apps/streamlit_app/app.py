from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from classroom_engagement.pipeline.engine import PipelineEngine
from classroom_engagement.settings import get_settings
from classroom_engagement.ui.formatting import format_event_rows
from classroom_engagement.video.source import VideoSource, VideoSourceConfig


st.set_page_config(page_title="Classroom Engagement Analytics", layout="wide")
SETTINGS = get_settings()
MODELS_CONFIG = SETTINGS.load_models_config().get("models", {})
APP_CONFIG = SETTINGS.load_app_config()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

        :root {
            --bg-top: #08131f;
            --bg-bottom: #10273b;
            --panel: #0b1927;
            --panel-soft: #12263a;
            --border: rgba(133, 173, 210, 0.18);
            --text: #eef6ff;
            --muted: #9eb4ca;
            --good: #37d39a;
            --warn: #ffbb45;
            --bad: #ff6d7a;
            --accent: #57c8ff;
        }

        html, body, [class*="css"]  {
            font-family: "Space Grotesk", sans-serif;
            -webkit-font-smoothing: antialiased;
            text-rendering: geometricPrecision;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(87, 200, 255, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(55, 211, 154, 0.10), transparent 20%),
                linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
        }

        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }

        section[data-testid="stSidebar"] {
            background: rgba(4, 10, 18, 0.88);
            border-right: 1px solid var(--border);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .panel-card {
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1rem 1rem 1.05rem 1rem;
            background: var(--panel);
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.18);
        }

        .section-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .status-panel {
            border: 1px solid var(--border);
            border-radius: 24px;
            background: #0c1b2b;
            padding: 1.1rem;
            min-height: 28rem;
        }

        .status-kicker {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.72rem;
            margin-bottom: 0.55rem;
        }

        .state-pill {
            display: inline-block;
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }

        .status-headline {
            color: var(--text);
            font-size: 1.8rem;
            line-height: 1.05;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .status-summary {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.45;
            margin-bottom: 1rem;
        }

        .status-mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.95rem 0 1rem 0;
        }

        .mini-card {
            border-radius: 18px;
            padding: 0.85rem;
            background: var(--panel-soft);
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        .mini-label {
            color: var(--muted);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.15rem;
        }

        .mini-value {
            color: var(--text);
            font-size: 1.45rem;
            font-weight: 700;
        }

        .gauge-row {
            margin-bottom: 0.75rem;
        }

        .gauge-head {
            display: flex;
            justify-content: space-between;
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 0.35rem;
        }

        .gauge-track {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            overflow: hidden;
        }

        .gauge-fill {
            height: 100%;
            border-radius: 999px;
        }

        .insight-box {
            margin-top: 0.95rem;
            border-radius: 18px;
            padding: 0.9rem;
            background: #163047;
            border: 1px solid rgba(87, 200, 255, 0.18);
        }

        .insight-title {
            color: #8adfff;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            margin-bottom: 0.25rem;
        }

        .insight-text {
            color: var(--text);
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .metric-card {
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 0.95rem 1rem;
            background: var(--panel);
            min-height: 7.5rem;
        }

        .metric-label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.09em;
            font-size: 0.74rem;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            color: var(--text);
            font-size: 2rem;
            line-height: 1;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .metric-foot {
            color: #89d2ff;
            font-size: 0.82rem;
        }

        .feed-caption {
            color: var(--muted);
            font-size: 0.84rem;
            margin: 0.35rem 0 0.2rem 0;
        }

        .small-note {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _recommended_detector_name(source_type: str) -> str:
    return "hybrid_person"


def _init_state() -> None:
    st.session_state.setdefault("video_source", None)
    st.session_state.setdefault("pipeline_engine", None)
    st.session_state.setdefault("event_rows", [])
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("source_signature", None)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_frame_rgb", None)
    st.session_state.setdefault("warning_message", "")
    exit_zone_defaults = APP_CONFIG.get("exit_zone", {})
    st.session_state.setdefault("exit_zone_enabled", bool(exit_zone_defaults.get("enabled", True)))
    st.session_state.setdefault("exit_x1", float(exit_zone_defaults.get("x1_ratio", 0.78)))
    st.session_state.setdefault("exit_y1", float(exit_zone_defaults.get("y1_ratio", 0.15)))
    st.session_state.setdefault("exit_x2", float(exit_zone_defaults.get("x2_ratio", 0.98)))
    st.session_state.setdefault("exit_y2", float(exit_zone_defaults.get("y2_ratio", 0.95)))
    st.session_state.setdefault("exit_min_frames", int(exit_zone_defaults.get("min_frames_in_zone", 1)))


def _release_source() -> None:
    source = st.session_state.get("video_source")
    if source is not None:
        source.release()
    st.session_state["video_source"] = None


def _reset_runtime() -> None:
    _release_source()
    st.session_state["pipeline_engine"] = None
    st.session_state["event_rows"] = []
    st.session_state["last_result"] = None
    st.session_state["last_frame_rgb"] = None
    st.session_state["warning_message"] = ""


def _build_source_config(source_type: str, source_uri: str, max_frame_width: int) -> VideoSourceConfig:
    return VideoSourceConfig(
        source_type=source_type,
        source_uri=source_uri,
        max_frame_width=max_frame_width,
    )


def _ensure_runtime(source_config: VideoSourceConfig) -> None:
    signature = (
        source_config.source_type,
        source_config.source_uri,
        source_config.max_frame_width,
        st.session_state["detector_name"],
        st.session_state["tracker_name"],
        st.session_state["exit_zone_enabled"],
        st.session_state["exit_x1"],
        st.session_state["exit_y1"],
        st.session_state["exit_x2"],
        st.session_state["exit_y2"],
        st.session_state["exit_min_frames"],
    )
    if st.session_state["source_signature"] != signature:
        _reset_runtime()
        st.session_state["source_signature"] = signature

    if st.session_state["pipeline_engine"] is None:
        runtime_models_config = dict(MODELS_CONFIG)
        runtime_models_config["app_runtime"] = {
            "exit_zone": {
                "enabled": st.session_state["exit_zone_enabled"],
                "x1_ratio": st.session_state["exit_x1"],
                "y1_ratio": st.session_state["exit_y1"],
                "x2_ratio": st.session_state["exit_x2"],
                "y2_ratio": st.session_state["exit_y2"],
                "min_frames_in_zone": st.session_state["exit_min_frames"],
            }
        }
        st.session_state["pipeline_engine"] = PipelineEngine(
            detector_name=st.session_state["detector_name"],
            tracker_name=st.session_state["tracker_name"],
            models_config=runtime_models_config,
        )

    if st.session_state["video_source"] is None:
        st.session_state["video_source"] = VideoSource(source_config)


def _append_events(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    st.session_state["event_rows"] = (rows + st.session_state["event_rows"])[:50]


def _run_single_iteration(source_config: VideoSourceConfig) -> None:
    try:
        _ensure_runtime(source_config)
        source: VideoSource = st.session_state["video_source"]
        engine: PipelineEngine = st.session_state["pipeline_engine"]

        frame, timestamp = source.read()
    except Exception as exc:
        st.session_state["warning_message"] = str(exc)
        st.session_state["running"] = False
        _release_source()
        return

    if frame is None:
        st.session_state["warning_message"] = "Video stream ended or frame could not be read."
        st.session_state["running"] = False
        _release_source()
        return

    result = engine.process_frame(frame=frame, timestamp=timestamp, source_name=source.describe())
    annotated = engine.annotate_frame(frame, result)

    st.session_state["last_result"] = result
    st.session_state["last_frame_rgb"] = annotated[:, :, ::-1]
    _append_events(format_event_rows(result.events))


def _status_style(state: str) -> tuple[str, str, str]:
    normalized = state.strip().lower()
    if normalized == "engaged":
        return "#0e3f2f", "#37d39a", "Students are attentive and absorbing the lecture."
    if normalized == "energized":
        return "#46310d", "#ffbb45", "The room feels lively, interactive, and emotionally active."
    if normalized == "confused":
        return "#3f2024", "#ff8a93", "Students appear uncertain. The teacher may need to slow down."
    if normalized == "disengaged":
        return "#431f27", "#ff6d7a", "Attention is dropping. This is a good moment to re-engage the class."
    if normalized == "neutral":
        return "#163044", "#57c8ff", "The class is stable, but interaction is still limited."
    return "#223343", "#b6c8d8", "The system is still gathering evidence from the room."


def _summary_text(result) -> str:
    snapshot = result.class_snapshot
    if snapshot.student_count == 0:
        return "No clear student tracks are active yet. Adjust framing or let the detector stabilize for a few seconds."
    if snapshot.state == "energized":
        return "Students look active and responsive. This is a strong moment for discussion, humor, or interactive prompts."
    if snapshot.state == "engaged":
        return "Class attention is healthy. Continue with the current teaching flow or move into the next important concept."
    if snapshot.state == "confused":
        return "Confusion signals are elevated. Consider repeating the last concept with a slower explanation or example."
    if snapshot.state == "disengaged":
        return "Participation is weak and class energy is fading. A question, demo, or movement break may help."
    return "The room is being monitored in real time. Watch attention, posture, and exits for the next shift in class mood."


def _teacher_tip(result) -> str:
    snapshot = result.class_snapshot
    if snapshot.student_count == 0:
        return "Move the camera slightly back and keep upper bodies visible for stronger detection and tracking."
    if snapshot.exit_events > 0:
        return "One or more students appear to have left the room. Check whether attendance drift is affecting engagement."
    if snapshot.attention_ratio < 0.35:
        return "Attention is low. Ask a direct question or switch to a more visual example."
    if snapshot.posture_ratio < 0.35:
        return "Body posture suggests the class is flattening. Try a faster pace or a short interaction."
    if snapshot.interaction_ratio > 0.25:
        return "Interaction level is rising. This is a good time to encourage responses from more students."
    return "The current classroom state is steady. Use the event timeline to watch for the next meaningful shift."


def _render_metric_card(title: str, value: str, foot: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_gauge(label: str, value: float, color: str) -> None:
    safe_value = max(0.0, min(1.0, float(value)))
    st.markdown(
        f"""
        <div class="gauge-row">
            <div class="gauge-head">
                <span>{label}</span>
                <span>{safe_value * 100:.0f}%</span>
            </div>
            <div class="gauge-track">
                <div class="gauge-fill" style="width:{safe_value * 100:.0f}%; background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_status_panel(result, source_config: VideoSourceConfig) -> None:
    if result is None:
        state = "unknown"
        pill_bg, pill_fg, headline = _status_style(state)
        summary = "Start the pipeline to populate the classroom status board."
        attention = 0.0
        posture = 0.0
        face_visible = 0.0
        interaction = 0.0
        confusion = 0.0
        tip = "For demos, keep the camera stable and make sure students' upper bodies remain visible."
        source_name = source_config.source_type
        frame_index = 0
    else:
        snapshot = result.class_snapshot
        state = snapshot.state
        pill_bg, pill_fg, headline = _status_style(state)
        summary = _summary_text(result)
        attention = snapshot.attention_ratio
        posture = snapshot.posture_ratio
        face_visible = snapshot.face_visible_ratio
        interaction = snapshot.interaction_ratio
        confusion = snapshot.confusion_ratio
        tip = _teacher_tip(result)
        source_name = result.source_name
        frame_index = result.frame_index

    st.markdown('<div class="status-kicker">Class Status Board</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="state-pill" style="background:{pill_bg}; color:{pill_fg};">{state.title()}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="status-headline">{headline}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="status-summary">{summary}</div>', unsafe_allow_html=True)

    _render_gauge("Attention", attention, "#57c8ff")
    _render_gauge("Posture", posture, "#37d39a")
    _render_gauge("Face Visibility", face_visible, "#ffbb45")
    _render_gauge("Interaction", interaction, "#f48fff")
    _render_gauge("Confusion", confusion, "#ff6d7a")

    st.markdown(
        f"""
        <div class="insight-box">
            <div class="insight-title">Live Teaching Hint</div>
            <div class="insight-text">{tip}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Source: {source_name} | Frame: {frame_index}")


_inject_styles()
_init_state()

available_detectors = MODELS_CONFIG.get(
    "detector", {}
).get("available", ["hybrid_person", "motion_blob", "hog_person", "yolox_onnx"])
available_trackers = MODELS_CONFIG.get("tracker", {}).get("available", ["centroid"])
default_detector = SETTINGS.resolved_detector_name()
default_tracker = SETTINGS.resolved_tracker_name()
st.session_state.setdefault("detector_name", default_detector)
st.session_state.setdefault("tracker_name", default_tracker)

with st.sidebar:
    st.header("Demo Control")
    source_type = st.radio("Input Type", options=["webcam", "video_file"], horizontal=True)
    if source_type == "webcam":
        webcam_index = st.number_input("Webcam Index", min_value=0, max_value=10, value=0, step=1)
        source_uri = str(webcam_index)
    elif source_type == "video_file":
        source_uri = st.text_input("Video File Path", value="")

    st.subheader("Runtime")
    detector_name = st.selectbox(
        "Detector Backend",
        options=available_detectors,
        index=available_detectors.index(st.session_state["detector_name"])
        if st.session_state["detector_name"] in available_detectors
        else 0,
    )
    tracker_name = st.selectbox(
        "Tracker Backend",
        options=available_trackers,
        index=available_trackers.index(st.session_state["tracker_name"])
        if st.session_state["tracker_name"] in available_trackers
        else 0,
    )
    max_frame_width = st.slider("Max Frame Width", min_value=640, max_value=1920, value=1280, step=160)
    target_fps = st.slider("Target Refresh FPS", min_value=1, max_value=20, value=8, step=1)
    st.session_state["detector_name"] = detector_name
    st.session_state["tracker_name"] = tracker_name

    st.subheader("Exit Zone")
    exit_zone_enabled = st.toggle("Enable Exit Zone", value=st.session_state["exit_zone_enabled"])
    exit_x1 = st.slider("Exit X1", min_value=0.0, max_value=1.0, value=st.session_state["exit_x1"], step=0.01)
    exit_y1 = st.slider("Exit Y1", min_value=0.0, max_value=1.0, value=st.session_state["exit_y1"], step=0.01)
    exit_x2 = st.slider("Exit X2", min_value=0.0, max_value=1.0, value=st.session_state["exit_x2"], step=0.01)
    exit_y2 = st.slider("Exit Y2", min_value=0.0, max_value=1.0, value=st.session_state["exit_y2"], step=0.01)
    exit_min_frames = st.slider(
        "Frames In Zone For Exit",
        min_value=1,
        max_value=20,
        value=st.session_state["exit_min_frames"],
        step=1,
    )
    st.session_state["exit_zone_enabled"] = exit_zone_enabled
    st.session_state["exit_x1"] = min(exit_x1, exit_x2)
    st.session_state["exit_y1"] = min(exit_y1, exit_y2)
    st.session_state["exit_x2"] = max(exit_x1, exit_x2)
    st.session_state["exit_y2"] = max(exit_y1, exit_y2)
    st.session_state["exit_min_frames"] = exit_min_frames

    start_col, stop_col = st.columns(2)
    if start_col.button("Start", width="stretch"):
        if source_type == "video_file" and not source_uri.strip():
            st.session_state["warning_message"] = "Provide a valid video file path before starting."
        else:
            st.session_state["warning_message"] = ""
            st.session_state["running"] = True
    if stop_col.button("Stop", width="stretch"):
        st.session_state["running"] = False
        _release_source()

    if st.button("Reset Session", width="stretch"):
        st.session_state["running"] = False
        _reset_runtime()

    st.caption("Recommended webcam mode: `hybrid_person + bytetrack`.")

source_config = _build_source_config(
    source_type=source_type,
    source_uri=source_uri.strip(),
    max_frame_width=max_frame_width,
)

if source_type == "video_file" and source_uri.strip():
    video_path = Path(source_uri.strip())
    if not video_path.exists():
        st.warning(f"Video file not found: {video_path}")

if st.session_state["warning_message"]:
    st.warning(st.session_state["warning_message"])

refresh_ms = max(50, int(1000 / max(target_fps, 1)))
status_refresh_ms = max(400, refresh_ms * 3)
events_refresh_ms = max(700, refresh_ms * 5)


@st.fragment(run_every=f"{refresh_ms}ms")
def pipeline_tick() -> None:
    if st.session_state["running"]:
        _run_single_iteration(source_config)


def _metric_data(result):
    if result is None:
        return [
            ("Student Count", "--", "Waiting for input"),
            ("Engagement Score", "--", "No active estimate"),
            ("Exited Students", "--", "No exit events yet"),
            ("Avg FPS", "--", "Runtime idle"),
        ]
    return [
        (
            "Student Count",
            str(result.class_snapshot.student_count),
            f"Tracks {result.session_metrics.active_tracks}",
        ),
        (
            "Engagement Score",
            f"{result.class_snapshot.engagement_score * 100:.0f}%",
            f"State: {result.class_snapshot.state.title()}",
        ),
        (
            "Exited Students",
            str(result.class_snapshot.exit_events),
            "Door-zone exits",
        ),
        (
            "Avg FPS",
            f"{result.session_metrics.average_fps:.1f}",
            "Local realtime loop",
        ),
    ]


@st.fragment(run_every=f"{status_refresh_ms}ms")
def render_metrics() -> None:
    result = st.session_state["last_result"]
    stats_cols = st.columns(4)
    for column, metric in zip(stats_cols, _metric_data(result)):
        with column:
            _render_metric_card(metric[0], metric[1], metric[2])


@st.fragment(run_every=f"{refresh_ms}ms")
def render_feed_panel() -> None:
    result = st.session_state["last_result"]
    st.markdown('<div class="section-title">Live Classroom Feed</div>', unsafe_allow_html=True)
    warning_message = st.session_state.get("warning_message", "")
    if warning_message and st.session_state["last_frame_rgb"] is None:
        st.warning(warning_message)
    if st.session_state["last_frame_rgb"] is not None:
        st.image(st.session_state["last_frame_rgb"], channels="RGB", width="stretch")
    else:
        st.info("Press Start to begin the demonstration feed.")
    if result is None:
        st.markdown(
            '<div class="feed-caption">No live frame yet. Start a source to begin the classroom demo.</div>',
            unsafe_allow_html=True,
        )
    else:
        detector_runtime = (
            "DirectML-assisted hybrid detector"
            if st.session_state["detector_name"] in {"yolox_onnx", "hybrid_person"}
            else "CPU/OpenCV detector"
        )
        st.markdown(
            f'<div class="feed-caption">Source: {result.source_name} | Frame: {result.frame_index} | '
            f'Loop FPS: {result.fps:.1f} | Runtime: {detector_runtime}</div>',
            unsafe_allow_html=True,
        )


@st.fragment(run_every=f"{status_refresh_ms}ms")
def render_status_board() -> None:
    _render_status_panel(st.session_state["last_result"], source_config)


@st.fragment(run_every=f"{events_refresh_ms}ms")
def render_events_panel() -> None:
    st.markdown('<div class="section-title">Recent Classroom Events</div>', unsafe_allow_html=True)
    event_rows = st.session_state["event_rows"]
    if event_rows:
        st.dataframe(event_rows[:10], width="stretch", hide_index=True)
    else:
        st.info("No events yet. Track entries, departures, and exit-zone activity will appear here.")


pipeline_tick()
render_metrics()
left_col, right_col = st.columns([1.7, 1], gap="large")
with left_col:
    render_feed_panel()
with right_col:
    render_status_board()
render_events_panel()
