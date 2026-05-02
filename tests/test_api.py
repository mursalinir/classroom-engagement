import time

from fastapi.testclient import TestClient

from apps.api.main import app


def test_api_session_routes() -> None:
    client = TestClient(app)
    client.post("/session/stop")

    root_response = client.get("/")
    assert root_response.status_code == 200
    assert root_response.json()["docs_url"] == "/docs"
    assert root_response.json()["session_frame_url"] == "/session/frame"

    start_response = client.post(
        "/session/start",
        json={
            "source_type": "synthetic",
            "source_uri": "demo",
            "detector_name": "motion_blob",
            "tracker_name": "centroid",
        },
    )
    assert start_response.status_code == 200
    assert start_response.json()["status"]["state"] == "running"

    snapshot_payload = None
    for _ in range(10):
        snapshot_response = client.get("/session/snapshot")
        assert snapshot_response.status_code == 200
        snapshot_payload = snapshot_response.json()
        if snapshot_payload["snapshot"] is not None and snapshot_payload["snapshot"]["frame_index"] >= 1:
            break
        time.sleep(0.1)

    assert snapshot_payload is not None
    assert snapshot_payload["snapshot"] is not None
    assert snapshot_payload["snapshot"]["source_name"] == "synthetic:demo"
    assert snapshot_payload["status"]["has_snapshot"] is True

    events_response = client.get("/session/events")
    assert events_response.status_code == 200
    assert isinstance(events_response.json()["events"], list)

    frame_response = client.get("/session/frame")
    assert frame_response.status_code == 200
    assert frame_response.headers["content-type"] == "image/jpeg"

    stop_response = client.post("/session/stop")
    assert stop_response.status_code == 200
    assert stop_response.json()["status"]["state"] == "stopped"


def test_api_session_routes_with_yolox_runtime() -> None:
    client = TestClient(app)
    client.post("/session/stop")

    start_response = client.post(
        "/session/start",
        json={
            "source_type": "synthetic",
            "source_uri": "demo",
            "detector_name": "yolox_onnx",
            "tracker_name": "bytetrack",
        },
    )
    assert start_response.status_code == 200
    assert start_response.json()["status"]["state"] == "running"

    snapshot_payload = None
    for _ in range(10):
        snapshot_response = client.get("/session/snapshot")
        assert snapshot_response.status_code == 200
        snapshot_payload = snapshot_response.json()
        if snapshot_payload["snapshot"] is not None and snapshot_payload["snapshot"]["frame_index"] >= 1:
            break
        time.sleep(0.1)

    assert snapshot_payload is not None
    assert snapshot_payload["snapshot"] is not None
    assert snapshot_payload["status"]["has_snapshot"] is True

    stop_response = client.post("/session/stop")
    assert stop_response.status_code == 200
    assert stop_response.json()["status"]["state"] == "stopped"
