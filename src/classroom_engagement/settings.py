from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any

from classroom_engagement.config import load_yaml


@dataclass(slots=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "dev")
    app_name: str = os.getenv("APP_NAME", "classroom-engagement-analytics")
    source_type: str = os.getenv("SOURCE_TYPE", "webcam")
    source_uri: str = os.getenv("SOURCE_URI", "0")
    session_output_dir: Path = Path(os.getenv("SESSION_OUTPUT_DIR", "outputs/sessions"))
    config_dir: Path = Path(os.getenv("CONFIG_DIR", "configs"))
    detector_name: str = os.getenv("DETECTOR_NAME", "")
    tracker_name: str = os.getenv("TRACKER_NAME", "")

    def models_config_path(self) -> Path:
        return self.config_dir / "models.yaml"

    def app_config_path(self) -> Path:
        return self.config_dir / "app.yaml"

    def load_models_config(self) -> dict[str, Any]:
        path = self.models_config_path()
        if not path.exists():
            return {}
        return load_yaml(path)

    def load_app_config(self) -> dict[str, Any]:
        path = self.app_config_path()
        if not path.exists():
            return {}
        return load_yaml(path)

    def resolved_detector_name(self) -> str:
        if self.detector_name:
            return self.detector_name
        models_config = self.load_models_config()
        return models_config.get("models", {}).get("detector", {}).get("name", "motion_blob")

    def resolved_tracker_name(self) -> str:
        if self.tracker_name:
            return self.tracker_name
        models_config = self.load_models_config()
        return models_config.get("models", {}).get("tracker", {}).get("name", "centroid")


def get_settings() -> Settings:
    return Settings()
