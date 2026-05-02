from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass(slots=True)
class VideoSourceConfig:
    source_type: str = "webcam"
    source_uri: str = "0"
    max_frame_width: int = 1280
    synthetic_width: int = 640
    synthetic_height: int = 360


class VideoSource:
    def __init__(self, config: VideoSourceConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None
        self._opened_source: str | int | None = None
        self._synthetic_mode = False
        self._synthetic_frame_index = 0

    @property
    def is_open(self) -> bool:
        return self._synthetic_mode or (self._capture is not None and self._capture.isOpened())

    def open(self) -> None:
        if self.is_open:
            return

        source = self._resolve_source()
        if self.config.source_type == "synthetic":
            self._synthetic_mode = True
            self._opened_source = str(source)
            self._synthetic_frame_index = 0
            return

        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        self._capture = capture
        self._opened_source = source

    def read(self) -> tuple[np.ndarray | None, float]:
        if not self.is_open:
            self.open()

        if self._synthetic_mode:
            return self._generate_synthetic_frame(), time.time()

        assert self._capture is not None
        ok, frame = self._capture.read()
        timestamp = time.time()
        if not ok or frame is None:
            return None, timestamp

        return self._resize_if_needed(frame), timestamp

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._opened_source = None
        self._synthetic_mode = False
        self._synthetic_frame_index = 0

    def describe(self) -> str:
        return f"{self.config.source_type}:{self.config.source_uri}"

    def _resolve_source(self) -> str | int:
        if self.config.source_type == "webcam":
            return int(self.config.source_uri)
        if self.config.source_type == "synthetic":
            return self.config.source_uri or "synthetic"

        path = Path(self.config.source_uri)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        return str(path)

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self.config.max_frame_width:
            return frame

        scale = self.config.max_frame_width / width
        target_size = (int(width * scale), int(height * scale))
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    def _generate_synthetic_frame(self) -> np.ndarray:
        width = min(self.config.synthetic_width, self.config.max_frame_width)
        height = self.config.synthetic_height
        frame = np.full((height, width, 3), 32, dtype=np.uint8)

        offset = (self._synthetic_frame_index * 12) % max(width - 140, 1)
        cv2.rectangle(frame, (20 + offset, 120), (95 + offset, 310), (0, 220, 0), -1)
        cv2.rectangle(
            frame,
            (width - 160 - offset // 2, 95),
            (width - 95 - offset // 2, 300),
            (0, 180, 255),
            -1,
        )
        cv2.putText(
            frame,
            "Synthetic Classroom Feed",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        self._synthetic_frame_index += 1
        return frame
