"""Domain models shared by YOLO26 detection layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


ImageArray = np.ndarray
PathLike = Union[str, Path]


@dataclass(frozen=True)
class CameraBackend:
    """OpenCV backend used for camera access."""

    name: str
    api_id: int


@dataclass(frozen=True)
class PlatformInfo:
    """Detected operating system and preferred camera backends."""

    system: str
    backends: Tuple[CameraBackend, ...]

    def label(self) -> str:
        names = ", ".join(backend.name for backend in self.backends)
        return f"{self.system} | backends: {names}"


@dataclass(frozen=True)
class CameraDevice:
    """A camera index opened through one backend."""

    index: int
    backend: CameraBackend
    name: str

    def key(self) -> str:
        return f"{self.backend.name}:{self.index}"

    def label(self) -> str:
        return f"{self.name} ({self.backend.name})"


@dataclass(frozen=True)
class ModelInfo:
    """A YOLO26 model file candidate."""

    name: str
    path: Path
    exists: bool
    is_official: bool = False

    def label(self) -> str:
        state = "" if self.exists else " [missing]"
        return f"{self.name}{state}"


@dataclass(frozen=True)
class DetectionSettings:
    """Runtime inference settings."""

    image_size: int = 640
    confidence: float = 0.25
    iou: float = 0.45
    device: str = "auto"

    def normalized_device(self) -> Optional[str]:
        value = self.device.strip().lower()
        return None if value in ("", "auto") else value


@dataclass(frozen=True)
class DetectionBox:
    """One detected object."""

    class_id: int
    class_name: str
    confidence: float
    xyxy: Tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionOutput:
    """Annotated frame plus structured detections."""

    annotated_frame: ImageArray
    detections: Tuple[DetectionBox, ...]
    inference_ms: float

    @property
    def detection_count(self) -> int:
        return len(self.detections)
