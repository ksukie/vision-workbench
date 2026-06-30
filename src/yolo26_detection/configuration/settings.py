"""Runtime configuration for YOLO26 detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_source_dir() -> Path:
    return project_root() / "third_party" / "yolo26_source"


def default_model_dir() -> Path:
    return project_root() / "models" / "yolo26_models"


def default_user_model_dir() -> Path:
    return Path.home() / ".vision_workbench" / "models" / "yolo26_models"


@dataclass(frozen=True)
class Yolo26DetectionConfig:
    """Camera, model, and inference defaults."""

    yolo26_source_dir: Path = field(default_factory=default_source_dir)
    model_dir: Path = field(default_factory=default_model_dir)
    user_model_dir: Path = field(default_factory=default_user_model_dir)
    official_model_names: Tuple[str, ...] = (
        "yolo26n.pt",
        "yolo26s.pt",
        "yolo26m.pt",
        "yolo26l.pt",
        "yolo26x.pt",
    )
    official_model_base_url: str = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
    camera_scan_start: int = 0
    camera_scan_stop: int = 10
    requested_capture_width: int = 640
    requested_capture_height: int = 480
    preview_size: Tuple[int, int] = (1120, 720)
    default_image_size: int = 640
    image_size_options: Tuple[int, ...] = (640, 960, 1280)
    default_confidence: float = 0.25
    default_iou: float = 0.45
    device_options: Tuple[str, ...] = ("auto", "cpu", "cuda", "mps")
    default_recording_fps: float = 30.0
    screenshot_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")
    video_extensions: Tuple[str, ...] = (".mp4", ".avi")

    def model_search_dirs(self) -> Tuple[Path, ...]:
        return (self.model_dir, self.user_model_dir)

