"""Runtime configuration for YOLO26 segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_source_dir() -> Path:
    return project_root() / "third_party" / "yolo26_source"


def default_model_dir() -> Path:
    return project_root() / "models" / "yolo26_segmentation_models"


def default_custom_model_dir() -> Path:
    return default_model_dir() / "custom"


@dataclass(frozen=True)
class Yolo26SegmentationConfig:
    """Paths and defaults for segmentation inference."""

    yolo26_source_dir: Path = field(default_factory=default_source_dir)
    model_dir: Path = field(default_factory=default_model_dir)
    custom_model_dir: Path = field(default_factory=default_custom_model_dir)
    official_segment_model_names: Tuple[str, ...] = (
        "yolo26n-seg.pt",
        "yolo26s-seg.pt",
        "yolo26m-seg.pt",
        "yolo26l-seg.pt",
        "yolo26x-seg.pt",
    )
    official_semantic_model_names: Tuple[str, ...] = (
        "yolo26n-sem.pt",
        "yolo26s-sem.pt",
        "yolo26m-sem.pt",
        "yolo26l-sem.pt",
        "yolo26x-sem.pt",
    )
    official_model_base_url: str = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
    task_options: Tuple[str, ...] = ("segment", "semantic")
    device_options: Tuple[str, ...] = ("auto", "cpu", "cuda", "mps")
    image_size_options: Tuple[int, ...] = (640, 960, 1280, 2048)
    default_image_size: int = 640
    default_confidence: float = 0.25
    default_iou: float = 0.45
    preview_size: Tuple[int, int] = (1120, 720)
    default_recording_fps: float = 30.0

    def model_names_for_task(self, task: str) -> Tuple[str, ...]:
        return self.official_semantic_model_names if task == "semantic" else self.official_segment_model_names

