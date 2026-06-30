"""Runtime configuration for YOLO26 training."""

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


def default_custom_model_dir() -> Path:
    return default_model_dir() / "custom"


def default_segmentation_model_dir() -> Path:
    return project_root() / "models" / "yolo26_segmentation_models"


def default_segmentation_custom_model_dir() -> Path:
    return default_segmentation_model_dir() / "custom"


def default_dataset_dir() -> Path:
    return project_root() / "datasets" / "yolo26_datasets"


def default_runs_dir() -> Path:
    return project_root() / "runs" / "yolo26_training"


@dataclass(frozen=True)
class Yolo26TrainingConfig:
    """Paths and defaults for training."""

    yolo26_source_dir: Path = field(default_factory=default_source_dir)
    model_dir: Path = field(default_factory=default_model_dir)
    custom_model_dir: Path = field(default_factory=default_custom_model_dir)
    segmentation_model_dir: Path = field(default_factory=default_segmentation_model_dir)
    segmentation_custom_model_dir: Path = field(default_factory=default_segmentation_custom_model_dir)
    dataset_dir: Path = field(default_factory=default_dataset_dir)
    runs_dir: Path = field(default_factory=default_runs_dir)
    task_options: Tuple[str, ...] = ("detect", "segment", "semantic")
    official_model_names: Tuple[str, ...] = (
        "yolo26n.pt",
        "yolo26s.pt",
        "yolo26m.pt",
        "yolo26l.pt",
        "yolo26x.pt",
    )
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
    image_extensions: Tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    )
    default_epochs: int = 100
    default_image_size: int = 640
    default_batch_size: int = 16
    default_workers: int = 8
    device_options: Tuple[str, ...] = ("auto", "cpu", "cuda", "mps")

    def model_dir_for_task(self, task: str) -> Path:
        return self.model_dir if task == "detect" else self.segmentation_model_dir

    def custom_model_dir_for_task(self, task: str) -> Path:
        return self.custom_model_dir if task == "detect" else self.segmentation_custom_model_dir

    def model_names_for_task(self, task: str) -> Tuple[str, ...]:
        if task == "segment":
            return self.official_segment_model_names
        if task == "semantic":
            return self.official_semantic_model_names
        return self.official_model_names
