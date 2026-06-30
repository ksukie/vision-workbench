"""Runtime configuration for image classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from ..domain import ClassificationModelName


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_model_dir() -> Path:
    return project_root() / "models" / "image_classification_models"


def default_custom_model_dir() -> Path:
    return default_model_dir() / "custom"


def default_pretrained_model_dir() -> Path:
    return default_model_dir() / "pretrained"


def default_dataset_dir() -> Path:
    return project_root() / "datasets" / "image_classification_datasets"


def default_runs_dir() -> Path:
    return project_root() / "runs" / "image_classification"


@dataclass(frozen=True)
class ImageClassificationConfig:
    """Paths and defaults for classification workflows."""

    model_dir: Path = field(default_factory=default_model_dir)
    custom_model_dir: Path = field(default_factory=default_custom_model_dir)
    pretrained_model_dir: Path = field(default_factory=default_pretrained_model_dir)
    dataset_dir: Path = field(default_factory=default_dataset_dir)
    runs_dir: Path = field(default_factory=default_runs_dir)
    supported_models: Tuple[str, ...] = (
        ClassificationModelName.RESNET18,
        ClassificationModelName.MOBILENET_V3_SMALL,
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
    default_model_name: str = ClassificationModelName.RESNET18
    default_epochs: int = 5
    default_image_size: int = 224
    default_batch_size: int = 16
    default_learning_rate: float = 0.001
    default_topk: int = 5
    device_options: Tuple[str, ...] = ("auto", "cpu", "cuda", "mps")
    preview_size: Tuple[int, int] = (520, 420)


__all__ = [
    "ImageClassificationConfig",
    "default_custom_model_dir",
    "default_dataset_dir",
    "default_model_dir",
    "default_pretrained_model_dir",
    "default_runs_dir",
    "project_root",
]
