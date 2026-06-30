"""Domain models for image classification workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


ImageArray = np.ndarray
PathLike = Union[str, Path]


class ClassificationModelName:
    """Built-in classification model names."""

    RESNET18 = "resnet18"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.RESNET18, cls.MOBILENET_V3_SMALL]


@dataclass(frozen=True)
class ClassCount:
    """Image count for one class."""

    name: str
    train_count: int = 0
    val_count: int = 0


@dataclass(frozen=True)
class DatasetValidationReport:
    """Result returned by classification dataset validation."""

    root: Path
    ok: bool
    messages: List[str] = field(default_factory=list)
    class_counts: List[ClassCount] = field(default_factory=list)
    checked_images: int = 0
    bad_images: List[Path] = field(default_factory=list)

    @property
    def class_names(self) -> List[str]:
        return [item.name for item in self.class_counts]

    @property
    def train_image_count(self) -> int:
        return sum(item.train_count for item in self.class_counts)

    @property
    def val_image_count(self) -> int:
        return sum(item.val_count for item in self.class_counts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": str(self.root),
            "ok": self.ok,
            "messages": list(self.messages),
            "classes": [
                {
                    "name": item.name,
                    "train": item.train_count,
                    "val": item.val_count,
                }
                for item in self.class_counts
            ],
            "checked_images": self.checked_images,
            "bad_images": [str(path) for path in self.bad_images],
        }

    def to_text(self) -> str:
        lines = [
            f"Dataset: {self.root}",
            f"Status: {'OK' if self.ok else 'FAILED'}",
            f"Classes: {len(self.class_counts)}",
            f"Train images: {self.train_image_count}",
            f"Val images: {self.val_image_count}",
            f"Checked images: {self.checked_images}",
        ]
        if self.class_counts:
            lines.append("")
            lines.append("Class counts:")
            for item in self.class_counts:
                lines.append(f"- {item.name}: train={item.train_count}, val={item.val_count}")
        if self.messages:
            lines.append("")
            lines.append("Messages:")
            for message in self.messages:
                lines.append(f"- {message}")
        if self.bad_images:
            lines.append("")
            lines.append("Bad images:")
            for path in self.bad_images:
                lines.append(f"- {path}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ClassificationModelInfo:
    """Model option shown by APIs and GUI."""

    name: str
    path: Optional[Path] = None
    source: str = "pretrained"
    class_count: Optional[int] = None

    def display_name(self) -> str:
        if self.path:
            return f"{self.name} ({self.path.name})"
        return f"{self.name} ({self.source})"


@dataclass(frozen=True)
class PretrainedWeightInfo:
    """Local status for one pretrained weight file."""

    model_name: str
    filename: str
    local_path: Path
    exists: bool
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "filename": self.filename,
            "local_path": str(self.local_path),
            "exists": self.exists,
            "url": self.url,
        }


@dataclass(frozen=True)
class ClassificationTrainingConfig:
    """Basic training options for a classification job."""

    model_name: str
    dataset_dir: Path
    output_dir: Path
    run_name: str = "classification_train"
    epochs: int = 5
    image_size: int = 224
    batch_size: int = 16
    device: str = "auto"
    learning_rate: float = 0.001
    workers: int = 0
    freeze_backbone: bool = True
    pretrained: bool = True
    pretrained_weight_path: Optional[Path] = None


@dataclass(frozen=True)
class PredictionItem:
    """One class prediction."""

    label: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "score": self.score}


@dataclass(frozen=True)
class PredictionResult:
    """Top-k prediction result."""

    image_path: Path
    model_name: str
    model_path: Optional[Path]
    inference_ms: float
    predictions: List[PredictionItem]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "model_name": self.model_name,
            "model_path": str(self.model_path) if self.model_path else None,
            "inference_ms": self.inference_ms,
            "predictions": [item.to_dict() for item in self.predictions],
        }
