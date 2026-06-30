"""Domain models shared by YOLO26 training layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union


PathLike = Union[str, Path]


@dataclass(frozen=True)
class DatasetValidationSummary:
    """Dataset counts collected during validation."""

    train_images: int = 0
    val_images: int = 0
    train_labels: int = 0
    val_labels: int = 0
    classes: int = 0


@dataclass(frozen=True)
class DatasetValidationReport:
    """Human-readable validation result for one YOLO dataset."""

    task: str
    data_yaml: Optional[Path]
    dataset_root: Optional[Path]
    class_names: Tuple[str, ...] = tuple()
    summary: DatasetValidationSummary = field(default_factory=DatasetValidationSummary)
    errors: Tuple[str, ...] = tuple()
    warnings: Tuple[str, ...] = tuple()

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_text(self) -> str:
        lines = []
        lines.append(f"task: {self.task}")
        lines.append("Dataset validation: PASS" if self.ok else "Dataset validation: FAILED")
        if self.data_yaml:
            lines.append(f"data.yaml: {self.data_yaml}")
        if self.dataset_root:
            lines.append(f"dataset root: {self.dataset_root}")
        lines.append(
            "summary: "
            f"classes={self.summary.classes}, "
            f"train_images={self.summary.train_images}, "
            f"val_images={self.summary.val_images}, "
            f"train_labels={self.summary.train_labels}, "
            f"val_labels={self.summary.val_labels}"
        )
        if self.class_names:
            lines.append(f"classes: {', '.join(self.class_names)}")
        if self.errors:
            lines.append("")
            lines.append("Errors:")
            lines.extend(f"- {item}" for item in self.errors)
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"- {item}" for item in self.warnings)
        return "\n".join(lines)


@dataclass(frozen=True)
class TrainingJobConfig:
    """Training arguments passed to the runner."""

    task: str
    data_yaml: Path
    model_path: Path
    project_dir: Path
    run_name: str
    epochs: int = 100
    image_size: int = 640
    batch_size: int = 16
    device: str = "auto"
    workers: int = 8
    resume: bool = False
    allow_missing_labels: bool = False
    extra_args: List[str] = field(default_factory=list)
