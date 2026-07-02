"""Domain models for YOLO26 segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


ImageArray = np.ndarray
PathLike = Union[str, Path]


@dataclass(frozen=True)
class ModelInfo:
    """A segmentation model file candidate."""

    name: str
    path: Path
    task: str
    exists: bool
    is_official: bool = False

    def label(self) -> str:
        state = "" if self.exists else "（模型未下载）"
        return f"{self.name}{state}"


@dataclass(frozen=True)
class SegmentationSettings:
    """Runtime segmentation settings."""

    task: str = "segment"
    image_size: int = 640
    confidence: float = 0.25
    iou: float = 0.45
    device: str = "auto"

    def normalized_device(self) -> Optional[str]:
        value = self.device.strip().lower()
        return None if value in ("", "auto") else value


@dataclass(frozen=True)
class SegmentationOutput:
    """Annotated segmentation frame."""

    annotated_frame: ImageArray
    item_count: int
    inference_ms: float
    names: Tuple[str, ...] = tuple()
