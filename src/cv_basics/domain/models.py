"""Domain models shared across layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


ImageArray = np.ndarray
PathLike = Union[str, Path]


class EffectName:
    """Canonical effect names used across layers."""

    GRAYSCALE = "Grayscale"
    BLUR = "Blur"
    EDGES = "Edges"
    THRESHOLD = "Threshold"
    CARTOON = "Cartoon"
    RGB_SPACE = "RGB Space"
    HSV_SPACE = "HSV Space"
    RED_CHANNEL = "Red Channel"
    GREEN_CHANNEL = "Green Channel"
    BLUE_CHANNEL = "Blue Channel"
    HUE_CHANNEL = "Hue Channel"
    SATURATION_CHANNEL = "Saturation Channel"
    VALUE_CHANNEL = "Value Channel"
    GRAY_HISTOGRAM = "Gray Histogram"
    RGB_HISTOGRAM = "RGB Histogram"
    ERODE = "Erode"
    DILATE = "Dilate"
    MORPH_OPEN = "Open Operation"
    MORPH_CLOSE = "Close Operation"
    ROTATE = "Rotate"
    SCALE = "Scale"
    CENTER_CROP = "Center Crop"
    PERSPECTIVE_WARP = "Perspective Warp"


@dataclass(frozen=True)
class ProcessingParams:
    """User-adjustable parameters for image-processing operations."""

    blur_kernel: int = 9
    edge_low: int = 80
    edge_high: int = 160
    threshold: int = 127
    morphology_kernel: int = 5
    morphology_iterations: int = 1
    rotate_angle: int = 30
    scale_percent: int = 120
    crop_percent: int = 70
    perspective_shift: int = 12


@dataclass(frozen=True)
class ImageInfo:
    """Basic image metadata returned by the service layer."""

    width: int
    height: int
    channels: int
    dtype: str
    min_value: float
    max_value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": self.dtype,
            "min": self.min_value,
            "max": self.max_value,
        }
