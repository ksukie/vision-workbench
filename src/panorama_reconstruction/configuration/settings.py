"""Runtime configuration for panorama reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


CHANNEL_CHOICES = ("gray", "r", "g", "b", "h", "s", "v")


@dataclass(frozen=True)
class PanoramaReconstructionConfig:
    """Configuration shared by the reconstruction service and GUI."""

    default_channel: str = "gray"
    preview_size: Tuple[int, int] = (360, 250)
    result_preview_size: Tuple[int, int] = (760, 420)
    supported_extensions: Tuple[str, ...] = (
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.bmp",
        "*.tif",
        "*.tiff",
    )
