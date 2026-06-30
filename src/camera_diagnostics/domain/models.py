"""Domain models shared by camera diagnostics layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

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
class CaptureProfile:
    """A concrete camera read mode."""

    width: int
    height: int
    fps: float
    fourcc: str
    backend_name: str = ""
    is_default: bool = False

    def label(self) -> str:
        if self.is_default:
            return "Default"

        fps_text = f"{self.fps:.0f}fps" if self.fps else "fps?"
        fourcc_text = self.fourcc or "format?"
        backend = f" [{self.backend_name}]" if self.backend_name else ""
        return f"{self.width}x{self.height} | {fps_text} | {fourcc_text}{backend}"
