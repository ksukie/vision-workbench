"""YOLO26 detection processor."""

from __future__ import annotations

from ..domain import DetectionOutput, DetectionSettings, ImageArray, PathLike
from ..infrastructure import UltralyticsYolo26Backend


class Yolo26Detector:
    """Thin processing wrapper around the selected detector backend."""

    def __init__(self, backend: UltralyticsYolo26Backend) -> None:
        self._backend = backend

    @property
    def model_path(self):
        return self._backend.model_path

    def load_model(self, model_path: PathLike) -> None:
        self._backend.load_model(model_path)

    def unload_model(self) -> None:
        self._backend.unload_model()

    def detect(self, frame: ImageArray, settings: DetectionSettings) -> DetectionOutput:
        return self._backend.detect(frame, settings)
