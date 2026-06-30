"""YOLO26 segmentation processor."""

from __future__ import annotations

from ..domain import ImageArray, PathLike, SegmentationOutput, SegmentationSettings
from ..infrastructure import UltralyticsYolo26SegmentationBackend


class Yolo26Segmenter:
    """Thin processing wrapper around the selected backend."""

    def __init__(self, backend: UltralyticsYolo26SegmentationBackend) -> None:
        self._backend = backend

    @property
    def model_path(self):
        return self._backend.model_path

    def load_model(self, model_path: PathLike) -> None:
        self._backend.load_model(model_path)

    def segment(self, image: ImageArray, settings: SegmentationSettings) -> SegmentationOutput:
        return self._backend.segment(image, settings)

