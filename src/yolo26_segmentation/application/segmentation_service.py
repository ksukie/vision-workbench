"""Application service for YOLO26 segmentation."""

from __future__ import annotations

from typing import List, Optional

import cv2

from ..configuration import Yolo26SegmentationConfig
from ..domain import ImageArray, ModelInfo, PathLike, SegmentationOutput, SegmentationSettings
from ..infrastructure import (
    UltralyticsYolo26SegmentationBackend,
    Yolo26SegmentationModelRegistry,
    load_image,
    save_image,
)
from ..processing import Yolo26Segmenter


class Yolo26SegmentationService:
    """Coordinates model discovery, image IO, camera frames, and segmentation."""

    def __init__(
        self,
        model_registry: Yolo26SegmentationModelRegistry,
        segmenter: Yolo26Segmenter,
        config: Yolo26SegmentationConfig = Yolo26SegmentationConfig(),
    ) -> None:
        self._model_registry = model_registry
        self._segmenter = segmenter
        self._config = config
        self._capture = None  # type: Optional[cv2.VideoCapture]

    def list_models(self, task: str = "segment", include_missing_official: bool = True) -> List[ModelInfo]:
        return self._model_registry.list_models(task, include_missing_official)

    def add_custom_model(self, path: PathLike, task: str = "segment") -> ModelInfo:
        return self._model_registry.add_custom_model(path, task)

    def download_official_model(self, name: str, task: str = "segment") -> ModelInfo:
        return self._model_registry.download_official_model(name, task)

    def load_image(self, path: PathLike) -> ImageArray:
        return load_image(path)

    def save_image(self, image: ImageArray, path: PathLike) -> None:
        save_image(image, path)

    def load_model(self, model_path: PathLike) -> None:
        self._segmenter.load_model(model_path)

    def segment_image(self, image: ImageArray, settings: SegmentationSettings) -> SegmentationOutput:
        return self._segmenter.segment(image, settings)

    def open_camera(self, index: int = 0) -> None:
        self.close_camera()
        capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open camera {index}.")
        self._capture = capture

    def close_camera(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def is_camera_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def read_frame(self) -> ImageArray:
        if self._capture is None:
            raise RuntimeError("No camera is open.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("Camera frame read failed.")
        return frame


def build_default_service(
    config: Yolo26SegmentationConfig = Yolo26SegmentationConfig(),
) -> Yolo26SegmentationService:
    backend = UltralyticsYolo26SegmentationBackend(config)
    return Yolo26SegmentationService(
        model_registry=Yolo26SegmentationModelRegistry(config),
        segmenter=Yolo26Segmenter(backend),
        config=config,
    )

