"""Application service for YOLO26 camera detection workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2

from ..configuration import Yolo26DetectionConfig
from ..domain import (
    CameraDevice,
    DetectionOutput,
    DetectionSettings,
    ImageArray,
    ModelInfo,
    PathLike,
    PlatformInfo,
)
from ..infrastructure import (
    OpenCvCameraSource,
    UltralyticsYolo26Backend,
    Yolo26ModelRegistry,
    detect_platform_info,
)
from ..processing import Yolo26Detector


class Yolo26DetectionService:
    """Coordinates camera access, YOLO26 inference, screenshots, and recording."""

    def __init__(
        self,
        camera_source: OpenCvCameraSource,
        model_registry: Yolo26ModelRegistry,
        detector: Yolo26Detector,
        config: Yolo26DetectionConfig = Yolo26DetectionConfig(),
    ) -> None:
        self._camera_source = camera_source
        self._model_registry = model_registry
        self._detector = detector
        self._config = config
        self._capture = None  # type: Optional[cv2.VideoCapture]
        self._writer = None  # type: Optional[cv2.VideoWriter]

    def get_platform_info(self) -> PlatformInfo:
        return detect_platform_info()

    def discover_cameras(self) -> List[CameraDevice]:
        return self._camera_source.discover_cameras()

    def list_models(self, include_missing_official: bool = True) -> List[ModelInfo]:
        return self._model_registry.list_models(include_missing_official)

    def add_custom_model(self, path: PathLike) -> ModelInfo:
        return self._model_registry.add_custom_model(path)

    def download_official_model(
        self,
        name: str,
        progress_callback: Callable[[int | None, int, int | None], None] | None = None,
    ) -> ModelInfo:
        return self._model_registry.download_official_model(name, progress_callback=progress_callback)

    def open_camera(
        self,
        device: CameraDevice,
        requested_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.close_camera()
        self._capture = self._camera_source.open_camera(device, requested_size)

    def close_camera(self) -> None:
        self.stop_recording()
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def is_camera_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def read_frame(self) -> ImageArray:
        if self._capture is None:
            raise RuntimeError("No camera is open.")
        return self._camera_source.read_frame(self._capture)

    def load_model(self, model_path: PathLike) -> None:
        self._detector.load_model(model_path)

    def loaded_model_path(self) -> Optional[Path]:
        return self._detector.model_path

    def unload_model(self) -> None:
        self._detector.unload_model()

    def detect_frame(self, frame: ImageArray, settings: DetectionSettings) -> DetectionOutput:
        return self._detector.detect(frame, settings)

    def save_screenshot(self, frame: ImageArray, path: PathLike) -> None:
        self._camera_source.save_frame(frame, path)

    def start_recording(
        self,
        path: PathLike,
        frame_size: Tuple[int, int],
        fps: float,
    ) -> None:
        self.stop_recording()
        self._writer = self._camera_source.create_video_writer(path, frame_size, fps)

    def write_recording_frame(self, frame: ImageArray) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def stop_recording(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def is_recording(self) -> bool:
        return self._writer is not None


def build_default_service(
    config: Yolo26DetectionConfig = Yolo26DetectionConfig(),
) -> Yolo26DetectionService:
    backend = UltralyticsYolo26Backend(config)
    return Yolo26DetectionService(
        camera_source=OpenCvCameraSource(config),
        model_registry=Yolo26ModelRegistry(config),
        detector=Yolo26Detector(backend),
        config=config,
    )
