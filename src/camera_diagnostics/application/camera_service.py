"""Application service for camera diagnostics workflows."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2

from ..configuration import CameraDiagnosticsConfig
from ..domain import CameraDevice, CaptureProfile, ImageArray, PathLike, PlatformInfo
from ..infrastructure import OpenCvCameraRepository, detect_platform_info


class CameraDiagnosticsService:
    """Coordinates camera discovery, preview, screenshots, and recording."""

    def __init__(
        self,
        repository: OpenCvCameraRepository,
        config: CameraDiagnosticsConfig = CameraDiagnosticsConfig(),
    ) -> None:
        self._repository = repository
        self._config = config
        self._capture = None  # type: Optional[cv2.VideoCapture]
        self._writer = None  # type: Optional[cv2.VideoWriter]

    def get_platform_info(self) -> PlatformInfo:
        return detect_platform_info()

    def discover_cameras(self) -> List[CameraDevice]:
        return self._repository.discover_cameras()

    def probe_profiles(self, device: CameraDevice) -> List[CaptureProfile]:
        return self._repository.probe_profiles(device)

    def open_camera(
        self,
        device: CameraDevice,
        profile: Optional[CaptureProfile] = None,
    ) -> CaptureProfile:
        self.close_camera()
        self._capture = self._repository.open_camera(device, profile)
        return self._repository.describe_capture(self._capture)

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
        return self._repository.read_frame(self._capture)

    def save_screenshot(self, frame: ImageArray, path: PathLike) -> None:
        self._repository.save_frame(frame, path)

    def start_recording(
        self,
        path: PathLike,
        frame_size: Tuple[int, int],
        fps: float,
    ) -> None:
        self.stop_recording()
        self._writer = self._repository.create_video_writer(path, frame_size, fps)

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
    config: CameraDiagnosticsConfig = CameraDiagnosticsConfig(),
) -> CameraDiagnosticsService:
    return CameraDiagnosticsService(
        repository=OpenCvCameraRepository(config),
        config=config,
    )
