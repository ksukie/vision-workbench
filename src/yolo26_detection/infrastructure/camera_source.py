"""OpenCV camera access used by YOLO26 detection."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from ..configuration import Yolo26DetectionConfig
from ..domain import CameraBackend, CameraDevice, ImageArray, PathLike
from .image_utils import ensure_uint8_image
from .platform_detector import detect_platform_info


class OpenCvCameraSource:
    """Camera adapter with platform backend fallbacks."""

    def __init__(self, config: Yolo26DetectionConfig = Yolo26DetectionConfig()) -> None:
        self._config = config

    def discover_cameras(self) -> List[CameraDevice]:
        devices = []
        seen = set()
        platform_info = detect_platform_info()
        for backend in platform_info.backends:
            for index in range(self._config.camera_scan_start, self._config.camera_scan_stop):
                key = (backend.name, index)
                if key in seen:
                    continue
                capture = self._open_capture(index, backend)
                try:
                    if not capture.isOpened():
                        continue
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        continue
                    devices.append(CameraDevice(index=index, backend=backend, name=f"Camera {index}"))
                    seen.add(key)
                finally:
                    capture.release()
        return devices

    def open_camera(
        self,
        device: CameraDevice,
        requested_size: Optional[Tuple[int, int]] = None,
    ) -> cv2.VideoCapture:
        capture = self._open_capture(device.index, device.backend)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open {device.label()}.")

        width, height = requested_size or (
            self._config.requested_capture_width,
            self._config.requested_capture_height,
        )
        if width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(
                f"{device.label()} opened, but no frame could be read. "
                "Try another camera backend or check OS camera permissions."
            )
        return capture

    def read_frame(self, capture: cv2.VideoCapture) -> ImageArray:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("Camera frame read failed.")
        return frame

    def save_frame(self, frame: ImageArray, path: PathLike) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        extension = output_path.suffix or ".png"
        ok, encoded = cv2.imencode(extension, ensure_uint8_image(frame))
        if not ok:
            raise ValueError(f"Cannot encode screenshot as {extension}.")
        encoded.tofile(str(output_path))

    def create_video_writer(
        self,
        path: PathLike,
        frame_size: Tuple[int, int],
        fps: float,
    ) -> cv2.VideoWriter:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = _video_fourcc_for_suffix(output_path.suffix)
        safe_fps = fps if fps and fps > 0 else self._config.default_recording_fps
        writer = cv2.VideoWriter(str(output_path), fourcc, safe_fps, frame_size)
        if not writer.isOpened():
            raise RuntimeError(
                f"Cannot create video writer for {output_path}. "
                "Try .avi if .mp4 is not supported on this system."
            )
        return writer

    def _open_capture(self, index: int, backend: CameraBackend) -> cv2.VideoCapture:
        return cv2.VideoCapture(index, backend.api_id)


def _video_fourcc_for_suffix(suffix: str) -> int:
    normalized = suffix.lower()
    if normalized == ".mp4":
        return cv2.VideoWriter_fourcc(*"mp4v")
    if normalized == ".avi":
        return cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter_fourcc(*"MJPG")

