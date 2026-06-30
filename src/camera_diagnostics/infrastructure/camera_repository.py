"""OpenCV-backed camera discovery and frame persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from ..configuration import CameraDiagnosticsConfig
from ..domain import CameraBackend, CameraDevice, CaptureProfile, ImageArray, PathLike
from .image_utils import ensure_uint8_image
from .platform_detector import detect_platform_info


class OpenCvCameraRepository:
    """OpenCV adapter with platform-specific backend fallbacks."""

    def __init__(self, config: CameraDiagnosticsConfig = CameraDiagnosticsConfig()) -> None:
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
                    devices.append(
                        CameraDevice(
                            index=index,
                            backend=backend,
                            name=f"Camera {index}",
                        )
                    )
                    seen.add(key)
                finally:
                    capture.release()
        return devices

    def probe_profiles(self, device: CameraDevice) -> List[CaptureProfile]:
        profiles = [
            CaptureProfile(
                width=0,
                height=0,
                fps=0.0,
                fourcc="DEFAULT",
                backend_name=device.backend.name,
                is_default=True,
            )
        ]
        seen = {profiles[0].label()}
        for width, height in self._config.probe_resolutions:
            for fps in self._config.probe_fps_values:
                for fourcc in self._config.probe_fourcc_values:
                    profile = CaptureProfile(width=width, height=height, fps=float(fps), fourcc=fourcc)
                    detected = self._try_profile(device, profile)
                    if detected is None:
                        continue
                    key = (detected.width, detected.height, int(round(detected.fps)), detected.fourcc)
                    if key in seen:
                        continue
                    seen.add(key)
                    profiles.append(detected)
        return profiles

    def open_camera(
        self,
        device: CameraDevice,
        profile: Optional[CaptureProfile] = None,
    ) -> cv2.VideoCapture:
        capture = self._open_capture(device.index, device.backend)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open {device.label()}.")

        if profile and not profile.is_default:
            self._apply_profile(capture, profile)

        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(
                f"{device.label()} opened, but no frame could be read. "
                "Try another backend or capture profile."
            )
        return capture

    def read_frame(self, capture: cv2.VideoCapture) -> ImageArray:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("Camera frame read failed.")
        return frame

    def describe_capture(self, capture: cv2.VideoCapture) -> CaptureProfile:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fourcc = _decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))
        return CaptureProfile(width=width, height=height, fps=fps, fourcc=fourcc)

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

    def _try_profile(
        self,
        device: CameraDevice,
        profile: CaptureProfile,
    ) -> Optional[CaptureProfile]:
        capture = self._open_capture(device.index, device.backend)
        try:
            if not capture.isOpened():
                return None
            self._apply_profile(capture, profile)
            ok, frame = capture.read()
            if not ok or frame is None:
                return None

            detected = self.describe_capture(capture)
            if detected.width <= 0 or detected.height <= 0:
                detected = CaptureProfile(
                    width=int(frame.shape[1]),
                    height=int(frame.shape[0]),
                    fps=profile.fps,
                    fourcc=profile.fourcc,
                    backend_name=device.backend.name,
                )
            if not _close_enough(detected.width, profile.width) or not _close_enough(detected.height, profile.height):
                return None
            fps = detected.fps if detected.fps > 0 else profile.fps
            fourcc = detected.fourcc if detected.fourcc else profile.fourcc
            return CaptureProfile(
                width=detected.width,
                height=detected.height,
                fps=fps,
                fourcc=fourcc,
                backend_name=device.backend.name,
            )
        finally:
            capture.release()

    def _apply_profile(self, capture: cv2.VideoCapture, profile: CaptureProfile) -> None:
        if profile.fourcc and profile.fourcc != "DEFAULT":
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*profile.fourcc[:4]))
        if profile.width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(profile.width))
        if profile.height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(profile.height))
        if profile.fps > 0:
            capture.set(cv2.CAP_PROP_FPS, float(profile.fps))


def _close_enough(actual: int, requested: int) -> bool:
    return abs(int(actual) - int(requested)) <= 16


def _decode_fourcc(raw_value: float) -> str:
    value = int(raw_value or 0)
    if value <= 0:
        return ""
    chars = [chr((value >> 8 * index) & 0xFF) for index in range(4)]
    decoded = "".join(chars).strip()
    return decoded if decoded.isprintable() else ""


def _video_fourcc_for_suffix(suffix: str) -> int:
    normalized = suffix.lower()
    if normalized == ".mp4":
        return cv2.VideoWriter_fourcc(*"mp4v")
    if normalized == ".avi":
        return cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter_fourcc(*"MJPG")
