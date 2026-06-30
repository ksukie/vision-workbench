"""Platform-specific OpenCV backend selection."""

from __future__ import annotations

import platform

import cv2

from ..domain import CameraBackend, PlatformInfo


def detect_platform_info() -> PlatformInfo:
    system = platform.system() or "Unknown"
    normalized = system.lower()
    if normalized == "windows":
        backends = (
            CameraBackend("DSHOW", cv2.CAP_DSHOW),
            CameraBackend("MSMF", cv2.CAP_MSMF),
            CameraBackend("ANY", cv2.CAP_ANY),
        )
    elif normalized == "linux":
        backends = (
            CameraBackend("V4L2", cv2.CAP_V4L2),
            CameraBackend("ANY", cv2.CAP_ANY),
        )
    elif normalized == "darwin":
        backends = (
            CameraBackend("AVFOUNDATION", cv2.CAP_AVFOUNDATION),
            CameraBackend("ANY", cv2.CAP_ANY),
        )
    else:
        backends = (CameraBackend("ANY", cv2.CAP_ANY),)
    return PlatformInfo(system=system, backends=backends)

