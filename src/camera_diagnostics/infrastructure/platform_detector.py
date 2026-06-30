"""Platform-aware OpenCV camera backend selection."""

from __future__ import annotations

import platform

import cv2

from ..domain import CameraBackend, PlatformInfo


def detect_platform_info() -> PlatformInfo:
    system = platform.system() or "Unknown"
    if system == "Windows":
        backends = (
            CameraBackend("DSHOW", cv2.CAP_DSHOW),
            CameraBackend("MSMF", cv2.CAP_MSMF),
            CameraBackend("ANY", cv2.CAP_ANY),
        )
    elif system == "Linux":
        backends = (
            CameraBackend("V4L2", cv2.CAP_V4L2),
            CameraBackend("ANY", cv2.CAP_ANY),
        )
    elif system == "Darwin":
        backends = (
            CameraBackend("AVFOUNDATION", cv2.CAP_AVFOUNDATION),
            CameraBackend("ANY", cv2.CAP_ANY),
        )
    else:
        backends = (CameraBackend("ANY", cv2.CAP_ANY),)

    return PlatformInfo(system=system, backends=backends)
