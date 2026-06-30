"""Camera diagnostics package for Vision Workbench."""

from .api import (
    create_camera_diagnostics_service,
    detect_platform,
    discover_cameras,
    get_default_service,
    probe_profiles,
)
from .domain import CameraBackend, CameraDevice, CaptureProfile, PlatformInfo

__all__ = [
    "CameraBackend",
    "CameraDevice",
    "CaptureProfile",
    "PlatformInfo",
    "create_camera_diagnostics_service",
    "detect_platform",
    "discover_cameras",
    "get_default_service",
    "probe_profiles",
]

__version__ = "0.1.0"
