"""Camera diagnostics package for Vision Workbench."""

from vision_workbench.versioning import current_version_info

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

__version__ = current_version_info().version
