"""Public API for camera diagnostics."""

from .facade import (
    create_camera_diagnostics_service,
    detect_platform,
    discover_cameras,
    get_default_service,
    probe_profiles,
)

__all__ = [
    "create_camera_diagnostics_service",
    "detect_platform",
    "discover_cameras",
    "get_default_service",
    "probe_profiles",
]
