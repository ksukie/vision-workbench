"""Public facade for camera diagnostics."""

from __future__ import annotations

from typing import List, Optional

from ..application import CameraDiagnosticsService, build_default_service
from ..configuration import CameraDiagnosticsConfig
from ..domain import CameraDevice, CaptureProfile, PlatformInfo


_default_service = None  # type: Optional[CameraDiagnosticsService]


def create_camera_diagnostics_service(
    config: Optional[CameraDiagnosticsConfig] = None,
) -> CameraDiagnosticsService:
    return build_default_service(config or CameraDiagnosticsConfig())


def get_default_service() -> CameraDiagnosticsService:
    global _default_service
    if _default_service is None:
        _default_service = create_camera_diagnostics_service()
    return _default_service


def detect_platform() -> PlatformInfo:
    return get_default_service().get_platform_info()


def discover_cameras() -> List[CameraDevice]:
    return get_default_service().discover_cameras()


def probe_profiles(device: CameraDevice) -> List[CaptureProfile]:
    return get_default_service().probe_profiles(device)
