from pathlib import Path

import numpy as np

from camera_diagnostics import api
from camera_diagnostics.domain import CaptureProfile


def test_detect_platform_has_backend_route() -> None:
    platform_info = api.detect_platform()

    assert platform_info.system
    assert platform_info.backends
    assert platform_info.backends[0].name


def test_capture_profile_labels() -> None:
    default = CaptureProfile(0, 0, 0.0, "DEFAULT", "DSHOW", is_default=True)
    profile = CaptureProfile(1280, 720, 30.0, "MJPG", "DSHOW")

    assert default.label() == "Default"
    assert "1280x720" in profile.label()
    assert "30fps" in profile.label()
    assert "MJPG" in profile.label()


def test_service_saves_screenshot(tmp_path: Path) -> None:
    service = api.create_camera_diagnostics_service()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    frame[8:24, 12:36] = [10, 120, 240]
    output = tmp_path / "frame.png"

    service.save_screenshot(frame, output)

    assert output.exists()
    assert output.stat().st_size > 0
