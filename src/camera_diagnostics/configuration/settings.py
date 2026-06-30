"""Runtime configuration for camera diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CameraDiagnosticsConfig:
    """Camera scan and recording defaults."""

    camera_scan_start: int = 0
    camera_scan_stop: int = 10
    preview_size: Tuple[int, int] = (960, 540)
    probe_resolutions: Tuple[Tuple[int, int], ...] = (
        (320, 240),
        (640, 480),
        (800, 600),
        (1280, 720),
        (1920, 1080),
    )
    probe_fps_values: Tuple[int, ...] = (15, 30, 60)
    probe_fourcc_values: Tuple[str, ...] = ("MJPG", "YUY2", "NV12", "H264")
    default_recording_fps: float = 30.0
    screenshot_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")
    video_extensions: Tuple[str, ...] = (".mp4", ".avi")
