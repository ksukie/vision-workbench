"""Runtime switches for distribution-specific desktop behavior."""

from __future__ import annotations

import os


BASE_EXE_ENV = "VISION_WORKBENCH_BASE_EXE"


def is_base_exe() -> bool:
    """Return whether the lightweight Windows executable is running."""

    return os.environ.get(BASE_EXE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
