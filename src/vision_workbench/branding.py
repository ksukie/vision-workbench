"""Shared Vision Workbench brand assets."""

from __future__ import annotations

from pathlib import Path


APPLICATION_ICON_FILENAME = "vision_workbench_icon.png"


def application_icon_path() -> Path:
    """Return the bundled application icon path."""

    path = Path(__file__).with_name("assets") / APPLICATION_ICON_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Bundled application icon is missing: {path}")
    return path


def application_icon():
    """Load the shared Qt application icon."""

    from PySide6.QtGui import QIcon

    icon = QIcon(str(application_icon_path()))
    if icon.isNull():
        raise RuntimeError("Bundled application icon could not be decoded.")
    return icon
