"""Tkinter image presentation helpers for classification."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image, ImageTk


class TkClassificationPresenter:
    """Converts image files into Tkinter previews."""

    def __init__(self, preview_size: Tuple[int, int]) -> None:
        self._preview_size = preview_size

    def path_to_photo_image(self, path: Path) -> ImageTk.PhotoImage:
        with Image.open(path) as image:
            image = image.convert("RGB")
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            image.thumbnail(self._preview_size, resampling)
            return ImageTk.PhotoImage(image.copy())
