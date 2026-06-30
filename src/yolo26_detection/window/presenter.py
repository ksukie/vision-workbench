"""Tkinter image presentation adapter for YOLO26 frames."""

from __future__ import annotations

from typing import Tuple

from PIL import Image, ImageTk

from ..domain import ImageArray
from ..infrastructure import bgr_to_rgb


class TkDetectionPresenter:
    """Converts BGR frames into Tkinter-compatible previews."""

    def __init__(self, preview_size: Tuple[int, int]) -> None:
        self._preview_size = preview_size

    def to_photo_image(self, frame: ImageArray) -> ImageTk.PhotoImage:
        pil_image = Image.fromarray(bgr_to_rgb(frame))
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        pil_image.thumbnail(self._preview_size, resampling)
        return ImageTk.PhotoImage(pil_image)

