"""Tkinter image presentation adapter for panorama reconstruction windows."""

from __future__ import annotations

from typing import Tuple

from PIL import Image, ImageTk

from ..domain import ImageArray
from ..infrastructure import to_rgb


class TkImagePresenter:
    """Converts BGR numpy images into Tkinter-compatible previews."""

    def __init__(self, preview_size: Tuple[int, int]) -> None:
        self._preview_size = preview_size

    def to_photo_image(self, image: ImageArray) -> ImageTk.PhotoImage:
        pil_image = Image.fromarray(to_rgb(image))
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        pil_image.thumbnail(self._preview_size, resampling)
        return ImageTk.PhotoImage(pil_image)
