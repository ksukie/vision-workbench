"""Qt image presentation adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from vision_workbench.input_limits import validate_image_file
from PySide6.QtGui import QImage, QPixmap


class QtImagePresenter:
    """Convert project image arrays and files into Qt pixmaps."""

    def __init__(self, preview_size: Tuple[int, int]) -> None:
        self._preview_size = preview_size

    def to_pixmap(self, image: np.ndarray, *, resize: bool = True) -> QPixmap:
        preview_image = self._resize_for_preview(image) if resize else np.asarray(image)
        rgb = np.ascontiguousarray(_to_rgb(preview_image))
        height, width, channels = rgb.shape
        if channels != 3:
            raise ValueError("Qt preview images must be RGB.")

        bytes_per_line = channels * width
        qimage = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(qimage)

    def path_to_pixmap(self, path: Path) -> QPixmap:
        path = validate_image_file(path)
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        return self.to_pixmap(rgb[:, :, ::-1])

    def _resize_for_preview(self, image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim not in (2, 3):
            return array

        height, width = array.shape[:2]
        max_width = max(1, int(self._preview_size[0]))
        max_height = max(1, int(self._preview_size[1]))
        if width <= max_width and height <= max_height:
            return array

        scale = min(max_width / width, max_height / height)
        target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(array, target_size, interpolation=cv2.INTER_AREA)


def _ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError("Image must be a 2D grayscale or 3D color array.")
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def _to_rgb(image: np.ndarray) -> np.ndarray:
    image = _ensure_uint8_image(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    raise ValueError("Color image must have 3 or 4 channels.")
