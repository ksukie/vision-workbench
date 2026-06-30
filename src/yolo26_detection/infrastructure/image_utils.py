"""Image helpers for camera frames and YOLO results."""

from __future__ import annotations

import cv2
import numpy as np

from ..domain import ImageArray


def ensure_uint8_image(image: ImageArray) -> ImageArray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    return np.clip(array, 0, 255).astype(np.uint8)


def bgr_to_rgb(image: ImageArray) -> ImageArray:
    image = ensure_uint8_image(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raise ValueError("Frame must be grayscale, BGR, or BGRA.")

