"""Image conversion helpers for the panorama reconstruction package."""

from __future__ import annotations

import cv2
import numpy as np

from ..domain import ImageArray


def ensure_uint8_image(image: ImageArray) -> ImageArray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    return np.clip(array, 0, 255).astype(np.uint8)


def ensure_color_bgr(image: ImageArray) -> ImageArray:
    image = ensure_uint8_image(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError("Image must be grayscale, BGR, or BGRA.")


def to_rgb(image: ImageArray) -> ImageArray:
    image = ensure_uint8_image(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raise ValueError("Image must be grayscale, BGR, or BGRA.")
