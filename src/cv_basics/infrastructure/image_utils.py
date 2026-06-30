"""Low-level image normalization and color conversion helpers."""

from __future__ import annotations

import cv2
import numpy as np

from ..domain import ImageArray


def ensure_uint8_image(image: ImageArray) -> ImageArray:
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError("Image must be a 2D grayscale or 3D color array.")
    if image.dtype == np.uint8:
        return image.copy()
    return np.clip(image, 0, 255).astype(np.uint8)


def ensure_color_bgr(image: ImageArray) -> ImageArray:
    image = ensure_uint8_image(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError("Color image must have 3 or 4 channels.")


def normalize_loaded_image(image: ImageArray) -> ImageArray:
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim in (2, 3):
        return ensure_uint8_image(image)
    raise ValueError("Loaded file is not a supported image.")


def to_rgb(image: ImageArray) -> ImageArray:
    image = ensure_uint8_image(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(ensure_color_bgr(image), cv2.COLOR_BGR2RGB)


def normalize_odd_kernel(ksize: int) -> int:
    kernel = max(1, int(ksize))
    if kernel % 2 == 0:
        kernel += 1
    return kernel
