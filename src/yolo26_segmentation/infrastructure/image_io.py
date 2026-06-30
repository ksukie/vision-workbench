"""Image IO helpers for YOLO26 segmentation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..domain import ImageArray, PathLike


def load_image(path: PathLike) -> ImageArray:
    image_path = Path(path)
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return image


def save_image(image: ImageArray, path: PathLike) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extension = output_path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, ensure_uint8_image(image))
    if not ok:
        raise ValueError(f"Cannot encode image as {extension}.")
    encoded.tofile(str(output_path))


def ensure_uint8_image(image: ImageArray) -> ImageArray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    return np.clip(array, 0, 255).astype(np.uint8)


def bgr_to_rgb(image: ImageArray) -> ImageArray:
    image = ensure_uint8_image(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    raise ValueError("Image must be grayscale, BGR, or BGRA.")

