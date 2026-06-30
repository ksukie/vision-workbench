"""OpenCV image persistence for panorama reconstruction workflows."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..domain import ImageArray, PathLike
from .image_utils import ensure_color_bgr, ensure_uint8_image


class OpenCvImageRepository:
    """Load and save images, including Windows paths with non-ASCII text."""

    def load(self, path: PathLike) -> ImageArray:
        image_path = Path(path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        data = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot decode image file: {image_path}")

        return ensure_color_bgr(image)

    def save(self, image: ImageArray, path: PathLike) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        extension = output_path.suffix or ".png"
        ok, encoded = cv2.imencode(extension, ensure_uint8_image(image))
        if not ok:
            raise ValueError(f"Cannot encode image as {extension}")

        encoded.tofile(str(output_path))
