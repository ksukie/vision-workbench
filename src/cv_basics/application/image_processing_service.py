"""Application service that coordinates image workflows."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..configuration import AppConfig
from ..domain import ImageArray, ImageInfo, PathLike, ProcessingParams
from ..infrastructure import OpenCvImageRepository
from ..ports import ImageRepository
from ..processing import OperationRegistry, build_default_registry


class ImageProcessingService:
    """Coordinates repositories, operations, and domain models."""

    def __init__(
        self,
        repository: ImageRepository,
        operations: OperationRegistry,
        config: AppConfig = AppConfig(),
    ) -> None:
        self._repository = repository
        self._operations = operations
        self._config = config

    def load_image(self, path: PathLike) -> ImageArray:
        return self._repository.load(path)

    def save_image(self, image: ImageArray, path: PathLike) -> None:
        self._repository.save(image, path)

    def apply_effect(
        self,
        image: ImageArray,
        effect_name: str,
        params: Optional[ProcessingParams] = None,
    ) -> ImageArray:
        operation = self._operations.get(effect_name)
        return operation.apply(image, params or self._config.processing_defaults)

    def get_image_info(self, image: ImageArray) -> ImageInfo:
        image = np.asarray(image)
        if image.ndim not in (2, 3):
            raise ValueError("Image must be a 2D grayscale or 3D color array.")

        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        return ImageInfo(
            width=int(width),
            height=int(height),
            channels=int(channels),
            dtype=str(image.dtype),
            min_value=float(np.min(image)) if image.size else 0.0,
            max_value=float(np.max(image)) if image.size else 0.0,
        )

    def list_effects(self) -> List[str]:
        return self._operations.names()


def build_default_service(config: AppConfig = AppConfig()) -> ImageProcessingService:
    return ImageProcessingService(
        repository=OpenCvImageRepository(),
        operations=build_default_registry(),
        config=config,
    )
