"""Interface definitions for dependency inversion."""

from __future__ import annotations

from typing import List, Protocol

from ..domain import ImageArray, ImageInfo, PathLike, ProcessingParams


class ImageRepository(Protocol):
    """Persistence boundary for image files."""

    def load(self, path: PathLike) -> ImageArray:
        ...

    def save(self, image: ImageArray, path: PathLike) -> None:
        ...


class ImageOperation(Protocol):
    """Strategy interface for a single image effect."""

    name: str

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        ...


class ImageProcessingServicePort(Protocol):
    """Application-service API consumed by presentation layers."""

    def load_image(self, path: PathLike) -> ImageArray:
        ...

    def save_image(self, image: ImageArray, path: PathLike) -> None:
        ...

    def apply_effect(
        self,
        image: ImageArray,
        effect_name: str,
        params: ProcessingParams = ProcessingParams(),
    ) -> ImageArray:
        ...

    def get_image_info(self, image: ImageArray) -> ImageInfo:
        ...

    def list_effects(self) -> List[str]:
        ...
