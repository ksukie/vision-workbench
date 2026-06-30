"""Application package for image classification."""

from .classification_service import ImageClassificationService, build_default_service

__all__ = [
    "ImageClassificationService",
    "build_default_service",
]
