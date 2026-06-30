"""Processing package for image classification."""

from .classifier_backend import LoadedClassifier, TorchVisionClassifierBackend

__all__ = [
    "LoadedClassifier",
    "TorchVisionClassifierBackend",
]
