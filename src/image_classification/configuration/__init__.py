"""Configuration package for image classification."""

from .settings import (
    ImageClassificationConfig,
    default_custom_model_dir,
    default_dataset_dir,
    default_model_dir,
    default_pretrained_model_dir,
    default_runs_dir,
    project_root,
)

__all__ = [
    "ImageClassificationConfig",
    "default_custom_model_dir",
    "default_dataset_dir",
    "default_model_dir",
    "default_pretrained_model_dir",
    "default_runs_dir",
    "project_root",
]
