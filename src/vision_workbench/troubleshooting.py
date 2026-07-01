"""Troubleshooting document hints for user-visible errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


ENVIRONMENT = "environment"
DEEP_LEARNING_DEPENDENCIES = "deep-learning-dependencies"
MODELS_AND_WEIGHTS = "models-and-weights"
DATA_AND_FILES = "data-and-files"
CAMERA_AND_VIDEO = "camera-and-video"
DATASETS_AND_TRAINING = "datasets-and-training"
MODULE_RUNTIME_ERRORS = "module-runtime-errors"
PACKAGING_AND_RELEASE = "packaging-and-release"


@dataclass(frozen=True)
class TroubleshootingDocs:
    """Relative paths to English and Chinese troubleshooting documents."""

    english: str
    chinese: str


_DOCS = {
    ENVIRONMENT: TroubleshootingDocs(
        "docs/troubleshooting/en/environment.md",
        "docs/troubleshooting/zh-CN/environment.zh-CN.md",
    ),
    DEEP_LEARNING_DEPENDENCIES: TroubleshootingDocs(
        "docs/troubleshooting/en/deep-learning-dependencies.md",
        "docs/troubleshooting/zh-CN/deep-learning-dependencies.zh-CN.md",
    ),
    MODELS_AND_WEIGHTS: TroubleshootingDocs(
        "docs/troubleshooting/en/models-and-weights.md",
        "docs/troubleshooting/zh-CN/models-and-weights.zh-CN.md",
    ),
    DATA_AND_FILES: TroubleshootingDocs(
        "docs/troubleshooting/en/data-and-files.md",
        "docs/troubleshooting/zh-CN/data-and-files.zh-CN.md",
    ),
    CAMERA_AND_VIDEO: TroubleshootingDocs(
        "docs/troubleshooting/en/camera-and-video.md",
        "docs/troubleshooting/zh-CN/camera-and-video.zh-CN.md",
    ),
    DATASETS_AND_TRAINING: TroubleshootingDocs(
        "docs/troubleshooting/en/datasets-and-training.md",
        "docs/troubleshooting/zh-CN/datasets-and-training.zh-CN.md",
    ),
    MODULE_RUNTIME_ERRORS: TroubleshootingDocs(
        "docs/troubleshooting/en/module-runtime-errors.md",
        "docs/troubleshooting/zh-CN/module-runtime-errors.zh-CN.md",
    ),
    PACKAGING_AND_RELEASE: TroubleshootingDocs(
        "docs/troubleshooting/en/packaging-and-release.md",
        "docs/troubleshooting/zh-CN/packaging-and-release.zh-CN.md",
    ),
}


def doc_paths(category: str) -> TroubleshootingDocs:
    """Return troubleshooting document paths for a category."""

    return _DOCS.get(category, _DOCS[MODULE_RUNTIME_ERRORS])


def all_doc_paths() -> Dict[str, TroubleshootingDocs]:
    """Return all known troubleshooting categories and document paths."""

    return dict(_DOCS)


def help_hint(category: str) -> str:
    """Return a short user-facing troubleshooting hint."""

    docs = doc_paths(category)
    return f"Troubleshooting: {docs.english}\nChinese guide: {docs.chinese}"


def with_help(message: object, category: str) -> str:
    """Append a troubleshooting hint to a message."""

    text = str(message).rstrip()
    hint = help_hint(category)
    if not text:
        return hint
    if docs_already_included(text):
        return text
    return f"{text}\n\n{hint}"


def docs_already_included(message: str) -> bool:
    """Return True when a message already includes a troubleshooting hint."""

    return "docs/troubleshooting/" in message
