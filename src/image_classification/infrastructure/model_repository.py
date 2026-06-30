"""Model discovery for classification."""

from __future__ import annotations

from pathlib import Path
from typing import List

from ..configuration import ImageClassificationConfig
from ..domain import ClassificationModelInfo


class ClassificationModelRepository:
    """Lists built-in and locally saved classification models."""

    def __init__(self, config: ImageClassificationConfig) -> None:
        self._config = config

    def list_pretrained_models(self) -> List[ClassificationModelInfo]:
        return [
            ClassificationModelInfo(name=name, source="torchvision")
            for name in self._config.supported_models
        ]

    def list_saved_models(self) -> List[ClassificationModelInfo]:
        models = []  # type: List[ClassificationModelInfo]
        for directory in (self._config.custom_model_dir, self._config.model_dir):
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in (".pt", ".pth"):
                    models.append(
                        ClassificationModelInfo(
                            name=path.stem,
                            path=path,
                            source="local",
                        )
                    )
        return _deduplicate_models(models)


def _deduplicate_models(models: List[ClassificationModelInfo]) -> List[ClassificationModelInfo]:
    seen = set()
    result = []  # type: List[ClassificationModelInfo]
    for model in models:
        key = str(model.path)
        if key in seen:
            continue
        seen.add(key)
        result.append(model)
    return result
