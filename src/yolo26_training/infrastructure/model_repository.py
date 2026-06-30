"""Model discovery for YOLO26 training."""

from __future__ import annotations

from pathlib import Path
from typing import List

from ..configuration import Yolo26TrainingConfig


class Yolo26ModelRepository:
    """Finds pretrained and custom .pt files."""

    def __init__(self, config: Yolo26TrainingConfig = Yolo26TrainingConfig()) -> None:
        self._config = config

    def list_models(self, task: str = "detect") -> List[Path]:
        task = _normalize_task(task)
        models = []
        model_dir = self._config.model_dir_for_task(task)
        custom_dir = self._config.custom_model_dir_for_task(task)
        for name in self._config.model_names_for_task(task):
            path = model_dir / name
            if path.exists():
                models.append(path.resolve())
        for directory in (model_dir, custom_dir):
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.pt")):
                resolved = path.resolve()
                if resolved not in models:
                    models.append(resolved)
        return models

    def default_model(self, task: str = "detect") -> Path:
        task = _normalize_task(task)
        models = self.list_models(task)
        if models:
            return models[0]
        return self._config.model_dir_for_task(task) / self._config.model_names_for_task(task)[0]


def _normalize_task(task: str) -> str:
    value = str(task or "detect").strip().lower()
    if value in ("seg", "instance", "instance_segmentation"):
        return "segment"
    if value in ("sem", "semantic_segmentation"):
        return "semantic"
    if value not in ("detect", "segment", "semantic"):
        return "detect"
    return value
