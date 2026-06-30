"""Model discovery and download for YOLO26 segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from urllib.request import urlretrieve

from ..configuration import Yolo26SegmentationConfig
from ..domain import ModelInfo, PathLike


class Yolo26SegmentationModelRegistry:
    """Finds official and user-provided segmentation model files."""

    def __init__(self, config: Yolo26SegmentationConfig = Yolo26SegmentationConfig()) -> None:
        self._config = config

    def list_models(self, task: str = "segment", include_missing_official: bool = True) -> List[ModelInfo]:
        task = _normalize_task(task)
        models = []
        by_path = set()
        for name in self._config.model_names_for_task(task):
            path = self._config.model_dir / name
            exists = path.exists()
            if include_missing_official or exists:
                models.append(ModelInfo(name=name, path=path, task=task, exists=exists, is_official=True))
                by_path.add(path.resolve())
        for directory in (self._config.model_dir, self._config.custom_model_dir):
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.pt")):
                resolved = path.resolve()
                if resolved in by_path:
                    continue
                if task == "segment" and "-sem" in path.name:
                    continue
                if task == "semantic" and "-seg" in path.name:
                    continue
                models.append(ModelInfo(name=path.name, path=path, task=task, exists=True, is_official=False))
                by_path.add(resolved)
        return models

    def official_model_urls(self, task: str = "segment") -> Dict[str, str]:
        task = _normalize_task(task)
        base = self._config.official_model_base_url.rstrip("/")
        return {name: f"{base}/{name}" for name in self._config.model_names_for_task(task)}

    def download_official_model(self, name: str, task: str = "segment") -> ModelInfo:
        task = _normalize_task(task)
        if name not in self._config.model_names_for_task(task):
            raise ValueError(f"{name} is not a configured official YOLO26 {task} model.")
        self._config.model_dir.mkdir(parents=True, exist_ok=True)
        path = self._config.model_dir / name
        urlretrieve(self.official_model_urls(task)[name], path)
        return ModelInfo(name=name, path=path, task=task, exists=True, is_official=True)

    def add_custom_model(self, path: PathLike, task: str = "segment") -> ModelInfo:
        model_path = Path(path)
        if model_path.suffix.lower() != ".pt":
            raise ValueError("YOLO26 model path must be a .pt file.")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return ModelInfo(name=model_path.name, path=model_path, task=_normalize_task(task), exists=True)


def _normalize_task(task: str) -> str:
    value = str(task or "segment").strip().lower()
    if value in ("semantic", "sem", "semantic_segmentation"):
        return "semantic"
    return "segment"
