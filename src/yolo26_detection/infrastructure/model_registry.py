"""Model discovery and download for YOLO26 detection weights."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Union

from vision_workbench.model_files import download_model_file, is_complete_model_file, validate_complete_model_file
from ..configuration import Yolo26DetectionConfig
from ..domain import ModelInfo


class Yolo26ModelRegistry:
    """Finds official and user-provided YOLO26 model files."""

    def __init__(self, config: Yolo26DetectionConfig = Yolo26DetectionConfig()) -> None:
        self._config = config

    def list_models(self, include_missing_official: bool = True) -> List[ModelInfo]:
        models = []
        by_path = set()
        for name in self._config.official_model_names:
            path = self._config.model_dir / name
            exists = is_complete_model_file(path)
            if include_missing_official or exists:
                models.append(ModelInfo(name=name, path=path, exists=exists, is_official=True))
                by_path.add(path.resolve())

        custom = []
        for directory in self._config.model_search_dirs():
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.pt")):
                resolved = path.resolve()
                if resolved in by_path:
                    continue
                if not is_complete_model_file(path):
                    continue
                custom.append(ModelInfo(name=path.name, path=path, exists=True, is_official=False))
                by_path.add(resolved)
        return models + custom

    def official_model_urls(self) -> Dict[str, str]:
        base = self._config.official_model_base_url.rstrip("/")
        return {name: f"{base}/{name}" for name in self._config.official_model_names}

    def download_official_model(
        self,
        name: str,
        progress_callback: Callable[[int | None, int, int | None], None] | None = None,
    ) -> ModelInfo:
        if name not in self._config.official_model_names:
            raise ValueError(f"{name} is not a configured official YOLO26 detection model.")
        self._config.model_dir.mkdir(parents=True, exist_ok=True)
        url = self.official_model_urls()[name]
        path = self._config.model_dir / name
        _download_to_path(url, path, progress_callback)
        return ModelInfo(name=name, path=path, exists=True, is_official=True)

    def add_custom_model(self, path: Union[Path, str]) -> ModelInfo:
        model_path = Path(path)
        if model_path.suffix.lower() != ".pt":
            raise ValueError("YOLO26 model path must be a .pt file.")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        validate_complete_model_file(model_path)
        return ModelInfo(name=model_path.name, path=model_path, exists=True, is_official=False)


def _download_to_path(
    url: str,
    path: Path,
    progress_callback: Callable[[int | None, int, int | None], None] | None,
) -> None:
    download_model_file(url, path, progress_callback)
