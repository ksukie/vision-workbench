"""Model discovery and download for YOLO26 detection weights."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Union

from vision_workbench.model_manifest import refresh_model_manifest, yolo26_model_entries_for_task
from vision_workbench.model_files import download_model_file, is_complete_model_file, validate_complete_model_file
from vision_workbench.trusted_models import trusted_model_download_hosts
from ..configuration import Yolo26DetectionConfig
from ..domain import ModelInfo


class Yolo26ModelRegistry:
    """Finds official and user-provided YOLO26 model files."""

    def __init__(self, config: Yolo26DetectionConfig = Yolo26DetectionConfig()) -> None:
        self._config = config

    def list_models(self, include_missing_official: bool = True) -> List[ModelInfo]:
        models = []
        by_path = set()
        for entry in self._official_model_entries():
            path = self._config.model_dir / entry.name
            exists = is_complete_model_file(path)
            if include_missing_official or exists:
                models.append(ModelInfo(name=entry.name, path=path, exists=exists, is_official=True))
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
        return {entry.name: entry.url for entry in self._official_model_entries()}

    def refresh_model_manifest(self) -> int:
        entries = refresh_model_manifest(
            self._config.official_model_manifest_url,
            self._config.model_manifest_cache_path,
        )
        return len(entries)

    def download_official_model(
        self,
        name: str,
        progress_callback: Callable[[int | None, int, int | None], None] | None = None,
    ) -> ModelInfo:
        model_entries = {entry.name: entry for entry in self._official_model_entries()}
        if name not in model_entries:
            raise ValueError(f"{name} is not a configured official YOLO26 detection model.")
        self._config.model_dir.mkdir(parents=True, exist_ok=True)
        entry = model_entries[name]
        path = self._config.model_dir / name
        download_model_file(
            entry.url,
            path,
            progress_callback,
            expected_sha256=entry.sha256,
            expected_size=entry.size_bytes,
            allowed_hosts=trusted_model_download_hosts(),
        )
        return ModelInfo(name=name, path=path, exists=True, is_official=True)

    def add_custom_model(self, path: Union[Path, str]) -> ModelInfo:
        model_path = Path(path)
        if model_path.suffix.lower() != ".pt":
            raise ValueError("YOLO26 model path must be a .pt file.")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        validate_complete_model_file(model_path)
        return ModelInfo(name=model_path.name, path=model_path, exists=True, is_official=False)

    def _official_model_entries(self):
        return yolo26_model_entries_for_task(
            "detect",
            fallback_names_by_task={"detect": self._config.official_model_names},
            base_url=self._config.official_model_base_url,
            cache_path=self._config.model_manifest_cache_path,
        )
