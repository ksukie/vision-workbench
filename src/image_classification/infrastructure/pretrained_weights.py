"""Pretrained weight file management for classification models."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from ..configuration import ImageClassificationConfig
from ..domain import ClassificationModelName, PathLike, PretrainedWeightInfo


DEFAULT_WEIGHT_FILENAMES = {
    ClassificationModelName.RESNET18: "resnet18-f37072fd.pth",
    ClassificationModelName.MOBILENET_V3_SMALL: "mobilenet_v3_small-047dcff4.pth",
}


class PretrainedWeightManager:
    """Checks, downloads, and imports TorchVision pretrained weights."""

    def __init__(self, config: ImageClassificationConfig) -> None:
        self._config = config

    def status(self, model_name: str) -> PretrainedWeightInfo:
        normalized = _normalize_model_name(model_name)
        filename = DEFAULT_WEIGHT_FILENAMES[normalized]
        local_path = self._config.pretrained_model_dir / filename
        return PretrainedWeightInfo(
            model_name=normalized,
            filename=filename,
            local_path=local_path,
            exists=local_path.exists(),
            url=_known_weight_url(normalized),
        )

    def status_all(self) -> List[PretrainedWeightInfo]:
        return [self.status(model_name) for model_name in self._config.supported_models]

    def import_weight(self, model_name: str, source_path: PathLike) -> PretrainedWeightInfo:
        normalized = _normalize_model_name(model_name)
        source = Path(source_path).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"Local weight file does not exist: {source}")
        if not source.is_file():
            raise ValueError(f"Local weight path is not a file: {source}")

        self._config.pretrained_model_dir.mkdir(parents=True, exist_ok=True)
        target = self.status(normalized).local_path
        if source.resolve() != target.resolve():
            shutil.copy2(str(source), str(target))
        return self.status(normalized)

    def download_weight(self, model_name: str) -> PretrainedWeightInfo:
        normalized = _normalize_model_name(model_name)
        self._config.pretrained_model_dir.mkdir(parents=True, exist_ok=True)
        filename = DEFAULT_WEIGHT_FILENAMES[normalized]
        weights = _torchvision_weights(normalized)
        url = str(weights.url)

        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "Downloading pretrained weights needs torch and torchvision. "
                "Please install them with: pip install -r requirements-classification.txt"
            ) from exc

        try:
            torch.hub.load_state_dict_from_url(
                url,
                model_dir=str(self._config.pretrained_model_dir),
                file_name=filename,
                progress=True,
            )
        except TypeError:
            torch.hub.load_state_dict_from_url(
                url,
                model_dir=str(self._config.pretrained_model_dir),
                progress=True,
            )
        return self.status(normalized)

    def local_weight_path(self, model_name: str) -> Optional[Path]:
        info = self.status(model_name)
        return info.local_path if info.exists else None


def _normalize_model_name(model_name: str) -> str:
    value = (model_name or "").strip().lower().replace("-", "_")
    if value in ("mobilenetv3small", "mobilenet_v3_small"):
        return ClassificationModelName.MOBILENET_V3_SMALL
    if value == "resnet18":
        return ClassificationModelName.RESNET18
    raise ValueError(f"Unsupported classification model: {model_name}")


def _known_weight_url(model_name: str) -> str:
    filename = DEFAULT_WEIGHT_FILENAMES[_normalize_model_name(model_name)]
    return f"https://download.pytorch.org/models/{filename}"


def _torchvision_weights(model_name: str) -> object:
    try:
        from torchvision import models
    except Exception as exc:
        raise RuntimeError(
            "TorchVision pretrained weights need torchvision. "
            "Please install it with: pip install -r requirements-classification.txt"
        ) from exc

    normalized = _normalize_model_name(model_name)
    if normalized == ClassificationModelName.RESNET18:
        return models.ResNet18_Weights.DEFAULT
    if normalized == ClassificationModelName.MOBILENET_V3_SMALL:
        return models.MobileNet_V3_Small_Weights.DEFAULT
    raise ValueError(f"Unsupported classification model: {model_name}")
