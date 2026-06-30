"""Configuration objects and optional JSON loading."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..domain import EffectName, ProcessingParams


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration shared by service and GUI layers."""

    default_effect: str = EffectName.GRAYSCALE
    preview_size: Tuple[int, int] = (470, 430)
    processing_defaults: ProcessingParams = ProcessingParams()
    supported_extensions: Tuple[str, ...] = (
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.bmp",
        "*.tif",
        "*.tiff",
    )


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load optional JSON config and merge it with safe defaults."""

    config = AppConfig()
    if not path:
        return config

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object.")

    return _merge_config(config, raw)


def _merge_config(config: AppConfig, raw: Dict[str, Any]) -> AppConfig:
    params = config.processing_defaults
    raw_params = raw.get("processing_defaults", {})
    if raw_params:
        if not isinstance(raw_params, dict):
            raise ValueError("processing_defaults must be a JSON object.")
        params = replace(
            params,
            blur_kernel=int(raw_params.get("blur_kernel", params.blur_kernel)),
            edge_low=int(raw_params.get("edge_low", params.edge_low)),
            edge_high=int(raw_params.get("edge_high", params.edge_high)),
            threshold=int(raw_params.get("threshold", params.threshold)),
            morphology_kernel=int(
                raw_params.get("morphology_kernel", params.morphology_kernel)
            ),
            morphology_iterations=int(
                raw_params.get(
                    "morphology_iterations",
                    params.morphology_iterations,
                )
            ),
            rotate_angle=int(raw_params.get("rotate_angle", params.rotate_angle)),
            scale_percent=int(raw_params.get("scale_percent", params.scale_percent)),
            crop_percent=int(raw_params.get("crop_percent", params.crop_percent)),
            perspective_shift=int(
                raw_params.get("perspective_shift", params.perspective_shift)
            ),
        )

    preview_size = raw.get("preview_size", config.preview_size)
    if not _is_pair(preview_size):
        raise ValueError("preview_size must contain exactly two integers.")

    extensions = raw.get("supported_extensions", config.supported_extensions)
    if isinstance(extensions, str) or not isinstance(extensions, Iterable):
        raise ValueError("supported_extensions must be a list of patterns.")

    return replace(
        config,
        default_effect=str(raw.get("default_effect", config.default_effect)),
        preview_size=(int(preview_size[0]), int(preview_size[1])),
        processing_defaults=params,
        supported_extensions=tuple(str(item) for item in extensions),
    )


def _is_pair(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2
