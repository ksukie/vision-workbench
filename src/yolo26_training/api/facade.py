"""Public facade for YOLO26 training."""

from __future__ import annotations

from typing import List, Optional

from pathlib import Path

from ..application import Yolo26TrainingService, build_default_service
from ..configuration import Yolo26TrainingConfig
from ..domain import DatasetValidationReport, PathLike


_default_service = None  # type: Optional[Yolo26TrainingService]


def create_yolo26_training_service(
    config: Optional[Yolo26TrainingConfig] = None,
) -> Yolo26TrainingService:
    return build_default_service(config or Yolo26TrainingConfig())


def get_default_service() -> Yolo26TrainingService:
    global _default_service
    if _default_service is None:
        _default_service = create_yolo26_training_service()
    return _default_service


def validate_dataset(
    data_path: PathLike,
    task: str = "detect",
    allow_missing_labels: bool = False,
) -> DatasetValidationReport:
    return get_default_service().validate_dataset(
        data_path,
        task=task,
        allow_missing_labels=allow_missing_labels,
    )


def list_models(task: str = "detect") -> List[Path]:
    return get_default_service().list_models(task)


def refresh_model_manifest() -> int:
    return get_default_service().refresh_model_manifest()


def copy_best_weight(
    run_dir: PathLike,
    target_name: Optional[str] = None,
    task: str = "detect",
) -> Path:
    return get_default_service().copy_best_weight(run_dir, target_name=target_name, task=task)
