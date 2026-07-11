"""YOLO26 training backend adapter."""

from __future__ import annotations

import sys
from pathlib import Path

from vision_workbench.runtime_security import (
    confined_child_path,
    configure_restricted_model_loading,
    validate_run_name,
)

from ..configuration import Yolo26TrainingConfig
from ..domain import TrainingJobConfig


class Yolo26TrainingBackend:
    """Runs Ultralytics training after validation has passed."""

    def __init__(self, config: Yolo26TrainingConfig = Yolo26TrainingConfig()) -> None:
        self._config = config

    def train(self, job: TrainingJobConfig) -> object:
        YOLO = self._import_yolo()
        model = YOLO(str(job.model_path))
        device = None if job.device.strip().lower() == "auto" else job.device
        run_name = validate_run_name(job.run_name, field_name="YOLO training run name")
        run_dir = confined_child_path(job.project_dir, run_name, field_name="YOLO training run name")
        return model.train(
            data=str(job.data_yaml),
            epochs=int(job.epochs),
            imgsz=int(job.image_size),
            batch=int(job.batch_size),
            device=device,
            workers=int(job.workers),
            project=str(run_dir.parent),
            name=run_name,
            resume=bool(job.resume),
        )

    def _import_yolo(self):
        configure_restricted_model_loading()
        source_dir = self._config.yolo26_source_dir
        if source_dir.exists():
            source_text = str(source_dir.resolve())
            if source_text not in sys.path:
                sys.path.insert(0, source_text)
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Cannot import Ultralytics YOLO26 runtime. "
                "Install it from the project root with: pip install -r requirements-yolo26.txt"
            ) from exc
        return YOLO
