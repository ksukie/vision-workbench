"""YOLO26 training backend adapter."""

from __future__ import annotations

import sys
from pathlib import Path

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
        return model.train(
            data=str(job.data_yaml),
            epochs=int(job.epochs),
            imgsz=int(job.image_size),
            batch=int(job.batch_size),
            device=device,
            workers=int(job.workers),
            project=str(job.project_dir),
            name=str(job.run_name),
            resume=bool(job.resume),
        )

    def _import_yolo(self):
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

