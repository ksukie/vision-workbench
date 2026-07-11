"""Application service for YOLO26 training workflows."""

from __future__ import annotations

import shutil
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional

from vision_workbench.model_files import validate_complete_model_file
from vision_workbench.runtime_security import confined_child_path, validate_run_name

from ..configuration import Yolo26TrainingConfig
from ..domain import DatasetValidationReport, PathLike, TrainingJobConfig
from ..infrastructure import Yolo26ModelRepository, Yolo26TrainingBackend, YoloDetectionDatasetValidator


class Yolo26TrainingService:
    """Coordinates dataset validation, command building, and training execution."""

    def __init__(
        self,
        validator: YoloDetectionDatasetValidator,
        model_repository: Yolo26ModelRepository,
        backend: Yolo26TrainingBackend,
        config: Yolo26TrainingConfig = Yolo26TrainingConfig(),
    ) -> None:
        self._validator = validator
        self._model_repository = model_repository
        self._backend = backend
        self._config = config

    def validate_dataset(
        self,
        data_path: PathLike,
        task: str = "detect",
        allow_missing_labels: bool = False,
    ) -> DatasetValidationReport:
        return self._validator.validate(
            data_path,
            task=task,
            allow_missing_labels=allow_missing_labels,
        )

    def list_models(self, task: str = "detect") -> List[Path]:
        return self._model_repository.list_models(task)

    def refresh_model_manifest(self) -> int:
        return self._model_repository.refresh_model_manifest()

    def default_model(self, task: str = "detect") -> Path:
        return self._model_repository.default_model(task)

    def train(self, job: TrainingJobConfig) -> object:
        self._validate_output_path(job)
        report = self.validate_dataset(
            job.data_yaml,
            task=job.task,
            allow_missing_labels=job.allow_missing_labels,
        )
        if not report.ok:
            raise ValueError(report.to_text())
        job.project_dir.mkdir(parents=True, exist_ok=True)
        return self._backend.train(job)

    def build_runner_command(self, job: TrainingJobConfig) -> List[str]:
        self._validate_output_path(job)
        command = [
            sys.executable,
            "-m",
            "yolo26_training.runner",
            "--task",
            job.task,
            "--data",
            str(job.data_yaml),
            "--model",
            str(job.model_path),
            "--epochs",
            str(job.epochs),
            "--imgsz",
            str(job.image_size),
            "--batch",
            str(job.batch_size),
            "--device",
            job.device,
            "--workers",
            str(job.workers),
            "--project",
            str(job.project_dir),
            "--name",
            job.run_name,
        ]
        if job.resume:
            command.append("--resume")
        if job.allow_missing_labels:
            command.append("--allow-missing-labels")
        command.extend(job.extra_args)
        return command

    @staticmethod
    def _validate_output_path(job: TrainingJobConfig) -> None:
        run_name = validate_run_name(job.run_name, field_name="YOLO training run name")
        confined_child_path(job.project_dir, run_name, field_name="YOLO training run name")

    def start_training_process(
        self,
        job: TrainingJobConfig,
        cwd: Optional[Path] = None,
    ) -> subprocess.Popen:
        command = self.build_runner_command(job)
        env = os.environ.copy()
        src_dir = Path(__file__).resolve().parents[2]
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(src_dir) if not current_pythonpath else str(src_dir) + os.pathsep + current_pythonpath
        )
        env["PYTHONNOUSERSITE"] = "1"
        env.setdefault("ULTRALYTICS_SAFE_LOAD", "true")
        env.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "1")
        return subprocess.Popen(
            command,
            cwd=str(cwd or self._config.yolo26_source_dir.parents[1]),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def copy_best_weight(
        self,
        run_dir: PathLike,
        target_name: Optional[str] = None,
        task: str = "detect",
    ) -> Path:
        task = _normalize_task(task)
        run_path = Path(run_dir)
        best_path = run_path / "weights" / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"best.pt not found: {best_path}")
        validate_complete_model_file(best_path)

        target_dir = self._config.custom_model_dir_for_task(task)
        target_dir.mkdir(parents=True, exist_ok=True)
        output_name = _registered_weight_name(run_path.name, task, target_name)
        output_path = _unique_path(target_dir / output_name)
        shutil.copy2(best_path, output_path)
        return output_path


def _normalize_task(task: str) -> str:
    value = str(task or "detect").strip().lower()
    if value in ("seg", "instance", "instance_segmentation"):
        return "segment"
    if value in ("sem", "semantic_segmentation"):
        return "semantic"
    if value not in ("detect", "segment", "semantic"):
        return "detect"
    return value


def _registered_weight_name(run_name: str, task: str, target_name: Optional[str]) -> str:
    if target_name:
        name = Path(target_name).name
        return name if name.lower().endswith(".pt") else f"{name}.pt"
    suffix = {
        "detect": "det",
        "segment": "seg",
        "semantic": "sem",
    }[task]
    stem = Path(run_name).stem or "best"
    return f"{stem}-{suffix}.pt"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Too many existing registered weights for {path.name}")


def build_default_service(
    config: Yolo26TrainingConfig = Yolo26TrainingConfig(),
) -> Yolo26TrainingService:
    return Yolo26TrainingService(
        validator=YoloDetectionDatasetValidator(config),
        model_repository=Yolo26ModelRepository(config),
        backend=Yolo26TrainingBackend(config),
        config=config,
    )
