"""Application service for image classification."""

from __future__ import annotations

import gc
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from vision_workbench.runtime_security import confined_child_path, validate_run_name

from ..configuration import ImageClassificationConfig
from ..domain import (
    ClassificationModelInfo,
    ClassificationModelName,
    ClassificationTrainingConfig,
    DatasetValidationReport,
    PathLike,
    PretrainedWeightInfo,
    PredictionResult,
)
from ..infrastructure import (
    ClassificationDatasetSplitter,
    ClassificationDatasetValidator,
    ClassificationModelRepository,
    PretrainedWeightManager,
)
from ..processing import LoadedClassifier, TorchVisionClassifierBackend


class ImageClassificationService:
    """Coordinates dataset, model, training, and prediction workflows."""

    def __init__(
        self,
        config: ImageClassificationConfig,
        validator: ClassificationDatasetValidator,
        splitter: ClassificationDatasetSplitter,
        model_repository: ClassificationModelRepository,
        weight_manager: PretrainedWeightManager,
        backend: TorchVisionClassifierBackend,
    ) -> None:
        self._config = config
        self._validator = validator
        self._splitter = splitter
        self._model_repository = model_repository
        self._weight_manager = weight_manager
        self._backend = backend
        self._pretrained_cache = {}  # type: Dict[Tuple[str, str, str, int], LoadedClassifier]
        self._checkpoint_cache = {}  # type: Dict[Tuple[str, str, int], LoadedClassifier]

    def supported_models(self) -> List[str]:
        return list(self._config.supported_models)

    def validate_dataset(
        self,
        dataset_dir: PathLike,
        check_images: bool = True,
    ) -> DatasetValidationReport:
        return self._validator.validate(dataset_dir, check_images=check_images)

    def split_dataset(
        self,
        input_dir: PathLike,
        output_dir: PathLike,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> Dict[str, Dict[str, int]]:
        return self._splitter.split(
            input_dir=input_dir,
            output_dir=output_dir,
            train_ratio=train_ratio,
            seed=seed,
        )

    def list_models(self) -> List[ClassificationModelInfo]:
        return self._model_repository.list_pretrained_models() + self._model_repository.list_saved_models()

    def list_saved_models(self) -> List[ClassificationModelInfo]:
        return self._model_repository.list_saved_models()

    def pretrained_weight_status(self, model_name: Optional[str] = None) -> List[PretrainedWeightInfo]:
        if model_name:
            return [self._weight_manager.status(model_name)]
        return self._weight_manager.status_all()

    def clear_pretrained_cache(self, model_name: Optional[str] = None) -> None:
        if model_name is None:
            if self._pretrained_cache:
                self._pretrained_cache.clear()
                _release_torch_memory()
            return

        normalized = self._weight_manager.status(model_name).model_name
        before_count = len(self._pretrained_cache)
        self._pretrained_cache = {
            key: classifier
            for key, classifier in self._pretrained_cache.items()
            if key[0] != normalized
        }
        if len(self._pretrained_cache) != before_count:
            _release_torch_memory()

    def download_pretrained_weight(
        self,
        model_name: str,
        progress_callback: Callable[[int | None, int, int | None], None] | None = None,
    ) -> PretrainedWeightInfo:
        info = self._weight_manager.download_weight(model_name, progress_callback=progress_callback)
        self._invalidate_pretrained_cache(info.model_name)
        return info

    def import_pretrained_weight(
        self,
        model_name: str,
        source_path: PathLike,
    ) -> PretrainedWeightInfo:
        info = self._weight_manager.import_weight(model_name, source_path)
        self._invalidate_pretrained_cache(info.model_name)
        return info

    def train(
        self,
        job: ClassificationTrainingConfig,
        progress_callback: Callable[[dict[str, float | int]], None] | None = None,
    ) -> Path:
        run_name = validate_run_name(job.run_name, field_name="classification training run name")
        confined_child_path(job.output_dir, run_name, field_name="classification training run name")
        if job.model_name not in ClassificationModelName.all():
            raise ValueError(f"Unsupported classification model: {job.model_name}")
        report = self.validate_dataset(job.dataset_dir)
        if not report.ok:
            raise ValueError(report.to_text())
        if job.pretrained and job.pretrained_weight_path is None:
            local_weight = self._weight_manager.local_weight_path(job.model_name)
            if local_weight is not None:
                job = replace(job, pretrained_weight_path=local_weight)
        job.output_dir.mkdir(parents=True, exist_ok=True)
        self._config.custom_model_dir.mkdir(parents=True, exist_ok=True)
        best_path = self._backend.train(job, progress_callback=progress_callback)
        custom_path = confined_child_path(
            self._config.custom_model_dir,
            f"{run_name}_best.pt",
            field_name="classification model file name",
        )
        if best_path != custom_path:
            custom_path.write_bytes(best_path.read_bytes())
        return best_path

    def build_runner_command(self, job: ClassificationTrainingConfig) -> List[str]:
        run_name = validate_run_name(job.run_name, field_name="classification training run name")
        confined_child_path(job.output_dir, run_name, field_name="classification training run name")
        command = [
            sys.executable,
            "-m",
            "image_classification.runner",
            "--data",
            str(job.dataset_dir),
            "--model",
            job.model_name,
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
            "--lr",
            str(job.learning_rate),
            "--project",
            str(job.output_dir),
            "--name",
            run_name,
        ]
        if not job.pretrained:
            command.append("--no-pretrained")
        if not job.freeze_backbone:
            command.append("--unfreeze")
        return command

    def load_pretrained_classifier(
        self,
        model_name: str,
        device: str = "auto",
    ) -> LoadedClassifier:
        status = self._weight_manager.status(model_name)
        local_weight = status.local_path if status.exists else None
        key = _pretrained_cache_key(status.model_name, device, local_weight)
        cached = self._pretrained_cache.get(key)
        if cached is not None:
            return cached
        self._evict_pretrained_cache_except(key)
        classifier = self._backend.load_pretrained(
            model_name=model_name,
            device=device,
            weight_path=local_weight,
        )
        self._pretrained_cache[key] = classifier
        return classifier

    def load_classifier(
        self,
        model_path: PathLike,
        device: str = "auto",
    ) -> LoadedClassifier:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Classification model does not exist: {path}")
        key = _checkpoint_cache_key(path, device)
        cached = self._checkpoint_cache.get(key)
        if cached is not None:
            return cached
        self._evict_checkpoint_cache_except(key)
        classifier = self._backend.load_checkpoint(model_path=path, device=device)
        self._checkpoint_cache[key] = classifier
        return classifier

    def predict_with_pretrained(
        self,
        model_name: str,
        image_path: PathLike,
        topk: Optional[int] = None,
        device: str = "auto",
    ) -> PredictionResult:
        classifier = self.load_pretrained_classifier(model_name=model_name, device=device)
        return self._backend.predict(
            classifier=classifier,
            image_path=Path(image_path).expanduser(),
            topk=topk or self._config.default_topk,
        )

    def _invalidate_pretrained_cache(self, model_name: str) -> None:
        self.clear_pretrained_cache(model_name)

    def _evict_pretrained_cache_except(self, keep_key: Tuple[str, str, str, int]) -> None:
        if any(key != keep_key for key in self._pretrained_cache):
            self._pretrained_cache = {
                key: classifier
                for key, classifier in self._pretrained_cache.items()
                if key == keep_key
            }
            _release_torch_memory()

    def _evict_checkpoint_cache_except(self, keep_key: Tuple[str, str, int]) -> None:
        if any(key != keep_key for key in self._checkpoint_cache):
            self._checkpoint_cache = {
                key: classifier
                for key, classifier in self._checkpoint_cache.items()
                if key == keep_key
            }
            _release_torch_memory()

    def predict_with_checkpoint(
        self,
        model_path: PathLike,
        image_path: PathLike,
        topk: Optional[int] = None,
        device: str = "auto",
    ) -> PredictionResult:
        classifier = self.load_classifier(model_path=model_path, device=device)
        return self._backend.predict(
            classifier=classifier,
            image_path=Path(image_path).expanduser(),
            topk=topk or self._config.default_topk,
        )


def build_default_service(
    config: Optional[ImageClassificationConfig] = None,
) -> ImageClassificationService:
    resolved_config = config or ImageClassificationConfig()
    return ImageClassificationService(
        config=resolved_config,
        validator=ClassificationDatasetValidator(resolved_config),
        splitter=ClassificationDatasetSplitter(resolved_config),
        model_repository=ClassificationModelRepository(resolved_config),
        weight_manager=PretrainedWeightManager(resolved_config),
        backend=TorchVisionClassifierBackend(),
    )


def _pretrained_cache_key(
    model_name: str,
    device: str,
    weight_path: Optional[Path],
) -> Tuple[str, str, str, int]:
    if weight_path is None:
        return (model_name, device, "", 0)
    resolved = weight_path.expanduser().resolve()
    return (model_name, device, str(resolved), resolved.stat().st_mtime_ns)


def _checkpoint_cache_key(model_path: Path, device: str) -> Tuple[str, str, int]:
    resolved = model_path.expanduser().resolve()
    return (str(resolved), device, resolved.stat().st_mtime_ns)


def _release_torch_memory() -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
