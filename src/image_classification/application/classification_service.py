"""Application service for image classification."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

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

    def download_pretrained_weight(self, model_name: str) -> PretrainedWeightInfo:
        return self._weight_manager.download_weight(model_name)

    def import_pretrained_weight(
        self,
        model_name: str,
        source_path: PathLike,
    ) -> PretrainedWeightInfo:
        return self._weight_manager.import_weight(model_name, source_path)

    def train(self, job: ClassificationTrainingConfig) -> Path:
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
        best_path = self._backend.train(job)
        custom_path = self._config.custom_model_dir / f"{job.run_name}_best.pt"
        if best_path != custom_path:
            custom_path.write_bytes(best_path.read_bytes())
        return best_path

    def load_pretrained_classifier(
        self,
        model_name: str,
        device: str = "auto",
    ) -> LoadedClassifier:
        local_weight = self._weight_manager.local_weight_path(model_name)
        return self._backend.load_pretrained(
            model_name=model_name,
            device=device,
            weight_path=local_weight,
        )

    def load_classifier(
        self,
        model_path: PathLike,
        device: str = "auto",
    ) -> LoadedClassifier:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Classification model does not exist: {path}")
        return self._backend.load_checkpoint(model_path=path, device=device)

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
