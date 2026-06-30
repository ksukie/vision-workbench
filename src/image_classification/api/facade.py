"""Public API facade for image classification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..application import ImageClassificationService, build_default_service
from ..configuration import ImageClassificationConfig
from ..domain import (
    ClassificationModelInfo,
    ClassificationTrainingConfig,
    DatasetValidationReport,
    PathLike,
    PretrainedWeightInfo,
    PredictionResult,
)
from ..processing import LoadedClassifier


_default_service = None  # type: Optional[ImageClassificationService]


def create_image_classification_service(
    config: Optional[ImageClassificationConfig] = None,
) -> ImageClassificationService:
    return build_default_service(config)


def get_default_service() -> ImageClassificationService:
    global _default_service
    if _default_service is None:
        _default_service = create_image_classification_service()
    return _default_service


def supported_models() -> List[str]:
    return get_default_service().supported_models()


def validate_classification_dataset(
    dataset_dir: PathLike,
    check_images: bool = True,
) -> DatasetValidationReport:
    return get_default_service().validate_dataset(dataset_dir, check_images=check_images)


def split_classification_dataset(
    input_dir: PathLike,
    output_dir: PathLike,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    return get_default_service().split_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        train_ratio=train_ratio,
        seed=seed,
    )


def list_classification_models() -> List[ClassificationModelInfo]:
    return get_default_service().list_models()


def pretrained_weight_status(
    model_name: Optional[str] = None,
) -> List[PretrainedWeightInfo]:
    return get_default_service().pretrained_weight_status(model_name)


def download_pretrained_weight(model_name: str) -> PretrainedWeightInfo:
    return get_default_service().download_pretrained_weight(model_name)


def import_pretrained_weight(
    model_name: str,
    source_path: PathLike,
) -> PretrainedWeightInfo:
    return get_default_service().import_pretrained_weight(model_name, source_path)


def train_classifier(config: ClassificationTrainingConfig) -> Path:
    return get_default_service().train(config)


def load_pretrained_classifier(
    model_name: str,
    device: str = "auto",
) -> LoadedClassifier:
    return get_default_service().load_pretrained_classifier(
        model_name=model_name,
        device=device,
    )


def load_classifier(
    model_path: PathLike,
    device: str = "auto",
) -> LoadedClassifier:
    return get_default_service().load_classifier(model_path=model_path, device=device)


def predict_with_pretrained(
    model_name: str,
    image_path: PathLike,
    topk: int = 5,
    device: str = "auto",
) -> PredictionResult:
    return get_default_service().predict_with_pretrained(
        model_name=model_name,
        image_path=image_path,
        topk=topk,
        device=device,
    )


def predict_image(
    model_path: PathLike,
    image_path: PathLike,
    topk: int = 5,
    device: str = "auto",
) -> PredictionResult:
    return get_default_service().predict_with_checkpoint(
        model_path=model_path,
        image_path=image_path,
        topk=topk,
        device=device,
    )
