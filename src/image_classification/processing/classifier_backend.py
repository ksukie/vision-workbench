"""TorchVision-backed image classification training and prediction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from PIL import Image

from ..domain import (
    ClassificationModelName,
    ClassificationTrainingConfig,
    PredictionItem,
    PredictionResult,
)


@dataclass
class LoadedClassifier:
    """In-memory classifier wrapper."""

    model: Any
    model_name: str
    class_names: List[str]
    image_size: int
    device: Any
    model_path: Optional[Path] = None


class TorchVisionClassifierBackend:
    """Training and prediction backend using torch and torchvision."""

    def train(self, config: ClassificationTrainingConfig) -> Path:
        torch, nn, optim, datasets, transforms, models = _require_torchvision()
        device = _normalize_device(torch, config.device)
        train_transform = _build_transform(transforms, config.image_size, is_train=True)
        val_transform = _build_transform(transforms, config.image_size, is_train=False)

        train_dataset = datasets.ImageFolder(str(config.dataset_dir / "train"), transform=train_transform)
        val_dataset = datasets.ImageFolder(str(config.dataset_dir / "val"), transform=val_transform)
        if train_dataset.classes != val_dataset.classes:
            raise ValueError("train/ and val/ class folders must be the same.")

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
        )

        model = _create_model(
            models=models,
            nn=nn,
            model_name=config.model_name,
            num_classes=len(train_dataset.classes),
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone,
            torch=torch,
            pretrained_weight_path=config.pretrained_weight_path,
        )
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=config.learning_rate,
        )

        run_dir = config.output_dir / config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        best_path = run_dir / "best.pt"
        last_path = run_dir / "last.pt"
        best_acc = -1.0

        for epoch in range(max(1, int(config.epochs))):
            model.train()
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            val_acc = _evaluate(torch, model, val_loader, device)
            _save_checkpoint(
                torch=torch,
                path=last_path,
                model=model,
                model_name=config.model_name,
                class_names=train_dataset.classes,
                image_size=config.image_size,
                epoch=epoch + 1,
                val_accuracy=val_acc,
            )
            if val_acc >= best_acc:
                best_acc = val_acc
                _save_checkpoint(
                    torch=torch,
                    path=best_path,
                    model=model,
                    model_name=config.model_name,
                    class_names=train_dataset.classes,
                    image_size=config.image_size,
                    epoch=epoch + 1,
                    val_accuracy=val_acc,
                )

        return best_path

    def load_pretrained(
        self,
        model_name: str,
        device: str = "auto",
        weight_path: Optional[Path] = None,
    ) -> LoadedClassifier:
        torch, nn, optim, datasets, transforms, models = _require_torchvision()
        resolved_device = _normalize_device(torch, device)
        model, class_names = _create_imagenet_model(
            torch=torch,
            models=models,
            model_name=model_name,
            weight_path=weight_path,
        )
        model.to(resolved_device)
        model.eval()
        return LoadedClassifier(
            model=model,
            model_name=model_name,
            class_names=class_names,
            image_size=224,
            device=resolved_device,
            model_path=weight_path,
        )

    def load_checkpoint(
        self,
        model_path: Path,
        device: str = "auto",
    ) -> LoadedClassifier:
        torch, nn, optim, datasets, transforms, models = _require_torchvision()
        resolved_device = _normalize_device(torch, device)
        checkpoint = torch.load(str(model_path), map_location=resolved_device)
        model_name = str(checkpoint.get("model_name", ClassificationModelName.RESNET18))
        class_names = list(checkpoint.get("class_names", []))
        image_size = int(checkpoint.get("image_size", 224))
        if not class_names:
            raise ValueError("Checkpoint does not contain class_names.")

        model = _create_model(
            models=models,
            nn=nn,
            model_name=model_name,
            num_classes=len(class_names),
            pretrained=False,
            freeze_backbone=False,
            torch=torch,
            pretrained_weight_path=None,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(resolved_device)
        model.eval()
        return LoadedClassifier(
            model=model,
            model_name=model_name,
            class_names=class_names,
            image_size=image_size,
            device=resolved_device,
            model_path=model_path,
        )

    def predict(
        self,
        classifier: LoadedClassifier,
        image_path: Path,
        topk: int = 5,
    ) -> PredictionResult:
        torch, nn, optim, datasets, transforms, models = _require_torchvision()
        transform = _build_transform(transforms, classifier.image_size, is_train=False)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = transform(image).unsqueeze(0).to(classifier.device)

        started_at = time.perf_counter()
        with torch.no_grad():
            logits = classifier.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            k = min(max(1, int(topk)), len(classifier.class_names))
            scores, indices = torch.topk(probabilities, k)
        inference_ms = (time.perf_counter() - started_at) * 1000.0

        predictions = []  # type: List[PredictionItem]
        for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist()):
            predictions.append(
                PredictionItem(
                    label=classifier.class_names[int(index)],
                    score=float(score),
                )
            )
        return PredictionResult(
            image_path=image_path,
            model_name=classifier.model_name,
            model_path=classifier.model_path,
            inference_ms=inference_ms,
            predictions=predictions,
        )


def _require_torchvision() -> Tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from torch import nn, optim
        from torchvision import datasets, models, transforms
    except Exception as exc:
        raise RuntimeError(
            "Image classification needs torch and torchvision. "
            "Please install them with: pip install -r requirements-classification.txt"
        ) from exc
    return torch, nn, optim, datasets, transforms, models


def _normalize_device(torch: Any, device: str) -> Any:
    requested = (device or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested in ("cuda", "0"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda:0")
    return torch.device(requested)


def _build_transform(transforms: Any, image_size: int, is_train: bool) -> Any:
    size = max(32, int(image_size))
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(size * 1.15)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def _create_model(
    models: Any,
    nn: Any,
    model_name: str,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
    torch: Any,
    pretrained_weight_path: Optional[Path],
) -> Any:
    normalized = _normalize_model_name(model_name)
    if normalized == ClassificationModelName.RESNET18:
        weights = None if pretrained_weight_path else (models.ResNet18_Weights.DEFAULT if pretrained else None)
        model = models.resnet18(weights=weights)
        if pretrained_weight_path:
            _load_weight_file(torch, model, pretrained_weight_path)
        if freeze_backbone:
            _freeze_parameters(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if normalized == ClassificationModelName.MOBILENET_V3_SMALL:
        weights = None if pretrained_weight_path else (models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        model = models.mobilenet_v3_small(weights=weights)
        if pretrained_weight_path:
            _load_weight_file(torch, model, pretrained_weight_path)
        if freeze_backbone:
            _freeze_parameters(model)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported classification model: {model_name}")


def _create_imagenet_model(
    torch: Any,
    models: Any,
    model_name: str,
    weight_path: Optional[Path],
) -> Tuple[Any, List[str]]:
    normalized = _normalize_model_name(model_name)
    if normalized == ClassificationModelName.RESNET18:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=None if weight_path else weights)
        if weight_path:
            _load_weight_file(torch, model, weight_path)
        return model, list(weights.meta["categories"])
    if normalized == ClassificationModelName.MOBILENET_V3_SMALL:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=None if weight_path else weights)
        if weight_path:
            _load_weight_file(torch, model, weight_path)
        return model, list(weights.meta["categories"])
    raise ValueError(f"Unsupported classification model: {model_name}")


def _normalize_model_name(model_name: str) -> str:
    value = (model_name or "").strip().lower().replace("-", "_")
    if value in ("mobilenetv3small", "mobilenet_v3_small"):
        return ClassificationModelName.MOBILENET_V3_SMALL
    if value == "resnet18":
        return ClassificationModelName.RESNET18
    raise ValueError(f"Unsupported classification model: {model_name}")


def _freeze_parameters(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def _load_weight_file(torch: Any, model: Any, weight_path: Path) -> None:
    raw = torch.load(str(weight_path), map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported pretrained weight file: {weight_path}")
    cleaned = {}
    for key, value in state_dict.items():
        cleaned_key = str(key)
        if cleaned_key.startswith("module."):
            cleaned_key = cleaned_key[len("module.") :]
        if cleaned_key.startswith("model."):
            cleaned_key = cleaned_key[len("model.") :]
        cleaned[cleaned_key] = value
    model.load_state_dict(cleaned)


def _evaluate(torch: Any, model: Any, val_loader: Any, device: Any) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    model.train()
    return float(correct / total) if total else 0.0


def _save_checkpoint(
    torch: Any,
    path: Path,
    model: Any,
    model_name: str,
    class_names: List[str],
    image_size: int,
    epoch: int,
    val_accuracy: float,
) -> None:
    torch.save(
        {
            "model_name": model_name,
            "class_names": class_names,
            "image_size": image_size,
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "state_dict": model.state_dict(),
        },
        str(path),
    )
