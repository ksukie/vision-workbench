"""Folder-based classification dataset validation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

from ..configuration import ImageClassificationConfig
from ..domain import ClassCount, DatasetValidationReport, PathLike


class ClassificationDatasetValidator:
    """Validates train/val folder classification datasets."""

    def __init__(self, config: ImageClassificationConfig) -> None:
        self._config = config

    def validate(self, root: PathLike, check_images: bool = True) -> DatasetValidationReport:
        dataset_root = Path(root).expanduser()
        messages = []  # type: List[str]
        bad_images = []  # type: List[Path]
        checked_images = 0

        if not dataset_root.exists():
            return DatasetValidationReport(
                root=dataset_root,
                ok=False,
                messages=[f"Dataset directory does not exist: {dataset_root}"],
            )
        if not dataset_root.is_dir():
            return DatasetValidationReport(
                root=dataset_root,
                ok=False,
                messages=[f"Dataset path is not a directory: {dataset_root}"],
            )

        train_dir = dataset_root / "train"
        val_dir = dataset_root / "val"
        if not train_dir.is_dir():
            messages.append("Missing train/ directory.")
        if not val_dir.is_dir():
            messages.append("Missing val/ directory.")

        train_counts = self._count_classes(train_dir) if train_dir.is_dir() else {}
        val_counts = self._count_classes(val_dir) if val_dir.is_dir() else {}
        class_names = sorted(set(train_counts) | set(val_counts))

        if not class_names:
            messages.append("No class folders were found under train/ or val/.")

        for class_name in sorted(set(train_counts) - set(val_counts)):
            messages.append(f"Class '{class_name}' exists in train/ but not in val/.")
        for class_name in sorted(set(val_counts) - set(train_counts)):
            messages.append(f"Class '{class_name}' exists in val/ but not in train/.")

        class_counts = []  # type: List[ClassCount]
        for class_name in class_names:
            train_count = train_counts.get(class_name, 0)
            val_count = val_counts.get(class_name, 0)
            if train_count == 0:
                messages.append(f"Class '{class_name}' has no training images.")
            if val_count == 0:
                messages.append(f"Class '{class_name}' has no validation images.")
            if train_count < 2:
                messages.append(f"Class '{class_name}' has very few training images.")
            class_counts.append(
                ClassCount(
                    name=class_name,
                    train_count=train_count,
                    val_count=val_count,
                )
            )

        if check_images:
            image_paths = list(self._iter_images(train_dir)) + list(self._iter_images(val_dir))
            for image_path in image_paths:
                checked_images += 1
                if not _can_open_image(image_path):
                    bad_images.append(image_path)
            if bad_images:
                messages.append(f"Found {len(bad_images)} unreadable image file(s).")

        has_errors = any(
            message.startswith("Missing")
            or message.startswith("No class")
            or "but not in" in message
            or "has no" in message
            or message.startswith("Found")
            for message in messages
        )
        return DatasetValidationReport(
            root=dataset_root,
            ok=not has_errors,
            messages=messages,
            class_counts=class_counts,
            checked_images=checked_images,
            bad_images=bad_images,
        )

    def _count_classes(self, split_dir: Path) -> Dict[str, int]:
        counts = {}  # type: Dict[str, int]
        for class_dir in _iter_class_dirs(split_dir):
            counts[class_dir.name] = len(list(self._iter_images(class_dir)))
        return counts

    def _iter_images(self, directory: Path) -> Iterable[Path]:
        if not directory.exists():
            return []
        return (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in self._config.image_extensions
        )


def _iter_class_dirs(split_dir: Path) -> List[Path]:
    if not split_dir.exists():
        return []
    return sorted(path for path in split_dir.iterdir() if path.is_dir())


def _can_open_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def infer_raw_class_counts(root: PathLike, extensions: Tuple[str, ...]) -> Dict[str, int]:
    """Count class folders in a raw classification dataset."""

    dataset_root = Path(root).expanduser()
    counts = {}  # type: Dict[str, int]
    for class_dir in _iter_class_dirs(dataset_root):
        counts[class_dir.name] = len(
            [
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in extensions
            ]
        )
    return counts
