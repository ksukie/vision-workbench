"""Create train/val folder datasets from raw class folders."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List

from ..configuration import ImageClassificationConfig
from ..domain import PathLike


class ClassificationDatasetSplitter:
    """Splits raw class-folder data into train/val folders."""

    def __init__(self, config: ImageClassificationConfig) -> None:
        self._config = config

    def split(
        self,
        input_dir: PathLike,
        output_dir: PathLike,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> Dict[str, Dict[str, int]]:
        source_root = Path(input_dir).expanduser()
        target_root = Path(output_dir).expanduser()
        if not source_root.is_dir():
            raise ValueError(f"Input dataset directory does not exist: {source_root}")
        if target_root.exists() and any(target_root.iterdir()):
            raise ValueError(f"Output directory is not empty: {target_root}")

        ratio = min(max(float(train_ratio), 0.1), 0.95)
        randomizer = random.Random(seed)
        summary = {}  # type: Dict[str, Dict[str, int]]

        class_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
        if not class_dirs:
            raise ValueError("Input directory must contain one folder per class.")

        for class_dir in class_dirs:
            image_paths = [
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in self._config.image_extensions
            ]
            if not image_paths:
                raise ValueError(f"Class '{class_dir.name}' has no images.")

            randomizer.shuffle(image_paths)
            split_index = int(len(image_paths) * ratio)
            if len(image_paths) > 1:
                split_index = min(max(split_index, 1), len(image_paths) - 1)
            train_images = image_paths[:split_index]
            val_images = image_paths[split_index:]

            self._copy_images(train_images, target_root / "train" / class_dir.name)
            self._copy_images(val_images, target_root / "val" / class_dir.name)
            summary[class_dir.name] = {
                "train": len(train_images),
                "val": len(val_images),
            }

        return summary

    def _copy_images(self, image_paths: List[Path], target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for index, image_path in enumerate(image_paths):
            target_path = target_dir / image_path.name
            if target_path.exists():
                target_path = target_dir / f"{image_path.stem}_{index}{image_path.suffix}"
            shutil.copy2(str(image_path), str(target_path))
