"""Dataset validation for basic YOLO training tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from vision_workbench.input_limits import MAX_DATASET_IMAGES, read_bounded_text

from ..configuration import Yolo26TrainingConfig
from ..domain import DatasetValidationReport, DatasetValidationSummary, PathLike


class YoloDetectionDatasetValidator:
    """Validates the YOLO dataset subsets supported by this project."""

    def __init__(self, config: Yolo26TrainingConfig = Yolo26TrainingConfig()) -> None:
        self._config = config

    def validate(
        self,
        data_path: PathLike,
        task: str = "detect",
        allow_missing_labels: bool = False,
        max_examples: int = 20,
    ) -> DatasetValidationReport:
        task = _normalize_task(task)
        errors = []  # type: List[str]
        warnings = []  # type: List[str]
        yaml_path = self._resolve_data_yaml(data_path, errors)
        if yaml_path is None:
            return DatasetValidationReport(
                task=task,
                data_yaml=None,
                dataset_root=None,
                errors=tuple(errors),
                warnings=tuple(warnings),
            )

        try:
            config = _load_dataset_yaml(yaml_path)
        except Exception as exc:
            return DatasetValidationReport(
                task=task,
                data_yaml=yaml_path,
                dataset_root=None,
                errors=(f"Cannot read dataset YAML: {exc}",),
            )

        self._check_task_keys(task, config, errors, warnings)
        class_names = self._parse_class_names(config, errors)
        root = self._resolve_dataset_root(yaml_path, config, errors)
        if root is None:
            return DatasetValidationReport(
                task=task,
                data_yaml=yaml_path,
                dataset_root=None,
                class_names=tuple(class_names),
                errors=tuple(errors),
                warnings=tuple(warnings),
            )

        train_images = self._collect_images(root, config.get("train"), "train", errors, warnings)
        val_images = self._collect_images(root, config.get("val"), "val", errors, warnings)
        if len(train_images) + len(val_images) > MAX_DATASET_IMAGES:
            errors.append(f"Dataset exceeds the {MAX_DATASET_IMAGES:,} image safety limit.")
            train_images = train_images[:MAX_DATASET_IMAGES]
            val_images = val_images[: max(0, MAX_DATASET_IMAGES - len(train_images))]
        if not train_images:
            errors.append("No training images found. Check the 'train' path in data.yaml.")
        if not val_images:
            errors.append("No validation images found. Check the 'val' path in data.yaml.")

        if task == "semantic" and config.get("masks_dir"):
            train_label_count = self._validate_semantic_masks(
                train_images,
                root,
                str(config.get("masks_dir")),
                "train",
                errors,
                warnings,
                max_examples,
            )
            val_label_count = self._validate_semantic_masks(
                val_images,
                root,
                str(config.get("masks_dir")),
                "val",
                errors,
                warnings,
                max_examples,
            )
        else:
            train_label_count = self._validate_images_and_labels(
                train_images,
                len(class_names),
                "train",
                task,
                allow_missing_labels,
                errors,
                warnings,
                max_examples,
            )
            val_label_count = self._validate_images_and_labels(
                val_images,
                len(class_names),
                "val",
                task,
                allow_missing_labels,
                errors,
                warnings,
                max_examples,
            )

        summary = DatasetValidationSummary(
            train_images=len(train_images),
            val_images=len(val_images),
            train_labels=train_label_count,
            val_labels=val_label_count,
            classes=len(class_names),
        )
        return DatasetValidationReport(
            task=task,
            data_yaml=yaml_path,
            dataset_root=root,
            class_names=tuple(class_names),
            summary=summary,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    def _resolve_data_yaml(self, data_path: PathLike, errors: List[str]) -> Optional[Path]:
        path = Path(data_path).expanduser()
        if path.is_file():
            if path.suffix.lower() not in (".yaml", ".yml"):
                errors.append("Dataset entry must be a .yaml/.yml file or a directory containing data.yaml.")
                return None
            return path.resolve()

        if not path.exists():
            errors.append(f"Dataset path does not exist: {path}")
            return None

        candidates = [
            path / "data.yaml",
            path / "data.yml",
            path / "dataset.yaml",
            path / "dataset.yml",
        ]
        found = [candidate for candidate in candidates if candidate.exists()]
        if not found:
            json_candidates = list(path.glob("*.json"))
            if json_candidates:
                errors.append(
                    "COCO JSON-style dataset detected, but this project currently supports only local YOLO "
                    "data.yaml with txt labels or semantic masks."
                )
            else:
                errors.append("No data.yaml found. Provide a YOLO dataset YAML file.")
            return None
        return found[0].resolve()

    def _check_task_keys(
        self,
        task: str,
        config: Dict[str, object],
        errors: List[str],
        warnings: List[str],
    ) -> None:
        yaml_task = str(config.get("task", "")).strip().lower()
        if yaml_task and yaml_task != task:
            errors.append(f"data.yaml task is '{yaml_task}', but selected task is '{task}'.")
        if "kpt_shape" in config or "kpt_names" in config:
            errors.append("Pose dataset keys detected. This training service supports detect/segment/semantic only.")
        if "download" in config:
            errors.append("Auto-download datasets are not supported here. Provide a local prepared dataset.")
        if task in ("detect", "segment") and "masks_dir" in config:
            errors.append("masks_dir is only supported for semantic segmentation datasets.")
        if task == "semantic" and "masks_dir" not in config:
            warnings.append(
                "Semantic task selected without masks_dir. The validator will expect YOLO polygon labels "
                "and Ultralytics will convert polygons to semantic masks during training."
            )

    def _parse_class_names(self, config: Dict[str, object], errors: List[str]) -> List[str]:
        raw_names = config.get("names")
        names = []  # type: List[str]
        if isinstance(raw_names, list):
            names = [str(item) for item in raw_names]
        elif isinstance(raw_names, dict):
            try:
                names = [str(raw_names[index]) for index in sorted(raw_names, key=lambda key: int(key))]
            except Exception:
                errors.append("The 'names' mapping must use numeric class ids, e.g. 0: defect.")
                return []
        else:
            errors.append("Missing or invalid 'names' in data.yaml.")
            return []

        if not names:
            errors.append("No class names found in data.yaml.")
        nc = config.get("nc")
        if nc is not None:
            try:
                if int(nc) != len(names):
                    errors.append(f"'nc' is {nc}, but names contains {len(names)} class(es).")
            except Exception:
                errors.append("'nc' must be an integer when provided.")
        return names

    def _resolve_dataset_root(
        self,
        yaml_path: Path,
        config: Dict[str, object],
        errors: List[str],
    ) -> Optional[Path]:
        raw_root = config.get("path")
        if raw_root in (None, ""):
            root = yaml_path.parent
        else:
            root = Path(str(raw_root)).expanduser()
            if not root.is_absolute():
                root = yaml_path.parent / root
        if not root.exists():
            errors.append(f"Dataset root does not exist: {root}")
            return None
        return root.resolve()

    def _collect_images(
        self,
        root: Path,
        raw_value: object,
        split_name: str,
        errors: List[str],
        warnings: List[str],
    ) -> List[Path]:
        if raw_value in (None, ""):
            errors.append(f"Missing '{split_name}' path in data.yaml.")
            return []
        entries = raw_value if isinstance(raw_value, list) else [raw_value]
        images = []  # type: List[Path]
        for entry in entries:
            item = Path(str(entry)).expanduser()
            if not item.is_absolute():
                item = root / item
            if item.is_file() and item.suffix.lower() == ".txt":
                images.extend(self._read_image_list(item, root, warnings))
            elif item.is_dir():
                images.extend(self._scan_image_dir(item))
            elif item.is_file() and item.suffix.lower() in self._config.image_extensions:
                images.append(item.resolve())
            elif item.suffix.lower() == ".json":
                errors.append(
                    f"{split_name} points to a JSON annotation file. COCO JSON is not supported by this basic trainer."
                )
            else:
                errors.append(f"{split_name} path not found or unsupported: {item}")
        return sorted(set(images))

    def _read_image_list(self, list_path: Path, root: Path, warnings: List[str]) -> List[Path]:
        images = []
        try:
            lines = read_bounded_text(list_path, encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = read_bounded_text(list_path, encoding="gbk").splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            image_path = Path(stripped).expanduser()
            if not image_path.is_absolute():
                image_path = root / image_path
            if image_path.exists() and image_path.suffix.lower() in self._config.image_extensions:
                images.append(image_path.resolve())
            else:
                warnings.append(f"Image listed in {list_path.name} does not exist or is unsupported: {stripped}")
        return images

    def _scan_image_dir(self, directory: Path) -> List[Path]:
        images = []
        for suffix in self._config.image_extensions:
            images.extend(directory.rglob(f"*{suffix}"))
            images.extend(directory.rglob(f"*{suffix.upper()}"))
        return [path.resolve() for path in images if path.is_file()]

    def _validate_images_and_labels(
        self,
        image_paths: Sequence[Path],
        class_count: int,
        split_name: str,
        task: str,
        allow_missing_labels: bool,
        errors: List[str],
        warnings: List[str],
        max_examples: int,
    ) -> int:
        label_count = 0
        missing_labels = []
        invalid_examples = 0
        for image_path in image_paths:
            label_path = _label_path_for_image(image_path)
            if label_path is None or not label_path.exists():
                message = f"Missing label for {split_name} image: {image_path}"
                if allow_missing_labels:
                    missing_labels.append(message)
                else:
                    if invalid_examples < max_examples:
                        errors.append(message)
                    invalid_examples += 1
                continue
            label_count += 1
            self._validate_label_file(label_path, class_count, task, errors, warnings, max_examples)

        if allow_missing_labels and missing_labels:
            shown = missing_labels[:max_examples]
            warnings.extend(shown)
            if len(missing_labels) > len(shown):
                warnings.append(f"{len(missing_labels) - len(shown)} more missing label warning(s) hidden.")
        if not allow_missing_labels and invalid_examples > max_examples:
            errors.append(f"{invalid_examples - max_examples} more missing label error(s) hidden.")
        return label_count

    def _validate_label_file(
        self,
        label_path: Path,
        class_count: int,
        task: str,
        errors: List[str],
        warnings: List[str],
        max_examples: int,
    ) -> None:
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = label_path.read_text(encoding="gbk").splitlines()

        if not any(line.strip() for line in lines):
            warnings.append(f"Empty label file: {label_path}")
            return

        local_errors = 0
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if task == "detect" and len(parts) != 5:
                errors.append(
                    f"{label_path}:{line_number} has {len(parts)} values. "
                    "Expected detect format: class x_center y_center width height."
                )
                local_errors += 1
                if local_errors >= max_examples:
                    return
                continue
            if task in ("segment", "semantic") and (len(parts) < 7 or len(parts) % 2 == 0):
                errors.append(
                    f"{label_path}:{line_number} has {len(parts)} values. "
                    "Expected polygon format: class x1 y1 x2 y2 x3 y3 ..."
                )
                local_errors += 1
                if local_errors >= max_examples:
                    return
                continue
            try:
                class_id = int(float(parts[0]))
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                errors.append(f"{label_path}:{line_number} contains non-numeric label values.")
                continue

            if class_id < 0 or class_id >= class_count:
                errors.append(
                    f"{label_path}:{line_number} class id {class_id} is outside valid range 0..{class_count - 1}."
                )
            if not all(0.0 <= value <= 1.0 for value in coords):
                errors.append(f"{label_path}:{line_number} box values must be normalized to 0..1.")
            if task == "detect":
                _x_center, _y_center, width, height = coords
                if width <= 0 or height <= 0:
                    errors.append(f"{label_path}:{line_number} width and height must be greater than 0.")
            else:
                points = len(coords) // 2
                if points < 3:
                    errors.append(f"{label_path}:{line_number} polygon must have at least 3 points.")

    def _validate_semantic_masks(
        self,
        image_paths: Sequence[Path],
        root: Path,
        masks_dir: str,
        split_name: str,
        errors: List[str],
        warnings: List[str],
        max_examples: int,
    ) -> int:
        mask_count = 0
        missing = 0
        for image_path in image_paths:
            mask_path = _semantic_mask_path_for_image(image_path, root, masks_dir)
            if mask_path is None or not mask_path.exists():
                if missing < max_examples:
                    errors.append(f"Missing semantic mask for {split_name} image: {image_path}")
                missing += 1
                continue
            if mask_path.suffix.lower() not in (".png", ".tif", ".tiff"):
                warnings.append(f"Semantic mask should use lossless format such as .png: {mask_path}")
            mask_count += 1
        if missing > max_examples:
            errors.append(f"{missing - max_examples} more missing semantic mask error(s) hidden.")
        return mask_count


def _label_path_for_image(image_path: Path) -> Optional[Path]:
    parts = list(image_path.parts)
    image_index = None
    for index, part in enumerate(parts):
        if part.lower() == "images":
            image_index = index
            break
    if image_index is None:
        return None
    parts[image_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _semantic_mask_path_for_image(image_path: Path, root: Path, masks_dir: str) -> Optional[Path]:
    parts = list(image_path.parts)
    image_index = None
    for index, part in enumerate(parts):
        if part.lower() == "images":
            image_index = index
            break
    if image_index is not None:
        parts[image_index] = masks_dir
        return Path(*parts).with_suffix(".png")

    try:
        relative = image_path.relative_to(root)
    except ValueError:
        return None
    relative_parts = list(relative.parts)
    if not relative_parts:
        return None
    if len(relative_parts) >= 2:
        relative_parts[0] = masks_dir
    else:
        relative_parts.insert(0, masks_dir)
    return (root / Path(*relative_parts)).with_suffix(".png")


def _normalize_task(task: str) -> str:
    value = str(task or "detect").strip().lower()
    aliases = {
        "detection": "detect",
        "instance": "segment",
        "instance_segmentation": "segment",
        "seg": "segment",
        "semantic_segmentation": "semantic",
        "sem": "semantic",
    }
    value = aliases.get(value, value)
    if value not in ("detect", "segment", "semantic"):
        raise ValueError("task must be detect, segment, or semantic.")
    return value


def _load_dataset_yaml(yaml_path: Path) -> Dict[str, object]:
    try:
        import yaml
    except Exception:
        return _load_minimal_yaml(yaml_path)

    data = yaml.safe_load(read_bounded_text(yaml_path, encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("data.yaml must contain a mapping.")
    return data


def _load_minimal_yaml(yaml_path: Path) -> Dict[str, object]:
    data = {}  # type: Dict[str, object]
    current_key = None  # type: Optional[str]
    lines = read_bounded_text(yaml_path, encoding="utf-8").splitlines()
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        is_nested = raw_line[:1].isspace()
        if is_nested and current_key:
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            nested = data.setdefault(current_key, {})
            if isinstance(nested, dict):
                nested[_parse_scalar(key.strip())] = _parse_scalar(value.strip())
            continue
        current_key = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            data[key] = {}
            current_key = key
        else:
            data[key] = _parse_scalar(value)
    return data


def _parse_scalar(value: str) -> object:
    stripped = value.strip().strip("'\"")
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("'\"") for item in inner.split(",")]
    if stripped.lower() in ("true", "false"):
        return stripped.lower() == "true"
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped
