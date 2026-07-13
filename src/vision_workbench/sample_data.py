"""Deterministic tiny datasets used to teach and smoke-test training flows."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def sample_image_path() -> Path:
    """Return the bundled image used by desktop quick-start controls."""

    path = Path(__file__).with_name("assets") / "vision_workbench_sample_street.png"
    if not path.is_file():
        raise FileNotFoundError(f"Bundled sample image is missing: {path}")
    return path


def create_classification_sample_dataset(root: Path) -> Path:
    """Create a two-class train/val dataset with simple geometric shapes."""

    dataset_root = Path(root).expanduser()
    counts = {"train": 4, "val": 2}
    for split, count in counts.items():
        for class_name in ("red_square", "blue_circle"):
            class_dir = dataset_root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                image = Image.new("RGB", (96, 96), (242, 245, 249))
                draw = ImageDraw.Draw(image)
                offset = 10 + index * 3
                if class_name == "red_square":
                    draw.rectangle((offset, offset, 78, 78), fill=(220, 55, 65), outline=(120, 20, 30), width=2)
                else:
                    draw.ellipse((offset, offset, 82, 82), fill=(45, 105, 220), outline=(20, 50, 130), width=2)
                image.save(class_dir / f"sample_{index + 1:02d}.png")
    (dataset_root / "README.txt").write_text(
        "Vision Workbench quick-start classification dataset.\n"
        "Classes: red_square, blue_circle. This dataset is only for workflow testing.\n",
        encoding="utf-8",
    )
    return dataset_root.resolve()


def create_yolo_sample_dataset(root: Path, task: str = "detect") -> Path:
    """Create a tiny detection, polygon segmentation, or semantic-mask dataset."""

    normalized = _normalize_task(task)
    dataset_root = Path(root).expanduser()
    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        if normalized == "semantic":
            annotation_dir = dataset_root / "masks" / split
        else:
            annotation_dir = dataset_root / "labels" / split
        annotation_dir.mkdir(parents=True, exist_ok=True)

        for index in range(2):
            left = 22 + index * 8
            top = 24 + index * 5
            right = 98 + index * 5
            bottom = 102 + index * 4
            image = Image.new("RGB", (128, 128), (236, 241, 247))
            draw = ImageDraw.Draw(image)
            draw.rectangle((left, top, right, bottom), fill=(30, 145, 90), outline=(10, 80, 45), width=3)
            stem = f"sample_{index + 1:02d}"
            image.save(image_dir / f"{stem}.png")

            if normalized == "detect":
                x_center = (left + right) / 2 / 128
                y_center = (top + bottom) / 2 / 128
                width = (right - left) / 128
                height = (bottom - top) / 128
                (annotation_dir / f"{stem}.txt").write_text(
                    f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n",
                    encoding="utf-8",
                )
            elif normalized == "segment":
                points = (
                    left / 128,
                    top / 128,
                    right / 128,
                    top / 128,
                    right / 128,
                    bottom / 128,
                    left / 128,
                    bottom / 128,
                )
                (annotation_dir / f"{stem}.txt").write_text(
                    "0 " + " ".join(f"{value:.6f}" for value in points) + "\n",
                    encoding="utf-8",
                )
            else:
                mask = Image.new("L", (128, 128), 0)
                ImageDraw.Draw(mask).rectangle((left, top, right, bottom), fill=1)
                mask.save(annotation_dir / f"{stem}.png")

    yaml_lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        f"task: {normalized}",
    ]
    if normalized == "semantic":
        yaml_lines.append("masks_dir: masks")
        yaml_lines.extend(("names:", "  0: background", "  1: rectangle"))
    else:
        yaml_lines.extend(("names:", "  0: rectangle"))
    data_yaml = dataset_root / "data.yaml"
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    (dataset_root / "README.txt").write_text(
        f"Vision Workbench quick-start YOLO {normalized} dataset. Workflow testing only.\n",
        encoding="utf-8",
    )
    return data_yaml.resolve()


def _normalize_task(task: str) -> str:
    value = str(task or "detect").strip().lower()
    if value not in {"detect", "segment", "semantic"}:
        raise ValueError("Sample YOLO task must be detect, segment, or semantic.")
    return value
