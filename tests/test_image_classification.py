from pathlib import Path
from typing import Tuple

from PIL import Image

from image_classification import api
from image_classification.configuration import ImageClassificationConfig
from image_classification.domain import ClassificationModelName
from image_classification.runner import main as runner_main


def make_image(path: Path, color: Tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color).save(path)


def make_classification_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        for class_name, color in (("cat", (200, 40, 40)), ("dog", (40, 90, 220))):
            make_image(root / split / class_name / "001.png", color)
            make_image(root / split / class_name / "002.png", color)
    return root


def test_supported_models_include_resnet18_and_mobilenet() -> None:
    models = api.supported_models()

    assert ClassificationModelName.RESNET18 in models
    assert ClassificationModelName.MOBILENET_V3_SMALL in models


def test_validate_classification_dataset_accepts_folder_layout(tmp_path: Path) -> None:
    dataset = make_classification_dataset(tmp_path / "dataset")

    report = api.validate_classification_dataset(dataset)

    assert report.ok
    assert report.class_names == ["cat", "dog"]
    assert report.train_image_count == 4
    assert report.val_image_count == 4
    assert report.checked_images == 8


def test_validate_classification_dataset_rejects_missing_val(tmp_path: Path) -> None:
    make_image(tmp_path / "dataset" / "train" / "cat" / "001.png", (200, 40, 40))

    report = api.validate_classification_dataset(tmp_path / "dataset")

    assert not report.ok
    assert any("Missing val" in message for message in report.messages)


def test_split_classification_dataset_creates_train_val_layout(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for class_name, color in (("cat", (200, 40, 40)), ("dog", (40, 90, 220))):
        for index in range(5):
            make_image(raw / class_name / f"{index}.png", color)
    output = tmp_path / "split"

    summary = api.split_classification_dataset(raw, output, train_ratio=0.8, seed=1)
    report = api.validate_classification_dataset(output)

    assert summary["cat"]["train"] == 4
    assert summary["cat"]["val"] == 1
    assert summary["dog"]["train"] == 4
    assert summary["dog"]["val"] == 1
    assert report.ok


def test_list_classification_models_contains_pretrained_options() -> None:
    service = api.create_image_classification_service(ImageClassificationConfig())

    model_names = [model.name for model in service.list_models()]

    assert "resnet18" in model_names
    assert "mobilenet_v3_small" in model_names


def test_pretrained_weight_status_and_import(tmp_path: Path) -> None:
    config = ImageClassificationConfig(
        model_dir=tmp_path / "models",
        custom_model_dir=tmp_path / "models" / "custom",
        pretrained_model_dir=tmp_path / "models" / "pretrained",
        dataset_dir=tmp_path / "datasets",
        runs_dir=tmp_path / "runs",
    )
    service = api.create_image_classification_service(config)
    source_weight = tmp_path / "source.pth"
    source_weight.write_bytes(b"fake local weight")

    status_before = service.pretrained_weight_status("resnet18")[0]
    status_after = service.import_pretrained_weight("resnet18", source_weight)

    assert not status_before.exists
    assert status_after.exists
    assert status_after.local_path.read_bytes() == b"fake local weight"
    assert status_after.filename == "resnet18-f37072fd.pth"


def test_runner_dry_run_validates_without_training(tmp_path: Path) -> None:
    dataset = make_classification_dataset(tmp_path / "dataset")

    exit_code = runner_main(
        [
            "--data",
            str(dataset),
            "--model",
            "mobilenet_v3_small",
            "--dry-run",
        ]
    )

    assert exit_code == 0
