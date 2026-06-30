from pathlib import Path

from yolo26_training.application import build_default_service
from yolo26_training.configuration import Yolo26TrainingConfig
from yolo26_training.infrastructure import YoloDetectionDatasetValidator
from yolo26_training.runner import main as runner_main


def _make_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / "sample.jpg").write_bytes(b"fake image")
        (root / "labels" / split / "sample.txt").write_text(
            "0 0.5 0.5 0.25 0.25\n",
            encoding="utf-8",
        )
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: defect",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def _make_segment_dataset(root: Path) -> Path:
    data_yaml = _make_dataset(root)
    for split in ("train", "val"):
        (root / "labels" / split / "sample.txt").write_text(
            "0 0.1 0.1 0.7 0.1 0.7 0.7 0.1 0.7\n",
            encoding="utf-8",
        )
    return data_yaml


def _make_semantic_mask_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "masks" / split).mkdir(parents=True)
        (root / "images" / split / "sample.jpg").write_bytes(b"fake image")
        (root / "masks" / split / "sample.png").write_bytes(b"fake mask")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "task: semantic",
                "train: images/train",
                "val: images/val",
                "masks_dir: masks",
                "names:",
                "  0: background",
                "  1: defect",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_yolo_training_dataset_validator_accepts_basic_detection_dataset(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path / "dataset")
    validator = YoloDetectionDatasetValidator(Yolo26TrainingConfig())

    report = validator.validate(data_yaml)

    assert report.ok
    assert report.summary.train_images == 1
    assert report.summary.val_images == 1
    assert report.summary.classes == 1


def test_yolo_training_dataset_validator_rejects_missing_labels(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path / "dataset")
    (tmp_path / "dataset" / "labels" / "train" / "sample.txt").unlink()
    validator = YoloDetectionDatasetValidator(Yolo26TrainingConfig())

    report = validator.validate(data_yaml)

    assert not report.ok
    assert any("Missing label" in error for error in report.errors)


def test_yolo_training_dataset_validator_rejects_unsupported_segment_label(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path / "dataset")
    (tmp_path / "dataset" / "labels" / "train" / "sample.txt").write_text(
        "0 0.1 0.1 0.2 0.2 0.3 0.3\n",
        encoding="utf-8",
    )
    validator = YoloDetectionDatasetValidator(Yolo26TrainingConfig())

    report = validator.validate(data_yaml)

    assert not report.ok
    assert any("Expected detect format" in error for error in report.errors)


def test_yolo_training_dataset_validator_accepts_instance_segmentation_dataset(tmp_path: Path) -> None:
    data_yaml = _make_segment_dataset(tmp_path / "dataset")
    validator = YoloDetectionDatasetValidator(Yolo26TrainingConfig())

    report = validator.validate(data_yaml, task="segment")

    assert report.ok
    assert report.task == "segment"


def test_yolo_training_dataset_validator_accepts_semantic_mask_dataset(tmp_path: Path) -> None:
    data_yaml = _make_semantic_mask_dataset(tmp_path / "dataset")
    validator = YoloDetectionDatasetValidator(Yolo26TrainingConfig())

    report = validator.validate(data_yaml, task="semantic")

    assert report.ok
    assert report.summary.train_labels == 1
    assert report.summary.val_labels == 1


def test_yolo_training_runner_dry_run_validates_without_training(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path / "dataset")
    model_path = tmp_path / "yolo26n.pt"
    model_path.write_bytes(b"fake model")

    exit_code = runner_main(
        [
            "--task",
            "detect",
            "--data",
            str(data_yaml),
            "--model",
            str(model_path),
            "--project",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )

    assert exit_code == 0


def test_yolo_training_service_builds_runner_command(tmp_path: Path) -> None:
    config = Yolo26TrainingConfig(
        yolo26_source_dir=tmp_path / "third_party" / "yolo26_source",
        model_dir=tmp_path / "models",
        custom_model_dir=tmp_path / "models" / "custom",
        segmentation_model_dir=tmp_path / "seg_models",
        segmentation_custom_model_dir=tmp_path / "seg_models" / "custom",
        dataset_dir=tmp_path / "datasets",
        runs_dir=tmp_path / "runs",
    )
    service = build_default_service(config)

    assert service.default_model().name == "yolo26n.pt"
    assert service.default_model("segment").name == "yolo26n-seg.pt"
    assert service.default_model("semantic").name == "yolo26n-sem.pt"
