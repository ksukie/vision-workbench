from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
from PIL import Image

import vision_workbench.input_limits as input_limits
import vision_workbench.runtime_diagnostics as runtime_diagnostics
from image_classification.configuration import ImageClassificationConfig
from image_classification.infrastructure import ClassificationDatasetValidator
from vision_workbench.input_limits import InputLimitError, read_bounded_text, validate_image_file
from vision_workbench.runtime_diagnostics import inspect_training_environment
from vision_workbench.sample_data import create_classification_sample_dataset, create_yolo_sample_dataset
from yolo26_training.configuration import Yolo26TrainingConfig
from yolo26_training.infrastructure import YoloDetectionDatasetValidator


def test_classification_sample_dataset_passes_validation(tmp_path: Path) -> None:
    root = create_classification_sample_dataset(tmp_path / "classification")
    report = ClassificationDatasetValidator(ImageClassificationConfig()).validate(root)

    assert report.ok, report.to_text()
    assert report.class_names == ["blue_circle", "red_square"]
    assert report.train_image_count == 8
    assert report.val_image_count == 4


@pytest.mark.parametrize("task", ["detect", "segment", "semantic"])
def test_yolo_sample_datasets_pass_validation(tmp_path: Path, task: str) -> None:
    yaml_path = create_yolo_sample_dataset(tmp_path / task, task=task)
    report = YoloDetectionDatasetValidator(Yolo26TrainingConfig()).validate(yaml_path, task=task)

    assert report.ok, report.to_text()
    assert report.summary.train_images == 2
    assert report.summary.val_images == 2
    assert report.summary.train_labels == 2
    assert report.summary.val_labels == 2


def test_image_file_size_limit_is_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (16, 16), "red").save(path)
    monkeypatch.setattr(input_limits, "MAX_IMAGE_FILE_BYTES", 8)

    with pytest.raises(InputLimitError, match="safety limit"):
        validate_image_file(path)


def test_image_pixel_limit_is_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (16, 16), "red").save(path)
    monkeypatch.setattr(input_limits, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(InputLimitError, match="pixel safety limit"):
        validate_image_file(path)


def test_dataset_text_limit_is_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "data.yaml"
    path.write_text("path: .\n", encoding="utf-8")
    monkeypatch.setattr(input_limits, "MAX_DATASET_TEXT_BYTES", 4)

    with pytest.raises(InputLimitError, match="metadata"):
        read_bounded_text(path)


def test_runtime_diagnostics_always_returns_actionable_report(tmp_path: Path) -> None:
    report = inspect_training_environment(tmp_path / "runs")

    assert report.python_executable.is_file()
    assert report.output_free_gib >= 0
    assert report.recommended_batch_size in {4, 8, 16}
    assert "建议批量" in report.to_text()


def test_runtime_diagnostics_timeout_returns_safe_fallback(tmp_path: Path, monkeypatch) -> None:
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("diagnostics", 1)

    monkeypatch.setattr(runtime_diagnostics.subprocess, "run", timeout)
    report = inspect_training_environment(tmp_path / "runs", timeout_seconds=1)

    assert not report.ok
    assert report.accelerator == "未知"
    assert "超过 1 秒" in report.issues[0]
