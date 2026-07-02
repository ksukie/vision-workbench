import zipfile
from pathlib import Path

import pytest

from yolo26_detection.api import create_yolo26_detection_service
from yolo26_detection.configuration import Yolo26DetectionConfig
from yolo26_detection.domain import DetectionSettings
from yolo26_detection.infrastructure import Yolo26ModelRegistry
from vision_workbench.model_files import ModelFileError


def write_model_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.pkl", b"fake model")


def test_model_registry_lists_official_and_custom_models(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    user_model_dir = tmp_path / "user_models"
    model_dir.mkdir()
    user_model_dir.mkdir()
    write_model_archive(model_dir / "yolo26n.pt")
    (model_dir / "yolo26s.pt").write_bytes(b"partial")
    write_model_archive(user_model_dir / "custom.pt")
    (user_model_dir / "broken.pt").write_bytes(b"fake")
    config = Yolo26DetectionConfig(
        yolo26_source_dir=tmp_path / "source",
        model_dir=model_dir,
        user_model_dir=user_model_dir,
    )

    registry = Yolo26ModelRegistry(config)
    models = registry.list_models(include_missing_official=True)
    by_name = {model.name: model for model in models}

    assert by_name["yolo26n.pt"].exists
    assert by_name["yolo26n.pt"].is_official
    assert not by_name["yolo26s.pt"].exists
    assert by_name["custom.pt"].exists
    assert not by_name["custom.pt"].is_official
    assert "broken.pt" not in by_name


def test_model_registry_rejects_invalid_custom_model(tmp_path: Path) -> None:
    registry = Yolo26ModelRegistry(
        Yolo26DetectionConfig(
            yolo26_source_dir=tmp_path / "source",
            model_dir=tmp_path / "models",
            user_model_dir=tmp_path / "user_models",
        )
    )

    with pytest.raises(ValueError):
        registry.add_custom_model(tmp_path / "bad.onnx")

    with pytest.raises(FileNotFoundError):
        registry.add_custom_model(tmp_path / "missing.pt")

    bad_pt = tmp_path / "bad.pt"
    bad_pt.write_bytes(b"partial")
    with pytest.raises(ModelFileError):
        registry.add_custom_model(bad_pt)


def test_detection_settings_normalized_device() -> None:
    assert DetectionSettings(device="auto").normalized_device() is None
    assert DetectionSettings(device="cpu").normalized_device() == "cpu"


def test_yolo26_service_can_be_created_without_loading_model(tmp_path: Path) -> None:
    config = Yolo26DetectionConfig(
        yolo26_source_dir=tmp_path / "source",
        model_dir=tmp_path / "models",
        user_model_dir=tmp_path / "user_models",
    )

    service = create_yolo26_detection_service(config)

    assert service.list_models()[0].name == "yolo26n.pt"
