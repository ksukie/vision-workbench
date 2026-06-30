from pathlib import Path

from yolo26_segmentation.api import create_yolo26_segmentation_service
from yolo26_segmentation.configuration import Yolo26SegmentationConfig
from yolo26_segmentation.domain import SegmentationSettings
from yolo26_segmentation.infrastructure import Yolo26SegmentationModelRegistry


def test_segmentation_model_registry_lists_task_models(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    custom_dir = model_dir / "custom"
    model_dir.mkdir()
    custom_dir.mkdir()
    (model_dir / "yolo26n-seg.pt").write_bytes(b"fake")
    (model_dir / "yolo26n-sem.pt").write_bytes(b"fake")
    config = Yolo26SegmentationConfig(
        yolo26_source_dir=tmp_path / "source",
        model_dir=model_dir,
        custom_model_dir=custom_dir,
    )

    registry = Yolo26SegmentationModelRegistry(config)

    segment_names = [model.name for model in registry.list_models("segment")]
    semantic_names = [model.name for model in registry.list_models("semantic")]
    assert "yolo26n-seg.pt" in segment_names
    assert "yolo26n-sem.pt" in semantic_names


def test_segmentation_settings_normalized_device() -> None:
    assert SegmentationSettings(device="auto").normalized_device() is None
    assert SegmentationSettings(device="cpu").normalized_device() == "cpu"


def test_segmentation_service_can_be_created_without_loading_model(tmp_path: Path) -> None:
    config = Yolo26SegmentationConfig(
        yolo26_source_dir=tmp_path / "source",
        model_dir=tmp_path / "models",
        custom_model_dir=tmp_path / "models" / "custom",
    )

    service = create_yolo26_segmentation_service(config)

    assert service.list_models("segment")[0].name == "yolo26n-seg.pt"
