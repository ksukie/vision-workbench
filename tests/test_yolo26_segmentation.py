import zipfile
from pathlib import Path

import numpy as np
import pytest

import yolo26_segmentation.api.facade as segmentation_facade
from yolo26_segmentation.api import create_yolo26_segmentation_service
from yolo26_segmentation.configuration import Yolo26SegmentationConfig
from yolo26_segmentation.domain import SegmentationOutput, SegmentationSettings
from yolo26_segmentation.infrastructure import Yolo26SegmentationModelRegistry
from yolo26_segmentation.infrastructure.segmentation_backend import _detected_class_names
from vision_workbench.model_files import ModelFileError


def write_model_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.pkl", b"fake model")


def test_segmentation_model_registry_lists_task_models(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    custom_dir = model_dir / "custom"
    model_dir.mkdir()
    custom_dir.mkdir()
    write_model_archive(model_dir / "yolo26n-seg.pt")
    write_model_archive(model_dir / "yolo26n-sem.pt")
    (model_dir / "yolo26s-seg.pt").write_bytes(b"partial")
    write_model_archive(custom_dir / "custom-seg.pt")
    write_model_archive(custom_dir / "custom-sem.pt")
    (custom_dir / "broken-seg.pt").write_bytes(b"partial")
    config = Yolo26SegmentationConfig(
        yolo26_source_dir=tmp_path / "source",
        model_dir=model_dir,
        custom_model_dir=custom_dir,
    )

    registry = Yolo26SegmentationModelRegistry(config)

    segment_names = [model.name for model in registry.list_models("segment")]
    semantic_names = [model.name for model in registry.list_models("semantic")]
    segment_by_name = {model.name: model for model in registry.list_models("segment")}
    assert "yolo26n-seg.pt" in segment_names
    assert "yolo26n-sem.pt" not in segment_names
    assert "custom-seg.pt" in segment_names
    assert "custom-sem.pt" not in segment_names
    assert "broken-seg.pt" not in segment_names
    assert "yolo26n-sem.pt" in semantic_names
    assert "yolo26n-seg.pt" not in semantic_names
    assert "custom-sem.pt" in semantic_names
    assert "custom-seg.pt" not in semantic_names
    assert not segment_by_name["yolo26s-seg.pt"].exists


def test_segmentation_model_registry_rejects_invalid_custom_model(tmp_path: Path) -> None:
    registry = Yolo26SegmentationModelRegistry(
        Yolo26SegmentationConfig(
            yolo26_source_dir=tmp_path / "source",
            model_dir=tmp_path / "models",
            custom_model_dir=tmp_path / "models" / "custom",
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


def test_segmentation_settings_normalized_device() -> None:
    assert SegmentationSettings(device="auto").normalized_device() is None
    assert SegmentationSettings(device="cpu").normalized_device() == "cpu"


def test_segment_image_facade_accepts_image_path(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"

    class FakeService:
        def __init__(self) -> None:
            self.loaded = None
            self.loaded_image = None
            self.segmented_shape = None

        def load_model(self, model_path):
            self.loaded = Path(model_path)

        def load_image(self, path):
            self.loaded_image = Path(path)
            return np.zeros((8, 10, 3), dtype=np.uint8)

        def segment_image(self, image, settings):
            self.segmented_shape = np.asarray(image).shape
            return SegmentationOutput(annotated_frame=image, item_count=0, inference_ms=1.0)

    service = FakeService()
    old_service = segmentation_facade._default_service
    segmentation_facade._default_service = service
    try:
        result = segmentation_facade.segment_image(image_path, model_path=tmp_path / "model.pt")
    finally:
        segmentation_facade._default_service = old_service

    assert service.loaded == tmp_path / "model.pt"
    assert service.loaded_image == image_path
    assert service.segmented_shape == (8, 10, 3)
    assert result.inference_ms == 1.0


def test_segmentation_service_can_be_created_without_loading_model(tmp_path: Path) -> None:
    config = Yolo26SegmentationConfig(
        yolo26_source_dir=tmp_path / "source",
        model_dir=tmp_path / "models",
        custom_model_dir=tmp_path / "models" / "custom",
    )

    service = create_yolo26_segmentation_service(config)

    assert service.list_models("segment")[0].name == "yolo26n-seg.pt"


def test_segmentation_detected_class_names_only_uses_detected_classes() -> None:
    class TensorLike:
        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [41.0, 39.0, 41.0]

    class Boxes:
        cls = TensorLike()

    class Result:
        boxes = Boxes()
        names = {
            9: "traffic light",
            10: "fire hydrant",
            39: "bottle",
            41: "cup",
        }

    assert _detected_class_names(Result()) == ("cup", "bottle", "cup")
