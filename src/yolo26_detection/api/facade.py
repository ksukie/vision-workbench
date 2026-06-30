"""Public facade for YOLO26 detection."""

from __future__ import annotations

from typing import List, Optional

from ..application import Yolo26DetectionService, build_default_service
from ..configuration import Yolo26DetectionConfig
from ..domain import CameraDevice, DetectionOutput, DetectionSettings, ImageArray, ModelInfo, PathLike


_default_service = None  # type: Optional[Yolo26DetectionService]


def create_yolo26_detection_service(
    config: Optional[Yolo26DetectionConfig] = None,
) -> Yolo26DetectionService:
    return build_default_service(config or Yolo26DetectionConfig())


def get_default_service() -> Yolo26DetectionService:
    global _default_service
    if _default_service is None:
        _default_service = create_yolo26_detection_service()
    return _default_service


def discover_cameras() -> List[CameraDevice]:
    return get_default_service().discover_cameras()


def list_models(include_missing_official: bool = True) -> List[ModelInfo]:
    return get_default_service().list_models(include_missing_official)


def add_custom_model(path: PathLike) -> ModelInfo:
    return get_default_service().add_custom_model(path)


def download_official_model(name: str) -> ModelInfo:
    return get_default_service().download_official_model(name)


def load_model(model_path: PathLike) -> None:
    get_default_service().load_model(model_path)


def detect_objects(
    frame: ImageArray,
    model_path: Optional[PathLike] = None,
    settings: DetectionSettings = DetectionSettings(),
) -> DetectionOutput:
    service = get_default_service()
    if model_path is not None:
        service.load_model(model_path)
    return service.detect_frame(frame, settings)

