"""Public facade for YOLO26 segmentation."""

from __future__ import annotations

from typing import List, Optional

from ..application import Yolo26SegmentationService, build_default_service
from ..configuration import Yolo26SegmentationConfig
from ..domain import ImageArray, ModelInfo, PathLike, SegmentationOutput, SegmentationSettings


_default_service = None  # type: Optional[Yolo26SegmentationService]


def create_yolo26_segmentation_service(
    config: Optional[Yolo26SegmentationConfig] = None,
) -> Yolo26SegmentationService:
    return build_default_service(config or Yolo26SegmentationConfig())


def get_default_service() -> Yolo26SegmentationService:
    global _default_service
    if _default_service is None:
        _default_service = create_yolo26_segmentation_service()
    return _default_service


def list_models(task: str = "segment") -> List[ModelInfo]:
    return get_default_service().list_models(task)


def segment_image(
    image: ImageArray,
    model_path: Optional[PathLike] = None,
    settings: SegmentationSettings = SegmentationSettings(),
) -> SegmentationOutput:
    service = get_default_service()
    if model_path is not None:
        service.load_model(model_path)
    return service.segment_image(image, settings)

