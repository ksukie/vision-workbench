"""Ultralytics-backed YOLO26 segmentation adapter."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from ..configuration import Yolo26SegmentationConfig
from ..domain import ImageArray, PathLike, SegmentationOutput, SegmentationSettings


class UltralyticsYolo26SegmentationBackend:
    """Loads YOLO26 segmentation weights and runs inference."""

    def __init__(self, config: Yolo26SegmentationConfig = Yolo26SegmentationConfig()) -> None:
        self._config = config
        self._model = None
        self._model_path = None

    def load_model(self, model_path: PathLike) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        YOLO = self._import_yolo()
        self._model = YOLO(str(path))
        self._model_path = path

    @property
    def model_path(self):
        return self._model_path

    def segment(self, image: ImageArray, settings: SegmentationSettings) -> SegmentationOutput:
        if self._model is None:
            raise RuntimeError("No YOLO26 segmentation model is loaded.")
        started = time.perf_counter()
        results = self._model.predict(
            source=image,
            imgsz=int(settings.image_size),
            conf=float(settings.confidence),
            iou=float(settings.iou),
            device=settings.normalized_device(),
            verbose=False,
        )
        inference_ms = (time.perf_counter() - started) * 1000.0
        result = results[0]
        annotated = result.plot()
        item_count = _count_result_items(result)
        names = tuple(str(value) for value in getattr(result, "names", {}).values())
        return SegmentationOutput(annotated_frame=annotated, item_count=item_count, inference_ms=inference_ms, names=names)

    def _import_yolo(self):
        source_dir = self._config.yolo26_source_dir
        if source_dir.exists():
            source_text = str(source_dir.resolve())
            if source_text not in sys.path:
                sys.path.insert(0, source_text)
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Cannot import Ultralytics YOLO26 runtime. "
                "Install it from the project root with: pip install -r requirements-yolo26.txt"
            ) from exc
        return YOLO


def _count_result_items(result) -> int:
    masks = getattr(result, "masks", None)
    if masks is not None and getattr(masks, "data", None) is not None:
        return int(len(masks.data))
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        return int(len(boxes))
    return 0

