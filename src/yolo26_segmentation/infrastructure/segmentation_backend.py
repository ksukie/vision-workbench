"""Ultralytics-backed YOLO26 segmentation adapter."""

from __future__ import annotations

import sys
import time
import gc
import threading
from pathlib import Path

from vision_workbench.runtime_security import configure_restricted_model_loading

from ..configuration import Yolo26SegmentationConfig
from ..domain import ImageArray, PathLike, SegmentationOutput, SegmentationSettings


class UltralyticsYolo26SegmentationBackend:
    """Loads YOLO26 segmentation weights and runs inference."""

    def __init__(self, config: Yolo26SegmentationConfig = Yolo26SegmentationConfig()) -> None:
        self._config = config
        self._model = None
        self._model_path = None
        self._model_lock = threading.RLock()

    def load_model(self, model_path: PathLike) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with self._model_lock:
            self.unload_model()
            YOLO = self._import_yolo()
            self._model = YOLO(str(path))
            self._model_path = path

    def unload_model(self) -> None:
        """Release the loaded YOLO model and cached accelerator memory."""

        with self._model_lock:
            model = self._model
            self._model = None
            self._model_path = None
            if model is None:
                _release_accelerator_memory()
                return

            try:
                model.to("cpu")
            except Exception:
                inner_model = getattr(model, "model", None)
                if inner_model is not None:
                    try:
                        inner_model.to("cpu")
                    except Exception:
                        pass
            try:
                predictor = getattr(model, "predictor", None)
                if predictor is not None:
                    predictor.model = None
                    predictor.results = None
            except Exception:
                pass
            del model
        _release_accelerator_memory()

    @property
    def model_path(self):
        return self._model_path

    def segment(self, image: ImageArray, settings: SegmentationSettings) -> SegmentationOutput:
        with self._model_lock:
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
        names = _detected_class_names(result)
        return SegmentationOutput(annotated_frame=annotated, item_count=item_count, inference_ms=inference_ms, names=names)

    def _import_yolo(self):
        configure_restricted_model_loading()
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


def _detected_class_names(result) -> tuple[str, ...]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return tuple()

    class_ids = []  # type: list[int]
    raw_classes = getattr(boxes, "cls", None)
    if raw_classes is not None:
        class_ids.extend(_to_int_values(raw_classes))
    else:
        for box in boxes:
            values = _to_int_values(getattr(box, "cls", None))
            if values:
                class_ids.append(values[0])

    names = _normalized_names(getattr(result, "names", {}) or {})
    return tuple(names.get(class_id, str(class_id)) for class_id in class_ids)


def _to_int_values(values) -> list[int]:
    if values is None:
        return []
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        values = [values]
    result = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _normalized_names(names) -> dict[int, str]:
    if isinstance(names, dict):
        items = names.items()
    else:
        items = enumerate(names)
    normalized = {}
    for key, value in items:
        try:
            normalized[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()
    except Exception:
        pass
