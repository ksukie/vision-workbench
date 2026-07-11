"""Ultralytics-backed YOLO26 detector adapter."""

from __future__ import annotations

import sys
import time
import gc
import threading
from pathlib import Path
from typing import Optional, Tuple

from vision_workbench.runtime_security import configure_restricted_model_loading

from ..configuration import Yolo26DetectionConfig
from ..domain import DetectionBox, DetectionOutput, DetectionSettings, ImageArray, PathLike


class UltralyticsYolo26Backend:
    """Loads YOLO26 weights and runs inference through Ultralytics."""

    def __init__(self, config: Yolo26DetectionConfig = Yolo26DetectionConfig()) -> None:
        self._config = config
        self._model = None
        self._model_path = None  # type: Optional[Path]
        self._model_lock = threading.RLock()

    @property
    def model_path(self) -> Optional[Path]:
        return self._model_path

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

    def detect(self, frame: ImageArray, settings: DetectionSettings) -> DetectionOutput:
        with self._model_lock:
            if self._model is None:
                raise RuntimeError("No YOLO26 model is loaded.")

            started = time.perf_counter()
            results = self._model.predict(
                source=frame,
                imgsz=int(settings.image_size),
                conf=float(settings.confidence),
                iou=float(settings.iou),
                device=settings.normalized_device(),
                verbose=False,
            )
        inference_ms = (time.perf_counter() - started) * 1000.0
        result = results[0]
        annotated = result.plot()
        detections = self._extract_boxes(result)
        return DetectionOutput(
            annotated_frame=annotated,
            detections=detections,
            inference_ms=inference_ms,
        )

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

    def _extract_boxes(self, result) -> Tuple[DetectionBox, ...]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return tuple()
        names = getattr(result, "names", {}) or {}
        detections = []
        for box in boxes:
            class_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else -1
            confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
            xyxy_values = box.xyxy[0].detach().cpu().tolist()
            class_name = str(names.get(class_id, class_id))
            detections.append(
                DetectionBox(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    xyxy=tuple(float(value) for value in xyxy_values),
                )
            )
        return tuple(detections)


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
