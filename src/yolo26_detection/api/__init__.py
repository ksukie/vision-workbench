"""Public API for YOLO26 detection."""

from .facade import (
    add_custom_model,
    create_yolo26_detection_service,
    detect_objects,
    discover_cameras,
    download_official_model,
    get_default_service,
    list_models,
    load_model,
)

__all__ = [
    "add_custom_model",
    "create_yolo26_detection_service",
    "detect_objects",
    "discover_cameras",
    "download_official_model",
    "get_default_service",
    "list_models",
    "load_model",
]

