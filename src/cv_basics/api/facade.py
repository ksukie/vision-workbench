"""Public facade for Vision Workbench."""

from __future__ import annotations

from typing import Dict, List, Optional

from ..application import build_default_service
from ..configuration import AppConfig, load_config
from ..domain import EffectName, ImageArray, PathLike, ProcessingParams
from ..ports import ImageProcessingServicePort


_default_service = None  # type: Optional[ImageProcessingServicePort]


def create_image_processing_service(
    config: Optional[AppConfig] = None,
) -> ImageProcessingServicePort:
    """Factory for a fully wired image-processing service."""

    return build_default_service(config or AppConfig())


def get_default_service() -> ImageProcessingServicePort:
    """Return a lazily built default service instance."""

    global _default_service
    if _default_service is None:
        _default_service = create_image_processing_service()
    return _default_service


def load_image(path: PathLike) -> ImageArray:
    return get_default_service().load_image(path)


def save_image(image: ImageArray, path: PathLike) -> None:
    get_default_service().save_image(image, path)


def apply_effect(
    image: ImageArray,
    effect_name: str,
    params: Optional[ProcessingParams] = None,
) -> ImageArray:
    return get_default_service().apply_effect(image, effect_name, params)


def to_grayscale(image: ImageArray) -> ImageArray:
    return apply_effect(image, EffectName.GRAYSCALE)


def apply_blur(image: ImageArray, ksize: int = 9) -> ImageArray:
    return apply_effect(image, EffectName.BLUR, ProcessingParams(blur_kernel=ksize))


def detect_edges(image: ImageArray, low: int = 80, high: int = 160) -> ImageArray:
    return apply_effect(
        image,
        EffectName.EDGES,
        ProcessingParams(edge_low=low, edge_high=high),
    )


def threshold_image(image: ImageArray, threshold: int = 127) -> ImageArray:
    return apply_effect(
        image,
        EffectName.THRESHOLD,
        ProcessingParams(threshold=threshold),
    )


def cartoonize(image: ImageArray) -> ImageArray:
    return apply_effect(image, EffectName.CARTOON)


def show_rgb_space(image: ImageArray) -> ImageArray:
    return apply_effect(image, EffectName.RGB_SPACE)


def show_hsv_space(image: ImageArray) -> ImageArray:
    return apply_effect(image, EffectName.HSV_SPACE)


def extract_red_channel(image: ImageArray) -> ImageArray:
    return apply_effect(image, EffectName.RED_CHANNEL)


def extract_green_channel(image: ImageArray) -> ImageArray:
    return apply_effect(image, EffectName.GREEN_CHANNEL)


def extract_blue_channel(image: ImageArray) -> ImageArray:
    return apply_effect(image, EffectName.BLUE_CHANNEL)


def extract_hue_channel(image: ImageArray) -> ImageArray:
    return apply_effect(image, EffectName.HUE_CHANNEL)


def extract_saturation_channel(image: ImageArray) -> ImageArray:
    return apply_effect(image, EffectName.SATURATION_CHANNEL)


def extract_value_channel(image: ImageArray) -> ImageArray:
    return apply_effect(image, EffectName.VALUE_CHANNEL)


def gray_histogram(image: ImageArray) -> ImageArray:
    return apply_effect(image, EffectName.GRAY_HISTOGRAM)


def rgb_histogram(image: ImageArray) -> ImageArray:
    return apply_effect(image, EffectName.RGB_HISTOGRAM)


def erode_image(
    image: ImageArray,
    kernel_size: int = 5,
    iterations: int = 1,
) -> ImageArray:
    return apply_effect(
        image,
        EffectName.ERODE,
        ProcessingParams(
            morphology_kernel=kernel_size,
            morphology_iterations=iterations,
        ),
    )


def dilate_image(
    image: ImageArray,
    kernel_size: int = 5,
    iterations: int = 1,
) -> ImageArray:
    return apply_effect(
        image,
        EffectName.DILATE,
        ProcessingParams(
            morphology_kernel=kernel_size,
            morphology_iterations=iterations,
        ),
    )


def morph_open_image(
    image: ImageArray,
    kernel_size: int = 5,
    iterations: int = 1,
) -> ImageArray:
    return apply_effect(
        image,
        EffectName.MORPH_OPEN,
        ProcessingParams(
            morphology_kernel=kernel_size,
            morphology_iterations=iterations,
        ),
    )


def morph_close_image(
    image: ImageArray,
    kernel_size: int = 5,
    iterations: int = 1,
) -> ImageArray:
    return apply_effect(
        image,
        EffectName.MORPH_CLOSE,
        ProcessingParams(
            morphology_kernel=kernel_size,
            morphology_iterations=iterations,
        ),
    )


def rotate_image(image: ImageArray, angle: int = 30) -> ImageArray:
    return apply_effect(
        image,
        EffectName.ROTATE,
        ProcessingParams(rotate_angle=angle),
    )


def scale_image(image: ImageArray, percent: int = 120) -> ImageArray:
    return apply_effect(
        image,
        EffectName.SCALE,
        ProcessingParams(scale_percent=percent),
    )


def center_crop_image(image: ImageArray, percent: int = 70) -> ImageArray:
    return apply_effect(
        image,
        EffectName.CENTER_CROP,
        ProcessingParams(crop_percent=percent),
    )


def perspective_warp(image: ImageArray, shift: int = 12) -> ImageArray:
    return apply_effect(
        image,
        EffectName.PERSPECTIVE_WARP,
        ProcessingParams(perspective_shift=shift),
    )


def get_image_info(image: ImageArray) -> Dict[str, object]:
    return get_default_service().get_image_info(image).to_dict()


def list_effects() -> List[str]:
    return get_default_service().list_effects()
