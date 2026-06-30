from pathlib import Path

import numpy as np

from cv_basics import api
from cv_basics.application import ImageProcessingService
from cv_basics.configuration import AppConfig
from cv_basics.domain import EffectName, ImageArray, ProcessingParams
from cv_basics.processing import OperationRegistry


def sample_image() -> np.ndarray:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = [30, 180, 240]
    image[28:36, :] = [255, 255, 255]
    return image


class MemoryImageRepository:
    def __init__(self, image: ImageArray) -> None:
        self.image = image
        self.saved = None

    def load(self, path: Path) -> ImageArray:
        return self.image.copy()

    def save(self, image: ImageArray, path: Path) -> None:
        self.saved = image.copy()


class InvertOperation:
    name = "Invert"

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        return 255 - image


def test_grayscale_shape() -> None:
    gray = api.to_grayscale(sample_image())

    assert gray.shape == (64, 64)
    assert gray.dtype == np.uint8


def test_blur_preserves_shape() -> None:
    image = sample_image()
    blurred = api.apply_blur(image, ksize=8)

    assert blurred.shape == image.shape
    assert blurred.dtype == np.uint8


def test_detect_edges_returns_binary_map() -> None:
    edges = api.detect_edges(sample_image(), low=40, high=120)

    assert edges.shape == (64, 64)
    assert edges.dtype == np.uint8
    assert set(np.unique(edges)).issubset({0, 255})


def test_threshold_returns_binary_map() -> None:
    binary = api.threshold_image(sample_image(), threshold=100)

    assert binary.shape == (64, 64)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 255})


def test_cartoonize_preserves_color_shape() -> None:
    image = sample_image()
    cartoon = api.cartoonize(image)

    assert cartoon.shape == image.shape
    assert cartoon.dtype == np.uint8


def test_image_info() -> None:
    info = api.get_image_info(sample_image())

    assert info["width"] == 64
    assert info["height"] == 64
    assert info["channels"] == 3
    assert info["dtype"] == "uint8"


def test_load_and_save_round_trip(tmp_path: Path) -> None:
    output_path = tmp_path / "round_trip.png"
    image = sample_image()

    api.save_image(image, output_path)
    loaded = api.load_image(output_path)

    assert loaded.shape == image.shape
    assert loaded.dtype == np.uint8


def test_service_lists_registered_effects() -> None:
    service = api.create_image_processing_service()

    assert service.list_effects() == [
        EffectName.GRAYSCALE,
        EffectName.BLUR,
        EffectName.EDGES,
        EffectName.THRESHOLD,
        EffectName.CARTOON,
        EffectName.RGB_SPACE,
        EffectName.HSV_SPACE,
        EffectName.RED_CHANNEL,
        EffectName.GREEN_CHANNEL,
        EffectName.BLUE_CHANNEL,
        EffectName.HUE_CHANNEL,
        EffectName.SATURATION_CHANNEL,
        EffectName.VALUE_CHANNEL,
        EffectName.GRAY_HISTOGRAM,
        EffectName.RGB_HISTOGRAM,
        EffectName.ERODE,
        EffectName.DILATE,
        EffectName.MORPH_OPEN,
        EffectName.MORPH_CLOSE,
        EffectName.ROTATE,
        EffectName.SCALE,
        EffectName.CENTER_CROP,
        EffectName.PERSPECTIVE_WARP,
    ]


def test_color_space_views_are_valid_images() -> None:
    image = sample_image()
    rgb_view = api.show_rgb_space(image)
    hsv_view = api.show_hsv_space(image)

    assert rgb_view.ndim == 3
    assert hsv_view.ndim == 3
    assert rgb_view.dtype == np.uint8
    assert hsv_view.dtype == np.uint8
    assert rgb_view.shape[0] == image.shape[0]
    assert hsv_view.shape[0] == image.shape[0]


def test_channel_extraction_returns_single_channel_images() -> None:
    image = sample_image()
    channels = [
        api.extract_red_channel(image),
        api.extract_green_channel(image),
        api.extract_blue_channel(image),
        api.extract_hue_channel(image),
        api.extract_saturation_channel(image),
        api.extract_value_channel(image),
    ]

    for channel in channels:
        assert channel.shape == image.shape[:2]
        assert channel.dtype == np.uint8


def test_histograms_return_display_images() -> None:
    gray_histogram = api.gray_histogram(sample_image())
    rgb_histogram = api.rgb_histogram(sample_image())

    assert gray_histogram.ndim == 3
    assert rgb_histogram.ndim == 3
    assert gray_histogram.dtype == np.uint8
    assert rgb_histogram.dtype == np.uint8


def test_morphology_operations_preserve_shape() -> None:
    image = sample_image()
    results = [
        api.erode_image(image, kernel_size=3),
        api.dilate_image(image, kernel_size=3),
        api.morph_open_image(image, kernel_size=3),
        api.morph_close_image(image, kernel_size=3),
    ]

    for result in results:
        assert result.shape == image.shape
        assert result.dtype == np.uint8


def test_geometry_operations_return_valid_images() -> None:
    image = sample_image()
    rotated = api.rotate_image(image, angle=30)
    scaled = api.scale_image(image, percent=150)
    cropped = api.center_crop_image(image, percent=50)
    warped = api.perspective_warp(image, shift=10)

    assert rotated.ndim == 3
    assert scaled.shape[:2] == (96, 96)
    assert cropped.shape[:2] == (32, 32)
    assert warped.shape == image.shape
    assert rotated.dtype == np.uint8
    assert scaled.dtype == np.uint8
    assert cropped.dtype == np.uint8
    assert warped.dtype == np.uint8


def test_service_can_be_extended_by_registering_new_operation() -> None:
    registry = OperationRegistry()
    registry.register(InvertOperation())
    service = ImageProcessingService(
        repository=MemoryImageRepository(sample_image()),
        operations=registry,
        config=AppConfig(),
    )

    result = service.apply_effect(sample_image(), "Invert")

    assert np.array_equal(result, 255 - sample_image())
