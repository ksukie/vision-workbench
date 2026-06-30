"""Image effect strategies and registry."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from ..domain import EffectName, ImageArray, ProcessingParams
from ..infrastructure import ensure_color_bgr, ensure_uint8_image, normalize_odd_kernel
from ..ports import ImageOperation


class GrayscaleOperation:
    name = EffectName.GRAYSCALE

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        image = ensure_uint8_image(image)
        if image.ndim == 2:
            return image.copy()
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class BlurOperation:
    name = EffectName.BLUR

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        image = ensure_uint8_image(image)
        kernel = normalize_odd_kernel(params.blur_kernel)
        return cv2.GaussianBlur(image, (kernel, kernel), 0)


class EdgeDetectionOperation:
    name = EffectName.EDGES

    def __init__(self) -> None:
        self._grayscale = GrayscaleOperation()

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        gray = self._grayscale.apply(image, params)
        low_value = int(np.clip(params.edge_low, 0, 255))
        high_value = int(np.clip(params.edge_high, low_value + 1, 255))
        return cv2.Canny(gray, low_value, high_value)


class ThresholdOperation:
    name = EffectName.THRESHOLD

    def __init__(self) -> None:
        self._grayscale = GrayscaleOperation()

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        gray = self._grayscale.apply(image, params)
        threshold_value = int(np.clip(params.threshold, 0, 255))
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        return binary


class CartoonOperation:
    name = EffectName.CARTOON

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        color = ensure_color_bgr(image)
        smoothed = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=9,
            C=2,
        )
        return cv2.bitwise_and(smoothed, smoothed, mask=edges)


class RgbSpaceOperation:
    name = EffectName.RGB_SPACE

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        color = ensure_color_bgr(image)
        blue, green, red = cv2.split(color)
        return _horizontal_montage(
            [
                _colorize_channel(red, 2),
                _colorize_channel(green, 1),
                _colorize_channel(blue, 0),
            ]
        )


class HsvSpaceOperation:
    name = EffectName.HSV_SPACE

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        color = ensure_color_bgr(image)
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        hue_scaled = cv2.convertScaleAbs(hue, alpha=255.0 / 179.0)
        hue_view = cv2.applyColorMap(hue_scaled, cv2.COLORMAP_HSV)
        return _horizontal_montage(
            [
                hue_view,
                cv2.cvtColor(saturation, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(value, cv2.COLOR_GRAY2BGR),
            ]
        )


class BgrChannelOperation:
    def __init__(self, name: str, channel_index: int) -> None:
        self.name = name
        self._channel_index = channel_index

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        color = ensure_color_bgr(image)
        return cv2.split(color)[self._channel_index]


class HsvChannelOperation:
    def __init__(self, name: str, channel_index: int) -> None:
        self.name = name
        self._channel_index = channel_index

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        color = ensure_color_bgr(image)
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        channel = cv2.split(hsv)[self._channel_index]
        if self._channel_index == 0:
            return cv2.convertScaleAbs(channel, alpha=255.0 / 179.0)
        return channel


class GrayHistogramOperation:
    name = EffectName.GRAY_HISTOGRAM

    def __init__(self) -> None:
        self._grayscale = GrayscaleOperation()

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        gray = self._grayscale.apply(image, params)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return _draw_histogram([hist], [(25, 25, 25)])


class RgbHistogramOperation:
    name = EffectName.RGB_HISTOGRAM

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        color = ensure_color_bgr(image)
        blue_hist = cv2.calcHist([color], [0], None, [256], [0, 256])
        green_hist = cv2.calcHist([color], [1], None, [256], [0, 256])
        red_hist = cv2.calcHist([color], [2], None, [256], [0, 256])
        return _draw_histogram(
            [red_hist, green_hist, blue_hist],
            [(0, 0, 220), (0, 160, 0), (220, 0, 0)],
        )


class MorphologyOperation:
    def __init__(self, name: str, op: int) -> None:
        self.name = name
        self._op = op

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        image = ensure_uint8_image(image)
        kernel_size = normalize_odd_kernel(params.morphology_kernel)
        iterations = int(np.clip(params.morphology_iterations, 1, 10))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        if self._op == cv2.MORPH_ERODE:
            return cv2.erode(image, kernel, iterations=iterations)
        if self._op == cv2.MORPH_DILATE:
            return cv2.dilate(image, kernel, iterations=iterations)
        return cv2.morphologyEx(image, self._op, kernel, iterations=iterations)


class RotateOperation:
    name = EffectName.ROTATE

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        image = ensure_uint8_image(image)
        angle = float(np.clip(params.rotate_angle, -180, 180))
        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_value = abs(matrix[0, 0])
        sin_value = abs(matrix[0, 1])
        new_width = int((height * sin_value) + (width * cos_value))
        new_height = int((height * cos_value) + (width * sin_value))
        matrix[0, 2] += (new_width / 2.0) - center[0]
        matrix[1, 2] += (new_height / 2.0) - center[1]
        return cv2.warpAffine(
            image,
            matrix,
            (max(1, new_width), max(1, new_height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )


class ScaleOperation:
    name = EffectName.SCALE

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        image = ensure_uint8_image(image)
        scale = float(np.clip(params.scale_percent, 10, 300)) / 100.0
        height, width = image.shape[:2]
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(image, new_size, interpolation=interpolation)


class CenterCropOperation:
    name = EffectName.CENTER_CROP

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        image = ensure_uint8_image(image)
        percent = float(np.clip(params.crop_percent, 10, 100)) / 100.0
        height, width = image.shape[:2]
        crop_width = max(1, int(width * percent))
        crop_height = max(1, int(height * percent))
        x1 = max(0, (width - crop_width) // 2)
        y1 = max(0, (height - crop_height) // 2)
        return image[y1 : y1 + crop_height, x1 : x1 + crop_width].copy()


class PerspectiveWarpOperation:
    name = EffectName.PERSPECTIVE_WARP

    def apply(self, image: ImageArray, params: ProcessingParams) -> ImageArray:
        image = ensure_uint8_image(image)
        height, width = image.shape[:2]
        shift_ratio = float(np.clip(params.perspective_shift, 0, 45)) / 100.0
        shift_x = width * shift_ratio
        shift_y = height * shift_ratio
        source = np.float32(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ]
        )
        target = np.float32(
            [
                [shift_x, shift_y],
                [width - 1 - shift_x, 0],
                [width - 1, height - 1 - shift_y],
                [0, height - 1],
            ]
        )
        matrix = cv2.getPerspectiveTransform(source, target)
        return cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )


class OperationRegistry:
    """Registry/factory for effect strategies."""

    def __init__(self) -> None:
        self._operations = {}  # type: Dict[str, ImageOperation]

    def register(self, operation: ImageOperation) -> None:
        if operation.name in self._operations:
            raise ValueError(f"Operation already registered: {operation.name}")
        self._operations[operation.name] = operation

    def get(self, name: str) -> ImageOperation:
        try:
            return self._operations[name]
        except KeyError:
            available = ", ".join(self.names())
            raise ValueError(f"Unsupported effect: {name}. Available: {available}")

    def names(self) -> List[str]:
        return list(self._operations.keys())


def build_default_registry() -> OperationRegistry:
    registry = OperationRegistry()
    registry.register(GrayscaleOperation())
    registry.register(BlurOperation())
    registry.register(EdgeDetectionOperation())
    registry.register(ThresholdOperation())
    registry.register(CartoonOperation())
    registry.register(RgbSpaceOperation())
    registry.register(HsvSpaceOperation())
    registry.register(BgrChannelOperation(EffectName.RED_CHANNEL, 2))
    registry.register(BgrChannelOperation(EffectName.GREEN_CHANNEL, 1))
    registry.register(BgrChannelOperation(EffectName.BLUE_CHANNEL, 0))
    registry.register(HsvChannelOperation(EffectName.HUE_CHANNEL, 0))
    registry.register(HsvChannelOperation(EffectName.SATURATION_CHANNEL, 1))
    registry.register(HsvChannelOperation(EffectName.VALUE_CHANNEL, 2))
    registry.register(GrayHistogramOperation())
    registry.register(RgbHistogramOperation())
    registry.register(MorphologyOperation(EffectName.ERODE, cv2.MORPH_ERODE))
    registry.register(MorphologyOperation(EffectName.DILATE, cv2.MORPH_DILATE))
    registry.register(MorphologyOperation(EffectName.MORPH_OPEN, cv2.MORPH_OPEN))
    registry.register(MorphologyOperation(EffectName.MORPH_CLOSE, cv2.MORPH_CLOSE))
    registry.register(RotateOperation())
    registry.register(ScaleOperation())
    registry.register(CenterCropOperation())
    registry.register(PerspectiveWarpOperation())
    return registry


def _colorize_channel(channel: ImageArray, bgr_index: int) -> ImageArray:
    colorized = np.zeros((channel.shape[0], channel.shape[1], 3), dtype=np.uint8)
    colorized[:, :, bgr_index] = ensure_uint8_image(channel)
    return colorized


def _horizontal_montage(images: Sequence[ImageArray]) -> ImageArray:
    color_images = [ensure_color_bgr(image) for image in images]
    if not color_images:
        raise ValueError("At least one image is required for a montage.")
    height = color_images[0].shape[0]
    normalized = []
    for image in color_images:
        if image.shape[0] != height:
            width = max(1, int(image.shape[1] * height / image.shape[0]))
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        normalized.append(image)
    separator = np.full((height, 6, 3), 230, dtype=np.uint8)
    pieces = []
    for index, image in enumerate(normalized):
        if index:
            pieces.append(separator)
        pieces.append(image)
    return np.hstack(pieces)


def _draw_histogram(
    histograms: Iterable[ImageArray],
    colors: Sequence[Tuple[int, int, int]],
    width: int = 512,
    height: int = 260,
) -> ImageArray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left_pad = 34
    bottom_pad = 28
    top_pad = 16
    plot_width = width - left_pad - 12
    plot_height = height - top_pad - bottom_pad
    cv2.line(canvas, (left_pad, top_pad), (left_pad, height - bottom_pad), (210, 210, 210), 1)
    cv2.line(
        canvas,
        (left_pad, height - bottom_pad),
        (width - 8, height - bottom_pad),
        (210, 210, 210),
        1,
    )

    for hist, color in zip(histograms, colors):
        hist = cv2.normalize(hist, None, 0, plot_height, cv2.NORM_MINMAX).flatten()
        points = []
        for bin_index, value in enumerate(hist):
            x_coord = left_pad + int(bin_index * plot_width / 255)
            y_coord = height - bottom_pad - int(value)
            points.append((x_coord, y_coord))
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(canvas, start, end, color, 2)
    return canvas
