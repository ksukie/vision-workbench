"""Qt page for the basic computer-vision workflow."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, cast

import numpy as np
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cv_basics.api import (
    AppConfig,
    EffectName,
    ImageArray,
    ImageProcessingServicePort,
    ProcessingParams,
    create_image_processing_service,
)
from vision_workbench.sample_data import sample_image_path
from vision_workbench.troubleshooting import DATA_AND_FILES, MODULE_RUNTIME_ERRORS, with_help

from ..image_presenter import QtImagePresenter
from ..task_runner import QtTaskRunner
from ..widgets import NoWheelComboBox as QComboBox
from ..widgets import ParameterSlider, PreviewPanel, SectionCard, make_button


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskTimings:
    load_ms: float = 0.0
    process_ms: float = 0.0
    save_ms: float = 0.0
    preview_ms: float = 0.0

    def with_preview(self, preview_ms: float) -> "TaskTimings":
        return TaskTimings(
            load_ms=self.load_ms,
            process_ms=self.process_ms,
            save_ms=self.save_ms,
            preview_ms=preview_ms,
        )

    def summary(self) -> str:
        parts = []
        if self.load_ms:
            parts.append(f"load {self.load_ms:.1f} ms")
        if self.process_ms:
            parts.append(f"process {self.process_ms:.1f} ms")
        if self.save_ms:
            parts.append(f"save {self.save_ms:.1f} ms")
        if self.preview_ms:
            parts.append(f"preview {self.preview_ms:.1f} ms")
        return " | ".join(parts) if parts else "timing unavailable"


@dataclass(frozen=True)
class EffectResultPayload:
    original: ImageArray
    result: ImageArray
    timings: TaskTimings


@dataclass(frozen=True)
class SaveResultPayload:
    image: ImageArray
    timings: TaskTimings


EFFECT_LABELS = [
    ("灰度图", EffectName.GRAYSCALE),
    ("模糊", EffectName.BLUR),
    ("边缘检测", EffectName.EDGES),
    ("阈值分割", EffectName.THRESHOLD),
    ("卡通效果", EffectName.CARTOON),
    ("RGB 色彩空间", EffectName.RGB_SPACE),
    ("HSV 色彩空间", EffectName.HSV_SPACE),
    ("红色通道", EffectName.RED_CHANNEL),
    ("绿色通道", EffectName.GREEN_CHANNEL),
    ("蓝色通道", EffectName.BLUE_CHANNEL),
    ("色相通道", EffectName.HUE_CHANNEL),
    ("饱和度通道", EffectName.SATURATION_CHANNEL),
    ("明度通道", EffectName.VALUE_CHANNEL),
    ("灰度直方图", EffectName.GRAY_HISTOGRAM),
    ("RGB 直方图", EffectName.RGB_HISTOGRAM),
    ("腐蚀", EffectName.ERODE),
    ("膨胀", EffectName.DILATE),
    ("开运算", EffectName.MORPH_OPEN),
    ("闭运算", EffectName.MORPH_CLOSE),
    ("旋转", EffectName.ROTATE),
    ("缩放", EffectName.SCALE),
    ("中心裁剪", EffectName.CENTER_CROP),
    ("透视变换", EffectName.PERSPECTIVE_WARP),
]

EFFECT_ZH_TO_NAME = dict(EFFECT_LABELS)
EFFECT_NAME_TO_ZH = {value: key for key, value in EFFECT_LABELS}

PARAMETERS_BY_EFFECT = {
    EffectName.BLUR: ("blur",),
    EffectName.EDGES: ("edge_low", "edge_high"),
    EffectName.THRESHOLD: ("threshold",),
    EffectName.ERODE: ("morph_kernel", "morph_iterations"),
    EffectName.DILATE: ("morph_kernel", "morph_iterations"),
    EffectName.MORPH_OPEN: ("morph_kernel", "morph_iterations"),
    EffectName.MORPH_CLOSE: ("morph_kernel", "morph_iterations"),
    EffectName.ROTATE: ("rotate",),
    EffectName.SCALE: ("scale",),
    EffectName.CENTER_CROP: ("crop",),
    EffectName.PERSPECTIVE_WARP: ("perspective",),
}


class CvBasicsPage(QWidget):
    """Native Qt implementation of the CV Basics workflow."""

    status_changed = Signal(str)

    def __init__(
        self,
        service: Optional[ImageProcessingServicePort] = None,
        config: AppConfig = AppConfig(),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.service = service or create_image_processing_service(config)
        self.presenter = QtImagePresenter(config.preview_size)
        self.tasks = QtTaskRunner(self)

        self.original_image = None  # type: Optional[ImageArray]
        self.result_image = None  # type: Optional[ImageArray]
        self.current_path = None  # type: Optional[Path]
        self.current_preview_size = None  # type: Optional[tuple[int, int]]
        self.last_timing_text = None  # type: Optional[str]
        self._completion_status = None  # type: Optional[str]
        self._busy = False

        defaults = config.processing_defaults
        self.effect_combo = QComboBox()
        self.effect_combo.setToolTip("选择要应用的图像处理效果")
        self.blur_slider = ParameterSlider("模糊核", 1, 31, defaults.blur_kernel)
        self.threshold_slider = ParameterSlider("阈值", 0, 255, defaults.threshold)
        self.edge_low_slider = ParameterSlider("边缘低阈值", 0, 254, defaults.edge_low)
        self.edge_high_slider = ParameterSlider("边缘高阈值", 1, 255, defaults.edge_high)
        self.morph_kernel_slider = ParameterSlider("形态学核", 1, 31, defaults.morphology_kernel)
        self.morph_iterations_slider = ParameterSlider("迭代次数", 1, 5, defaults.morphology_iterations)
        self.rotate_slider = ParameterSlider("旋转角度", -180, 180, defaults.rotate_angle)
        self.scale_slider = ParameterSlider("缩放比例", 10, 200, defaults.scale_percent)
        self.crop_slider = ParameterSlider("裁剪比例", 10, 100, defaults.crop_percent)
        self.perspective_slider = ParameterSlider("透视偏移", 0, 40, defaults.perspective_shift)
        self.parameter_widgets = {
            "blur": self.blur_slider,
            "threshold": self.threshold_slider,
            "edge_low": self.edge_low_slider,
            "edge_high": self.edge_high_slider,
            "morph_kernel": self.morph_kernel_slider,
            "morph_iterations": self.morph_iterations_slider,
            "rotate": self.rotate_slider,
            "scale": self.scale_slider,
            "crop": self.crop_slider,
            "perspective": self.perspective_slider,
        }
        self.parameter_hint = QLabel("该效果无需额外参数，直接应用即可。")
        self.parameter_hint.setObjectName("ParameterHint")
        self.parameter_hint.setWordWrap(True)

        self.original_preview = PreviewPanel("原图", "请打开一张图片")
        self.result_preview = PreviewPanel("结果", "处理结果会显示在这里")
        self.info_label = QLabel("")
        self.info_label.setObjectName("MutedText")
        self.info_label.setWordWrap(True)

        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setObjectName("PageScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        content = QWidget()
        content.setObjectName("PageContent")
        content.setMinimumWidth(760)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 22, 24, 18)
        layout.setSpacing(16)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("基础图像处理")
        title.setObjectName("PageTitle")
        subtitle = QLabel("打开图像，调整参数，并预览传统 OpenCV 处理结果。")
        subtitle.setObjectName("PageSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block, 1)
        layout.addLayout(header)

        controls_card = SectionCard("操作")
        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.open_button = make_button("打开图片", primary=True)
        self.sample_button = make_button("加载示例图")
        self.save_button = make_button("保存结果")
        self.reset_button = make_button("重置")
        self.apply_button = make_button("应用效果", primary=True)
        self.open_button.setToolTip("选择本地图片并载入预览")
        self.sample_button.setToolTip("加载内置测试图，立即体验图像处理流程")
        self.save_button.setToolTip("保存当前处理结果")
        self.reset_button.setToolTip("恢复为原图")
        self.apply_button.setToolTip("使用当前参数处理图片")
        self.open_button.clicked.connect(self.open_image)
        self.sample_button.clicked.connect(self.load_sample_image)
        self.save_button.clicked.connect(self.save_result)
        self.reset_button.clicked.connect(self.reset_result)
        self.apply_button.clicked.connect(self.apply_effect)
        controls.addWidget(self.open_button)
        controls.addWidget(self.sample_button)
        controls.addWidget(self.save_button)
        controls.addWidget(self.reset_button)
        controls.addStretch(1)
        controls.addWidget(self.apply_button)
        controls_card.content_layout.addLayout(controls)
        layout.addWidget(controls_card)

        params_card = SectionCard("参数")
        params_grid = QGridLayout()
        params_grid.setHorizontalSpacing(18)
        params_grid.setVerticalSpacing(12)

        effect_label = QLabel("效果")
        effect_label.setMinimumWidth(76)
        effect_label.setBuddy(self.effect_combo)
        self.effect_combo.setAccessibleName("效果")
        for label, effect_name in EFFECT_LABELS:
            self.effect_combo.addItem(label, effect_name)
        default_label = EFFECT_NAME_TO_ZH.get(self.config.default_effect)
        if default_label:
            self.effect_combo.setCurrentText(default_label)
        self.effect_combo.currentIndexChanged.connect(self._update_parameter_visibility)
        params_grid.addWidget(effect_label, 0, 0)
        params_grid.addWidget(self.effect_combo, 0, 1, 1, 3)
        params_grid.addWidget(self.parameter_hint, 1, 0, 1, 4)
        params_grid.addWidget(self.blur_slider, 2, 0, 1, 2)
        params_grid.addWidget(self.threshold_slider, 2, 2, 1, 2)
        params_grid.addWidget(self.edge_low_slider, 3, 0, 1, 2)
        params_grid.addWidget(self.edge_high_slider, 3, 2, 1, 2)
        params_grid.addWidget(self.morph_kernel_slider, 4, 0, 1, 2)
        params_grid.addWidget(self.morph_iterations_slider, 4, 2, 1, 2)
        params_grid.addWidget(self.rotate_slider, 5, 0, 1, 2)
        params_grid.addWidget(self.scale_slider, 5, 2, 1, 2)
        params_grid.addWidget(self.crop_slider, 6, 0, 1, 2)
        params_grid.addWidget(self.perspective_slider, 6, 2, 1, 2)
        params_grid.setColumnStretch(1, 1)
        params_grid.setColumnStretch(3, 1)
        params_card.content_layout.addLayout(params_grid)
        layout.addWidget(params_card)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.original_preview)
        splitter.addWidget(self.result_preview)
        splitter.setSizes([1, 1])
        splitter.setMinimumHeight(420)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.info_label)

        self.scroll_area.setWidget(content)
        root_layout.addWidget(self.scroll_area)
        self._update_parameter_visibility()
        self._update_action_states()

    def open_image(self) -> None:
        patterns = " ".join(self.config.supported_extensions)
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "打开图片",
            "",
            f"图像文件 ({patterns});;所有文件 (*.*)",
        )
        if not path:
            return

        self._open_preview(Path(path))


    def load_sample_image(self) -> None:
        self._open_preview(sample_image_path())

    def save_result(self) -> None:
        if not self._has_open_image():
            QMessageBox.information(self, "没有结果", "请先打开一张图片。")
            return

        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存结果",
            "",
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg *.jpeg);;位图 (*.bmp);;所有文件 (*.*)",
        )
        if not path:
            return

        image = self.result_image if self.result_image is not None else self.original_image
        if image is not None:
            self._run_task(
                task=lambda: self._save_image(image, path),
                on_success=lambda payload: self._on_image_saved(payload, path),
                busy_text="正在保存图片...",
                error_title="保存失败",
                error_category=DATA_AND_FILES,
            )
            return

        image_path = self.current_path
        if image_path is None:
            QMessageBox.information(self, "没有结果", "请先打开一张图片。")
            return

        self._run_task(
            task=lambda: self._load_and_save_original(image_path, path),
            on_success=lambda image: self._on_original_saved(image, path),
            busy_text="正在读取并保存图片...",
            error_title="保存失败",
            error_category=DATA_AND_FILES,
        )

    def reset_result(self) -> None:
        if not self._has_open_image():
            QMessageBox.information(self, "没有图片", "请先打开一张图片。")
            return

        if self.original_image is None:
            self.result_image = None
            if self.current_path is not None:
                try:
                    self.result_preview.set_pixmap(self._pixmap_from_path(self.current_path))
                except ValueError as exc:
                    QMessageBox.critical(self, "打开失败", with_help(exc, DATA_AND_FILES))
                    return
        else:
            self.result_image = self.original_image
            self._show_result(self.result_image)
        self.last_timing_text = None
        self._update_info()
        self._set_status("已重置为原图。")

    def apply_effect(self) -> None:
        if not self._has_open_image():
            QMessageBox.information(self, "没有图片", "请先打开一张图片。")
            return

        effect = self.current_effect_name()
        params = self.current_params()
        label = EFFECT_NAME_TO_ZH.get(effect, effect)
        image_path = self.current_path
        cached_image = self.original_image
        self._run_task(
            task=lambda: self._load_and_apply_effect(cached_image, image_path, effect, params),
            on_success=self._on_effect_applied,
            busy_text=f"正在应用：{label}...",
            error_title="处理失败",
            error_category=MODULE_RUNTIME_ERRORS,
        )

    def current_effect_name(self) -> str:
        return str(self.effect_combo.currentData())

    def current_params(self) -> ProcessingParams:
        return ProcessingParams(
            blur_kernel=self.blur_slider.value(),
            edge_low=self.edge_low_slider.value(),
            edge_high=self.edge_high_slider.value(),
            threshold=self.threshold_slider.value(),
            morphology_kernel=self.morph_kernel_slider.value(),
            morphology_iterations=self.morph_iterations_slider.value(),
            rotate_angle=self.rotate_slider.value(),
            scale_percent=self.scale_slider.value(),
            crop_percent=self.crop_slider.value(),
            perspective_shift=self.perspective_slider.value(),
        )

    def shutdown(self) -> None:
        self.tasks.shutdown()

    def _run_task(
        self,
        task: Callable[[], object],
        on_success: Callable[[object], None],
        busy_text: str,
        error_title: str,
        error_category: str,
    ) -> None:
        self._completion_status = None
        accepted = self.tasks.run(
            task=task,
            on_success=lambda value: self._task_success(value, on_success),
            on_error=lambda exc: self._task_error(exc, error_title, error_category),
        )
        if not accepted:
            QMessageBox.information(self, "正在处理", "请等待当前操作完成。")
            return
        self._set_busy(True)
        self._set_status(busy_text)

    def _open_preview(self, image_path: Path) -> None:
        try:
            started_at = time.perf_counter()
            preview = self._pixmap_from_path(image_path)
            preview_ms = _elapsed_ms(started_at)
        except ValueError as exc:
            QMessageBox.critical(self, "打开失败", with_help(exc, DATA_AND_FILES))
            return

        self.current_path = image_path
        self.current_preview_size = (preview.width(), preview.height())
        self.original_image = None
        self.result_image = None
        self.last_timing_text = f"preview open {preview_ms:.1f} ms"
        self.original_preview.set_pixmap(preview)
        self.result_preview.set_pixmap(preview)
        self._update_info()
        self._update_action_states()
        self._set_status(f"Preview ready ({self.last_timing_text}); image data loads on apply.")

    def _pixmap_from_path(self, image_path: Path) -> QPixmap:
        preview = QPixmap(str(image_path))
        if preview.isNull():
            raise ValueError(f"Cannot preview image file: {image_path}")
        return preview

    def _save_image(self, image: ImageArray, output_path: str) -> SaveResultPayload:
        save_started_at = time.perf_counter()
        self.service.save_image(image, output_path)
        save_ms = _elapsed_ms(save_started_at)
        return SaveResultPayload(image=image, timings=TaskTimings(save_ms=save_ms))

    def _load_and_save_original(self, image_path: Path, output_path: str) -> SaveResultPayload:
        load_started_at = time.perf_counter()
        image = self.service.load_image(image_path)
        load_ms = _elapsed_ms(load_started_at)
        save_started_at = time.perf_counter()
        self.service.save_image(image, output_path)
        save_ms = _elapsed_ms(save_started_at)
        return SaveResultPayload(image=image, timings=TaskTimings(load_ms=load_ms, save_ms=save_ms))

    def _load_and_apply_effect(
        self,
        cached_image: ImageArray | None,
        image_path: Path | None,
        effect: str,
        params: ProcessingParams,
    ) -> EffectResultPayload:
        original = cached_image
        load_ms = 0.0
        if original is None:
            if image_path is None:
                raise ValueError("No image path is available.")
            load_started_at = time.perf_counter()
            original = self.service.load_image(image_path)
            load_ms = _elapsed_ms(load_started_at)
        process_started_at = time.perf_counter()
        result = self.service.apply_effect(original, effect, params)
        process_ms = _elapsed_ms(process_started_at)
        return EffectResultPayload(
            original=original,
            result=result,
            timings=TaskTimings(load_ms=load_ms, process_ms=process_ms),
        )

    def _task_success(self, value: object, callback: Callable[[object], None]) -> None:
        try:
            callback(value)
        finally:
            self._set_busy(False)
            self._set_status(self._completion_status or "就绪。")
            self._completion_status = None

    def _task_error(self, exc: Exception, title: str, category: str) -> None:
        self._set_busy(False)
        self._set_status("就绪。")
        QMessageBox.critical(self, title, with_help(exc, category))

    def _on_original_saved(self, payload: object, output_path: str) -> None:
        save_payload = cast(SaveResultPayload, payload)
        self.original_image = save_payload.image
        self.last_timing_text = save_payload.timings.summary()
        self._completion_status = f"Saved ({self.last_timing_text})."
        logger.info("CV Basics save timings: %s", self.last_timing_text)
        self._update_info()
        QMessageBox.information(self, "保存完成", f"结果已保存到：\n{output_path}")

    def _on_image_saved(self, payload: object, output_path: str) -> None:
        save_payload = cast(SaveResultPayload, payload)
        if self.original_image is None:
            self.original_image = save_payload.image
        self.last_timing_text = save_payload.timings.summary()
        self._completion_status = f"Saved ({self.last_timing_text})."
        logger.info("CV Basics save timings: %s", self.last_timing_text)
        self._update_info()
        QMessageBox.information(self, "保存完成", f"结果已保存到：\n{output_path}")

    def _on_effect_applied(self, payload: object) -> None:
        result_payload = cast(EffectResultPayload, payload)
        self.original_image = result_payload.original
        self.result_image = result_payload.result
        preview_started_at = time.perf_counter()
        self._show_result(self.result_image)
        timings = result_payload.timings.with_preview(_elapsed_ms(preview_started_at))
        self.last_timing_text = timings.summary()
        self._completion_status = f"Done ({self.last_timing_text})."
        logger.info("CV Basics effect timings: %s", self.last_timing_text)
        self._update_info()
        self._update_action_states()

    def _show_original(self, image: ImageArray) -> None:
        self.original_preview.set_pixmap(self.presenter.to_pixmap(image))

    def _show_result(self, image: ImageArray) -> None:
        self.result_preview.set_pixmap(self.presenter.to_pixmap(image))

    def _update_info(self) -> None:
        if self.result_image is None and self.original_image is None and self.current_path is None:
            self.info_label.setText("")
            return

        source = str(self.current_path) if self.current_path else "内存"
        image = self.result_image if self.result_image is not None else self.original_image
        if image is None:
            size = self.current_preview_size
            parts = [f"来源：{source}", "处理数据：点击应用效果时加载"]
            if size is not None:
                parts.insert(1, f"预览尺寸：{size[0]}x{size[1]}")
            if self.last_timing_text:
                parts.append(self.last_timing_text)
            self.info_label.setText(" | ".join(parts))
            return

        array = np.asarray(image)
        height, width = array.shape[:2]
        channels = 1 if array.ndim == 2 else array.shape[2]
        self.info_label.setText(
            " | ".join(
                [
                    f"来源：{source}",
                    f"尺寸：{width}x{height}",
                    f"通道：{channels}",
                    f"类型：{array.dtype}",
                ]
            )
        )
        if self.last_timing_text:
            self.info_label.setText(f"{self.info_label.text()} | {self.last_timing_text}")

    def _set_status(self, text: str) -> None:
        self.status_changed.emit(text)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_action_states()

    def _has_open_image(self) -> bool:
        return self.current_path is not None or self.original_image is not None

    def _has_displayable_result(self) -> bool:
        return self.result_image is not None or self.original_image is not None or self.current_path is not None

    def _update_action_states(self) -> None:
        self.open_button.setEnabled(not self._busy)
        self.sample_button.setEnabled(not self._busy)
        self.save_button.setEnabled(self._has_displayable_result() and not self._busy)
        self.reset_button.setEnabled(self._has_open_image() and not self._busy)
        self.apply_button.setEnabled(self._has_open_image() and not self._busy)

    def _update_parameter_visibility(self, _index: int | None = None) -> None:
        active_keys = set(PARAMETERS_BY_EFFECT.get(self.current_effect_name(), ()))
        for key, widget in self.parameter_widgets.items():
            widget.setVisible(key in active_keys)
        self.parameter_hint.setVisible(not active_keys)


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0
