"""Qt page for YOLO26 segmentation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, cast

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from vision_workbench.model_files import model_file_issue
from vision_workbench.troubleshooting import DATA_AND_FILES, MODELS_AND_WEIGHTS, MODULE_RUNTIME_ERRORS, with_help
from yolo26_segmentation.api import create_yolo26_segmentation_service
from yolo26_segmentation.application import Yolo26SegmentationService
from yolo26_segmentation.configuration import Yolo26SegmentationConfig
from yolo26_segmentation.domain import ImageArray, ModelInfo, SegmentationOutput, SegmentationSettings

from ..image_presenter import QtImagePresenter
from ..task_runner import QtTaskRunner
from ..widgets import SELECTED_DISPLAY_ROLE
from ..widgets import NoWheelComboBox as QComboBox
from ..widgets import NoWheelDoubleSpinBox as QDoubleSpinBox
from ..widgets import NoWheelSpinBox as QSpinBox
from ..widgets import PreviewPanel, SectionCard, make_button, set_download_progress, style_form_label


logger = logging.getLogger(__name__)


TASK_LABELS = {
    "segment": "实例分割",
    "semantic": "语义分割",
}
TASK_BY_LABEL = {label: task for task, label in TASK_LABELS.items()}


@dataclass(frozen=True)
class ImageLoadPayload:
    path: Path
    image: ImageArray
    total_ms: float


@dataclass(frozen=True)
class ModelPayload:
    model: ModelInfo
    total_ms: float


@dataclass(frozen=True)
class SegmentationPayload:
    output: SegmentationOutput
    image: ImageArray
    model: ModelInfo
    total_ms: float
    loaded_model: bool


@dataclass(frozen=True)
class SaveSegmentationPayload:
    path: Path
    total_ms: float


class YoloSegmentationPage(QWidget):
    """Native Qt implementation of single-image YOLO26 segmentation."""

    status_changed = Signal(str)
    download_progress_changed = Signal(object)

    def __init__(
        self,
        service: Optional[Yolo26SegmentationService] = None,
        config: Yolo26SegmentationConfig = Yolo26SegmentationConfig(),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.service = service or create_yolo26_segmentation_service(config)
        self.presenter = QtImagePresenter(config.preview_size)
        self.tasks = QtTaskRunner(self)

        self.models = []  # type: list[ModelInfo]
        self.image_path = None  # type: Optional[Path]
        self.source_image = None  # type: Optional[ImageArray]
        self.result_image = None  # type: Optional[ImageArray]
        self.last_timing_text = None  # type: Optional[str]
        self._loaded_model_path = None  # type: Optional[Path]
        self._busy = False
        self._completion_status = None  # type: Optional[str]

        self.task_label = QLabel("任务")
        self.task_combo = QComboBox()
        for task in config.task_options:
            self.task_combo.addItem(TASK_LABELS.get(task, task), task)
        self.task_combo.setMinimumWidth(132)

        self.model_label = QLabel("模型")
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(180)
        self.model_combo.setToolTip("选择本地或官方 YOLO26 分割模型")

        self.refresh_models_button = make_button("查找模型")
        self.browse_model_button = make_button("选择模型文件")
        self.download_model_button = make_button("下载所选模型")

        self.select_image_button = make_button("选择图片", primary=True)
        self.segment_button = make_button("分割图片", primary=True)
        self.save_result_button = make_button("保存分割图")

        self.device_label = QLabel("设备")
        self.device_combo = QComboBox()
        self.device_combo.addItems(config.device_options)
        self.device_combo.setCurrentText("auto")
        self.device_combo.setMinimumWidth(96)

        self.image_size_label = QLabel("尺寸")
        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(128, 4096)
        self.image_size_spin.setSingleStep(32)
        self.image_size_spin.setValue(config.default_image_size)
        self.image_size_spin.setMinimumWidth(88)

        self.conf_label = QLabel("置信度")
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(config.default_confidence)
        self.conf_spin.setMinimumWidth(82)

        self.iou_label = QLabel("IoU")
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 0.99)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(config.default_iou)
        self.iou_spin.setMinimumWidth(82)
        for label in (
            self.task_label,
            self.model_label,
            self.device_label,
            self.image_size_label,
            self.conf_label,
            self.iou_label,
        ):
            style_form_label(label)

        for button in (
            self.refresh_models_button,
            self.browse_model_button,
            self.download_model_button,
            self.select_image_button,
            self.segment_button,
            self.save_result_button,
        ):
            button.setMinimumWidth(112)

        self.model_status_label = QLabel("")
        self.model_status_label.setObjectName("MutedText")
        self.model_status_label.setWordWrap(False)
        self.model_status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.busy_label = QLabel("")
        self.busy_label.setObjectName("BusyNotice")
        self.busy_label.setWordWrap(True)
        self.busy_label.setVisible(False)
        self.busy_progress = QProgressBar()
        self.busy_progress.setObjectName("InlineBusyProgress")
        self.busy_progress.setRange(0, 0)
        self.busy_progress.setFixedHeight(6)
        self.busy_progress.setTextVisible(False)
        self.busy_progress.setVisible(False)

        self.preview_panel = PreviewPanel("分割预览", "选择图片后显示画面")
        self.items_list = QListWidget()
        self.items_list.setObjectName("SegmentationItems")
        self.items_list.setAlternatingRowColors(True)
        self.items_list.setMinimumHeight(160)

        self.info_label = QLabel("")
        self.info_label.setObjectName("MutedText")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.model_layout = None  # type: QGridLayout | None
        self.controls_layout = None  # type: QGridLayout | None
        self._compact_layout = None  # type: bool | None

        self._build_ui()
        self._connect_signals()
        self.refresh_models()
        self._update_action_states()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        scroll_area = QScrollArea(self)
        scroll_area.setObjectName("PageScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        content = QWidget()
        content.setObjectName("PageContent")
        content.setMinimumWidth(760)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 22, 24, 18)
        layout.setSpacing(16)

        title = QLabel("YOLO 分割")
        title.setObjectName("PageTitle")
        subtitle = QLabel("选择 YOLO26 分割模型，对单张图片执行实例或语义分割。")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        model_card = SectionCard("模型")
        self.model_layout = QGridLayout()
        self.model_layout.setHorizontalSpacing(10)
        self.model_layout.setVerticalSpacing(10)
        model_card.content_layout.addLayout(self.model_layout)
        layout.addWidget(model_card)

        controls_card = SectionCard("操作")
        self.controls_layout = QGridLayout()
        self.controls_layout.setHorizontalSpacing(10)
        self.controls_layout.setVerticalSpacing(10)
        controls_card.content_layout.addLayout(self.controls_layout)
        layout.addWidget(controls_card)
        layout.addWidget(self.busy_label)
        layout.addWidget(self.busy_progress)

        self.preview_panel.setMinimumHeight(460)
        layout.addWidget(self.preview_panel, 1)

        items_card = SectionCard("分割结果")
        items_card.content_layout.addWidget(self.items_list)
        layout.addWidget(items_card)
        layout.addWidget(self.info_label)

        scroll_area.setWidget(content)
        root_layout.addWidget(scroll_area)
        self._apply_responsive_layout(force=True)

    def _connect_signals(self) -> None:
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.refresh_models_button.clicked.connect(self.refresh_models)
        self.browse_model_button.clicked.connect(self.browse_model)
        self.download_model_button.clicked.connect(self.download_selected_model)
        self.select_image_button.clicked.connect(self.select_image)
        self.segment_button.clicked.connect(self.segment_image)
        self.save_result_button.clicked.connect(self.save_result)
        self.download_progress_changed.connect(self._on_download_progress)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def _apply_responsive_layout(self, *, force: bool = False) -> None:
        model_layout = self.model_layout
        controls = self.controls_layout
        if model_layout is None or controls is None:
            return

        compact = self.width() < 1180
        if not force and compact == self._compact_layout:
            return
        self._compact_layout = compact

        self._reset_grid(model_layout, 8)
        self._reset_grid(controls, 8)
        if compact:
            self._layout_compact_model(model_layout)
            self._layout_compact_controls(controls)
            self.preview_panel.setMinimumHeight(360)
        else:
            self._layout_wide_model(model_layout)
            self._layout_wide_controls(controls)
            self.preview_panel.setMinimumHeight(460)

    def _layout_wide_model(self, layout: QGridLayout) -> None:
        layout.addWidget(self.task_label, 0, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.task_combo, 0, 1)
        layout.addWidget(self.model_label, 0, 2, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.model_combo, 0, 3, 1, 2)
        layout.addWidget(self.browse_model_button, 0, 5)
        layout.addWidget(self.download_model_button, 0, 6)
        layout.addWidget(self.refresh_models_button, 0, 7)
        layout.addWidget(self.model_status_label, 1, 1, 1, 7)
        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(4, 1)

    def _layout_compact_model(self, layout: QGridLayout) -> None:
        layout.addWidget(self.task_label, 0, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.task_combo, 0, 1, 1, 3)
        layout.addWidget(self.model_label, 1, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.model_combo, 1, 1, 1, 3)
        layout.addWidget(self.browse_model_button, 2, 0)
        layout.addWidget(self.download_model_button, 2, 1, 1, 2)
        layout.addWidget(self.refresh_models_button, 2, 3)
        layout.addWidget(self.model_status_label, 3, 0, 1, 4)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)

    def _layout_wide_controls(self, controls: QGridLayout) -> None:
        controls.addWidget(self.select_image_button, 0, 0)
        controls.addWidget(self.segment_button, 0, 1, 1, 3)
        controls.addWidget(self.save_result_button, 0, 4)
        controls.addWidget(self.device_label, 1, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.device_combo, 1, 1)
        controls.addWidget(self.image_size_label, 1, 2, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.image_size_spin, 1, 3)
        controls.addWidget(self.conf_label, 1, 4, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.conf_spin, 1, 5)
        controls.addWidget(self.iou_label, 1, 6, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.iou_spin, 1, 7)
        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(3, 1)

    def _layout_compact_controls(self, controls: QGridLayout) -> None:
        controls.addWidget(self.select_image_button, 0, 0)
        controls.addWidget(self.segment_button, 0, 1, 1, 2)
        controls.addWidget(self.save_result_button, 0, 3)
        controls.addWidget(self.device_label, 1, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.device_combo, 1, 1)
        controls.addWidget(self.image_size_label, 1, 2, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.image_size_spin, 1, 3)
        controls.addWidget(self.conf_label, 2, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.conf_spin, 2, 1)
        controls.addWidget(self.iou_label, 2, 2, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.iou_spin, 2, 3)
        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(3, 1)

    def _reset_grid(self, grid: QGridLayout, columns: int) -> None:
        while grid.count():
            item = grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        for index in range(columns):
            grid.setColumnStretch(index, 0)
            grid.setColumnMinimumWidth(index, 0)
        for index in range(6):
            grid.setRowStretch(index, 0)

    def refresh_models(self) -> None:
        task = self.current_task()
        current = self.selected_model()
        current_path = current.path if current is not None else None
        try:
            self.models = list(self.service.list_models(task, include_missing_official=True))
        except Exception as exc:
            self.models = []
            self.model_combo.clear()
            self.model_status_label.setText(f"无法读取模型列表：{exc}")
            self._update_action_states()
            return

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for model in self.models:
            self.model_combo.addItem(_model_combo_label(model))
            index = self.model_combo.count() - 1
            self.model_combo.setItemData(index, model.name, SELECTED_DISPLAY_ROLE)
            self.model_combo.setItemData(index, str(model.path), Qt.ItemDataRole.ToolTipRole)
        self.model_combo.blockSignals(False)
        if self.models:
            self.model_combo.setCurrentIndex(_preferred_model_index(self.models, current_path))
        self._show_model_status()
        self._update_info()
        self._update_action_states()

    def browse_model(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择 YOLO26 分割模型",
            "",
            "PyTorch 模型 (*.pt);;所有文件 (*.*)",
        )
        if not path:
            return
        try:
            model = self.service.add_custom_model(path, self.current_task())
        except Exception as exc:
            QMessageBox.critical(self, "模型失败", with_help(exc, MODELS_AND_WEIGHTS))
            return
        self.models.append(model)
        self.model_combo.addItem(_model_combo_label(model))
        index = self.model_combo.count() - 1
        self.model_combo.setItemData(index, model.name, SELECTED_DISPLAY_ROLE)
        self.model_combo.setItemData(index, str(model.path), Qt.ItemDataRole.ToolTipRole)
        self.model_combo.setCurrentIndex(index)
        self._show_model_status()
        self._update_action_states()
        self._set_status(f"已选择模型：{model.name}")

    def download_selected_model(self) -> None:
        model = self.selected_model()
        if model is None:
            QMessageBox.information(self, "没有模型", "请先选择模型。")
            return
        if not model.is_official:
            QMessageBox.information(self, "自定义模型", "这里只能下载官方模型。")
            return
        if model.exists:
            self._set_status(f"模型已存在：{model.path}")
            return
        self._run_task(
            task=lambda: self._download_model(model),
            on_success=self._on_model_downloaded,
            busy_text=f"正在下载 {model.name}...",
            progress_text=f"正在下载 {model.name}...\n保存路径：{model.path}",
            error_title="下载失败",
            error_category=MODELS_AND_WEIGHTS,
        )

    def select_image(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;所有文件 (*.*)",
        )
        if not path:
            return
        image_path = Path(path)
        self._run_task(
            task=lambda: self._load_image(image_path),
            on_success=self._on_image_loaded,
            busy_text="正在加载图片...",
            error_title="打开失败",
            error_category=DATA_AND_FILES,
        )

    def segment_image(self) -> None:
        model = self.selected_model()
        if self.source_image is None:
            QMessageBox.information(self, "没有图片", "请先选择图片。")
            return
        if model is None:
            QMessageBox.information(self, "没有模型", "请先选择模型。")
            return
        if not model.exists:
            QMessageBox.information(self, "模型缺失", "请先下载该模型，或选择本地 .pt 文件。")
            return

        image = self.source_image.copy()
        settings = self.current_settings()
        first_load = self._loaded_model_path != model.path
        self._run_task(
            task=lambda: self._segment_image(image, model, settings, first_load),
            on_success=self._on_segmentation_ready,
            busy_text=(
                f"首次加载 {model.name} 并分割，可能需要数秒..."
                if first_load
                else f"正在使用已加载的 {model.name} 分割..."
            ),
            error_title="分割失败",
            error_category=MODULE_RUNTIME_ERRORS,
        )

    def save_result(self) -> None:
        if self.result_image is None:
            QMessageBox.information(self, "没有结果", "请先完成一次分割。")
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存分割图",
            "",
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg *.jpeg);;位图 (*.bmp);;所有文件 (*.*)",
        )
        if not path:
            return
        result = self.result_image.copy()
        output_path = Path(path)
        self._run_task(
            task=lambda: self._save_result(result, output_path),
            on_success=self._on_result_saved,
            busy_text="正在保存分割图...",
            error_title="保存失败",
            error_category=DATA_AND_FILES,
        )

    def current_task(self) -> str:
        return str(self.task_combo.currentData() or "segment")

    def selected_model(self) -> ModelInfo | None:
        index = self.model_combo.currentIndex()
        if index < 0 or index >= len(self.models):
            return None
        return self.models[index]

    def current_settings(self) -> SegmentationSettings:
        return SegmentationSettings(
            task=self.current_task(),
            image_size=int(self.image_size_spin.value()),
            confidence=float(self.conf_spin.value()),
            iou=float(self.iou_spin.value()),
            device=self.device_combo.currentText(),
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
        progress_text: str | None = None,
    ) -> None:
        self._completion_status = None
        accepted = self.tasks.run(
            task=task,
            on_success=lambda value: self._task_success(value, on_success),
            on_error=lambda exc: self._task_error(exc, error_title, error_category),
        )
        if not accepted:
            QMessageBox.information(self, "正在处理", "请等待当前任务完成。")
            return
        self.busy_label.setText(progress_text or _busy_notice_text(busy_text))
        self.busy_label.setVisible(True)
        self.busy_progress.setVisible(progress_text is not None)
        self._set_busy(True)
        self._set_status(busy_text)

    def _download_model(self, model: ModelInfo) -> ModelPayload:
        started_at = time.perf_counter()
        downloaded = self.service.download_official_model(
            model.name,
            model.task,
            progress_callback=self._download_progress_callback(f"正在下载 {model.name}", model.path),
        )
        return ModelPayload(model=downloaded, total_ms=_elapsed_ms(started_at))

    def _load_image(self, image_path: Path) -> ImageLoadPayload:
        started_at = time.perf_counter()
        image = self.service.load_image(image_path)
        return ImageLoadPayload(path=image_path, image=image, total_ms=_elapsed_ms(started_at))

    def _segment_image(
        self,
        image: ImageArray,
        model: ModelInfo,
        settings: SegmentationSettings,
        first_load: bool,
    ) -> SegmentationPayload:
        started_at = time.perf_counter()
        if first_load:
            self.service.load_model(model.path)
        output = self.service.segment_image(image, settings)
        return SegmentationPayload(
            output=output,
            image=image,
            model=model,
            total_ms=_elapsed_ms(started_at),
            loaded_model=first_load,
        )

    def _save_result(self, image: ImageArray, path: Path) -> SaveSegmentationPayload:
        started_at = time.perf_counter()
        self.service.save_image(image, path)
        return SaveSegmentationPayload(path=path, total_ms=_elapsed_ms(started_at))

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

    def _on_model_downloaded(self, value: object) -> None:
        payload = cast(ModelPayload, value)
        self.refresh_models()
        for index, model in enumerate(self.models):
            if model.path == payload.model.path:
                self.model_combo.setCurrentIndex(index)
                break
        self.last_timing_text = f"下载 {payload.total_ms:.1f} ms"
        self._completion_status = f"模型已下载到 {payload.model.path}（{self.last_timing_text}）。"
        self._show_model_status()
        QMessageBox.information(self, "下载完成", f"模型已下载到：\n{payload.model.path}")

    def _download_progress_callback(self, action_text: str, path: Path):
        def callback(percent: int | None, downloaded_bytes: int, total_bytes: int | None) -> None:
            self.download_progress_changed.emit(
                (action_text, str(path), percent, downloaded_bytes, total_bytes)
            )

        return callback

    def _on_download_progress(self, value: object) -> None:
        action_text, path, percent, downloaded_bytes, total_bytes = cast(
            tuple[str, str, int | None, int, int | None],
            value,
        )
        self.busy_label.setVisible(True)
        self.busy_progress.setVisible(True)
        set_download_progress(
            self.busy_label,
            self.busy_progress,
            action_text,
            path,
            percent,
            downloaded_bytes,
            total_bytes,
        )

    def _on_image_loaded(self, value: object) -> None:
        payload = cast(ImageLoadPayload, value)
        self.image_path = payload.path
        self.source_image = payload.image
        self.result_image = None
        self.items_list.clear()
        self.preview_panel.set_pixmap(self.presenter.to_pixmap(payload.image))
        self.last_timing_text = f"加载图片 {payload.total_ms:.1f} ms"
        self._completion_status = f"图片已就绪（{self.last_timing_text}）。"
        self._update_info()
        self._update_action_states()

    def _on_segmentation_ready(self, value: object) -> None:
        payload = cast(SegmentationPayload, value)
        self._loaded_model_path = payload.model.path
        self.result_image = payload.output.annotated_frame
        self.preview_panel.set_pixmap(self.presenter.to_pixmap(self.result_image))
        self._show_items(payload.output)
        prepare_ms = max(0.0, payload.total_ms - payload.output.inference_ms)
        self.last_timing_text = (
            f"加载/准备 {prepare_ms:.1f} ms | "
            f"推理 {payload.output.inference_ms:.1f} ms | "
            f"总计 {payload.total_ms:.1f} ms"
        )
        cache_text = "模型已加载" if payload.loaded_model else "已复用模型"
        self._completion_status = f"分割完成，{cache_text}（{self.last_timing_text}）。"
        logger.info("YOLO segmentation timing: %s", self.last_timing_text)
        self._update_info()
        self._update_action_states()

    def _on_result_saved(self, value: object) -> None:
        payload = cast(SaveSegmentationPayload, value)
        self.last_timing_text = f"保存 {payload.total_ms:.1f} ms"
        self._completion_status = f"分割图已保存（{self.last_timing_text}）。"
        self._update_info()
        QMessageBox.information(self, "保存完成", f"分割图已保存到：\n{payload.path}")

    def _on_task_changed(self, _index: int) -> None:
        self.result_image = None
        self.items_list.clear()
        self._loaded_model_path = None
        self.refresh_models()
        self._restore_source_preview()

    def _on_model_changed(self, _index: int) -> None:
        self.result_image = None
        self.items_list.clear()
        self.last_timing_text = None
        self._restore_source_preview()
        self._show_model_status()
        self._update_info()
        self._update_action_states()

    def _show_items(self, output: SegmentationOutput) -> None:
        self.items_list.clear()
        if not output.names:
            item = QListWidgetItem(f"分割目标：{output.item_count}")
            self.items_list.addItem(item)
            return
        for index, name in enumerate(output.names, start=1):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(8, 6, 8, 6)
            layout.setSpacing(10)
            rank_label = QLabel(str(index))
            rank_label.setMinimumWidth(24)
            name_label = QLabel(name)
            layout.addWidget(rank_label)
            layout.addWidget(name_label, 1)
            item = QListWidgetItem()
            item.setSizeHint(row.sizeHint())
            self.items_list.addItem(item)
            self.items_list.setItemWidget(item, row)

    def _show_model_status(self) -> None:
        model = self.selected_model()
        if model is None:
            self.model_status_label.setText("模型：未找到")
            self.model_status_label.setToolTip("")
            return
        kind = "官方" if model.is_official else "自定义"
        state = _model_state_text(model)
        task = TASK_LABELS.get(model.task, model.task)
        self.model_status_label.setText(f"{kind}{task}模型：{state} | {_shorten_text(model.name, 28)}")
        self.model_status_label.setToolTip(str(model.path))

    def _restore_source_preview(self) -> None:
        if self.source_image is None:
            self.preview_panel.clear()
            return
        self.preview_panel.set_pixmap(self.presenter.to_pixmap(self.source_image))

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if not busy:
            self.busy_label.setVisible(False)
            self.busy_progress.setVisible(False)
        self._update_action_states()

    def _update_action_states(self) -> None:
        model = self.selected_model()
        has_model = model is not None and model.exists
        can_download = model is not None and model.is_official and not model.exists
        has_image = self.source_image is not None
        has_result = self.result_image is not None
        self.download_model_button.setText(_download_button_text(model))

        self.task_combo.setEnabled(not self._busy)
        self.model_combo.setEnabled(bool(self.models) and not self._busy)
        self.refresh_models_button.setEnabled(not self._busy)
        self.browse_model_button.setEnabled(not self._busy)
        self.download_model_button.setEnabled(can_download and not self._busy)
        self.select_image_button.setEnabled(not self._busy)
        self.segment_button.setEnabled(has_image and has_model and not self._busy)
        self.save_result_button.setEnabled(has_result and not self._busy)
        self.device_combo.setEnabled(not self._busy)
        self.image_size_spin.setEnabled(not self._busy)
        self.conf_spin.setEnabled(not self._busy)
        self.iou_spin.setEnabled(not self._busy)

    def _update_info(self) -> None:
        parts = []
        if self.image_path is not None:
            parts.append(f"图片：{_shorten_path(self.image_path, 68)}")
        model = self.selected_model()
        if model is not None:
            parts.append(f"模型：{model.name}")
        parts.append(f"任务：{self.task_combo.currentText()}")
        if self.result_image is not None:
            parts.append(f"分割结果：{self.items_list.count()} 项")
        if self.last_timing_text:
            parts.append(self.last_timing_text)
        self.info_label.setText(" | ".join(parts))
        if self.image_path is not None:
            self.info_label.setToolTip(str(self.image_path))

    def _set_status(self, text: str) -> None:
        self.status_changed.emit(text)


def _preferred_model_index(models: list[ModelInfo], current_path: Path | None) -> int:
    if current_path is not None:
        for index, model in enumerate(models):
            if model.path == current_path:
                return index
    for index, model in enumerate(models):
        if model.exists:
            return index
    return 0


def _model_combo_label(model: ModelInfo) -> str:
    if model.exists:
        return str(model.path)
    issue = model_file_issue(model.path)
    state = issue if issue is not None else "模型未下载"
    return f"{model.path}（{state}）"


def _model_state_text(model: ModelInfo) -> str:
    if model.exists:
        return "本地可用"
    issue = model_file_issue(model.path)
    return f"{issue}，可重新下载" if issue is not None else "未下载"


def _download_button_text(model: ModelInfo | None) -> str:
    if model is None or model.exists or model_file_issue(model.path) is None:
        return "下载所选模型"
    return "重新下载模型"


def _busy_notice_text(text: str) -> str:
    lower = text.lower()
    if "加载" in text and ("模型" in text or ".pt" in lower or "yolo" in lower):
        return f"{text} 窗口仍在工作，请稍等。"
    return text


def _shorten_path(path: Path, max_chars: int) -> str:
    text = str(path)
    if len(text) <= max_chars:
        return text
    head = max(12, max_chars // 2 - 3)
    tail = max(18, max_chars - head - 3)
    return f"{text[:head]}...{text[-tail:]}"


def _shorten_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0
