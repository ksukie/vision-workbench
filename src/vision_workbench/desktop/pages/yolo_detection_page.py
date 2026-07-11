"""Qt page for YOLO26 object detection."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, cast

import cv2
import numpy as np
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap
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

from vision_workbench.troubleshooting import (
    CAMERA_AND_VIDEO,
    DATA_AND_FILES,
    MODELS_AND_WEIGHTS,
    MODULE_RUNTIME_ERRORS,
    with_help,
)
from vision_workbench.model_files import model_file_issue
from vision_workbench.input_limits import validate_image_file
from yolo26_detection.api import create_yolo26_detection_service
from yolo26_detection.application import Yolo26DetectionService
from yolo26_detection.configuration import Yolo26DetectionConfig
from yolo26_detection.domain import CameraBackend, CameraDevice, DetectionOutput, DetectionSettings, ImageArray, ModelInfo

from ..camera_resource import CameraResourceCoordinator, shared_camera_coordinator
from ..image_presenter import QtImagePresenter
from ..task_runner import QtTaskRunner
from ..widgets import SELECTED_DISPLAY_ROLE
from ..widgets import NoWheelComboBox as QComboBox
from ..widgets import NoWheelDoubleSpinBox as QDoubleSpinBox
from ..widgets import NoWheelSpinBox as QSpinBox
from ..widgets import (
    PreviewPanel,
    SectionCard,
    associate_form_label,
    make_button,
    set_download_progress,
    style_form_label,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionPayload:
    output: DetectionOutput
    image: ImageArray
    model: ModelInfo
    total_ms: float
    loaded_model: bool


@dataclass(frozen=True)
class ModelPayload:
    model: ModelInfo
    total_ms: float


@dataclass(frozen=True)
class SaveDetectionPayload:
    path: Path
    total_ms: float


@dataclass(frozen=True)
class CameraScanPayload:
    cameras: list[CameraDevice]
    total_ms: float


@dataclass(frozen=True)
class ModelManifestPayload:
    entry_count: int
    total_ms: float


@dataclass(frozen=True)
class OpenCameraPayload:
    camera: CameraDevice
    total_ms: float


@dataclass(frozen=True)
class LiveDetectionStartPayload:
    model: ModelInfo
    settings: DetectionSettings
    total_ms: float
    loaded_model: bool


@dataclass(frozen=True)
class LiveFramePayload:
    frame: ImageArray
    output: DetectionOutput | None
    camera_fps: float
    inference_fps: float


@dataclass(frozen=True)
class LiveDetectionPayload:
    output: DetectionOutput
    inference_fps: float


class YoloDetectionPage(QWidget):
    """Native Qt implementation of YOLO26 image and camera detection."""

    status_changed = Signal(str)
    live_frame_ready = Signal(object)
    live_detection_ready = Signal(object)
    live_failed = Signal(object)
    download_progress_changed = Signal(object)

    def __init__(
        self,
        service: Optional[Yolo26DetectionService] = None,
        config: Yolo26DetectionConfig = Yolo26DetectionConfig(),
        camera_coordinator: CameraResourceCoordinator | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.service = service or create_yolo26_detection_service(config)
        self.presenter = QtImagePresenter(config.preview_size)
        self.tasks = QtTaskRunner(self)
        self.camera_coordinator = camera_coordinator or shared_camera_coordinator()
        self._camera_owner_id = "yolo_detection"
        self._camera_owner_label = "YOLO 检测"

        self.models = []  # type: list[ModelInfo]
        self.cameras = []  # type: list[CameraDevice]
        self.image_path = None  # type: Optional[Path]
        self.result_image = None  # type: Optional[ImageArray]
        self.last_timing_text = None  # type: Optional[str]
        self._busy = False
        self._completion_status = None  # type: Optional[str]
        self._loaded_model_path = self._service_loaded_model_path()
        self._camera_open = False
        self._live_detection_enabled = False
        self._live_camera_fps = 0.0
        self._live_inference_fps = 0.0
        self._live_detection_count = 0
        self._live_read_failures = 0
        self._last_live_output = None  # type: Optional[DetectionOutput]
        self._camera_lock = threading.Lock()
        self._live_settings_lock = threading.Lock()
        self._live_frame_condition = threading.Condition()
        self._latest_inference_frame = None  # type: Optional[ImageArray]
        self._latest_frame_sequence = 0
        self._live_settings = DetectionSettings(
            image_size=config.default_image_size,
            confidence=config.default_confidence,
            iou=config.default_iou,
            device="auto",
        )
        self._live_stop = None  # type: Optional[threading.Event]
        self._live_thread = None  # type: Optional[threading.Thread]
        self._inference_thread = None  # type: Optional[threading.Thread]

        self.open_image_button = make_button("选择图片", primary=True)
        self.detect_button = make_button("检测图片", primary=True)
        self.save_result_button = make_button("保存检测图")
        self.refresh_models_button = make_button("查找模型")
        self.browse_model_button = make_button("选择本地模型文件")
        self.download_model_button = make_button("下载所选模型")
        self.refresh_cameras_button = make_button("查找相机", primary=True)
        self.open_camera_button = make_button("打开相机", primary=True)
        self.close_camera_button = make_button("关闭相机")
        self.start_live_button = make_button("开始摄像头检测", primary=True)
        self.stop_live_button = make_button("停止摄像头检测")
        for button in (
            self.open_image_button,
            self.detect_button,
            self.save_result_button,
            self.refresh_models_button,
            self.browse_model_button,
            self.download_model_button,
            self.refresh_cameras_button,
            self.open_camera_button,
            self.close_camera_button,
            self.start_live_button,
            self.stop_live_button,
        ):
            button.setMinimumWidth(112)

        self.browse_model_button.setToolTip("从本地磁盘选择一个 .pt 模型文件")

        self.model_label = QLabel("模型")
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(180)
        self.model_combo.setToolTip("选择本地或官方 YOLO26 检测模型")

        self.device_label = QLabel("设备")
        self.device_combo = QComboBox()
        self.device_combo.addItems(config.device_options)
        self.device_combo.setCurrentText("auto")
        self.device_combo.setMinimumWidth(96)

        self.image_size_label = QLabel("尺寸")
        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(128, 2048)
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

        self.camera_label = QLabel("相机")
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(180)
        self.camera_combo.setToolTip("选择实时检测摄像头")
        for label in (
            self.model_label,
            self.device_label,
            self.image_size_label,
            self.conf_label,
            self.iou_label,
            self.camera_label,
        ):
            style_form_label(label)
        for label, control in (
            (self.model_label, self.model_combo),
            (self.device_label, self.device_combo),
            (self.image_size_label, self.image_size_spin),
            (self.conf_label, self.conf_spin),
            (self.iou_label, self.iou_spin),
            (self.camera_label, self.camera_combo),
        ):
            associate_form_label(label, control)

        self.live_status_label = QLabel("实时：未打开")
        self.live_status_label.setObjectName("MutedText")
        self.live_status_label.setWordWrap(True)

        self.model_status_label = QLabel("")
        self.model_status_label.setObjectName("MutedText")
        self.model_status_label.setWordWrap(False)
        self.model_status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.preview_panel = PreviewPanel("检测预览", "打开图片或相机后显示画面")
        self.original_preview = self.preview_panel
        self.result_preview = self.preview_panel
        self.results_list = QListWidget()
        self.results_list.setObjectName("DetectionList")
        self.results_list.setAlternatingRowColors(True)
        self.results_list.setMinimumHeight(240)

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

        self.info_label = QLabel("")
        self.info_label.setObjectName("MutedText")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.content = None  # type: QWidget | None
        self.controls_layout = None  # type: QGridLayout | None
        self.model_layout = None  # type: QGridLayout | None
        self.results_panel = None  # type: QWidget | None
        self._compact_layout = None  # type: bool | None

        self._build_ui()
        self._connect_signals()
        self.camera_coordinator.changed.connect(self._update_action_states)
        self.camera_coordinator.devices_changed.connect(self._apply_cached_cameras)
        self._apply_cached_cameras()
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
        self.content = content

        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 22, 24, 18)
        layout.setSpacing(16)

        title = QLabel("YOLO 检测")
        title.setObjectName("PageTitle")
        subtitle = QLabel("选择 YOLO26 模型，对图片或摄像头画面执行目标检测并查看结果。")
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

        self.results_panel = SectionCard("检测框")
        self.results_panel.content_layout.addWidget(self.results_list, 1)
        layout.addWidget(self.results_panel)
        layout.addWidget(self.info_label)

        scroll_area.setWidget(content)
        root_layout.addWidget(scroll_area)
        self._apply_responsive_layout(force=True)

    def _connect_signals(self) -> None:
        self.open_image_button.clicked.connect(self.open_image)
        self.detect_button.clicked.connect(self.detect_image)
        self.save_result_button.clicked.connect(self.save_result)
        self.refresh_models_button.clicked.connect(self.refresh_model_catalog)
        self.browse_model_button.clicked.connect(self.browse_model)
        self.download_model_button.clicked.connect(self.download_selected_model)
        self.refresh_cameras_button.clicked.connect(self.refresh_cameras)
        self.open_camera_button.clicked.connect(self.open_live_camera)
        self.close_camera_button.clicked.connect(self.close_live_camera)
        self.start_live_button.clicked.connect(self.start_live_detection)
        self.stop_live_button.clicked.connect(self.stop_live_detection)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.live_frame_ready.connect(self._on_live_frame_ready)
        self.live_detection_ready.connect(self._on_live_detection_ready)
        self.live_failed.connect(self._on_live_failed)
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
        layout.addWidget(self.model_label, 0, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.model_combo, 0, 1, 1, 2)
        layout.addWidget(self.browse_model_button, 0, 3)
        layout.addWidget(self.download_model_button, 0, 4)
        layout.addWidget(self.refresh_models_button, 0, 5)
        layout.addWidget(self.model_status_label, 1, 1, 1, 5)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(5, 1)

    def _layout_compact_model(self, layout: QGridLayout) -> None:
        layout.addWidget(self.model_label, 0, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.model_combo, 0, 1, 1, 3)
        layout.addWidget(self.refresh_models_button, 1, 0)
        layout.addWidget(self.browse_model_button, 1, 1)
        layout.addWidget(self.download_model_button, 1, 2, 1, 2)
        layout.addWidget(self.model_status_label, 2, 0, 1, 4)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)

    def _layout_wide_controls(self, controls: QGridLayout) -> None:
        controls.addWidget(self.open_image_button, 0, 0)
        controls.addWidget(self.detect_button, 0, 1, 1, 3)
        controls.addWidget(self.save_result_button, 0, 4)

        controls.addWidget(self.camera_label, 1, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.camera_combo, 1, 1, 1, 2)
        controls.addWidget(self.refresh_cameras_button, 1, 3)
        controls.addWidget(self.open_camera_button, 1, 4)
        controls.addWidget(self.start_live_button, 1, 5)
        controls.addWidget(self.stop_live_button, 1, 6)
        controls.addWidget(self.close_camera_button, 1, 7)

        controls.addWidget(self.device_label, 2, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.device_combo, 2, 1)
        controls.addWidget(self.image_size_label, 2, 2, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.image_size_spin, 2, 3)
        controls.addWidget(self.conf_label, 2, 4, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.conf_spin, 2, 5)
        controls.addWidget(self.iou_label, 2, 6, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.iou_spin, 2, 7)

        controls.addWidget(self.live_status_label, 3, 0, 1, 8)
        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(3, 1)

    def _layout_compact_controls(self, controls: QGridLayout) -> None:
        controls.addWidget(self.open_image_button, 0, 0)
        controls.addWidget(self.detect_button, 0, 1, 1, 2)
        controls.addWidget(self.save_result_button, 0, 3)

        controls.addWidget(self.camera_label, 1, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.camera_combo, 1, 1, 1, 3)
        controls.addWidget(self.refresh_cameras_button, 2, 0)
        controls.addWidget(self.open_camera_button, 2, 1)
        controls.addWidget(self.start_live_button, 2, 2, 1, 2)
        controls.addWidget(self.stop_live_button, 3, 0, 1, 2)
        controls.addWidget(self.close_camera_button, 3, 2, 1, 2)

        controls.addWidget(self.device_label, 4, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.device_combo, 4, 1)
        controls.addWidget(self.image_size_label, 4, 2, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.image_size_spin, 4, 3)
        controls.addWidget(self.conf_label, 5, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.conf_spin, 5, 1)
        controls.addWidget(self.iou_label, 5, 2, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.iou_spin, 5, 3)
        controls.addWidget(self.live_status_label, 6, 0, 1, 4)
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
        for index in range(8):
            grid.setRowStretch(index, 0)

    def refresh_models(self) -> None:
        try:
            self.models = list(self.service.list_models(include_missing_official=True))
        except Exception as exc:
            self.models = []
            self.model_combo.clear()
            self.model_status_label.setText(f"无法读取模型列表：{exc}")
            self._update_action_states()
            return

        current_path = self.selected_model().path if self.selected_model() is not None else None
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for model in self.models:
            self.model_combo.addItem(_model_combo_label(model))
            index = self.model_combo.count() - 1
            self.model_combo.setItemData(index, model.name, SELECTED_DISPLAY_ROLE)
            self.model_combo.setItemData(index, str(model.path), Qt.ItemDataRole.ToolTipRole)
        self.model_combo.blockSignals(False)
        if self.models:
            selected_index = _preferred_model_index(self.models, current_path)
            self.model_combo.setCurrentIndex(selected_index)
        self._show_model_status()
        self._update_action_states()

    def refresh_model_catalog(self) -> None:
        refresher = getattr(self.service, "refresh_model_manifest", None)
        if not callable(refresher):
            self.refresh_models()
            return
        self._run_task(
            task=self._refresh_model_manifest,
            on_success=self._on_model_manifest_refreshed,
            busy_text="正在刷新模型目录...",
            error_title="刷新模型目录失败",
            error_category=MODELS_AND_WEIGHTS,
        )

    def refresh_cameras(self) -> None:
        self.close_live_camera(show_status=False)
        if not self._reserve_camera_resource():
            return
        self._run_task(
            task=lambda: self._run_releasing_camera_on_error(self._discover_cameras),
            on_success=lambda value: self._on_camera_scan_ready(value, release_camera=True),
            busy_text="正在扫描摄像头...",
            error_title="扫描失败",
            error_category=CAMERA_AND_VIDEO,
        )

    def open_live_camera(self) -> None:
        camera = self.selected_camera()
        if camera is None:
            QMessageBox.information(self, "没有相机", "请先刷新并选择相机。")
            return

        self.close_live_camera(show_status=False)
        if not self._reserve_camera_resource():
            return
        self._run_task(
            task=lambda: self._run_releasing_camera_on_error(lambda: self._open_live_camera(camera)),
            on_success=self._on_live_camera_opened,
            busy_text=f"正在打开 {camera.label()}...",
            error_title="打开失败",
            error_category=CAMERA_AND_VIDEO,
        )

    def close_live_camera(self, *, show_status: bool = True) -> None:
        was_open = self._camera_open or self._live_thread is not None
        self.stop_live_detection(show_status=False)
        self._stop_live_loop()
        self._close_service_camera()
        self._camera_open = False
        self._live_camera_fps = 0.0
        self._live_inference_fps = 0.0
        self._live_detection_count = 0
        self._live_read_failures = 0
        self._last_live_output = None
        self.result_image = None
        self.results_list.clear()
        self._restore_input_preview()
        self.live_status_label.setText("实时：未打开")
        self._update_info()
        self._update_action_states()
        self._release_camera_resource()
        if show_status and was_open:
            self._set_status("相机已停止。")

    def start_live_detection(self) -> None:
        if not self._camera_open:
            QMessageBox.information(self, "没有相机", "请先打开相机。")
            return
        model = self.selected_model()
        if model is None:
            QMessageBox.information(self, "没有模型", "请先选择模型。")
            return
        if not model.exists:
            QMessageBox.information(self, "模型缺失", "请先下载该模型，或选择本地 .pt 文件。")
            return

        settings = self.current_settings()
        if self._loaded_model_path == model.path:
            self._enable_live_detection(model, settings, loaded_model=False)
            return

        self._run_task(
            task=lambda: self._load_live_model(model, settings),
            on_success=self._on_live_model_ready,
            busy_text=f"正在加载 {model.name}，实时检测即将开始...",
            error_title="模型加载失败",
            error_category=MODELS_AND_WEIGHTS,
        )

    def stop_live_detection(self, *, show_status: bool = True) -> None:
        was_enabled = self._live_detection_enabled
        self._live_detection_enabled = False
        with self._live_frame_condition:
            self._live_frame_condition.notify_all()
        self._live_inference_fps = 0.0
        self._live_detection_count = 0
        self._last_live_output = None
        self.live_status_label.setText(
            f"实时：预览中 | FPS {_format_fps(self._live_camera_fps)}"
            if self._camera_open
            else "实时：未打开"
        )
        self._update_action_states()
        if show_status and was_enabled:
            self._set_status("实时检测已停止，相机预览仍在运行。")

    def open_image(self) -> None:
        if self._camera_open:
            QMessageBox.information(self, "相机运行中", "请先停止相机，再打开图片。")
            return
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "打开图片",
            "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*.*)",
        )
        if not path:
            return

        image_path = Path(path)
        try:
            pixmap = self._pixmap_from_path(image_path)
        except ValueError as exc:
            QMessageBox.critical(self, "打开失败", with_help(exc, DATA_AND_FILES))
            return

        self.image_path = image_path
        self.result_image = None
        self.last_timing_text = None
        self.results_list.clear()
        self.preview_panel.set_pixmap(pixmap)
        self._update_info()
        self._update_action_states()
        self._set_status("图片已就绪。")

    def browse_model(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择本地 YOLO26 PT 模型",
            "",
            "PyTorch 模型 (*.pt);;所有文件 (*.*)",
        )
        if not path:
            return
        try:
            model = self.service.add_custom_model(path)
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
            task=lambda: self._download_model(model.name, model.path),
            on_success=self._on_model_downloaded,
            busy_text=f"正在下载 {model.name}...",
            progress_text=f"正在下载 {model.name}...\n保存路径：{model.path}",
            error_title="下载失败",
            error_category=MODELS_AND_WEIGHTS,
        )

    def detect_image(self) -> None:
        if self._camera_open:
            QMessageBox.information(self, "相机运行中", "请先停止相机，再执行图片检测。")
            return
        if self.image_path is None:
            QMessageBox.information(self, "没有图片", "请先打开一张图片。")
            return
        model = self.selected_model()
        if model is None:
            QMessageBox.information(self, "没有模型", "请先选择模型。")
            return
        if not model.exists:
            QMessageBox.information(self, "模型缺失", "请先下载该模型，或选择本地 .pt 文件。")
            return

        image_path = self.image_path
        settings = self.current_settings()
        first_load = self._loaded_model_path != model.path
        self._run_task(
            task=lambda: self._detect_image(image_path, model, settings, first_load),
            on_success=self._on_detection_ready,
            busy_text=(
                f"首次加载 {model.name} 并检测，可能需要数秒..."
                if first_load
                else f"正在使用已加载的 {model.name} 检测..."
            ),
            error_title="检测失败",
            error_category=MODULE_RUNTIME_ERRORS,
        )

    def save_result(self) -> None:
        if self.result_image is None:
            QMessageBox.information(self, "没有结果", "请先完成一次检测。")
            return

        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存检测图",
            "",
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg *.jpeg);;位图 (*.bmp);;所有文件 (*.*)",
        )
        if not path:
            return

        result = self.result_image.copy()
        output_path = Path(path)
        self._run_task(
            task=lambda: self._save_detection(result, output_path),
            on_success=self._on_detection_saved,
            busy_text="正在保存检测图...",
            error_title="保存失败",
            error_category=DATA_AND_FILES,
        )

    def selected_model(self) -> ModelInfo | None:
        index = self.model_combo.currentIndex()
        if index < 0 or index >= len(self.models):
            return None
        return self.models[index]

    def selected_camera(self) -> CameraDevice | None:
        index = self.camera_combo.currentIndex()
        if index < 0 or index >= len(self.cameras):
            return None
        return self.cameras[index]

    def current_settings(self) -> DetectionSettings:
        return DetectionSettings(
            image_size=int(self.image_size_spin.value()),
            confidence=float(self.conf_spin.value()),
            iou=float(self.iou_spin.value()),
            device=self.device_combo.currentText(),
        )

    def shutdown(self) -> None:
        self.close_live_camera(show_status=False)
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

    def _download_model(self, name: str, expected_path: Path | None = None) -> ModelPayload:
        started_at = time.perf_counter()
        path = expected_path or self.config.model_dir / name
        model = self.service.download_official_model(
            name,
            progress_callback=self._download_progress_callback(f"正在下载 {name}", path),
        )
        return ModelPayload(model=model, total_ms=_elapsed_ms(started_at))

    def _refresh_model_manifest(self) -> ModelManifestPayload:
        started_at = time.perf_counter()
        refresher = getattr(self.service, "refresh_model_manifest")
        entry_count = int(refresher())
        return ModelManifestPayload(entry_count=entry_count, total_ms=_elapsed_ms(started_at))

    def _discover_cameras(self) -> CameraScanPayload:
        started_at = time.perf_counter()
        cameras = self.service.discover_cameras()
        return CameraScanPayload(cameras=list(cameras), total_ms=_elapsed_ms(started_at))

    def _open_live_camera(self, camera: CameraDevice) -> OpenCameraPayload:
        started_at = time.perf_counter()
        with self._camera_lock:
            self.service.open_camera(
                camera,
                (self.config.requested_capture_width, self.config.requested_capture_height),
            )
        return OpenCameraPayload(camera=camera, total_ms=_elapsed_ms(started_at))

    def _load_live_model(
        self,
        model: ModelInfo,
        settings: DetectionSettings,
    ) -> LiveDetectionStartPayload:
        started_at = time.perf_counter()
        self.service.load_model(model.path)
        return LiveDetectionStartPayload(
            model=model,
            settings=settings,
            total_ms=_elapsed_ms(started_at),
            loaded_model=True,
        )

    def _detect_image(
        self,
        image_path: Path,
        model: ModelInfo,
        settings: DetectionSettings,
        first_load: bool,
    ) -> DetectionPayload:
        started_at = time.perf_counter()
        image = _load_image_bgr(image_path)
        if first_load:
            self.service.load_model(model.path)
        output = self.service.detect_frame(image, settings)
        return DetectionPayload(
            output=output,
            image=image,
            model=model,
            total_ms=_elapsed_ms(started_at),
            loaded_model=first_load,
        )

    def _save_detection(self, image: ImageArray, path: Path) -> SaveDetectionPayload:
        started_at = time.perf_counter()
        self.service.save_screenshot(image, path)
        return SaveDetectionPayload(path=path, total_ms=_elapsed_ms(started_at))

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

    def _on_cameras_ready(self, value: object) -> None:
        payload = cast(CameraScanPayload, value)
        self.camera_coordinator.update_devices(payload.cameras)
        self._set_cameras(payload.cameras)
        if self.cameras:
            self._completion_status = f"发现 {len(self.cameras)} 条相机路由。"
        else:
            self._completion_status = "未发现相机，请检查权限或连接。"
        self.last_timing_text = f"扫描 {payload.total_ms:.1f} ms"
        self._update_info()
        self._update_action_states()

    def _on_camera_scan_ready(self, value: object, *, release_camera: bool) -> None:
        try:
            self._on_cameras_ready(value)
        finally:
            if release_camera:
                self._release_camera_resource()

    def _apply_cached_cameras(self) -> None:
        cameras = self.camera_coordinator.cached_devices_as(CameraDevice, CameraBackend)
        current = self.selected_camera()
        current_key = current.key() if current is not None else ""
        self._set_cameras(cameras, preferred_key=current_key)
        self._update_info()
        self._update_action_states()

    def _set_cameras(self, cameras: list[CameraDevice], preferred_key: str = "") -> None:
        self.cameras = list(cameras)
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for camera in self.cameras:
            self.camera_combo.addItem(camera.label())
        selected_index = 0
        if preferred_key:
            for index, camera in enumerate(self.cameras):
                if camera.key() == preferred_key:
                    selected_index = index
                    break
        if self.cameras:
            self.camera_combo.setCurrentIndex(selected_index)
        self.camera_combo.blockSignals(False)

    def _on_live_camera_opened(self, value: object) -> None:
        payload = cast(OpenCameraPayload, value)
        self._camera_open = True
        self._live_read_failures = 0
        self._live_camera_fps = 0.0
        self._live_inference_fps = 0.0
        self._live_detection_count = 0
        self.last_timing_text = f"打开相机 {payload.total_ms:.1f} ms"
        self._completion_status = f"已打开 {payload.camera.label()}。"
        self.live_status_label.setText("实时：预览中")
        self._start_live_loop()
        self._update_info()
        self._update_action_states()

    def _on_live_model_ready(self, value: object) -> None:
        payload = cast(LiveDetectionStartPayload, value)
        self._loaded_model_path = payload.model.path
        self.last_timing_text = f"加载模型 {payload.total_ms:.1f} ms"
        self._enable_live_detection(payload.model, payload.settings, loaded_model=True)

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

    def _on_model_manifest_refreshed(self, value: object) -> None:
        payload = cast(ModelManifestPayload, value)
        self.refresh_models()
        if payload.entry_count:
            self._completion_status = f"模型目录已刷新：{payload.entry_count} 项（{payload.total_ms:.1f} ms）。"
        else:
            self._completion_status = f"模型列表已重新扫描（{payload.total_ms:.1f} ms）。"

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

    def _on_detection_ready(self, value: object) -> None:
        payload = cast(DetectionPayload, value)
        self._loaded_model_path = payload.model.path
        self.result_image = payload.output.annotated_frame
        self.preview_panel.set_pixmap(self.presenter.to_pixmap(self.result_image))
        self._show_detections(payload.output)
        prepare_ms = max(0.0, payload.total_ms - payload.output.inference_ms)
        self.last_timing_text = (
            f"加载/准备 {prepare_ms:.1f} ms | "
            f"推理 {payload.output.inference_ms:.1f} ms | "
            f"总计 {payload.total_ms:.1f} ms"
        )
        cache_text = "模型已加载" if payload.loaded_model else "已复用模型"
        self._completion_status = f"检测完成，{cache_text}（{self.last_timing_text}）。"
        logger.info("YOLO detection timing: %s", self.last_timing_text)
        self._update_info()
        self._update_action_states()

    def _on_detection_saved(self, value: object) -> None:
        payload = cast(SaveDetectionPayload, value)
        self.last_timing_text = f"保存 {payload.total_ms:.1f} ms"
        self._completion_status = f"检测结果已保存（{self.last_timing_text}）。"
        self._update_info()
        QMessageBox.information(self, "保存完成", f"检测结果已保存到：\n{payload.path}")

    def _enable_live_detection(
        self,
        model: ModelInfo,
        settings: DetectionSettings,
        *,
        loaded_model: bool,
    ) -> None:
        with self._live_settings_lock:
            self._live_settings = settings
        self._live_detection_enabled = True
        self._live_inference_fps = 0.0
        self._live_detection_count = 0
        self.result_image = None
        self._last_live_output = None
        with self._live_frame_condition:
            self._live_frame_condition.notify_all()
        self.results_list.clear()
        cache_text = "模型已加载" if loaded_model else "已复用模型"
        self._completion_status = f"实时检测已开始，{cache_text}：{model.name}"
        self.live_status_label.setText(f"实时：检测中 | {cache_text}")
        self._set_status(self._completion_status)
        self._update_action_states()

    def _start_live_loop(self) -> None:
        self._stop_live_loop()
        stop_event = threading.Event()
        self._live_stop = stop_event
        self._live_thread = threading.Thread(
            target=self._live_loop,
            args=(stop_event,),
            name="yolo-camera-preview",
            daemon=True,
        )
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            args=(stop_event,),
            name="yolo-live-detection",
            daemon=True,
        )
        self._live_thread.start()
        self._inference_thread.start()

    def _stop_live_loop(self) -> None:
        stop_event = self._live_stop
        if stop_event is not None:
            stop_event.set()
        with self._live_frame_condition:
            self._latest_inference_frame = None
            self._latest_frame_sequence = 0
            self._live_frame_condition.notify_all()
        thread = self._live_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.8)
        inference_thread = self._inference_thread
        if (
            inference_thread is not None
            and inference_thread.is_alive()
            and inference_thread is not threading.current_thread()
        ):
            inference_thread.join(timeout=0.8)
        self._live_stop = None
        self._live_thread = None
        self._inference_thread = None

    def _live_loop(self, stop_event: threading.Event) -> None:
        last_frame_at = None  # type: Optional[float]
        camera_fps = 0.0
        while not stop_event.is_set():
            try:
                with self._camera_lock:
                    if stop_event.is_set() or not self.service.is_camera_open():
                        break
                    frame = self.service.read_frame()
            except Exception as exc:
                self.live_failed.emit(exc)
                time.sleep(0.08)
                continue

            now = time.perf_counter()
            if last_frame_at is not None and now > last_frame_at:
                instant_fps = 1.0 / (now - last_frame_at)
                camera_fps = _smooth_fps(camera_fps, instant_fps, 0.15)
            last_frame_at = now

            with self._live_frame_condition:
                self._latest_inference_frame = frame.copy()
                self._latest_frame_sequence += 1
                self._live_frame_condition.notify_all()

            self.live_frame_ready.emit(
                LiveFramePayload(
                    frame=frame,
                    output=None,
                    camera_fps=camera_fps,
                    inference_fps=self._live_inference_fps,
                )
            )

    def _inference_loop(self, stop_event: threading.Event) -> None:
        last_sequence = 0
        inference_fps = 0.0
        while not stop_event.is_set():
            with self._live_frame_condition:
                self._live_frame_condition.wait_for(
                    lambda: (
                        stop_event.is_set()
                        or (
                            self._live_detection_enabled
                            and self._latest_inference_frame is not None
                            and self._latest_frame_sequence != last_sequence
                        )
                    ),
                    timeout=0.2,
                )
                if stop_event.is_set():
                    break
                if (
                    not self._live_detection_enabled
                    or self._latest_inference_frame is None
                    or self._latest_frame_sequence == last_sequence
                ):
                    continue
                frame = self._latest_inference_frame.copy()
                last_sequence = self._latest_frame_sequence

            with self._live_settings_lock:
                settings = self._live_settings

            started_at = time.perf_counter()
            try:
                output = self.service.detect_frame(frame, settings)
            except Exception as exc:
                self._live_detection_enabled = False
                self.live_failed.emit(exc)
                continue

            elapsed = time.perf_counter() - started_at
            if elapsed > 0:
                inference_fps = _smooth_fps(inference_fps, 1.0 / elapsed, 0.20)
            self.live_detection_ready.emit(
                LiveDetectionPayload(output=output, inference_fps=inference_fps)
            )

    def _on_live_frame_ready(self, value: object) -> None:
        if self._live_stop is None or not self._camera_open:
            return
        payload = cast(LiveFramePayload, value)
        self._live_read_failures = 0
        self._live_camera_fps = payload.camera_fps
        if payload.output is not None:
            self._live_inference_fps = payload.inference_fps
            self._live_detection_count = payload.output.detection_count
            self._last_live_output = payload.output
            self._show_detections(payload.output)
        if self._live_detection_enabled and self._last_live_output is not None:
            display_frame = _draw_detection_overlay(payload.frame, self._last_live_output)
            self.result_image = display_frame
        else:
            display_frame = payload.frame
            if not self._live_detection_enabled:
                self.result_image = None
        self.preview_panel.set_pixmap(self.presenter.to_pixmap(display_frame))
        self.live_status_label.setText(
            "实时："
            + ("检测中" if self._live_detection_enabled else "预览中")
            + f" | 相机 {_format_fps(self._live_camera_fps)}"
            + (
                f" | 推理 {_format_fps(self._live_inference_fps)} | 检测 {self._live_detection_count}"
                if self._live_detection_enabled or self._live_detection_count
                else ""
            )
        )
        self._update_info()
        self._update_action_states()

    def _on_live_detection_ready(self, value: object) -> None:
        if self._live_stop is None or not self._camera_open or not self._live_detection_enabled:
            return
        payload = cast(LiveDetectionPayload, value)
        self._live_inference_fps = payload.inference_fps
        self._live_detection_count = payload.output.detection_count
        self._last_live_output = payload.output
        self._show_detections(payload.output)
        self.live_status_label.setText(
            "实时：检测中"
            + f" | 相机 {_format_fps(self._live_camera_fps)}"
            + f" | 推理 {_format_fps(self._live_inference_fps)} | 检测 {self._live_detection_count}"
        )
        self._update_info()
        self._update_action_states()

    def _on_live_failed(self, value: object) -> None:
        if self._live_stop is None or not self._camera_open:
            return
        self._live_read_failures += 1
        if self._live_read_failures < 5:
            return
        exc = cast(Exception, value)
        self.close_live_camera(show_status=False)
        QMessageBox.critical(self, "实时检测失败", with_help(exc, MODULE_RUNTIME_ERRORS))
        self._set_status("实时检测失败，已停止相机。")

    def _close_service_camera(self) -> None:
        acquired = self._camera_lock.acquire(timeout=0.8)
        if not acquired:
            logger.warning("Skipped immediate YOLO camera close because frame read is still active.")
            return
        try:
            self.service.close_camera()
        finally:
            self._camera_lock.release()

    def _on_model_changed(self, _index: int) -> None:
        if self._live_detection_enabled:
            self.stop_live_detection(show_status=False)
        self.results_list.clear()
        self.result_image = None
        self._restore_input_preview()
        self.last_timing_text = None
        self._show_model_status()
        self._update_info()
        self._update_action_states()

    def _show_model_status(self) -> None:
        model = self.selected_model()
        if model is None:
            self.model_status_label.setText("模型：未找到")
            return
        state = _model_state_text(model)
        kind = "官方" if model.is_official else "自定义"
        self.model_status_label.setText(f"{kind}模型：{state} | {_shorten_text(model.name, 28)}")
        self.model_status_label.setToolTip(str(model.path))

    def _show_detections(self, output: DetectionOutput) -> None:
        self.results_list.clear()
        for index, detection in enumerate(output.detections, start=1):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(8, 6, 8, 6)
            layout.setSpacing(10)

            rank_label = QLabel(str(index))
            rank_label.setMinimumWidth(22)
            name_label = QLabel(detection.class_name)
            score_label = QLabel(f"{detection.confidence * 100:.1f}%")
            box_label = QLabel(_box_text(detection.xyxy))
            box_label.setObjectName("MutedText")
            box_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

            layout.addWidget(rank_label)
            layout.addWidget(name_label, 1)
            layout.addWidget(score_label)
            layout.addWidget(box_label, 2)

            item = QListWidgetItem()
            item.setSizeHint(row.sizeHint())
            self.results_list.addItem(item)
            self.results_list.setItemWidget(item, row)

    def _pixmap_from_path(self, image_path: Path) -> QPixmap:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            raise ValueError(f"Cannot preview image file: {image_path}")
        return pixmap

    def _service_loaded_model_path(self) -> Path | None:
        loaded = getattr(self.service, "loaded_model_path", None)
        if not callable(loaded):
            return None
        return loaded()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if not busy:
            self.busy_label.setVisible(False)
            self.busy_progress.setVisible(False)
        self._update_action_states()

    def _update_action_states(self) -> None:
        model = self.selected_model()
        has_image = self.image_path is not None
        has_model = model is not None and model.exists
        can_download = model is not None and model.is_official and not model.exists
        can_edit_inference = not self._busy and not self._live_detection_enabled
        self.download_model_button.setText(_download_button_text(model))
        self.open_image_button.setEnabled(not self._busy and not self._camera_open)
        self.detect_button.setEnabled(has_image and has_model and not self._busy and not self._camera_open)
        self.save_result_button.setEnabled(self.result_image is not None and not self._busy)
        self.refresh_models_button.setEnabled(not self._busy)
        self.browse_model_button.setEnabled(not self._busy and not self._live_detection_enabled)
        self.download_model_button.setEnabled(can_download and not self._busy)
        self.refresh_cameras_button.setEnabled(not self._busy and not self._camera_open)
        self.open_camera_button.setEnabled(
            self.selected_camera() is not None and not self._busy and not self._camera_open
        )
        self.close_camera_button.setEnabled(self._camera_open and not self._busy)
        self.start_live_button.setEnabled(self._camera_open and has_model and not self._busy and not self._live_detection_enabled)
        self.stop_live_button.setEnabled(self._live_detection_enabled and not self._busy)
        self.model_combo.setEnabled(can_edit_inference)
        self.device_combo.setEnabled(can_edit_inference)
        self.image_size_spin.setEnabled(can_edit_inference)
        self.conf_spin.setEnabled(can_edit_inference)
        self.iou_spin.setEnabled(can_edit_inference)
        self.camera_combo.setEnabled(not self._busy and not self._camera_open)

    def _reserve_camera_resource(self) -> bool:
        if self.camera_coordinator.reserve(self._camera_owner_id, self._camera_owner_label):
            return True
        message = self.camera_coordinator.busy_message(self._camera_owner_id) or "相机正在被其他页面使用。"
        self._update_action_states()
        QMessageBox.information(self, "相机正在使用", message)
        return False

    def _release_camera_resource(self) -> None:
        self.camera_coordinator.release(self._camera_owner_id)

    def _run_releasing_camera_on_error(self, task: Callable[[], object]) -> object:
        try:
            return task()
        except Exception:
            self._release_camera_resource()
            raise

    def _update_info(self) -> None:
        parts = []
        if self.image_path is not None:
            parts.append(f"图片：{_shorten_path(self.image_path, 68)}")
        camera = self.selected_camera()
        if camera is not None:
            parts.append(f"相机：{camera.label()}")
        model = self.selected_model()
        if model is not None:
            parts.append(f"模型：{model.name}")
        if self._camera_open:
            parts.append("实时预览：运行中")
        if self._live_detection_enabled or self._live_detection_count:
            parts.append(f"实时检测数：{self._live_detection_count}")
        if self.result_image is not None:
            parts.append(f"检测数：{self.results_list.count()}")
        if self.last_timing_text:
            parts.append(self.last_timing_text)
        self.info_label.setText(" | ".join(parts))
        if self.image_path is not None:
            self.info_label.setToolTip(str(self.image_path))

    def _set_status(self, text: str) -> None:
        self.status_changed.emit(text)

    def _restore_input_preview(self) -> None:
        if self._camera_open:
            return
        if self.image_path is None:
            self.preview_panel.clear()
            return
        try:
            self.preview_panel.set_pixmap(self._pixmap_from_path(self.image_path))
        except ValueError:
            self.preview_panel.clear()


def _load_image_bgr(path: Path) -> ImageArray:
    path = validate_image_file(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image file: {path}")
    return image


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


def _box_text(xyxy: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = xyxy
    return f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]"


def _draw_detection_overlay(frame: ImageArray, output: DetectionOutput) -> ImageArray:
    display = np.asarray(frame).copy()
    height, width = display.shape[:2]
    for detection in output.detections:
        x1, y1, x2, y2 = detection.xyxy
        left = max(0, min(width - 1, int(round(x1))))
        top = max(0, min(height - 1, int(round(y1))))
        right = max(0, min(width - 1, int(round(x2))))
        bottom = max(0, min(height - 1, int(round(y2))))
        color = _class_color(detection.class_id)
        cv2.rectangle(display, (left, top), (right, bottom), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        text_origin = (left, max(12, top - 5))
        cv2.putText(
            display,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return display


def _class_color(class_id: int) -> tuple[int, int, int]:
    palette = (
        (0, 128, 255),
        (255, 128, 0),
        (80, 220, 80),
        (220, 80, 220),
        (60, 180, 220),
    )
    return palette[class_id % len(palette)]


def _smooth_fps(current: float, instant: float, alpha: float) -> float:
    return instant if current <= 0.0 else current * (1.0 - alpha) + instant * alpha


def _format_fps(value: float) -> str:
    return f"{value:.1f} FPS" if value > 0 else "--"


def _busy_notice_text(text: str) -> str:
    lower = text.lower()
    if "加载" in text and ("模型" in text or ".pt" in lower or "yolo" in lower):
        return f"{text} 窗口仍在工作，请稍等。"
    if "打开" in text and ("相机" in text or "Camera" in text):
        return f"{text} 正在请求摄像头权限和第一帧。"
    if "扫描摄像头" in text:
        return f"{text} 这可能需要几秒。"
    return text


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0
