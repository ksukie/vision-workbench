"""Qt page for image classification prediction."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, cast

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Signal, Qt
from PySide6.QtGui import QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from image_classification.api import (
    ImageClassificationConfig,
    ImageClassificationService,
    PredictionResult,
    PretrainedWeightInfo,
    create_image_classification_service,
)
from image_classification.domain import ClassificationTrainingConfig, DatasetValidationReport
from vision_workbench.model_files import model_file_issue
from vision_workbench.troubleshooting import DATASETS_AND_TRAINING, DATA_AND_FILES, MODELS_AND_WEIGHTS, with_help
from vision_workbench.runtime_security import validate_run_name
from vision_workbench.sample_data import create_classification_sample_dataset
from vision_workbench.runtime_diagnostics import TrainingEnvironmentReport, inspect_training_environment

from ..task_runner import QtTaskRunner
from ..widgets import NoWheelDoubleSpinBox as QDoubleSpinBox
from ..widgets import NoWheelComboBox as QComboBox
from ..widgets import NoWheelSpinBox as QSpinBox
from ..widgets import PreviewPanel, SectionCard, make_button, set_download_progress, style_form_label


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionPayload:
    result: PredictionResult
    total_ms: float


class ClassificationPage(QWidget):
    """Native Qt implementation of the image-classification prediction workflow."""

    status_changed = Signal(str)
    download_progress_changed = Signal(object)

    def __init__(
        self,
        service: Optional[ImageClassificationService] = None,
        config: ImageClassificationConfig = ImageClassificationConfig(),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.service = service or create_image_classification_service(config)
        self.tasks = QtTaskRunner(self)
        self.training_process = None  # type: QProcess | None
        self._active_training_job = None  # type: ClassificationTrainingConfig | None
        self._training_output_buffer = ""
        self._training_stop_requested = False
        self._recommended_batch_size = None  # type: int | None

        self.image_path = None  # type: Optional[Path]
        self.checkpoint_path = None  # type: Optional[Path]
        self.last_timing_text = None  # type: Optional[str]
        self._busy = False
        self._completion_status = None  # type: Optional[str]
        self._warmed_pretrained_keys = set()  # type: set[tuple[str, str]]

        self.model_combo = QComboBox()
        self.model_combo.addItems(self.service.supported_models())
        self.model_combo.setCurrentText(config.default_model_name)
        self.model_combo.setToolTip("选择用于预测的预训练模型")
        self.model_combo.setMinimumWidth(170)

        self.device_combo = QComboBox()
        self.device_combo.addItems(config.device_options)
        self.device_combo.setCurrentText("auto")
        self.device_combo.setToolTip("选择推理设备")
        self.device_combo.setMinimumWidth(112)

        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 10)
        self.topk_spin.setValue(config.default_topk)
        self.topk_spin.setToolTip("预测结果行数")
        self.topk_spin.setMinimumWidth(78)

        self.model_label = QLabel("模型")
        self.device_label = QLabel("设备")
        self.topk_label = QLabel("Top-K")
        for label in (self.model_label, self.device_label, self.topk_label):
            style_form_label(label)
        for label, control in (
            (self.model_label, self.model_combo),
            (self.device_label, self.device_combo),
            (self.topk_label, self.topk_spin),
        ):
            label.setBuddy(control)
            control.setAccessibleName(label.text())

        self.preview_panel = PreviewPanel("输入图像", "请打开一张图片")
        self.results_list = QListWidget()
        self.results_list.setObjectName("PredictionList")
        self.results_list.setAlternatingRowColors(True)
        self.results_list.setMinimumHeight(260)

        self.weight_status_label = QLabel("")
        self.weight_status_label.setObjectName("MutedText")
        self.weight_status_label.setWordWrap(True)
        self.weight_status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.checkpoint_label = QLabel("模型文件：未选择")
        self.checkpoint_label.setObjectName("MutedText")
        self.checkpoint_label.setWordWrap(True)
        self.checkpoint_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.info_label = QLabel("")
        self.info_label.setObjectName("MutedText")
        self.info_label.setWordWrap(True)

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

        self.open_button = make_button("打开图片", primary=True)
        self.predict_pretrained_button = make_button("预测", primary=True)
        self.check_weights_button = make_button("检查权重")
        self.download_weights_button = make_button("下载权重")
        self.import_weights_button = make_button("导入权重")
        self.browse_checkpoint_button = make_button("选择本地模型文件")
        self.browse_checkpoint_button.setToolTip("从本地磁盘选择一个 .pt 或 .pth checkpoint")
        self.predict_checkpoint_button = make_button("预测自定义模型")
        for button in (
            self.open_button,
            self.predict_pretrained_button,
            self.check_weights_button,
            self.download_weights_button,
            self.import_weights_button,
            self.browse_checkpoint_button,
            self.predict_checkpoint_button,
        ):
            button.setMinimumWidth(132)

        self.train_dataset_label = QLabel("数据集")
        self.train_dataset_edit = QLineEdit()
        self.train_dataset_edit.setPlaceholderText("选择包含 train/ 和 val/ 的分类数据集")
        self.train_dataset_edit.setMinimumWidth(320)
        self.select_dataset_button = make_button("选择数据集", primary=True)
        self.validate_dataset_button = make_button("验证数据集")
        self.sample_dataset_button = make_button("创建示例数据")

        self.train_model_label = QLabel("模型")
        self.train_model_combo = QComboBox()
        self.train_model_combo.addItems(self.service.supported_models())
        self.train_model_combo.setCurrentText(config.default_model_name)
        self.train_model_combo.setMinimumWidth(170)

        self.train_device_label = QLabel("设备")
        self.train_device_combo = QComboBox()
        self.train_device_combo.addItems(config.device_options)
        self.train_device_combo.setCurrentText("auto")
        self.train_device_combo.setMinimumWidth(112)

        self.train_epochs_label = QLabel("轮数")
        self.train_epochs_spin = QSpinBox()
        self.train_epochs_spin.setRange(1, 500)
        self.train_epochs_spin.setValue(config.default_epochs)
        self.train_epochs_spin.setMinimumWidth(92)
        self.train_epochs_spin.setToolTip("完整遍历训练集的次数。先用 3～5 轮确认数据和流程。")

        self.train_image_size_label = QLabel("图像尺寸")
        self.train_image_size_spin = QSpinBox()
        self.train_image_size_spin.setRange(64, 1024)
        self.train_image_size_spin.setSingleStep(32)
        self.train_image_size_spin.setValue(config.default_image_size)
        self.train_image_size_spin.setMinimumWidth(92)
        self.train_image_size_spin.setToolTip("分类输入尺寸。尺寸越大越占显存，通常从 224 开始。")

        self.train_batch_label = QLabel("批量")
        self.train_batch_spin = QSpinBox()
        self.train_batch_spin.setRange(1, 256)
        self.train_batch_spin.setValue(config.default_batch_size)
        self.train_batch_spin.setMinimumWidth(92)
        self.train_batch_spin.setToolTip("每批图片数。显存不足时降低到 8、4 或 1。")

        self.train_lr_label = QLabel("学习率")
        self.train_lr_spin = QDoubleSpinBox()
        self.train_lr_spin.setRange(0.000001, 1.0)
        self.train_lr_spin.setDecimals(6)
        self.train_lr_spin.setSingleStep(0.0005)
        self.train_lr_spin.setValue(config.default_learning_rate)
        self.train_lr_spin.setMinimumWidth(118)
        self.train_lr_spin.setToolTip("优化器步长。迁移学习通常先保留默认值 0.001。")

        self.train_run_name_label = QLabel("运行名称")
        self.train_run_name_edit = QLineEdit("classification_train")
        self.train_run_name_edit.setMinimumWidth(180)
        self.train_run_name_edit.setToolTip("仅填写名称，不要输入目录、斜杠或 ..。")

        self.train_pretrained_check = QCheckBox("使用预训练权重")
        self.train_pretrained_check.setChecked(True)
        self.train_pretrained_check.setToolTip("新手和小数据集建议保持启用。")
        self.train_freeze_check = QCheckBox("冻结 backbone")
        self.train_freeze_check.setChecked(True)
        self.train_freeze_check.setToolTip("只训练分类头，速度更快且更不容易过拟合。")
        self.start_training_button = make_button("开始训练", primary=True)
        self.start_training_button.setMinimumWidth(132)
        self.stop_training_button = make_button("停止训练", danger=True)
        self.stop_training_button.setMinimumWidth(112)
        self.jump_to_training_button = make_button("前往训练设置")
        self.jump_to_training_button.setToolTip("滚动到本页下方的分类训练区域")

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMinimumHeight(220)
        self.train_log.setPlaceholderText("数据集验证和训练结果会显示在这里。")
        self.train_log.setPlainText("选择分类数据集，验证后开始训练。")

        self.train_status_label = QLabel("")
        self.train_status_label.setObjectName("MutedText")
        self.train_status_label.setWordWrap(True)

        self.train_hint_label = QLabel(
            "新手建议：保持“预训练权重”和“冻结 backbone”，先训练 3～5 轮；显存不足时降低批量。"
        )
        self.train_hint_label.setObjectName("ParameterHint")
        self.train_hint_label.setWordWrap(True)

        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, max(1, config.default_epochs))
        self.train_progress.setValue(0)
        self.train_progress.setFormat("等待训练")
        self.train_metric_label = QLabel("尚未开始训练。")
        self.train_metric_label.setObjectName("MutedText")
        self.train_metric_label.setWordWrap(True)
        self.check_environment_button = make_button("检查训练环境")
        self.apply_batch_button = make_button("应用推荐批量")
        self.apply_batch_button.setEnabled(False)
        self.environment_label = QLabel("尚未检查训练环境。")
        self.environment_label.setObjectName("MutedText")
        self.environment_label.setWordWrap(True)

        for label in (
            self.train_dataset_label,
            self.train_model_label,
            self.train_device_label,
            self.train_epochs_label,
            self.train_image_size_label,
            self.train_batch_label,
            self.train_lr_label,
            self.train_run_name_label,
        ):
            style_form_label(label)

        for label, control in (
            (self.train_dataset_label, self.train_dataset_edit),
            (self.train_model_label, self.train_model_combo),
            (self.train_device_label, self.train_device_combo),
            (self.train_epochs_label, self.train_epochs_spin),
            (self.train_image_size_label, self.train_image_size_spin),
            (self.train_batch_label, self.train_batch_spin),
            (self.train_lr_label, self.train_lr_spin),
            (self.train_run_name_label, self.train_run_name_edit),
        ):
            label.setBuddy(control)
            control.setAccessibleName(label.text())

        self.content = None  # type: QWidget | None
        self.controls_layout = None  # type: QGridLayout | None
        self.splitter = None  # type: QSplitter | None
        self.results_panel = None  # type: QWidget | None
        self.page_scroll_area = None  # type: QScrollArea | None
        self.training_panel = None  # type: QWidget | None
        self.mode_tabs = None  # type: QTabWidget | None
        self._compact_layout = None  # type: bool | None

        self._build_ui()
        self._connect_signals()
        self._update_action_states()
        self.check_weight_status()

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
        self.page_scroll_area = scroll_area

        content = QWidget()
        content.setObjectName("PageContent")
        content.setMinimumWidth(760)
        self.content = content

        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 22, 24, 18)
        layout.setSpacing(16)

        title = QLabel("图像分类")
        title.setObjectName("PageTitle")
        subtitle = QLabel("使用预训练模型或自定义模型文件进行 Top-K 分类预测。")
        subtitle.setObjectName("PageSubtitle")
        title_row = QHBoxLayout()
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self.jump_to_training_button)
        layout.addLayout(title_row)
        layout.addWidget(subtitle)

        self.mode_tabs = QTabWidget()
        self.mode_tabs.setObjectName("WorkflowTabs")
        prediction_tab = QWidget()
        prediction_layout = QVBoxLayout(prediction_tab)
        prediction_layout.setContentsMargins(0, 14, 0, 0)
        prediction_layout.setSpacing(16)
        training_tab = QWidget()
        training_layout = QVBoxLayout(training_tab)
        training_layout.setContentsMargins(0, 14, 0, 0)
        training_layout.setSpacing(16)

        controls_card = SectionCard("预测")
        self.controls_layout = QGridLayout()
        self.controls_layout.setHorizontalSpacing(10)
        self.controls_layout.setVerticalSpacing(10)
        controls_card.content_layout.addLayout(self.controls_layout)
        prediction_layout.addWidget(controls_card)
        prediction_layout.addWidget(self.busy_label)
        prediction_layout.addWidget(self.busy_progress)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.preview_panel)
        self.results_panel = self._build_results_panel()
        self.splitter.addWidget(self.results_panel)
        self.splitter.setSizes([2, 1])
        self.splitter.setMinimumHeight(420)
        prediction_layout.addWidget(self.splitter, 1)
        prediction_layout.addWidget(self.info_label)
        self.training_panel = self._build_training_panel()
        training_layout.addWidget(self.training_panel)
        training_layout.addStretch(1)
        self.mode_tabs.addTab(prediction_tab, "预测")
        self.mode_tabs.addTab(training_tab, "训练")
        layout.addWidget(self.mode_tabs, 1)

        scroll_area.setWidget(content)
        root_layout.addWidget(scroll_area)
        self._apply_responsive_layout(force=True)

    def _build_results_panel(self) -> QWidget:
        panel = SectionCard("Top-K 结果")
        panel.setMinimumWidth(280)
        panel.content_layout.addWidget(self.results_list, 1)
        return panel

    def _build_training_panel(self) -> QWidget:
        panel = SectionCard("训练", "数据集需包含 train/ 和 val/，两者下的类别文件夹必须一致。")
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        grid.addWidget(self.train_dataset_label, 0, 0, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.train_dataset_edit, 0, 1, 1, 4)
        grid.addWidget(self.select_dataset_button, 0, 5)
        grid.addWidget(self.validate_dataset_button, 0, 6)
        grid.addWidget(self.sample_dataset_button, 0, 7)

        grid.addWidget(self.train_model_label, 1, 0, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.train_model_combo, 1, 1)
        grid.addWidget(self.train_device_label, 1, 2, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.train_device_combo, 1, 3)
        grid.addWidget(self.train_run_name_label, 1, 4, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.train_run_name_edit, 1, 5, 1, 2)

        grid.addWidget(self.train_epochs_label, 2, 0, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.train_epochs_spin, 2, 1)
        grid.addWidget(self.train_image_size_label, 2, 2, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.train_image_size_spin, 2, 3)
        grid.addWidget(self.train_batch_label, 2, 4, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.train_batch_spin, 2, 5)
        grid.addWidget(self.train_lr_label, 2, 6, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.train_lr_spin, 2, 7)

        options = QHBoxLayout()
        options.setSpacing(14)
        options.addWidget(self.train_pretrained_check)
        options.addWidget(self.train_freeze_check)
        options.addStretch(1)
        options.addWidget(self.start_training_button)
        options.addWidget(self.stop_training_button)

        panel.content_layout.addLayout(grid)
        panel.content_layout.addLayout(options)
        panel.content_layout.addWidget(self.train_hint_label)
        environment_row = QHBoxLayout()
        environment_row.addWidget(self.check_environment_button)
        environment_row.addWidget(self.apply_batch_button)
        environment_row.addWidget(self.environment_label, 1)
        panel.content_layout.addLayout(environment_row)
        panel.content_layout.addWidget(self.train_progress)
        panel.content_layout.addWidget(self.train_metric_label)
        panel.content_layout.addWidget(self.train_log)
        panel.content_layout.addWidget(self.train_status_label)
        return panel

    def _connect_signals(self) -> None:
        self.open_button.clicked.connect(self.open_image)
        self.predict_pretrained_button.clicked.connect(self.predict_pretrained)
        self.check_weights_button.clicked.connect(self.check_weight_status)
        self.download_weights_button.clicked.connect(self.download_pretrained_weight)
        self.import_weights_button.clicked.connect(self.import_local_weight)
        self.browse_checkpoint_button.clicked.connect(self.browse_checkpoint)
        self.predict_checkpoint_button.clicked.connect(self.predict_checkpoint)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.download_progress_changed.connect(self._on_download_progress)
        self.select_dataset_button.clicked.connect(self.select_training_dataset)
        self.validate_dataset_button.clicked.connect(self.validate_training_dataset)
        self.sample_dataset_button.clicked.connect(self.create_sample_training_dataset)
        self.start_training_button.clicked.connect(self.start_training)
        self.stop_training_button.clicked.connect(self.stop_training)
        self.train_dataset_edit.textChanged.connect(self._update_action_states)
        self.jump_to_training_button.clicked.connect(self.show_training_panel)
        self.check_environment_button.clicked.connect(self.check_training_environment)
        self.apply_batch_button.clicked.connect(self.apply_recommended_batch)

    def show_training_panel(self) -> None:
        if self.mode_tabs is not None:
            self.mode_tabs.setCurrentIndex(1)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def _apply_responsive_layout(self, *, force: bool = False) -> None:
        controls = self.controls_layout
        splitter = self.splitter
        if controls is None or splitter is None:
            return

        width = self.width()
        compact = width < 1180
        if not force and compact == self._compact_layout:
            return
        self._compact_layout = compact

        self._reset_grid(controls)
        if compact:
            self._layout_compact_controls(controls)
            splitter.setOrientation(Qt.Orientation.Vertical)
            splitter.setSizes([3, 2])
            self.preview_panel.setMinimumHeight(320)
            if self.results_panel is not None:
                self.results_panel.setMinimumWidth(0)
                self.results_panel.setMinimumHeight(260)
        else:
            self._layout_wide_controls(controls)
            splitter.setOrientation(Qt.Orientation.Horizontal)
            splitter.setSizes([2, 1])
            self.preview_panel.setMinimumHeight(380)
            if self.results_panel is not None:
                self.results_panel.setMinimumWidth(280)
                self.results_panel.setMinimumHeight(0)

    def _layout_wide_controls(self, controls: QGridLayout) -> None:
        controls.addWidget(self.open_button, 0, 0)
        controls.addWidget(self.model_label, 0, 1, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.model_combo, 0, 2)
        controls.addWidget(self.device_label, 0, 3, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.device_combo, 0, 4)
        controls.addWidget(self.topk_label, 0, 5, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.topk_spin, 0, 6)
        controls.addWidget(self.predict_pretrained_button, 0, 7)

        controls.addWidget(self.check_weights_button, 1, 0)
        controls.addWidget(self.download_weights_button, 1, 1, 1, 2)
        controls.addWidget(self.import_weights_button, 1, 3, 1, 2)
        controls.addWidget(self.weight_status_label, 1, 5, 1, 3)

        controls.addWidget(self.browse_checkpoint_button, 2, 0)
        controls.addWidget(self.predict_checkpoint_button, 2, 1, 1, 2)
        controls.addWidget(self.checkpoint_label, 2, 3, 1, 5)

        controls.setColumnStretch(2, 1)
        controls.setColumnStretch(7, 1)

    def _layout_compact_controls(self, controls: QGridLayout) -> None:
        controls.addWidget(self.open_button, 0, 0)
        controls.addWidget(self.predict_pretrained_button, 0, 1, 1, 3)

        controls.addWidget(self.model_label, 1, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.model_combo, 1, 1, 1, 3)
        controls.addWidget(self.device_label, 2, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.device_combo, 2, 1)
        controls.addWidget(self.topk_label, 2, 2, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.topk_spin, 2, 3)

        controls.addWidget(self.check_weights_button, 3, 0)
        controls.addWidget(self.download_weights_button, 3, 1)
        controls.addWidget(self.import_weights_button, 3, 2, 1, 2)
        controls.addWidget(self.weight_status_label, 4, 0, 1, 4)

        controls.addWidget(self.browse_checkpoint_button, 5, 0)
        controls.addWidget(self.predict_checkpoint_button, 5, 1, 1, 3)
        controls.addWidget(self.checkpoint_label, 6, 0, 1, 4)

        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(3, 1)

    def _reset_grid(self, controls: QGridLayout) -> None:
        for index in range(8):
            controls.setColumnStretch(index, 0)
            controls.setColumnMinimumWidth(index, 0)
        for index in range(8):
            controls.setRowStretch(index, 0)

    def open_image(self) -> None:
        patterns = " ".join(f"*{extension}" for extension in self.config.image_extensions)
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "打开图片",
            "",
            f"图像文件 ({patterns});;所有文件 (*.*)",
        )
        if not path:
            return

        image_path = Path(path)
        try:
            started_at = time.perf_counter()
            pixmap = self._pixmap_from_path(image_path)
            preview_ms = _elapsed_ms(started_at)
        except ValueError as exc:
            QMessageBox.critical(self, "打开失败", with_help(exc, DATA_AND_FILES))
            return

        self.image_path = image_path
        self.preview_panel.set_pixmap(pixmap)
        self.last_timing_text = f"预览打开 {preview_ms:.1f} ms"
        self.results_list.clear()
        self._update_info()
        self._update_action_states()
        self._set_status(f"图像已就绪（{self.last_timing_text}）。")

    def browse_checkpoint(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择本地分类模型文件",
            "",
            "PyTorch 模型 (*.pt *.pth);;所有文件 (*.*)",
        )
        if not path:
            return
        self.checkpoint_path = Path(path)
        self._set_path_label(self.checkpoint_label, "模型文件", self.checkpoint_path)
        self._update_action_states()

    def check_weight_status(self) -> None:
        try:
            info = self.service.pretrained_weight_status(self.current_model_name())[0]
        except Exception as exc:
            self.weight_status_label.setText(f"无法读取权重状态：{exc}")
            return
        self._show_weight_status(info)

    def _on_model_changed(self, _index: int) -> None:
        self.results_list.clear()
        self.last_timing_text = None
        self.check_weight_status()
        self._update_info()

    def download_pretrained_weight(self) -> None:
        model_name = self.current_model_name()
        try:
            info = self.service.pretrained_weight_status(model_name)[0]
        except Exception as exc:
            QMessageBox.critical(self, "下载失败", with_help(exc, MODELS_AND_WEIGHTS))
            return
        self._run_task(
            task=lambda: self.service.download_pretrained_weight(
                model_name,
                progress_callback=self._download_progress_callback(
                    f"正在下载 {model_name} 权重",
                    info.local_path,
                ),
            ),
            on_success=lambda value: self._on_weight_ready(value, "下载"),
            busy_text=f"正在下载 {model_name} 权重...",
            progress_text=f"正在下载 {model_name} 权重...\n保存路径：{info.local_path}",
            error_title="下载失败",
            error_category=MODELS_AND_WEIGHTS,
        )

    def import_local_weight(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择本地预训练权重",
            "",
            "PyTorch 权重 (*.pth *.pt);;所有文件 (*.*)",
        )
        if not path:
            return
        model_name = self.current_model_name()
        self._run_task(
            task=lambda: self.service.import_pretrained_weight(model_name, path),
            on_success=lambda value: self._on_weight_ready(value, "导入"),
            busy_text=f"正在导入 {model_name} 权重...",
            error_title="导入失败",
            error_category=MODELS_AND_WEIGHTS,
        )

    def predict_pretrained(self) -> None:
        if self.image_path is None:
            QMessageBox.information(self, "没有图片", "请先打开一张图片。")
            return
        model_name = self.current_model_name()
        image_path = self.image_path
        topk = self.topk_spin.value()
        device = self.device_combo.currentText()
        model_key = (model_name, device)
        first_load = model_key not in self._warmed_pretrained_keys
        self._run_task(
            task=lambda: self._predict_pretrained(model_name, image_path, topk, device),
            on_success=lambda value: self._show_pretrained_prediction(value, model_key, first_load),
            busy_text=(
                f"首次加载 {model_name} 模型，可能需要数秒..."
                if first_load
                else f"正在使用已缓存的 {model_name} 模型预测..."
            ),
            error_title="预测失败",
            error_category=MODELS_AND_WEIGHTS,
        )

    def predict_checkpoint(self) -> None:
        if self.image_path is None:
            QMessageBox.information(self, "没有图片", "请先打开一张图片。")
            return
        if self.checkpoint_path is None:
            QMessageBox.information(self, "没有模型文件", "请先选择一个训练好的模型文件。")
            return
        image_path = self.image_path
        checkpoint_path = self.checkpoint_path
        topk = self.topk_spin.value()
        device = self.device_combo.currentText()
        self._run_task(
            task=lambda: self._predict_checkpoint(checkpoint_path, image_path, topk, device),
            on_success=self._show_prediction,
            busy_text="正在使用自定义模型预测...",
            error_title="预测失败",
            error_category=MODELS_AND_WEIGHTS,
        )

    def select_training_dataset(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择分类数据集",
            str(self.config.dataset_dir),
        )
        if not directory:
            return
        dataset_dir = Path(directory)
        self.train_dataset_edit.setText(str(dataset_dir))
        self._set_train_log(f"已选择数据集：\n{dataset_dir}\n\n验证通过后即可开始训练。")
        self.train_status_label.setText("数据集已选择。")

    def validate_training_dataset(self) -> None:
        dataset_dir = self._training_dataset_path()
        if dataset_dir is None:
            QMessageBox.information(self, "没有数据集", "请先选择分类数据集目录。")
            return
        self._run_task(
            task=lambda: self.service.validate_dataset(dataset_dir),
            on_success=self._show_training_validation,
            busy_text="正在验证分类数据集...",
            progress_text=f"正在验证分类数据集...\n路径：{dataset_dir}",
            error_title="数据集验证失败",
            error_category=DATASETS_AND_TRAINING,
        )

    def create_sample_training_dataset(self) -> None:
        target = self.config.dataset_dir / "quickstart"
        try:
            dataset_dir = create_classification_sample_dataset(target)
        except Exception as exc:
            QMessageBox.critical(self, "示例数据创建失败", with_help(exc, DATASETS_AND_TRAINING))
            return
        self.train_dataset_edit.setText(str(dataset_dir))
        self._set_train_log(
            f"已创建分类示例数据集：\n{dataset_dir}\n\n"
            "该数据集只用于验证训练流程，不代表真实模型效果。"
        )
        self.train_status_label.setText("示例数据已创建，可以验证后试跑。")

    def check_training_environment(self) -> None:
        self._run_task(
            task=lambda: inspect_training_environment(self.config.runs_dir),
            on_success=self._on_environment_checked,
            busy_text="正在检查训练环境...",
            progress_text="正在检查 Torch、加速器和输出磁盘...",
            error_title="环境检查失败",
            error_category=DATASETS_AND_TRAINING,
        )

    def _on_environment_checked(self, value: object) -> None:
        report = value
        if not isinstance(report, TrainingEnvironmentReport):
            return
        self._recommended_batch_size = report.recommended_batch_size
        self.environment_label.setText(report.to_text())
        self.environment_label.setToolTip(report.to_text())
        self.apply_batch_button.setEnabled(True)
        self._completion_status = "训练环境检查完成。" if report.ok else "训练环境存在需要处理的问题。"

    def apply_recommended_batch(self) -> None:
        if self._recommended_batch_size is not None:
            self.train_batch_spin.setValue(self._recommended_batch_size)

    def start_training(self) -> None:
        dataset_dir = self._training_dataset_path()
        if dataset_dir is None:
            QMessageBox.information(self, "没有数据集", "请先选择分类数据集目录。")
            return

        try:
            run_name = validate_run_name(
                self.train_run_name_edit.text().strip() or "classification_train",
                field_name="训练运行名",
            )
        except ValueError as exc:
            QMessageBox.critical(self, "运行名称无效", with_help(exc, DATASETS_AND_TRAINING))
            return

        job = ClassificationTrainingConfig(
            model_name=self.train_model_combo.currentText(),
            dataset_dir=dataset_dir,
            output_dir=self.config.runs_dir,
            run_name=run_name,
            epochs=self.train_epochs_spin.value(),
            image_size=self.train_image_size_spin.value(),
            batch_size=self.train_batch_spin.value(),
            device=self.train_device_combo.currentText(),
            learning_rate=float(self.train_lr_spin.value()),
            workers=0,
            pretrained=self.train_pretrained_check.isChecked(),
            freeze_backbone=self.train_freeze_check.isChecked(),
        )
        self._set_train_log(
            "训练开始：\n"
            f"数据集：{dataset_dir}\n"
            f"模型：{job.model_name}\n"
            f"设备：{job.device}\n"
            f"Epochs：{job.epochs}\n"
            f"输出目录：{job.output_dir / job.run_name}\n"
        )
        self.train_status_label.setText("训练运行中...")
        self.train_progress.setRange(0, job.epochs)
        self.train_progress.setValue(0)
        self.train_progress.setFormat("准备训练 %v/%m")
        self.train_metric_label.setText("正在验证数据集并初始化模型...")
        self._start_training_process(job)

    def stop_training(self) -> None:
        if not self._training_running() or self.training_process is None:
            return
        self._training_stop_requested = True
        self.train_status_label.setText("正在停止训练...")
        self._append_train_log("\n已请求停止训练。\n")
        self.training_process.terminate()
        QTimer.singleShot(5000, self._force_stop_training)
        self._update_action_states()

    def current_model_name(self) -> str:
        return self.model_combo.currentText()

    def shutdown(self) -> None:
        if self._training_running() and self.training_process is not None:
            self.training_process.terminate()
            if not self.training_process.waitForFinished(3000):
                self.training_process.kill()
                self.training_process.waitForFinished(1000)
        self.tasks.shutdown()

    def _training_dataset_path(self) -> Path | None:
        text = self.train_dataset_edit.text().strip()
        if not text:
            return None
        return Path(text).expanduser()

    def _predict_pretrained(
        self,
        model_name: str,
        image_path: Path,
        topk: int,
        device: str,
    ) -> PredictionPayload:
        started_at = time.perf_counter()
        result = self.service.predict_with_pretrained(
            model_name=model_name,
            image_path=image_path,
            topk=topk,
            device=device,
        )
        return PredictionPayload(result=result, total_ms=_elapsed_ms(started_at))

    def _predict_checkpoint(
        self,
        checkpoint_path: Path,
        image_path: Path,
        topk: int,
        device: str,
    ) -> PredictionPayload:
        started_at = time.perf_counter()
        result = self.service.predict_with_checkpoint(
            model_path=checkpoint_path,
            image_path=image_path,
            topk=topk,
            device=device,
        )
        return PredictionPayload(result=result, total_ms=_elapsed_ms(started_at))

    def _show_training_validation(self, value: object) -> None:
        report = cast(DatasetValidationReport, value)
        self._set_train_log(report.to_text())
        status = "数据集验证通过。" if report.ok else "数据集验证失败。"
        self.train_status_label.setText(status)
        self._completion_status = status

    def _start_training_process(self, job: ClassificationTrainingConfig) -> None:
        if self._training_running():
            QMessageBox.information(self, "正在训练", "当前分类训练任务仍在运行。")
            return
        command = self.service.build_runner_command(job)
        process = QProcess(self)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        project_root = Path(__file__).resolve().parents[4]
        process.setWorkingDirectory(str(project_root))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        src_dir = project_root / "src"
        existing_pythonpath = environment.value("PYTHONPATH")
        environment.insert(
            "PYTHONPATH",
            str(src_dir) if not existing_pythonpath else str(src_dir) + os.pathsep + existing_pythonpath,
        )
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PYTHONNOUSERSITE", "1")
        environment.insert("ULTRALYTICS_SAFE_LOAD", "true")
        environment.insert("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "1")
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read_training_output)
        process.finished.connect(self._on_training_process_finished)
        process.errorOccurred.connect(self._on_training_process_error)
        self.training_process = process
        self._active_training_job = job
        self._training_output_buffer = ""
        self._training_stop_requested = False
        self._set_busy(True)
        self._set_status("分类训练运行中...")
        process.start()

    def _read_training_output(self) -> None:
        if self.training_process is None:
            return
        text = bytes(self.training_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._training_output_buffer += text
        lines = self._training_output_buffer.split("\n")
        self._training_output_buffer = lines.pop()
        for line in lines:
            self._handle_training_line(line.rstrip("\r"))

    def _handle_training_line(self, line: str) -> None:
        if line.startswith("VW_METRIC "):
            try:
                metrics = json.loads(line[len("VW_METRIC ") :])
                epoch = int(metrics["epoch"])
                epochs = int(metrics["epochs"])
                loss = float(metrics["train_loss"])
                accuracy = float(metrics["val_accuracy"])
                best = float(metrics["best_accuracy"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self._append_train_log(line + "\n")
                return
            self.train_progress.setRange(0, epochs)
            self.train_progress.setValue(epoch)
            self.train_progress.setFormat(f"第 {epoch}/{epochs} 轮")
            self.train_metric_label.setText(
                f"训练损失：{loss:.4f} | 验证准确率：{accuracy:.2%} | 最佳准确率：{best:.2%}"
            )
            self._append_train_log(
                f"Epoch {epoch}/{epochs} | loss={loss:.4f} | val_acc={accuracy:.2%} | best={best:.2%}\n"
            )
            return
        if line:
            self._append_train_log(line + "\n")

    def _on_training_process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        if self._training_output_buffer:
            self._handle_training_line(self._training_output_buffer)
            self._training_output_buffer = ""
        job = self._active_training_job
        stopped = self._training_stop_requested
        process = self.training_process
        self.training_process = None
        self._active_training_job = None
        if process is not None:
            process.deleteLater()
        self._set_busy(False)
        if stopped:
            self.train_status_label.setText("训练已停止。")
            self.train_metric_label.setText("训练由用户停止，已保留已写入的 checkpoint。")
            self._set_status("分类训练已停止。")
            return
        if exit_code != 0 or job is None:
            self.train_status_label.setText(f"训练失败，退出码：{exit_code}")
            self._set_status("分类训练失败。")
            QMessageBox.critical(
                self,
                "训练失败",
                with_help("分类训练进程失败，请查看训练日志。", DATASETS_AND_TRAINING),
            )
            return
        best_path = job.output_dir / job.run_name / "best.pt"
        self.train_progress.setValue(job.epochs)
        self.train_status_label.setText(f"训练完成：{_shorten_path(best_path, 70)}")
        self.train_metric_label.setText(f"训练完成，最佳模型：{best_path}")
        self._set_status(f"分类训练完成：{best_path.name}")

    def _on_training_process_error(self, error: QProcess.ProcessError) -> None:
        if self.training_process is None:
            return
        message = self.training_process.errorString()
        self._append_train_log(f"\n训练进程错误：{message}\n")
        if error == QProcess.ProcessError.FailedToStart:
            process = self.training_process
            self.training_process = None
            self._active_training_job = None
            self._set_busy(False)
            self.train_status_label.setText("训练进程无法启动。")
            self._set_status("分类训练启动失败。")
            process.deleteLater()
            QMessageBox.critical(
                self,
                "训练启动失败",
                with_help(f"无法启动分类训练进程：{message}", DATASETS_AND_TRAINING),
            )

    def _force_stop_training(self) -> None:
        if self._training_running() and self.training_process is not None:
            self.training_process.kill()

    def _training_running(self) -> bool:
        return self.training_process is not None and self.training_process.state() != QProcess.ProcessState.NotRunning

    def _set_train_log(self, text: str) -> None:
        self.train_log.setPlainText(text)
        self.train_log.moveCursor(QTextCursor.MoveOperation.End)

    def _append_train_log(self, text: str) -> None:
        self.train_log.moveCursor(QTextCursor.MoveOperation.End)
        self.train_log.insertPlainText(text)
        self.train_log.ensureCursorVisible()

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
        self.busy_label.setText(progress_text or "")
        self.busy_label.setVisible(progress_text is not None)
        self.busy_progress.setVisible(progress_text is not None)
        self._set_busy(True)
        self._set_status(busy_text)

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

    def _on_weight_ready(self, value: object, action: str) -> None:
        info = cast(PretrainedWeightInfo, value)
        self._show_weight_status(info)
        self._warmed_pretrained_keys = {
            key for key in self._warmed_pretrained_keys if key[0] != info.model_name
        }
        title = f"{action}完成"
        self._completion_status = f"{title}：{info.local_path}"
        QMessageBox.information(self, title, f"预训练权重已{action}：\n{info.local_path}")

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

    def _show_weight_status(self, info: PretrainedWeightInfo) -> None:
        if info.exists:
            self._set_path_label(self.weight_status_label, "本地权重", info.local_path)
            self.download_weights_button.setText("下载权重")
        elif issue := model_file_issue(info.local_path):
            self._set_path_label(self.weight_status_label, f"本地权重{issue}，可重新下载", info.local_path)
            self.download_weights_button.setText("重新下载权重")
        else:
            self._set_path_label(self.weight_status_label, "未找到本地权重，预期路径", info.local_path)
            self.download_weights_button.setText("下载权重")

    def _show_pretrained_prediction(
        self,
        value: object,
        model_key: tuple[str, str],
        first_load: bool,
    ) -> None:
        self._show_prediction(value)
        self._warmed_pretrained_keys.add(model_key)
        cache_text = "模型已缓存，后续预测会更快" if first_load else "已使用缓存模型"
        self._completion_status = f"预测完成，{cache_text}（{self.last_timing_text}）。"

    def _show_prediction(self, value: object) -> None:
        payload = cast(PredictionPayload, value)
        result = payload.result
        self.results_list.clear()
        for index, item in enumerate(result.predictions, start=1):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(8, 6, 8, 6)
            layout.setSpacing(10)

            rank_label = QLabel(f"{index}")
            rank_label.setMinimumWidth(22)
            label = QLabel(item.label)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            score = QProgressBar()
            score.setRange(0, 1000)
            score.setValue(max(0, min(1000, int(item.score * 1000))))
            score.setFormat(f"{item.score * 100:.2f}%")
            score.setMinimumWidth(140)

            layout.addWidget(rank_label)
            layout.addWidget(label, 1)
            layout.addWidget(score)

            list_item = QListWidgetItem()
            list_item.setSizeHint(row.sizeHint())
            self.results_list.addItem(list_item)
            self.results_list.setItemWidget(list_item, row)

        prepare_ms = max(0.0, payload.total_ms - result.inference_ms)
        self.last_timing_text = (
            f"加载/准备 {prepare_ms:.1f} ms | "
            f"推理 {result.inference_ms:.1f} ms | "
            f"总计 {payload.total_ms:.1f} ms"
        )
        logger.info("Classification prediction timings: %s", self.last_timing_text)
        self._completion_status = f"预测完成（{self.last_timing_text}）。"
        self._update_info(result)

    def _pixmap_from_path(self, image_path: Path) -> QPixmap:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            raise ValueError(f"Cannot preview image file: {image_path}")
        return pixmap

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if not busy:
            self.busy_label.setVisible(False)
            self.busy_progress.setVisible(False)
        self._update_action_states()

    def _update_action_states(self) -> None:
        training_running = self._training_running()
        has_image = self.image_path is not None
        has_checkpoint = self.checkpoint_path is not None
        self.open_button.setEnabled(not self._busy)
        self.predict_pretrained_button.setEnabled(has_image and not self._busy)
        self.check_weights_button.setEnabled(not self._busy)
        self.download_weights_button.setEnabled(not self._busy)
        self.import_weights_button.setEnabled(not self._busy)
        self.browse_checkpoint_button.setEnabled(not self._busy)
        self.predict_checkpoint_button.setEnabled(has_image and has_checkpoint and not self._busy)
        self.model_combo.setEnabled(not self._busy)
        self.device_combo.setEnabled(not self._busy)
        self.topk_spin.setEnabled(not self._busy)
        self.jump_to_training_button.setEnabled(not self._busy)
        self.check_environment_button.setEnabled(not self._busy and not training_running)
        self.apply_batch_button.setEnabled(
            not self._busy and not training_running and self._recommended_batch_size is not None
        )
        has_training_dataset = bool(self.train_dataset_edit.text().strip())
        for widget in (
            self.train_dataset_edit,
            self.select_dataset_button,
            self.validate_dataset_button,
            self.sample_dataset_button,
            self.train_model_combo,
            self.train_device_combo,
            self.train_epochs_spin,
            self.train_image_size_spin,
            self.train_batch_spin,
            self.train_lr_spin,
            self.train_run_name_edit,
            self.train_pretrained_check,
            self.train_freeze_check,
        ):
            widget.setEnabled(not self._busy)
        self.start_training_button.setEnabled(not self._busy and not training_running and has_training_dataset)
        self.stop_training_button.setEnabled(training_running)

    def _update_info(self, result: PredictionResult | None = None) -> None:
        parts = []
        if self.image_path:
            parts.append(f"图片：{_shorten_path(self.image_path, 70)}")
        if result is not None:
            parts.append(f"模型：{result.model_name}")
        else:
            parts.append(f"模型：{self.current_model_name()}")
        if self.last_timing_text:
            parts.append(self.last_timing_text)
        self.info_label.setText(" | ".join(parts))
        if self.image_path:
            self.info_label.setToolTip(str(self.image_path))

    def _set_status(self, text: str) -> None:
        self.status_changed.emit(text)

    def _set_path_label(self, label: QLabel, prefix: str, path: Path) -> None:
        label.setText(f"{prefix}: {_shorten_path(path, 68)}")
        label.setToolTip(str(path))


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0


def _shorten_path(path: Path, max_chars: int) -> str:
    text = str(path)
    if len(text) <= max_chars:
        return text
    head = max(12, max_chars // 2 - 3)
    tail = max(18, max_chars - head - 3)
    return f"{text[:head]}...{text[-tail:]}"
