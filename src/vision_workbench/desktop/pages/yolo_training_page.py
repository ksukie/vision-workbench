"""Qt page for YOLO26 training."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Signal, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vision_workbench.model_files import validate_complete_model_file
from vision_workbench.runtime_security import validate_run_name
from vision_workbench.sample_data import create_yolo_sample_dataset
from vision_workbench.runtime_diagnostics import TrainingEnvironmentReport, inspect_training_environment
from vision_workbench.troubleshooting import DATASETS_AND_TRAINING, MODELS_AND_WEIGHTS, help_hint, with_help
from yolo26_training.api import create_yolo26_training_service
from yolo26_training.application import Yolo26TrainingService
from yolo26_training.configuration import Yolo26TrainingConfig
from yolo26_training.domain import TrainingJobConfig

from ..task_runner import QtTaskRunner
from ..widgets import SELECTED_DISPLAY_ROLE
from ..widgets import NoWheelComboBox as QComboBox
from ..widgets import NoWheelSpinBox as QSpinBox
from ..widgets import SectionCard, make_button, style_form_label


TASK_LABELS = {
    "detect": "目标检测",
    "segment": "实例分割",
    "semantic": "语义分割",
}


@dataclass(frozen=True)
class TrainingProcessPayload:
    command: list[str]
    started_ms: float


@dataclass(frozen=True)
class ModelManifestPayload:
    entry_count: int
    total_ms: float


class YoloTrainingPage(QWidget):
    """Native Qt implementation of the YOLO26 training workflow."""

    status_changed = Signal(str)

    def __init__(
        self,
        service: Yolo26TrainingService | None = None,
        config: Yolo26TrainingConfig = Yolo26TrainingConfig(),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.service = service or create_yolo26_training_service(config)
        self.tasks = QtTaskRunner(self)
        self.process = None  # type: QProcess | None
        self._process_started_at = None  # type: float | None
        self._active_job = None  # type: TrainingJobConfig | None
        self._last_completed_job = None  # type: TrainingJobConfig | None
        self.models = []  # type: list[Path]
        self._recommended_batch_size = None  # type: int | None

        self.task_label = QLabel("任务")
        self.task_combo = QComboBox()
        for task in config.task_options:
            self.task_combo.addItem(TASK_LABELS.get(task, task), task)
        self.task_combo.setMinimumWidth(132)

        self.dataset_label = QLabel("数据集")
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText("选择 YOLO data.yaml")
        self.dataset_edit.setMinimumWidth(320)
        self.choose_dataset_button = make_button("选择 data.yaml", primary=True)
        self.validate_button = make_button("验证数据集")
        self.sample_dataset_button = make_button("创建示例数据")
        self.allow_missing_labels_check = QCheckBox("允许缺失标签")
        self.allow_missing_labels_check.setToolTip(
            "仅当无标签图片确实代表背景/负样本时启用；否则会掩盖数据集标注缺失。"
        )

        self.model_label = QLabel("预训练权重")
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(260)
        self.browse_model_button = make_button("选择本地权重文件")
        self.browse_model_button.setToolTip("从本地磁盘选择一个 .pt 训练初始权重")
        self.refresh_models_button = make_button("查找权重")

        self.epochs_label = QLabel("轮数")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(config.default_epochs)
        self.epochs_spin.setToolTip("完整遍历训练集的次数。新手可先用 10～30 轮确认流程。")

        self.image_size_label = QLabel("尺寸")
        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(64, 4096)
        self.image_size_spin.setSingleStep(32)
        self.image_size_spin.setValue(config.default_image_size)
        self.image_size_spin.setToolTip("输入图像边长。数值越大越占显存，遇到显存不足时优先降低。")

        self.batch_label = QLabel("批量")
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(config.default_batch_size)
        self.batch_spin.setToolTip("每次送入模型的图片数。显存不足时可从 16 降到 8、4 或 1。")

        self.workers_label = QLabel("线程")
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 256)
        self.workers_spin.setValue(config.default_workers)
        self.workers_spin.setToolTip("数据加载进程数。Windows 出现卡住时可改为 0。")

        self.device_label = QLabel("设备")
        self.device_combo = QComboBox()
        self.device_combo.addItems(config.device_options)
        self.device_combo.setCurrentText("auto")
        self.device_combo.setToolTip("auto 会优先使用可用加速器；排查环境问题时可选 cpu。")

        self.run_name_label = QLabel("运行名")
        self.run_name_edit = QLineEdit()
        self.run_name_edit.setPlaceholderText("留空时使用数据集和权重名")
        self.run_name_edit.setToolTip("仅填写名称，不要输入目录、斜杠或 ..。")
        self.resume_check = QCheckBox("继续训练")
        self.resume_check.setToolTip("从兼容的上次训练状态继续；首次训练请保持关闭。")

        self.parameter_hint = QLabel(
            "新手建议：先验证数据集，用较小轮数完成一次试跑。若显存不足，优先降低批量和图像尺寸。"
        )
        self.parameter_hint.setObjectName("ParameterHint")
        self.parameter_hint.setWordWrap(True)
        self.check_environment_button = make_button("检查训练环境")
        self.apply_batch_button = make_button("应用推荐批量")
        self.apply_batch_button.setEnabled(False)
        self.environment_label = QLabel("尚未检查训练环境。")
        self.environment_label.setObjectName("MutedText")
        self.environment_label.setWordWrap(True)

        for label in (
            self.task_label,
            self.dataset_label,
            self.model_label,
            self.epochs_label,
            self.image_size_label,
            self.batch_label,
            self.workers_label,
            self.device_label,
            self.run_name_label,
        ):
            style_form_label(label)

        for label, control in (
            (self.task_label, self.task_combo),
            (self.dataset_label, self.dataset_edit),
            (self.model_label, self.model_combo),
            (self.epochs_label, self.epochs_spin),
            (self.image_size_label, self.image_size_spin),
            (self.batch_label, self.batch_spin),
            (self.workers_label, self.workers_spin),
            (self.device_label, self.device_combo),
            (self.run_name_label, self.run_name_edit),
        ):
            label.setBuddy(control)
            control.setAccessibleName(label.text())

        self.start_button = make_button("开始训练", primary=True)
        self.stop_button = make_button("停止训练", danger=True)
        self.register_best_button = make_button("加入模型库")
        self.register_best_button.setToolTip("将最近一次成功训练的 weights/best.pt 复制到当前任务的自定义模型目录")
        self.open_runs_button = make_button("打开输出目录")

        for button in (
            self.choose_dataset_button,
            self.validate_button,
            self.sample_dataset_button,
            self.browse_model_button,
            self.refresh_models_button,
            self.start_button,
            self.stop_button,
            self.register_best_button,
            self.open_runs_button,
        ):
            button.setMinimumWidth(112)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(360)
        self.log_text.setPlaceholderText("数据集验证和训练日志会显示在这里。")

        self.info_label = QLabel("")
        self.info_label.setObjectName("MutedText")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._build_ui()
        self._connect_signals()
        self.refresh_models()
        self._append_log("选择 data.yaml，验证数据集后开始训练。\n")
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

        title = QLabel("YOLO 模型训练")
        title.setObjectName("PageTitle")
        subtitle = QLabel("验证 YOLO data.yaml，并启动检测、实例分割或语义分割训练任务。")
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        dataset_card = SectionCard("数据集")
        dataset_grid = QGridLayout()
        dataset_grid.setHorizontalSpacing(10)
        dataset_grid.setVerticalSpacing(10)
        dataset_grid.addWidget(self.task_label, 0, 0, Qt.AlignmentFlag.AlignVCenter)
        dataset_grid.addWidget(self.task_combo, 0, 1)
        dataset_grid.addWidget(self.dataset_label, 1, 0, Qt.AlignmentFlag.AlignVCenter)
        dataset_grid.addWidget(self.dataset_edit, 1, 1, 1, 3)
        dataset_grid.addWidget(self.choose_dataset_button, 1, 4)
        dataset_grid.addWidget(self.validate_button, 1, 5)
        dataset_grid.addWidget(self.allow_missing_labels_check, 2, 1, 1, 2)
        dataset_grid.addWidget(self.sample_dataset_button, 2, 4, 1, 2)
        dataset_grid.setColumnStretch(1, 1)
        dataset_grid.setColumnStretch(3, 1)
        dataset_card.content_layout.addLayout(dataset_grid)
        layout.addWidget(dataset_card)

        params_card = SectionCard("训练参数")
        params_grid = QGridLayout()
        params_grid.setHorizontalSpacing(10)
        params_grid.setVerticalSpacing(10)
        params_grid.addWidget(self.model_label, 0, 0, Qt.AlignmentFlag.AlignVCenter)
        params_grid.addWidget(self.model_combo, 0, 1, 1, 3)
        params_grid.addWidget(self.browse_model_button, 0, 4)
        params_grid.addWidget(self.refresh_models_button, 0, 5)
        params_grid.addWidget(self.epochs_label, 1, 0, Qt.AlignmentFlag.AlignVCenter)
        params_grid.addWidget(self.epochs_spin, 1, 1)
        params_grid.addWidget(self.image_size_label, 1, 2, Qt.AlignmentFlag.AlignVCenter)
        params_grid.addWidget(self.image_size_spin, 1, 3)
        params_grid.addWidget(self.batch_label, 1, 4, Qt.AlignmentFlag.AlignVCenter)
        params_grid.addWidget(self.batch_spin, 1, 5)
        params_grid.addWidget(self.workers_label, 2, 0, Qt.AlignmentFlag.AlignVCenter)
        params_grid.addWidget(self.workers_spin, 2, 1)
        params_grid.addWidget(self.device_label, 2, 2, Qt.AlignmentFlag.AlignVCenter)
        params_grid.addWidget(self.device_combo, 2, 3)
        params_grid.addWidget(self.run_name_label, 3, 0, Qt.AlignmentFlag.AlignVCenter)
        params_grid.addWidget(self.run_name_edit, 3, 1, 1, 3)
        params_grid.addWidget(self.resume_check, 3, 4)
        params_grid.setColumnStretch(1, 1)
        params_grid.setColumnStretch(3, 1)
        params_card.content_layout.addLayout(params_grid)
        params_card.content_layout.addWidget(self.parameter_hint)
        environment_row = QHBoxLayout()
        environment_row.addWidget(self.check_environment_button)
        environment_row.addWidget(self.apply_batch_button)
        environment_row.addWidget(self.environment_label, 1)
        params_card.content_layout.addLayout(environment_row)
        layout.addWidget(params_card)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.stop_button)
        action_row.addWidget(self.register_best_button)
        action_row.addWidget(self.open_runs_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        log_card = SectionCard("训练日志")
        log_card.content_layout.addWidget(self.log_text)
        layout.addWidget(log_card, 1)
        layout.addWidget(self.info_label)

        scroll_area.setWidget(content)
        root_layout.addWidget(scroll_area)

    def _connect_signals(self) -> None:
        self.task_combo.currentIndexChanged.connect(self.refresh_models)
        self.choose_dataset_button.clicked.connect(self.choose_dataset)
        self.validate_button.clicked.connect(self.validate_dataset)
        self.sample_dataset_button.clicked.connect(self.create_sample_dataset)
        self.browse_model_button.clicked.connect(self.choose_model)
        self.refresh_models_button.clicked.connect(self.refresh_model_catalog)
        self.start_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.stop_training)
        self.register_best_button.clicked.connect(self.register_best_weight)
        self.open_runs_button.clicked.connect(self.open_runs_dir)
        self.dataset_edit.textChanged.connect(self._update_action_states)
        self.check_environment_button.clicked.connect(self.check_training_environment)
        self.apply_batch_button.clicked.connect(self.apply_recommended_batch)

    def current_task(self) -> str:
        return str(self.task_combo.currentData() or "detect")

    def choose_dataset(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择 YOLO data.yaml",
            str(self.config.dataset_dir),
            "YAML 数据集 (*.yaml *.yml);;所有文件 (*.*)",
        )
        if path:
            self.dataset_edit.setText(path)
            self._update_info()

    def choose_model(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择本地 YOLO26 预训练权重",
            str(self.config.model_dir_for_task(self.current_task())),
            "PyTorch 权重 (*.pt);;所有文件 (*.*)",
        )
        if not path:
            return
        model_path = Path(path)
        self._add_model_option(model_path)
        self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
        self._update_info()

    def create_sample_dataset(self) -> None:
        task = self.current_task()
        target = self.config.dataset_dir / f"quickstart_{task}"
        try:
            data_yaml = create_yolo_sample_dataset(target, task)
        except Exception as exc:
            QMessageBox.critical(self, "示例数据创建失败", with_help(exc, DATASETS_AND_TRAINING))
            return
        self.dataset_edit.setText(str(data_yaml))
        self._append_log(f"\n已创建 {TASK_LABELS.get(task, task)} 示例数据集：\n{data_yaml}\n")
        self._set_status("示例数据集已创建，可以先验证数据集。")
        self._update_info("示例数据集已创建。")

    def check_training_environment(self) -> None:
        accepted = self.tasks.run(
            task=lambda: inspect_training_environment(self.config.runs_dir),
            on_success=self._on_environment_checked,
            on_error=self._on_environment_check_error,
        )
        if not accepted:
            QMessageBox.information(self, "正在处理", "请等待当前任务完成。")
            return
        self.environment_label.setText("正在检查 Torch、加速器和磁盘空间...")
        self.check_environment_button.setEnabled(False)
        self._set_status("正在检查训练环境...")

    def _on_environment_checked(self, value: object) -> None:
        report = value
        if not isinstance(report, TrainingEnvironmentReport):
            return
        self._recommended_batch_size = report.recommended_batch_size
        self.environment_label.setText(report.to_text())
        self.environment_label.setToolTip(report.to_text())
        self.apply_batch_button.setEnabled(True)
        self.check_environment_button.setEnabled(True)
        self._set_status("训练环境检查完成。" if report.ok else "训练环境存在需要处理的问题。")

    def _on_environment_check_error(self, exc: Exception) -> None:
        self.check_environment_button.setEnabled(True)
        self.environment_label.setText(f"环境检查失败：{exc}")
        self._set_status("训练环境检查失败。")

    def apply_recommended_batch(self) -> None:
        if self._recommended_batch_size is not None:
            self.batch_spin.setValue(self._recommended_batch_size)

    def refresh_models(self) -> None:
        task = self.current_task()
        current = self.current_model_path()
        self.models = self.service.list_models(task)

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for model_path in self.models:
            self._add_model_option(model_path)
        if self.model_combo.count() == 0:
            default_path = self.service.default_model(task)
            self._add_model_option(default_path)
        self.model_combo.blockSignals(False)

        if current is not None:
            for index in range(self.model_combo.count()):
                if Path(str(self.model_combo.itemData(index))) == current:
                    self.model_combo.setCurrentIndex(index)
                    break
        self._update_info()
        self._update_action_states()

    def refresh_model_catalog(self) -> None:
        refresher = getattr(self.service, "refresh_model_manifest", None)
        if not callable(refresher):
            self.refresh_models()
            return
        accepted = self.tasks.run(
            task=self._refresh_model_manifest,
            on_success=self._on_model_manifest_refreshed,
            on_error=self._on_model_manifest_error,
        )
        if not accepted:
            QMessageBox.information(self, "正在处理", "请等待当前任务完成。")
            return
        self._set_status("正在刷新模型目录...")

    def validate_dataset(self) -> bool:
        data_path = self.dataset_edit.text().strip()
        if not data_path:
            QMessageBox.information(self, "没有数据集", "请先选择 data.yaml。")
            return False
        report = self.service.validate_dataset(
            data_path,
            task=self.current_task(),
            allow_missing_labels=self.allow_missing_labels_check.isChecked(),
        )
        self._append_log("\n" + report.to_text() + "\n")
        status = "数据集验证通过。" if report.ok else "数据集验证失败。"
        self._set_status(status)
        self._update_info(status)
        if not report.ok:
            QMessageBox.critical(
                self,
                "数据集无效",
                with_help("数据集验证失败，请查看训练日志。", DATASETS_AND_TRAINING),
            )
        return report.ok

    def start_training(self) -> None:
        if self._training_running():
            QMessageBox.information(self, "正在训练", "当前训练任务仍在运行。")
            return
        if not self.validate_dataset():
            return
        try:
            job = self._current_job()
        except Exception as exc:
            QMessageBox.critical(self, "参数无效", with_help(exc, DATASETS_AND_TRAINING))
            return

        command = self.service.build_runner_command(job)
        self._append_log("\n开始训练：\n" + " ".join(command) + "\n\n")
        self._active_job = job
        self._last_completed_job = None
        self._start_process(command)

    def stop_training(self) -> None:
        if not self._training_running() or self.process is None:
            self._set_status("没有正在运行的训练任务。")
            return
        self._append_log("\n已请求停止训练。\n")
        self._set_status("正在停止训练...")
        self.process.terminate()
        self._update_action_states()

    def open_runs_dir(self) -> None:
        self.config.runs_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(self.config.runs_dir))
        else:
            self._append_log(f"\n输出目录：{self.config.runs_dir}\n")

    def register_best_weight(self) -> None:
        job = self._last_completed_job
        if job is None:
            QMessageBox.information(self, "没有可登记权重", "请先完成一次训练。")
            return
        run_dir = job.project_dir / job.run_name
        try:
            output_path = self.service.copy_best_weight(run_dir, task=job.task)
        except Exception as exc:
            QMessageBox.critical(self, "登记失败", with_help(exc, MODELS_AND_WEIGHTS))
            return

        self._append_log(f"\n最佳权重已加入模型库：{output_path}\n")
        self.refresh_models()
        self._select_model_path(output_path)
        self._set_status(f"最佳权重已加入模型库：{output_path.name}")
        self._update_info("最佳权重已加入模型库。")

    def current_model_path(self) -> Path | None:
        value = self.model_combo.currentData()
        if value:
            return Path(str(value))
        text = self.model_combo.currentText().strip()
        return Path(text) if text else None

    def shutdown(self) -> None:
        if self._training_running() and self.process is not None:
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()
                self.process.waitForFinished(1000)
        self.tasks.shutdown()

    def _refresh_model_manifest(self) -> ModelManifestPayload:
        started_at = time.perf_counter()
        refresher = getattr(self.service, "refresh_model_manifest")
        entry_count = int(refresher())
        return ModelManifestPayload(entry_count=entry_count, total_ms=(time.perf_counter() - started_at) * 1000.0)

    def _on_model_manifest_refreshed(self, payload: ModelManifestPayload) -> None:
        self.refresh_models()
        if payload.entry_count:
            self._set_status(f"模型目录已刷新：{payload.entry_count} 项（{payload.total_ms:.1f} ms）。")
        else:
            self._set_status(f"模型列表已重新扫描（{payload.total_ms:.1f} ms）。")

    def _on_model_manifest_error(self, exc: Exception) -> None:
        self._set_status("模型目录刷新失败。")
        QMessageBox.critical(self, "刷新模型目录失败", with_help(exc, MODELS_AND_WEIGHTS))

    def _current_job(self) -> TrainingJobConfig:
        data_path = Path(self.dataset_edit.text().strip())
        model_path = self.current_model_path()
        if model_path is None:
            raise FileNotFoundError("请选择预训练权重文件。")
        if not model_path.exists():
            raise FileNotFoundError(f"预训练权重文件不存在：{model_path}")
        validate_complete_model_file(model_path)
        run_name = validate_run_name(
            self.run_name_edit.text().strip() or f"{data_path.stem}_{model_path.stem}",
            field_name="训练运行名",
        )
        return TrainingJobConfig(
            task=self.current_task(),
            data_yaml=data_path,
            model_path=model_path,
            project_dir=self.config.runs_dir,
            run_name=run_name,
            epochs=int(self.epochs_spin.value()),
            image_size=int(self.image_size_spin.value()),
            batch_size=int(self.batch_spin.value()),
            device=self.device_combo.currentText(),
            workers=int(self.workers_spin.value()),
            resume=self.resume_check.isChecked(),
            allow_missing_labels=self.allow_missing_labels_check.isChecked(),
        )

    def _start_process(self, command: list[str]) -> None:
        process = QProcess(self)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setWorkingDirectory(str(self.config.yolo26_source_dir.parents[1]))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

        environment = QProcessEnvironment.systemEnvironment()
        src_dir = Path(__file__).resolve().parents[3]
        existing_pythonpath = environment.value("PYTHONPATH")
        environment.insert(
            "PYTHONPATH",
            str(src_dir) if not existing_pythonpath else str(src_dir) + os.pathsep + existing_pythonpath,
        )
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read_process_output)
        process.finished.connect(self._on_process_finished)
        process.errorOccurred.connect(self._on_process_error)

        self.process = process
        self._process_started_at = time.perf_counter()
        process.start()
        if not process.waitForStarted(3000):
            self.process = None
            self._process_started_at = None
            self._active_job = None
            QMessageBox.critical(self, "训练失败", with_help("训练进程未能启动。", DATASETS_AND_TRAINING))
            self._update_action_states()
            return
        self._set_status("训练运行中...")
        self._update_action_states()

    def _read_process_output(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        if text:
            self._append_log(text)

    def _on_process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        elapsed = ""
        if self._process_started_at is not None:
            elapsed = f"（{(time.perf_counter() - self._process_started_at) * 1000.0:.1f} ms）"
        self._append_log(f"\n训练进程退出，代码：{exit_code} {elapsed}\n")
        if exit_code == 0:
            self._last_completed_job = self._active_job
            status = "训练完成，可加入模型库。"
        else:
            self._last_completed_job = None
            status = "训练停止或失败。"
            self._append_log("\n" + help_hint(DATASETS_AND_TRAINING) + "\n")
        self._active_job = None
        self.process = None
        self._process_started_at = None
        self._set_status(status)
        self._update_info(status)
        self._update_action_states()

    def _on_process_error(self, _error: QProcess.ProcessError) -> None:
        if self.process is None:
            return
        message = self.process.errorString()
        self._append_log(f"\n训练进程错误：{message}\n")
        QMessageBox.critical(self, "训练进程错误", with_help(message, DATASETS_AND_TRAINING))

    def _training_running(self) -> bool:
        return self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning

    def _add_model_option(self, path: Path) -> None:
        text = str(path)
        self.model_combo.addItem(text, str(path))
        index = self.model_combo.count() - 1
        self.model_combo.setItemData(index, path.name, SELECTED_DISPLAY_ROLE)
        self.model_combo.setItemData(index, str(path), Qt.ItemDataRole.ToolTipRole)

    def _select_model_path(self, path: Path) -> None:
        target = path.resolve()
        for index in range(self.model_combo.count()):
            try:
                candidate = Path(str(self.model_combo.itemData(index))).resolve()
            except OSError:
                continue
            if candidate == target:
                self.model_combo.setCurrentIndex(index)
                return

    def _append_log(self, text: str) -> None:
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)
        self.log_text.insertPlainText(text)
        self.log_text.ensureCursorVisible()

    def _set_status(self, text: str) -> None:
        self.status_changed.emit(text)

    def _update_info(self, status: str | None = None) -> None:
        parts = [f"任务：{self.task_combo.currentText()}"]
        data_path = self.dataset_edit.text().strip()
        if data_path:
            parts.append(f"数据集：{_shorten_path(Path(data_path), 68)}")
        model_path = self.current_model_path()
        if model_path is not None:
            parts.append(f"预训练权重：{model_path.name}")
        parts.append(f"输出：{_shorten_path(self.config.runs_dir, 68)}")
        if status:
            parts.append(status)
        self.info_label.setText(" | ".join(parts))
        self.info_label.setToolTip(str(self.config.runs_dir))

    def _update_action_states(self) -> None:
        running = self._training_running()
        self.task_combo.setEnabled(not running)
        self.dataset_edit.setEnabled(not running)
        self.choose_dataset_button.setEnabled(not running)
        self.sample_dataset_button.setEnabled(not running)
        has_dataset = bool(self.dataset_edit.text().strip())
        self.validate_button.setEnabled(not running and has_dataset)
        self.allow_missing_labels_check.setEnabled(not running)
        self.model_combo.setEnabled(not running)
        self.browse_model_button.setEnabled(not running)
        self.refresh_models_button.setEnabled(not running)
        self.epochs_spin.setEnabled(not running)
        self.image_size_spin.setEnabled(not running)
        self.batch_spin.setEnabled(not running)
        self.workers_spin.setEnabled(not running)
        self.device_combo.setEnabled(not running)
        self.run_name_edit.setEnabled(not running)
        self.resume_check.setEnabled(not running)
        self.check_environment_button.setEnabled(not running)
        self.apply_batch_button.setEnabled(not running and self._recommended_batch_size is not None)
        self.start_button.setEnabled(not running and has_dataset)
        self.stop_button.setEnabled(running)
        self.register_best_button.setEnabled(not running and self._last_completed_job is not None)
        self.open_runs_button.setEnabled(not running)


def _shorten_path(path: Path, max_chars: int) -> str:
    text = str(path)
    if len(text) <= max_chars:
        return text
    head = max(12, max_chars // 2 - 3)
    tail = max(18, max_chars - head - 3)
    return f"{text[:head]}...{text[-tail:]}"
