"""Qt page for panorama reconstruction."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, cast

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPoint, QRect, Signal, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from panorama_reconstruction.application import PanoramaReconstructionService
from panorama_reconstruction.api import create_panorama_reconstruction_service
from panorama_reconstruction.configuration import CHANNEL_CHOICES, PanoramaReconstructionConfig
from panorama_reconstruction.domain import (
    ControlPointReconstructionParams,
    ImageArray,
    PanoramaReconstructionParams,
    PanoramaResult,
    Point,
    PointPair,
)
from vision_workbench.troubleshooting import DATA_AND_FILES, MODULE_RUNTIME_ERRORS, with_help

from ..image_presenter import QtImagePresenter
from ..task_runner import QtTaskRunner
from ..widgets import NoWheelComboBox as QComboBox
from ..widgets import PreviewPanel, SectionCard, make_button, style_form_label


logger = logging.getLogger(__name__)


MODE_AUTO = "自动 SIFT"
MODE_MANUAL = "手动控制点"
MODE_ASSISTED = "手动+辅助"
MANUAL_MODES = {MODE_MANUAL, MODE_ASSISTED}

CHANNEL_LABELS = {
    "gray": "灰度",
    "r": "红色",
    "g": "绿色",
    "b": "蓝色",
    "h": "色相",
    "s": "饱和度",
    "v": "明度",
}


@dataclass(frozen=True)
class ImageLoadPayload:
    side: str
    path: Path
    image: ImageArray
    load_ms: float


@dataclass(frozen=True)
class ImagePairPayload:
    left_path: Path
    right_path: Path
    left: ImageArray
    right: ImageArray
    load_ms: float


@dataclass(frozen=True)
class ReconstructionPayload:
    result: PanoramaResult
    total_ms: float


@dataclass(frozen=True)
class SavePayload:
    output: object
    total_ms: float


class ClickablePreviewPanel(PreviewPanel):
    """Preview panel that maps canvas clicks back to original image coordinates."""

    point_clicked = Signal(float, float)

    def __init__(self, title: str, empty_text: str, parent: QWidget | None = None) -> None:
        super().__init__(title, empty_text, parent)
        self._image_size = None  # type: tuple[int, int] | None
        self.canvas.installEventFilter(self)

    def set_image(self, pixmap: QPixmap, image: ImageArray) -> None:
        height, width = image.shape[:2]
        self._image_size = (int(width), int(height))
        self.set_pixmap(pixmap)
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)

    def clear(self) -> None:
        self._image_size = None
        self.canvas.unsetCursor()
        super().clear()

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # noqa: N802
        if watched is self.canvas and event.type() == QEvent.Type.MouseButtonPress:
            if getattr(event, "button")() != Qt.MouseButton.LeftButton:
                return False
            position = getattr(event, "position", None)
            point = position().toPoint() if callable(position) else getattr(event, "pos")()
            image_point = self.map_canvas_point(point)
            if image_point is None:
                return False
            self.point_clicked.emit(image_point[0], image_point[1])
            return True
        return super().eventFilter(watched, event)

    def map_canvas_point(self, point: QPoint) -> Point | None:
        pixmap = self._source_pixmap
        image_size = self._image_size
        if pixmap is None or pixmap.isNull() or image_size is None:
            return None

        display_rect = self._scaled_pixmap_rect()
        if display_rect.isNull() or not display_rect.contains(point):
            return None

        x_in_display = point.x() - display_rect.x()
        y_in_display = point.y() - display_rect.y()
        source_x = float(x_in_display) * float(pixmap.width()) / float(display_rect.width())
        source_y = float(y_in_display) * float(pixmap.height()) / float(display_rect.height())

        original_width, original_height = image_size
        original_x = source_x * float(original_width) / float(pixmap.width())
        original_y = source_y * float(original_height) / float(pixmap.height())
        return (
            min(max(original_x, 0.0), float(original_width - 1)),
            min(max(original_y, 0.0), float(original_height - 1)),
        )

    def _scaled_pixmap_rect(self) -> QRect:
        pixmap = self._source_pixmap
        if pixmap is None or pixmap.isNull():
            return QRect()
        scaled_size = pixmap.size()
        scaled_size.scale(self.canvas.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = (self.canvas.width() - scaled_size.width()) // 2
        y = (self.canvas.height() - scaled_size.height()) // 2
        return QRect(x, y, scaled_size.width(), scaled_size.height())


class PanoramaPage(QWidget):
    """Native Qt implementation of the panorama reconstruction workflow."""

    status_changed = Signal(str)

    def __init__(
        self,
        service: Optional[PanoramaReconstructionService] = None,
        config: PanoramaReconstructionConfig = PanoramaReconstructionConfig(),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.service = service or create_panorama_reconstruction_service(config)
        self.input_presenter = QtImagePresenter(config.preview_size)
        self.result_presenter = QtImagePresenter(config.result_preview_size)
        self.tasks = QtTaskRunner(self)

        self.left_path = None  # type: Optional[Path]
        self.right_path = None  # type: Optional[Path]
        self.left_image = None  # type: Optional[ImageArray]
        self.right_image = None  # type: Optional[ImageArray]
        self.result = None  # type: Optional[PanoramaResult]
        self.point_pairs = []  # type: list[PointPair]
        self.pending_left_point = None  # type: Optional[Point]
        self.last_timing_text = None  # type: Optional[str]
        self._busy = False
        self._completion_status = None  # type: Optional[str]

        self.open_left_button = make_button("打开左图", primary=True)
        self.open_right_button = make_button("打开右图", primary=True)
        self.sample_button = make_button("加载示例")
        self.reconstruct_button = make_button("重建全景", primary=True)
        self.save_panorama_button = make_button("保存全景")
        self.save_all_button = make_button("保存全部")
        self.undo_point_button = make_button("撤销点")
        self.clear_points_button = make_button("清空点")
        self.load_points_button = make_button("加载点对")
        self.save_points_button = make_button("保存点对")

        for button in (
            self.open_left_button,
            self.open_right_button,
            self.sample_button,
            self.reconstruct_button,
            self.save_panorama_button,
            self.save_all_button,
            self.undo_point_button,
            self.clear_points_button,
            self.load_points_button,
            self.save_points_button,
        ):
            button.setMinimumWidth(112)
            button.setMinimumHeight(38)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.mode_label = QLabel("模式")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([MODE_AUTO, MODE_MANUAL, MODE_ASSISTED])
        self.mode_combo.setToolTip("选择自动 SIFT 或控制点重建方式")
        self.mode_combo.setMinimumWidth(132)

        self.channel_label = QLabel("通道")
        self.channel_combo = QComboBox()
        for channel_name in CHANNEL_CHOICES:
            self.channel_combo.addItem(CHANNEL_LABELS.get(channel_name, channel_name), channel_name)
        self.channel_combo.setCurrentText(CHANNEL_LABELS.get(config.default_channel, config.default_channel))
        self.channel_combo.setToolTip("自动 SIFT 模式使用的特征通道")
        self.channel_combo.setMinimumWidth(112)
        for label in (self.mode_label, self.channel_label):
            style_form_label(label)

        self.left_preview = ClickablePreviewPanel("左图", "请打开左图")
        self.right_preview = ClickablePreviewPanel("右图", "请打开右图")
        self.point_count_label = QLabel("")
        self.point_count_label.setObjectName("MutedText")
        self.point_count_label.setWordWrap(True)
        self.point_hint_label = QLabel("手动模式下先点左图，再点右图，至少 3 组点对。")
        self.point_hint_label.setObjectName("MutedText")
        self.point_hint_label.setWordWrap(True)
        self.info_label = QLabel("")
        self.info_label.setObjectName("MutedText")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.result_panels = {}  # type: Dict[str, PreviewPanel]
        self.content = None  # type: QWidget | None
        self.controls_layout = None  # type: QGridLayout | None
        self.points_layout = None  # type: QGridLayout | None
        self.points_card = None  # type: SectionCard | None
        self.input_splitter = None  # type: QSplitter | None
        self.result_tabs = None  # type: QTabWidget | None
        self._compact_layout = None  # type: bool | None

        self._build_ui()
        self._connect_signals()
        self._update_mode_state()
        self._update_metrics()
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

        title = QLabel("全景拼接")
        title.setObjectName("PageTitle")
        subtitle = QLabel("加载左右图像，选择自动 SIFT 或控制点方式生成全景图。")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        controls_card = SectionCard("输入与重建")
        self.controls_layout = QGridLayout()
        self.controls_layout.setHorizontalSpacing(10)
        self.controls_layout.setVerticalSpacing(10)
        controls_card.content_layout.addLayout(self.controls_layout)
        layout.addWidget(controls_card)

        points_card = SectionCard("控制点")
        self.points_card = points_card
        self.points_layout = QGridLayout()
        self.points_layout.setHorizontalSpacing(10)
        self.points_layout.setVerticalSpacing(10)
        points_card.content_layout.addLayout(self.points_layout)
        layout.addWidget(points_card)

        self.input_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.input_splitter.addWidget(self.left_preview)
        self.input_splitter.addWidget(self.right_preview)
        self.input_splitter.setSizes([1, 1])
        self.input_splitter.setMinimumHeight(360)
        layout.addWidget(self.input_splitter, 1)

        self.result_tabs = QTabWidget()
        self.result_tabs.setMinimumHeight(420)
        for key, title_text, empty_text in [
            ("panorama", "全景图", "重建结果会显示在这里"),
            ("matches", "匹配关系", "匹配或控制点可视化会显示在这里"),
            ("mapped", "映射检查", "映射检查图会显示在这里"),
            ("warped", "右图变换", "右图变换结果会显示在这里"),
        ]:
            panel = PreviewPanel(title_text, empty_text)
            panel.setMinimumHeight(360)
            self.result_tabs.addTab(panel, title_text)
            self.result_panels[key] = panel
        layout.addWidget(self.result_tabs)
        layout.addWidget(self.info_label)

        scroll_area.setWidget(content)
        root_layout.addWidget(scroll_area)
        self._apply_responsive_layout(force=True)

    def _connect_signals(self) -> None:
        self.open_left_button.clicked.connect(lambda: self.open_image("left"))
        self.open_right_button.clicked.connect(lambda: self.open_image("right"))
        self.sample_button.clicked.connect(self.load_sample_pair)
        self.reconstruct_button.clicked.connect(self.reconstruct_panorama)
        self.save_panorama_button.clicked.connect(self.save_panorama)
        self.save_all_button.clicked.connect(self.save_all_outputs)
        self.undo_point_button.clicked.connect(self.undo_point)
        self.clear_points_button.clicked.connect(self.clear_points)
        self.load_points_button.clicked.connect(self.load_points)
        self.save_points_button.clicked.connect(self.save_points)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.left_preview.point_clicked.connect(lambda x, y: self._on_input_click("left", (x, y)))
        self.right_preview.point_clicked.connect(lambda x, y: self._on_input_click("right", (x, y)))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def _apply_responsive_layout(self, *, force: bool = False) -> None:
        controls = self.controls_layout
        points = self.points_layout
        splitter = self.input_splitter
        result_tabs = self.result_tabs
        if controls is None or points is None or splitter is None or result_tabs is None:
            return

        compact = self.width() < 1180
        if not force and compact == self._compact_layout:
            return
        self._compact_layout = compact

        self._reset_grid(controls, 4)
        self._reset_grid(points, 4)
        if compact:
            self._layout_compact_controls(controls)
            self._layout_compact_points(points)
            splitter.setOrientation(Qt.Orientation.Vertical)
            splitter.setMinimumHeight(620)
            self.left_preview.setMinimumHeight(300)
            self.right_preview.setMinimumHeight(300)
            result_tabs.setMinimumHeight(360)
        else:
            self._layout_wide_controls(controls)
            self._layout_wide_points(points)
            splitter.setOrientation(Qt.Orientation.Horizontal)
            splitter.setMinimumHeight(360)
            self.left_preview.setMinimumHeight(340)
            self.right_preview.setMinimumHeight(340)
            result_tabs.setMinimumHeight(420)

    def _layout_wide_controls(self, controls: QGridLayout) -> None:
        controls.addWidget(self.open_left_button, 0, 0)
        controls.addWidget(self.open_right_button, 0, 1)
        controls.addWidget(self.sample_button, 0, 2)
        controls.addWidget(self.reconstruct_button, 0, 3)

        controls.addWidget(self.mode_label, 1, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.mode_combo, 1, 1)
        controls.addWidget(self.channel_label, 1, 2, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.channel_combo, 1, 3)

        controls.addWidget(self.save_panorama_button, 2, 0, 1, 2)
        controls.addWidget(self.save_all_button, 2, 2, 1, 2)
        for index in range(4):
            controls.setColumnStretch(index, 1)

    def _layout_compact_controls(self, controls: QGridLayout) -> None:
        controls.addWidget(self.open_left_button, 0, 0)
        controls.addWidget(self.open_right_button, 0, 1)
        controls.addWidget(self.sample_button, 1, 0)
        controls.addWidget(self.reconstruct_button, 1, 1)

        controls.addWidget(self.mode_label, 2, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.mode_combo, 2, 1)
        controls.addWidget(self.channel_label, 3, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.channel_combo, 3, 1)

        controls.addWidget(self.save_panorama_button, 4, 0)
        controls.addWidget(self.save_all_button, 4, 1)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)

    def _layout_wide_points(self, points: QGridLayout) -> None:
        points.addWidget(self.point_count_label, 0, 0)
        points.addWidget(self.point_hint_label, 0, 1, 1, 3)
        points.addWidget(self.undo_point_button, 1, 0)
        points.addWidget(self.clear_points_button, 1, 1)
        points.addWidget(self.load_points_button, 1, 2)
        points.addWidget(self.save_points_button, 1, 3)
        for index in range(4):
            points.setColumnStretch(index, 1)

    def _layout_compact_points(self, points: QGridLayout) -> None:
        points.addWidget(self.point_count_label, 0, 0, 1, 2)
        points.addWidget(self.point_hint_label, 1, 0, 1, 2)
        points.addWidget(self.undo_point_button, 2, 0)
        points.addWidget(self.clear_points_button, 2, 1)
        points.addWidget(self.load_points_button, 3, 0)
        points.addWidget(self.save_points_button, 3, 1)
        points.setColumnStretch(0, 1)
        points.setColumnStretch(1, 1)

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

    def open_image(self, side: str) -> None:
        patterns = " ".join(self.config.supported_extensions)
        side_label = _side_label(side)
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            f"打开{side_label}",
            "",
            f"图像文件 ({patterns});;所有文件 (*.*)",
        )
        if not path:
            return

        image_path = Path(path)
        self._run_task(
            task=lambda: self._load_single_image(side, image_path),
            on_success=self._on_single_image_loaded,
            busy_text=f"正在加载{side_label}...",
            error_title="打开失败",
            error_category=DATA_AND_FILES,
        )

    def load_sample_pair(self) -> None:
        pair = self.service.get_sample_image_paths()
        self._run_task(
            task=lambda: self._load_pair_from_paths(pair.left, pair.right),
            on_success=self._on_pair_loaded,
            busy_text="正在加载示例图像...",
            error_title="打开失败",
            error_category=DATA_AND_FILES,
        )

    def reconstruct_panorama(self) -> None:
        if self.left_image is None or self.right_image is None:
            QMessageBox.information(self, "缺少图像", "请先加载左图和右图。")
            return

        mode = self.current_mode()
        if mode in MANUAL_MODES:
            if len(self.point_pairs) < 3:
                QMessageBox.information(self, "控制点不足", "请至少添加 3 组左右控制点。")
                return
            if self.pending_left_point is not None:
                QMessageBox.information(self, "点对未完成", "请先在右图点击匹配点。")
                return

        left = self.left_image.copy()
        right = self.right_image.copy()
        pairs = list(self.point_pairs)
        channel_name = self.current_channel_name()
        self._run_task(
            task=lambda: self._reconstruct(mode, channel_name, left, right, pairs),
            on_success=self._on_panorama_ready,
            busy_text=_reconstruction_busy_text(mode),
            error_title="重建失败",
            error_category=MODULE_RUNTIME_ERRORS,
        )

    def save_panorama(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "没有结果", "请先完成一次全景重建。")
            return

        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存全景图",
            "",
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg *.jpeg);;位图 (*.bmp);;所有文件 (*.*)",
        )
        if not path:
            return

        panorama = self.result.panorama.copy()
        self._run_task(
            task=lambda: self._save_image(panorama, Path(path)),
            on_success=lambda payload: self._on_panorama_saved(payload, Path(path)),
            busy_text="正在保存全景图...",
            error_title="保存失败",
            error_category=DATA_AND_FILES,
        )

    def save_all_outputs(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "没有结果", "请先完成一次全景重建。")
            return

        directory = QFileDialog.getExistingDirectory(self, "选择输出文件夹", "")
        if not directory:
            return

        result = self.result
        self._run_task(
            task=lambda: self._save_outputs(result, Path(directory)),
            on_success=lambda payload: self._on_outputs_saved(payload, Path(directory)),
            busy_text="正在保存全部结果...",
            error_title="保存失败",
            error_category=DATA_AND_FILES,
        )

    def undo_point(self) -> None:
        if self.pending_left_point is not None:
            self.pending_left_point = None
            self._set_status("已撤销待匹配的左图点。")
        elif self.point_pairs:
            self.point_pairs.pop()
            self._set_status("已撤销上一组控制点。")
        self._clear_result()
        self._refresh_input_images()
        self._update_metrics()
        self._update_action_states()

    def clear_points(self) -> None:
        self.point_pairs.clear()
        self.pending_left_point = None
        self._clear_result()
        self._refresh_input_images()
        self._update_metrics()
        self._update_action_states()
        self._set_status("控制点已清空。")

    def load_points(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "加载控制点",
            "",
            "JSON 文件 (*.json);;所有文件 (*.*)",
        )
        if not path:
            return

        point_path = Path(path)
        self._run_task(
            task=lambda: self.service.load_point_pairs(point_path),
            on_success=lambda value: self._on_points_loaded(value, point_path),
            busy_text="正在加载控制点...",
            error_title="加载失败",
            error_category=DATA_AND_FILES,
        )

    def save_points(self) -> None:
        if not self.point_pairs:
            QMessageBox.information(self, "没有控制点", "请先添加控制点。")
            return

        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存控制点",
            "",
            "JSON 文件 (*.json);;所有文件 (*.*)",
        )
        if not path:
            return

        point_path = Path(path)
        pairs = list(self.point_pairs)
        self._run_task(
            task=lambda: self._save_point_pairs(point_path, pairs),
            on_success=lambda payload: self._on_points_saved(payload, point_path),
            busy_text="正在保存控制点...",
            error_title="保存失败",
            error_category=DATA_AND_FILES,
        )

    def current_mode(self) -> str:
        return self.mode_combo.currentText()

    def current_channel_name(self) -> str:
        return str(self.channel_combo.currentData())

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
            QMessageBox.information(self, "正在处理", "请等待当前任务完成。")
            return
        self._set_busy(True)
        self._set_status(busy_text)

    def _load_single_image(self, side: str, path: Path) -> ImageLoadPayload:
        started_at = time.perf_counter()
        image = self.service.load_image(path)
        return ImageLoadPayload(side=side, path=path, image=image, load_ms=_elapsed_ms(started_at))

    def _load_pair_from_paths(self, left_path: Path, right_path: Path) -> ImagePairPayload:
        started_at = time.perf_counter()
        left = self.service.load_image(left_path)
        right = self.service.load_image(right_path)
        return ImagePairPayload(
            left_path=left_path,
            right_path=right_path,
            left=left,
            right=right,
            load_ms=_elapsed_ms(started_at),
        )

    def _reconstruct(
        self,
        mode: str,
        channel_name: str,
        left: ImageArray,
        right: ImageArray,
        point_pairs: Sequence[PointPair],
    ) -> ReconstructionPayload:
        started_at = time.perf_counter()
        if mode == MODE_AUTO:
            result = self.service.reconstruct(
                left,
                right,
                PanoramaReconstructionParams(channel_name=channel_name),
            )
        elif mode == MODE_ASSISTED:
            result = self.service.reconstruct_assisted_from_points(
                left,
                right,
                list(point_pairs),
                ControlPointReconstructionParams(),
            )
        else:
            result = self.service.reconstruct_from_points(left, right, list(point_pairs))
        return ReconstructionPayload(result=result, total_ms=_elapsed_ms(started_at))

    def _save_image(self, image: ImageArray, path: Path) -> SavePayload:
        started_at = time.perf_counter()
        self.service.save_image(image, path)
        return SavePayload(output=path, total_ms=_elapsed_ms(started_at))

    def _save_outputs(self, result: PanoramaResult, directory: Path) -> SavePayload:
        started_at = time.perf_counter()
        outputs = self.service.save_outputs(result, directory)
        return SavePayload(output=outputs, total_ms=_elapsed_ms(started_at))

    def _save_point_pairs(self, path: Path, pairs: Sequence[PointPair]) -> SavePayload:
        started_at = time.perf_counter()
        self.service.save_point_pairs(path, list(pairs))
        return SavePayload(output=path, total_ms=_elapsed_ms(started_at))

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

    def _on_single_image_loaded(self, value: object) -> None:
        payload = cast(ImageLoadPayload, value)
        if payload.side == "left":
            self.left_path = payload.path
            self.left_image = payload.image
        else:
            self.right_path = payload.path
            self.right_image = payload.image
        self.point_pairs.clear()
        self.pending_left_point = None
        self._clear_result()
        self._refresh_input_images()
        self.last_timing_text = f"加载 {payload.load_ms:.1f} ms"
        self._completion_status = f"{_side_label(payload.side)}已加载（{self.last_timing_text}）。"
        self._update_metrics()
        self._update_action_states()

    def _on_pair_loaded(self, value: object) -> None:
        payload = cast(ImagePairPayload, value)
        self.left_path = payload.left_path
        self.right_path = payload.right_path
        self.left_image = payload.left
        self.right_image = payload.right
        self.point_pairs.clear()
        self.pending_left_point = None
        self._clear_result()
        self._refresh_input_images()
        self.last_timing_text = f"加载 {payload.load_ms:.1f} ms"
        self._completion_status = f"左右图像已加载（{self.last_timing_text}）。"
        self._update_metrics()
        self._update_action_states()

    def _on_panorama_ready(self, value: object) -> None:
        payload = cast(ReconstructionPayload, value)
        self.result = payload.result
        self._show_output("panorama", self.result.panorama)
        self._show_output("matches", self.result.match_visualization)
        self._show_output("mapped", self.result.mapped_points_visualization)
        self._show_output("warped", self.result.warped_right)
        self.last_timing_text = f"重建 {payload.total_ms:.1f} ms"
        self._completion_status = f"全景拼接完成（{self.last_timing_text}）。"
        logger.info("Panorama reconstruction timing: %s", self.last_timing_text)
        self._update_metrics()
        self._update_action_states()

    def _on_panorama_saved(self, value: object, path: Path) -> None:
        payload = cast(SavePayload, value)
        self.last_timing_text = f"保存 {payload.total_ms:.1f} ms"
        self._completion_status = f"全景图已保存（{self.last_timing_text}）。"
        self._update_metrics()
        QMessageBox.information(self, "保存完成", f"全景图已保存到：\n{path}")

    def _on_outputs_saved(self, value: object, directory: Path) -> None:
        payload = cast(SavePayload, value)
        outputs = cast(Dict[str, Path], payload.output)
        self.last_timing_text = f"保存 {payload.total_ms:.1f} ms"
        self._completion_status = f"全部结果已保存（{self.last_timing_text}）。"
        self._update_metrics()
        QMessageBox.information(
            self,
            "保存完成",
            "结果已保存到：\n"
            + str(directory)
            + "\n\n"
            + "\n".join(f"{name}: {path.name}" for name, path in outputs.items()),
        )

    def _on_points_loaded(self, value: object, path: Path) -> None:
        self.point_pairs = list(cast(Sequence[PointPair], value))
        self.pending_left_point = None
        self._clear_result()
        self._refresh_input_images()
        self._completion_status = f"已加载控制点：{path.name}"
        self._update_metrics()
        self._update_action_states()

    def _on_points_saved(self, value: object, path: Path) -> None:
        payload = cast(SavePayload, value)
        self.last_timing_text = f"保存 {payload.total_ms:.1f} ms"
        self._completion_status = f"控制点已保存（{self.last_timing_text}）。"
        self._update_metrics()
        QMessageBox.information(self, "保存完成", f"控制点已保存到：\n{path}")

    def _on_input_click(self, side: str, point: Point) -> None:
        if self.left_image is None or self.right_image is None:
            return

        if self.pending_left_point is None:
            if side != "left":
                QMessageBox.information(self, "点选顺序", "请先在左图选择一个点。")
                return
            self.pending_left_point = point
            self._set_status("已选择左图点，请在右图选择匹配点。")
        else:
            if side != "right":
                QMessageBox.information(self, "点选顺序", "请在右图选择匹配点。")
                return
            self.point_pairs.append((self.pending_left_point, point))
            self.pending_left_point = None
            self._set_status(f"已添加第 {len(self.point_pairs)} 组控制点。")

        self._clear_result()
        self._refresh_input_images()
        self._update_metrics()
        self._update_action_states()

    def _on_mode_changed(self, _index: int) -> None:
        self._update_mode_state()
        self._update_metrics()
        self._update_action_states()

    def _refresh_input_images(self) -> None:
        if self.left_image is not None:
            self.left_preview.set_image(
                self.input_presenter.to_pixmap(self._annotated_input_image("left", self.left_image)),
                self.left_image,
            )
        else:
            self.left_preview.clear()

        if self.right_image is not None:
            self.right_preview.set_image(
                self.input_presenter.to_pixmap(self._annotated_input_image("right", self.right_image)),
                self.right_image,
            )
        else:
            self.right_preview.clear()

    def _annotated_input_image(self, side: str, image: ImageArray) -> ImageArray:
        annotated = np.asarray(image).copy()
        for index, (left_point, right_point) in enumerate(self.point_pairs, start=1):
            point = left_point if side == "left" else right_point
            color = (0, 255, 255) if side == "left" else (0, 0, 255)
            _draw_point_marker(annotated, point, color, str(index))

        if side == "left" and self.pending_left_point is not None:
            _draw_point_marker(annotated, self.pending_left_point, (0, 255, 0), "next")
        return annotated

    def _show_output(self, key: str, image: ImageArray) -> None:
        panel = self.result_panels[key]
        panel.set_pixmap(self.result_presenter.to_pixmap(image))

    def _clear_result(self) -> None:
        self.result = None
        for panel in self.result_panels.values():
            panel.clear()

    def _update_mode_state(self) -> None:
        auto_mode = self.current_mode() == MODE_AUTO
        self.channel_label.setEnabled(auto_mode)
        self.channel_combo.setEnabled(auto_mode and not self._busy)
        if self.points_card is not None:
            self.points_card.setVisible(not auto_mode)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_mode_state()
        self._update_action_states()

    def _update_action_states(self) -> None:
        has_pair = self.left_image is not None and self.right_image is not None
        has_result = self.result is not None
        has_points = bool(self.point_pairs)
        manual_ready = len(self.point_pairs) >= 3 and self.pending_left_point is None
        can_reconstruct = has_pair and (self.current_mode() == MODE_AUTO or manual_ready)

        self.open_left_button.setEnabled(not self._busy)
        self.open_right_button.setEnabled(not self._busy)
        self.sample_button.setEnabled(not self._busy)
        self.reconstruct_button.setEnabled(can_reconstruct and not self._busy)
        self.save_panorama_button.setEnabled(has_result and not self._busy)
        self.save_all_button.setEnabled(has_result and not self._busy)
        self.undo_point_button.setEnabled((has_points or self.pending_left_point is not None) and not self._busy)
        self.clear_points_button.setEnabled((has_points or self.pending_left_point is not None) and not self._busy)
        self.load_points_button.setEnabled(not self._busy)
        self.save_points_button.setEnabled(has_points and not self._busy)
        self.mode_combo.setEnabled(not self._busy)
        self.channel_combo.setEnabled(self.current_mode() == MODE_AUTO and not self._busy)

    def _update_metrics(self) -> None:
        point_text = f"点对：{len(self.point_pairs)}"
        if self.pending_left_point is not None:
            point_text += " | 左图点待匹配"
        self.point_count_label.setText(point_text)

        parts = []
        if self.left_path is not None:
            parts.append(f"左图：{self.left_path.name}")
        if self.right_path is not None:
            parts.append(f"右图：{self.right_path.name}")
        parts.append(f"模式：{self.current_mode()}")
        if self.current_mode() == MODE_AUTO:
            parts.append(f"通道：{self.channel_combo.currentText()}")
        parts.append(point_text)
        if self.result is not None:
            metrics = self.result.metrics()
            parts.append(f"方法：{metrics['method']}")
            parts.append(f"匹配/点数：{metrics['inliers']}")
            parts.append(f"尺寸：{metrics['panorama_shape']}")
        if self.last_timing_text:
            parts.append(self.last_timing_text)
        self.info_label.setText(" | ".join(parts))

    def _set_status(self, text: str) -> None:
        self.status_changed.emit(text)


def _side_label(side: str) -> str:
    return "左图" if side == "left" else "右图"


def _reconstruction_busy_text(mode: str) -> str:
    if mode == MODE_AUTO:
        return "正在执行自动 SIFT 拼接..."
    if mode == MODE_ASSISTED:
        return "正在使用控制点和辅助匹配拼接..."
    return "正在使用手动控制点拼接..."


def _draw_point_marker(
    image: ImageArray,
    point: Point,
    color: tuple[int, int, int],
    label: str,
) -> None:
    x, y = np.round(point).astype(int)
    height, width = image.shape[:2]
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    cv2.line(image, (x - 8, y), (x + 8, y), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (x, y - 8), (x, y + 8), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(image, (x, y), 5, color, -1)
    cv2.circle(image, (x, y), 9, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        (x + 11, max(12, y - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0
