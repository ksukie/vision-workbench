"""Main PySide6 window for Vision Workbench."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys
from dataclasses import dataclass
from typing import Dict

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .pages.camera_page import CameraPage
from .pages.classification_page import ClassificationPage
from .pages.cv_basics_page import CvBasicsPage
from .pages.panorama_page import PanoramaPage
from .pages.yolo_detection_page import YoloDetectionPage
from .pages.yolo_segmentation_page import YoloSegmentationPage
from .pages.yolo_training_page import YoloTrainingPage


@dataclass(frozen=True)
class NavItem:
    key: str
    label: str
    description: str


NAV_ITEMS = [
    NavItem("cv_basics", "基础 CV", "传统图像处理、色彩空间、直方图、形态学与几何变换。"),
    NavItem("panorama", "全景拼接", "左右图像拼接、SIFT 匹配和手动控制点。"),
    NavItem("camera", "相机诊断", "摄像头枚举、读流模式、FPS、截图与录制。"),
    NavItem("detection", "YOLO 检测", "YOLO26 摄像头实时目标检测。"),
    NavItem("segmentation", "YOLO 分割", "YOLO26 实例分割和语义分割。"),
    NavItem("training", "模型训练", "YOLO26 检测、分割与语义分割训练入口。"),
    NavItem("classification", "图像分类", "ResNet18、MobileNetV3 预测和基础训练。"),
]
NAV_ITEM_BY_KEY = {item.key: item for item in NAV_ITEMS}


class WindowTitleBar(QFrame):
    """Custom title bar used by the frameless main window."""

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.window = window
        self._drag_offset = None  # type: QPoint | None
        self.setObjectName("WindowTitleBar")
        self.setFixedHeight(32)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 10, 4)
        layout.setSpacing(5)

        self.title_label = QLabel("Vision Workbench")
        self.title_label.setObjectName("WindowTitle")
        layout.addWidget(self.title_label)
        layout.addStretch(1)

        self.status_label = QLabel("")
        self.status_label.setObjectName("WindowStatus")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.status_label.setMaximumWidth(360)
        layout.addWidget(self.status_label)

        self.minimize_button = self._make_window_button("-")
        self.maximize_button = self._make_window_button("□")
        self.close_button = self._make_window_button("×", close=True)
        self.minimize_button.setToolTip("最小化")
        self.maximize_button.setToolTip("最大化 / 还原")
        self.close_button.setToolTip("关闭")
        self.minimize_button.setAccessibleName("最小化窗口")
        self.maximize_button.setAccessibleName("最大化或还原窗口")
        self.close_button.setAccessibleName("关闭窗口")
        self.minimize_button.clicked.connect(window.showMinimized)
        self.maximize_button.clicked.connect(window.toggle_maximized)
        self.close_button.clicked.connect(window.close)

        layout.addWidget(self.minimize_button)
        layout.addWidget(self.maximize_button)
        layout.addWidget(self.close_button)

    def set_status(self, text: str) -> None:
        display_text = "就绪" if text == "就绪。" else _compact_status_text(text)
        self.status_label.setText(self._elide_status(display_text))
        self.status_label.setToolTip(text if text != display_text else "")

    def update_maximize_button(self, maximized: bool) -> None:
        self.maximize_button.setText("❐" if maximized else "□")

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if sys.platform == "win32":
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.window.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if sys.platform == "win32":
            super().mouseMoveEvent(event)
            return
        if self._drag_offset is None or not event.buttons() & Qt.MouseButton.LeftButton:
            super().mouseMoveEvent(event)
            return
        if not self.window.isMaximized():
            self.window.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if sys.platform == "win32":
            super().mouseReleaseEvent(event)
            return
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.window.toggle_maximized()
        super().mouseDoubleClickEvent(event)

    def _make_window_button(self, text: str, *, close: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("WindowCloseButton" if close else "WindowControlButton")
        button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _elide_status(self, text: str) -> str:
        available_width = max(80, self.status_label.maximumWidth() - 18)
        return self.status_label.fontMetrics().elidedText(
            text,
            Qt.TextElideMode.ElideRight,
            available_width,
        )


class MainWindow(QMainWindow):
    """Unified frameless Qt shell with native pages."""

    def __init__(self) -> None:
        super().__init__()
        self.nav_buttons = {}  # type: Dict[str, QPushButton]
        self.pages = {}  # type: Dict[str, QWidget]
        self.title_bar = None  # type: WindowTitleBar | None
        self.root_layout = None  # type: QHBoxLayout | None
        self.shell = None  # type: QFrame | None
        self.sidebar = None  # type: QFrame | None
        self.shell_shadow = None  # type: QGraphicsDropShadowEffect | None
        self.nav_detail_label = None  # type: QLabel | None

        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowTitle("Vision Workbench")
        self.resize(1240, 820)
        self.setMinimumSize(1040, 680)

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

        self.stack = QStackedWidget()
        self._build_ui()
        self.set_current_page("cv_basics")
        self.set_status("就绪。")

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("WindowRoot")
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 8)
        root_layout.setSpacing(0)
        self.root_layout = root_layout

        shell = QFrame()
        shell.setObjectName("AppShell")
        self.shell = shell
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        shadow = QGraphicsDropShadowEffect(shell)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(15, 23, 42, 28))
        shell.setGraphicsEffect(shadow)
        self.shell_shadow = shadow

        self.title_bar = WindowTitleBar(self)
        shell_layout.addWidget(self.title_bar)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        self.sidebar = sidebar
        sidebar.setFixedWidth(240)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(18, 22, 18, 18)
        sidebar_layout.setSpacing(10)

        title = QLabel("Vision Workbench")
        title.setObjectName("AppTitle")
        title.setWordWrap(True)
        subtitle = QLabel("计算机视觉学习工作台")
        subtitle.setObjectName("AppSubtitle")
        sidebar_layout.addWidget(title)
        sidebar_layout.addWidget(subtitle)
        sidebar_layout.addSpacing(14)

        section_label = QLabel("工作流")
        section_label.setObjectName("SidebarSectionLabel")
        sidebar_layout.addWidget(section_label)

        for index, item in enumerate(NAV_ITEMS, start=1):
            button = QPushButton(item.label)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(item.description)
            button.setAccessibleName(f"工作流：{item.label}")
            button.setAccessibleDescription(item.description)
            button.setShortcut(f"Alt+{index}")
            button.clicked.connect(lambda _checked=False, key=item.key: self.set_current_page(key))
            self.button_group.addButton(button)
            self.nav_buttons[item.key] = button
            sidebar_layout.addWidget(button)

        sidebar_layout.addStretch(1)
        self.nav_detail_label = QLabel("")
        self.nav_detail_label.setObjectName("SidebarDetail")
        self.nav_detail_label.setWordWrap(True)
        self.nav_detail_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        sidebar_layout.addWidget(self.nav_detail_label)

        version = QLabel("Qt / PySide6")
        version.setObjectName("MutedText")
        sidebar_layout.addWidget(version)

        for item in NAV_ITEMS:
            if item.key == "cv_basics":
                page = CvBasicsPage()
                page.status_changed.connect(self.set_status)
            elif item.key == "panorama":
                page = PanoramaPage()
                page.status_changed.connect(self.set_status)
            elif item.key == "camera":
                page = CameraPage()
                page.status_changed.connect(self.set_status)
            elif item.key == "detection":
                page = YoloDetectionPage()
                page.status_changed.connect(self.set_status)
            elif item.key == "segmentation":
                page = YoloSegmentationPage()
                page.status_changed.connect(self.set_status)
            elif item.key == "training":
                page = YoloTrainingPage()
                page.status_changed.connect(self.set_status)
            elif item.key == "classification":
                page = ClassificationPage()
                page.status_changed.connect(self.set_status)
            else:
                raise ValueError(f"Unsupported navigation item: {item.key}")
            self.pages[item.key] = page
            self.stack.addWidget(page)

        body_layout.addWidget(sidebar)
        body_layout.addWidget(self.stack, 1)
        shell_layout.addWidget(body, 1)
        root_layout.addWidget(shell, 1)
        self.setCentralWidget(central)

    def set_current_page(self, key: str) -> None:
        page = self.pages[key]
        self.stack.setCurrentWidget(page)
        self.nav_buttons[key].setChecked(True)
        item = NAV_ITEM_BY_KEY.get(key)
        if item is not None:
            self.setWindowTitle(f"{item.label} - Vision Workbench")
            if self.nav_detail_label is not None:
                self.nav_detail_label.setText(item.description)

    def set_status(self, text: str) -> None:
        if self.title_bar is not None:
            self.title_bar.set_status(text)

    def toggle_maximized(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
        self._update_window_state()

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        self._update_window_state()

    def closeEvent(self, event) -> None:  # noqa: N802
        for page in self.pages.values():
            shutdown = getattr(page, "shutdown", None)
            if callable(shutdown):
                shutdown()
        super().closeEvent(event)

    def nativeEvent(self, event_type, message):  # noqa: N802
        if sys.platform != "win32":
            return super().nativeEvent(event_type, message)

        msg = ctypes.wintypes.MSG.from_address(int(message))
        if msg.message != 0x0084:  # WM_NCHITTEST
            return super().nativeEvent(event_type, message)

        x = ctypes.c_short(msg.lParam & 0xFFFF).value
        y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
        if not self.isMaximized():
            border = 8
            rect = self.frameGeometry()
            left = x - rect.left() <= border
            right = rect.right() - x <= border
            top = y - rect.top() <= border
            bottom = rect.bottom() - y <= border

            if top and left:
                return True, 13  # HTTOPLEFT
            if top and right:
                return True, 14  # HTTOPRIGHT
            if bottom and left:
                return True, 16  # HTBOTTOMLEFT
            if bottom and right:
                return True, 17  # HTBOTTOMRIGHT
            if left:
                return True, 10  # HTLEFT
            if right:
                return True, 11  # HTRIGHT
            if top:
                return True, 12  # HTTOP
            if bottom:
                return True, 15  # HTBOTTOM
        if self._is_title_bar_drag_area(QPoint(x, y)):
            return True, 2  # HTCAPTION
        return super().nativeEvent(event_type, message)

    def _is_title_bar_drag_area(self, global_pos: QPoint) -> bool:
        title_bar = getattr(self, "title_bar", None)
        if title_bar is None or not title_bar.isVisible():
            return False
        if not title_bar.rect().contains(title_bar.mapFromGlobal(global_pos)):
            return False
        for button in (title_bar.minimize_button, title_bar.maximize_button, title_bar.close_button):
            if button.isVisible() and button.rect().contains(button.mapFromGlobal(global_pos)):
                return False
        return True

    def _update_window_state(self) -> None:
        title_bar = getattr(self, "title_bar", None)
        root_layout = getattr(self, "root_layout", None)
        shell = getattr(self, "shell", None)
        shell_shadow = getattr(self, "shell_shadow", None)
        sidebar = getattr(self, "sidebar", None)

        if title_bar is not None:
            title_bar.update_maximize_button(self.isMaximized())
        maximized = self.isMaximized()
        if root_layout is not None:
            if maximized:
                root_layout.setContentsMargins(0, 0, 0, 0)
            else:
                root_layout.setContentsMargins(10, 10, 10, 8)
        if shell_shadow is not None:
            shell_shadow.setEnabled(not maximized)
        for widget in (shell, title_bar, sidebar):
            if widget is None:
                continue
            widget.setProperty("maximized", maximized)
            widget.style().unpolish(widget)
            widget.style().polish(widget)


def _compact_status_text(text: str) -> str:
    if text.startswith("预测完成，"):
        if "已使用缓存模型" in text:
            return "预测完成，已使用缓存模型。"
        if "模型已缓存" in text:
            return "预测完成，模型已缓存。"
    if "ms" in text:
        for marker in ("（", "("):
            if marker in text:
                prefix = text.split(marker, 1)[0].strip()
                if prefix:
                    return _ensure_sentence_end(prefix)
    return text


def _ensure_sentence_end(text: str) -> str:
    return text if text.endswith(("。", ".", "!", "！", "?", "？")) else f"{text}。"
