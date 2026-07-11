"""Reusable Qt widgets for the Vision Workbench UI."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPalette, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionComboBox,
    QStyleOptionViewItem,
    QStylePainter,
    QVBoxLayout,
    QWidget,
)


_USER_ROLE = Qt.ItemDataRole.UserRole
SELECTED_DISPLAY_ROLE = _USER_ROLE.value + 101 if hasattr(_USER_ROLE, "value") else int(_USER_ROLE) + 101


def make_button(text: str, *, primary: bool = False, danger: bool = False) -> QPushButton:
    """Create a styled push button with an optional visual role."""

    button = QPushButton(text)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setAccessibleName(text)
    shortcut = {
        "打开图片": "Ctrl+O",
        "保存结果": "Ctrl+S",
        "开始训练": "Ctrl+Return",
    }.get(text)
    if shortcut:
        button.setShortcut(shortcut)
    if primary:
        button.setProperty("variant", "primary")
    elif danger:
        button.setProperty("variant", "danger")
    return button


def style_form_label(label: QLabel) -> QLabel:
    """Align compact form labels with adjacent input controls."""

    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    label.setMinimumHeight(42)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
    return label


def associate_form_label(label: QLabel, control: QWidget) -> None:
    """Expose a visual form label as the accessible name and keyboard buddy."""

    label.setBuddy(control)
    control.setAccessibleName(label.text().strip())


class _CenteredItemDelegate(QStyledItemDelegate):
    """Center-align combo popup rows."""

    def initStyleOption(self, option: QStyleOptionViewItem, index) -> None:  # noqa: N802
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignmentFlag.AlignCenter


class NoWheelComboBox(QComboBox):
    """Combo box whose value is not changed by accidental wheel scrolling."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setItemDelegate(_CenteredItemDelegate(self))

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)

        current_text = option.currentText
        selected_text = self.itemData(self.currentIndex(), SELECTED_DISPLAY_ROLE)
        if selected_text:
            current_text = str(selected_text)
        option.currentText = ""
        option.currentIcon = QIcon()
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)

        text_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            option,
            QStyle.SubControl.SC_ComboBoxEditField,
            self,
        )
        text = option.fontMetrics.elidedText(current_text, Qt.TextElideMode.ElideRight, text_rect.width())
        palette_group = QPalette.ColorGroup.Disabled if not self.isEnabled() else QPalette.ColorGroup.Active
        painter.setPen(option.palette.color(palette_group, QPalette.ColorRole.Text))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine, text)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """Integer spin box whose value is not changed by accidental wheel scrolling."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Decimal spin box whose value is not changed by accidental wheel scrolling."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        event.ignore()


class SectionCard(QFrame):
    """Rounded white section used to group controls."""

    def __init__(self, title: str = "", subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SectionCard")
        self.content_layout = QVBoxLayout(self)
        self.content_layout.setContentsMargins(18, 16, 18, 16)
        self.content_layout.setSpacing(12)
        if title:
            label = QLabel(title)
            label.setObjectName("SectionTitle")
            self.content_layout.addWidget(label)
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("MutedText")
            subtitle_label.setWordWrap(True)
            self.content_layout.addWidget(subtitle_label)


class ParameterSlider(QWidget):
    """Integer slider with a fixed label and live value display."""

    def __init__(
        self,
        label: str,
        minimum: int,
        maximum: int,
        value: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.label = QLabel(label)
        self.label.setObjectName("ParameterLabel")
        self.label.setMinimumWidth(76)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.setMinimumWidth(130)
        self.slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.label.setBuddy(self.slider)
        self.slider.setAccessibleName(label)

        self.value_label = QLabel(str(value))
        self.value_label.setObjectName("SliderValue")
        self.value_label.setAccessibleName(f"{label}当前值")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setMinimumWidth(38)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)

        self.slider.valueChanged.connect(lambda new_value: self.value_label.setText(str(new_value)))

    def value(self) -> int:
        return int(self.slider.value())

    def set_value(self, value: int) -> None:
        self.slider.setValue(value)


class PreviewPanel(QFrame):
    """Rounded image preview panel that scales pixmaps to its canvas."""

    def __init__(self, title: str, empty_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PreviewPanel")
        self.setMinimumHeight(380)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._source_pixmap = None  # type: QPixmap | None
        self._empty_text = empty_text

        self.title_label = QLabel(title)
        self.title_label.setObjectName("PreviewTitle")

        self.canvas = QLabel(empty_text)
        self.canvas.setObjectName("PreviewCanvas")
        self.canvas.setProperty("hasImage", False)
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.setMargin(18)
        self.canvas.setWordWrap(True)
        self.canvas.setMinimumSize(300, 320)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(self.title_label)
        layout.addWidget(self.canvas, 1)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self.canvas.setText("")
        self._set_canvas_has_image(True)
        self._apply_scaled_pixmap()

    def clear(self) -> None:
        self._source_pixmap = None
        self.canvas.clear()
        self.canvas.setText(self._empty_text)
        self._set_canvas_has_image(False)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        size = self.canvas.size()
        if size.width() <= 0 or size.height() <= 0:
            return
        scaled = self._source_pixmap.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.canvas.setPixmap(scaled)

    def _set_canvas_has_image(self, has_image: bool) -> None:
        self.canvas.setProperty("hasImage", has_image)
        self.canvas.style().unpolish(self.canvas)
        self.canvas.style().polish(self.canvas)


def add_grid_widget(grid: QGridLayout, widget: QWidget, row: int, column: int) -> None:
    """Add a widget to a two-column parameter grid."""

    grid.addWidget(widget, row, column)


def set_download_progress(
    label: QLabel,
    progress_bar: QProgressBar,
    action_text: str,
    path: object,
    percent: int | None,
    downloaded_bytes: int,
    total_bytes: int | None,
) -> None:
    """Update a compact download status label and progress line."""

    if percent is None:
        progress_bar.setRange(0, 0)
        progress_text = ""
    else:
        bounded_percent = max(0, min(100, int(percent)))
        progress_bar.setRange(0, 100)
        progress_bar.setValue(bounded_percent)
        progress_text = f" {bounded_percent}%"

    if total_bytes and total_bytes > 0:
        size_text = f"（{_format_bytes(downloaded_bytes)} / {_format_bytes(total_bytes)}）"
    elif downloaded_bytes > 0:
        size_text = f"（已下载 {_format_bytes(downloaded_bytes)}）"
    else:
        size_text = ""

    label.setText(f"{action_text}...{progress_text}{size_text}\n保存路径：{path}")


def _format_bytes(value: int) -> str:
    size = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} GB"
