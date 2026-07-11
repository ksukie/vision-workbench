"""Qt theme helpers for the Vision Workbench desktop UI."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication


APP_QSS = """
* {
    font-family: "Microsoft YaHei UI", "Segoe UI", "PingFang SC", sans-serif;
    font-size: 13px;
    color: #172033;
}

QMainWindow {
    background: transparent;
}

QWidget#WindowRoot {
    background: transparent;
}

QFrame#AppShell {
    background: #f6f8fb;
    border: 1px solid #d6deea;
    border-radius: 18px;
}

QFrame#AppShell[maximized="true"] {
    border: 0;
    border-radius: 0;
}

QFrame#Sidebar {
    background: #eef3f7;
    border-right: 1px solid #dae3ef;
    border-bottom-left-radius: 17px;
}

QFrame#Sidebar[maximized="true"] {
    border-bottom-left-radius: 0;
}

QFrame#WindowTitleBar {
    background: #f6f8fb;
    border: 0;
    border-top-left-radius: 17px;
    border-top-right-radius: 17px;
}

QFrame#WindowTitleBar[maximized="true"] {
    border-top-left-radius: 0;
    border-top-right-radius: 0;
}

QLabel#WindowTitle {
    color: #334155;
    font-size: 12px;
    font-weight: 600;
}

QLabel#WindowStatus {
    background: #edf2f7;
    border: 1px solid #dde5f0;
    border-radius: 8px;
    padding: 2px 8px;
    color: #64748b;
    font-size: 12px;
}

QPushButton#WindowControlButton {
    background: transparent;
    border: 0;
    border-radius: 7px;
    padding: 0;
    min-width: 30px;
    max-width: 30px;
    min-height: 24px;
    max-height: 24px;
    color: #64748b;
    font-size: 14px;
}

QPushButton#WindowControlButton:hover {
    background: #e6ebf2;
    color: #334155;
}

QPushButton#WindowControlButton:focus,
QPushButton#WindowCloseButton:focus {
    border: 1px solid #0a84ff;
    background: #eef6ff;
}

QPushButton#WindowCloseButton {
    background: transparent;
    border: 0;
    border-radius: 7px;
    padding: 0;
    min-width: 30px;
    max-width: 30px;
    min-height: 24px;
    max-height: 24px;
    color: #64748b;
    font-size: 15px;
}

QPushButton#WindowCloseButton:hover {
    background: #fee2e2;
    color: #7f1d1d;
}

QLabel#AppTitle {
    font-size: 22px;
    font-weight: 700;
    color: #111827;
}

QLabel#SidebarSectionLabel {
    color: #64748b;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0;
}

QLabel#SidebarDetail {
    background: transparent;
    border: 0;
    padding: 4px 2px;
    color: #64748b;
    font-size: 12px;
}

QLabel#AppSubtitle,
QLabel#PageSubtitle,
QLabel#MutedText {
    color: #6b7280;
}

QLabel#PageTitle {
    font-size: 24px;
    font-weight: 700;
    color: #111827;
}

QFrame#SectionCard,
QFrame#PlaceholderCard {
    background: #ffffff;
    border: 1px solid #dde5f0;
    border-radius: 8px;
}

QFrame#PreviewPanel {
    background: #ffffff;
    border: 1px solid #dde5f0;
    border-radius: 8px;
}

QTabWidget#WorkflowTabs::pane {
    border: 0;
    background: transparent;
}

QTabWidget#WorkflowTabs QTabBar::tab {
    background: #e9eef5;
    border: 1px solid #d6deea;
    border-bottom: 0;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 9px 24px;
    margin-right: 6px;
    color: #475569;
}

QTabWidget#WorkflowTabs QTabBar::tab:selected {
    background: #ffffff;
    color: #075fbd;
    font-weight: 600;
}

QTabWidget#WorkflowTabs QTabBar::tab:focus {
    border-color: #0a84ff;
}

QScrollArea#PageScrollArea {
    background: transparent;
    border: 0;
}

QWidget#PageContent {
    background: #f6f8fb;
}

QScrollBar:vertical {
    background: transparent;
    width: 12px;
    margin: 8px 3px 8px 3px;
}

QScrollBar::handle:vertical {
    background: #cbd5e1;
    border-radius: 4px;
    min-height: 40px;
}

QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: transparent;
    border: 0;
}

QScrollBar:horizontal {
    background: transparent;
    height: 12px;
    margin: 3px 8px 3px 8px;
}

QScrollBar::handle:horizontal {
    background: #cbd5e1;
    border-radius: 4px;
    min-width: 40px;
}

QScrollBar::handle:horizontal:hover {
    background: #94a3b8;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background: transparent;
    border: 0;
}

QLabel#PreviewCanvas {
    border-radius: 8px;
}

QLabel#PreviewCanvas[hasImage="false"] {
    background: #f8fafc;
    border: 1px dashed #b8c5d6;
    color: #8292a8;
}

QLabel#PreviewCanvas[hasImage="true"] {
    background: #161b22;
    border: 0;
}

QLabel#SectionTitle,
QLabel#PreviewTitle {
    font-size: 14px;
    font-weight: 600;
    color: #111827;
}

QLabel#ParameterLabel {
    color: #475569;
}

QLabel#SliderValue {
    background: #f1f5f9;
    border: 1px solid #dde5f0;
    border-radius: 7px;
    padding: 2px 6px;
    color: #334155;
}

QLabel#ParameterHint {
    background: #f8fafc;
    border: 1px dashed #cad5e2;
    border-radius: 8px;
    padding: 10px 12px;
    color: #64748b;
}

QLabel#BusyNotice {
    background: #eef6ff;
    border: 1px solid #b9dcff;
    border-radius: 8px;
    padding: 10px 12px;
    color: #075fbd;
    font-weight: 600;
}

QPushButton {
    background: #ffffff;
    border: 1px solid #d4dbe7;
    border-radius: 8px;
    padding: 8px 14px;
    min-height: 20px;
}

QPushButton:hover {
    background: #f8fafc;
    border-color: #b8c2d2;
}

QPushButton:pressed {
    background: #eef2f7;
}

QPushButton:focus {
    border-color: #0a84ff;
}

QCheckBox:focus {
    color: #075fbd;
}

QPushButton:disabled {
    background: #eef2f7;
    border-color: #e2e8f0;
    color: #94a3b8;
}

QPushButton[variant="primary"] {
    background: #0a84ff;
    border-color: #0a84ff;
    color: #ffffff;
    font-weight: 600;
}

QPushButton[variant="primary"]:hover {
    background: #0077ed;
    border-color: #0077ed;
}

QPushButton[variant="danger"] {
    color: #b42318;
}

QPushButton[variant="primary"]:disabled,
QPushButton[variant="primary"]:disabled:hover,
QPushButton[variant="danger"]:disabled,
QPushButton[variant="danger"]:disabled:hover {
    background: #eef2f7;
    border-color: #e2e8f0;
    color: #94a3b8;
}

QPushButton#NavButton {
    text-align: left;
    padding: 10px 12px;
    border: 0;
    border-radius: 8px;
    background: transparent;
    color: #374151;
}

QPushButton#NavButton:hover {
    background: #e4e9f1;
}

QPushButton#NavButton:checked {
    background: #dcecff;
    color: #075fbd;
    font-weight: 600;
}

QSpinBox,
QDoubleSpinBox,
QLineEdit {
    background: #ffffff;
    border: 1px solid #d4dbe7;
    border-radius: 8px;
    padding: 6px 10px;
    min-height: 24px;
}

QSpinBox:hover,
QDoubleSpinBox:hover,
QLineEdit:hover {
    border-color: #aeb9ca;
}

QSpinBox:focus,
QDoubleSpinBox:focus,
QLineEdit:focus {
    border-color: #0a84ff;
}

QComboBox {
    background: #ffffff;
    border: 1px solid #d4dbe7;
    border-radius: 8px;
    padding: 7px 38px 7px 12px;
    min-height: 26px;
    selection-background-color: #dcecff;
    selection-color: #111827;
}

QComboBox:hover {
    background: #fbfdff;
    border-color: #aeb9ca;
}

QComboBox:focus,
QComboBox:on {
    border-color: #0a84ff;
}

QComboBox:disabled {
    background: #f1f5f9;
    color: #94a3b8;
}

QComboBox::drop-down {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 36px;
    border: 0;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
    background: transparent;
}

QComboBox::drop-down:hover {
    background: #eef5ff;
}

QComboBox::down-arrow {
    width: 12px;
    height: 12px;
}

QComboBox QAbstractItemView {
    background: #ffffff;
    border: 1px solid #d4dbe7;
    border-radius: 8px;
    padding: 6px;
    outline: 0;
    selection-background-color: #dcecff;
    selection-color: #075fbd;
}

QProgressBar#InlineBusyProgress {
    background: #e8eef7;
    border: 0;
    border-radius: 3px;
    min-height: 6px;
    max-height: 6px;
}

QProgressBar#InlineBusyProgress::chunk {
    background: #0a84ff;
    border-radius: 3px;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #e5e7eb;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #0a84ff;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 1px solid #b8c2d2;
    width: 18px;
    margin: -7px 0;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    border-color: #0a84ff;
}

QSplitter::handle {
    background: #eef1f5;
}

QSplitter::handle:horizontal {
    width: 8px;
}

QToolTip {
    background: #111827;
    color: #ffffff;
    border: 0;
    border-radius: 6px;
    padding: 6px 8px;
}

"""


def apply_theme(app: QApplication) -> None:
    """Apply the shared Qt Widgets style."""

    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
