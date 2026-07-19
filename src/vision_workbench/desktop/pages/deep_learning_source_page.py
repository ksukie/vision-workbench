"""Upgrade guidance shown for deep-learning workflows in the base Windows EXE."""

from __future__ import annotations

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...versioning import source_archive_url
from ..widgets import make_button


def source_install_commands(target: str) -> str:
    """Return the supported source-install route for an optional dependency group."""

    return "\n".join(
        (
            "# 先下载并解压完整源码：",
            f"# {source_archive_url()}",
            "# 在解压后的 vision-workbench 目录打开 PowerShell：",
            "py -3.11 -m venv .venv",
            f".\\.venv\\Scripts\\python.exe scripts\\install_dependencies.py {target}",
            ".\\.venv\\Scripts\\vision-workbench.exe",
        )
    )


class DeepLearningSourcePage(QWidget):
    """Explain how to enable a workflow intentionally excluded from the base EXE."""

    status_changed = Signal(str)

    def __init__(self, feature_name: str, dependency_target: str) -> None:
        super().__init__()
        self.feature_name = feature_name
        self.dependency_target = dependency_target
        self.setObjectName("DeepLearningSourcePage")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(24, 24, 24, 24)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        root_layout.addWidget(self.scroll_area)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(16)

        title = QLabel(f"{feature_name}需要完整源码环境")
        title.setObjectName("PageTitle")
        content_layout.addWidget(title)

        description = QLabel(
            "此 Windows 基础版只包含传统 CV、全景拼接和相机诊断，"
            "未打包 PyTorch、YOLO26 或训练模型，以保持 EXE 体积和安全扫描风险可控。"
        )
        description.setWordWrap(True)
        content_layout.addWidget(description)

        guidance = QLabel(
            "请下载完整源码后执行下方命令。该路径会安装项目、选择合适的 Torch 环境，"
            "并为 YOLO26 安装仓库内置的依赖源码。"
        )
        guidance.setObjectName("MutedText")
        guidance.setWordWrap(True)
        content_layout.addWidget(guidance)

        command_label = QLabel("完整源码安装命令")
        command_label.setObjectName("SidebarSectionLabel")
        content_layout.addWidget(command_label)

        self.install_commands = QPlainTextEdit(source_install_commands(dependency_target))
        self.install_commands.setObjectName("DeepLearningInstallCommands")
        self.install_commands.setAccessibleName("完整源码安装命令")
        self.install_commands.setReadOnly(True)
        self.install_commands.setMinimumHeight(190)
        content_layout.addWidget(self.install_commands)

        actions = QHBoxLayout()
        self.download_source_button = make_button("下载完整源码", primary=True)
        self.copy_commands_button = make_button("复制安装命令")
        self.download_source_button.clicked.connect(self.open_source_download)
        self.copy_commands_button.clicked.connect(self.copy_install_commands)
        actions.addWidget(self.download_source_button)
        actions.addWidget(self.copy_commands_button)
        actions.addStretch(1)
        content_layout.addLayout(actions)

        content_layout.addStretch(1)
        self.scroll_area.setWidget(content)

    def open_source_download(self) -> None:
        QDesktopServices.openUrl(QUrl(source_archive_url()))
        self.status_changed.emit("已打开完整源码下载页。")

    def copy_install_commands(self) -> None:
        QApplication.clipboard().setText(self.install_commands.toPlainText())
        self.status_changed.emit("完整源码安装命令已复制。")
