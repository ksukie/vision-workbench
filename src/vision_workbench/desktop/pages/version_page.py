"""Qt page for runtime version details and trusted release updates."""

from __future__ import annotations

from datetime import datetime
from typing import cast

from PySide6.QtCore import QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ...update_installer import PreparedUpdate, launch_update_helper, prepare_update
from ...update_service import UpdateCheckResult, UpdateClient, UpdateState
from ...versioning import RuntimeVersionInfo, current_version_info
from ..task_runner import QtTaskRunner
from ..widgets import SectionCard, make_button


_ACTION_TWO_COLUMN_MIN_WIDTH = 280
_ACTION_FOUR_COLUMN_MIN_WIDTH = 500


class VersionPage(QWidget):
    """Display the running release identity and manage explicit updates."""

    status_changed = Signal(str)
    download_progress_changed = Signal(object, object)

    def __init__(
        self,
        version_info: RuntimeVersionInfo | None = None,
        update_client: UpdateClient | None = None,
    ) -> None:
        super().__init__()
        self.setObjectName("VersionPage")
        self.version_info = version_info or current_version_info()
        self.update_client = update_client or UpdateClient()
        self.task_runner = QtTaskRunner(self)
        self.update_result = None  # type: UpdateCheckResult | None
        self._active_operation = None  # type: str | None
        self._action_columns = None  # type: int | None

        self._build_ui()
        self.download_progress_changed.connect(self._show_download_progress)
        self._apply_action_layout(force=True)

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(24, 24, 24, 24)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("PageScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        root_layout.addWidget(self.scroll_area)

        content = QWidget()
        content.setObjectName("PageContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(16)

        title = QLabel("版本信息")
        title.setObjectName("PageTitle")
        content_layout.addWidget(title)

        subtitle = QLabel("查看当前运行版本、更新时间、项目仓库和可用正式更新。")
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        content_layout.addWidget(subtitle)

        info_card = SectionCard("当前版本")
        info_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        info_grid = QGridLayout()
        info_grid.setHorizontalSpacing(18)
        info_grid.setVerticalSpacing(12)
        info_grid.setColumnStretch(1, 1)

        self.current_version_label = self._make_value_label(f"v{self.version_info.version}", "当前版本")
        self.current_updated_label = self._make_value_label(self.version_info.updated_at, "当前版本更新时间")
        self.current_mode_label = self._make_value_label(
            _display_install_mode(self.version_info.install_mode),
            "当前运行方式",
        )
        self.repository_label = QLabel(
            f'<a href="{self.version_info.repository_url}">{self.version_info.repository_url}</a>'
        )
        self.repository_label.setObjectName("VersionValue")
        self.repository_label.setAccessibleName("项目仓库链接")
        self.repository_label.setWordWrap(True)
        self.repository_label.setOpenExternalLinks(True)
        self.repository_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)

        self._add_info_row(info_grid, 0, "当前版本", self.current_version_label)
        self._add_info_row(info_grid, 1, "更新时间", self.current_updated_label)
        self._add_info_row(info_grid, 2, "运行方式", self.current_mode_label)
        self._add_info_row(info_grid, 3, "项目仓库", self.repository_label)
        info_card.content_layout.addLayout(info_grid)
        info_card.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        content_layout.addWidget(info_card)

        update_card = SectionCard("检查更新", "检查只在后台访问官方 GitHub Release，不会阻塞程序启动。")
        update_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        update_grid = QGridLayout()
        update_grid.setHorizontalSpacing(18)
        update_grid.setVerticalSpacing(12)
        update_grid.setColumnStretch(1, 1)

        self.latest_version_label = self._make_value_label("尚未检查", "最新版本")
        self.latest_updated_label = self._make_value_label("尚未检查", "最新版本发布时间")
        self.last_checked_label = self._make_value_label("尚未检查", "上次检查时间")
        self.update_status_label = QLabel("点击“检查更新”查询最新正式版本。")
        self.update_status_label.setObjectName("ParameterHint")
        self.update_status_label.setAccessibleName("更新状态")
        self.update_status_label.setWordWrap(True)

        self._add_info_row(update_grid, 0, "最新版本", self.latest_version_label)
        self._add_info_row(update_grid, 1, "发布时间", self.latest_updated_label)
        self._add_info_row(update_grid, 2, "上次检查", self.last_checked_label)
        update_card.content_layout.addLayout(update_grid)
        update_card.content_layout.addWidget(self.update_status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("InlineBusyProgress")
        self.progress_bar.setAccessibleName("更新进度")
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        update_card.content_layout.addWidget(self.progress_bar)

        self.actions_widget = QWidget()
        self.actions_layout = QGridLayout(self.actions_widget)
        self.actions_layout.setContentsMargins(0, 0, 0, 0)
        self.actions_layout.setHorizontalSpacing(10)
        self.actions_layout.setVerticalSpacing(10)

        self.check_button = make_button("检查更新", primary=True)
        self.update_button = make_button("一键更新", primary=True)
        self.repository_button = make_button("打开项目仓库")
        self.release_button = make_button("打开最新发布页面")
        for button in (
            self.check_button,
            self.update_button,
            self.repository_button,
            self.release_button,
        ):
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.update_button.hide()
        self.release_button.hide()
        self.check_button.clicked.connect(self.check_for_updates)
        self.update_button.clicked.connect(self.install_update)
        self.repository_button.clicked.connect(self.open_repository)
        self.release_button.clicked.connect(self.open_latest_release)
        update_card.content_layout.addWidget(self.actions_widget)
        update_card.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        content_layout.addWidget(update_card)
        content_layout.addStretch(1)

        self.scroll_area.setWidget(content)

    @staticmethod
    def _make_value_label(text: str, accessible_name: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("VersionValue")
        label.setAccessibleName(accessible_name)
        label.setWordWrap(True)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        return label

    @staticmethod
    def _add_info_row(layout: QGridLayout, row: int, title: str, value: QLabel) -> None:
        title_label = QLabel(title)
        title_label.setObjectName("MutedText")
        title_label.setMinimumWidth(80)
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        layout.addWidget(title_label, row, 0)
        layout.addWidget(value, row, 1)

    def check_for_updates(self) -> None:
        if not self.task_runner.run(
            lambda: self.update_client.check(self.version_info),
            self._on_check_success,
            self._on_task_error,
        ):
            return
        self._set_busy(True, "正在查询官方最新版本…", indeterminate=True)
        self._active_operation = "check"
        self.status_changed.emit("正在检查更新。")

    def _on_check_success(self, value: object) -> None:
        result = cast(UpdateCheckResult, value)
        self._active_operation = None
        self.update_result = result
        self.latest_version_label.setText(f"v{result.latest.version}")
        self.latest_updated_label.setText(_display_datetime(result.latest.published_at))
        self.last_checked_label.setText(_display_datetime(result.checked_at))
        self.release_button.show()

        if result.state is UpdateState.UP_TO_DATE:
            message = "当前已是最新正式版本。"
            self.update_button.hide()
        elif result.state is UpdateState.CURRENT_AHEAD:
            message = "当前版本高于公开正式版本，可能是待发布或开发版本。"
            self.update_button.hide()
        elif result.can_install:
            message = f"发现正式版本 v{result.latest.version}，已通过更新元数据校验。"
            self.update_button.setText(f"更新到 v{result.latest.version}")
            self.update_button.show()
        elif (
            result.compatible_asset is not None
            and result.compatible_asset.installable
            and not result.dependencies_compatible
        ):
            message = "发现新版本，但运行依赖契约缺失或已变化；为避免不完整升级，请打开发布页面手动更新。"
            self.update_button.hide()
        else:
            message = "发现新版本，但发布资产不完整、大小超限或缺少 SHA-256；请打开发布页面手动更新。"
            self.update_button.hide()

        self._set_busy(False, message)
        self.status_changed.emit(message)
        self._apply_action_layout(force=True)

    def install_update(self) -> None:
        result = self.update_result
        if result is None or not result.can_install:
            return
        mode_note = ""
        if self.version_info.install_mode == "editable":
            mode_note = "\n\n当前为源码开发模式。更新会把本环境切换为经过校验的正式 wheel；不会修改源码仓库。"
        answer = QMessageBox.question(
            self,
            "确认更新 Vision Workbench",
            f"将从 v{self.version_info.version} 更新到 v{result.latest.version}。"
            "\n程序会在下载并校验完成后退出，由独立更新助手安装并重新启动。"
            f"{mode_note}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        if not self.task_runner.run(
            lambda: prepare_update(result, self.download_progress_changed.emit),
            self._on_update_prepared,
            self._on_task_error,
        ):
            return
        self._set_busy(True, f"正在下载 v{result.latest.version}…", indeterminate=False)
        self._active_operation = "download"
        self.status_changed.emit(f"正在下载 v{result.latest.version}。")

    def _on_update_prepared(self, value: object) -> None:
        prepared = cast(PreparedUpdate, value)
        self._active_operation = None
        try:
            launch_update_helper(prepared)
        except Exception as exc:
            self._on_task_error(exc)
            return
        message = f"v{prepared.version} 已下载并校验，正在退出并交给更新助手安装。"
        self._set_busy(True, message, indeterminate=True)
        self.status_changed.emit(message)
        app = QApplication.instance()
        if app is not None:
            QTimer.singleShot(0, app.quit)

    def _on_task_error(self, exc: Exception) -> None:
        checking_for_updates = self._active_operation == "check"
        if checking_for_updates:
            self.update_result = None
            self.update_button.hide()
            self.release_button.hide()
            self.latest_version_label.setText("查询失败")
            self.latest_updated_label.setText("查询失败")
        self._active_operation = None
        if checking_for_updates:
            reason = str(exc).strip() or "未知原因"
            message = (
                f"检查更新失败：{reason}\n"
                "抱歉，暂时无法自动查询最新版本，请点击上方“项目仓库”链接手动查询。"
            )
        else:
            message = f"更新操作失败：{exc}"
        self._set_busy(False, message)
        self.status_changed.emit(message)

    def _show_download_progress(self, downloaded: object, total: object) -> None:
        downloaded_bytes = max(0, int(downloaded))
        total_bytes = int(total) if total is not None else 0
        self.progress_bar.show()
        if total_bytes <= 0:
            self.progress_bar.setRange(0, 0)
            return
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(min(100, int(downloaded_bytes * 100 / total_bytes)))
        self.update_status_label.setText(
            f"正在下载更新：{_format_megabytes(downloaded_bytes)} / {_format_megabytes(total_bytes)}"
        )

    def _set_busy(self, busy: bool, message: str, *, indeterminate: bool = False) -> None:
        self.check_button.setEnabled(not busy)
        self.update_button.setEnabled(not busy)
        self.repository_button.setEnabled(not busy)
        self.release_button.setEnabled(not busy)
        self.update_status_label.setText(message)
        if busy:
            self.progress_bar.show()
            if indeterminate:
                self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.hide()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)

    def open_repository(self) -> None:
        QDesktopServices.openUrl(QUrl(self.version_info.repository_url))
        self.status_changed.emit("已打开项目仓库。")

    def open_latest_release(self) -> None:
        if self.update_result is None:
            return
        QDesktopServices.openUrl(QUrl(self.update_result.latest.release_url))
        self.status_changed.emit("已打开最新发布页面。")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_action_layout()

    def _apply_action_layout(self, *, force: bool = False) -> None:
        columns = self._preferred_action_columns()
        if not force and columns == self._action_columns:
            return
        self._action_columns = columns
        buttons = (
            self.check_button,
            self.update_button,
            self.repository_button,
            self.release_button,
        )
        for button in buttons:
            self.actions_layout.removeWidget(button)
        for index in range(4):
            self.actions_layout.setColumnStretch(index, 0)
        for index, button in enumerate(buttons):
            self.actions_layout.addWidget(button, index // columns, index % columns)
        for index in range(columns):
            self.actions_layout.setColumnStretch(index, 1)
        for first, second in zip(buttons, buttons[1:]):
            QWidget.setTabOrder(first, second)

    def _preferred_action_columns(self) -> int:
        """Choose 1, 2, or 4 columns without compressing button labels."""

        buttons = (
            self.check_button,
            self.update_button,
            self.repository_button,
            self.release_button,
        )
        widths = [button.minimumSizeHint().width() for button in buttons]
        spacing = self.actions_layout.horizontalSpacing()
        scrollbar = self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        available = max(0, self.width() - 24 * 2 - 8 * 2 - 18 * 2 - scrollbar)
        four_columns = sum(widths) + spacing * 3
        two_columns = max(widths[0], widths[2]) + max(widths[1], widths[3]) + spacing
        if max(_ACTION_FOUR_COLUMN_MIN_WIDTH, four_columns) <= available:
            return 4
        if max(_ACTION_TWO_COLUMN_MIN_WIDTH, two_columns) <= available:
            return 2
        return 1

    def shutdown(self) -> None:
        self.task_runner.shutdown()


def _display_datetime(value: str) -> str:
    text = value.strip()
    if not text:
        return "未知"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    return parsed.strftime("%Y-%m-%d %H:%M")


def _format_megabytes(value: int) -> str:
    return f"{max(0, value) / (1024 * 1024):.1f} MB"


def _display_install_mode(value: str) -> str:
    return {
        "editable": "editable 源码（当前工作区）",
        "wheel": "Python wheel 安装",
        "single-file": "Windows 单文件 EXE",
    }.get(value, value)
