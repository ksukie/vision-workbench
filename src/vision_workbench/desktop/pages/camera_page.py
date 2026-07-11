"""Qt page for camera diagnostics."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, cast

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from camera_diagnostics.api import create_camera_diagnostics_service
from camera_diagnostics.application import CameraDiagnosticsService
from camera_diagnostics.configuration import CameraDiagnosticsConfig
from camera_diagnostics.domain import CameraBackend, CameraDevice, CaptureProfile, ImageArray, PlatformInfo
from vision_workbench.troubleshooting import CAMERA_AND_VIDEO, DATA_AND_FILES, with_help

from ..camera_resource import CameraResourceCoordinator, shared_camera_coordinator
from ..image_presenter import QtImagePresenter
from ..task_runner import QtTaskRunner
from ..widgets import NoWheelComboBox as QComboBox
from ..widgets import PreviewPanel, SectionCard, associate_form_label, make_button, style_form_label


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceScanPayload:
    devices: list[CameraDevice]
    total_ms: float


@dataclass(frozen=True)
class ProfileProbePayload:
    device: CameraDevice
    profiles: list[CaptureProfile]
    total_ms: float


@dataclass(frozen=True)
class OpenCameraPayload:
    device: CameraDevice
    requested_profile: CaptureProfile
    actual_profile: CaptureProfile
    total_ms: float


@dataclass(frozen=True)
class SaveFramePayload:
    path: Path
    total_ms: float


class CameraPage(QWidget):
    """Native Qt implementation of the camera diagnostics workflow."""

    status_changed = Signal(str)
    frame_ready = Signal(object, float)
    frame_failed = Signal(object)

    def __init__(
        self,
        service: Optional[CameraDiagnosticsService] = None,
        config: CameraDiagnosticsConfig = CameraDiagnosticsConfig(),
        camera_coordinator: CameraResourceCoordinator | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.service = service or create_camera_diagnostics_service(config)
        self.presenter = QtImagePresenter(config.preview_size)
        self.tasks = QtTaskRunner(self)
        self.camera_coordinator = camera_coordinator or shared_camera_coordinator()
        self._camera_owner_id = "camera_diagnostics"
        self._camera_owner_label = "相机诊断"
        self.platform_info = self.service.get_platform_info()

        self.devices = []  # type: list[CameraDevice]
        self.profiles = []  # type: list[CaptureProfile]
        self.current_frame = None  # type: Optional[ImageArray]
        self.actual_profile = None  # type: Optional[CaptureProfile]
        self.last_timing_text = None  # type: Optional[str]
        self._completion_status = None  # type: Optional[str]
        self._busy = False
        self._camera_open = False
        self._display_fps = 0.0
        self._last_frame_time = None  # type: Optional[float]
        self._read_failures = 0
        self._camera_lock = threading.Lock()
        self._preview_stop = None  # type: Optional[threading.Event]
        self._preview_thread = None  # type: Optional[threading.Thread]
        self._preview_interval = 1.0 / max(1.0, config.default_recording_fps)

        self.refresh_button = make_button("查找相机", primary=True)
        self.probe_button = make_button("查询相机支持格式")
        self.open_button = make_button("打开相机", primary=True)
        self.close_button = make_button("停止预览")
        self.screenshot_button = make_button("保存截图")

        for button in (
            self.refresh_button,
            self.probe_button,
            self.open_button,
            self.close_button,
            self.screenshot_button,
        ):
            button.setMinimumWidth(112)

        self.device_label = QLabel("相机")
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(220)
        self.device_combo.setToolTip("选择要打开的摄像头路由")

        self.profile_label = QLabel("支持格式")
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(260)
        self.profile_combo.setToolTip("选择相机支持的读取格式")
        for label in (self.device_label, self.profile_label):
            style_form_label(label)
        associate_form_label(self.device_label, self.device_combo)
        associate_form_label(self.profile_label, self.profile_combo)

        self.platform_label = QLabel(_platform_text(self.platform_info))
        self.platform_label.setObjectName("MutedText")
        self.platform_label.setWordWrap(True)

        self.fps_label = QLabel("FPS：--")
        self.fps_label.setObjectName("MutedText")
        self.profile_info_label = QLabel("格式：--")
        self.profile_info_label.setObjectName("MutedText")
        self.profile_info_label.setWordWrap(True)

        self.preview_panel = PreviewPanel("相机预览", "查找并打开相机后显示画面")
        self.info_label = QLabel("")
        self.info_label.setObjectName("MutedText")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.content = None  # type: QWidget | None
        self.controls_layout = None  # type: QGridLayout | None
        self.stats_layout = None  # type: QGridLayout | None
        self._compact_layout = None  # type: bool | None

        self._build_ui()
        self._connect_signals()
        self.camera_coordinator.changed.connect(self._update_action_states)
        self.camera_coordinator.devices_changed.connect(self._apply_cached_camera_devices)
        self.camera_coordinator.profiles_changed.connect(self._apply_cached_profiles)
        self._apply_cached_camera_devices()
        self._update_info()
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

        title = QLabel("相机诊断")
        title.setObjectName("PageTitle")
        subtitle = QLabel("枚举摄像头、查询相机支持格式，并进行实时预览与截图。")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        controls_card = SectionCard("相机")
        self.controls_layout = QGridLayout()
        self.controls_layout.setHorizontalSpacing(10)
        self.controls_layout.setVerticalSpacing(10)
        controls_card.content_layout.addLayout(self.controls_layout)
        layout.addWidget(controls_card)

        stats_card = SectionCard("状态")
        self.stats_layout = QGridLayout()
        self.stats_layout.setHorizontalSpacing(14)
        self.stats_layout.setVerticalSpacing(10)
        stats_card.content_layout.addLayout(self.stats_layout)
        layout.addWidget(stats_card)

        self.preview_panel.setMinimumHeight(460)
        layout.addWidget(self.preview_panel, 1)
        layout.addWidget(self.info_label)

        scroll_area.setWidget(content)
        root_layout.addWidget(scroll_area)
        self._apply_responsive_layout(force=True)

    def _connect_signals(self) -> None:
        self.refresh_button.clicked.connect(self.refresh_cameras)
        self.probe_button.clicked.connect(self.probe_profiles)
        self.open_button.clicked.connect(self.open_camera)
        self.close_button.clicked.connect(self.close_camera)
        self.screenshot_button.clicked.connect(self.save_screenshot)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.frame_ready.connect(self._on_frame_ready)
        self.frame_failed.connect(self._on_frame_failed)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def _apply_responsive_layout(self, *, force: bool = False) -> None:
        controls = self.controls_layout
        stats = self.stats_layout
        if controls is None or stats is None:
            return

        compact = self.width() < 1180
        if not force and compact == self._compact_layout:
            return
        self._compact_layout = compact

        self._reset_grid(controls, 8)
        self._reset_grid(stats, 4)
        if compact:
            self._layout_compact_controls(controls)
            self._layout_compact_stats(stats)
            self.preview_panel.setMinimumHeight(360)
        else:
            self._layout_wide_controls(controls)
            self._layout_wide_stats(stats)
            self.preview_panel.setMinimumHeight(460)

    def _layout_wide_controls(self, controls: QGridLayout) -> None:
        controls.addWidget(self.refresh_button, 0, 0)
        controls.addWidget(self.device_label, 0, 1, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.device_combo, 0, 2)
        controls.addWidget(self.probe_button, 0, 3)
        controls.addWidget(self.profile_label, 0, 4, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.profile_combo, 0, 5)
        controls.addWidget(self.open_button, 0, 6)
        controls.addWidget(self.close_button, 0, 7)
        controls.addWidget(self.screenshot_button, 1, 0)
        controls.setColumnStretch(2, 1)
        controls.setColumnStretch(5, 1)

    def _layout_compact_controls(self, controls: QGridLayout) -> None:
        controls.addWidget(self.refresh_button, 0, 0)
        controls.addWidget(self.probe_button, 0, 1)
        controls.addWidget(self.open_button, 0, 2)
        controls.addWidget(self.close_button, 0, 3)
        controls.addWidget(self.device_label, 1, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.device_combo, 1, 1, 1, 3)
        controls.addWidget(self.profile_label, 2, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.profile_combo, 2, 1, 1, 3)
        controls.addWidget(self.screenshot_button, 3, 0, 1, 4)
        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(3, 1)

    def _layout_wide_stats(self, stats: QGridLayout) -> None:
        stats.addWidget(self.fps_label, 0, 0)
        stats.addWidget(self.profile_info_label, 0, 1)
        stats.addWidget(self.platform_label, 0, 2)
        stats.setColumnStretch(1, 1)
        stats.setColumnStretch(2, 1)

    def _layout_compact_stats(self, stats: QGridLayout) -> None:
        stats.addWidget(self.fps_label, 0, 0)
        stats.addWidget(self.profile_info_label, 1, 0)
        stats.addWidget(self.platform_label, 2, 0)
        stats.setColumnStretch(0, 1)

    def _reset_grid(self, grid: QGridLayout, columns: int) -> None:
        while grid.count():
            item = grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        for index in range(columns):
            grid.setColumnStretch(index, 0)
            grid.setColumnMinimumWidth(index, 0)
        for index in range(5):
            grid.setRowStretch(index, 0)

    def refresh_cameras(self) -> None:
        self.close_camera(show_status=False)
        if not self._reserve_camera_resource():
            return
        self._run_task(
            task=lambda: self._run_releasing_camera_on_error(self._discover_cameras),
            on_success=lambda value: self._on_camera_scan_ready(value, release_camera=True),
            busy_text="正在扫描摄像头...",
            error_title="扫描失败",
            error_category=CAMERA_AND_VIDEO,
        )

    def probe_profiles(self) -> None:
        device = self.selected_device()
        if device is None:
            QMessageBox.information(self, "没有相机", "请先刷新并选择相机。")
            return

        self.close_camera(show_status=False)
        if not self._reserve_camera_resource():
            return
        self._run_task(
            task=lambda: self._run_releasing_camera_on_error(lambda: self._probe_profiles(device)),
            on_success=lambda value: self._on_camera_profiles_ready(value, release_camera=True),
            busy_text=f"正在查询 {device.label()} 的支持格式...",
            error_title="查询失败",
            error_category=CAMERA_AND_VIDEO,
        )

    def open_camera(self) -> None:
        device = self.selected_device()
        if device is None:
            QMessageBox.information(self, "没有相机", "请先刷新并选择相机。")
            return

        profile = self.selected_profile() or _default_profile(device)
        self.close_camera(show_status=False)
        if not self._reserve_camera_resource():
            return
        self._run_task(
            task=lambda: self._run_releasing_camera_on_error(lambda: self._open_selected_camera(device, profile)),
            on_success=self._on_camera_opened,
            busy_text=f"正在打开 {device.label()}...",
            error_title="打开失败",
            error_category=CAMERA_AND_VIDEO,
        )

    def close_camera(self, *, show_status: bool = True) -> None:
        was_open = self._camera_open or self._preview_thread is not None
        self._stop_preview_loop()
        closed = self._close_service_camera()
        self._camera_open = False
        self.current_frame = None
        self.actual_profile = None
        self._display_fps = 0.0
        self._last_frame_time = None
        self._read_failures = 0
        self.preview_panel.clear()
        self.fps_label.setText("FPS：--")
        self.profile_info_label.setText("格式：--")
        self._update_info()
        self._update_action_states()
        self._release_camera_resource()
        if show_status and was_open:
            self._set_status("相机已停止。" if closed else "相机停止请求已发送。")

    def save_screenshot(self) -> None:
        if self.current_frame is None:
            QMessageBox.information(self, "没有画面", "请先打开相机。")
            return

        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存截图",
            "",
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg *.jpeg);;位图 (*.bmp);;所有文件 (*.*)",
        )
        if not path:
            return

        frame = self.current_frame.copy()
        output_path = Path(path)
        self._run_task(
            task=lambda: self._save_frame(frame, output_path),
            on_success=self._on_screenshot_saved,
            busy_text="正在保存截图...",
            error_title="保存失败",
            error_category=DATA_AND_FILES,
        )

    def selected_device(self) -> CameraDevice | None:
        index = self.device_combo.currentIndex()
        if index < 0 or index >= len(self.devices):
            return None
        return self.devices[index]

    def selected_profile(self) -> CaptureProfile | None:
        index = self.profile_combo.currentIndex()
        if index < 0 or index >= len(self.profiles):
            return None
        return self.profiles[index]

    def shutdown(self) -> None:
        self.close_camera(show_status=False)
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

    def _discover_cameras(self) -> DeviceScanPayload:
        started_at = time.perf_counter()
        devices = self.service.discover_cameras()
        return DeviceScanPayload(devices=list(devices), total_ms=_elapsed_ms(started_at))

    def _probe_profiles(self, device: CameraDevice) -> ProfileProbePayload:
        started_at = time.perf_counter()
        profiles = self.service.probe_profiles(device)
        if not profiles:
            profiles = [_default_profile(device)]
        return ProfileProbePayload(
            device=device,
            profiles=list(profiles),
            total_ms=_elapsed_ms(started_at),
        )

    def _open_selected_camera(
        self,
        device: CameraDevice,
        profile: CaptureProfile,
    ) -> OpenCameraPayload:
        started_at = time.perf_counter()
        with self._camera_lock:
            actual = self.service.open_camera(device, profile)
        return OpenCameraPayload(
            device=device,
            requested_profile=profile,
            actual_profile=actual,
            total_ms=_elapsed_ms(started_at),
        )

    def _save_frame(self, frame: ImageArray, path: Path) -> SaveFramePayload:
        started_at = time.perf_counter()
        self.service.save_screenshot(frame, path)
        return SaveFramePayload(path=path, total_ms=_elapsed_ms(started_at))

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

    def _on_devices_ready(self, value: object) -> None:
        payload = cast(DeviceScanPayload, value)
        self.camera_coordinator.update_devices(payload.devices)
        self._set_devices(payload.devices)
        self.last_timing_text = f"扫描 {payload.total_ms:.1f} ms"
        if not self.devices:
            self._completion_status = "未发现相机，请检查权限或连接。"
        else:
            self._completion_status = f"发现 {len(self.devices)} 条相机路由（{self.last_timing_text}）。"
        self._update_info()
        self._update_action_states()

    def _on_camera_scan_ready(self, value: object, *, release_camera: bool) -> None:
        try:
            self._on_devices_ready(value)
        finally:
            if release_camera:
                self._release_camera_resource()

    def _on_profiles_ready(self, value: object) -> None:
        payload = cast(ProfileProbePayload, value)
        self.profiles = payload.profiles or [_default_profile(payload.device)]
        self.camera_coordinator.update_profiles(payload.device, self.profiles)
        self._refresh_profile_combo()
        self.last_timing_text = f"查询 {payload.total_ms:.1f} ms"
        self._completion_status = f"支持格式已更新，共 {len(self.profiles)} 项（{self.last_timing_text}）。"
        self._update_info()
        self._update_action_states()

    def _on_camera_profiles_ready(self, value: object, *, release_camera: bool) -> None:
        try:
            self._on_profiles_ready(value)
        finally:
            if release_camera:
                self._release_camera_resource()

    def _on_camera_opened(self, value: object) -> None:
        payload = cast(OpenCameraPayload, value)
        self._camera_open = True
        self.actual_profile = payload.actual_profile
        self.current_frame = None
        self._display_fps = 0.0
        self._last_frame_time = None
        self._read_failures = 0
        self._preview_interval = _profile_interval(payload.actual_profile, self.config.default_recording_fps)
        self.profile_info_label.setText(f"格式：{payload.actual_profile.label()}")
        self.last_timing_text = f"打开 {payload.total_ms:.1f} ms"
        self._completion_status = f"已打开 {payload.device.label()}（{self.last_timing_text}）。"
        self._update_info()
        self._update_action_states()
        self._start_preview_loop()

    def _on_screenshot_saved(self, value: object) -> None:
        payload = cast(SaveFramePayload, value)
        self.last_timing_text = f"保存 {payload.total_ms:.1f} ms"
        self._completion_status = f"截图已保存（{self.last_timing_text}）。"
        self._update_info()
        QMessageBox.information(self, "保存完成", f"截图已保存到：\n{payload.path}")

    def _on_device_changed(self, _index: int) -> None:
        device = self.selected_device()
        cached_profiles = self.camera_coordinator.cached_profiles_as(device, CaptureProfile)
        self.profiles = cached_profiles or ([_default_profile(device)] if device is not None else [])
        self._refresh_profile_combo()
        self._update_info()
        self._update_action_states()

    def _apply_cached_camera_devices(self) -> None:
        devices = self.camera_coordinator.cached_devices_as(CameraDevice, CameraBackend)
        current = self.selected_device()
        current_key = current.key() if current is not None else ""
        self._set_devices(devices, preferred_key=current_key)
        self._update_info()
        self._update_action_states()

    def _apply_cached_profiles(self, device_key: str) -> None:
        device = self.selected_device()
        if device is None or device.key() != device_key or self._camera_open:
            return
        cached_profiles = self.camera_coordinator.cached_profiles_as(device, CaptureProfile)
        if not cached_profiles:
            return
        self.profiles = cached_profiles
        self._refresh_profile_combo()
        self._update_info()
        self._update_action_states()

    def _set_devices(self, devices: Sequence[CameraDevice], preferred_key: str = "") -> None:
        self.devices = list(devices)
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for device in self.devices:
            self.device_combo.addItem(device.label())
        selected_index = 0
        if preferred_key:
            for index, device in enumerate(self.devices):
                if device.key() == preferred_key:
                    selected_index = index
                    break
        if self.devices:
            self.device_combo.setCurrentIndex(selected_index)
        self.device_combo.blockSignals(False)

        device = self.selected_device()
        cached_profiles = self.camera_coordinator.cached_profiles_as(device, CaptureProfile)
        self.profiles = cached_profiles or ([_default_profile(device)] if device is not None else [])
        self._refresh_profile_combo()

    def _refresh_profile_combo(self) -> None:
        self.profile_combo.clear()
        for profile in self.profiles:
            self.profile_combo.addItem(profile.label())
        if self.profiles:
            self.profile_combo.setCurrentIndex(0)

    def _start_preview_loop(self) -> None:
        self._stop_preview_loop()
        stop_event = threading.Event()
        self._preview_stop = stop_event
        self._preview_thread = threading.Thread(
            target=self._preview_loop,
            args=(stop_event,),
            name="camera-preview",
            daemon=True,
        )
        self._preview_thread.start()

    def _stop_preview_loop(self) -> None:
        stop_event = self._preview_stop
        if stop_event is not None:
            stop_event.set()
        thread = self._preview_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        self._preview_stop = None
        self._preview_thread = None

    def _preview_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            loop_started_at = time.perf_counter()
            try:
                with self._camera_lock:
                    if stop_event.is_set() or not self.service.is_camera_open():
                        break
                    frame = self.service.read_frame()
            except Exception as exc:
                self.frame_failed.emit(exc)
                time.sleep(0.08)
                continue

            self.frame_ready.emit(frame, time.perf_counter())
            elapsed = time.perf_counter() - loop_started_at
            delay = max(0.0, self._preview_interval - elapsed)
            if delay:
                stop_event.wait(delay)

    def _on_frame_ready(self, value: object, received_at: float) -> None:
        if self._preview_stop is None or not self._camera_open:
            return
        frame = cast(ImageArray, value)
        self.current_frame = frame
        self._read_failures = 0
        self._update_fps(received_at)
        self.preview_panel.set_pixmap(self.presenter.to_pixmap(frame))
        self._update_info()
        self._update_action_states()

    def _on_frame_failed(self, value: object) -> None:
        if self._preview_stop is None or not self._camera_open:
            return
        self._read_failures += 1
        if self._read_failures < 5:
            return
        exc = cast(Exception, value)
        self.close_camera(show_status=False)
        QMessageBox.critical(self, "读取失败", with_help(exc, CAMERA_AND_VIDEO))
        self._set_status("相机读取失败，预览已停止。")

    def _update_fps(self, now: float) -> None:
        if self._last_frame_time is not None:
            elapsed = now - self._last_frame_time
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                self._display_fps = (
                    instant_fps
                    if self._display_fps == 0.0
                    else self._display_fps * 0.85 + instant_fps * 0.15
                )
                self.fps_label.setText(f"FPS：{self._display_fps:.1f}")
        self._last_frame_time = now

    def _close_service_camera(self) -> bool:
        acquired = self._camera_lock.acquire(timeout=0.5)
        if not acquired:
            logger.warning("Skipped immediate camera close because frame read is still active.")
            return False
        try:
            self.service.close_camera()
        finally:
            self._camera_lock.release()
        return True

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_action_states()

    def _update_action_states(self) -> None:
        has_device = self.selected_device() is not None
        has_frame = self.current_frame is not None
        self.refresh_button.setEnabled(not self._busy)
        self.probe_button.setEnabled(has_device and not self._busy and not self._camera_open)
        self.open_button.setEnabled(has_device and not self._busy and not self._camera_open)
        self.close_button.setEnabled(self._camera_open and not self._busy)
        self.screenshot_button.setEnabled(has_frame and not self._busy)
        self.device_combo.setEnabled(not self._busy and not self._camera_open)
        self.profile_combo.setEnabled(bool(self.profiles) and not self._busy and not self._camera_open)

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
        selected = self.selected_device()
        if selected is not None:
            parts.append(f"相机：{selected.label()}")
        if self.actual_profile is not None:
            parts.append(f"当前格式：{self.actual_profile.label()}")
        elif self.selected_profile() is not None:
            parts.append(f"待打开格式：{self.selected_profile().label()}")
        if self.current_frame is not None:
            height, width = self.current_frame.shape[:2]
            parts.append(f"画面：{width}x{height}")
        if self.last_timing_text:
            parts.append(self.last_timing_text)
        self.info_label.setText(" | ".join(parts))

    def _set_status(self, text: str) -> None:
        self.status_changed.emit(text)


def _default_profile(device: CameraDevice) -> CaptureProfile:
    return CaptureProfile(
        width=0,
        height=0,
        fps=0.0,
        fourcc="DEFAULT",
        backend_name=device.backend.name,
        is_default=True,
    )


def _platform_text(platform_info: PlatformInfo) -> str:
    return f"系统：{platform_info.label()}"


def _profile_interval(profile: CaptureProfile, fallback_fps: float) -> float:
    fps = profile.fps if profile.fps and profile.fps > 1 else fallback_fps
    fps = min(max(float(fps), 1.0), 60.0)
    return 1.0 / fps


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0
