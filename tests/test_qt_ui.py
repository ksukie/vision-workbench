import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

qt_probe = subprocess.run(
    [sys.executable, "-c", "from PySide6.QtWidgets import QApplication"],
    env={**os.environ, "QT_QPA_PLATFORM": os.environ["QT_QPA_PLATFORM"]},
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    check=False,
)
if qt_probe.returncode != 0:
    pytest.skip("PySide6 QtWidgets is not available in this environment.", allow_module_level=True)

from PySide6.QtCore import QEventLoop, QPoint, QPointF, QTimer, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QMessageBox, QPushButton, QScrollArea

from camera_diagnostics.domain import CameraBackend, CameraDevice, CaptureProfile, PlatformInfo
from cv_basics.api import EffectName
from image_classification.api import PredictionItem, PredictionResult, PretrainedWeightInfo
from panorama_reconstruction.domain import ImagePairPaths, PanoramaResult
from vision_workbench.desktop.image_presenter import QtImagePresenter
from vision_workbench.desktop.main_window import MainWindow, NAV_ITEM_BY_KEY
from vision_workbench.desktop.pages.camera_page import CameraPage
from vision_workbench.desktop.pages.classification_page import ClassificationPage
from vision_workbench.desktop.pages.cv_basics_page import CvBasicsPage, EFFECT_ZH_TO_NAME
from vision_workbench.desktop.pages.panorama_page import MODE_MANUAL, PanoramaPage
from vision_workbench.desktop.pages.yolo_detection_page import (
    LiveDetectionPayload,
    LiveFramePayload,
    YoloDetectionPage,
    _busy_notice_text,
)
from vision_workbench.desktop.pages.yolo_segmentation_page import YoloSegmentationPage
from vision_workbench.desktop.pages.yolo_training_page import YoloTrainingPage
from vision_workbench.desktop.task_runner import QtTaskRunner
from vision_workbench.desktop.theme import APP_QSS
from vision_workbench.desktop.widgets import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
    SELECTED_DISPLAY_ROLE,
)
from yolo26_detection.domain import DetectionBox, DetectionOutput, ModelInfo
from yolo26_segmentation.domain import ModelInfo as SegmentationModelInfo, SegmentationOutput
from vision_workbench.desktop.camera_resource import CameraResourceCoordinator


class LazyLoadService:
    def __init__(self):
        self.load_calls = []
        self.apply_calls = []

    def load_image(self, path):
        self.load_calls.append(Path(path))
        return np.full((12, 16, 3), 64, dtype=np.uint8)

    def save_image(self, image, path):
        pass

    def apply_effect(self, image, effect_name, params):
        self.apply_calls.append(effect_name)
        return image

    def get_image_info(self, image):
        raise NotImplementedError

    def list_effects(self):
        return []


class FakeClassificationService:
    def __init__(self):
        self.predict_calls = []
        self.clear_cache_calls = []

    def supported_models(self):
        return ["resnet18", "mobilenet_v3_small"]

    def pretrained_weight_status(self, model_name=None):
        return [
            PretrainedWeightInfo(
                model_name=model_name or "resnet18",
                filename="fake.pth",
                local_path=Path("models") / "fake.pth",
                exists=True,
            )
        ]

    def download_pretrained_weight(self, model_name, progress_callback=None):
        if progress_callback is not None:
            progress_callback(100, 10, 10)
        return self.pretrained_weight_status(model_name)[0]

    def import_pretrained_weight(self, model_name, source_path):
        return self.pretrained_weight_status(model_name)[0]

    def clear_pretrained_cache(self, model_name=None):
        self.clear_cache_calls.append(model_name)

    def predict_with_pretrained(self, model_name, image_path, topk=5, device="auto"):
        self.predict_calls.append((model_name, Path(image_path), topk, device))
        return PredictionResult(
            image_path=Path(image_path),
            model_name=model_name,
            model_path=None,
            inference_ms=12.5,
            predictions=[
                PredictionItem("class-a", 0.8),
                PredictionItem("class-b", 0.2),
            ],
        )

    def predict_with_checkpoint(self, model_path, image_path, topk=5, device="auto"):
        return self.predict_with_pretrained(str(model_path), image_path, topk, device)


class FakePanoramaService:
    def __init__(self):
        self.load_calls = []
        self.auto_calls = []
        self.manual_calls = []
        self.assisted_calls = []
        self.save_image_calls = []
        self.save_outputs_calls = []
        self.saved_point_pairs = []
        self.sample_paths = ImagePairPaths(left=Path("sample-left.png"), right=Path("sample-right.png"))

    def load_image(self, path):
        image_path = Path(path)
        self.load_calls.append(image_path)
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, :, 0] = 80 if "left" in image_path.name else 120
        return image

    def save_image(self, image, path):
        self.save_image_calls.append((np.asarray(image).shape, Path(path)))

    def save_outputs(self, result, output_dir):
        self.save_outputs_calls.append((result.method, Path(output_dir)))
        return {
            "panorama": Path(output_dir) / "panorama.png",
            "feature_matches": Path(output_dir) / "feature_matches.png",
        }

    def load_point_pairs(self, path):
        return [
            ((1.0, 1.0), (2.0, 2.0)),
            ((6.0, 1.0), (7.0, 2.0)),
            ((1.0, 6.0), (2.0, 7.0)),
        ]

    def save_point_pairs(self, path, point_pairs):
        self.saved_point_pairs.append((Path(path), list(point_pairs)))

    def get_sample_image_paths(self):
        return self.sample_paths

    def reconstruct(self, left, right, params=None):
        self.auto_calls.append(params.channel_name if params is not None else None)
        return self._result("automatic")

    def reconstruct_from_points(self, left, right, point_pairs):
        self.manual_calls.append(list(point_pairs))
        return self._result("manual-perspective")

    def reconstruct_assisted_from_points(self, left, right, point_pairs, params=None):
        self.assisted_calls.append(list(point_pairs))
        return self._result("manual-assisted-tps")

    def _result(self, method):
        image = np.full((18, 28, 3), 96, dtype=np.uint8)
        return PanoramaResult(
            panorama=image,
            warped_right=image.copy(),
            match_visualization=image.copy(),
            mapped_points_visualization=image.copy(),
            raw_match_count=4,
            balanced_match_count=4,
            inlier_count=4,
            channel_name="gray",
            method=method,
        )


class FakeCameraService:
    def __init__(self):
        self.backend = CameraBackend("FAKE", 0)
        self.devices = [CameraDevice(index=0, backend=self.backend, name="Fake Camera")]
        self.profiles = [
            CaptureProfile(width=320, height=240, fps=30.0, fourcc="MJPG", backend_name="FAKE")
        ]
        self.discover_calls = 0
        self.probe_calls = []
        self.open_calls = []
        self.close_calls = 0
        self.read_calls = 0
        self.saved_screenshots = []
        self.open = False

    def get_platform_info(self):
        return PlatformInfo(system="TestOS", backends=(self.backend,))

    def discover_cameras(self):
        self.discover_calls += 1
        return list(self.devices)

    def probe_profiles(self, device):
        self.probe_calls.append(device)
        return list(self.profiles)

    def open_camera(self, device, profile=None):
        self.open_calls.append((device, profile))
        self.open = True
        return self.profiles[0]

    def close_camera(self):
        self.close_calls += 1
        self.open = False

    def is_camera_open(self):
        return self.open

    def read_frame(self):
        self.read_calls += 1
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, 1] = 160
        return frame

    def save_screenshot(self, frame, path):
        self.saved_screenshots.append((np.asarray(frame).shape, Path(path)))


class FakeYoloDetectionService:
    def __init__(self):
        self.backend = CameraBackend("FAKE", 0)
        self.cameras = [CameraDevice(index=0, backend=self.backend, name="Fake YOLO Camera")]
        self.models = [
            ModelInfo("yolo26n.pt", Path("models") / "yolo26n.pt", True, True),
            ModelInfo("yolo26s.pt", Path("models") / "yolo26s.pt", False, True),
        ]
        self.discover_calls = 0
        self.open_camera_calls = []
        self.close_calls = 0
        self.read_calls = 0
        self.list_calls = []
        self.add_calls = []
        self.download_calls = []
        self.load_calls = []
        self.detect_calls = []
        self.saved_screenshots = []
        self.loaded_path = None
        self.camera_open = False

    def discover_cameras(self):
        self.discover_calls += 1
        return list(self.cameras)

    def open_camera(self, camera, requested_size=None):
        self.open_camera_calls.append((camera, requested_size))
        self.camera_open = True

    def close_camera(self):
        self.close_calls += 1
        self.camera_open = False

    def is_camera_open(self):
        return self.camera_open

    def read_frame(self):
        self.read_calls += 1
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = 90
        return frame

    def list_models(self, include_missing_official=True):
        self.list_calls.append(include_missing_official)
        return list(self.models)

    def add_custom_model(self, path):
        self.add_calls.append(Path(path))
        model = ModelInfo(Path(path).name, Path(path), True, False)
        self.models.append(model)
        return model

    def download_official_model(self, name, progress_callback=None):
        if progress_callback is not None:
            progress_callback(100, 10, 10)
        self.download_calls.append(name)
        model = ModelInfo(name, Path("models") / name, True, True)
        self.models = [model if item.name == name else item for item in self.models]
        return model

    def load_model(self, model_path):
        self.loaded_path = Path(model_path)
        self.load_calls.append(self.loaded_path)

    def loaded_model_path(self):
        return self.loaded_path

    def detect_frame(self, frame, settings):
        self.detect_calls.append((np.asarray(frame).shape, settings))
        annotated = np.asarray(frame).copy()
        annotated[:, :, 2] = 255
        return DetectionOutput(
            annotated_frame=annotated,
            detections=(
                DetectionBox(0, "cup", 0.82, (1.0, 2.0, 20.0, 22.0)),
                DetectionBox(1, "phone", 0.51, (5.0, 6.0, 18.0, 20.0)),
            ),
            inference_ms=14.0,
        )

    def save_screenshot(self, frame, path):
        self.saved_screenshots.append((np.asarray(frame).shape, Path(path)))


class FakeYoloSegmentationService:
    def __init__(self):
        self.models = [
            SegmentationModelInfo("yolo26n-seg.pt", Path("models") / "yolo26n-seg.pt", "segment", True, True),
            SegmentationModelInfo("yolo26n-sem.pt", Path("models") / "yolo26n-sem.pt", "semantic", False, True),
        ]
        self.list_calls = []
        self.add_calls = []
        self.download_calls = []
        self.load_image_calls = []
        self.save_image_calls = []
        self.load_model_calls = []
        self.segment_calls = []
        self.loaded_path = None

    def list_models(self, task="segment", include_missing_official=True):
        self.list_calls.append((task, include_missing_official))
        return [
            model
            for model in self.models
            if model.task == task and (include_missing_official or model.exists)
        ]

    def add_custom_model(self, path, task="segment"):
        self.add_calls.append((Path(path), task))
        model = SegmentationModelInfo(Path(path).name, Path(path), task, True, False)
        self.models.append(model)
        return model

    def download_official_model(self, name, task="segment", progress_callback=None):
        if progress_callback is not None:
            progress_callback(100, 10, 10)
        self.download_calls.append((name, task))
        downloaded = SegmentationModelInfo(name, Path("models") / name, task, True, True)
        self.models = [
            downloaded if model.name == name and model.task == task else model
            for model in self.models
        ]
        if downloaded not in self.models:
            self.models.append(downloaded)
        return downloaded

    def load_image(self, path):
        self.load_image_calls.append(Path(path))
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, :, 1] = 120
        return image

    def save_image(self, image, path):
        self.save_image_calls.append((np.asarray(image).shape, Path(path)))

    def load_model(self, model_path):
        self.loaded_path = Path(model_path)
        self.load_model_calls.append(self.loaded_path)

    def unload_model(self):
        self.loaded_path = None

    def segment_image(self, image, settings):
        self.segment_calls.append((np.asarray(image).shape, settings))
        annotated = np.asarray(image).copy()
        annotated[:, :, 2] = 210
        return SegmentationOutput(
            annotated_frame=annotated,
            item_count=2,
            inference_ms=18.0,
            names=("person", "chair"),
        )


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _wheel_event():
    return QWheelEvent(
        QPointF(4, 4),
        QPointF(4, 4),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )


def test_qt_main_window_smoke(qt_app):
    window = MainWindow()
    try:
        window.set_current_page("cv_basics")
        assert window.stack.currentWidget() is window.pages["cv_basics"]
        assert window.findChild(QFrame, "AppShell") is not None
        assert window.findChild(QFrame, "WindowTitleBar") is not None
        assert window.pages["cv_basics"].findChild(QScrollArea) is not None
        assert isinstance(window.pages["panorama"], PanoramaPage)
        assert isinstance(window.pages["camera"], CameraPage)
        assert isinstance(window.pages["detection"], YoloDetectionPage)
        assert isinstance(window.pages["segmentation"], YoloSegmentationPage)
        assert isinstance(window.pages["training"], YoloTrainingPage)
        assert isinstance(window.pages["classification"], ClassificationPage)
        assert all(page.findChild(QScrollArea) is not None for page in window.pages.values())
        assert window.windowFlags() & Qt.WindowType.FramelessWindowHint

        button_texts = {button.text() for button in window.findChildren(QPushButton)}
        assert {"打开图片", "保存结果", "重置", "应用效果"}.issubset(button_texts)
        assert {"-", "□", "×"}.issubset(button_texts)
        assert window.findChild(QLabel, "SidebarDetail").text() == NAV_ITEM_BY_KEY["cv_basics"].description

        window.set_current_page("panorama")
        assert window.findChild(QLabel, "SidebarDetail").text() == NAV_ITEM_BY_KEY["panorama"].description
    finally:
        window.close()


def test_qt_main_window_maximize_restore_keeps_shell_state(qt_app):
    window = MainWindow()
    try:
        shell = window.findChild(QFrame, "AppShell")
        assert shell is not None
        shadow = shell.graphicsEffect()
        assert shadow is not None

        window.showMaximized()
        qt_app.processEvents()
        window._update_window_state()

        assert shell.graphicsEffect() is shadow
        assert not shadow.isEnabled()

        window.showNormal()
        qt_app.processEvents()
        window._update_window_state()

        assert shell.graphicsEffect() is shadow
        assert shadow.isEnabled()
        assert window.stack.currentWidget() is window.pages["cv_basics"]
    finally:
        window.close()


def test_qt_main_window_page_roundtrip_updates_navigation(qt_app):
    window = MainWindow()
    try:
        for key, button in window.nav_buttons.items():
            window.set_current_page(key)

            assert window.stack.currentWidget() is window.pages[key]
            assert button.isChecked()
            assert window.nav_detail_label is not None
            assert window.nav_detail_label.text() == NAV_ITEM_BY_KEY[key].description

        window.pages["cv_basics"].status_changed.emit("处理中")

        assert window.title_bar is not None
        assert window.title_bar.status_label.text() == "处理中"
    finally:
        window.close()


def test_qt_title_bar_compacts_long_timing_status(qt_app):
    window = MainWindow()
    try:
        text = "预测完成，已使用缓存模型（加载/准备 53.2 ms | 推理 25.1 ms | 总计 78.3 ms）。"

        window.set_status(text)

        assert window.title_bar is not None
        assert window.title_bar.status_label.text() == "预测完成，已使用缓存模型。"
        assert window.title_bar.status_label.toolTip() == text
    finally:
        window.close()


def test_primary_buttons_have_disabled_style_after_primary_style():
    primary_index = APP_QSS.index('QPushButton[variant="primary"]')
    disabled_index = APP_QSS.index('QPushButton[variant="primary"]:disabled')

    assert disabled_index > primary_index
    assert "color: #94a3b8;" in APP_QSS[disabled_index:]


def test_choice_and_numeric_inputs_ignore_mouse_wheel(qt_app):
    combo = NoWheelComboBox()
    combo.addItems(["A", "B", "C"])
    combo.setCurrentIndex(0)

    spin = NoWheelSpinBox()
    spin.setRange(0, 10)
    spin.setValue(5)

    double_spin = NoWheelDoubleSpinBox()
    double_spin.setRange(0.0, 1.0)
    double_spin.setSingleStep(0.1)
    double_spin.setValue(0.5)

    combo.wheelEvent(_wheel_event())
    spin.wheelEvent(_wheel_event())
    double_spin.wheelEvent(_wheel_event())

    assert combo.currentIndex() == 0
    assert spin.value() == 5
    assert double_spin.value() == 0.5
    assert spin.alignment() & Qt.AlignmentFlag.AlignHCenter
    assert double_spin.alignment() & Qt.AlignmentFlag.AlignHCenter


def test_qt_main_window_has_no_legacy_launcher(qt_app):
    window = MainWindow()
    try:
        assert not hasattr(window, "launch_legacy_module")
        assert not hasattr(window, "child_processes")
    finally:
        window.close()


def test_cv_effect_labels_map_to_service_names():
    assert EFFECT_ZH_TO_NAME["灰度图"] == EffectName.GRAYSCALE
    assert EFFECT_ZH_TO_NAME["边缘检测"] == EffectName.EDGES
    assert EFFECT_ZH_TO_NAME["透视变换"] == EffectName.PERSPECTIVE_WARP


def test_cv_page_builds_processing_params(qt_app):
    page = CvBasicsPage()
    try:
        page.blur_slider.set_value(13)
        page.edge_low_slider.set_value(44)
        page.edge_high_slider.set_value(144)
        page.threshold_slider.set_value(99)
        page.morph_kernel_slider.set_value(7)
        page.morph_iterations_slider.set_value(3)
        page.rotate_slider.set_value(45)
        page.scale_slider.set_value(150)
        page.crop_slider.set_value(80)
        page.perspective_slider.set_value(16)

        params = page.current_params()

        assert page.scroll_area.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
        assert page.open_button.isEnabled()
        assert not page.save_button.isEnabled()
        assert not page.reset_button.isEnabled()
        assert not page.apply_button.isEnabled()
        assert params.blur_kernel == 13
        assert params.edge_low == 44
        assert params.edge_high == 144
        assert params.threshold == 99
        assert params.morphology_kernel == 7
        assert params.morphology_iterations == 3
        assert params.rotate_angle == 45
        assert params.scale_percent == 150
        assert params.crop_percent == 80
        assert params.perspective_shift == 16
    finally:
        page.shutdown()
        page.close()


def test_cv_page_open_preview_defers_service_load(qt_app, tmp_path):
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((10, 14, 3), dtype=np.uint8)).save(image_path)
    service = LazyLoadService()
    page = CvBasicsPage(service=service)
    try:
        page._open_preview(image_path)

        assert service.load_calls == []
        assert page.current_path == image_path
        assert page.original_image is None
        assert page.result_image is None
        assert page.save_button.isEnabled()
        assert page.reset_button.isEnabled()
        assert page.apply_button.isEnabled()

        payload = page._load_and_apply_effect(
            None,
            image_path,
            EffectName.GRAYSCALE,
            page.current_params(),
        )

        assert service.load_calls == [image_path]
        assert service.apply_calls == [EffectName.GRAYSCALE]
        assert payload.result.shape == payload.original.shape
        assert payload.timings.load_ms >= 0
        assert payload.timings.process_ms >= 0
    finally:
        page.shutdown()
        page.close()


def test_panorama_page_builds_without_loading_images_and_resizes(qt_app):
    service = FakePanoramaService()
    page = PanoramaPage(service=service)
    try:
        assert service.load_calls == []
        assert page.open_left_button.text() == "打开左图（参考）"
        assert page.open_right_button.text() == "打开右图（待拼接）"
        assert page.reconstruct_button.text() == "重建全景"
        assert not page.reconstruct_button.isEnabled()
        assert page.channel_combo.isEnabled()
        assert page.points_card is not None
        assert page.points_card.isHidden()

        page.mode_combo.setCurrentText(MODE_MANUAL)

        assert not page.points_card.isHidden()

        page.resize(900, 700)
        page._apply_responsive_layout(force=True)

        assert page.input_splitter is not None
        assert page.input_splitter.orientation() == Qt.Orientation.Vertical

        page.resize(1280, 800)
        page._apply_responsive_layout(force=True)

        assert page.input_splitter.orientation() == Qt.Orientation.Horizontal
    finally:
        page.shutdown()
        page.close()


def test_panorama_page_pair_loading_and_manual_reconstruction_flow(qt_app, tmp_path):
    left_path = tmp_path / "left.png"
    right_path = tmp_path / "right.png"
    service = FakePanoramaService()
    page = PanoramaPage(service=service)
    try:
        payload = page._load_pair_from_paths(left_path, right_path)
        page._on_pair_loaded(payload)

        assert service.load_calls == [left_path, right_path]
        assert page.left_path == left_path
        assert page.right_path == right_path
        assert page.reconstruct_button.isEnabled()

        page.mode_combo.setCurrentText(MODE_MANUAL)

        assert not page.reconstruct_button.isEnabled()

        for index in range(3):
            page._on_input_click("left", (float(index + 1), float(index + 2)))
            page._on_input_click("right", (float(index + 3), float(index + 4)))

        assert len(page.point_pairs) == 3
        assert page.pending_left_point is None
        assert page.reconstruct_button.isEnabled()

        result = page._reconstruct(
            page.current_mode(),
            page.current_channel_name(),
            page.left_image.copy(),
            page.right_image.copy(),
            page.point_pairs,
        )
        page._on_panorama_ready(result)

        assert len(service.manual_calls) == 1
        assert service.manual_calls[0] == page.point_pairs
        assert page.result is not None
        assert page.save_panorama_button.isEnabled()
        assert page.save_all_button.isEnabled()
        assert "重建" in page.last_timing_text
    finally:
        page.shutdown()
        page.close()


def test_panorama_page_click_mapping_tracks_original_image_size(qt_app, tmp_path):
    service = FakePanoramaService()
    page = PanoramaPage(service=service)
    try:
        payload = page._load_pair_from_paths(tmp_path / "left.png", tmp_path / "right.png")
        page._on_pair_loaded(payload)
        page.left_preview.canvas.resize(300, 200)
        display_rect = page.left_preview._scaled_pixmap_rect()

        mapped = page.left_preview.map_canvas_point(display_rect.center())

        assert mapped is not None
        assert 14.0 <= mapped[0] <= 18.0
        assert 10.0 <= mapped[1] <= 14.0
    finally:
        page.shutdown()
        page.close()


def test_panorama_page_auto_reconstruction_uses_selected_channel(qt_app, tmp_path):
    service = FakePanoramaService()
    page = PanoramaPage(service=service)
    try:
        payload = page._load_pair_from_paths(tmp_path / "left.png", tmp_path / "right.png")
        page._on_pair_loaded(payload)
        page.channel_combo.setCurrentIndex(page.channel_combo.findData("r"))

        result = page._reconstruct(
            page.current_mode(),
            page.current_channel_name(),
            page.left_image.copy(),
            page.right_image.copy(),
            [],
        )
        page._on_panorama_ready(result)

        assert service.auto_calls == ["r"]
        assert page.result is not None
        assert page.info_label.text()
    finally:
        page.shutdown()
        page.close()


def test_camera_page_builds_without_scanning_and_resizes(qt_app):
    service = FakeCameraService()
    page = CameraPage(service=service, camera_coordinator=CameraResourceCoordinator())
    try:
        assert service.discover_calls == 0
        assert page.refresh_button.text() == "查找相机"
        assert page.probe_button.text() == "查询相机支持格式"
        assert page.open_button.text() == "打开相机"
        assert page.close_button.text() == "停止预览"
        assert page.profile_label.text() == "支持格式"
        assert not page.open_button.isEnabled()
        assert "TestOS" in page.platform_label.text()

        page.resize(900, 700)
        page._apply_responsive_layout(force=True)

        assert page.preview_panel.minimumHeight() == 360

        page.resize(1280, 800)
        page._apply_responsive_layout(force=True)

        assert page.preview_panel.minimumHeight() == 460
    finally:
        page.shutdown()
        page.close()


def test_camera_page_scan_probe_open_frame_and_close_flow(qt_app):
    service = FakeCameraService()
    page = CameraPage(service=service, camera_coordinator=CameraResourceCoordinator())
    try:
        scan = page._discover_cameras()
        page._on_devices_ready(scan)

        assert service.discover_calls == 1
        assert page.selected_device() == service.devices[0]
        assert page.open_button.isEnabled()

        probe = page._probe_profiles(service.devices[0])
        page._on_profiles_ready(probe)

        assert service.probe_calls == [service.devices[0]]
        assert page.selected_profile() == service.profiles[0]

        opened = page._open_selected_camera(page.selected_device(), page.selected_profile())
        page._on_camera_opened(opened)

        assert service.open_calls == [(service.devices[0], service.profiles[0])]
        assert page.close_button.isEnabled()
        assert not page.open_button.isEnabled()

        frame = service.read_frame()
        page._on_frame_ready(frame, time.perf_counter())

        assert page.current_frame is frame
        assert page.screenshot_button.isEnabled()
        assert "画面：32x24" in page.info_label.text()

        page.close_camera()

        assert service.close_calls >= 1
        assert page.current_frame is None
        assert not page.close_button.isEnabled()
    finally:
        page.shutdown()
        page.close()


def test_camera_page_screenshot_delegates_to_service(qt_app, tmp_path):
    service = FakeCameraService()
    page = CameraPage(service=service, camera_coordinator=CameraResourceCoordinator())
    try:
        frame = np.zeros((8, 10, 3), dtype=np.uint8)
        output = tmp_path / "frame.png"

        payload = page._save_frame(frame, output)

        assert payload.path == output
        assert service.saved_screenshots == [((8, 10, 3), output)]
    finally:
        page.shutdown()
        page.close()


def test_camera_resource_coordinator_prompts_other_camera_pages(qt_app, monkeypatch):
    coordinator = CameraResourceCoordinator()
    camera_page = CameraPage(service=FakeCameraService(), camera_coordinator=coordinator)
    yolo_page = YoloDetectionPage(service=FakeYoloDetectionService(), camera_coordinator=coordinator)
    classification_page = ClassificationPage(service=FakeClassificationService())
    messages = []

    def fake_information(_parent, title, text):
        messages.append((title, text))
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "information", fake_information)
    try:
        camera_scan = camera_page._discover_cameras()
        camera_page._on_devices_ready(camera_scan)
        yolo_scan = yolo_page._discover_cameras()
        yolo_page._on_cameras_ready(yolo_scan)

        assert camera_page.refresh_button.isEnabled()
        assert camera_page.open_button.isEnabled()
        assert yolo_page.refresh_cameras_button.isEnabled()
        assert yolo_page.open_camera_button.isEnabled()

        assert coordinator.reserve(yolo_page._camera_owner_id, yolo_page._camera_owner_label)
        yolo_page._camera_open = True
        camera_page._update_action_states()
        yolo_page._update_action_states()

        assert camera_page.refresh_button.isEnabled()
        assert camera_page.probe_button.isEnabled()
        assert camera_page.open_button.isEnabled()
        assert classification_page.open_button.isEnabled()
        assert classification_page.import_weights_button.isEnabled()
        assert classification_page.browse_checkpoint_button.isEnabled()
        assert yolo_page.close_camera_button.isEnabled()
        assert not camera_page._reserve_camera_resource()
        assert "YOLO 检测" in messages[-1][1]

        yolo_page.close_live_camera(show_status=False)

        assert camera_page.refresh_button.isEnabled()
        assert camera_page.open_button.isEnabled()
        assert classification_page.open_button.isEnabled()

        assert coordinator.reserve(camera_page._camera_owner_id, camera_page._camera_owner_label)
        camera_page._camera_open = True
        camera_page._update_action_states()
        yolo_page._update_action_states()

        assert yolo_page.refresh_cameras_button.isEnabled()
        assert yolo_page.open_camera_button.isEnabled()
        assert camera_page.close_button.isEnabled()
        assert not yolo_page._reserve_camera_resource()
        assert "相机诊断" in messages[-1][1]

        camera_page.close_camera(show_status=False)

        assert yolo_page.refresh_cameras_button.isEnabled()
        assert yolo_page.open_camera_button.isEnabled()

        assert coordinator.reserve(camera_page._camera_owner_id, camera_page._camera_owner_label)
        with pytest.raises(RuntimeError):
            camera_page._run_releasing_camera_on_error(lambda: (_ for _ in ()).throw(RuntimeError("scan failed")))

        assert coordinator.owner_id() is None
    finally:
        camera_page.shutdown()
        camera_page.close()
        yolo_page.shutdown()
        yolo_page.close()
        classification_page.shutdown()
        classification_page.close()


def test_camera_discovery_cache_is_shared_across_camera_pages(qt_app):
    coordinator = CameraResourceCoordinator()
    camera_service = FakeCameraService()
    yolo_service = FakeYoloDetectionService()
    camera_page = CameraPage(service=camera_service, camera_coordinator=coordinator)
    yolo_page = YoloDetectionPage(service=yolo_service, camera_coordinator=coordinator)
    extra_camera_page = None
    try:
        yolo_page._on_cameras_ready(yolo_page._discover_cameras())

        assert camera_page.selected_device() is not None
        assert camera_page.selected_device().key() == yolo_service.cameras[0].key()
        assert camera_page.open_button.isEnabled()

        camera_page._on_profiles_ready(camera_page._probe_profiles(camera_page.selected_device()))
        extra_camera_page = CameraPage(service=FakeCameraService(), camera_coordinator=coordinator)

        assert extra_camera_page.selected_device() is not None
        assert extra_camera_page.selected_device().key() == yolo_service.cameras[0].key()
        assert extra_camera_page.selected_profile() == camera_service.profiles[0]

        camera_service.devices = []
        camera_page._on_devices_ready(camera_page._discover_cameras())

        assert yolo_page.selected_camera() is None
        assert not yolo_page.open_camera_button.isEnabled()
    finally:
        if extra_camera_page is not None:
            extra_camera_page.shutdown()
            extra_camera_page.close()
        camera_page.shutdown()
        camera_page.close()
        yolo_page.shutdown()
        yolo_page.close()


def test_yolo_detection_page_builds_without_loading_model_and_resizes(qt_app):
    service = FakeYoloDetectionService()
    page = YoloDetectionPage(service=service, camera_coordinator=CameraResourceCoordinator())
    try:
        assert service.discover_calls == 0
        assert service.list_calls == [True]
        assert service.load_calls == []
        assert service.detect_calls == []
        assert page.open_image_button.text() == "选择图片"
        assert page.detect_button.text() == "检测图片"
        assert page.save_result_button.text() == "保存检测图"
        assert page.refresh_cameras_button.text() == "查找相机"
        assert page.refresh_cameras_button.property("variant") == "primary"
        assert page.open_camera_button.text() == "打开相机"
        assert page.start_live_button.text() == "开始摄像头检测"
        assert page.stop_live_button.text() == "停止摄像头检测"
        assert page.close_camera_button.text() == "关闭相机"
        assert page.model_combo.currentText() == str(Path("models") / "yolo26n.pt")
        assert page.model_combo.itemData(page.model_combo.currentIndex(), SELECTED_DISPLAY_ROLE) == "yolo26n.pt"
        missing_model_text = page.model_combo.itemText(1)
        assert str(Path("models") / "yolo26s.pt") in missing_model_text
        assert page.model_combo.itemData(1, SELECTED_DISPLAY_ROLE) == "yolo26s.pt"
        assert "missing" not in missing_model_text.lower()
        assert page.model_status_label.text() == "官方模型：本地可用 | yolo26n.pt"
        assert page.model_status_label.toolTip().endswith(str(Path("models") / "yolo26n.pt"))
        assert "\\" not in page.model_status_label.text()
        assert page.detect_button.isEnabled() is False
        assert page.open_camera_button.isEnabled() is False
        assert page.start_live_button.isEnabled() is False
        assert page.stop_live_button.isEnabled() is False
        assert page.close_camera_button.isEnabled() is False
        assert page.original_preview is page.result_preview
        assert not hasattr(page, "splitter")
        assert "窗口仍在工作" in _busy_notice_text("正在加载 yolo26n.pt，实时检测即将开始...")

        page.resize(900, 700)
        page._apply_responsive_layout(force=True)

        assert page.preview_panel.minimumHeight() == 360

        page.resize(1280, 800)
        page._apply_responsive_layout(force=True)

        assert page.preview_panel.minimumHeight() == 460
    finally:
        page.shutdown()
        page.close()


def test_yolo_detection_page_image_detection_loads_then_reuses_model(qt_app, tmp_path):
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((24, 32, 3), dtype=np.uint8)).save(image_path)
    service = FakeYoloDetectionService()
    page = YoloDetectionPage(service=service, camera_coordinator=CameraResourceCoordinator())
    try:
        page.image_path = image_path
        page.preview_panel.set_pixmap(page._pixmap_from_path(image_path))
        page._update_action_states()

        assert page.detect_button.isEnabled()

        model = page.selected_model()
        payload = page._detect_image(image_path, model, page.current_settings(), first_load=True)
        page._on_detection_ready(payload)

        assert service.load_calls == [model.path]
        assert len(service.detect_calls) == 1
        assert service.detect_calls[0][0] == (24, 32, 3)
        assert page.results_list.count() == 2
        assert page.save_result_button.isEnabled()
        assert "推理" in page.last_timing_text

        second = page._detect_image(image_path, model, page.current_settings(), first_load=False)
        page._on_detection_ready(second)

        assert service.load_calls == [model.path]
        assert len(service.detect_calls) == 2
        assert page._loaded_model_path == model.path
    finally:
        page.shutdown()
        page.close()


def test_yolo_detection_page_live_camera_detection_flow(qt_app, monkeypatch):
    service = FakeYoloDetectionService()
    page = YoloDetectionPage(service=service, camera_coordinator=CameraResourceCoordinator())
    monkeypatch.setattr(page, "_start_live_loop", lambda: None)
    try:
        scan = page._discover_cameras()
        page._on_cameras_ready(scan)

        assert service.discover_calls == 1
        assert page.selected_camera() == service.cameras[0]
        assert page.open_camera_button.isEnabled()
        assert not page.start_live_button.isEnabled()
        assert not page.stop_live_button.isEnabled()
        assert not page.close_camera_button.isEnabled()

        opened = page._open_live_camera(page.selected_camera())
        page._on_live_camera_opened(opened)
        page._live_stop = threading.Event()

        assert service.open_camera_calls == [
            (service.cameras[0], (page.config.requested_capture_width, page.config.requested_capture_height))
        ]
        assert page.start_live_button.isEnabled()
        assert not page.detect_button.isEnabled()

        model = page.selected_model()
        live_start = page._load_live_model(model, page.current_settings())
        page._on_live_model_ready(live_start)

        assert service.load_calls == [model.path]
        assert page._live_detection_enabled
        assert not page.model_combo.isEnabled()

        frame = service.read_frame()
        output = service.detect_frame(frame, page.current_settings())
        page._on_live_detection_ready(
            LiveDetectionPayload(
                output=output,
                inference_fps=14.0,
            )
        )
        page._on_live_frame_ready(
            LiveFramePayload(
                frame=frame,
                output=None,
                camera_fps=12.5,
                inference_fps=14.0,
            )
        )

        assert page.result_image is not output.annotated_frame
        assert page.results_list.count() == 2
        assert page.save_result_button.isEnabled()
        assert "推理 14.0 FPS" in page.live_status_label.text()
        assert "检测 2" in page.live_status_label.text()

        next_frame = frame.copy()
        next_frame[:, :, 0] = 33
        page._on_live_frame_ready(
            LiveFramePayload(
                frame=next_frame,
                output=None,
                camera_fps=12.5,
                inference_fps=4.0,
            )
        )

        assert page.result_image[0, 0, 0] == 33
        assert tuple(page.result_image[2, 1]) == (0, 128, 255)

        page.stop_live_detection()

        assert not page._live_detection_enabled
        assert page._camera_open

        page.close_live_camera()

        assert not page._camera_open
        assert page.result_image is None
        assert page.results_list.count() == 0
        assert page.preview_panel._source_pixmap is None
        assert not page.save_result_button.isEnabled()
        assert not page.start_live_button.isEnabled()
        assert not page.stop_live_button.isEnabled()
        assert not page.close_camera_button.isEnabled()
        assert service.close_calls >= 1
    finally:
        page.shutdown()
        page.close()


def test_yolo_detection_page_download_and_save_delegate_to_service(qt_app, tmp_path, monkeypatch):
    service = FakeYoloDetectionService()
    page = YoloDetectionPage(service=service, camera_coordinator=CameraResourceCoordinator())
    messages = []

    def fake_information(_parent, title, text):
        messages.append((title, text))
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "information", fake_information)
    try:
        page.model_combo.setCurrentIndex(1)
        missing_model = page.selected_model()

        payload = page._download_model(missing_model.name)
        page._on_model_downloaded(payload)

        assert service.download_calls == [missing_model.name]
        assert page.selected_model().exists
        assert messages[-1][0] == "下载完成"
        assert str(page.selected_model().path) in messages[-1][1]

        result = np.zeros((8, 10, 3), dtype=np.uint8)
        output = tmp_path / "result.png"
        saved = page._save_detection(result, output)

        assert saved.path == output
        assert service.saved_screenshots == [((8, 10, 3), output)]
    finally:
        page.shutdown()
        page.close()


def test_yolo_segmentation_page_builds_without_loading_model_and_resizes(qt_app):
    service = FakeYoloSegmentationService()
    page = YoloSegmentationPage(service=service)
    try:
        assert service.list_calls == [("segment", True)]
        assert service.load_image_calls == []
        assert service.load_model_calls == []
        assert service.segment_calls == []
        assert page.task_combo.currentText() == "实例分割"
        assert page.model_combo.currentText() == str(Path("models") / "yolo26n-seg.pt")
        assert page.model_combo.itemData(page.model_combo.currentIndex(), SELECTED_DISPLAY_ROLE) == "yolo26n-seg.pt"
        assert page.model_status_label.text() == "官方实例分割模型：本地可用 | yolo26n-seg.pt"
        assert page.model_status_label.toolTip().endswith(str(Path("models") / "yolo26n-seg.pt"))
        assert "\\" not in page.model_status_label.text()
        assert page.segment_button.isEnabled() is False
        assert page.save_result_button.isEnabled() is False
        assert page.download_model_button.isEnabled() is False
        assert page.select_image_button.isEnabled()

        page.resize(900, 700)
        page._apply_responsive_layout(force=True)

        assert page.preview_panel.minimumHeight() == 360

        page.resize(1280, 800)
        page._apply_responsive_layout(force=True)

        assert page.preview_panel.minimumHeight() == 460
    finally:
        page.shutdown()
        page.close()


def test_yolo_segmentation_page_load_segment_reuse_and_save(qt_app, tmp_path):
    image_path = tmp_path / "sample.png"
    output_path = tmp_path / "segmented.png"
    service = FakeYoloSegmentationService()
    page = YoloSegmentationPage(service=service)
    try:
        image_payload = page._load_image(image_path)
        page._on_image_loaded(image_payload)

        assert service.load_image_calls == [image_path]
        assert page.source_image is not None
        assert page.segment_button.isEnabled()
        assert not page.save_result_button.isEnabled()

        model = page.selected_model()
        payload = page._segment_image(page.source_image.copy(), model, page.current_settings(), first_load=True)
        page._on_segmentation_ready(payload)

        assert service.load_model_calls == [model.path]
        assert len(service.segment_calls) == 1
        assert service.segment_calls[0][0] == (24, 32, 3)
        assert page.items_list.count() == 2
        assert page.save_result_button.isEnabled()
        assert "推理" in page.last_timing_text

        second = page._segment_image(page.source_image.copy(), model, page.current_settings(), first_load=False)
        page._on_segmentation_ready(second)

        assert service.load_model_calls == [model.path]
        assert len(service.segment_calls) == 2
        assert page._loaded_model_path == model.path

        saved = page._save_result(page.result_image.copy(), output_path)

        assert saved.path == output_path
        assert service.save_image_calls == [((24, 32, 3), output_path)]
    finally:
        page.shutdown()
        page.close()


def test_yolo_segmentation_page_task_switch_controls_model_chain(qt_app, tmp_path, monkeypatch):
    service = FakeYoloSegmentationService()
    page = YoloSegmentationPage(service=service)
    messages = []

    def fake_information(_parent, title, text):
        messages.append((title, text))
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "information", fake_information)
    try:
        semantic_index = page.task_combo.findData("semantic")
        assert semantic_index >= 0

        page.task_combo.setCurrentIndex(semantic_index)
        image_payload = page._load_image(tmp_path / "sample.png")
        page._on_image_loaded(image_payload)

        assert service.list_calls[-1] == ("semantic", True)
        assert page.task_combo.currentText() == "语义分割"
        semantic_model_text = page.model_combo.currentText()
        assert str(Path("models") / "yolo26n-sem.pt") in semantic_model_text
        assert page.model_combo.itemData(page.model_combo.currentIndex(), SELECTED_DISPLAY_ROLE) == "yolo26n-sem.pt"
        assert "missing" not in semantic_model_text.lower()
        assert page.selected_model().name == "yolo26n-sem.pt"
        assert not page.selected_model().exists
        assert page.segment_button.isEnabled() is False
        assert page.download_model_button.isEnabled()

        missing_model = page.selected_model()
        downloaded = page._download_model(missing_model)
        page._on_model_downloaded(downloaded)

        assert service.download_calls == [("yolo26n-sem.pt", "semantic")]
        assert page.selected_model().exists
        assert messages[-1][0] == "下载完成"
        assert str(page.selected_model().path) in messages[-1][1]
        assert page.download_model_button.isEnabled() is False
        assert page.segment_button.isEnabled()
    finally:
        page.shutdown()
        page.close()


def test_classification_page_open_preview_and_prediction(qt_app, tmp_path):
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((10, 14, 3), dtype=np.uint8)).save(image_path)
    service = FakeClassificationService()
    page = ClassificationPage(service=service)
    try:
        pixmap = page._pixmap_from_path(image_path)
        page.image_path = image_path
        page.preview_panel.set_pixmap(pixmap)
        page._update_action_states()

        assert service.predict_calls == []
        assert page.predict_pretrained_button.isEnabled()

        payload = page._predict_pretrained("resnet18", image_path, 2, "cpu")
        page._show_prediction(payload)

        assert service.predict_calls == [("resnet18", image_path, 2, "cpu")]
        assert page.results_list.count() == 2
        assert "推理" in page.last_timing_text

        page.model_combo.setCurrentText("mobilenet_v3_small")

        assert service.clear_cache_calls == []
        assert page.results_list.count() == 0
        assert page.last_timing_text is None
    finally:
        page.shutdown()
        page.close()


def test_classification_page_uses_chinese_labels_and_compact_layout(qt_app):
    page = ClassificationPage(service=FakeClassificationService())
    try:
        assert page.open_button.text() == "打开图片"
        assert page.predict_pretrained_button.text() == "预测"
        assert page.check_weights_button.text() == "检查权重"
        assert page.browse_checkpoint_button.text() == "选择本地模型文件"
        assert page.predict_checkpoint_button.text() == "预测自定义模型"
        assert page.model_label.text() == "模型"
        assert page.device_label.text() == "设备"
        assert page.checkpoint_label.text().startswith("模型文件")

        page.resize(900, 700)
        page._apply_responsive_layout(force=True)

        assert page.splitter is not None
        assert page.splitter.orientation() == Qt.Orientation.Vertical
        assert page.results_panel is not None
        assert page.results_panel.minimumWidth() == 0
        assert page.preview_panel.minimumHeight() == 320

        page.resize(1280, 800)
        page._apply_responsive_layout(force=True)

        assert page.splitter.orientation() == Qt.Orientation.Horizontal
        assert page.results_panel.minimumWidth() == 280
    finally:
        page.shutdown()
        page.close()


def test_classification_page_pretrained_prediction_status_mentions_cache(qt_app, tmp_path):
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((10, 14, 3), dtype=np.uint8)).save(image_path)
    page = ClassificationPage(service=FakeClassificationService())
    statuses = []

    def collect_status(text):
        statuses.append(text)
        if text.startswith("预测完成"):
            loop.quit()

    page.status_changed.connect(collect_status)
    try:
        page.image_path = image_path
        page._update_action_states()

        loop = QEventLoop()
        page.predict_pretrained()
        QTimer.singleShot(1000, loop.quit)
        loop.exec()

        assert ("resnet18", "auto") in page._warmed_pretrained_keys
        assert any("首次加载 resnet18 模型" in text for text in statuses)
        assert any("模型已缓存" in text for text in statuses)

        statuses.clear()
        loop = QEventLoop()
        page.predict_pretrained()
        QTimer.singleShot(1000, loop.quit)
        loop.exec()

        assert any("正在使用已缓存的 resnet18 模型预测" in text for text in statuses)
        assert any("已使用缓存模型" in text for text in statuses)
    finally:
        page.shutdown()
        page.close()


def test_cv_effect_parameters_follow_selected_effect(qt_app):
    page = CvBasicsPage()
    try:
        assert not page.parameter_hint.isHidden()
        assert page.blur_slider.isHidden()
        assert page.edge_low_slider.isHidden()

        page.effect_combo.setCurrentText("边缘检测")

        assert page.parameter_hint.isHidden()
        assert page.blur_slider.isHidden()
        assert not page.edge_low_slider.isHidden()
        assert not page.edge_high_slider.isHidden()
    finally:
        page.shutdown()
        page.close()


def test_qt_image_presenter_creates_non_empty_pixmap(qt_app):
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[:, :, 0] = 40
    image[:, :, 1] = 160
    image[:, :, 2] = 220

    pixmap = QtImagePresenter((120, 120)).to_pixmap(image)

    assert not pixmap.isNull()
    assert pixmap.width() > 0
    assert pixmap.height() > 0


def test_qt_task_runner_drops_callbacks_after_shutdown(qt_app):
    runner = QtTaskRunner()
    callbacks = []
    loop = QEventLoop()

    accepted = runner.run(
        task=lambda: (time.sleep(0.05), "done")[1],
        on_success=lambda value: callbacks.append(value),
        on_error=lambda exc: callbacks.append(exc),
    )
    runner.shutdown()
    QTimer.singleShot(200, loop.quit)
    loop.exec()

    assert accepted
    assert callbacks == []


def test_qt_task_runner_can_start_next_task_from_callback(qt_app):
    runner = QtTaskRunner()
    callbacks = []
    loop = QEventLoop()

    def on_success(value):
        callbacks.append(value)
        if value == "first":
            assert runner.run(lambda: "second", on_success, callbacks.append)
        else:
            loop.quit()

    accepted = runner.run(lambda: "first", on_success, callbacks.append)
    QTimer.singleShot(1000, loop.quit)
    loop.exec()
    runner.shutdown()

    assert accepted
    assert callbacks == ["first", "second"]
