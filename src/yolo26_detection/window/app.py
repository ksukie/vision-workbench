"""Tkinter GUI for YOLO26 camera object detection."""

from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from yolo26_detection.api import create_yolo26_detection_service
    from yolo26_detection.configuration import Yolo26DetectionConfig
    from yolo26_detection.domain import CameraDevice, DetectionSettings, ImageArray, ModelInfo
    from yolo26_detection.window.presenter import TkDetectionPresenter
    from cv_basics.window.process_exit import arm_forced_process_exit, close_window
else:
    from ..api import create_yolo26_detection_service
    from ..configuration import Yolo26DetectionConfig
    from ..domain import CameraDevice, DetectionSettings, ImageArray, ModelInfo
    from .presenter import TkDetectionPresenter
    from cv_basics.window.process_exit import arm_forced_process_exit, close_window

from vision_workbench.troubleshooting import CAMERA_AND_VIDEO, MODELS_AND_WEIGHTS, MODULE_RUNTIME_ERRORS, with_help


class Yolo26DetectionWindow:
    """Camera GUI that delegates inference to the YOLO26 application service."""

    def __init__(
        self,
        root: tk.Misc,
        config: Yolo26DetectionConfig = Yolo26DetectionConfig(),
        exit_on_close: bool = True,
    ) -> None:
        self.root = root
        self.config = config
        self.exit_on_close = exit_on_close
        self.service = create_yolo26_detection_service(config)
        self.presenter = TkDetectionPresenter(config.preview_size)

        if isinstance(root, (tk.Tk, tk.Toplevel)):
            root.title("YOLO26 Detection Workbench")
            root.geometry("1280x860")
            root.minsize(1040, 720)
            root.protocol("WM_DELETE_WINDOW", self.close)
            root.bind("<Destroy>", self._on_destroy, add="+")

        self.devices = []  # type: List[CameraDevice]
        self.models = []  # type: List[ModelInfo]
        self.current_photo = None
        self.current_display_frame = None  # type: Optional[ImageArray]
        self.latest_frame = None  # type: Optional[ImageArray]
        self.latest_camera_fps = 0.0
        self.latest_inference_fps = 0.0
        self.latest_detection_count = 0
        self.read_failures = 0
        self.loaded_model_path = None  # type: Optional[Path]
        self.recording_path = None  # type: Optional[Path]

        self.frame_lock = threading.Lock()
        self.settings_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.camera_thread = None  # type: Optional[threading.Thread]
        self.preview_after_id = None  # type: Optional[str]
        self.detection_enabled = False
        self.closed = False
        self.active_settings = DetectionSettings(
            image_size=config.default_image_size,
            confidence=config.default_confidence,
            iou=config.default_iou,
            device="auto",
        )

        self.device_var = tk.StringVar(value="")
        self.model_var = tk.StringVar(value="")
        self.device_choice_var = tk.StringVar(value="auto")
        self.image_size_var = tk.StringVar(value=str(config.default_image_size))
        self.conf_var = tk.DoubleVar(value=config.default_confidence)
        self.iou_var = tk.DoubleVar(value=config.default_iou)
        self.platform_var = tk.StringVar(value="")
        self.camera_fps_var = tk.StringVar(value="Camera FPS: --")
        self.inference_fps_var = tk.StringVar(value="Infer FPS: --")
        self.detections_var = tk.StringVar(value="Detections: --")
        self.model_status_var = tk.StringVar(value="Model: --")
        self.recording_var = tk.StringVar(value="Recording: off")
        self.status_var = tk.StringVar(value="Refresh cameras and choose a YOLO26 model.")

        self._build_ui()
        self._update_platform_label()
        self.refresh_models()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(10, 10, 10, 6))
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="Refresh Cameras", command=self.refresh_cameras).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Label(toolbar, text="Camera").pack(side=tk.LEFT, padx=(0, 6))
        self.device_box = ttk.Combobox(
            toolbar,
            textvariable=self.device_var,
            values=[],
            state="readonly",
            width=26,
        )
        self.device_box.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Open", command=self.open_camera).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Close", command=self.close_camera).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(toolbar, text="Model").pack(side=tk.LEFT, padx=(0, 6))
        self.model_box = ttk.Combobox(
            toolbar,
            textvariable=self.model_var,
            values=[],
            state="readonly",
            width=22,
        )
        self.model_box.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Refresh Models", command=self.refresh_models).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Browse PT", command=self.browse_model).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Download Selected", command=self.download_selected_model).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )

        controls = ttk.Frame(self.root, padding=(10, 4, 10, 6))
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Device").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        ttk.Combobox(
            controls,
            textvariable=self.device_choice_var,
            values=self.config.device_options,
            state="readonly",
            width=8,
        ).grid(row=0, column=1, sticky=tk.W, padx=(0, 14))

        ttk.Label(controls, text="Image size").grid(row=0, column=2, sticky=tk.W, padx=(0, 6))
        ttk.Combobox(
            controls,
            textvariable=self.image_size_var,
            values=[str(value) for value in self.config.image_size_options],
            width=8,
        ).grid(row=0, column=3, sticky=tk.W, padx=(0, 14))

        self._add_float_slider(controls, "Conf", self.conf_var, 0.05, 0.95, 4)
        self._add_float_slider(controls, "IoU", self.iou_var, 0.10, 0.95, 8)

        ttk.Button(controls, text="Start Detection", command=self.start_detection).grid(
            row=0,
            column=12,
            sticky=tk.W,
            padx=(12, 8),
        )
        ttk.Button(controls, text="Stop Detection", command=self.stop_detection).grid(
            row=0,
            column=13,
            sticky=tk.W,
            padx=(0, 12),
        )
        ttk.Button(controls, text="Screenshot", command=self.save_screenshot).grid(
            row=0,
            column=14,
            sticky=tk.W,
            padx=(0, 8),
        )
        ttk.Button(controls, text="Start Recording", command=self.start_recording).grid(
            row=0,
            column=15,
            sticky=tk.W,
            padx=(0, 8),
        )
        ttk.Button(controls, text="Stop Recording", command=self.stop_recording).grid(
            row=0,
            column=16,
            sticky=tk.W,
        )

        stats = ttk.Frame(self.root, padding=(10, 2, 10, 8))
        stats.pack(fill=tk.X)
        ttk.Label(stats, textvariable=self.platform_var, width=30).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(stats, textvariable=self.camera_fps_var, width=18).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(stats, textvariable=self.inference_fps_var, width=18).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(stats, textvariable=self.detections_var, width=18).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(stats, textvariable=self.model_status_var, width=30).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(stats, textvariable=self.recording_var, width=28).pack(side=tk.LEFT, padx=(0, 12))

        status = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        status.pack(fill=tk.X)
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Separator(self.root).pack(fill=tk.X)

        body = ttk.Frame(self.root, padding=(10, 10, 10, 10))
        body.pack(fill=tk.BOTH, expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        self.preview_label = tk.Label(
            body,
            bg="#202226",
            fg="#d9dde3",
            text="No camera frame",
            anchor=tk.CENTER,
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        self.conf_var.trace_add("write", lambda *_args: self._update_active_settings())
        self.iou_var.trace_add("write", lambda *_args: self._update_active_settings())
        self.device_choice_var.trace_add("write", lambda *_args: self._update_active_settings())
        self.image_size_var.trace_add("write", lambda *_args: self._update_active_settings())

    def _add_float_slider(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.DoubleVar,
        from_: float,
        to: float,
        base_column: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=0, column=base_column, sticky=tk.W, padx=(0, 6))
        ttk.Scale(parent, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL, length=120).grid(
            row=0,
            column=base_column + 1,
            sticky=tk.W,
            padx=(0, 6),
        )
        value_label = ttk.Label(parent, width=5)
        value_label.grid(row=0, column=base_column + 2, sticky=tk.W, padx=(0, 12))

        def update_label(*_args) -> None:
            value_label.configure(text=f"{variable.get():.2f}")

        variable.trace_add("write", update_label)
        update_label()

    def _update_platform_label(self) -> None:
        platform_info = self.service.get_platform_info()
        self.platform_var.set(f"System: {platform_info.label()}")

    def refresh_cameras(self) -> None:
        self.close_camera()
        self.status_var.set("Scanning cameras...")
        self.root.update_idletasks()
        try:
            self.devices = self.service.discover_cameras()
        except Exception as exc:
            messagebox.showerror("Camera scan failed", with_help(exc, CAMERA_AND_VIDEO))
            self.status_var.set("Camera scan failed.")
            return

        labels = [device.label() for device in self.devices]
        self.device_box.configure(values=labels)
        if not self.devices:
            self.device_var.set("")
            self.status_var.set("No camera found. Check permissions, USB connection, or OS privacy settings.")
            return
        self.device_var.set(labels[0])
        self.status_var.set(f"Found {len(self.devices)} camera route(s).")

    def refresh_models(self) -> None:
        self.models = self.service.list_models(include_missing_official=True)
        labels = [model.label() for model in self.models]
        self.model_box.configure(values=labels)
        selected = next((model for model in self.models if model.exists), None)
        if selected is None and self.models:
            selected = self.models[0]
        self.model_var.set(selected.label() if selected else "")
        if selected and selected.exists:
            self.status_var.set(f"Found YOLO26 model: {selected.path}")
        elif selected:
            self.status_var.set("Official model is listed but missing. Click Download Selected or browse a .pt file.")

    def browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose YOLO26 .pt model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            model = self.service.add_custom_model(path)
        except Exception as exc:
            messagebox.showerror("Model failed", with_help(exc, MODELS_AND_WEIGHTS))
            return
        self.models.append(model)
        self.model_box.configure(values=[item.label() for item in self.models])
        self.model_var.set(model.label())
        self.status_var.set(f"Selected model: {model.path}")

    def download_selected_model(self) -> None:
        model = self._selected_model()
        if model is None:
            messagebox.showinfo("No model", "Choose a YOLO26 model first.")
            return
        if not model.is_official:
            messagebox.showinfo("Custom model", "Only official YOLO26 models can be downloaded here.")
            return
        if model.exists:
            self.status_var.set(f"Model already exists: {model.path}")
            return

        self.status_var.set(f"Downloading {model.name}...")
        thread = threading.Thread(target=self._download_model_worker, args=(model.name,), daemon=True)
        thread.start()

    def _download_model_worker(self, name: str) -> None:
        try:
            model = self.service.download_official_model(name)
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Download failed", with_help(exc, MODELS_AND_WEIGHTS)))
            self.root.after(0, lambda: self.status_var.set("Model download failed."))
            return
        self.root.after(0, lambda: self._on_model_downloaded(model))

    def _on_model_downloaded(self, model: ModelInfo) -> None:
        self.refresh_models()
        self.model_var.set(model.label())
        self.status_var.set(f"Downloaded model: {model.path}")

    def open_camera(self) -> None:
        device = self._selected_device()
        if device is None:
            messagebox.showinfo("No camera", "Refresh and select a camera first.")
            return

        try:
            self.service.open_camera(
                device,
                (self.config.requested_capture_width, self.config.requested_capture_height),
            )
        except Exception as exc:
            messagebox.showerror("Open failed", with_help(exc, CAMERA_AND_VIDEO))
            self.status_var.set("Camera open failed.")
            return

        self.stop_event.clear()
        self.read_failures = 0
        self.latest_camera_fps = 0.0
        self.latest_inference_fps = 0.0
        self.camera_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.camera_thread.start()
        self._schedule_preview()
        self.status_var.set(f"Opened {device.label()} at requested 640x480.")

    def close_camera(self) -> None:
        self.stop_detection()
        self.stop_event.set()
        if self.preview_after_id is not None:
            try:
                self.root.after_cancel(self.preview_after_id)
            except Exception:
                pass
            self.preview_after_id = None
        self.service.close_camera()
        if self.camera_thread is not None and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=10.0)
        if self.camera_thread is not None and not self.camera_thread.is_alive():
            self.camera_thread = None
        with self.frame_lock:
            self.latest_frame = None
            self.current_display_frame = None
        self.current_photo = None
        self.preview_label.configure(image="", text="No camera frame")
        self.camera_fps_var.set("Camera FPS: --")
        self.inference_fps_var.set("Infer FPS: --")
        self.detections_var.set("Detections: --")
        self.recording_var.set("Recording: off")
        self.recording_path = None

    def start_detection(self) -> None:
        if not self.service.is_camera_open():
            messagebox.showinfo("No camera", "Open a camera first.")
            return
        model = self._selected_model()
        if model is None:
            messagebox.showinfo("No model", "Choose or browse a YOLO26 .pt model first.")
            return
        if not model.exists:
            messagebox.showinfo("Missing model", "Download this model first or browse a local .pt file.")
            return

        try:
            settings = self._current_settings()
        except Exception as exc:
            messagebox.showerror("Invalid settings", with_help(exc, MODULE_RUNTIME_ERRORS))
            return

        if self.loaded_model_path != model.path:
            self.status_var.set(f"Loading model: {model.name}...")
            self.root.update_idletasks()
            try:
                self.service.load_model(model.path)
            except Exception as exc:
                messagebox.showerror("Model load failed", with_help(exc, MODELS_AND_WEIGHTS))
                self.status_var.set("Model load failed.")
                return
            self.loaded_model_path = model.path

        with self.settings_lock:
            self.active_settings = settings
        self.detection_enabled = True
        self.model_status_var.set(f"Model: {model.name}")
        self.status_var.set("YOLO26 detection running.")

    def stop_detection(self) -> None:
        self.detection_enabled = False

    def save_screenshot(self) -> None:
        frame = self._current_frame_copy()
        if frame is None:
            messagebox.showinfo("No frame", "Open a camera first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save screenshot",
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("JPEG image", "*.jpg;*.jpeg"),
                ("Bitmap image", "*.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.service.save_screenshot(frame, path)
        except Exception as exc:
            messagebox.showerror("Screenshot failed", with_help(exc, CAMERA_AND_VIDEO))
            return
        self.status_var.set(f"Saved screenshot: {path}")

    def start_recording(self) -> None:
        frame = self._current_frame_copy()
        if frame is None:
            messagebox.showinfo("No frame", "Open a camera first.")
            return
        if self.service.is_recording():
            messagebox.showinfo("Recording", "Recording is already running.")
            return
        path = filedialog.asksaveasfilename(
            title="Save detection recording",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 video", "*.mp4"),
                ("AVI video", "*.avi"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        height, width = frame.shape[:2]
        fps = self.latest_camera_fps if self.latest_camera_fps > 1 else self.config.default_recording_fps
        try:
            self.service.start_recording(path, (width, height), fps)
        except Exception as exc:
            messagebox.showerror("Recording failed", with_help(exc, CAMERA_AND_VIDEO))
            return
        self.recording_path = Path(path)
        self.recording_var.set(f"Recording: {self.recording_path.name}")
        self.status_var.set(f"Recording to {path}")

    def stop_recording(self) -> None:
        if not self.service.is_recording():
            self.recording_var.set("Recording: off")
            return
        self.service.stop_recording()
        saved = str(self.recording_path) if self.recording_path else "video file"
        self.recording_path = None
        self.recording_var.set("Recording: off")
        self.status_var.set(f"Saved recording: {saved}")

    def close(self, destroy_root: bool = True) -> None:
        if self.exit_on_close:
            arm_forced_process_exit()
        if self.closed:
            return
        self.closed = True
        self.close_camera()
        self.service.unload_model()
        self.loaded_model_path = None
        close_window(self.root, exit_on_close=self.exit_on_close, destroy_root=destroy_root)

    def _on_destroy(self, event) -> None:
        if event.widget is self.root and not self.closed:
            self.close(destroy_root=False)

    def _camera_worker(self) -> None:
        last_camera_time = None  # type: Optional[float]
        inference_fps = 0.0
        while not self.stop_event.is_set() and self.service.is_camera_open():
            try:
                frame = self.service.read_frame()
            except Exception as exc:
                self._handle_worker_error(exc)
                return

            now = time.perf_counter()
            camera_fps = self.latest_camera_fps
            if last_camera_time is not None:
                elapsed = now - last_camera_time
                if elapsed > 0:
                    instant = 1.0 / elapsed
                    camera_fps = instant if camera_fps == 0 else camera_fps * 0.85 + instant * 0.15
            last_camera_time = now

            display_frame = frame
            detection_count = 0
            if self.detection_enabled:
                with self.settings_lock:
                    settings = self.active_settings
                try:
                    output = self.service.detect_frame(frame, settings)
                except Exception as exc:
                    self.detection_enabled = False
                    self._handle_worker_error(exc)
                    output = None
                if output is not None:
                    display_frame = output.annotated_frame
                    detection_count = output.detection_count
                    if output.inference_ms > 0:
                        instant_inference_fps = 1000.0 / output.inference_ms
                        inference_fps = (
                            instant_inference_fps
                            if inference_fps == 0
                            else inference_fps * 0.80 + instant_inference_fps * 0.20
                        )

            if self.service.is_recording():
                self.service.write_recording_frame(display_frame)

            with self.frame_lock:
                self.latest_frame = display_frame.copy()
                self.current_display_frame = display_frame.copy()
                self.latest_camera_fps = camera_fps
                self.latest_inference_fps = inference_fps
                self.latest_detection_count = detection_count

    def _handle_worker_error(self, exc: Exception) -> None:
        self.root.after(0, lambda: self.status_var.set(str(exc)))
        self.root.after(0, lambda: messagebox.showerror("Runtime failed", with_help(exc, MODULE_RUNTIME_ERRORS)))
        self.root.after(0, self.close_camera)

    def _schedule_preview(self) -> None:
        if self.preview_after_id is None:
            self.preview_after_id = self.root.after(20, self._refresh_preview)

    def _refresh_preview(self) -> None:
        self.preview_after_id = None
        frame = None
        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
                camera_fps = self.latest_camera_fps
                inference_fps = self.latest_inference_fps
                detection_count = self.latest_detection_count
            else:
                camera_fps = self.latest_camera_fps
                inference_fps = self.latest_inference_fps
                detection_count = self.latest_detection_count

        if frame is not None:
            self.current_photo = self.presenter.to_photo_image(frame)
            self.preview_label.configure(image=self.current_photo, text="")
        self.camera_fps_var.set(f"Camera FPS: {camera_fps:.1f}" if camera_fps else "Camera FPS: --")
        self.inference_fps_var.set(f"Infer FPS: {inference_fps:.1f}" if inference_fps else "Infer FPS: --")
        self.detections_var.set(f"Detections: {detection_count}")

        if self.service.is_camera_open():
            self.preview_after_id = self.root.after(20, self._refresh_preview)

    def _current_settings(self) -> DetectionSettings:
        image_size = int(self.image_size_var.get())
        if image_size <= 0:
            raise ValueError("Image size must be positive.")
        return DetectionSettings(
            image_size=image_size,
            confidence=float(self.conf_var.get()),
            iou=float(self.iou_var.get()),
            device=self.device_choice_var.get(),
        )

    def _update_active_settings(self) -> None:
        try:
            settings = self._current_settings()
        except Exception:
            return
        with self.settings_lock:
            self.active_settings = settings

    def _selected_device(self) -> Optional[CameraDevice]:
        label = self.device_var.get()
        for device in self.devices:
            if device.label() == label:
                return device
        return None

    def _selected_model(self) -> Optional[ModelInfo]:
        label = self.model_var.get()
        for model in self.models:
            if model.label() == label:
                return model
        return None

    def _current_frame_copy(self) -> Optional[ImageArray]:
        with self.frame_lock:
            if self.current_display_frame is None:
                return None
            return self.current_display_frame.copy()


def main() -> None:
    root = tk.Tk()
    Yolo26DetectionWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
