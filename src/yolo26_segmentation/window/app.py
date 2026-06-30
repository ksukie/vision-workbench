"""Tkinter GUI for YOLO26 segmentation."""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from yolo26_segmentation.api import create_yolo26_segmentation_service
    from yolo26_segmentation.configuration import Yolo26SegmentationConfig
    from yolo26_segmentation.domain import ImageArray, ModelInfo, SegmentationSettings
    from yolo26_segmentation.window.presenter import TkSegmentationPresenter
    from cv_basics.window.process_exit import arm_forced_process_exit, terminate_process
else:
    from ..api import create_yolo26_segmentation_service
    from ..configuration import Yolo26SegmentationConfig
    from ..domain import ImageArray, ModelInfo, SegmentationSettings
    from .presenter import TkSegmentationPresenter
    from cv_basics.window.process_exit import arm_forced_process_exit, terminate_process


class Yolo26SegmentationWindow:
    """Independent GUI for instance and semantic segmentation."""

    def __init__(
        self,
        root: tk.Misc,
        config: Yolo26SegmentationConfig = Yolo26SegmentationConfig(),
    ) -> None:
        self.root = root
        self.config = config
        self.service = create_yolo26_segmentation_service(config)
        self.presenter = TkSegmentationPresenter(config.preview_size)

        if isinstance(root, (tk.Tk, tk.Toplevel)):
            root.title("YOLO26 Segmentation Workbench")
            root.geometry("1260x840")
            root.minsize(1040, 700)
            root.protocol("WM_DELETE_WINDOW", self.close)
            root.bind("<Destroy>", self._on_destroy, add="+")

        self.models = []  # type: List[ModelInfo]
        self.source_image = None  # type: Optional[ImageArray]
        self.result_image = None  # type: Optional[ImageArray]
        self.loaded_model_path = None  # type: Optional[Path]
        self.current_photo = None
        self.after_id = None  # type: Optional[str]
        self.segmenting_camera = False
        self.closed = False
        self.last_frame_time = None  # type: Optional[float]
        self.display_fps = 0.0

        self.task_var = tk.StringVar(value="segment")
        self.model_var = tk.StringVar(value="")
        self.image_size_var = tk.StringVar(value=str(config.default_image_size))
        self.conf_var = tk.DoubleVar(value=config.default_confidence)
        self.iou_var = tk.DoubleVar(value=config.default_iou)
        self.device_var = tk.StringVar(value="auto")
        self.camera_index_var = tk.IntVar(value=0)
        self.fps_var = tk.StringVar(value="FPS: --")
        self.items_var = tk.StringVar(value="Items: --")
        self.status_var = tk.StringVar(value="Choose an image or open a camera.")

        self._build_ui()
        self.refresh_models()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(10, 10, 10, 6))
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(toolbar, text="Camera").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(toolbar, textvariable=self.camera_index_var, width=4).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="Open Camera", command=self.open_camera).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Close Camera", command=self.close_camera).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(toolbar, text="Task").pack(side=tk.LEFT, padx=(0, 6))
        self.task_box = ttk.Combobox(
            toolbar,
            textvariable=self.task_var,
            values=self.config.task_options,
            state="readonly",
            width=10,
        )
        self.task_box.pack(side=tk.LEFT, padx=(0, 8))
        self.task_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh_models())

        ttk.Label(toolbar, text="Model").pack(side=tk.LEFT, padx=(0, 6))
        self.model_box = ttk.Combobox(toolbar, textvariable=self.model_var, values=[], state="readonly", width=24)
        self.model_box.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Refresh Models", command=self.refresh_models).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Browse PT", command=self.browse_model).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Download Selected", command=self.download_selected_model).pack(side=tk.LEFT)

        controls = ttk.Frame(self.root, padding=(10, 4, 10, 8))
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Device").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        ttk.Combobox(
            controls,
            textvariable=self.device_var,
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
        ttk.Button(controls, text="Run Once", command=self.run_once).grid(row=0, column=12, sticky=tk.W, padx=(12, 8))
        ttk.Button(controls, text="Start Live", command=self.start_live).grid(row=0, column=13, sticky=tk.W, padx=(0, 8))
        ttk.Button(controls, text="Stop Live", command=self.stop_live).grid(row=0, column=14, sticky=tk.W)

        stats = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        stats.pack(fill=tk.X)
        ttk.Label(stats, textvariable=self.fps_var, width=16).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(stats, textvariable=self.items_var, width=18).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(stats, textvariable=self.status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Separator(self.root).pack(fill=tk.X)
        body = ttk.Frame(self.root, padding=(10, 10, 10, 10))
        body.pack(fill=tk.BOTH, expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        self.preview_label = tk.Label(body, bg="#202226", fg="#d9dde3", text="No image", anchor=tk.CENTER)
        self.preview_label.grid(row=0, column=0, sticky="nsew")

    def _add_float_slider(self, parent, label, variable, from_, to, base_column) -> None:
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

    def refresh_models(self) -> None:
        task = self.task_var.get()
        self.models = self.service.list_models(task, include_missing_official=True)
        labels = [model.label() for model in self.models]
        self.model_box.configure(values=labels)
        selected = next((model for model in self.models if model.exists), None)
        if selected is None and self.models:
            selected = self.models[0]
        self.model_var.set(selected.label() if selected else "")

    def browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose YOLO26 segmentation .pt model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            model = self.service.add_custom_model(path, self.task_var.get())
        except Exception as exc:
            messagebox.showerror("Model failed", str(exc))
            return
        self.models.append(model)
        self.model_box.configure(values=[item.label() for item in self.models])
        self.model_var.set(model.label())

    def download_selected_model(self) -> None:
        model = self._selected_model()
        if model is None:
            messagebox.showinfo("No model", "Choose a model first.")
            return
        if not model.is_official:
            messagebox.showinfo("Custom model", "Only official models can be downloaded here.")
            return
        if model.exists:
            self.status_var.set(f"Model already exists: {model.path}")
            return
        try:
            downloaded = self.service.download_official_model(model.name, model.task)
        except Exception as exc:
            messagebox.showerror("Download failed", str(exc))
            return
        self.refresh_models()
        self.model_var.set(downloaded.label())
        self.status_var.set(f"Downloaded: {downloaded.path}")

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.source_image = self.service.load_image(path)
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))
            return
        self.result_image = self.source_image.copy()
        self._show_image(self.result_image)
        self.status_var.set(f"Loaded image: {path}")

    def save_result(self) -> None:
        if self.result_image is None:
            messagebox.showinfo("No result", "Run segmentation first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save result",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg;*.jpeg"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.service.save_image(self.result_image, path)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.status_var.set(f"Saved result: {path}")

    def open_camera(self) -> None:
        try:
            self.service.open_camera(int(self.camera_index_var.get()))
        except Exception as exc:
            messagebox.showerror("Camera failed", str(exc))
            return
        self.status_var.set("Camera opened.")
        self._schedule_camera_preview()

    def close_camera(self) -> None:
        self.stop_live()
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        self.service.close_camera()
        self.fps_var.set("FPS: --")

    def run_once(self) -> None:
        if self.source_image is None:
            messagebox.showinfo("No image", "Open an image first, or open a camera and use Start Live.")
            return
        self._run_segmentation(self.source_image)

    def start_live(self) -> None:
        if not self.service.is_camera_open():
            messagebox.showinfo("No camera", "Open a camera first.")
            return
        self.segmenting_camera = True
        self.status_var.set("Live segmentation running.")

    def stop_live(self) -> None:
        self.segmenting_camera = False

    def close(self, destroy_root: bool = True) -> None:
        arm_forced_process_exit()
        if self.closed:
            return
        self.closed = True
        self.close_camera()
        self.service.unload_model()
        self.loaded_model_path = None
        terminate_process(self.root, destroy_root=destroy_root)

    def _on_destroy(self, event) -> None:
        if event.widget is self.root and not self.closed:
            self.close(destroy_root=False)

    def _schedule_camera_preview(self) -> None:
        self.after_id = self.root.after(10, self._read_camera_frame)

    def _read_camera_frame(self) -> None:
        if not self.service.is_camera_open():
            self.after_id = None
            return
        try:
            frame = self.service.read_frame()
        except Exception as exc:
            messagebox.showerror("Camera read failed", str(exc))
            self.close_camera()
            return
        self.source_image = frame
        if self.segmenting_camera:
            self._run_segmentation(frame, show_errors=False)
        else:
            self.result_image = frame
            self._show_image(frame)
        self._update_camera_fps()
        self._schedule_camera_preview()

    def _run_segmentation(self, image: ImageArray, show_errors: bool = True) -> None:
        model = self._selected_model()
        if model is None:
            if show_errors:
                messagebox.showinfo("No model", "Choose a segmentation model first.")
            return
        if not model.exists:
            if show_errors:
                messagebox.showinfo("Missing model", "Download this model first or browse a local .pt file.")
            return
        settings = self._current_settings()
        try:
            if self.loaded_model_path != model.path:
                self.service.load_model(model.path)
                self.loaded_model_path = model.path
            output = self.service.segment_image(image, settings)
        except Exception as exc:
            self.segmenting_camera = False
            if show_errors:
                messagebox.showerror("Segmentation failed", str(exc))
            else:
                self.status_var.set(str(exc))
            return
        self.result_image = output.annotated_frame
        self.items_var.set(f"Items: {output.item_count}")
        self.status_var.set(f"{settings.task} | {output.inference_ms:.1f} ms")
        self._show_image(output.annotated_frame)

    def _current_settings(self) -> SegmentationSettings:
        return SegmentationSettings(
            task=self.task_var.get(),
            image_size=int(self.image_size_var.get()),
            confidence=float(self.conf_var.get()),
            iou=float(self.iou_var.get()),
            device=self.device_var.get(),
        )

    def _show_image(self, image: ImageArray) -> None:
        self.current_photo = self.presenter.to_photo_image(image)
        self.preview_label.configure(image=self.current_photo, text="")

    def _update_camera_fps(self) -> None:
        now = time.perf_counter()
        if self.last_frame_time is not None:
            elapsed = now - self.last_frame_time
            if elapsed > 0:
                instant = 1.0 / elapsed
                self.display_fps = instant if self.display_fps == 0 else self.display_fps * 0.85 + instant * 0.15
                self.fps_var.set(f"FPS: {self.display_fps:.1f}")
        self.last_frame_time = now

    def _selected_model(self) -> Optional[ModelInfo]:
        label = self.model_var.get()
        for model in self.models:
            if model.label() == label:
                return model
        return None


def main() -> None:
    root = tk.Tk()
    Yolo26SegmentationWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
