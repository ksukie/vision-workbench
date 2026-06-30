"""Tkinter desktop window for Vision Workbench."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional, cast

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from cv_basics.api import create_image_processing_service
    from cv_basics.configuration import AppConfig
    from cv_basics.domain import ImageArray, ProcessingParams
    from cv_basics.ports import ImageProcessingServicePort
    from cv_basics.window.presenter import TkImagePresenter
    from cv_basics.window.task_runner import TkTaskRunner
else:
    from ..api import create_image_processing_service
    from ..configuration import AppConfig
    from ..domain import ImageArray, ProcessingParams
    from ..ports import ImageProcessingServicePort
    from .presenter import TkImagePresenter
    from .task_runner import TkTaskRunner


class CvDemoWindow:
    """GUI that delegates image work to the application service."""

    def __init__(
        self,
        root: tk.Tk,
        service: Optional[ImageProcessingServicePort] = None,
        config: AppConfig = AppConfig(),
    ) -> None:
        self.root = root
        self.service = service or create_image_processing_service(config)
        self.config = config
        self.presenter = TkImagePresenter(config.preview_size)
        self.tasks = TkTaskRunner(root)

        self.root.title("Vision Workbench")
        self.root.geometry("1180x760")
        self.root.minsize(1000, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.original_image = None  # type: Optional[ImageArray]
        self.result_image = None  # type: Optional[ImageArray]
        self.current_path = None  # type: Optional[Path]
        self.original_photo = None
        self.result_photo = None

        defaults = config.processing_defaults
        self.effect_var = tk.StringVar(value=config.default_effect)
        self.blur_var = tk.IntVar(value=defaults.blur_kernel)
        self.low_var = tk.IntVar(value=defaults.edge_low)
        self.high_var = tk.IntVar(value=defaults.edge_high)
        self.threshold_var = tk.IntVar(value=defaults.threshold)
        self.morph_kernel_var = tk.IntVar(value=defaults.morphology_kernel)
        self.morph_iterations_var = tk.IntVar(value=defaults.morphology_iterations)
        self.rotate_angle_var = tk.IntVar(value=defaults.rotate_angle)
        self.scale_percent_var = tk.IntVar(value=defaults.scale_percent)
        self.crop_percent_var = tk.IntVar(value=defaults.crop_percent)
        self.perspective_shift_var = tk.IntVar(value=defaults.perspective_shift)
        self.status_var = tk.StringVar(value="Open an image to begin.")
        self.info_var = tk.StringVar(value="")

        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(10, 10, 10, 6))
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="Open", command=self.open_image).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Reset", command=self.reset_result).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            toolbar,
            text="Panorama Reconstruction",
            command=self.open_panorama_reconstruction,
        ).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(
            toolbar,
            text="Camera Diagnostics",
            command=self.open_camera_diagnostics,
        ).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(
            toolbar,
            text="YOLO26 Detection",
            command=self.open_yolo26_detection,
        ).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(
            toolbar,
            text="Image Classification",
            command=self.open_image_classification,
        ).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.LEFT, padx=(16, 0))

        ttk.Separator(self.root).pack(fill=tk.X)

        controls = ttk.Frame(self.root, padding=(10, 8))
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Effect").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        effect_box = ttk.Combobox(
            controls,
            textvariable=self.effect_var,
            values=self.service.list_effects(),
            state="readonly",
            width=24,
        )
        effect_box.grid(row=0, column=1, sticky=tk.W, padx=(0, 12))
        ttk.Button(controls, text="Apply Effect", command=self.apply_effect).grid(
            row=0,
            column=2,
            sticky=tk.W,
            padx=(0, 16),
        )

        self._add_slider(controls, "Blur kernel", self.blur_var, 1, 31, 0, 1)
        self._add_slider(controls, "Threshold", self.threshold_var, 0, 255, 3, 1)
        self._add_slider(controls, "Morph kernel", self.morph_kernel_var, 1, 31, 6, 1)
        self._add_slider(controls, "Edge low", self.low_var, 0, 254, 0, 2)
        self._add_slider(controls, "Edge high", self.high_var, 1, 255, 3, 2)
        self._add_slider(
            controls,
            "Morph iter",
            self.morph_iterations_var,
            1,
            5,
            6,
            2,
        )
        self._add_slider(controls, "Rotate deg", self.rotate_angle_var, -180, 180, 0, 3)
        self._add_slider(controls, "Scale %", self.scale_percent_var, 10, 200, 3, 3)
        self._add_slider(controls, "Crop %", self.crop_percent_var, 10, 100, 6, 3)
        self._add_slider(
            controls,
            "Perspective %",
            self.perspective_shift_var,
            0,
            40,
            0,
            4,
        )

        body = ttk.Frame(self.root, padding=(10, 6, 10, 10))
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        ttk.Label(body, text="Original").grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
        ttk.Label(body, text="Result").grid(row=0, column=1, sticky=tk.W, pady=(0, 4))

        self.original_label = tk.Label(body, bg="#202226", fg="#d9dde3", text="No image", width=54, height=24)
        self.result_label = tk.Label(body, bg="#202226", fg="#d9dde3", text="No result", width=54, height=24)
        self.original_label.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.result_label.grid(row=1, column=1, sticky="nsew", padx=(8, 0))

        info = ttk.Label(self.root, textvariable=self.info_var, padding=(10, 4, 10, 10), anchor=tk.W)
        info.pack(fill=tk.X)

    def _add_slider(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.IntVar,
        from_: int,
        to: int,
        base_column: int,
        row: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=base_column, sticky=tk.W, padx=(8, 6))
        scale = ttk.Scale(parent, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL, length=120)
        scale.grid(row=row, column=base_column + 1, sticky=tk.W, padx=(0, 6))
        ttk.Label(parent, textvariable=variable, width=4).grid(
            row=row,
            column=base_column + 2,
            sticky=tk.W,
            padx=(0, 12),
        )

    def open_image(self) -> None:
        patterns = ";".join(self.config.supported_extensions)
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Image files", patterns), ("All files", "*.*")],
        )
        if not path:
            return

        image_path = Path(path)
        self._run_task(
            task=lambda: self.service.load_image(image_path),
            on_success=lambda image: self._on_image_loaded(image_path, image),
            busy_text="Loading image...",
            error_title="Open failed",
        )

    def save_result(self) -> None:
        if self.result_image is None:
            messagebox.showinfo("No result", "Open an image and apply an effect first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save result",
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

        image = self.result_image.copy()
        self._run_task(
            task=lambda: self.service.save_image(image, path),
            on_success=lambda _: messagebox.showinfo("Saved", f"Saved result to:\n{path}"),
            busy_text="Saving image...",
            error_title="Save failed",
        )

    def reset_result(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No image", "Open an image first.")
            return

        self.result_image = self.original_image.copy()
        self._show_result(self.result_image)
        self._update_info()
        self.status_var.set("Reset to original.")

    def open_panorama_reconstruction(self) -> None:
        try:
            from panorama_reconstruction.window.app import PanoramaReconstructionWindow
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))
            return

        window = tk.Toplevel(self.root)
        window.title("Panorama Reconstruction Workbench")
        window.geometry("1180x760")
        window.minsize(980, 660)
        window._panorama_reconstruction_app = PanoramaReconstructionWindow(window)

    def open_camera_diagnostics(self) -> None:
        try:
            from camera_diagnostics.window.app import CameraDiagnosticsWindow
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))
            return

        window = tk.Toplevel(self.root)
        window.title("Camera Diagnostics Workbench")
        window.geometry("1120x760")
        window.minsize(920, 620)
        window._camera_diagnostics_app = CameraDiagnosticsWindow(window)

    def open_yolo26_detection(self) -> None:
        try:
            from yolo26_detection.window.app import Yolo26DetectionWindow
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))
            return

        window = tk.Toplevel(self.root)
        window.title("YOLO26 Detection Workbench")
        window.geometry("1280x860")
        window.minsize(1040, 720)
        window._yolo26_detection_app = Yolo26DetectionWindow(window)

    def open_image_classification(self) -> None:
        try:
            from image_classification.window.app import ImageClassificationWindow
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))
            return

        window = tk.Toplevel(self.root)
        window.title("Image Classification Workbench")
        window.geometry("1160x760")
        window.minsize(980, 640)
        window._image_classification_app = ImageClassificationWindow(window)

    def apply_effect(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No image", "Open an image first.")
            return

        effect = self.effect_var.get()
        params = self._current_params()
        image = self.original_image.copy()
        self._run_task(
            task=lambda: self.service.apply_effect(image, effect, params),
            on_success=self._on_effect_applied,
            busy_text=f"Applying {effect}...",
            error_title="Processing failed",
        )

    def close(self) -> None:
        self.tasks.shutdown()
        self.root.destroy()

    def _run_task(
        self,
        task: Callable[[], object],
        on_success: Callable[[object], None],
        busy_text: str,
        error_title: str,
    ) -> None:
        accepted = self.tasks.run(
            task=task,
            on_success=lambda value: self._task_success(value, on_success),
            on_error=lambda exc: self._task_error(exc, error_title),
        )
        if not accepted:
            messagebox.showinfo("Busy", "Please wait for the current operation to finish.")
            return
        self.status_var.set(busy_text)

    def _task_success(self, value: object, callback: Callable[[object], None]) -> None:
        callback(value)
        self.status_var.set("Ready.")

    def _task_error(self, exc: Exception, title: str) -> None:
        self.status_var.set("Ready.")
        messagebox.showerror(title, str(exc))

    def _on_image_loaded(self, path: Path, image: object) -> None:
        self.current_path = path
        self.original_image = cast(ImageArray, image)
        self.result_image = self.original_image.copy()
        self._show_original(self.original_image)
        self._show_result(self.result_image)
        self._update_info()

    def _on_effect_applied(self, image: object) -> None:
        self.result_image = cast(ImageArray, image)
        self._show_result(self.result_image)
        self._update_info()

    def _current_params(self) -> ProcessingParams:
        return ProcessingParams(
            blur_kernel=self.blur_var.get(),
            edge_low=self.low_var.get(),
            edge_high=self.high_var.get(),
            threshold=self.threshold_var.get(),
            morphology_kernel=self.morph_kernel_var.get(),
            morphology_iterations=self.morph_iterations_var.get(),
            rotate_angle=self.rotate_angle_var.get(),
            scale_percent=self.scale_percent_var.get(),
            crop_percent=self.crop_percent_var.get(),
            perspective_shift=self.perspective_shift_var.get(),
        )

    def _show_original(self, image: ImageArray) -> None:
        self.original_photo = self.presenter.to_photo_image(image)
        self.original_label.configure(image=self.original_photo, text="")

    def _show_result(self, image: ImageArray) -> None:
        self.result_photo = self.presenter.to_photo_image(image)
        self.result_label.configure(image=self.result_photo, text="")

    def _update_info(self) -> None:
        if self.result_image is None:
            self.info_var.set("")
            return

        info = self.service.get_image_info(self.result_image)
        source = str(self.current_path) if self.current_path else "memory"
        self.info_var.set(
            " | ".join(
                [
                    f"Source: {source}",
                    f"Size: {info.width}x{info.height}",
                    f"Channels: {info.channels}",
                    f"Dtype: {info.dtype}",
                    f"Range: {info.min_value}..{info.max_value}",
                ]
            )
        )


def main() -> None:
    root = tk.Tk()
    CvDemoWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
