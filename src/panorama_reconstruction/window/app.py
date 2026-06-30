"""Tkinter GUI for panorama reconstruction."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional, Tuple, cast

import cv2
import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from panorama_reconstruction.api import create_panorama_reconstruction_service
    from panorama_reconstruction.configuration import (
        CHANNEL_CHOICES,
        PanoramaReconstructionConfig,
    )
    from panorama_reconstruction.domain import (
        ControlPointReconstructionParams,
        ImageArray,
        PanoramaResult,
        Point,
        PointPair,
    )
    from panorama_reconstruction.window.presenter import TkImagePresenter
    from panorama_reconstruction.window.task_runner import TkTaskRunner
    from cv_basics.window.process_exit import arm_forced_process_exit, terminate_process
else:
    from ..api import create_panorama_reconstruction_service
    from ..configuration import CHANNEL_CHOICES, PanoramaReconstructionConfig
    from ..domain import (
        ControlPointReconstructionParams,
        ImageArray,
        PanoramaResult,
        Point,
        PointPair,
    )
    from .presenter import TkImagePresenter
    from .task_runner import TkTaskRunner
    from cv_basics.window.process_exit import arm_forced_process_exit, terminate_process


class PanoramaReconstructionWindow:
    """GUI that delegates reconstruction to the application service."""

    MODE_MANUAL = "手动"
    MODE_ASSISTED = "手动+辅助"

    def __init__(
        self,
        root: tk.Misc,
        config: PanoramaReconstructionConfig = PanoramaReconstructionConfig(),
    ) -> None:
        self.root = root
        self.config = config
        self.service = create_panorama_reconstruction_service(config)
        self.input_presenter = TkImagePresenter(config.preview_size)
        self.result_presenter = TkImagePresenter(config.result_preview_size)
        self.tasks = TkTaskRunner(root)

        if isinstance(root, (tk.Tk, tk.Toplevel)):
            root.title("Panorama Reconstruction Workbench")
            root.geometry("1180x760")
            root.minsize(980, 660)
            root.protocol("WM_DELETE_WINDOW", self.close)

        self.left_path = None  # type: Optional[Path]
        self.right_path = None  # type: Optional[Path]
        self.left_image = None  # type: Optional[ImageArray]
        self.right_image = None  # type: Optional[ImageArray]
        self.result = None  # type: Optional[PanoramaResult]
        self.photos = {}  # type: Dict[str, object]
        self.input_display_sizes = {}  # type: Dict[str, Tuple[int, int]]
        self.point_pairs = []  # type: List[PointPair]
        self.pending_left_point = None  # type: Optional[Point]

        self.mode_var = tk.StringVar(value=self.MODE_MANUAL)
        self.status_var = tk.StringVar(value="Load a left/right image pair.")
        self.metrics_var = tk.StringVar(value="")
        self.point_count_var = tk.StringVar(value="Pairs: 0")

        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(10, 10, 10, 6))
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="Open Left", command=self.open_left_image).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(toolbar, text="Open Right", command=self.open_right_image).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(toolbar, text="Load Sample Pair", command=self.load_sample_pair).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(toolbar, text="Reconstruct", command=self.reconstruct_panorama).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(toolbar, text="Save Panorama", command=self.save_panorama).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(toolbar, text="Save All Outputs", command=self.save_all_outputs).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.LEFT, padx=(16, 0))

        controls = ttk.Frame(self.root, padding=(10, 4, 10, 8))
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Mode").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=(self.MODE_MANUAL, self.MODE_ASSISTED),
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(controls, text="Undo Point", command=self.undo_point).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="Clear Points", command=self.clear_points).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="Load Points", command=self.load_points).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="Save Points", command=self.save_points).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(controls, textvariable=self.point_count_var).pack(side=tk.LEFT)

        ttk.Separator(self.root).pack(fill=tk.X)

        body = ttk.Frame(self.root, padding=(10, 8, 10, 8))
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=0)
        body.rowconfigure(3, weight=1)

        ttk.Label(body, text="Left Image").grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
        ttk.Label(body, text="Right Image").grid(row=0, column=1, sticky=tk.W, pady=(0, 4))

        self.left_label = tk.Label(
            body,
            anchor=tk.NW,
            bg="#202226",
            fg="#d9dde3",
            text="No left image",
        )
        self.right_label = tk.Label(
            body,
            anchor=tk.NW,
            bg="#202226",
            fg="#d9dde3",
            text="No right image",
        )
        self.left_label.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(0, 10))
        self.right_label.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(0, 10))
        self.left_label.bind("<Button-1>", lambda event: self._on_input_click("left", event))
        self.right_label.bind("<Button-1>", lambda event: self._on_input_click("right", event))

        ttk.Label(body, text="Output").grid(row=2, column=0, columnspan=2, sticky=tk.W)
        notebook = ttk.Notebook(body)
        notebook.grid(row=3, column=0, columnspan=2, sticky="nsew")

        self.output_labels = {}
        for key, title, empty_text in [
            ("panorama", "Panorama", "No panorama"),
            ("matches", "Feature Matches", "No match visualization"),
            ("mapped", "Mapped Points", "No mapped points"),
            ("warped", "Warped Right", "No warped image"),
        ]:
            frame = ttk.Frame(notebook, padding=8)
            label = tk.Label(frame, bg="#202226", fg="#d9dde3", text=empty_text)
            label.pack(fill=tk.BOTH, expand=True)
            notebook.add(frame, text=title)
            self.output_labels[key] = label

        ttk.Label(
            self.root,
            textvariable=self.metrics_var,
            padding=(10, 2, 10, 8),
            anchor=tk.W,
        ).pack(fill=tk.X)

    def open_left_image(self) -> None:
        self._open_image("left")

    def open_right_image(self) -> None:
        self._open_image("right")

    def _open_image(self, side: str) -> None:
        patterns = ";".join(self.config.supported_extensions)
        path = filedialog.askopenfilename(
            title=f"Open {side} image",
            filetypes=[("Image files", patterns), ("All files", "*.*")],
        )
        if not path:
            return

        image_path = Path(path)
        self._run_task(
            task=lambda: self.service.load_image(image_path),
            on_success=lambda image: self._on_image_loaded(side, image_path, image),
            busy_text=f"Loading {side} image...",
            error_title="Open failed",
        )

    def load_sample_pair(self) -> None:
        pair = self.service.get_sample_image_paths()

        def task() -> Tuple[ImageArray, ImageArray]:
            return self.service.load_image(pair.left), self.service.load_image(pair.right)

        self._run_task(
            task=task,
            on_success=lambda images: self._on_sample_pair_loaded(pair.left, pair.right, images),
            busy_text="Loading sample pair...",
            error_title="Open failed",
        )

    def reconstruct_panorama(self) -> None:
        if self.left_image is None or self.right_image is None:
            messagebox.showinfo("Missing images", "Load both left and right images first.")
            return

        if len(self.point_pairs) < 3:
            messagebox.showinfo("Missing points", "Add at least 3 left/right point pairs first.")
            return

        if self.pending_left_point is not None:
            messagebox.showinfo("Incomplete pair", "Click the matching point on the right image first.")
            return

        left = self.left_image.copy()
        right = self.right_image.copy()
        pairs = list(self.point_pairs)
        mode = self.mode_var.get()
        if mode == self.MODE_ASSISTED:
            task = lambda: self.service.reconstruct_assisted_from_points(
                left,
                right,
                pairs,
                ControlPointReconstructionParams(),
            )
            busy_text = "Reconstructing with assisted points..."
        else:
            task = lambda: self.service.reconstruct_from_points(left, right, pairs)
            busy_text = "Reconstructing from manual points..."

        self._run_task(
            task=task,
            on_success=self._on_panorama_ready,
            busy_text=busy_text,
            error_title="Reconstruction failed",
        )

    def undo_point(self) -> None:
        if self.pending_left_point is not None:
            self.pending_left_point = None
        elif self.point_pairs:
            self.point_pairs.pop()
        self._clear_result()
        self._refresh_input_images()
        self._update_metrics()

    def clear_points(self) -> None:
        self.point_pairs.clear()
        self.pending_left_point = None
        self._clear_result()
        self._refresh_input_images()
        self._update_metrics()

    def load_points(self) -> None:
        path = filedialog.askopenfilename(
            title="Load point pairs",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.point_pairs = self.service.load_point_pairs(path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return

        self.pending_left_point = None
        self._clear_result()
        self._refresh_input_images()
        self._update_metrics()
        self.status_var.set(f"Loaded point pairs from {Path(path).name}.")

    def save_points(self) -> None:
        if not self.point_pairs:
            messagebox.showinfo("No points", "Add point pairs first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save point pairs",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.service.save_point_pairs(path, self.point_pairs)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        messagebox.showinfo("Saved", f"Saved point pairs to:\n{path}")

    def save_panorama(self) -> None:
        if self.result is None:
            messagebox.showinfo("No result", "Create a panorama first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save panorama",
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

        panorama = self.result.panorama.copy()
        self._run_task(
            task=lambda: self.service.save_image(panorama, path),
            on_success=lambda _: messagebox.showinfo("Saved", f"Saved panorama to:\n{path}"),
            busy_text="Saving panorama...",
            error_title="Save failed",
        )

    def save_all_outputs(self) -> None:
        if self.result is None:
            messagebox.showinfo("No result", "Create a panorama first.")
            return

        directory = filedialog.askdirectory(title="Choose output folder")
        if not directory:
            return

        result = self.result
        self._run_task(
            task=lambda: self.service.save_outputs(result, directory),
            on_success=lambda outputs: messagebox.showinfo(
                "Saved",
                "Saved outputs to:\n" + "\n".join(str(path) for path in outputs.values()),
            ),
            busy_text="Saving all outputs...",
            error_title="Save failed",
        )

    def close(self) -> None:
        arm_forced_process_exit()
        self.tasks.shutdown()
        terminate_process(self.root)

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

    def _on_image_loaded(self, side: str, path: Path, image: object) -> None:
        image_array = cast(ImageArray, image)
        if side == "left":
            self.left_path = path
            self.left_image = image_array
        else:
            self.right_path = path
            self.right_image = image_array
        self.point_pairs.clear()
        self.pending_left_point = None
        self._refresh_input_images()
        self._clear_result()
        self._update_metrics()

    def _on_sample_pair_loaded(
        self,
        left_path: Path,
        right_path: Path,
        images: object,
    ) -> None:
        left, right = cast(Tuple[ImageArray, ImageArray], images)
        self.left_path = left_path
        self.right_path = right_path
        self.left_image = left
        self.right_image = right
        self.point_pairs.clear()
        self.pending_left_point = None
        self._refresh_input_images()
        self._clear_result()
        self._update_metrics()

    def _on_panorama_ready(self, value: object) -> None:
        self.result = cast(PanoramaResult, value)
        self._show_output("panorama", self.result.panorama)
        self._show_output("matches", self.result.match_visualization)
        self._show_output("mapped", self.result.mapped_points_visualization)
        self._show_output("warped", self.result.warped_right)
        self._update_metrics()

    def _show_input(self, side: str, image: ImageArray) -> None:
        photo = self.input_presenter.to_photo_image(image)
        self.photos[side] = photo
        self.input_display_sizes[side] = (int(photo.width()), int(photo.height()))
        label = self.left_label if side == "left" else self.right_label
        label.configure(image=photo, text="")

    def _refresh_input_images(self) -> None:
        if self.left_image is not None:
            self._show_input("left", self._annotated_input_image("left", self.left_image))
        if self.right_image is not None:
            self._show_input("right", self._annotated_input_image("right", self.right_image))

    def _annotated_input_image(self, side: str, image: ImageArray) -> ImageArray:
        annotated = image.copy()
        for index, (left_point, right_point) in enumerate(self.point_pairs, start=1):
            point = left_point if side == "left" else right_point
            color = (0, 255, 255) if side == "left" else (0, 0, 255)
            self._draw_point(annotated, point, color, str(index))

        if side == "left" and self.pending_left_point is not None:
            self._draw_point(annotated, self.pending_left_point, (0, 255, 0), "next")
        return annotated

    def _draw_point(
        self,
        image: ImageArray,
        point: Point,
        color: Tuple[int, int, int],
        label: str,
    ) -> None:
        x, y = np.round(point).astype(int)
        cv2.line(image, (x - 8, y), (x + 8, y), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(image, (x, y - 8), (x, y + 8), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 5, color, -1)
        cv2.circle(image, (x, y), 9, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (x + 11, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    def _on_input_click(self, side: str, event: tk.Event) -> None:
        point = self._display_to_original_point(side, int(event.x), int(event.y))
        if point is None:
            return

        if self.pending_left_point is None:
            if side != "left":
                messagebox.showinfo("Point order", "Click a point on the left image first.")
                return
            self.pending_left_point = point
            self.status_var.set("Now click the matching point on the right image.")
        else:
            if side != "right":
                messagebox.showinfo("Point order", "Click the matching point on the right image.")
                return
            self.point_pairs.append((self.pending_left_point, point))
            self.pending_left_point = None
            self.status_var.set("Point pair added.")

        self._clear_result()
        self._refresh_input_images()
        self._update_metrics()

    def _display_to_original_point(self, side: str, x: int, y: int) -> Optional[Point]:
        image = self.left_image if side == "left" else self.right_image
        display_size = self.input_display_sizes.get(side)
        if image is None or display_size is None:
            return None

        display_width, display_height = display_size
        if x < 0 or y < 0 or x >= display_width or y >= display_height:
            return None

        image_height, image_width = image.shape[:2]
        return (
            float(x) * float(image_width) / float(display_width),
            float(y) * float(image_height) / float(display_height),
        )

    def _show_output(self, key: str, image: ImageArray) -> None:
        photo = self.result_presenter.to_photo_image(image)
        self.photos[key] = photo
        self.output_labels[key].configure(image=photo, text="")

    def _clear_result(self) -> None:
        self.result = None
        for key, label in self.output_labels.items():
            self.photos.pop(key, None)
            empty_text = {
                "panorama": "No panorama",
                "matches": "No match visualization",
                "mapped": "No mapped points",
                "warped": "No warped image",
            }[key]
            label.configure(image="", text=empty_text)

    def _update_metrics(self) -> None:
        parts = []
        if self.left_path:
            parts.append(f"Left: {self.left_path.name}")
        if self.right_path:
            parts.append(f"Right: {self.right_path.name}")
        parts.append(f"Mode: {self.mode_var.get()}")
        point_text = f"Pairs: {len(self.point_pairs)}"
        if self.pending_left_point is not None:
            point_text += " | pending left"
        self.point_count_var.set(point_text)
        parts.append(point_text)
        if self.result:
            metrics = self.result.metrics()
            parts.append(f"Method: {metrics['method']}")
            if "manual_pairs" in metrics:
                parts.append(f"Manual: {metrics['manual_pairs']}")
            if "assisted_pairs" in metrics:
                parts.append(f"Assisted: {metrics['assisted_pairs']}")
            parts.append(f"Pairs used: {metrics['inliers']}")
            parts.append(f"Panorama shape: {metrics['panorama_shape']}")
        self.metrics_var.set(" | ".join(parts))


def main() -> None:
    root = tk.Tk()
    PanoramaReconstructionWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
