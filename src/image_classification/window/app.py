"""Tkinter GUI for image classification."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from image_classification.api import create_image_classification_service
    from image_classification.configuration import ImageClassificationConfig
    from image_classification.domain import ClassificationTrainingConfig, PredictionResult
    from image_classification.window.presenter import TkClassificationPresenter
    from cv_basics.window.process_exit import arm_forced_process_exit, close_window
else:
    from ..api import create_image_classification_service
    from ..configuration import ImageClassificationConfig
    from ..domain import ClassificationTrainingConfig, PredictionResult
    from .presenter import TkClassificationPresenter
    from cv_basics.window.process_exit import arm_forced_process_exit, close_window


class ImageClassificationWindow:
    """Beginner-friendly GUI for classification prediction and training."""

    def __init__(
        self,
        root: tk.Misc,
        config: ImageClassificationConfig = ImageClassificationConfig(),
        exit_on_close: bool = True,
    ) -> None:
        self.root = root
        self.config = config
        self.exit_on_close = exit_on_close
        self.service = create_image_classification_service(config)
        self.presenter = TkClassificationPresenter(config.preview_size)

        if isinstance(root, (tk.Tk, tk.Toplevel)):
            root.title("Image Classification Workbench")
            root.geometry("1160x760")
            root.minsize(980, 640)
            root.protocol("WM_DELETE_WINDOW", self.close)

        self.image_path = None  # type: Optional[Path]
        self.checkpoint_path = None  # type: Optional[Path]
        self.preview_photo = None

        self.predict_model_var = tk.StringVar(value=config.default_model_name)
        self.predict_device_var = tk.StringVar(value="auto")
        self.topk_var = tk.IntVar(value=config.default_topk)
        self.checkpoint_var = tk.StringVar(value="")
        self.weight_status_var = tk.StringVar(value="")
        self.predict_status_var = tk.StringVar(value="Open an image to begin.")

        self.dataset_var = tk.StringVar(value="")
        self.train_model_var = tk.StringVar(value=config.default_model_name)
        self.train_device_var = tk.StringVar(value="auto")
        self.epochs_var = tk.IntVar(value=config.default_epochs)
        self.image_size_var = tk.IntVar(value=config.default_image_size)
        self.batch_var = tk.IntVar(value=config.default_batch_size)
        self.lr_var = tk.StringVar(value=str(config.default_learning_rate))
        self.run_name_var = tk.StringVar(value="classification_train")
        self.pretrained_var = tk.BooleanVar(value=True)
        self.freeze_var = tk.BooleanVar(value=True)
        self.train_status_var = tk.StringVar(value="Select a dataset to validate or train.")

        self._build_ui()
        self.check_weight_status()

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        predict_tab = ttk.Frame(notebook, padding=10)
        train_tab = ttk.Frame(notebook, padding=10)
        notebook.add(predict_tab, text="Predict")
        notebook.add(train_tab, text="Train")

        self._build_predict_tab(predict_tab)
        self._build_train_tab(train_tab)

    def _build_predict_tab(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(toolbar, text="Model").pack(side=tk.LEFT, padx=(8, 4))
        model_box = ttk.Combobox(
            toolbar,
            textvariable=self.predict_model_var,
            values=self.service.supported_models(),
            state="readonly",
            width=22,
        )
        model_box.pack(side=tk.LEFT, padx=(0, 8))
        model_box.bind("<<ComboboxSelected>>", lambda event: self.check_weight_status())
        ttk.Label(toolbar, text="Device").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Combobox(
            toolbar,
            textvariable=self.predict_device_var,
            values=self.config.device_options,
            state="readonly",
            width=8,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(toolbar, text="Top-K").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Spinbox(toolbar, from_=1, to=10, textvariable=self.topk_var, width=5).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Predict Pretrained", command=self.predict_pretrained).pack(
            side=tk.LEFT,
            padx=(12, 8),
        )
        ttk.Button(toolbar, text="Browse Checkpoint", command=self.browse_checkpoint).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(toolbar, text="Predict Checkpoint", command=self.predict_checkpoint).pack(side=tk.LEFT)

        weights_bar = ttk.Frame(parent)
        weights_bar.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(weights_bar, text="Check Weights", command=self.check_weight_status).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(weights_bar, text="Download Pretrained Weights", command=self.download_pretrained_weight).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(weights_bar, text="Import Local Weights", command=self.import_local_weight).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Label(weights_bar, textvariable=self.weight_status_var).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(parent, textvariable=self.checkpoint_var).pack(fill=tk.X, pady=(8, 4))

        body = ttk.Frame(parent)
        body.pack(fill=tk.BOTH, expand=True, pady=(8, 4))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.image_label = tk.Label(body, bg="#202226", fg="#d9dde3", text="No image", width=54, height=24)
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        result_frame = ttk.Frame(body)
        result_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)
        self.result_text = tk.Text(result_frame, height=20, wrap=tk.WORD)
        self.result_text.grid(row=0, column=0, sticky="nsew")
        self.result_text.insert(tk.END, "Prediction results will appear here.")
        self.result_text.configure(state=tk.DISABLED)

        ttk.Label(parent, textvariable=self.predict_status_var).pack(fill=tk.X, pady=(6, 0))

    def _build_train_tab(self, parent: ttk.Frame) -> None:
        form = ttk.Frame(parent)
        form.pack(fill=tk.X)

        ttk.Label(form, text="Dataset").grid(row=0, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        ttk.Entry(form, textvariable=self.dataset_var, width=70).grid(row=0, column=1, sticky=tk.W, pady=4)
        ttk.Button(form, text="Browse", command=self.browse_dataset).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(form, text="Validate", command=self.validate_dataset).grid(row=0, column=3, padx=(8, 0))

        ttk.Label(form, text="Model").grid(row=1, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        ttk.Combobox(
            form,
            textvariable=self.train_model_var,
            values=self.service.supported_models(),
            state="readonly",
            width=22,
        ).grid(row=1, column=1, sticky=tk.W, pady=4)

        ttk.Label(form, text="Device").grid(row=1, column=2, sticky=tk.W, padx=(8, 6), pady=4)
        ttk.Combobox(
            form,
            textvariable=self.train_device_var,
            values=self.config.device_options,
            state="readonly",
            width=8,
        ).grid(row=1, column=3, sticky=tk.W, pady=4)

        self._add_spinbox(form, "Epochs", self.epochs_var, 1, 500, 2, 0)
        self._add_spinbox(form, "Image size", self.image_size_var, 64, 1024, 2, 2)
        self._add_spinbox(form, "Batch", self.batch_var, 1, 256, 3, 0)
        ttk.Label(form, text="Learning rate").grid(row=3, column=2, sticky=tk.W, padx=(8, 6), pady=4)
        ttk.Entry(form, textvariable=self.lr_var, width=10).grid(row=3, column=3, sticky=tk.W, pady=4)

        ttk.Label(form, text="Run name").grid(row=4, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        ttk.Entry(form, textvariable=self.run_name_var, width=32).grid(row=4, column=1, sticky=tk.W, pady=4)
        ttk.Checkbutton(form, text="Use pretrained weights", variable=self.pretrained_var).grid(
            row=4,
            column=2,
            sticky=tk.W,
            padx=(8, 0),
        )
        ttk.Checkbutton(form, text="Freeze backbone", variable=self.freeze_var).grid(
            row=4,
            column=3,
            sticky=tk.W,
        )

        ttk.Button(form, text="Start Training", command=self.start_training).grid(
            row=5,
            column=1,
            sticky=tk.W,
            pady=(8, 4),
        )

        log_frame = ttk.Frame(parent)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.train_log = tk.Text(log_frame, height=20, wrap=tk.WORD)
        self.train_log.grid(row=0, column=0, sticky="nsew")
        self.train_log.insert(tk.END, "Dataset validation and training messages will appear here.")
        self.train_log.configure(state=tk.DISABLED)
        ttk.Label(parent, textvariable=self.train_status_var).pack(fill=tk.X, pady=(6, 0))

    def _add_spinbox(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.IntVar,
        from_: int,
        to: int,
        row: int,
        column: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky=tk.W, padx=(0, 6), pady=4)
        ttk.Spinbox(parent, from_=from_, to=to, textvariable=variable, width=10).grid(
            row=row,
            column=column + 1,
            sticky=tk.W,
            pady=4,
        )

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        self.image_path = Path(path)
        try:
            self.preview_photo = self.presenter.path_to_photo_image(self.image_path)
            self.image_label.configure(image=self.preview_photo, text="")
            self.predict_status_var.set(f"Loaded image: {self.image_path}")
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))

    def browse_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="Select classification checkpoint",
            filetypes=[("PyTorch model", "*.pt;*.pth"), ("All files", "*.*")],
        )
        if not path:
            return
        self.checkpoint_path = Path(path)
        self.checkpoint_var.set(f"Checkpoint: {self.checkpoint_path}")

    def check_weight_status(self) -> None:
        try:
            info = self.service.pretrained_weight_status(self.predict_model_var.get())[0]
        except Exception as exc:
            self.weight_status_var.set(f"Weight status unavailable: {exc}")
            return
        if info.exists:
            self.weight_status_var.set(f"Local pretrained weight: {info.local_path}")
        else:
            self.weight_status_var.set(f"No local pretrained weight. Expected: {info.local_path}")

    def download_pretrained_weight(self) -> None:
        model_name = self.predict_model_var.get()
        self._run_worker(
            task=lambda: self.service.download_pretrained_weight(model_name),
            on_success=lambda info: self._on_weight_ready(info, "Downloaded"),
            busy_text=f"Downloading {model_name} weights...",
            error_title="Download failed",
        )

    def import_local_weight(self) -> None:
        path = filedialog.askopenfilename(
            title="Select local pretrained weight",
            filetypes=[("PyTorch weight", "*.pth;*.pt"), ("All files", "*.*")],
        )
        if not path:
            return
        model_name = self.predict_model_var.get()
        self._run_worker(
            task=lambda: self.service.import_pretrained_weight(model_name, path),
            on_success=lambda info: self._on_weight_ready(info, "Imported"),
            busy_text=f"Importing {model_name} weights...",
            error_title="Import failed",
        )

    def _on_weight_ready(self, info: object, action: str) -> None:
        weight_info = info
        self.weight_status_var.set(f"{action}: {weight_info.local_path}")
        messagebox.showinfo(action, f"{action} pretrained weight:\n{weight_info.local_path}")

    def predict_pretrained(self) -> None:
        if self.image_path is None:
            messagebox.showinfo("No image", "Please open an image first.")
            return
        self._run_worker(
            task=lambda: self.service.predict_with_pretrained(
                model_name=self.predict_model_var.get(),
                image_path=self.image_path,
                topk=self.topk_var.get(),
                device=self.predict_device_var.get(),
            ),
            on_success=self._show_prediction,
            busy_text="Running pretrained prediction...",
            error_title="Prediction failed",
        )

    def predict_checkpoint(self) -> None:
        if self.image_path is None:
            messagebox.showinfo("No image", "Please open an image first.")
            return
        if self.checkpoint_path is None:
            messagebox.showinfo("No checkpoint", "Please select a trained checkpoint first.")
            return
        self._run_worker(
            task=lambda: self.service.predict_with_checkpoint(
                model_path=self.checkpoint_path,
                image_path=self.image_path,
                topk=self.topk_var.get(),
                device=self.predict_device_var.get(),
            ),
            on_success=self._show_prediction,
            busy_text="Running checkpoint prediction...",
            error_title="Prediction failed",
        )

    def browse_dataset(self) -> None:
        path = filedialog.askdirectory(title="Select classification dataset")
        if path:
            self.dataset_var.set(path)

    def validate_dataset(self) -> None:
        if not self.dataset_var.get().strip():
            messagebox.showinfo("No dataset", "Please select a dataset directory.")
            return
        report = self.service.validate_dataset(self.dataset_var.get().strip())
        self._set_train_log(report.to_text())
        self.train_status_var.set("Dataset validation passed." if report.ok else "Dataset validation failed.")

    def start_training(self) -> None:
        if not self.dataset_var.get().strip():
            messagebox.showinfo("No dataset", "Please select a dataset directory.")
            return
        try:
            lr = float(self.lr_var.get())
        except ValueError:
            messagebox.showerror("Invalid value", "Learning rate must be a number.")
            return

        job = ClassificationTrainingConfig(
            model_name=self.train_model_var.get(),
            dataset_dir=Path(self.dataset_var.get().strip()),
            output_dir=self.config.runs_dir,
            run_name=self.run_name_var.get().strip() or "classification_train",
            epochs=self.epochs_var.get(),
            image_size=self.image_size_var.get(),
            batch_size=self.batch_var.get(),
            device=self.train_device_var.get(),
            learning_rate=lr,
            workers=0,
            pretrained=self.pretrained_var.get(),
            freeze_backbone=self.freeze_var.get(),
        )
        self._run_worker(
            task=lambda: self.service.train(job),
            on_success=lambda path: self._set_train_log(f"Training finished.\nBest model:\n{path}"),
            busy_text="Training started...",
            error_title="Training failed",
            status_var=self.train_status_var,
        )

    def _run_worker(
        self,
        task: Callable[[], object],
        on_success: Callable[[object], None],
        busy_text: str,
        error_title: str,
        status_var: Optional[tk.StringVar] = None,
    ) -> None:
        target_status = status_var or self.predict_status_var
        target_status.set(busy_text)

        def worker() -> None:
            try:
                result = task()
            except Exception as exc:
                self.root.after(0, lambda: self._show_error(error_title, exc, target_status))
                return
            self.root.after(0, lambda: self._show_success(result, on_success, target_status))

        threading.Thread(target=worker, daemon=True).start()

    def _show_error(self, title: str, exc: Exception, status_var: tk.StringVar) -> None:
        status_var.set("Ready.")
        messagebox.showerror(title, str(exc))

    def _show_success(
        self,
        result: object,
        callback: Callable[[object], None],
        status_var: tk.StringVar,
    ) -> None:
        callback(result)
        status_var.set("Ready.")

    def _show_prediction(self, result: object) -> None:
        prediction = result  # type: PredictionResult
        lines = [
            f"Model: {prediction.model_name}",
            f"Inference: {prediction.inference_ms:.1f} ms",
            "",
            "Top predictions:",
        ]
        for index, item in enumerate(prediction.predictions, start=1):
            lines.append(f"{index}. {item.label}: {item.score * 100:.2f}%")
        self._set_prediction_text("\n".join(lines))

    def _set_prediction_text(self, text: str) -> None:
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state=tk.DISABLED)

    def _set_train_log(self, text: str) -> None:
        self.train_log.configure(state=tk.NORMAL)
        self.train_log.delete("1.0", tk.END)
        self.train_log.insert(tk.END, text)
        self.train_log.configure(state=tk.DISABLED)

    def close(self) -> None:
        if self.exit_on_close:
            arm_forced_process_exit()
        close_window(self.root, exit_on_close=self.exit_on_close)


def main() -> None:
    root = tk.Tk()
    ImageClassificationWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
