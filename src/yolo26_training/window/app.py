"""Tkinter GUI for YOLO26 training."""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from yolo26_training.api import create_yolo26_training_service
    from yolo26_training.configuration import Yolo26TrainingConfig
    from yolo26_training.domain import TrainingJobConfig
    from cv_basics.window.process_exit import arm_forced_process_exit, close_window
else:
    from ..api import create_yolo26_training_service
    from ..configuration import Yolo26TrainingConfig
    from ..domain import TrainingJobConfig
    from cv_basics.window.process_exit import arm_forced_process_exit, close_window

from vision_workbench.troubleshooting import DATASETS_AND_TRAINING, help_hint, with_help


class Yolo26TrainingWindow:
    """Independent GUI for dataset validation and basic YOLO26 training."""

    def __init__(
        self,
        root: tk.Misc,
        config: Yolo26TrainingConfig = Yolo26TrainingConfig(),
        exit_on_close: bool = True,
    ) -> None:
        self.root = root
        self.config = config
        self.exit_on_close = exit_on_close
        self.service = create_yolo26_training_service(config)
        self.process = None  # type: Optional[subprocess.Popen]
        self.reader_thread = None  # type: Optional[threading.Thread]
        self.log_queue = queue.Queue()
        self.models = []  # type: List[Path]

        if isinstance(root, (tk.Tk, tk.Toplevel)):
            root.title("YOLO26 Training Workbench")
            root.geometry("1180x780")
            root.minsize(980, 660)
            root.protocol("WM_DELETE_WINDOW", self.close)

        self.task_var = tk.StringVar(value="detect")
        self.data_var = tk.StringVar(value="")
        self.model_var = tk.StringVar(value="")
        self.epochs_var = tk.IntVar(value=config.default_epochs)
        self.imgsz_var = tk.IntVar(value=config.default_image_size)
        self.batch_var = tk.IntVar(value=config.default_batch_size)
        self.workers_var = tk.IntVar(value=config.default_workers)
        self.device_var = tk.StringVar(value="auto")
        self.run_name_var = tk.StringVar(value="")
        self.allow_missing_labels_var = tk.BooleanVar(value=False)
        self.resume_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Choose a dataset data.yaml to begin.")
        self.output_var = tk.StringVar(value=f"Output: {config.runs_dir}")

        self._build_ui()
        self.refresh_models()
        self._poll_log_queue()

    def _build_ui(self) -> None:
        task_frame = ttk.LabelFrame(self.root, text="Task", padding=(10, 8))
        task_frame.pack(fill=tk.X, padx=10, pady=(10, 6))
        ttk.Label(task_frame, text="Task").pack(side=tk.LEFT, padx=(0, 6))
        self.task_box = ttk.Combobox(
            task_frame,
            textvariable=self.task_var,
            values=self.config.task_options,
            state="readonly",
            width=14,
        )
        self.task_box.pack(side=tk.LEFT, padx=(0, 12))
        self.task_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh_models())
        ttk.Label(
            task_frame,
            text="detect: box labels | segment: polygon labels | semantic: masks_dir masks or polygon labels",
        ).pack(side=tk.LEFT)

        dataset_frame = ttk.LabelFrame(self.root, text="Dataset", padding=(10, 8))
        dataset_frame.pack(fill=tk.X, padx=10, pady=6)
        dataset_frame.columnconfigure(1, weight=1)

        ttk.Button(dataset_frame, text="Choose data.yaml", command=self.choose_dataset).grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=(0, 8),
        )
        ttk.Entry(dataset_frame, textvariable=self.data_var).grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Button(dataset_frame, text="Validate", command=self.validate_dataset).grid(
            row=0,
            column=2,
            sticky=tk.W,
        )
        ttk.Checkbutton(
            dataset_frame,
            text="Allow missing labels",
            variable=self.allow_missing_labels_var,
        ).grid(row=0, column=3, sticky=tk.W, padx=(12, 0))

        params = ttk.LabelFrame(self.root, text="Training", padding=(10, 8))
        params.pack(fill=tk.X, padx=10, pady=6)
        params.columnconfigure(1, weight=1)

        ttk.Label(params, text="Model").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        self.model_box = ttk.Combobox(params, textvariable=self.model_var, values=[], width=48)
        self.model_box.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Button(params, text="Browse PT", command=self.choose_model).grid(row=0, column=2, sticky=tk.W)
        ttk.Button(params, text="Refresh Models", command=self.refresh_models).grid(
            row=0,
            column=3,
            sticky=tk.W,
            padx=(8, 0),
        )

        self._add_int_entry(params, "Epochs", self.epochs_var, 1, 1)
        self._add_int_entry(params, "Image size", self.imgsz_var, 3, 1)
        self._add_int_entry(params, "Batch", self.batch_var, 5, 1)
        self._add_int_entry(params, "Workers", self.workers_var, 7, 1)

        ttk.Label(params, text="Device").grid(row=1, column=9, sticky=tk.W, padx=(12, 6), pady=(8, 0))
        ttk.Combobox(
            params,
            textvariable=self.device_var,
            values=self.config.device_options,
            width=10,
            state="readonly",
        ).grid(row=1, column=10, sticky=tk.W, pady=(8, 0))

        ttk.Label(params, text="Run name").grid(row=2, column=0, sticky=tk.W, padx=(0, 6), pady=(8, 0))
        ttk.Entry(params, textvariable=self.run_name_var, width=32).grid(
            row=2,
            column=1,
            sticky=tk.W,
            pady=(8, 0),
        )
        ttk.Checkbutton(params, text="Resume", variable=self.resume_var).grid(
            row=2,
            column=2,
            sticky=tk.W,
            padx=(8, 0),
            pady=(8, 0),
        )

        actions = ttk.Frame(self.root, padding=(10, 4, 10, 8))
        actions.pack(fill=tk.X)
        ttk.Button(actions, text="Start Training", command=self.start_training).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions, text="Stop Training", command=self.stop_training).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions, text="Open Runs", command=self.open_runs_dir).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(actions, textvariable=self.output_var).pack(side=tk.LEFT, padx=(12, 0))

        ttk.Label(self.root, textvariable=self.status_var, padding=(10, 0, 10, 6)).pack(fill=tk.X)

        log_frame = ttk.Frame(self.root, padding=(10, 4, 10, 10))
        log_frame.pack(fill=tk.BOTH, expand=True)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=24)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _add_int_entry(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.IntVar,
        column: int,
        row: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky=tk.W, padx=(12, 6), pady=(8, 0))
        ttk.Entry(parent, textvariable=variable, width=8).grid(
            row=row,
            column=column + 1,
            sticky=tk.W,
            pady=(8, 0),
        )

    def choose_dataset(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose YOLO data.yaml",
            initialdir=str(self.config.dataset_dir),
            filetypes=[("YAML dataset", "*.yaml;*.yml"), ("All files", "*.*")],
        )
        if path:
            self.data_var.set(path)

    def choose_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose YOLO26 .pt model",
            initialdir=str(self.config.model_dir),
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.model_var.set(path)

    def refresh_models(self) -> None:
        task = self.task_var.get()
        self.models = self.service.list_models(task)
        labels = [str(path) for path in self.models]
        self.model_box.configure(values=labels)
        current = self.model_var.get()
        if labels and (not current or Path(current).parent != self.config.model_dir_for_task(task)):
            self.model_var.set(labels[0])
        elif not labels:
            self.model_var.set(str(self.service.default_model(task)))

    def validate_dataset(self) -> bool:
        data_path = self.data_var.get().strip()
        if not data_path:
            messagebox.showinfo("No dataset", "Choose a data.yaml first.")
            return False
        report = self.service.validate_dataset(
            data_path,
            task=self.task_var.get(),
            allow_missing_labels=self.allow_missing_labels_var.get(),
        )
        self._append_log("\n" + report.to_text() + "\n")
        self.status_var.set("Dataset validation passed." if report.ok else "Dataset validation failed.")
        if not report.ok:
            messagebox.showerror(
                "Dataset invalid",
                with_help("Dataset validation failed. See the log for details.", DATASETS_AND_TRAINING),
            )
        return report.ok

    def start_training(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo("Training", "Training is already running.")
            return
        if not self.validate_dataset():
            return
        try:
            job = self._current_job()
        except Exception as exc:
            messagebox.showerror("Invalid settings", with_help(exc, DATASETS_AND_TRAINING))
            return

        command = self.service.build_runner_command(job)
        self._append_log("\nStarting training:\n" + " ".join(command) + "\n\n")
        try:
            self.process = self.service.start_training_process(job, cwd=self.config.yolo26_source_dir.parents[1])
        except Exception as exc:
            messagebox.showerror("Training failed", with_help(exc, DATASETS_AND_TRAINING))
            self.status_var.set("Training failed to start.")
            return

        self.status_var.set("Training running...")
        self.reader_thread = threading.Thread(target=self._read_process_output, daemon=True)
        self.reader_thread.start()

    def stop_training(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self.status_var.set("No training process is running.")
            return
        self.process.terminate()
        self.status_var.set("Stopping training...")
        self._append_log("\nStop requested.\n")

    def open_runs_dir(self) -> None:
        self.config.runs_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(self.config.runs_dir))
        else:
            self._append_log(f"\nRuns directory: {self.config.runs_dir}\n")

    def close(self) -> None:
        if self.exit_on_close:
            arm_forced_process_exit()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
        close_window(self.root, exit_on_close=self.exit_on_close)

    def _current_job(self) -> TrainingJobConfig:
        data_path = Path(self.data_var.get().strip())
        model_path = Path(self.model_var.get().strip())
        if not model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        epochs = int(self.epochs_var.get())
        image_size = int(self.imgsz_var.get())
        batch_size = int(self.batch_var.get())
        workers = int(self.workers_var.get())
        if epochs <= 0 or image_size <= 0 or batch_size <= 0 or workers < 0:
            raise ValueError("Epochs, image size, batch, and workers must be valid positive numbers.")
        run_name = self.run_name_var.get().strip() or f"{data_path.stem}_{model_path.stem}"
        return TrainingJobConfig(
            task=self.task_var.get(),
            data_yaml=data_path,
            model_path=model_path,
            project_dir=self.config.runs_dir,
            run_name=run_name,
            epochs=epochs,
            image_size=image_size,
            batch_size=batch_size,
            device=self.device_var.get(),
            workers=workers,
            resume=self.resume_var.get(),
            allow_missing_labels=self.allow_missing_labels_var.get(),
        )

    def _read_process_output(self) -> None:
        assert self.process is not None
        if self.process.stdout is not None:
            for line in self.process.stdout:
                self.log_queue.put(line)
        code = self.process.wait()
        self.log_queue.put(f"\nTraining process exited with code {code}.\n")
        self.log_queue.put("__TRAINING_DONE__")

    def _poll_log_queue(self) -> None:
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item == "__TRAINING_DONE__":
                    if self.process and self.process.returncode == 0:
                        self.status_var.set("Training finished.")
                    else:
                        self.status_var.set("Training stopped or failed.")
                        self._append_log("\n" + help_hint(DATASETS_AND_TRAINING) + "\n")
                else:
                    self._append_log(item)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)


def main() -> None:
    root = tk.Tk()
    Yolo26TrainingWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
