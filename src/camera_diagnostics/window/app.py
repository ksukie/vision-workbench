"""Tkinter GUI for camera diagnostics."""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from camera_diagnostics.api import create_camera_diagnostics_service
    from camera_diagnostics.configuration import CameraDiagnosticsConfig
    from camera_diagnostics.domain import CameraDevice, CaptureProfile, ImageArray
    from camera_diagnostics.window.presenter import TkFramePresenter
else:
    from ..api import create_camera_diagnostics_service
    from ..configuration import CameraDiagnosticsConfig
    from ..domain import CameraDevice, CaptureProfile, ImageArray
    from .presenter import TkFramePresenter


class CameraDiagnosticsWindow:
    """Camera testing GUI backed by the camera diagnostics service."""

    def __init__(
        self,
        root: tk.Misc,
        config: CameraDiagnosticsConfig = CameraDiagnosticsConfig(),
    ) -> None:
        self.root = root
        self.config = config
        self.service = create_camera_diagnostics_service(config)
        self.presenter = TkFramePresenter(config.preview_size)

        if isinstance(root, (tk.Tk, tk.Toplevel)):
            root.title("Camera Diagnostics Workbench")
            root.geometry("1120x760")
            root.minsize(920, 620)
            root.protocol("WM_DELETE_WINDOW", self.close)

        self.devices = []  # type: List[CameraDevice]
        self.profiles = []  # type: List[CaptureProfile]
        self.current_frame = None  # type: Optional[ImageArray]
        self.current_photo = None
        self.after_id = None  # type: Optional[str]
        self.last_frame_time = None  # type: Optional[float]
        self.display_fps = 0.0
        self.read_failures = 0
        self.recording_path = None  # type: Optional[Path]

        self.device_var = tk.StringVar(value="")
        self.profile_var = tk.StringVar(value="")
        self.platform_var = tk.StringVar(value="")
        self.fps_var = tk.StringVar(value="FPS: --")
        self.profile_info_var = tk.StringVar(value="Profile: --")
        self.status_var = tk.StringVar(value="Refresh cameras to begin.")
        self.recording_var = tk.StringVar(value="Recording: off")

        self._build_ui()
        self._update_platform_label()

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
            width=28,
        )
        self.device_box.pack(side=tk.LEFT, padx=(0, 8))
        self.device_box.bind("<<ComboboxSelected>>", lambda _event: self.probe_profiles())

        ttk.Button(toolbar, text="Probe Modes", command=self.probe_profiles).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(toolbar, text="Open", command=self.open_camera).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Close", command=self.close_camera).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Screenshot", command=self.save_screenshot).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(toolbar, text="Start Recording", command=self.start_recording).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(toolbar, text="Stop Recording", command=self.stop_recording).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )

        controls = ttk.Frame(self.root, padding=(10, 4, 10, 8))
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Read mode").pack(side=tk.LEFT, padx=(0, 6))
        self.profile_box = ttk.Combobox(
            controls,
            textvariable=self.profile_var,
            values=[],
            state="readonly",
            width=44,
        )
        self.profile_box.pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(controls, textvariable=self.platform_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(controls, textvariable=self.profile_info_var).pack(side=tk.LEFT)

        stats = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        stats.pack(fill=tk.X)
        ttk.Label(stats, textvariable=self.fps_var, width=16).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(stats, textvariable=self.recording_var, width=30).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(stats, textvariable=self.status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

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
            messagebox.showerror("Camera scan failed", str(exc))
            self.status_var.set("Camera scan failed.")
            return

        labels = [device.label() for device in self.devices]
        self.device_box.configure(values=labels)
        self.profiles = []
        self.profile_box.configure(values=[])
        self.profile_var.set("")
        if not self.devices:
            self.device_var.set("")
            self.status_var.set("No camera found. Check permissions, USB connection, or OS privacy settings.")
            return

        self.device_var.set(labels[0])
        self.status_var.set(f"Found {len(self.devices)} camera route(s).")
        self.probe_profiles()

    def probe_profiles(self) -> None:
        device = self._selected_device()
        if device is None:
            messagebox.showinfo("No camera", "Refresh and select a camera first.")
            return

        self.close_camera()
        self.status_var.set("Probing read modes...")
        self.root.update_idletasks()
        try:
            self.profiles = self.service.probe_profiles(device)
        except Exception as exc:
            messagebox.showerror("Mode probe failed", str(exc))
            self.profiles = [self._default_profile(device)]

        if not self.profiles:
            self.profiles = [self._default_profile(device)]

        labels = [profile.label() for profile in self.profiles]
        self.profile_box.configure(values=labels)
        self.profile_var.set(labels[0])
        self.status_var.set(f"Detected {len(self.profiles)} read mode(s).")

    def open_camera(self) -> None:
        device = self._selected_device()
        if device is None:
            messagebox.showinfo("No camera", "Refresh and select a camera first.")
            return

        profile = self._selected_profile() or self._default_profile(device)
        try:
            actual_profile = self.service.open_camera(device, profile)
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))
            self.status_var.set("Camera open failed.")
            return

        self.profile_info_var.set(f"Profile: {actual_profile.label()}")
        self.status_var.set(f"Opened {device.label()}.")
        self.last_frame_time = None
        self.display_fps = 0.0
        self.read_failures = 0
        self._schedule_frame()

    def close_camera(self) -> None:
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        self.service.close_camera()
        self.current_frame = None
        self.current_photo = None
        self.preview_label.configure(image="", text="No camera frame")
        self.fps_var.set("FPS: --")
        self.recording_path = None
        self.recording_var.set("Recording: off")

    def save_screenshot(self) -> None:
        if self.current_frame is None:
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
            self.service.save_screenshot(self.current_frame, path)
        except Exception as exc:
            messagebox.showerror("Screenshot failed", str(exc))
            return
        self.status_var.set(f"Saved screenshot: {path}")

    def start_recording(self) -> None:
        if self.current_frame is None:
            messagebox.showinfo("No frame", "Open a camera first.")
            return
        if self.service.is_recording():
            messagebox.showinfo("Recording", "Recording is already running.")
            return

        path = filedialog.asksaveasfilename(
            title="Save recording",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 video", "*.mp4"),
                ("AVI video", "*.avi"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        frame_height, frame_width = self.current_frame.shape[:2]
        fps = self.display_fps if self.display_fps > 1 else self.config.default_recording_fps
        try:
            self.service.start_recording(path, (frame_width, frame_height), fps)
        except Exception as exc:
            messagebox.showerror("Recording failed", str(exc))
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

    def close(self) -> None:
        self.close_camera()
        if hasattr(self.root, "destroy"):
            self.root.destroy()

    def _schedule_frame(self) -> None:
        self.after_id = self.root.after(10, self._read_next_frame)

    def _read_next_frame(self) -> None:
        if not self.service.is_camera_open():
            self.after_id = None
            return

        try:
            frame = self.service.read_frame()
        except Exception as exc:
            self.read_failures += 1
            if self.read_failures >= 5:
                self.status_var.set(str(exc))
                messagebox.showerror("Read failed", str(exc))
                self.close_camera()
                return
            self._schedule_frame()
            return

        self.read_failures = 0
        self.current_frame = frame
        self._update_fps()
        if self.service.is_recording():
            self.service.write_recording_frame(frame)

        self.current_photo = self.presenter.to_photo_image(frame)
        self.preview_label.configure(image=self.current_photo, text="")
        self._schedule_frame()

    def _update_fps(self) -> None:
        now = time.perf_counter()
        if self.last_frame_time is not None:
            elapsed = now - self.last_frame_time
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                self.display_fps = instant_fps if self.display_fps == 0 else self.display_fps * 0.85 + instant_fps * 0.15
                self.fps_var.set(f"FPS: {self.display_fps:.1f}")
        self.last_frame_time = now

    def _selected_device(self) -> Optional[CameraDevice]:
        label = self.device_var.get()
        for device in self.devices:
            if device.label() == label:
                return device
        return None

    def _selected_profile(self) -> Optional[CaptureProfile]:
        label = self.profile_var.get()
        for profile in self.profiles:
            if profile.label() == label:
                return profile
        return None

    def _default_profile(self, device: CameraDevice) -> CaptureProfile:
        return CaptureProfile(
            width=0,
            height=0,
            fps=0.0,
            fourcc="DEFAULT",
            backend_name=device.backend.name,
            is_default=True,
        )


def main() -> None:
    root = tk.Tk()
    CameraDiagnosticsWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
