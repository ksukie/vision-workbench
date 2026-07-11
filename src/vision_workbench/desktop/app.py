"""Application entry point for the PySide6 Vision Workbench UI."""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

from vision_workbench.runtime_security import (
    configure_isolated_python_environment,
    configure_restricted_model_loading,
)


def create_app():
    """Create or return the active Qt application."""

    configure_isolated_python_environment()
    configure_restricted_model_loading()
    from PySide6.QtWidgets import QApplication

    from .theme import apply_theme

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    apply_theme(app)
    return app


def main() -> None:
    try:
        app = create_app()
        from .main_window import MainWindow

        window = MainWindow()
        window.show()
        raise SystemExit(app.exec())
    except SystemExit:
        raise
    except Exception as exc:
        _report_startup_failure(exc)
        raise SystemExit(1)


def _report_startup_failure(exc: Exception) -> None:
    log_path = _write_startup_log(exc)
    message = (
        "Vision Workbench 启动失败。\n\n"
        f"{type(exc).__name__}: {exc}\n\n"
        f"启动日志：{log_path}\n\n"
        "如果错误包含 PySide6 或 DLL load failed，请重新安装基础依赖，"
        "或在当前 conda 环境中安装 PySide6。"
    )
    try:
        print(message, file=sys.stderr)
    except Exception:
        pass
    _show_fallback_error_dialog(message)


def _write_startup_log(exc: Exception) -> Path:
    configured_path = os.environ.get("VISION_WORKBENCH_STARTUP_LOG")
    log_path = Path(configured_path) if configured_path else Path(tempfile.gettempdir()) / "vision-workbench-startup.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "Vision Workbench startup failure\n\n"
        f"Python: {sys.executable}\n"
        f"Version: {sys.version}\n\n"
        + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        encoding="utf-8",
    )
    return log_path


def _show_fallback_error_dialog(message: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Vision Workbench 启动失败", message)
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
