"""Process-level exit helper for Tkinter GUI windows."""

from __future__ import annotations

import os
import threading
from typing import Any


_exit_timer_started = False


def arm_forced_process_exit(exit_code: int = 0, delay_seconds: float = 5.0) -> None:
    """Schedule a hard process exit in case GUI cleanup blocks."""

    global _exit_timer_started
    if _exit_timer_started:
        return
    _exit_timer_started = True
    timer = threading.Timer(delay_seconds, lambda: os._exit(exit_code))
    timer.daemon = True
    timer.start()


def terminate_process(root: Any, exit_code: int = 0, destroy_root: bool = True) -> None:
    """Close Tk resources, then terminate the current Python process."""

    arm_forced_process_exit(exit_code=exit_code, delay_seconds=1.0)

    try:
        if hasattr(root, "quit"):
            root.quit()
    except Exception:
        pass

    if destroy_root:
        try:
            if hasattr(root, "destroy"):
                root.destroy()
        except Exception:
            pass

    os._exit(exit_code)


def close_window(root: Any, exit_on_close: bool, exit_code: int = 0, destroy_root: bool = True) -> None:
    """Close one Tk window, or the whole process for standalone entry points."""

    if exit_on_close:
        terminate_process(root, exit_code=exit_code, destroy_root=destroy_root)
        return

    if not destroy_root:
        return

    try:
        if hasattr(root, "destroy"):
            root.destroy()
    except Exception:
        pass
