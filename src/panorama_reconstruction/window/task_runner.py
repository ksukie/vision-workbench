"""Small Tkinter task runner for non-blocking image processing."""

from __future__ import annotations

import queue
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable


class TkTaskRunner:
    """Run slow OpenCV work outside the Tkinter event loop."""

    def __init__(self, root: tk.Misc) -> None:
        self._root = root
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._events = queue.Queue()
        self._busy = False

    def run(
        self,
        task: Callable[[], object],
        on_success: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> bool:
        if self._busy:
            return False

        self._busy = True
        future = self._executor.submit(task)
        future.add_done_callback(
            lambda item: self._events.put((item, on_success, on_error))
        )
        self._root.after(80, self._poll)
        return True

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _poll(self) -> None:
        try:
            future, on_success, on_error = self._events.get_nowait()
        except queue.Empty:
            self._root.after(80, self._poll)
            return

        self._busy = False
        future = future  # type: Future[object]
        try:
            on_success(future.result())
        except Exception as exc:
            on_error(exc)
