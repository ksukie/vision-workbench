"""Small Tkinter async task adapter."""

from __future__ import annotations

import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable


class TkTaskRunner:
    """Runs background work and returns results to the Tk event loop."""

    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._executor = ThreadPoolExecutor(max_workers=1)
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
        self._poll(future, on_success, on_error)
        return True

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    def _poll(
        self,
        future: Future,
        on_success: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        if not future.done():
            self._root.after(40, lambda: self._poll(future, on_success, on_error))
            return

        self._busy = False
        try:
            on_success(future.result())
        except Exception as exc:
            on_error(exc)
