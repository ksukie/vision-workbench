"""Small Qt background task adapter."""

from __future__ import annotations

import threading
from typing import Callable

from PySide6.QtCore import QObject, Signal


class _TaskSignals(QObject):
    finished = Signal(object, object)
    error = Signal(object, object)


class QtTaskRunner(QObject):
    """Runs one background task at a time and reports back on the Qt event loop."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._signals = _TaskSignals(self)
        self._signals.finished.connect(self._finish)
        self._signals.error.connect(self._fail)
        self._busy = False
        self._active_thread = None  # type: threading.Thread | None
        self._active_token = None  # type: object | None
        self._on_success = None  # type: Callable[[object], None] | None
        self._on_error = None  # type: Callable[[Exception], None] | None
        self._shutting_down = False

    def run(
        self,
        task: Callable[[], object],
        on_success: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> bool:
        if self._busy or self._shutting_down:
            return False

        token = object()
        self._active_token = token
        self._on_success = on_success
        self._on_error = on_error
        self._busy = True
        thread = threading.Thread(target=self._execute, args=(token, task), daemon=True)
        self._active_thread = thread
        thread.start()
        return True

    def shutdown(self) -> None:
        self._shutting_down = True
        self._busy = False
        self._active_token = None
        self._on_success = None
        self._on_error = None
        self._active_thread = None

    def _execute(self, token: object, task: Callable[[], object]) -> None:
        try:
            self._signals.finished.emit(token, task())
        except Exception as exc:  # pragma: no cover - exercised through UI behavior
            self._signals.error.emit(token, exc)

    def _finish(self, token: object, value: object) -> None:
        if self._shutting_down or token is not self._active_token:
            return
        callback = self._on_success
        if callback is None:
            return
        self._busy = False
        self._active_thread = None
        self._active_token = None
        self._on_success = None
        self._on_error = None
        callback(value)

    def _fail(self, token: object, exc: Exception) -> None:
        if self._shutting_down or token is not self._active_token:
            return
        callback = self._on_error
        if callback is None:
            return
        self._busy = False
        self._active_thread = None
        self._active_token = None
        self._on_success = None
        self._on_error = None
        callback(exc)
