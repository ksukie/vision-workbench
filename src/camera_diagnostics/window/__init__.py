"""Tkinter window for camera diagnostics."""

__all__ = ["CameraDiagnosticsWindow", "main"]


def __getattr__(name: str):
    if name in __all__:
        from .app import CameraDiagnosticsWindow, main

        values = {
            "CameraDiagnosticsWindow": CameraDiagnosticsWindow,
            "main": main,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
