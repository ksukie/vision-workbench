"""Tkinter window for panorama reconstruction."""

__all__ = ["PanoramaReconstructionWindow", "main"]


def __getattr__(name: str):
    if name in __all__:
        from .app import PanoramaReconstructionWindow, main

        values = {
            "PanoramaReconstructionWindow": PanoramaReconstructionWindow,
            "main": main,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
