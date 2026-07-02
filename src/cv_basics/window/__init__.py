"""Desktop window package."""

__all__ = ["CvDemoWindow", "main"]


def __getattr__(name: str):
    if name in __all__:
        from .app import CvDemoWindow, main

        values = {
            "CvDemoWindow": CvDemoWindow,
            "main": main,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
