"""Window package for image classification."""

__all__ = [
    "ImageClassificationWindow",
    "TkClassificationPresenter",
    "main",
]


def __getattr__(name: str):
    if name in ("ImageClassificationWindow", "main"):
        from .app import ImageClassificationWindow, main

        values = {
            "ImageClassificationWindow": ImageClassificationWindow,
            "main": main,
        }
        return values[name]
    if name == "TkClassificationPresenter":
        from .presenter import TkClassificationPresenter

        return TkClassificationPresenter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
