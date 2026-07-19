"""Shared helpers for Vision Workbench."""

__all__ = ["__version__"]


def __getattr__(name: str):
    if name != "__version__":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .versioning import current_version_info

    version = current_version_info().version
    globals()[name] = version
    return version
