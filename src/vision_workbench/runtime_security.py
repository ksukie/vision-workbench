"""Runtime security helpers shared by desktop and command-line entry points."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path


ALLOW_USER_SITE_ENV = "VISION_WORKBENCH_ALLOW_USER_SITE"
SAFE_LOAD_ENV = "ULTRALYTICS_SAFE_LOAD"
TORCH_SAFE_LOAD_ENV = "TORCH_FORCE_WEIGHTS_ONLY_LOAD"


def configure_isolated_python_environment() -> tuple[str, ...]:
    """Keep an active Conda/virtual environment from importing user-site packages.

    A user-level Torch or Qt installation can otherwise shadow the packages from
    the selected environment. Set ``VISION_WORKBENCH_ALLOW_USER_SITE=1`` to opt
    back into the normal Python user-site behavior.
    """

    if _env_flag(ALLOW_USER_SITE_ENV):
        return tuple()
    if not _inside_isolated_environment():
        return tuple()

    os.environ["PYTHONNOUSERSITE"] = "1"
    user_sites = site.getusersitepackages()
    if isinstance(user_sites, str):
        user_sites = [user_sites]
    normalized_user_sites = {_normalized_path(value) for value in user_sites}
    removed = []
    retained = []
    for entry in sys.path:
        if _normalized_path(entry) in normalized_user_sites:
            removed.append(entry)
        else:
            retained.append(entry)
    if removed:
        sys.path[:] = retained
    return tuple(removed)


def configure_restricted_model_loading() -> None:
    """Default PyTorch and Ultralytics checkpoint loading to restricted mode."""

    os.environ.setdefault(SAFE_LOAD_ENV, "true")
    os.environ.setdefault(TORCH_SAFE_LOAD_ENV, "1")


def validate_run_name(value: str, *, field_name: str = "run name") -> str:
    """Return a safe single path component for a training run."""

    name = str(value or "").strip()
    if not name:
        raise ValueError(f"{field_name} cannot be empty.")
    if len(name) > 128:
        raise ValueError(f"{field_name} must be 128 characters or fewer.")
    if name in {".", ".."} or Path(name).is_absolute() or Path(name).name != name:
        raise ValueError(f"{field_name} must be a name, not a path.")
    if any(ord(char) < 32 or char in '<>:"/\\|?*' for char in name):
        raise ValueError(f"{field_name} contains characters that are not allowed in file names.")
    if name.endswith((" ", ".")):
        raise ValueError(f"{field_name} cannot end with a space or period.")

    stem = name.split(".", 1)[0].upper()
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update({f"COM{index}" for index in range(1, 10)})
    reserved.update({f"LPT{index}" for index in range(1, 10)})
    if stem in reserved:
        raise ValueError(f"{field_name} uses a reserved Windows file name.")
    return name


def confined_child_path(root: Path, name: str, *, field_name: str = "run name") -> Path:
    """Return ``root/name`` after proving that it stays within ``root``."""

    safe_name = validate_run_name(name, field_name=field_name)
    root_path = Path(root).expanduser().resolve()
    candidate = (root_path / safe_name).resolve()
    if not candidate.is_relative_to(root_path):
        raise ValueError(f"{field_name} resolves outside the configured output directory.")
    return candidate


def _inside_isolated_environment() -> bool:
    return bool(os.environ.get("CONDA_PREFIX")) or sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _normalized_path(value: object) -> str:
    try:
        return os.path.normcase(os.path.abspath(os.fspath(value)))
    except (TypeError, ValueError):
        return ""


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
