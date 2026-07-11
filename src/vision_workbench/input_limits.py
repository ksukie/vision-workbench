"""Shared limits for untrusted local images and dataset metadata."""

from __future__ import annotations

from pathlib import Path


MAX_IMAGE_FILE_BYTES = 256 * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000
MAX_DATASET_IMAGES = 200_000
MAX_DATASET_TEXT_BYTES = 16 * 1024 * 1024


class InputLimitError(ValueError):
    """Raised when an input exceeds a bounded local processing limit."""


def validate_image_file(path: Path) -> Path:
    """Validate file size and decoded dimensions before heavy image processing."""

    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(f"Image file does not exist: {candidate}")
    size = candidate.stat().st_size
    if size <= 0:
        raise InputLimitError(f"Image file is empty: {candidate}")
    if size > MAX_IMAGE_FILE_BYTES:
        raise InputLimitError(
            f"Image file exceeds the {MAX_IMAGE_FILE_BYTES // (1024 * 1024)} MiB safety limit: {candidate}"
        )

    from PIL import Image

    with Image.open(candidate) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise InputLimitError(f"Image dimensions are invalid: {candidate}")
    if width * height > MAX_IMAGE_PIXELS:
        raise InputLimitError(
            f"Image contains {width * height:,} pixels, above the {MAX_IMAGE_PIXELS:,} pixel safety limit."
        )
    return candidate


def validate_dataset_count(count: int, *, label: str = "dataset images") -> None:
    if count > MAX_DATASET_IMAGES:
        raise InputLimitError(f"{label} exceeds the {MAX_DATASET_IMAGES:,} file safety limit.")


def read_bounded_text(path: Path, *, encoding: str = "utf-8") -> str:
    candidate = Path(path)
    size = candidate.stat().st_size
    if size > MAX_DATASET_TEXT_BYTES:
        raise InputLimitError(
            f"Dataset metadata exceeds the {MAX_DATASET_TEXT_BYTES // (1024 * 1024)} MiB safety limit: {candidate}"
        )
    return candidate.read_text(encoding=encoding)
