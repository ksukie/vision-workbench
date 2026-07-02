"""Helpers for safely downloading and recognizing model files."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Callable
from urllib.request import urlopen


ProgressCallback = Callable[[int | None, int, int | None], None]
CHUNK_SIZE = 1024 * 1024


class ModelFileError(ValueError):
    """Raised when a downloaded model file is incomplete or corrupt."""


def is_complete_model_file(path: Path) -> bool:
    """Return True for complete modern PyTorch zip-based model files."""

    candidate = Path(path)
    if not candidate.is_file():
        return False
    try:
        if candidate.stat().st_size <= 0:
            return False
        with zipfile.ZipFile(candidate) as archive:
            return bool(archive.infolist())
    except (OSError, zipfile.BadZipFile):
        return False


def partial_model_file_path(path: Path) -> Path:
    """Return the sidecar path used for an in-progress download."""

    target = Path(path)
    return target.with_name(f"{target.name}.part")


def model_file_issue(path: Path) -> str | None:
    """Return a user-facing issue for an unavailable model file."""

    target = Path(path)
    if is_complete_model_file(target):
        return None
    if target.is_file():
        return "文件不完整/损坏"
    if partial_model_file_path(target).is_file():
        return "上次下载未完成"
    return None


def validate_complete_model_file(path: Path) -> None:
    """Raise if a model archive cannot be read end-to-end."""

    candidate = Path(path)
    if not candidate.is_file():
        raise ModelFileError(f"Model file does not exist: {candidate}")
    try:
        with zipfile.ZipFile(candidate) as archive:
            if not archive.infolist():
                raise ModelFileError(f"Model file is empty: {candidate}")
            bad_entry = archive.testzip()
    except zipfile.BadZipFile as exc:
        raise ModelFileError(f"Model file is incomplete or corrupt: {candidate}") from exc
    except OSError as exc:
        raise ModelFileError(f"Cannot read model file: {candidate}") from exc
    if bad_entry is not None:
        raise ModelFileError(f"Model file contains a corrupt entry: {bad_entry}")


def download_model_file(
    url: str,
    path: Path,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Download to a .part file, validate it, then atomically promote it."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f"{target.name}.part")
    if temp_path.exists():
        temp_path.unlink(missing_ok=True)

    downloaded = 0
    total = None  # type: int | None
    try:
        with urlopen(url, timeout=30) as response:
            total = _content_length(response)
            with temp_path.open("wb") as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    _emit_progress(progress_callback, downloaded, total)

        if total is not None and downloaded != total:
            raise ModelFileError(
                f"Partial download for {target.name}: {downloaded}/{total} bytes"
            )
        validate_complete_model_file(temp_path)
        temp_path.replace(target)
        size = target.stat().st_size if target.exists() else downloaded
        _emit_progress(progress_callback, size, size or total)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def _content_length(response: object) -> int | None:
    headers = getattr(response, "headers", None)
    value = None
    if headers is not None:
        value = headers.get("Content-Length")
    if not value:
        return None
    try:
        total = int(value)
    except (TypeError, ValueError):
        return None
    return total if total > 0 else None


def _emit_progress(
    progress_callback: ProgressCallback | None,
    downloaded: int,
    total: int | None,
) -> None:
    if progress_callback is None:
        return
    if total is None or total <= 0:
        progress_callback(None, downloaded, total)
        return
    percent = min(100, int(downloaded * 100 / total))
    progress_callback(percent, min(downloaded, total), total)
