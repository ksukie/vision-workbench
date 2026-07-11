"""Helpers for safely downloading and recognizing model files."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from typing import Callable, Collection
from urllib.parse import urlparse
from urllib.request import urlopen


ProgressCallback = Callable[[int | None, int, int | None], None]
CHUNK_SIZE = 1024 * 1024
DEFAULT_MAX_MODEL_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_MAX_UNCOMPRESSED_BYTES = 16 * 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 1000
MIN_HASH_PREFIX_LENGTH = 8


class ModelFileError(ValueError):
    """Raised when a downloaded model file is incomplete or corrupt."""


def is_complete_model_file(path: Path) -> bool:
    """Return True for complete modern PyTorch zip-based model files."""

    candidate = Path(path)
    if not candidate.is_file():
        return False
    try:
        size = candidate.stat().st_size
        if size <= 0 or size > DEFAULT_MAX_MODEL_BYTES:
            return False
        with zipfile.ZipFile(candidate) as archive:
            entries = archive.infolist()
            return bool(entries) and _archive_layout_is_reasonable(entries)
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
    size = candidate.stat().st_size
    if size <= 0:
        raise ModelFileError(f"Model file is empty: {candidate}")
    if size > DEFAULT_MAX_MODEL_BYTES:
        raise ModelFileError(
            f"Model file exceeds the {DEFAULT_MAX_MODEL_BYTES // (1024 ** 3)} GiB safety limit: {candidate}"
        )
    try:
        with zipfile.ZipFile(candidate) as archive:
            entries = archive.infolist()
            if not entries:
                raise ModelFileError(f"Model file is empty: {candidate}")
            if not _archive_layout_is_reasonable(entries):
                raise ModelFileError(f"Model archive expands beyond the configured safety limit: {candidate}")
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
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    max_bytes: int = DEFAULT_MAX_MODEL_BYTES,
    allowed_hosts: Collection[str] | None = None,
) -> None:
    """Download to a .part file, verify it, then atomically promote it."""

    _validate_download_url(url, allowed_hosts)
    normalized_hash = _normalize_expected_hash(expected_sha256)
    if expected_size is not None and expected_size <= 0:
        raise ModelFileError("Expected model size must be greater than zero.")
    if expected_size is not None and expected_size > max_bytes:
        raise ModelFileError(f"Expected model size exceeds the {max_bytes} byte safety limit.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f"{target.name}.part")
    if temp_path.exists():
        temp_path.unlink(missing_ok=True)

    downloaded = 0
    total = None  # type: int | None
    digest = hashlib.sha256()
    try:
        with urlopen(url, timeout=30) as response:
            final_url = response.geturl() if hasattr(response, "geturl") else url
            _validate_download_url(final_url, allowed_hosts)
            total = _content_length(response)
            if total is not None and total > max_bytes:
                raise ModelFileError(f"Model download exceeds the {max_bytes} byte safety limit.")
            with temp_path.open("wb") as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    if downloaded + len(chunk) > max_bytes:
                        raise ModelFileError(f"Model download exceeds the {max_bytes} byte safety limit.")
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    _emit_progress(progress_callback, downloaded, total)

        if total is not None and downloaded != total:
            raise ModelFileError(
                f"Partial download for {target.name}: {downloaded}/{total} bytes"
            )
        if expected_size is not None and downloaded != expected_size:
            raise ModelFileError(
                f"Model size mismatch for {target.name}: expected {expected_size}, received {downloaded} bytes"
            )
        if normalized_hash is not None and not digest.hexdigest().startswith(normalized_hash):
            raise ModelFileError(f"SHA-256 mismatch for downloaded model: {target.name}")
        validate_complete_model_file(temp_path)
        temp_path.replace(target)
        size = target.stat().st_size if target.exists() else downloaded
        _emit_progress(progress_callback, size, size or total)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def sha256_prefix_from_filename(filename: str) -> str | None:
    """Return an embedded hexadecimal hash prefix such as ``f37072fd``."""

    match = re.search(r"-([0-9a-fA-F]{8,64})(?=\.[^.]+$)", Path(filename).name)
    return match.group(1).lower() if match else None


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


def _validate_download_url(url: str, allowed_hosts: Collection[str] | None = None) -> None:
    parsed = urlparse(str(url))
    if parsed.scheme not in {"https", "file"}:
        raise ModelFileError("Model downloads require HTTPS or an explicit local file URL.")
    if parsed.scheme == "https" and not parsed.netloc:
        raise ModelFileError("Model download URL is missing a host.")
    if parsed.scheme == "https" and allowed_hosts is not None:
        normalized_hosts = {host.strip().lower() for host in allowed_hosts}
        if (parsed.hostname or "").lower() not in normalized_hosts:
            raise ModelFileError(f"Model download host is not trusted: {parsed.hostname or parsed.netloc}")


def _normalize_expected_hash(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if len(normalized) < MIN_HASH_PREFIX_LENGTH or len(normalized) > 64:
        raise ModelFileError("Expected SHA-256 must contain 8 to 64 hexadecimal characters.")
    if any(char not in "0123456789abcdef" for char in normalized):
        raise ModelFileError("Expected SHA-256 contains non-hexadecimal characters.")
    return normalized


def _archive_layout_is_reasonable(entries: list[zipfile.ZipInfo]) -> bool:
    total_compressed = sum(max(0, entry.compress_size) for entry in entries)
    total_uncompressed = sum(max(0, entry.file_size) for entry in entries)
    if total_uncompressed > DEFAULT_MAX_UNCOMPRESSED_BYTES:
        return False
    if total_compressed == 0:
        return total_uncompressed == 0
    return total_uncompressed <= total_compressed * MAX_COMPRESSION_RATIO
