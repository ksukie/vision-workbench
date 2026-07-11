"""Cached model manifest helpers.

The UI should be able to show supported-but-not-downloaded models without
depending on a live network request.  This module keeps that path explicit:
callers read the local cache plus bundled defaults for normal dropdowns, and
only refresh the cache when the user asks for it or a startup policy does so.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from vision_workbench.trusted_models import YOLO26_ASSETS_V8_4_0


MODEL_MANIFEST_ENV_VAR = "VISION_WORKBENCH_MODEL_MANIFEST_URL"
MODEL_MANIFEST_USER_AGENT = "VisionWorkbench/0.2 model-manifest"
YOLO26_FAMILY = "yolo26"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024


class ModelManifestError(ValueError):
    """Raised when a model manifest cannot be parsed safely."""


@dataclass(frozen=True)
class ModelManifestEntry:
    """One downloadable model described by the manifest."""

    family: str
    task: str
    name: str
    url: str
    size_bytes: int | None = None
    sha256: str | None = None
    license_name: str | None = None
    source: str | None = None


def default_model_manifest_url() -> str | None:
    """Return the optional remote manifest URL configured for this machine."""

    value = os.environ.get(MODEL_MANIFEST_ENV_VAR, "").strip()
    return value or None


def default_model_manifest_cache_path() -> Path:
    """Return the shared per-user model manifest cache path."""

    return Path.home() / ".vision_workbench" / "cache" / "model_manifest.json"


def refresh_model_manifest(
    manifest_url: str | None,
    cache_path: Path,
    *,
    timeout_seconds: float = 8.0,
) -> tuple[ModelManifestEntry, ...]:
    """Download, validate, and cache a remote model manifest.

    Returns the parsed entries.  If no URL is configured, this is a no-op and
    returns an empty tuple so callers can still rescan local files.
    """

    if not manifest_url:
        return tuple()

    _validate_manifest_url(manifest_url)

    request = Request(
        manifest_url,
        headers={
            "Accept": "application/json",
            "User-Agent": MODEL_MANIFEST_USER_AGENT,
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        final_url = response.geturl() if hasattr(response, "geturl") else manifest_url
        _validate_manifest_url(final_url)
        raw_payload = response.read(MAX_MANIFEST_BYTES + 1)
    if len(raw_payload) > MAX_MANIFEST_BYTES:
        raise ModelManifestError("Model manifest exceeds the 2 MiB safety limit.")
    payload = json.loads(raw_payload.decode("utf-8"))

    entries = parse_model_manifest(payload)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(cache_path.name + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(cache_path)
    return entries


def load_cached_model_manifest(cache_path: Path) -> tuple[ModelManifestEntry, ...]:
    """Return parsed cache entries, or an empty tuple when cache is absent/bad."""

    if not cache_path.is_file():
        return tuple()
    try:
        if cache_path.stat().st_size > MAX_MANIFEST_BYTES:
            return tuple()
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return parse_model_manifest(payload)
    except Exception:
        return tuple()


def yolo26_model_entries_for_task(
    task: str,
    *,
    fallback_names_by_task: Mapping[str, Sequence[str]],
    base_url: str,
    cache_path: Path,
) -> tuple[ModelManifestEntry, ...]:
    """Return cached/default YOLO26 entries for one normalized task."""

    normalized_task = _normalize_yolo26_task(task)
    fallback_entries = default_yolo26_model_entries(fallback_names_by_task, base_url)
    cached_entries = load_cached_model_manifest(cache_path)
    merged = merge_model_entries(fallback_entries, cached_entries)
    return tuple(
        entry
        for entry in merged
        if entry.family == YOLO26_FAMILY and entry.task == normalized_task
    )


def default_yolo26_model_entries(
    names_by_task: Mapping[str, Sequence[str]],
    base_url: str,
) -> tuple[ModelManifestEntry, ...]:
    """Build bundled fallback entries from the stable config model names."""

    base = base_url.rstrip("/")
    entries = []
    for task, names in names_by_task.items():
        normalized_task = _normalize_yolo26_task(task)
        for name in names:
            safe_name = _safe_model_name(name, YOLO26_FAMILY)
            pinned = YOLO26_ASSETS_V8_4_0.get(safe_name)
            if base == "https://github.com/ultralytics/assets/releases/download/v8.4.0" and pinned is None:
                raise ModelManifestError(f"Official YOLO26 asset is missing pinned metadata: {safe_name}")
            entries.append(
                ModelManifestEntry(
                    family=YOLO26_FAMILY,
                    task=normalized_task,
                    name=safe_name,
                    url=f"{base}/{safe_name}",
                    size_bytes=pinned[0] if pinned else None,
                    sha256=pinned[1] if pinned else None,
                    source="bundled-default",
                )
            )
    return tuple(entries)


def merge_model_entries(
    fallback_entries: Iterable[ModelManifestEntry],
    cached_entries: Iterable[ModelManifestEntry],
) -> tuple[ModelManifestEntry, ...]:
    """Merge cached manifest entries over fallback entries while preserving order."""

    cached_by_key = {_entry_key(entry): entry for entry in cached_entries}
    merged = []
    seen = set()
    for fallback in fallback_entries:
        key = _entry_key(fallback)
        cached = cached_by_key.get(key)
        if fallback.sha256:
            if cached is not None and cached.sha256 == fallback.sha256:
                merged.append(
                    ModelManifestEntry(
                        family=fallback.family,
                        task=fallback.task,
                        name=fallback.name,
                        url=cached.url,
                        size_bytes=fallback.size_bytes,
                        sha256=fallback.sha256,
                        license_name=cached.license_name or fallback.license_name,
                        source=cached.source or fallback.source,
                    )
                )
            else:
                merged.append(fallback)
        else:
            merged.append(cached or fallback)
        seen.add(key)
    for cached in cached_entries:
        key = _entry_key(cached)
        if key not in seen and cached.sha256 and cached.size_bytes:
            merged.append(cached)
            seen.add(key)
    return tuple(merged)


def parse_model_manifest(payload: object) -> tuple[ModelManifestEntry, ...]:
    """Parse the supported model manifest JSON schema.

    Supported shape:

    {
      "schema_version": 1,
      "models": [
        {"family": "yolo26", "task": "detect", "name": "yolo26n.pt", "url": "..."},
        {"family": "yolo26", "tasks": ["segment"], "filename": "x.pt", "download_url": "..."}
      ]
    }
    """

    if isinstance(payload, list):
        raw_models = payload
    elif isinstance(payload, Mapping):
        raw_models = payload.get("models", [])
    else:
        raise ModelManifestError("Model manifest must be a JSON object or list.")

    if not isinstance(raw_models, list):
        raise ModelManifestError("Model manifest field 'models' must be a list.")

    entries = []
    for raw_model in raw_models:
        entries.extend(_entries_from_raw_model(raw_model))

    if not entries:
        raise ModelManifestError("Model manifest does not contain any valid model entries.")
    return tuple(entries)


def _entries_from_raw_model(raw_model: object) -> tuple[ModelManifestEntry, ...]:
    if not isinstance(raw_model, Mapping):
        raise ModelManifestError("Each model manifest entry must be an object.")

    family = str(raw_model.get("family") or YOLO26_FAMILY).strip().lower()
    name = _safe_model_name(str(raw_model.get("name") or raw_model.get("filename") or ""), family)
    url = str(raw_model.get("url") or raw_model.get("download_url") or "").strip()
    if not url:
        raise ModelManifestError(f"Model manifest entry for {name} is missing a URL.")
    _validate_model_url(url, name)

    tasks = _manifest_tasks(raw_model)
    size_bytes = _optional_int(raw_model.get("size_bytes") or raw_model.get("size"))
    sha256 = _optional_sha256(raw_model.get("sha256"))
    license_name = _optional_str(raw_model.get("license") or raw_model.get("license_name"))
    source = _optional_str(raw_model.get("source"))

    return tuple(
        ModelManifestEntry(
            family=family,
            task=_normalize_yolo26_task(task) if family == YOLO26_FAMILY else str(task).strip().lower(),
            name=name,
            url=url,
            size_bytes=size_bytes,
            sha256=sha256,
            license_name=license_name,
            source=source,
        )
        for task in tasks
    )


def _manifest_tasks(raw_model: Mapping[object, object]) -> tuple[str, ...]:
    raw_tasks = raw_model.get("tasks")
    if raw_tasks is None:
        raw_tasks = raw_model.get("task")
    if raw_tasks is None:
        raise ModelManifestError("Model manifest entry is missing 'task' or 'tasks'.")
    if isinstance(raw_tasks, str):
        tasks = (raw_tasks,)
    elif isinstance(raw_tasks, list):
        tasks = tuple(str(task) for task in raw_tasks)
    else:
        raise ModelManifestError("Model manifest tasks must be a string or list.")
    normalized = tuple(task.strip().lower() for task in tasks if task and task.strip())
    if not normalized:
        raise ModelManifestError("Model manifest entry has no valid tasks.")
    return normalized


def _safe_model_name(name: str, family: str) -> str:
    value = name.strip()
    if not value:
        raise ModelManifestError("Model manifest entry is missing a model name.")
    if "/" in value or "\\" in value or Path(value).name != value:
        raise ModelManifestError(f"Unsafe model manifest filename: {value}")
    if value in (".", "..") or any(char in value for char in ':*?"<>|'):
        raise ModelManifestError(f"Unsafe model manifest filename: {value}")
    if family == YOLO26_FAMILY and not value.lower().endswith(".pt"):
        raise ModelManifestError(f"YOLO26 model manifest filename must end with .pt: {value}")
    return value


def _normalize_yolo26_task(task: str) -> str:
    value = str(task or "detect").strip().lower()
    if value in ("seg", "instance", "instance_segmentation"):
        return "segment"
    if value in ("sem", "semantic_segmentation"):
        return "semantic"
    if value not in ("detect", "segment", "semantic"):
        return "detect"
    return value


def _entry_key(entry: ModelManifestEntry) -> tuple[str, str, str]:
    return (entry.family, entry.task, entry.name)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _optional_sha256(value: object) -> str | None:
    text = _optional_str(value)
    if text is None:
        return None
    normalized = text.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ModelManifestError("Model manifest sha256 values must contain exactly 64 hexadecimal characters.")
    return normalized


def _validate_model_url(url: str, name: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "file"}:
        raise ModelManifestError(f"Model manifest URL for {name} must use HTTPS or file://.")
    if parsed.scheme == "https" and not parsed.netloc:
        raise ModelManifestError(f"Model manifest URL for {name} is missing a host.")


def _validate_manifest_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "file"}:
        raise ModelManifestError("Remote model manifests require HTTPS or an explicit local file URL.")
    if parsed.scheme == "https" and not parsed.netloc:
        raise ModelManifestError("Remote model manifest URL is missing a host.")
