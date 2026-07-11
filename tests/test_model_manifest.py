import json
from pathlib import Path

import pytest

from vision_workbench.model_manifest import (
    default_yolo26_model_entries,
    load_cached_model_manifest,
    ModelManifestError,
    parse_model_manifest,
    refresh_model_manifest,
    yolo26_model_entries_for_task,
)
from vision_workbench.trusted_models import YOLO26_ASSETS_V8_4_0


def test_refresh_model_manifest_caches_and_merges_yolo_entries(tmp_path: Path) -> None:
    remote_manifest = tmp_path / "remote_manifest.json"
    cache_path = tmp_path / "cache" / "model_manifest.json"
    remote_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "family": "yolo26",
                        "task": "detect",
                        "name": "yolo26s.pt",
                        "url": "https://mirror.example/yolo26s.pt",
                        "size_bytes": YOLO26_ASSETS_V8_4_0["yolo26s.pt"][0],
                        "sha256": YOLO26_ASSETS_V8_4_0["yolo26s.pt"][1],
                    },
                    {
                        "family": "yolo26",
                        "tasks": ["detect"],
                        "name": "yolo26tiny.pt",
                        "url": "https://mirror.example/yolo26tiny.pt",
                        "size_bytes": 123456,
                        "sha256": "a" * 64,
                    },
                    {
                        "family": "yolo26",
                        "task": "semantic",
                        "name": "yolo26n-sem.pt",
                        "url": "https://mirror.example/yolo26n-sem.pt",
                        "size_bytes": YOLO26_ASSETS_V8_4_0["yolo26n-sem.pt"][0],
                        "sha256": YOLO26_ASSETS_V8_4_0["yolo26n-sem.pt"][1],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    refreshed = refresh_model_manifest(remote_manifest.as_uri(), cache_path)

    assert len(refreshed) == 3
    assert cache_path.is_file()

    detect_entries = yolo26_model_entries_for_task(
        "detect",
        fallback_names_by_task={"detect": ("yolo26n.pt", "yolo26s.pt")},
        base_url="https://fallback.example/models",
        cache_path=cache_path,
    )

    assert [entry.name for entry in detect_entries] == [
        "yolo26n.pt",
        "yolo26s.pt",
        "yolo26tiny.pt",
    ]
    assert detect_entries[0].url == "https://fallback.example/models/yolo26n.pt"
    assert detect_entries[1].url == "https://mirror.example/yolo26s.pt"


def test_refresh_model_manifest_noops_without_url(tmp_path: Path) -> None:
    cache_path = tmp_path / "model_manifest.json"

    assert refresh_model_manifest(None, cache_path) == ()
    assert not cache_path.exists()


def test_model_manifest_rejects_unsafe_model_names() -> None:
    with pytest.raises(ModelManifestError):
        parse_model_manifest(
            {
                "models": [
                    {
                        "family": "yolo26",
                        "task": "detect",
                        "name": "../bad.pt",
                        "url": "https://example.test/bad.pt",
                    }
                ]
            }
        )


def test_model_manifest_rejects_insecure_model_url() -> None:
    with pytest.raises(ModelManifestError, match="must use HTTPS"):
        parse_model_manifest(
            {
                "models": [
                    {
                        "family": "yolo26",
                        "task": "detect",
                        "name": "model.pt",
                        "url": "http://example.test/model.pt",
                    }
                ]
            }
        )


def test_model_manifest_rejects_invalid_sha256() -> None:
    with pytest.raises(ModelManifestError, match="exactly 64"):
        parse_model_manifest(
            {
                "models": [
                    {
                        "family": "yolo26",
                        "task": "detect",
                        "name": "model.pt",
                        "url": "https://example.test/model.pt",
                        "sha256": "abcd",
                    }
                ]
            }
        )


def test_official_default_entries_always_include_pinned_metadata() -> None:
    entries = default_yolo26_model_entries(
        {"detect": ("yolo26n.pt",), "segment": ("yolo26n-seg.pt",)},
        "https://github.com/ultralytics/assets/releases/download/v8.4.0",
    )

    assert all(entry.sha256 and len(entry.sha256) == 64 for entry in entries)
    assert all(entry.size_bytes and entry.size_bytes > 0 for entry in entries)


def test_cached_hash_mismatch_cannot_override_official_asset(tmp_path: Path) -> None:
    cache_path = tmp_path / "manifest.json"
    cache_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "family": "yolo26",
                        "task": "detect",
                        "name": "yolo26n.pt",
                        "url": "https://attacker.invalid/yolo26n.pt",
                        "size_bytes": YOLO26_ASSETS_V8_4_0["yolo26n.pt"][0],
                        "sha256": "0" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    entries = yolo26_model_entries_for_task(
        "detect",
        fallback_names_by_task={"detect": ("yolo26n.pt",)},
        base_url="https://github.com/ultralytics/assets/releases/download/v8.4.0",
        cache_path=cache_path,
    )

    assert entries[0].url.startswith("https://github.com/ultralytics/assets/")
    assert entries[0].sha256 == YOLO26_ASSETS_V8_4_0["yolo26n.pt"][1]


def test_oversized_cached_manifest_is_ignored(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "manifest.json"
    cache_path.write_text('{"models": []}', encoding="utf-8")
    monkeypatch.setattr("vision_workbench.model_manifest.MAX_MANIFEST_BYTES", 4)

    assert load_cached_model_manifest(cache_path) == ()
