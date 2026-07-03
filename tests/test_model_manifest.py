import json
from pathlib import Path

import pytest

from vision_workbench.model_manifest import (
    ModelManifestError,
    parse_model_manifest,
    refresh_model_manifest,
    yolo26_model_entries_for_task,
)


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
                    },
                    {
                        "family": "yolo26",
                        "tasks": ["detect"],
                        "name": "yolo26tiny.pt",
                        "url": "https://mirror.example/yolo26tiny.pt",
                    },
                    {
                        "family": "yolo26",
                        "task": "semantic",
                        "name": "yolo26n-sem.pt",
                        "url": "https://mirror.example/yolo26n-sem.pt",
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
