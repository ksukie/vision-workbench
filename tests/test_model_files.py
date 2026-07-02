from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pytest

import vision_workbench.model_files as model_files
from vision_workbench.model_files import ModelFileError, download_model_file, is_complete_model_file


class FakeResponse:
    def __init__(self, data: bytes, content_length: int | None = None) -> None:
        self._body = BytesIO(data)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)


def archive_bytes(payload: bytes = b"fake model") -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("data.pkl", payload)
    return output.getvalue()


def test_download_model_file_promotes_valid_archive(tmp_path: Path, monkeypatch) -> None:
    data = archive_bytes()
    progress = []

    def fake_urlopen(_url: str, timeout: int = 30) -> FakeResponse:
        return FakeResponse(data, len(data))

    monkeypatch.setattr(model_files, "urlopen", fake_urlopen)
    target = tmp_path / "model.pt"

    download_model_file("https://example.test/model.pt", target, lambda *value: progress.append(value))

    assert target.read_bytes() == data
    assert is_complete_model_file(target)
    assert not target.with_name("model.pt.part").exists()
    assert progress[-1] == (100, len(data), len(data))


def test_download_model_file_rejects_short_response(tmp_path: Path, monkeypatch) -> None:
    data = archive_bytes()

    def fake_urlopen(_url: str, timeout: int = 30) -> FakeResponse:
        return FakeResponse(data, len(data) + 1)

    monkeypatch.setattr(model_files, "urlopen", fake_urlopen)
    target = tmp_path / "model.pt"

    with pytest.raises(ModelFileError, match="Partial download"):
        download_model_file("https://example.test/model.pt", target)

    assert not target.exists()
    assert not target.with_name("model.pt.part").exists()


def test_download_model_file_rejects_corrupt_archive_without_overwriting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original = archive_bytes(b"original")

    def fake_urlopen(_url: str, timeout: int = 30) -> FakeResponse:
        return FakeResponse(b"not a zip", len(b"not a zip"))

    monkeypatch.setattr(model_files, "urlopen", fake_urlopen)
    target = tmp_path / "model.pt"
    target.write_bytes(original)

    with pytest.raises(ModelFileError, match="incomplete or corrupt"):
        download_model_file("https://example.test/model.pt", target)

    assert target.read_bytes() == original
    assert not target.with_name("model.pt.part").exists()
