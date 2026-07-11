from __future__ import annotations

import hashlib
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

import vision_workbench.model_files as model_files
from vision_workbench.model_files import ModelFileError, download_model_file, is_complete_model_file


class FakeResponse:
    def __init__(
        self,
        data: bytes,
        content_length: int | None = None,
        final_url: str | None = None,
    ) -> None:
        self._body = BytesIO(data)
        self._final_url = final_url
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def geturl(self) -> str:
        return self._final_url or "https://example.test/model.pt"


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


def test_download_model_file_verifies_sha256_and_size(tmp_path: Path, monkeypatch) -> None:
    data = archive_bytes()

    def fake_urlopen(_url: str, timeout: int = 30) -> FakeResponse:
        return FakeResponse(data, len(data))

    monkeypatch.setattr(model_files, "urlopen", fake_urlopen)
    target = tmp_path / "model.pt"

    download_model_file(
        "https://example.test/model.pt",
        target,
        expected_sha256=hashlib.sha256(data).hexdigest(),
        expected_size=len(data),
    )

    assert target.read_bytes() == data


def test_download_model_file_rejects_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    data = archive_bytes()

    def fake_urlopen(_url: str, timeout: int = 30) -> FakeResponse:
        return FakeResponse(data, len(data))

    monkeypatch.setattr(model_files, "urlopen", fake_urlopen)

    with pytest.raises(ModelFileError, match="SHA-256 mismatch"):
        download_model_file(
            "https://example.test/model.pt",
            tmp_path / "model.pt",
            expected_sha256="0" * 64,
        )


def test_download_model_file_rejects_insecure_url(tmp_path: Path) -> None:
    with pytest.raises(ModelFileError, match="require HTTPS"):
        download_model_file("http://example.test/model.pt", tmp_path / "model.pt")


def test_download_model_file_rejects_response_over_limit(tmp_path: Path, monkeypatch) -> None:
    data = archive_bytes()

    def fake_urlopen(_url: str, timeout: int = 30) -> FakeResponse:
        return FakeResponse(data, len(data))

    monkeypatch.setattr(model_files, "urlopen", fake_urlopen)

    with pytest.raises(ModelFileError, match="safety limit"):
        download_model_file("https://example.test/model.pt", tmp_path / "model.pt", max_bytes=8)


def test_download_model_file_rejects_untrusted_host(tmp_path: Path) -> None:
    with pytest.raises(ModelFileError, match="not trusted"):
        download_model_file(
            "https://example.test/model.pt",
            tmp_path / "model.pt",
            allowed_hosts={"download.pytorch.org"},
        )


def test_download_model_file_rejects_redirect_to_untrusted_host(tmp_path: Path, monkeypatch) -> None:
    data = archive_bytes()

    def fake_urlopen(_url: str, timeout: int = 30) -> FakeResponse:
        return FakeResponse(data, len(data), "https://attacker.invalid/model.pt")

    monkeypatch.setattr(model_files, "urlopen", fake_urlopen)

    with pytest.raises(ModelFileError, match="not trusted"):
        download_model_file(
            "https://download.pytorch.org/models/model.pt",
            tmp_path / "model.pt",
            allowed_hosts={"download.pytorch.org"},
        )


def test_model_archive_compression_ratio_is_bounded(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "compressed.pt"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.pkl", b"0" * 100_000)
    monkeypatch.setattr(model_files, "MAX_COMPRESSION_RATIO", 2)

    assert not is_complete_model_file(target)
    with pytest.raises(ModelFileError, match="safety limit"):
        model_files.validate_complete_model_file(target)
