import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError

import pytest

import vision_workbench
from scripts import check_version_contract, generate_update_manifest
from vision_workbench import self_test, update_helper, update_installer, versioning
from vision_workbench.update_installer import (
    PreparedUpdate,
    UpdatePreparationError,
    launch_update_helper,
    prepare_update,
)
from vision_workbench.update_service import (
    GITHUB_API_LATEST_URL,
    ReleaseAsset,
    ReleaseInfo,
    UpdateCheckError,
    UpdateCheckResult,
    UpdateClient,
    UpdateState,
)
from vision_workbench.versioning import (
    LATEST_MANIFEST_URL,
    RuntimeVersionInfo,
    current_version_info,
    source_archive_url,
    stable_version_tuple,
)


DEPENDENCY_CONTRACT = "d" * 64


class FakeResponse:
    def __init__(self, payload: bytes, *, url: str, content_length: int | None = None):
        self.payload = payload
        self.url = url
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size=-1):
        if size is None or size < 0:
            size = len(self.payload) - self._offset
        chunk = self.payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def geturl(self):
        return self.url


def _runtime(version="1.0.0", mode="editable", source_root=None):
    return RuntimeVersionInfo(
        version=version,
        updated_at="2026-07-19",
        repository_url="https://github.com/ksukie/vision-workbench",
        install_mode=mode,
        source_root=source_root,
        dependency_contract_sha256=DEPENDENCY_CONTRACT,
    )


def _manifest(version="1.1.0", *, sha256="a" * 64):
    return {
        "schema_version": 1,
        "channel": "stable",
        "version": version,
        "published_at": "2026-08-01T12:00:00Z",
        "tag": f"v{version}",
        "commit": "1" * 40,
        "repository_url": "https://github.com/ksukie/vision-workbench",
        "release_url": f"https://github.com/ksukie/vision-workbench/releases/tag/v{version}",
        "dependency_contract_sha256": DEPENDENCY_CONTRACT,
        "assets": [
            {
                "kind": "python-wheel",
                "name": f"vision_workbench-{version}-py3-none-any.whl",
                "url": (
                    "https://github.com/ksukie/vision-workbench/releases/download/"
                    f"v{version}/vision_workbench-{version}-py3-none-any.whl"
                ),
                "size": 100,
                "sha256": sha256,
            }
        ],
    }


def _json_response(payload):
    encoded = json.dumps(payload).encode("utf-8")
    return FakeResponse(encoded, url=LATEST_MANIFEST_URL, content_length=len(encoded))


def _wheel_bytes(version="1.1.0", dependency_contract=DEPENDENCY_CONTRACT):
    stream = io.BytesIO()
    release_info = {
        "schema_version": 1,
        "version": version,
        "updated_at": "2026-08-01",
        "repository_url": "https://github.com/ksukie/vision-workbench",
        "dependency_contract_sha256": dependency_contract,
    }
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            f"vision_workbench-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: vision-workbench\nVersion: {version}\n",
        )
        archive.writestr(
            f"vision_workbench-{version}.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"vision_workbench-{version}.dist-info/RECORD", "")
        archive.writestr("vision_workbench/release_info.json", json.dumps(release_info))
    return stream.getvalue()


def test_runtime_version_reads_the_current_editable_source():
    info = current_version_info()

    assert info.version == "1.0.0"
    assert info.install_mode == "editable"
    assert info.source_root is not None
    assert (info.source_root / "pyproject.toml").is_file()
    assert source_archive_url() == (
        "https://github.com/ksukie/vision-workbench/archive/refs/tags/v1.0.0.zip"
    )
    assert vision_workbench.__version__ == info.version


def test_editable_source_resolution_is_bound_to_the_imported_module(tmp_path, monkeypatch):
    live_root = tmp_path / "live"
    stale_root = tmp_path / "stale"
    live_file = live_root / "src" / "vision_workbench" / "versioning.py"
    stale_file = stale_root / "src" / "vision_workbench" / "versioning.py"
    for root, source_file in ((live_root, live_file), (stale_root, stale_file)):
        source_file.parent.mkdir(parents=True)
        source_file.write_text("# test\n", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]\nname='vision-workbench'\n", encoding="utf-8")

    class StaleDistribution:
        @staticmethod
        def read_text(_name):
            return json.dumps({"url": stale_root.as_uri(), "dir_info": {"editable": True}})

    monkeypatch.setattr(versioning, "__file__", str(live_file))
    monkeypatch.setattr(versioning.metadata, "distribution", lambda _name: StaleDistribution())

    assert versioning._editable_source_root() == live_root.resolve()


def test_editable_runtime_rejects_unsynchronized_dependency_contract(monkeypatch):
    current_version_info.cache_clear()
    monkeypatch.setattr(versioning, "project_dependency_contract", lambda _root: "e" * 64)
    try:
        with pytest.raises(RuntimeError, match="release_info.json"):
            current_version_info()
    finally:
        current_version_info.cache_clear()


def test_runtime_version_uses_bundled_identity_for_frozen_exe(monkeypatch):
    current_version_info.cache_clear()
    monkeypatch.setattr(
        versioning,
        "_editable_source_root",
        lambda: pytest.fail("frozen runtime must not inspect editable metadata"),
    )
    monkeypatch.setattr(versioning.sys, "frozen", True, raising=False)
    try:
        info = current_version_info()
        assert info.version == "1.0.0"
        assert info.install_mode == "single-file"
    finally:
        current_version_info.cache_clear()


def test_base_exe_environment_flag_alone_does_not_claim_a_frozen_identity(monkeypatch):
    current_version_info.cache_clear()
    monkeypatch.delattr(versioning.sys, "frozen", raising=False)
    monkeypatch.setenv("VISION_WORKBENCH_BASE_EXE", "1")
    try:
        info = current_version_info()
        assert info.install_mode == "editable"
    finally:
        current_version_info.cache_clear()


def test_runtime_version_uses_distribution_metadata_for_wheel(monkeypatch):
    current_version_info.cache_clear()
    monkeypatch.setattr(versioning, "_editable_source_root", lambda: None)
    monkeypatch.delattr(versioning.sys, "frozen", raising=False)
    monkeypatch.delenv("VISION_WORKBENCH_BASE_EXE", raising=False)
    monkeypatch.setattr(versioning, "_running_distribution_version", lambda: "1.2.3")
    monkeypatch.setattr(
        versioning,
        "_read_bundled_release_info",
        lambda: {
            "schema_version": 1,
            "version": "1.2.3",
            "updated_at": "2026-08-01",
            "repository_url": "https://github.com/ksukie/vision-workbench",
            "dependency_contract_sha256": "a" * 64,
        },
    )
    try:
        info = current_version_info()
        assert info.version == "1.2.3"
        assert info.install_mode == "wheel"
    finally:
        current_version_info.cache_clear()


def test_wheel_runtime_rejects_mismatched_distribution_and_bundled_versions(monkeypatch):
    current_version_info.cache_clear()
    monkeypatch.setattr(versioning, "_editable_source_root", lambda: None)
    monkeypatch.delattr(versioning.sys, "frozen", raising=False)
    monkeypatch.setattr(versioning, "_running_distribution_version", lambda: "1.2.3")
    try:
        with pytest.raises(RuntimeError, match="不一致"):
            current_version_info()
    finally:
        current_version_info.cache_clear()


def test_wheel_distribution_identity_is_bound_to_the_imported_file(tmp_path, monkeypatch):
    stale_root = tmp_path / "stale-site"
    live_root = tmp_path / "live-site"
    relative = Path("vision_workbench/versioning.py")
    stale_file = stale_root / relative
    live_file = live_root / relative
    for path in (stale_file, live_file):
        path.parent.mkdir(parents=True)
        path.write_text("# test\n", encoding="utf-8")

    class FakeDistribution:
        def __init__(self, root, distribution_version):
            self.root = root
            self.version = distribution_version

        def locate_file(self, path):
            return self.root / path

    monkeypatch.setattr(versioning, "__file__", str(live_file))
    monkeypatch.setattr(
        versioning.metadata,
        "distributions",
        lambda **_kwargs: (
            FakeDistribution(stale_root, "9.9.9"),
            FakeDistribution(live_root, "1.2.3"),
        ),
    )

    assert versioning._running_distribution_version() == "1.2.3"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1.0.0", (1, 0, 0)), ("v12.3.45", (12, 3, 45)), ("0.0.1", (0, 0, 1))],
)
def test_stable_version_tuple(value, expected):
    assert stable_version_tuple(value) == expected


@pytest.mark.parametrize("value", ["1.0", "1.0.0.dev1", "01.0.0", "latest", ""])
def test_stable_version_tuple_rejects_non_release_versions(value):
    with pytest.raises(ValueError):
        stable_version_tuple(value)


def test_update_client_selects_a_hashed_wheel_for_editable_mode():
    client = UpdateClient(opener=lambda _request, timeout: _json_response(_manifest()))

    result = client.check(_runtime())

    assert result.state is UpdateState.UPDATE_AVAILABLE
    assert result.can_install
    assert result.dependencies_compatible
    assert result.compatible_asset is not None
    assert result.compatible_asset.kind == "python-wheel"


def test_update_client_selects_the_stable_name_exe_for_frozen_mode():
    payload = _manifest()
    payload["assets"] = [
        {
            "kind": "windows-x64-exe",
            "name": "Vision-Workbench-win-x64.exe",
            "url": (
                "https://github.com/ksukie/vision-workbench/releases/download/"
                "v1.1.0/Vision-Workbench-win-x64.exe"
            ),
            "size": 100,
            "sha256": "a" * 64,
        }
    ]
    client = UpdateClient(opener=lambda _request, timeout: _json_response(payload))

    result = client.check(_runtime(mode="single-file"))

    assert result.can_install
    assert result.compatible_asset is not None
    assert result.compatible_asset.name == "Vision-Workbench-win-x64.exe"


def test_update_client_reports_new_release_without_digest_as_not_installable():
    payload = _manifest(sha256=None)
    client = UpdateClient(opener=lambda _request, timeout: _json_response(payload))

    result = client.check(_runtime())

    assert result.state is UpdateState.UPDATE_AVAILABLE
    assert not result.can_install


def test_update_client_rejects_zero_length_asset_for_one_click():
    payload = _manifest()
    payload["assets"][0]["size"] = 0
    client = UpdateClient(opener=lambda _request, timeout: _json_response(payload))

    result = client.check(_runtime())

    assert result.compatible_asset is not None
    assert not result.compatible_asset.installable
    assert not result.can_install


def test_update_client_blocks_wheel_when_dependency_contract_changes():
    payload = _manifest()
    payload["dependency_contract_sha256"] = "e" * 64
    client = UpdateClient(opener=lambda _request, timeout: _json_response(payload))

    result = client.check(_runtime())

    assert result.state is UpdateState.UPDATE_AVAILABLE
    assert result.compatible_asset is not None
    assert result.compatible_asset.installable
    assert not result.dependencies_compatible
    assert not result.can_install


def test_update_client_rejects_nonofficial_asset_url():
    payload = _manifest()
    payload["assets"][0]["url"] = "https://example.com/update.whl"
    client = UpdateClient(opener=lambda _request, timeout: _json_response(payload))

    with pytest.raises(UpdateCheckError, match="不是允许|官方"):
        client.fetch_latest()


def test_update_client_rejects_nonstandard_official_host_url():
    payload = _manifest()
    payload["assets"][0]["url"] = (
        "https://github.com:444/ksukie/vision-workbench/releases/download/"
        "v1.1.0/vision_workbench-1.1.0-py3-none-any.whl"
    )
    client = UpdateClient(opener=lambda _request, timeout: _json_response(payload))

    with pytest.raises(UpdateCheckError, match="HTTPS"):
        client.fetch_latest()


def test_update_client_falls_back_to_github_release_api():
    api_payload = {
        "draft": False,
        "prerelease": False,
        "tag_name": "v1.1.0",
        "html_url": "https://github.com/ksukie/vision-workbench/releases/tag/v1.1.0",
        "published_at": "2026-08-01T12:00:00Z",
        "target_commitish": "main",
        "assets": [
            {
                "name": "vision_workbench-1.1.0-py3-none-any.whl",
                "browser_download_url": (
                    "https://github.com/ksukie/vision-workbench/releases/download/"
                    "v1.1.0/vision_workbench-1.1.0-py3-none-any.whl"
                ),
                "size": 100,
                "digest": f"sha256:{'b' * 64}",
            },
            {
                "name": "Vision-Workbench-1.1.0-win-x64.exe",
                "browser_download_url": (
                    "https://github.com/ksukie/vision-workbench/releases/download/"
                    "v1.1.0/Vision-Workbench-1.1.0-win-x64.exe"
                ),
                "size": 100,
                "digest": f"sha256:{'c' * 64}",
            },
        ],
    }

    def opener(request, timeout):
        if request.full_url == LATEST_MANIFEST_URL:
            raise HTTPError(request.full_url, 404, "missing", {}, None)
        assert request.full_url == GITHUB_API_LATEST_URL
        encoded = json.dumps(api_payload).encode("utf-8")
        return FakeResponse(encoded, url=GITHUB_API_LATEST_URL, content_length=len(encoded))

    result = UpdateClient(opener=opener).check(_runtime())

    assert result.state is UpdateState.UPDATE_AVAILABLE
    assert not result.can_install
    assert not result.dependencies_compatible


def test_update_client_rejects_github_release_url_for_a_different_tag():
    api_payload = {
        "draft": False,
        "prerelease": False,
        "tag_name": "v1.1.0",
        "html_url": "https://github.com/ksukie/vision-workbench/releases/tag/v9.9.9",
        "published_at": "2026-08-01T12:00:00Z",
        "assets": [],
    }

    def opener(request, timeout):
        if request.full_url == LATEST_MANIFEST_URL:
            raise HTTPError(request.full_url, 404, "missing", {}, None)
        encoded = json.dumps(api_payload).encode("utf-8")
        return FakeResponse(encoded, url=GITHUB_API_LATEST_URL, content_length=len(encoded))

    with pytest.raises(UpdateCheckError, match="URL 与版本不一致"):
        UpdateClient(opener=opener).fetch_latest()


def test_prepare_update_downloads_and_rechecks_sha256(tmp_path, monkeypatch):
    payload = _wheel_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    asset = ReleaseAsset(
        kind="python-wheel",
        name="vision_workbench-1.1.0-py3-none-any.whl",
        url=(
            "https://github.com/ksukie/vision-workbench/releases/download/"
            "v1.1.0/vision_workbench-1.1.0-py3-none-any.whl"
        ),
        size=len(payload),
        sha256=digest,
    )
    release = ReleaseInfo(
        version="1.1.0",
        published_at="2026-08-01T12:00:00Z",
        release_url="https://github.com/ksukie/vision-workbench/releases/tag/v1.1.0",
        tag="v1.1.0",
        commit="1" * 40,
        assets=(asset,),
        dependency_contract_sha256=DEPENDENCY_CONTRACT,
    )
    result = UpdateCheckResult(UpdateState.UPDATE_AVAILABLE, _runtime(), release, "now", asset)
    monkeypatch.setenv("VISION_WORKBENCH_UPDATE_DIR", str(tmp_path))
    progress = []

    prepared = prepare_update(
        result,
        lambda downloaded, total: progress.append((downloaded, total)),
        opener=lambda _request, timeout: FakeResponse(
            payload,
            url="https://release-assets.githubusercontent.com/release.whl",
            content_length=len(payload),
        ),
    )

    assert prepared.asset_path.read_bytes() == payload
    assert json.loads(prepared.plan_path.read_text(encoding="utf-8"))["target_version"] == "1.1.0"
    assert progress[-1] == (len(payload), len(payload))


def test_prepare_update_rejects_mislabeled_wheel_identity(tmp_path, monkeypatch):
    payload = _wheel_bytes(version="9.9.9")
    digest = hashlib.sha256(payload).hexdigest()
    asset = ReleaseAsset(
        kind="python-wheel",
        name="vision_workbench-1.1.0-py3-none-any.whl",
        url=(
            "https://github.com/ksukie/vision-workbench/releases/download/"
            "v1.1.0/vision_workbench-1.1.0-py3-none-any.whl"
        ),
        size=len(payload),
        sha256=digest,
    )
    release = ReleaseInfo(
        "1.1.0",
        "2026-08-01T12:00:00Z",
        "https://github.com/ksukie/vision-workbench/releases/tag/v1.1.0",
        "v1.1.0",
        None,
        (asset,),
        DEPENDENCY_CONTRACT,
    )
    result = UpdateCheckResult(UpdateState.UPDATE_AVAILABLE, _runtime(), release, "now", asset)
    monkeypatch.setenv("VISION_WORKBENCH_UPDATE_DIR", str(tmp_path))

    with pytest.raises(UpdatePreparationError, match="版本元数据|版本不一致"):
        prepare_update(
            result,
            opener=lambda _request, timeout: FakeResponse(
                payload,
                url="https://release-assets.githubusercontent.com/release.whl",
                content_length=len(payload),
            ),
        )


def test_prepare_update_removes_corrupt_partial_download(tmp_path, monkeypatch):
    expected = b"expected"
    actual = b"tampered"
    asset = ReleaseAsset(
        kind="python-wheel",
        name="vision_workbench-1.1.0-py3-none-any.whl",
        url=(
            "https://github.com/ksukie/vision-workbench/releases/download/"
            "v1.1.0/vision_workbench-1.1.0-py3-none-any.whl"
        ),
        size=len(actual),
        sha256=hashlib.sha256(expected).hexdigest(),
    )
    release = ReleaseInfo(
        "1.1.0",
        "2026-08-01T12:00:00Z",
        "https://github.com/ksukie/vision-workbench/releases/tag/v1.1.0",
        "v1.1.0",
        None,
        (asset,),
        DEPENDENCY_CONTRACT,
    )
    result = UpdateCheckResult(UpdateState.UPDATE_AVAILABLE, _runtime(), release, "now", asset)
    monkeypatch.setenv("VISION_WORKBENCH_UPDATE_DIR", str(tmp_path))

    with pytest.raises(UpdatePreparationError, match="SHA-256"):
        prepare_update(
            result,
            opener=lambda _request, timeout: FakeResponse(
                actual,
                url="https://release-assets.githubusercontent.com/release.whl",
                content_length=len(actual),
            ),
        )

    assert not list(tmp_path.rglob("*.part"))


def test_launch_update_helper_uses_an_external_python_process(tmp_path, monkeypatch):
    plan = tmp_path / "update-plan.json"
    plan.write_text("{}", encoding="utf-8")
    prepared = PreparedUpdate("1.1.0", tmp_path / "release.whl", plan)
    calls = []
    monkeypatch.setattr(update_installer.subprocess, "Popen", lambda command, **kwargs: calls.append((command, kwargs)))

    launch_update_helper(prepared)

    command, kwargs = calls[0]
    assert command[:3] == [update_installer.sys.executable, "-m", "vision_workbench.update_helper"]
    assert "--parent-pid" in command
    assert kwargs["cwd"] == str(tmp_path)


def test_update_installer_prefers_console_python_for_windows_gui_launcher(tmp_path, monkeypatch):
    python = tmp_path / "python.exe"
    pythonw = tmp_path / "pythonw.exe"
    python.write_bytes(b"")
    pythonw.write_bytes(b"")
    monkeypatch.setattr(update_installer.sys, "platform", "win32")
    monkeypatch.setattr(update_installer.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(update_installer.sys, "executable", str(tmp_path / "pythonw.exe"))

    assert update_installer._console_python_executable() == python.resolve()
    assert update_installer._gui_python_executable() == pythonw.resolve()


def test_update_helper_installs_wheel_and_restarts(tmp_path, monkeypatch):
    wheel = tmp_path / "vision_workbench-1.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    python = tmp_path / "python.exe"
    pythonw = tmp_path / "pythonw.exe"
    python.write_bytes(b"")
    pythonw.write_bytes(b"")
    plan = {
        "asset_kind": "python-wheel",
        "asset_path": str(wheel),
        "python_executable": str(python),
        "application_executable": str(pythonw),
        "source_root": None,
        "target_version": "1.1.0",
    }
    commands = []
    restarts = []
    monkeypatch.setattr(update_helper, "_run_logged", lambda command, _log: commands.append(command) or 0)
    monkeypatch.setattr(update_helper.subprocess, "Popen", lambda command, **_kwargs: restarts.append(command))

    update_helper._apply_python_wheel(plan, tmp_path / "update.log")

    assert commands[0] == [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--force-reinstall",
        str(wheel),
    ]
    assert commands[1][:2] == [str(python), "-c"]
    assert len(commands[1]) == 4
    assert commands[1][-1] == "1.1.0"
    assert restarts == [[str(pythonw), "-m", "vision_workbench.desktop.app"]]


def test_update_helper_restores_editable_registration_after_pip_failure(tmp_path, monkeypatch):
    wheel = tmp_path / "vision_workbench-1.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    source_root = tmp_path / "source"
    source_root.mkdir()
    python = tmp_path / "python.exe"
    pythonw = tmp_path / "pythonw.exe"
    python.write_bytes(b"")
    pythonw.write_bytes(b"")
    plan = {
        "asset_kind": "python-wheel",
        "asset_path": str(wheel),
        "python_executable": str(python),
        "application_executable": str(pythonw),
        "source_root": str(source_root),
        "target_version": "1.1.0",
    }
    commands = []
    restarts = []

    def run_logged(command, _log):
        commands.append(command)
        return 1 if len(commands) == 1 else 0

    monkeypatch.setattr(update_helper, "_run_logged", run_logged)
    monkeypatch.setattr(update_helper.subprocess, "Popen", lambda command, **_kwargs: restarts.append(command))

    with pytest.raises(update_helper.UpdateApplyError, match="pip"):
        update_helper._apply_python_wheel(plan, tmp_path / "update.log")

    assert commands[1] == [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-build-isolation",
        "--force-reinstall",
        "--editable",
        str(source_root),
    ]
    assert restarts == [[str(pythonw), "-m", "vision_workbench.desktop.app"]]


def test_update_helper_restores_editable_when_post_install_identity_probe_fails(tmp_path, monkeypatch):
    wheel = tmp_path / "vision_workbench-1.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    source_root = tmp_path / "source"
    source_root.mkdir()
    python = tmp_path / "python.exe"
    pythonw = tmp_path / "pythonw.exe"
    python.write_bytes(b"")
    pythonw.write_bytes(b"")
    plan = {
        "asset_kind": "python-wheel",
        "asset_path": str(wheel),
        "python_executable": str(python),
        "application_executable": str(pythonw),
        "source_root": str(source_root),
        "target_version": "1.1.0",
    }
    commands = []
    restarts = []

    def run_logged(command, _log):
        commands.append(command)
        return 1 if len(commands) == 2 else 0

    monkeypatch.setattr(update_helper, "_run_logged", run_logged)
    monkeypatch.setattr(update_helper.subprocess, "Popen", lambda command, **_kwargs: restarts.append(command))

    with pytest.raises(update_helper.UpdateApplyError, match="版本身份复核失败"):
        update_helper._apply_python_wheel(plan, tmp_path / "update.log")

    assert "--editable" in commands[2]
    assert restarts == [[str(pythonw), "-m", "vision_workbench.desktop.app"]]


def test_update_helper_stages_single_file_beside_target_before_atomic_replace(tmp_path, monkeypatch):
    install_dir = tmp_path / "portable"
    cache_dir = tmp_path / "cache"
    install_dir.mkdir()
    cache_dir.mkdir()
    current = install_dir / "Vision-Workbench.exe"
    downloaded = cache_dir / "Vision-Workbench-win-x64.exe"
    current.write_bytes(b"old executable")
    downloaded.write_bytes(b"new executable")
    plan = {
        "asset_kind": "windows-x64-exe",
        "asset_path": str(downloaded),
        "asset_sha256": hashlib.sha256(downloaded.read_bytes()).hexdigest(),
        "application_executable": str(current),
        "target_version": "1.1.0",
    }
    replacements = []
    restarts = []

    def replace_with_backup(target, replacement, backup):
        replacements.append((target, replacement, backup, replacement.read_bytes()))
        target.replace(backup)
        replacement.replace(target)

    monkeypatch.setattr(update_helper, "_is_windows", lambda: True)
    monkeypatch.setattr(update_helper, "_self_test_executable", lambda *_args: None)
    monkeypatch.setattr(update_helper, "_replace_file_with_backup", replace_with_backup)
    monkeypatch.setattr(update_helper.subprocess, "Popen", lambda command, **_kwargs: restarts.append(command))

    update_helper._apply_single_file(plan, tmp_path / "update.log")

    target, staged, backup, staged_bytes = replacements[0]
    assert target == current
    assert staged.parent == current.parent
    assert staged_bytes == b"new executable"
    assert current.read_bytes() == b"new executable"
    assert backup.read_bytes() == b"old executable"
    assert downloaded.read_bytes() == b"new executable"
    assert restarts == [[str(current)]]


def test_update_helper_keeps_existing_backup_when_new_exe_self_test_fails(tmp_path, monkeypatch):
    current = tmp_path / "Vision-Workbench.exe"
    downloaded = tmp_path / "Vision-Workbench-win-x64.exe"
    backup = tmp_path / "Vision-Workbench.previous.exe"
    current.write_bytes(b"current")
    downloaded.write_bytes(b"new")
    backup.write_bytes(b"previous")
    plan = {
        "asset_kind": "windows-x64-exe",
        "asset_path": str(downloaded),
        "asset_sha256": hashlib.sha256(downloaded.read_bytes()).hexdigest(),
        "application_executable": str(current),
        "target_version": "1.1.0",
    }
    monkeypatch.setattr(update_helper, "_is_windows", lambda: True)
    monkeypatch.setattr(
        update_helper,
        "_self_test_executable",
        lambda *_args: (_ for _ in ()).throw(update_helper.UpdateApplyError("self-test failed")),
    )

    with pytest.raises(update_helper.UpdateApplyError, match="self-test failed"):
        update_helper._apply_single_file(plan, tmp_path / "update.log")

    assert current.read_bytes() == b"current"
    assert backup.read_bytes() == b"previous"


@pytest.mark.skipif(os.name != "nt", reason="ReplaceFileW is Windows-only")
def test_update_helper_windows_atomic_replace_preserves_backup(tmp_path):
    current = tmp_path / "Vision-Workbench.exe"
    replacement = tmp_path / ".Vision-Workbench.exe.update"
    backup = tmp_path / "Vision-Workbench.previous.exe"
    current.write_bytes(b"old executable")
    replacement.write_bytes(b"new executable")

    update_helper._replace_file_with_backup(current, replacement, backup)

    assert current.read_bytes() == b"new executable"
    assert backup.read_bytes() == b"old executable"
    assert not replacement.exists()


def test_update_helper_rejects_tampered_plan_asset(tmp_path):
    asset = tmp_path / "vision_workbench-1.1.0-py3-none-any.whl"
    asset.write_bytes(b"tampered")
    plan = tmp_path / "update-plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "wheel",
                "asset_path": str(asset),
                "asset_sha256": hashlib.sha256(b"expected").hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(update_helper.UpdateApplyError, match="SHA-256"):
        update_helper._read_plan(plan)


def test_update_helper_accepts_a_complete_verified_editable_plan(tmp_path):
    wheel = tmp_path / "vision_workbench-1.1.0-py3-none-any.whl"
    wheel.write_bytes(_wheel_bytes())
    source_root = Path(__file__).resolve().parents[1]
    plan = tmp_path / "update-plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "editable",
                "current_version": "1.0.0",
                "target_version": "1.1.0",
                "asset_kind": "python-wheel",
                "asset_path": str(wheel),
                "asset_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "current_dependency_contract_sha256": DEPENDENCY_CONTRACT,
                "target_dependency_contract_sha256": DEPENDENCY_CONTRACT,
                "python_executable": str(Path(sys.executable).resolve()),
                "application_executable": str(Path(sys.executable).resolve()),
                "source_root": str(source_root),
                "release_url": "https://github.com/ksukie/vision-workbench/releases/tag/v1.1.0",
            }
        ),
        encoding="utf-8",
    )

    assert update_helper._read_plan(plan)["target_version"] == "1.1.0"


def test_update_helper_sanitizes_python_and_pip_environment(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "untrusted-source")
    monkeypatch.setenv("PIP_TARGET", "redirected-install")
    monkeypatch.setenv("PIP_USER", "1")

    environment = update_helper._update_subprocess_environment()

    assert "PYTHONPATH" not in environment
    assert "PIP_TARGET" not in environment
    assert "PIP_USER" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PIP_NO_INDEX"] == "1"


def test_generate_update_manifest_hashes_exact_1_0_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_update_manifest, "project_version", lambda: "1.0.0")
    exe = tmp_path / "Vision-Workbench-win-x64.exe"
    wheel = tmp_path / "vision_workbench-1.0.0-py3-none-any.whl"
    exe.write_bytes(b"exe")
    dependency_contract = generate_update_manifest.project_dependency_contract(
        generate_update_manifest.PROJECT_ROOT
    )
    wheel.write_bytes(_wheel_bytes("1.0.0", dependency_contract))

    manifest = generate_update_manifest.build_manifest(
        tmp_path,
        "2026-07-19T12:00:00Z",
        "1" * 40,
    )

    assert manifest["version"] == "1.0.0"
    assert len(manifest["dependency_contract_sha256"]) == 64
    assert {item["name"] for item in manifest["assets"]} == {exe.name, wheel.name}
    assert all(len(item["sha256"]) == 64 for item in manifest["assets"])


def test_generate_update_manifest_rejects_ambiguous_release_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_update_manifest, "project_version", lambda: "1.0.0")

    with pytest.raises(ValueError, match="timezone"):
        generate_update_manifest.build_manifest(tmp_path, "2026-07-19", "1" * 40)
    with pytest.raises(ValueError, match="40-character"):
        generate_update_manifest.build_manifest(tmp_path, "2026-07-19T12:00:00Z", "short")


def test_generate_update_manifest_derives_commit_from_clean_exact_tag(monkeypatch):
    monkeypatch.setattr(generate_update_manifest, "project_version", lambda: "1.0.0")
    values = {
        ("describe", "--tags", "--exact-match", "HEAD"): "v1.0.0",
        ("cat-file", "-t", "v1.0.0"): "tag",
        ("status", "--porcelain"): "",
        ("rev-parse", "HEAD"): "1" * 40,
    }
    monkeypatch.setattr(generate_update_manifest, "_git", lambda *args: values[args])

    assert generate_update_manifest.release_commit() == "1" * 40


def test_generate_update_manifest_rejects_mismatched_release_tag(monkeypatch):
    monkeypatch.setattr(generate_update_manifest, "project_version", lambda: "1.0.0")
    monkeypatch.setattr(generate_update_manifest, "_git", lambda *args: "v0.4.0")

    with pytest.raises(ValueError, match="does not match v1.0.0"):
        generate_update_manifest.release_commit()


def test_generate_update_manifest_rejects_lightweight_release_tag(monkeypatch):
    monkeypatch.setattr(generate_update_manifest, "project_version", lambda: "1.0.0")
    values = {
        ("describe", "--tags", "--exact-match", "HEAD"): "v1.0.0",
        ("cat-file", "-t", "v1.0.0"): "commit",
    }
    monkeypatch.setattr(generate_update_manifest, "_git", lambda *args: values[args])

    with pytest.raises(ValueError, match="annotated"):
        generate_update_manifest.release_commit()


def test_repository_version_contract_is_consistent():
    assert check_version_contract.contract_errors() == []


def test_version_contract_reports_the_expected_dependency_fingerprint(monkeypatch):
    monkeypatch.setattr(check_version_contract, "project_dependency_contract", lambda _root: "f" * 64)

    errors = check_version_contract.contract_errors()

    assert any("expected " + "f" * 64 in error for error in errors)


def test_source_installation_self_test_accepts_stale_pip_metadata():
    assert self_test.installation_errors(expected_version="1.0.0", expected_mode="editable") == []
