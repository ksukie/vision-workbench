"""Download, verify, and hand off a Vision Workbench update safely."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path
from pathlib import PurePosixPath
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .update_service import ALLOWED_DOWNLOAD_HOSTS, MAX_UPDATE_ASSET_BYTES, UpdateCheckResult
from .versioning import DISTRIBUTION_NAME, REPOSITORY_URL, stable_version_tuple


MAX_UPDATE_BYTES = MAX_UPDATE_ASSET_BYTES
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MAX_WHEEL_METADATA_BYTES = 2 * 1024 * 1024
MAX_RELEASE_INFO_BYTES = 64 * 1024


class UpdatePreparationError(RuntimeError):
    """Raised before any installer is launched when an update is unsafe."""


@dataclass(frozen=True)
class PreparedUpdate:
    version: str
    asset_path: Path
    plan_path: Path


def prepare_update(
    result: UpdateCheckResult,
    progress: Callable[[int, int | None], None] | None = None,
    *,
    opener: Callable[..., object] = urlopen,
) -> PreparedUpdate:
    """Download a compatible release asset and create a local update plan."""

    asset = result.compatible_asset
    if not result.can_install or asset is None or asset.sha256 is None:
        raise UpdatePreparationError("当前更新缺少兼容资产或 SHA-256，不能一键安装。")
    if asset.size <= 0 or asset.size > MAX_UPDATE_BYTES:
        raise UpdatePreparationError("更新资产大小无效或超过 2 GB 安全限制。")
    if Path(asset.name).name != asset.name:
        raise UpdatePreparationError("更新资产文件名不安全。")

    version_dir = update_cache_root() / result.latest.version
    version_dir.mkdir(parents=True, exist_ok=True)
    destination = version_dir / asset.name
    if not _matches_sha256(destination, asset.sha256):
        _download_verified(asset.url, destination, asset.size, asset.sha256, progress, opener=opener)
    elif progress is not None:
        progress(destination.stat().st_size, destination.stat().st_size)
    if asset.kind == "python-wheel":
        target_contract = result.latest.dependency_contract_sha256
        if target_contract is None:
            raise UpdatePreparationError("Python 更新缺少目标运行依赖契约。")
        validate_python_wheel(destination, result.latest.version, target_contract)

    python_executable = _console_python_executable()
    application_executable = (
        Path(sys.executable).resolve() if bool(getattr(sys, "frozen", False)) else _gui_python_executable()
    )
    plan = {
        "schema_version": 1,
        "mode": result.current.install_mode,
        "current_version": result.current.version,
        "target_version": result.latest.version,
        "asset_kind": asset.kind,
        "asset_path": str(destination.resolve()),
        "asset_sha256": asset.sha256,
        "current_dependency_contract_sha256": result.current.dependency_contract_sha256,
        "target_dependency_contract_sha256": result.latest.dependency_contract_sha256,
        "python_executable": str(python_executable),
        "application_executable": str(application_executable),
        "source_root": str(result.current.source_root.resolve()) if result.current.source_root else None,
        "release_url": result.latest.release_url,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    plan_path = version_dir / "update-plan.json"
    _write_json_atomic(plan_path, plan)
    return PreparedUpdate(result.latest.version, destination, plan_path)


def launch_update_helper(prepared: PreparedUpdate) -> None:
    """Start an out-of-process updater that waits for this application to exit."""

    common_args = ["--plan", str(prepared.plan_path), "--parent-pid", str(os.getpid())]
    kwargs: dict[str, object] = {
        "cwd": str(prepared.plan_path.parent),
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    try:
        if bool(getattr(sys, "frozen", False)):
            helper_path = prepared.plan_path.parent / f"VisionWorkbenchUpdateHelper-{os.getpid()}.exe"
            shutil.copy2(sys.executable, helper_path)
            command = [str(helper_path), "--vision-workbench-apply-update", *common_args]
        else:
            command = [str(_console_python_executable()), "-m", "vision_workbench.update_helper", *common_args]
        subprocess.Popen(command, **kwargs)
    except OSError as exc:
        raise UpdatePreparationError(f"无法启动更新助手：{exc}") from exc


def validate_python_wheel(path: Path, expected_version: str, expected_dependency_contract: str) -> None:
    """Fail closed when a downloaded wheel's internal identity is inconsistent."""

    stable_version_tuple(expected_version)
    if re.fullmatch(r"[0-9a-f]{64}", expected_dependency_contract) is None:
        raise UpdatePreparationError("目标运行依赖契约指纹无效。")
    expected_name = f"vision_workbench-{expected_version}-py3-none-any.whl"
    if path.name != expected_name:
        raise UpdatePreparationError("Python 更新 wheel 的文件名与目标版本不一致。")

    dist_info = f"vision_workbench-{expected_version}.dist-info"
    metadata_name = f"{dist_info}/METADATA"
    wheel_metadata_name = f"{dist_info}/WHEEL"
    record_name = f"{dist_info}/RECORD"
    release_info_name = "vision_workbench/release_info.json"
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise UpdatePreparationError("Python 更新 wheel 包含重复路径。")
            uncompressed_bytes = 0
            for info in archive.infolist():
                name = info.filename
                member = PurePosixPath(name)
                if "\\" in name or member.is_absolute() or ".." in member.parts:
                    raise UpdatePreparationError("Python 更新 wheel 包含不安全路径。")
                if info.flag_bits & 0x1:
                    raise UpdatePreparationError("Python 更新 wheel 不允许加密条目。")
                if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
                    raise UpdatePreparationError("Python 更新 wheel 不允许符号链接。")
                uncompressed_bytes += info.file_size
                if uncompressed_bytes > MAX_UPDATE_BYTES:
                    raise UpdatePreparationError("Python 更新 wheel 的解压后大小超过 2 GB。")
            required_members = (metadata_name, wheel_metadata_name, record_name, release_info_name)
            if any(names.count(name) != 1 for name in required_members):
                raise UpdatePreparationError("Python 更新 wheel 缺少唯一的版本元数据。")
            if archive.getinfo(metadata_name).file_size > MAX_WHEEL_METADATA_BYTES:
                raise UpdatePreparationError("Python 更新 wheel 的 METADATA 过大。")
            if archive.getinfo(release_info_name).file_size > MAX_RELEASE_INFO_BYTES:
                raise UpdatePreparationError("Python 更新 wheel 的 release_info.json 过大。")
            package_metadata = BytesParser(policy=email_policy).parsebytes(archive.read(metadata_name))
            wheel_metadata = BytesParser(policy=email_policy).parsebytes(archive.read(wheel_metadata_name))
            release_info = json.loads(archive.read(release_info_name).decode("utf-8"))
    except UpdatePreparationError:
        raise
    except (OSError, KeyError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise UpdatePreparationError("Python 更新 wheel 不是有效的正式发布包。") from exc

    distribution_name = str(package_metadata.get("Name") or "").strip().lower().replace("_", "-")
    metadata_version = str(package_metadata.get("Version") or "").strip()
    if distribution_name != DISTRIBUTION_NAME or metadata_version != expected_version:
        raise UpdatePreparationError("Python 更新 wheel 的包名或版本元数据不一致。")
    if (
        str(wheel_metadata.get("Wheel-Version") or "").strip() != "1.0"
        or str(wheel_metadata.get("Root-Is-Purelib") or "").strip().lower() != "true"
        or "py3-none-any" not in [str(value).strip() for value in wheel_metadata.get_all("Tag", [])]
    ):
        raise UpdatePreparationError("Python 更新 wheel 的兼容性标签与正式文件名不一致。")
    if not isinstance(release_info, dict) or release_info.get("schema_version") != 1:
        raise UpdatePreparationError("Python 更新 wheel 的内置发布信息格式无效。")
    if release_info.get("version") != expected_version:
        raise UpdatePreparationError("Python 更新 wheel 的内置版本不一致。")
    if release_info.get("repository_url") != REPOSITORY_URL:
        raise UpdatePreparationError("Python 更新 wheel 指向了非官方仓库。")
    if release_info.get("dependency_contract_sha256") != expected_dependency_contract:
        raise UpdatePreparationError("Python 更新 wheel 的运行依赖契约不一致。")


def update_cache_root() -> Path:
    configured = os.environ.get("VISION_WORKBENCH_UPDATE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "VisionWorkbench" / "updates"
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "vision-workbench" / "updates"


def _console_python_executable() -> Path:
    """Return the environment's console Python even from a GUI launcher."""

    executable = Path(sys.executable).resolve()
    if sys.platform != "win32":
        return executable
    candidate = Path(sys.prefix).resolve() / "python.exe"
    return candidate if candidate.is_file() else executable


def _gui_python_executable() -> Path:
    """Return the environment's no-console Python for restarting the Qt app."""

    executable = Path(sys.executable).resolve()
    if sys.platform != "win32":
        return executable
    candidate = Path(sys.prefix).resolve() / "pythonw.exe"
    return candidate if candidate.is_file() else executable


def _download_verified(
    url: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
    progress: Callable[[int, int | None], None] | None,
    *,
    opener: Callable[..., object],
) -> None:
    request = Request(url, headers={"User-Agent": "Vision-Workbench-Update-Client/1.0"})
    partial = destination.with_name(f"{destination.name}.part")
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with opener(request, timeout=30) as response, partial.open("wb") as stream:
            _validate_final_download_url(response.geturl())
            length_header = response.headers.get("Content-Length")
            response_size = int(length_header) if length_header else None
            if response_size is not None and response_size != expected_size:
                raise UpdatePreparationError("服务器返回的更新大小与发布清单不一致。")
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > MAX_UPDATE_BYTES or downloaded > expected_size:
                    raise UpdatePreparationError("下载内容超过发布清单声明的大小。")
                digest.update(chunk)
                stream.write(chunk)
                if progress is not None:
                    progress(downloaded, expected_size)
    except HTTPError as exc:
        raise UpdatePreparationError(f"更新下载返回 HTTP {exc.code}。") from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise UpdatePreparationError(f"更新下载失败：{exc}") from exc
    finally:
        if partial.exists() and (downloaded != expected_size or digest.hexdigest() != expected_sha256):
            partial.unlink(missing_ok=True)

    if downloaded != expected_size:
        raise UpdatePreparationError("下载文件大小与发布清单不一致。")
    if digest.hexdigest() != expected_sha256:
        raise UpdatePreparationError("下载文件 SHA-256 校验失败。")
    os.replace(partial, destination)


def _validate_final_download_url(value: str) -> None:
    parsed = urlparse(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise UpdatePreparationError("更新下载被重定向到了无效地址。") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        raise UpdatePreparationError("更新下载被重定向到了非允许地址。")


def _matches_sha256(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(DOWNLOAD_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError:
        return False
    return digest.hexdigest() == expected


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
