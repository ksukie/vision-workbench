"""Authoritative runtime and release identity for Vision Workbench."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from importlib import metadata, resources
from pathlib import Path
from urllib.parse import unquote, urlparse


DISTRIBUTION_NAME = "vision-workbench"
REPOSITORY_URL = "https://github.com/ksukie/vision-workbench"
RELEASES_URL = f"{REPOSITORY_URL}/releases"
LATEST_RELEASE_URL = f"{RELEASES_URL}/latest"
LATEST_MANIFEST_URL = f"{LATEST_RELEASE_URL}/download/update-manifest.json"
_STABLE_VERSION_PATTERN = re.compile(r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$")
_RUNTIME_DEPENDENCY_FILES = (
    "requirements.txt",
    "requirements-classification.txt",
    "requirements-yolo26.txt",
    "third_party/yolo26_source/pyproject.toml",
)
_NON_RUNTIME_OPTIONAL_GROUPS = {"test"}


@dataclass(frozen=True)
class RuntimeVersionInfo:
    """Version information bound to the code that is actually running."""

    version: str
    updated_at: str
    repository_url: str
    install_mode: str
    source_root: Path | None = None
    dependency_contract_sha256: str | None = None


def stable_version_tuple(value: str) -> tuple[int, int, int]:
    """Parse the project's supported stable ``major.minor.patch`` format."""

    normalized = value.strip().removeprefix("v")
    match = _STABLE_VERSION_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(f"不支持的版本格式：{value!r}")
    return tuple(int(match.group(name)) for name in ("major", "minor", "patch"))


@lru_cache(maxsize=1)
def current_version_info() -> RuntimeVersionInfo:
    """Return identity for the running source, wheel, or frozen executable."""

    release_info = _read_bundled_release_info()
    frozen = bool(getattr(sys, "frozen", False))
    source_root = None if frozen else _editable_source_root()
    bundled_version = _required_release_value(release_info, "version")
    stable_version_tuple(bundled_version)

    if frozen:
        mode = "single-file"
        version = bundled_version
    elif source_root is not None:
        mode = "editable"
        version = _read_project_version(source_root)
    else:
        mode = "wheel"
        version = _running_distribution_version()

    stable_version_tuple(version)
    if version != bundled_version:
        raise RuntimeError(
            f"运行版本 {version} 与内置发布信息 {bundled_version} 不一致；请重新构建或刷新安装元数据。"
        )
    bundled_dependency_contract = _required_release_value(release_info, "dependency_contract_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", bundled_dependency_contract) is None:
        raise RuntimeError("内置运行依赖契约指纹无效。")
    dependency_contract = (
        project_dependency_contract(source_root) if source_root is not None else bundled_dependency_contract
    )
    if source_root is not None and dependency_contract != bundled_dependency_contract:
        raise RuntimeError("当前源码的运行依赖契约与内置发布信息不一致；请同步 release_info.json。")
    updated_at = _required_release_value(release_info, "updated_at")
    try:
        if date.fromisoformat(updated_at).isoformat() != updated_at:
            raise ValueError
    except ValueError as exc:
        raise RuntimeError("内置版本更新时间必须是 YYYY-MM-DD 格式。") from exc
    repository_url = _required_release_value(release_info, "repository_url")
    if repository_url != REPOSITORY_URL:
        raise RuntimeError("内置版本信息指向了非官方仓库。")
    return RuntimeVersionInfo(
        version=version,
        updated_at=updated_at,
        repository_url=repository_url,
        install_mode=mode,
        source_root=source_root,
        dependency_contract_sha256=dependency_contract,
    )


def source_archive_url(version: str | None = None) -> str:
    """Return the canonical source archive URL for a stable release."""

    resolved_version = version or current_version_info().version
    stable_version_tuple(resolved_version)
    return f"{REPOSITORY_URL}/archive/refs/tags/v{resolved_version}.zip"


def project_dependency_contract(root: Path) -> str:
    """Hash normalized runtime dependency declarations for safe no-deps updates."""

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
        import tomli as tomllib

    try:
        payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        project = payload["project"]
        dependencies = project["dependencies"]
        optional_dependencies = project.get("optional-dependencies", {})
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"无法从 {root / 'pyproject.toml'} 读取运行依赖契约。") from exc
    if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
        raise RuntimeError("pyproject.toml 中的 project.dependencies 无效。")
    if not isinstance(optional_dependencies, dict):
        raise RuntimeError("pyproject.toml 中的可选依赖契约无效。")
    normalized_optional = {}
    for group, values in optional_dependencies.items():
        if group in _NON_RUNTIME_OPTIONAL_GROUPS:
            continue
        if not isinstance(group, str) or not isinstance(values, list) or any(
            not isinstance(item, str) for item in values
        ):
            raise RuntimeError("pyproject.toml 中的可选依赖契约无效。")
        normalized_optional[group] = sorted(item.strip() for item in values)

    declared_files = {}
    for relative_path in _RUNTIME_DEPENDENCY_FILES:
        path = root / relative_path
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"无法读取运行依赖声明：{path}") from exc
        normalized_lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        declared_files[relative_path] = "\n".join(normalized_lines).rstrip() + "\n"

    contract = {
        "project_dependencies": sorted(item.strip() for item in dependencies),
        "project_optional_dependencies": normalized_optional,
        "runtime_dependency_files": declared_files,
    }
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_bundled_release_info() -> dict[str, object]:
    try:
        text = resources.files("vision_workbench").joinpath("release_info.json").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise RuntimeError("缺少内置版本信息 release_info.json。") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("内置版本信息不是有效 JSON。") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("不支持的内置版本信息格式。")
    return payload


def _required_release_value(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"内置版本信息缺少 {key}。")
    return value.strip()


def _editable_source_root() -> Path | None:
    current_file = Path(__file__).resolve()
    candidate = current_file.parents[2]
    if _source_file_for_root(candidate) == current_file:
        return candidate

    try:
        distribution = metadata.distribution(DISTRIBUTION_NAME)
        direct_url_text = distribution.read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        direct_url_text = None

    if direct_url_text:
        try:
            payload = json.loads(direct_url_text)
        except json.JSONDecodeError:
            payload = {}
        dir_info = payload.get("dir_info")
        if isinstance(dir_info, dict) and dir_info.get("editable") is True:
            path = _path_from_file_url(payload.get("url"))
            if path is not None and _source_file_for_root(path) == current_file:
                return path
    return None


def _source_file_for_root(root: Path) -> Path | None:
    if not (root / "pyproject.toml").is_file():
        return None
    candidate = root / "src" / "vision_workbench" / "versioning.py"
    try:
        return candidate.resolve() if candidate.is_file() else None
    except OSError:
        return None


def _running_distribution_version() -> str:
    current_file = Path(__file__).resolve()
    for distribution in metadata.distributions(name=DISTRIBUTION_NAME):
        try:
            candidate = Path(distribution.locate_file("vision_workbench/versioning.py")).resolve()
        except (OSError, TypeError, ValueError):
            continue
        if candidate == current_file:
            return str(distribution.version)
    raise RuntimeError("找不到与当前运行代码匹配的 vision-workbench 安装元数据。")


def _path_from_file_url(value: object) -> Path | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    if parsed.scheme.lower() != "file":
        return None
    path_text = unquote(parsed.path)
    if os.name == "nt" and re.match(r"^/[A-Za-z]:/", path_text):
        path_text = path_text[1:]
    return Path(path_text).resolve()


def _read_project_version(root: Path) -> str:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
        import tomli as tomllib

    try:
        payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        version = payload["project"]["version"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"无法从 {root / 'pyproject.toml'} 读取项目版本。") from exc
    if not isinstance(version, str):
        raise RuntimeError("pyproject.toml 中的项目版本不是字符串。")
    return version.strip()
