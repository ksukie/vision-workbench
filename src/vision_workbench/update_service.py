"""Release discovery and strict update metadata validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .versioning import (
    LATEST_MANIFEST_URL,
    REPOSITORY_URL,
    UPDATE_REPOSITORY_URL,
    RuntimeVersionInfo,
    stable_version_tuple,
)


GITHUB_API_LATEST_URL = "https://api.github.com/repos/ksukie/Vision-WorkBench/releases/latest"
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_UPDATE_ASSET_BYTES = 2 * 1024 * 1024 * 1024
ALLOWED_DOWNLOAD_HOSTS = {
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class UpdateState(str, Enum):
    UP_TO_DATE = "up-to-date"
    UPDATE_AVAILABLE = "update-available"
    CURRENT_AHEAD = "current-ahead"


class UpdateCheckError(RuntimeError):
    """Raised when update metadata cannot be obtained or trusted."""


@dataclass(frozen=True)
class ReleaseAsset:
    kind: str
    name: str
    url: str
    size: int
    sha256: str | None

    @property
    def installable(self) -> bool:
        return self.sha256 is not None and 0 < self.size <= MAX_UPDATE_ASSET_BYTES


@dataclass(frozen=True)
class ReleaseInfo:
    version: str
    published_at: str
    release_url: str
    tag: str
    commit: str | None
    assets: tuple[ReleaseAsset, ...]
    dependency_contract_sha256: str | None = None


@dataclass(frozen=True)
class UpdateCheckResult:
    state: UpdateState
    current: RuntimeVersionInfo
    latest: ReleaseInfo
    checked_at: str
    compatible_asset: ReleaseAsset | None

    @property
    def dependencies_compatible(self) -> bool:
        return (
            self.current.install_mode == "single-file"
            or (
                self.current.dependency_contract_sha256 is not None
                and self.latest.dependency_contract_sha256 == self.current.dependency_contract_sha256
            )
        )

    @property
    def can_install(self) -> bool:
        return (
            self.state is UpdateState.UPDATE_AVAILABLE
            and self.compatible_asset is not None
            and self.compatible_asset.installable
            and self.dependencies_compatible
        )


class UpdateClient:
    """Fetch the stable release manifest, with the GitHub API as a fallback."""

    def __init__(self, timeout: float = 10.0, opener: Callable[..., object] = urlopen) -> None:
        self.timeout = timeout
        self._opener = opener

    def check(self, current: RuntimeVersionInfo) -> UpdateCheckResult:
        latest = self.fetch_latest()
        current_version = stable_version_tuple(current.version)
        latest_version = stable_version_tuple(latest.version)
        if latest_version > current_version:
            state = UpdateState.UPDATE_AVAILABLE
        elif latest_version == current_version:
            state = UpdateState.UP_TO_DATE
        else:
            state = UpdateState.CURRENT_AHEAD
        return UpdateCheckResult(
            state=state,
            current=current,
            latest=latest,
            checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            compatible_asset=_compatible_asset(latest.assets, current.install_mode),
        )

    def fetch_latest(self) -> ReleaseInfo:
        manifest_error = None
        try:
            return _parse_manifest(self._get_json(LATEST_MANIFEST_URL))
        except UpdateCheckError as exc:
            manifest_error = exc

        try:
            return _parse_github_release(self._get_json(GITHUB_API_LATEST_URL))
        except UpdateCheckError as api_error:
            raise UpdateCheckError(
                f"无法查询最新版本。更新清单：{manifest_error}；GitHub API：{api_error}"
            ) from api_error

    def _get_json(self, url: str) -> dict[str, object]:
        request = Request(
            url,
            headers={
                "Accept": "application/vnd.github+json, application/json",
                "User-Agent": "Vision-Workbench-Update-Client/1.0",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                _validate_metadata_response_url(url, response.geturl())
                length_header = response.headers.get("Content-Length")
                if length_header and int(length_header) > MAX_METADATA_BYTES:
                    raise UpdateCheckError("更新元数据超过大小限制。")
                raw = response.read(MAX_METADATA_BYTES + 1)
        except HTTPError as exc:
            if exc.code == 403:
                raise UpdateCheckError("GitHub 请求受到限流，请稍后重试。") from exc
            raise UpdateCheckError(f"服务器返回 HTTP {exc.code}。") from exc
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise UpdateCheckError(f"网络请求失败：{exc}") from exc
        if len(raw) > MAX_METADATA_BYTES:
            raise UpdateCheckError("更新元数据超过大小限制。")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UpdateCheckError("服务器返回的更新元数据无效。") from exc
        if not isinstance(payload, dict):
            raise UpdateCheckError("服务器返回的更新元数据不是对象。")
        return payload


def _parse_manifest(payload: dict[str, object]) -> ReleaseInfo:
    if payload.get("schema_version") != 1 or payload.get("channel") != "stable":
        raise UpdateCheckError("不支持的更新清单格式或发布通道。")
    version = _stable_version(payload.get("version"))
    tag = _required_text(payload, "tag")
    if tag != f"v{version}":
        raise UpdateCheckError("更新清单的 tag 与版本不一致。")
    repository_url = payload.get("repository_url")
    if repository_url != UPDATE_REPOSITORY_URL:
        raise UpdateCheckError("更新清单指向了非官方仓库。")
    release_url = _official_release_url(_required_text(payload, "release_url"))
    if release_url != f"{REPOSITORY_URL}/releases/tag/{tag}":
        raise UpdateCheckError("更新清单的 Release URL 与版本不一致。")
    published_at = _iso_timestamp(_required_text(payload, "published_at"))
    commit_value = payload.get("commit")
    commit = commit_value.strip() if isinstance(commit_value, str) and commit_value.strip() else None
    if commit is not None and re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise UpdateCheckError("更新清单的 commit 不是完整 Git SHA。")
    raw_assets = payload.get("assets")
    if not isinstance(raw_assets, list):
        raise UpdateCheckError("更新清单缺少资产列表。")
    assets = tuple(_parse_asset(item) for item in raw_assets)
    _validate_assets_for_version(assets, version)
    dependency_contract = _sha256(payload.get("dependency_contract_sha256"))
    if dependency_contract is None:
        raise UpdateCheckError("更新清单缺少运行依赖契约指纹。")
    return ReleaseInfo(version, published_at, release_url, tag, commit, assets, dependency_contract)


def _parse_github_release(payload: dict[str, object]) -> ReleaseInfo:
    if payload.get("draft") is True or payload.get("prerelease") is True:
        raise UpdateCheckError("GitHub 返回的不是稳定正式版本。")
    tag = _required_text(payload, "tag_name")
    version = _stable_version(tag)
    if tag != f"v{version}":
        raise UpdateCheckError("GitHub Release 的 tag 与正式版本格式不一致。")
    release_url = _official_release_url(_required_text(payload, "html_url"))
    if release_url != f"{REPOSITORY_URL}/releases/tag/{tag}":
        raise UpdateCheckError("GitHub Release URL 与版本不一致。")
    published_at = _iso_timestamp(_required_text(payload, "published_at"))
    raw_assets = payload.get("assets")
    if not isinstance(raw_assets, list):
        raise UpdateCheckError("GitHub Release 缺少资产列表。")
    assets = tuple(_parse_github_asset(item) for item in raw_assets)
    _validate_assets_for_version(assets, version)
    target = payload.get("target_commitish")
    commit = target.strip() if isinstance(target, str) and re.fullmatch(r"[0-9a-fA-F]{7,40}", target.strip()) else None
    return ReleaseInfo(version, published_at, release_url, f"v{version}", commit, assets)


def _parse_asset(value: object) -> ReleaseAsset:
    if not isinstance(value, dict):
        raise UpdateCheckError("更新资产条目无效。")
    kind = _required_text(value, "kind")
    name = _required_text(value, "name")
    url = _official_download_url(_required_text(value, "url"))
    size = _nonnegative_int(value.get("size"), "size")
    sha256 = _sha256(value.get("sha256"))
    return ReleaseAsset(kind, name, url, size, sha256)


def _parse_github_asset(value: object) -> ReleaseAsset:
    if not isinstance(value, dict):
        raise UpdateCheckError("GitHub 资产条目无效。")
    name = _required_text(value, "name")
    url = _official_download_url(_required_text(value, "browser_download_url"))
    size = _nonnegative_int(value.get("size"), "size")
    digest = value.get("digest")
    sha256 = None
    if isinstance(digest, str) and digest.startswith("sha256:"):
        sha256 = _sha256(digest.removeprefix("sha256:"))
    return ReleaseAsset(_asset_kind(name), name, url, size, sha256)


def _compatible_asset(assets: tuple[ReleaseAsset, ...], install_mode: str) -> ReleaseAsset | None:
    wanted_kind = "windows-x64-exe" if install_mode == "single-file" else "python-wheel"
    return next((asset for asset in assets if asset.kind == wanted_kind), None)


def _validate_assets_for_version(assets: tuple[ReleaseAsset, ...], version: str) -> None:
    expected_names = {
        "python-wheel": f"vision_workbench-{version}-py3-none-any.whl",
        "windows-x64-exe": "Vision-Workbench-win-x64.exe",
    }
    seen = set()
    for asset in assets:
        expected = expected_names.get(asset.kind)
        if expected is None:
            continue
        if asset.name != expected:
            raise UpdateCheckError(f"更新资产名称与版本不一致：{asset.name}")
        parsed_url = urlparse(asset.url)
        expected_suffix = f"/v{version}/{asset.name}"
        if parsed_url.hostname == "github.com" and not parsed_url.path.endswith(expected_suffix):
            raise UpdateCheckError(f"更新资产 URL 与版本不一致：{asset.name}")
        if asset.kind in seen:
            raise UpdateCheckError(f"更新清单包含重复资产：{asset.kind}")
        seen.add(asset.kind)


def _asset_kind(name: str) -> str:
    lowered = name.lower()
    if lowered == "vision-workbench-win-x64.exe":
        return "windows-x64-exe"
    if lowered.startswith("vision_workbench-") and lowered.endswith("-py3-none-any.whl"):
        return "python-wheel"
    return "other"


def _stable_version(value: object) -> str:
    if not isinstance(value, str):
        raise UpdateCheckError("更新元数据缺少版本号。")
    normalized = value.strip().removeprefix("v")
    try:
        stable_version_tuple(normalized)
    except ValueError as exc:
        raise UpdateCheckError(str(exc)) from exc
    return normalized


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise UpdateCheckError(f"更新元数据缺少 {key}。")
    return value.strip()


def _official_release_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.query
        or parsed.fragment
    ):
        raise UpdateCheckError("Release URL 不是官方 HTTPS 地址。")
    prefix = "/ksukie/vision-workbench/releases/"
    if not parsed.path.casefold().startswith(prefix):
        raise UpdateCheckError("Release URL 不属于官方仓库。")
    return f"{REPOSITORY_URL}/releases/{parsed.path[len(prefix):]}"


def _official_download_url(value: str) -> str:
    parsed = urlparse(value)
    if not _standard_https_host(parsed) or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
        raise UpdateCheckError("更新资产不是允许的 HTTPS 下载地址。")
    if parsed.hostname == "github.com" and not parsed.path.casefold().startswith(
        "/ksukie/vision-workbench/releases/download/"
    ):
        raise UpdateCheckError("更新资产不属于官方仓库。")
    return value


def _validate_metadata_response_url(requested: str, final: str) -> None:
    parsed = urlparse(final)
    if not _standard_https_host(parsed):
        raise UpdateCheckError("更新元数据被重定向到了非 HTTPS 地址。")
    if requested == GITHUB_API_LATEST_URL:
        if parsed.hostname != "api.github.com":
            raise UpdateCheckError("GitHub API 被重定向到了非官方地址。")
        return
    if parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
        raise UpdateCheckError("更新清单被重定向到了非允许地址。")


def _standard_https_host(parsed) -> bool:
    try:
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.hostname is not None
        and parsed.username is None
        and parsed.password is None
        and port in {None, 443}
    )


def _iso_timestamp(value: str) -> str:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise UpdateCheckError("发布时间不是有效 ISO-8601 时间。") from exc
    if parsed.tzinfo is None:
        raise UpdateCheckError("发布时间必须包含时区。")
    return value


def _sha256(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value.lower()) is None:
        raise UpdateCheckError("更新资产的 SHA-256 无效。")
    return value.lower()


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UpdateCheckError(f"更新资产的 {name} 无效。")
    return value
