"""Generate the strict stable-update manifest for a GitHub release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep the alias embedded in 1.0.0 so released clients can validate and install
# every later official update after the repository's display-case change.
UPDATE_REPOSITORY_URL = "https://github.com/ksukie/vision-workbench"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_workbench.update_installer import (  # noqa: E402
    MAX_UPDATE_BYTES,
    UpdatePreparationError,
    validate_python_wheel,
)
from vision_workbench.versioning import project_dependency_contract  # noqa: E402


def project_version() -> str:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
        import tomli as tomllib

    payload = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(payload["project"]["version"])


def build_manifest(assets_dir: Path, published_at: str, commit: str) -> dict[str, object]:
    version = project_version()
    try:
        parsed_time = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("published_at must be a valid ISO-8601 timestamp") from exc
    if parsed_time.tzinfo is None:
        raise ValueError("published_at must include a timezone")
    if re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise ValueError("commit must be a full 40-character Git SHA")
    tag = f"v{version}"
    expected_assets = (
        ("windows-x64-exe", "Vision-Workbench-win-x64.exe"),
        ("python-wheel", f"vision_workbench-{version}-py3-none-any.whl"),
    )
    dependency_contract = project_dependency_contract(PROJECT_ROOT)
    assets = []
    for kind, name in expected_assets:
        path = assets_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing required release asset: {path}")
        size = path.stat().st_size
        if size <= 0 or size > MAX_UPDATE_BYTES:
            raise ValueError(f"Release asset size is invalid or exceeds 2 GB: {path}")
        if kind == "python-wheel":
            try:
                validate_python_wheel(path, version, dependency_contract)
            except UpdatePreparationError as exc:
                raise ValueError(str(exc)) from exc
        assets.append(
            {
                "kind": kind,
                "name": name,
                "url": f"{UPDATE_REPOSITORY_URL}/releases/download/{tag}/{quote(name)}",
                "size": size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "channel": "stable",
        "version": version,
        "published_at": published_at,
        "tag": tag,
        "commit": commit,
        "repository_url": UPDATE_REPOSITORY_URL,
        "release_url": f"{UPDATE_REPOSITORY_URL}/releases/tag/{tag}",
        "dependency_contract_sha256": dependency_contract,
        "assets": assets,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_commit() -> str:
    """Return the exact clean tagged commit allowed to produce a manifest."""

    version = project_version()
    expected_tag = f"v{version}"
    tag = _git("describe", "--tags", "--exact-match", "HEAD")
    if tag != expected_tag:
        raise ValueError(f"HEAD tag {tag or '<none>'} does not match {expected_tag}")
    if _git("cat-file", "-t", expected_tag) != "tag":
        raise ValueError(f"release tag {expected_tag} must be annotated")
    if _git("status", "--porcelain"):
        raise ValueError("release worktree must be clean before generating a manifest")
    commit = _git("rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise ValueError("could not resolve the full release commit SHA")
    return commit


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=PROJECT_ROOT / "dist")
    parser.add_argument("--published-at", required=True, help="UTC ISO-8601 release timestamp")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "dist" / "update-manifest.json")
    args = parser.parse_args()

    try:
        commit = release_commit()
        manifest = build_manifest(args.assets_dir.resolve(), args.published_at, commit)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
