"""Validate the Vision Workbench 1.0+ version and release contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/ksukie/Vision-WorkBench"
UPDATE_REPOSITORY_URL = "https://github.com/ksukie/vision-workbench"


def _load_dependency_contract_function():
    module_name = "_vision_workbench_contract_versioning"
    path = PROJECT_ROOT / "src" / "vision_workbench" / "versioning.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load version contract module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.project_dependency_contract


project_dependency_contract = _load_dependency_contract_function()


def project_metadata() -> dict[str, object]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
        import tomli as tomllib

    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def project_version() -> str:
    return str(project_metadata()["project"]["version"])


def contract_errors(*, release: bool = False) -> list[str]:
    metadata_payload = project_metadata()
    project = metadata_payload["project"]
    version = str(project["version"])
    errors = []
    if project.get("name") != "vision-workbench":
        errors.append("pyproject.toml project name must be vision-workbench")
    if re.fullmatch(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", version) is None:
        errors.append(f"pyproject.toml version is not stable major.minor.patch: {version}")
    if project.get("license") != "AGPL-3.0-only":
        errors.append("pyproject.toml must use the AGPL-3.0-only SPDX license expression")
    classifiers = project.get("classifiers")
    if not isinstance(classifiers, list) or "Development Status :: 5 - Production/Stable" not in classifiers:
        errors.append("pyproject.toml must declare the Production/Stable classifier")
    urls = project.get("urls")
    expected_project_urls = {
        "Homepage": REPOSITORY_URL,
        "Repository": REPOSITORY_URL,
        "Documentation": f"{REPOSITORY_URL}/tree/main/docs",
        "Issues": f"{REPOSITORY_URL}/issues",
        "Changelog": f"{REPOSITORY_URL}/blob/main/CHANGELOG.md",
    }
    if not isinstance(urls, dict):
        errors.append("pyproject.toml project.urls is missing")
    else:
        for label, expected_url in expected_project_urls.items():
            if urls.get(label) != expected_url:
                errors.append(f"pyproject.toml project.urls.{label} is not canonical")
    package_data = metadata_payload.get("tool", {}).get("setuptools", {}).get("package-data", {})
    vision_workbench_data = package_data.get("vision_workbench", [])
    for required_pattern in ("assets/*.png", "assets/*.ico", "release_info.json"):
        if required_pattern not in vision_workbench_data:
            errors.append(
                "pyproject.toml package data must include "
                f"vision_workbench/{required_pattern}"
            )
    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    required_sdist_declarations = (
        "requirements.txt",
        "requirements-classification.txt",
        "requirements-yolo26.txt",
        "third_party/yolo26_source/pyproject.toml",
    )
    for declaration in required_sdist_declarations:
        if f"include {declaration}" not in manifest:
            errors.append(f"MANIFEST.in must include runtime dependency declaration: {declaration}")

    release_info = json.loads(
        (PROJECT_ROOT / "src" / "vision_workbench" / "release_info.json").read_text(encoding="utf-8")
    )
    if release_info.get("version") != version:
        errors.append(f"release_info.json version {release_info.get('version')} != {version}")
    if release_info.get("schema_version") != 1:
        errors.append("release_info.json schema_version must be 1")
    if release_info.get("repository_url") != UPDATE_REPOSITORY_URL:
        errors.append("release_info.json repository_url is not the official repository")

    icon_png = PROJECT_ROOT / "src" / "vision_workbench" / "assets" / "vision_workbench_icon.png"
    icon_ico = PROJECT_ROOT / "src" / "vision_workbench" / "assets" / "vision_workbench_icon.ico"
    if not icon_png.is_file() or not icon_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        errors.append("the bundled application PNG icon is missing or invalid")
    if not icon_ico.is_file() or not icon_ico.read_bytes().startswith(b"\x00\x00\x01\x00"):
        errors.append("the Windows application ICO is missing or invalid")
    dependency_contract = project_dependency_contract(PROJECT_ROOT)
    if release_info.get("dependency_contract_sha256") != dependency_contract:
        errors.append(
            "release_info.json runtime dependency contract does not match declarations; "
            f"expected {dependency_contract}"
        )
    updated_at = release_info.get("updated_at")
    try:
        if not isinstance(updated_at, str) or date.fromisoformat(updated_at).isoformat() != updated_at:
            raise ValueError
    except ValueError:
        errors.append("release_info.json updated_at must be a valid YYYY-MM-DD date")

    citation = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if re.search(rf"(?m)^version:\s*{re.escape(version)}\s*$", citation) is None:
        errors.append("CITATION.cff version does not match pyproject.toml")
    if re.search(r"(?m)^license:\s*AGPL-3\.0-only\s*$", citation) is None:
        errors.append("CITATION.cff must use the AGPL-3.0-only SPDX identifier")
    if f'repository-code: "{REPOSITORY_URL}"' not in citation:
        errors.append("CITATION.cff repository-code is not canonical")

    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    first_heading = re.search(r"(?m)^## \[([^]]+)] - (\d{4}-\d{2}-\d{2})$", changelog)
    if first_heading is None or first_heading.group(1) != version:
        errors.append("the first CHANGELOG.md release does not match pyproject.toml")
    elif updated_at != first_heading.group(2):
        errors.append("release_info.json updated_at does not match the first changelog release date")
    current_headings = [
        (heading_version, heading_date)
        for heading_version, heading_date in re.findall(
            r"(?m)^## \[([^]]+)] - (\d{4}-\d{2}-\d{2})$",
            changelog,
        )
        if heading_version == version
    ]
    if current_headings != [(version, updated_at), (version, updated_at)]:
        errors.append("CHANGELOG.md must contain matching English and Chinese current release headings")
    if re.search(
        rf"(?m)^date-released:\s*{re.escape(str(updated_at))}\s*$",
        citation,
    ) is None:
        errors.append("CITATION.cff date-released does not match release_info.json")

    expected_wheel = f"vision_workbench-{version}-py3-none-any.whl"
    for name in ("README.md", "README.en.md"):
        readme = (PROJECT_ROOT / name).read_text(encoding="utf-8")
        if f"{REPOSITORY_URL}/releases" not in readme:
            errors.append(f"{name} does not reference the canonical GitHub Releases URL")
        if expected_wheel not in readme:
            errors.append(f"{name} does not reference {expected_wheel}")
        wheel_versions = set(re.findall(r"vision_workbench-(\d+\.\d+\.\d+)-py3-none-any\.whl", readme))
        if wheel_versions != {version}:
            errors.append(f"{name} contains stale or ambiguous wheel versions: {sorted(wheel_versions)}")

    stale_versions = []
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"__version__\s*=\s*['\"]\d+\.\d+\.\d+['\"]", text):
            stale_versions.append(path.relative_to(PROJECT_ROOT).as_posix())
    if stale_versions:
        errors.append("hard-coded package __version__ values remain: " + ", ".join(stale_versions))

    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    if "datasets/" not in {line.strip() for line in gitignore}:
        errors.append(".gitignore must protect the local datasets/ directory")

    exe_build_script = (PROJECT_ROOT / "scripts" / "build_windows_base_exe.py").read_text(encoding="utf-8")
    for build_only_module in ('"pkg_resources"', '"setuptools"'):
        if build_only_module not in exe_build_script:
            errors.append(f"Windows EXE must exclude build-only module {build_only_module}")
    if '"--collect-data"' in exe_build_script:
        errors.append("Windows EXE package data must not depend on an installed project package")
    for required_data in (
        '"vision_workbench/release_info.json"',
        '"vision_workbench/assets"',
        '"panorama_reconstruction/assets"',
        '"vision_workbench_icon.ico"',
        '"--icon"',
        '"--add-data"',
    ):
        if required_data not in exe_build_script:
            errors.append(f"Windows EXE is missing explicit package data: {required_data}")

    release_workflow = PROJECT_ROOT / ".github" / "workflows" / "release.yml"
    if not release_workflow.is_file():
        errors.append("the tagged release-draft workflow is missing")
    else:
        workflow_text = release_workflow.read_text(encoding="utf-8")
        required_steps = (
            "python -W error -m compileall",
            "check_version_contract.py --release",
            "generate_update_manifest.py",
            "libegl1",
            'VISION_WORKBENCH_REQUIRE_QT_TESTS: "1"',
            "--vision-workbench-self-test",
            "--report",
            "Vision-Workbench-win-x64.exe",
            "--draft",
        )
        for required_step in required_steps:
            if required_step not in workflow_text:
                errors.append(f"release workflow is missing required gate: {required_step}")
        checkout_count = workflow_text.count("uses: actions/checkout@v4")
        explicit_tag_ref_count = workflow_text.count("ref: ${{ github.ref }}")
        if checkout_count == 0 or explicit_tag_ref_count != checkout_count:
            errors.append("every release checkout must explicitly preserve the triggering tag ref")

    ci_workflow = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    if not ci_workflow.is_file():
        errors.append("the cross-platform CI workflow is missing")
    else:
        ci_text = ci_workflow.read_text(encoding="utf-8")
        for required_step in (
            "libegl1",
            'VISION_WORKBENCH_REQUIRE_QT_TESTS: "1"',
            "vision_workbench.self_test --expected-mode editable --qt",
            "python -W error -m compileall",
        ):
            if required_step not in ci_text:
                errors.append(f"CI workflow is missing required Qt gate: {required_step}")

    if release:
        expected_tag = f"v{version}"
        tag = _git("describe", "--tags", "--exact-match", "HEAD")
        if tag != expected_tag:
            errors.append(
                f"HEAD tag {tag or '<none>'} != {expected_tag} from pyproject.toml; "
                "commit the synchronized release version before creating the annotated tag"
            )
        elif _git("cat-file", "-t", expected_tag) != "tag":
            errors.append(f"release tag {expected_tag} must be an annotated tag object")
        if _git("status", "--porcelain"):
            errors.append("release worktree is not clean")
    return errors


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
    parser.add_argument("--release", action="store_true", help="also require a clean exact release tag")
    args = parser.parse_args()
    errors = contract_errors(release=args.release)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Version contract OK: {project_version()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
