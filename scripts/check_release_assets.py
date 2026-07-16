"""Check release files for the Vision Workbench publishing policy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_workbench.troubleshooting import PACKAGING_AND_RELEASE, help_hint

MODEL_ROOT = ROOT / "models"
MAX_MODEL_BYTES = 100 * 1024 * 1024


def is_git_ignored(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", str(path.relative_to(ROOT))],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def main() -> int:
    oversized = []
    if MODEL_ROOT.exists():
        for path in MODEL_ROOT.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".pt", ".pth", ".onnx", ".engine"}:
                continue
            if is_git_ignored(path):
                continue
            if path.stat().st_size > MAX_MODEL_BYTES:
                oversized.append(path)

    if not oversized:
        print("Release asset check passed: no publishable model file is larger than 100 MB.")
        return 0

    print("Release asset check failed: these model files are larger than 100 MB:")
    for path in oversized:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"- {path.relative_to(ROOT)} ({size_mb:.2f} MB)")
    print("Move oversized files to local download, Git LFS, or release assets before publishing.")
    print(help_hint(PACKAGING_AND_RELEASE))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
