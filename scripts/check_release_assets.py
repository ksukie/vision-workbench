"""Check release files for the Vision Workbench publishing policy."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models"
MAX_MODEL_BYTES = 100 * 1024 * 1024


def main() -> int:
    oversized = []
    if MODEL_ROOT.exists():
        for path in MODEL_ROOT.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".pt", ".pth", ".onnx", ".engine"}:
                continue
            if path.stat().st_size > MAX_MODEL_BYTES:
                oversized.append(path)

    if not oversized:
        print("Release asset check passed: no model file is larger than 100 MB.")
        return 0

    print("Release asset check failed: these model files are larger than 100 MB:")
    for path in oversized:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"- {path.relative_to(ROOT)} ({size_mb:.2f} MB)")
    print("Move oversized files to local download, Git LFS, or release assets before publishing.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
