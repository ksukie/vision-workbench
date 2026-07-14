"""Build the single-file Windows EXE without deep-learning dependencies."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import PyInstaller.__main__


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEEP_LEARNING_MODULES = (
    "image_classification",
    "torch",
    "torchvision",
    "ultralytics",
    "yolo26_detection",
    "yolo26_segmentation",
    "yolo26_training",
)
CONDA_RUNTIME_DLLS = (
    "ffi.dll",
    "libbz2.dll",
    "libcrypto-3-x64.dll",
    "libexpat.dll",
    "liblzma.dll",
    "libssl-3-x64.dll",
    "tcl86t.dll",
    "tk86t.dll",
)


def project_version() -> str:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def add_conda_runtime_dlls(arguments: list[str]) -> None:
    """Bundle Conda DLLs needed by Python extension modules when available."""

    runtime_dir = Path(sys.base_prefix) / "Library" / "bin"
    for name in CONDA_RUNTIME_DLLS:
        path = runtime_dir / name
        if path.is_file():
            arguments.extend(("--add-binary", f"{path}{os.pathsep}."))


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("The base EXE must be built on Windows.")

    version = project_version()
    name = f"Vision-Workbench-{version}-win-x64"
    work_path = PROJECT_ROOT / "build" / "windows-base-exe"
    arguments = [
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--noupx",
        "--name",
        name,
        "--paths",
        str(PROJECT_ROOT / "src"),
        "--distpath",
        str(PROJECT_ROOT / "dist"),
        "--workpath",
        str(work_path),
        "--specpath",
        str(work_path),
        "--collect-data",
        "vision_workbench",
        "--collect-data",
        "panorama_reconstruction",
    ]
    for module in DEEP_LEARNING_MODULES:
        arguments.extend(("--exclude-module", module))
    add_conda_runtime_dlls(arguments)
    arguments.append(str(PROJECT_ROOT / "scripts" / "windows_base_exe.py"))

    PyInstaller.__main__.run(arguments)
    print(PROJECT_ROOT / "dist" / f"{name}.exe")


if __name__ == "__main__":
    main()
