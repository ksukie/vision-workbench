"""Build the single-file Windows EXE without deep-learning dependencies."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import PyInstaller.__main__


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPLICATION_ICON = (
    PROJECT_ROOT / "src" / "vision_workbench" / "assets" / "vision_workbench_icon.ico"
)
EXCLUDED_MODULES = (
    "image_classification",
    "pkg_resources",
    "setuptools",
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
PACKAGE_DATA = (
    ("vision_workbench/release_info.json", "vision_workbench"),
    ("vision_workbench/assets", "vision_workbench/assets"),
    ("panorama_reconstruction/assets", "panorama_reconstruction/assets"),
)


def add_conda_runtime_dlls(arguments: list[str]) -> None:
    """Bundle Conda DLLs needed by Python extension modules when available."""

    runtime_dir = Path(sys.base_prefix) / "Library" / "bin"
    for name in CONDA_RUNTIME_DLLS:
        path = runtime_dir / name
        if path.is_file():
            arguments.extend(("--add-binary", f"{path}{os.pathsep}."))


def add_package_data(arguments: list[str]) -> None:
    """Bundle required source data without relying on an installed package."""

    source_root = PROJECT_ROOT / "src"
    for relative_source, destination in PACKAGE_DATA:
        source = source_root / relative_source
        if not source.exists():
            raise FileNotFoundError(f"Required EXE package data is missing: {source}")
        arguments.extend(("--add-data", f"{source}{os.pathsep}{destination}"))


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("The base EXE must be built on Windows.")

    name = "Vision-Workbench-win-x64"
    work_path = PROJECT_ROOT / "build" / "windows-base-exe"
    if not APPLICATION_ICON.is_file():
        raise FileNotFoundError(f"Required Windows application icon is missing: {APPLICATION_ICON}")
    arguments = [
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--noupx",
        "--name",
        name,
        "--icon",
        str(APPLICATION_ICON),
        "--paths",
        str(PROJECT_ROOT / "src"),
        "--distpath",
        str(PROJECT_ROOT / "dist"),
        "--workpath",
        str(work_path),
        "--specpath",
        str(work_path),
    ]
    for module in EXCLUDED_MODULES:
        arguments.extend(("--exclude-module", module))
    add_package_data(arguments)
    add_conda_runtime_dlls(arguments)
    arguments.append(str(PROJECT_ROOT / "scripts" / "windows_base_exe.py"))

    PyInstaller.__main__.run(arguments)
    print(PROJECT_ROOT / "dist" / f"{name}.exe")


if __name__ == "__main__":
    main()
