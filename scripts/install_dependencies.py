"""Install Vision Workbench dependencies with automatic Torch selection.

The plain requirements files cannot detect whether the machine has an NVIDIA
GPU, so this helper chooses the Torch wheel source before invoking pip.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


ROOT = Path(__file__).resolve().parents[1]

PYPI_INDEX_URL = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
PYTORCH_OFFICIAL_BASE = "https://download.pytorch.org/whl"
PYTORCH_MIRROR_BASE = "https://mirrors.aliyun.com/pytorch-wheels"

TORCH_VERSION = "2.12.1"
TORCHVISION_VERSION = "0.27.1"
CUDA_TAG = "cu126"
CPU_TAG = "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install project dependencies and choose CUDA or CPU Torch automatically."
    )
    parser.add_argument(
        "target",
        nargs="?",
        choices=("base", "classification", "yolo26", "all"),
        default="all",
        help="Dependency group to install. Default: all.",
    )
    parser.add_argument(
        "--torch",
        choices=("auto", "cuda", "cpu"),
        default=os.environ.get("VISION_WORKBENCH_TORCH", "auto"),
        help="Torch build selection. Default: auto, or VISION_WORKBENCH_TORCH.",
    )
    parser.add_argument(
        "--skip-project",
        action="store_true",
        help="Do not run 'pip install -e .' for the main project.",
    )
    args = parser.parse_args()

    install_base(skip_project=args.skip_project)

    if args.target in {"classification", "yolo26", "all"}:
        selected_tag = install_torch(args.torch)
        verify_torch(selected_tag)

    if args.target in {"yolo26", "all"}:
        install_yolo26()

    print("\nDone.")
    return 0


def install_base(skip_project: bool) -> None:
    pip_install("-r", "requirements.txt")
    if not skip_project:
        pip_install("--index-url", PYPI_INDEX_URL, "-e", ".")


def install_torch(requested: str) -> str:
    tag = select_torch_tag(requested)
    packages = [
        f"torch=={TORCH_VERSION}+{tag}",
        f"torchvision=={TORCHVISION_VERSION}+{tag}",
    ]

    sources = [
        (
            f"PyTorch official {tag}",
            ["--index-url", f"{PYTORCH_OFFICIAL_BASE}/{tag}"],
        ),
        (
            f"Aliyun PyTorch mirror {tag}",
            ["--no-index", "--find-links", f"{PYTORCH_MIRROR_BASE}/{tag}/"],
        ),
    ]

    last_error: Optional[subprocess.CalledProcessError] = None
    for label, source_args in sources:
        print(f"\nInstalling Torch from {label}...")
        try:
            pip_install(*source_args, *packages)
            return tag
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"Torch install failed from {label}; trying the next source.")

    raise SystemExit(last_error.returncode if last_error else 1)


def select_torch_tag(requested: str) -> str:
    if requested == "cuda":
        return CUDA_TAG
    if requested == "cpu":
        return CPU_TAG
    return CUDA_TAG if has_nvidia_gpu() else CPU_TAG


def has_nvidia_gpu() -> bool:
    candidates = [
        shutil.which("nvidia-smi"),
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            result = subprocess.run(
                [candidate, "-L"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        output = f"{result.stdout}\n{result.stderr}".lower()
        if result.returncode == 0 and ("gpu" in output or "nvidia" in output):
            print("NVIDIA GPU detected; selecting CUDA Torch.")
            return True
    print("No NVIDIA GPU detected; selecting CPU Torch.")
    return False


def install_yolo26() -> None:
    pip_install("--index-url", PYPI_INDEX_URL, "-e", "./third_party/yolo26_source")


def verify_torch(expected_tag: str) -> None:
    code = (
        "import torch, torchvision; "
        "print('torch:', torch.__version__); "
        "print('torchvision:', torchvision.__version__); "
        "print('torch.version.cuda:', torch.version.cuda); "
        "print('torch.cuda.is_available():', torch.cuda.is_available())"
    )
    subprocess.check_call([sys.executable, "-c", code], cwd=ROOT, env=child_env())
    if expected_tag.startswith("cu"):
        print("CUDA Torch was installed. If cuda.is_available() is False, update the NVIDIA driver.")


def pip_install(*args: str) -> None:
    command = [sys.executable, "-m", "pip", "install", *args]
    print("\n+ " + " ".join(command))
    subprocess.check_call(command, cwd=ROOT, env=child_env())


def child_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    return env


if __name__ == "__main__":
    raise SystemExit(main())
