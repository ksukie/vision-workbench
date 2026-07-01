"""Install Vision Workbench dependencies with automatic Torch selection.

The plain requirements files cannot detect whether the machine has an NVIDIA
GPU, so this helper chooses the Torch wheel source before invoking pip.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_workbench.troubleshooting import DEEP_LEARNING_DEPENDENCIES, ENVIRONMENT, with_help

PYPI_INDEX_URL = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
PYTORCH_OFFICIAL_BASE = "https://download.pytorch.org/whl"
PYTORCH_MIRROR_BASE = "https://mirrors.aliyun.com/pytorch-wheels"

TORCH_VERSION = "2.12.1"
TORCHVISION_VERSION = "0.27.1"
CUDA_TAG = "cu126"
CPU_TAG = "cpu"
DEFAULT_TAG = "default"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install project dependencies and choose CUDA or CPU Torch automatically."
    )
    parser.add_argument(
        "target",
        nargs="?",
        choices=("base", "classification", "yolo26", "all", "doctor"),
        default="all",
        help="Dependency group to install, or 'doctor' to check Torch. Default: all.",
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

    if args.target == "doctor":
        diagnose_torch(args.torch)
        return 0

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
    tag, packages, sources = torch_install_plan(requested)

    last_error: Optional[subprocess.CalledProcessError] = None
    for label, source_args in sources:
        print(f"\nInstalling Torch from {label}...")
        try:
            pip_install(*source_args, *packages, help_category=DEEP_LEARNING_DEPENDENCIES)
            return tag
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"Torch install failed from {label}; trying the next source.")

    print(with_help("Torch install failed from every configured source.", DEEP_LEARNING_DEPENDENCIES))
    raise SystemExit(last_error.returncode if last_error else 1)


def torch_install_plan(requested: str) -> Tuple[str, List[str], List[Tuple[str, List[str]]]]:
    tag = select_torch_tag(requested)
    if tag == DEFAULT_TAG:
        packages = [
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
        ]
        return tag, packages, [(f"Tsinghua PyPI {tag}", ["--index-url", PYPI_INDEX_URL])]

    packages = [
        f"torch=={TORCH_VERSION}+{tag}",
        f"torchvision=={TORCHVISION_VERSION}+{tag}",
    ]
    return tag, packages, [
        (
            f"PyTorch official {tag}",
            ["--index-url", f"{PYTORCH_OFFICIAL_BASE}/{tag}"],
        ),
        (
            f"Aliyun PyTorch mirror {tag}",
            ["--no-index", "--find-links", f"{PYTORCH_MIRROR_BASE}/{tag}/"],
        ),
    ]


def select_torch_tag(requested: str) -> str:
    if requested == "cuda":
        if not supports_pytorch_cuda_wheels():
            raise SystemExit(
                with_help(
                    "CUDA Torch pip wheels are only selected automatically on Windows/Linux x86_64. "
                    "Use --torch cpu or install the platform-specific Torch build manually.",
                    DEEP_LEARNING_DEPENDENCIES,
                )
            )
        return CUDA_TAG
    if requested == "cpu":
        return CPU_TAG if supports_pytorch_tagged_wheels() else DEFAULT_TAG
    if supports_pytorch_cuda_wheels() and has_nvidia_gpu():
        return CUDA_TAG
    if supports_pytorch_tagged_wheels():
        return CPU_TAG
    print("Using the default PyPI Torch build for this platform.")
    return DEFAULT_TAG


def supports_pytorch_cuda_wheels() -> bool:
    system = platform.system()
    machine = platform.machine().lower()
    return system in {"Windows", "Linux"} and machine in {"amd64", "x86_64"}


def supports_pytorch_tagged_wheels() -> bool:
    system = platform.system()
    machine = platform.machine().lower()
    return system in {"Windows", "Linux"} and machine in {"amd64", "x86_64"}


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
    pip_install("--index-url", PYPI_INDEX_URL, "-e", "./third_party/yolo26_source", help_category=DEEP_LEARNING_DEPENDENCIES)


def diagnose_torch(requested: str) -> None:
    expected_tag = select_torch_tag(requested)
    info = inspect_torch()
    print("\nTorch environment check")
    print(f"Expected build: {expected_tag}")
    for key in ("torch", "torchvision", "torch.version.cuda", "torch.cuda.is_available"):
        print(f"{key}: {info.get(key, '--')}")

    error = info.get("error")
    if error:
        print(with_help(f"Import error: {error}", DEEP_LEARNING_DEPENDENCIES))

    if torch_needs_reinstall(expected_tag, info):
        answer = input("\nTorch does not match this machine. Reinstall Torch now? [y/N] ").strip().lower()
        if answer in {"y", "yes"}:
            installed_tag = install_torch(requested)
            verify_torch(installed_tag)
        else:
            print("Skipped reinstall.")
        return

    print("\nTorch looks OK.")


def inspect_torch() -> Dict[str, str]:
    code = (
        "import json\n"
        "try:\n"
        "    import torch, torchvision\n"
        "    data = {\n"
        "        'torch': torch.__version__,\n"
        "        'torchvision': torchvision.__version__,\n"
        "        'torch.version.cuda': str(torch.version.cuda),\n"
        "        'torch.cuda.is_available': str(torch.cuda.is_available()),\n"
        "    }\n"
        "except Exception as exc:\n"
        "    data = {'error': f'{type(exc).__name__}: {exc}'}\n"
        "print(json.dumps(data))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    import json

    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": (result.stderr or result.stdout or "unknown error").strip()}


def torch_needs_reinstall(expected_tag: str, info: Dict[str, str]) -> bool:
    if info.get("error"):
        return True

    torch_version = info.get("torch", "")
    cuda_available = info.get("torch.cuda.is_available") == "True"

    if expected_tag.startswith("cu"):
        return f"+{expected_tag}" not in torch_version or not cuda_available
    if expected_tag == CPU_TAG:
        return "+cu" in torch_version and not has_nvidia_gpu()
    return False


def verify_torch(expected_tag: str) -> None:
    code = (
        "import torch, torchvision; "
        "print('torch:', torch.__version__); "
        "print('torchvision:', torchvision.__version__); "
        "print('torch.version.cuda:', torch.version.cuda); "
        "print('torch.cuda.is_available():', torch.cuda.is_available())"
    )
    try:
        subprocess.check_call([sys.executable, "-c", code], cwd=ROOT, env=child_env())
    except subprocess.CalledProcessError as exc:
        print(with_help(f"Torch verification failed with exit code {exc.returncode}.", DEEP_LEARNING_DEPENDENCIES))
        raise
    if expected_tag.startswith("cu"):
        print(
            with_help(
                "CUDA Torch was installed. If cuda.is_available() is False, update the NVIDIA driver.",
                DEEP_LEARNING_DEPENDENCIES,
            )
        )


def pip_install(*args: str, help_category: str = ENVIRONMENT) -> None:
    command = [sys.executable, "-m", "pip", "install", *args]
    print("\n+ " + " ".join(command))
    try:
        subprocess.check_call(command, cwd=ROOT, env=child_env())
    except subprocess.CalledProcessError as exc:
        print(
            with_help(
                f"Dependency command failed with exit code {exc.returncode}: {' '.join(command)}",
                help_category,
            ),
            file=sys.stderr,
        )
        raise


def child_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    return env


if __name__ == "__main__":
    raise SystemExit(main())
