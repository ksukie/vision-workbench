"""Training environment diagnostics used by both CLIs and Qt pages."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingEnvironmentReport:
    python_executable: Path
    torch_version: str | None
    torchvision_version: str | None
    accelerator: str
    accelerator_memory_gib: float | None
    output_free_gib: float
    recommended_batch_size: int
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_text(self) -> str:
        memory = (
            f"，显存约 {self.accelerator_memory_gib:.1f} GiB"
            if self.accelerator_memory_gib is not None
            else ""
        )
        parts = [
            f"Python：{self.python_executable}",
            f"Torch：{self.torch_version or '不可用'}",
            f"TorchVision：{self.torchvision_version or '不可用'}",
            f"设备：{self.accelerator}{memory}",
            f"输出磁盘可用：{self.output_free_gib:.1f} GiB",
            f"建议批量：{self.recommended_batch_size}",
        ]
        if self.issues:
            parts.append("问题：" + "；".join(self.issues))
        return " | ".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "python_executable": str(self.python_executable),
            "torch_version": self.torch_version,
            "torchvision_version": self.torchvision_version,
            "accelerator": self.accelerator,
            "accelerator_memory_gib": self.accelerator_memory_gib,
            "output_free_gib": self.output_free_gib,
            "recommended_batch_size": self.recommended_batch_size,
            "issues": list(self.issues),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TrainingEnvironmentReport":
        return cls(
            python_executable=Path(str(payload["python_executable"])),
            torch_version=str(payload["torch_version"]) if payload.get("torch_version") else None,
            torchvision_version=(
                str(payload["torchvision_version"]) if payload.get("torchvision_version") else None
            ),
            accelerator=str(payload["accelerator"]),
            accelerator_memory_gib=(
                float(payload["accelerator_memory_gib"])
                if payload.get("accelerator_memory_gib") is not None
                else None
            ),
            output_free_gib=float(payload["output_free_gib"]),
            recommended_batch_size=int(payload["recommended_batch_size"]),
            issues=tuple(str(item) for item in payload.get("issues", [])),
        )


def inspect_training_environment(
    output_dir: Path,
    *,
    timeout_seconds: float = 45.0,
) -> TrainingEnvironmentReport:
    """Inspect Torch in a bounded subprocess so a broken driver cannot hang the UI."""

    output_path = Path(output_dir).expanduser().resolve()
    command = [
        sys.executable,
        "-m",
        "vision_workbench.runtime_diagnostics",
        "--direct-json",
        str(output_path),
    ]
    environment = os.environ.copy()
    environment.setdefault("PYTHONNOUSERSITE", "1")
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            env=environment,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired:
        return _fallback_report(output_path, f"环境检查超过 {timeout_seconds:.0f} 秒，已停止设备探测。")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return _fallback_report(output_path, f"环境检查进程失败：{detail or completed.returncode}")
    marker = "VW_ENV_REPORT "
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            try:
                return TrainingEnvironmentReport.from_dict(json.loads(line[len(marker) :]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                return _fallback_report(output_path, f"环境检查结果无法解析：{exc}")
    return _fallback_report(output_path, "环境检查进程未返回结果。")


def _inspect_training_environment_direct(output_dir: Path) -> TrainingEnvironmentReport:
    """Inspect the interpreter, Torch stack, accelerator and output disk in this process."""

    issues = []
    torch_version = None
    torchvision_version = None
    accelerator = "CPU"
    memory_gib = None
    recommended_batch = 4
    try:
        import torch
        import torchvision

        torch_version = str(torch.__version__)
        torchvision_version = str(torchvision.__version__)
        torch_path = Path(torch.__file__).resolve()
        prefix = Path(sys.prefix).resolve()
        if not torch_path.is_relative_to(prefix):
            issues.append(f"Torch 来自当前环境之外：{torch_path}")
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            accelerator = f"CUDA / {properties.name}"
            memory_gib = float(properties.total_memory) / (1024 ** 3)
            recommended_batch = 16 if memory_gib >= 10 else 8 if memory_gib >= 6 else 4
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator = "Apple MPS"
            recommended_batch = 8
    except Exception as exc:
        issues.append(f"Torch/TorchVision 导入失败：{type(exc).__name__}: {exc}")

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output_path).free / (1024 ** 3)
    if free_gib < 5:
        issues.append(f"训练输出磁盘空间偏低：仅剩 {free_gib:.1f} GiB")

    return TrainingEnvironmentReport(
        python_executable=Path(sys.executable).resolve(),
        torch_version=torch_version,
        torchvision_version=torchvision_version,
        accelerator=accelerator,
        accelerator_memory_gib=memory_gib,
        output_free_gib=free_gib,
        recommended_batch_size=recommended_batch,
        issues=tuple(issues),
    )


def _fallback_report(output_dir: Path, issue: str) -> TrainingEnvironmentReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output_dir).free / (1024 ** 3)
    return TrainingEnvironmentReport(
        python_executable=Path(sys.executable).resolve(),
        torch_version=None,
        torchvision_version=None,
        accelerator="未知",
        accelerator_memory_gib=None,
        output_free_gib=free_gib,
        recommended_batch_size=4,
        issues=(issue,),
    )


def _main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--direct-json":
        report = _inspect_training_environment_direct(Path(argv[1]))
        print("VW_ENV_REPORT " + json.dumps(report.to_dict(), ensure_ascii=False), flush=True)
        return 0
    print("Usage: python -m vision_workbench.runtime_diagnostics --direct-json OUTPUT_DIR", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
