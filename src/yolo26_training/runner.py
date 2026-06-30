"""Command-line YOLO26 training runner with dataset validation."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from yolo26_training.application import build_default_service
    from yolo26_training.configuration import Yolo26TrainingConfig
    from yolo26_training.domain import TrainingJobConfig
else:
    from .application import build_default_service
    from .configuration import Yolo26TrainingConfig
    from .domain import TrainingJobConfig


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a YOLO dataset and train a YOLO26 model.",
    )
    parser.add_argument("--task", default="detect", choices=["detect", "segment", "semantic"])
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml or dataset directory.")
    parser.add_argument("--model", required=True, help="Path to YOLO26 .pt model.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or CUDA device id.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=None, help="Training output root directory.")
    parser.add_argument("--name", default=None, help="Training run name.")
    parser.add_argument("--resume", action="store_true", help="Resume training.")
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Treat missing label files as warnings instead of errors.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset and print the training command without starting training.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = Yolo26TrainingConfig()
    service = build_default_service(config)

    data_path = Path(args.data).expanduser()
    model_path = Path(args.model).expanduser()
    project_dir = Path(args.project).expanduser() if args.project else config.runs_dir
    run_name = args.name or _default_run_name(data_path, model_path)
    job = TrainingJobConfig(
        task=args.task,
        data_yaml=data_path,
        model_path=model_path,
        project_dir=project_dir,
        run_name=run_name,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        workers=args.workers,
        resume=args.resume,
        allow_missing_labels=args.allow_missing_labels,
    )

    report = service.validate_dataset(
        job.data_yaml,
        task=job.task,
        allow_missing_labels=job.allow_missing_labels,
    )
    print(report.to_text(), flush=True)
    if not report.ok:
        print("\nTraining aborted because dataset validation failed.", file=sys.stderr, flush=True)
        return 2

    if not model_path.exists():
        print(f"\nTraining aborted because model file does not exist: {model_path}", file=sys.stderr, flush=True)
        return 2

    if args.dry_run:
        print("\nDry run passed. Training was not started.", flush=True)
        print("Output root:", project_dir, flush=True)
        print("Run name:", run_name, flush=True)
        return 0

    try:
        service.train(job)
    except Exception as exc:
        print(f"\nTraining failed: {exc}", file=sys.stderr, flush=True)
        return 1

    print("\nTraining finished.", flush=True)
    print("Expected weights:", project_dir / run_name / "weights", flush=True)
    return 0


def _default_run_name(data_path: Path, model_path: Path) -> str:
    dataset_name = data_path.stem if data_path.is_file() else data_path.name
    model_name = model_path.stem
    raw = f"{dataset_name}_{model_name}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return safe or "yolo26_train"


if __name__ == "__main__":
    raise SystemExit(main())
