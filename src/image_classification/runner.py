"""Command-line image classification training runner."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_workbench.runtime_security import (
    configure_isolated_python_environment,
    configure_restricted_model_loading,
    validate_run_name,
)

# Apply isolation before importing PIL, NumPy, Torch adapters, or application services.
configure_isolated_python_environment()
configure_restricted_model_loading()

if __package__ in (None, ""):
    from image_classification.application import build_default_service
    from image_classification.configuration import ImageClassificationConfig
    from image_classification.domain import ClassificationTrainingConfig
else:
    from .application import build_default_service
    from .configuration import ImageClassificationConfig
    from .domain import ClassificationTrainingConfig

from vision_workbench.troubleshooting import DATASETS_AND_TRAINING, help_hint, with_help


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a folder classification dataset and train a classifier.",
    )
    parser.add_argument("--data", required=True, help="Dataset root with train/ and val/ folders.")
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=["resnet18", "mobilenet_v3_small"],
        help="Classification backbone.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or CUDA device id.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--project", default=None, help="Training output root directory.")
    parser.add_argument("--name", default=None, help="Training run name.")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train without TorchVision pretrained weights.",
    )
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="Train the whole network instead of only the classification head.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset and print configuration without starting training.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    configure_isolated_python_environment()
    args = parse_args(argv)
    config = ImageClassificationConfig()
    service = build_default_service(config)

    dataset_dir = Path(args.data).expanduser()
    project_dir = Path(args.project).expanduser() if args.project else config.runs_dir
    try:
        run_name = validate_run_name(
            args.name or _default_run_name(dataset_dir, args.model),
            field_name="training run name",
        )
    except ValueError as exc:
        print(with_help(f"Invalid training run name: {exc}", DATASETS_AND_TRAINING), file=sys.stderr)
        return 2
    job = ClassificationTrainingConfig(
        model_name=args.model,
        dataset_dir=dataset_dir,
        output_dir=project_dir,
        run_name=run_name,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        learning_rate=args.lr,
        workers=args.workers,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.unfreeze,
    )

    report = service.validate_dataset(dataset_dir)
    print(report.to_text(), flush=True)
    if not report.ok:
        print("\nTraining aborted because dataset validation failed.", file=sys.stderr, flush=True)
        print(help_hint(DATASETS_AND_TRAINING), file=sys.stderr, flush=True)
        return 2

    if args.dry_run:
        print("\nDry run passed. Training was not started.", flush=True)
        print("Model:", job.model_name, flush=True)
        print("Output root:", job.output_dir, flush=True)
        print("Run name:", job.run_name, flush=True)
        return 0

    try:
        best_path = service.train(job, progress_callback=_print_progress)
    except Exception as exc:
        print(with_help(f"\nTraining failed: {exc}", DATASETS_AND_TRAINING), file=sys.stderr, flush=True)
        return 1

    print("\nTraining finished.", flush=True)
    print("Best model:", best_path, flush=True)
    return 0


def _default_run_name(dataset_dir: Path, model_name: str) -> str:
    raw = f"{dataset_dir.name}_{model_name}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return safe or "classification_train"


def _print_progress(metrics: dict[str, float | int]) -> None:
    print("VW_METRIC " + json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
