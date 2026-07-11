"""Create all Vision Workbench quick-start training datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_workbench.sample_data import create_classification_sample_dataset, create_yolo_sample_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create deterministic quick-start training datasets.")
    parser.add_argument(
        "--task",
        choices=("all", "classification", "detect", "segment", "semantic"),
        default="all",
        help="dataset type to create (default: all)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "datasets",
        help="parent directory for generated datasets",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    if args.task in {"all", "classification"}:
        classification = create_classification_sample_dataset(
            output_root / "image_classification_datasets" / "quickstart"
        )
        print(f"Classification: {classification}")
    yolo_tasks = ("detect", "segment", "semantic") if args.task == "all" else (args.task,)
    for task in yolo_tasks:
        if task == "classification":
            continue
        data_yaml = create_yolo_sample_dataset(
            output_root / "yolo26_datasets" / f"quickstart_{task}",
            task,
        )
        print(f"YOLO {task}: {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
