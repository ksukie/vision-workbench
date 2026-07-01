"""Simple image classification training script.

Change the variables below, then run:

python src\image_classification\train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_classification.api import create_image_classification_service
    from image_classification.configuration import ImageClassificationConfig
    from image_classification.domain import ClassificationTrainingConfig
else:
    from .api import create_image_classification_service
    from .configuration import ImageClassificationConfig
    from .domain import ClassificationTrainingConfig

from vision_workbench.troubleshooting import DATASETS_AND_TRAINING, with_help


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Choose: "resnet18" or "mobilenet_v3_small"
MODEL_NAME = "resnet18"

# Dataset must contain train/ and val/ folders.
DATASET_DIR = r"C:\path\to\classification_dataset"

EPOCHS = 5
IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = "auto"
LEARNING_RATE = 0.001
RUN_NAME = "my_classification_train"
USE_PRETRAINED = True
FREEZE_BACKBONE = True


def main() -> int:
    if DATASET_DIR == r"C:\path\to\classification_dataset":
        print(with_help("Please edit DATASET_DIR in train.py before training.", DATASETS_AND_TRAINING))
        return 2

    config = ImageClassificationConfig()
    service = create_image_classification_service(config)
    dataset_dir = Path(DATASET_DIR).expanduser()
    report = service.validate_dataset(dataset_dir)
    print(report.to_text())
    if not report.ok:
        print(with_help("\nTraining stopped because the dataset is not valid.", DATASETS_AND_TRAINING))
        return 2

    job = ClassificationTrainingConfig(
        model_name=MODEL_NAME,
        dataset_dir=dataset_dir,
        output_dir=config.runs_dir,
        run_name=RUN_NAME,
        epochs=EPOCHS,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        workers=0,
        pretrained=USE_PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE,
    )
    try:
        best_model = service.train(job)
    except Exception as exc:
        print(with_help(f"\nTraining failed: {exc}", DATASETS_AND_TRAINING))
        return 1
    print("\nTraining finished.")
    print("Best model:", best_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
