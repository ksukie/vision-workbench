# YOLO26 Training README

[Back to README](../../../README.md) | [中文文档](../zh-CN/YOLO26训练.md) | [Extension Guide](../../adding_custom_features_README.md#yolo26-training-extensions)

## Overview

YOLO26 Training provides dataset validation and basic training for detection, instance segmentation, and semantic segmentation. The user-facing workflow is the native Qt page inside the unified Vision Workbench desktop, while `yolo26-train` remains available for command-line training.

## Setup

Install the optional YOLO26 dependency group once:

```bash
python scripts/install_dependencies.py yolo26
```

If dependencies were installed manually from `requirements-yolo26.txt`, run `python scripts/install_dependencies.py doctor` afterward to verify the Torch build. GPU training requires a CUDA-enabled PyTorch build; CPU training is supported but slower.

## Launch

Qt GUI:

```bash
vision-workbench
```

Open **YOLO Training** from the left navigation, choose `detect`, `segment`, or `semantic`, select `data.yaml`, choose a pretrained weight file, validate the dataset, and start training.

CLI:

```bash
yolo26-train --task detect --data path/to/dataset/data.yaml --model models/yolo26_models/yolo26n.pt
```

## Task Types

| Task | Label format |
| --- | --- |
| `detect` | `class x_center y_center width height` |
| `segment` | `class x1 y1 x2 y2 x3 y3 ...` |
| `semantic` | PNG/TIF mask or polygon labels |

The training model list is task-aware: detection weights are separate from `-seg` and `-sem` weights, and incomplete `.pt` files are skipped.

## CLI Training

```bash
yolo26-train --task detect --data path/to/dataset/data.yaml --model path/to/model.pt --epochs 100 --imgsz 640 --batch 16 --device auto
```

Validation only:

```bash
yolo26-train --task detect --data path/to/data.yaml --model path/to/model.pt --dry-run
```

## Output Directory

```text
runs/yolo26_training/
```

YOLO26 training outputs commonly include weights, configuration files, logs, and metric plots.

Common weight paths:

```text
runs/yolo26_training/<run_name>/weights/best.pt
runs/yolo26_training/<run_name>/weights/last.pt
```

Prefer `best.pt` for later inference. To make trained weights appear in the detection or segmentation dropdowns, copy and rename the file into the task-specific folder, then click the page refresh button:

| Task | Recommended destination | Naming suggestion |
| --- | --- | --- |
| `detect` | `models/yolo26_models/custom/` | `my-det.pt` |
| `segment` | `models/yolo26_segmentation_models/custom/` | `my-seg.pt` |
| `semantic` | `models/yolo26_segmentation_models/custom/` | `my-sem.pt` |

See [Trained Model Loading And API Notes](./trained_model_loading_and_api_README.md) for trained-weight placement, dropdown discovery, and Python API notes.

## Python API

```python
from yolo26_training.api import validate_dataset, list_models

report = validate_dataset("path/to/dataset/data.yaml", task="detect")
models = list_models(task="detect")
```

## Source Layout

```text
src/yolo26_training/api/             Public API
src/yolo26_training/application/     Training workflows
src/yolo26_training/configuration/   Defaults and paths
src/yolo26_training/domain/          Data models
src/yolo26_training/infrastructure/  Dataset validation, model discovery, YOLO backend
src/yolo26_training/runner.py        CLI training entry point
src/yolo26_training/train.py         Basic training script
src/vision_workbench/desktop/        Unified Qt UI
src/yolo26_training/window/          Legacy Tkinter compatibility/reference code
```

## Secondary Development

For training tasks, parameters, dataset validation rules, or log-saving strategies, update the service/infrastructure layers and expose controls in the Qt desktop page. See [Adding Custom Features - YOLO26 Training Extensions](../../adding_custom_features_README.md#yolo26-training-extensions).
