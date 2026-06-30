# YOLO26 Training README

[Back to README](../../../README.md) | [中文文档](../zh-CN/yolo26_training_README.zh-CN.md) | [Extension Guide](../../adding_custom_features_README.md#yolo26-training-add)

## Overview

The YOLO26 Training module provides basic training entry points for detection, instance segmentation, and semantic segmentation tasks. Dataset validation runs before training and stops the workflow when the dataset is invalid.

## Environment

```bash
conda create -n vision-yolo-train python=3.11 -y
conda activate vision-yolo-train
cd path/to/vision-workbench
pip install -e .
pip install -r requirements-yolo26.txt
```

GPU training requires a PyTorch build that matches the local CUDA version. CPU training is supported but slower.

## Launch

GUI:

```bash
yolo26-training-workbench
```

CLI:

```bash
yolo26-train --task detect --data C:\path\to\dataset\data.yaml --model models\yolo26_models\yolo26n.pt
```

Basic script:

```bash
python .\src\yolo26_training\train.py
```

## Task Types

| Task | Label format |
| --- | --- |
| `detect` | `class x_center y_center width height` |
| `segment` | `class x1 y1 x2 y2 x3 y3 ...` |
| `semantic` | PNG/TIF mask or polygon labels |

Datasets must provide `data.yaml` with image paths, class count, and class names aligned with the actual dataset.

## GUI Workflow

1. Select `detect`, `segment`, or `semantic` from `Task`.
2. Select dataset `data.yaml`.
3. Select an initial model.
4. Configure `Epochs`, `Image size`, `Batch size`, `Device`, and related parameters.
5. Run dataset validation.
6. Start training after validation succeeds.
7. Review logs in the GUI.
8. Review outputs under `runs/yolo26_training/`.

## CLI Training

```bash
yolo26-train ^
  --task detect ^
  --data C:\path\to\dataset\data.yaml ^
  --model C:\path\to\model.pt ^
  --epochs 100 ^
  --imgsz 640 ^
  --batch 16 ^
  --device auto
```

Validation only:

```bash
yolo26-train --task detect --data C:\path\to\data.yaml --model C:\path\to\model.pt --dry-run
```

## Basic Script

Configuration variables are defined at the top of:

```text
src/yolo26_training/train.py
```

Run:

```bash
python .\src\yolo26_training\train.py
```

## Output Directory

```text
runs/yolo26_training/
```

The YOLO26 source generates training outputs, which commonly include weights, configuration files, logs, and metric plots.

## Python API

```python
from yolo26_training.api import validate_dataset, list_models

report = validate_dataset("C:/path/to/dataset/data.yaml", task="detect")
models = list_models(task="detect")
```

## Source Layout

```text
src/yolo26_training/api/             Public API
src/yolo26_training/application/     Training workflows
src/yolo26_training/configuration/   Defaults and paths
src/yolo26_training/domain/          Data models
src/yolo26_training/infrastructure/  Dataset validation, model discovery, YOLO26 training backend
src/yolo26_training/runner.py        CLI entry point
src/yolo26_training/train.py         Basic training script
src/yolo26_training/window/          Tkinter GUI
```

## Secondary Development

For training tasks, parameters, dataset validation rules, or log-saving strategies, see [Adding Custom Features - YOLO26 Training Extensions](../../adding_custom_features_README.md#yolo26-training-add).
