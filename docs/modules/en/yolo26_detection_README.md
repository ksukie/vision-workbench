# YOLO26 Detection README

[Back to README](../../../README.md) | [中文文档](../zh-CN/yolo26_detection_README.zh-CN.md) | [Extension Guide](../../adding_custom_features_README.md#yolo26-detection-add)

## Overview

The YOLO26 Detection module loads YOLO26 object-detection models and runs live camera inference. It supports camera selection, model selection, inference parameters, screenshots, and video recording.

## Setup

Use the shared project environment from the root [Quick Start](../../../README.md#quick-start), then install the YOLO26 dependency group once:

```bash
python scripts/install_dependencies.py yolo26
```

If dependencies were installed manually from `requirements-yolo26.txt`, run `python scripts/install_dependencies.py doctor` afterward to verify the Torch build.

## Launch

```bash
yolo26-detection-workbench
```

Source entry:

```bash
python -m yolo26_detection.window.app
```

## Model Directories

```text
third_party/yolo26_source/   YOLO26 source
models/yolo26_models/        YOLO26 detection models
```

Supported model files:

```text
yolo26n.pt
yolo26s.pt
yolo26m.pt
yolo26l.pt
yolo26x.pt
```

Custom `.pt` files can be placed under `models/yolo26_models/` or selected through the GUI.

## Workflow

1. Select `Refresh Cameras`.
2. Select a device from `Camera`.
3. Select a detection model from `Model`.
4. Use `Browse PT` for a custom model.
5. Configure `Device`, `Image size`, `Conf`, and `IoU`.
6. Select `Open`.
7. Select `Start Detection`.
8. Select `Screenshot` to save the current frame.
9. Use `Start Recording` and `Stop Recording` for video output.

## Parameters

| Parameter | Description |
| --- | --- |
| `Device` | Inference device, such as `auto`, `cpu`, `cuda`, or `mps` |
| `Image size` | Model input size |
| `Conf` | Confidence threshold |
| `IoU` | NMS IoU threshold |

## Python API

```python
from yolo26_detection.api import list_models, load_model, detect_objects

models = list_models()
detector = load_model(models[0].path)
result = detect_objects(detector, image)
```

## Source Layout

```text
src/yolo26_detection/api/             Public API
src/yolo26_detection/application/     Detection workflows
src/yolo26_detection/configuration/   Defaults and model directories
src/yolo26_detection/domain/          Data models
src/yolo26_detection/infrastructure/  Model discovery, camera source, YOLO26 backend
src/yolo26_detection/processing/      Detection processor
src/yolo26_detection/window/          Tkinter GUI
```

## Secondary Development

For detection models, inference parameters, class filtering, or post-processing logic, see [Adding Custom Features - YOLO26 Detection Extensions](../../adding_custom_features_README.md#yolo26-detection-add).
