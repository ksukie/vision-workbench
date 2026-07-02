# YOLO26 Detection README

[Back to README](../../../README.md) | [中文文档](../zh-CN/YOLO26目标检测.md) | [Extension Guide](../../adding_custom_features_README.md#yolo26-detection-extensions)

## Overview

YOLO26 Detection loads official or local `.pt` weights, downloads missing official weights safely, runs single-image detection, and supports live camera detection. The user-facing workflow is the native Qt page inside the unified Vision Workbench desktop.

## Setup

Install the optional YOLO26 dependency group once:

```bash
python scripts/install_dependencies.py yolo26
```

If dependencies were installed manually from `requirements-yolo26.txt`, run `python scripts/install_dependencies.py doctor` afterward to verify the Torch build.

## Launch

```bash
vision-workbench
```

Open **YOLO Detection** from the left navigation. The model dropdown shows full paths when expanded and only the selected file name when collapsed. Incomplete or corrupt model files are not treated as usable weights.

## Python API

```python
from yolo26_detection.api import detect_image

result = detect_image("image.jpg", model_path="models/yolo26_models/yolo26n.pt")
```

## Source Layout

```text
src/yolo26_detection/api/             Public API
src/yolo26_detection/application/     Detection workflows
src/yolo26_detection/configuration/   Defaults and model paths
src/yolo26_detection/domain/          Data models
src/yolo26_detection/infrastructure/  Camera, model, and YOLO adapters
src/yolo26_detection/processing/      Detector wrapper
src/vision_workbench/desktop/         Unified Qt UI
src/yolo26_detection/window/          Legacy Tkinter compatibility/reference code
```

## Secondary Development

For model discovery, camera inference, result rendering, downloads, or runtime settings, update the application/infrastructure layers and expose controls in the Qt desktop page. See [Adding Custom Features - YOLO26 Detection Extensions](../../adding_custom_features_README.md#yolo26-detection-extensions).
