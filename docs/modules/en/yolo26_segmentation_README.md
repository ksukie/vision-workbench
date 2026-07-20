# YOLO26 Segmentation README

[Back to README](../../../README.en.md) | [中文文档](../zh-CN/YOLO26分割.md) | [Extension Guide](../../adding_custom_features_README.md#yolo26-segmentation-extensions)

## Overview

YOLO26 Segmentation supports instance segmentation and semantic segmentation for images. It safely downloads official weights, validates local `.pt` files, and separates `-seg` and `-sem` model choices by task. The user-facing workflow is the native Qt page inside the unified Vision Workbench desktop.

## Setup

Install the optional YOLO26 dependency group once:

```bash
python scripts/install_dependencies.py yolo26
```

## Launch

```bash
vision-workbench
```

Open **YOLO Segmentation** from the left navigation, choose the task and model, load an image, run segmentation, and save the result.

## Custom Weight Locations

The segmentation page scans:

```text
models/yolo26_segmentation_models/
models/yolo26_segmentation_models/custom/
```

After instance segmentation training, copy `best.pt` to a name like `models/yolo26_segmentation_models/custom/my-seg.pt`. For semantic segmentation, prefer a name like `my-sem.pt`. The `-seg` and `-sem` suffixes help the page filter models by task.

## Python API

```python
from yolo26_segmentation.api import segment_image

result = segment_image("image.jpg", model_path="models/yolo26_segmentation_models/yolo26n-seg.pt")
```

## Source Layout

```text
src/yolo26_segmentation/api/             Public API
src/yolo26_segmentation/application/     Segmentation workflows
src/yolo26_segmentation/configuration/   Defaults and model paths
src/yolo26_segmentation/domain/          Data models
src/yolo26_segmentation/infrastructure/  Image/model/YOLO adapters
src/yolo26_segmentation/processing/      Segmenter wrapper
src/vision_workbench/desktop/            Unified Qt UI
src/yolo26_segmentation/window/          Legacy Tkinter compatibility/reference code
```

## Secondary Development

For task-specific model rules, rendering, saving, downloads, or runtime settings, update the service/infrastructure layers and expose controls in the Qt desktop page. See [Adding Custom Features - YOLO26 Segmentation Extensions](../../adding_custom_features_README.md#yolo26-segmentation-extensions).
