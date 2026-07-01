# YOLO26 Segmentation README

[Back to README](../../../README.md) | [中文文档](../zh-CN/yolo26_segmentation_README.zh-CN.md) | [Extension Guide](../../adding_custom_features_README.md#yolo26-segmentation-add)

## Overview

The YOLO26 Segmentation module provides instance segmentation and semantic segmentation for image and camera inputs. It displays mask visualization results and supports saving the current output image.

## Setup

Use the shared project environment from the root [Quick Start](../../../README.md#quick-start), then install the YOLO26 dependency group once:

```bash
python scripts/install_dependencies.py yolo26
```

If dependencies were installed manually from `requirements-yolo26.txt`, run `python scripts/install_dependencies.py doctor` afterward to verify the Torch build.

## Launch

```bash
yolo26-segmentation-workbench
```

Source entry:

```bash
python -m yolo26_segmentation.window.app
```

## Model Directories

```text
third_party/yolo26_source/                 YOLO26 source
models/yolo26_segmentation_models/         YOLO26 segmentation models
models/yolo26_segmentation_models/custom/  Custom segmentation models
```

Model naming:

```text
*-seg.pt  Instance segmentation
*-sem.pt  Semantic segmentation
```

## Task Types

| Task | Description |
| --- | --- |
| `segment` | Instance segmentation with object-level masks |
| `semantic` | Semantic segmentation with pixel-level class regions |

## Workflow

1. Select `segment` or `semantic` from `Task`.
2. Select a matching model from `Model`.
3. Use `Browse PT` for a custom model.
4. Select `Open Image` or select a camera.
5. Select `Run Once` for one inference pass.
6. Select `Start Live` for live segmentation.
7. Select `Save Result` to save the current output.
8. Select `Stop` to stop live processing.

## Python API

```python
from yolo26_segmentation.api import list_models, segment_image

models = list_models(task="segment")
result = segment_image("input.jpg", model_path=models[0].path)
```

## Source Layout

```text
src/yolo26_segmentation/api/             Public API
src/yolo26_segmentation/application/     Segmentation workflows
src/yolo26_segmentation/configuration/   Defaults and model directories
src/yolo26_segmentation/domain/          Data models
src/yolo26_segmentation/infrastructure/  Model discovery, image I/O, YOLO26 backend
src/yolo26_segmentation/processing/      Segmentation processor
src/yolo26_segmentation/window/          Tkinter GUI
```

## Secondary Development

For segmentation models, mask saving, visualization, or batch workflows, see [Adding Custom Features - YOLO26 Segmentation Extensions](../../adding_custom_features_README.md#yolo26-segmentation-add).
