# Image Classification README

[Back to README](../../../README.md) | [中文文档](../zh-CN/图像分类.md) | [Extension Guide](../../adding_custom_features_README.md#image-classification-extensions)

## Overview

Image Classification provides ResNet18 and MobileNetV3 Small prediction, pretrained weight management, dataset validation, and basic training. The user-facing workflow is the native Qt page inside the unified Vision Workbench desktop.

## Setup

Install the optional classification dependency group once:

```bash
python scripts/install_dependencies.py classification
```

If dependencies were installed manually from `requirements-classification.txt`, run `python scripts/install_dependencies.py doctor` afterward to verify the Torch build.

## Launch

```bash
vision-workbench
```

Open **Image Classification** from the left navigation. Use the prediction controls for pretrained or custom checkpoints, and the training section for classification datasets with `train/` and `val/` class folders.

Prediction and training are separate tabs. New users can create the tiny sample dataset, check the training environment, apply the recommended batch size, validate, and train. The isolated training process reports loss and validation accuracy per epoch and can be stopped from the page.

## CLI Training

```bash
image-classification-train --dataset path/to/dataset --model resnet18 --epochs 5
```

## Python API

```python
from image_classification.api import predict_with_pretrained

result = predict_with_pretrained("resnet18", "image.jpg", topk=5)
```

## Source Layout

```text
src/image_classification/api/             Public API
src/image_classification/application/     Prediction and training workflows
src/image_classification/configuration/   Defaults and paths
src/image_classification/domain/          Data models
src/image_classification/infrastructure/  Dataset/model/weight adapters
src/image_classification/processing/      TorchVision backend
src/image_classification/runner.py        CLI training entry point
src/vision_workbench/desktop/             Unified Qt UI
src/image_classification/window/          Legacy Tkinter compatibility/reference code
```

## Secondary Development

For new backbones, pretrained weights, metrics, dataset rules, batch prediction, or export support, update the service/backend layers and expose new UI controls in the Qt desktop page. See [Adding Custom Features - Image Classification Extensions](../../adding_custom_features_README.md#image-classification-extensions).
