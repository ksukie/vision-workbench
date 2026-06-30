# Image Classification README

[Back to README](../../../README.md) | [中文文档](../zh-CN/image_classification_README.zh-CN.md) | [Extension Guide](../../adding_custom_features_README.md#image-classification-add)

## Overview

The Image Classification module predicts the category of an entire image. It provides ResNet18 and MobileNetV3 Small backbones, pretrained ImageNet prediction, offline weight import, folder-based dataset validation, basic transfer learning, and Top-K result display.

## Feature Scope

| Feature | Description |
| --- | --- |
| Pretrained prediction | Single-image prediction with ResNet18 or MobileNetV3 Small ImageNet weights |
| Weight management | Check, download, and import pretrained weights |
| Dataset validation | Validate `train/val/class_name` folder datasets |
| Dataset split | Split raw class folders into `train/val` |
| Training | Basic transfer learning from pretrained weights |
| Result display | Display Top-K labels, confidence scores, and inference time |

## Models

| Model | Purpose |
| --- | --- |
| `resnet18` | Classic CNN backbone for teaching and baseline experiments |
| `mobilenet_v3_small` | Lightweight backbone for low-resource environments and quick demos |

## Environment

```bash
conda create -n vision-classification python=3.11 -y
conda activate vision-classification
cd path/to/vision-workbench
pip install -e .
python scripts/install_dependencies.py classification
```

## Launch

```bash
image-classification-workbench
```

Main GUI entry:

```bash
vision-workbench
```

Source entry:

```bash
python .\src\image_classification\window\app.py
```

## Dataset Format

The classification dataset uses folder names as class names:

```text
dataset/
  train/
    cat/
      001.jpg
    dog/
      001.jpg
  val/
    cat/
      101.jpg
    dog/
      101.jpg
```

Validation checks:

- `train/` and `val/` existence.
- Class consistency between training and validation splits.
- Image count per class.
- Image readability.
- Very small class counts.

## Pretrained Weights

Pretrained weight directory:

```text
models/image_classification_models/pretrained/
```

Supported files:

```text
resnet18-f37072fd.pth
mobilenet_v3_small-047dcff4.pth
```

GUI weight-management controls:

| Control | Function |
| --- | --- |
| `Check Weights` | Check local weight status for the selected backbone |
| `Download Pretrained Weights` | Download TorchVision official weights for the selected backbone |
| `Import Local Weights` | Import an offline `.pth` or `.pt` weight file |

Prediction and training prefer local weights from the project directory. If local weights are missing, TorchVision may use its default download mechanism.

TorchVision default cache location:

```text
Windows: C:\Users\username\.cache\torch\hub\checkpoints
Linux/macOS: ~/.cache/torch/hub/checkpoints
```

## Prediction Workflow

1. Open the `Predict` tab.
2. Select `Open Image`.
3. Select a backbone from `Model`.
4. Select `Check Weights`.
5. Download or import weights when required.
6. Select `Predict Pretrained`.
7. Review the Top-K prediction results.

## Training Workflow

1. Open the `Train` tab.
2. Select the classification dataset root.
3. Select `Validate`.
4. Select backbone, device, epochs, image size, and batch size.
5. Configure pretrained weights and backbone freezing.
6. Select `Start Training`.
7. Review outputs under `runs/image_classification/`.

## CLI Training

```bash
image-classification-train ^
  --data C:\path\to\classification_dataset ^
  --model resnet18 ^
  --epochs 5 ^
  --imgsz 224 ^
  --batch 16 ^
  --device auto
```

Dataset validation only:

```bash
image-classification-train --data C:\path\to\classification_dataset --model mobilenet_v3_small --dry-run
```

## Python API

```python
from image_classification.api import (
    validate_classification_dataset,
    predict_with_pretrained,
)

report = validate_classification_dataset("C:/path/to/dataset")
result = predict_with_pretrained("resnet18", "C:/path/to/image.jpg", topk=5)
```

Common functions:

```text
supported_models()
validate_classification_dataset(dataset_dir)
split_classification_dataset(input_dir, output_dir, train_ratio=0.8)
pretrained_weight_status(model_name=None)
download_pretrained_weight(model_name)
import_pretrained_weight(model_name, source_path)
list_classification_models()
train_classifier(config)
load_pretrained_classifier(model_name)
load_classifier(model_path)
predict_with_pretrained(model_name, image_path, topk=5)
predict_image(model_path, image_path, topk=5)
```

## Source Layout

```text
src/image_classification/api/             Public API
src/image_classification/application/     Classification workflows
src/image_classification/configuration/   Paths and defaults
src/image_classification/domain/          Data models
src/image_classification/infrastructure/  Dataset validation, splitting, model discovery, weight management
src/image_classification/processing/      TorchVision training and prediction backend
src/image_classification/window/          Tkinter GUI
```

## Secondary Development

For new backbones, pretrained weight rules, augmentation, or export logic, see [Adding Custom Features - Image Classification Extensions](../../adding_custom_features_README.md#image-classification-add).
