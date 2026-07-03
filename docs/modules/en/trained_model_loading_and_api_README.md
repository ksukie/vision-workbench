# Trained Model Loading And API Notes

[中文](../zh-CN/训练后模型加载与接口说明.md) | [Back to YOLO26 Training](./yolo26_training_README.md)

This note covers where to place YOLO26 training outputs, how model dropdowns discover local files, and the common Python APIs. Project installation and first launch remain in the root README Quick Start. Developers extending the project can continue with the [Extension Guide](../../adding_custom_features_README.md).

## API Layers

Feature packages follow the same rough structure:

```text
src/<feature>/api/             Public Python API for users
src/<feature>/application/     Workflow orchestration
src/<feature>/domain/          Data structures and settings objects
src/<feature>/infrastructure/  File, model, camera, and third-party adapters
src/<feature>/processing/      Algorithm or model wrappers
src/vision_workbench/desktop/  Unified Qt desktop UI
src/<feature>/window/          Legacy Tkinter compatibility/reference code
```

Most users should start with:

- GUI: run `vision-workbench`
- Python: import from each feature package's `api` module

## Common Python APIs

Detection:

```python
from yolo26_detection.api import detect_image

result = detect_image(
    "image.jpg",
    model_path="models/yolo26_models/yolo26n.pt",
)
print(result.detection_count)
```

Segmentation:

```python
from yolo26_segmentation.api import segment_image
from yolo26_segmentation.domain import SegmentationSettings

result = segment_image(
    "image.jpg",
    model_path="models/yolo26_segmentation_models/yolo26n-seg.pt",
    settings=SegmentationSettings(task="segment"),
)
print(result.item_count)
```

Training dataset validation:

```python
from yolo26_training.api import validate_dataset, list_models

report = validate_dataset("path/to/data.yaml", task="detect")
print(report.to_text())
print(list_models(task="detect"))
```

Image classification:

```python
from image_classification.api import predict_with_pretrained

result = predict_with_pretrained("resnet18", "image.jpg")
print(result.predictions[:3])
```

## Where To Put Trained YOLO Weights

After YOLO training finishes, weights are usually written to:

```text
runs/yolo26_training/<run_name>/weights/best.pt
runs/yolo26_training/<run_name>/weights/last.pt
```

Use `best.pt` in most cases. To make it appear in detection, segmentation, or training dropdowns, copy and rename it into the task-specific model folder, then click the page refresh button.

| Task | Recommended destination | Naming suggestion | Visible in |
| --- | --- | --- | --- |
| `detect` | `models/yolo26_models/custom/` | `my-det.pt` | YOLO Detection, YOLO Training `detect` |
| `segment` | `models/yolo26_segmentation_models/custom/` | `my-seg.pt` | YOLO Segmentation, YOLO Training `segment` |
| `semantic` | `models/yolo26_segmentation_models/custom/` | `my-sem.pt` | YOLO Segmentation, YOLO Training `semantic` |

You can also use the per-user directory when you do not want to modify the project folder:

```text
~/.vision_workbench/models/yolo26_models/
```

Notes:

- Detection weights should not use `-seg` or `-sem` suffixes.
- Instance segmentation weights should preferably use `-seg`.
- Semantic segmentation weights should preferably use `-sem`.
- Incomplete or corrupt `.pt` files are skipped and will not appear as usable models.

Windows example:

```powershell
New-Item -ItemType Directory -Force models\yolo26_models\custom
Copy-Item runs\yolo26_training\my_run\weights\best.pt models\yolo26_models\custom\my-det.pt
```

## Where To Put Classification Checkpoints

Classification training usually writes:

```text
runs/image_classification/<run_name>/best.pt
```

The GUI training workflow also copies a saved checkpoint into:

```text
models/image_classification_models/custom/
```

The classification page can also load any compatible `.pt` or `.pth` checkpoint through its custom model picker.

## Remote Model Catalog

If you maintain a remote model manifest, set:

```powershell
$env:VISION_WORKBENCH_MODEL_MANIFEST_URL="https://example.com/model_manifest.json"
```

When users click refresh in the YOLO pages, Vision Workbench updates:

```text
~/.vision_workbench/cache/model_manifest.json
```

Without network access, cached and bundled model lists are still used.
